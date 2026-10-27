"""Evaluation harness: run several extractors on the same gold set and compare them fairly.

Metrics per system:
1. json_valid_rate, schema_valid_rate (an invalid output scores zero on every field).
2. Per field macro F1 (guidance 4 classes, tone 3 classes, booleans as 2 classes) and accuracy.
3. extraction_score: mean macro F1 over the four fields, the single number the CI gate protects.
4. quote_grounding_rate: share of quotes that appear verbatim in the paragraph they cite.
5. Latency p50 and p95, and cost per 1,000 filings (API tokens, or GPU hours for self hosted models).
Comparisons between systems use a paired bootstrap over gold items, so "fine tuned beats base" comes with a
confidence interval rather than a single point estimate.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .providers import ExtractionResult, Extractor
from .schema import LABEL_FIELDS

CLASSES = {
    "guidance_direction": ["raised", "lowered", "maintained", "none"],
    "management_tone": ["positive", "neutral", "negative"],
    "liquidity_concern": [True, False],
    "litigation_risk": [True, False],
}
INVALID = "__invalid__"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s%$.,]", "", s.lower())).strip()


def run_system(ex: Extractor, gold, workers: int = 1, cache: Path | None = None) -> list[ExtractionResult]:
    """Run one extractor over the gold set; results are cached as JSONL so reruns are free."""
    if cache and cache.exists():
        return [ExtractionResult(**json.loads(x)) for x in cache.read_text().splitlines() if x.strip()]
    inputs = [inp for inp, _ in gold]
    if workers > 1:
        with ThreadPoolExecutor(workers) as pool:
            results = list(pool.map(ex.extract, inputs))
    else:
        results = [ex.extract(i) for i in inputs]
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text("\n".join(json.dumps(r.to_dict()) for r in results) + "\n")
    return results


def item_table(results: list[ExtractionResult], gold) -> pd.DataFrame:
    rows = []
    for r, (inp, lab) in zip(results, gold, strict=True):
        pred = r.signals or {}
        row = {"accession": inp.accession, "form": inp.form, "json_valid": r.json_valid, "schema_valid": r.schema_valid,
               "latency_s": r.latency_s, "cost_usd": r.cost_usd, "error": r.error}
        for f in LABEL_FIELDS:
            row[f"gold_{f}"] = lab[f]
            row[f"pred_{f}"] = pred.get(f, INVALID) if r.schema_valid else INVALID
        quotes = pred.get("key_quotes", []) if r.schema_valid else []
        grounded = 0
        for q in quotes:
            k = q.get("paragraph", 0) - 1
            if 0 <= k < len(inp.paragraphs) and _norm(q.get("text", "")) and _norm(q["text"]) in _norm(inp.paragraphs[k]["text"]):
                grounded += 1
        row["n_quotes"], row["n_grounded"] = len(quotes), grounded
        rows.append(row)
    return pd.DataFrame(rows)


def macro_f1(gold: np.ndarray, pred: np.ndarray, classes: list) -> float:
    f1s = []
    for c in classes:
        tp = np.sum((pred == c) & (gold == c))
        fp = np.sum((pred == c) & (gold != c))
        fn = np.sum((pred != c) & (gold == c))
        if tp + fp + fn == 0:
            continue  # class absent from both gold and predictions
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else float("nan")


def field_scores(t: pd.DataFrame) -> dict[str, float]:
    out = {}
    for f in LABEL_FIELDS:
        g, p = t[f"gold_{f}"].to_numpy(dtype=object), t[f"pred_{f}"].to_numpy(dtype=object)
        out[f"{f}_f1"] = macro_f1(g, p, CLASSES[f])
        out[f"{f}_acc"] = float(np.mean(g == p))
    out["extraction_score"] = float(np.nanmean([out[f"{f}_f1"] for f in LABEL_FIELDS]))
    return out


def summarize(t: pd.DataFrame, gpu_hourly_usd: float | None = None, concurrency: int = 1) -> dict:
    s = {"n": int(len(t)), "json_valid_rate": float(t["json_valid"].mean()),
         "schema_valid_rate": float(t["schema_valid"].mean()), **field_scores(t)}
    s["quote_grounding_rate"] = float(t["n_grounded"].sum() / t["n_quotes"].sum()) if t["n_quotes"].sum() else float("nan")
    s["latency_p50_s"] = float(t["latency_s"].quantile(0.5))
    s["latency_p95_s"] = float(t["latency_s"].quantile(0.95))
    if t["cost_usd"].sum() > 0:
        s["cost_per_1k_usd"] = float(t["cost_usd"].mean() * 1000)
    elif gpu_hourly_usd:
        s["cost_per_1k_usd"] = float(t["latency_s"].mean() * 1000 / 3600 * gpu_hourly_usd / max(concurrency, 1))
    else:
        s["cost_per_1k_usd"] = 0.0
    return s


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    """CI for extraction_score(a) minus extraction_score(b), resampling the same gold items for both."""
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(field_scores(a.iloc[idx])["extraction_score"] - field_scores(b.iloc[idx])["extraction_score"])
    diffs = np.asarray(diffs)
    point = field_scores(a)["extraction_score"] - field_scores(b)["extraction_score"]
    return {"diff": float(point), "ci_low": float(np.quantile(diffs, 0.025)), "ci_high": float(np.quantile(diffs, 0.975)),
            "p_le_0": float(np.mean(diffs <= 0))}


def evaluate(systems: list[Extractor], gold, out_dir: str | Path, reference: str | None = None, workers: int = 1,
             gpu_hourly_usd: dict | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables, summary = {}, {}
    for ex in systems:
        safe = re.sub(r"[^\w.-]", "_", ex.name)
        res = run_system(ex, gold, workers, cache=out / "predictions" / f"{safe}.jsonl")
        tables[ex.name] = item_table(res, gold)
        summary[ex.name] = summarize(tables[ex.name], (gpu_hourly_usd or {}).get(ex.name))
    comparisons = {}
    if reference and reference in tables:
        for name, t in tables.items():
            if name != reference:
                comparisons[f"{name} vs {reference}"] = paired_bootstrap(t, tables[reference])
    result = {"summary": summary, "comparisons": comparisons}
    (out / "eval.json").write_text(json.dumps(result, indent=2))
    (out / "report.md").write_text(render(result))
    items = pd.concat([t.assign(system=k) for k, t in tables.items()], ignore_index=True)
    label_cols = [c for c in items.columns if c.startswith(("gold_", "pred_"))]
    items[label_cols] = items[label_cols].astype(str)  # predictions mix bools with the invalid marker
    items.to_parquet(out / "items.parquet", index=False)
    return result


def render(result: dict) -> str:
    cols = ["n", "json_valid_rate", "schema_valid_rate", "extraction_score", "guidance_direction_f1",
            "management_tone_f1", "liquidity_concern_f1", "litigation_risk_f1", "quote_grounding_rate",
            "latency_p50_s", "latency_p95_s", "cost_per_1k_usd"]
    df = pd.DataFrame(result["summary"]).T[cols]
    lines = ["# Phase 3 evaluation", "", df.round(4).to_markdown(), ""]
    if result["comparisons"]:
        lines += ["## Paired bootstrap on extraction_score", "", "| comparison | diff | 95% CI | P(diff <= 0) |", "|---|---|---|---|"]
        for k, v in result["comparisons"].items():
            lines.append(f"| {k} | {v['diff']:.4f} | [{v['ci_low']:.4f}, {v['ci_high']:.4f}] | {v['p_le_0']:.3f} |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------ gate
GATED = {  # metric -> (rule, value)
    "extraction_score": ("max", 0.01),        # may not drop more than 0.01 below production
    "schema_valid_rate": ("max", 0.005),
    "quote_grounding_rate": ("max", 0.02),
    "guidance_direction_f1": ("max", 0.02),
    "latency_p95_s": ("slo", 8.0),            # absolute service level objective, not relative to production
}


def gate(candidate: dict, baseline: dict, rules: dict = GATED) -> tuple[bool, list[str]]:
    """A candidate model may not be worse than production on any gated metric beyond its tolerance."""
    failures = []
    for metric, (rule, tol) in rules.items():
        c, b = candidate.get(metric), baseline.get(metric)
        if c is None or (isinstance(c, float) and np.isnan(c)):
            continue
        if rule == "slo":
            if c > tol:
                failures.append(f"{metric}: candidate {c:.4f} exceeds the SLO of {tol}")
            continue
        if b is None:
            continue
        worse = (c < b - tol) if rule == "max" else (c > b + tol)
        if worse:
            failures.append(f"{metric}: candidate {c:.4f} vs production {b:.4f} (tolerance {tol})")
    return not failures, failures
