"""Build extraction inputs from the lake, attach labels, split without leakage and write SFT data.

Splits:
1. test: filings from the last ``test_years`` years, plus every filing of a held out 10% of companies.
2. val: the year before the test period (companies not held out).
3. train: everything earlier.
Holding out companies as well as time catches a model that memorises company specific boilerplate.

Training data is stratified: the rare classes (guidance changes, liquidity, litigation) are all kept and the
dominant "nothing happened" filings are downsampled, otherwise a small model learns to always say "none".
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .prompts import ExtractionInput, chat_messages, select_paragraphs
from .schema import FilingSignals, Quote

TEXT_SECTIONS = {"mdna", "legal", "8k_results_of_operations", "8k_material_agreement", "8k_officer_change",
                 "8k_other_events", "full_text"}


def build_inputs(paragraphs: pd.DataFrame, index: pd.DataFrame, budget_chars: int = 12_000) -> dict[str, ExtractionInput]:
    p = paragraphs[paragraphs["section"].isin(TEXT_SECTIONS)].sort_values(["accession", "section", "para_idx"])
    company = index.set_index("accession")["company"].to_dict() if "company" in index else {}
    out = {}
    for (acc, form), g in p.groupby(["accession", "form"], sort=False):
        paras = g[["paragraph_id", "section", "text"]].to_dict("records")
        out[acc] = ExtractionInput(acc, form, company.get(acc, ""), select_paragraphs(paras, budget_chars))
    return out


def _find_quote(inp: ExtractionInput, sentence: str | None) -> list[Quote]:
    if isinstance(sentence, str) and sentence:
        for i, p in enumerate(inp.paragraphs):
            if sentence in p["text"]:
                return [Quote(paragraph=i + 1, text=sentence[:300])]
    if inp.paragraphs:
        first = re.split(r"(?<=[.!?])\s+", inp.paragraphs[0]["text"])[0]
        return [Quote(paragraph=1, text=first[:300])]
    return []


def labels_from_synthetic_truth(truth: pd.DataFrame, inputs: dict[str, ExtractionInput]) -> dict[str, dict]:
    labels = {}
    for r in truth.itertuples(index=False):
        inp = inputs.get(r.accession)
        if inp is None:
            continue
        flags = [n for n, v in (("liquidity concern", r.liquidity_concern), ("litigation", r.litigation_risk)) if v]
        summary = (f"Guidance {r.guidance_direction}; management tone {r.management_tone}"
                   + (f"; flags: {', '.join(flags)}." if flags else "; no liquidity or litigation flags."))
        sig = FilingSignals(guidance_direction=r.guidance_direction, management_tone=r.management_tone,
                            liquidity_concern=bool(r.liquidity_concern), litigation_risk=bool(r.litigation_risk),
                            key_quotes=_find_quote(inp, r.evidence), summary=summary)
        labels[r.accession] = sig.model_dump()
    return labels


def assign_splits(index: pd.DataFrame, test_years: int = 2, holdout_frac: float = 0.1, seed: int = 0) -> pd.Series:
    year = pd.to_datetime(index["date_filed"]).dt.year
    last = int(year.max())
    rng = np.random.default_rng(seed)
    ciks = index["cik"].unique()
    held = set(rng.choice(ciks, size=max(1, int(len(ciks) * holdout_frac)), replace=False))
    split = np.where(year > last - test_years, "test", np.where(year == last - test_years, "val", "train"))
    split = np.where(index["cik"].isin(held), "test", split)
    return pd.Series(split, index=index["accession"].to_numpy(), name="split")


def is_rare(lab: dict) -> bool:
    return lab["guidance_direction"] in ("raised", "lowered") or lab["liquidity_concern"] or lab["litigation_risk"]


def stratified(accs: list[str], labels: dict[str, dict], common_per_rare: float = 1.5, seed: int = 0) -> list[str]:
    rare = [a for a in accs if is_rare(labels[a])]
    common = [a for a in accs if not is_rare(labels[a])]
    rng = np.random.default_rng(seed)
    k = min(len(common), max(int(len(rare) * common_per_rare), 50))
    keep = rare + list(rng.choice(common, size=k, replace=False)) if common else rare
    rng.shuffle(keep)
    return list(keep)


def sft_record(inp: ExtractionInput, label: dict) -> dict:
    """TRL conversational prompt completion format: loss is computed on the completion only."""
    return {
        "prompt": chat_messages(inp),
        "completion": [{"role": "assistant", "content": json.dumps(label, separators=(",", ":"))}],
        "accession": inp.accession,
        "form": inp.form,
    }


def write_dataset(inputs: dict[str, ExtractionInput], labels: dict[str, dict], splits: pd.Series, out_dir: str | Path,
                  max_train: int | None = None, seed: int = 0) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name in ("train", "val", "test"):
        accs = [a for a in splits[splits == name].index if a in labels and a in inputs]
        if name == "train":
            accs = stratified(accs, labels, seed=seed)[: max_train or None]
        with open(out / f"{name}.jsonl", "w") as fh:
            for a in accs:
                fh.write(json.dumps(sft_record(inputs[a], labels[a])) + "\n")
        stats[name] = len(accs)
        if name == "test":  # the gold set for the eval harness: inputs and labels, no prompt formatting
            gold = [{"accession": a, "form": inputs[a].form, "company": inputs[a].company,
                     "paragraphs": inputs[a].paragraphs, "label": labels[a]} for a in accs]
            (out / "gold.jsonl").write_text("\n".join(json.dumps(g) for g in gold) + "\n")
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def load_gold(path: str | Path) -> list[tuple[ExtractionInput, dict]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [(ExtractionInput(r["accession"], r["form"], r.get("company", ""), r["paragraphs"]), r["label"]) for r in rows]
