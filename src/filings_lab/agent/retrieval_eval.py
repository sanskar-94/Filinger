"""Retrieval evaluation on the synthetic world.

The generator writes every fact with a sentence from a known bank, so which paragraphs state a fact is
known exactly: a paragraph is relevant to an information need when it contains one of that need's
sentences. Each need is asked two ways:

* ``lexical`` queries reuse words of the filing sentence, as a keyword search would be typed;
* ``paraphrase`` queries say the same thing in other words ("cut its sales forecast" for "lowering its
  revenue guidance"), as an analyst would ask it;
* ``exact`` queries look a specific figure up ("2019 revenue guidance $11 billion to $14 billion"). They are
  generated from the paragraphs themselves, and only the paragraph with those very numbers is relevant.

Every query runs once per company that has a relevant paragraph, scoped to that company as the agent's
searches are. Queries repeat across companies, so intervals come from a bootstrap that resamples whole
queries, not single runs.

Results are scored on distinct texts: a paragraph repeated verbatim in later filings counts once, and a
second copy in a result list earns nothing.

What this can show: whether dense retrieval and fusion recover paragraphs that keyword search cannot reach,
and whether keyword search still wins where an exact figure matters.
What it cannot show: how large that gain is on real filings. The paraphrases were written with the
generator's sentences in view, so the lexical set is easy and the paraphrase set hard for BM25 by design.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..synthetic import EIGHTK_OTHER, GUIDANCE, LIQUIDITY, LITIGATION, NEG

METRICS = ["hit@1", "hit@5", "recall@10", "mrr@10", "ndcg@10"]


@dataclass(frozen=True)
class Need:
    name: str
    templates: tuple[str, ...]
    lexical: tuple[str, ...]
    paraphrase: tuple[str, ...]


NEEDS: tuple[Need, ...] = (
    Need("guidance_lowered", tuple(GUIDANCE["lowered"]),
         ("lowering full year revenue guidance", "reduced earnings outlook below previously issued outlook"),
         ("management cut its sales forecast", "trimmed its annual projections", "expects to miss its earlier targets")),
    Need("guidance_raised", tuple(GUIDANCE["raised"]),
         ("raising full year revenue guidance", "increased outlook for earnings per share"),
         ("management lifted its sales forecast", "boosted its annual projections", "now predicts a better year than it did before")),
    Need("guidance_maintained", tuple(GUIDANCE["maintained"]),
         ("reaffirms full year revenue guidance", "outlook unchanged from guidance provided last quarter"),
         ("kept its forecast the same", "stuck with its earlier annual targets", "no revision to its projections")),
    Need("going_concern", (LIQUIDITY[0],),
         ("substantial doubt about ability to continue as a going concern",),
         ("the business may not survive", "danger of insolvency or bankruptcy", "might have to shut down for lack of money")),
    Need("covenant_breach", (LIQUIDITY[1],),
         ("not in compliance with financial covenants requested a waiver from lenders",),
         ("violated the terms of its loans", "broke its debt agreements and asked the banks for relief", "defaulted on borrowing conditions")),
    Need("capital_need", (LIQUIDITY[2],),
         ("need to raise additional capital within the next twelve months to fund operations",),
         ("running out of cash and must find new financing", "requires fresh funding soon", "may have to sell shares or borrow to pay its bills")),
    Need("class_action", (LITIGATION[0],),
         ("putative securities class action filed against the Company and certain officers",),
         ("shareholders sued the firm and its executives", "investor lawsuit accusing leadership of lying", "stockholders went to court over deceptive disclosures")),
    Need("sec_subpoena", (LITIGATION[1],),
         ("subpoena from the Securities and Exchange Commission requesting documents",),
         ("regulators are investigating the accounting", "government probe into how sales were booked", "federal watchdog demanded records")),
    Need("patent_verdict", (LITIGATION[2],),
         ("jury returned a verdict in a patent infringement case and awarded damages",),
         ("lost a court case over intellectual property", "ordered to pay after a trial about copied technology", "court ruled against the firm in an IP dispute")),
    Need("ceo_departure", (EIGHTK_OTHER["5.02"][0],),
         ("Chief Executive Officer will step down effective immediately",),
         ("the CEO resigned", "top boss is leaving", "the head of the business quit")),
    Need("cfo_appointed", (EIGHTK_OTHER["5.02"][1],),
         ("Board of Directors appointed a new Chief Financial Officer",),
         ("hired a new finance chief", "named a new CFO", "someone new now runs the finance department")),
    Need("supply_contract", (EIGHTK_OTHER["1.01"][0],),
         ("definitive agreement to supply a large customer under a multi year contract",),
         ("won a big long term deal with a major buyer", "signed a major sales pact", "landed a significant client for several years")),
    Need("credit_facility", (EIGHTK_OTHER["1.01"][1],),
         ("new credit agreement providing a revolving credit facility",),
         ("secured a new bank loan line", "arranged new borrowing capacity", "set up a fresh line of financing with banks")),
    Need("buyback", (EIGHTK_OTHER["8.01"][0],),
         ("press release announcing a new share repurchase authorization",),
         ("plans to buy back its own stock", "returning cash to investors through buybacks", "board approved a stock buyback")),
    Need("new_facility", (EIGHTK_OTHER["8.01"][1],),
         ("announced the opening of a new facility",),
         ("opened a new plant", "expanding with an additional site", "started operating at a new location")),
    Need("impairment", (NEG[2],),
         ("recorded an impairment charge related to goodwill",),
         ("wrote down the value of an acquired business", "took a writedown on intangible assets", "admitted it overpaid for an acquisition")),
    Need("restructuring", (NEG[5],),
         ("restructuring charges increased as we reduced headcount and closed underperforming facilities",),
         ("laid off staff and shut sites", "job cuts and plant closures", "downsizing the workforce")),
)


def template_regex(template: str) -> re.Pattern:
    """A generator sentence with its slots (product, percent, year, amounts, date) turned into wildcards."""
    return re.compile(re.sub(r"\\\{[a-z]\\\}", ".+?", re.escape(template)))


def label_paragraphs(texts: pd.Series, needs: tuple[Need, ...] = NEEDS) -> dict[str, np.ndarray]:
    """need name -> boolean array over paragraphs: does the paragraph contain one of the need's sentences."""
    out = {}
    for need in needs:
        pattern = "|".join(f"(?:{template_regex(t).pattern})" for t in need.templates)
        out[need.name] = texts.str.contains(pattern, regex=True, na=False).to_numpy()
    return out


def build_cases(meta: pd.DataFrame, labels: dict[str, np.ndarray], companies_per_need: int = 12, seed: int = 0,
                needs: tuple[Need, ...] = NEEDS) -> list[dict]:
    """One case per (query, company): the query text, the company it is scoped to and the relevant paragraphs."""
    rng = np.random.default_rng(seed)
    tickers = meta["ticker"].to_numpy()
    cases = []
    for need in needs:
        rel = labels[need.name]
        have = sorted(set(tickers[rel]))
        chosen = have if len(have) <= companies_per_need else sorted(rng.choice(have, companies_per_need, replace=False))
        for ticker in chosen:
            relevant = np.flatnonzero(rel & (tickers == ticker))
            for style, queries in (("lexical", need.lexical), ("paraphrase", need.paraphrase)):
                for q in queries:
                    cases.append({"need": need.name, "style": style, "query": q, "ticker": ticker,
                                  "relevant": relevant, "n_relevant": int(len(relevant))})
    return cases


# figure lookups: (need the paragraph belongs to, pattern that pulls the figures out, query built from them)
EXACT_FAMILIES: tuple[tuple[str, str, str], ...] = (
    ("guidance range", r"full year (\d{4}) revenue guidance (?:to|of) \$(\d+) billion to \$(\d+) billion",
     "{0} revenue guidance ${1} billion to ${2} billion"),
    ("credit facility size", r"revolving credit facility of \$(\d+) million", "revolving credit facility of ${0} million"),
    ("buyback size", r"share repurchase authorization of \$(\d+) million", "share repurchase authorization ${0} million"),
    ("damages awarded", r"awarded damages of \$(\d+) million", "patent verdict damages ${0} million"),
)


def build_exact_cases(meta: pd.DataFrame, per_family: int = 40, seed: int = 0) -> list[dict]:
    """Lookups of a specific figure. Relevant: the company's paragraphs that state exactly those figures."""
    rng = np.random.default_rng(seed)
    texts, tickers = meta["text"].fillna(""), meta["ticker"].to_numpy()
    cases = []
    for name, pattern, query in EXACT_FAMILIES:
        found = texts.str.extract(pattern).to_numpy(dtype=object)
        pos = np.flatnonzero(pd.notna(found).all(axis=1))
        for i in sorted(rng.choice(pos, min(per_family, len(pos)), replace=False)):
            values = tuple(found[i])
            same = (found == found[i]).all(axis=1) & (tickers == tickers[i])
            cases.append({"need": name, "style": "exact", "query": query.format(*values), "ticker": tickers[i],
                          "relevant": np.flatnonzero(same), "n_relevant": int(same.sum())})
    return cases


def rank_metrics(ranked: np.ndarray, relevant: np.ndarray, k: int = 10) -> dict[str, float]:
    """``ranked`` and ``relevant`` are ids of distinct texts; a repeat of a text already listed earns nothing."""
    rel = set(int(i) for i in relevant)
    seen: set[int] = set()
    earned = []
    for i in ranked[:k]:
        earned.append(1.0 if int(i) in rel and int(i) not in seen else 0.0)
        seen.add(int(i))
    gains = np.asarray(earned if earned else [0.0])
    first = int(np.argmax(gains)) + 1 if gains.any() else 0
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    ideal = discount[: min(len(rel), k)].sum()
    return {"hit@1": float(gains[:1].sum() > 0), "hit@5": float(gains[:5].sum() > 0),
            "recall@10": float(gains.sum() / max(min(len(rel), k), 1)),
            "mrr@10": 1.0 / first if first else 0.0,
            "ndcg@10": float((gains * discount[: len(gains)]).sum() / ideal) if ideal else 0.0}


def run_cases(indexes: dict[str, Any], cases: list[dict], groups: np.ndarray, k: int = 10, diversify: bool = False,
              collapse: bool = True, suffix: str = "") -> pd.DataFrame:
    rows = []
    for name, index in indexes.items():
        for i, c in enumerate(cases):
            t0 = time.perf_counter()
            hits = index.search(c["query"], k=k, diversify=diversify, collapse=collapse, ticker=c["ticker"])
            ms = (time.perf_counter() - t0) * 1000
            rows.append({"system": name + suffix, "case": i, "need": c["need"], "style": c["style"], "query": c["query"],
                         "ticker": c["ticker"], "n_relevant": int(len(set(groups[c["relevant"]]))), "ms": ms,
                         **rank_metrics(groups[hits.index.to_numpy()], groups[c["relevant"]], k)})
    return pd.DataFrame(rows)


def _cluster_sums(df: pd.DataFrame, metric: str, queries: list[str]) -> tuple[np.ndarray, np.ndarray]:
    g = df.groupby("query")[metric].agg(["sum", "count"]).reindex(queries)
    return g["sum"].to_numpy(), g["count"].to_numpy()


def bootstrap_mean(df: pd.DataFrame, metric: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean of ``metric`` with a 95% interval from resampling whole queries (each query ran on several companies)."""
    queries = sorted(df["query"].unique())
    s, n = _cluster_sums(df, metric, queries)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(queries), (n_boot, len(queries)))
    boot = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    return {"mean": float(s.sum() / n.sum()), "ci_low": float(np.quantile(boot, 0.025)), "ci_high": float(np.quantile(boot, 0.975))}


def bootstrap_diff(a: pd.DataFrame, b: pd.DataFrame, metric: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired difference a minus b: the same resampled queries score both systems."""
    queries = sorted(a["query"].unique())
    sa, n = _cluster_sums(a, metric, queries)
    sb, _ = _cluster_sums(b, metric, queries)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(queries), (n_boot, len(queries)))
    boot = (sa[idx].sum(axis=1) - sb[idx].sum(axis=1)) / n[idx].sum(axis=1)
    return {"diff": float((sa.sum() - sb.sum()) / n.sum()), "ci_low": float(np.quantile(boot, 0.025)),
            "ci_high": float(np.quantile(boot, 0.975)), "p_le_0": float(np.mean(boot <= 0))}


STYLES = (("lexical", "Keyword style queries (words taken from the filing sentence)"),
          ("paraphrase", "Paraphrased queries (the same need in other words)"),
          ("exact", "Exact figure lookups (generated from the paragraphs)"),
          ("all", "All queries"))


def summarise(rows: pd.DataFrame, served: pd.DataFrame | None = None, uncollapsed: pd.DataFrame | None = None,
              ungated: pd.DataFrame | None = None) -> dict:
    systems = list(dict.fromkeys(rows["system"]))
    out: dict = {"systems": systems, "by_style": {}, "paired": {}, "by_need": {}, "latency_ms": {}}
    for style, _ in STYLES:
        part = rows if style == "all" else rows[rows["style"] == style]
        if part.empty:
            continue
        out["by_style"][style] = {
            s: {"cases": int((part["system"] == s).sum()), "queries": int(part[part["system"] == s]["query"].nunique()),
                **{m: bootstrap_mean(part[part["system"] == s], m) for m in METRICS}} for s in systems}
        if "hybrid" in systems:
            for other in (s for s in systems if s != "hybrid"):
                out["paired"].setdefault(style, {})[f"hybrid minus {other}"] = {
                    m: bootstrap_diff(part[part["system"] == "hybrid"], part[part["system"] == other], m)
                    for m in ("hit@5", "ndcg@10")}
    para = rows[rows["style"] == "paraphrase"]
    out["by_need"] = {need: {s: float(g[g["system"] == s]["ndcg@10"].mean()) for s in systems}
                      for need, g in para.groupby("need")}
    out["latency_ms"] = {s: {"p50": float(rows[rows["system"] == s]["ms"].median()),
                             "p95": float(rows[rows["system"] == s]["ms"].quantile(0.95))} for s in systems}

    def by_style(df: pd.DataFrame, metric: str) -> dict:
        return {s: {style: float(df[(df["system"] == s) & (df["style"] == style)][metric].mean())
                    for style, _ in STYLES[:3] if (df["style"] == style).any()} for s in dict.fromkeys(df["system"])}

    if served is not None:
        out["served"] = by_style(served, "hit@5")
    if uncollapsed is not None:
        out["uncollapsed"] = by_style(uncollapsed, "ndcg@10")
        out["collapsed"] = by_style(rows, "ndcg@10")
    if ungated is not None:
        out["ungated"] = {**by_style(ungated, "ndcg@10"), **by_style(rows[rows["system"].isin(["dense", "hybrid"])], "ndcg@10")}
    return out


def _ci(d: dict) -> str:
    return f"{d['mean']:.3f} [{d['ci_low']:.3f}, {d['ci_high']:.3f}]"


def report_md(summary: dict, info: dict) -> str:
    lines = ["# Retrieval evaluation (synthetic world)", "",
             f"{info['paragraphs']:,} paragraphs ({info['distinct_texts']:,} distinct texts) from {info['companies']} companies; "
             f"{info['queries']} distinct queries, {info['cases']:,} query and company pairs. "
             f"Dense model: `{info['embed_model']}` ({info['dim']} dimensions). Every search is scoped to one company, "
             "as the agent's searches are. Scores count distinct texts. Intervals are 95%, from a bootstrap over whole queries.", ""]
    for style, title in STYLES:
        if style not in summary["by_style"]:
            continue
        lines += [f"## {title}", "", "| System | hit@1 | hit@5 | recall@10 | MRR@10 | nDCG@10 |", "|---|---|---|---|---|---|"]
        for s, m in summary["by_style"][style].items():
            lines.append(f"| {s} | {m['hit@1']['mean']:.3f} | {_ci(m['hit@5'])} | {m['recall@10']['mean']:.3f} | "
                         f"{m['mrr@10']['mean']:.3f} | {_ci(m['ndcg@10'])} |")
        paired = summary["paired"].get(style, {})
        if paired:
            lines += ["", "| Paired difference | hit@5 | nDCG@10 | P(nDCG difference <= 0) |", "|---|---|---|---|"]
            for name, d in paired.items():
                h, n = d["hit@5"], d["ndcg@10"]
                lines.append(f"| {name} | {h['diff']:+.3f} [{h['ci_low']:+.3f}, {h['ci_high']:+.3f}] | "
                             f"{n['diff']:+.3f} [{n['ci_low']:+.3f}, {n['ci_high']:+.3f}] | {n['p_le_0']:.3f} |")
        lines.append("")
    systems = summary["systems"]
    lines += ["## Paraphrased queries by need (nDCG@10)", "", "| Need | " + " | ".join(systems) + " |",
              "|---|" + "---|" * len(systems)]
    for need, d in sorted(summary["by_need"].items()):
        lines.append(f"| {need} | " + " | ".join(f"{d[s]:.3f}" for s in systems) + " |")

    def style_table(title: str, table: dict, other: dict | None = None) -> list[str]:
        styles = list(next(iter(table.values())))
        names = {"lexical": "keyword style", "paraphrase": "paraphrased", "exact": "exact figure"}
        head = "| System | " + " | ".join(names[x] for x in styles) + " |"
        rows_ = [f"| {s} | " + " | ".join(
            f"{d[x]:.3f}" + (f" (was {other[s][x]:.3f})" if other else "") for x in styles) + " |" for s, d in table.items()]
        return ["", f"## {title}", "", head, "|---|" + "---|" * len(styles), *rows_]

    if "uncollapsed" in summary:
        lines += style_table("Effect of collapsing repeated paragraphs (nDCG@10, in brackets without it)",
                             summary["collapsed"], summary["uncollapsed"])
    if "ungated" in summary:
        lines += style_table("Effect of the coverage gate on fusion (nDCG@10)", summary["ungated"])
    if "served" in summary:
        lines += style_table("As the agent serves it (top 6 with the diversity step, hit@5)", summary["served"])
    lines += ["", "## Search time per query on this machine (milliseconds)", "", "| System | p50 | p95 |", "|---|---|---|"]
    lines += [f"| {s} | {d['p50']:.1f} | {d['p95']:.1f} |" for s, d in summary["latency_ms"].items()]
    lines += ["", f"Dense and hybrid times exclude encoding the query, which took a median of {info['query_embed_ms_p50']:.1f} ms "
              f"(p95 {info['query_embed_ms_p95']:.1f} ms) and is cached per query text.", "",
              "Relevance is exact (the generator's own sentences). The paraphrases were written with those sentences in "
              "view, so read the gap between the query sets as the mechanism at work, not as an estimate for real filings."]
    return "\n".join(lines) + "\n"


def evaluate(paragraphs: pd.DataFrame, lake: str, out_dir: str | Path, modes=("bm25", "dense", "hybrid"),
             companies_per_need: int = 12, seed: int = 0, embedder=None, log=print) -> dict:
    from .retrieval import BM25Index, HybridIndex, build_index

    bm25 = BM25Index(paragraphs)
    indexes = {m: build_index(paragraphs, m, lake=lake, embedder=embedder, bm25=bm25, log=log) for m in modes}
    meta = bm25.meta
    labels = label_paragraphs(meta["text"])
    cases = build_cases(meta, labels, companies_per_need, seed) + build_exact_cases(meta, seed=seed)
    dense = next((ix for ix in indexes.values() if hasattr(ix, "embedder")), None) or getattr(indexes.get("hybrid"), "dense", None)
    embed_ms = [0.0]
    if dense is not None:  # encode each distinct query once, timed, so search times below are the search alone
        embed_ms = []
        for q in dict.fromkeys(c["query"] for c in cases):
            t0 = time.perf_counter()
            dense.scores(q, ticker=cases[0]["ticker"])
            embed_ms.append((time.perf_counter() - t0) * 1000)
    rows = run_cases(indexes, cases, bm25.groups, k=10)
    served = run_cases(indexes, cases, bm25.groups, k=6, diversify=True)
    uncollapsed = run_cases(indexes, cases, bm25.groups, k=10, collapse=False)
    ungated = None
    if isinstance(indexes.get("hybrid"), HybridIndex):  # the same fusion with every keyword hit let in
        plain = HybridIndex(bm25, indexes["hybrid"].dense, min_coverage=0.0)
        ungated = run_cases({"hybrid, no coverage gate": plain}, cases, bm25.groups, k=10)
    summary = summarise(rows, served, uncollapsed, ungated)
    info = {"paragraphs": int(len(meta)), "distinct_texts": int(meta["text"].nunique()), "companies": int(meta["ticker"].nunique()),
            "needs": len(NEEDS), "queries": int(rows["query"].nunique()), "cases": len(cases), "seed": seed,
            "cases_by_style": {k: int(v) for k, v in pd.Series([c["style"] for c in cases]).value_counts().items()},
            "companies_per_need": companies_per_need,
            "embed_model": getattr(getattr(dense, "embedder", None), "name", "none"),
            "dim": int(dense.unique.shape[1]) if dense is not None else 0,
            "query_embed_ms_p50": float(np.median(embed_ms)), "query_embed_ms_p95": float(np.quantile(embed_ms, 0.95)),
            "relevant_paragraphs": {n.name: int(labels[n.name].sum()) for n in NEEDS}}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"info": info, **summary}, indent=2))
    (out / "report.md").write_text(report_md(summary, info))
    (out / "queries.json").write_text(json.dumps(
        {"needs": [{"need": n.name, "sentences": list(n.templates), "lexical": list(n.lexical), "paraphrase": list(n.paraphrase)}
                   for n in NEEDS],
         "exact": [{"family": name, "pattern": pattern, "query": query} for name, pattern, query in EXACT_FAMILIES]}, indent=2))
    rows.drop(columns=["ms"]).round(4).to_csv(out / "cases.csv", index=False)
    return {"info": info, **summary}
