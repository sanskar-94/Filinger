"""Would a cross encoder reranker on top of hybrid retrieval pay for itself? A one off trial.

Takes the paraphrased and keyword style cases of ``flab retrieval-eval``, gathers the top 30 distinct texts
from BM25 and from the dense index, scores each (query, paragraph) pair with a cross encoder and keeps the
best ten. Writes eval/retrieval/rerank_trial.json next to the main report.

    python scripts/rerank_trial.py --lake data/lake

Needs ``pip install fastembed`` and downloads the reranker (about 80 MB) on first use.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from filings_lab.agent.retrieval import BM25Index, build_index, ranked
from filings_lab.agent.retrieval_eval import METRICS, build_cases, label_paragraphs, rank_metrics
from filings_lab.agent.store import DataStore

MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake", default="data/lake")
    ap.add_argument("--out", default="eval/retrieval/rerank_trial.json")
    ap.add_argument("--depth", type=int, default=30)
    a = ap.parse_args()
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    paragraphs = DataStore(a.lake).paragraphs
    bm25 = BM25Index(paragraphs)
    hybrid = build_index(paragraphs, "hybrid", lake=a.lake, bm25=bm25)
    dense, meta = hybrid.dense, bm25.meta
    cases = build_cases(meta, label_paragraphs(meta["text"]), 12, 0)
    reranker = TextCrossEncoder(MODEL)
    rows = []
    for c in cases:
        relevant = bm25.groups[c["relevant"]]
        t0 = time.perf_counter()
        pool: list[int] = []
        for index, floor in ((bm25, 0.0), (dense, -1.0)):
            pool += ranked(index.scores(c["query"], ticker=c["ticker"]), floor, index.recency, index.groups, a.depth).tolist()
        pool = list(dict.fromkeys(pool))
        scores = np.asarray(list(reranker.rerank(c["query"], meta["text"].iloc[pool].tolist())))
        top = np.asarray(pool)[np.argsort(-scores, kind="stable")][:10]
        seconds = time.perf_counter() - t0
        base = {"style": c["style"], "need": c["need"], "seconds": seconds}
        rows.append({"system": "hybrid + reranker", **base, **rank_metrics(bm25.groups[top], relevant)})
        for name, index in (("dense", dense), ("hybrid", hybrid)):
            hits = index.search(c["query"], k=10, diversify=False, ticker=c["ticker"])
            rows.append({"system": name, **base, **rank_metrics(bm25.groups[hits.index.to_numpy()], relevant)})
    df = pd.DataFrame(rows)
    out = {"model": MODEL, "depth": a.depth, "cases": len(cases),
           "by_style": {style: {s: {m: round(float(g[g["system"] == s][m].mean()), 4) for m in METRICS}
                                for s in ("dense", "hybrid", "hybrid + reranker")} for style, g in df.groupby("style")},
           "rerank_seconds_per_query": {"p50": round(float(df[df["system"] == "hybrid + reranker"]["seconds"].median()), 3),
                                        "p95": round(float(df[df["system"] == "hybrid + reranker"]["seconds"].quantile(0.95)), 3)}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
