"""A fixed evaluation sample for systems that label by hand or outside the API path.

Without an API key the frontier teacher cannot run through ``flab llm-label``. This script draws a small, fixed
sample of the compact gold set and writes two files:

1. ``sample_gold.jsonl``: the sampled gold items, labels included, for ``flab llm-eval --gold``.
2. ``sample_inputs.md``: the same filings without labels (company, form and the numbered paragraphs only), for
   an annotator, human or model, to label blind.

Labels written blind against ``sample_inputs.md`` (one JSON object per line: ``accession`` and ``signals``) are
then scored next to every other system with the ``replay:<labels.jsonl>`` system spec.

The sample keeps every liquidity and litigation positive (they are rare, about 1% and 3%, so a purely random
draw would leave their F1 undefined) and fills the rest at random with a fixed seed. Compare systems on the same
sample only; its class mix is not the full gold set's.

    python scripts/frontier_sample.py --gold data/llm_dataset_compact/gold.jsonl --out data/frontier_sample
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def draw(rows: list[dict], size: int, seed: int) -> list[dict]:
    rare = [i for i, r in enumerate(rows) if r["label"]["liquidity_concern"] or r["label"]["litigation_risk"]]
    rest = [i for i in range(len(rows)) if i not in set(rare)]
    rng = random.Random(seed)
    picked = rare + rng.sample(rest, max(0, size - len(rare)))
    rng.shuffle(picked)  # the annotator must not be able to tell the rare positives by their position
    return [rows[i] for i in picked]


def inputs_markdown(sample: list[dict]) -> str:
    out = ["# Filings to label (no labels in this file)", ""]
    for n, r in enumerate(sample, start=1):
        out += [f"## {n}. {r['accession']} | {r['form']} | {r.get('company', '')}", ""]
        out += [f"[P{i}] ({p['section']}) {p['text']}" for i, p in enumerate(r["paragraphs"], start=1)]
        out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", default="data/llm_dataset_compact/gold.jsonl")
    ap.add_argument("--limit", type=int, default=400, help="draw from the first N gold items, as the gate does")
    ap.add_argument("--size", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="data/frontier_sample")
    a = ap.parse_args(argv)
    rows = [json.loads(x) for x in Path(a.gold).read_text().splitlines() if x.strip()][: a.limit]
    sample = draw(rows, a.size, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sample_gold.jsonl").write_text("\n".join(json.dumps(r) for r in sample) + "\n")
    (out / "sample_inputs.md").write_text(inputs_markdown(sample))
    print(f"{len(sample)} filings -> {out}/sample_gold.jsonl (with labels) and {out}/sample_inputs.md (without)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
