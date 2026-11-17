"""The two datasets of the distillation experiment, built so the students differ only in who wrote the labels.

    python scripts/distill_datasets.py --labels data/teacher/nemotron-3-ultra.jsonl --n 1000 \
        --source data/llm_dataset_compact --out data/distill

Takes the first N filings a teacher labelled (``flab llm-label`` labels in a fixed shuffled order, so these are a
random sample of the student's train and val splits) and writes, under ``--out``:

* ``teacher/``: those filings with the teacher's labels as the answers to learn;
* ``generator/``: the same filings, the same prompts, with the synthetic generator's labels (the matched control).

The control can be built before the labels are in, from the labelling order (``--which generator --accessions``),
so it trains while the teacher is still labelling. A filing whose teacher answer then fails the schema is counted
in ``stats.json``. Neither dataset has a test split: both students are scored on the source dataset's gold
filings, against the generator's truth, which no teacher saw.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _records(source: Path, split: str) -> list[dict]:
    return [json.loads(x) for x in (source / f"{split}.jsonl").read_text().splitlines() if x.strip()]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", help="a teacher's labels from flab llm-label, in labelling order")
    p.add_argument("--accessions", help="the labelling order, one accession per line (to build the control early)")
    p.add_argument("--which", choices=["both", "teacher", "generator"], default="both")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--source", default="data/llm_dataset_compact")
    p.add_argument("--out", default="data/distill")
    a = p.parse_args(argv)

    rows = [json.loads(x) for x in Path(a.labels).read_text().splitlines() if x.strip()][: a.n] if a.labels else []
    if a.accessions:
        order = [x.strip() for x in Path(a.accessions).read_text().splitlines() if x.strip()][: a.n]
    else:
        order = [r["accession"] for r in rows]
    if a.which != "generator" and len(rows) < a.n:
        raise SystemExit(f"only {len(rows)} labels so far; the teacher set needs {a.n}")
    teacher = {r["accession"]: r["signals"] for r in rows if r.get("schema_valid") and r.get("signals")}
    chosen = set(order)
    stats_path = Path(a.out) / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    for split in ("train", "val"):
        records = [r for r in _records(Path(a.source), split) if r["accession"] in chosen]
        for name in ("teacher", "generator"):
            if a.which not in ("both", name):
                continue
            out = Path(a.out) / name
            out.mkdir(parents=True, exist_ok=True)
            kept = [r for r in records if name == "generator" or r["accession"] in teacher]
            with open(out / f"{split}.jsonl", "w") as fh:
                for r in kept:
                    if name == "teacher":
                        answer = json.dumps(teacher[r["accession"]], separators=(",", ":"))
                        r = {**r, "completion": [{"role": "assistant", "content": answer}]}
                    fh.write(json.dumps(r) + "\n")
            stats.setdefault(name, {})[split] = len(kept)
    stats["filings"] = len(order)
    if rows:
        stats["teacher_answers_outside_the_schema"] = sum(1 for r in rows if r["accession"] in chosen and r["accession"] not in teacher)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
