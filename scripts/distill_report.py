"""The distillation result, with the training seed treated as the noise it is.

    python scripts/distill_report.py --arm teacher=eval/distill/teacher_seed0,eval/distill/teacher_seed1 \
        --arm generator=eval/distill/generator,eval/distill/generator_seed1 --out eval/distill/report.md

Each folder is a ``flab llm-eval`` run holding one student and the rules on the same gold filings. Two students
that differ only in their training seed scored 0.899 and 0.807 here, so one run per arm says little: the report
gives every run, each arm's mean and spread over seeds, and the difference of the arm means two ways, across
seeds (Welch's t, which carries the seed noise) and over filings (a paired bootstrap that holds the seeds fixed).
With ``--data`` (the folder ``scripts/distill_datasets.py`` wrote) it also compares the labels the two arms learned
from, field by field, which says where a difference between the students can come from at all.
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from filings_lab.llm.eval_harness import LABEL_FIELDS, field_scores

METRIC_NAMES = {"extraction_score": "Extraction score", "guidance_direction_f1": "Guidance F1",
                "management_tone_f1": "Tone F1", "liquidity_concern_f1": "Liquidity F1", "litigation_risk_f1": "Litigation F1"}


def _student(spec: str) -> pd.DataFrame:
    """``folder`` holding one student beside the rules, or ``folder@system`` to pick one of several."""
    folder, _, system = spec.partition("@")
    items = pd.read_parquet(Path(folder) / "items.parquet")
    names = [system] if system else [s for s in items["system"].unique() if s != "rules"]
    if len(names) != 1 or names[0] not in set(items["system"]):
        raise SystemExit(f"{folder}: expected one student beside the rules, found {sorted(set(items['system']))}")
    return _booleans(items[items["system"] == names[0]].sort_values("accession").reset_index(drop=True))


def _booleans(t: pd.DataFrame) -> pd.DataFrame:
    """items.parquet stores the two yes or no fields as text; the scorer compares them with True and False."""
    t = t.copy()
    for f in ("liquidity_concern", "litigation_risk"):
        for side in ("gold", "pred"):
            t[f"{side}_{f}"] = t[f"{side}_{f}"].map({"True": True, "False": False, True: True, False: False}).astype(object)
    return t


def _final_val_loss(folder: str) -> str:
    """The last validation loss in the run's train.log, on its own arm's labels (so not comparable across arms)."""
    log = Path(folder.partition("@")[0]) / "train.log"
    losses = re.findall(r"^\s*\d+\s+val\s+([0-9.]+)", log.read_text(), re.M) if log.exists() else []
    return losses[-1] if losses else "n/a"


def _answers(folder: Path) -> dict[str, dict]:
    out = {}
    for split in ("train", "val"):
        f = folder / f"{split}.jsonl"
        for line in f.read_text().splitlines() if f.exists() else []:
            if line.strip():
                r = json.loads(line)
                out[r["accession"]] = json.loads(r["completion"][-1]["content"])
    return out


def _label_agreement(data: Path) -> tuple[list[str], dict]:
    """Where the teacher's labels and the generator's differ, field by field, on the filings both arms trained on."""
    teacher, generator = _answers(data / "teacher"), _answers(data / "generator")
    shared = sorted(set(teacher) & set(generator))
    if not shared:
        return [], {}

    def counts(side: dict[str, dict], f: str) -> str:
        c = Counter(str(side[a].get(f)) for a in shared)
        return ", ".join(f"{k} {v}" for k, v in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))

    lines = ["", "## The labels each arm learned from", "",
             f"{len(shared)} filings in both training sets (train and val). Same label: the share of filings where "
             "the teacher gave the generator's label.", "",
             "| Field | Same label | Teacher's labels | Generator's labels |", "|---|---:|---|---|"]
    same: dict[str, float] = {}
    for f in LABEL_FIELDS:
        same[f] = float(np.mean([teacher[a].get(f) == generator[a].get(f) for a in shared]))
        lines.append(f"| {f} | {same[f]:.1%} | {counts(teacher, f)} | {counts(generator, f)} |")
    tone = pd.crosstab(pd.Series([str(generator[a].get("management_tone")) for a in shared], name="generator"),
                       pd.Series([str(teacher[a].get("management_tone")) for a in shared], name="teacher"))
    lines += ["", "Management tone, the generator's label down the side and the teacher's across:", "",
              "| generator \\ teacher | " + " | ".join(map(str, tone.columns)) + " |",
              "|---|" + "---:|" * len(tone.columns)]
    for g, row in tone.iterrows():
        lines.append(f"| {g} | " + " | ".join(str(int(v)) for v in row) + " |")
    table = {str(g): {str(t): int(v) for t, v in row.items()} for g, row in tone.iterrows()}
    return lines, {"filings": len(shared), "same_label": same, "tone_generator_by_teacher": table}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", action="append", required=True, help="name=folder,folder,... (one per arm)")
    p.add_argument("--rules", help="a folder whose rules rows to show as the baseline (default: the first run's)")
    p.add_argument("--out", default="eval/distill/report.md")
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--data", help="the datasets folder of scripts/distill_datasets.py, to compare the arms' labels")
    a = p.parse_args(argv)

    arms: dict[str, list[tuple[str, pd.DataFrame]]] = {}
    for spec in a.arm:
        name, _, folders = spec.partition("=")
        runs = [(f, _student(f)) for f in folders.split(",") if f]
        if runs:  # an arm with no finished run yet is left out
            arms[name] = runs
    if not arms:
        raise SystemExit("no finished runs")
    first = next(iter(arms.values()))[0][0]
    rules_items = pd.read_parquet(Path((a.rules or first).partition("@")[0]) / "items.parquet")
    rules = _booleans(rules_items[rules_items["system"] == "rules"].sort_values("accession").reset_index(drop=True))
    accessions = rules["accession"].tolist()
    for runs in arms.values():
        for folder, t in runs:
            if t["accession"].tolist() != accessions:
                raise SystemExit(f"{folder}: not the same gold filings as the rules run")

    lines = ["# Distillation: students trained on a teacher's labels against students trained on the generator's", "",
             f"{len(accessions)} gold filings, scored against the generator's truth. One row per training run.", "",
             "| Arm | Run | Final val loss | Schema valid | Extraction score | Guidance F1 | Tone F1 | Liquidity F1 "
             "| Litigation F1 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    rs = field_scores(rules)
    lines.append(f"| rules | production | | {rules['schema_valid'].mean():.1%} | {rs['extraction_score']:.4f} | "
                 + " | ".join(f"{rs[f'{f}_f1']:.4f}" for f in LABEL_FIELDS) + " |")
    metrics = ["extraction_score"] + [f"{f}_f1" for f in LABEL_FIELDS]
    per_run: dict[str, list[dict]] = {}
    for name, runs in arms.items():
        for folder, t in runs:
            s = field_scores(t)
            per_run.setdefault(name, []).append(s)
            lines.append(f"| {name} | `{folder}` | {_final_val_loss(folder)} | {t['schema_valid'].mean():.1%} | "
                         f"{s['extraction_score']:.4f} | "
                         + " | ".join(f"{s[f'{f}_f1']:.4f}" for f in LABEL_FIELDS) + " |")
    lines += ["", "Final val loss is the last validation loss of training, on the run's own labels: the arms learned "
              "different answers, so it compares seeds within an arm, not the arms."]

    lines += ["", "Each arm's mean, with the spread over its seeds (sd) in brackets:", "",
              "| Arm | Runs | " + " | ".join(METRIC_NAMES[m] for m in metrics) + " |", "|---|---:|" + "---:|" * len(metrics)]
    for name, runs in per_run.items():
        cells = []
        for m in metrics:
            x = np.asarray([r[m] for r in runs])
            cells.append(f"{x.mean():.4f} ({x.std(ddof=1):.4f})" if len(x) > 1 else f"{x.mean():.4f}")
        lines.append(f"| {name} | {len(runs)} | " + " | ".join(cells) + " |")

    names = list(arms)
    summary: dict = {"runs": per_run}
    if len(names) == 2:
        from scipy import stats

        lines += ["", f"{names[0]} minus {names[1]}, by measure: the difference of the arm means; Welch's t across seeds, "
                  "which carries the seed noise; and a 95% interval over filings from a paired bootstrap that holds the "
                  "seeds fixed.", "", "| Measure | Difference | Welch t | p | 95% CI over filings |", "|---|---:|---:|---:|---|"]
        rng = np.random.default_rng(0)
        draws = [rng.integers(0, len(accessions), len(accessions)) for _ in range(a.boot)]
        boot = {m: [] for m in metrics}
        for idx in draws:
            means = []
            for n in names:
                scored = [field_scores(t.iloc[idx]) for _, t in arms[n]]
                means.append({m: np.nanmean([sc[m] for sc in scored]) for m in metrics})
            for m in metrics:
                boot[m].append(means[0][m] - means[1][m])
        summary["differences"] = {}
        for m in metrics:
            x, y = (np.asarray([r[m] for r in per_run[n]]) for n in names)
            diff = float(x.mean() - y.mean())
            lo, hi = np.nanquantile(boot[m], [0.025, 0.975])
            tt = pp = "n/a"  # one run in an arm, or every run of both arms on the same score: no spread to test
            if len(x) > 1 and len(y) > 1 and (x.std() > 0 or y.std() > 0):
                with warnings.catch_warnings():  # scipy warns when one arm's runs all score the same
                    warnings.simplefilter("ignore", RuntimeWarning)
                    t = stats.ttest_ind(x, y, equal_var=False)
                tt, pp = f"{t.statistic:.2f}", f"{t.pvalue:.3f}"
            lines.append(f"| {METRIC_NAMES[m]} | {diff:+.4f} | {tt} | {pp} | [{lo:+.4f}, {hi:+.4f}] |")
            summary["differences"][m] = {"diff": diff, "ci_filings": [float(lo), float(hi)], "welch_t": tt, "welch_p": pp}
    if a.data:
        label_lines, summary["labels"] = _label_agreement(Path(a.data))
        lines += label_lines
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n")
    Path(a.out).with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
