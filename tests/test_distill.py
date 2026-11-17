"""The distillation scripts: the two training sets differ only in their answers, and the report reads the runs right."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from filings_lab.llm.eval_harness import field_scores

ROOT = Path(__file__).parents[1]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _signals(tone: str, liquidity: bool = False) -> dict:
    return {"guidance_direction": "none", "management_tone": tone, "liquidity_concern": liquidity,
            "litigation_risk": False, "key_quotes": [], "summary": "s"}


def _record(accession: str, tone: str) -> dict:
    return {"accession": accession, "form": "8-K", "prompt": [{"role": "user", "content": f"filing {accession}"}],
            "completion": [{"role": "assistant", "content": json.dumps(_signals(tone))}]}


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def test_datasets_take_the_first_labelled_filings_and_change_only_the_answers(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _jsonl(src / "train.jsonl", [_record(f"a{i}", "neutral") for i in range(4)])
    _jsonl(src / "val.jsonl", [_record("v0", "neutral")])
    # labelling order v0, a2, a0, a1: with --n 3 the last is left out, and a0's answer failed the schema
    _jsonl(tmp_path / "labels.jsonl", [
        {"accession": "v0", "schema_valid": True, "signals": _signals("positive")},
        {"accession": "a2", "schema_valid": True, "signals": _signals("negative")},
        {"accession": "a0", "schema_valid": False, "signals": None},
        {"accession": "a1", "schema_valid": True, "signals": _signals("positive")}])
    _script("distill_datasets").main(["--labels", str(tmp_path / "labels.jsonl"), "--n", "3", "--source", str(src),
                                      "--out", str(out)])

    control, taught = _read(out / "generator" / "train.jsonl"), _read(out / "teacher" / "train.jsonl")
    assert [r["accession"] for r in control] == ["a0", "a2"]
    assert [r["accession"] for r in taught] == ["a2"]
    assert taught[0]["prompt"] == control[1]["prompt"]
    assert json.loads(taught[0]["completion"][0]["content"])["management_tone"] == "negative"
    assert json.loads(control[1]["completion"][0]["content"])["management_tone"] == "neutral"
    assert [r["accession"] for r in _read(out / "teacher" / "val.jsonl")] == ["v0"]
    stats = json.loads((out / "stats.json").read_text())
    assert stats["filings"] == 3 and stats["teacher_answers_outside_the_schema"] == 1
    assert stats["teacher"] == {"train": 1, "val": 1} and stats["generator"] == {"train": 2, "val": 1}


GOLD_TONE = ["positive", "neutral", "negative", "positive", "neutral", "negative"]
GOLD_LIQUIDITY = [True, False, False, False, False, False]


def _run(folder: Path, student: str, tones: list[str], liquidity: list[bool]) -> pd.DataFrame:
    """An llm-eval folder as the report reads it: the rules and one student, the yes or no fields stored as text."""
    rows = []
    for system, pred_tone, pred_liq in (("rules", GOLD_TONE, GOLD_LIQUIDITY), (student, tones, liquidity)):
        for i in range(len(GOLD_TONE)):
            rows.append({"accession": f"x{i}", "system": system, "schema_valid": True,
                         "gold_guidance_direction": "none", "pred_guidance_direction": "none",
                         "gold_management_tone": GOLD_TONE[i], "pred_management_tone": pred_tone[i],
                         "gold_liquidity_concern": str(GOLD_LIQUIDITY[i]), "pred_liquidity_concern": str(pred_liq[i]),
                         "gold_litigation_risk": "False", "pred_litigation_risk": "False"})
    folder.mkdir(parents=True)
    t = pd.DataFrame(rows)
    t.to_parquet(folder / "items.parquet")
    return t


def test_report_compares_the_arms_over_seeds_and_filings(tmp_path):
    report = _script("distill_report")
    never = [False] * 6
    runs = {"t0": (GOLD_TONE, GOLD_LIQUIDITY), "t1": (GOLD_TONE[:5] + ["positive"], GOLD_LIQUIDITY),
            "g0": (["neutral"] * 6, GOLD_LIQUIDITY), "g1": (["neutral"] * 6, never)}
    for name, (tones, liq) in runs.items():
        _run(tmp_path / name, f"student-{name}", tones, liq)
    (tmp_path / "t0" / "train.log").write_text("     1    val 2.061    28.40s\n  1000    val 0.014    28.74s\n")
    data = tmp_path / "data"
    _jsonl(data / "teacher" / "train.jsonl", [_record("a", "positive"), _record("b", "neutral")])
    _jsonl(data / "generator" / "train.jsonl", [_record("a", "neutral"), _record("b", "neutral")])

    out = tmp_path / "report.md"
    report.main(["--arm", f"teacher={tmp_path / 't0'},{tmp_path / 't1'}",
                 "--arm", f"generator={tmp_path / 'g0'},{tmp_path / 'g1'}",
                 "--data", str(data), "--boot", "50", "--out", str(out)])

    summary = json.loads(out.with_suffix(".json").read_text())

    def score(name: str, measure: str) -> float:
        t = report._booleans(pd.read_parquet(tmp_path / name / "items.parquet").query("system != 'rules'"))
        return field_scores(t)[measure]

    for measure in ("management_tone_f1", "liquidity_concern_f1", "extraction_score"):
        want = (score("t0", measure) + score("t1", measure) - score("g0", measure) - score("g1", measure)) / 2
        assert abs(summary["differences"][measure]["diff"] - want) < 1e-9
    assert score("t0", "liquidity_concern_f1") == 1.0  # the yes or no fields read back as booleans, not text
    lo, hi = summary["differences"]["management_tone_f1"]["ci_filings"]
    assert lo <= hi and summary["differences"]["management_tone_f1"]["welch_t"] != "n/a"
    assert summary["labels"]["filings"] == 2 and summary["labels"]["same_label"]["management_tone"] == 0.5
    assert summary["labels"]["tone_generator_by_teacher"] == {"neutral": {"neutral": 1, "positive": 1}}
    text = out.read_text()
    assert "| teacher | `" + str(tmp_path / "t0") + "` | 0.014 |" in text
    assert "## The labels each arm learned from" in text
