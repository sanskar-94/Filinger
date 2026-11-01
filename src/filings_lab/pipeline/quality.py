"""Data quality gates for the lakehouse tables.

Each check returns pass, warn or fail with the observed value and its threshold. ``flab dq`` exits non zero
on any failure, so the Databricks job, the model gate workflow and a nightly run all stop before a model is
trained on broken data. Thresholds are deliberately explicit in one table (``THRESHOLDS``) so a change to
them is a reviewed code change, not a silent drift.

The leakage canary deserves a note: filing text effects are small, so no single feature should correlate
strongly with the label. A feature whose rank correlation with CAR exceeds the threshold is far more likely
to be a timing bug (information from after day 0) than a discovery.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLDS = {
    "index_null_rate": 0.0,             # cik, form, accession, date_filed must always be present
    "accepted_parse_rate": 0.98,        # acceptance timestamps that parse
    "periodic_mdna_rate": 0.90,         # 10-K and 10-Q filings with an extracted MD&A section
    "tenk_risk_rate": 0.80,             # 10-K filings with an extracted risk factors section
    "earnings_section_rate": 0.90,      # 8-K item 2.02 filings with an extracted results section
    "event_coverage": 0.60,             # filings that end up as labelled events
    "car_abs_max": 1.0,                 # a 3 day abnormal return above 100% is a data error
    "label_balance": (0.35, 0.65),      # share of positive labels
    "price_abs_return_max": 0.75,       # daily moves above 75% are almost always split errors
    "feature_null_rate_warn": 0.60,     # warn when a feature is mostly missing
    "leakage_abs_ic": 0.30,             # any single feature this predictive is a timing bug until proven otherwise
}


@dataclass
class Check:
    name: str
    table: str
    status: str  # pass, warn, fail
    observed: float | str
    threshold: float | str
    detail: str = ""


def _rate(ok: pd.Series) -> float:
    return float(ok.mean()) if len(ok) else 0.0


def _ge(name, table, observed, threshold, detail="", warn_only=False) -> Check:
    ok = observed >= threshold
    return Check(name, table, "pass" if ok else ("warn" if warn_only else "fail"), round(observed, 6), threshold, detail)


def check_index(idx: pd.DataFrame) -> list[Check]:
    out = []
    req = ["cik", "form", "accession", "date_filed"]
    null_rate = float(idx[req].isna().any(axis=1).mean()) if len(idx) else 1.0
    out.append(Check("required_fields_present", "bronze_index", "pass" if null_rate <= THRESHOLDS["index_null_rate"] else "fail",
                     round(null_rate, 6), THRESHOLDS["index_null_rate"], "share of rows missing cik, form, accession or date"))
    dup = int(idx["accession"].duplicated().sum())
    out.append(Check("accession_unique", "bronze_index", "pass" if dup == 0 else "fail", dup, 0, "duplicate accession numbers"))
    parsed = pd.to_datetime(idx["accepted_at"], utc=True, errors="coerce").notna()
    out.append(_ge("accepted_at_parses", "bronze_index", _rate(parsed), THRESHOLDS["accepted_parse_rate"]))
    return out


def check_sections(idx: pd.DataFrame, sections: pd.DataFrame) -> list[Check]:
    out = []
    have = sections.groupby("accession")["section"].apply(set)

    def rate(mask, section):
        accs = idx.loc[mask, "accession"]
        return _rate(accs.map(lambda a: section in have.get(a, set())))

    periodic = idx["form"].str.replace("/A", "").isin(["10-K", "10-Q"])
    out.append(_ge("periodic_mdna_extracted", "silver_sections", rate(periodic, "mdna"), THRESHOLDS["periodic_mdna_rate"]))
    tenk = idx["form"].str.replace("/A", "") == "10-K"
    if tenk.any():
        out.append(_ge("tenk_risk_factors_extracted", "silver_sections", rate(tenk, "risk_factors"), THRESHOLDS["tenk_risk_rate"]))
    earnings = idx["form"].str.startswith("8-K") & idx["items"].fillna("").astype(str).str.contains("2.02", regex=False)
    if earnings.any():
        out.append(_ge("earnings_results_extracted", "silver_sections", rate(earnings, "8k_results_of_operations"),
                       THRESHOLDS["earnings_section_rate"]))
    empty = int((sections["n_words"] <= 0).sum())
    out.append(Check("no_empty_sections", "silver_sections", "pass" if empty == 0 else "warn", empty, 0))
    return out


def check_paragraphs(paragraphs: pd.DataFrame) -> list[Check]:
    dup = int(paragraphs["paragraph_id"].duplicated().sum())
    blank = int((paragraphs["text"].fillna("").str.strip() == "").sum())
    return [Check("paragraph_id_unique", "silver_paragraphs", "pass" if dup == 0 else "fail", dup, 0, "citations depend on it"),
            Check("no_blank_paragraphs", "silver_paragraphs", "pass" if blank == 0 else "fail", blank, 0)]


def check_prices(prices: pd.DataFrame) -> list[Check]:
    out = []
    dup = int(prices.duplicated(["ticker", "date"]).sum())
    out.append(Check("one_price_per_day", "silver_prices", "pass" if dup == 0 else "fail", dup, 0))
    nonpos = int((prices["close"] <= 0).sum())
    out.append(Check("positive_closes", "silver_prices", "pass" if nonpos == 0 else "fail", nonpos, 0))
    p = prices.sort_values(["ticker", "date"])
    r = p.groupby("ticker")["close"].pct_change().abs()
    big = int((r > THRESHOLDS["price_abs_return_max"]).sum())
    out.append(Check("no_extreme_daily_moves", "silver_prices", "pass" if big == 0 else "warn", big,
                     THRESHOLDS["price_abs_return_max"], "unadjusted splits show up here"))
    return out


def check_events(idx: pd.DataFrame, events: pd.DataFrame) -> list[Check]:
    out = []
    out.append(_ge("event_coverage", "gold_events", len(events) / max(len(idx), 1), THRESHOLDS["event_coverage"],
                   "first year has no estimation window, so coverage below 100% is expected"))
    finite = np.isfinite(events["car"].astype(float))
    out.append(Check("car_finite", "gold_events", "pass" if finite.all() else "fail", int((~finite).sum()), 0))
    too_big = int((events["car"].abs() > THRESHOLDS["car_abs_max"]).sum())
    out.append(Check("car_plausible", "gold_events", "pass" if too_big == 0 else "fail", too_big, THRESHOLDS["car_abs_max"]))
    acc = pd.to_datetime(events["accepted_at"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    early = int((pd.to_datetime(events["event_date"]) < acc).sum())
    out.append(Check("event_not_before_acceptance", "gold_events", "pass" if early == 0 else "fail", early, 0,
                     "a reaction dated before the filing was public is look ahead"))
    lo, hi = THRESHOLDS["label_balance"]
    bal = float(events["label_up"].astype(float).mean())
    out.append(Check("label_balance", "gold_events", "pass" if lo <= bal <= hi else "warn", round(bal, 4), f"{lo} to {hi}"))
    return out


def check_features(features: pd.DataFrame) -> list[Check]:
    from scipy import stats

    from ..features.build import feature_groups

    out = []
    dup = int(features["accession"].duplicated().sum())
    out.append(Check("one_row_per_filing", "gold_features", "pass" if dup == 0 else "fail", dup, 0))
    cols = [c for g, cs in feature_groups(features.columns).items() if g != "meta" for c in cs]
    X = features[cols].apply(pd.to_numeric, errors="coerce")
    inf = int(np.isinf(X.to_numpy(dtype=float)).sum())
    out.append(Check("no_infinite_values", "gold_features", "pass" if inf == 0 else "fail", inf, 0))
    order = (pd.to_datetime(features["t1"]) > pd.to_datetime(features["event_date"])).all()
    out.append(Check("label_window_after_event", "gold_features", "pass" if order else "fail", str(bool(order)), "True"))
    mostly_null = [c for c in cols if X[c].isna().mean() > THRESHOLDS["feature_null_rate_warn"]]
    out.append(Check("feature_null_rates", "gold_features", "pass" if not mostly_null else "warn", len(mostly_null),
                     THRESHOLDS["feature_null_rate_warn"], ", ".join(mostly_null[:8])))
    # judge the lower 95% bound of |IC| (Fisher z), so small samples do not raise false alarms
    worst, worst_ic, worst_lo = None, 0.0, 0.0
    for c in cols:
        x = X[c]
        ok = x.notna() & features["car"].notna()
        n = int(ok.sum())
        if n < 50 or x[ok].nunique() < 3:
            continue
        ic = abs(stats.spearmanr(x[ok], features.loc[ok, "car"])[0])
        lo = float(np.tanh(np.arctanh(min(ic, 0.999999)) - 1.96 / np.sqrt(n - 3)))
        if lo > worst_lo or worst is None:
            worst, worst_ic, worst_lo = c, ic, lo
    out.append(Check("leakage_canary", "gold_features", "pass" if worst_lo <= THRESHOLDS["leakage_abs_ic"] else "fail",
                     round(worst_lo, 4), THRESHOLDS["leakage_abs_ic"],
                     f"lower 95% bound of |IC|; strongest feature {worst} with |IC| {worst_ic:.3f}"))
    return out


def run_checks(lake: str) -> list[Check]:
    lake = lake.rstrip("/")
    idx = pd.read_parquet(f"{lake}/bronze_index.parquet")
    checks = check_index(idx)
    secs = f"{lake}/silver/sections"
    if Path(secs).exists():
        checks += check_sections(idx, pd.read_parquet(secs))
        checks += check_paragraphs(pd.read_parquet(f"{lake}/silver/paragraphs"))
    if Path(f"{lake}/silver/prices.parquet").exists():
        checks += check_prices(pd.read_parquet(f"{lake}/silver/prices.parquet"))
    if Path(f"{lake}/gold/events").exists():
        checks += check_events(idx, pd.read_parquet(f"{lake}/gold/events"))
    if Path(f"{lake}/gold/features").exists():
        checks += check_features(pd.read_parquet(f"{lake}/gold/features"))
    return checks


def report(checks: list[Check]) -> str:
    rows = ["| status | table | check | observed | threshold | detail |", "|---|---|---|---|---|---|"]
    for c in sorted(checks, key=lambda c: {"fail": 0, "warn": 1, "pass": 2}[c.status]):
        rows.append(f"| {c.status.upper()} | {c.table} | {c.name} | {c.observed} | {c.threshold} | {c.detail} |")
    n = {s: sum(c.status == s for c in checks) for s in ("pass", "warn", "fail")}
    return f"# Data quality\n\n{n['pass']} passed, {n['warn']} warnings, {n['fail']} failed.\n\n" + "\n".join(rows) + "\n"


def save(checks: list[Check], out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dq.json").write_text(json.dumps([asdict(c) for c in checks], indent=2, default=str))
    (out / "dq.md").write_text(report(checks))
