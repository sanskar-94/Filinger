"""Point in time fundamental features from XBRL facts.

Every feature for a filing uses only facts *filed strictly before* that filing's own filing date, so the
filing's own numbers (whose surprise is part of the reaction we are predicting) never leak in as a
feature, and nothing filed later can either.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METRICS = ["revenue", "net_income", "assets", "liabilities", "cash", "op_cash_flow"]


def _yoy_growth(facts: pd.DataFrame) -> pd.DataFrame:
    """Revenue growth versus the same fiscal period one year earlier (period ends 330 to 400 days apart)."""
    rev = facts[facts["metric"] == "revenue"][["cik", "fp", "period_end", "filed", "value"]].sort_values("period_end")
    prev = rev.rename(columns={"period_end": "prev_end", "value": "prev_value"})[["cik", "fp", "prev_end", "prev_value"]]
    m = rev.merge(prev, on=["cik", "fp"])
    lag = (m["period_end"] - m["prev_end"]).dt.days
    m = m[(lag >= 330) & (lag <= 400)]
    m = m.assign(fin_rev_growth=m["value"] / m["prev_value"] - 1)
    return m[["cik", "filed", "fin_rev_growth"]].drop_duplicates(["cik", "filed"], keep="last")


def fundamentals_asof(facts: pd.DataFrame, events: pd.DataFrame, date_col: str = "date_filed") -> pd.DataFrame:
    """Return ``events[["accession"]]`` plus fin_* features known strictly before ``date_col``."""
    if facts is None or facts.empty:
        return events[["accession"]].copy()
    f = facts[facts["metric"].isin(METRICS)].copy()
    f["filed"] = pd.to_datetime(f["filed"])
    # latest known value of each metric at each filing date of the company
    wide = (
        f.sort_values(["filed", "period_end"])
        .pivot_table(index=["cik", "filed"], columns="metric", values="value", aggfunc="last")
        .groupby(level="cik").ffill()
        .reindex(columns=METRICS)  # a company may not report every concept
        .reset_index()
    )
    g = _yoy_growth(f)
    wide = wide.merge(g, on=["cik", "filed"], how="left")
    wide["fin_rev_growth"] = wide.groupby("cik")["fin_rev_growth"].ffill()
    a = wide["assets"]
    out = pd.DataFrame({"cik": wide["cik"], "filed": wide["filed"]})
    out["fin_log_assets"] = np.log(a.clip(lower=1))
    out["fin_leverage"] = wide["liabilities"] / a
    out["fin_roa"] = wide["net_income"] / a
    out["fin_cash_ratio"] = wide["cash"] / a
    out["fin_accruals"] = (wide["net_income"] - wide["op_cash_flow"]) / a
    out["fin_rev_growth"] = wide["fin_rev_growth"]
    ev = events[["accession", "cik", date_col]].copy()
    ev[date_col] = pd.to_datetime(ev[date_col])
    ev = ev.sort_values(date_col)
    out = out.sort_values("filed")
    merged = pd.merge_asof(
        ev, out, left_on=date_col, right_on="filed", by="cik", allow_exact_matches=False, direction="backward"
    )
    return merged.drop(columns=["cik", date_col, "filed"])
