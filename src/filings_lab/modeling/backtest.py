"""Cost aware event backtest of the out of sample predictions.

Each month, filings in the top prediction quintile are bought and those in the bottom quintile are sold
short, equal weight, entering at the day 0 close and exiting two trading days later, so the return per
filing is the tradable abnormal return CAR[+1,+2]. A long and a short leg each trade in and out, so one
round costs 4 x the per side cost on unit capital. Months are treated as independent bets: the number is
an average monthly spread, not a compounded equity curve.

Reported: mean gross and net spread, annualised Sharpe of the monthly net series, Newey West t statistic,
hit rate, maximum drawdown of the cumulative net spread, and the break even cost per side.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import newey_west

COSTS_BPS = (0, 5, 10, 20, 40)


def monthly_spreads(oof: pd.DataFrame, target: str = "car_tradable", quantiles: int = 5, min_events: int = 10) -> pd.DataFrame:
    df = oof.dropna(subset=["pred", target]).copy()
    df["month"] = pd.to_datetime(df["event_date"]).dt.to_period("M")
    rows = []
    for m, g in df.groupby("month"):
        if len(g) < min_events or g["pred"].nunique() < quantiles:
            continue
        q = pd.qcut(g["pred"].rank(method="first"), quantiles, labels=False)
        long, short = g.loc[q == quantiles - 1, target], g.loc[q == 0, target]
        rows.append({"month": str(m), "n_long": len(long), "n_short": len(short), "gross": long.mean() - short.mean()})
    return pd.DataFrame(rows)


def summarize(monthly: pd.DataFrame, cost_bps: float) -> dict:
    net = monthly["gross"] - 4 * cost_bps / 1e4
    cum = net.cumsum()
    nw = newey_west(net)
    return {
        "months": int(len(net)), "cost_bps_per_side": cost_bps,
        "mean_gross_bps": float(monthly["gross"].mean() * 1e4), "mean_net_bps": float(net.mean() * 1e4),
        "sharpe_annual": float(net.mean() / net.std() * np.sqrt(12)) if net.std() > 0 else float("nan"),
        "nw_t": nw["t"], "hit_rate": float((net > 0).mean()),
        "max_drawdown_bps": float((cum - cum.cummax()).min() * 1e4),
        "break_even_cost_bps": float(monthly["gross"].mean() * 1e4 / 4),
    }


def backtest(oof: pd.DataFrame, cost_bps: float = 10.0, target: str = "car_tradable") -> dict:
    if target not in oof.columns:
        target = "car"
    monthly = monthly_spreads(oof, target)
    if monthly.empty:
        return {"summary": {}, "monthly": monthly, "cost_curve": pd.DataFrame()}
    monthly["net"] = monthly["gross"] - 4 * cost_bps / 1e4
    monthly["cum_net"] = monthly["net"].cumsum()
    curve = pd.DataFrame([summarize(monthly, c) for c in COSTS_BPS])
    return {"summary": {**summarize(monthly, cost_bps), "target": target}, "monthly": monthly, "cost_curve": curve}
