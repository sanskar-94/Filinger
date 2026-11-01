"""Probability calibration: is a predicted 60% chance of a positive reaction right 60% of the time?

1. Reliability table on equal count bins and the expected calibration error (ECE).
2. Murphy decomposition of the Brier score: reliability (lower is better), resolution (higher is better)
   and uncertainty (fixed by the base rate).
3. Out of sample isotonic recalibration along the walk forward folds: fold k is recalibrated with an
   isotonic map fitted only on folds before k, so the recalibrated score is still a fair forecast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def reliability(y, p, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)}).dropna()
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    t = df.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean")).reset_index()
    return t


def ece(table: pd.DataFrame) -> float:
    return float((table["n"] * (table["observed"] - table["mean_pred"]).abs()).sum() / table["n"].sum())


def brier_decomposition(y, p, bins: int = 10) -> dict:
    t = reliability(y, p, bins)
    n = t["n"].sum()
    base = float(np.mean(y))
    rel = float((t["n"] * (t["mean_pred"] - t["observed"]) ** 2).sum() / n)
    res = float((t["n"] * (t["observed"] - base) ** 2).sum() / n)
    unc = base * (1 - base)
    return {"brier": float(np.mean((np.asarray(p) - np.asarray(y)) ** 2)), "reliability": rel, "resolution": res,
            "uncertainty": unc}


def isotonic_walk_forward(oof: pd.DataFrame) -> pd.Series:
    """Recalibrated predictions; the first fold has no history and keeps its raw score."""
    from sklearn.isotonic import IsotonicRegression

    out = oof["pred"].copy()
    folds = sorted(oof["fold"].unique())
    for k in folds[1:]:
        hist = oof[oof["fold"] < k]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(hist["pred"], hist["y"])
        mask = oof["fold"] == k
        out[mask] = iso.predict(oof.loc[mask, "pred"])
    return out


def calibration_report(oof: pd.DataFrame) -> dict:
    later = oof[oof["fold"] > oof["fold"].min()].copy()
    later["pred_iso"] = isotonic_walk_forward(oof)[later.index]
    raw_t, iso_t = reliability(later["y"], later["pred"]), reliability(later["y"], later["pred_iso"])
    return {
        "raw": {"ece": ece(raw_t), **brier_decomposition(later["y"], later["pred"])},
        "isotonic": {"ece": ece(iso_t), **brier_decomposition(later["y"], later["pred_iso"])},
        "table": raw_t.assign(observed_iso=iso_t["observed"], mean_pred_iso=iso_t["mean_pred"]),
    }
