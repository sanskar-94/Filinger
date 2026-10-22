"""Statistical tests for small, noisy, autocorrelated predictive signals.

Everything is implemented with numpy and scipy so the formulas are visible and testable:
1. DeLong variance and paired test for AUC (Sun and Xu 2014 fast algorithm).
2. Newey West (Bartlett kernel) HAC t statistic for the mean of a time series (monthly IC, long short).
3. Circular block bootstrap and cluster (by month) bootstrap confidence intervals.
4. Diebold Mariano test with the Harvey, Leybourne and Newbold small sample correction.
5. Permutation test with labels shuffled within time blocks.
6. Benjamini Hochberg false discovery rate control.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------------------------- DeLong AUC
def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    z = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong(y: np.ndarray, preds: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """AUCs and their covariance matrix for one or more score vectors on the same labels."""
    y = np.asarray(y).astype(int)
    p = np.vstack([np.asarray(s, dtype=float) for s in preds])
    pos, neg = p[:, y == 1], p[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = p.shape[0]
    tx = np.vstack([_midrank(pos[i]) for i in range(k)])
    ty = np.vstack([_midrank(neg[i]) for i in range(k)])
    tz = np.vstack([_midrank(np.concatenate([pos[i], neg[i]])) for i in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def auc_ci(y, pred, alpha: float = 0.05) -> dict:
    aucs, cov = delong(y, [pred])
    se = float(np.sqrt(cov[0, 0]))
    z = stats.norm.ppf(1 - alpha / 2)
    auc = float(aucs[0])
    p_vs_half = 2 * (1 - stats.norm.cdf(abs(auc - 0.5) / se)) if se > 0 else float("nan")
    return {"auc": auc, "se": se, "ci_low": auc - z * se, "ci_high": auc + z * se, "p_vs_0.5": p_vs_half}


def delong_test(y, pred_a, pred_b) -> dict:
    aucs, cov = delong(y, [pred_a, pred_b])
    diff = float(aucs[0] - aucs[1])
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = float(np.sqrt(max(var, 1e-18)))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff, "se": se,
            "p_value": float(2 * (1 - stats.norm.cdf(abs(diff) / se)))}


# ------------------------------------------------------------------------------------------- Newey West
def newey_west_lags(n: int) -> int:
    return int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def newey_west(x, lags: int | None = None) -> dict:
    """HAC t statistic for H0: mean(x) = 0 with Bartlett weights."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    if n < 3:
        return {"mean": float(np.mean(x)) if n else float("nan"), "t": float("nan"), "p_value": float("nan"), "n": n}
    lags = newey_west_lags(n) if lags is None else lags
    e = x - x.mean()
    s = e @ e / n
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        s += 2 * w * (e[lag:] @ e[:-lag]) / n
    se = np.sqrt(max(s, 1e-18) / n)
    t = x.mean() / se
    return {"mean": float(x.mean()), "se": float(se), "t": float(t), "p_value": float(2 * (1 - stats.t.cdf(abs(t), n - 1))),
            "n": n, "lags": lags}


# ------------------------------------------------------------------------------------------- bootstraps
def block_bootstrap_ci(x, stat: Callable = np.mean, block: int | None = None, n_boot: int = 2000,
                       alpha: float = 0.05, seed: int = 0) -> dict:
    """Circular block bootstrap CI for a statistic of a time series."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    block = block or max(1, int(round(n ** (1 / 3))))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    idx = (rng.integers(0, n, size=(n_boot, n_blocks))[:, :, None] + np.arange(block)) % n
    samples = x[idx.reshape(n_boot, -1)[:, :n]]
    boots = np.apply_along_axis(stat, 1, samples)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"estimate": float(stat(x)), "ci_low": float(lo), "ci_high": float(hi), "block": block}


def cluster_bootstrap_ci(df: pd.DataFrame, cluster: str, stat: Callable[[pd.DataFrame], float],
                         n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Resample whole clusters (e.g. months) with replacement; robust to within month correlation."""
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby(cluster)}
    keys = list(groups)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        sample = pd.concat([groups[keys[i]] for i in pick], ignore_index=True)
        try:
            boots.append(stat(sample))
        except ValueError:
            continue
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"estimate": float(stat(df)), "ci_low": float(lo), "ci_high": float(hi), "n_boot": len(boots)}


# ----------------------------------------------------------------------------------------- Diebold Mariano
def diebold_mariano(loss_a, loss_b, h: int = 1) -> dict:
    """H0: equal expected loss. Negative statistic means model A has lower loss (is better)."""
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    nw = newey_west(d, lags=max(h - 1, newey_west_lags(n)))
    dm = nw["t"]
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    return {"mean_diff": float(d.mean()), "dm": float(stat), "p_value": float(2 * (1 - stats.t.cdf(abs(stat), n - 1))), "n": n}


# ------------------------------------------------------------------------------------------- permutation
def permutation_test(y: np.ndarray, blocks: np.ndarray, score_fn: Callable[[np.ndarray], float],
                     n_perm: int = 100, seed: int = 0) -> dict:
    """Shuffle labels within blocks (keeps the per block base rate), recompute the score each time."""
    rng = np.random.default_rng(seed)
    observed = score_fn(y)
    y = np.asarray(y)
    blocks = np.asarray(blocks)
    groups = [np.where(blocks == b)[0] for b in np.unique(blocks)]
    null = []
    for _ in range(n_perm):
        yp = y.copy()
        for g in groups:
            yp[g] = y[rng.permutation(g)]
        null.append(score_fn(yp))
    null = np.asarray(null)
    return {"observed": float(observed), "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "p_value": float((1 + (null >= observed).sum()) / (1 + n_perm)), "n_perm": n_perm}


# --------------------------------------------------------------------------------------------- multiple
def benjamini_hochberg(pvals, alpha: float = 0.05) -> pd.DataFrame:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    out = np.empty(n)
    out[order] = adj
    return pd.DataFrame({"p_value": p, "q_value": out, "reject": out <= alpha})


# --------------------------------------------------------------------------------------- finance metrics
def monthly_ic(df: pd.DataFrame, pred: str = "pred", target: str = "car", date: str = "event_date",
               min_n: int = 10) -> pd.Series:
    month = pd.to_datetime(df[date]).dt.to_period("M")
    def one(g):
        if len(g) < min_n or g[pred].nunique() < 2 or g[target].nunique() < 2:
            return np.nan
        return stats.spearmanr(g[pred], g[target])[0]

    ic = df.groupby(month).apply(one, include_groups=False)
    return ic.dropna()


def long_short(df: pd.DataFrame, pred: str = "pred", target: str = "car", date: str = "event_date",
               quantiles: int = 5, min_n: int = 10) -> pd.Series:
    """Monthly mean target of the top prediction quantile minus the bottom quantile."""
    month = pd.to_datetime(df[date]).dt.to_period("M")

    def one(g):
        if len(g) < min_n:
            return np.nan
        q = pd.qcut(g[pred].rank(method="first"), quantiles, labels=False)
        return g.loc[q == quantiles - 1, target].mean() - g.loc[q == 0, target].mean()

    return df.groupby(month).apply(one, include_groups=False).dropna()


@dataclass
class SignalSummary:
    auc: dict
    ic: dict
    long_short: dict
    log_loss: float
    brier: float

    def to_dict(self) -> dict:
        return {"auc": self.auc, "ic": self.ic, "long_short": self.long_short, "log_loss": self.log_loss, "brier": self.brier}
