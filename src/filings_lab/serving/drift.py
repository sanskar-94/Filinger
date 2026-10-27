"""Distribution drift between the training reference and live traffic.

1. Numeric features: Population Stability Index on reference quantile bins (PSI < 0.1 stable, 0.1 to 0.25
   moderate, > 0.25 investigate) and a two sample Kolmogorov Smirnov test.
2. Categorical outputs (for example the share of filings labelled "guidance lowered"): PSI on category shares.
The monitor keeps a sliding window of recent observations and exposes one PSI per feature, which the service
publishes as Prometheus gauges and Grafana alerts on.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-4


def reference_stats(df: pd.DataFrame, numeric: list[str], categorical: list[str] | None = None, bins: int = 10) -> dict:
    ref = {"numeric": {}, "categorical": {}}
    for c in numeric:
        x = pd.to_numeric(df[c], errors="coerce").dropna().to_numpy()
        if len(x) < bins * 5:
            continue
        edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            continue
        counts = np.histogram(np.clip(x, edges[0], edges[-1]), edges)[0]
        ref["numeric"][c] = {"edges": edges.tolist(), "probs": (counts / counts.sum()).tolist(),
                             "sample": np.random.default_rng(0).choice(x, size=min(len(x), 2000), replace=False).tolist()}
    for c in categorical or []:
        shares = df[c].astype(str).value_counts(normalize=True)
        ref["categorical"][c] = shares.to_dict()
    return ref


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, float), EPS, None)
    a = np.clip(np.asarray(actual, float), EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def numeric_psi(ref: dict, values) -> float:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    if len(x) == 0:
        return float("nan")
    edges = np.asarray(ref["edges"])
    counts = np.histogram(np.clip(x, edges[0], edges[-1]), edges)[0]
    return psi(ref["probs"], counts / max(counts.sum(), 1))


def categorical_psi(ref_shares: dict, values) -> float:
    cur = pd.Series(values).astype(str).value_counts(normalize=True)
    cats = sorted(set(ref_shares) | set(cur.index))
    return psi([ref_shares.get(c, 0.0) for c in cats], [cur.get(c, 0.0) for c in cats])


def ks_pvalue(ref: dict, values) -> float:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    if len(x) < 20:
        return float("nan")
    return float(stats.ks_2samp(ref["sample"], x).pvalue)


class DriftMonitor:
    def __init__(self, reference: dict, window: int = 500, min_obs: int = 100):
        self.ref = reference
        self.window = {c: deque(maxlen=window) for c in [*reference.get("numeric", {}), *reference.get("categorical", {})]}
        self.min_obs = min_obs

    def observe(self, row: dict) -> None:
        for c, buf in self.window.items():
            v = row.get(c)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                buf.append(v)

    def report(self) -> dict[str, dict]:
        out = {}
        for c, ref in self.ref.get("numeric", {}).items():
            vals = list(self.window[c])
            if len(vals) >= self.min_obs:
                out[c] = {"psi": numeric_psi(ref, vals), "ks_p": ks_pvalue(ref, vals), "n": len(vals)}
        for c, ref in self.ref.get("categorical", {}).items():
            vals = list(self.window[c])
            if len(vals) >= self.min_obs:
                out[c] = {"psi": categorical_psi(ref, vals), "n": len(vals)}
        return out
