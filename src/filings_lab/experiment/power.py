"""Power analysis for the A/B test, done before the test starts.

The primary metric is a proportion (thumbs up rate among rated answers). Requests from the same user are
correlated, so the per arm sample size from the textbook formula is inflated by the design effect
1 + (m - 1) * ICC, where m is the mean number of rated answers per user.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from scipy import stats


def _z(q: float) -> float:
    return float(stats.norm.ppf(q))


def n_per_arm(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8, ratio: float = 1.0) -> int:
    """Two sided two proportion z test, ``ratio`` = n2 / n1. Returns n1 (n2 = ratio * n1)."""
    if p1 == p2:
        raise ValueError("p1 and p2 must differ")
    pbar = (p1 + ratio * p2) / (1 + ratio)
    za, zb = _z(1 - alpha / 2), _z(power)
    num = (za * math.sqrt(pbar * (1 - pbar) * (1 + 1 / ratio)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / ratio)) ** 2
    return math.ceil(num / (p2 - p1) ** 2)


def achieved_power(n1: int, p1: float, p2: float, alpha: float = 0.05, ratio: float = 1.0) -> float:
    n2 = n1 * ratio
    pbar = (p1 * n1 + p2 * n2) / (n1 + n2)
    se0 = math.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n2))
    se1 = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    za = _z(1 - alpha / 2)
    d = abs(p2 - p1)
    return float(stats.norm.cdf((d - za * se0) / se1) + stats.norm.cdf((-d - za * se0) / se1))


def mde(n1: int, p1: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest absolute lift detectable with ``n1`` per arm (bisection on achieved power)."""
    lo, hi = 1e-5, min(1 - p1, 0.5) - 1e-6
    for _ in range(80):
        mid = (lo + hi) / 2
        if achieved_power(n1, p1, p1 + mid, alpha) < power:
            lo = mid
        else:
            hi = mid
    return hi


def n_continuous(sd: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Per arm n for a difference in means (for example latency), normal approximation."""
    return math.ceil(2 * (_z(1 - alpha / 2) + _z(power)) ** 2 * sd**2 / delta**2)


@dataclass
class Plan:
    baseline: float
    mde_abs: float
    alpha: float
    power: float
    rated_per_user: float
    icc: float
    rating_rate: float
    n_rated_per_arm_iid: int
    design_effect: float
    n_rated_per_arm: int
    n_exposures_per_arm: int
    n_users_per_arm: int
    days_at_traffic: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def plan(baseline: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8, rated_per_user: float = 1.0,
         icc: float = 0.0, rating_rate: float = 1.0, daily_users: float | None = None) -> Plan:
    n_iid = n_per_arm(baseline, baseline + mde_abs, alpha, power)
    deff = 1 + max(rated_per_user - 1, 0) * icc
    n_rated = math.ceil(n_iid * deff)
    n_exp = math.ceil(n_rated / rating_rate)
    n_users = math.ceil(n_rated / max(rated_per_user, 1e-9))
    days = (2 * n_users / daily_users) if daily_users else None
    return Plan(baseline, mde_abs, alpha, power, rated_per_user, icc, rating_rate, n_iid, deff, n_rated, n_exp, n_users, days)
