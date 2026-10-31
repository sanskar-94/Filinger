"""Simulated traffic for the A/B test, run through the real agent service.

Each simulated analyst asks a Poisson number of questions over the test window. Every request goes through
``AgentService.ask``, so assignment, retrieval, citations, latency and logging are the production code
paths and the guardrail metrics are real measurements. Only the human part is simulated: whether the
analyst rates the answer and whether the rating is positive. Positive ratings follow
logit(p) = logit(base_rate[variant]) + user_effect, with user_effect ~ N(0, user_sd): users differ in how
generous they are, which correlates ratings within a user and is why the analysis clusters by user.

The planted rates make the readout checkable against a known truth. They describe nothing about real users.
"""
from __future__ import annotations

import time

import numpy as np

QUESTIONS = [
    "Did the company change its revenue guidance, and how did the stock react?",
    "Has management flagged any liquidity or going concern issues?",
    "Summarise any litigation or regulatory investigations disclosed.",
    "How has management described demand and margins recently?",
    "What new risk factors were added this year?",
    "Did the company announce any executive departures?",
]


def _logit(p):
    return np.log(p / (1 - p))


def simulate(service, n_users: int = 3000, mean_requests: float = 4.0, rating_rate: float = 0.45,
             base_rates: dict[str, float] | None = None, user_sd: float = 0.6, days: int = 14,
             start_ts: float | None = None, seed: int = 0, progress: bool = False) -> dict:
    base_rates = base_rates or {"A": 0.60, "B": 0.64}
    rng = np.random.default_rng(seed)
    tickers = service.store.companies()["ticker"].tolist()
    t0 = start_ts or time.time() - days * 86400
    expected = {v: [] for v in base_rates}
    n_req = 0
    for u in range(n_users):
        user = f"sim-user-{seed}-{u}"
        effect = rng.normal(0, user_sd)
        k = max(1, rng.poisson(mean_requests))
        first = t0 + rng.uniform(0, days * 86400 * 0.8)
        for j in range(k):
            ts = min(first + j * rng.exponential(6 * 3600), t0 + days * 86400 - 60)
            out = service.ask(QUESTIONS[rng.integers(len(QUESTIONS))], tickers[rng.integers(len(tickers))], user, ts=ts)
            n_req += 1
            if rng.random() < rating_rate:
                p = 1 / (1 + np.exp(-(_logit(base_rates[out["variant"]]) + effect)))
                expected[out["variant"]].append(p)
                service.log.feedback(out["request_id"], 1 if rng.random() < p else -1, ts=ts + rng.uniform(5, 120))
        if progress and (u + 1) % 500 == 0:
            print(f"{u + 1}/{n_users} users, {n_req} requests")
    truth = {v: float(np.mean(ps)) for v, ps in expected.items() if ps}
    return {"users": n_users, "requests": n_req, "true_expected_rates": truth,
            "true_diff": truth.get("B", np.nan) - truth.get("A", np.nan)}
