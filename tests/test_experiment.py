import math

import numpy as np
import pandas as pd
import pytest

from filings_lab.experiment.analysis import (
    analyze,
    msprt_path,
    newcombe_diff,
    ratio_delta,
    render_markdown,
    two_proportion_z,
    wilson,
)
from filings_lab.experiment.power import achieved_power, mde, n_continuous, n_per_arm, plan


def test_sample_size_matches_statsmodels():
    from statsmodels.stats.power import NormalIndPower
    from statsmodels.stats.proportion import proportion_effectsize

    ours = n_per_arm(0.5, 0.55)
    ref = NormalIndPower().solve_power(proportion_effectsize(0.55, 0.5), alpha=0.05, power=0.8)
    assert abs(ours - ref) <= 2
    assert achieved_power(ours, 0.5, 0.55) == pytest.approx(0.8, abs=0.01)
    assert mde(ours, 0.5) == pytest.approx(0.05, abs=0.002)


def test_design_effect_inflates_plan():
    iid = plan(0.6, 0.04, rated_per_user=1.0, icc=0.2)
    clustered = plan(0.6, 0.04, rated_per_user=3.0, icc=0.2)
    assert clustered.design_effect == pytest.approx(1.4)
    assert clustered.n_rated_per_arm == math.ceil(iid.n_rated_per_arm_iid * 1.4)
    assert n_continuous(sd=1.0, delta=0.2) == 393


def test_wilson_and_newcombe_known_values():
    lo, hi = wilson(81, 263)  # Newcombe (1998) example: 0.2553 to 0.3662
    assert lo == pytest.approx(0.2553, abs=1e-3) and hi == pytest.approx(0.3662, abs=1e-3)
    lo, hi = newcombe_diff(48, 80, 56, 70)  # Newcombe (1998) example (a), method 10: 0.0524 to 0.3339
    assert lo == pytest.approx(0.0524, abs=1e-3) and hi == pytest.approx(0.3339, abs=1e-3)
    z, p = two_proportion_z(50, 100, 65, 100)
    assert z > 0 and 0.02 < p < 0.04


def test_delta_method_matches_cluster_bootstrap():
    rng = np.random.default_rng(0)
    y = rng.integers(1, 6, 800).astype(float)
    p_user = 1 / (1 + np.exp(-(0.4 + rng.normal(0, 0.8, 800))))
    x = rng.binomial(y.astype(int), p_user).astype(float)
    r, var = ratio_delta(x, y)
    boots = []
    for _ in range(600):
        i = rng.integers(0, 800, 800)
        boots.append(x[i].sum() / y[i].sum())
    assert math.sqrt(var) == pytest.approx(np.std(boots), rel=0.15)


def _events(n, pa, pb, seed):
    rng = np.random.default_rng(seed)
    variant = np.where(rng.random(n) < 0.5, "A", "B")
    p = np.where(variant == "A", pa, pb)
    return pd.DataFrame({"ts": np.arange(n, dtype=float), "variant": variant, "thumbs_up": (rng.random(n) < p).astype(int)})


def test_msprt_controls_false_positives_under_peeking():
    # under the null, a fixed horizon z test checked after every observation rejects far more than 5%;
    # the always valid p value must not
    rejections = sum(msprt_path(_events(2000, 0.6, 0.6, s), "A", "B")["p_always_valid"].min() < 0.05 for s in range(60))
    assert rejections / 60 <= 0.1
    assert msprt_path(_events(6000, 0.55, 0.65, 1), "A", "B")["p_always_valid"].iloc[-1] < 0.01


def _log_frame(n_users, pa, pb, seed, weights=(0.5, 0.5), user_sd=0.6):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        v = "A" if rng.random() < weights[0] / sum(weights) else "B"
        eff = rng.normal(0, user_sd)
        base = pa if v == "A" else pb
        for _ in range(max(1, rng.poisson(3))):
            p = 1 / (1 + np.exp(-(np.log(base / (1 - base)) + eff)))
            rated = rng.random() < 0.5
            rows.append({"request_id": f"{u}-{len(rows)}", "ts": float(len(rows)), "experiment": "x", "variant": v,
                         "user_id": f"u{u}", "latency_s": rng.lognormal(-3, 0.3), "citation_valid_rate": 1.0,
                         "cost_usd": 0.0, "rating": (1 if rng.random() < p else -1) if rated else np.nan})
    return pd.DataFrame(rows)


def test_analysis_recovers_planted_lift_and_clusters_users():
    df = _log_frame(4000, 0.55, 0.65, seed=3)
    r = analyze(df, "x", {"A": 0.5, "B": 0.5})
    lo, hi = r.primary["ci_clustered"]
    assert lo > 0 and r.decision == "ship treatment"
    assert r.primary["design_effect"] > 1.05  # repeat raters make the naive interval too narrow
    nlo, nhi = r.primary["ci_newcombe"]
    assert (hi - lo) > (nhi - nlo)
    assert not r.srm["mismatch"]


def test_analysis_flags_sample_ratio_mismatch_and_null():
    df = _log_frame(3000, 0.6, 0.6, seed=4, weights=(0.6, 0.4))
    r = analyze(df, "x", {"A": 0.5, "B": 0.5})
    assert r.srm["mismatch"] and r.decision == "do not ship"
    r2 = analyze(_log_frame(3000, 0.6, 0.6, seed=5), "x", {"A": 0.5, "B": 0.5})
    assert r2.decision in ("inconclusive", "keep control", "ship treatment")
    assert r2.primary["ci_clustered"][0] < 0.03


def test_confidence_sequence_covers_truth_and_sequential_decisions():
    path = msprt_path(_events(8000, 0.55, 0.60, 7), "A", "B")
    lo, hi = path["cs_low"].iloc[-1], path["cs_high"].iloc[-1]
    assert lo <= 0.05 <= hi and (path["cs_high"].diff().dropna() <= 1e-12).all()
    df = _log_frame(800, 0.6, 0.6, seed=8)
    r = analyze(df, "x", {"A": 0.5, "B": 0.5}, sequential=True, max_rated_per_arm=100_000)
    assert r.decision in ("continue", "ship treatment", "keep control")
    strong = analyze(_log_frame(5000, 0.5, 0.65, seed=9), "x", {"A": 0.5, "B": 0.5}, sequential=True)
    assert strong.decision == "ship treatment"


def test_sequential_readout_stops_at_maximum_sample():
    df = _log_frame(3000, 0.6, 0.6, seed=12)
    rated_min = df[df["rating"].notna()].groupby("variant").size().min()
    cap = int(rated_min * 0.6)
    r = analyze(df, "x", {"A": 0.5, "B": 0.5}, sequential=True, max_rated_per_arm=cap)
    assert r.primary["truncated_at_ts"] is not None
    assert min(r.arms["A"]["rated"], r.arms["B"]["rated"]) == cap
    assert r.decision in ("inconclusive", "ship treatment", "keep control")


def test_readout_refuses_a_variant_whose_model_did_not_answer():
    df = _log_frame(4000, 0.55, 0.65, seed=3)
    df["n_extractions"] = 1
    df["n_extraction_fallbacks"] = 0
    ok = analyze(df, "x", {"A": 0.5, "B": 0.5})
    assert ok.decision == "ship treatment" and ok.integrity["B"]["served_by_variant_model"] == 1.0
    df.loc[df["variant"] == "B", "n_extraction_fallbacks"] = 1  # B's model always fell back to the rules
    r = analyze(df, "x", {"A": 0.5, "B": 0.5})
    assert r.decision == "do not ship" and not r.integrity["B"]["ok"]
    assert any("treatment integrity" in x for x in r.reasons)
    assert "## Treatment integrity" in render_markdown(r)
    # logs without the columns (older deployments, offline simulations) are analysed as before
    assert analyze(_log_frame(4000, 0.55, 0.65, seed=3), "x", {"A": 0.5, "B": 0.5}).integrity == {}
