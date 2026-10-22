import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from filings_lab.modeling import stats
from filings_lab.modeling.cv import PurgedKFold, PurgedWalkForward, assert_no_leakage


def _events(n=600, seed=0):
    rng = np.random.default_rng(seed)
    t0 = pd.Series(pd.to_datetime("2015-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 2500, n)), unit="D"))
    t1 = t0 + pd.offsets.BDay(2)
    return t0, t1


def test_walk_forward_is_purged_and_forward_only():
    t0, t1 = _events()
    splits = list(PurgedWalkForward(n_splits=5).split(t0, t1))
    assert len(splits) == 5
    for train, test in splits:
        assert_no_leakage(t0, t1, train, test)
        assert t0.iloc[train].max() < t0.iloc[test].min()
    tests = np.concatenate([te for _, te in splits])
    assert len(tests) == len(np.unique(tests))  # each row is tested at most once


def test_purged_kfold_purges_and_embargoes():
    t0, t1 = _events()
    for train, test in PurgedKFold(n_splits=4, embargo=pd.Timedelta(days=10)).split(t0, t1):
        assert_no_leakage(t0, t1, train, test)
        hi = t1.iloc[test].max()
        after = t0.iloc[train]
        assert not ((after > hi) & (after <= hi + pd.Timedelta(days=10))).any()


def test_leak_detector_fires():
    t0, t1 = _events()
    with pytest.raises(AssertionError):
        assert_no_leakage(t0, t1, np.arange(0, 400), np.arange(300, 600))


def test_delong_matches_sklearn_and_detects_difference():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 3000)
    good = y + rng.normal(0, 1.2, 3000)
    bad = y + rng.normal(0, 4.0, 3000)
    ci = stats.auc_ci(y, good)
    assert abs(ci["auc"] - roc_auc_score(y, good)) < 1e-12
    assert ci["ci_low"] < ci["auc"] < ci["ci_high"]
    t = stats.delong_test(y, good, bad)
    assert t["diff"] > 0 and t["p_value"] < 1e-3
    same = stats.delong_test(y, good, good + 1e-9 * rng.normal(size=3000))
    assert same["p_value"] > 0.5


def test_delong_se_close_to_bootstrap():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 1500)
    s = y * 0.3 + rng.normal(size=1500)
    se = stats.auc_ci(y, s)["se"]
    boots = [roc_auc_score(y[i], s[i]) for i in (rng.integers(0, 1500, 1500) for _ in range(400))]
    assert abs(se - np.std(boots)) / se < 0.2


def test_newey_west_inflates_se_for_autocorrelated_series():
    rng = np.random.default_rng(3)
    e = rng.normal(size=500)
    ar = np.zeros(500)
    for i in range(1, 500):
        ar[i] = 0.8 * ar[i - 1] + e[i]
    iid = stats.newey_west(e, lags=0)
    assert abs(iid["t"] - e.mean() / (e.std() / np.sqrt(500))) < 1e-6
    assert stats.newey_west(ar)["se"] > stats.newey_west(ar, lags=0)["se"] * 1.5


def test_benjamini_hochberg_known_example():
    out = stats.benjamini_hochberg([0.01, 0.04, 0.03, 0.005], alpha=0.05)
    assert out["reject"].all()
    out = stats.benjamini_hochberg([0.01, 0.20, 0.02, 0.80], alpha=0.05)  # q = 0.04, 0.27, 0.04, 0.80
    assert out["reject"].tolist() == [True, False, True, False]


def test_diebold_mariano_sign():
    rng = np.random.default_rng(4)
    a = rng.normal(1.0, 0.2, 400)
    b = a + 0.05 + rng.normal(0, 0.05, 400)
    r = stats.diebold_mariano(a, b)
    assert r["dm"] < 0 and r["p_value"] < 0.01


def test_permutation_test_null_and_signal():
    rng = np.random.default_rng(5)
    x = rng.normal(size=800)
    blocks = np.repeat(np.arange(40), 20)
    y_signal = (x + rng.normal(0, 1, 800) > 0).astype(int)
    y_null = rng.integers(0, 2, 800)
    assert stats.permutation_test(y_signal, blocks, lambda y: roc_auc_score(y, x), n_perm=99)["p_value"] < 0.05
    assert stats.permutation_test(y_null, blocks, lambda y: roc_auc_score(y, x), n_perm=99)["p_value"] > 0.05


def test_block_bootstrap_covers_mean():
    rng = np.random.default_rng(6)
    x = rng.normal(0.5, 1, 300)
    ci = stats.block_bootstrap_ci(x)
    assert ci["ci_low"] < 0.5 < ci["ci_high"]
