import numpy as np
import pandas as pd
import pytest

from filings_lab.modeling.backtest import backtest
from filings_lab.modeling.calibration import brier_decomposition, calibration_report, ece, reliability
from filings_lab.modeling.tuning import sample_configs, select


def _oof(n=4000, signal=0.5, seed=0):
    rng = np.random.default_rng(seed)
    pred = rng.uniform(0.3, 0.7, n)
    car = 0.02 * signal * (pred - 0.5) / 0.2 + rng.normal(0, 0.03, n)
    dates = pd.to_datetime("2016-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 2000, n)), unit="D")
    return pd.DataFrame({"pred": pred, "car": car, "car_tradable": car, "y": (car > 0).astype(int), "event_date": dates,
                         "fold": np.minimum(np.arange(n) * 5 // n, 4)})


def test_costs_reduce_returns_by_four_times_cost():
    r = backtest(_oof(), cost_bps=10)
    c = r["cost_curve"].set_index("cost_bps_per_side")
    assert c.loc[0, "mean_net_bps"] - c.loc[10, "mean_net_bps"] == pytest.approx(40.0)
    assert r["summary"]["mean_gross_bps"] > 0 and r["summary"]["break_even_cost_bps"] > 0


def test_no_signal_has_no_edge():
    r = backtest(_oof(signal=0.0, seed=3), cost_bps=0)
    assert abs(r["summary"]["nw_t"]) < 3


def test_calibration_metrics():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)
    assert ece(reliability(y, p)) < 0.02
    off = np.clip(p * 0.5 + 0.4, 0, 1)
    assert ece(reliability(y, off)) > 0.1
    d = brier_decomposition(y, p)
    assert d["brier"] == pytest.approx(d["reliability"] - d["resolution"] + d["uncertainty"], abs=0.01)


def test_isotonic_recalibration_is_out_of_sample_and_helps():
    df = _oof(n=6000)
    df["pred"] = np.clip(df["pred"] * 0.3 + 0.6, 0, 1)  # systematically overconfident
    rep = calibration_report(df)
    assert rep["isotonic"]["ece"] < rep["raw"]["ece"]


def test_tuning_selects_a_config_from_the_space():
    rng = np.random.default_rng(2)
    n = 900
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = (X["a"] + rng.normal(0, 1, n) > 0).astype(int).to_numpy()
    t0 = pd.Series(pd.to_datetime("2018-01-01") + pd.to_timedelta(np.arange(n), unit="D"))
    best, scores = select(X, y, t0, t0 + pd.offsets.BDay(2), {"objective": "binary", "verbose": -1}, sample_configs(3), rounds=50)
    assert len(scores) == 3 and max(scores) > 0.6 and best["num_leaves"] in (7, 15, 31)
