import numpy as np
import pandas as pd

from filings_lab.prices.events import add_labels, event_day, event_returns

DAYS = pd.bdate_range("2020-01-01", "2021-12-31")


def ts(s):
    return pd.Timestamp(s, tz="America/New_York")


def test_event_day_rules():
    assert event_day(ts("2021-03-03 10:00"), DAYS) == pd.Timestamp("2021-03-03")
    assert event_day(ts("2021-03-03 16:30"), DAYS) == pd.Timestamp("2021-03-04")
    assert event_day(ts("2021-03-05 17:00"), DAYS) == pd.Timestamp("2021-03-08")  # Friday evening
    assert event_day(ts("2021-03-06 09:00"), DAYS) == pd.Timestamp("2021-03-08")  # Saturday
    assert event_day(pd.Timestamp("2021-03-03"), DAYS) == pd.Timestamp("2021-03-04")  # date only: be conservative
    assert event_day(pd.Timestamp("2021-03-03 14:05:00+00:00"), DAYS) == pd.Timestamp("2021-03-03")  # 09:05 ET


def _returns(shock_day: str, shock: float):
    rng = np.random.default_rng(0)
    mkt = rng.normal(0, 0.01, len(DAYS))
    stock = 1.2 * mkt + rng.normal(0, 0.001, len(DAYS))
    i = DAYS.get_loc(pd.Timestamp(shock_day))
    stock[i] += shock
    return pd.DataFrame({"AAA": stock, "SPY": mkt}, index=DAYS)


def test_car_captures_planted_shock_and_features_are_pre_event():
    rets = _returns("2021-06-15", 0.05)
    ev = pd.DataFrame({"ticker": ["AAA"], "accepted_at": [ts("2021-06-15 08:00")]})
    out = event_returns(ev, rets, "SPY")
    row = out.iloc[0]
    assert abs(row["car_mm"] - 0.05) < 0.01
    assert abs(row["beta"] - 1.2) < 0.05
    # moving the shock to day 0 must not change any pre event feature
    rets2 = _returns("2021-06-15", -0.30)
    row2 = event_returns(ev, rets2, "SPY").iloc[0]
    for f in ("ret_5_1", "vol_60_1", "mom_250_30", "mkt_ret_20_1"):
        assert row[f] == row2[f]


def test_labels():
    df = pd.DataFrame({"car": [0.1, -0.1, 0.0, 0.2, -0.3, 0.05], "event_date": pd.to_datetime(["2021-01-05"] * 6)})
    out = add_labels(df)
    assert out["label_up"].tolist() == [1, 0, 0, 1, 0, 1]
    assert out["label_tercile"].tolist() == [2, 0, 1, 2, 0, 1]
    # percent_rank: [0.8, 0.2, 0.4, 1.0, 0.0, 0.6]
