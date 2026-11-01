"""Event study machinery: align a filing to its first tradable day and measure the reaction.

Timing rule (docs/RESEARCH.md section 3): a filing accepted before 16:00 America/New_York on a trading
day has that day as day 0. Anything later, or on a non trading day, maps to the next trading day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MARKET_CLOSE_HOUR = 16


def event_day(accepted_at: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp | None:
    """First trading day on which the market could react to the filing."""
    if accepted_at is None or pd.isna(accepted_at):
        return None
    ts = pd.Timestamp(accepted_at)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("America/New_York")
        after_close = ts.hour >= MARKET_CLOSE_HOUR
    else:
        # A naive midnight timestamp means only the filing date is known. Assume it arrived after the
        # close: being a day late costs a little signal, being a day early leaks the label.
        no_time = ts.hour == 0 and ts.minute == 0 and ts.second == 0
        after_close = no_time or ts.hour >= MARKET_CLOSE_HOUR
    day = pd.Timestamp(ts.date())
    pos = trading_days.searchsorted(day)
    if pos >= len(trading_days):
        return None
    if trading_days[pos] == day and not after_close:
        return trading_days[pos]
    nxt = pos + 1 if trading_days[pos] == day else pos
    return trading_days[nxt] if nxt < len(trading_days) else None


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Wide frame of simple daily returns: index date, one column per ticker."""
    wide = prices.pivot_table(index="date", columns="ticker", values="close").sort_index()
    return wide.pct_change(fill_method=None)


def market_model(stock: pd.Series, market: pd.Series, min_obs: int = 60) -> tuple[float, float] | None:
    df = pd.concat([stock, market], axis=1).dropna()
    if len(df) < min_obs:
        return None
    x, y = df.iloc[:, 1].to_numpy(), df.iloc[:, 0].to_numpy()
    beta = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    alpha = y.mean() - beta * x.mean()
    return float(alpha), float(beta)


def event_returns(
    events: pd.DataFrame,
    returns: pd.DataFrame,
    market_col: str,
    window: tuple[int, int] = (0, 2),
    estimation: tuple[int, int] = (-250, -30),
    volume: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute reaction labels and strictly pre event price features for each filing.

    ``events`` needs: ticker, accepted_at. Returns the events frame with added columns:
    event_date, car (market adjusted CAR over ``window``), car_mm (market model CAR), ret_0,
    and pre event features mom_250_30, ret_5_1, vol_60_1, beta, abn_volume_5_1.
    """
    days = returns.index
    mkt = returns[market_col]
    out = []
    for row in events.itertuples(index=False):
        rec = row._asdict()
        t = rec.get("ticker")
        d0 = event_day(rec.get("accepted_at"), days)
        rec["event_date"] = d0
        if d0 is None or t not in returns.columns:
            out.append(rec)
            continue
        i0 = days.get_loc(d0)
        lo, hi = i0 + window[0], i0 + window[1]
        if hi >= len(days) or i0 + estimation[0] < 0:
            out.append(rec)
            continue
        r = returns[t]
        win_r, win_m = r.iloc[lo : hi + 1], mkt.iloc[lo : hi + 1]
        if win_r.isna().any():
            out.append(rec)
            continue
        rec["ret_0"] = float(r.iloc[i0])
        rec["car"] = float((win_r - win_m).sum())
        # tradable version: enter at the day 0 close, so day 0 itself is not captured
        rec["car_tradable"] = float((win_r.iloc[1:] - win_m.iloc[1:]).sum()) if len(win_r) > 1 else np.nan
        est_r = r.iloc[i0 + estimation[0] : i0 + estimation[1]]
        est_m = mkt.iloc[i0 + estimation[0] : i0 + estimation[1]]
        mm = market_model(est_r, est_m)
        if mm:
            alpha, beta = mm
            rec["beta"] = beta
            rec["car_mm"] = float((win_r - (alpha + beta * win_m)).sum())
        # pre event features: everything ends at day -1
        pre = r.iloc[: i0]
        rec["mom_250_30"] = float((1 + pre.iloc[-250:-30]).prod() - 1) if len(pre) >= 250 else np.nan
        rec["ret_5_1"] = float((1 + pre.iloc[-5:]).prod() - 1)
        rec["vol_60_1"] = float(pre.iloc[-60:].std())
        rec["mkt_ret_20_1"] = float((1 + mkt.iloc[i0 - 20 : i0]).prod() - 1)
        if volume is not None and t in volume.columns:
            v = volume[t]
            base = v.iloc[max(0, i0 - 60) : i0 - 5].mean()
            rec["abn_volume_5_1"] = float(v.iloc[i0 - 5 : i0].mean() / base) if base and base > 0 else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def add_labels(df: pd.DataFrame, col: str = "car") -> pd.DataFrame:
    """Binary sign label and within month tercile label (removes market wide drift)."""
    df = df.copy()
    df["label_up"] = (df[col] > 0).astype("Int8").where(df[col].notna())
    month = pd.to_datetime(df["event_date"]).dt.to_period("M")
    # Same definition as Spark's percent_rank(): (rank - 1) / (n - 1), ties take the lowest rank.
    rank = df.groupby(month)[col].rank(method="min")
    n = df.groupby(month)[col].transform("count")
    pct = ((rank - 1) / (n - 1)).where(n > 1, 0.0)
    tercile = np.where(pct < 1 / 3, 0, np.where(pct < 2 / 3, 1, 2))
    df["label_tercile"] = pd.Series(tercile, index=df.index).astype("Int8").where(df[col].notna())
    return df
