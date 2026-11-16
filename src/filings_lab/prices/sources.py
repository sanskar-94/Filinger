"""Pluggable daily price sources. All return a long frame: ticker, date, close, volume.

``close`` must be split and dividend adjusted. The canonical identifier elsewhere in the lab is CIK, so a
ticker map (from ``company_tickers.json`` or your vendor's security master) joins prices to filings.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Protocol

import pandas as pd

PRICE_COLUMNS = ["ticker", "date", "close", "volume"]


class PriceSource(Protocol):
    def history(self, tickers: list[str], start: str, end: str) -> pd.DataFrame: ...


def _standardise(df: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    cols = {c.lower().replace(" ", "_"): c for c in df.columns}
    out = pd.DataFrame(index=df.index)  # rows first, so a scalar ticker fills every row
    out["ticker"] = df[cols["ticker"]] if "ticker" in cols else ticker
    out["date"] = pd.to_datetime(df[cols["date"]])
    close_col = cols.get("adj_close") or cols.get("adjclose") or cols.get("close")
    out["close"] = pd.to_numeric(df[close_col], errors="coerce")
    out["volume"] = pd.to_numeric(df[cols["volume"]], errors="coerce") if "volume" in cols else float("nan")
    return out.dropna(subset=["close"]).sort_values(["ticker", "date"]).reset_index(drop=True)


class CsvPriceSource:
    """Reads either a single long file (csv/parquet with a ticker column) or a directory of TICKER.csv files."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def history(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        if self.path.is_dir():
            frames = []
            for t in tickers:
                f = self.path / f"{t}.csv"
                if f.exists():
                    frames.append(_standardise(pd.read_csv(f), t))
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PRICE_COLUMNS)
        else:
            raw = pd.read_parquet(self.path) if self.path.suffix == ".parquet" else pd.read_csv(self.path)
            df = _standardise(raw)
            df = df[df["ticker"].isin(tickers)]
        return df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)


class YahooPriceSource:
    """Free daily bars from Yahoo Finance's chart endpoint, for research use. Not survivorship bias free.

    The endpoint is unofficial and unversioned, so treat it as a convenience and keep a vendor file for
    anything that matters. ``close`` is the split and dividend adjusted close, which the event study needs:
    an unadjusted close turns every split into a fake minus 50% day.
    """

    URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

    def __init__(self, session=None, pause: float = 0.25, retries: int = 3):
        import requests

        self.session = session or requests.Session()
        self.pause, self.retries = pause, retries

    @staticmethod
    def parse(payload: dict, ticker: str) -> pd.DataFrame:
        result = ((payload.get("chart") or {}).get("result") or [None])[0]
        if not result or not result.get("timestamp"):
            return pd.DataFrame(columns=PRICE_COLUMNS)
        quote = result["indicators"]["quote"][0]
        adjusted = (result["indicators"].get("adjclose") or [{}])[0].get("adjclose")
        # bars are stamped at the exchange's open in UTC; the offset turns them into the local trading date
        local = pd.to_datetime(result["timestamp"], unit="s") + pd.Timedelta(seconds=result["meta"].get("gmtoffset", 0))
        df = pd.DataFrame({"ticker": ticker, "date": local.normalize(),
                           "close": pd.to_numeric(pd.Series(adjusted or quote["close"]), errors="coerce").to_numpy(),
                           "volume": pd.to_numeric(pd.Series(quote.get("volume")), errors="coerce").to_numpy()})
        return df.dropna(subset=["close"]).drop_duplicates("date", keep="last").reset_index(drop=True)

    def history(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        import time

        params: dict[str, str | int] = {"period1": int(pd.Timestamp(start).timestamp()),
                  "period2": int((pd.Timestamp(end) + pd.Timedelta(days=1)).timestamp()),
                  "interval": "1d", "events": "div,splits"}
        frames = []
        for t in tickers:
            for attempt in range(self.retries):
                resp = self.session.get(self.URL.format(symbol=t.upper().replace(".", "-")), params=params, timeout=30,
                                        headers={"User-Agent": "Mozilla/5.0 (compatible; filings-lab)"})
                if resp.status_code == 200:
                    frames.append(self.parse(resp.json(), t.upper()))
                    break
                if resp.status_code not in (429, 500, 502, 503) or attempt == self.retries - 1:
                    break  # an unknown symbol is a 404: reported as missing by the coverage report
                time.sleep(2.0 * (attempt + 1))
            time.sleep(self.pause)
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PRICE_COLUMNS)
        return df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)


class StooqPriceSource:
    """Daily bars from stooq.com. Since late 2026 the site answers scripts with a browser check, so this
    source now finds nothing unless the caller supplies a session that passes it; ``YahooPriceSource`` is the
    default free source. Not survivorship bias free."""

    URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"

    def __init__(self, session=None, suffix: str = ".us"):
        import requests

        self.session = session or requests.Session()
        self.suffix = suffix

    def history(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        frames = []
        for t in tickers:
            resp = self.session.get(self.URL.format(symbol=t.lower() + self.suffix), timeout=30)
            if resp.status_code != 200 or not resp.text.startswith("Date"):
                continue
            frames.append(_standardise(pd.read_csv(io.StringIO(resp.text)), t.upper()))
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PRICE_COLUMNS)
        return df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)


class FramePriceSource:
    """Wraps an in memory frame; used by the synthetic data generator and in tests."""

    def __init__(self, df: pd.DataFrame):
        self.df = _standardise(df)

    def history(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        d = self.df
        return d[d["ticker"].isin(tickers) & (d["date"] >= start) & (d["date"] <= end)].reset_index(drop=True)
