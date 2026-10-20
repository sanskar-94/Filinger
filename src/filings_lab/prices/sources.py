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
    out = pd.DataFrame()
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


class StooqPriceSource:
    """Free daily bars from stooq.com for experimentation. Not survivorship bias free."""

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
