"""Bronze prices for every ticker in the filing index plus the market proxy, with a coverage report."""
from __future__ import annotations

import pandas as pd


def build_prices(index: pd.DataFrame, source, start: str, end: str, market: str = "SPY") -> tuple[pd.DataFrame, dict]:
    tickers = sorted({t for t in index["ticker"].dropna().astype(str).str.upper() if t} | {market.upper()})
    prices = source.history(tickers, start, end)
    got = set(prices["ticker"].unique()) if len(prices) else set()
    missing = [t for t in tickers if t not in got]
    filings_covered = float(index["ticker"].astype(str).str.upper().isin(got).mean()) if len(index) else 0.0
    report = {"tickers_requested": len(tickers), "tickers_found": len(got), "missing": missing[:200],
              "market_found": market.upper() in got, "filings_with_prices": filings_covered, "rows": int(len(prices))}
    if not report["market_found"]:
        raise ValueError(f"market proxy {market} has no prices; abnormal returns cannot be computed")
    return prices, report
