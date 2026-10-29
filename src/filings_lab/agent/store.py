"""Read side of the lake for the agent: paragraphs with filing metadata, events, prices and model signals.

Everything is loaded into memory from Parquet, which is right for this corpus size (tens of thousands of
paragraphs per few hundred companies). At full EDGAR scale the same interface sits on OpenSearch (paragraph
search), a warehouse table (events, prices) and the online feature store.
"""
from __future__ import annotations

import os
from functools import cached_property

import pandas as pd


class DataStore:
    def __init__(self, lake: str = "data/lake"):
        self.lake = lake.rstrip("/")

    def _read(self, rel: str) -> pd.DataFrame:
        path = f"{self.lake}/{rel}"
        return pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()

    @cached_property
    def index(self) -> pd.DataFrame:
        df = self._read("bronze_index.parquet")
        if not df.empty:
            df["date_filed"] = pd.to_datetime(df["date_filed"])
        return df

    @cached_property
    def paragraphs(self) -> pd.DataFrame:
        p = self._read("silver/paragraphs")
        if p.empty:
            return p
        meta = self.index[["accession", "ticker", "company", "date_filed", "primary_document"]]
        p = p.drop(columns=[c for c in ("year",) if c in p.columns]).merge(meta, on="accession", how="left")
        p["form"] = p["form"].astype(str)
        return p.sort_values(["accession", "section", "para_idx"]).reset_index(drop=True)

    @cached_property
    def by_id(self) -> pd.DataFrame:
        return self.paragraphs.set_index("paragraph_id")

    @cached_property
    def events(self) -> pd.DataFrame:
        e = self._read("gold/events")
        return e.set_index("accession") if not e.empty else e

    @cached_property
    def prices(self) -> pd.DataFrame:
        p = self._read("silver/prices.parquet")
        if not p.empty:
            p["date"] = pd.to_datetime(p["date"])
        return p

    @cached_property
    def signals(self) -> pd.DataFrame:
        s = self._read("gold/llm_signals.parquet")
        return s.set_index("accession") if not s.empty else s

    @cached_property
    def features(self) -> pd.DataFrame:
        f = self._read("gold/features")
        return f.set_index("accession") if not f.empty else f

    def companies(self) -> pd.DataFrame:
        idx = self.index
        if idx.empty:
            return idx
        g = idx.groupby(["ticker", "company", "cik"]).agg(filings=("accession", "count"), first=("date_filed", "min"),
                                                        last=("date_filed", "max"))
        return g.reset_index().sort_values("ticker")


def edgar_url(cik, accession: str, document: str | None = None) -> str:
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}"
    return f"{base}/{document}" if document else f"{base}/{accession}-index.htm"
