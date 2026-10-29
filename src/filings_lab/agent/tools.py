"""The agent's tools: definitions (strict JSON schemas) and their implementations.

Every tool that returns filing text returns paragraph ids, and the ToolBox remembers every id it handed
out during a session. The answer validator accepts a citation only if its id is in that set, which makes
a fabricated citation detectable rather than merely unlikely.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .retrieval import BM25Index
from .store import DataStore, edgar_url

NULLABLE_STR = {"type": ["string", "null"]}

TOOL_DEFS = [
    {
        "name": "search_filings",
        "description": "Full text search over paragraphs of SEC filings (10-K, 10-Q, 8-K MD&A, risk factors, legal proceedings and 8-K items). "
                       "Returns the best matching paragraphs with their paragraph_id, which is what you cite. Use filters whenever the question names a company, form or period.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "keywords or a short natural language query"},
                "ticker": {**NULLABLE_STR, "description": "restrict to one company, e.g. ACME; null for all"},
                "form": {**NULLABLE_STR, "description": "10-K, 10-Q or 8-K; null for all"},
                "date_from": {**NULLABLE_STR, "description": "YYYY-MM-DD or null"},
                "date_to": {**NULLABLE_STR, "description": "YYYY-MM-DD or null"},
                "k": {"type": "integer", "description": "number of results, 1 to 10"},
            },
            "required": ["query", "ticker", "form", "date_from", "date_to", "k"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_paragraph",
        "description": "Fetch one paragraph by paragraph_id with its filing metadata and the ids of the neighbouring paragraphs, to read surrounding context before citing.",
        "input_schema": {"type": "object", "properties": {"paragraph_id": {"type": "string"}},
                         "required": ["paragraph_id"], "additionalProperties": False},
    },
    {
        "name": "list_filings",
        "description": "List a company's filings (newest first) with accession numbers, form types, filing dates and the market reaction when known.",
        "input_schema": {"type": "object",
                         "properties": {"ticker": {"type": "string"}, "form": NULLABLE_STR, "limit": {"type": "integer"}},
                         "required": ["ticker", "form", "limit"], "additionalProperties": False},
    },
    {
        "name": "get_price_history",
        "description": "Daily closing prices for a ticker between two dates, summarised (start, end, return, max drawdown, volatility) with a weekly series.",
        "input_schema": {"type": "object",
                         "properties": {"ticker": {"type": "string"}, "start": {"type": "string"}, "end": {"type": "string"}},
                         "required": ["ticker", "start", "end"], "additionalProperties": False},
    },
    {
        "name": "get_event_return",
        "description": "The stock's reaction to one filing: first tradable day, day 0 return and the 3 day cumulative abnormal return versus the market.",
        "input_schema": {"type": "object", "properties": {"accession": {"type": "string"}},
                         "required": ["accession"], "additionalProperties": False},
    },
    {
        "name": "get_model_signal",
        "description": "The lab's model outputs for one filing: LLM extracted signals (guidance, tone, liquidity, litigation) and the predicted probability of a positive reaction with its main drivers.",
        "input_schema": {"type": "object", "properties": {"accession": {"type": "string"}},
                         "required": ["accession"], "additionalProperties": False},
    },
]


def _clean(v):
    if isinstance(v, (pd.Timestamp,)):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 6)
    if isinstance(v, np.integer):
        return int(v)
    return v


class ToolBox:
    def __init__(self, store: DataStore, index: BM25Index | None = None, scorer=None, diversify: bool = True):
        self.store = store
        self.index = index or BM25Index(store.paragraphs)
        self.scorer = scorer  # callable(features dict) -> {"probability_up", "drivers"}; optional
        self.diversify = diversify
        self.seen_ids: set[str] = set()

    def reset(self) -> None:
        self.seen_ids = set()

    def execute(self, name: str, args: dict) -> str:
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            raise ValueError(f"unknown tool {name}")
        return json.dumps(fn(**args), default=_clean)

    def _para(self, row, full: bool = False) -> dict:
        pid = row["paragraph_id"]
        self.seen_ids.add(pid)
        text = row["text"] if full else (row["text"][:600] + ("..." if len(row["text"]) > 600 else ""))
        return {"paragraph_id": pid, "ticker": row.get("ticker"), "company": row.get("company"), "form": row["form"],
                "date_filed": _clean(row.get("date_filed")), "accession": row["accession"], "section": row["section"],
                "text": text}

    # ---------------------------------------------------------------------------------------- tools
    def t_search_filings(self, query, ticker=None, form=None, date_from=None, date_to=None, k=6):
        k = int(min(max(k or 6, 1), 10))
        hits = self.index.search(query, k=k, diversify=self.diversify, ticker=ticker, form=form,
                                 date_from=date_from, date_to=date_to)
        return {"results": [self._para(r) | {"score": round(float(r["score"]), 3)} for _, r in hits.iterrows()]}

    def t_get_paragraph(self, paragraph_id):
        if paragraph_id not in self.store.by_id.index:
            return {"error": f"unknown paragraph_id {paragraph_id}"}
        row = self.store.by_id.loc[paragraph_id]
        row = row.to_dict() | {"paragraph_id": paragraph_id}
        same = self.store.paragraphs[(self.store.paragraphs["accession"] == row["accession"])
                                     & (self.store.paragraphs["section"] == row["section"])]
        idx = int(row["para_idx"])
        neighbours = same[same["para_idx"].isin([idx - 1, idx + 1])]["paragraph_id"].tolist()
        cik = int(row["cik"])
        return self._para(row, full=True) | {"neighbour_ids": neighbours,
                                             "url": edgar_url(cik, row["accession"], row.get("primary_document"))}

    def t_list_filings(self, ticker, form=None, limit=10):
        idx = self.store.index
        d = idx[idx["ticker"].str.upper() == ticker.upper()]
        if form:
            d = d[d["form"].str.upper() == form.upper()]
        d = d.sort_values("date_filed", ascending=False).head(int(min(max(limit or 10, 1), 50)))
        ev = self.store.events
        rows = []
        for r in d.itertuples(index=False):
            e = ev.loc[r.accession] if not ev.empty and r.accession in ev.index else None
            rows.append({"accession": r.accession, "form": r.form, "date_filed": _clean(r.date_filed),
                         "items": r.items or None, "car_3d": _clean(e["car"]) if e is not None else None})
        return {"ticker": ticker.upper(), "filings": rows}

    def t_get_price_history(self, ticker, start, end):
        p = self.store.prices
        s = p[(p["ticker"].str.upper() == ticker.upper()) & (p["date"] >= start) & (p["date"] <= end)].sort_values("date")
        if s.empty:
            return {"error": f"no prices for {ticker} between {start} and {end}"}
        close = s.set_index("date")["close"]
        r = close.pct_change().dropna()
        weekly = close.resample("W-FRI").last().dropna()
        return {"ticker": ticker.upper(), "start": _clean(close.index[0]), "end": _clean(close.index[-1]),
                "start_close": _clean(close.iloc[0]), "end_close": _clean(close.iloc[-1]),
                "total_return": _clean(close.iloc[-1] / close.iloc[0] - 1),
                "max_drawdown": _clean((close / close.cummax() - 1).min()),
                "annualised_vol": _clean(r.std() * np.sqrt(252)) if len(r) > 2 else None,
                "weekly": [{"date": _clean(d), "close": _clean(v)} for d, v in weekly.tail(60).items()]}

    def t_get_event_return(self, accession):
        ev = self.store.events
        if ev.empty or accession not in ev.index:
            return {"error": f"no reaction data for {accession}"}
        e = ev.loc[accession]
        return {"accession": accession, "ticker": e["ticker"], "form": e["form"], "event_date": _clean(e["event_date"]),
                "return_day0": _clean(e["ret_0"]), "car_0_2": _clean(e["car"]), "car_market_model": _clean(e.get("car_mm")),
                "note": "CAR is the 3 trading day return in excess of the market (SPY) starting at the first tradable day."}

    def t_get_model_signal(self, accession):
        out: dict = {"accession": accession}
        sig = self.store.signals
        if not sig.empty and accession in sig.index:
            s = sig.loc[accession]
            out["llm_signals"] = {
                "guidance": "raised" if s["llm_guid_up"] else "lowered" if s["llm_guid_down"] else "maintained" if s["llm_guid_flat"] else "none",
                "tone": {1.0: "positive", 0.0: "neutral", -1.0: "negative"}.get(_clean(s["llm_tone"]), "unknown"),
                "liquidity_concern": bool(s["llm_liquidity"]), "litigation_risk": bool(s["llm_litigation"])}
        feats = self.store.features
        if self.scorer is not None and not feats.empty and accession in feats.index:
            out["reaction_model"] = self.scorer(feats.loc[accession].to_dict())
        if len(out) == 1:
            out["error"] = f"no model outputs for {accession}"
        return out
