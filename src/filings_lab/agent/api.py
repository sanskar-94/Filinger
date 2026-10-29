"""HTTP routes for the analyst app (mounted into the Phase 4 FastAPI service)."""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    ticker: str | None = None
    user_id: str = Field(default="anonymous", max_length=128)


class FeedbackRequest(BaseModel):
    request_id: str
    rating: int = Field(description="1 for helpful, -1 for not helpful")
    comment: str | None = Field(default=None, max_length=2000)


def build_router(service, auth, metrics) -> APIRouter:
    r = APIRouter(prefix="/v1", dependencies=[Depends(auth)])

    @r.post("/agent/ask")
    def ask(req: AskRequest):
        out = service.ask(req.question, req.ticker, req.user_id)
        metrics.ab_requests.labels(out.get("experiment") or "none", out["variant"]).inc()
        metrics.agent_citation_valid.labels(out["variant"]).observe(out["citation_valid_rate"])
        return out

    @r.post("/feedback")
    def feedback(req: FeedbackRequest):
        if req.rating not in (-1, 1):
            raise HTTPException(422, "rating must be 1 or -1")
        if service.log is None or not service.log.feedback(req.request_id, req.rating, req.comment):
            raise HTTPException(404, "unknown request_id")
        return {"ok": True}

    @r.get("/experiments/current")
    def experiment():
        e = service.experiment
        return {"name": e.name, "variants": sorted(e.variants), "active": e.active} if e else {"name": None}

    @r.get("/companies")
    def companies():
        c = service.store.companies()
        return [{"ticker": x.ticker, "company": x.company, "cik": int(x.cik), "filings": int(x.filings),
                 "first": x.first.strftime("%Y-%m-%d"), "last": x.last.strftime("%Y-%m-%d")} for x in c.itertuples(index=False)]

    @r.get("/companies/{ticker}/filings")
    def filings(ticker: str, form: str | None = None, limit: int = 40):
        import json

        from .tools import ToolBox

        tb = ToolBox(service.store, service.index)
        return json.loads(tb.execute("list_filings", {"ticker": ticker, "form": form, "limit": limit}))

    @r.get("/companies/{ticker}/prices")
    def prices(ticker: str, start: str = "2000-01-01", end: str = "2100-01-01"):
        p = service.store.prices
        s = p[(p["ticker"].str.upper() == ticker.upper()) & (p["date"] >= start) & (p["date"] <= end)].sort_values("date")
        if s.empty:
            raise HTTPException(404, f"no prices for {ticker}")
        w = s.set_index("date")["close"].resample("W-FRI").last().dropna()
        return {"ticker": ticker.upper(), "series": [{"date": d.strftime("%Y-%m-%d"), "close": round(float(v), 4)} for d, v in w.items()]}

    @r.get("/paragraphs/{paragraph_id}")
    def paragraph(paragraph_id: str):
        by_id = service.store.by_id
        if paragraph_id not in by_id.index:
            raise HTTPException(404, "unknown paragraph")
        row = by_id.loc[paragraph_id]
        from .store import edgar_url

        return {"paragraph_id": paragraph_id, "text": row["text"], "ticker": row.get("ticker"), "company": row.get("company"),
                "form": row["form"], "section": row["section"], "accession": row["accession"],
                "date_filed": pd.Timestamp(row["date_filed"]).strftime("%Y-%m-%d"),
                "url": edgar_url(int(row["cik"]), row["accession"], row.get("primary_document"))}

    return r
