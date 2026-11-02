"""HTTP routes for the analyst app (mounted into the Phase 4 FastAPI service)."""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    ticker: str | None = None
    user_id: str = Field(default="anonymous", max_length=128)


class ReviewRequest(BaseModel):
    accession: str
    action: str = Field(description="accept, correct or skip")
    label: dict | None = None
    reviewer: str = Field(default="anonymous", max_length=128)


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

    @r.get("/filings/{accession}/paragraphs")
    def filing_paragraphs(accession: str):
        from .store import edgar_url

        p = service.store.paragraphs
        rows = p[p["accession"] == accession]
        if rows.empty:
            raise HTTPException(404, f"unknown accession {accession}")
        first = rows.iloc[0]
        order = ["mdna", "risk_factors", "legal", "8k_results_of_operations", "8k_material_agreement",
                 "8k_officer_change", "8k_other_events", "8k_exhibits", "full_text"]
        rank = {s: i for i, s in enumerate(order)}
        sections = []
        for sec, g in sorted(rows.groupby("section"), key=lambda kv: rank.get(kv[0], 99)):
            g = g.sort_values("para_idx")
            sections.append({"section": sec, "paragraphs": [{"paragraph_id": x.paragraph_id, "para_idx": int(x.para_idx), "text": x.text}
                                                            for x in g.itertuples(index=False)]})
        return {"accession": accession, "ticker": first.get("ticker"), "company": first.get("company"), "form": first["form"],
                "date_filed": pd.Timestamp(first["date_filed"]).strftime("%Y-%m-%d"),
                "url": edgar_url(int(first["cik"]), accession, first.get("primary_document")), "sections": sections}

    @r.get("/experiments/current/readout")
    def readout():
        import time as _t

        e = service.experiment
        if e is None or service.log is None:
            raise HTTPException(404, "no active experiment")
        cached = getattr(service, "_readout_cache", None)
        if cached and _t.time() - cached[0] < 60:
            return cached[1]
        from ..experiment.analysis import analyze

        df = service.log.frame(e.name)
        if df["rating"].notna().sum() < 40:
            raise HTTPException(409, "not enough rated answers yet")
        res = analyze(df, e.name, {k: v.get("weight", 1.0) for k, v in e.variants.items()}, planned_mde=e.planned_mde,
                      guardrails=e.guardrails if isinstance(e.guardrails, dict) else None,
                      sequential=e.analysis == "sequential", max_rated_per_arm=e.max_rated_per_arm).to_dict()
        res["descriptions"] = {k: v.get("description", "") for k, v in e.variants.items()}
        service._readout_cache = (_t.time(), res)
        return res

    def _review_store():
        import os

        from ..llm.review import ReviewStore

        if getattr(service, "_review", None) is None:
            service._review = ReviewStore(os.environ.get("FLAB_REVIEW_QUEUE", "data/llm_dataset/gold.jsonl"),
                                          os.environ.get("FLAB_REVIEWED", "data/llm_dataset/reviewed.jsonl"))
        return service._review

    @r.get("/review/next")
    def review_next():
        item = _review_store().next_item()
        return item or {"done": True, **_review_store().stats()}

    @r.post("/review")
    def review_submit(req: ReviewRequest):
        from ..llm.schema import GUIDANCE, TONE

        lab = req.label or {}
        if "guidance_direction" in lab and lab["guidance_direction"] not in GUIDANCE:
            raise HTTPException(422, "invalid guidance_direction")
        if "management_tone" in lab and lab["management_tone"] not in TONE:
            raise HTTPException(422, "invalid management_tone")
        try:
            return _review_store().submit(req.accession, req.action, lab, req.reviewer)
        except KeyError:
            raise HTTPException(404, "unknown accession") from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @r.get("/review/stats")
    def review_stats():
        return _review_store().stats()

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
