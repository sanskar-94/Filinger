"""FastAPI service in front of the extraction LLM (vLLM) and the reaction model.

Routes:
  GET  /healthz                    liveness
  GET  /readyz                     readiness (model loaded)
  GET  /metrics                    Prometheus exposition, drift gauges refreshed on scrape
  GET  /v1/model                   active model version, feature set, training window, metrics
  POST /v1/extract                 filing paragraphs -> structured signals (vLLM, rules fallback)
  POST /v1/score                   accession or feature vector -> P(positive reaction) + top drivers
  GET  /v1/filings/{accession}     everything known about one filing: features, score, cached signals

Resilience: the LLM call sits behind a circuit breaker; on error, invalid JSON or an open breaker the rule
extractor answers and the response says so (``fallback: true``), so clients never see a 500 because a GPU
pod is restarting.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from ..llm.prompts import ExtractionInput, select_paragraphs
from ..llm.providers import Extractor, OpenAICompatExtractor, RuleExtractor
from .drift import DriftMonitor


# ------------------------------------------------------------------------------------------------ config
@dataclass
class ServiceConfig:
    model_uri: str = field(default_factory=lambda: os.environ.get("FLAB_MODEL_URI", "models:/filings-reaction@champion"))
    features_path: str = field(default_factory=lambda: os.environ.get("FLAB_FEATURES", "data/lake/gold/features"))
    signals_path: str = field(default_factory=lambda: os.environ.get("FLAB_SIGNALS", "data/lake/gold/llm_signals.parquet"))
    llm_base_url: str = field(default_factory=lambda: os.environ.get("FLAB_LLM_BASE_URL", ""))
    llm_model: str = field(default_factory=lambda: os.environ.get("FLAB_LLM_MODEL", "filings-extractor"))
    api_keys: frozenset = field(default_factory=lambda: frozenset(
        k for k in os.environ.get("FLAB_API_KEYS", "").split(",") if k))
    breaker_failures: int = 5
    breaker_cooldown_s: float = 30.0
    max_input_chars: int = 12_000


class CircuitBreaker:
    def __init__(self, failures: int, cooldown_s: float):
        self.failures, self.cooldown, self.count, self.opened_at = failures, cooldown_s, 0, None

    @property
    def open(self) -> bool:
        if self.opened_at is None:
            return False
        if time.monotonic() - self.opened_at > self.cooldown:  # half open: let one request through
            self.opened_at, self.count = None, self.failures - 1
            return False
        return True

    def record(self, ok: bool) -> None:
        if ok:
            self.count = 0
            return
        self.count += 1
        if self.count >= self.failures:
            self.opened_at = time.monotonic()


class OfflineFeatureStore:
    """Accession -> feature dict from the gold Parquet tables. In production this interface is backed by a
    low latency store (DynamoDB or Redis) filled by the same Spark job."""

    def __init__(self, features_path: str | None, signals_path: str | None = None):
        self.df = pd.DataFrame()
        if features_path and os.path.exists(features_path):
            self.df = pd.read_parquet(features_path).set_index("accession")
            if signals_path and os.path.exists(signals_path):
                self.df = self.df.join(pd.read_parquet(signals_path).set_index("accession"), how="left")

    def get(self, accession: str) -> dict | None:
        if accession not in self.df.index:
            return None
        row = self.df.loc[accession]
        return {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in row.items()}


# ------------------------------------------------------------------------------------------------ schemas
class Paragraph(BaseModel):
    paragraph_id: str | None = None
    section: str = "unknown"
    text: str


class ExtractRequest(BaseModel):
    accession: str = "adhoc"
    form: str = "10-K"
    company: str = ""
    paragraphs: list[Paragraph] = Field(default_factory=list)
    text: str | None = Field(default=None, description="raw text, split on blank lines when paragraphs are not given")


class ScoreRequest(BaseModel):
    accession: str | None = None
    features: dict[str, float | int | None] | None = None


# ------------------------------------------------------------------------------------------------ metrics
class Metrics:
    def __init__(self):
        r = self.registry = CollectorRegistry()
        self.requests = Counter("flab_requests_total", "HTTP requests", ["route", "status"], registry=r)
        self.latency = Histogram("flab_request_latency_seconds", "Request latency", ["route"], registry=r,
                                 buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30))
        self.extractions = Counter("flab_extractions_total", "Extraction outcomes", ["system", "outcome"], registry=r)
        self.predictions = Counter("flab_predictions_total", "Scored filings", ["model_version"], registry=r)
        self.score = Histogram("flab_prediction_score", "Predicted probability", registry=r,
                               buckets=tuple(np.round(np.linspace(0.05, 0.95, 19), 2)))
        self.psi = Gauge("flab_feature_psi", "Population stability index vs training", ["feature"], registry=r)
        self.ks_p = Gauge("flab_feature_ks_pvalue", "KS test p value vs training", ["feature"], registry=r)
        self.model_info = Gauge("flab_model_info", "Active model", ["version", "feature_set", "trained_through"], registry=r)
        self.breaker = Gauge("flab_llm_breaker_open", "1 when the LLM circuit breaker is open", registry=r)


# ------------------------------------------------------------------------------------------------ app
def create_app(config: ServiceConfig | None = None, model=None, extractor: Extractor | None = None,
               feature_store: OfflineFeatureStore | None = None) -> FastAPI:
    cfg = config or ServiceConfig()
    app = FastAPI(title="Filings Intelligence Lab API", version="1.0.0")
    m = Metrics()
    state = {"model": model, "monitor": None}
    if state["model"] is None:
        try:
            from .model import ReactionModel
            from .registry import get_registry

            state["model"] = ReactionModel.load(get_registry().resolve(cfg.model_uri))
        except Exception as exc:  # noqa: BLE001 - readiness reports it
            state["load_error"] = str(exc)
    if state["model"] is not None:
        mdl = state["model"]
        state["monitor"] = DriftMonitor(mdl.reference)
        m.model_info.labels(mdl.version, mdl.meta.get("feature_set", ""), mdl.meta.get("trained_through", "")).set(1)
    store = feature_store or OfflineFeatureStore(cfg.features_path, cfg.signals_path)
    primary = extractor or (OpenAICompatExtractor(cfg.llm_base_url, cfg.llm_model, name="llm", guided=True)
                            if cfg.llm_base_url else None)
    fallback = RuleExtractor()
    breaker = CircuitBreaker(cfg.breaker_failures, cfg.breaker_cooldown_s)

    def auth(x_api_key: str | None = Header(default=None)):
        if cfg.api_keys and x_api_key not in cfg.api_keys:
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    @app.middleware("http")
    async def observe(request: Request, call_next):
        route = request.scope.get("route")
        t0 = time.perf_counter()
        response = await call_next(request)
        path = getattr(request.scope.get("route"), "path", None) or (route.path if route else request.url.path)
        m.requests.labels(path, str(response.status_code)).inc()
        m.latency.labels(path).observe(time.perf_counter() - t0)
        return response

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz():
        if state["model"] is None:
            raise HTTPException(503, detail=f"model not loaded: {state.get('load_error', 'unknown')}")
        return {"status": "ready", "model_version": state["model"].version}

    @app.get("/metrics")
    def metrics():
        if state["monitor"] is not None:
            for feat, rep in state["monitor"].report().items():
                m.psi.labels(feat).set(rep["psi"])
                if "ks_p" in rep and not np.isnan(rep["ks_p"]):
                    m.ks_p.labels(feat).set(rep["ks_p"])
        m.breaker.set(1.0 if breaker.open else 0.0)
        return Response(generate_latest(m.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/model", dependencies=[Depends(auth)])
    def model_info():
        if state["model"] is None:
            raise HTTPException(503, "model not loaded")
        meta = state["model"].meta
        return {k: meta.get(k) for k in ("version", "feature_set", "trained_through", "n_train", "num_rounds")} | {
            "n_features": len(state["model"].features), "metrics": meta.get("metrics", {}).get("summary", {})}

    @app.post("/v1/extract", dependencies=[Depends(auth)])
    def extract(req: ExtractRequest):
        paras = [p.model_dump() for p in req.paragraphs]
        if not paras and req.text:
            paras = [{"paragraph_id": None, "section": "unknown", "text": t.strip()}
                     for t in req.text.split("\n\n") if t.strip()]
        if not paras:
            raise HTTPException(422, "provide paragraphs or text")
        inp = ExtractionInput(req.accession, req.form, req.company, select_paragraphs(paras, cfg.max_input_chars))
        result, used_fallback = None, False
        if primary is not None and not breaker.open:
            result = primary.extract(inp)
            breaker.record(result.schema_valid)
            m.extractions.labels(primary.name, "valid" if result.schema_valid else "invalid" if result.json_valid else "error").inc()
        if result is None or not result.schema_valid:
            used_fallback = primary is not None
            result = fallback.extract(inp)
            m.extractions.labels(fallback.name, "fallback" if used_fallback else "valid").inc()
        if state["monitor"] is not None and result.signals:
            state["monitor"].observe({"guidance_direction": result.signals["guidance_direction"]})
        return {"accession": req.accession, "system": result.system, "fallback": used_fallback,
                "signals": result.signals, "latency_s": round(result.latency_s, 4),
                "paragraph_ids": [p.get("paragraph_id") for p in inp.paragraphs]}

    def _score(features: dict) -> dict:
        mdl = state["model"]
        if mdl is None:
            raise HTTPException(503, "model not loaded")
        X = pd.DataFrame([features]).reindex(columns=mdl.features).apply(pd.to_numeric, errors="coerce")
        p = float(mdl.booster.predict(X)[0])
        contrib = mdl.booster.predict(X, pred_contrib=True)[0][:-1]
        top = np.argsort(-np.abs(contrib))[:5]
        m.predictions.labels(mdl.version).inc()
        m.score.observe(p)
        if state["monitor"] is not None:
            state["monitor"].observe({**features, "prediction": p})
        return {"probability_up": p, "model_version": mdl.version,
                "drivers": [{"feature": mdl.features[i], "value": features.get(mdl.features[i]),
                             "contribution": float(contrib[i])} for i in top],
                "missing_features": int(X.isna().sum(axis=1).iloc[0])}

    @app.post("/v1/score", dependencies=[Depends(auth)])
    def score(req: ScoreRequest):
        feats = req.features
        if feats is None and req.accession:
            feats = store.get(req.accession)
            if feats is None:
                raise HTTPException(404, f"unknown accession {req.accession}")
        if feats is None:
            raise HTTPException(422, "provide accession or features")
        return {"accession": req.accession, **_score(feats)}

    @app.get("/v1/filings/{accession}", dependencies=[Depends(auth)])
    def filing(accession: str):
        feats = store.get(accession)
        if feats is None:
            raise HTTPException(404, f"unknown accession {accession}")
        signals = {k: v for k, v in feats.items() if k.startswith("llm_")}
        keep = ("cik", "ticker", "form", "event_date", "car", "label_up")
        return {"accession": accession, **{k: _jsonable(feats.get(k)) for k in keep},
                "signals": {k: _jsonable(v) for k, v in signals.items()}, "score": _score(feats)}

    app.state.config, app.state.metrics, app.state.breaker = cfg, m, breaker
    return app


def _jsonable(v):
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v
