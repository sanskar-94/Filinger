"""Agent service: experiment assignment, per request tool box, agent run, exposure logging."""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import uuid
from typing import Any

from ..experiment.assign import Experiment
from ..experiment.log import EventLog
from ..openai_compat import api_key as openai_api_key
from .agent import AGENT_MODEL, ClaudeAgent, ExtractiveAgent
from .openai_agent import OpenAIChatAgent
from .retrieval import BM25Index, build_index
from .store import DataStore
from .tools import ToolBox


class AgentService:
    def __init__(self, store: DataStore, experiment: Experiment | None = None, log: EventLog | None = None,
                 mode: str = "auto", scorer=None, client=None):
        self.store = store
        self._bm25 = BM25Index(store.paragraphs)
        self.retrieval = os.environ.get("FLAB_RETRIEVAL", "bm25").lower()
        self._indexes: dict[str, Any] = {"bm25": self._bm25}
        self.index = self.index_for(self.retrieval)
        self.experiment, self.log, self.scorer, self.client = experiment, log, scorer, client
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
        # auto: the hosted model when its key is set, else a free OpenAI compatible API when that key is set, else offline
        auto = "claude" if (has_key or client is not None) else "openai" if openai_api_key() else "offline"
        self.mode = auto if mode == "auto" else mode
        self._extractors: dict[str, object] = {}
        self.serve_signals = True  # attach each variant's extractor to its tool box
        self.llm_base_url = os.environ.get("FLAB_LLM_BASE_URL", "")
        # shared by every replica (a Kubernetes secret) so a rating can be accepted by any of them
        self._signing_key = (os.environ.get("FLAB_REQUEST_SIGNING_KEY") or secrets.token_hex(32)).encode()

    def _signature(self, rid: str) -> str:
        return hmac.new(self._signing_key, rid.encode(), hashlib.sha256).hexdigest()[:24]

    def new_request_id(self) -> str:
        rid = uuid.uuid4().hex
        return f"{rid}.{self._signature(rid)}"

    def verify_request_id(self, request_id: str) -> bool:
        rid, _, sig = request_id.partition(".")
        return bool(rid and sig) and hmac.compare_digest(sig, self._signature(rid))

    def feedback(self, request_id: str, rating: int, comment: str | None = None, ts: float | None = None) -> bool:
        """Accept a rating for an answer this service issued, whichever replica served it."""
        if self.log is None:
            return False
        return self.log.feedback(request_id, rating, comment, ts=ts, known=self.verify_request_id(request_id))

    def extractor_for(self, variant: str, cfg: dict):
        """The model version a variant serves: its LoRA adapter on vLLM when configured, rules otherwise."""
        if variant not in self._extractors:
            from ..llm.providers import OpenAICompatExtractor, RuleExtractor

            model = cfg.get("extractor_model")
            if self.llm_base_url and model:
                self._extractors[variant] = OpenAICompatExtractor(self.llm_base_url, model, name=f"vllm:{model}", guided=True,
                                                                  style=os.environ.get("FLAB_LLM_PROMPT_STYLE", "full"))
            else:
                self._extractors[variant] = RuleExtractor()
        return self._extractors[variant]

    def index_for(self, mode: str):
        """The index for a retrieval mode, built once. A dense mode that cannot load (no embedding model in this
        image, no cache in the lake) falls back to BM25 and says so, so a missing model degrades search
        instead of taking the service down."""
        if mode not in self._indexes:
            try:
                self._indexes[mode] = build_index(self.store.paragraphs, mode, lake=self.store.lake, bm25=self._bm25)
            except Exception as exc:  # noqa: BLE001 - any failure to build the optional index
                print(f"retrieval mode {mode!r} unavailable ({exc}); serving bm25")
                self._indexes[mode] = self._bm25
        return self._indexes[mode]

    def toolbox_for(self, variant: str, cfg: dict) -> ToolBox:
        index = self.index_for(cfg["retrieval"]) if cfg.get("retrieval") else self.index
        return ToolBox(self.store, index, scorer=self.scorer, diversify=cfg.get("diversify", True),
                       extractor=self.extractor_for(variant, cfg) if self.serve_signals else None)

    @classmethod
    def from_env(cls, lake: str, scorer=None) -> AgentService:
        exp_path = os.environ.get("FLAB_EXPERIMENT", "configs/experiment.json")
        exp = Experiment.load(exp_path) if os.path.exists(exp_path) else None
        log = EventLog(os.environ.get("FLAB_EVENT_DB", "data/experiments.db"))
        return cls(DataStore(lake), exp, log, os.environ.get("FLAB_AGENT_MODE", "auto"), scorer)

    def variant_for(self, user_id: str) -> tuple[str, dict]:
        if self.experiment is None or not self.experiment.active:
            return "control", {}
        v = self.experiment.assign(user_id)
        return v, self.experiment.config(v)

    def ask(self, question: str, ticker: str | None = None, user_id: str = "anonymous", ts: float | None = None) -> dict:
        variant, cfg = self.variant_for(user_id)
        tb = self.toolbox_for(variant, cfg)
        if self.mode in ("claude", "openai"):
            if self.mode == "claude":
                ans = ClaudeAgent(tb, model=cfg.get("agent_model", AGENT_MODEL), client=self.client).ask(question, ticker)
            else:
                ans = OpenAIChatAgent(tb, model=cfg.get("openai_agent_model"), client=self.client).ask(question, ticker)
            if ans.error:  # API unavailable: answer offline rather than fail the analyst
                ans = ExtractiveAgent(tb, k=cfg.get("k", 4)).ask(question, ticker)
                ans.meta["fallback_from_error"] = True
        else:
            ans = ExtractiveAgent(tb, k=cfg.get("k", 4)).ask(question, ticker)
        request_id = self.new_request_id()
        # the configured model version, and how often it actually answered: a variant whose model keeps falling back
        # to rules is not testing that model, and the readout has to be able to see it
        extractions = {"n": len(tb.extraction_log), "fallbacks": sum(e["fallback"] for e in tb.extraction_log)}
        if self.log is not None:
            self.log.exposure(**({"ts": ts} if ts is not None else {}), request_id=request_id,
                              experiment=self.experiment.name if self.experiment else "none",
                              variant=variant, user_id=user_id, ticker=ticker, mode=ans.mode,
                              model_version=getattr(tb.extractor, "name", "rules"),
                              latency_s=ans.latency_s, n_citations=len(ans.citations),
                              citation_valid_rate=ans.citation_valid_rate, cost_usd=ans.cost_usd, error=ans.error,
                              n_extractions=extractions["n"], n_extraction_fallbacks=extractions["fallbacks"])
        return {"request_id": request_id, "variant": variant,
                "experiment": self.experiment.name if self.experiment else None, "extractions": extractions,
                **ans.to_dict()}
