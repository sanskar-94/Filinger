"""The analyst agent.

``ClaudeAgent`` runs a tool use loop with Claude. Design choices:
1. Manual loop rather than the SDK tool runner, because the lab needs per request provenance (every
   paragraph id a tool returned), a citation repair turn, iteration caps and cost accounting in one place,
   and a client that tests can replace.
2. ``strict`` tool schemas with ``tool_choice`` left at auto (forcing a tool is rejected on this model);
   the system prompt steers tool use.
3. The full ``response.content`` is appended every turn, so thinking blocks are replayed unchanged.
4. Server side refusal fallback (``fallbacks="default"``) and explicit ``stop_reason`` handling.

``ExtractiveAgent`` answers without any API: retrieval plus quoted sentences with citations. It is the
offline demo mode and the fallback when the API is unavailable.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass, field

from ..llm.providers import token_cost
from .store import edgar_url
from .tools import TOOL_DEFS, ToolBox

CITE = re.compile(r"\[cite:([0-9a-f]{16})\]")
AGENT_MODEL = os.environ.get("FLAB_AGENT_MODEL", "claude-opus-5-5")

SYSTEM_PROMPT = """You are an equity research assistant with access to a company's SEC filings (10-K, 10-Q, 8-K), daily prices and the lab's model outputs, all through tools.

How to work:
1. Use the tools to find evidence before answering. Search with filters when the question names a company, form or period, and read neighbouring paragraphs when context matters.
2. Every statement that comes from a filing must be followed immediately by a citation in the form [cite:<paragraph_id>], using only paragraph ids that a tool returned in this conversation. Never invent or alter an id.
3. Numbers from get_event_return, get_price_history or get_model_signal are cited as (market data) or (model output) instead.
4. If the filings do not answer the question, say so plainly rather than guessing.
5. Keep answers short and specific: lead with the answer, then the supporting points. This is research support, not investment advice."""


@dataclass
class Citation:
    paragraph_id: str
    ticker: str | None
    company: str | None
    form: str
    date_filed: str | None
    section: str
    accession: str
    url: str
    snippet: str


@dataclass
class AgentAnswer:
    answer: str
    citations: list[Citation]
    invalid_citations: list[str]
    tool_calls: list[dict]
    mode: str
    turns: int = 0
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    stop_reason: str | None = None
    error: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def citation_valid_rate(self) -> float:
        n = len(self.citations) + len(self.invalid_citations)
        return 1.0 if n == 0 else len(self.citations) / n

    def to_dict(self) -> dict:
        d = asdict(self)
        d["citation_valid_rate"] = self.citation_valid_rate
        return d


def build_citations(text: str, toolbox: ToolBox) -> tuple[list[Citation], list[str]]:
    ids = list(dict.fromkeys(CITE.findall(text)))
    valid, invalid = [], []
    by_id = toolbox.store.by_id
    for pid in ids:
        if pid not in toolbox.seen_ids or pid not in by_id.index:
            invalid.append(pid)
            continue
        r = by_id.loc[pid]
        valid.append(Citation(pid, r.get("ticker"), r.get("company"), str(r["form"]),
                              r["date_filed"].strftime("%Y-%m-%d") if hasattr(r["date_filed"], "strftime") else None,
                              r["section"], r["accession"], edgar_url(int(r["cik"]), r["accession"], r.get("primary_document")),
                              r["text"][:400]))
    return valid, invalid


class ClaudeAgent:
    def __init__(self, toolbox: ToolBox, model: str = AGENT_MODEL, effort: str = "high", max_turns: int = 8,
                 client=None, use_fallbacks: bool = True):
        import anthropic

        self.tb, self.model, self.effort, self.max_turns = toolbox, model, effort, max_turns
        self.client = client or anthropic.Anthropic()
        self.use_fallbacks = use_fallbacks
        self.tools = [{**t, "strict": True} for t in TOOL_DEFS]

    def _create(self, messages):
        params = {
            "model": self.model, "max_tokens": 16000, "tools": self.tools, "messages": messages,
            "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            "output_config": {"effort": self.effort},
        }
        if self.use_fallbacks:
            return self.client.beta.messages.create(**params, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        return self.client.messages.create(**params)

    def ask(self, question: str, ticker: str | None = None) -> AgentAnswer:
        import anthropic

        t0 = time.perf_counter()
        self.tb.reset()
        content = question if not ticker else f"{question}\n\n(Company in focus: ticker {ticker.upper()})"
        messages: list[dict] = [{"role": "user", "content": content}]
        calls: list[dict] = []
        usage = {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0}
        repaired, turns, final_text, stop = False, 0, "", None
        try:
            while turns < self.max_turns:
                turns += 1
                resp = self._create(messages)
                u = resp.usage
                usage["in"] += u.input_tokens or 0
                usage["out"] += u.output_tokens or 0
                usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
                usage["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
                stop = resp.stop_reason
                if stop == "refusal":
                    final_text = "The request was declined by the model's safety system, so no answer was produced."
                    break
                messages.append({"role": "assistant", "content": resp.content})  # keep thinking blocks intact
                if stop == "pause_turn":
                    continue
                tool_uses = [b for b in resp.content if b.type == "tool_use"]
                if stop == "tool_use" and tool_uses:
                    results = []
                    for b in tool_uses:  # all results go back in one user message
                        c0 = time.perf_counter()
                        try:
                            out, err = self.tb.execute(b.name, dict(b.input)), None
                        except Exception as exc:  # noqa: BLE001 - report to the model, let it recover
                            out, err = f"Error: {exc}", str(exc)
                        calls.append({"tool": b.name, "input": dict(b.input), "ms": round((time.perf_counter() - c0) * 1000, 1),
                                      "error": err})
                        results.append({"type": "tool_result", "tool_use_id": b.id, "content": out, **({"is_error": True} if err else {})})
                    messages.append({"role": "user", "content": results})
                    continue
                final_text = "".join(b.text for b in resp.content if b.type == "text")
                _, invalid = build_citations(final_text, self.tb)
                if invalid and not repaired and turns < self.max_turns:
                    repaired = True
                    messages.append({"role": "user", "content": (
                        "These citations were not returned by any tool in this conversation: "
                        + ", ".join(invalid) + ". Rewrite the answer citing only retrieved paragraphs; search again if needed.")})
                    continue
                break
        except anthropic.APIError as exc:
            return AgentAnswer("", [], [], calls, f"claude:{self.model}", turns, time.perf_counter() - t0, error=f"api: {exc}")
        valid, invalid = build_citations(final_text, self.tb)
        return AgentAnswer(final_text, valid, invalid, calls, f"claude:{self.model}", turns, time.perf_counter() - t0,
                           usage["in"], usage["out"],
                           token_cost(self.model, usage["in"], usage["out"], usage["cache_read"], usage["cache_write"]),
                           stop, meta={"repaired": repaired})


class ExtractiveAgent:
    """No LLM: retrieve, quote the most relevant sentence of each hit, attach the market reaction."""

    PRICE_WORDS = re.compile(r"\b(react|reaction|stock|share price|return|move|moved|market)\b", re.I)
    SIGNAL_WORDS = re.compile(r"\b(guidance|outlook|tone|liquidity|going concern|covenant|litigation|lawsuit|investigation)\b", re.I)

    def __init__(self, toolbox: ToolBox, k: int = 4):
        self.tb, self.k = toolbox, k

    def _guess_ticker(self, question: str) -> str | None:
        comps = self.tb.store.companies()
        for r in comps.itertuples(index=False):
            if re.search(rf"\b{re.escape(r.ticker)}\b", question) or r.company.lower().split(" inc")[0] in question.lower():
                return r.ticker
        return None

    def ask(self, question: str, ticker: str | None = None) -> AgentAnswer:
        import json

        t0 = time.perf_counter()
        self.tb.reset()
        ticker = ticker or self._guess_ticker(question)
        args = {"query": question, "ticker": ticker, "form": None, "date_from": None, "date_to": None, "k": self.k}
        res = json.loads(self.tb.execute("search_filings", args))["results"]
        calls = [{"tool": "search_filings", "input": args}]
        if not res:
            return AgentAnswer("No filing paragraphs matched the question.", [], [], calls, "offline-extractive",
                               1, time.perf_counter() - t0)
        terms = {t for t in re.findall(r"[a-z]{4,}", question.lower())}
        lines = [f"Most relevant passages{f' for {ticker}' if ticker else ''}:"]
        for n, r in enumerate(res, start=1):
            sents = re.split(r"(?<=[.!?])\s+", r["text"])
            best = max(sents, key=lambda s: len(terms & set(re.findall(r"[a-z]{4,}", s.lower()))))
            lines.append(f"{n}. {r['form']} filed {r['date_filed']} ({r['section']}): \"{best.strip()}\" [cite:{r['paragraph_id']}]")
        if self.SIGNAL_WORDS.search(question) and self.tb.extractor is not None:
            sig = json.loads(self.tb.execute("extract_signals", {"accession": res[0]["accession"]}))
            calls.append({"tool": "extract_signals", "input": {"accession": res[0]["accession"]}})
            if "signals" in sig:
                v = sig["signals"]
                flags = [n for n, on in (("liquidity concern", v.get("liquidity_concern")), ("litigation risk", v.get("litigation_risk"))) if on]
                lines.append(f"Extracted signals for the top filing ({sig['model']}): guidance {v.get('guidance_direction')}, "
                             f"tone {v.get('management_tone')}" + (f", flags: {', '.join(flags)}" if flags else ", no liquidity or litigation flags")
                             + " (model output).")
        if self.PRICE_WORDS.search(question):
            ev = json.loads(self.tb.execute("get_event_return", {"accession": res[0]["accession"]}))
            calls.append({"tool": "get_event_return", "input": {"accession": res[0]["accession"]}})
            if "car_0_2" in ev and ev["car_0_2"] is not None:
                lines.append(f"Market reaction to the top filing: {ev['car_0_2'] * 100:+.2f}% abnormal return over 3 days "
                             f"starting {ev['event_date']} (market data).")
        text = "\n".join(lines)
        valid, invalid = build_citations(text, self.tb)
        return AgentAnswer(text, valid, invalid, calls, "offline-extractive", 1, time.perf_counter() - t0)
