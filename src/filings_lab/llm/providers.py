"""Extractors: every system compared in Phase 3 implements ``Extractor.extract``.

1. ``RuleExtractor``: regex and lexicon baseline, free and deterministic.
2. ``AnthropicExtractor``: the frontier model and teacher, Claude with structured outputs (schema enforced
   by the API), server side refusal fallback, effort control and token based cost accounting.
3. ``AnthropicBatchLabeler``: the same request through the Message Batches API at half price, for
   labelling thousands of training examples.
4. ``OpenAICompatExtractor``: any OpenAI compatible server, in practice vLLM serving the base model or the
   LoRA fine tuned adapter. Constrained decoding is optional so raw JSON validity can be measured.
5. ``HFExtractor``: local transformers generation with an optional PEFT adapter, for offline evaluation.
6. ``LabelExtractor``: replays known labels with optional noise (tests, simulated teachers).
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass

from .prompts import SYSTEM_PROMPT, ExtractionInput, chat_messages, user_message
from .schema import FilingSignals, Quote, json_schema, parse

FRONTIER_MODEL = os.environ.get("FLAB_FRONTIER_MODEL", "claude-opus-5-5")
# USD per million tokens: input, output, cache read. Cache writes bill at 1.25x input; batches at 50%.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}


@dataclass
class ExtractionResult:
    system: str
    accession: str
    raw: str | None
    signals: dict | None
    json_valid: bool
    schema_valid: bool
    error: str | None = None
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _result(system: str, inp: ExtractionInput, raw: str | None, latency: float, **usage) -> ExtractionResult:
    sig, json_ok, err = parse(raw) if raw is not None else (None, False, usage.pop("error", "no output"))
    usage.pop("error", None)
    return ExtractionResult(system, inp.accession, raw, sig.model_dump() if sig else None, json_ok, sig is not None,
                            err, latency, **usage)


def token_cost(model: str, input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0,
               batch: bool = False) -> float:
    if model not in PRICES:
        return 0.0
    pin, pout, pread = PRICES[model]
    usd = (input_tokens * pin + output_tokens * pout + cache_read * pread + cache_write * pin * 1.25) / 1e6
    return usd * (0.5 if batch else 1.0)


class Extractor:
    name = "extractor"

    def extract(self, inp: ExtractionInput) -> ExtractionResult:  # pragma: no cover - interface
        raise NotImplementedError


# ------------------------------------------------------------------------------------------------ rules
class RuleExtractor(Extractor):
    name = "rules"

    def __init__(self):
        from ..features.lexicon import Lexicon, tokenize
        from ..features.text_features import GOING_CONCERN, GUIDANCE_DOWN, GUIDANCE_FLAT, GUIDANCE_UP

        self.lex, self.tok = Lexicon(), tokenize
        self.rx = {"up": GUIDANCE_UP, "down": GUIDANCE_DOWN, "flat": GUIDANCE_FLAT, "gc": GOING_CONCERN}
        self.lit = re.compile(r"class action|subpoena|verdict|lawsuit was filed|investigation", re.I)

    def extract(self, inp: ExtractionInput) -> ExtractionResult:
        t0 = time.perf_counter()
        quotes: list[Quote] = []
        hits = {k: False for k in ("up", "down", "flat", "gc", "lit")}
        for i, p in enumerate(inp.paragraphs):
            for sent in re.split(r"(?<=[.!?])\s+", p["text"]):
                for k, rx in [*self.rx.items(), ("lit", self.lit)]:
                    if rx.search(sent) and not hits[k]:
                        hits[k] = True
                        if len(quotes) < 3:
                            quotes.append(Quote(paragraph=i + 1, text=sent[:300]))
        guidance = "raised" if hits["up"] else "lowered" if hits["down"] else "maintained" if hits["flat"] else "none"
        tone_text = " ".join(p["text"] for p in inp.paragraphs if p["section"] in ("mdna", "8k_results_of_operations"))
        net = self.lex.score(self.tok(tone_text or " ".join(p["text"] for p in inp.paragraphs)))["lex_net_tone"]
        tone = "positive" if net > 0.2 else "negative" if net < -0.2 else "neutral"
        sig = FilingSignals(guidance_direction=guidance, management_tone=tone, liquidity_concern=hits["gc"],
                            litigation_risk=hits["lit"], key_quotes=quotes,
                            summary=f"Rule based extraction: guidance {guidance}, tone {tone}.")
        return _result(self.name, inp, sig.model_dump_json(), time.perf_counter() - t0)


# --------------------------------------------------------------------------------------------- frontier
class AnthropicExtractor(Extractor):
    """Claude as frontier model and teacher.

    Structured outputs (``output_config.format``) make the API return JSON that matches the schema, so the
    frontier system's JSON validity is guaranteed by construction; the eval harness still checks it.
    ``fallbacks="default"`` (beta ``server-side-fallback-2026-07-01``) re-runs a request declined by a
    safety classifier on Anthropic's recommended fallback model instead of failing the label.
    """

    name = "frontier"

    def __init__(self, model: str = FRONTIER_MODEL, effort: str = "medium", max_tokens: int = 16000,
                 use_fallbacks: bool = True, client=None):
        import anthropic

        self.model, self.effort, self.max_tokens, self.use_fallbacks = model, effort, max_tokens, use_fallbacks
        self.client = client or anthropic.Anthropic()
        self.name = f"frontier:{model}"

    def request_params(self, inp: ExtractionInput) -> dict:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            # identical across requests, so it is cacheable once it exceeds the model's minimum cache prefix
            "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user_message(inp)}],
            "output_config": {"effort": self.effort, "format": {"type": "json_schema", "schema": json_schema()}},
        }

    def extract(self, inp: ExtractionInput) -> ExtractionResult:
        import anthropic

        t0 = time.perf_counter()
        try:
            if self.use_fallbacks:
                resp = self.client.beta.messages.create(
                    **self.request_params(inp), betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            else:
                resp = self.client.messages.create(**self.request_params(inp))
        except anthropic.BadRequestError as exc:
            return _result(self.name, inp, None, time.perf_counter() - t0, error=f"bad request: {exc.message}")
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APIStatusError) as exc:
            # the SDK already retried 429, 5xx and connection errors with backoff
            return _result(self.name, inp, None, time.perf_counter() - t0, error=f"api: {type(exc).__name__}")
        latency = time.perf_counter() - t0
        return self.from_message(inp, resp, latency, batch=False)

    def from_message(self, inp: ExtractionInput, resp, latency: float, batch: bool) -> ExtractionResult:
        u = resp.usage
        usage = {
            "input_tokens": u.input_tokens or 0, "output_tokens": u.output_tokens or 0,
            "cache_read_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_write_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        }
        usage["cost_usd"] = token_cost(self.model, usage["input_tokens"], usage["output_tokens"],
                                       usage["cache_read_tokens"], usage["cache_write_tokens"], batch=batch)
        if resp.stop_reason == "refusal":  # check before reading content
            cat = getattr(getattr(resp, "stop_details", None), "category", None)
            return _result(self.name, inp, None, latency, error=f"refusal: {cat}", **usage)
        if resp.stop_reason == "max_tokens":
            return _result(self.name, inp, None, latency, error="max_tokens", **usage)
        text = next((b.text for b in resp.content if b.type == "text"), None)
        return _result(self.name, inp, text, latency, **usage)


class AnthropicBatchLabeler:
    """Teacher labelling through the Message Batches API (50% price, results within 24 hours).

    Server side fallbacks are not accepted on the Batches API, so refused or failed items come back with an
    error and can be re run through ``AnthropicExtractor`` (which has the fallback) afterwards.
    """

    def __init__(self, extractor: AnthropicExtractor | None = None):
        self.ex = extractor or AnthropicExtractor(use_fallbacks=False)

    def submit(self, inputs: list[ExtractionInput]) -> list[str]:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        ids = []
        for start in range(0, len(inputs), 50_000):
            reqs = [Request(custom_id=inp.accession, params=MessageCreateParamsNonStreaming(**self.ex.request_params(inp)))
                    for inp in inputs[start:start + 50_000]]
            ids.append(self.ex.client.messages.batches.create(requests=reqs).id)
        return ids

    def wait(self, batch_id: str, poll_s: int = 60) -> None:
        while self.ex.client.messages.batches.retrieve(batch_id).processing_status != "ended":
            time.sleep(poll_s)

    def collect(self, batch_id: str, inputs: dict[str, ExtractionInput]) -> list[ExtractionResult]:
        out = []
        for r in self.ex.client.messages.batches.results(batch_id):  # results arrive in any order
            inp = inputs[r.custom_id]
            if r.result.type == "succeeded":
                out.append(self.ex.from_message(inp, r.result.message, 0.0, batch=True))
            else:
                out.append(_result(self.ex.name, inp, None, 0.0, error=f"batch: {r.result.type}"))
        return out


# --------------------------------------------------------------------------------------- self hosted
class OpenAICompatExtractor(Extractor):
    """vLLM (or any OpenAI compatible server). ``guided=True`` uses vLLM's JSON schema constrained decoding."""

    def __init__(self, base_url: str, model: str, name: str | None = None, guided: bool = False,
                 max_tokens: int = 768, timeout: float = 120.0, session=None, style: str = "full"):
        import requests

        self.base_url, self.model, self.guided, self.style = base_url.rstrip("/"), model, guided, style
        self.max_tokens, self.timeout = max_tokens, timeout
        self.session = session or requests.Session()
        self.name = name or f"vllm:{model}{':guided' if guided else ''}"

    def extract(self, inp: ExtractionInput) -> ExtractionResult:
        body = {"model": self.model, "messages": chat_messages(inp, style=self.style), "temperature": 0.0, "max_tokens": self.max_tokens}
        if self.guided:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "filing_signals", "schema": json_schema(), "strict": True}}
        t0 = time.perf_counter()
        try:
            r = self.session.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:  # noqa: BLE001
            return _result(self.name, inp, None, time.perf_counter() - t0, error=f"http: {exc}")
        usage = data.get("usage", {})
        return _result(self.name, inp, data["choices"][0]["message"]["content"], time.perf_counter() - t0,
                       input_tokens=usage.get("prompt_tokens", 0), output_tokens=usage.get("completion_tokens", 0))


class HFExtractor(Extractor):
    """Local transformers inference (greedy), optionally with a LoRA adapter on top of the base model."""

    def __init__(self, model_path: str, adapter_path: str | None = None, max_new_tokens: int = 512, name: str | None = None,
                 style: str = "full"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_path)
        from .train_lora import _dtype_kwarg

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_path, **{_dtype_kwarg(): dtype},
                                                     device_map="auto" if torch.cuda.is_available() else None)
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model.eval()
        self.max_new_tokens, self.style = max_new_tokens, style
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.name = name or f"hf:{model_path.rstrip('/').split('/')[-1]}{'+lora' if adapter_path else ''}"

    def extract(self, inp: ExtractionInput) -> ExtractionResult:
        import torch

        ids = self.tok.apply_chat_template(chat_messages(inp, style=self.style), add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True)
        ids = {k: v.to(self.model.device) for k, v in ids.items()}
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        new = out[0, ids["input_ids"].shape[1]:]
        text = self.tok.decode(new, skip_special_tokens=True)
        return _result(self.name, inp, text, time.perf_counter() - t0,
                       input_tokens=int(ids["input_ids"].shape[1]), output_tokens=int(new.shape[0]))

    def extract_many(self, inputs: list[ExtractionInput], batch_size: int = 16) -> list[ExtractionResult]:
        """Batched greedy generation (left padded), several times faster than one at a time."""
        import torch

        self.tok.padding_side = "left"
        out: list[ExtractionResult] = []
        for i in range(0, len(inputs), batch_size):
            chunk = inputs[i:i + batch_size]
            prompts = [self.tok.apply_chat_template(chat_messages(x, style=self.style), add_generation_prompt=True, tokenize=False)
                       for x in chunk]
            enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id, eos_token_id=self.tok.eos_token_id)
            per = (time.perf_counter() - t0) / len(chunk)
            for k, x in enumerate(chunk):
                new = gen[k, enc["input_ids"].shape[1]:]
                text = self.tok.decode(new, skip_special_tokens=True)
                out.append(_result(self.name, x, text, per, input_tokens=int(enc["attention_mask"][k].sum()),
                                   output_tokens=int((new != self.tok.pad_token_id).sum())))
        return out


# ------------------------------------------------------------------------------------------- replay
class LabelExtractor(Extractor):
    """Replays known labels. With ``noise > 0`` each categorical field is resampled with that probability,
    which simulates an imperfect teacher or annotator for testing the harness."""

    def __init__(self, labels: dict[str, dict], name: str = "labels", noise: float = 0.0, seed: int = 0):
        self.labels, self.name, self.noise = labels, name, noise
        self.rng = random.Random(seed)

    def extract(self, inp: ExtractionInput) -> ExtractionResult:
        lab = dict(self.labels.get(inp.accession) or {})
        if not lab:
            return _result(self.name, inp, None, 0.0, error="no label")
        if self.noise:
            choices = {"guidance_direction": ["raised", "lowered", "maintained", "none"],
                       "management_tone": ["positive", "neutral", "negative"],
                       "liquidity_concern": [True, False], "litigation_risk": [True, False]}
            for f, opts in choices.items():
                if self.rng.random() < self.noise:
                    lab[f] = self.rng.choice(opts)
        return _result(self.name, inp, json.dumps(lab), 0.0)
