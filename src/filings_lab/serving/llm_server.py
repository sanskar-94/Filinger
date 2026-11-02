"""OpenAI compatible chat completions server for local models: the development and CI stand in for vLLM.

It serves one base model plus any number of LoRA adapters under their own model names, the way vLLM's
``--lora-modules`` does, so the API, its circuit breaker and both A/B variants run unchanged against it. It
exports the vLLM metric names the Grafana dashboard and the KEDA scaler use (``vllm:num_requests_running``,
``vllm:num_requests_waiting``, ``vllm:prompt_tokens_total``, ``vllm:generation_tokens_total``), so the
monitoring path can be exercised without a GPU. Decoding is greedy; ``response_format`` is accepted and the
output is validated by the caller, as it would be without constrained decoding.

    flab llm-serve --base artifacts/tiny-base --adapter filings-extractor=artifacts/lora/tiny-champion/adapter
"""
from __future__ import annotations

import threading
import time
import uuid

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = 0.0
    response_format: dict | None = None


def create_llm_app(base_path: str, adapters: dict[str, str] | None = None, base_name: str = "base",
                   max_concurrency: int = 1) -> FastAPI:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_path)
    adapters = adapters or {}
    if adapters:
        from peft import PeftModel

        names = list(adapters)
        model = PeftModel.from_pretrained(model, adapters[names[0]], adapter_name=names[0])
        for n in names[1:]:
            model.load_adapter(adapters[n], adapter_name=n)
    model.eval()
    gate = threading.Semaphore(max_concurrency)

    reg = CollectorRegistry()
    running = Gauge("vllm:num_requests_running", "Requests being generated", registry=reg)
    waiting = Gauge("vllm:num_requests_waiting", "Requests queued for a generation slot", registry=reg)
    prompt_tokens = Counter("vllm:prompt_tokens", "Prompt tokens processed", ["model_name"], registry=reg)
    gen_tokens = Counter("vllm:generation_tokens", "Tokens generated", ["model_name"], registry=reg)
    latency = Histogram("vllm:e2e_request_latency_seconds", "End to end request latency", ["model_name"], registry=reg,
                        buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30))
    app = FastAPI(title="Local OpenAI compatible LLM server")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "filings-lab"} for m in [base_name, *adapters]]}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(reg), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        if req.model != base_name and req.model not in adapters:
            raise HTTPException(404, f"model {req.model} is not served; available: {[base_name, *adapters]}")
        prompt = tok.apply_chat_template([m.model_dump() for m in req.messages], add_generation_prompt=True, tokenize=False)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
        t0 = time.perf_counter()
        waiting.inc()
        with gate:
            waiting.dec()
            running.inc()
            try:
                with torch.no_grad():
                    if adapters and req.model == base_name:
                        with model.disable_adapter():
                            out = model.generate(**enc, max_new_tokens=req.max_tokens, do_sample=False,
                                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
                    else:
                        if adapters:
                            model.set_adapter(req.model)
                        out = model.generate(**enc, max_new_tokens=req.max_tokens, do_sample=False,
                                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            finally:
                running.dec()
        new = out[0, enc["input_ids"].shape[1]:]
        text = tok.decode(new, skip_special_tokens=True)
        n_in, n_out = int(enc["input_ids"].shape[1]), int(new.shape[0])
        prompt_tokens.labels(req.model).inc(n_in)
        gen_tokens.labels(req.model).inc(n_out)
        latency.labels(req.model).observe(time.perf_counter() - t0)
        finished = "length" if n_out >= req.max_tokens else "stop"
        return {"id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion", "created": int(time.time()),
                "model": req.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finished}],
                "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out}}

    return app
