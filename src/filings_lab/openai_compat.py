"""A small client for any OpenAI compatible chat API.

The lab's hosted model paths were written for one paid provider. This client lets the agent and the labelling
run against a free one instead: NVIDIA's hosted endpoints by default (a free key from build.nvidia.com), or
any server that speaks ``POST /chat/completions``, vLLM included. Free tiers allow about 40 requests a minute
and answer 429 above that, so the client spaces its requests and backs off.

Settings, from the environment or a local ``.env``:
``NVIDIA_API_KEY`` (or ``FLAB_OPENAI_API_KEY``), ``FLAB_OPENAI_BASE_URL``, ``FLAB_OPENAI_MODEL``.
"""
from __future__ import annotations

import os
import time

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"
RETRY_STATUS = (429, 500, 502, 503, 504)


class ChatError(RuntimeError):
    pass


def api_key() -> str | None:
    return os.environ.get("FLAB_OPENAI_API_KEY") or os.environ.get("NVIDIA_API_KEY") or None


def default_model() -> str:
    return os.environ.get("FLAB_OPENAI_MODEL", DEFAULT_MODEL)


class ChatClient:
    def __init__(self, base_url: str | None = None, key: str | None = None, rpm: float = 30.0, session=None,
                 timeout: float = 120.0, max_retries: int = 6, sleep=time.sleep):
        import requests

        from .edgar.client import RateLimiter

        self.base_url = (base_url or os.environ.get("FLAB_OPENAI_BASE_URL", NIM_BASE_URL)).rstrip("/")
        self.key = key or api_key()
        self.session = session or requests.Session()
        self.limiter = RateLimiter(rpm / 60.0) if rpm > 0 else None
        self.timeout, self.max_retries, self.sleep = timeout, max_retries, sleep
        self.requests = 0

    def complete(self, **body) -> dict:
        """One chat completion. Retries a rate limit or a server error, honouring ``Retry-After`` when given."""
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        delay = 2.0
        for attempt in range(self.max_retries):
            if self.limiter:
                self.limiter.acquire()
            self.requests += 1
            resp = self.session.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in RETRY_STATUS and attempt < self.max_retries - 1:
                try:
                    wait = float(resp.headers.get("Retry-After", delay))
                except (TypeError, ValueError):
                    wait = delay
                self.sleep(min(max(wait, 0.0), 60.0))
                delay = min(delay * 2, 60.0)
                continue
            raise ChatError(f"{resp.status_code} from {self.base_url}: {str(getattr(resp, 'text', ''))[:300]}")
        raise ChatError("unreachable")
