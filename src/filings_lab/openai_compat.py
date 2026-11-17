"""A small client for any OpenAI compatible chat API.

The lab's hosted model paths were written for one paid provider. This client lets the agent and the labelling
run against a free one instead: NVIDIA's hosted endpoints by default (a free key from build.nvidia.com), or
any server that speaks ``POST /chat/completions``, vLLM included. Free tiers allow about 40 requests a minute
and answer 429 above that, so the client spaces its requests and backs off. The published limit is not the
whole story: the largest free model answered 429 for minutes at 17 requests a minute, with no ``Retry-After``.
So the pace adapts. A 429 seen by one thread holds all of them and halves the request rate, and the rate
climbs back a step for every twenty answers that follow.

Settings, from the environment or a local ``.env``:
``NVIDIA_API_KEY`` (or ``FLAB_OPENAI_API_KEY``), ``FLAB_OPENAI_BASE_URL``, ``FLAB_OPENAI_MODEL``.
"""
from __future__ import annotations

import os
import threading
import time

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
RETRY_STATUS = (429, 500, 502, 503, 504)


class ChatError(RuntimeError):
    pass


def api_key() -> str | None:
    return os.environ.get("FLAB_OPENAI_API_KEY") or os.environ.get("NVIDIA_API_KEY") or None


def default_model() -> str:
    return os.environ.get("FLAB_OPENAI_MODEL", DEFAULT_MODEL)


class ChatClient:
    def __init__(self, base_url: str | None = None, key: str | None = None, rpm: float = 30.0, session=None,
                 timeout: float = 120.0, max_retries: int = 10, sleep=time.sleep, min_rpm: float = 4.0):
        import requests

        from .edgar.client import RateLimiter

        self.base_url = (base_url or os.environ.get("FLAB_OPENAI_BASE_URL", NIM_BASE_URL)).rstrip("/")
        self.key = key or api_key()
        self.session = session or requests.Session()
        self.limiter = RateLimiter(rpm / 60.0) if rpm > 0 else None
        self.timeout, self.max_retries, self.sleep = timeout, max_retries, sleep
        self.requests = self.limited = 0  # requests sent, and how many of them the API answered with a rate limit
        self.base_rate, self.min_rate = rpm / 60.0, min_rpm / 60.0
        self._lock = threading.Lock()
        self._held_until, self._streak = 0.0, 0

    @property
    def rpm(self) -> float:
        """The pace in force now, in requests a minute (0 when requests are not spaced)."""
        return self.limiter.rate * 60.0 if self.limiter else 0.0

    def _take_turn(self, retrying: bool) -> None:
        if not retrying:  # a thread that is retrying has just slept its own wait
            with self._lock:
                gap = self._held_until - time.monotonic()
            if gap > 0:
                self.sleep(gap)
        if self.limiter:
            self.limiter.acquire()

    def _slow_down(self, wait: float) -> None:
        with self._lock:
            self.limited += 1
            self._streak = 0
            self._held_until = max(self._held_until, time.monotonic() + wait)
            if self.limiter:
                self.limiter.rate = max(self.limiter.rate / 2.0, self.min_rate)

    def _answered(self) -> None:
        with self._lock:
            self._streak += 1
            if self.limiter and self._streak % 20 == 0:
                self.limiter.rate = min(self.limiter.rate * 1.25, self.base_rate)

    def complete(self, **body) -> dict:
        """One chat completion. Retries a rate limit, a server error, a timeout or a dropped connection, honouring
        ``Retry-After`` when given."""
        import requests

        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        delay = 2.0
        for attempt in range(self.max_retries):
            self._take_turn(retrying=attempt > 0)
            self.requests += 1
            try:
                resp = self.session.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=self.timeout)
            except requests.RequestException as exc:  # a timeout or a dropped connection: the same remedy as a 5xx
                if attempt == self.max_retries - 1:
                    raise ChatError(f"no answer from {self.base_url}: {type(exc).__name__}") from exc
                self.sleep(delay)
                delay = min(delay * 2, 60.0)
                continue
            if resp.status_code == 200:
                self._answered()
                return resp.json()
            if resp.status_code in RETRY_STATUS and attempt < self.max_retries - 1:
                try:
                    wait = float(resp.headers.get("Retry-After", delay))
                except (TypeError, ValueError):
                    wait = delay
                wait = min(max(wait, 0.0), 60.0)
                if resp.status_code == 429:
                    self._slow_down(wait)
                self.sleep(wait)
                delay = min(delay * 2, 60.0)
                continue
            hint = " Hosted models are retired often: list the live ones with `flab api-models`." if resp.status_code in (404, 410) else ""
            raise ChatError(f"{resp.status_code} from {self.base_url}: {str(getattr(resp, 'text', ''))[:300]}{hint}")
        raise ChatError("unreachable")

    def models(self) -> list[str]:
        """Ids of the models this key can call right now."""
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        resp = self.session.get(f"{self.base_url}/models", headers=headers, timeout=self.timeout)
        if resp.status_code != 200:
            raise ChatError(f"{resp.status_code} from {self.base_url}/models")
        return sorted(m["id"] for m in resp.json().get("data", []))
