"""A polite EDGAR HTTP client.

Implements the SEC fair access policy: a declared User-Agent, at most ``max_rps`` requests per
second (token bucket shared by all threads of this process), retries with exponential backoff on
429 and 5xx, and an optional on disk cache so re runs never refetch the same document.
"""
from __future__ import annotations

import hashlib
import re
import threading
import time
from pathlib import Path

import requests

from ..config import get_settings

_UA_RE = re.compile(r".+\s+\S+@\S+\.\S+")


class RateLimiter:
    """Thread safe token bucket."""

    def __init__(self, rate: float, burst: int = 1):
        self.rate = rate
        self.capacity = max(1, burst)
        self.tokens = float(self.capacity)
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                wait = (1 - self.tokens) / self.rate
            time.sleep(wait)


class EdgarClient:
    def __init__(
        self,
        user_agent: str | None = None,
        max_rps: float | None = None,
        cache_dir: str | Path | None = None,
        session: requests.Session | None = None,
        max_retries: int = 5,
    ):
        settings = get_settings()
        ua = user_agent or settings.sec_user_agent
        if not ua or not _UA_RE.match(ua):
            raise ValueError(
                "SEC requires a User-Agent with an organisation name and contact email, "
                "for example 'Acme Research ops@acme.com'. Set SEC_USER_AGENT."
            )
        rps = max_rps or settings.sec_max_rps
        if rps > 10:
            raise ValueError("SEC fair access policy caps clients at 10 requests per second")
        self.limiter = RateLimiter(rps)
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_retries = max_retries

    def _cache_path(self, url: str) -> Path | None:
        if not self.cache_dir:
            return None
        digest = hashlib.sha256(url.encode()).hexdigest()
        return self.cache_dir / digest[:2] / digest

    def get_bytes(self, url: str) -> bytes:
        cached = self._cache_path(url)
        if cached and cached.exists():
            return cached.read_bytes()
        delay = 1.0
        for attempt in range(self.max_retries):
            self.limiter.acquire()
            resp = self.session.get(url, timeout=30)
            if resp.status_code == 200:
                if cached:
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    cached.write_bytes(resp.content)
                return resp.content
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
        raise RuntimeError(f"unreachable: {url}")

    def get_text(self, url: str) -> str:
        raw = self.get_bytes(url)
        for enc in ("utf-8", "latin-1"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")

    def get_json(self, url: str) -> dict:
        import json

        return json.loads(self.get_bytes(url))
