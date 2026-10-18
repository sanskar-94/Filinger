"""Central configuration, read from environment variables with sane local defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    sec_user_agent: str = field(default_factory=lambda: _env("SEC_USER_AGENT", ""))
    sec_max_rps: float = field(default_factory=lambda: float(_env("SEC_MAX_RPS", "8")))
    data_root: Path = field(default_factory=lambda: Path(_env("FLAB_DATA_ROOT", "./data")))
    lake_root: str = field(default_factory=lambda: _env("FLAB_LAKE_ROOT", "./data/lake"))
    forms: tuple[str, ...] = ("10-K", "10-Q", "8-K")
    llm_base_url: str = field(default_factory=lambda: _env("FLAB_LLM_BASE_URL", "http://localhost:8000/v1"))
    llm_model: str = field(default_factory=lambda: _env("FLAB_LLM_MODEL", "filings-extractor"))
    mlflow_uri: str = field(default_factory=lambda: _env("MLFLOW_TRACKING_URI", "./mlruns"))

    def lake(self, *parts: str) -> str:
        root = self.lake_root.rstrip("/")
        return "/".join([root, *parts])


def get_settings() -> Settings:
    return Settings()
