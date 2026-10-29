"""Deterministic, sticky variant assignment.

A user is hashed with the experiment's salt into [0, 1); the same user always gets the same variant for
the same experiment, different experiments are independent (different salts), and no assignment table is
needed. Changing weights mid experiment would reshuffle users, so the config is frozen once the test starts.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Experiment:
    name: str
    salt: str
    variants: dict[str, dict] = field(default_factory=dict)  # variant -> {"weight": float, ...config}
    primary_metric: str = "thumbs_up_rate"
    guardrails: list[str] = field(default_factory=list)
    active: bool = True

    @classmethod
    def load(cls, path: str | Path) -> Experiment:
        return cls(**json.loads(Path(path).read_text()))

    def bucket(self, unit_id: str) -> float:
        h = hashlib.sha256(f"{self.salt}:{unit_id}".encode()).digest()
        return int.from_bytes(h[:8], "big") / 2**64

    def assign(self, unit_id: str) -> str:
        names = sorted(self.variants)
        total = sum(self.variants[n].get("weight", 1.0) for n in names)
        x, acc = self.bucket(unit_id) * total, 0.0
        for n in names:
            acc += self.variants[n].get("weight", 1.0)
            if x < acc:
                return n
        return names[-1]

    def config(self, variant: str) -> dict:
        return {k: v for k, v in self.variants[variant].items() if k != "weight"}
