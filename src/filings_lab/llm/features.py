"""Turn extracted signals into model features so Phase 2 can test whether the LLM adds information."""
from __future__ import annotations

import pandas as pd

from .providers import ExtractionResult

TONE_NUM = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


def signals_to_features(results: list[ExtractionResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        s = r.signals or {}
        ok = r.schema_valid
        g = s.get("guidance_direction")
        rows.append({
            "accession": r.accession,
            "llm_valid": float(ok),
            "llm_guid_up": float(ok and g == "raised"),
            "llm_guid_down": float(ok and g == "lowered"),
            "llm_guid_flat": float(ok and g == "maintained"),
            "llm_tone": TONE_NUM.get(s.get("management_tone"), float("nan")) if ok else float("nan"),
            "llm_liquidity": float(ok and bool(s.get("liquidity_concern"))),
            "llm_litigation": float(ok and bool(s.get("litigation_risk"))),
        })
    return pd.DataFrame(rows)
