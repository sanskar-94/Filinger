"""The contract between every extractor (rules, base model, fine tuned model, frontier model) and the rest of
the lab. Kept small on purpose: every field has a definition precise enough for two humans to agree on it."""
from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

GUIDANCE = ("raised", "lowered", "maintained", "none")
TONE = ("positive", "neutral", "negative")
LABEL_FIELDS = ("guidance_direction", "management_tone", "liquidity_concern", "litigation_risk")


class Quote(BaseModel):
    paragraph: int = Field(description="number of the [P#] paragraph the quote comes from")
    text: str = Field(description="verbatim excerpt from that paragraph, at most 300 characters")


class FilingSignals(BaseModel):
    guidance_direction: Literal["raised", "lowered", "maintained", "none"]
    management_tone: Literal["positive", "neutral", "negative"]
    liquidity_concern: bool
    litigation_risk: bool
    key_quotes: list[Quote] = Field(default_factory=list, description="one to three verbatim quotes supporting the labels")
    summary: str = Field(description="one or two sentences, at most 60 words")


def json_schema() -> dict:
    """Strict JSON schema (no additional properties, all fields required) for constrained decoding."""
    def strict(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for k in ("title", "default"):
                node.pop(k, None)
            for v in node.values():
                strict(v)
        elif isinstance(node, list):
            for v in node:
                strict(v)
        return node

    schema = FilingSignals.model_json_schema()
    defs = schema.pop("$defs", {})
    text = json.dumps(schema)
    for name, d in defs.items():  # inline refs: some constrained decoders do not resolve $ref
        text = text.replace(json.dumps({"$ref": f"#/$defs/{name}"}), json.dumps(d))
    return strict(json.loads(text))


_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def parse(raw: str) -> tuple[FilingSignals | None, bool, str | None]:
    """Returns (signals, json_valid, error). Tolerates code fences and prose around the JSON object."""
    if raw is None:
        return None, False, "empty"
    m = _JSON_BLOCK.search(raw)
    if not m:
        return None, False, "no json object"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        return None, False, f"json: {exc.msg}"
    try:
        return FilingSignals.model_validate(obj), True, None
    except ValidationError as exc:
        return None, True, f"schema: {exc.errors()[0]['loc']} {exc.errors()[0]['msg']}"
