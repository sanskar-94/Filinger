"""Prompt construction with evidence selection.

Real MD&A sections run to tens of thousands of words, more than a 3B model handles well. The input is
therefore the subset of paragraphs most likely to carry the labelled signals, kept in document order and
numbered [P1], [P2]... so quotes can be traced back to a paragraph id (the agent's citation target).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .schema import json_schema

SYSTEM_PROMPT = """You are a financial disclosure analyst. You read excerpts of a company's SEC filing and extract structured signals.

Field definitions:
1. guidance_direction: what management did to its forward financial outlook in THIS document.
   raised = outlook increased, or results now expected above the prior range.
   lowered = outlook decreased, or results now expected below the prior range.
   maintained = outlook explicitly reaffirmed or unchanged.
   none = no statement about the company's own forward outlook. Generic forward looking boilerplate is none.
2. management_tone: how management characterises recent results and conditions: positive, neutral or negative.
   Judge the substance (growth, margins, demand, losses, impairments), not risk factor boilerplate, and read negations carefully: "we do not expect a material adverse effect" is not negative.
3. liquidity_concern: true only for going concern doubt, covenant breaches or waivers, or an explicit need to raise capital to fund operations.
4. litigation_risk: true only for a specific, non routine proceeding: a named lawsuit, class action, regulator subpoena or investigation, or an adverse verdict. Ordinary course boilerplate is false.
5. key_quotes: one to three verbatim excerpts (copied exactly) that support the labels, each with the number of the paragraph it comes from.
6. summary: one or two sentences, at most 60 words.

Respond with a single JSON object matching the schema. Do not add any text outside the JSON."""

KEYWORDS = re.compile(
    r"guidance|outlook|expect|forecast|reaffirm|going concern|substantial doubt|covenant|waiver|raise additional capital|"
    r"lawsuit|class action|subpoena|verdict|investigation|litigation|increased|decreased|declined|record|impairment",
    re.I,
)


@dataclass
class ExtractionInput:
    accession: str
    form: str
    company: str = ""
    paragraphs: list[dict] = field(default_factory=list)  # each: paragraph_id, section, text

    def numbered(self) -> str:
        return "\n\n".join(f"[P{i + 1}] ({p['section']}) {p['text']}" for i, p in enumerate(self.paragraphs))


def select_paragraphs(paragraphs: list[dict], budget_chars: int = 12_000) -> list[dict]:
    """Keep every paragraph if they fit; otherwise rank by keyword density, keep the best within budget and
    restore document order."""
    total = sum(len(p["text"]) for p in paragraphs)
    if total <= budget_chars:
        return paragraphs
    scored = sorted(
        range(len(paragraphs)),
        key=lambda i: (-len(KEYWORDS.findall(paragraphs[i]["text"])), i),
    )
    keep, used = [], 0
    for i in scored:
        n = len(paragraphs[i]["text"])
        if used + n > budget_chars:
            continue
        keep.append(i)
        used += n
    return [paragraphs[i] for i in sorted(keep)]


def user_message(inp: ExtractionInput) -> str:
    return (
        f"Company: {inp.company or 'unknown'}\nForm: {inp.form}\nAccession: {inp.accession}\n\n"
        f"Filing excerpts:\n\n{inp.numbered()}\n\nReturn the JSON object."
    )


def chat_messages(inp: ExtractionInput, include_schema: bool = True) -> list[dict]:
    """OpenAI style chat messages, used for fine tuning data and for vLLM / local inference.
    Small models benefit from seeing the schema; the frontier path enforces it with structured outputs."""
    system = SYSTEM_PROMPT
    if include_schema:
        system += "\n\nJSON schema:\n" + json.dumps(json_schema(), separators=(",", ":"))
    return [{"role": "system", "content": system}, {"role": "user", "content": user_message(inp)}]
