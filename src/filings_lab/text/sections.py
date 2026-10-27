"""Locate the sections that carry most of the language signal.

Filings contain a table of contents that repeats every item header, so the first regex hit is usually
the TOC line, not the section. For each item we consider every header match, measure the body that
follows it up to the next item header, and keep the longest. This is robust to TOCs, running page
headers and cross references such as "see Item 1A".
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# canonical section name -> header pattern (applied per line, case insensitive)
TENK_ITEMS: dict[str, str] = {
    "item_1": r"item\s*1\s*[\.\:\-—–]?\s*business\b",
    "item_1a": r"item\s*1a\s*[\.\:\-—–]?\s*risk\s+factors",
    "item_1b": r"item\s*1b\s*[\.\:\-—–]?\s*unresolved",
    "item_1c": r"item\s*1c\s*[\.\:\-—–]?\s*cybersecurity",
    "item_2": r"item\s*2\s*[\.\:\-—–]?\s*properties",
    "item_3": r"item\s*3\s*[\.\:\-—–]?\s*legal\s+proceedings",
    "item_7": r"item\s*7\s*[\.\:\-—–]?\s*management.?s\s+discussion",
    "item_7a": r"item\s*7a\s*[\.\:\-—–]?\s*quantitative",
    "item_8": r"item\s*8\s*[\.\:\-—–]?\s*financial\s+statements",
    "item_9a": r"item\s*9a\s*[\.\:\-—–]?\s*controls",
}
TENQ_ITEMS: dict[str, str] = {
    "p1_item_1": r"item\s*1\s*[\.\:\-—–]?\s*financial\s+statements",
    "p1_item_2": r"item\s*2\s*[\.\:\-—–]?\s*management.?s\s+discussion",
    "p1_item_3": r"item\s*3\s*[\.\:\-—–]?\s*quantitative",
    "p1_item_4": r"item\s*4\s*[\.\:\-—–]?\s*controls",
    "p2_item_1": r"item\s*1\s*[\.\:\-—–]?\s*legal\s+proceedings",
    "p2_item_1a": r"item\s*1a\s*[\.\:\-—–]?\s*risk\s+factors",
    "p2_item_2": r"item\s*2\s*[\.\:\-—–]?\s*unregistered",
    "p2_item_6": r"item\s*6\s*[\.\:\-—–]?\s*exhibits",
}
EIGHTK_ITEM_RE = r"item\s*(\d\.\d{2})"
# Boundary only headers: they end the previous section but are never returned as sections themselves.
BOUNDARIES: dict[str, str] = {
    "_part": r"part\s+(i{1,3}|iv)\b\s*[\.\:\-\u2014\u2013]?",
    "_signatures": r"signatures?\s*$",
}

# The sections the rest of the lab consumes, with friendly names.
KEY_SECTIONS = {
    "10-K": {"item_1a": "risk_factors", "item_7": "mdna", "item_3": "legal"},
    "10-Q": {"p1_item_2": "mdna", "p2_item_1a": "risk_factors", "p2_item_1": "legal"},
}

EIGHTK_ITEM_NAMES = {
    "1.01": "material_agreement", "1.02": "agreement_termination", "1.03": "bankruptcy",
    "2.01": "acquisition_disposition", "2.02": "results_of_operations", "2.03": "direct_obligation",
    "2.05": "exit_costs", "2.06": "impairment", "3.01": "delisting_notice", "4.01": "auditor_change",
    "4.02": "non_reliance", "5.02": "officer_change", "5.07": "shareholder_vote", "7.01": "reg_fd",
    "8.01": "other_events", "9.01": "exhibits",
}

MIN_BODY_CHARS = 40  # the longest candidate wins, so this only filters empty TOC rows


@dataclass
class Section:
    name: str
    text: str
    start: int


def _header_hits(text: str, patterns: dict[str, str]) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    offset = 0
    for line in text.split("\n"):
        stripped = line.strip().lower()
        if stripped and len(stripped) < 200:  # headers are short lines
            for name, pat in patterns.items():
                if re.match(pat, stripped):
                    hits.append((offset, name))
                    break
        offset += len(line) + 1
    return sorted(hits)


def extract_sections(text: str, form: str) -> dict[str, Section]:
    base = form.replace("/A", "").upper()
    if base.startswith("8-K"):
        return extract_8k_items(text)
    patterns = TENK_ITEMS if base.startswith("10-K") else TENQ_ITEMS
    hits = _header_hits(text, {**patterns, **BOUNDARIES})
    if base.startswith("10-Q"):
        hits = _resolve_10q_parts(text, hits)
    best: dict[str, Section] = {}
    for i, (pos, name) in enumerate(hits):
        if name.startswith("_"):
            continue
        end = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        body = text[pos:end]
        # drop the header line itself
        body = body.split("\n", 1)[1].strip() if "\n" in body else ""
        if len(body) < MIN_BODY_CHARS:
            continue
        if name not in best or len(body) > len(best[name].text):
            best[name] = Section(name=name, text=body, start=pos)
    return best


def _resolve_10q_parts(text: str, hits: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """10-Q item numbers repeat in Part I and Part II. Use the 'Part II' marker to disambiguate."""
    marks = list(re.finditer(r"^\s*part\s+ii\b", text, re.I | re.M))
    part2 = marks[-1].start() if marks else len(text)  # last occurrence skips the TOC
    out = []
    for pos, name in hits:
        if name.startswith("_"):
            out.append((pos, name))
            continue
        if name.startswith("p1_") and pos >= part2:
            continue
        if name.startswith("p2_") and pos < part2:
            continue
        out.append((pos, name))
    return out


def extract_8k_items(text: str) -> dict[str, Section]:
    hits = []
    offset = 0
    for line in text.split("\n"):
        m = re.match(EIGHTK_ITEM_RE, line.strip().lower())
        if m and len(line) < 250:
            hits.append((offset, m.group(1)))
        offset += len(line) + 1
    out: dict[str, Section] = {}
    for i, (pos, code) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        body = text[pos:end].split("\n", 1)
        body_text = body[1].strip() if len(body) > 1 else ""
        name = f"8k_{code}"
        if body_text and (name not in out or len(body_text) > len(out[name].text)):
            out[name] = Section(name=name, text=body_text, start=pos)
    return out


def key_sections(text: str, form: str) -> dict[str, str]:
    """Friendly name -> text for the sections used downstream. 8-K returns every item."""
    base = form.replace("/A", "").upper()
    secs = extract_sections(text, base)
    if base.startswith("8-K"):
        return {f"8k_{EIGHTK_ITEM_NAMES.get(s.name[3:], s.name[3:])}": s.text for s in secs.values()}
    mapping = KEY_SECTIONS["10-K" if base.startswith("10-K") else "10-Q"]
    return {friendly: secs[raw].text for raw, friendly in mapping.items() if raw in secs}
