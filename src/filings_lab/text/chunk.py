"""Paragraph chunking with stable ids. These ids are the citation targets of the analyst agent."""
from __future__ import annotations

import hashlib
import re

MIN_PARA_CHARS = 80
MAX_PARA_CHARS = 2000
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+(?=[A-Z\"\(])")


def paragraph_id(accession: str, section: str, idx: int) -> str:
    return hashlib.sha1(f"{accession}|{section}|{idx}".encode()).hexdigest()[:16]


def _split_long(p: str) -> list[str]:
    if len(p) <= MAX_PARA_CHARS:
        return [p]
    out, cur = [], ""
    for sent in _SENT_SPLIT.split(p):
        if cur and len(cur) + len(sent) + 1 > MAX_PARA_CHARS:
            out.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        out.append(cur)
    return out


def split_paragraphs(text: str) -> list[str]:
    raw = [p.replace("\n", " ").strip() for p in re.split(r"\n\s*\n|\n(?=[A-Z][^\n]{0,120}\n)", text)]
    raw = [p for p in raw if p]
    merged: list[str] = []
    carry = ""
    for p in raw:
        p = f"{carry} {p}".strip() if carry else p
        if len(p) < MIN_PARA_CHARS:
            carry = p  # short heading or fragment: glue it to the next paragraph
            continue
        carry = ""
        merged.extend(_split_long(p))
    if carry:
        if merged:
            merged[-1] = f"{merged[-1]} {carry}"
        else:
            merged.append(carry)
    return merged


def chunk_section(accession: str, section: str, text: str) -> list[dict]:
    return [
        {"paragraph_id": paragraph_id(accession, section, i), "section": section, "para_idx": i, "text": p}
        for i, p in enumerate(split_paragraphs(text))
    ]
