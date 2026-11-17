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
    # words may sit between the number and the title: "Item 7. Bank of America Corporation and Subsidiaries
    # Management's Discussion and Analysis ..."; the \b keeps Item 7A out
    "item_7": r"item\s*7\b\s*[\.\:\-—–]?\s*(?:[^\n]{0,80}?\s)?management.?s\s+discussion",
    "item_7a": r"item\s*7a\s*[\.\:\-—–]?\s*quantitative",
    "item_8": r"item\s*8\s*[\.\:\-—–]?\s*financial\s+statements",
    "item_9a": r"item\s*9a\s*[\.\:\-—–]?\s*controls",
}
TENQ_ITEMS: dict[str, str] = {
    "p1_item_1": r"item\s*1\s*[\.\:\-—–]?\s*financial\s+statements",
    "p1_item_2": r"item\s*2\b\s*[\.\:\-—–]?\s*(?:[^\n]{0,80}?\s)?management.?s\s+discussion",
    "p1_item_3": r"item\s*3\s*[\.\:\-—–]?\s*quantitative",
    "p1_item_4": r"item\s*4\s*[\.\:\-—–]?\s*controls",
    "p2_item_1": r"item\s*1\s*[\.\:\-—–]?\s*legal\s+proceedings",
    "p2_item_1a": r"item\s*1a\s*[\.\:\-—–]?\s*risk\s+factors",
    "p2_item_2": r"item\s*2\s*[\.\:\-—–]?\s*unregistered",
    "p2_item_6": r"item\s*6\s*[\.\:\-—–]?\s*exhibits",
}
EIGHTK_ITEM_RE = r"item\s*(\d\.\d{2})"
# The official item titles. With its title an item header is recognised anywhere, not only at the start of a
# short line: real 8-Ks run the header into its text ("ITEM 2.02 RESULTS OF OPERATIONS AND FINANCIAL CONDITION On
# October 18, ...") or put two headers on one line before the text they share ("Item 2.02 Results of Operations
# and Financial Condition Item 7.01 Regulation FD Disclosure The following information ..."). A cross reference
# such as 'pursuant to Item 2.02, "Results of Operations ..."' has a comma and a quote after the number, so it
# does not match.
EIGHTK_TITLES = {
    "1.01": r"entry\s+into\s+a\s+material\s+definitive\s+agreement",
    "1.02": r"termination\s+of\s+a\s+material\s+definitive\s+agreement",
    "1.03": r"bankruptcy\s+or\s+receivership",
    "2.01": r"completion\s+of\s+(?:an\s+)?acquisition\s+or\s+disposition\s+of\s+assets",
    "2.02": r"results\s+of\s+operations\s+and\s+financial\s+condition",
    "2.03": r"creation\s+of\s+a\s+direct\s+financial\s+obligation(?:\s+or\s+an\s+obligation\s+under\s+an\s+off.balance\s+"
            r"sheet\s+arrangement\s+of\s+a\s+registrant)?",
    "2.05": r"costs\s+associated\s+with\s+exit\s+or\s+disposal\s+activities",
    "2.06": r"material\s+impairments",
    "3.01": r"notice\s+of\s+delisting\s+or\s+failure\s+to\s+satisfy\s+a\s+continued\s+listing\s+rule\s+or\s+standard"
            r"(?:;\s*transfer\s+of\s+listing)?",
    "4.01": r"changes?\s+in\s+registrant.s\s+certifying\s+accountant",
    "4.02": r"non.reliance\s+on\s+previously\s+issued\s+financial\s+statements(?:\s+or\s+a\s+related\s+audit\s+report\s+or"
            r"\s+completed\s+interim\s+review)?",
    "5.02": r"departure\s+of\s+directors\s+or\s+(?:certain\s+)?officers(?:;\s*election\s+of\s+directors;\s*appointment\s+of"
            r"\s+(?:certain\s+)?officers;\s*compensatory\s+arrangements\s+of\s+certain\s+officers)?",
    "5.07": r"submission\s+of\s+matters\s+to\s+a\s+vote\s+of\s+security\s+holders",
    "7.01": r"regulation\s+fd\s+disclosure",
    "8.01": r"other\s+events",
    "9.01": r"financial\s+statements\s+and\s+exhibits",
}
_EIGHTK_INLINE = re.compile(
    "|".join(rf"item\s*{re.escape(code)}\s*[\.\:\-—–]?\s*(?:{title})\s*[\.\:]?" for code, title in EIGHTK_TITLES.items()),
    re.I)
_EIGHTK_CODE = re.compile(r"item\s*(\d\.\d{2})", re.I)
# Management's discussion in a report laid out without item headers (JPMorgan's 10-Q opens "The following is
# Management's discussion and analysis ..." and names its items only in a cross reference index): used only when
# no item header was found, and it ends where the financial statements begin.
_MDNA_OPENING = re.compile(r"^[ \t]*(?:the\s+following\s+is\s+)?management.?s\s+discussion\s+and\s+analysis", re.I | re.M)
# the heading of the income statement, which opens the financial statements: a heading line, so the discussion's
# own "Consolidated balance sheets analysis" or "Consolidated results of operations" do not end it
_STATEMENTS_START = re.compile(
    r"^[ \t]*(?:condensed\s+)?consolidated\s+statements?\s+of\s+(?:income|operations|earnings)\b[^\n]{0,60}$", re.I | re.M)
MDNA_FALLBACK_MIN_CHARS = 2_000  # a table of contents line followed by the next one is not the discussion
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
    mdna = "item_7" if base.startswith("10-K") else "p1_item_2"
    # no item header, or one whose text is only a pointer ("... appears on pages 52-165" in a cross reference index)
    if mdna not in best or len(best[mdna].text) < MDNA_FALLBACK_MIN_CHARS:
        found = _mdna_without_item_header(text, [pos for pos, _ in hits])
        if found and (mdna not in best or len(found.text) > len(best[mdna].text)):
            best[mdna] = found
    return best


def _mdna_without_item_header(text: str, header_positions: list[int]) -> Section | None:
    """The longest stretch from a line opening "Management's discussion and analysis" to the start of the
    financial statements or the next item header, whichever comes first."""
    ends = sorted([m.start() for m in _STATEMENTS_START.finditer(text)] + header_positions)
    best = None
    for m in _MDNA_OPENING.finditer(text):
        end = next((e for e in ends if e > m.start()), len(text))
        body = text[m.start():end].strip()
        if len(body) >= MDNA_FALLBACK_MIN_CHARS and (best is None or len(body) > len(best.text)):
            best = Section(name="mdna_fallback", text=body, start=m.start())
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
    # each header as (start, where its text begins, item code)
    hits: list[tuple[int, int, str]] = []
    offset = 0
    for line in text.split("\n"):
        m = re.match(EIGHTK_ITEM_RE, line.strip().lower())
        if m and len(line) < 250:  # a header on a short line of its own; with its official title, text after it counts
            lead = len(line) - len(line.lstrip())
            titled = _EIGHTK_INLINE.match(line, lead)
            hits.append((offset, offset + (titled.end() if titled else len(line)), m.group(1)))
        offset += len(line) + 1
    for m in _EIGHTK_INLINE.finditer(text):  # a titled header inside a longer line
        if not any(start <= m.start() < body for start, body, _ in hits):
            number = _EIGHTK_CODE.match(m.group(0))
            if number:
                hits.append((m.start(), m.end(), number.group(1)))
    hits.sort()
    out: dict[str, Section] = {}
    for i, (pos, body_start, code) in enumerate(hits):
        j = i + 1
        # headers with nothing between them share the text that follows the last of them
        while j < len(hits) and not text[body_start:hits[j][0]].strip(" \t\n.:;-—–"):
            body_start = hits[j][1]
            j += 1
        end = hits[j][0] if j < len(hits) else len(text)
        body_text = text[body_start:end].strip().lstrip(".:;-—–").strip()
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
