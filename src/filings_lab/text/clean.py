"""Turn raw EDGAR documents (HTML, inline XBRL or legacy SGML text) into clean plain text.

Design choices (see docs/RESEARCH.md section 2):
1. Block level tags become line breaks so paragraph structure survives.
2. ``script``, ``style`` and the hidden inline XBRL header (``ix:header``) are dropped.
3. Tables are kept only when they are mostly prose. Numeric tables are financial statements, which the
   lab takes from XBRL instead; leaving them in pollutes word counts and tone scores.
"""
from __future__ import annotations

import html
import re
import unicodedata
from html.parser import HTMLParser

_BLOCK_TAGS = {
    "p", "div", "br", "tr", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
    "table", "section", "article", "center", "blockquote", "pre", "hr", "title",
}
_SKIP_TAGS = {"script", "style", "head", "ix:header", "xbrli:context", "xbrli:unit"}
_DOC_RE = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.S | re.I)
_TYPE_RE = re.compile(r"<TYPE>\s*([^\s<]+)", re.I)
_TEXT_RE = re.compile(r"<TEXT>(.*?)</TEXT>", re.S | re.I)
_NUMERIC_TOKEN = re.compile(r"^[\(\)\$%,.\-—–\d\s]+$")
_WS = re.compile(r"[ \t    ]+")
_MULTI_NL = re.compile(r"\n{3,}")
_PAGE_NOISE = re.compile(r"^\s*(page\s*)?\d{1,3}\s*$|^\s*table of contents\s*$", re.I)


_VOID_TAGS = {"br", "hr", "img", "meta", "input", "link", "col", "area", "base", "wbr"}


class _Extractor(HTMLParser):
    """Streaming HTML to text extractor that tolerates the malformed markup common in EDGAR."""

    def __init__(self, numeric_table_threshold: float):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.stack: list[str] = []
        self.skip_level: int | None = None  # stack depth at which a skipped subtree started
        self.table_depth = 0
        self.table_buf: list[str] = []
        self.threshold = numeric_table_threshold

    def _emit(self, s: str) -> None:
        if self.skip_level is not None:
            return
        (self.table_buf if self.table_depth else self.out).append(s)

    def handle_startendtag(self, tag, attrs):
        if tag.lower() in _BLOCK_TAGS:
            self._emit("\n")

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in _VOID_TAGS:
            if tag in _BLOCK_TAGS:
                self._emit("\n")
            return
        style = (dict(attrs).get("style") or "").replace(" ", "").lower()
        self.stack.append(tag)
        if self.skip_level is None and (tag in _SKIP_TAGS or "display:none" in style):
            self.skip_level = len(self.stack)
            return
        if self.skip_level is not None:
            return
        if tag == "table":
            self.table_depth += 1
            if self.table_depth == 1:
                self.table_buf = []
        if tag in ("td", "th"):
            self._emit(" | ")
        if tag in _BLOCK_TAGS:
            self._emit("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in _VOID_TAGS or tag not in self.stack:
            return  # stray closing tag
        # pop up to and including the most recent matching open tag (implicitly closes unclosed children)
        while self.stack:
            popped = self.stack.pop()
            if self.skip_level is not None and len(self.stack) < self.skip_level:
                self.skip_level = None
                if popped == tag:
                    return
                continue
            if self.skip_level is None:
                if popped in _BLOCK_TAGS:
                    self._emit("\n")
                if popped == "table" and self.table_depth:
                    self.table_depth -= 1
                    if self.table_depth == 0:
                        self._flush_table()
            if popped == tag:
                return

    def _flush_table(self) -> None:
        text = "".join(self.table_buf)
        self.table_buf = []
        cells = [c.strip() for c in re.split(r"\||\n", text) if c.strip()]
        if not cells:
            return
        numeric = sum(1 for c in cells if _NUMERIC_TOKEN.match(c))
        if numeric / len(cells) >= self.threshold:
            self.out.append("\n")
            return
        # prose table (often used purely for layout): keep its text without the cell separators
        self.out.append("\n" + re.sub(r"\s*\|\s*", " ", text) + "\n")

    def handle_data(self, data):
        self._emit(data)

    def close(self):
        super().close()
        if self.table_depth and self.table_buf:  # unterminated table at end of document
            self.table_depth = 0
            self._flush_table()


def select_main_document(raw: str, form: str | None = None) -> str:
    """Pick the main document out of a full submission ``.txt`` (SGML wrapped) file.

    Returns the input unchanged when it is a single HTML document.
    """
    docs = _DOC_RE.findall(raw)
    if not docs:
        return raw
    target = (form or "").upper()
    for doc in docs:
        m = _TYPE_RE.search(doc)
        dtype = m.group(1).upper() if m else ""
        if target and (dtype == target or dtype.replace("/A", "") == target.replace("/A", "")):
            body = _TEXT_RE.search(doc)
            return body.group(1) if body else doc
    body = _TEXT_RE.search(docs[0])
    return body.group(1) if body else docs[0]


def looks_like_html(raw: str) -> bool:
    head = raw[:5000].lower()
    return "<html" in head or "<div" in head or "<p" in head or "<font" in head or "<xbrl" in head


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = _WS.sub(" ", text)
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if not _PAGE_NOISE.match(ln)]
    text = "\n".join(lines)
    return _MULTI_NL.sub("\n\n", text).strip()


def html_to_text(raw_html: str, numeric_table_threshold: float = 0.4) -> str:
    parser = _Extractor(numeric_table_threshold)
    parser.feed(raw_html)
    parser.close()
    return normalise(html.unescape("".join(parser.out)))


def clean_filing(raw: str, form: str | None = None) -> str:
    """Full cleaning entry point used by both the Spark job and the local pipeline."""
    doc = select_main_document(raw, form)
    if looks_like_html(doc):
        return html_to_text(doc)
    return normalise(html.unescape(doc))
