"""Pure functions shared by the Spark job and the local runner, so both produce identical tables."""
from __future__ import annotations

import re

import pandas as pd

from ..prices.events import add_labels, daily_returns, event_returns
from ..text.chunk import chunk_section
from ..text.clean import clean_filing
from ..text.sections import key_sections

FALLBACK_CHARS = 60_000
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]+")

SECTION_COLUMNS = ["accession", "cik", "form", "section", "text", "n_chars", "n_words"]
PARAGRAPH_COLUMNS = ["paragraph_id", "accession", "cik", "form", "section", "para_idx", "text"]


def process_document(raw: str, accession: str, cik: int, form: str) -> tuple[list[dict], list[dict]]:
    text = clean_filing(raw, form)
    secs = key_sections(text, form)
    if not secs:  # keep something searchable even when headers are non standard
        secs = {"full_text": text[:FALLBACK_CHARS]}
    sections, paragraphs = [], []
    for name, body in secs.items():
        sections.append(
            {"accession": accession, "cik": int(cik), "form": form, "section": name, "text": body,
             "n_chars": len(body), "n_words": len(_WORD.findall(body))}
        )
        for p in chunk_section(accession, name, body):
            paragraphs.append({"accession": accession, "cik": int(cik), "form": form, **p})
    return sections, paragraphs


def process_partition(batch: pd.DataFrame, reader) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``batch`` has accession, cik, form and either ``content`` (bytes) or ``raw_uri`` read via ``reader``."""
    s_rows, p_rows = [], []
    for row in batch.itertuples(index=False):
        raw = row.content if "content" in batch.columns else reader(row.raw_uri)
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        try:
            s, p = process_document(raw, row.accession, row.cik, row.form)
        except Exception as exc:  # one bad document must not kill a 10 year backfill
            print(f"failed {row.accession}: {exc}")
            continue
        s_rows.extend(s)
        p_rows.extend(p)
    return (
        pd.DataFrame(s_rows, columns=SECTION_COLUMNS),
        pd.DataFrame(p_rows, columns=PARAGRAPH_COLUMNS),
    )


EVENT_COLUMNS = [
    "accession", "cik", "ticker", "form", "accepted_at", "event_date", "ret_0", "car", "car_tradable", "car_mm", "beta",
    "mom_250_30", "ret_5_1", "vol_60_1", "mkt_ret_20_1", "abn_volume_5_1",
]


def compute_events(index: pd.DataFrame, prices: pd.DataFrame, market: str = "SPY") -> pd.DataFrame:
    """Gold events table: one row per filing with reaction labels and pre event price features."""
    rets = daily_returns(prices)
    vol = prices.pivot_table(index="date", columns="ticker", values="volume").sort_index()
    ev = index[["accession", "cik", "ticker", "form", "accepted_at"]].copy()
    ev["accepted_at"] = pd.to_datetime(ev["accepted_at"], utc=True)
    out = event_returns(ev, rets, market_col=market, volume=vol)
    for c in EVENT_COLUMNS:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[EVENT_COLUMNS]
    out["event_date"] = pd.to_datetime(out["event_date"])
    return add_labels(out.dropna(subset=["car"]))
