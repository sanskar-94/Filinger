"""Build the bronze filing index: which filings, when they were accepted, and where the document lives.

Two routes with identical output:
1. ``index_from_api``: one submissions request per company. Fine for a few hundred companies.
2. ``index_from_bulk``: reads the nightly ``submissions.zip`` (all filers, about 1.5 GB). The right way to
   backfill thousands of companies: one download instead of thousands of requests.
"""
from __future__ import annotations

import json
import zipfile
from collections.abc import Iterable

import pandas as pd

from . import urls
from .submissions import fetch_submissions, parse_submissions

DEFAULT_FORMS = ("10-K", "10-Q", "8-K")
INDEX_COLUMNS = [
    "cik", "company", "ticker", "sic", "form", "accession", "date_filed", "accepted_at",
    "primary_document", "items", "period_of_report",
]


def load_company_tickers(client) -> pd.DataFrame:
    payload = client.get_json(urls.company_tickers())
    df = pd.DataFrame(payload.values()).rename(columns={"cik_str": "cik", "title": "company"})
    df["cik"] = df["cik"].astype(int)
    return df[["cik", "ticker", "company"]]


def _filter(df: pd.DataFrame, forms: Iterable[str], start: str, end: str) -> pd.DataFrame:
    forms = set(forms)
    keep = df["form"].isin(forms) & (df["date_filed"] >= start) & (df["date_filed"] <= end)
    out = df.loc[keep].copy()
    out["ticker"] = out["tickers"].str.split(",").str[0].replace("", pd.NA)
    for c in INDEX_COLUMNS:
        if c not in out.columns:
            out[c] = pd.NA
    return out[INDEX_COLUMNS].reset_index(drop=True)


def index_from_api(client, ciks: Iterable[int], start: str, end: str, forms=DEFAULT_FORMS) -> pd.DataFrame:
    frames = []
    for cik in ciks:
        try:
            frames.append(_filter(fetch_submissions(client, int(cik)), forms, start, end))
        except Exception as exc:
            print(f"cik {cik}: {exc}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=INDEX_COLUMNS)


def index_from_bulk(zip_path: str, start: str, end: str, forms=DEFAULT_FORMS, ciks: set[int] | None = None) -> pd.DataFrame:
    """Parse the bulk submissions archive. Main files are ``CIK##########.json``; overflow pages are
    ``CIK##########-submissions-###.json`` and contain the bare filings block."""
    frames = []
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        pages: dict[str, list[str]] = {}
        for n in names:
            if "-submissions-" in n:
                pages.setdefault(n.split("-submissions-")[0], []).append(n)
        for n in names:
            if "-submissions-" in n or not n.endswith(".json"):
                continue
            cik = int(n[3:13])
            if ciks is not None and cik not in ciks:
                continue
            payload = json.loads(zf.read(n))
            if not payload.get("filings"):
                continue
            older = [json.loads(zf.read(p)) for p in pages.get(n[:-5], [])]
            frames.append(_filter(parse_submissions(payload, older), forms, start, end))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=INDEX_COLUMNS)
