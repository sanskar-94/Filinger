"""Company submissions JSON: acceptance timestamps and primary documents."""
from __future__ import annotations

import pandas as pd

from . import urls

_COLS = ["accessionNumber", "form", "filingDate", "acceptanceDateTime", "primaryDocument", "items", "reportDate"]


def _frame(block: dict) -> pd.DataFrame:
    df = pd.DataFrame({c: block.get(c, []) for c in _COLS})
    return df


def parse_submissions(payload: dict, older_pages: list[dict] | None = None) -> pd.DataFrame:
    frames = [_frame(payload.get("filings", {}).get("recent", {}))]
    for page in older_pages or []:
        frames.append(_frame(page))
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(
        columns={
            "accessionNumber": "accession",
            "filingDate": "date_filed",
            "acceptanceDateTime": "accepted_at",
            "primaryDocument": "primary_document",
            "reportDate": "period_of_report",
        }
    )
    df["cik"] = int(payload["cik"])
    df["company"] = payload.get("name")
    df["sic"] = payload.get("sic")
    df["tickers"] = ",".join(payload.get("tickers", []) or [])
    # acceptanceDateTime is UTC ISO; convert to US/Eastern for the 16:00 cutoff rule
    df["accepted_at"] = pd.to_datetime(df["accepted_at"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    df["date_filed"] = pd.to_datetime(df["date_filed"])
    return df


def fetch_submissions(client, cik: int) -> pd.DataFrame:
    payload = client.get_json(urls.submissions(cik))
    pages = [client.get_json(urls.submissions_page(f["name"])) for f in payload.get("filings", {}).get("files", [])]
    return parse_submissions(payload, pages)
