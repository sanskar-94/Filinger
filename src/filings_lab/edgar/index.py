"""Parse EDGAR quarterly master.idx files into a tidy DataFrame."""
from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

from . import urls

TARGET_FORMS = {"10-K", "10-K405", "10-KT", "10-Q", "8-K", "10-K/A", "10-Q/A", "8-K/A"}
_ACC_RE = re.compile(r"(\d{10}-\d{2}-\d{6})")


def parse_master_index(text: str, forms: Iterable[str] | None = None) -> pd.DataFrame:
    """Parse the pipe delimited body of a master.idx file.

    The file has a free text preamble, then a header ``CIK|Company Name|Form Type|Date Filed|Filename``
    followed by a dashed line. Everything after the dashes is data.
    """
    wanted = set(forms) if forms else TARGET_FORMS
    rows = []
    in_body = False
    for line in text.splitlines():
        if not in_body:
            if line.startswith("-----"):
                in_body = True
            continue
        parts = line.split("|")
        if len(parts) != 5:
            continue
        cik, company, form, date_filed, filename = parts
        if form not in wanted:
            continue
        acc = _ACC_RE.search(filename)
        rows.append(
            {
                "cik": int(cik),
                "company": company.strip(),
                "form": form,
                "base_form": form.replace("/A", "").replace("405", "").replace("10-KT", "10-K"),
                "is_amendment": form.endswith("/A"),
                "date_filed": pd.Timestamp(date_filed),
                "path": filename,
                "accession": acc.group(1) if acc else None,
            }
        )
    return pd.DataFrame(rows)


def quarters(start_year: int, end_year: int) -> list[tuple[int, int]]:
    return [(y, q) for y in range(start_year, end_year + 1) for q in range(1, 5)]


def fetch_universe(client, start_year: int, end_year: int, forms=None, ciks: set[int] | None = None) -> pd.DataFrame:
    frames = []
    for year, q in quarters(start_year, end_year):
        try:
            text = client.get_text(urls.master_index(year, q))
        except Exception as exc:  # future quarters 404
            print(f"skip {year}Q{q}: {exc}")
            continue
        df = parse_master_index(text, forms)
        if ciks is not None:
            df = df[df["cik"].isin(ciks)]
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
