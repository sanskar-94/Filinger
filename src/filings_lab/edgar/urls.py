"""URL builders for EDGAR endpoints."""
from __future__ import annotations

SEC = "https://www.sec.gov"
DATA = "https://data.sec.gov"


def pad_cik(cik: int | str) -> str:
    return str(int(cik)).zfill(10)


def master_index(year: int, quarter: int) -> str:
    return f"{SEC}/Archives/edgar/full-index/{year}/QTR{quarter}/master.idx"


def submissions(cik: int | str) -> str:
    return f"{DATA}/submissions/CIK{pad_cik(cik)}.json"


def submissions_page(name: str) -> str:
    return f"{DATA}/submissions/{name}"


def company_facts(cik: int | str) -> str:
    return f"{DATA}/api/xbrl/companyfacts/CIK{pad_cik(cik)}.json"


def company_tickers() -> str:
    return f"{SEC}/files/company_tickers.json"


def filing_document(cik: int | str, accession: str, document: str) -> str:
    return f"{SEC}/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{document}"


def full_submission_txt(path: str) -> str:
    """``path`` is the Filename column of master.idx, e.g. edgar/data/320193/0000320193-20-000096.txt"""
    return f"{SEC}/Archives/{path}"
