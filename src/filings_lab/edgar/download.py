"""Bronze stage: download primary filing documents to the raw store, politely and resumably.

Downloading is deliberately *not* done inside Spark. EDGAR allows 10 requests per second per client, so
fanning out across executors would get the cluster's IPs blocked. One rate limited process (or a few,
with a shared budget) fills ``raw/``; Spark then processes it at full parallelism.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from .. import storage
from . import urls


def raw_uri(raw_root: str, cik: int, accession: str) -> str:
    return storage.join(raw_root, str(int(cik)), f"{accession}.txt")


def download_filings(client, index: pd.DataFrame, raw_root: str, workers: int = 4) -> pd.DataFrame:
    """Download each filing in ``index`` (needs cik, accession and either primary_document or path).

    Returns the index with a ``raw_uri`` column and a ``download_status`` column. Already present files are
    skipped, so the job can be resumed after interruption.
    """

    def _one(row) -> tuple[str, str]:
        dest = raw_uri(raw_root, row.cik, row.accession)
        if storage.exists(dest):
            return dest, "cached"
        doc = getattr(row, "primary_document", None)
        url = (
            urls.filing_document(row.cik, row.accession, doc)
            if isinstance(doc, str) and doc
            else urls.full_submission_txt(row.path)
        )
        try:
            storage.write_bytes(dest, client.get_bytes(url))
            return dest, "ok"
        except Exception as exc:  # keep going; failures are reported, not fatal
            return dest, f"error: {exc}"

    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:  # the client's limiter caps total rate
        futures = {pool.submit(_one, row): i for i, row in enumerate(index.itertuples(index=False))}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    out = index.copy()
    out["raw_uri"] = [results[i][0] for i in range(len(index))]
    out["download_status"] = [results[i][1] for i in range(len(index))]
    return out
