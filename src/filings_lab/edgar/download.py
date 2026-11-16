"""Bronze stage: download primary filing documents to the raw store, politely and resumably.

Downloading is deliberately *not* done inside Spark. EDGAR allows 10 requests per second per client, so
fanning out across executors would get the cluster's IPs blocked. One rate limited process (or a few,
with a shared budget) fills ``raw/``; Spark then processes it at full parallelism.
"""
from __future__ import annotations

import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from .. import storage
from ..text.clean import prune_submission
from . import urls

EXHIBIT_FORMS = ("8-K",)  # forms whose substance is in an exhibit, so the whole submission is fetched


def raw_uri(raw_root: str, cik: int, accession: str) -> str:
    return storage.join(raw_root, str(int(cik)), f"{accession}.txt")


def download_filings(client, index: pd.DataFrame, raw_root: str, workers: int = 4, exhibits: bool = True,
                     compress: bool = False) -> pd.DataFrame:
    """Download each filing in ``index`` (needs cik, accession and either primary_document or path).

    Returns the index with a ``raw_uri`` column and a ``download_status`` column. Already present files are
    skipped, so the job can be resumed after interruption.

    With ``exhibits`` an 8-K is fetched as its full submission and pruned to the 8-K itself plus its EX-99
    exhibits, because the press release is where an earnings 8-K says anything. ``compress`` stores gzip
    under the same name; every reader detects it, so a lake can mix both.
    """

    def _one(row) -> tuple[str, str]:
        dest = raw_uri(raw_root, row.cik, row.accession)
        if storage.exists(dest):
            return dest, "cached"
        doc = getattr(row, "primary_document", None)
        form = str(getattr(row, "form", "") or "")
        whole = exhibits and form.upper().startswith(EXHIBIT_FORMS)
        if whole:
            url = urls.submission_txt(row.cik, row.accession)
        elif isinstance(doc, str) and doc:
            url = urls.filing_document(row.cik, row.accession, doc)
        else:
            url = urls.full_submission_txt(row.path)
        try:
            data = client.get_bytes(url)
            if whole:
                data = prune_submission(data.decode("utf-8", errors="replace"), form).encode("utf-8")
            storage.write_bytes(dest, gzip.compress(data, 6) if compress else data)
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
