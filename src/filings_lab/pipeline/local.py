"""Single machine runner with the same outputs as the Spark job. Good up to a few hundred thousand filings."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .. import storage
from .core import compute_events, process_partition


def _work(batch: pd.DataFrame):
    return process_partition(batch, lambda uri: storage.read_bytes(uri))


def write_partitioned(df: pd.DataFrame, path: str, partition_cols: list[str]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, partition_cols=partition_cols, index=False, existing_data_behavior="delete_matching")


def run(index: pd.DataFrame, prices: pd.DataFrame, lake_root: str, workers: int = 4, market: str = "SPY") -> dict:
    index = index.copy()
    index["year"] = pd.to_datetime(index["date_filed"]).dt.year
    n_batches = max(1, min(len(index), workers * 8))
    bounds = np.linspace(0, len(index), n_batches + 1).astype(int)
    batches = [index.iloc[a:b] for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    s_parts, p_parts = [], []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for s, p in pool.map(_work, batches):
                s_parts.append(s)
                p_parts.append(p)
    else:
        for b in batches:
            s, p = _work(b)
            s_parts.append(s)
            p_parts.append(p)
    sections = pd.concat(s_parts, ignore_index=True)
    paragraphs = pd.concat(p_parts, ignore_index=True)
    years = index.set_index("accession")["year"]
    sections["year"] = sections["accession"].map(years)
    paragraphs["year"] = paragraphs["accession"].map(years)

    events = compute_events(index, prices, market=market)
    events["year"] = events["event_date"].dt.year

    write_partitioned(sections, storage.join(lake_root, "silver", "sections"), ["form", "year"])
    write_partitioned(paragraphs, storage.join(lake_root, "silver", "paragraphs"), ["form", "year"])
    Path(storage.join(lake_root, "silver")).mkdir(parents=True, exist_ok=True)
    prices.to_parquet(storage.join(lake_root, "silver", "prices.parquet"), index=False)
    index.to_parquet(storage.join(lake_root, "bronze_index.parquet"), index=False)
    write_partitioned(events, storage.join(lake_root, "gold", "events"), ["year"])
    return {"sections": len(sections), "paragraphs": len(paragraphs), "events": len(events)}
