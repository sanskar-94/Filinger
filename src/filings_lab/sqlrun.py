"""Run the analysis SQL in ``sql/`` against the Parquet lake with DuckDB (no warehouse needed)."""
from __future__ import annotations

from pathlib import Path


def run_sql(path: str, lake: str = "data/lake", artifacts: str = "artifacts"):
    import duckdb

    query = Path(path).read_text().format(lake=lake.rstrip("/"), artifacts=artifacts.rstrip("/"))
    return duckdb.sql(query).df()
