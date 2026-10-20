"""Runs only where PySpark and Java are available (CI job ``spark``). Proves the cluster code path and the
local runner write identical tables."""
import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")


def test_spark_matches_local(small_world, small_lake, tmp_path):
    from filings_lab.pipeline import spark_job

    lake = str(tmp_path / "spark_lake")
    d = small_world["dir"]
    spark_job.run(str(d / "index.parquet"), str(d / "raw"), str(d / "prices.parquet"), lake)

    se, le = pd.read_parquet(f"{lake}/gold/events"), pd.read_parquet(small_lake / "gold" / "events")
    m = se.merge(le, on="accession", suffixes=("_s", "_l"))
    assert len(m) == len(le) == len(se)
    assert (m["car_s"] - m["car_l"]).abs().max() < 1e-12
    assert (m["label_tercile_s"].astype(int) == m["label_tercile_l"].astype(int)).all()

    sp = pd.read_parquet(f"{lake}/silver/paragraphs")
    lp = pd.read_parquet(small_lake / "silver" / "paragraphs")
    assert set(sp["paragraph_id"]) == set(lp["paragraph_id"])
