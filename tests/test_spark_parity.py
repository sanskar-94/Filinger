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


def test_spark_features_match_pandas(small_world, small_lake, tmp_path):
    import numpy as np

    from filings_lab.features.build import build_features
    from filings_lab.features.spark_features import build_features_spark
    from filings_lab.pipeline.spark_job import build_spark

    facts_path = tmp_path / "facts.parquet"
    small_world["facts"].to_parquet(facts_path, index=False)
    spark = build_spark("parity-features")
    sp = build_features_spark(spark, str(small_lake), str(facts_path)).toPandas()
    sections = pd.read_parquet(small_lake / "silver" / "sections")
    events = pd.read_parquet(small_lake / "gold" / "events").drop(columns=["year"])
    pdf = build_features(sections, events, small_world["index"], small_world["facts"])
    assert set(sp["accession"]) == set(pdf["accession"])
    m = pdf.merge(sp, on="accession", suffixes=("_p", "_s"))
    numeric = [c for c in pdf.columns if c != "accession" and pd.api.types.is_numeric_dtype(pdf[c]) and f"{c}_s" in m]
    for c in numeric:
        a, b = m[f"{c}_p"].astype(float), m[f"{c}_s"].astype(float)
        assert np.allclose(a.fillna(-999), b.fillna(-999), atol=1e-9), c
    assert (pd.to_datetime(m["t1_p"]) == pd.to_datetime(m["t1_s"])).all()
