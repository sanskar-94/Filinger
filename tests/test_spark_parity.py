"""Runs only where PySpark and Java are available (CI job ``spark``). Proves the cluster code path and the
local runner write identical tables."""
import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")


def test_spark_matches_local(small_world, small_lake, tmp_path):
    from filings_lab.pipeline import spark_job

    lake = str(tmp_path / "spark_lake")
    d = small_world["dir"]
    spark_job.build_spark().conf.set("spark.sql.session.timeZone", "Asia/Kolkata")  # a machine that is not on UTC
    spark_job.run(str(d / "index.parquet"), str(d / "raw"), str(d / "prices.parquet"), lake)

    se, le = pd.read_parquet(f"{lake}/gold/events"), pd.read_parquet(small_lake / "gold" / "events")
    m = se.merge(le, on="accession", suffixes=("_s", "_l"))
    assert len(m) == len(le) == len(se)
    assert (m["car_s"] - m["car_l"]).abs().max() < 1e-12
    assert (pd.to_datetime(m["event_date_s"]) == pd.to_datetime(m["event_date_l"])).all()
    assert (pd.to_datetime(m["accepted_at_s"], utc=True) == pd.to_datetime(m["accepted_at_l"], utc=True)).all()
    assert (m["label_tercile_s"].astype(int) == m["label_tercile_l"].astype(int)).all()

    sp = pd.read_parquet(f"{lake}/silver/paragraphs")
    lp = pd.read_parquet(small_lake / "silver" / "paragraphs")
    assert set(sp["paragraph_id"]) == set(lp["paragraph_id"])

    # the cluster lake has every table the downstream stages read, at the same paths as the local lake
    for table, key in [("bronze_index.parquet", "accession"), ("silver/prices.parquet", ["ticker", "date"]),
                       ("silver/sections", ["accession", "section"])]:
        s_t, l_t = pd.read_parquet(f"{lake}/{table}"), pd.read_parquet(small_lake / table)
        assert len(s_t) == len(l_t), table
        assert s_t[key].astype(str).drop_duplicates().shape[0] == l_t[key].astype(str).drop_duplicates().shape[0], table
    si, li = pd.read_parquet(f"{lake}/bronze_index.parquet"), pd.read_parquet(small_lake / "bronze_index.parquet")
    assert (si.set_index("accession")["year"].sort_index().astype(int) == li.set_index("accession")["year"].sort_index().astype(int)).all()


def test_cluster_chain_runs_on_the_spark_lake(small_world, tmp_path):
    """Spark lake, then Spark features and the data quality gate on that same lake, as the Databricks bundle runs
    them; no table borrowed from the local runner."""
    from filings_lab.features.spark_features import run as spark_features
    from filings_lab.pipeline import spark_job
    from filings_lab.pipeline.quality import run_checks

    lake = str(tmp_path / "cluster_lake")
    d = small_world["dir"]
    spark_job.run(str(d / "index.parquet"), str(d / "raw"), str(d / "prices.parquet"), lake)
    facts = tmp_path / "facts.parquet"
    small_world["facts"].to_parquet(facts, index=False)
    spark_features(lake, str(facts))
    feats = pd.read_parquet(f"{lake}/gold/features")
    assert len(feats) == len(pd.read_parquet(f"{lake}/gold/events"))
    checks = run_checks(lake)
    assert {"bronze_index", "silver_sections", "silver_paragraphs", "silver_prices", "gold_events", "gold_features"} <= {c.table for c in checks}
    assert not [c for c in checks if c.status == "fail"], [c for c in checks if c.status == "fail"]


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
