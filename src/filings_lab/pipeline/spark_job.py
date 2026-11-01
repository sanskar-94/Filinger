"""PySpark job: raw filings on S3 to silver and gold Parquet tables.

Runs unchanged on Databricks (see ``infra/databricks/databricks.yml``), on AWS EMR (see
``infra/emr/``) or locally with ``spark-submit``. All parsing logic lives in ``pipeline.core`` and is shared
with the local runner, so unit tests on the pure functions cover the cluster code path too.

Usage::

    spark-submit -m filings_lab.pipeline.spark_job \
        --index s3://bucket/filings-lab/bronze/index.parquet \
        --raw s3://bucket/filings-lab/bronze/raw \
        --prices s3://bucket/filings-lab/bronze/prices.parquet \
        --lake s3://bucket/filings-lab
"""
from __future__ import annotations

import argparse

import pandas as pd

from .core import PARAGRAPH_COLUMNS, SECTION_COLUMNS, process_partition

SECTION_SCHEMA = "accession string, cik long, form string, section string, text string, n_chars long, n_words long"
PARAGRAPH_SCHEMA = (
    "paragraph_id string, accession string, cik long, form string, section string, para_idx long, text string"
)
EVENT_SCHEMA = (
    "accession string, cik long, ticker string, form string, accepted_at timestamp, event_date timestamp, "
    "ret_0 double, car double, car_tradable double, car_mm double, beta double, mom_250_30 double, ret_5_1 double, vol_60_1 double, "
    "mkt_ret_20_1 double, abn_volume_5_1 double"
)


def build_spark(app: str = "filings-lab"):
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName(app)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.parquet.compression.codec", "zstd")
        .getOrCreate()
    )


def load_raw_documents(spark, raw_root: str, index_df):
    """Read raw documents with the binaryFile source and attach index metadata by accession."""
    from pyspark.sql import functions as F

    docs = (
        spark.read.format("binaryFile")
        .option("recursiveFileLookup", "true")
        .option("pathGlobFilter", "*.txt")
        .load(raw_root)
        .select(
            F.regexp_extract("path", r"([0-9]{10}-[0-9]{2}-[0-9]{6})\.txt$", 1).alias("accession"),
            "content",
        )
    )
    meta = index_df.select("accession", F.col("cik").cast("long").alias("cik"), "form", "date_filed")
    return docs.join(F.broadcast(meta) if index_df.count() < 2_000_000 else meta, "accession")


def _sections_fn(batches):
    for b in batches:
        s, _ = process_partition(b, reader=None)
        yield s[SECTION_COLUMNS]


def _paragraphs_fn(batches):
    for b in batches:
        _, p = process_partition(b, reader=None)
        yield p[PARAGRAPH_COLUMNS]


def silver_text(spark, docs):
    """Sections and paragraphs. Parsing is CPU bound Python, so repartition to use every core."""
    from pyspark.sql import functions as F

    n = max(spark.sparkContext.defaultParallelism * 3, 8)
    docs = docs.repartition(n, "accession").cache()
    year = F.year(F.col("date_filed"))
    meta = docs.select("accession", year.alias("year"))
    base = docs.select("accession", "cik", "form", "content")
    sections = base.mapInPandas(_sections_fn, SECTION_SCHEMA).join(meta, "accession")
    paragraphs = base.mapInPandas(_paragraphs_fn, PARAGRAPH_SCHEMA).join(meta, "accession")
    return sections, paragraphs


def gold_events(spark, index_df, prices_df, market: str = "SPY"):
    """Per ticker event study using cogroup: each task sees one ticker's prices and its filings."""
    from pyspark.sql import functions as F

    from ..prices.events import event_returns

    mkt_pd = prices_df.filter(F.col("ticker") == market).select("date", "close").toPandas()
    mkt_pd["date"] = pd.to_datetime(mkt_pd["date"])
    mkt_ret = mkt_pd.set_index("date")["close"].sort_index().pct_change()
    bmkt = spark.sparkContext.broadcast(mkt_ret)

    def per_ticker(px: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
        cols = [c.split(" ")[0] for c in EVENT_SCHEMA.split(", ")]
        if px.empty or ev.empty:
            return pd.DataFrame(columns=cols)
        t = ev["ticker"].iloc[0]
        px = px.assign(date=pd.to_datetime(px["date"])).sort_values("date").set_index("date")
        rets = pd.concat([px["close"].pct_change().rename(t), bmkt.value.rename("__mkt__")], axis=1).sort_index()
        vol = px[["volume"]].rename(columns={"volume": t})
        ev = ev.assign(accepted_at=pd.to_datetime(ev["accepted_at"], utc=True))
        out = event_returns(ev, rets, market_col="__mkt__", volume=vol)
        for c in cols:
            if c not in out.columns:
                out[c] = None
        out["accepted_at"] = out["accepted_at"].dt.tz_convert("UTC").dt.tz_localize(None)
        return out[cols]

    events = index_df.select("accession", F.col("cik").cast("long"), "ticker", "form", "accepted_at")
    return (
        prices_df.groupBy("ticker")
        .cogroup(events.groupBy("ticker"))
        .applyInPandas(per_ticker, EVENT_SCHEMA)
        .filter(F.col("car").isNotNull())
        .withColumn("year", F.year("event_date"))
    )


def add_spark_labels(events):
    """Sign label and within month tercile label, computed with window functions."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    w = Window.partitionBy(F.date_trunc("month", "event_date"))
    pct = F.percent_rank().over(w.orderBy("car"))
    return events.withColumn("label_up", (F.col("car") > 0).cast("tinyint")).withColumn(
        "label_tercile", F.when(pct < 1 / 3, 0).when(pct < 2 / 3, 1).otherwise(2).cast("tinyint")
    )


def run(index_path: str, raw_root: str, prices_path: str, lake_root: str, market: str = "SPY") -> None:
    spark = build_spark()
    index_df = spark.read.parquet(index_path)
    prices_df = spark.read.parquet(prices_path)
    docs = load_raw_documents(spark, raw_root, index_df)
    sections, paragraphs = silver_text(spark, docs)
    sections.write.mode("overwrite").partitionBy("form", "year").parquet(f"{lake_root}/silver/sections")
    paragraphs.write.mode("overwrite").partitionBy("form", "year").parquet(f"{lake_root}/silver/paragraphs")
    events = add_spark_labels(gold_events(spark, index_df, prices_df, market))
    events.write.mode("overwrite").partitionBy("year").parquet(f"{lake_root}/gold/events")
    prices_df.write.mode("overwrite").parquet(f"{lake_root}/silver/prices")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--lake", required=True)
    ap.add_argument("--market", default="SPY")
    a = ap.parse_args(argv)
    run(a.index, a.raw, a.prices, a.lake, a.market)


if __name__ == "__main__":
    main()
