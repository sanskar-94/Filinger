"""Spark build of the gold feature table, equal to ``features.build.build_features`` (see the parity test).

1. Text features: ``groupBy("cik").applyInPandas`` runs the same ``company_text_features`` per company, so the
   year over year comparisons see a company's whole history in one task.
2. Fundamentals: ``events.groupBy("cik").cogroup(facts.groupBy("cik"))`` runs the same point in time join.
3. Derived columns (8-K item flags, form dummies, month, label end date) are native Spark expressions.
"""
from __future__ import annotations

import pandas as pd

from .build import EIGHTK_ITEMS, company_text_features, text_feature_columns

FIN_COLUMNS = ["fin_log_assets", "fin_leverage", "fin_roa", "fin_cash_ratio", "fin_accruals", "fin_rev_growth"]


def text_features_spark(spark, sections_df, index_df):
    from pyspark.sql import functions as F

    cols = text_feature_columns()
    schema = "accession string, " + ", ".join(f"{c} double" for c in cols)
    sec = sections_df.select("accession", "cik", "form", "section", "text").join(
        index_df.select("accession", F.to_date("date_filed").alias("date_filed")), "accession")

    def fn(pdf: pd.DataFrame) -> pd.DataFrame:
        pdf = pdf.assign(date_filed=pd.to_datetime(pdf["date_filed"]))
        out = company_text_features(pdf)
        return out.reindex(columns=["accession", *cols]).astype({c: "float64" for c in cols})

    return sec.groupBy("cik").applyInPandas(fn, schema)


def fundamentals_spark(spark, events_df, index_df, facts_df):
    from pyspark.sql import functions as F

    from .financial import fundamentals_asof

    schema = "accession string, " + ", ".join(f"{c} double" for c in FIN_COLUMNS)
    ev = events_df.select("accession", "cik").join(index_df.select("accession", F.to_date("date_filed").alias("date_filed")), "accession")

    def fn(events: pd.DataFrame, facts: pd.DataFrame) -> pd.DataFrame:
        if events.empty:
            return pd.DataFrame(columns=["accession", *FIN_COLUMNS])
        events = events.assign(date_filed=pd.to_datetime(events["date_filed"]))
        out = fundamentals_asof(facts if not facts.empty else None, events)
        return out.reindex(columns=["accession", *FIN_COLUMNS]).astype({c: "float64" for c in FIN_COLUMNS})

    return ev.groupBy("cik").cogroup(facts_df.groupBy("cik")).applyInPandas(fn, schema)


def build_features_spark(spark, lake: str, facts_path: str | None = None):
    from pyspark.sql import functions as F

    lake = lake.rstrip("/")
    index_df = spark.read.parquet(f"{lake}/bronze_index.parquet")
    sections = spark.read.parquet(f"{lake}/silver/sections")
    events = spark.read.parquet(f"{lake}/gold/events").drop("year")
    text = text_features_spark(spark, sections, index_df)
    meta = index_df.select("accession", F.to_timestamp("date_filed").alias("date_filed"), F.coalesce("items", F.lit("")).alias("items"))
    df = events.join(meta, "accession", "left").join(text, "accession", "left")
    if facts_path:
        facts = spark.read.parquet(facts_path)
        df = df.join(fundamentals_spark(spark, events, index_df, facts), "accession", "left")
    for it in EIGHTK_ITEMS:
        df = df.withColumn(f"item_{it.replace('.', '_')}", F.col("items").contains(it).cast("tinyint"))
    base_form = F.regexp_replace("form", "/A", "")
    for f in ("10-K", "10-Q", "8-K"):
        df = df.withColumn(f"form_{f.replace('-', '').lower()}", (base_form == f).cast("tinyint"))
    dow = F.dayofweek("event_date")  # 1 Sunday ... 7 Saturday; event dates are trading days
    df = (df.withColumn("month", F.month("event_date").cast("tinyint"))
            .withColumn("t1", F.expr("date_add(event_date, 0)").cast("timestamp"))
            .withColumn("t1", (F.col("t1") + F.when(dow.isin(5, 6), F.expr("INTERVAL 4 DAYS")).otherwise(F.expr("INTERVAL 2 DAYS"))))
            .withColumn("year", F.year("event_date"))
            .drop("items"))
    return df


def run(lake: str, facts_path: str | None = None) -> None:
    from ..pipeline.spark_job import build_spark

    spark = build_spark("filings-lab-features")
    build_features_spark(spark, lake, facts_path).write.mode("overwrite").partitionBy("year").parquet(f"{lake.rstrip('/')}/gold/features")
