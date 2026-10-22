"""Spark version of the feature build: per company text features via groupBy(cik).applyInPandas."""
from __future__ import annotations

import pandas as pd

from .build import company_text_features


def text_features_spark(spark, sections_df, index_df):
    """Returns a Spark DataFrame of per filing text features. The output schema is inferred from a
    sample company so that adding a feature never requires editing a schema string."""
    from pyspark.sql import functions as F

    sec = sections_df.join(index_df.select("accession", F.to_date("date_filed").alias("date_filed")), "accession")
    sample_cik = sec.select("cik").limit(1).collect()[0][0]
    sample = company_text_features(sec.filter(F.col("cik") == sample_cik).toPandas())
    schema = spark.createDataFrame(sample.head(1)).schema

    def fn(pdf: pd.DataFrame) -> pd.DataFrame:
        out = company_text_features(pdf)
        for f in schema.fieldNames():
            if f not in out.columns:
                out[f] = None
        return out[schema.fieldNames()]

    return sec.groupBy("cik").applyInPandas(fn, schema)
