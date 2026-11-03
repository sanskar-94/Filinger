"""``flab features``: build gold/features from the lake."""
from __future__ import annotations

from pathlib import Path


def _run(a):
    from .. import storage
    from .build import build_features

    lake = a.lake.rstrip("/")
    sections = storage.read_parquet(f"{lake}/silver/sections")
    events = storage.read_parquet(f"{lake}/gold/events")
    index = storage.read_parquet(f"{lake}/bronze_index.parquet")
    facts = storage.read_parquet(a.facts) if a.facts and storage.exists(a.facts) else None
    feats = build_features(sections, events.drop(columns=["year"], errors="ignore"), index, facts)
    out = f"{lake}/gold/features"
    storage.write_parquet(feats, out, ["year"])
    print(f"{len(feats)} rows x {feats.shape[1]} columns -> {out}")


def _register(p):
    p.add_argument("--lake", default="data/lake")
    p.add_argument("--facts", default="data/synthetic/facts.parquet", help="XBRL facts parquet (long format)")
    p.set_defaults(fn=_run)


def _run_spark(a):
    from .spark_features import run

    run(a.lake, a.facts if a.facts and Path(a.facts).exists() else None)


def _register_spark(p):
    p.add_argument("--lake", default="data/lake")
    p.add_argument("--facts", default="data/synthetic/facts.parquet")
    p.set_defaults(fn=_run_spark)


COMMANDS = {"features": (_register, "build the gold feature table"),
            "spark-features": (_register_spark, "build the gold feature table with PySpark")}
