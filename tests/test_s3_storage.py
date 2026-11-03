"""The lake runs on S3, not only on local disk.

A moto server stands in for S3 over real HTTP, so this exercises the same code paths as production: boto3 for raw
filings and PyArrow's native S3 filesystem for the Parquet partitions. Raw filings are uploaded, the pipeline reads
them from S3 and writes the lake to S3, then the feature build and the data quality gate read it back. Every table
must equal the one from a local run of the same inputs.
"""
import socket

import pandas as pd
import pytest

pytest.importorskip("moto.server")
boto3 = pytest.importorskip("boto3")

from filings_lab import storage  # noqa: E402
from filings_lab.pipeline import local  # noqa: E402
from filings_lab.pipeline.quality import run_checks  # noqa: E402

BUCKET = "flab-lake-test"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def s3(monkeypatch):
    from moto.server import ThreadedMotoServer

    port = _free_port()
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=port, verbose=False)
    server.start()
    endpoint = f"http://127.0.0.1:{port}"
    for k, v in {"AWS_ENDPOINT_URL": endpoint, "AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing",
                 "AWS_REGION": "us-east-1"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    boto3.client("s3", endpoint_url=endpoint).create_bucket(Bucket=BUCKET)
    yield f"s3://{BUCKET}"
    server.stop()


def _canon(df: pd.DataFrame, key: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:  # partition columns come back as categoricals; compare values, not dtypes
        if isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].astype(str)
        elif c in ("year",):
            df[c] = df[c].astype(str)
    return df.sort_values(key).reset_index(drop=True)[sorted(df.columns)]


def test_storage_primitives_round_trip(s3):
    uri = storage.join(s3, "raw", "320193", "0000320193-24-000001.txt")
    assert not storage.exists(uri)
    storage.write_bytes(uri, b"<SEC-DOCUMENT>hello")
    assert storage.exists(uri) and storage.read_bytes(uri) == b"<SEC-DOCUMENT>hello"
    assert storage.exists(storage.join(s3, "raw"))  # prefix with objects under it counts as present
    assert not storage.exists(storage.join(s3, "nothing-here"))
    storage.makedirs(storage.join(s3, "silver"))  # no op on object stores
    df = pd.DataFrame({"a": [1, 2, 3], "year": [2020, 2020, 2021]})
    storage.write_parquet(df, storage.join(s3, "t"), ["year"])
    back = storage.read_parquet(storage.join(s3, "t"))
    assert sorted(back["a"].tolist()) == [1, 2, 3]
    storage.write_parquet(df.iloc[:1], storage.join(s3, "t"), ["year"])  # rewrites 2020, keeps 2021
    assert sorted(storage.read_parquet(storage.join(s3, "t"))["a"].tolist()) == [1, 3]


def test_pipeline_features_and_quality_on_s3_match_local(s3, small_world, small_lake, tmp_path):
    from filings_lab.features.build import build_features

    idx = small_world["index"].copy()
    s3_uris = []
    for cik, acc, path in zip(idx["cik"], idx["accession"], idx["raw_uri"], strict=True):
        uri = storage.join(s3, "raw", str(int(cik)), f"{acc}.txt")
        storage.write_bytes(uri, open(path, "rb").read())
        s3_uris.append(uri)
    idx["raw_uri"] = s3_uris
    lake = storage.join(s3, "lake")
    counts = local.run(idx, small_world["prices"], lake, workers=1)
    assert counts["paragraphs"] > 0

    for table, key in [("silver/sections", ["accession", "section"]), ("silver/paragraphs", ["paragraph_id"]),
                       ("gold/events", ["accession"])]:
        remote = storage.read_parquet(storage.join(lake, table))
        here = pd.read_parquet(small_lake / table)
        pd.testing.assert_frame_equal(_canon(remote, key), _canon(here, key), check_dtype=False)

    # the feature build reads the S3 lake and writes gold/features back to it
    from argparse import Namespace

    from filings_lab.features.cli import _run

    _run(Namespace(lake=lake, facts=None))
    remote_feats = storage.read_parquet(storage.join(lake, "gold", "features"))
    local_feats = build_features(pd.read_parquet(small_lake / "silver/sections"),
                                 pd.read_parquet(small_lake / "gold/events").drop(columns=["year"]),
                                 pd.read_parquet(small_lake / "bronze_index.parquet"), None)
    assert len(remote_feats) == len(local_feats)
    pd.testing.assert_frame_equal(_canon(remote_feats, ["accession"]), _canon(local_feats, ["accession"]),
                                  check_dtype=False, check_like=True)

    checks = run_checks(lake)
    assert {c.table for c in checks} == {"bronze_index", "silver_sections", "silver_paragraphs", "silver_prices",
                                         "gold_events", "gold_features"}
    assert not [c for c in checks if c.status == "fail"], [c for c in checks if c.status == "fail"]
