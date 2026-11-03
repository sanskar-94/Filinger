"""Minimal storage layer: the same code reads and writes local paths or s3:// URIs.

Raw filings go through boto3 (one object per filing). Parquet tables go through PyArrow's native S3 filesystem,
which writes partitioned datasets directly and needs no extra Python dependency. Both honour the standard AWS
environment: credentials from the default chain, ``AWS_REGION``, and ``AWS_ENDPOINT_URL_S3`` or
``AWS_ENDPOINT_URL`` for S3 compatible stores (MinIO, LocalStack, the moto server used in the tests).
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse


def _s3_split(uri: str) -> tuple[str, str]:
    bucket, _, key = uri[len("s3://"):].partition("/")
    return bucket, key


def is_remote(uri: str) -> bool:
    return str(uri).startswith("s3://")


def _endpoint() -> str | None:
    return os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL") or None


def _client():
    import boto3

    return boto3.client("s3", endpoint_url=_endpoint())


def write_bytes(uri: str, data: bytes) -> None:
    if is_remote(uri):
        bucket, key = _s3_split(uri)
        _client().put_object(Bucket=bucket, Key=key, Body=data)
        return
    p = Path(uri)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def read_bytes(uri: str) -> bytes:
    if is_remote(uri):
        bucket, key = _s3_split(uri)
        return _client().get_object(Bucket=bucket, Key=key)["Body"].read()
    return Path(uri).read_bytes()


def exists(uri: str) -> bool:
    """True for an existing file or object, or a directory or prefix with at least one object under it."""
    if is_remote(uri):
        from botocore.exceptions import ClientError

        bucket, key = _s3_split(uri)
        client = _client()
        try:
            client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            pass
        resp = client.list_objects_v2(Bucket=bucket, Prefix=key.rstrip("/") + "/", MaxKeys=1)
        return resp.get("KeyCount", 0) > 0
    return Path(uri).exists()


def list_uris(prefix: str, suffix: str = "") -> list[str]:
    """Every file or object under a directory or prefix whose name ends with ``suffix``, sorted."""
    if is_remote(prefix):
        bucket, key = _s3_split(prefix)
        pages = _client().get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=key)
        return sorted(f"s3://{bucket}/{o['Key']}" for page in pages for o in page.get("Contents", [])
                      if o["Key"].endswith(suffix))
    p = Path(prefix)
    if p.is_dir():
        return sorted(str(x) for x in p.rglob(f"*{suffix}") if x.is_file())
    return [str(p)] if p.exists() else []


def makedirs(uri: str) -> None:
    """Create a local directory; a no op for object stores, where prefixes need no creation."""
    if not is_remote(uri):
        Path(uri).mkdir(parents=True, exist_ok=True)


def join(root: str, *parts: str) -> str:
    return "/".join([root.rstrip("/"), *[p.strip("/") for p in parts]])


def arrow_fs(uri: str):
    """(pyarrow filesystem, path) for Parquet IO; ``(None, uri)`` for local paths."""
    if not is_remote(uri):
        return None, uri
    from pyarrow import fs

    kwargs = {"region": os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"}
    if endpoint := _endpoint():
        u = urlparse(endpoint)
        kwargs.update(endpoint_override=u.netloc or u.path, scheme=u.scheme or "https")
    return fs.S3FileSystem(**kwargs), uri[len("s3://"):]


def read_parquet(uri: str, **kwargs):
    import pandas as pd

    filesystem, path = arrow_fs(uri)
    return pd.read_parquet(path, filesystem=filesystem, **kwargs)


def write_parquet(df, uri: str, partition_cols: list[str] | None = None) -> None:
    """Write a table, or a Hive partitioned dataset when ``partition_cols`` is given. Re-running replaces the
    partitions it writes and leaves the others, so a backfill of one year does not wipe the lake."""
    filesystem, path = arrow_fs(uri)
    if filesystem is None:
        Path(uri if partition_cols else Path(uri).parent).mkdir(parents=True, exist_ok=True)
    if partition_cols:
        df.to_parquet(path, filesystem=filesystem, partition_cols=partition_cols, index=False,
                      existing_data_behavior="delete_matching")
    else:
        df.to_parquet(path, filesystem=filesystem, index=False)
