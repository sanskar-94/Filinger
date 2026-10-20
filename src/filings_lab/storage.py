"""Minimal storage layer: the same code reads and writes local paths or s3:// URIs."""
from __future__ import annotations

from pathlib import Path


def _s3_split(uri: str) -> tuple[str, str]:
    bucket, _, key = uri[len("s3://"):].partition("/")
    return bucket, key


def write_bytes(uri: str, data: bytes) -> None:
    if uri.startswith("s3://"):
        import boto3

        bucket, key = _s3_split(uri)
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=data)
        return
    p = Path(uri)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def read_bytes(uri: str) -> bytes:
    if uri.startswith("s3://"):
        import boto3

        bucket, key = _s3_split(uri)
        return boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
    return Path(uri).read_bytes()


def exists(uri: str) -> bool:
    if uri.startswith("s3://"):
        import boto3
        from botocore.exceptions import ClientError

        bucket, key = _s3_split(uri)
        try:
            boto3.client("s3").head_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False
    return Path(uri).exists()


def join(root: str, *parts: str) -> str:
    return "/".join([root.rstrip("/"), *[p.strip("/") for p in parts]])
