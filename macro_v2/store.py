"""Local storage in the Bronze/Silver/Gold layout, plus upload to S3."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

BRONZE = "bronze/macro_v2"
SILVER = "silver/macro_v2"
GOLD = "gold/macro_v2"


class Store:
    """Writes everything under one local root that mirrors the S3 key layout."""

    def __init__(self, local_dir: str | Path):
        self.root = Path(local_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, rel: str) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def exists(self, rel: str) -> bool:
        return (self.root / rel).exists()

    def write_parquet(self, df: pd.DataFrame, rel: str, index: bool = False) -> Path:
        p = self.path(rel)
        df.to_parquet(p, index=index)
        log.info("wrote %s (%d rows)", rel, len(df))
        return p

    def read_parquet(self, rel: str) -> pd.DataFrame:
        return pd.read_parquet(self.root / rel)

    def write_json(self, obj, rel: str) -> Path:
        p = self.path(rel)
        p.write_text(json.dumps(obj, indent=2, default=str))
        log.info("wrote %s", rel)
        return p

    def read_json(self, rel: str):
        return json.loads((self.root / rel).read_text())

    def write_csv(self, df: pd.DataFrame, rel: str) -> Path:
        p = self.path(rel)
        df.to_csv(p, index=False)
        log.info("wrote %s (%d rows)", rel, len(df))
        return p


def upload_to_s3(local_root: str | Path, bucket: str, region: str = "us-east-2",
                 prefixes: tuple[str, ...] = (BRONZE, SILVER, GOLD)) -> int:
    """Upload every file under the given prefixes, keeping the same relative keys.

    Credentials come from the standard boto3 chain (the EC2 instance role on AWS).
    """
    import boto3

    s3 = boto3.client("s3", region_name=region)
    root = Path(local_root)
    count = 0
    for prefix in prefixes:
        base = root / prefix
        if not base.exists():
            continue
        for f in sorted(base.rglob("*")):
            if f.is_file():
                key = f.relative_to(root).as_posix()
                s3.upload_file(str(f), bucket, key)
                count += 1
                log.info("uploaded s3://%s/%s", bucket, key)
    return count
