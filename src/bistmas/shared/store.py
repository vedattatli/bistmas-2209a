"""Storage helpers: atomic Parquet/JSON writes and append-only JSONL manifests.

The raw layer (data/raw) is immutable: every fetched payload is written once, gzip-compressed,
with a manifest line recording source, request, fetch time and sha256. The feature store
(data/store) is rebuilt from raw and versioned by the raw-manifest hash.
"""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from bistmas.shared.hashing import sha256_bytes


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Atomic Parquet write; ``DataFrame.attrs`` are made JSON-safe (Timestamps -> ISO strings) first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if df.attrs:
        df = df.copy()
        df.attrs = json.loads(json.dumps(df.attrs, default=str))
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def write_json(obj: Any, path: Path) -> Path:
    _atomic_write_bytes(path, json.dumps(obj, indent=2, ensure_ascii=False, default=str).encode("utf-8"))
    return path


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_raw_gzip(payload: bytes, path: Path) -> str:
    """Write a raw response payload gzip-compressed; returns sha256 of the *uncompressed* payload."""
    _atomic_write_bytes(path, gzip.compress(payload, compresslevel=6))
    return sha256_bytes(payload)


def read_raw_gzip(path: Path) -> bytes:
    with gzip.open(path, "rb") as fh:
        return fh.read()


def append_manifest(manifest_path: Path, record: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"written_at": utc_now_iso(), **record}
    with open(manifest_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def read_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return []
    with open(manifest_path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
