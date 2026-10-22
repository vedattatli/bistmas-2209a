import pandas as pd

from bistmas.shared.store import (
    append_manifest,
    read_manifest,
    read_parquet,
    read_raw_gzip,
    write_parquet,
    write_raw_gzip,
)


def test_parquet_roundtrip(tmp_path):
    df = pd.DataFrame({"ticker": ["THYAO", "ASELS"], "close": [1.5, 2.5]})
    path = write_parquet(df, tmp_path / "x" / "prices.parquet")
    pd.testing.assert_frame_equal(read_parquet(path), df)


def test_raw_gzip_roundtrip_and_manifest(tmp_path):
    payload = b'{"value": [1, 2, 3]}'
    digest = write_raw_gzip(payload, tmp_path / "raw" / "a.json.gz")
    assert read_raw_gzip(tmp_path / "raw" / "a.json.gz") == payload
    append_manifest(tmp_path / "raw" / "manifest.jsonl", {"source": "test", "sha256": digest})
    rows = read_manifest(tmp_path / "raw" / "manifest.jsonl")
    assert rows[0]["sha256"] == digest and "written_at" in rows[0]
