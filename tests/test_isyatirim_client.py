"""Offline tests for bistmas.data.clients.isyatirim (fixture trimmed from the live THYAO 2018 payload of 2026-09-23)
plus one network test that re-uses the shared raw layer."""

from __future__ import annotations

import gzip
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from bistmas.data.clients import isyatirim as iy
from bistmas.data.clients.http import PoliteSession
from bistmas.data.clients.isyatirim import (
    CONTRACT_COLUMNS,
    FIELD_MAP,
    HISSETEKIL_URL,
    PARSED_COLUMNS,
    IsYatirimClient,
    IsYatirimError,
    build_url,
    concat_years,
    iter_cached_files,
    parse_hissetekil,
    parse_session_date,
    validate_payload,
    volume_semantics,
    year_range,
)
from bistmas.shared.store import read_manifest, write_raw_gzip

FIXTURES = Path(__file__).parent / "fixtures" / "isyatirim"
TODAY = date(2026, 9, 23)


@pytest.fixture(scope="module")
def thyao_payload() -> bytes:
    return (FIXTURES / "THYAO_2018_head.json").read_bytes()


@pytest.fixture(scope="module")
def empty_payload() -> bytes:
    return (FIXTURES / "MRDIN_2018_empty.json").read_bytes()


class OfflineSession(PoliteSession):
    """PoliteSession that refuses to touch the network (cache hits only)."""

    def request(self, method: str, url: str, **kwargs: Any) -> Any:  # pragma: no cover - guard
        raise AssertionError(f"unexpected network call: {method} {url}")


class FakeResponse:
    def __init__(self, content: bytes, status: int = 200) -> None:
        self.content = content
        self.status_code = status


class FakeSession(PoliteSession):
    """Serves canned bodies per URL through the real ``get_cached`` (raw layer + manifest), no sleeping."""

    def __init__(self, payloads: dict[str, bytes]) -> None:
        super().__init__(min_interval_s=1.5)
        self.payloads = payloads
        self.calls: list[str] = []

    def _wait_for_slot(self, host: str) -> None:
        return None

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        self.calls.append(url)
        if url not in self.payloads:
            return FakeResponse(b"not found", 404)
        return FakeResponse(self.payloads[url])


# ----------------------------------------------------------------------------- fixtures / URL
def test_fixtures_stay_small_and_keep_the_envelope(thyao_payload: bytes) -> None:
    for path in FIXTURES.iterdir():
        assert path.stat().st_size < 150_000, path
    obj = json.loads(thyao_payload)
    assert list(obj) == ["ok", "errorCode", "errorDescription", "transactionId", "value"]
    assert obj["ok"] is True and len(obj["value"]) == 20
    assert set(FIELD_MAP) <= set(obj["value"][0]), "every mapped field exists in the real payload"
    assert len(obj["value"][0]) == 31


def test_year_range_and_url() -> None:
    assert year_range(2018, TODAY) == ("01-01-2018", "31-12-2018")
    assert year_range(2026, TODAY) == ("01-01-2026", "23-09-2026")  # current year ends today
    with pytest.raises(ValueError):
        year_range(2027, TODAY)
    with pytest.raises(ValueError):
        year_range(1999, TODAY)
    assert build_url("thyao.e", 2018, TODAY) == f"{HISSETEKIL_URL}?hisse=THYAO&startdate=01-01-2018&enddate=31-12-2018"
    with pytest.raises(ValueError):
        build_url("TGB", 2018, TODAY)  # three-letter bank instrument, not an equity


# ----------------------------------------------------------------------------- dates
def test_parse_session_date_formats() -> None:
    assert parse_session_date("02-01-2018") == pd.Timestamp("2018-01-02")
    assert parse_session_date("2-1-2018") == pd.Timestamp("2018-01-02")
    # epoch ms at Istanbul midnight (the END_TARIH / DD_TARIH format): 1514840400000 = 2018-01-02 00:00 +03
    assert parse_session_date(1514840400000) == pd.Timestamp("2018-01-02")
    assert parse_session_date("/Date(1514840400000)/") == pd.Timestamp("2018-01-02")
    assert parse_session_date("2018-01-02T00:00:00") == pd.Timestamp("2018-01-02")
    assert pd.isna(parse_session_date(None))
    assert pd.isna(parse_session_date(float("nan")))
    assert pd.isna(parse_session_date("31-02-2018"))
    assert pd.isna(parse_session_date("garbage"))


# ----------------------------------------------------------------------------- parsing
def test_parse_hissetekil_columns_dtypes_values(thyao_payload: bytes) -> None:
    df = parse_hissetekil(thyao_payload, "THYAO")
    assert list(df.columns) == list(PARSED_COLUMNS)
    assert len(df) == 20
    assert str(df["date"].dtype) == "datetime64[ns]"
    assert df["date"].is_monotonic_increasing and df["date"].is_unique
    assert df["date"].iloc[0] == pd.Timestamp("2018-01-02")
    assert df["date"].iloc[-1] == pd.Timestamp("2018-01-29")
    assert (df["ticker"] == "THYAO").all()
    for col in CONTRACT_COLUMNS:
        assert df[col].dtype == np.float64, col
    first = df.iloc[0]
    assert first["close_raw"] == 16.08 and first["close_adj"] == 15.7083
    assert first["vwap_raw"] == 16.022 and first["low_raw"] == 15.67 and first["high_raw"] == 16.18
    assert first["shares"] == 1_380_000_000.0
    assert first["usdtry"] == 3.7719 and first["xu100"] == 1175.242 and first["price_usd"] == 4.1646
    assert first["mcap_tl"] == pytest.approx(22_190_399_894.71)
    assert first["ffloat_mcap_tl"] == pytest.approx(11_095_199_947.36)
    assert first["mcap_usd"] == pytest.approx(5_883_082_768.56)
    assert first["ffloat_mcap_usd"] == pytest.approx(2_941_541_384.28)
    assert first["volume_raw"] == first["volume_adj"] == 709_012_777.0
    assert first["volume_usd"] == pytest.approx(187_972_315.54)
    assert first["xu100_session"] == 2.0 and first["price_index_rel"] == 0.0134
    assert not df.isna().any().any()


def test_parse_hissetekil_nulls_missing_fields_and_duplicates(thyao_payload: bytes) -> None:
    obj = json.loads(thyao_payload)
    rows = obj["value"][:3]
    rows[0]["HAO_PD"] = None  # nullable in early years
    rows[1]["HG_KAPANIS"] = ""  # blank -> NaN
    del rows[2]["END_DEGER"]  # missing key on one row
    rows.append(dict(rows[0], HG_KAPANIS=99.0))  # duplicate date: last wins
    rows.append({"HGDG_TARIH": "bad-date", "HG_KAPANIS": 1.0})
    for r in rows:
        r.pop("DOLAR_HACIM", None)  # field absent from the whole payload -> all-NaN column
    obj["value"] = rows
    df = parse_hissetekil(json.dumps(obj).encode(), "THYAO")
    assert len(df) == 3  # bad date dropped, duplicate merged
    assert list(df.columns) == list(PARSED_COLUMNS)
    assert np.isnan(df.loc[0, "ffloat_mcap_tl"])
    assert df.loc[0, "close_raw"] == 99.0
    assert np.isnan(df.loc[1, "close_raw"])
    assert np.isnan(df.loc[2, "xu100"])
    assert df["volume_usd"].isna().all()


def test_parse_hissetekil_empty_payload(empty_payload: bytes) -> None:
    df = parse_hissetekil(empty_payload, "MRDIN")
    assert len(df) == 0
    assert list(df.columns) == list(PARSED_COLUMNS)
    assert str(df["date"].dtype) == "datetime64[ns]"
    assert df["close_raw"].dtype == np.float64


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"<html><body>Access denied</body></html>",
        b"\n  <!DOCTYPE html><html>WAF</html>",
        b"not json at all",
        b'{"ok": true}',
        b'{"value": {"a": 1}}',
        b'{"value": [1, 2]}',
        b'{"ok": false, "errorCode": "E1", "errorDescription": "Teknik Hata", "value": []}',
        b'[{"HGDG_TARIH": "02-01-2018"}]',
    ],
)
def test_validate_payload_rejects_non_hissetekil_bodies(body: bytes) -> None:
    with pytest.raises(IsYatirimError):
        validate_payload(body)


def test_validate_payload_accepts_bom_and_rows_without_date_field_are_reported() -> None:
    assert validate_payload(b'\xef\xbb\xbf{"value": []}') == []
    with pytest.raises(IsYatirimError, match="HGDG_TARIH"):
        parse_hissetekil(b'{"value": [{"HG_KAPANIS": 1.0}]}', "THYAO")


# ----------------------------------------------------------------------------- client / raw layer
def test_fetch_year_writes_raw_layer_then_serves_from_cache(tmp_path: Path, thyao_payload: bytes) -> None:
    url = build_url("THYAO", 2018, TODAY)
    session = FakeSession({url: thyao_payload})
    client = IsYatirimClient(session, tmp_path)
    df = client.fetch_year("THYAO", 2018, today=TODAY)
    assert len(df) == 20
    raw = tmp_path / "isyatirim" / "THYAO" / "2018.json.gz"
    assert raw.exists() and client.raw_path("thyao", 2018) == raw
    assert gzip.open(raw).read() == thyao_payload
    manifest = read_manifest(tmp_path / "isyatirim" / "manifest.jsonl")
    assert len(manifest) == 1 and manifest[0]["url"] == url and manifest[0]["source"] == "isyatirim"
    assert session.calls == [url]
    again = client.fetch_year("THYAO", 2018, today=TODAY)
    assert session.calls == [url], "second call is a cache hit"
    pd.testing.assert_frame_equal(again, df)
    assert client.is_cached("THYAO", 2018) and not client.is_cached("THYAO", 2019)
    assert list(iter_cached_files(tmp_path)) == [("THYAO", 2018, raw)]
    pd.testing.assert_frame_equal(client.load_cached("THYAO", 2018), df)
    assert client.load_cached("THYAO", 2019) is None


def test_fetch_year_offline_from_prepopulated_cache(tmp_path: Path, thyao_payload: bytes) -> None:
    write_raw_gzip(thyao_payload, tmp_path / "isyatirim" / "THYAO" / "2018.json.gz")
    client = IsYatirimClient(OfflineSession(min_interval_s=1.5), tmp_path)
    df = client.fetch_year("THYAO", 2018)
    assert len(df) == 20 and df["ticker"].iloc[0] == "THYAO"


def test_fetch_year_caches_empty_result_as_valid(tmp_path: Path, empty_payload: bytes) -> None:
    url = build_url("MRDIN", 2018, TODAY)
    session = FakeSession({url: empty_payload})
    client = IsYatirimClient(session, tmp_path)
    df = client.fetch_year("MRDIN", 2018, today=TODAY)
    assert len(df) == 0 and list(df.columns) == list(PARSED_COLUMNS)
    assert client.is_cached("MRDIN", 2018)
    client.fetch_year("MRDIN", 2018, today=TODAY)
    assert session.calls == [url], "an empty answer is never re-asked"


def test_fetch_year_discards_html_body_and_raises(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    url = build_url("THYAO", 2018, TODAY)
    session = FakeSession({url: b"<html><body>Please enable JavaScript</body></html>"})
    client = IsYatirimClient(session, tmp_path)
    with caplog.at_level(logging.WARNING, logger="bistmas.data.isyatirim"):
        with pytest.raises(IsYatirimError, match="THYAO 2018"):
            client.fetch_year("THYAO", 2018, today=TODAY)
    assert not client.raw_path("THYAO", 2018).exists(), "invalid body must not stay in the raw layer"
    assert "discarding" in caplog.text
    # the next call asks again (nothing cached)
    session.payloads[url] = (FIXTURES / "THYAO_2018_head.json").read_bytes()
    assert len(client.fetch_year("THYAO", 2018, today=TODAY)) == 20
    assert session.calls == [url, url]


def test_fetch_year_http_error_propagates_without_caching(tmp_path: Path) -> None:
    import requests

    session = FakeSession({})  # every URL -> 404
    client = IsYatirimClient(session, tmp_path)
    with pytest.raises(requests.HTTPError):
        client.fetch_year("THYAO", 2018, today=TODAY)
    assert not client.raw_path("THYAO", 2018).exists()


def test_client_enforces_spacing_floor(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    session = OfflineSession(min_interval_s=0.5)
    with caplog.at_level(logging.WARNING, logger="bistmas.data.isyatirim"):
        IsYatirimClient(session, tmp_path)
    assert session.min_interval_s == iy.MIN_INTERVAL_S
    assert "raising session min_interval_s" in caplog.text
    polite = iy.make_session()
    assert polite.min_interval_s >= 1.5 and polite.timeout_s == 60.0 and polite.max_retries == 4


def test_fetch_history_concatenates_and_dedups(tmp_path: Path, thyao_payload: bytes) -> None:
    obj = json.loads(thyao_payload)
    rows_2018 = obj["value"]
    rows_2019 = [dict(r, HGDG_TARIH=r["HGDG_TARIH"].replace("2018", "2019")) for r in rows_2018[:5]]
    rows_2019.insert(0, dict(rows_2018[-1], HG_KAPANIS=123.0))  # overlap with 2018's last date
    payloads = {
        build_url("THYAO", 2018, TODAY): thyao_payload,
        build_url("THYAO", 2019, TODAY): json.dumps(dict(obj, value=rows_2019)).encode(),
        build_url("THYAO", 2020, TODAY): json.dumps(dict(obj, value=[])).encode(),
    }
    client = IsYatirimClient(FakeSession(payloads), tmp_path)
    df = client.fetch_history("THYAO", 2018, 2020)
    assert len(df) == 25
    assert df["date"].is_monotonic_increasing and df["date"].is_unique
    assert df.loc[df["date"] == pd.Timestamp("2018-01-29"), "close_raw"].item() == 123.0  # later chunk wins
    assert df.index.tolist() == list(range(25))
    assert len(concat_years([])) == 0 and list(concat_years([]).columns) == list(PARSED_COLUMNS)


# ----------------------------------------------------------------------------- volume semantics
def test_volume_semantics_detects_tl_value_and_shares(thyao_payload: bytes) -> None:
    df = parse_hissetekil(thyao_payload, "THYAO")
    # real payload: HG_HACIM == HGDG_HACIM although close_adj/close_raw = 0.977 (dividend adjustment)
    real = volume_semantics(df)
    assert real["verdict"] == "tl_value"
    assert real["median_volume_ratio"] == pytest.approx(1.0)
    assert real["median_adj_factor"] == pytest.approx(0.9769, abs=1e-3)
    assert real["median_value_ratio"] == pytest.approx(0.9769, abs=2e-3)
    # synthetic share-count semantics: adjusted shares scale inversely with the price factor
    shares = df.copy()
    shares["volume_raw"] = shares["volume_raw"] / shares["vwap_raw"]
    shares["volume_adj"] = shares["volume_raw"] * shares["vwap_raw"] / shares["vwap_adj"]
    assert volume_semantics(shares)["verdict"] == "shares"
    # no adjusted day at all -> undecidable
    flat = df.copy()
    flat["close_adj"] = flat["close_raw"]
    assert volume_semantics(flat)["verdict"] == "unknown"
    assert volume_semantics(df.iloc[0:0])["verdict"] == "unknown"


# ----------------------------------------------------------------------------- network
@pytest.mark.network
def test_live_thyao_2018_from_shared_raw_layer(settings) -> None:
    """Uses the project raw layer, so after the first run this is a cache hit (no extra load on the site)."""
    client = IsYatirimClient(iy.make_session(), settings.paths.raw)
    df = client.fetch_year("THYAO", 2018)
    assert len(df) == 251
    assert df["date"].iloc[0] == pd.Timestamp("2018-01-02") and df["date"].iloc[-1] == pd.Timestamp("2018-12-31")
    assert volume_semantics(df)["verdict"] == "tl_value"
    assert client.is_cached("THYAO", 2018)
