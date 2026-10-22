"""Offline tests for the EVDS v3 client (synthetic fixtures in the documented response shape) + one network test.

The fixtures under ``tests/fixtures/macro`` are *illustrative*: they follow the EVDS JSON layout
(``items`` / ``Tarih`` / ``TP_FG_J0`` / ``UNIXTIME``) but their values are not authoritative TÜİK/TCMB
figures; no key was available when they were written.
"""

from __future__ import annotations

import gzip
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

from bistmas.data.clients import evds
from bistmas.data.clients.evds import (
    CPI,
    ENV_KEY_NAME,
    USDTRY_BUY,
    EvdsClient,
    EvdsError,
    MissingEvdsKeyError,
    build_url,
    load_api_key,
    parse_env_file,
    parse_evds_date,
    parse_response,
    pivot_series,
    raw_file_name,
)
from bistmas.shared.store import append_manifest, read_manifest, write_raw_gzip

FIXTURES = Path(__file__).parent / "fixtures" / "macro"
CPI_PAYLOAD = (FIXTURES / "evds_TP_FG_J0_monthly.json").read_bytes()
FX_PAYLOAD = (FIXTURES / "evds_TP_DK_USD_A_YTL_daily.json").read_bytes()


class _Resp:
    def __init__(self, status: int) -> None:
        self.status_code = status


class FakeSession:
    """Mimics PoliteSession.get_cached: payloads are matched by a substring of the URL."""

    def __init__(self, payloads: dict[str, bytes], status: int = 200) -> None:
        self.payloads = payloads
        self.status = status
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_cached(self, url: str, raw_path: Path, manifest_path=None, source="", force=False, **kw: Any) -> bytes:
        self.calls.append((url, kw))
        if raw_path.exists() and not force:
            with gzip.open(raw_path, "rb") as fh:
                return fh.read()
        if self.status != 200:
            raise requests.HTTPError(f"{self.status} for {url}", response=_Resp(self.status))
        for needle, payload in self.payloads.items():
            if needle in url:
                digest = write_raw_gzip(payload, raw_path)
                append_manifest(
                    manifest_path or raw_path.parent / "manifest.jsonl", {"source": source, "sha256": digest}
                )
                return payload
        raise requests.HTTPError(f"404 for {url}", response=_Resp(404))


# ----------------------------------------------------------------------------- key handling
def test_parse_env_file_handles_quotes_comments_and_export(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# comment\nexport EVDS_API_KEY='abc123'\nOTHER = \"x y\"\nBARE=value # trailing\nINVALID LINE\n\n",
        encoding="utf-8",
    )
    parsed = parse_env_file(env)
    assert parsed == {"EVDS_API_KEY": "abc123", "OTHER": "x y", "BARE": "value"}
    assert parse_env_file(tmp_path / "missing.env") == {}


def test_load_api_key_prefers_environment_then_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text(f"{ENV_KEY_NAME}=fromfile\n", encoding="utf-8")
    assert load_api_key(env_file=env, environ={}) == "fromfile"
    assert load_api_key(env_file=env, environ={ENV_KEY_NAME: " fromenv "}) == "fromenv"
    assert load_api_key(env_file=tmp_path / "none.env", environ={ENV_KEY_NAME: ""}) is None
    monkeypatch.setenv(ENV_KEY_NAME, "viaos")
    assert load_api_key(env_file=tmp_path / "none.env") == "viaos"


def test_missing_key_message_is_actionable() -> None:
    msg = evds.missing_key_message()
    assert evds.EVDS_REGISTER_URL in msg and ENV_KEY_NAME in msg and ".env" in msg and "DD_DEGER" in msg


# ----------------------------------------------------------------------------- request building
def test_build_url_and_raw_file_name() -> None:
    url = build_url([CPI, USDTRY_BUY], date(2018, 1, 1), "2018-12-31", frequency="monthly", aggregation="avg")
    assert url.startswith(evds.EVDS_BASE_URL)
    assert "series=TP.FG.J0-TP.DK.USD.A.YTL" in url
    assert "startDate=01-01-2018" in url and "endDate=31-12-2018" in url
    assert "type=json" in url and "frequency=5" in url and "aggregationTypes=avg" in url
    assert "key" not in url.lower().replace("keyless", "")
    assert raw_file_name(CPI, date(2018, 1, 1), date(2018, 12, 31)) == "TP_FG_J0_20180101_20181231.json.gz"
    assert raw_file_name(CPI, "2018-01-01", "2018-12-31", 5, "avg") == "TP_FG_J0_20180101_20181231_f5_avg.json.gz"
    with pytest.raises(ValueError):
        build_url("not a code", date(2018, 1, 1), date(2018, 1, 2))
    with pytest.raises(ValueError):
        build_url(CPI, date(2018, 1, 1), date(2018, 1, 2), frequency="fortnightly")
    with pytest.raises(ValueError):
        build_url(CPI, date(2018, 1, 1), date(2018, 1, 2), aggregation="median")


def test_parse_evds_date_formats() -> None:
    assert parse_evds_date("05-01-2018") == pd.Timestamp("2018-01-05")
    assert parse_evds_date("2018-1") == pd.Timestamp("2018-01-01")
    assert parse_evds_date("2018-12") == pd.Timestamp("2018-12-01")
    assert parse_evds_date("2018-Q3") == pd.Timestamp("2018-07-01")
    assert parse_evds_date("2018") == pd.Timestamp("2018-01-01")
    with pytest.raises(EvdsError):
        parse_evds_date("yesterday")


# ----------------------------------------------------------------------------- response parsing
def test_parse_response_monthly_fixture_drops_nulls_and_uses_first_of_month() -> None:
    df = parse_response(CPI_PAYLOAD, CPI)
    assert list(df.columns) == ["series", "date", "value"]
    assert len(df) == 7  # the null April row is dropped
    assert df["series"].unique().tolist() == [CPI]
    assert df["date"].iloc[0] == pd.Timestamp("2017-09-01") and df["date"].iloc[-1] == pd.Timestamp("2018-03-01")
    assert df["date"].dtype == "datetime64[ns]"
    assert df["value"].iloc[-1] == pytest.approx(340.24)


def test_parse_response_daily_fixture_and_dotted_keys() -> None:
    df = parse_response(FX_PAYLOAD, USDTRY_BUY)
    assert len(df) == 6  # nulls and the empty string are dropped
    assert df["date"].tolist()[:2] == [pd.Timestamp("2017-12-29"), pd.Timestamp("2018-01-02")]
    dotted = {"items": [{"Tarih": "02-01-2018", "TP.DK.USD.A.YTL": "3.5"}], "totalCount": 1}
    assert parse_response(json.dumps(dotted), USDTRY_BUY)["value"].tolist() == [3.5]
    assert parse_response(json.dumps(dotted), CPI).empty  # series not in the items


def test_parse_response_rejects_non_series_bodies() -> None:
    with pytest.raises(EvdsError):
        parse_response(b"<html>login</html>", CPI)
    with pytest.raises(EvdsError):
        parse_response(b'{"status":"403","message":"Required request header \'key\' is not present"}', CPI)


def test_pivot_series() -> None:
    long_df = pd.concat([parse_response(CPI_PAYLOAD, CPI), parse_response(FX_PAYLOAD, USDTRY_BUY)])
    wide = pivot_series(long_df)
    assert list(wide.columns) == [USDTRY_BUY, CPI]
    assert wide.loc[pd.Timestamp("2018-01-01"), CPI] == pytest.approx(334.44)
    assert pivot_series(evds.empty_series_frame()).empty


# ----------------------------------------------------------------------------- client
def test_client_fetches_once_then_serves_the_cache_without_a_key(tmp_path: Path) -> None:
    session = FakeSession({"series=TP.FG.J0": CPI_PAYLOAD})
    client = EvdsClient(session, tmp_path, api_key="secret")
    df = client.fetch_series([CPI], date(2018, 1, 1), date(2018, 12, 31))
    assert len(df) == 7
    raw = tmp_path / "evds" / "TP_FG_J0_20180101_20181231.json.gz"
    assert raw.exists()
    url, kwargs = session.calls[0]
    assert kwargs["headers"] == {"key": "secret"} and "key=" not in url
    manifest = read_manifest(tmp_path / "evds" / "manifest.jsonl")
    assert len(manifest) == 1 and "secret" not in json.dumps(manifest)

    keyless = EvdsClient(FakeSession({}), tmp_path, api_key=None, env_file=tmp_path / "absent.env")
    keyless.api_key = None
    assert not keyless.has_key
    again = keyless.fetch_series(CPI, date(2018, 1, 1), date(2018, 12, 31))
    pd.testing.assert_frame_equal(again, df)


def test_client_without_key_and_cache_raises_friendly_error(tmp_path: Path) -> None:
    client = EvdsClient(FakeSession({}), tmp_path, api_key=None, env_file=tmp_path / "absent.env")
    client.api_key = None
    with pytest.raises(MissingEvdsKeyError) as exc:
        client.fetch_series(CPI, date(2018, 1, 1), date(2018, 12, 31))
    assert "EVDS_API_KEY" in str(exc.value) and "not cached" in str(exc.value)
    assert not (tmp_path / "evds").exists()


def test_client_translates_403_into_missing_key_error(tmp_path: Path) -> None:
    client = EvdsClient(FakeSession({}, status=403), tmp_path, api_key="bad")
    with pytest.raises(MissingEvdsKeyError) as exc:
        client.fetch_one(CPI, date(2018, 1, 1), date(2018, 12, 31))
    assert "rejected" in str(exc.value)
    client = EvdsClient(FakeSession({}, status=500), tmp_path, api_key="bad")
    with pytest.raises(requests.HTTPError):
        client.fetch_one(CPI, date(2018, 1, 1), date(2018, 12, 31))


def test_client_discards_a_non_json_payload(tmp_path: Path) -> None:
    client = EvdsClient(FakeSession({"series=TP.FG.J0": b"<html>maintenance</html>"}), tmp_path, api_key="k")
    with pytest.raises(EvdsError):
        client.fetch_one(CPI, date(2018, 1, 1), date(2018, 12, 31))
    assert not (tmp_path / "evds" / "TP_FG_J0_20180101_20181231.json.gz").exists()


def test_fetch_series_by_year_chunks_requests_per_calendar_year(tmp_path: Path) -> None:
    session = FakeSession({"series=TP.DK.USD.A.YTL": FX_PAYLOAD})
    client = EvdsClient(session, tmp_path, api_key="k")
    df = client.fetch_series_by_year(USDTRY_BUY, date(2017, 11, 15), date(2018, 3, 1))
    names = sorted(p.name for p in (tmp_path / "evds").glob("*.json.gz"))
    assert names == ["TP_DK_USD_A_YTL_20171115_20171231.json.gz", "TP_DK_USD_A_YTL_20180101_20180301.json.gz"]
    assert len(df) == 6 and not df.duplicated(["series", "date"]).any()  # same fixture twice -> deduplicated
    with pytest.raises(ValueError):
        client.fetch_series_by_year(USDTRY_BUY, date(2018, 3, 1), date(2018, 1, 1))
    assert client.fetch_series([], date(2018, 1, 1), date(2018, 1, 2)).empty


# ----------------------------------------------------------------------------- network
@pytest.mark.network
def test_network_usdtry_january_2018(settings: Any) -> None:
    from bistmas.data.clients.http import PoliteSession

    if load_api_key() is None:
        pytest.skip("no EVDS_API_KEY configured (register at https://evds3.tcmb.gov.tr)")
    client = EvdsClient(PoliteSession(min_interval_s=1.5, timeout_s=60.0), settings.paths.raw)
    df = client.fetch_series(USDTRY_BUY, date(2018, 1, 1), date(2018, 1, 10))
    assert len(df) >= 5
    assert df["value"].between(3.5, 4.0).all()
