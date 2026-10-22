"""Offline tests for the daily macro table (CPI availability lag, FX alignment, policy rate, store build)."""

from __future__ import annotations

import gzip
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from bistmas.data import macro
from bistmas.data.clients.evds import CPI, USDTRY_BUY, EvdsClient, parse_response
from bistmas.data.clients.tcmb_rates import POLICY_COLUMNS, POLICY_RATE_URL
from bistmas.data.macro import (
    MACRO_COLUMNS,
    MACRO_FILENAME,
    MACRO_META_FILENAME,
    build_macro_frame,
    build_macro_store,
    calendar_dates_from_store,
    cpi_release_date,
    cpi_to_sessions,
    fx_to_sessions,
    load_macro,
    splice_cpi,
)
from bistmas.shared.config import PathsConfig, Settings
from bistmas.shared.store import append_manifest, write_parquet, write_raw_gzip

FIXTURES = Path(__file__).parent / "fixtures" / "macro"
CPI_PAYLOAD = (FIXTURES / "evds_TP_FG_J0_monthly.json").read_bytes()
FX_PAYLOAD = (FIXTURES / "evds_TP_DK_USD_A_YTL_daily.json").read_bytes()
TCMB_PAGE = (FIXTURES / "tcmb_one_week_repo_snippet.html").read_bytes()


class FakeSession:
    """Serves EVDS payloads by URL substring and the TCMB page by exact URL; writes the raw layer like get_cached."""

    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.fetches: list[str] = []

    def get_cached(self, url: str, raw_path: Path, manifest_path=None, source="", force=False, **kw: Any) -> bytes:
        if raw_path.exists() and not force:
            with gzip.open(raw_path, "rb") as fh:
                return fh.read()
        for needle, payload in self.payloads.items():
            if needle in url:
                self.fetches.append(url)
                digest = write_raw_gzip(payload, raw_path)
                append_manifest(
                    manifest_path or raw_path.parent / "manifest.jsonl", {"source": source, "sha256": digest}
                )
                return payload
        raise AssertionError(f"unexpected network call: {url}")


def _settings(tmp_path: Path) -> Settings:
    paths = PathsConfig().resolve(tmp_path)
    for p in (paths.raw, paths.store, paths.runs):
        p.mkdir(parents=True, exist_ok=True)
    return Settings(root=tmp_path, paths=paths)


def _sessions(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


# ----------------------------------------------------------------------------- CPI
def test_cpi_release_date_is_third_of_next_month_rolled_to_a_session() -> None:
    assert cpi_release_date("2017-12-01") == pd.Timestamp("2018-01-03")  # Wednesday
    assert cpi_release_date("2018-01-15") == pd.Timestamp("2018-02-05")  # 3 Feb 2018 is a Saturday -> Monday
    sessions = pd.DatetimeIndex(["2018-01-02", "2018-01-04", "2018-01-05"])  # no session on the 3rd
    assert cpi_release_date("2017-12-01", sessions) == pd.Timestamp("2018-01-04")
    assert cpi_release_date("2018-01-01", sessions) == pd.Timestamp("2018-02-05")  # beyond the sessions: weekday roll


def test_cpi_to_sessions_spreads_the_month_and_lags_the_pit_value() -> None:
    monthly = parse_response(CPI_PAYLOAD, CPI)[["date", "value"]]
    sessions = _sessions("2018-01-02", "2018-03-30")
    out = cpi_to_sessions(monthly, sessions)
    assert list(out.columns) == ["date", "cpi", "cpi_available_from", "cpi_pit"]
    jan = out[out["date"].dt.month == 1]
    assert (jan["cpi"] == 334.44).all()
    assert (jan["cpi_available_from"] == pd.Timestamp("2018-02-05")).all()
    # point in time: November's index until December's release on Wed 3 Jan, December's until the January
    # release on Monday 5 Feb (the 3rd is a Saturday), then January's
    assert out.loc[out["date"] == pd.Timestamp("2018-01-02"), "cpi_pit"].item() == 330.46
    assert out.loc[out["date"] == pd.Timestamp("2018-01-03"), "cpi_pit"].item() == 331.10
    assert out.loc[out["date"] == pd.Timestamp("2018-02-02"), "cpi_pit"].item() == 331.10
    assert out.loc[out["date"] == pd.Timestamp("2018-02-05"), "cpi_pit"].item() == 334.44
    assert out.loc[out["date"] == pd.Timestamp("2018-03-05"), "cpi_pit"].item() == 336.88
    assert out["cpi_pit"].notna().all() and out["cpi"].notna().all()
    # a month without a published value: cpi NaN, pit keeps the last published value
    later = cpi_to_sessions(monthly, _sessions("2018-04-02", "2018-04-06"))
    # March's index is released on Tue 3 Apr: Monday 2 Apr still sees February's value
    assert later["cpi"].isna().all() and later["cpi_pit"].tolist() == [336.88] + [340.24] * 4
    empty = cpi_to_sessions(None, sessions)
    assert empty["cpi"].isna().all() and empty["cpi_available_from"].isna().all() and empty["cpi_pit"].isna().all()


# ----------------------------------------------------------------------------- FX
def test_fx_to_sessions_forward_fills_short_gaps_only() -> None:
    fx = parse_response(FX_PAYLOAD, USDTRY_BUY)[["date", "value"]]
    sessions = pd.DatetimeIndex(["2018-01-02", "2018-01-03", "2018-01-09", "2018-01-10", "2018-01-22"])
    got = fx_to_sessions(fx, sessions)
    assert got.index.name == "date" and got.name == "usdtry_evds"
    assert got.tolist()[:4] == pytest.approx([3.7581, 3.7534, 3.7409, 3.7409])  # shifted one business day back
    assert np.isnan(got.iloc[-1])  # 22 Jan is more than 5 days after the last observation
    assert fx_to_sessions(None, sessions).isna().all()


# ----------------------------------------------------------------------------- assembly
def test_build_macro_frame_without_any_source_has_all_columns_nan() -> None:
    sessions = _sessions("2023-02-06", "2023-02-17")
    out = build_macro_frame(sessions)
    assert list(out.columns) == MACRO_COLUMNS
    assert out["date"].tolist() == [
        pd.Timestamp(d) for d in ("2023-02-06", "2023-02-07", "2023-02-15", "2023-02-16", "2023-02-17")
    ]
    assert out.drop(columns=["date"]).isna().all().all()


def test_build_macro_frame_combines_sources() -> None:
    policy = pd.DataFrame([("2017-11-25", 8.0), ("2018-06-01", 16.5)], columns=POLICY_COLUMNS)
    out = build_macro_frame(
        _sessions("2018-01-02", "2018-01-05"),
        cpi_monthly=parse_response(CPI_PAYLOAD, CPI)[["date", "value"]],
        usdtry_daily=parse_response(FX_PAYLOAD, USDTRY_BUY)[["date", "value"]],
        policy=policy,
    )
    assert out["policy_rate"].tolist() == [8.0] * 4
    assert out["rf_daily"].tolist() == pytest.approx([0.08 / 365] * 4)
    assert out["usdtry_evds"].tolist() == pytest.approx([3.7581, 3.7534, 3.7393, 3.7409])
    assert (out["cpi"] == 334.44).all() and out["cpi_pit"].tolist() == [330.46, 331.10, 331.10, 331.10]


def test_build_macro_store_offline_end_to_end(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    session = FakeSession(
        {"series=TP.FG.J0": CPI_PAYLOAD, "series=TP.DK.USD.A.YTL": FX_PAYLOAD, POLICY_RATE_URL: TCMB_PAGE}
    )
    client = EvdsClient(session, settings.paths.raw, api_key="test-key")
    sessions = _sessions("2018-01-02", "2018-01-10")
    frame = build_macro_store(settings, sessions, session=session, evds_client=client)
    assert list(frame.columns) == MACRO_COLUMNS and len(frame) == 7
    assert (frame["policy_rate"] == 8.0).all()  # 25.11.2016 = 8.00 until 01.06.2018
    assert frame["cpi"].tolist() == [334.44] * 7 and frame["cpi_pit"].tolist() == [330.46] + [331.10] * 6
    assert frame["usdtry_evds"].iloc[0] == pytest.approx(3.7581)
    assert (settings.paths.store / MACRO_FILENAME).exists()
    meta = json.loads((settings.paths.store / MACRO_META_FILENAME).read_text(encoding="utf-8"))
    assert meta["evds_key_configured"] is True
    assert meta["sources"]["cpi"]["status"] == "ok" and meta["sources"]["policy_rate"]["rows"] == 57
    assert meta["coverage"]["policy_rate"] == 1.0
    # CPI is fetched from three months before the first session, year by year
    evds_files = sorted(p.name for p in (settings.paths.raw / "evds").glob("*.json.gz"))
    assert evds_files == [
        "TP_DK_USD_A_YTL_20171001_20171231.json.gz",
        "TP_DK_USD_A_YTL_20180101_20180110.json.gz",
        "TP_FG_J0_20171001_20171231.json.gz",
        "TP_FG_J0_20180101_20180110.json.gz",
    ]
    assert (settings.paths.raw / "tcmb" / "one_week_repo.html.gz").exists()
    loaded = load_macro(settings.paths.store)
    pd.testing.assert_frame_equal(loaded, frame)
    # second build is served from the raw layer: no new fetches
    n_fetches = len(session.fetches)
    build_macro_store(settings, sessions, session=session, evds_client=client)
    assert len(session.fetches) == n_fetches


def test_build_macro_store_without_key_leaves_evds_columns_nan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVDS_API_KEY", raising=False)
    settings = _settings(tmp_path)
    session = FakeSession({POLICY_RATE_URL: TCMB_PAGE})
    client = EvdsClient(session, settings.paths.raw, api_key=None, env_file=tmp_path / "absent.env")
    assert not client.has_key
    frame = build_macro_store(settings, _sessions("2024-03-18", "2024-03-29"), session=session, evds_client=client)
    assert frame["cpi"].isna().all() and frame["cpi_pit"].isna().all() and frame["usdtry_evds"].isna().all()
    assert frame.loc[frame["date"] == pd.Timestamp("2024-03-21"), "policy_rate"].item() == 45.0
    assert frame.loc[frame["date"] == pd.Timestamp("2024-03-22"), "policy_rate"].item() == 50.0
    meta = json.loads((settings.paths.store / MACRO_META_FILENAME).read_text(encoding="utf-8"))
    assert meta["evds_key_configured"] is False
    assert meta["sources"]["cpi"]["status"] == "missing_key" and meta["sources"]["usdtry"]["status"] == "missing_key"
    assert not (settings.paths.raw / "evds").exists()


def test_build_macro_store_rejects_empty_calendar(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_macro_store(_settings(tmp_path), [], session=FakeSession({}))


def test_calendar_dates_from_store(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        calendar_dates_from_store(tmp_path)
    write_parquet(
        pd.DataFrame({"date": pd.to_datetime(["2018-01-03", "2018-01-02", "2018-01-02"])}),
        tmp_path / macro.CALENDAR_FILENAME,
    )
    assert calendar_dates_from_store(tmp_path) == [date(2018, 1, 2), date(2018, 1, 3)]
    with pytest.raises(FileNotFoundError):
        load_macro(tmp_path)


def test_fx_evds_value_dated_monday_is_the_friday_fixing() -> None:
    # raw EVDS 2018: 13-08-2018 (Mon) = 5.9389 is the Friday 10-Aug fixing; the crash level is dated 14-08
    fx = pd.DataFrame(
        {"date": pd.to_datetime(["2018-08-10", "2018-08-13", "2018-08-14"]), "value": [5.9, 5.9389, 6.8798]}
    )
    sessions = pd.DatetimeIndex(["2018-08-10", "2018-08-13", "2018-08-14"])
    got = fx_to_sessions(fx, sessions)
    assert got.tolist() == pytest.approx(
        [5.9389, 6.8798, 6.8798]
    )  # the 14-Aug row is the 13-Aug fixing, 14-Aug carried


def test_cpi_pit_is_nan_once_the_latest_print_is_stale() -> None:
    monthly = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "value": [100.0]})
    out = cpi_to_sessions(monthly, pd.DatetimeIndex(["2026-02-04", "2026-03-30", "2026-04-15", "2026-09-01"]), 60)
    assert out["cpi_pit"].tolist()[:2] == [100.0, 100.0]
    assert out["cpi_pit"].isna().tolist() == [False, False, True, True]


def test_splice_cpi_uses_overlap_ratio_and_extends() -> None:
    old = pd.DataFrame(
        {"date": pd.to_datetime(["2025-11-01", "2025-12-01", "2026-01-01"]), "value": [3000.0, 3100.0, 3200.0]}
    )
    new = pd.DataFrame(
        {"date": pd.to_datetime(["2025-12-01", "2026-01-01", "2026-02-01"]), "value": [100.0, 3200.0 / 31.0, 105.0]}
    )
    out, ratio = splice_cpi(old, new)
    assert ratio == pytest.approx(31.0)
    assert out is not None and out["value"].iloc[-1] == pytest.approx(105.0 * 31.0) and len(out) == 4
    # no overlap: nothing is guessed
    far = pd.DataFrame({"date": pd.to_datetime(["2027-01-01"]), "value": [1.0]})
    assert splice_cpi(old, far) == (old, None)
