"""Offline tests for bistmas.data.prices_fetch (planning, alias resolution, checkpointed job) with a fake client."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

from bistmas.data.clients.http import RateLimitedError
from bistmas.data.clients.isyatirim import PARSED_COLUMNS, IsYatirimError, parse_hissetekil
from bistmas.data.prices_fetch import (
    DEFAULT_START_YEAR,
    EST_SECONDS_PER_REQUEST,
    MAX_CONSECUTIVE_FAILURES,
    estimate,
    fetch_prices,
    load_progress,
    pending_pairs,
    plan_tickers,
    progress_path,
    resolve_fetch_codes,
    run_fetch,
    year_span,
)
from bistmas.data.security_master import COLUMNS as SM_COLUMNS
from bistmas.data.security_master import write_security_master
from bistmas.data.universe.official import load_aliases, write_membership
from bistmas.shared.config import CONFIG_DIR, PathsConfig, Settings

FIXTURES = Path(__file__).parent / "fixtures" / "isyatirim"
TODAY = date(2026, 9, 23)
THIS_YEAR = date.today().year


@pytest.fixture(scope="module")
def thyao_frame() -> pd.DataFrame:
    return parse_hissetekil((FIXTURES / "THYAO_2018_head.json").read_bytes(), "THYAO")


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    return Settings(root=tmp_path, paths=PathsConfig().resolve(tmp_path))


class FakeClient:
    """Stands in for IsYatirimClient: records calls, serves frames or raises per (ticker, year)."""

    def __init__(self, raw_dir: Path, frame: pd.DataFrame, errors: dict[tuple[str, int], Any] | None = None) -> None:
        self.raw_dir = Path(raw_dir)
        self.frame = frame
        self.errors = dict(errors or {})
        self.calls: list[tuple[str, int]] = []
        self.empty: set[tuple[str, int]] = set()

    def raw_path(self, ticker: str, year: int) -> Path:
        return self.raw_dir / "isyatirim" / ticker / f"{year}.json.gz"

    def is_cached(self, ticker: str, year: int) -> bool:
        return self.raw_path(ticker, year).exists()

    def fetch_year(self, ticker: str, year: int, force: bool = False, today: date | None = None) -> pd.DataFrame:
        if self.is_cached(ticker, year) and not force:
            return self.frame
        self.calls.append((ticker, year))
        err = self.errors.get((ticker, year))
        if err is not None:
            if isinstance(err, list) and err:  # transient: raise once, then succeed
                exc = err.pop(0)
                if exc is not None:
                    raise exc
            elif not isinstance(err, list):
                raise err
        path = self.raw_path(ticker, year)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"cached")
        if (ticker, year) in self.empty:
            return self.frame.iloc[0:0]
        return self.frame


def _membership(tickers_by_index: dict[str, list[str]]) -> pd.DataFrame:
    rows = [
        {
            "index_code": idx,
            "ticker": t,
            "effective_start": pd.Timestamp("2018-01-02"),
            "effective_end": pd.NaT,
            "membership_known_since": pd.Timestamp("2017-12-29"),
            "source": "test",
        }
        for idx, tickers in tickers_by_index.items()
        for t in tickers
    ]
    return pd.DataFrame(rows)


def _security_master(current: list[str], no_sector: list[str]) -> pd.DataFrame:
    rows = []
    for t in current + no_sector:
        row = {c: "" for c in SM_COLUMNS}
        row.update(
            {
                "ticker": t,
                "is_current": True,
                "sector_group": "OTHER",
                "template": "industrial",
                "note": "no_kap_sector" if t in no_sector else "",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=SM_COLUMNS)


# ----------------------------------------------------------------------------- planning
def test_resolve_fetch_codes_applies_renames_only() -> None:
    aliases = load_aliases(CONFIG_DIR / "ticker_aliases.csv")
    codes, applied = resolve_fetch_codes(
        ["AVISA", "gusgr", "YAZIC.E", "TRKCM", "THYAO", "agesa", "TGB", "", "A1CAP"], aliases
    )
    assert codes == ["A1CAP", "AGESA", "AGHOL", "THYAO", "TRKCM", "TURSG"]
    assert applied == {"AVISA": "AGESA", "GUSGR": "TURSG", "YAZIC": "AGHOL"}  # mergers (TRKCM) keep their own code
    codes2, applied2 = resolve_fetch_codes(["THYAO"], aliases)
    assert codes2 == ["THYAO"] and applied2 == {}


def test_resolve_fetch_codes_uses_repo_aliases_by_default() -> None:
    codes, applied = resolve_fetch_codes(["AVISA"])
    assert codes == ["AGESA"] and applied == {"AVISA": "AGESA"}


def test_plan_tickers_union_and_all_equities(tmp_settings: Settings) -> None:
    store = tmp_settings.paths.store
    write_membership(_membership({"XU100": ["THYAO", "AVISA", "TRKCM"], "XU030": ["THYAO", "GARAN"]}), store)
    write_security_master(_security_master(["THYAO", "GARAN", "SISE", "ZZZZQ"], ["BONDX"]), store)
    assert plan_tickers(tmp_settings) == ["AGESA", "GARAN", "THYAO", "TRKCM"]
    assert plan_tickers(tmp_settings, include_all_equities=True) == [
        "AGESA",
        "GARAN",
        "SISE",
        "THYAO",
        "TRKCM",
        "ZZZZQ",
    ]


def test_plan_tickers_against_real_store(settings: Settings) -> None:
    """The real store (built in M1) must yield the membership union with renames resolved."""
    store = settings.paths.store / "universe_membership.parquet"
    if not store.exists():
        pytest.skip("universe_membership.parquet not built")
    codes = plan_tickers(settings)
    assert len(codes) >= 150
    assert "TURSG" in codes and "AGHOL" in codes  # GUSGR / YAZIC spells resolved to the served codes
    assert not ({"AVISA", "GUSGR", "YAZIC"} & set(codes))
    assert "TRKCM" in codes  # merged-away codes keep their own history


def test_year_span_defaults_and_validation() -> None:
    assert year_span(2017, 2019, TODAY) == [2017, 2018, 2019]
    assert year_span(2025, None, TODAY) == [2025, 2026]
    assert year_span()[0] == DEFAULT_START_YEAR and year_span()[-1] == THIS_YEAR
    with pytest.raises(ValueError):
        year_span(2020, 2019, TODAY)
    with pytest.raises(ValueError):
        year_span(2017, 2027, TODAY)


def test_estimate_counts_only_uncached_years(tmp_settings: Settings, thyao_frame: pd.DataFrame) -> None:
    client = FakeClient(tmp_settings.paths.raw, thyao_frame)
    n, seconds = estimate(tmp_settings, ["THYAO", "ASELS"], 2018, 2020, client=client)
    assert n == 6 and seconds == pytest.approx(6 * EST_SECONDS_PER_REQUEST)
    client.fetch_year("THYAO", 2018)
    client.fetch_year("THYAO", 2019)
    n, seconds = estimate(tmp_settings, ["THYAO", "ASELS"], 2018, 2020, client=client)
    assert n == 4
    assert pending_pairs(client, ["THYAO", "ASELS"], [2018, 2019, 2020]) == [
        ("THYAO", 2020),
        ("ASELS", 2018),
        ("ASELS", 2019),
        ("ASELS", 2020),
    ]


# ----------------------------------------------------------------------------- the job
def test_run_fetch_dry_run_makes_no_calls(tmp_settings: Settings, thyao_frame: pd.DataFrame) -> None:
    client = FakeClient(tmp_settings.paths.raw, thyao_frame)
    summary = run_fetch(tmp_settings, ["THYAO", "AVISA"], 2018, 2019, dry_run=True, client=client)
    assert client.calls == []
    assert summary["tickers"] == 2 and summary["aliases"] == {"AVISA": "AGESA"}
    assert summary["planned"] == 4 and summary["pending"] == 4 and summary["cached"] == 0
    assert summary["stopped_reason"] == "dry_run" and summary["requested"] == 0
    assert summary["estimate_seconds"] == pytest.approx(4 * EST_SECONDS_PER_REQUEST)
    assert not progress_path(tmp_settings).exists()


def test_run_fetch_full_run_writes_progress_and_resumes(tmp_settings: Settings, thyao_frame: pd.DataFrame) -> None:
    client = FakeClient(tmp_settings.paths.raw, thyao_frame)
    client.empty.add(("AKBNK", 2019))
    summary = run_fetch(tmp_settings, ["THYAO", "AKBNK"], 2018, 2019, client=client)
    assert client.calls == [("AKBNK", 2018), ("AKBNK", 2019), ("THYAO", 2018), ("THYAO", 2019)]  # ticker-major
    assert summary["requested"] == 4 and summary["fetched"] == 4 and summary["empty"] == 1 and summary["failed"] == 0
    assert summary["stopped_reason"] == "complete" and summary["remaining"] == 0
    progress = load_progress(progress_path(tmp_settings))
    assert progress["version"] == 1 and progress["requests_made"] == 4
    assert set(progress["done"]) == {"AKBNK/2018", "AKBNK/2019", "THYAO/2018", "THYAO/2019"}
    assert progress["done"]["AKBNK/2019"] == {**progress["done"]["AKBNK/2019"], "rows": 0, "empty": True}
    assert progress["done"]["THYAO/2018"]["rows"] == 20
    assert progress["runs"][-1]["stopped_reason"] == "complete"
    # second run: everything cached -> no calls, complete
    again = run_fetch(tmp_settings, ["THYAO", "AKBNK"], 2018, 2019, client=client)
    assert len(client.calls) == 4 and again["stopped_reason"] == "complete" and again["cached"] == 4
    # widening the span only fetches the new years and appends a run entry
    wider = run_fetch(tmp_settings, ["THYAO", "AKBNK"], 2018, 2020, client=client)
    assert client.calls[4:] == [("AKBNK", 2020), ("THYAO", 2020)] and wider["cached"] == 4 and wider["fetched"] == 2
    assert len(load_progress(progress_path(tmp_settings))["runs"]) == 2  # the all-cached run made no requests


def test_run_fetch_max_requests_slices(tmp_settings: Settings, thyao_frame: pd.DataFrame) -> None:
    client = FakeClient(tmp_settings.paths.raw, thyao_frame)
    first = run_fetch(tmp_settings, ["THYAO", "ASELS"], 2018, 2020, max_requests=2, client=client)
    assert first["requested"] == 2 and first["stopped_reason"] == "max_requests" and first["remaining"] == 4
    assert client.calls == [("ASELS", 2018), ("ASELS", 2019)]
    second = run_fetch(tmp_settings, ["THYAO", "ASELS"], 2018, 2020, max_requests=2, client=client)
    assert client.calls[2:] == [("ASELS", 2020), ("THYAO", 2018)] and second["cached"] == 2
    third = run_fetch(tmp_settings, ["THYAO", "ASELS"], 2018, 2020, client=client)
    assert third["requested"] == 2 and third["stopped_reason"] == "complete" and third["remaining"] == 0
    assert run_fetch(tmp_settings, ["THYAO", "ASELS"], 2018, 2020, max_requests=0, client=client)["requested"] == 0


def test_run_fetch_records_failures_and_continues(tmp_settings: Settings, thyao_frame: pd.DataFrame) -> None:
    errors = {
        ("ASELS", 2019): requests.ConnectionError("reset"),
        ("THYAO", 2018): IsYatirimError("HTML body"),
    }
    client = FakeClient(tmp_settings.paths.raw, thyao_frame, errors)
    summary = run_fetch(tmp_settings, ["THYAO", "ASELS"], 2018, 2019, client=client)
    assert summary["requested"] == 4 and summary["fetched"] == 2 and summary["failed"] == 2
    assert summary["stopped_reason"] == "complete_with_failures" and summary["remaining"] == 2
    progress = load_progress(progress_path(tmp_settings))
    assert set(progress["failed"]) == {"ASELS/2019", "THYAO/2018"}
    assert progress["failed"]["THYAO/2018"]["attempts"] == 1
    assert progress["failed"]["THYAO/2018"]["error"].startswith("IsYatirimError")
    # retry: the failures are re-asked (they are not cached), succeed and leave the failed map
    client.errors.clear()
    retry = run_fetch(tmp_settings, ["THYAO", "ASELS"], 2018, 2019, client=client)
    assert retry["fetched"] == 2 and retry["cached"] == 2 and retry["stopped_reason"] == "complete"
    assert load_progress(progress_path(tmp_settings))["failed"] == {}


def test_run_fetch_stops_on_rate_limit_and_on_forbidden(tmp_settings: Settings, thyao_frame: pd.DataFrame) -> None:
    client = FakeClient(tmp_settings.paths.raw, thyao_frame, {("ASELS", 2019): RateLimitedError("429")})
    summary = run_fetch(tmp_settings, ["THYAO", "ASELS"], 2018, 2020, client=client)
    assert summary["stopped_reason"] == "blocked" and summary["requested"] == 2 and summary["failed"] == 1
    assert client.calls == [("ASELS", 2018), ("ASELS", 2019)]

    resp = requests.Response()
    resp.status_code = 403
    forbidden = requests.HTTPError("403", response=resp)
    client2 = FakeClient(tmp_settings.paths.raw, thyao_frame, {("THYAO", 2018): forbidden})
    summary2 = run_fetch(tmp_settings, ["THYAO"], 2018, 2020, client=client2)
    assert summary2["stopped_reason"] == "blocked" and summary2["requested"] == 1

    resp5 = requests.Response()
    resp5.status_code = 500
    server_err = requests.HTTPError("500", response=resp5)
    client3 = FakeClient(tmp_settings.paths.raw, thyao_frame, {("GARAN", 2018): server_err})
    summary3 = run_fetch(tmp_settings, ["GARAN"], 2018, 2019, client=client3)
    assert summary3["stopped_reason"] == "complete_with_failures" and summary3["fetched"] == 1


def test_run_fetch_aborts_after_consecutive_failures(tmp_settings: Settings, thyao_frame: pd.DataFrame) -> None:
    years = list(range(2010, 2020))
    errors = {("THYAO", y): requests.Timeout("slow") for y in years}
    client = FakeClient(tmp_settings.paths.raw, thyao_frame, errors)
    summary = run_fetch(tmp_settings, ["THYAO"], 2010, 2019, client=client)
    assert summary["stopped_reason"] == "failures"
    assert summary["requested"] == MAX_CONSECUTIVE_FAILURES and summary["failed"] == MAX_CONSECUTIVE_FAILURES


def test_run_fetch_keyboard_interrupt_saves_progress(tmp_settings: Settings, thyao_frame: pd.DataFrame) -> None:
    client = FakeClient(tmp_settings.paths.raw, thyao_frame, {("ASELS", 2019): KeyboardInterrupt()})
    summary = run_fetch(tmp_settings, ["THYAO", "ASELS"], 2018, 2020, client=client)
    assert summary["stopped_reason"] == "interrupted"
    assert summary["fetched"] == 1 and summary["remaining"] == 5
    progress = load_progress(progress_path(tmp_settings))
    assert list(progress["done"]) == ["ASELS/2018"]
    assert progress["runs"][-1]["stopped_reason"] == "interrupted"
    # resume picks up exactly where it stopped
    client.errors.clear()
    resumed = run_fetch(tmp_settings, ["THYAO", "ASELS"], 2018, 2020, client=client)
    assert resumed["cached"] == 1 and resumed["fetched"] == 5 and resumed["stopped_reason"] == "complete"


def test_run_fetch_force_refetches_cached_years(tmp_settings: Settings, thyao_frame: pd.DataFrame) -> None:
    client = FakeClient(tmp_settings.paths.raw, thyao_frame)
    run_fetch(tmp_settings, ["THYAO"], 2018, 2019, client=client)
    assert len(client.calls) == 2
    forced = run_fetch(tmp_settings, ["THYAO"], 2018, 2019, client=client, force=True)
    assert len(client.calls) == 4 and forced["pending"] == 2 and forced["fetched"] == 2


def test_estimate_uses_measured_rate_from_progress(tmp_settings: Settings, thyao_frame: pd.DataFrame) -> None:
    client = FakeClient(tmp_settings.paths.raw, thyao_frame)
    run_fetch(tmp_settings, ["THYAO"], 2015, 2019, client=client)  # 5 requests -> average available
    path = progress_path(tmp_settings)
    progress = json.loads(path.read_text())
    progress["seconds_spent"] = 100.0  # 20 s/request
    path.write_text(json.dumps(progress))
    n, seconds = estimate(tmp_settings, ["THYAO", "ASELS"], 2015, 2019, client=client)
    assert n == 5 and seconds == pytest.approx(100.0)


def test_load_progress_tolerates_garbage(tmp_settings: Settings) -> None:
    path = progress_path(tmp_settings)
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    assert load_progress(path) == {}
    path.write_text(json.dumps({"version": 99}))
    assert load_progress(path) == {}
    assert load_progress(path.with_name("missing.json")) == {}


def test_fetch_prices_entry_point(
    tmp_settings: Settings, thyao_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_settings.paths.store
    write_membership(_membership({"XU100": ["THYAO", "AVISA"]}), store)
    write_security_master(_security_master(["THYAO"], []), store)
    captured: dict[str, Any] = {}

    def fake_run(settings: Settings, tickers: list[str], **kwargs: Any) -> dict[str, Any]:
        captured["tickers"] = tickers
        captured.update(kwargs)
        return {"stopped_reason": "dry_run"}

    monkeypatch.setattr("bistmas.data.prices_fetch.run_fetch", fake_run)
    out = fetch_prices(tmp_settings, start_year=2018, end_year=2019, max_requests=3, dry_run=True)
    assert out == {"stopped_reason": "dry_run"}
    assert captured["tickers"] == ["AGESA", "THYAO"]
    assert captured["start_year"] == 2018 and captured["end_year"] == 2019 and captured["max_requests"] == 3
    assert captured["dry_run"] is True and captured["force"] is False


def test_parsed_columns_are_the_contract_prefix(thyao_frame: pd.DataFrame) -> None:
    assert list(thyao_frame.columns)[:20] == [
        "ticker",
        "date",
        "close_raw",
        "vwap_raw",
        "low_raw",
        "high_raw",
        "volume_raw",
        "close_adj",
        "vwap_adj",
        "low_adj",
        "high_adj",
        "volume_adj",
        "shares",
        "mcap_tl",
        "ffloat_mcap_tl",
        "mcap_usd",
        "ffloat_mcap_usd",
        "usdtry",
        "xu100",
        "price_usd",
    ]
    assert len(PARSED_COLUMNS) == 26
