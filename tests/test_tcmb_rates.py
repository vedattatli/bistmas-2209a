"""Tests for the TCMB 1-week repo table client (fixture trimmed from the live page of 2026-09-23) + rf conventions."""

from __future__ import annotations

import gzip
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from bistmas.data.clients import tcmb_rates as tr
from bistmas.data.clients.tcmb_rates import (
    POLICY_COLUMNS,
    POLICY_RATE_URL,
    PolicyRateParseError,
    accrual_between_sessions,
    coerce_policy_frame,
    fetch_policy_rate_table,
    overnight_rate_daily,
    parse_policy_rate_html,
    policy_rate_from_evds,
    policy_rate_on,
)
from bistmas.shared.store import append_manifest, write_raw_gzip

FIXTURES = Path(__file__).parent / "fixtures" / "macro"
PAGE = (FIXTURES / "tcmb_one_week_repo_snippet.html").read_bytes()


class FakeSession:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.fetches: list[str] = []

    def get_cached(self, url: str, raw_path: Path, manifest_path=None, source="", force=False, **kw: Any) -> bytes:
        if raw_path.exists() and not force:
            with gzip.open(raw_path, "rb") as fh:
                return fh.read()
        self.fetches.append(url)
        payload = self.payloads[url]
        digest = write_raw_gzip(payload, raw_path)
        append_manifest(manifest_path or raw_path.parent / "manifest.jsonl", {"source": source, "sha256": digest})
        return payload


def _policy(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=POLICY_COLUMNS)


# ----------------------------------------------------------------------------- parsing
def test_parse_fixture_table() -> None:
    df = parse_policy_rate_html(PAGE)
    assert list(df.columns) == POLICY_COLUMNS
    assert df["effective_date"].dtype == "datetime64[ns]"
    assert len(df) == 57
    assert (df["effective_date"].iloc[0], df["rate"].iloc[0]) == (pd.Timestamp("2010-05-20"), 7.0)
    assert (df["effective_date"].iloc[-1], df["rate"].iloc[-1]) == (pd.Timestamp("2026-01-23"), 37.0)
    assert df["effective_date"].is_monotonic_increasing and df["effective_date"].is_unique
    by_date = df.set_index("effective_date")["rate"]
    assert by_date[pd.Timestamp("2018-09-14")] == 24.0
    assert by_date[pd.Timestamp("2024-03-22")] == 50.0
    assert by_date[pd.Timestamp("2021-12-17")] == 14.0  # decision 16 Dec 2021, effective the next day


def test_parse_tolerates_th_headers_comma_decimals_and_bad_rows() -> None:
    html = """
    <html><body><table>
      <tr><th>Tarih</th><th>Borç Alma</th><th>Borç Verme</th></tr>
      <tr><td>20/05/2010</td><td>-</td><td>7,00</td></tr>
      <tr><td>17.12.2010</td><td>-</td><td>6.50</td></tr>
      <tr><td>not a date</td><td>-</td><td>1.00</td></tr>
      <tr><td>21.01.2011</td><td>-</td><td>-</td></tr>
      <tr><td>05.08.2011</td><td>-</td><td>5.75</td></tr>
      <tr><td>19.12.2012</td><td>-</td><td>5.50</td></tr>
      <tr><td>17.04.2013</td><td>-</td><td>5.00</td></tr>
      <tr><td>17.04.2013</td><td>-</td><td>5.25</td></tr>
    </table></body></html>"""
    df = parse_policy_rate_html(html)
    assert df["rate"].tolist() == [7.0, 6.5, 5.75, 5.5, 5.25]  # duplicate date keeps the last row
    assert df["effective_date"].iloc[0] == pd.Timestamp("2010-05-20")


def test_parse_rejects_pages_without_the_table() -> None:
    with pytest.raises(PolicyRateParseError):
        parse_policy_rate_html("<html><body><p>Bakım çalışması</p></body></html>")
    with pytest.raises(PolicyRateParseError):
        parse_policy_rate_html("")
    short = "<table><tr><td>Tarih</td><td>Borç Verme</td></tr><tr><td>20.05.2010</td><td>7.00</td></tr></table>"
    with pytest.raises(PolicyRateParseError):
        parse_policy_rate_html(short)
    huge = "<table><tr><td>Tarih</td><td>Borç Verme</td></tr>" + "".join(
        f"<tr><td>0{i}.01.2020</td><td>{150 + i}</td></tr>" for i in range(1, 8)
    )
    with pytest.raises(PolicyRateParseError):
        parse_policy_rate_html(huge + "</table>")


# ----------------------------------------------------------------------------- step series and rf
def test_policy_rate_on_uses_effective_dated_steps() -> None:
    policy = _policy([("2018-06-01", 16.5), ("2018-06-08", 17.75), ("2018-09-14", 24.0)])
    dates = [date(2018, 5, 31), date(2018, 6, 1), date(2018, 6, 7), date(2018, 6, 8), date(2019, 1, 2)]
    got = policy_rate_on(policy, dates)
    assert got.index.name == "date" and got.name == "policy_rate"
    assert pd.isna(got.iloc[0])
    assert got.iloc[1:].tolist() == [16.5, 16.5, 17.75, 24.0]
    # unsorted probe dates keep their order
    assert policy_rate_on(policy, [date(2019, 1, 2), date(2018, 6, 1)]).tolist() == [24.0, 16.5]
    assert policy_rate_on(None, dates).isna().all()


def test_overnight_rate_and_session_accrual_convention() -> None:
    policy = _policy([("2018-01-01", 36.5)])
    sessions = pd.bdate_range("2018-01-04", "2018-01-09")  # Thu, Fri, Mon, Tue
    daily = overnight_rate_daily(policy, sessions)
    assert daily.name == "rf_daily"
    assert daily.tolist() == pytest.approx([0.365 / 365] * 4)
    accrual = accrual_between_sessions(policy, sessions)
    assert accrual.name == "rf_session"
    assert accrual.tolist() == pytest.approx([0.001, 0.001, 0.003, 0.001])  # weekend earns three days
    assert overnight_rate_daily(policy, sessions, day_count=360).iloc[0] == pytest.approx(0.365 / 360)


def test_coerce_policy_frame_validates_and_normalises() -> None:
    raw = pd.DataFrame({"effective_date": ["2020-01-17", "2020-01-17", "2019-12-13"], "rate": ["11.25", "11.25", 12.0]})
    out = coerce_policy_frame(raw)
    assert out["effective_date"].tolist() == [pd.Timestamp("2019-12-13"), pd.Timestamp("2020-01-17")]
    assert out["rate"].tolist() == [12.0, 11.25]
    with pytest.raises(ValueError):
        coerce_policy_frame(pd.DataFrame({"date": ["2020-01-01"], "rate": [1.0]}))
    assert coerce_policy_frame(pd.DataFrame(columns=POLICY_COLUMNS)).empty


# ----------------------------------------------------------------------------- fetching
def test_fetch_policy_rate_table_caches_in_raw_layer(tmp_path: Path) -> None:
    session = FakeSession({POLICY_RATE_URL: PAGE})
    df = fetch_policy_rate_table(session, tmp_path)
    assert len(df) == 57
    raw = tmp_path / "tcmb" / "one_week_repo.html.gz"
    assert raw.exists() and (tmp_path / "tcmb" / "manifest.jsonl").exists()
    again = fetch_policy_rate_table(FakeSession({}), tmp_path)  # cache hit: no fetch
    pd.testing.assert_frame_equal(again, df)
    assert session.fetches == [POLICY_RATE_URL]


def test_fetch_policy_rate_table_discards_unparseable_page(tmp_path: Path) -> None:
    session = FakeSession({POLICY_RATE_URL: b"<html><body>Sayfa bulunamadi</body></html>"})
    with pytest.raises(PolicyRateParseError):
        fetch_policy_rate_table(session, tmp_path)
    assert not (tmp_path / "tcmb" / "one_week_repo.html.gz").exists()


def test_policy_rate_from_evds_keeps_change_points() -> None:
    class FakeClient:
        def fetch_series_by_year(self, series, start, end, force=False):
            dates = pd.bdate_range("2025-03-03", "2025-03-14")
            values = [42.5] * 5 + [46.0] * 5
            return pd.DataFrame({"series": series, "date": dates, "value": values})

    df = policy_rate_from_evds(FakeClient(), date(2025, 3, 1), date(2025, 3, 31))
    assert df["effective_date"].tolist() == [pd.Timestamp("2025-03-03"), pd.Timestamp("2025-03-10")]
    assert df["rate"].tolist() == [42.5, 46.0]
    assert tr.EVDS_FUNDING_COST_SERIES == "TP.APIFON4"

    class EmptyClient:
        def fetch_series_by_year(self, *a, **k):
            return pd.DataFrame(columns=["series", "date", "value"])

    assert policy_rate_from_evds(EmptyClient(), date(2025, 3, 1), date(2025, 3, 31)).empty


# ----------------------------------------------------------------------------- network
@pytest.mark.network
def test_network_live_table(settings: Any) -> None:
    from bistmas.data.clients.http import PoliteSession

    df = fetch_policy_rate_table(PoliteSession(min_interval_s=1.5, timeout_s=60.0), settings.paths.raw)
    assert len(df) >= 57
    by_date = df.set_index("effective_date")["rate"]
    assert by_date[pd.Timestamp("2010-05-20")] == 7.0
    assert by_date[pd.Timestamp("2024-03-22")] == 50.0
