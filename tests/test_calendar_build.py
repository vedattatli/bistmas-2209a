"""Offline tests for bistmas.data.calendar_build."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from bistmas.data import calendar_build as cb
from bistmas.data.prices import build_prices_frame, parse_hissetekil
from bistmas.shared.calendar import CLOSURES, TradingCalendar

FIXTURES = Path(__file__).parent / "fixtures" / "prices"


def _frame(*names: str) -> pd.DataFrame:
    vendor = pd.concat(
        [parse_hissetekil((FIXTURES / f"hissetekil_{n}.json").read_bytes(), n.split("_")[0]) for n in names],
        ignore_index=True,
    )
    return build_prices_frame(vendor)


def _panel(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ticker", "date", "traded_value_tl"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_calendar_excludes_cancelled_session_and_closures():
    cal = cb.build_trading_calendar(_frame("SYNC_2023"))
    assert list(cal.columns) == ["date"] and str(cal["date"].dtype) == "datetime64[ns]"
    dates = set(cal["date"])
    assert pd.Timestamp("2023-02-08") not in dates
    assert all(pd.Timestamp(c) not in dates for c in CLOSURES)
    assert len(cal) == len(pd.bdate_range("2023-01-02", "2023-03-31")) - 5
    assert cal["date"].is_monotonic_increasing and cal["date"].is_unique


def test_top_n_by_traded_value_controls_the_union():
    panel = _panel(
        [
            ("BIG", "2018-01-02", 100.0),
            ("BIG", "2018-01-03", 100.0),
            ("MID", "2018-01-02", 10.0),
            ("MID", "2018-01-04", 10.0),
            ("TINY", "2018-01-01", 1.0),  # a holiday only the illiquid name "trades" on
            ("TINY", "2018-01-02", 1.0),
        ]
    )
    assert cb.top_liquid_tickers(panel, 2) == ["BIG", "MID"]
    two = cb.build_trading_calendar(panel, top_n=2)
    assert [d.date() for d in two["date"]] == [date(2018, 1, 2), date(2018, 1, 3), date(2018, 1, 4)]
    three = cb.build_trading_calendar(panel, top_n=3)
    assert three["date"].iloc[0] == pd.Timestamp("2018-01-01")
    with pytest.raises(ValueError):
        cb.build_trading_calendar(panel.iloc[0:0])
    with pytest.raises(ValueError):
        cb.build_trading_calendar(panel.drop(columns=["traded_value_tl"]))


def test_liquidity_falls_back_to_volume_times_vwap():
    panel = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "date": pd.to_datetime(["2018-01-02", "2018-01-02"]),
            "volume_raw": [10.0, 1.0],
            "vwap_raw": [1.0, 100.0],
        }
    )
    assert cb.top_liquid_tickers(panel, 1) == ["B"]


def test_anomaly_report_flags_only_the_feb_2023_closure():
    prices = _frame("SYNC_2023")
    cal = cb.build_trading_calendar(prices)
    report = cb.calendar_anomalies(cal, prices)
    assert report["n_sessions"] == len(cal)
    assert report["gaps"] == [{"from": "2023-02-07", "to": "2023-02-15", "missing_weekdays": 5, "expected": True}]
    assert report["weekend_sessions"] == []
    assert report["month_end_anomalies"] == []
    assert report["half_day_candidates"] == []


def test_half_day_candidates_from_low_traded_value():
    prices = _frame("SYNC_2023")
    prices.loc[prices["date"] == pd.Timestamp("2023-03-15"), "traded_value_tl"] *= 0.1
    cal = cb.build_trading_calendar(prices)
    report = cb.calendar_anomalies(cal, prices)
    assert [c["date"] for c in report["half_day_candidates"]] == ["2023-03-15"]
    assert report["half_day_candidates"][0]["ratio"] < cb.HALF_DAY_RATIO


def test_month_end_and_weekend_anomalies_and_unexpected_gaps():
    days = [d for d in pd.bdate_range("2018-01-02", "2018-02-28") if d < pd.Timestamp("2018-01-29")]
    cal = pd.DataFrame({"date": [*days, pd.Timestamp("2018-02-03")]})  # Saturday row, Jan ends on the 26th
    report = cb.calendar_anomalies(cal)
    assert report["weekend_sessions"] == ["2018-02-03"]
    assert report["month_end_anomalies"] == [
        {"month": "2018-01", "last_session": "2018-01-26", "month_end": "2018-01-31"}
    ]
    assert report["gaps"] == [{"from": "2018-01-26", "to": "2018-02-03", "missing_weekdays": 5, "expected": False}]

    multi = cb.build_trading_calendar(_frame("SYNB_2018", "SYNC_2023"), top_n=30)
    gaps = cb.calendar_anomalies(multi)["gaps"]
    assert [g["expected"] for g in gaps] == [False, True]
    assert gaps[0]["from"] == "2018-03-30" and gaps[0]["to"] == "2023-01-02"


def test_write_load_roundtrip_and_trading_calendar(tmp_path):
    prices = _frame("SYNC_2023")
    cal = cb.build_and_write_calendar(prices, tmp_path)
    assert "report" in cal.attrs and cal.attrs["report"]["gaps"][0]["expected"] is True
    frame = cb.load_calendar_frame(tmp_path)
    pd.testing.assert_frame_equal(frame, cal[["date"]])
    tc = cb.load_calendar(tmp_path)
    assert isinstance(tc, TradingCalendar)
    assert tc.is_session(date(2023, 2, 7)) and not tc.is_session(date(2023, 2, 8))
    assert tc.next_session(date(2023, 2, 7)) == date(2023, 2, 15)
    assert tc.month_ends(date(2023, 1, 1), date(2023, 3, 31)) == [
        date(2023, 1, 31),
        date(2023, 2, 28),
        date(2023, 3, 31),
    ]
    with pytest.raises(FileNotFoundError):
        cb.load_calendar(tmp_path / "missing")
    with pytest.raises(ValueError):
        cb.write_calendar(pd.DataFrame({"day": []}), tmp_path)


def test_write_calendar_normalises_duplicates_and_times(tmp_path):
    cb.write_calendar(pd.DataFrame({"date": ["2018-01-03 10:00", "2018-01-02", "2018-01-02"]}), tmp_path)
    out = cb.load_calendar_frame(tmp_path)
    assert [d.date() for d in out["date"]] == [date(2018, 1, 2), date(2018, 1, 3)]
