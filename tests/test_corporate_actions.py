"""Offline tests for bistmas.data.corporate_actions."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bistmas.data import corporate_actions as ca
from bistmas.data.prices import build_prices_frame, parse_hissetekil
from bistmas.shared.calendar import TradingCalendar

FIXTURES = Path(__file__).parent / "fixtures" / "prices"


def _frame(*names: str) -> pd.DataFrame:
    vendor = pd.concat(
        [parse_hissetekil((FIXTURES / f"hissetekil_{n}.json").read_bytes(), n.split("_")[0]) for n in names],
        ignore_index=True,
    )
    return build_prices_frame(vendor)


def _only(df: pd.DataFrame, ticker: str, kind: str) -> pd.DataFrame:
    return df[(df["ticker"] == ticker) & (df["kind"] == kind)].reset_index(drop=True)


def test_contract_columns_and_sorting():
    events = ca.infer_corporate_actions(_frame("SYNB_2018", "SYNR_2018", "SYND_2020", "SYNC_2023"))
    assert list(events.columns) == ca.CA_COLUMNS
    assert str(events["date"].dtype) == "datetime64[ns]"
    assert set(events["kind"]) <= set(ca.KINDS)
    key = events[["ticker", "date", "kind"]]
    assert key.equals(key.sort_values(["ticker", "date", "kind"]).reset_index(drop=True))
    assert not events[["ticker", "date", "kind"]].duplicated().any()


def test_bonus_issue_is_dated_on_the_ex_date_and_marked_early():
    events = ca.infer_corporate_actions(_frame("SYNB_2018"))
    cap = _only(events, "SYNB", "capital_change")
    assert len(cap) == 1
    row = cap.iloc[0]
    assert row["date"] == pd.Timestamp("2018-02-15")
    assert (row["shares_before"], row["shares_after"]) == (100e6, 200e6)
    assert row["ret_raw"] == pytest.approx(-0.5, abs=0.03)
    assert abs(row["ret_adj"]) < 0.05
    assert "ratio=2" in row["note"] and "class=bonus" in row["note"] and "vendor_step_one_day_early" in row["note"]
    assert _only(events, "SYNB", "gap").empty


def test_dividend_is_an_adj_jump():
    events = ca.infer_corporate_actions(_frame("SYNB_2018"))
    jumps = _only(events, "SYNB", "adj_jump")
    assert list(jumps["date"]) == [pd.Timestamp("2018-03-12")]
    assert "dividend?" in jumps.iloc[0]["note"]
    assert np.isnan(jumps.iloc[0]["shares_before"])
    assert ca.infer_corporate_actions(_frame("SYNB_2018"), adj_tol=0.2).pipe(_only, "SYNB", "adj_jump").empty


def test_rights_issue_classification():
    events = ca.infer_corporate_actions(_frame("SYNR_2018"))
    cap = _only(events, "SYNR", "capital_change")
    assert len(cap) == 1
    note = cap.iloc[0]["note"]
    assert "class=rights_or_mixed" in note and "vendor_step_one_day_early" not in note
    assert cap.iloc[0]["ret_raw"] == pytest.approx(-0.375, abs=0.03)
    assert _only(events, "SYNR", "adj_jump").empty


def test_gap_inside_a_delisted_tickers_life():
    prices = _frame("SYND_2020")
    events = ca.infer_corporate_actions(prices)
    gaps = _only(events, "SYND", "gap")
    assert len(gaps) == 1
    assert gaps.iloc[0]["date"] == pd.Timestamp("2020-03-16")
    assert gaps.iloc[0]["note"] == "10 sessions missing after 2020-02-28"
    assert prices["date"].max() == pd.Timestamp("2020-05-18")  # delisting: the series simply ends, no event
    # with an explicit calendar the count is the same; a looser threshold hides the gap
    cal = TradingCalendar([d.date() for d in pd.bdate_range("2020-01-01", "2020-12-31")])
    same = ca.infer_corporate_actions(prices, calendar=cal)
    assert _only(same, "SYND", "gap").iloc[0]["note"] == gaps.iloc[0]["note"]
    assert _only(ca.infer_corporate_actions(prices, max_gap_sessions=15), "SYND", "gap").empty


def test_feb_2023_closure_is_not_a_gap():
    events = ca.infer_corporate_actions(_frame("SYNC_2023"))
    assert events.empty


def test_gap_uses_calendar_sessions_not_panel_dates():
    prices = _frame("SYND_2020")
    # a calendar in which the hole is not a hole (those days are not sessions) -> no gap
    hole = set(pd.bdate_range("2020-03-02", "2020-03-13").date)
    cal = TradingCalendar([d.date() for d in pd.bdate_range("2020-01-01", "2020-12-31") if d.date() not in hole])
    assert _only(ca.infer_corporate_actions(prices, calendar=cal), "SYND", "gap").empty


def test_falls_back_to_vendor_shares_when_aligned_missing():
    prices = _frame("SYNB_2018").drop(columns=["shares_aligned"])
    cap = _only(ca.infer_corporate_actions(prices), "SYNB", "capital_change")
    assert list(cap["date"]) == [pd.Timestamp("2018-02-14")]  # vendor date, one day early
    assert "class=no_price_effect" in cap.iloc[0]["note"]
    with pytest.raises(ValueError):
        ca.infer_corporate_actions(prices.drop(columns=["close_adj"]))


def test_classify_capital_change():
    assert ca.classify_capital_change(2.0, 2.0) == "bonus"
    assert ca.classify_capital_change(2.0, 1.6) == "rights_or_mixed"
    assert ca.classify_capital_change(0.5, 0.5) == "reduction"
    assert ca.classify_capital_change(1.05, 1.0) == "no_price_effect"
    assert ca.classify_capital_change(2.0, 3.0) == "other"
    assert ca.classify_capital_change(float("nan"), 1.0) == "unknown"


def test_summary_events_for_and_store_roundtrip(tmp_path):
    events = ca.infer_corporate_actions(_frame("SYNB_2018", "SYNR_2018", "SYND_2020"))
    summary = ca.summarise_corporate_actions(events)
    assert summary["by_kind"] == {"capital_change": 2, "adj_jump": 1, "gap": 1}
    assert summary["capital_change_classes"] == {"bonus": 1, "rights_or_mixed": 1}
    assert summary["vendor_step_one_day_early"] == 1
    assert summary["tickers_with_events"] == 3
    assert summary["largest_adj_jumps"][0]["ticker"] == "SYNB"
    assert ca.events_for(events, "synb") == [date(2018, 2, 15), date(2018, 3, 12)]
    assert ca.events_for(events, "SYND", kinds=("gap",)) == [date(2020, 3, 16)]
    ca.write_corporate_actions(events, tmp_path)
    pd.testing.assert_frame_equal(ca.load_corporate_actions(tmp_path), events)
    with pytest.raises(FileNotFoundError):
        ca.load_corporate_actions(tmp_path / "missing")
    with pytest.raises(ValueError):
        ca.write_corporate_actions(events.drop(columns=["note"]), tmp_path)
    empty = ca.summarise_corporate_actions(events.iloc[0:0])
    assert empty["n_events"] == 0 and empty["largest_adj_jumps"] == []
