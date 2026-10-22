"""Secondary tables on the hand-built price world of the engine tests."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from bistmas.backtest.engine import BacktestConfig, Backtester
from bistmas.backtest.tables import (
    align_rf,
    calendar_year_table,
    cost_sensitivity,
    leave_one_year_out,
    minimum_detectable_effect,
    secondary_tables,
    tables_to_json,
)
from bistmas.shared.calendar import TradingCalendar


def _world():
    sessions = pd.bdate_range("2018-01-01", "2018-06-29")
    cal = TradingCalendar([d.date() for d in sessions])
    close = pd.DataFrame(index=sessions)
    close["AAA"] = np.linspace(10, 20, len(sessions))
    close["BBB"] = 10.0
    return cal, close * 1.0, close


def _results():
    cal, vwap, close = _world()
    bt = Backtester(cal, vwap, close)
    first, last = date(2018, 1, 31), date(2018, 5, 31)
    res = {
        "det_top1": bt.run(
            lambda t: ["AAA"], BacktestConfig(first, last, cost_bps_one_way=0.0, top_n=1, label="det_top1")
        ),
        "bench_ew_pit": bt.run(
            lambda t: ["AAA", "BBB"], BacktestConfig(first, last, cost_bps_one_way=0.0, label="bench_ew_pit")
        ),
    }
    return bt, first, last, res


def test_calendar_year_table_compounds_periods_and_reports_active():
    _, _, _, res = _results()
    t = calendar_year_table(res)
    assert list(t["year"]) == [2018]
    expected = float((1 + res["det_top1"].periods["net_ret"]).prod() - 1)
    assert t["det_top1"].iloc[0] == pytest.approx(expected)
    assert t["active_det_top1"].iloc[0] == pytest.approx(expected - t["bench_ew_pit"].iloc[0])


def test_leave_one_year_out_and_mde_shapes():
    rng = np.random.default_rng(0)
    dates = pd.Series(pd.date_range("2018-01-31", periods=36, freq="ME"))
    active = pd.Series(rng.normal(0.002, 0.03, 36))
    loyo = leave_one_year_out(active, dates)
    assert list(loyo["year_excluded"]) == [2018, 2019, 2020] and (loyo["n"] == 24).all()
    mde = minimum_detectable_effect(active)
    assert mde["n"] == 36 and mde["mde_monthly"] > 0 and mde["mde_annualised"] == pytest.approx(12 * mde["mde_monthly"])
    assert mde["mde_monthly"] == pytest.approx(mde["se"] * (1.959964 + 0.841621), rel=1e-4)


def test_cost_sensitivity_monotone_in_cost_and_json_safe():
    bt, first, last, res = _results()
    sel = {"det_top1": lambda t: ["AAA"], "bench_ew_pit": lambda t: ["AAA", "BBB"]}
    tbl = cost_sensitivity(bt, sel, {"first": first, "last": last, "top_n": {"det_top1": 1}}, [0.0, 50.0, 100.0])
    d = tbl[tbl["strategy"] == "det_top1"].sort_values("cost_bps_one_way")
    assert len(d) == 3 and d["cagr_tl"].is_monotonic_decreasing and "nw_t" in d.columns
    sec = secondary_tables(res)
    js = tables_to_json({**sec, "cost_sensitivity": tbl})
    assert isinstance(js["calendar_year"], list) and isinstance(js["leave_one_year_out"]["det_top1"], list)
    assert isinstance(js["mde"]["det_top1"]["mde_monthly"], float)


def test_align_rf_forward_fills_on_strategy_dates():
    rf = pd.Series([0.001, 0.002], index=pd.to_datetime(["2018-01-01", "2018-03-01"]))
    daily = pd.DataFrame({"date": pd.to_datetime(["2018-01-15", "2018-03-15"])})
    out = align_rf(rf, daily)
    # date-indexed (sharpe() reindexes by date) and accrued over the 59 calendar days between the sessions
    assert list(out.index) == list(daily["date"]) and align_rf(0.0, daily) == 0.0
    assert out.iloc[0] == 0.001 and abs(out.iloc[1] - 0.002 * 59) < 1e-12


def test_prepare_master_keeps_alias_rows_and_warns_on_missing(caplog):
    import logging

    import pandas as pd

    from bistmas.backtest.run import prepare_master

    m = pd.DataFrame(
        {
            "ticker": ["SISE", "SODA", "SODA"],
            "sector_group": ["HOLDINGS", "MANUFACTURING", "OTHER"],
            "template": ["other_financial", "industrial", "industrial"],
            "is_current": [True, False, True],
        }
    )
    with caplog.at_level(logging.WARNING):
        out = prepare_master(m, {"SISE", "SODA", "GHOST"})
    assert list(out.index) == ["SODA", "SISE"] or set(out.index) == {"SISE", "SODA"}
    assert out.loc["SODA", "is_current"]  # current row preferred when a ticker appears twice
    assert "GHOST" in caplog.text
    alias_only = prepare_master(m.iloc[[1]], {"SODA"})
    assert alias_only.loc["SODA", "sector_group"] == "MANUFACTURING"


def test_sortino_downside_deviation_over_all_observations():
    from bistmas.backtest.metrics import TRADING_DAYS, sortino

    r = pd.Series([0.01, -0.01, 0.01, 0.01])
    assert abs(sortino(r) - TRADING_DAYS**0.5 * 0.005 / (0.0001 / 4) ** 0.5) < 1e-9
