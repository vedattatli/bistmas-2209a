"""Backtest engine on a hand-built price world where every number can be checked by hand."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from bistmas.backtest.engine import BacktestConfig, Backtester
from bistmas.backtest.metrics import active_stats, summarise, to_numeraire
from bistmas.backtest.stats import deflated_sharpe, newey_west_ttest, stationary_bootstrap_ci
from bistmas.shared.calendar import TradingCalendar


def _world():
    sessions = pd.bdate_range("2018-01-01", "2018-06-29")
    cal = TradingCalendar([d.date() for d in sessions])
    # AAA doubles linearly, BBB flat, CCC delists after 2018-03-15
    close = pd.DataFrame(index=sessions)
    close["AAA"] = np.linspace(10, 20, len(sessions))
    close["BBB"] = 10.0
    close["CCC"] = 5.0
    close.loc[close.index > pd.Timestamp("2018-03-15"), "CCC"] = np.nan
    vwap = close * 1.0  # fills equal closes to keep arithmetic simple
    return cal, vwap, close


def test_fills_costs_and_delisting_by_hand():
    cal, vwap, close = _world()
    bt = Backtester(cal, vwap, close)
    cfg = BacktestConfig(date(2018, 1, 31), date(2018, 3, 30), cost_bps_one_way=100.0, delisting_haircut=0.5, label="t")
    res = bt.run(lambda t: ["AAA", "BBB", "CCC"], cfg)
    p = res.periods
    assert list(p["decision_date"].dt.date) == [date(2018, 1, 31), date(2018, 2, 28), date(2018, 3, 30)]
    assert p["exec_date"].iloc[0] == pd.Timestamp("2018-02-01") and p["exit_date"].iloc[0] == pd.Timestamp("2018-03-01")
    # first period: buy 3 names at 1/3 each from nothing -> turnover 0.5, cost = 2 * 0.5 * 1% = 1%
    assert p["turnover"].iloc[0] == pytest.approx(0.5) and p["cost"].iloc[0] == pytest.approx(0.01)
    a0 = vwap.at[pd.Timestamp("2018-03-01"), "AAA"] / vwap.at[pd.Timestamp("2018-02-01"), "AAA"] - 1
    assert p["gross_ret"].iloc[0] == pytest.approx(a0 / 3)  # BBB and CCC flat
    assert p["net_ret"].iloc[0] == pytest.approx((1 + a0 / 3) * 0.99 - 1)
    # second period (Mar): CCC has no VWAP at the April exit -> liquidated at last close 5 * (1 - 0.5)
    h = res.holdings[
        (res.holdings["decision_date"] == pd.Timestamp("2018-02-28")) & (res.holdings["ticker"] == "CCC")
    ].iloc[0]
    assert h["liquidated"] and h["exit_price"] == pytest.approx(2.5) and h["period_ret"] == pytest.approx(-0.5)
    # third period: CCC cannot be filled (no VWAP on 2018-04-02) -> its 1/3 slice stays in cash
    assert p["n_holdings"].iloc[2] == 3 and p["holdings"].iloc[2] == "AAA,BBB,CCC"
    h3 = res.holdings[(res.holdings["decision_date"] == p["decision_date"].iloc[2]) & (res.holdings["ticker"] == "CCC")]
    assert h3["weight"].iloc[0] == pytest.approx(1 / 3) and h3["period_ret"].iloc[0] == 0.0
    # daily NAV reconciles with the period NAV path
    assert res.daily["nav"].iloc[-1] == pytest.approx(p["nav_end"].iloc[-1], rel=1e-9)
    assert (res.daily["date"].diff().dropna() > pd.Timedelta(0)).all()


def test_empty_selection_holds_cash_and_metrics_are_finite():
    cal, vwap, close = _world()
    bt = Backtester(cal, vwap, close)
    cfg = BacktestConfig(date(2018, 1, 31), date(2018, 4, 30), cost_bps_one_way=0.0, label="cash")
    res = bt.run(lambda t: [], cfg)
    assert (res.periods["net_ret"] == 0).all() and (res.daily["ret"] == 0).all()
    m = summarise(res.daily, res.periods)
    assert m["total_return"] == 0 and m["max_drawdown"] == 0 and np.isnan(m["sharpe"])
    res2 = bt.run(lambda t: ["AAA"], cfg)
    m2 = summarise(res2.daily, res2.periods, rf_daily=0.0)
    assert m2["cagr"] > 0 and m2["sharpe"] > 0 and m2["max_drawdown"] <= 0 and m2["hit_rate_monthly"] == 1.0
    usd = to_numeraire(res2.daily, pd.Series(2.0, index=pd.to_datetime(res2.daily["date"])))
    assert usd["nav"].iloc[-1] == pytest.approx(res2.daily["nav"].iloc[-1])  # constant FX: no change


def test_active_stats_and_inference():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2018-01-31", periods=84, freq="ME")
    strat = pd.DataFrame({"decision_date": dates, "net_ret": rng.normal(0.02, 0.05, 84)})
    bench = pd.DataFrame({"decision_date": dates, "net_ret": rng.normal(0.01, 0.05, 84)})
    act = active_stats(strat, bench)
    assert list(act.columns) == ["strategy", "benchmark", "active"] and len(act) == 84
    nw = newey_west_ttest(act["active"], lags=3)
    assert nw["n"] == 84 and 0 <= nw["p"] <= 1 and np.isfinite(nw["t"])
    ci = stationary_bootstrap_ci(act["active"], n_boot=200)
    assert ci["lo"] < nw["mean"] < ci["hi"]
    d = deflated_sharpe(sharpe_hat=0.3, n_obs=84, n_trials=20)
    assert 0 <= d["psr"] <= 1 and d["sr_benchmark"] > 0
    assert deflated_sharpe(0.3, 84, 1)["sr_benchmark"] == 0.0


def test_halted_name_is_filled_when_it_trades_again_and_zero_vwap_is_not_a_price():
    cal, vwap, close = _world()
    # DDD is suspended on the first two sessions of February (vendor writes VWAP 0, close = last close)
    close["DDD"] = 8.0
    vwap["DDD"] = 8.0
    vwap.loc[pd.Timestamp("2018-02-01") : pd.Timestamp("2018-02-02"), "DDD"] = 0.0
    vwap.loc[pd.Timestamp("2018-02-05") :, "DDD"] = 9.0
    close.loc[pd.Timestamp("2018-02-05") :, "DDD"] = 9.0
    bt = Backtester(cal, vwap, close)
    cfg = BacktestConfig(date(2018, 1, 31), date(2018, 2, 28), cost_bps_one_way=0.0, label="t")
    res = bt.run(lambda t: ["DDD", "BBB"], cfg)
    h = res.holdings[res.holdings["ticker"] == "DDD"].iloc[0]
    assert h["fill_date"] == pd.Timestamp("2018-02-05") and h["fill_price"] == pytest.approx(9.0)
    assert h["period_ret"] == pytest.approx(0.0)  # bought at 9 on the first tradable session, still 9 at the exit
    assert res.periods["n_holdings"].iloc[0] == 2
    # cash until the delayed fill: the daily NAV never moves through the zero-VWAP sessions
    nav = res.daily.set_index("date")["nav"]
    assert nav.loc[pd.Timestamp("2018-02-02")] == pytest.approx(1.0)
    # a name that never trades before the exit keeps its slice as cash (no re-normalisation, no division by zero)
    vwap.loc[pd.Timestamp("2018-02-01") : pd.Timestamp("2018-03-01"), "DDD"] = 0.0
    res2 = Backtester(cal, vwap, close).run(lambda t: ["DDD", "BBB"], cfg)
    assert res2.periods["n_holdings"].iloc[0] == 2
    h2 = res2.holdings[
        (res2.holdings["ticker"] == "DDD") & (res2.holdings["decision_date"] == res2.periods["decision_date"].iloc[0])
    ]
    assert h2["weight"].iloc[0] == pytest.approx(0.5) and h2["period_ret"].iloc[0] == 0.0
    bbb = res2.holdings[(res2.holdings["ticker"] == "BBB")].iloc[0]
    assert res2.periods["gross_ret"].iloc[0] == pytest.approx(0.5 * bbb["period_ret"])
