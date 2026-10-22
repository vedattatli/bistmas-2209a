"""Pre-registered secondary analyses on synthetic data."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from bistmas.backtest.engine import BacktestConfig, Backtester
from bistmas.experiments.secondary import (
    agreement,
    bonferroni,
    buy_only,
    incremental_ic,
    kish_effective,
    krippendorff_alpha_interval,
    next_month_returns,
    spearman_j_composite,
)
from bistmas.shared.calendar import TradingCalendar


def _cons(n_months=8, n=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m in pd.date_range("2018-01-31", periods=n_months, freq="BME"):
        for i in range(n):
            a = rng.uniform(0, 1, 3)
            rows.append(
                {
                    "as_of": m,
                    "ticker": f"T{i}",
                    "composite_rank": i + 1,
                    "quality_growth": a[0],
                    "value": a[1],
                    "risk_timing": a[2],
                    "J": a.mean(),
                }
            )
    return pd.DataFrame(rows)


def test_bonferroni():
    r = bonferroni({"a": {"newey_west": {"p": 0.01}}, "b": {"newey_west": {"p": 0.2}}, "c": {}}, 10)
    assert r["alpha_adjusted"] == pytest.approx(0.005)
    assert r["strategies"]["a"]["p_bonferroni"] == pytest.approx(0.1)
    assert not r["strategies"]["a"]["significant_bonferroni"] and r["strategies"]["a"]["significant_raw"]
    assert "c" not in r["strategies"]


def test_echo_spearman_perfect_and_inverse():
    c = _cons()
    c["J"] = -c["composite_rank"].astype(float)  # J agrees with composite quality
    assert spearman_j_composite(c)["mean"] == pytest.approx(1.0)
    c["J"] = c["composite_rank"].astype(float)
    assert spearman_j_composite(c)["mean"] == pytest.approx(-1.0)


def test_krippendorff_known_values():
    assert krippendorff_alpha_interval(np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])) == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    assert abs(krippendorff_alpha_interval(rng.normal(size=(2000, 3)))) < 0.05
    v = np.array([[1.0, 1.0, np.nan], [2.0, np.nan, np.nan], [3.0, 3.0, 3.0]])
    assert krippendorff_alpha_interval(v) == pytest.approx(1.0)


def test_kish_and_agreement():
    assert kish_effective(np.array([1, 1, 1])) == pytest.approx(3.0)
    assert kish_effective(np.array([1, 0, 0])) == pytest.approx(1.0)
    assert kish_effective(np.array([3.0, 1.0])) == pytest.approx(16 / 10)
    c = _cons()
    c["value"] = c["quality_growth"]
    c["risk_timing"] = c["quality_growth"]
    a = agreement(c)
    assert a["mean_pairwise_spearman"] == pytest.approx(1.0)
    assert a["effective_votes_corr_adjusted"] == pytest.approx(1.0)
    assert a["krippendorff_alpha_interval"] == pytest.approx(1.0)
    assert a["kish_votes_mean"] == pytest.approx(3.0)
    c.loc[c.index[:5], "value"] = np.nan
    assert agreement(c)["kish_votes_mean"] < 3.0


def test_incremental_ic_detects_j_signal():
    c = _cons(n_months=14, n=20)
    decisions = sorted(c["as_of"].unique())
    rng = np.random.default_rng(3)
    fwd = {}
    for m in decisions[:-1]:
        g = c[c["as_of"] == m].set_index("ticker")
        fwd[pd.Timestamp(m)] = g["J"] + rng.normal(0, 0.05, len(g))  # returns follow J, not composite
    r = incremental_ic(c, fwd)
    assert r["n_months"] == 13
    assert r["coef_J"]["mean"] > 0.5 and r["coef_J"]["p"] < 0.01
    assert abs(r["coef_composite"]["mean"]) < 0.2


def test_next_month_returns():
    idx = pd.bdate_range("2018-01-01", "2018-03-30")
    close = pd.DataFrame({"A": np.linspace(10, 20, len(idx))}, index=idx)
    d = [pd.Timestamp("2018-01-31"), pd.Timestamp("2018-02-28")]
    r = next_month_returns(close, d)
    assert r[d[0]]["A"] == pytest.approx(close.loc[d[1], "A"] / close.loc[d[0], "A"] - 1)


def test_buy_only_counts_and_cash():
    sessions = pd.bdate_range("2018-01-01", "2018-08-31")
    cal = TradingCalendar([d.date() for d in sessions])
    close = pd.DataFrame({"AAA": np.linspace(10, 20, len(sessions)), "BBB": 10.0}, index=sessions)
    bt = Backtester(cal, close.copy(), close)
    months = [pd.Timestamp(d) for d in cal.month_ends(date(2018, 1, 31), date(2018, 6, 29))]
    rows = []
    for k, m in enumerate(months):
        rows += [
            {"as_of": m, "ticker": "AAA", "J": 0.80 if k else 0.5},
            {"as_of": m, "ticker": "BBB", "J": 0.7499999 if k else 0.4},  # rounds to 0.75
        ]
    cons = pd.DataFrame(rows)
    bench = bt.run(lambda t: ["AAA", "BBB"], BacktestConfig(date(2018, 1, 31), date(2018, 6, 29), 0.0))
    out = buy_only(bt, cons, bench.periods, [0.70, 0.75, 0.80], date(2018, 1, 31), date(2018, 6, 29), 0.0, 0.0)
    assert out["0.75"]["buy_count_max"] == 2 and out["0.75"]["buy_count_min"] == 0
    assert out["0.75"]["months_in_cash"] == 1
    assert out["0.80"]["buy_count_max"] == 1 and out["0.80"]["months_in_cash"] == 1
    assert out["0.70"]["buy_count_mean"] == pytest.approx(5 / 6 * 2)
