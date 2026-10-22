"""Regression tests for the prereg-v2 backtest/reporting corrections (wp5)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from bistmas.backtest.benchmark import eligible_selector
from bistmas.backtest.engine import BacktestConfig, Backtester
from bistmas.backtest.metrics import convert_periods
from bistmas.backtest.run import build_entries, load_membership
from bistmas.backtest.tables import cpi_series, secondary_tables, usdtry_series
from bistmas.reporting.results import collect_runs, to_markdown
from bistmas.shared.calendar import TradingCalendar
from bistmas.shared.config import PathsConfig, Settings


def _results():
    sessions = pd.bdate_range("2018-01-01", "2018-06-29")
    cal = TradingCalendar([d.date() for d in sessions])
    close = pd.DataFrame(index=sessions)
    close["AAA"] = np.linspace(10, 20, len(sessions))
    close["BBB"] = 10.0
    bt = Backtester(cal, close * 1.0, close)
    first, last = date(2017, 12, 29), date(2018, 4, 30)
    res = {
        "det_top1": bt.run(lambda t: ["AAA"], BacktestConfig(first, last, 0.0, top_n=1, label="det_top1")),
        "bench_ew_pit": bt.run(lambda t: ["AAA", "BBB"], BacktestConfig(first, last, 0.0, label="bench_ew_pit")),
    }
    return res


def test_leave_one_year_out_groups_by_holding_year():
    res = _results()
    # the December-2017 decision is held in January 2018: no phantom 2017 row
    loyo = secondary_tables(res)["leave_one_year_out"]["det_top1"]
    assert list(loyo["year_excluded"]) == [2018]


def test_cpi_series_interpolates_geometrically_without_jumps(tmp_path):
    idx = pd.bdate_range("2018-01-01", "2018-04-30")
    pit = np.where(idx < "2018-02-05", 100.0, np.where(idx < "2018-03-05", 110.0, 121.0))
    pd.DataFrame({"date": idx, "cpi_pit": pit}).to_parquet(tmp_path / "macro_daily.parquet")
    s = cpi_series(tmp_path)
    assert s.pct_change().dropna().abs().max() < 0.02  # no 10 % step on a release day
    assert s.loc["2018-02-05"] == pytest.approx(110.0) and s.loc["2018-03-05"] == pytest.approx(121.0)
    assert s.loc["2018-01-01"] == pytest.approx(100.0) and s.iloc[-1] == pytest.approx(121.0)


def test_usdtry_series_prefers_evds_and_repairs_stale_vendor(tmp_path):
    idx = pd.bdate_range("2018-08-06", "2018-08-17")
    true = pd.Series(np.linspace(5.0, 5.5, len(idx)), index=idx)
    vendor = true * 1.0018
    stale = idx[4]
    vendor.loc[stale] = vendor.iloc[3]
    fixed = usdtry_series(tmp_path, vendor)  # no EVDS: neighbours interpolate the stale session
    assert abs(fixed.loc[stale] / (true.loc[stale] * 1.0018) - 1) < 1e-3
    pd.DataFrame({"date": idx, "usdtry_evds": true.to_numpy()}).to_parquet(tmp_path / "macro_daily.parquet")
    out = usdtry_series(tmp_path, vendor)
    assert out.loc[stale] / out.iloc[0] == pytest.approx(true.loc[stale] / true.iloc[0], rel=1e-6)


def test_convert_periods_and_per_numeraire_active_statistics():
    res = _results()
    d_idx = pd.to_datetime(res["det_top1"].daily["date"])
    fx = pd.Series(np.linspace(1.0, 2.0, len(d_idx)), index=d_idx)
    p = res["det_top1"].periods
    conv = convert_periods(p, fx)
    nav_tl = float((1 + p["net_ret"]).prod())
    x0, x1 = fx.loc[p["exec_date"].iloc[0]], fx.loc[p["exit_date"].iloc[-1]]
    assert float((1 + conv["net_ret"]).prod()) == pytest.approx(nav_tl * x0 / x1)
    e = build_entries(res, 0.0, fx, None, n_trials=3)["det_top1"]
    assert e["usd"]["active"]["mean"] != pytest.approx(e["tl"]["active"]["mean"])
    assert "real" not in e and e["newey_west"] == e["tl"]["active"]["newey_west"]
    assert "active" not in build_entries(res, 0.0, fx, None, 3)["bench_ew_pit"]["usd"]


def test_eligible_selector_intersects_members_and_screen():
    m = pd.DataFrame(
        {
            "index_code": ["XU100"] * 3,
            "ticker": ["AAA", "BBB", "CCC"],
            "effective_start": pd.to_datetime(["2017-01-01"] * 3),
            "effective_end": pd.NaT,
            "membership_known_since": pd.to_datetime(["2017-01-01"] * 3),
            "source": "t",
        }
    )
    scr = pd.DataFrame(
        {"as_of": pd.to_datetime(["2018-01-31"] * 3), "ticker": ["AAA", "BBB", "ZZZ"], "eligible": [True, False, True]}
    )
    assert eligible_selector(m, scr)(pd.Timestamp("2018-01-31")) == ["AAA"]
    assert eligible_selector(m, scr)(pd.Timestamp("2018-02-28")) == []


def test_collect_runs_reads_per_numeraire_statistics_and_h1(tmp_path):
    import json

    def act(m):
        return {"mean": m, "newey_west": {"t": m * 10, "p": 0.5}, "deflated_sharpe": {"psr": 0.3, "n_trials": 5}}

    rep = {
        "run": "r",
        "strategies": {
            "hybrid_top3": {
                "tl": {"cagr": 0.1, "active": act(0.01)},
                "usd": {"cagr": 0.05, "active": act(0.02)},
                "vs_deterministic": {"mean_monthly_diff": 0.003, "newey_west": {"t": 1.5, "p": 0.14}},
            }
        },
        "random_jury_null": {"n": 10, "cagr_mean": 0.07, "cagr_p05": 0.0, "cagr_p95": 0.14, "hybrid_percentile": 0.6},
    }
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "report.json").write_text(json.dumps(rep))
    df = collect_runs(tmp_path)
    h = df[df["strategy"] == "hybrid_top3"].set_index("numeraire")
    assert h.loc["tl", "active_mean_monthly"] == 0.01 and h.loc["usd", "active_mean_monthly"] == 0.02
    assert h.loc["tl", "vs_det_nw_p"] == 0.14 and h.loc["tl", "n_trials"] == 5
    md = to_markdown(df)
    assert "vs_det_nw_p" in md and "Random-jury null" in md and "0.600" in md
    assert df[df["strategy"] == "random_jury_null"]["active_mean_monthly"].isna().all()


def test_universe_provider_switch(tmp_path):
    s = Settings(root=tmp_path, paths=PathsConfig().resolve(tmp_path))
    s.paths.store.mkdir(parents=True)
    s.universe.provider = "rule_based"
    with pytest.raises(FileNotFoundError, match="rule_based"):
        load_membership(s)
    s.universe.provider = "bogus"
    with pytest.raises(ValueError):
        load_membership(s)


def test_build_entries_tl_sharpe_uses_risk_free():
    res = _results()
    d = pd.to_datetime(res["det_top1"].daily["date"])
    rf = pd.Series(0.0002, index=d)
    e0 = build_entries(res, 0.0, None, None, 3)["det_top1"]["tl"]
    e1 = build_entries(res, rf, None, None, 3)["det_top1"]["tl"]
    assert e1["sharpe"] != pytest.approx(e0["sharpe"])
    assert e1["sharpe"] < e0["sharpe"]
