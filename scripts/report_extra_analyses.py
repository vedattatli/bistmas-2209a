"""Exploratory analyses requested during the final-report review (not pre-registered; reported as such).

    BISTMAS_CONFIG=configs/prereg_v2.toml uv run python scripts/report_extra_analyses.py [jury_run] [hybrid_run]

1. Sharpe difference hybrid Top-3 − rule-based Top-3 (TL, rf = TCMB policy rate) with a paired stationary
   block bootstrap on daily excess returns (Politis-Romano, mean block 21 sessions, 2,000 draws).
2. Sensitivity of the hybrid Top-3 to the way the three agents' scores are combined: the pre-registered raw
   mean J, the mean of per-agent z-scores within each month, and the mean of per-agent ranks within each month
   (ties broken by the composite rank as in the pre-registered rule). Each variant runs through the same
   backtest engine, costs and fills as the headline result.

Writes data/runs/<hybrid_run>/extra_analyses.json and prints a summary.
"""

from __future__ import annotations

import json
import sys
from datetime import date

import numpy as np
import pandas as pd

from bistmas.backtest.engine import BacktestConfig, Backtester
from bistmas.backtest.metrics import TRADING_DAYS, active_stats, summarise
from bistmas.backtest.run import _vwap_panel, load_inputs
from bistmas.backtest.stats import newey_west_ttest, stationary_bootstrap_ci
from bistmas.backtest.tables import align_rf, rf_daily_series
from bistmas.data.calendar_build import load_calendar
from bistmas.shared.config import load_settings

AGENTS = ["quality_growth", "value", "risk_timing"]


def sharpe_of(ex: np.ndarray) -> float:
    sd = ex.std(ddof=1)
    return float(np.sqrt(TRADING_DAYS) * ex.mean() / sd) if sd > 0 else np.nan


def sharpe_diff_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, mean_block: float = 21.0, seed: int = 7):
    """Paired stationary bootstrap of Sharpe(a) − Sharpe(b); a and b are aligned daily excess returns."""
    n = len(a)
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    out = np.empty(n_boot)
    for k in range(n_boot):
        starts = rng.random(n) < p
        starts[0] = True
        idx = np.empty(n, dtype=int)
        i = 0
        for j in range(n):
            i = rng.integers(n) if starts[j] else (i + 1) % n
            idx[j] = i
        out[k] = sharpe_of(a[idx]) - sharpe_of(b[idx])
    lo, hi = np.quantile(out, [0.025, 0.975])
    return {
        "diff": sharpe_of(a) - sharpe_of(b),
        "lo": float(lo),
        "hi": float(hi),
        "n_boot": n_boot,
        "mean_block_sessions": mean_block,
        "share_draws_above_zero": float((out > 0).mean()),
    }


def ranking(cons: pd.DataFrame, scores: pd.DataFrame, method: str) -> dict[pd.Timestamp, list[str]]:
    wide = cons[["as_of", "ticker", "composite_rank", "J"]].copy()
    if method == "raw_mean":
        wide["agg"] = wide["J"]
    else:
        s = scores[~scores["abstain"]].copy()
        if method == "z_mean":
            g = s.groupby(["as_of", "agent"])["score"]
            sd = g.transform("std").replace(0, np.nan)
            s["v"] = ((s["score"] - g.transform("mean")) / sd).fillna(0.0)
        else:  # rank_mean: higher score → higher rank value (average ranks for ties)
            s["v"] = s.groupby(["as_of", "agent"])["score"].rank(method="average", pct=True)
        agg = s.groupby(["as_of", "ticker"])["v"].mean().rename("agg").reset_index()
        wide = wide.merge(agg, on=["as_of", "ticker"], how="left")
    wide = wide.dropna(subset=["agg"])
    return {
        m: g.sort_values(["agg", "composite_rank"], ascending=[False, True])["ticker"].tolist()
        for m, g in wide.groupby("as_of")
    }


def main() -> None:
    jury_run = sys.argv[1] if len(sys.argv) > 1 else "prereg-v2"
    hybrid_run = sys.argv[2] if len(sys.argv) > 2 else "hybrid-v2"
    settings = load_settings()
    runs = settings.paths.runs
    out: dict = {"jury_run": jury_run, "hybrid_run": hybrid_run, "pre_registered": False}

    # 1. Sharpe difference from the stored daily series of the headline run
    rf = rf_daily_series(settings.paths.store)
    daily = {}
    for label in ("hybrid_top3", "det_top3"):
        d = pd.read_csv(runs / hybrid_run / f"{label}_daily.csv", parse_dates=["date"])
        r = pd.Series(d["ret"].to_numpy(), index=pd.DatetimeIndex(d["date"]))
        daily[label] = r - align_rf(rf, d).reindex(r.index).fillna(0.0)
    both = pd.concat(daily, axis=1, join="inner").dropna()
    out["sharpe_difference_top3"] = sharpe_diff_bootstrap(both["hybrid_top3"].to_numpy(), both["det_top3"].to_numpy())
    out["sharpe_difference_top3"]["sharpe_hybrid"] = sharpe_of(both["hybrid_top3"].to_numpy())
    out["sharpe_difference_top3"]["sharpe_det"] = sharpe_of(both["det_top3"].to_numpy())

    # 2. aggregation variants through the same engine
    inputs, prices, _ = load_inputs(settings)
    bt = Backtester(load_calendar(settings.paths.store), _vwap_panel(prices), inputs.panel.close_adj)
    first = date.fromisoformat(settings.rebalance.first_decision)
    last = date.fromisoformat(settings.rebalance.last_decision)
    cost = settings.backtest.cost_bps_commission + settings.backtest.cost_bps_slippage
    cons = pd.read_parquet(runs / jury_run / "jury_consensus.parquet")
    scores = pd.read_parquet(runs / jury_run / "jury_scores.parquet")
    for df in (cons, scores):
        df["as_of"] = pd.to_datetime(df["as_of"])
    cons["composite_rank"] = pd.to_numeric(cons["composite_rank"], errors="coerce")
    det_by_month = {
        m: g.dropna(subset=["composite_rank"]).sort_values("composite_rank")["ticker"].tolist()
        for m, g in cons.groupby("as_of")
    }

    def run(selmap, label):
        return bt.run(
            lambda t: selmap.get(pd.Timestamp(t), [])[:3],
            BacktestConfig(first, last, cost, settings.backtest.delisting_haircut, 3, label),
        )

    det = run(det_by_month, "det_top3")
    out["aggregation_variants_top3"] = {}
    for method in ("raw_mean", "z_mean", "rank_mean"):
        res = run(ranking(cons, scores, method), f"hybrid_top3_{method}")
        diff = active_stats(res.periods, det.periods)["active"]
        nw = newey_west_ttest(diff, lags=settings.backtest.newey_west_lags)
        s = summarise(res.daily, res.periods, rf_daily=align_rf(rf, res.daily))
        out["aggregation_variants_top3"][method] = {
            "cagr_tl": s["cagr"],
            "sharpe_tl": s["sharpe"],
            "mean_monthly_diff_vs_det": float(diff.mean()),
            "nw_t": nw["t"],
            "nw_p": nw["p"],
            "bootstrap_ci_95": stationary_bootstrap_ci(diff, n_boot=1000),
            "mean_turnover": float(res.periods["turnover"].mean()),
        }
    out["det_top3_cagr_tl"] = summarise(det.daily, det.periods, rf_daily=align_rf(rf, det.daily))["cagr"]
    path = runs / hybrid_run / "extra_analyses.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
