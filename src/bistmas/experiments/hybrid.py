"""Experiments that combine the jury with the backtest (the only package allowed to import both).

``run_hybrid``: Top-N by jury score J among each month's screened candidates, the deterministic Top-N with
the same N, the point-in-time equal-weight benchmark, H1 paired tests and the random-jury null.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from bistmas.backtest.benchmark import eligible_selector, universe_selector
from bistmas.backtest.engine import BacktestConfig, Backtester, BacktestResult
from bistmas.backtest.metrics import active_stats, summarise
from bistmas.backtest.run import (
    _vendor_usdtry,
    _vwap_panel,
    build_entries,
    ledger_summary,
    load_inputs,
    run_screens,
)
from bistmas.backtest.stats import newey_west_ttest, stationary_bootstrap_ci
from bistmas.backtest.tables import (
    cost_sensitivity,
    cpi_series,
    rf_daily_series,
    secondary_tables,
    tables_to_json,
    usdtry_series,
)
from bistmas.data.calendar_build import load_calendar
from bistmas.experiments.secondary import run_secondary
from bistmas.shared.config import Settings
from bistmas.shared.ledger import record_run, trial_count
from bistmas.shared.log import get_logger
from bistmas.shared.store import utc_now_iso, write_json

log = get_logger("experiments.hybrid")


def run_hybrid(
    settings: Settings,
    jury_run: str,
    run_name: str | None = None,
    top_ns: list[int] | None = None,
    n_random: int | None = None,
) -> dict[str, Any]:
    """Hybrid backtest: Top-N by jury score J (rank_J) among each month's screened candidates, plus the
    deterministic Top-N with the same N, the equal-weight benchmark and the random-jury null distribution."""
    from bistmas.jury.consensus import consensus
    from bistmas.jury.random_jury import random_scores
    from bistmas.jury.run import load_consensus

    run_name = run_name or f"hybrid_{jury_run}"
    inputs, prices, provenance = load_inputs(settings)
    calendar = load_calendar(settings.paths.store)
    first, last = (
        date.fromisoformat(settings.rebalance.first_decision),
        date.fromisoformat(settings.rebalance.last_decision),
    )
    bt = Backtester(calendar, _vwap_panel(prices), inputs.panel.close_adj)
    cons = load_consensus(settings.paths.runs, jury_run)
    cons["composite_rank"] = pd.to_numeric(cons["composite_rank"], errors="coerce")
    cons["rank_J"] = pd.to_numeric(cons["rank_J"], errors="coerce")
    cons["as_of"] = pd.to_datetime(cons["as_of"])
    by_month = {
        m: g.dropna(subset=["rank_J"]).sort_values("rank_J")["ticker"].tolist() for m, g in cons.groupby("as_of")
    }
    det_by_month = {
        m: g.dropna(subset=["composite_rank"]).sort_values("composite_rank")["ticker"].tolist()
        for m, g in cons.groupby("as_of")
    }
    cost = settings.backtest.cost_bps_commission + settings.backtest.cost_bps_slippage
    out_dir = settings.paths.runs / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, BacktestResult] = {}
    for n in top_ns or settings.backtest.top_n:
        results[f"hybrid_top{n}"] = bt.run(
            lambda t, n=n: by_month.get(pd.Timestamp(t), [])[:n],
            BacktestConfig(first, last, cost, settings.backtest.delisting_haircut, n, f"hybrid_top{n}"),
        )
        results[f"det_top{n}"] = bt.run(
            lambda t, n=n: det_by_month.get(pd.Timestamp(t), [])[:n],
            BacktestConfig(first, last, cost, settings.backtest.delisting_haircut, n, f"det_top{n}"),
        )
    results["bench_ew_pit"] = bt.run(
        universe_selector(inputs.membership, settings.universe.index),
        BacktestConfig(first, last, cost, settings.backtest.delisting_haircut, None, "bench_ew_pit"),
    )
    # secondary benchmark: equal weight of the eligible members only (screens are rebuilt from the store, they
    # do not depend on the jury)
    screens = run_screens(settings, inputs, bt.decision_dates(first, last))
    results["bench_ew_eligible"] = bt.run(
        eligible_selector(inputs.membership, screens, settings.universe.index),
        BacktestConfig(first, last, cost, settings.backtest.delisting_haircut, None, "bench_ew_eligible"),
    )
    usdtry = usdtry_series(settings.paths.store, _vendor_usdtry(prices))
    report: dict[str, Any] = {
        "run": run_name,
        "jury_run": jury_run,
        "generated_at": utc_now_iso(),
        "provenance": provenance,
        "strategies": {},
    }
    n_trials = trial_count(settings, extra_labels=list(results))
    rf = rf_daily_series(settings.paths.store)
    cpi = cpi_series(settings.paths.store)
    for label, res in results.items():
        res.periods.to_csv(out_dir / f"{label}_periods.csv", index=False)
        res.daily.to_csv(out_dir / f"{label}_daily.csv", index=False)
    report["strategies"] = build_entries(results, rf, usdtry, cpi, n_trials, lags=settings.backtest.newey_west_lags)
    # hybrid vs deterministic with the same N (H1), paired on decision dates
    for n in top_ns or settings.backtest.top_n:
        h, d = results[f"hybrid_top{n}"].periods, results[f"det_top{n}"].periods
        diff = active_stats(h, d)["active"]
        report["strategies"][f"hybrid_top{n}"]["vs_deterministic"] = {
            "mean_monthly_diff": float(diff.mean()),
            "newey_west": newey_west_ttest(diff, lags=settings.backtest.newey_west_lags),
            "bootstrap_ci_95": stationary_bootstrap_ci(diff, n_boot=1000),
        }
    tables = secondary_tables(results)
    # pre-registered sensitivities: one-way cost grid and involuntary-exit haircut, for hybrid and det Top-N
    nlist = top_ns or settings.backtest.top_n
    selectors: dict[str, Any] = {}
    for n in nlist:
        selectors[f"hybrid_top{n}"] = lambda t, n=n: by_month.get(pd.Timestamp(t), [])[:n]
        selectors[f"det_top{n}"] = lambda t, n=n: det_by_month.get(pd.Timestamp(t), [])[:n]
    selectors["bench_ew_pit"] = universe_selector(inputs.membership, settings.universe.index)
    top_n_map = {f"{k}_top{n}": n for n in nlist for k in ("hybrid", "det")}
    sens_base = {"first": first, "last": last, "top_n": top_n_map}
    lags = settings.backtest.newey_west_lags
    tables["cost_sensitivity"] = cost_sensitivity(
        bt,
        selectors,
        {**sens_base, "haircut": settings.backtest.delisting_haircut},
        [float(c) for c in settings.backtest.cost_sensitivity_bps],
        benchmark="bench_ew_pit",
        lags=lags,
    )
    tables["haircut_sensitivity"] = cost_sensitivity(
        bt,
        selectors,
        {**sens_base, "haircut": settings.backtest.haircut_sensitivity},
        [cost],
        benchmark="bench_ew_pit",
        lags=lags,
    )
    for name, tbl in tables.items():
        if isinstance(tbl, pd.DataFrame):
            tbl.to_csv(out_dir / f"table_{name}.csv", index=False)
    report["tables"] = tables_to_json(tables)
    report["rf"] = {"source": "tcmb_policy_rate" if isinstance(rf, pd.Series) else "none", "applies_to": "tl"}
    # random-jury null: same candidates, uniform scores, identical consensus + portfolio path
    n_null = n_random if n_random is not None else settings.backtest.random_jury_seeds
    headline = settings.backtest.headline_top_n
    cands = cons[["as_of", "ticker", "composite_rank"]].copy()
    cands["as_of"] = cands["as_of"].dt.strftime("%Y-%m-%d")
    null_cagr: list[float] = []
    for seed in range(n_null):
        rc = consensus(random_scores(cands, seed), theta=settings.jury.theta, min_votes=settings.jury.min_votes)
        rc["as_of"] = pd.to_datetime(rc["as_of"])
        rmonth = {
            m: g.dropna(subset=["rank_J"]).sort_values("rank_J")["ticker"].tolist() for m, g in rc.groupby("as_of")
        }
        rr = bt.run(
            lambda t, rm=rmonth: rm.get(pd.Timestamp(t), [])[:headline],
            BacktestConfig(first, last, cost, settings.backtest.delisting_haircut, headline, f"random_{seed}"),
        )
        null_cagr.append(summarise(rr.daily)["cagr"])
    if null_cagr:
        hy = report["strategies"][f"hybrid_top{headline}"]["tl"]["cagr"]
        arr = np.array(null_cagr)
        report["random_jury_null"] = {
            "n": int(len(arr)),
            "cagr_mean": float(arr.mean()),
            "cagr_p05": float(np.quantile(arr, 0.05)),
            "cagr_p95": float(np.quantile(arr, 0.95)),
            "hybrid_cagr": hy,
            "hybrid_percentile": float((arr < hy).mean()),
        }
    report["secondary"] = run_secondary(settings, bt, cons, results, report, inputs.panel.close_adj, n_trials)
    write_json(report, out_dir / "report.json")
    record_run(
        settings,
        "hybrid",
        run_name,
        ledger_summary(report["strategies"]),
        {"jury_run": jury_run, "n_trials": n_trials, "random_jury_null": report.get("random_jury_null")},
    )
    log.info("hybrid run %s written to %s", run_name, out_dir)
    return report
