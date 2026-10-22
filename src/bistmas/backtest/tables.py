"""Secondary tables promised in the pre-registration: calendar-year returns, leave-one-year-out inference,
minimum detectable effect, cost sensitivity and the risk-free series used for Sharpe ratios."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as sps

from bistmas.backtest.engine import BacktestConfig, Backtester, BacktestResult
from bistmas.backtest.metrics import active_stats, summarise
from bistmas.backtest.stats import newey_west_ttest
from bistmas.shared.log import get_logger
from bistmas.shared.store import read_parquet

log = get_logger("backtest.tables")

Selector = Callable[[pd.Timestamp], list[str]]


def rf_daily_series(store_dir: Path) -> pd.Series | float:
    """Daily simple risk-free rate (TCMB policy rate, ACT/365) indexed by session; 0.0 when the macro store
    is missing."""
    path = Path(store_dir) / "macro_daily.parquet"
    if not path.exists():
        return 0.0
    m = read_parquet(path, columns=["date", "rf_daily"])
    s = pd.Series(pd.to_numeric(m["rf_daily"], errors="coerce").to_numpy(), index=pd.to_datetime(m["date"]))
    s = s.dropna()
    return s if len(s) else 0.0


def cpi_series(store_dir: Path) -> pd.Series | None:
    """CPI deflator for the CPI-deflated numeraire, indexed by session; ``None`` without CPI (no EVDS key).

    The point-in-time CPI (``cpi_pit``) is a step function: the whole monthly inflation lands on the release
    session, which would put 84 artificial one-day jumps (up to -13 %) into the real daily returns. The
    deflator used here is the ``cpi_pit`` knots (one per release) interpolated GEOMETRICALLY (log-linear in
    calendar days) between consecutive releases and held flat after the last one. Nothing is read from the
    unreleased future of a knot, so the series stays a function of public CPI prints only; it is still an
    ex-post deflator (the interpolation at ``t`` uses the next release), which is what real performance needs.
    """
    path = Path(store_dir) / "macro_daily.parquet"
    if not path.exists():
        return None
    m = read_parquet(path, columns=["date", "cpi_pit"])
    s = pd.Series(pd.to_numeric(m["cpi_pit"], errors="coerce").to_numpy(), index=pd.to_datetime(m["date"])).dropna()
    if s.empty:
        return None
    return geometric_interpolate_steps(s.sort_index())


def geometric_interpolate_steps(step: pd.Series) -> pd.Series:
    """Replace a step series by the log-linear interpolation between its change points (flat outside)."""
    knots = step[step != step.shift()]
    if len(knots) < 2:
        return step.astype(float)
    x = knots.index.asi8.astype(float)
    y = np.log(knots.to_numpy(dtype=float))
    out = np.exp(np.interp(step.index.asi8.astype(float), x, y))
    return pd.Series(out, index=step.index)


def usdtry_series(store_dir: Path, vendor: pd.Series) -> pd.Series:
    """USD/TRY for the USD numeraire: the EVDS series (``macro_daily.usdtry_evds``) where present, else the
    vendor rate (``prices_daily.usdtry``) rescaled to the EVDS level on the overlap.

    The vendor series has sessions that carry a neighbouring day's rate (e.g. 2018-08-10). Vendor sessions
    whose value is identical to the previous session's are flagged (a genuine repeat of a 4-decimal rate
    is practically impossible), logged, and replaced by the geometric interpolation of their neighbours when
    EVDS cannot supply the day.
    """
    vendor = vendor.dropna().sort_index()
    vendor = vendor[vendor > 0]
    stale = vendor.eq(vendor.shift())
    if stale.any():
        log.warning(
            "vendor USD/TRY repeats the previous session on %d dates: %s",
            int(stale.sum()),
            ",".join(str(d.date()) for d in vendor.index[stale][:10]),
        )
    repaired = vendor.mask(stale)
    repaired = pd.Series(np.exp(np.log(repaired).interpolate(limit_area="inside")), index=repaired.index).fillna(vendor)
    path = Path(store_dir) / "macro_daily.parquet"
    if not path.exists():
        return repaired
    m = read_parquet(path)
    if "usdtry_evds" not in m.columns:
        return repaired
    evds = pd.Series(pd.to_numeric(m["usdtry_evds"], errors="coerce").to_numpy(), index=pd.to_datetime(m["date"]))
    evds = evds.dropna()
    evds = evds[evds > 0]
    if evds.empty:
        return repaired
    common = repaired.index.intersection(evds.index)
    scale = float((repaired.loc[common] / evds.loc[common]).median()) if len(common) else 1.0
    out = repaired.copy()
    out.loc[out.index.intersection(evds.index)] = evds.loc[out.index.intersection(evds.index)] * scale
    # EVDS sessions the vendor lacks extend the series
    extra = evds.index.difference(out.index)
    return pd.concat([out, evds.loc[extra] * scale]).sort_index()


def align_rf(rf: pd.Series | float, daily: pd.DataFrame) -> pd.Series | float:
    """Risk-free return earned by each of the strategy's sessions, indexed by date (the one-calendar-day
    rate forward-filled, times the calendar days since the previous session; the first session counts one
    day), or the scalar unchanged."""
    if not isinstance(rf, pd.Series):
        return rf
    idx = pd.DatetimeIndex(pd.to_datetime(daily["date"]))
    per_day = rf.sort_index().reindex(idx, method="ffill").fillna(0.0)
    gaps = pd.Series(idx, index=idx).diff().dt.days.fillna(1.0)
    return per_day * gaps


def calendar_year_table(results: dict[str, BacktestResult], benchmark: str = "bench_ew_pit") -> pd.DataFrame:
    """Net return per holding year (periods grouped by execution-session year) for every strategy, plus the
    active return vs the benchmark. Columns: ``year, <label>..., active_<label>...``."""
    frames = {}
    for label, res in results.items():
        p = res.periods
        if p.empty:
            continue
        year = pd.to_datetime(p["exec_date"]).dt.year
        frames[label] = (1 + p["net_ret"]).groupby(year).prod() - 1
    table = pd.DataFrame(frames)
    table.index.name = "year"
    if benchmark in table.columns:
        for label in list(table.columns):
            if label != benchmark:
                table[f"active_{label}"] = table[label] - table[benchmark]
    return table.reset_index()


def leave_one_year_out(active: pd.Series, dates: pd.Series, lags: int = 3) -> pd.DataFrame:
    """Newey-West mean test of the monthly active return with each calendar year removed in turn.

    ``dates`` must be the HOLDING dates (execution sessions): a December decision is held in January, and
    the pre-registered sample is 84 holding months Jan-2018 .. Dec-2024."""
    years = pd.to_datetime(dates).dt.year.to_numpy()
    a = pd.Series(active).to_numpy(dtype=float)
    rows = []
    for y in sorted(set(years)):
        keep = years != y
        nw = newey_west_ttest(pd.Series(a[keep]), lags=lags)
        rows.append({"year_excluded": int(y), "n": int(keep.sum()), "mean": nw["mean"], "t": nw["t"], "p": nw["p"]})
    return pd.DataFrame(rows, columns=["year_excluded", "n", "mean", "t", "p"])


def minimum_detectable_effect(
    active: pd.Series, lags: int = 3, alpha: float = 0.05, power: float = 0.8
) -> dict[str, Any]:
    """Smallest monthly active return the paired Newey-West test would detect with ``power`` at ``alpha``
    (two-sided), given the realised HAC standard error: ``MDE = se * (z_{1-alpha/2} + z_{power})``."""
    nw = newey_west_ttest(pd.Series(active), lags=lags)
    se = nw.get("se", np.nan)
    if se is None or not np.isfinite(se):
        return {"mde_monthly": np.nan, "mde_annualised": np.nan, "se": np.nan, "n": nw.get("n", 0)}
    z = sps.norm.ppf(1 - alpha / 2) + sps.norm.ppf(power)
    mde = float(se * z)
    return {"mde_monthly": mde, "mde_annualised": mde * 12, "se": float(se), "n": int(nw["n"]), "power": power}


def cost_sensitivity(
    bt: Backtester,
    selectors: dict[str, Selector],
    base_cfg: dict[str, Any],
    costs_bps: list[float],
    benchmark: str = "bench_ew_pit",
    lags: int = 3,
) -> pd.DataFrame:
    """Re-run every strategy (and the benchmark) at each one-way cost; report CAGR, mean monthly active return
    and its Newey-West t. ``selectors`` maps label -> selector; ``base_cfg`` holds the shared BacktestConfig
    fields (first/last decision, haircut) and per-label ``top_n`` under ``base_cfg['top_n'][label]``."""
    rows = []
    for c in costs_bps:
        runs: dict[str, BacktestResult] = {}
        for label, sel in selectors.items():
            cfg = BacktestConfig(
                base_cfg["first"],
                base_cfg["last"],
                cost_bps_one_way=float(c),
                delisting_haircut=base_cfg.get("haircut", 0.0),
                top_n=base_cfg.get("top_n", {}).get(label),
                label=label,
            )
            runs[label] = bt.run(sel, cfg)
        bench = runs.get(benchmark)
        for label, res in runs.items():
            summ = summarise(res.daily, res.periods)
            row = {
                "cost_bps_one_way": float(c),
                "strategy": label,
                "cagr_tl": summ.get("cagr"),
                "total_cost": summ.get("total_cost"),
            }
            if bench is not None and label != benchmark:
                act = active_stats(res.periods, bench.periods)["active"]
                nw = newey_west_ttest(act, lags=lags)
                row.update({"active_monthly_mean": float(act.mean()), "nw_t": nw["t"], "nw_p": nw["p"]})
            rows.append(row)
    return pd.DataFrame(rows)


def secondary_tables(
    results: dict[str, BacktestResult], benchmark: str = "bench_ew_pit", lags: int = 3
) -> dict[str, Any]:
    """Calendar-year table, leave-one-year-out and MDE for every strategy vs ``benchmark``."""
    out: dict[str, Any] = {"calendar_year": calendar_year_table(results, benchmark)}
    loyo: dict[str, pd.DataFrame] = {}
    mde: dict[str, dict[str, Any]] = {}
    bench = results.get(benchmark)
    if bench is not None:
        for label, res in results.items():
            if label == benchmark or res.periods.empty:
                continue
            act = active_stats(res.periods, bench.periods)
            exec_by_decision = res.periods.set_index("decision_date")["exec_date"]
            loyo[label] = leave_one_year_out(
                act["active"], pd.Series(exec_by_decision.reindex(act.index).to_numpy()), lags=lags
            )
            mde[label] = minimum_detectable_effect(act["active"], lags=lags)
    out["leave_one_year_out"] = loyo
    out["mde"] = mde
    return out


def tables_to_json(tables: dict[str, Any]) -> dict[str, Any]:
    """JSON-friendly copy (DataFrames -> records)."""
    out: dict[str, Any] = {}
    for k, v in tables.items():
        if isinstance(v, pd.DataFrame):
            out[k] = v.to_dict("records")
        elif isinstance(v, dict):
            out[k] = {kk: (vv.to_dict("records") if isinstance(vv, pd.DataFrame) else vv) for kk, vv in v.items()}
        else:
            out[k] = v
    return out
