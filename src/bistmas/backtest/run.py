"""Orchestration: assemble the feature inputs from the store, run the deterministic screen for every
decision date, back-test the Top-N portfolios against the point-in-time equal-weight benchmark and
write the run artefacts under ``data/runs/<run_name>/``.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from bistmas.backtest.benchmark import eligible_selector, index_daily, universe_selector
from bistmas.backtest.engine import BacktestConfig, Backtester, BacktestResult
from bistmas.backtest.metrics import active_stats, convert_periods, summarise, to_numeraire
from bistmas.backtest.stats import deflated_sharpe, newey_west_ttest, stationary_bootstrap_ci
from bistmas.backtest.tables import (
    align_rf,
    cost_sensitivity,
    cpi_series,
    rf_daily_series,
    secondary_tables,
    tables_to_json,
    usdtry_series,
)
from bistmas.data.calendar_build import load_calendar
from bistmas.data.prices import load_prices
from bistmas.data.security_master import load_security_master
from bistmas.data.statements_fetch import load_statements
from bistmas.data.universe.official import read_membership
from bistmas.data.universe.provider import coerce_membership
from bistmas.data.universe.rule_based import RULE_BASED_MEMBERSHIP_FILENAME
from bistmas.scoring.features import FeatureInputs, PricePanel
from bistmas.scoring.screen import screen_one
from bistmas.scoring.statements_view import add_ttm, filings_wide, load_concept_map
from bistmas.shared.config import Settings
from bistmas.shared.hashing import sha256_json
from bistmas.shared.ledger import record_run, trial_count
from bistmas.shared.log import get_logger
from bistmas.shared.store import read_parquet, utc_now_iso, write_json, write_parquet

log = get_logger("backtest.run")


def prepare_master(master: pd.DataFrame, members: set[str]) -> pd.DataFrame:
    """Ticker-indexed master for the features: alias rows kept (one row per ticker, current preferred).

    Warns loudly when a universe member has no row and would silently fall back to OTHER/industrial.
    """
    out = master.sort_values("is_current", ascending=False, kind="stable").drop_duplicates("ticker").set_index("ticker")
    missing = sorted(members - set(out.index))
    if missing:
        log.warning(
            "%d universe members have no security-master row and fall back to OTHER/industrial: %s",
            len(missing),
            missing[:20],
        )
    return out


def load_membership(settings: Settings) -> pd.DataFrame:
    """Point-in-time membership of the provider selected by ``settings.universe.provider``.

    ``official``: the rolled-forward official XU100/50/30 lists. ``rule_based``: the replica written by
    ``RuleBasedProvider.write`` (``universe_membership_rule_based.parquet``); it must be built first.

    Limitation of the rule-based robustness row: the price panel (and therefore the replica's candidate
    pool) contains only the codes that are, or later become, official index members, so the replica can never
    pick a name that qualified on the rules but was never an official member, and it can pick names that
    entered an index only later. Pull prices for all equities before reporting it as unbiased.
    """
    provider = settings.universe.provider
    if provider == "official":
        return read_membership(settings.paths.store)
    if provider == "rule_based":
        path = settings.paths.store / RULE_BASED_MEMBERSHIP_FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"universe.provider = 'rule_based' but {path} has not been built; run "
                "RuleBasedProvider(store).write(store) first, or set provider = 'official'"
            )
        return coerce_membership(read_parquet(path))
    raise ValueError(f"unknown universe.provider {provider!r} (official | rule_based)")


def load_inputs(settings: Settings) -> tuple[FeatureInputs, pd.DataFrame, dict[str, Any]]:
    """Feature inputs + raw prices + provenance from the store tables."""
    store = settings.paths.store
    membership = load_membership(settings)
    prices = load_prices(store)
    master = load_security_master(store)
    master = prepare_master(master, set(membership["ticker"]))
    cfg = load_concept_map()
    statements = load_statements(store)
    templates = master["template"]
    wide = filings_wide(statements, templates, cfg)
    flow_aliases = sorted({a for t in ("industrial", "bank", "insurer", "other_financial") for a in cfg[t]["flows"]})
    fallback_path = store / "statements_fallback_wide.parquet"
    if fallback_path.exists():
        fb = read_parquet(fallback_path)
        fb = fb[~fb["ticker"].isin(set(wide["ticker"]))] if len(wide) else fb
        for col in ("period_end", "publish_ts", "availability_date"):
            fb[col] = pd.to_datetime(fb[col])
        wide = pd.concat([wide, fb], ignore_index=True) if len(wide) else fb
        log.info("fallback statements: %d rows for %d tickers", len(fb), fb["ticker"].nunique())
    wide = add_ttm(wide, [a for a in flow_aliases if f"{a}_cur" in wide.columns]) if len(wide) else wide
    tax = {int(k): float(v) for k, v in cfg["tax"]["statutory"].items()}
    inputs = FeatureInputs(membership, PricePanel.from_long(prices), wide, master, tax, settings.universe.index)
    provenance = {
        "universe_provider": settings.universe.provider,
        "membership_rows": int(len(membership)),
        "price_rows": int(len(prices)),
        "price_vintage": str(prices["source_vintage"].iloc[0])
        if "source_vintage" in prices.columns and len(prices)
        else "",
        "statement_rows": int(len(statements)),
        "filings": int(len(wide)),
        "concept_map_hash": sha256_json(cfg)[:12],
    }
    return inputs, prices, provenance


def run_screens(settings: Settings, inputs: FeatureInputs, decisions: list[date]) -> pd.DataFrame:
    frames = []
    for d in decisions:
        s = screen_one(d, inputs, settings)
        if len(s):
            frames.append(s)
        log.info(
            "screen %s: %d members, %d eligible, top-%d ready",
            d,
            len(s),
            int(s["eligible"].sum()) if len(s) else 0,
            settings.scoring.top_n_screen,
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_deterministic(
    settings: Settings, run_name: str = "deterministic", top_ns: list[int] | None = None
) -> dict[str, Any]:
    """Deterministic-only backtest: Top-N by composite for every N in ``top_ns`` + benchmarks + stats."""
    inputs, prices, provenance = load_inputs(settings)
    calendar = load_calendar(settings.paths.store)
    first, last = (
        date.fromisoformat(settings.rebalance.first_decision),
        date.fromisoformat(settings.rebalance.last_decision),
    )
    bt = Backtester(calendar, inputs.panel.close_adj * 0 + _vwap_panel(prices), inputs.panel.close_adj)
    decisions = bt.decision_dates(first, last)
    screens = run_screens(settings, inputs, decisions)
    out_dir = settings.paths.runs / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(_parquet_safe(screens), out_dir / "screens.parquet")
    ranked = (
        {
            pd.Timestamp(d): g.assign(rank=pd.to_numeric(g["rank"], errors="coerce"))
            .dropna(subset=["rank"])
            .sort_values("rank")["ticker"]
            .tolist()
            for d, g in screens.groupby("as_of")
        }
        if len(screens)
        else {}
    )
    cost = settings.backtest.cost_bps_commission + settings.backtest.cost_bps_slippage
    results: dict[str, BacktestResult] = {}
    for n in top_ns or settings.backtest.top_n:
        cfg = BacktestConfig(
            first,
            last,
            cost_bps_one_way=cost,
            delisting_haircut=settings.backtest.delisting_haircut,
            top_n=n,
            label=f"det_top{n}",
        )
        results[cfg.label] = bt.run(lambda t, n=n: ranked.get(pd.Timestamp(t), [])[:n], cfg)
    bench_cfg = BacktestConfig(
        first,
        last,
        cost_bps_one_way=cost,
        delisting_haircut=settings.backtest.delisting_haircut,
        top_n=None,
        label="bench_ew_pit",
    )
    results["bench_ew_pit"] = bt.run(universe_selector(inputs.membership, settings.universe.index), bench_cfg)
    results["bench_ew_eligible"] = bt.run(
        eligible_selector(inputs.membership, screens, settings.universe.index),
        BacktestConfig(
            first,
            last,
            cost_bps_one_way=cost,
            delisting_haircut=settings.backtest.delisting_haircut,
            top_n=None,
            label="bench_ew_eligible",
        ),
    )
    xu = index_daily(prices)
    report: dict[str, Any] = {
        "run": run_name,
        "generated_at": utc_now_iso(),
        "provenance": provenance,
        "decisions": [str(d) for d in decisions],
        "strategies": {},
    }
    n_trials = trial_count(settings, extra_labels=list(results))
    rf = rf_daily_series(settings.paths.store)  # TCMB policy rate for TL Sharpe/Sortino (USD leg: rf = 0)
    cpi = cpi_series(settings.paths.store)  # geometrically interpolated PIT CPI (EVDS), CPI-deflated numeraire
    usdtry = usdtry_series(settings.paths.store, _vendor_usdtry(prices))
    for label, res in results.items():
        res.periods.to_csv(out_dir / f"{label}_periods.csv", index=False)
        res.daily.to_csv(out_dir / f"{label}_daily.csv", index=False)
    report["strategies"] = build_entries(results, rf, usdtry, cpi, n_trials, lags=settings.backtest.newey_west_lags)
    tables = secondary_tables(results)
    selectors = {
        label: (lambda t, n=n: ranked.get(pd.Timestamp(t), [])[:n])
        for n, label in ((n, f"det_top{n}") for n in (top_ns or settings.backtest.top_n))
    }
    selectors["bench_ew_pit"] = universe_selector(inputs.membership, settings.universe.index)
    selectors["bench_ew_eligible"] = eligible_selector(inputs.membership, screens, settings.universe.index)
    tables["cost_sensitivity"] = cost_sensitivity(
        bt,
        selectors,
        {
            "first": first,
            "last": last,
            "haircut": settings.backtest.delisting_haircut,
            "top_n": {f"det_top{n}": n for n in (top_ns or settings.backtest.top_n)},
        },
        [float(c) for c in settings.backtest.cost_sensitivity_bps],
        lags=settings.backtest.newey_west_lags,
    )
    tables["haircut_sensitivity"] = cost_sensitivity(
        bt,
        selectors,
        {
            "first": first,
            "last": last,
            "haircut": settings.backtest.haircut_sensitivity,
            "top_n": {f"det_top{n}": n for n in (top_ns or settings.backtest.top_n)},
        },
        [cost],
        lags=settings.backtest.newey_west_lags,
    )
    for name, tbl in tables.items():
        if isinstance(tbl, pd.DataFrame):
            tbl.to_csv(out_dir / f"table_{name}.csv", index=False)
    report["tables"] = tables_to_json(tables)
    report["rf"] = {"source": "tcmb_policy_rate" if isinstance(rf, pd.Series) else "none", "applies_to": "tl"}
    report["numeraires"] = {
        "usd": "evds_preferred_vendor_repaired" if usdtry is not None else "none",
        "real": "cpi_pit_geometric_interpolation" if cpi is not None else "none",
    }
    report["xu100_index"] = summarise(
        xu[
            (xu["date"] >= pd.Timestamp(results["bench_ew_pit"].daily["date"].min()))
            & (xu["date"] <= pd.Timestamp(results["bench_ew_pit"].daily["date"].max()))
        ]
    )
    write_json(report, out_dir / "report.json")
    record_run(settings, "deterministic", run_name, ledger_summary(report["strategies"]), {"n_trials": n_trials})
    log.info("run %s written to %s", run_name, out_dir)
    return report


BENCHMARKS = ("bench_ew_pit", "bench_ew_eligible")


def _vendor_usdtry(prices: pd.DataFrame) -> pd.Series:
    s = prices.groupby("date")["usdtry"].first().dropna()
    s.index = pd.to_datetime(s.index)
    return s


def active_block(active: pd.Series, n_trials: int, lags: int = 3) -> dict[str, Any]:
    """Pre-registered inference on one monthly active-return series (any numeraire)."""
    sd = active.std()
    return {
        "mean": float(active.mean()),
        "newey_west": newey_west_ttest(active, lags=lags),
        "bootstrap_ci_95": stationary_bootstrap_ci(active, n_boot=1000),
        "information_ratio": float(active.mean() / sd * (12**0.5)) if sd > 0 else None,
        "deflated_sharpe": monthly_deflated_sharpe(active, n_trials),
    }


def build_entries(
    results: dict[str, BacktestResult],
    rf: pd.Series | float,
    usdtry: pd.Series | None,
    cpi: pd.Series | None,
    n_trials: int,
    benchmark: str = "bench_ew_pit",
    secondary_benchmark: str = "bench_ew_eligible",
    lags: int = 3,
) -> dict[str, dict[str, Any]]:
    """Per-strategy report entries. Every numeraire (``tl``, ``usd``, ``real``) carries level metrics of the
    re-denominated daily NAV AND the active-return statistics of the period returns converted to that
    numeraire (``entry[num]["active"]``, vs the headline benchmark). The TL block is mirrored at the top level
    (``active_monthly_mean``, ``newey_west``, ...) for older readers. The eligible-only benchmark is a
    secondary row: ``entry["active_vs_eligible"]`` (TL)."""
    numeraires: dict[str, pd.Series | None] = {"tl": None, "usd": usdtry, "real": cpi}
    out: dict[str, dict[str, Any]] = {}
    for label, res in results.items():
        entry: dict[str, Any] = {}
        for num, conv in numeraires.items():
            if num != "tl" and conv is None:
                continue
            periods = convert_periods(res.periods, conv)
            daily = res.daily if num == "tl" else to_numeraire(res.daily, conv)
            block = summarise(daily, periods, rf_daily=align_rf(rf, res.daily) if num == "tl" else 0.0)
            if label not in BENCHMARKS and benchmark in results:
                bench_p = convert_periods(results[benchmark].periods, conv)
                block["active"] = active_block(active_stats(periods, bench_p)["active"], n_trials, lags)
            entry[num] = block
        if label not in BENCHMARKS and "active" in entry.get("tl", {}):
            a = entry["tl"]["active"]
            entry["active_monthly_mean"] = a["mean"]
            for k in ("newey_west", "bootstrap_ci_95", "information_ratio", "deflated_sharpe"):
                entry[k] = a[k]
            if secondary_benchmark in results:
                act2 = active_stats(res.periods, results[secondary_benchmark].periods)["active"]
                entry["active_vs_eligible"] = {"mean": float(act2.mean()), "newey_west": newey_west_ttest(act2, lags)}
        out[label] = entry
    return out


def monthly_deflated_sharpe(monthly_ret: pd.Series, n_trials: int) -> dict[str, Any]:
    """Deflated Sharpe of a monthly return series (non-annualised, with sample skew and kurtosis)."""
    r = pd.to_numeric(monthly_ret, errors="coerce").dropna()
    if len(r) < 3 or r.std() == 0:
        return {"psr": None, "sr_benchmark": None, "n_trials": n_trials}
    sr = float(r.mean() / r.std())
    out = deflated_sharpe(sr, int(len(r)), int(n_trials), skew=float(r.skew()), kurt=float(r.kurt() + 3.0))
    return {**out, "sharpe_monthly": sr}


def ledger_summary(strategies: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compact per-strategy metrics for the ledger line."""
    out: dict[str, dict[str, Any]] = {}
    for label, e in strategies.items():
        tl, usd = e.get("tl", {}), e.get("usd", {})
        out[label] = {
            "cagr_tl": tl.get("cagr"),
            "sharpe_tl": tl.get("sharpe"),
            "mdd_tl": tl.get("max_drawdown"),
            "cagr_usd": usd.get("cagr"),
            "active_t": (e.get("newey_west") or {}).get("t"),
            "information_ratio": e.get("information_ratio"),
            "psr": (e.get("deflated_sharpe") or {}).get("psr"),
        }
    return out


def _vwap_panel(prices: pd.DataFrame) -> pd.DataFrame:
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"])
    if "flag_cancelled_session" in p.columns:
        p = p[~p["flag_cancelled_session"].fillna(False)]
    return p.pivot_table(index="date", columns="ticker", values="vwap_adj", aggfunc="last").sort_index()


def _parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype(str)
    return out
