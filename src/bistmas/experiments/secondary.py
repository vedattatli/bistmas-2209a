"""Pre-registered secondary analyses of the hybrid run (docs/preregistration.md, "Secondary").

Post-processing of the consensus table and the backtest periods plus the BUY-only re-runs through the same
``Backtester``; no LLM call. Items: Bonferroni thresholds next to raw p-values, BUY-only variants
(theta, theta +- step), per-month Spearman(J, composite) (echo check), incremental IC of J beyond the
composite, Kish effective votes, pairwise agent Spearman and Krippendorff's alpha (interval).
"""

from __future__ import annotations

from datetime import date
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from bistmas.backtest.engine import BacktestConfig, Backtester
from bistmas.backtest.metrics import active_stats, summarise
from bistmas.backtest.stats import newey_west_ttest
from bistmas.backtest.tables import align_rf, rf_daily_series
from bistmas.shared.config import Settings
from bistmas.shared.log import get_logger

log = get_logger("experiments.secondary")

AGENT_COLS = ("quality_growth", "value", "risk_timing")
J_DECIMALS = 6


def bonferroni(strategies: dict[str, Any], n_trials: int, alpha: float = 0.05) -> dict[str, Any]:
    """Bonferroni threshold alpha/m next to each raw Newey-West p-value; m = configurations in the ledger."""
    m = max(int(n_trials), 1)
    rows = {}
    for label, entry in strategies.items():
        p = (entry.get("newey_west") or {}).get("p")
        if p is None or not np.isfinite(p):
            continue
        rows[label] = {
            "p_raw": float(p),
            "p_bonferroni": float(min(1.0, p * m)),
            "significant_raw": bool(p < alpha),
            "significant_bonferroni": bool(p < alpha / m),
        }
    return {"n_configurations": m, "alpha": alpha, "alpha_adjusted": alpha / m, "strategies": rows}


def spearman_j_composite(cons: pd.DataFrame) -> dict[str, Any]:
    """Spearman(J, composite score) per month; composite score = -composite_rank (higher = better)."""
    rows = {}
    for m, g in cons.dropna(subset=["J", "composite_rank"]).groupby("as_of"):
        if len(g) >= 4 and g["J"].nunique() > 1:
            rows[pd.Timestamp(m)] = float(g["J"].rank().corr(-g["composite_rank"].rank()))
    s = pd.Series(rows, dtype=float)
    return {
        "n_months": int(len(s)),
        "mean": float(s.mean()) if len(s) else None,
        "min": float(s.min()) if len(s) else None,
        "max": float(s.max()) if len(s) else None,
        "by_month": {k.strftime("%Y-%m-%d"): v for k, v in s.items()},
    }


def next_month_returns(close_adj: pd.DataFrame, decisions: list[pd.Timestamp]) -> dict[pd.Timestamp, pd.Series]:
    """Close-to-close return of every ticker from each decision date to the next one (last has none)."""
    close = close_adj.sort_index()
    close.index = pd.to_datetime(close.index)
    out = {}
    for t0, t1 in zip(decisions[:-1], decisions[1:], strict=True):
        a, b = close.loc[:t0], close.loc[:t1]
        if len(a) and len(b):
            out[t0] = (b.iloc[-1] / a.iloc[-1] - 1).replace([np.inf, -np.inf], np.nan)
    return out


def incremental_ic(cons: pd.DataFrame, fwd: dict[pd.Timestamp, pd.Series], lags: int = 3) -> dict[str, Any]:
    """Monthly cross-section: pct-rank(next-month return) ~ 1 + pct-rank(composite) + pct-rank(J); the J
    coefficients are tested with Newey-West over months (incremental information of J beyond composite)."""
    b_j, b_c = {}, {}
    for m, g in cons.dropna(subset=["J", "composite_rank"]).groupby("as_of"):
        r = fwd.get(pd.Timestamp(m))
        if r is None:
            continue
        g = g.assign(ret=g["ticker"].map(r)).dropna(subset=["ret"])
        if len(g) < 5 or g["J"].nunique() < 2:
            continue
        y = g["ret"].rank(pct=True).to_numpy()
        x = np.column_stack([np.ones(len(g)), (-g["composite_rank"]).rank(pct=True), g["J"].rank(pct=True)])
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        b_c[pd.Timestamp(m)], b_j[pd.Timestamp(m)] = float(beta[1]), float(beta[2])
    sj, sc = pd.Series(b_j, dtype=float), pd.Series(b_c, dtype=float)
    return {
        "n_months": int(len(sj)),
        "coef_J": newey_west_ttest(sj, lags=lags),
        "coef_composite": newey_west_ttest(sc, lags=lags),
    }


def krippendorff_alpha_interval(values: np.ndarray) -> float:
    """Krippendorff's alpha, interval metric. ``values``: units x raters, NaN = missing."""
    v = np.asarray(values, dtype=float)
    m_u = np.isfinite(v).sum(axis=1)
    keep = m_u >= 2
    v, m_u = v[keep], m_u[keep]
    n = m_u.sum()
    if n <= 1:
        return float("nan")
    d_o = 0.0
    for row, m in zip(v, m_u, strict=True):
        x = row[np.isfinite(row)]
        d_o += ((x[:, None] - x[None, :]) ** 2).sum() / (m - 1)
    d_o /= n
    allv = v[np.isfinite(v)]
    d_e = ((allv[:, None] - allv[None, :]) ** 2).sum() / (n * (n - 1))
    if d_e == 0:
        return float("nan")
    return float(1 - d_o / d_e)


def kish_effective(w: np.ndarray) -> float:
    """Kish effective sample size (sum w)^2 / sum w^2 of the positive finite weights."""
    w = np.asarray(w, dtype=float)
    w = w[np.isfinite(w) & (w > 0)]
    return float(w.sum() ** 2 / (w**2).sum()) if len(w) else 0.0


def agreement(cons: pd.DataFrame, agents: tuple[str, ...] = AGENT_COLS) -> dict[str, Any]:
    """Kish effective votes (equal vote weights, so it reflects abstentions) and the correlation-adjusted
    effective votes k/(1+(k-1)*mean pairwise rho); pairwise agent Spearman (pooled and mean of monthly);
    Krippendorff alpha (interval) over (month, ticker) units."""
    cols = [a for a in agents if a in cons.columns]
    sc = cons[cols].apply(pd.to_numeric, errors="coerce")
    kish = sc.notna().astype(float).apply(lambda r: kish_effective(r.to_numpy()), axis=1)
    pair: dict[str, Any] = {}
    rhos = []
    for a, b in combinations(cols, 2):
        pooled = sc[a].corr(sc[b], method="spearman")
        monthly = []
        for _, g in cons.assign(_a=sc[a], _b=sc[b]).groupby("as_of"):
            if g["_a"].nunique() > 1 and g["_b"].nunique() > 1:
                monthly.append(g["_a"].corr(g["_b"], method="spearman"))
        pair[f"{a}~{b}"] = {
            "spearman_pooled": None if pd.isna(pooled) else float(pooled),
            "spearman_monthly_mean": float(np.nanmean(monthly)) if monthly else None,
        }
        if not pd.isna(pooled):
            rhos.append(pooled)
    k = len(cols)
    rho_bar = float(np.mean(rhos)) if rhos else None
    return {
        "kish_votes_mean": float(kish.mean()) if len(kish) else None,
        "effective_votes_corr_adjusted": float(k / (1 + (k - 1) * max(rho_bar, 0.0))) if rhos else None,
        "pairwise": pair,
        "mean_pairwise_spearman": rho_bar,
        "krippendorff_alpha_interval": krippendorff_alpha_interval(sc.to_numpy()),
    }


def buy_only(
    bt: Backtester,
    cons: pd.DataFrame,
    bench_periods: pd.DataFrame,
    thetas: list[float],
    first: date,
    last: date,
    cost_bps: float,
    haircut: float,
    lags: int = 3,
    rf: pd.Series | float = 0.0,
) -> dict[str, Any]:
    """Equal-weight portfolio of every candidate with J >= theta (J rounded as in the ranking); cash when
    there is no BUY. Returns BUY counts, summary and the active test vs the benchmark per theta."""
    j = cons.dropna(subset=["J"]).assign(J=lambda d: d["J"].round(J_DECIMALS))
    out: dict[str, Any] = {}
    for th in thetas:
        buys = {
            pd.Timestamp(m): g.loc[g["J"] >= round(th, J_DECIMALS), "ticker"].tolist() for m, g in j.groupby("as_of")
        }
        res = bt.run(
            lambda t, b=buys: b.get(pd.Timestamp(t), []),
            BacktestConfig(first, last, cost_bps, haircut, None, f"buy_only_{th:.2f}"),
        )
        counts = [len(v) for v in buys.values()]
        act = active_stats(res.periods, bench_periods)
        out[f"{th:.2f}"] = {
            "theta": th,
            "buy_count_mean": float(np.mean(counts)) if counts else None,
            "buy_count_min": min(counts, default=None),
            "buy_count_max": max(counts, default=None),
            "months_in_cash": int((res.periods["n_holdings"] == 0).sum()),
            "tl": summarise(res.daily, res.periods, rf_daily=align_rf(rf, res.daily)),
            "newey_west": newey_west_ttest(act["active"], lags=lags),
        }
    return out


def run_secondary(
    settings: Settings,
    bt: Backtester,
    cons: pd.DataFrame,
    results: dict[str, Any],
    report: dict[str, Any],
    close_adj: pd.DataFrame,
    n_trials: int,
) -> dict[str, Any]:
    """All secondary analyses; returns the dict stored under report["secondary"]."""
    cfg = settings.backtest
    first = date.fromisoformat(settings.rebalance.first_decision)
    last = date.fromisoformat(settings.rebalance.last_decision)
    step, th = cfg.theta_step, settings.jury.theta
    decisions = sorted(pd.Timestamp(d) for d in cons["as_of"].unique())
    cost = cfg.cost_bps_commission + cfg.cost_bps_slippage
    return {
        "bonferroni": bonferroni(report["strategies"], n_trials),
        "buy_only": buy_only(
            bt,
            cons,
            results["bench_ew_pit"].periods,
            [th - step, th, th + step],
            first,
            last,
            cost,
            cfg.delisting_haircut,
            cfg.newey_west_lags,
            rf_daily_series(settings.paths.store),
        ),
        "echo_spearman": spearman_j_composite(cons),
        "incremental_ic": incremental_ic(cons, next_month_returns(close_adj, decisions), cfg.newey_west_lags),
        "agreement": agreement(cons),
    }
