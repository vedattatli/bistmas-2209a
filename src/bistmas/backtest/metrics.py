"""Performance metrics on daily and monthly series (TL, USD and CPI-deflated numeraires)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(nav: pd.Series, dates: pd.Series) -> float:
    """Compound annual growth from a starting value of 1.0 (the book before its first session's return) to
    the last NAV. Dividing by ``nav.iloc[0]`` instead would silently drop the first session's return and the
    initial transaction cost."""
    if len(nav) < 2:
        return np.nan
    years = (pd.Timestamp(dates.iloc[-1]) - pd.Timestamp(dates.iloc[0])).days / 365.25
    return float(nav.iloc[-1]) ** (1 / years) - 1 if years > 0 else np.nan


def max_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()
    return float((nav / peak - 1).min())


def sharpe(daily_ret: pd.Series, rf_daily: pd.Series | float = 0.0) -> float:
    ex = daily_ret - (rf_daily if np.isscalar(rf_daily) else rf_daily.reindex(daily_ret.index).fillna(0.0))
    sd = ex.std()
    return float(np.sqrt(TRADING_DAYS) * ex.mean() / sd) if sd and sd > 0 else np.nan


def sortino(daily_ret: pd.Series, rf_daily: pd.Series | float = 0.0) -> float:
    ex = daily_ret - (rf_daily if np.isscalar(rf_daily) else rf_daily.reindex(daily_ret.index).fillna(0.0))
    dd = np.sqrt((np.minimum(ex, 0.0) ** 2).mean()) if len(ex) else np.nan  # downside deviation over all N
    return float(np.sqrt(TRADING_DAYS) * ex.mean() / dd) if dd and dd > 0 else np.nan


def to_numeraire(daily: pd.DataFrame, converter: pd.Series | None) -> pd.DataFrame:
    """Re-express a daily NAV in another numeraire (e.g. divide by USD/TRY or by a CPI index)."""
    out = daily.copy()
    if converter is None:
        return out
    conv = converter.reindex(pd.to_datetime(out["date"])).ffill().bfill().to_numpy()
    nav = out["nav"].to_numpy() / conv * conv[0]
    out["nav"] = nav
    out["ret"] = pd.Series(nav).pct_change().fillna(nav[0] - 1.0).to_numpy()
    return out


def convert_periods(periods: pd.DataFrame, converter: pd.Series | None) -> pd.DataFrame:
    """Period returns re-expressed in another numeraire: ``(1 + r) * X(exec) / X(exit) - 1`` for the net and
    gross return (``X`` = USD/TRY or the CPI deflator on the execution and exit sessions)."""
    out = periods.copy()
    if converter is None or out.empty:
        return out
    c = converter.sort_index()
    x0 = c.reindex(pd.DatetimeIndex(pd.to_datetime(out["exec_date"])), method="ffill").bfill().to_numpy()
    x1 = c.reindex(pd.DatetimeIndex(pd.to_datetime(out["exit_date"])), method="ffill").bfill().to_numpy()
    ratio = x0 / x1
    for col in ("net_ret", "gross_ret"):
        if col in out.columns:
            out[col] = (1 + out[col].to_numpy()) * ratio - 1
    return out


def summarise(
    daily: pd.DataFrame, periods: pd.DataFrame | None = None, rf_daily: pd.Series | float = 0.0
) -> dict[str, Any]:
    d = daily.dropna(subset=["ret"])
    if d.empty:
        return {}
    ret = d.set_index(pd.to_datetime(d["date"]))["ret"]
    nav = (1 + ret).cumprod()
    out = {
        "cagr": cagr(nav, d["date"]),
        "total_return": float(nav.iloc[-1] - 1),
        "vol": float(ret.std() * np.sqrt(TRADING_DAYS)),
        "sharpe": sharpe(ret, rf_daily),
        "sortino": sortino(ret, rf_daily),
        "max_drawdown": max_drawdown(nav),
        "n_days": int(len(ret)),
    }
    if periods is not None and len(periods):
        out["n_periods"] = int(len(periods))
        out["avg_turnover"] = float(periods["turnover"].mean())
        out["total_cost"] = float(periods["cost"].sum())
        out["hit_rate_monthly"] = float((periods["net_ret"] > 0).mean())
    return out


def active_stats(strategy_periods: pd.DataFrame, benchmark_periods: pd.DataFrame) -> pd.DataFrame:
    """Monthly active returns aligned on decision dates."""
    a = strategy_periods.set_index("decision_date")["net_ret"]
    b = benchmark_periods.set_index("decision_date")["net_ret"]
    df = pd.concat([a.rename("strategy"), b.rename("benchmark")], axis=1, join="inner")
    df["active"] = df["strategy"] - df["benchmark"]
    return df
