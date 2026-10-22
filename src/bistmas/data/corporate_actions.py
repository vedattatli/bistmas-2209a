"""Corporate-action audit table inferred from the daily price panel.

İş Yatırım exposes no corporate-action feed anonymously, so events are *inferred* from the price
store (:mod:`bistmas.data.prices`) and written to ``data/store/corporate_actions.parquet`` with one row
per (ticker, date, kind):

``capital_change``
    ``shares_aligned`` (the ex-date-aligned paid-in capital) differs from the previous session. The
    note carries the capital ratio, the implied price ratio ``(1 + ret_adj) / (1 + ret_raw)`` and a
    coarse class: ``bonus`` when the price ratio matches the capital ratio (bedelsiz), ``rights_or_mixed``
    when the price fell by less than the capital ratio (bedelli: new shares were paid for),
    ``reduction`` when capital shrank, ``no_price_effect`` otherwise (private placement at market,
    tiny changes). ``vendor_step_one_day_early`` marks steps that the vendor dated one session early.
``adj_jump``
    ``|ret_raw - ret_adj| > adj_tol`` (default 5 %) on a session without a capital change; the note
    holds the adjustment-factor ratio (``> 1`` is the signature of a cash dividend).
``gap``
    more than ``max_gap_sessions`` calendar sessions missing between two consecutive rows of a ticker
    (suspensions, vendor holes). ``date`` is the first session after the gap and ``ret_*`` the return
    across it; the February-2023 closure never counts because the calendar excludes it.

Columns (``CA_COLUMNS``): ``ticker, date, kind, shares_before, shares_after, ret_raw, ret_adj, note``.
Rows are sorted by ticker, date, kind. Momentum/volatility windows should exclude ``capital_change``
and ``adj_jump`` days (raw returns) — see the red-team controls in the plan.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bistmas.shared.calendar import CANCELLED_SESSIONS, CLOSURES, TradingCalendar
from bistmas.shared.log import get_logger
from bistmas.shared.store import read_parquet, write_parquet

log = get_logger("data.corporate_actions")

STORE_FILENAME = "corporate_actions.parquet"
CA_COLUMNS: list[str] = ["ticker", "date", "kind", "shares_before", "shares_after", "ret_raw", "ret_adj", "note"]
KINDS: tuple[str, ...] = ("capital_change", "adj_jump", "gap")
ADJ_TOL = 0.05
MAX_GAP_SESSIONS = 5
BONUS_MATCH_TOL = 0.03
"""Relative tolerance between the capital ratio and the implied price ratio to call a step ``bonus``."""


def classify_capital_change(ratio: float, implied_price_ratio: float) -> str:
    """Coarse event class from the capital ratio and the implied raw/adjusted price ratio."""
    if np.isnan(ratio):
        return "unknown"
    if ratio < 1:
        return "reduction"
    if np.isnan(implied_price_ratio) or abs(implied_price_ratio - 1) < BONUS_MATCH_TOL:
        return "no_price_effect"
    if abs(implied_price_ratio / ratio - 1) < BONUS_MATCH_TOL:
        return "bonus"
    if 1 < implied_price_ratio < ratio:
        return "rights_or_mixed"
    return "other"


def _session_list(prices_df: pd.DataFrame, calendar: TradingCalendar | None) -> np.ndarray:
    if calendar is not None:
        return np.array(calendar.sessions, dtype="datetime64[D]")
    # Without a real calendar, weekdays stand in for sessions: the panel's own dates cannot reveal a
    # gap when only one (or a few) tickers are present.
    excluded = {pd.Timestamp(d) for d in CANCELLED_SESSIONS | CLOSURES}
    all_dates = pd.to_datetime(prices_df["date"]).dt.normalize()
    weekdays = pd.bdate_range(all_dates.min(), all_dates.max())
    dates = sorted(set(weekdays) - excluded)
    return np.array([d.to_datetime64() for d in dates], dtype="datetime64[D]")


def infer_corporate_actions(
    prices_df: pd.DataFrame,
    adj_tol: float = ADJ_TOL,
    max_gap_sessions: int = MAX_GAP_SESSIONS,
    calendar: TradingCalendar | None = None,
) -> pd.DataFrame:
    """Infer capital changes, adjustment jumps and data gaps from the price panel (see module docstring).

    ``prices_df`` needs ``ticker, date, close_raw, close_adj`` and ``shares_aligned`` (falls back to
    ``shares``); ``flag_cancelled_session`` rows are ignored. Without a ``calendar`` the union of all
    panel dates (minus the hard-coded closures) stands in for the session list when measuring gaps.
    """
    required = {"ticker", "date", "close_raw", "close_adj"}
    missing = sorted(required - set(prices_df.columns))
    if missing:
        raise ValueError(f"prices frame is missing columns {missing}")
    df = prices_df
    if "flag_cancelled_session" in df.columns:
        df = df[~df["flag_cancelled_session"].astype(bool)]
    df = df.sort_values(["ticker", "date"], kind="stable")
    shares_col = "shares_aligned" if "shares_aligned" in df.columns else "shares"
    sessions = _session_list(df, calendar)

    records: list[dict[str, Any]] = []
    for ticker, g in df.groupby("ticker", sort=True):
        dates = pd.to_datetime(g["date"]).to_numpy()
        close_raw = g["close_raw"].to_numpy(dtype=float)
        close_adj = g["close_adj"].to_numpy(dtype=float)
        shares = g[shares_col].to_numpy(dtype=float) if shares_col in g.columns else np.full(len(g), np.nan)
        vendor = g["shares"].to_numpy(dtype=float) if "shares" in g.columns else shares
        with np.errstate(divide="ignore", invalid="ignore"):
            ret_raw = np.concatenate([[np.nan], close_raw[1:] / close_raw[:-1] - 1])
            ret_adj = np.concatenate([[np.nan], close_adj[1:] / close_adj[:-1] - 1])
            adj_factor = np.where(close_raw > 0, close_adj / close_raw, np.nan)
        positions = np.searchsorted(sessions, dates.astype("datetime64[D]"))
        for t in range(1, len(g)):
            cap = not np.isnan(shares[t]) and not np.isnan(shares[t - 1]) and shares[t] != shares[t - 1]
            mismatch = abs(ret_raw[t] - ret_adj[t]) if not np.isnan(ret_raw[t]) and not np.isnan(ret_adj[t]) else np.nan
            if cap:
                ratio = shares[t] / shares[t - 1]
                with np.errstate(divide="ignore", invalid="ignore"):
                    implied = (1 + ret_adj[t]) / (1 + ret_raw[t])
                klass = classify_capital_change(ratio, implied)
                early = not np.isnan(vendor[t - 1]) and vendor[t - 1] == shares[t] and vendor[t - 1] != shares[t - 1]
                note = f"ratio={ratio:.6g}; implied_price_ratio={implied:.6g}; class={klass}"
                if early:
                    note += "; vendor_step_one_day_early"
                elif t + 1 < len(g) and vendor[t + 1] == shares[t] and vendor[t] != shares[t]:
                    note += "; vendor_step_one_day_late"
                if klass == "other":
                    log.warning(
                        "%s %s: capital change class=other (ratio %.4g, implied price ratio %.4g): "
                        "check the vendor adjustment, see configs/adjustment_overrides.csv",
                        ticker,
                        pd.Timestamp(dates[t]).date(),
                        ratio,
                        implied,
                    )
                records.append(
                    {
                        "ticker": ticker,
                        "date": dates[t],
                        "kind": "capital_change",
                        "shares_before": shares[t - 1],
                        "shares_after": shares[t],
                        "ret_raw": ret_raw[t],
                        "ret_adj": ret_adj[t],
                        "note": note,
                    }
                )
            elif not np.isnan(mismatch) and mismatch > adj_tol:
                with np.errstate(divide="ignore", invalid="ignore"):
                    af_ratio = adj_factor[t] / adj_factor[t - 1]
                hint = "dividend?" if af_ratio > 1 else "unexplained"
                records.append(
                    {
                        "ticker": ticker,
                        "date": dates[t],
                        "kind": "adj_jump",
                        "shares_before": np.nan,
                        "shares_after": np.nan,
                        "ret_raw": ret_raw[t],
                        "ret_adj": ret_adj[t],
                        "note": f"adj_factor_ratio={af_ratio:.6g}; mismatch={mismatch:.4f}; {hint}",
                    }
                )
            missing_sessions = int(positions[t] - positions[t - 1] - 1)
            if missing_sessions > max_gap_sessions:
                records.append(
                    {
                        "ticker": ticker,
                        "date": dates[t],
                        "kind": "gap",
                        "shares_before": np.nan,
                        "shares_after": np.nan,
                        "ret_raw": ret_raw[t],
                        "ret_adj": ret_adj[t],
                        "note": f"{missing_sessions} sessions missing after {pd.Timestamp(dates[t - 1]).date()}",
                    }
                )
    out = pd.DataFrame(records, columns=CA_COLUMNS)
    out["ticker"] = out["ticker"].astype(str)
    out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
    out["kind"] = out["kind"].astype(str)
    out["note"] = out["note"].astype(str)
    for col in ("shares_before", "shares_after", "ret_raw", "ret_adj"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    out = out.sort_values(["ticker", "date", "kind"], kind="stable").reset_index(drop=True)
    log.info("corporate actions: %s", summarise_corporate_actions(out)["by_kind"])
    return out


def summarise_corporate_actions(ca_df: pd.DataFrame) -> dict[str, Any]:
    """Counts per kind, per class (capital changes), tickers with the most events and the largest jumps."""
    by_kind = {k: int((ca_df["kind"] == k).sum()) for k in KINDS}
    cap = ca_df[ca_df["kind"] == "capital_change"]
    classes: dict[str, int] = {}
    early = 0
    for note in cap["note"]:
        for part in str(note).split(";"):
            part = part.strip()
            if part.startswith("class="):
                classes[part[6:]] = classes.get(part[6:], 0) + 1
            elif part == "vendor_step_one_day_early":
                early += 1
    jumps = ca_df[ca_df["kind"] == "adj_jump"]
    return {
        "n_events": int(len(ca_df)),
        "by_kind": by_kind,
        "capital_change_classes": classes,
        "vendor_step_one_day_early": early,
        "tickers_with_events": int(ca_df["ticker"].nunique()),
        "top_tickers": ca_df["ticker"].value_counts().head(10).to_dict(),
        "largest_adj_jumps": [
            {"ticker": r.ticker, "date": str(pd.Timestamp(r.date).date()), "ret_raw": round(float(r.ret_raw), 4)}
            for r in jumps.reindex(jumps["ret_raw"].abs().sort_values(ascending=False).index).head(5).itertuples()
        ],
    }


def events_for(ca_df: pd.DataFrame, ticker: str, kinds: tuple[str, ...] = ("capital_change", "adj_jump")) -> list[date]:
    """Sorted event dates of one ticker for the given kinds (to exclude from return windows)."""
    sub = ca_df[(ca_df["ticker"] == ticker.upper()) & (ca_df["kind"].isin(kinds))]
    return sorted({pd.Timestamp(d).date() for d in sub["date"]})


# --------------------------------------------------------------------------- store I/O
def write_corporate_actions(ca_df: pd.DataFrame, store_dir: Path) -> Path:
    """Write ``<store_dir>/corporate_actions.parquet`` atomically and return its path."""
    missing = [c for c in CA_COLUMNS if c not in ca_df.columns]
    if missing:
        raise ValueError(f"corporate-actions frame is missing columns {missing}")
    return write_parquet(ca_df[CA_COLUMNS], Path(store_dir) / STORE_FILENAME)


def load_corporate_actions(store_dir: Path) -> pd.DataFrame:
    path = Path(store_dir) / STORE_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; build the price store first")
    return read_parquet(path)
