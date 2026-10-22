"""Data-derived BIST trading calendar.

Borsa İstanbul has half days, ad-hoc closures and session-time changes, so the calendar is not a
holiday list: it is the **union of the session dates of the most liquid names** in the price panel,
minus the hard-coded facts in :mod:`bistmas.shared.calendar` (the cancelled 2023-02-08 session and the
2023-02-09..14 earthquake closure). Liquidity is total ``traded_value_tl`` over the panel; the top-N
names (default 30) are traded every session, so their union is the full session list without illiquid
names' gaps and without any single name's suspension days.

Output: ``data/store/trading_calendar.parquet`` with one ``date`` column (``datetime64[ns]``, sorted,
unique). :func:`load_calendar` turns it into :class:`bistmas.shared.calendar.TradingCalendar`.

:func:`calendar_anomalies` reports what a reviewer should look at: gaps longer than ``max_gap_weekdays``
weekdays (only the February-2023 closure is expected), sessions on a weekend, months whose last session
is more than two weekdays before the calendar month end, and *half-day candidates*: sessions whose
cross-sectional traded value of the top-N names is below ``half_day_ratio`` of the trailing median
(BIST closes at noon on the eve of religious holidays, and the cancelled 2023-02-08 shows up the same
way if it were kept).
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bistmas.shared.calendar import CANCELLED_SESSIONS, CLOSURES, TradingCalendar
from bistmas.shared.log import get_logger
from bistmas.shared.store import read_parquet, write_parquet

log = get_logger("data.calendar_build")

STORE_FILENAME = "trading_calendar.parquet"
CALENDAR_COLUMNS: list[str] = ["date"]
DEFAULT_TOP_N = 30
MAX_GAP_WEEKDAYS = 4
HALF_DAY_RATIO = 0.4
HALF_DAY_WINDOW = 20


def _liquidity_rank(prices_df: pd.DataFrame) -> pd.Series:
    """Total traded value per ticker (falls back to ``volume_raw * vwap_raw`` when the column is absent)."""
    if "traded_value_tl" in prices_df.columns:
        value = prices_df["traded_value_tl"]
    elif {"volume_raw", "vwap_raw"} <= set(prices_df.columns):
        value = prices_df["volume_raw"] * prices_df["vwap_raw"]
    else:
        raise ValueError("prices frame needs traded_value_tl (or volume_raw and vwap_raw)")
    return value.groupby(prices_df["ticker"]).sum().sort_values(ascending=False)


def top_liquid_tickers(prices_df: pd.DataFrame, top_n: int = DEFAULT_TOP_N) -> list[str]:
    """Tickers ranked by total traded value, first ``top_n`` (all tickers when fewer exist)."""
    ranked = _liquidity_rank(prices_df)
    return [str(t) for t in ranked.index[:top_n]]


def build_trading_calendar(prices_df: pd.DataFrame, top_n: int = DEFAULT_TOP_N) -> pd.DataFrame:
    """Union of the session dates of the ``top_n`` most liquid tickers, minus cancelled sessions and closures.

    Rows flagged ``flag_cancelled_session`` are ignored even before the explicit removal. Returns a
    frame with one sorted, unique ``date`` column (``datetime64[ns]``).
    """
    if prices_df.empty:
        raise ValueError("cannot build a calendar from an empty price panel")
    df = prices_df
    if "flag_cancelled_session" in df.columns:
        df = df[~df["flag_cancelled_session"].astype(bool)]
    tickers = top_liquid_tickers(df, top_n)
    dates = pd.to_datetime(df.loc[df["ticker"].isin(tickers), "date"]).dt.normalize()
    excluded = {pd.Timestamp(d) for d in CANCELLED_SESSIONS | CLOSURES}
    sessions = sorted(set(dates) - excluded)
    out = pd.DataFrame({"date": pd.to_datetime(sessions).astype("datetime64[ns]")})
    log.info(
        "trading calendar: %d sessions %s..%s from %d tickers (%s ...)",
        len(out),
        out["date"].min().date() if len(out) else None,
        out["date"].max().date() if len(out) else None,
        len(tickers),
        ", ".join(tickers[:5]),
    )
    return out


def _weekdays_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    """Number of weekdays strictly between a and b."""
    return int(np.busday_count(a.date() + timedelta(days=1), b.date()))


def calendar_anomalies(
    calendar_df: pd.DataFrame,
    prices_df: pd.DataFrame | None = None,
    top_n: int = DEFAULT_TOP_N,
    max_gap_weekdays: int = MAX_GAP_WEEKDAYS,
    half_day_ratio: float = HALF_DAY_RATIO,
) -> dict[str, Any]:
    """Reviewer report for a calendar (see module docstring); every list holds ISO date strings."""
    dates = pd.to_datetime(calendar_df["date"]).sort_values().reset_index(drop=True)
    report: dict[str, Any] = {
        "n_sessions": int(len(dates)),
        "first": str(dates.iloc[0].date()) if len(dates) else None,
        "last": str(dates.iloc[-1].date()) if len(dates) else None,
        "gaps": [],
        "weekend_sessions": [str(d.date()) for d in dates if d.weekday() >= 5],
        "month_end_anomalies": [],
        "half_day_candidates": [],
    }
    for prev, cur in zip(dates[:-1], dates[1:], strict=True):
        gap = _weekdays_between(prev, cur)
        if gap > max_gap_weekdays:
            expected = all(prev < pd.Timestamp(c) < cur for c in CLOSURES)
            report["gaps"].append(
                {
                    "from": str(prev.date()),
                    "to": str(cur.date()),
                    "missing_weekdays": gap,
                    "expected": bool(expected),
                }
            )
    if len(dates):
        by_month = dates.groupby([dates.dt.year, dates.dt.month]).max()
        for (y, m), last in by_month.items():
            month_end = (pd.Timestamp(year=int(y), month=int(m), day=1) + pd.offsets.MonthEnd(0)).normalize()
            if month_end > max(dates):
                continue  # the panel ends inside this month: not an anomaly
            if last.date() < month_end.date() and _weekdays_between(last, month_end + pd.Timedelta(days=1)) > 2:
                report["month_end_anomalies"].append(
                    {
                        "month": f"{int(y)}-{int(m):02d}",
                        "last_session": str(last.date()),
                        "month_end": str(month_end.date()),
                    }
                )
    if prices_df is not None and len(prices_df):
        tickers = top_liquid_tickers(prices_df, top_n)
        sub = prices_df[prices_df["ticker"].isin(tickers)]
        value = sub["traded_value_tl"] if "traded_value_tl" in sub.columns else sub["volume_raw"] * sub["vwap_raw"]
        daily = value.groupby(pd.to_datetime(sub["date"]).dt.normalize()).sum().sort_index()
        daily = daily[daily.index.isin(set(dates))]
        trailing = daily.rolling(HALF_DAY_WINDOW, min_periods=5).median().shift(1)
        ratio = daily / trailing
        low = ratio[(ratio < half_day_ratio) & trailing.notna()]
        report["half_day_candidates"] = [{"date": str(d.date()), "ratio": round(float(r), 3)} for d, r in low.items()]
    return report


# --------------------------------------------------------------------------- store I/O
def write_calendar(calendar_df: pd.DataFrame, store_dir: Path) -> Path:
    """Write ``<store_dir>/trading_calendar.parquet`` atomically and return its path."""
    if "date" not in calendar_df.columns:
        raise ValueError("calendar frame needs a 'date' column")
    out = pd.DataFrame(
        {"date": pd.to_datetime(calendar_df["date"], format="mixed").dt.normalize().astype("datetime64[ns]")}
    )
    out = out.drop_duplicates().sort_values("date").reset_index(drop=True)
    return write_parquet(out, Path(store_dir) / STORE_FILENAME)


def load_calendar_frame(store_dir: Path) -> pd.DataFrame:
    path = Path(store_dir) / STORE_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; build the price store first")
    return read_parquet(path)


def load_calendar(store_dir: Path) -> TradingCalendar:
    """Load ``<store_dir>/trading_calendar.parquet`` into a :class:`TradingCalendar`."""
    df = load_calendar_frame(store_dir)
    return calendar_from_frame(df)


def calendar_from_frame(calendar_df: pd.DataFrame) -> TradingCalendar:
    sessions: list[date] = [pd.Timestamp(d).date() for d in calendar_df["date"]]
    return TradingCalendar(sessions)


def build_and_write_calendar(prices_df: pd.DataFrame, store_dir: Path, top_n: int = DEFAULT_TOP_N) -> pd.DataFrame:
    """Build the calendar from the price panel, log its anomaly report and write it to the store."""
    cal = build_trading_calendar(prices_df, top_n)
    report = calendar_anomalies(cal, prices_df, top_n)
    unexpected = [g for g in report["gaps"] if not g["expected"]]
    if unexpected:
        log.warning("calendar: %d unexpected gaps > %d weekdays: %s", len(unexpected), MAX_GAP_WEEKDAYS, unexpected[:5])
    if report["weekend_sessions"]:
        log.warning("calendar: weekend sessions %s", report["weekend_sessions"][:5])
    if report["half_day_candidates"]:
        log.info("calendar: %d half-day candidates (low traded value)", len(report["half_day_candidates"]))
    write_calendar(cal, store_dir)
    cal.attrs["report"] = report
    return cal
