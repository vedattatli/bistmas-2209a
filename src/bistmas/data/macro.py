"""Daily macro table (``data/store/macro_daily.parquet``): CPI, USD/TRY and the policy rate on BIST sessions.

Columns (one row per trading session, sorted by ``date``):

``date``
    Session (``datetime64[ns]``, tz-naive midnight). Sessions come from the data-derived trading
    calendar (``trading_calendar.parquet``); 2023-02-08 and the earthquake closure are never rows.
``cpi``
    TÜİK consumer price index (EVDS ``MacroConfig.cpi_series`` = ``TP.FG.J0``, 2003=100; months after that series
    stops are filled from ``cpi_continuation_series`` rescaled by the median old/new ratio over the overlap) of the
    **calendar month containing the session** (the monthly value is spread over every session of its month).
    This is the deflator for
    ex-post real (CPI-deflated) performance and must *not* be used as a feature as-is: it is only known
    after the month ends.
``cpi_available_from``
    First session on/after the 3rd calendar day of the following month, i.e. when ``cpi`` became public
    (TÜİK releases CPI at 10:00 on the 3rd, before the 18:00 decision cut-off; a weekend 3rd rolls to the
    next session). Point-in-time code lags ``cpi`` with this column.
``cpi_pit``
    Latest CPI value whose ``cpi_available_from`` is on/before the session (the value a decision maker
    could see): the previous month's index from the 3rd onward, the month-before's before that. ``NaN`` when
    that print is older than ``cpi_stale_days`` (an overdue release is never silently carried forward).
``usdtry_evds``
    TCMB buying rate (EVDS ``TP.DK.USD.A.YTL``) *announced* on the session (the 15:30 indicative rate of that
    session). EVDS dates this series by validity day, i.e. the value dated D is the fixing of the previous
    business day (verified on the raw 2018 file: 2018-08-13 = 5.9389 is the 10-Aug fixing, the 13-Aug crash level
    appears on 2018-08-14), so :func:`fx_to_sessions` moves every observation back one business day.
    Forward-filled over gaps of at most ``FX_MAX_FFILL_DAYS`` calendar days. ``NaN`` when EVDS is unavailable:
    ``prices_daily.usdtry``
    (İş Yatırım ``DD_DEGER``) is the documented fallback for the USD benchmark.
``policy_rate``
    CBRT 1-week repo rate in percent in force on the session (effective-dated step series from
    :mod:`bistmas.data.clients.tcmb_rates`).
``rf_daily``
    ``policy_rate / 100 / 365`` (simple ACT/365 rate for one calendar day; see ``tcmb_rates``).

Without an EVDS key the table is still built (``policy_rate``/``rf_daily`` come from the TCMB HTML page)
and the EVDS columns are ``NaN`` with a warning that explains how to register; every EVDS payload that
was fetched once is reused from ``data/raw/evds`` without a key.
"""

from __future__ import annotations

import calendar as _calendar
from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bistmas.data.clients.evds import USDTRY_BUY, EvdsClient, MissingEvdsKeyError
from bistmas.data.clients.http import PoliteSession
from bistmas.data.clients.tcmb_rates import (
    coerce_policy_frame,
    fetch_policy_rate_table,
    overnight_rate_daily,
    policy_rate_on,
)
from bistmas.shared.calendar import TradingCalendar
from bistmas.shared.config import MacroConfig, Settings
from bistmas.shared.log import get_logger
from bistmas.shared.store import read_parquet, utc_now_iso, write_json, write_parquet

log = get_logger("data.macro")

MACRO_FILENAME = "macro_daily.parquet"
MACRO_META_FILENAME = "macro_meta.json"
CALENDAR_FILENAME = "trading_calendar.parquet"
MACRO_COLUMNS: list[str] = [
    "date",
    "cpi",
    "cpi_available_from",
    "cpi_pit",
    "usdtry_evds",
    "policy_rate",
    "rf_daily",
]
CPI_RELEASE_DAY = 3  # TÜİK publishes the CPI of month M on the 3rd of month M+1 (10:00)
CPI_LOOKBACK_MONTHS = 3  # fetch CPI from this many months before the first session so cpi_pit is defined
FX_MAX_FFILL_DAYS = 5


# --------------------------------------------------------------------------- helpers
def _sessions_index(dates: Iterable[date | pd.Timestamp | str] | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Sorted, unique, tz-naive session index with the hard-coded cancellations/closures removed."""
    values = list(dates)
    if not values:
        return pd.DatetimeIndex([], dtype="datetime64[ns]", name="date")
    cal = TradingCalendar(pd.to_datetime(values))
    return pd.DatetimeIndex(pd.to_datetime(cal.sessions).astype("datetime64[ns]"), name="date")


def _month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts.year, ts.month, 1)


def _add_months(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    y, m = divmod(ts.month - 1 + months, 12)
    year, month = ts.year + y, m + 1
    day = min(ts.day, _calendar.monthrange(year, month)[1])
    return pd.Timestamp(year, month, day)


def cpi_release_date(month: date | pd.Timestamp | str, sessions: pd.DatetimeIndex | None = None) -> pd.Timestamp:
    """Date from which the CPI of ``month`` is public: the 3rd of the next month, rolled to a session.

    With ``sessions`` the first session on/after the 3rd is returned (or the plain calendar date when
    the sessions end before it); without, a Saturday/Sunday rolls to Monday.
    """
    m = _month_start(pd.Timestamp(month))
    release = _add_months(m, 1).replace(day=CPI_RELEASE_DAY)
    if sessions is not None and len(sessions):
        pos = sessions.searchsorted(release, side="left")
        if pos < len(sessions):
            return pd.Timestamp(sessions[pos])
    while release.weekday() >= 5:
        release += timedelta(days=1)
    return release


def splice_cpi(old: pd.DataFrame | None, new: pd.DataFrame | None) -> tuple[pd.DataFrame | None, float | None]:
    """Extend the monthly ``old`` index past its last month with ``new`` rescaled by the overlap ratio.

    Both frames are ``[date, value]``. The ratio old/new is the median over months present in both (it is
    constant for a pure re-basing); without an overlap nothing is spliced (no guessing). Returns the
    spliced frame and the ratio.
    """
    if old is None or len(old) == 0 or new is None or len(new) == 0:
        return old, None

    def _m(df: pd.DataFrame) -> pd.Series:
        f = df.loc[:, ["date", "value"]].copy()
        f["month"] = pd.to_datetime(f["date"]).dt.to_period("M").dt.to_timestamp()
        f["value"] = pd.to_numeric(f["value"], errors="coerce")
        return f.dropna().drop_duplicates("month", keep="last").set_index("month")["value"].sort_index()

    o, n = _m(old), _m(new)
    both = o.index.intersection(n.index)
    if len(both) == 0:
        return old, None
    ratio = float((o.loc[both] / n.loc[both]).median())
    extra = n[n.index > o.index.max()] * ratio
    if extra.empty:
        return old, ratio
    add = pd.DataFrame({"date": extra.index, "value": extra.to_numpy()})
    return pd.concat([old.loc[:, ["date", "value"]], add], ignore_index=True), ratio


def cpi_to_sessions(
    cpi_monthly: pd.DataFrame | None, sessions: pd.DatetimeIndex, stale_days: int | None = None
) -> pd.DataFrame:
    """Spread the monthly CPI over sessions: ``[date, cpi, cpi_available_from, cpi_pit]``.

    ``cpi_monthly`` has ``[date, value]`` with ``date`` = any day of the month (EVDS gives the 1st).
    ``stale_days`` (default ``MacroConfig.cpi_stale_days``) is how long a print stays "latest" in ``cpi_pit``.
    """
    stale_days = MacroConfig().cpi_stale_days if stale_days is None else stale_days
    idx = _sessions_index(sessions)
    out = pd.DataFrame({"date": idx})
    out["cpi"] = np.nan
    out["cpi_available_from"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    out["cpi_pit"] = np.nan
    if cpi_monthly is None or len(cpi_monthly) == 0 or len(idx) == 0:
        return out
    monthly = cpi_monthly.loc[:, ["date", "value"]].copy()
    monthly["month"] = pd.to_datetime(monthly["date"]).dt.to_period("M").dt.to_timestamp()
    monthly["value"] = pd.to_numeric(monthly["value"], errors="coerce")
    monthly = monthly.dropna(subset=["value"]).drop_duplicates("month", keep="last").sort_values("month")
    monthly["available_from"] = [cpi_release_date(m, idx) for m in monthly["month"]]

    session_month = idx.to_period("M").to_timestamp()
    by_month = monthly.set_index("month")
    out["cpi"] = by_month["value"].reindex(session_month).to_numpy(dtype=float)
    avail = by_month["available_from"].reindex(session_month)
    out["cpi_available_from"] = pd.to_datetime(avail.to_numpy()).astype("datetime64[ns]")

    pit = monthly.sort_values(["available_from", "month"])[["available_from", "value"]]
    pit = pit.drop_duplicates("available_from", keep="last")
    merged = pd.merge_asof(
        out[["date"]],
        pit,
        left_on="date",
        right_on="available_from",
        direction="backward",
        tolerance=pd.Timedelta(days=stale_days),
    )
    out["cpi_pit"] = merged["value"].to_numpy(dtype=float)
    return out


def fx_to_sessions(
    fx_daily: pd.DataFrame | None, sessions: pd.DatetimeIndex, max_ffill_days: int = FX_MAX_FFILL_DAYS
) -> pd.Series:
    """Daily EVDS FX (``[date, value]``, dated by validity day) on the session whose 15:30 fixing it is.

    Each observation is moved back one business day (see the module docstring), then forward-filled onto sessions
    over gaps of at most ``max_ffill_days`` calendar days.
    """
    idx = _sessions_index(sessions)
    if fx_daily is None or len(fx_daily) == 0 or len(idx) == 0:
        return pd.Series(np.nan, index=idx, name="usdtry_evds", dtype=float)
    fx = fx_daily.loc[:, ["date", "value"]].copy()
    fx["date"] = pd.to_datetime(fx["date"]).dt.normalize().astype("datetime64[ns]")
    fx["value"] = pd.to_numeric(fx["value"], errors="coerce")
    # EVDS dates the TCMB indicative rate by validity day: the value dated D is the previous business day's fixing
    fx["date"] = fx["date"] - pd.offsets.BDay(1)
    fx = fx.dropna().sort_values("date").drop_duplicates("date", keep="last")
    merged = pd.merge_asof(
        pd.DataFrame({"date": idx}),
        fx.rename(columns={"date": "fx_date"}),
        left_on="date",
        right_on="fx_date",
        direction="backward",
        tolerance=pd.Timedelta(days=max_ffill_days),
    )
    return pd.Series(merged["value"].to_numpy(dtype=float), index=idx, name="usdtry_evds")


def build_macro_frame(
    sessions: Iterable[date | pd.Timestamp | str] | pd.DatetimeIndex,
    cpi_monthly: pd.DataFrame | None = None,
    usdtry_daily: pd.DataFrame | None = None,
    policy: pd.DataFrame | None = None,
    cpi_stale_days: int | None = None,
) -> pd.DataFrame:
    """Pure assembly of :data:`MACRO_COLUMNS` from the three inputs (any of which may be ``None``)."""
    idx = _sessions_index(sessions)
    out = cpi_to_sessions(cpi_monthly, idx, cpi_stale_days)
    out["usdtry_evds"] = fx_to_sessions(usdtry_daily, idx).to_numpy()
    pol = coerce_policy_frame(policy)
    out["policy_rate"] = policy_rate_on(pol, idx).to_numpy()
    out["rf_daily"] = overnight_rate_daily(pol, idx).to_numpy()
    return out.loc[:, MACRO_COLUMNS].reset_index(drop=True)


# --------------------------------------------------------------------------- store
def calendar_dates_from_store(store_dir: Path) -> list[date]:
    """Sessions from ``<store_dir>/trading_calendar.parquet`` (column ``date``, written by the price layer)."""
    path = Path(store_dir) / CALENDAR_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; build the price layer (bistmas fetch prices) first")
    df = read_parquet(path, columns=["date"])
    return [pd.Timestamp(d).date() for d in pd.to_datetime(df["date"]).sort_values().unique()]


def build_macro_store(
    settings: Settings,
    calendar_dates: Iterable[date | pd.Timestamp | str],
    session: PoliteSession | None = None,
    evds_client: EvdsClient | None = None,
    policy: pd.DataFrame | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch (cached) EVDS CPI + USD/TRY and the TCMB policy table, build and write ``macro_daily.parquet``.

    ``calendar_dates`` are the trading sessions (see :func:`calendar_dates_from_store`). ``session``,
    ``evds_client`` and ``policy`` can be injected for tests; ``force`` re-fetches every raw payload.
    Returns the frame that was written; ``macro_meta.json`` records what was available.
    """
    idx = _sessions_index(calendar_dates)
    if len(idx) == 0:
        raise ValueError("calendar_dates is empty")
    http = session or PoliteSession(min_interval_s=1.0, timeout_s=60.0)
    client = evds_client or EvdsClient(http, settings.paths.raw)
    first, last = idx[0], idx[-1]
    evds_start = _add_months(_month_start(first), -CPI_LOOKBACK_MONTHS)
    meta: dict[str, Any] = {
        "built_at": utc_now_iso(),
        "sessions": int(len(idx)),
        "first_session": str(first.date()),
        "last_session": str(last.date()),
        "evds_key_configured": client.has_key,
        "sources": {},
    }

    cpi_monthly: pd.DataFrame | None = None
    fx_daily: pd.DataFrame | None = None
    mc = settings.macro
    fetch_plan = [("cpi", mc.cpi_series, "cpi_monthly"), ("usdtry", USDTRY_BUY, "fx_daily")]
    if mc.cpi_continuation_series:
        fetch_plan.append(("cpi_continuation", mc.cpi_continuation_series, "cpi_cont"))
    cpi_cont: pd.DataFrame | None = None
    for name, code, target in fetch_plan:
        try:
            frame = client.fetch_series_by_year(code, evds_start, last, force=force)
        except MissingEvdsKeyError as exc:
            log.warning("%s: %s", name, exc)
            meta["sources"][name] = {"series": code, "status": "missing_key", "rows": 0}
            continue
        except Exception as exc:  # noqa: BLE001 - the optional continuation series never breaks the build
            if target != "cpi_cont":
                raise
            log.warning("optional CPI continuation %s not available: %s", code, exc)
            meta["sources"][name] = {"series": code, "status": "unavailable", "rows": 0}
            continue
        meta["sources"][name] = {"series": code, "status": "ok", "rows": int(len(frame))}
        if target == "cpi_monthly":
            cpi_monthly = frame
        elif target == "cpi_cont":
            cpi_cont = frame
        else:
            fx_daily = frame

    if cpi_cont is not None and len(cpi_cont):
        cpi_monthly, ratio = splice_cpi(cpi_monthly, cpi_cont)
        meta["sources"]["cpi_continuation"]["splice_ratio"] = ratio
        if ratio is None:
            log.warning(
                "CPI continuation %s has no overlap with %s: not spliced", mc.cpi_continuation_series, mc.cpi_series
            )

    if policy is None:
        policy = fetch_policy_rate_table(http, settings.paths.raw, force=force)
        meta["sources"]["policy_rate"] = {"url": "tcmb_1_week_repo_html", "rows": int(len(policy))}
    else:
        meta["sources"]["policy_rate"] = {"url": "injected", "rows": int(len(policy))}

    frame = build_macro_frame(idx, cpi_monthly, fx_daily, policy, mc.cpi_stale_days)
    meta["coverage"] = {c: float(frame[c].notna().mean()) for c in MACRO_COLUMNS if c != "date"}
    if cpi_monthly is not None and len(cpi_monthly):
        last_month = pd.to_datetime(cpi_monthly["date"]).max()
        overdue = (last - cpi_release_date(last_month, idx)).days
        meta["cpi_last_month"] = str(last_month.date())
        meta["cpi_days_since_last_release"] = int(overdue)
        if overdue > mc.cpi_stale_days:
            log.warning(
                "CPI series %s: last observation %s, %d days before the last session (> %d): cpi_pit is NaN after "
                "the print goes stale. Set macro.cpi_continuation_series to the re-based successor.",
                mc.cpi_series,
                last_month.date(),
                overdue,
                mc.cpi_stale_days,
            )
    store = settings.paths.store
    write_parquet(frame, store / MACRO_FILENAME)
    write_json(meta, store / MACRO_META_FILENAME)
    log.info("macro_daily: %d sessions, coverage %s", len(frame), {k: round(v, 3) for k, v in meta["coverage"].items()})
    return frame


def load_macro(store_dir: Path) -> pd.DataFrame:
    """Read ``<store_dir>/macro_daily.parquet`` (raises FileNotFoundError when it was never built)."""
    path = Path(store_dir) / MACRO_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `bistmas fetch macro` first")
    return read_parquet(path)
