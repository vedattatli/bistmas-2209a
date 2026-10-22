"""Point-in-time rules: when a filing became usable, and conservative fallbacks when the date is unknown.

* :func:`availability_from_publish` — a filing published before the session cut-off (09:55, BIST opens
  at 10:00) is usable on that session; anything later is usable from the next session.
* :func:`fallback_availability` — when the true publication timestamp is unknown, a lag chosen above the
  empirical 95th percentile of KAP publication lags (2019 sample: FY p95 88 d, Q1 p95 43 d) plus the
  regulatory extensions (COVID for 2019Q4/2020Q1; TMS-29 inflation-accounting transition for FY2023 and
  2024Q1/Q2). Using the fallback can only make information available LATER than it really was, so it
  never leaks the future.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pandas as pd

from bistmas.shared.calendar import TradingCalendar

SESSION_CUTOFF = time(9, 55)

#: days after the period end, by period (1..4) and template group; H1 (period 2) carries the audit extension
FALLBACK_LAG_DAYS = {
    "industrial": {1: 45, 2: 55, 3: 45, 4: 90},
    "insurer": {1: 45, 2: 55, 3: 45, 4: 90},
    "bank": {1: 55, 2: 65, 3: 55, 4: 90},
    "other_financial": {1: 55, 2: 65, 3: 55, 4: 90},
}
#: extra days for specific period ends (regulatory extensions)
EXTRA_LAG_DAYS = {
    date(2019, 12, 31): 30,  # COVID: SPK decision 19.03.2020 (+30 d)
    date(2020, 3, 31): 30,  # COVID
    date(2023, 12, 31): 70,  # TMS-29 first annual application (+10 weeks)
    date(2024, 3, 31): 42,  # TMS-29 (+6 weeks)
    date(2024, 6, 30): 42,  # TMS-29 (+6 weeks)
}


def period_end_date(fiscal_year: int, period: int, fiscal_year_end_month: int = 12) -> date:
    """Last calendar day of fiscal ``period`` (1..4) of ``fiscal_year`` for a fiscal year ending in that month."""
    if period not in (1, 2, 3, 4):
        raise ValueError(f"period must be 1..4, got {period}")
    end_month = fiscal_year_end_month
    # quarter ends counted backwards from the fiscal year end
    month = (end_month - 3 * (4 - period) - 1) % 12 + 1
    year = fiscal_year if month <= end_month or fiscal_year_end_month == 12 else fiscal_year - 1
    if fiscal_year_end_month != 12 and month > end_month:
        year = fiscal_year - 1
    first_next = date(year + (month // 12), month % 12 + 1, 1)
    return first_next - timedelta(days=1)


def availability_from_publish(publish_ts: datetime | pd.Timestamp, calendar: TradingCalendar) -> date:
    """Session on which a filing published at ``publish_ts`` (local time) could first be used."""
    ts = pd.Timestamp(publish_ts)
    if pd.isna(ts):
        raise ValueError("publish_ts is NaT")
    day = ts.date()
    if ts.time() < SESSION_CUTOFF and calendar.is_session(day):
        return day
    return calendar.next_session(day, strict=True)


def fallback_availability(
    period_end: date,
    period: int,
    template: str,
    calendar: TradingCalendar | None = None,
) -> date:
    """Conservative availability date for a statement whose KAP publication timestamp is unknown."""
    lags = FALLBACK_LAG_DAYS.get(template, FALLBACK_LAG_DAYS["industrial"])
    d = period_end + timedelta(days=lags[period] + EXTRA_LAG_DAYS.get(period_end, 0))
    if calendar is not None:
        try:
            return calendar.next_session(d, strict=False)
        except ValueError:
            return d
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d
