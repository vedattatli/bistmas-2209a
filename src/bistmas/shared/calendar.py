"""BIST trading calendar.

The calendar is *derived from data* (union of trading dates of the most liquid names, built in
M2) rather than from a holiday list, because Borsa İstanbul has half days, ad-hoc closures and
session-time changes. Two facts are hard-coded because vendors get them wrong:

* 2023-02-08: all trades were cancelled after the Kahramanmaraş earthquake; rows for that day
  must be dropped.
* 2023-02-09 .. 2023-02-14: the exchange was closed.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterable
from datetime import date, timedelta

import pandas as pd

CANCELLED_SESSIONS: frozenset[date] = frozenset({date(2023, 2, 8)})
CLOSURES: frozenset[date] = frozenset({date(2023, 2, 9), date(2023, 2, 10), date(2023, 2, 13), date(2023, 2, 14)})


class TradingCalendar:
    """Sorted list of trading sessions with next/previous lookups."""

    def __init__(self, sessions: Iterable[date]) -> None:
        cleaned = sorted({pd.Timestamp(s).date() for s in sessions} - CANCELLED_SESSIONS - CLOSURES)
        if not cleaned:
            raise ValueError("calendar needs at least one session")
        self._sessions: list[date] = cleaned

    @property
    def sessions(self) -> list[date]:
        return list(self._sessions)

    def is_session(self, d: date) -> bool:
        i = bisect_left(self._sessions, d)
        return i < len(self._sessions) and self._sessions[i] == d

    def next_session(self, d: date, strict: bool = True) -> date:
        """First session strictly after d (or on/after d when strict=False)."""
        i = bisect_right(self._sessions, d) if strict else bisect_left(self._sessions, d)
        if i >= len(self._sessions):
            raise ValueError(f"no session after {d}")
        return self._sessions[i]

    def prev_session(self, d: date, strict: bool = True) -> date:
        """Last session strictly before d (or on/before d when strict=False)."""
        i = bisect_left(self._sessions, d) if strict else bisect_right(self._sessions, d)
        if i == 0:
            raise ValueError(f"no session before {d}")
        return self._sessions[i - 1]

    def month_ends(self, start: date, end: date) -> list[date]:
        """Last session of every calendar month between start and end (inclusive)."""
        idx = pd.DatetimeIndex(self._sessions)
        idx = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
        return [ts.date() for ts in idx.to_series().groupby([idx.year, idx.month]).max()]

    def sessions_between(self, start: date, end: date) -> list[date]:
        lo = bisect_left(self._sessions, start)
        hi = bisect_right(self._sessions, end)
        return self._sessions[lo:hi]


# --------------------------------------------------------------------------- future sessions (rule based)
# Official Turkish public holidays on which Borsa İstanbul is closed, 2026-2027 (static list; verify against
# the exchange's annual holiday announcement before relying on a date more than a few months ahead).
# Religious holidays follow the Diyanet calendar: Ramazan Bayramı 2026-03-20..22 and 2027-03-09..11,
# Kurban Bayramı 2026-05-27..30 and 2027-05-17..19 (weekend days are irrelevant and omitted).
FUTURE_HOLIDAYS: frozenset[date] = frozenset(
    {
        # 2026
        date(2026, 1, 1),
        date(2026, 3, 20),
        date(2026, 4, 23),
        date(2026, 5, 1),
        date(2026, 5, 19),
        date(2026, 5, 27),
        date(2026, 5, 28),
        date(2026, 5, 29),
        date(2026, 7, 15),
        date(2026, 10, 29),
        # 2027
        date(2027, 1, 1),
        date(2027, 3, 9),
        date(2027, 3, 10),
        date(2027, 3, 11),
        date(2027, 4, 23),
        date(2027, 5, 17),
        date(2027, 5, 18),
        date(2027, 5, 19),
        date(2027, 7, 15),
        date(2027, 8, 30),
        date(2027, 10, 29),
    }
)
# Half-day sessions (arife): still sessions, listed for documentation only.
FUTURE_HALF_DAYS: frozenset[date] = frozenset(
    {
        date(2026, 3, 19),
        date(2026, 5, 26),
        date(2026, 10, 28),
        date(2027, 3, 8),
        date(2027, 10, 28),
    }
)
FUTURE_CALENDAR_END = date(2027, 12, 31)


def rule_sessions(start: date, end: date) -> list[date]:
    """Weekdays between start and end (inclusive) that are not in :data:`FUTURE_HOLIDAYS`."""
    days = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))
    return [d.date() for d in days if d.date() not in FUTURE_HOLIDAYS and d.date() not in CLOSURES]


def with_future_sessions(cal: TradingCalendar, until: date = FUTURE_CALENDAR_END) -> TradingCalendar:
    """Data-derived sessions up to the last observed one, holiday-aware weekday rule afterwards.

    Used only to look *forward* (planned execution session, whether a month has ended); every date up to
    the last data session still comes from the data, never from the rule.
    """
    sessions = cal.sessions
    last = sessions[-1]
    if until <= last:
        return cal
    return TradingCalendar(sessions + rule_sessions(last + timedelta(days=1), until))
