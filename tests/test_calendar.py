from datetime import date

import pandas as pd
import pytest

from bistmas.shared.calendar import CLOSURES, TradingCalendar


def _weekdays(start: str, end: str) -> list[date]:
    return [d.date() for d in pd.bdate_range(start, end)]


def test_closures_and_cancelled_session_are_removed():
    cal = TradingCalendar(_weekdays("2023-02-01", "2023-02-28"))
    assert not cal.is_session(date(2023, 2, 8))
    for d in CLOSURES:
        assert not cal.is_session(d)
    assert cal.next_session(date(2023, 2, 7)) == date(2023, 2, 15)


def test_next_prev_and_month_ends():
    cal = TradingCalendar(_weekdays("2018-01-01", "2018-03-31"))
    assert cal.next_session(date(2018, 1, 31)) == date(2018, 2, 1)
    assert cal.prev_session(date(2018, 2, 1)) == date(2018, 1, 31)
    assert cal.next_session(date(2018, 2, 3), strict=False) == date(2018, 2, 5)  # Saturday -> Monday
    assert cal.month_ends(date(2018, 1, 1), date(2018, 3, 31)) == [
        date(2018, 1, 31),
        date(2018, 2, 28),
        date(2018, 3, 30),
    ]


def test_bounds_raise():
    cal = TradingCalendar(_weekdays("2018-01-01", "2018-01-10"))
    with pytest.raises(ValueError):
        cal.next_session(date(2018, 1, 10))
    with pytest.raises(ValueError):
        cal.prev_session(date(2018, 1, 1))
