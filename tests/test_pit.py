from datetime import date, datetime

import pandas as pd
import pytest

from bistmas.data.pit import availability_from_publish, fallback_availability, period_end_date
from bistmas.shared.calendar import TradingCalendar


def _cal():
    return TradingCalendar([d.date() for d in pd.bdate_range("2019-01-01", "2020-12-31")])


def test_period_end_dates():
    assert period_end_date(2018, 1) == date(2018, 3, 31)
    assert period_end_date(2018, 2) == date(2018, 6, 30)
    assert period_end_date(2018, 3) == date(2018, 9, 30)
    assert period_end_date(2018, 4) == date(2018, 12, 31)
    with pytest.raises(ValueError):
        period_end_date(2018, 5)


def test_availability_before_and_after_cutoff():
    cal = _cal()
    assert availability_from_publish(datetime(2019, 3, 5, 8, 9, 36), cal) == date(2019, 3, 5)  # before 09:55
    assert availability_from_publish(datetime(2019, 3, 5, 18, 30), cal) == date(2019, 3, 6)  # after the close
    assert availability_from_publish(datetime(2019, 3, 8, 23, 59), cal) == date(2019, 3, 11)  # Friday night -> Monday
    assert availability_from_publish(datetime(2019, 3, 9, 8, 0), cal) == date(2019, 3, 11)  # Saturday morning -> Monday


def test_fallback_is_conservative_and_rolls_to_sessions():
    cal = _cal()
    assert fallback_availability(date(2019, 3, 31), 1, "industrial", cal) == date(2019, 5, 15)  # +45 d
    assert fallback_availability(date(2019, 3, 31), 1, "bank", cal) == date(
        2019, 5, 27
    )  # +55 d -> Saturday 25 -> Monday 27
    assert fallback_availability(date(2018, 12, 31), 4, "industrial", cal) == date(
        2019, 4, 1
    )  # +90 d = Sunday 31 Mar -> Monday
    assert fallback_availability(date(2019, 12, 31), 4, "industrial", cal) == date(2020, 4, 29)  # +90 +30 COVID
    assert fallback_availability(date(2023, 12, 31), 4, "industrial") == date(2024, 6, 10)  # +90 +70 TMS-29 -> weekday
