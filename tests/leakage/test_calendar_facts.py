"""Leakage suite placeholder: hard facts every later test relies on."""

from datetime import date

from bistmas.shared.calendar import CANCELLED_SESSIONS, CLOSURES


def test_earthquake_closure_is_hard_coded():
    assert date(2023, 2, 8) in CANCELLED_SESSIONS
    assert {date(2023, 2, 9), date(2023, 2, 14)} <= CLOSURES
