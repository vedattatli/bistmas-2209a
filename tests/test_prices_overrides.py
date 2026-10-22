"""Late SERMAYE alignment and vendor adjustment overrides (offline, synthetic)."""

from __future__ import annotations

import pandas as pd
import pytest

from bistmas.data import prices


def _synthetic(close_raw, close_adj, shares) -> pd.DataFrame:
    n = len(close_raw)
    df = pd.DataFrame(
        {
            "ticker": "TST",
            "date": pd.bdate_range("2020-01-06", periods=n),
            "close_raw": close_raw,
            "close_adj": close_adj,
            "shares": shares,
        }
    )
    for c in ("vwap_raw", "low_raw", "high_raw"):
        df[c] = df["close_raw"]
    for c in ("vwap_adj", "low_adj", "high_adj"):
        df[c] = df["close_adj"]
    for c in ("volume_raw", "volume_adj", "mcap_tl", "ffloat_mcap_tl", "mcap_usd", "ffloat_mcap_usd"):
        df[c] = 1.0
    for c in ("usdtry", "xu100", "price_usd", "volume_usd"):
        df[c] = 1.0
    return df


def test_align_shares_handles_vendor_update_one_session_late():
    # 2:1 bonus: price ex-date is index 3 (raw halves, adjusted flat) but the vendor raises SERMAYE a day late
    df = _synthetic([10, 10, 10, 5, 5, 5], [5, 5, 5, 5, 5, 5], [100, 100, 100, 100, 200, 200])
    aligned, shifts = prices.align_shares(df)
    assert list(aligned) == [100, 100, 100, 200, 200, 200]
    assert shifts[0]["ex_date"] == df["date"].iloc[3]


def test_align_shares_late_rule_ignores_cash_dividend():
    # 6 % dividend mismatch on index 3, unrelated 1.01x capital step on index 4: no shift
    df = _synthetic([10, 10, 10, 9.4, 9.4, 9.4], [5, 5, 5, 5, 5, 5], [100, 100, 100, 100, 101, 101])
    aligned, shifts = prices.align_shares(df)
    assert list(aligned) == [100, 100, 100, 100, 101, 101] and not shifts


def test_adjustment_override_rescales_history_to_the_correct_step():
    # 1.5x bonus but the vendor adjustment factor steps by 2.25 on index 3
    close_raw = [6.0, 6.0, 6.0, 4.0, 4.0]
    close_adj = [1.0, 1.0, 1.0, 1.5, 1.5]  # factor 1/6 -> 3/8: step 2.25
    df = _synthetic(close_raw, close_adj, [60, 60, 60, 90, 90])
    ov = pd.DataFrame({"ticker": ["TST"], "ex_date": [df["date"].iloc[3]], "correct_factor_step": [1.5]})
    out = prices.build_prices_frame(df, adjustment_overrides=ov)
    assert out["adj_factor"].iloc[3] / out["adj_factor"].iloc[2] == pytest.approx(1.5)
    assert out["close_adj"].iloc[3:].tolist() == pytest.approx([1.5, 1.5])
    assert out.attrs["adjustment_overrides"][0]["multiplier"] == pytest.approx(2.25 / 1.5)


def test_shipped_adjustment_overrides_load():
    ov = prices.load_adjustment_overrides()
    assert (ov["ticker"] == "LRSHO").any() and float(ov["correct_factor_step"].iloc[0]) == 1.5
