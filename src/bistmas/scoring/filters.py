"""Eligibility filters applied before ranking (all thresholds are pre-registered config values)."""

from __future__ import annotations

import numpy as np
import pandas as pd

REASONS = ("no_price", "listed", "liquidity", "negative_equity", "stale_statement", "no_statement", "no_composite")


def apply_filters(
    df: pd.DataFrame,
    min_listed_sessions: int = 250,
    adv_percentile_floor: float = 0.10,
    min_adv_tl: float = 0.0,
    max_statement_age_days: int = 400,
    require_positive_equity: bool = True,
) -> pd.DataFrame:
    """Add ``eligible`` and ``filter_reason`` (first failed test) columns."""
    out = df.copy()
    reason = pd.Series("", index=out.index, dtype=object)
    adv = pd.to_numeric(out["adv_6m_tl"], errors="coerce")
    floor = max(float(min_adv_tl), float(np.nanquantile(adv, adv_percentile_floor)) if adv.notna().any() else 0.0)
    tests = [
        ("no_price", out["close_raw"].isna()),
        ("listed", pd.to_numeric(out["listed_sessions"], errors="coerce").fillna(0) < min_listed_sessions),
        ("liquidity", adv.isna() | (adv < floor)),
        ("no_statement", out["statement_period_end"].isna()),
        ("stale_statement", pd.to_numeric(out["statement_age_days"], errors="coerce") > max_statement_age_days),
        ("negative_equity", require_positive_equity & (pd.to_numeric(out["book_to_market"], errors="coerce") <= 0)),
    ]
    for name, failed in tests:
        reason = reason.where(~(failed.fillna(False) & (reason == "")), name)
    out["filter_reason"] = reason.astype(str)
    out["eligible"] = out["filter_reason"] == ""
    out.attrs["adv_floor_tl"] = floor
    return out
