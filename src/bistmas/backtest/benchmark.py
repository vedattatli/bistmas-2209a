"""Benchmarks: equal weight of the point-in-time universe (same mechanics) and the XU100 index level."""

from __future__ import annotations

import pandas as pd

from bistmas.data.universe.provider import members_at


def universe_selector(membership: pd.DataFrame, index_code: str = "XU100"):
    def _select(decision: pd.Timestamp) -> list[str]:
        return members_at(membership, decision, index_code, known_only=True)

    return _select


def eligible_selector(membership: pd.DataFrame, screens: pd.DataFrame, index_code: str = "XU100"):
    """Equal weight of the point-in-time members that also pass the pre-registered eligibility filters
    (listing age, liquidity, one class per issuer, positive equity, statement present) at the decision date;
    the names the strategies can actually hold. ``screens`` is the screen table (``as_of, ticker, eligible``)."""
    elig: dict[pd.Timestamp, set[str]] = {}
    if len(screens):
        s = screens[screens["eligible"].astype(bool)]
        for d, g in s.groupby(pd.to_datetime(s["as_of"])):
            elig[pd.Timestamp(d)] = set(g["ticker"])

    def _select(decision: pd.Timestamp) -> list[str]:
        ok = elig.get(pd.Timestamp(decision), set())
        return [t for t in members_at(membership, decision, index_code, known_only=True) if t in ok]

    return _select


def index_daily(
    prices: pd.DataFrame, column: str = "xu100", rescale_date: str = "2020-07-27", factor: float = 100.0
) -> pd.DataFrame:
    """Daily XU100 levels from the price panel, spliced at the July-2020 /100 rescale into one continuous series."""
    s = prices.groupby("date")[column].first().dropna().sort_index()
    s.index = pd.to_datetime(s.index)
    before = s.index < pd.Timestamp(rescale_date)
    spliced = s.copy()
    if before.any() and (~before).any():
        # Borsa İstanbul divided the index by 100 on 2020-07-27. Some vendors (İş Yatırım's END_DEGER) restate
        # the whole history into the new scale, others keep the old scale before the date: splice only when
        # the series actually jumps by the factor at the rescale date.
        ratio = float(s[~before].iloc[0]) / float(s[before].iloc[-1])
        if ratio < 0.5 / factor * 10:  # ~1/100 jump (allow a normal daily move on top)
            spliced[before] = s[before] / factor
    out = pd.DataFrame({"date": spliced.index, "level": spliced.values})
    out["ret"] = out["level"].pct_change().fillna(0.0)
    out["nav"] = (1 + out["ret"]).cumprod()
    return out
