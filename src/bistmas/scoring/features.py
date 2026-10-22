"""Decision-date feature build (pure function of the as-of date and the input panels).

``build_features_one`` never reads a row dated after ``as_of``: price metrics use sessions <= as_of, the
statement is the latest filing with ``availability_date <= as_of``, and every output row carries
``max_source_available_date`` so the run certificate can prove it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from bistmas.data.universe.provider import coerce_membership, members_at
from bistmas.scoring.ratios import RATIO_COLUMNS, ratios_frame
from bistmas.scoring.statements_view import latest_filing_as_of
from bistmas.shared.log import get_logger

log = get_logger("scoring.features")

MARKET_FEATURES = [
    "momentum_12_1",
    "vol_6m",
    "mdd_6m",
    "adv_6m_tl",
    "listed_sessions",
    "mcap_tl",
    "ffloat_mcap_tl",
    "close_raw",
]
FEATURE_COLUMNS = [
    "as_of",
    "ticker",
    "sector_group",
    "template",
    *MARKET_FEATURES,
    *RATIO_COLUMNS,
    "statement_period_end",
    "statement_availability_date",
    "statement_age_days",
    "ttm_method",
    "basis",
    "price_date",
    "max_source_available_date",
]


@dataclass
class PricePanel:
    """Wide matrices (DatetimeIndex sessions x tickers) built once from ``prices_daily``."""

    close_adj: pd.DataFrame
    traded_value: pd.DataFrame
    mcap: pd.DataFrame
    ffloat_mcap: pd.DataFrame
    close_raw: pd.DataFrame
    flags: pd.DataFrame | None = None
    """Boolean matrix (same shape as ``close_adj``): session carries a corporate-action / adjustment-mismatch
    flag; its return is excluded from vol/mdd and contributes 0 to momentum (D4)."""

    @classmethod
    def from_long(cls, prices: pd.DataFrame) -> PricePanel:
        p = prices.copy()
        p["date"] = pd.to_datetime(p["date"])
        p = p[~p["flag_cancelled_session"].fillna(False)] if "flag_cancelled_session" in p.columns else p

        def piv(col: str) -> pd.DataFrame:
            return p.pivot_table(index="date", columns="ticker", values=col, aggfunc="last").sort_index()

        mcap_col = "mcap_tl_aligned" if "mcap_tl_aligned" in p.columns else "mcap_tl"
        close_adj = piv("close_adj")
        # every matrix shares the close_adj index/columns: pivot_table drops all-NaN dates, and metrics_at
        # slices positionally, so differing indexes would let a window overshoot the as-of date (finding 32)
        shape = {"index": close_adj.index, "columns": close_adj.columns}

        def aligned(col: str) -> pd.DataFrame:
            return piv(col).reindex(**shape)

        flag_cols = [c for c in ("flag_capital_change", "flag_adj_mismatch") if c in p.columns]
        flags = None
        if flag_cols:
            fl = p[flag_cols].fillna(False).astype(bool).any(axis=1)
            flags = (
                pd.DataFrame({"date": p["date"], "ticker": p["ticker"], "f": fl})
                .pivot_table(index="date", columns="ticker", values="f", aggfunc="max")
                .reindex(**shape)
                .fillna(0)
                .astype(bool)
            )
        return cls(
            close_adj,
            aligned("traded_value_tl"),
            aligned(mcap_col),
            aligned("ffloat_mcap_tl"),
            aligned("close_raw"),
            flags,
        )

    def metrics_at(
        self,
        as_of: date | pd.Timestamp,
        tickers: list[str],
        listed_since: Mapping[str, pd.Timestamp] | None = None,
    ) -> pd.DataFrame:
        """Momentum, volatility, drawdown, liquidity and size for ``tickers`` on the last session <= as_of.

        Flagged corporate-action sessions are excluded: NaN in the volatility sample, a zero return in the
        momentum chain and in the drawdown path. ``listed_since`` (ticker -> date) overrides the first
        vendor price as the start of the listing-age count when it is earlier.
        """
        ts = pd.Timestamp(as_of)
        idx = self.close_adj.index
        pos = idx.searchsorted(ts, side="right") - 1
        if pos < 0:
            return pd.DataFrame(columns=["ticker", *MARKET_FEATURES, "price_date"]).set_index("ticker")
        end = pos + 1
        cols = [t for t in tickers if t in self.close_adj.columns]
        ca = self.close_adj.iloc[max(0, end - 260) : end][cols]
        fl = (
            self.flags.iloc[max(0, end - 260) : end][cols]
            if self.flags is not None
            else pd.DataFrame(False, index=ca.index, columns=cols)
        )
        caf = ca.ffill()
        lr_f = np.log(caf / caf.shift(1)).where(fl, 0.0).fillna(0.0)  # log return carried by flagged sessions
        tv = self.traded_value.iloc[max(0, end - 126) : end][cols]
        out = pd.DataFrame(index=pd.Index(cols, name="ticker"))
        last = ca.ffill().iloc[-1]
        # momentum 12-1: close 21 sessions ago vs 252 sessions ago
        if len(ca) >= 252:
            removed = lr_f.iloc[-251:-21].sum()  # flagged returns between close[-252] (excl.) and close[-22]
            out["momentum_12_1"] = np.exp(np.log(ca.iloc[-22] / ca.iloc[-252]) - removed) - 1
        else:
            out["momentum_12_1"] = np.nan
        window = ca.iloc[-126:]
        wfl = fl.iloc[-126:]
        rets = np.log(window / window.shift(1)).where(~wfl)
        out["vol_6m"] = rets.std() * np.sqrt(252)
        wf = window.ffill()
        step = np.log(wf / wf.shift(1)).where(~wfl, 0.0).fillna(0.0)
        path = np.exp(step.cumsum()).where(window.notna())
        out["mdd_6m"] = (path / path.cummax() - 1).min()
        out["adv_6m_tl"] = tv.mean()
        first_valid = self.close_adj[cols].apply(lambda s: s.first_valid_index())
        listed_since = listed_since or {}
        counts = []
        for t, fv in zip(cols, first_valid, strict=True):
            start = fv
            ls = listed_since.get(t)
            if ls is not None and pd.notna(ls) and (start is None or pd.Timestamp(ls) < start):
                start = pd.Timestamp(ls)
            if start is None or start > ts:
                counts.append(0)
                continue
            n = int(((idx <= ts) & (idx >= start)).sum())
            if start < idx[0]:  # listed before the price panel starts: add the weekdays not in the panel
                n += int(np.busday_count(start.date(), idx[0].date()))
            counts.append(n)
        out["listed_sessions"] = counts
        out["mcap_tl"] = self.mcap.iloc[:end][cols].ffill().iloc[-1]
        out["ffloat_mcap_tl"] = self.ffloat_mcap.iloc[:end][cols].ffill().iloc[-1]
        out["close_raw"] = self.close_raw.iloc[:end][cols].ffill().iloc[-1]
        # stale prices (no trade on the as-of session) keep the last close but are flagged through price_date
        last_dates = self.close_adj.iloc[:end][cols].apply(lambda s: s.last_valid_index())
        out["price_date"] = [pd.Timestamp(d) if d is not None else pd.NaT for d in last_dates]
        out.loc[last.isna(), MARKET_FEATURES] = np.nan
        return out


@dataclass
class FeatureInputs:
    membership: pd.DataFrame
    panel: PricePanel
    filings: pd.DataFrame  # statements_view wide + ttm
    master: pd.DataFrame  # security master (current rows), index ticker
    tax_rates: dict[int, float]
    index_code: str = "XU100"
    max_statement_age_days: int = 400


def listed_since_from_membership(
    membership: pd.DataFrame, index_code: str = "XU100", as_of: pd.Timestamp | None = None
) -> dict[str, pd.Timestamp]:
    """First official index membership start per ticker (the other candidate for ``listed_since``, D5)."""
    df = coerce_membership(membership)
    df = df[df["index_code"] == index_code.strip().upper()]
    if as_of is not None and "membership_known_since" in df.columns:  # only rows already known at the decision date
        ks = pd.to_datetime(df["membership_known_since"], errors="coerce")
        df = df[ks.isna() | (ks <= as_of)]
    first = df.groupby("ticker")["effective_start"].min()
    return {str(t): pd.Timestamp(d) for t, d in first.items() if pd.notna(d)}


def build_features_one(as_of: date | pd.Timestamp, inputs: FeatureInputs) -> pd.DataFrame:
    """Feature rows for every universe member at ``as_of`` (``FEATURE_COLUMNS``)."""
    ts = pd.Timestamp(as_of)
    members = members_at(inputs.membership, ts, inputs.index_code, known_only=True)
    if not members:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    market = inputs.panel.metrics_at(
        ts, members, listed_since_from_membership(inputs.membership, inputs.index_code, ts)
    )
    latest = latest_filing_as_of(
        inputs.filings[inputs.filings["ticker"].isin(members)], ts, inputs.max_statement_age_days
    )
    ratios = (
        ratios_frame(latest, market, inputs.tax_rates).set_index("ticker")
        if len(latest)
        else pd.DataFrame(
            columns=[*RATIO_COLUMNS, "statement_period_end", "statement_availability_date", "ttm_method", "basis"]
        )
    )
    rows: list[dict[str, Any]] = []
    for t in members:
        m = market.loc[t] if t in market.index else None
        r = ratios.loc[t] if t in ratios.index else None
        sm = inputs.master.loc[t] if t in inputs.master.index else None
        row: dict[str, Any] = {"as_of": ts, "ticker": t}
        row["sector_group"] = str(sm["sector_group"]) if sm is not None else "OTHER"
        row["template"] = str(sm["template"]) if sm is not None else "industrial"
        for c in MARKET_FEATURES:
            row[c] = float(m[c]) if m is not None and pd.notna(m[c]) else np.nan
        for c in RATIO_COLUMNS:
            row[c] = float(r[c]) if r is not None and pd.notna(r[c]) else np.nan
        row["statement_period_end"] = r["statement_period_end"] if r is not None else pd.NaT
        row["statement_availability_date"] = r["statement_availability_date"] if r is not None else pd.NaT
        row["statement_age_days"] = (
            int((ts - pd.Timestamp(r["statement_period_end"])).days)
            if r is not None and pd.notna(r["statement_period_end"])
            else np.nan
        )
        row["ttm_method"] = str(r["ttm_method"]) if r is not None else ""
        row["basis"] = str(r["basis"]) if r is not None else ""
        row["price_date"] = m["price_date"] if m is not None else pd.NaT
        sources = [d for d in (row["price_date"], row["statement_availability_date"]) if d is not None and pd.notna(d)]
        row["max_source_available_date"] = max(sources) if sources else pd.NaT
        rows.append(row)
    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    if (df["max_source_available_date"].dropna() > ts).any():
        raise RuntimeError(f"feature build at {ts.date()} used a source dated after as_of")
    return df
