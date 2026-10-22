"""Universe provider contract and point-in-time membership helpers.

Every universe provider (official roll-forward, DataStore files, rule-based replica) produces the
same *membership interval table* so that the backtest can ask "who was in the index on day t,
and was that fact already public on day t?" without knowing where the list came from.

Membership table (``MEMBERSHIP_COLUMNS``), one row per (index, ticker, membership spell):

``index_code``
    BIST index code: ``XU100`` (BIST 100), ``XU030`` (BIST 30), ``XU050``, ``XUTUM``, ``XYLDZ``.
``ticker``
    BIST equity code (4-5 uppercase letters, no ``.E`` suffix).
``effective_start``
    First session on which the ticker is a member (``datetime64[ns]``, inclusive).
``effective_end``
    First session on which the ticker is *no longer* a member (``datetime64[ns]``, exclusive);
    ``NaT`` means the spell is still open.
``membership_known_since``
    Date from which the spell was public knowledge: the announcement date of a periodic change,
    the snapshot date of an anchor file, or the effective date of a hand-resolved override. A
    backtest that decides on day ``t`` must only use spells with ``membership_known_since <= t``
    (``known_only=True``), otherwise a list that was published later leaks into the past.
``source``
    Provenance of the opening event (``anchor:<name>``, ``announcement:2018Q2``,
    ``override:entry``, ``override:rename``, ``datastore:exsrk2018`` ...).

Date convention: the three date columns are stored as ``datetime64[ns]`` (midnight, tz-naive) so
that ``NaT`` can express an open interval and Parquet round-trips them unchanged. Functions accept
``datetime.date``, ``pd.Timestamp`` or ISO strings for ``as_of`` arguments.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

MEMBERSHIP_COLUMNS: list[str] = [
    "index_code",
    "ticker",
    "effective_start",
    "effective_end",
    "membership_known_since",
    "source",
]
DATE_COLUMNS: tuple[str, ...] = ("effective_start", "effective_end", "membership_known_since")


@runtime_checkable
class UniverseProvider(Protocol):
    """A source of point-in-time index membership.

    ``name`` identifies the provider in run certificates (``official`` | ``rule_based`` ...) and
    ``membership()`` returns a DataFrame with exactly ``MEMBERSHIP_COLUMNS``.
    """

    name: str

    def membership(self) -> pd.DataFrame:
        """Return the membership interval table (see module docstring)."""
        ...


# --------------------------------------------------------------------------- normalisation
def normalise_ticker(value: Any) -> str:
    """Upper-case, strip whitespace and drop a trailing ``.E`` (Borsa İstanbul equity suffix)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip().upper()
    return text[:-2] if text.endswith(".E") else text


def normalise_ticker_series(series: pd.Series) -> pd.Series:
    """Vectorised :func:`normalise_ticker`; missing values become empty strings."""
    return series.astype("string").fillna("").str.strip().str.upper().str.replace(r"\.E$", "", regex=True).astype(str)


def to_datetime_column(series: pd.Series | Iterable[Any]) -> pd.Series:
    """Coerce dates/strings/Timestamps to tz-naive ``datetime64[ns]`` at midnight; blanks -> ``NaT``."""
    s = series if isinstance(series, pd.Series) else pd.Series(list(series), dtype=object)
    if len(s) == 0:
        return pd.Series(pd.DatetimeIndex([], dtype="datetime64[ns]"), index=s.index)
    out = pd.to_datetime(s, errors="coerce")
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out.dt.normalize().astype("datetime64[ns]")


def to_timestamp(value: date | pd.Timestamp | str) -> pd.Timestamp:
    """Single-value counterpart of :func:`to_datetime_column`."""
    ts = pd.Timestamp(value)
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def coerce_membership(df: pd.DataFrame | None) -> pd.DataFrame:
    """Return a copy of ``df`` with exactly ``MEMBERSHIP_COLUMNS``, normalised codes and date dtypes.

    Raises ``ValueError`` when a required column is missing. An empty/None input yields an empty
    frame with the right columns and dtypes.
    """
    if df is None or len(df) == 0:
        empty = pd.DataFrame(
            {c: pd.Series(dtype="datetime64[ns]" if c in DATE_COLUMNS else str) for c in MEMBERSHIP_COLUMNS}
        )
        return empty
    missing = [c for c in MEMBERSHIP_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"membership table is missing columns {missing}")
    out = df.loc[:, MEMBERSHIP_COLUMNS].copy()
    out["index_code"] = out["index_code"].astype("string").fillna("").str.strip().str.upper().astype(str)
    out["ticker"] = normalise_ticker_series(out["ticker"])
    for col in DATE_COLUMNS:
        out[col] = to_datetime_column(out[col])
    out["source"] = out["source"].astype("string").fillna("").astype(str)
    out = out.sort_values(["index_code", "effective_start", "ticker"], kind="stable").reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- queries
def _active_mask(df: pd.DataFrame, ts: pd.Timestamp, known_only: bool) -> pd.Series:
    mask = (df["effective_start"] <= ts) & (df["effective_end"].isna() | (df["effective_end"] > ts))
    if known_only:
        # NaT known-since is treated as "known from effective_start" (lenient); check_membership flags it.
        mask &= df["membership_known_since"].isna() | (df["membership_known_since"] <= ts)
    return mask


def members_at(
    membership_df: pd.DataFrame,
    as_of: date | pd.Timestamp | str,
    index_code: str = "XU100",
    known_only: bool = True,
) -> list[str]:
    """Sorted tickers that are members of ``index_code`` on ``as_of``.

    A ticker is a member when ``effective_start <= as_of < effective_end`` (``NaT`` end = open)
    and, with ``known_only=True`` (the backtest default), ``membership_known_since <= as_of``.
    """
    df = coerce_membership(membership_df)
    df = df[df["index_code"] == index_code.strip().upper()]
    ts = to_timestamp(as_of)
    return sorted(set(df.loc[_active_mask(df, ts, known_only), "ticker"]))


def membership_matrix(
    membership_df: pd.DataFrame,
    dates: Iterable[date | pd.Timestamp | str],
    index_code: str = "XU100",
    known_only: bool = True,
) -> pd.DataFrame:
    """Boolean matrix (rows = ``dates``, columns = tickers) of membership in ``index_code``.

    Columns are every ticker that has a spell in the index (sorted); rows keep the order of
    ``dates`` as a ``DatetimeIndex``. Uses the same rule as :func:`members_at`.
    """
    df = coerce_membership(membership_df)
    df = df[df["index_code"] == index_code.strip().upper()]
    idx = pd.DatetimeIndex(to_datetime_column(list(dates)).to_numpy(), name="date")
    tickers = sorted(set(df["ticker"]))
    values = np.zeros((len(idx), len(tickers)), dtype=bool)
    if len(idx) and len(df):
        col = {t: j for j, t in enumerate(tickers)}
        d = idx.to_numpy(dtype="datetime64[ns]")
        for row in df.itertuples(index=False):
            mask = d >= np.datetime64(row.effective_start, "ns")
            if not pd.isna(row.effective_end):
                mask &= d < np.datetime64(row.effective_end, "ns")
            if known_only and not pd.isna(row.membership_known_since):
                mask &= d >= np.datetime64(row.membership_known_since, "ns")
            values[mask, col[row.ticker]] = True
    return pd.DataFrame(values, index=idx, columns=tickers)


def check_membership(membership_df: pd.DataFrame) -> list[str]:
    """Return human-readable invariant violations (empty list = table is consistent).

    Checks: required columns, non-empty codes, ``effective_end > effective_start`` when set,
    missing ``membership_known_since`` and overlapping spells of the same (index, ticker).
    """
    problems: list[str] = []
    try:
        df = coerce_membership(membership_df)
    except ValueError as exc:
        return [str(exc)]
    if (df["ticker"] == "").any():
        problems.append(f"{int((df['ticker'] == '').sum())} rows with an empty ticker")
    if (df["index_code"] == "").any():
        problems.append(f"{int((df['index_code'] == '').sum())} rows with an empty index_code")
    if df["effective_start"].isna().any():
        problems.append(f"{int(df['effective_start'].isna().sum())} rows with a missing effective_start")
    bad_end = df["effective_end"].notna() & (df["effective_end"] <= df["effective_start"])
    if bad_end.any():
        problems.append(f"{int(bad_end.sum())} rows with effective_end <= effective_start")
    if df["membership_known_since"].isna().any():
        problems.append(f"{int(df['membership_known_since'].isna().sum())} rows with a missing membership_known_since")
    for (code, ticker), grp in df.groupby(["index_code", "ticker"], sort=True):
        if len(grp) < 2:
            continue
        g = grp.sort_values("effective_start")
        prev_end = g["effective_end"].shift(1)
        overlap = prev_end.isna() | (g["effective_start"] < prev_end)
        overlap.iloc[0] = False
        if overlap.any():
            problems.append(f"overlapping spells for {code}/{ticker}")
    return problems
