"""TCMB policy rate (1-week repo auction rate) from the central bank's public HTML table.

The CBRT has used the one-week repo auction rate as its policy rate since 2010-05-20 and publishes
every change as a plain table at :data:`POLICY_RATE_URL` (page title "1 Hafta Repo"; verified live on
2026-09-23: 57 data rows from 20.05.2010 = 7.00 to 23.01.2026 = 37.00). Markup, as served::

    <table class="table-condensed ..." id="midTable">
      <tr class="MerkezBankasiTableHeader"><td>Tarih</td><td>Borç Alma</td><td>Borç Verme</td></tr>
      <tr class="zebra1"><td>20.05.2010</td><td>-</td><td>7.00</td></tr>
      ...

``Tarih`` is the effective date of the decision (``DD.MM.YYYY``), ``Borç Verme`` (lending) holds the
auction rate in percent, ``Borç Alma`` (borrowing) is always ``-`` for this instrument. The parser looks
the table up by its header cells, not by ``id`` or CSS class, so cosmetic changes do not break it; a
page without a recognisable table raises :class:`PolicyRateParseError` and the cached copy is removed.

No EVDS series carries the 1-week repo rate itself (checked 2026-09-23; ``TP.APIFON4`` is the weighted
average cost of CBRT funding, which equals the policy rate only when all funding goes through the
1-week repo and exceeds it whenever the bank funds via the overnight window, e.g. 2017-2018 and
March 2025). :func:`policy_rate_from_evds` therefore offers the funding cost as a *documented
alternative* risk-free proxy, not as a drop-in replacement; the default ``rf`` of the project is the
policy rate from this table (``configs/base.toml`` ``backtest.rf_source = "tcmb_policy_rate"``).

Risk-free convention (:func:`overnight_rate_daily`): ``rate / 100 / 365`` is the simple rate for one
*calendar* day (ACT/365, no compounding); :func:`accrual_between_sessions` multiplies it by the number
of calendar days since the previous session, so cash held over a weekend earns three days. The step
series is effective-dated: the new rate applies from its ``effective_date`` (inclusive); dates before
the first row are ``NaN``.

Raw layer: ``<raw_dir>/tcmb/one_week_repo.html.gz`` + ``manifest.jsonl`` through
:meth:`bistmas.data.clients.http.PoliteSession.get_cached` (fetched once; ``force=True`` refreshes).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from lxml import html as lh

from bistmas.data.clients.evds import FUNDING_COST
from bistmas.shared.log import get_logger

if TYPE_CHECKING:
    from bistmas.data.clients.evds import EvdsClient
    from bistmas.data.clients.http import PoliteSession

log = get_logger("data.tcmb_rates")

POLICY_RATE_URL = (
    "https://www.tcmb.gov.tr/wps/wcm/connect/TR/TCMB+TR/Main+Menu/Temel+Faaliyetler/Para+Politikasi/"
    "Merkez+Bankasi+Faiz+Oranlari/1+Hafta+Repo"
)
SOURCE_NAME = "tcmb"
RAW_SUBDIR = "tcmb"
RAW_FILE_NAME = "one_week_repo.html.gz"
POLICY_COLUMNS: list[str] = ["effective_date", "rate"]
DAY_COUNT = 365
POLICY_RATE_START = date(2010, 5, 20)
EVDS_FUNDING_COST_SERIES = FUNDING_COST

_DATE_RE = re.compile(r"^(\d{1,2})[./-](\d{1,2})[./-](\d{4})$")
_HEADER_DATE = "TARIH"
_HEADER_LENDING = "VERME"
_HEADER_BORROWING = "ALMA"
_MIN_ROWS = 5


class PolicyRateParseError(RuntimeError):
    """Raised when the TCMB page does not contain the expected rate table."""


# --------------------------------------------------------------------------- parsing
def _norm(text: str) -> str:
    s = re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip()
    return s.replace("ı", "I").replace("i", "İ").upper().replace("İ", "I")


def _parse_rate(text: str) -> float:
    s = (text or "").strip().replace("%", "").replace(" ", "")
    if s in ("", "-", "—", "–"):
        return float("nan")
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _parse_date(text: str) -> pd.Timestamp | None:
    m = _DATE_RE.match((text or "").strip())
    if not m:
        return None
    return pd.Timestamp(int(m.group(3)), int(m.group(2)), int(m.group(1)))


def _cells(row: Any) -> list[str]:
    return [" ".join(c.text_content().split()) for c in row.xpath("./td|./th")]


def parse_policy_rate_html(html: str | bytes) -> pd.DataFrame:
    """Parse the 1-week repo page into ``[effective_date, rate]`` (sorted, one row per date).

    The table is recognised by a header row whose first cell is ``Tarih``; the rate column is the one
    headed ``Borç Verme`` (falling back to the last column). Rows whose date or rate cannot be parsed
    are skipped with a warning; fewer than five rows, rates outside ``(0, 100)`` or an empty table raise
    :class:`PolicyRateParseError`.
    """
    text = html.decode("utf-8", "replace") if isinstance(html, bytes) else html
    if not text.strip():
        raise PolicyRateParseError("empty page")
    try:
        doc = lh.fromstring(text)
    except Exception as exc:  # noqa: BLE001 - lxml raises several parser error types
        raise PolicyRateParseError(f"page is not HTML: {exc}") from exc
    for table in doc.xpath("//table"):
        rows = table.xpath(".//tr")
        if not rows:
            continue
        header = [_norm(c) for c in _cells(rows[0])]
        if not header or not header[0].startswith(_HEADER_DATE):
            continue
        rate_col = next((i for i, h in enumerate(header) if _HEADER_LENDING in h), len(header) - 1)
        borrow_col = next((i for i, h in enumerate(header) if _HEADER_BORROWING in h), None)
        records: list[tuple[pd.Timestamp, float]] = []
        for row in rows[1:]:
            cells = _cells(row)
            if len(cells) <= rate_col:
                continue
            when = _parse_date(cells[0])
            if when is None:
                log.warning("policy rate table: unparseable date %r skipped", cells[0])
                continue
            rate = _parse_rate(cells[rate_col])
            if np.isnan(rate) and borrow_col is not None and borrow_col < len(cells):
                rate = _parse_rate(cells[borrow_col])
            if np.isnan(rate):
                log.warning("policy rate table: no rate on %s skipped", when.date())
                continue
            records.append((when, rate))
        if not records:
            continue
        df = pd.DataFrame(records, columns=POLICY_COLUMNS)
        df["effective_date"] = pd.to_datetime(df["effective_date"]).astype("datetime64[ns]")
        df["rate"] = df["rate"].astype(float)
        df = df.sort_values("effective_date", kind="stable").drop_duplicates("effective_date", keep="last")
        df = df.reset_index(drop=True)
        if len(df) < _MIN_ROWS:
            raise PolicyRateParseError(f"policy rate table has only {len(df)} rows")
        if not ((df["rate"] > 0) & (df["rate"] < 100)).all():
            raise PolicyRateParseError(f"policy rates out of range: {df['rate'].describe().to_dict()}")
        return df
    raise PolicyRateParseError("no table with a 'Tarih' header found on the page")


def empty_policy_frame() -> pd.DataFrame:
    return pd.DataFrame({"effective_date": pd.Series(dtype="datetime64[ns]"), "rate": pd.Series(dtype=float)})


def coerce_policy_frame(policy_df: pd.DataFrame | None) -> pd.DataFrame:
    """Normalise a ``[effective_date, rate]`` frame (any date-like input) to the module's dtypes."""
    if policy_df is None or len(policy_df) == 0:
        return empty_policy_frame()
    missing = [c for c in POLICY_COLUMNS if c not in policy_df.columns]
    if missing:
        raise ValueError(f"policy rate table is missing columns {missing}")
    out = policy_df.loc[:, POLICY_COLUMNS].copy()
    out["effective_date"] = pd.to_datetime(out["effective_date"]).dt.normalize().astype("datetime64[ns]")
    out["rate"] = pd.to_numeric(out["rate"], errors="coerce").astype(float)
    out = out.dropna().sort_values("effective_date", kind="stable").drop_duplicates("effective_date", keep="last")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- fetching
def fetch_policy_rate_table(session: PoliteSession, raw_dir: Path, force: bool = False) -> pd.DataFrame:
    """Return ``[effective_date, rate]`` from the raw layer (``raw_dir/tcmb``) or from one live fetch.

    A page that no longer parses is removed from the raw layer (so the next call re-fetches) and the
    error is raised; callers may then fall back to :func:`policy_rate_from_evds`.
    """
    raw_path = Path(raw_dir) / RAW_SUBDIR / RAW_FILE_NAME
    manifest = Path(raw_dir) / RAW_SUBDIR / "manifest.jsonl"
    cached = raw_path.exists() and not force
    payload = session.get_cached(POLICY_RATE_URL, raw_path, manifest, source=SOURCE_NAME, force=force)
    try:
        df = parse_policy_rate_html(payload)
    except PolicyRateParseError as exc:
        log.warning("TCMB policy rate page: %s; discarding %s", exc, raw_path)
        raw_path.unlink(missing_ok=True)
        raise
    log.info(
        "TCMB policy rate: %d changes %s..%s, latest %.2f%% (%s)",
        len(df),
        df["effective_date"].iloc[0].date(),
        df["effective_date"].iloc[-1].date(),
        df["rate"].iloc[-1],
        "cache" if cached else "fetched",
    )
    return df


def policy_rate_from_evds(
    client: EvdsClient,
    start: date | pd.Timestamp | str,
    end: date | pd.Timestamp | str,
    series: str = EVDS_FUNDING_COST_SERIES,
    force: bool = False,
) -> pd.DataFrame:
    """Documented alternative: an EVDS daily rate series reduced to its change points.

    Default ``series`` is ``TP.APIFON4`` (weighted average cost of CBRT funding). It is *not* the 1-week
    repo rate (it sits above it whenever funding shifts to the overnight window), so use it only when the
    HTML table is unavailable and say so in the run report.
    """
    long_df = client.fetch_series_by_year(series, start, end, force=force)
    if long_df.empty:
        return empty_policy_frame()
    s = long_df.sort_values("date")
    changed = s["value"].ne(s["value"].shift(1))
    out = s.loc[changed, ["date", "value"]].rename(columns={"date": "effective_date", "value": "rate"})
    return coerce_policy_frame(out)


# --------------------------------------------------------------------------- step series and rf
def _dates_index(dates: Iterable[date | pd.Timestamp | str] | pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)) if not isinstance(dates, pd.DatetimeIndex) else dates)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return pd.DatetimeIndex(idx.normalize().astype("datetime64[ns]"), name="date")


def policy_rate_on(policy_df: pd.DataFrame, dates: Iterable[date | pd.Timestamp | str]) -> pd.Series:
    """Annual policy rate (percent) in force on each date: last change with ``effective_date <= date``."""
    pol = coerce_policy_frame(policy_df)
    idx = _dates_index(dates)
    if pol.empty or len(idx) == 0:
        return pd.Series(np.nan, index=idx, name="policy_rate", dtype=float)
    probe = pd.DataFrame({"date": idx, "_order": np.arange(len(idx))}).sort_values("date")
    merged = pd.merge_asof(probe, pol, left_on="date", right_on="effective_date", direction="backward")
    merged = merged.sort_values("_order")
    return pd.Series(merged["rate"].to_numpy(dtype=float), index=idx, name="policy_rate")


def overnight_rate_daily(
    policy_df: pd.DataFrame, dates: Iterable[date | pd.Timestamp | str], day_count: int = DAY_COUNT
) -> pd.Series:
    """Simple risk-free rate for one calendar day: ``policy_rate / 100 / day_count`` (ACT/365 by default)."""
    return (policy_rate_on(policy_df, dates) / 100.0 / float(day_count)).rename("rf_daily")


def accrual_between_sessions(
    policy_df: pd.DataFrame, sessions: Iterable[date | pd.Timestamp | str], day_count: int = DAY_COUNT
) -> pd.Series:
    """Cash return earned *by* each session since the previous one: ``rf_daily * calendar days elapsed``.

    The first session counts one day. Sessions must be given in ascending order (they are sorted here);
    the rate used is the one in force on the session itself (a change effective on day t is earned from t).
    """
    idx = _dates_index(sessions).sort_values()
    daily = overnight_rate_daily(policy_df, idx, day_count)
    gaps = np.diff(idx.to_numpy(dtype="datetime64[D]").astype("int64"), prepend=np.nan)
    gaps = np.where(np.isnan(gaps), 1.0, gaps)
    return (daily * gaps).rename("rf_session")
