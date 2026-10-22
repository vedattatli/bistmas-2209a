"""TCMB EVDS (Elektronik Veri Dağıtım Sistemi) v3 client: CPI, USD/TRY and other macro series.

EVDS is the central bank's free statistics API. Access needs a personal key (register at
https://evds3.tcmb.gov.tr, then *Profile -> API Key*), which is sent in the HTTP header ``key`` (since
2024-04-05 it is no longer accepted as a query parameter). The key is a secret: it is read from the
environment variable ``EVDS_API_KEY`` or from a ``.env`` file at the repository root (plain ``KEY=VALUE``
lines; no python-dotenv), never from ``configs/``, and it never appears in a URL or in the raw manifest.

Request shape (verified live on 2026-09-23 without a key: the v3 host answers
``403 {"status":"403","message":"Required request header 'key' is not present"}``, while the legacy
``evds2.tcmb.gov.tr/service/evds/`` path now serves the web application's HTML and is kept only for
the record)::

    GET https://evds3.tcmb.gov.tr/igmevdsms-dis/series=TP.DK.USD.A.YTL&startDate=01-01-2018&endDate=31-12-2018&type=json
    key: <api key>

Optional parameters: ``frequency`` (1 daily, 2 business days, 3 weekly, 4 twice-monthly, 5 monthly,
6 quarterly, 7 semi-annual, 8 annual), ``aggregationTypes`` (avg, min, max, first, last, sum) and
``formulas`` (0 level, 1 percent change, ...). Several series may be joined with ``-`` in one request;
this client issues **one request per series** so that every raw file is keyed by (series, start, end,
frequency) and can be reused by any later job.

Response shape (EVDS user guide)::

    {"totalCount": 2,
     "items": [{"Tarih": "01-01-2018", "TP_DK_USD_A_YTL": "3.7719", "UNIXTIME": {"$numberLong": "1514754000"}},
               {"Tarih": "02-01-2018", "TP_DK_USD_A_YTL": null, "UNIXTIME": {"$numberLong": "1514840400"}}]}

Series codes appear in the items with dots replaced by underscores; values are strings (``null`` on
days without an observation); ``Tarih`` is ``DD-MM-YYYY`` for daily/weekly data, ``YYYY-M`` for monthly,
``YYYY-Qn`` for quarterly and ``YYYY`` for annual data. :func:`parse_response` returns a long frame
``[series, date, value]`` with ``date`` as tz-naive ``datetime64[ns]`` (first day of the period for
monthly and coarser data) and ``value`` as float; rows without a value are dropped.

Raw layer: ``<raw_dir>/evds/<series>_<start>_<end>[_f<frequency>[_<aggregation>]].json.gz`` plus
``manifest.jsonl`` through :meth:`bistmas.data.clients.http.PoliteSession.get_cached`. A cached file is
served even when no key is configured, so a rebuild never needs the key.

Series constants used by the project: :data:`CPI` (``TP.FG.J0``, monthly CPI index, 2003=100),
:data:`USDTRY_BUY` / :data:`USDTRY_SELL` (``TP.DK.USD.A.YTL`` / ``TP.DK.USD.S.YTL``, daily TCMB buying /
selling rates) and :data:`FUNDING_COST` (``TP.APIFON4``, daily weighted average cost of TCMB funding, an
*alternative* risk-free proxy, not the 1-week repo policy rate; see ``tcmb_rates.py``).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import requests

from bistmas.data.clients.http import PoliteSession
from bistmas.shared.config import REPO_ROOT
from bistmas.shared.log import get_logger

log = get_logger("data.evds")

EVDS_BASE_URL = "https://evds3.tcmb.gov.tr/igmevdsms-dis/"
EVDS_LEGACY_BASE_URL = "https://evds2.tcmb.gov.tr/service/evds/"  # retired (serves HTML since the v3 migration)
EVDS_REGISTER_URL = "https://evds3.tcmb.gov.tr"
ENV_KEY_NAME = "EVDS_API_KEY"
KEY_HEADER = "key"
SOURCE_NAME = "evds"
RAW_SUBDIR = "evds"

CPI = "TP.FG.J0"
USDTRY_BUY = "TP.DK.USD.A.YTL"
USDTRY_SELL = "TP.DK.USD.S.YTL"
FUNDING_COST = "TP.APIFON4"

FREQUENCY_CODES: dict[str, int] = {
    "daily": 1,
    "business": 2,
    "weekly": 3,
    "semimonthly": 4,
    "monthly": 5,
    "quarterly": 6,
    "semiannual": 7,
    "annual": 8,
}
AGGREGATIONS: frozenset[str] = frozenset({"avg", "min", "max", "first", "last", "sum"})
SERIES_COLUMNS: list[str] = ["series", "date", "value"]

_SERIES_RE = re.compile(r"^[A-Z0-9]+(?:\.[A-Z0-9]+)+$")
_DAILY_RE = re.compile(r"^(\d{2})-(\d{2})-(\d{4})$")
_MONTHLY_RE = re.compile(r"^(\d{4})-(\d{1,2})$")
_QUARTERLY_RE = re.compile(r"^(\d{4})-Q([1-4])$")
_ANNUAL_RE = re.compile(r"^(\d{4})$")
_ENV_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


class MissingEvdsKeyError(RuntimeError):
    """Raised when EVDS must be contacted but no API key is configured (or the key was rejected)."""


class EvdsError(RuntimeError):
    """Raised when EVDS answers with something that is not a series payload."""


# --------------------------------------------------------------------------- key handling
def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a ``.env`` file: ``KEY=VALUE`` lines, optional ``export``, ``#`` comments, single/double quotes."""
    values: dict[str, str] = {}
    if not Path(path).exists():
        return values
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _ENV_LINE_RE.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


def load_api_key(env_file: Path | None = None, environ: Mapping[str, str] | None = None) -> str | None:
    """Return the EVDS key from ``$EVDS_API_KEY``, else from ``<repo>/.env`` (or ``env_file``); ``None`` if unset."""
    env = os.environ if environ is None else environ
    key = (env.get(ENV_KEY_NAME) or "").strip()
    if key:
        return key
    key = parse_env_file(env_file or REPO_ROOT / ".env").get(ENV_KEY_NAME, "").strip()
    return key or None


def missing_key_message(reason: str = "no EVDS API key is configured") -> str:
    return (
        f"{reason}. EVDS (CPI, USD/TRY) needs a free personal key: register at {EVDS_REGISTER_URL}, open "
        f"Profile -> API Key, then either `export {ENV_KEY_NAME}=<key>` or add `{ENV_KEY_NAME}=<key>` to "
        f"{REPO_ROOT / '.env'} (git-ignored). Until then macro columns that come from EVDS stay NaN and "
        "prices_daily.usdtry (İş Yatırım DD_DEGER) is the USD/TRY fallback."
    )


# --------------------------------------------------------------------------- request building
def _fmt_date(d: date | pd.Timestamp | str) -> str:
    return pd.Timestamp(d).strftime("%d-%m-%Y")


def _frequency_code(frequency: int | str | None) -> int | None:
    if frequency is None or frequency == "":
        return None
    if isinstance(frequency, str) and not frequency.isdigit():
        try:
            return FREQUENCY_CODES[frequency.lower()]
        except KeyError as exc:
            raise ValueError(f"unknown EVDS frequency {frequency!r}; use one of {sorted(FREQUENCY_CODES)}") from exc
    code = int(frequency)
    if code not in FREQUENCY_CODES.values():
        raise ValueError(f"EVDS frequency code must be 1..8, got {code}")
    return code


def validate_series_code(code: str) -> str:
    text = str(code).strip().upper()
    if not _SERIES_RE.match(text):
        raise ValueError(f"{code!r} is not an EVDS series code (expected e.g. TP.FG.J0)")
    return text


def build_url(
    series: str | Sequence[str],
    start: date | pd.Timestamp | str,
    end: date | pd.Timestamp | str,
    frequency: int | str | None = None,
    aggregation: str | None = None,
    formulas: int | None = None,
    base_url: str = EVDS_BASE_URL,
) -> str:
    """EVDS query URL (the key is *not* part of it; it travels in the ``key`` header)."""
    codes = [series] if isinstance(series, str) else list(series)
    params: dict[str, Any] = {
        "series": "-".join(validate_series_code(c) for c in codes),
        "startDate": _fmt_date(start),
        "endDate": _fmt_date(end),
        "type": "json",
    }
    freq = _frequency_code(frequency)
    if freq is not None:
        params["frequency"] = freq
    if aggregation:
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {sorted(AGGREGATIONS)}, got {aggregation!r}")
        params["aggregationTypes"] = aggregation
    if formulas is not None:
        params["formulas"] = int(formulas)
    return base_url + urlencode(params, safe=".-")


def raw_file_name(
    series: str,
    start: date | pd.Timestamp | str,
    end: date | pd.Timestamp | str,
    frequency: int | str | None = None,
    aggregation: str | None = None,
) -> str:
    """Raw-layer file name for one (series, start, end, frequency, aggregation) request."""
    stem = validate_series_code(series).replace(".", "_")
    name = f"{stem}_{pd.Timestamp(start):%Y%m%d}_{pd.Timestamp(end):%Y%m%d}"
    freq = _frequency_code(frequency)
    if freq is not None:
        name += f"_f{freq}"
        if aggregation:
            name += f"_{aggregation}"
    return name + ".json.gz"


# --------------------------------------------------------------------------- response parsing
def parse_evds_date(text: Any) -> pd.Timestamp:
    """``DD-MM-YYYY`` -> that day; ``YYYY-M`` -> first of month; ``YYYY-Qn`` -> quarter start; ``YYYY`` -> 1 Jan."""
    s = str(text).strip()
    if m := _DAILY_RE.match(s):
        return pd.Timestamp(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    if m := _MONTHLY_RE.match(s):
        return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
    if m := _QUARTERLY_RE.match(s):
        return pd.Timestamp(int(m.group(1)), 3 * (int(m.group(2)) - 1) + 1, 1)
    if m := _ANNUAL_RE.match(s):
        return pd.Timestamp(int(m.group(1)), 1, 1)
    ts = pd.to_datetime(s, dayfirst=True, errors="coerce")
    if pd.isna(ts):
        raise EvdsError(f"unrecognised EVDS date {text!r}")
    return pd.Timestamp(ts).normalize()


def _item_key(item: Mapping[str, Any], code: str) -> str | None:
    """Key under which ``code`` appears in an item (``TP_FG_J0`` normally; ``TP.FG.J0`` tolerated)."""
    for cand in (code.replace(".", "_"), code):
        if cand in item:
            return cand
    return None


def parse_response(payload: bytes | str | Mapping[str, Any], series: str | Sequence[str]) -> pd.DataFrame:
    """Turn an EVDS JSON body into the long frame ``[series, date, value]`` (rows without a value dropped)."""
    if isinstance(payload, bytes | str):
        text = payload.decode("utf-8", "replace") if isinstance(payload, bytes) else payload
        try:
            body = json.loads(text)
        except json.JSONDecodeError as exc:
            raise EvdsError(f"EVDS body is not JSON (first bytes: {text[:120]!r})") from exc
    else:
        body = payload
    if not isinstance(body, Mapping) or "items" not in body:
        raise EvdsError(f"EVDS body has no 'items' (got {str(body)[:200]!r})")
    items = body.get("items") or []
    codes = [validate_series_code(series)] if isinstance(series, str) else [validate_series_code(c) for c in series]
    records: list[tuple[str, pd.Timestamp, float]] = []
    for item in items:
        if not isinstance(item, Mapping) or "Tarih" not in item:
            continue
        when = parse_evds_date(item["Tarih"])
        for code in codes:
            key = _item_key(item, code)
            if key is None:
                continue
            value = pd.to_numeric(pd.Series([item[key]]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            records.append((code, when, float(value)))
    df = pd.DataFrame.from_records(records, columns=SERIES_COLUMNS)
    df["series"] = df["series"].astype(str)
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    df["value"] = df["value"].astype(float)
    return df.sort_values(["series", "date"], kind="stable").drop_duplicates(["series", "date"]).reset_index(drop=True)


def empty_series_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"series": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]"), "value": pd.Series(dtype=float)}
    )


# --------------------------------------------------------------------------- client
class EvdsClient:
    """Cached EVDS series fetcher.

    ``session`` is the shared :class:`PoliteSession`; ``raw_dir`` is ``settings.paths.raw`` (files go under
    ``raw_dir/evds``). ``api_key=None`` means :func:`load_api_key`; ``has_key`` tells callers whether an
    uncached request can be made. Requests are paced by the session (EVDS allows generous use, but the
    project never re-fetches a cached file).
    """

    def __init__(
        self,
        session: PoliteSession,
        raw_dir: Path,
        api_key: str | None = None,
        base_url: str = EVDS_BASE_URL,
        env_file: Path | None = None,
    ) -> None:
        self.session = session
        self.raw_dir = Path(raw_dir) / RAW_SUBDIR
        self.manifest_path = self.raw_dir / "manifest.jsonl"
        self.base_url = base_url
        self.api_key = (api_key or "").strip() or load_api_key(env_file)

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)

    def raw_path(
        self,
        series: str,
        start: date | pd.Timestamp | str,
        end: date | pd.Timestamp | str,
        frequency: int | str | None = None,
        aggregation: str | None = None,
    ) -> Path:
        return self.raw_dir / raw_file_name(series, start, end, frequency, aggregation)

    def fetch_one(
        self,
        series: str,
        start: date | pd.Timestamp | str,
        end: date | pd.Timestamp | str,
        frequency: int | str | None = None,
        aggregation: str | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """Fetch (or read from the raw layer) one series and return ``[series, date, value]``."""
        code = validate_series_code(series)
        raw_path = self.raw_path(code, start, end, frequency, aggregation)
        cached = raw_path.exists() and not force
        if not cached and not self.has_key:
            raise MissingEvdsKeyError(
                missing_key_message(f"{code} {_fmt_date(start)}..{_fmt_date(end)} is not cached and")
            )
        url = build_url(code, start, end, frequency, aggregation, base_url=self.base_url)
        headers = {KEY_HEADER: self.api_key} if self.api_key else {}
        try:
            payload = self.session.get_cached(
                url, raw_path, self.manifest_path, source=SOURCE_NAME, force=force, headers=headers
            )
        except requests.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (401, 403):
                raise MissingEvdsKeyError(missing_key_message(f"EVDS rejected the request ({status})")) from exc
            raise
        try:
            df = parse_response(payload, code)
        except EvdsError as exc:
            log.warning("EVDS %s: %s; discarding %s", code, exc, raw_path)
            raw_path.unlink(missing_ok=True)
            raise
        log.info(
            "EVDS %s %s..%s: %d rows (%s)",
            code,
            _fmt_date(start),
            _fmt_date(end),
            len(df),
            "cache" if cached else "fetched",
        )
        return df

    def fetch_series(
        self,
        series: Sequence[str] | str,
        start: date | pd.Timestamp | str,
        end: date | pd.Timestamp | str,
        frequency: int | str | None = None,
        aggregation: str | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """Fetch several series (one request each) and return one long frame ``[series, date, value]``."""
        codes = [series] if isinstance(series, str) else list(series)
        if not codes:
            return empty_series_frame()
        frames = [self.fetch_one(c, start, end, frequency, aggregation, force=force) for c in codes]
        out = pd.concat(frames, ignore_index=True) if frames else empty_series_frame()
        return out.sort_values(["series", "date"], kind="stable").reset_index(drop=True)

    def fetch_series_by_year(
        self,
        series: Sequence[str] | str,
        start: date | pd.Timestamp | str,
        end: date | pd.Timestamp | str,
        frequency: int | str | None = None,
        aggregation: str | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """Like :meth:`fetch_series` but in calendar-year chunks so raw files stay small and reusable."""
        first, last = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        if last < first:
            raise ValueError(f"end {end} precedes start {start}")
        frames = []
        for year in range(first.year, last.year + 1):
            lo = max(first, pd.Timestamp(year, 1, 1))
            hi = min(last, pd.Timestamp(year, 12, 31))
            frames.append(self.fetch_series(series, lo, hi, frequency, aggregation, force=force))
        out = pd.concat(frames, ignore_index=True) if frames else empty_series_frame()
        return (
            out.drop_duplicates(["series", "date"])
            .sort_values(["series", "date"], kind="stable")
            .reset_index(drop=True)
        )


def pivot_series(long_df: pd.DataFrame) -> pd.DataFrame:
    """Wide view of a long ``[series, date, value]`` frame: one column per series code, indexed by date."""
    if long_df.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
    wide = long_df.pivot(index="date", columns="series", values="value").sort_index()
    wide.columns.name = None
    return wide
