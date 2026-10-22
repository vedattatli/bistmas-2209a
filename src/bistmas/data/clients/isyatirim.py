"""İş Yatırım ``HisseTekil`` daily price client (unofficial JSON endpoint, cached raw layer).

Endpoint (verified 2026-09-23)::

    GET https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/HisseTekil
        ?hisse=<TICKER>&startdate=DD-MM-YYYY&enddate=DD-MM-YYYY
    -> {"value": [ {31 fields}, ... ]}           one object per trading session

Governance (``docs/data_governance.md``): the site's terms forbid automated access, so the project makes
ONE slow, cached research pull. Every request goes through :class:`~bistmas.data.clients.http.PoliteSession`
with ``min_interval_s >= 1.5`` (:data:`MIN_INTERVAL_S`), a 60 s timeout and bounded retries, and every body
is written once to ``<raw_dir>/isyatirim/<TICKER>/<YYYY>.json.gz`` with a manifest line. Cached bodies are
never re-requested, **including empty ones**: ``{"value": []}`` is the endpoint's answer for unknown codes
(MRDIN, ITTFH, AVISA, GUSGR, YAZIC) and for years after a delisting, and re-asking would only add load.

Why one-year chunks: a full-history request (2000–today, 4–5 MB) takes 17–55 s and frequently dies with a
connection reset; a one-year request (~250 rows, ~175 KB) takes 8–12 s and is reliable.

Why ``Connection: close`` (:data:`REQUEST_HEADERS`): with a keep-alive session every second request hung
until the 60 s read timeout and only succeeded on the retry (measured 2026-09-23: 0.4 s / 64 s / 0.9 s /
64 s ...); the server drops the idle pooled connection. Opening a fresh connection per request gives a
steady ~11 s per one-year chunk.

Payload facts (inspected on the live THYAO/ASELS/YKBNK/TRKCM payloads, see the module constants):

* ``HGDG_TARIH`` is the session date as ``DD-MM-YYYY`` text; ``DD_TARIH``/``END_TARIH`` are ``/Date(ms)/``
  or epoch-millisecond stamps of the USD/TRY and index observations and are not used (they carry the same day).
* ``HG_*`` are raw (as traded) close / VWAP / low / high / volume; ``HGDG_*`` are adjusted for rights issues,
  bonus issues **and cash dividends**, restated retroactively after every event.
* ``SERMAYE`` is paid-in capital in TL (= share count at 1 TL nominal), updated one session *before* the
  ex-date; ``PD`` / ``HAO_PD`` are market cap and free-float market cap in TL (``HAO_PD`` may be null in
  early years); ``DD_DEGER`` is USD/TRY; ``END_DEGER`` the XU100 close (÷100 rescale on 2020-07-27).
* There is no Open price.
* Volume semantics (``HG_HACIM`` / ``HGDG_HACIM``): see :func:`volume_semantics`, which measures on a parsed
  frame whether the field is a share count (``volume * vwap`` equal for raw and adjusted legs) or a TL value
  (raw and adjusted volume identical). The finding is documented in the M2 report and in ``prices_fetch``.

Field -> store column mapping is :data:`FIELD_MAP`; :func:`parse_hissetekil` returns :data:`PARSED_COLUMNS`.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from bistmas.data.clients.http import PoliteSession
from bistmas.shared.codes import EQUITY_CODE_RE
from bistmas.shared.log import get_logger

log = get_logger("data.isyatirim")

HISSETEKIL_URL = "https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/HisseTekil"
SOURCE_NAME = "isyatirim"
RAW_MANIFEST = "manifest.jsonl"
MIN_INTERVAL_S = 1.5
TIMEOUT_S = 60.0
MAX_RETRIES = 4
FIRST_YEAR = 2000  # the endpoint's history starts 2000-01-04
DATE_FIELD = "HGDG_TARIH"
REQUEST_HEADERS: dict[str, str] = {"Connection": "close"}  # see module docstring: keep-alive hangs every 2nd call

# Raw field -> store column (docs/data_dictionary.md, M2 store contract).
FIELD_MAP: dict[str, str] = {
    "HG_KAPANIS": "close_raw",
    "HG_AOF": "vwap_raw",
    "HG_MIN": "low_raw",
    "HG_MAX": "high_raw",
    "HG_HACIM": "volume_raw",
    "HGDG_KAPANIS": "close_adj",
    "HGDG_AOF": "vwap_adj",
    "HGDG_MIN": "low_adj",
    "HGDG_MAX": "high_adj",
    "HGDG_HACIM": "volume_adj",
    "SERMAYE": "shares",
    "PD": "mcap_tl",
    "HAO_PD": "ffloat_mcap_tl",
    "PD_USD": "mcap_usd",
    "HAO_PD_USD": "ffloat_mcap_usd",
    "DD_DEGER": "usdtry",
    "END_DEGER": "xu100",
    "DOLAR_BAZLI_FIYAT": "price_usd",
    # extra fields kept for completeness (not in the prices_daily contract)
    "DOLAR_BAZLI_MIN": "low_usd",
    "DOLAR_BAZLI_MAX": "high_usd",
    "DOLAR_BAZLI_AOF": "vwap_usd",
    "DOLAR_HACIM": "volume_usd",
    "ENDEKS_BAZLI_FIYAT": "price_index_rel",
    "END_SEANS": "xu100_session",
}
CONTRACT_COLUMNS: tuple[str, ...] = (
    "close_raw",
    "vwap_raw",
    "low_raw",
    "high_raw",
    "volume_raw",
    "close_adj",
    "vwap_adj",
    "low_adj",
    "high_adj",
    "volume_adj",
    "shares",
    "mcap_tl",
    "ffloat_mcap_tl",
    "mcap_usd",
    "ffloat_mcap_usd",
    "usdtry",
    "xu100",
    "price_usd",
)
EXTRA_COLUMNS: tuple[str, ...] = tuple(c for c in FIELD_MAP.values() if c not in CONTRACT_COLUMNS)
PARSED_COLUMNS: tuple[str, ...] = ("ticker", "date", *CONTRACT_COLUMNS, *EXTRA_COLUMNS)

_ASPNET_DATE_RE = re.compile(r"/Date\((-?\d+)(?:[+-]\d{4})?\)/")
_DMY_RE = re.compile(r"^(\d{1,2})-(\d{1,2})-(\d{4})$")


class IsYatirimError(RuntimeError):
    """Raised when the endpoint answers with something that is not a ``HisseTekil`` JSON payload."""


# ----------------------------------------------------------------------------- request helpers
def normalise_ticker(ticker: str) -> str:
    """Upper-case equity code without the ``.E`` suffix; raises ``ValueError`` for non-equity codes."""
    code = str(ticker).strip().upper()
    if code.endswith(".E"):
        code = code[:-2]
    if not EQUITY_CODE_RE.fullmatch(code):
        raise ValueError(f"{ticker!r} is not a BIST equity code (expected {EQUITY_CODE_RE.pattern})")
    return code


def year_range(year: int, today: date | None = None) -> tuple[str, str]:
    """``(startdate, enddate)`` in ``DD-MM-YYYY`` for one calendar year (the current year ends today)."""
    today = today or date.today()
    if year < FIRST_YEAR or year > today.year:
        raise ValueError(f"year {year} outside {FIRST_YEAR}..{today.year}")
    end = date(year, 12, 31) if year < today.year else today
    return f"01-01-{year}", end.strftime("%d-%m-%Y")


def build_url(ticker: str, year: int, today: date | None = None) -> str:
    """Full request URL for ``ticker`` and calendar ``year`` (the manifest records it verbatim)."""
    start, end = year_range(year, today)
    return f"{HISSETEKIL_URL}?hisse={normalise_ticker(ticker)}&startdate={start}&enddate={end}"


def make_session() -> PoliteSession:
    """The session every price pull must use: ≥ 1.5 s spacing, 60 s timeout, 4 retries."""
    return PoliteSession(min_interval_s=MIN_INTERVAL_S, timeout_s=TIMEOUT_S, max_retries=MAX_RETRIES)


# ----------------------------------------------------------------------------- payload validation / parsing
def validate_payload(payload: bytes) -> list[dict[str, Any]]:
    """Return the ``value`` list of a ``HisseTekil`` body; raise :class:`IsYatirimError` otherwise.

    HTML (login page, WAF challenge, SharePoint error page) and any JSON without a ``value`` list are
    rejected so that they are never kept in the raw layer.
    """
    head = payload[:512].lstrip().lower()
    if not head or head.startswith(b"<") or b"<html" in head or b"<!doctype" in head:
        raise IsYatirimError(f"HTML/empty body instead of JSON (first bytes: {payload[:80]!r})")
    try:
        obj = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IsYatirimError(f"body is not JSON: {exc}") from exc
    if not isinstance(obj, dict) or not isinstance(obj.get("value"), list):
        raise IsYatirimError(
            f"JSON without a 'value' list (keys: {list(obj)[:5] if isinstance(obj, dict) else type(obj)})"
        )
    # envelope: {"ok": true, "errorCode": null, "errorDescription": null, "transactionId": "...", "value": [...]}
    if obj.get("ok") is False or obj.get("errorCode") not in (None, "", 0):
        raise IsYatirimError(f"endpoint error: code={obj.get('errorCode')!r} {obj.get('errorDescription')!r}")
    rows = obj["value"]
    if any(not isinstance(r, dict) for r in rows):
        raise IsYatirimError("'value' contains non-object rows")
    return rows


def parse_session_date(value: Any) -> pd.Timestamp:
    """Parse one date cell: ``DD-MM-YYYY`` text (the ``HGDG_TARIH`` format), ``/Date(ms)/``, epoch ms or ISO.

    Epoch stamps are interpreted in Europe/Istanbul (the server's local midnight) and reduced to the
    calendar day; the result is tz-naive midnight or ``NaT``.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    if isinstance(value, str):
        text = value.strip()
        m = _DMY_RE.match(text)
        if m:
            day, month, year = (int(g) for g in m.groups())
            try:
                return pd.Timestamp(year=year, month=month, day=day)
            except ValueError:
                return pd.NaT
        m = _ASPNET_DATE_RE.match(text)
        if m:
            value = int(m.group(1))
        else:
            ts = pd.to_datetime(text, errors="coerce")
            if pd.isna(ts):
                return pd.NaT
            if ts.tz is not None:
                ts = ts.tz_convert("Europe/Istanbul").tz_localize(None)
            return ts.normalize()
    if isinstance(value, int | float | np.integer | np.floating):
        ts = pd.Timestamp(int(value), unit="ms", tz="UTC").tz_convert("Europe/Istanbul").tz_localize(None)
        return ts.normalize()
    return pd.NaT


def parse_hissetekil(payload: bytes, ticker: str | None = None) -> pd.DataFrame:
    """Parse a raw ``HisseTekil`` body into a frame with :data:`PARSED_COLUMNS`.

    ``date`` comes from ``HGDG_TARIH`` (``DD-MM-YYYY``) as tz-naive ``datetime64[ns]``; every other field is
    coerced to ``float64`` with ``NaN`` for nulls/blanks. Fields absent from the payload become all-``NaN``
    columns so the schema is stable. Rows without a parsable date are dropped (logged). The frame is sorted
    by date with one row per date (the last one wins). An empty ``value`` list gives an empty frame with the
    same columns and dtypes.
    """
    rows = validate_payload(payload)
    code = normalise_ticker(ticker) if ticker else ""
    if not rows:
        return _empty_frame(code)
    raw = pd.DataFrame.from_records(rows)
    if DATE_FIELD not in raw.columns:
        raise IsYatirimError(f"payload rows lack {DATE_FIELD} (fields: {list(raw.columns)[:8]})")
    out = pd.DataFrame(index=raw.index)
    out["ticker"] = code
    out["date"] = pd.Series([parse_session_date(v) for v in raw[DATE_FIELD]], index=raw.index, dtype="datetime64[ns]")
    for field, column in FIELD_MAP.items():
        out[column] = pd.to_numeric(raw[field], errors="coerce").astype("float64") if field in raw.columns else np.nan
    bad = out["date"].isna()
    if bad.any():
        log.warning("%s: dropping %d rows without a parsable %s", code or "?", int(bad.sum()), DATE_FIELD)
        out = out[~bad]
    out = out.sort_values("date", kind="stable").drop_duplicates(subset="date", keep="last")
    return out.loc[:, list(PARSED_COLUMNS)].reset_index(drop=True)


def _empty_frame(ticker: str = "") -> pd.DataFrame:
    data: dict[str, Any] = {"ticker": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]")}
    for column in (*CONTRACT_COLUMNS, *EXTRA_COLUMNS):
        data[column] = pd.Series(dtype="float64")
    frame = pd.DataFrame(data)
    frame.attrs["ticker"] = ticker
    return frame


# ----------------------------------------------------------------------------- diagnostics
def volume_semantics(df: pd.DataFrame, tolerance: float = 0.02) -> dict[str, Any]:
    """Decide whether ``HACIM`` is a share count or a TL traded value from a parsed frame.

    If ``HACIM`` is a share count, ``volume_raw * vwap_raw`` and ``volume_adj * vwap_adj`` are the same TL
    value (the adjustment multiplies shares and divides prices by the same factor) while ``volume_raw`` and
    ``volume_adj`` differ wherever ``adj_factor != 1``. If it is a TL value the two volume columns are equal
    and the products differ. Returns the medians of both ratios and a verdict in
    ``{"shares", "tl_value", "unknown"}`` (``unknown`` when there is no day with ``adj_factor != 1`` to
    tell them apart).
    """
    d = df.dropna(subset=["volume_raw", "volume_adj", "vwap_raw", "vwap_adj", "close_raw", "close_adj"])
    d = d[(d["volume_raw"] > 0) & (d["volume_adj"] > 0) & (d["vwap_raw"] > 0) & (d["vwap_adj"] > 0)]
    if d.empty:
        return {"verdict": "unknown", "n": 0}
    adj_factor = d["close_adj"] / d["close_raw"]
    value_ratio = (d["volume_adj"] * d["vwap_adj"]) / (d["volume_raw"] * d["vwap_raw"])
    volume_ratio = d["volume_adj"] / d["volume_raw"]
    distinct = d[(adj_factor - 1.0).abs() > tolerance]
    result: dict[str, Any] = {
        "n": int(len(d)),
        "n_adjusted_days": int(len(distinct)),
        "median_adj_factor": float(adj_factor.median()),
        "median_value_ratio": float(value_ratio.median()),
        "median_volume_ratio": float(volume_ratio.median()),
    }
    if distinct.empty:
        result["verdict"] = "unknown"
        return result
    vr = value_ratio.loc[distinct.index]
    volr = volume_ratio.loc[distinct.index]
    result["adjusted_days_value_ratio_median"] = float(vr.median())
    result["adjusted_days_volume_ratio_median"] = float(volr.median())
    if ((vr - 1.0).abs() <= tolerance).mean() > 0.9:
        result["verdict"] = "shares"
    elif ((volr - 1.0).abs() <= tolerance).mean() > 0.9:
        result["verdict"] = "tl_value"
    else:
        result["verdict"] = "unknown"
    return result


# ----------------------------------------------------------------------------- client
class IsYatirimClient:
    """Cached, paced access to ``HisseTekil`` through a :class:`PoliteSession`.

    ``raw_dir`` is the project raw layer (``data/raw``); bodies live under ``<raw_dir>/isyatirim/<TICKER>/
    <YYYY>.json.gz`` with one shared ``manifest.jsonl``. The session's spacing is raised to
    :data:`MIN_INTERVAL_S` when it is lower (governance floor).
    """

    def __init__(self, session: PoliteSession, raw_dir: Path, headers: dict[str, str] | None = None) -> None:
        self.session = session
        self.raw_dir = Path(raw_dir)
        self.headers = {**REQUEST_HEADERS, **(headers or {})}
        interval = getattr(session, "min_interval_s", None)
        if interval is not None and interval < MIN_INTERVAL_S:
            log.warning(
                "raising session min_interval_s from %.2f to %.2f s (İş Yatırım floor)", interval, MIN_INTERVAL_S
            )
            session.min_interval_s = MIN_INTERVAL_S

    # ------------------------------------------------------------------ paths
    @property
    def source_dir(self) -> Path:
        return self.raw_dir / SOURCE_NAME

    @property
    def manifest_path(self) -> Path:
        return self.source_dir / RAW_MANIFEST

    def raw_path(self, ticker: str, year: int) -> Path:
        return self.source_dir / normalise_ticker(ticker) / f"{int(year)}.json.gz"

    def is_cached(self, ticker: str, year: int) -> bool:
        return self.raw_path(ticker, year).exists()

    # ------------------------------------------------------------------ fetching
    def _get_cached(self, url: str, raw_path: Path, force: bool) -> bytes:
        attempts = 0
        while True:
            try:
                return self.session.get_cached(
                    url,
                    raw_path,
                    self.manifest_path,
                    source=SOURCE_NAME,
                    force=force,
                    timeout=TIMEOUT_S,
                    headers=self.headers,
                )
            except requests.exceptions.ChunkedEncodingError as exc:
                # a reset in the middle of the body is not retried by PoliteSession
                attempts += 1
                if attempts > 2:
                    raise
                log.warning("%s: body truncated (%s); retry %d/2", url, exc.__class__.__name__, attempts)
                time.sleep(5.0 * attempts)

    def fetch_year(self, ticker: str, year: int, force: bool = False, today: date | None = None) -> pd.DataFrame:
        """Daily rows of ``ticker`` for calendar ``year`` (cached; parsed with :func:`parse_hissetekil`).

        A body that is not a ``HisseTekil`` JSON payload is removed from the raw layer and raised as
        :class:`IsYatirimError`; an empty ``value`` list is a valid, cached answer (empty frame).
        """
        code = normalise_ticker(ticker)
        url = build_url(code, year, today)
        raw_path = self.raw_path(code, year)
        cached = raw_path.exists() and not force
        t0 = time.monotonic()
        payload = self._get_cached(url, raw_path, force)
        try:
            frame = parse_hissetekil(payload, code)
        except IsYatirimError as exc:
            log.warning("%s %d: %s; discarding %s", code, year, exc, raw_path)
            raw_path.unlink(missing_ok=True)
            raise IsYatirimError(f"{code} {year}: {exc}") from exc
        log.log(
            10 if cached else 20,
            "%s %d: %d rows, %d bytes (%s%s)",
            code,
            year,
            len(frame),
            len(payload),
            "cache" if cached else "fetched",
            "" if cached else f" in {time.monotonic() - t0:.1f}s",
        )
        return frame

    def load_cached(self, ticker: str, year: int) -> pd.DataFrame | None:
        """Parse a cached year without touching the network; ``None`` when it was never fetched."""
        raw_path = self.raw_path(ticker, year)
        if not raw_path.exists():
            return None
        from bistmas.shared.store import read_raw_gzip

        return parse_hissetekil(read_raw_gzip(raw_path), normalise_ticker(ticker))

    def fetch_history(
        self, ticker: str, start_year: int, end_year: int | None = None, force: bool = False
    ) -> pd.DataFrame:
        """Concatenate ``fetch_year`` over ``start_year..end_year`` (default: this year), one row per date."""
        end_year = end_year or date.today().year
        frames = [self.fetch_year(ticker, year, force=force) for year in range(int(start_year), int(end_year) + 1)]
        return concat_years(frames, normalise_ticker(ticker))


def concat_years(frames: list[pd.DataFrame], ticker: str = "") -> pd.DataFrame:
    """Concatenate parsed yearly frames, sorted by date with duplicates (overlapping requests) dropped."""
    parts = [f for f in frames if len(f)]
    if not parts:
        return _empty_frame(ticker)
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values("date", kind="stable").drop_duplicates(subset="date", keep="last")
    return out.reset_index(drop=True)


def iter_cached_files(raw_dir: Path) -> Iterator[tuple[str, int, Path]]:
    """Yield ``(ticker, year, path)`` for every cached ``<raw_dir>/isyatirim/<TICKER>/<YYYY>.json.gz``."""
    source_dir = Path(raw_dir) / SOURCE_NAME
    if not source_dir.exists():
        return
    for ticker_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
        for path in sorted(ticker_dir.glob("*.json.gz")):
            stem = path.name[: -len(".json.gz")]
            if stem.isdigit():
                yield ticker_dir.name, int(stem), path
