"""Official Borsa İstanbul index constituents-and-weights files ("anchors").

Borsa İstanbul publishes one file per valuation period, ``endeks_agirlik_ds_genel*.csv|xlsx`` (index
weightings, BIST series), listing every member of every stock index with its free-float ratio,
share count, weighting factor and weight. The live file is overwritten every quarter, so historical
copies come from the Wayback Machine. These full lists anchor the roll-forward of the quarterly
announcements (``bist_announcements.py``) and let the universe builder verify its reconstruction.

File facts (inspected 2026-09-23 on the real files; see ``ANCHOR_SPECS``):

* Two header lines: Turkish names, then English names. Data rows use ``;`` (CSV) or one row per
  xlsx line. Column sets vary by vintage, so columns are mapped **by Turkish header name**:

  ==================================================  ================================  ================
  Turkish header                                      English header                    canonical column
  ==================================================  ================================  ================
  TARIH                                               DATE                              file_date
  ENDEKS KODU                                         INDEX CODE                        index_code
  ENDEKSLER                                           INDEX NAMES IN TURKISH            index_name_tr
  ENDEKSLERIN INGILIZCE ISIMLERI                      INDICES                           index_name_en
  ENDEKS KUR TURU                                     INDEX CURRENCY TYPE               index_currency
  ENDEKS KURU                                         INDEX CURRENCY RATE               index_fx_rate
  DUZELTME ONCESI/DUZELTME SONRASI/YENI GUN           BEFORE/AFTER ADJUSTMENT/NEW DAY   adjustment_stage
  PAY KUR TURU                                        EQUITY CURRENCY TYPE              equity_currency
  PAY KURU                                            EQUITY CURRENCY RATE              equity_fx_rate
  PAY KODU                                            EQUITY CODE                       equity_code (e.g. ``THYAO.E``)
  PAY ADI                                             EQUITY NAME                       security_name
  TOPLAM PAY SAYISI                                   TOTAL NUMBER OF SHARES            shares_total
  KAPANIS/AOF                                         CLOSE/ VWAP                       price_basis
  FIYAT (2023-05 xlsx, live csv only)                 PRICE                             price
  FIILI DOLASIMDAKI PAY ORANI                         FREE FLOAT RATIO                  free_float_ratio
  KATSAYI                                             WEIGHTING FACTOR                  weighting_factor
  TOPLAM PIYASA DEGERI (2023-05 xlsx only)            TOTAL MARKET VALUE                market_value_total
  FDP KATSAYILI PIYASA DEGERI (2023-05 xlsx only)     WEIGHTED FREE FLOAT MARKET VALUE  market_value_weighted_free_float
  AGIRLIK (%)                                         WEIGHT (%)                        weight
  ==================================================  ================================  ================

  Derived columns: ``ticker`` = ``equity_code`` without the ``.E`` share suffix;
  ``shares_in_index`` = ``shares_total * free_float_ratio * weighting_factor`` (the free-float,
  factor-adjusted share count that enters the index computation). Columns absent from a vintage are
  present as NaN so every parser returns the same column list (``CANONICAL_COLUMNS``).
* Dates are ``DD/MM/YYYY`` or ``DD.MM.YYYY`` in CSVs and real datetimes in xlsx. They are returned
  as Python ``datetime.date`` objects.
* A file dated on the **last session of a valuation period** (e.g. 30/06/2017) holds the composition
  of the **next** period (effective 2017-07-03). Mid-period snapshots (e.g. 2023-05-31, 2024-02-06)
  hold the composition in effect on that day. ``composition_effective_date`` and
  ``snapshot_valid_from`` encode this.
* The 2017 anchor is served as ``...zip`` but is a RAR archive (``Rar!`` magic); ``bsdtar`` extracts
  its members from stdin. The archive also holds ``endeks_bolen_ds_genel_20170630.csv`` (index
  divisors, not used here).
* Equity codes are 4–5 characters and may contain digits (``A1CAP``, ``A1YEN``), so tickers are
  validated against ``[A-Z0-9]{4,6}`` and only logged when odd, never dropped.
"""

from __future__ import annotations

import csv
import io
import json
import re
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from bistmas.data.clients.http import PoliteSession
from bistmas.shared.calendar import TradingCalendar
from bistmas.shared.log import get_logger

log = get_logger("data.index_anchors")

SOURCE_NAME = "index_anchors"
ANCHOR_TIMEOUT_S = 120.0
CDX_API = "https://web.archive.org/cdx/search/cdx"
TICKER_RE = re.compile(r"^[A-Z0-9]{4,6}$")

AnchorKind = Literal["rar", "xlsx", "csv"]


class AnchorError(RuntimeError):
    """Raised when an anchor cannot be fetched, extracted or parsed."""


@dataclass(frozen=True)
class AnchorSpec:
    """One official constituents-and-weights file.

    Attributes:
        name: unique id; also the raw-layer file stem (``data/raw/index_anchors/<name>.<kind>.gz``).
        url: download URL (Wayback ``id_`` URLs return the original bytes; https is required).
        kind: ``rar`` (archive holding the CSV in ``member``), ``xlsx`` or ``csv``.
        snapshot_date: expected ``TARIH`` of the rows, or None to take it from the file (live file).
        member: archive member to extract when ``kind == "rar"``.
        mandatory: ``load_anchor_memberships`` raises when this anchor cannot be loaded.
        period_end: True when the file is dated on the last session of a valuation period (holds the
            next period's composition); False for a mid-period snapshot; None = decide from the date
            with ``is_period_end_date``.
        note: provenance / caveats for the data-governance table.
    """

    name: str
    url: str
    kind: AnchorKind
    snapshot_date: date | None
    member: str | None = None
    mandatory: bool = False
    period_end: bool | None = None
    note: str = ""

    @property
    def raw_file_name(self) -> str:
        return f"{self.name}.{self.kind}.gz"


def _wayback(timestamp: str, original: str) -> str:
    return f"https://web.archive.org/web/{timestamp}id_/{original}"


ANCHOR_SPECS: tuple[AnchorSpec, ...] = (
    AnchorSpec(
        name="bist_weights_20170630",
        url=_wayback(
            "20170731190725",
            "http://www.borsaistanbul.com/docs/default-source/endeksler/"
            "bist-30-bist-50-bist-100-indices-constituents-and-weights.zip?sfvrsn=20",
        ),
        kind="rar",
        snapshot_date=date(2017, 6, 30),
        member="endeks_agirlik_ds_genel_20170630.csv",
        mandatory=True,
        period_end=True,
        note="Wayback capture of the old-site file; RAR despite the .zip name; 55 indices, XU100=100, "
        "XUTUM=310; composition effective 2017-07-03.",
    ),
    AnchorSpec(
        name="bist_weights_20230531",
        url=_wayback("20240330143720", "https://borsaistanbul.com/files/endeks_agirlik_ds_genel_4.xlsx"),
        kind="xlsx",
        snapshot_date=date(2023, 5, 31),
        period_end=False,
        note="Mid-period snapshot (19-column layout with PRICE and market values); 74 indices, XUTUM=464.",
    ),
    AnchorSpec(
        name="bist_weights_20230929",
        url=_wayback("20240330125346", "https://borsaistanbul.com/files/endeks_agirlik_ds_genel_3.xlsx"),
        kind="xlsx",
        snapshot_date=date(2023, 9, 29),
        period_end=True,
        note="2023Q4 composition (effective 2023-10-02); 76 indices, XUTUM=485.",
    ),
    AnchorSpec(
        name="bist_weights_20231229",
        url=_wayback("20240331074837", "https://borsaistanbul.com/files/endeks_agirlik_ds_genel_1.xlsx"),
        kind="xlsx",
        snapshot_date=date(2023, 12, 29),
        mandatory=True,
        period_end=True,
        note="2024Q1 composition (effective 2024-01-02); 77 indices, XU100=100, XUTUM=521.",
    ),
    AnchorSpec(
        name="bist_weights_20240206",
        url=_wayback("20240530191302", "https://www.borsaistanbul.com/files/endeks_agirlik_ds_genel_5.csv"),
        kind="csv",
        snapshot_date=date(2024, 2, 6),
        period_end=False,
        note="Mid-period snapshot dated 06/02/2024 (2024Q1 in effect); 77 indices.",
    ),
    AnchorSpec(
        name="bist_weights_20250328",
        url=_wayback("20250527024804", "https://www.borsaistanbul.com/files/endeks_agirlik_ds_genel_1.csv"),
        kind="csv",
        snapshot_date=date(2025, 3, 28),
        period_end=True,
        note="2025Q2 composition (dated 28.03.2025, last session before the Ramazan holiday); 79 indices.",
    ),
    AnchorSpec(
        name="bist_weights_20250930",
        url=_wayback("20251220002254", "https://www.borsaistanbul.com/files/endeks_agirlik_ds_genel_1.csv"),
        kind="csv",
        snapshot_date=date(2025, 9, 30),
        period_end=True,
        note="2025Q4 composition (effective 2025-10-01); 79 indices.",
    ),
    AnchorSpec(
        name="bist_weights_live",
        url="https://www.borsaistanbul.com/files/endeks_agirlik_ds_genel.csv",
        kind="csv",
        snapshot_date=None,
        period_end=None,
        note="Live file, overwritten by the exchange every period; cached once (pass force=True to refresh). "
        "On 2026-09-23 it was dated 30/06/2026 (17-column layout with PRICE), 86 indices, XUTUM=567.",
    ),
)

MANDATORY_ANCHOR_NAMES: frozenset[str] = frozenset(s.name for s in ANCHOR_SPECS if s.mandatory)

# Turkish header -> canonical column (see module docstring).
COLUMN_MAP: dict[str, str] = {
    "TARIH": "file_date",
    "ENDEKS KODU": "index_code",
    "ENDEKSLER": "index_name_tr",
    "ENDEKSLERIN INGILIZCE ISIMLERI": "index_name_en",
    "ENDEKS KUR TURU": "index_currency",
    "ENDEKS KURU": "index_fx_rate",
    "DUZELTME ONCESI/DUZELTME SONRASI/YENI GUN": "adjustment_stage",
    "PAY KUR TURU": "equity_currency",
    "PAY KURU": "equity_fx_rate",
    "PAY KODU": "equity_code",
    "PAY ADI": "security_name",
    "TOPLAM PAY SAYISI": "shares_total",
    "KAPANIS/AOF": "price_basis",
    "FIYAT": "price",
    "FIILI DOLASIMDAKI PAY ORANI": "free_float_ratio",
    "KATSAYI": "weighting_factor",
    "TOPLAM PIYASA DEGERI": "market_value_total",
    "FDP KATSAYILI PIYASA DEGERI": "market_value_weighted_free_float",
    "AGIRLIK (%)": "weight",
}
NUMERIC_COLUMNS: tuple[str, ...] = (
    "index_fx_rate",
    "equity_fx_rate",
    "shares_total",
    "price",
    "free_float_ratio",
    "weighting_factor",
    "market_value_total",
    "market_value_weighted_free_float",
    "weight",
)
STRING_COLUMNS: tuple[str, ...] = (
    "index_code",
    "index_name_tr",
    "index_name_en",
    "index_currency",
    "adjustment_stage",
    "equity_currency",
    "equity_code",
    "security_name",
    "price_basis",
)
CANONICAL_COLUMNS: tuple[str, ...] = (
    "file_date",
    "index_code",
    "ticker",
    "equity_code",
    "security_name",
    "weight",
    "shares_total",
    "shares_in_index",
    "free_float_ratio",
    "weighting_factor",
    "price",
    "price_basis",
    "market_value_total",
    "market_value_weighted_free_float",
    "index_name_tr",
    "index_name_en",
    "index_currency",
    "index_fx_rate",
    "equity_currency",
    "equity_fx_rate",
    "adjustment_stage",
)
MEMBERSHIP_COLUMNS: tuple[str, ...] = (
    "snapshot_date",
    "composition_effective_date",
    "index_code",
    "ticker",
    "weight",
    "source_name",
)


# ----------------------------------------------------------------------------- dates
def quarter_start(d: date) -> date:
    """First calendar day of the valuation period (calendar quarter) containing d."""
    return date(d.year, 3 * ((d.month - 1) // 3) + 1, 1)


def next_quarter_start(d: date) -> date:
    """First calendar day of the valuation period following the one containing d."""
    qs = quarter_start(d)
    return date(qs.year + 1, 1, 1) if qs.month == 10 else date(qs.year, qs.month + 3, 1)


def first_session_on_or_after(d: date, calendar: TradingCalendar | None = None) -> date:
    """First trading session on/after d: from the calendar when given, else the first weekday."""
    if calendar is not None:
        return calendar.next_session(d, strict=False)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def composition_effective_date(file_date: date, calendar: TradingCalendar | None = None) -> date:
    """First session of the valuation period after ``file_date``.

    A weights file dated on the last session of a period (30/06/2017, 29/12/2023, ...) lists the
    composition that becomes effective on the first session of the next period. Without a trading
    calendar the first weekday is used (wrong when 1 January or a Bayram day falls on a weekday, e.g.
    2024-01-01); pass the data-derived ``TradingCalendar`` once M2 provides it.
    """
    return first_session_on_or_after(next_quarter_start(file_date), calendar)


def is_period_end_date(file_date: date, tolerance_days: int = 7) -> bool:
    """True when file_date lies in the last ``tolerance_days`` calendar days of a valuation period."""
    return (next_quarter_start(file_date) - file_date).days <= tolerance_days


def snapshot_valid_from(
    file_date: date, period_end: bool | None = None, calendar: TradingCalendar | None = None
) -> date:
    """Date from which the composition listed in a file dated ``file_date`` is in effect.

    Period-end files -> ``composition_effective_date``; mid-period snapshots -> the file date itself
    (the list was observed in effect on that day, which is the only claim the file supports).
    """
    if period_end is None:
        period_end = is_period_end_date(file_date)
    return composition_effective_date(file_date, calendar) if period_end else file_date


# ----------------------------------------------------------------------------- fetch / extract
def _looks_like_weights_text(data: bytes) -> bool:
    head = data[:400].decode("utf-8", errors="replace").upper().lstrip("﻿")
    return head.startswith("TARIH") or head.startswith("DATE")


def validate_anchor_payload(data: bytes, kind: AnchorKind) -> None:
    """Raise AnchorError when ``data`` is not a payload of ``kind`` (Wayback/HTML stubs are common)."""
    if kind == "rar":
        ok = data.startswith(b"Rar!") or data.startswith(b"PK\x03\x04")
    elif kind == "xlsx":
        ok = data.startswith(b"PK\x03\x04")
    elif kind == "csv":
        ok = _looks_like_weights_text(data)
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"unknown anchor kind {kind!r}")
    if not ok:
        raise AnchorError(f"payload is not a {kind} file (first bytes: {data[:80]!r})")


def fetch_anchor(session: PoliteSession, spec: AnchorSpec, raw_dir: Path, force: bool = False) -> bytes:
    """Return the raw bytes of an anchor, from ``data/raw/index_anchors`` when cached, else fetched once.

    The Wayback Machine is slow and sometimes offline; the session's retries and a 120 s timeout cover
    that. An invalid payload (HTML stub) is removed from the raw layer so the next call re-fetches.
    """
    raw_path = Path(raw_dir) / SOURCE_NAME / spec.raw_file_name
    manifest_path = Path(raw_dir) / SOURCE_NAME / "manifest.jsonl"
    cached = raw_path.exists() and not force
    data = session.get_cached(
        spec.url, raw_path, manifest_path, source=SOURCE_NAME, force=force, timeout=ANCHOR_TIMEOUT_S
    )
    try:
        validate_anchor_payload(data, spec.kind)
    except AnchorError as exc:
        log.warning("anchor %s: %s; discarding %s", spec.name, exc, raw_path)
        raw_path.unlink(missing_ok=True)
        raise AnchorError(f"anchor {spec.name}: {exc}") from exc
    log.info("anchor %s: %d bytes (%s)", spec.name, len(data), "cache" if cached else "fetched")
    return data


def _run_bsdtar(args: Sequence[str], data: bytes) -> subprocess.CompletedProcess[bytes]:
    if shutil.which("bsdtar") is None:
        raise AnchorError("bsdtar not found on PATH (macOS ships it in /usr/bin; Linux: apt install libarchive-tools)")
    return subprocess.run(["bsdtar", *args], input=data, capture_output=True, check=False)


def list_archive_members(data: bytes) -> list[str]:
    """Member names of a RAR/ZIP archive held in memory (``bsdtar -tf -``)."""
    proc = _run_bsdtar(["-tf", "-"], data)
    if proc.returncode != 0:
        raise AnchorError(f"bsdtar could not list the archive: {proc.stderr.decode(errors='replace').strip()}")
    return [line for line in proc.stdout.decode("utf-8", errors="replace").splitlines() if line]


def extract_rar_member(data: bytes, member: str) -> bytes:
    """Extract one member of a RAR (or ZIP) archive from memory via ``bsdtar -xOf - <member>``."""
    proc = _run_bsdtar(["-xOf", "-", member], data)
    if proc.returncode != 0 or not proc.stdout:
        raise AnchorError(
            f"bsdtar could not extract {member!r}: {proc.stderr.decode(errors='replace').strip() or 'empty output'}"
        )
    return proc.stdout


# ----------------------------------------------------------------------------- parsing
def decode_text(data: bytes) -> str:
    """Decode CSV bytes: UTF-8 (with/without BOM) first, then Windows-1254 (Turkish)."""
    for encoding in ("utf-8-sig", "cp1254"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _parse_date(value: Any) -> date | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%d/%m/%Y", "%d.%m.%Y", "%Y-%m-%d", "%d-%m-%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise AnchorError(f"unrecognised date {value!r}")


def _to_float(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return float("nan")
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip().replace(" ", "")
    if not text:
        return float("nan")
    if "," in text and "." not in text:
        text = text.replace(",", ".")
    elif "," in text and "." in text:  # 1.234.567,89 style
        text = text.replace(".", "").replace(",", ".")
    try:
        return float(text)
    except ValueError as exc:
        raise AnchorError(f"unparseable number {value!r}") from exc


def _normalise_header(cell: Any) -> str:
    return re.sub(r"\s+", " ", str(cell or "")).strip().upper()


def _cell(cells: Sequence[Any], positions: dict[str, int], col: str) -> Any:
    i = positions.get(col)
    return cells[i] if i is not None and i < len(cells) else None


def _frame_from_rows(header: Sequence[Any], rows: Iterable[Sequence[Any]]) -> pd.DataFrame:
    """Map a Turkish header + data rows onto ``CANONICAL_COLUMNS``."""
    names = [_normalise_header(h) for h in header]
    unknown = [h for h in names if h and h not in COLUMN_MAP]
    if unknown:
        log.info("weights file: unmapped columns ignored: %s", unknown)
    positions = {COLUMN_MAP[h]: i for i, h in enumerate(names) if h in COLUMN_MAP}
    for required in ("file_date", "index_code", "equity_code", "weight"):
        if required not in positions:
            raise AnchorError(f"weights file lacks the {required!r} column; header was {names}")

    records: list[dict[str, Any]] = []
    for row in rows:
        cells = list(row)
        if not cells or all(c is None or str(c).strip() == "" for c in cells):
            continue
        index_code = _normalise_header(_cell(cells, positions, "index_code"))
        if not index_code:
            continue
        rec: dict[str, Any] = {"file_date": _parse_date(_cell(cells, positions, "file_date"))}
        for col in STRING_COLUMNS:
            raw = _cell(cells, positions, col)
            rec[col] = None if raw is None else str(raw).strip()
        rec["index_code"] = index_code
        rec["equity_code"] = (rec["equity_code"] or "").upper()
        rec["ticker"] = rec["equity_code"].split(".")[0]
        for col in NUMERIC_COLUMNS:
            rec[col] = _to_float(_cell(cells, positions, col))
        rec["shares_in_index"] = rec["shares_total"] * rec["free_float_ratio"] * rec["weighting_factor"]
        records.append(rec)

    df = pd.DataFrame.from_records(records, columns=list(CANONICAL_COLUMNS))
    for col in STRING_COLUMNS + ("ticker",):
        df[col] = df[col].astype("string")
    for col in NUMERIC_COLUMNS + ("shares_in_index",):
        df[col] = df[col].astype("float64")
    odd = df.loc[~df["ticker"].fillna("").str.match(TICKER_RE.pattern), "ticker"].unique().tolist()
    if odd:
        log.warning("weights file: %d equity codes do not match %s: %s", len(odd), TICKER_RE.pattern, odd[:10])
    return df


def parse_weights_csv(text: str) -> pd.DataFrame:
    """Parse an ``endeks_agirlik_ds_genel*.csv`` text (two header lines, ``;``-separated)."""
    lines = [line for line in text.lstrip("﻿").splitlines() if line.strip()]
    if len(lines) < 2:
        raise AnchorError("weights csv has no data rows")
    reader = csv.reader(lines, delimiter=";")
    header = next(reader)
    rows = list(reader)
    if rows and _normalise_header(rows[0][0]) == "DATE":  # second (English) header line
        rows = rows[1:]
    return _frame_from_rows(header, rows)


def parse_weights_xlsx(data: bytes) -> pd.DataFrame:
    """Parse an ``endeks_agirlik_ds_genel*.xlsx`` workbook (first sheet, two header rows)."""
    import openpyxl

    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    try:
        ws = wb.worksheets[0]
        it = ws.iter_rows(values_only=True)
        header = next(it, None)
        if header is None:
            raise AnchorError("weights xlsx is empty")
        rows = list(it)
    finally:
        wb.close()
    if rows and _normalise_header(rows[0][0]) == "DATE":
        rows = rows[1:]
    return _frame_from_rows(header, rows)


def parse_anchor(data: bytes, spec: AnchorSpec) -> pd.DataFrame:
    """Turn the raw bytes of ``spec`` into the canonical weights frame (extracting RAR members first)."""
    if spec.kind == "rar":
        if not spec.member:
            raise AnchorError(f"anchor {spec.name}: rar specs need a member name")
        return parse_weights_csv(decode_text(extract_rar_member(data, spec.member)))
    if spec.kind == "xlsx":
        return parse_weights_xlsx(data)
    return parse_weights_csv(decode_text(data))


def weight_sums(df: pd.DataFrame) -> pd.Series:
    """Sum of ``weight`` per (file_date, index_code); each should be ~100 for a complete list."""
    return df.groupby(["file_date", "index_code"], observed=True)["weight"].sum()


# ----------------------------------------------------------------------------- discovery
def discover_wayback_captures(
    session: PoliteSession,
    pattern: str = "borsaistanbul.com/files/endeks_agirlik_ds_genel*",
    drop_html: bool = True,
) -> pd.DataFrame:
    """List Wayback captures matching ``pattern`` via the CDX API.

    Returns columns [timestamp, original, mimetype, statuscode, digest, length, capture_url]; empty
    (with a warning) when the archive is offline or answers garbage, so callers can carry on with
    the static ``ANCHOR_SPECS``. HTML captures are Wayback error stubs and are dropped by default.
    """
    params = {"url": pattern, "output": "json", "filter": "statuscode:200"}
    columns = ["timestamp", "original", "mimetype", "statuscode", "digest", "length", "capture_url"]
    try:
        resp = session.get(CDX_API, params=params, timeout=ANCHOR_TIMEOUT_S)
        resp.raise_for_status()
        payload = json.loads(resp.content.decode("utf-8")) if resp.content.strip() else []
    except Exception as exc:  # noqa: BLE001 - archive outages are expected and non-fatal
        log.warning("CDX lookup for %s failed: %s", pattern, exc)
        return pd.DataFrame(columns=columns)
    if not payload:
        return pd.DataFrame(columns=columns)
    header, *rows = payload
    df = pd.DataFrame(rows, columns=header)
    for col in ("timestamp", "original", "mimetype", "statuscode", "digest", "length"):
        if col not in df.columns:
            df[col] = None
    if drop_html:
        df = df[~df["mimetype"].astype(str).str.startswith("text/html")]
    df = df.copy()
    df["capture_url"] = [_wayback(ts, orig) for ts, orig in zip(df["timestamp"], df["original"], strict=True)]
    return df[columns].reset_index(drop=True)


# ----------------------------------------------------------------------------- loading
def load_anchor_frame(session: PoliteSession, raw_dir: Path, spec: AnchorSpec, force: bool = False) -> pd.DataFrame:
    """Fetch (cached) + parse one anchor into the full canonical weights frame."""
    df = parse_anchor(fetch_anchor(session, spec, raw_dir, force=force), spec)
    dates = df["file_date"].dropna().unique().tolist()
    if len(dates) != 1:
        raise AnchorError(f"anchor {spec.name}: expected one file date, found {dates[:5]}")
    if spec.snapshot_date is not None and dates[0] != spec.snapshot_date:
        log.warning("anchor %s: file is dated %s, spec expected %s", spec.name, dates[0], spec.snapshot_date)
    counts = df["index_code"].value_counts()
    log.info(
        "anchor %s: date %s, %d rows, %d indices, XU100=%d XU050=%d XU030=%d",
        spec.name,
        dates[0],
        len(df),
        counts.size,
        counts.get("XU100", 0),
        counts.get("XU050", 0),
        counts.get("XU030", 0),
    )
    return df


def load_anchor_memberships(
    session: PoliteSession,
    raw_dir: Path,
    specs: Sequence[AnchorSpec] | None = None,
    calendar: TradingCalendar | None = None,
) -> pd.DataFrame:
    """Membership rows of every anchor that can be loaded.

    Columns: ``snapshot_date`` (the file's TARIH, a ``datetime.date``), ``composition_effective_date``
    (see ``snapshot_valid_from``), ``index_code``, ``ticker``, ``weight`` (percent) and ``source_name``
    (the spec name). Optional anchors that fail are logged and skipped; mandatory ones
    (``MANDATORY_ANCHOR_NAMES``) raise ``AnchorError``.
    """
    frames: list[pd.DataFrame] = []
    for spec in specs if specs is not None else ANCHOR_SPECS:
        try:
            df = load_anchor_frame(session, raw_dir, spec)
        except Exception as exc:
            if spec.mandatory:
                raise AnchorError(f"mandatory anchor {spec.name} could not be loaded: {exc}") from exc
            log.warning("optional anchor %s skipped: %s", spec.name, exc)
            continue
        snapshot = df["file_date"].iloc[0]
        valid_from = snapshot_valid_from(snapshot, spec.period_end, calendar)
        out = pd.DataFrame(
            {
                "snapshot_date": [snapshot] * len(df),
                "composition_effective_date": [valid_from] * len(df),
                "index_code": df["index_code"].to_numpy(),
                "ticker": df["ticker"].to_numpy(),
                "weight": df["weight"].to_numpy(),
                "source_name": [spec.name] * len(df),
            }
        )
        frames.append(out)
    if not frames:
        return pd.DataFrame(columns=list(MEMBERSHIP_COLUMNS))
    result = pd.concat(frames, ignore_index=True)
    result["index_code"] = result["index_code"].astype("string")
    result["ticker"] = result["ticker"].astype("string")
    result["source_name"] = result["source_name"].astype("string")
    return result.sort_values(["snapshot_date", "index_code", "ticker"], kind="stable").reset_index(drop=True)
