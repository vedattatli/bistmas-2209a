"""Readers for the free Borsa İstanbul DataStore files the project lead downloads by hand.

The DataStore (datastore.borsaistanbul.com) needs an account, so nothing here touches the network.
Files are expected under ``data/raw/datastore/`` exactly as downloaded:

* ``exsrkYYYY.zip`` — product-type 3184 "Constituent Companies of Borsa Istanbul Equity Indices
  (as from year 2000)". Documented fields: ``PAY KODU; PAY ADI; BULUNDUĞU ENDEKS``. The member
  format inside the zip is not documented (CSV with ``;`` in UTF-8 or Windows-1254, or xlsx are
  handled); index names may be Turkish ("BIST 100") or codes ("XU100"); a cell may list several
  indices ("BIST 30, BIST 50, BIST 100"). Whether one zip holds one snapshot or several
  (quarterly) tables is unknown until the first download: every member is read separately and
  labelled (``snapshot_label``) from a date/period column when present, else from digits in the
  member name, else from the zip name.
* Free "Delisted companies", "Equity code changes" and "First trading data" files — format unknown;
  ``read_delisted_file`` / ``read_code_changes_file`` / ``read_first_trading_file`` are tolerant
  generic readers (CSV ``;``/``,``/tab, xlsx, or a zip of those) that map Turkish/English headers
  onto normalised columns and log the mapping so surprises are visible in the run log.

Assumptions to verify on the real files (see the M1 notes): header row may be preceded by title
lines (located by content, not position); dates are ``DD.MM.YYYY`` / ``DD/MM/YYYY`` / ISO or Excel
datetimes; ``.xls`` (BIFF) members cannot be read (openpyxl only) and are reported, not parsed.
"""

from __future__ import annotations

import csv
import io
import re
import unicodedata
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from bistmas.shared.log import get_logger

log = get_logger("data.datastore_files")

DATASTORE_SUBDIR = "datastore"
TABLE_SUFFIXES = {".csv", ".txt", ".xlsx", ".xlsm"}
CONSTITUENT_COLUMNS: tuple[str, ...] = (
    "file_name",
    "member_name",
    "snapshot_label",
    "ticker",
    "security_name",
    "index_code",
)
GENERIC_COLUMNS: tuple[str, ...] = ("ticker", "name", "date", "new_ticker", "new_name", "note", "source_file")

# Turkish index names (normalised with fold_text) -> BIST index codes.
INDEX_NAME_MAP: dict[str, str] = {
    "BIST 100": "XU100",
    "BIST100": "XU100",
    "XU100": "XU100",
    "ULUSAL 100": "XU100",
    "IMKB 100": "XU100",
    "IMKB ULUSAL 100": "XU100",
    "BIST 50": "XU050",
    "BIST50": "XU050",
    "XU050": "XU050",
    "ULUSAL 50": "XU050",
    "IMKB 50": "XU050",
    "IMKB ULUSAL 50": "XU050",
    "BIST 30": "XU030",
    "BIST30": "XU030",
    "XU030": "XU030",
    "ULUSAL 30": "XU030",
    "IMKB 30": "XU030",
    "IMKB ULUSAL 30": "XU030",
    "BIST TUM": "XUTUM",
    "XUTUM": "XUTUM",
    "BIST YILDIZ": "XYLDZ",
    "XYLDZ": "XYLDZ",
}
INDEX_CODE_RE = re.compile(r"^X[A-Z0-9]{4,5}$")
INDEX_SPLIT_RE = re.compile(r"[,;/|]|\s+VE\s+|\s+AND\s+|\s*&\s*")
TICKER_RE = re.compile(r"^[A-Z0-9]{4,6}$")

_TR_FOLD = str.maketrans(
    {
        "İ": "I",
        "ı": "i",
        "Ş": "S",
        "ş": "s",
        "Ğ": "G",
        "ğ": "g",
        "Ü": "U",
        "ü": "u",
        "Ö": "O",
        "ö": "o",
        "Ç": "C",
        "ç": "c",
        "Â": "A",
        "â": "a",
        "Î": "I",
        "î": "i",
    }
)


def fold_text(value: Any) -> str:
    """Upper-case ASCII fold of a header/name (Turkish letters mapped, whitespace collapsed)."""
    text = str(value if value is not None else "").translate(_TR_FOLD)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text).strip().upper()


def normalise_index_code(name: Any) -> str:
    """Map an index label ('BIST 100', 'XU100', 'BIST 100 ENDEKSİ') to its code; unknown labels pass through folded."""
    text = fold_text(name)
    text = re.sub(r"\b(ENDEKSI|ENDEKS|INDEX)\b", "", text).strip()
    text = re.sub(r"\s*-\s*", "-", text)
    if text in INDEX_NAME_MAP:
        return INDEX_NAME_MAP[text]
    compact = text.replace(" ", "")
    if compact in INDEX_NAME_MAP:
        return INDEX_NAME_MAP[compact]
    if INDEX_CODE_RE.match(compact):
        return compact
    return text


def split_index_cell(cell: Any) -> list[str]:
    """Explode 'BIST 30, BIST 50 VE BIST 100' into index codes."""
    text = fold_text(cell)
    if not text:
        return []
    parts = [p.strip() for p in INDEX_SPLIT_RE.split(text) if p and p.strip()]
    return [normalise_index_code(p) for p in parts]


def normalise_ticker(value: Any) -> str:
    """Upper-case equity code without the '.E' share suffix."""
    text = fold_text(value)
    return text.split(".")[0].strip()


# ----------------------------------------------------------------------------- low-level table reading
def decode_text(data: bytes) -> str:
    """UTF-8 (BOM tolerated) first, then Windows-1254, then Latin-1 (never fails)."""
    for encoding in ("utf-8-sig", "cp1254"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def sniff_delimiter(sample: str) -> str:
    """Most frequent of ';', ',', tab, '|' in the first non-empty lines."""
    lines = [line for line in sample.splitlines() if line.strip()][:5]
    head = "\n".join(lines)
    counts = {d: head.count(d) for d in (";", "\t", "|", ",")}
    best = max(counts, key=lambda d: counts[d])
    return best if counts[best] > 0 else ";"


def read_table_bytes(name: str, data: bytes) -> list[list[Any]]:
    """Rows (lists of cells) of a CSV/TXT or XLSX payload; unsupported types raise ValueError."""
    suffix = Path(name).suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        import openpyxl

        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        try:
            rows: list[list[Any]] = []
            for ws in wb.worksheets:
                sheet_rows = [list(r) for r in ws.iter_rows(values_only=True)]
                if sheet_rows:
                    rows.extend(sheet_rows)
                    break  # first non-empty sheet only
            return rows
        finally:
            wb.close()
    if suffix in {".csv", ".txt", ""}:
        text = decode_text(data)
        return [list(r) for r in csv.reader(io.StringIO(text), delimiter=sniff_delimiter(text))]
    raise ValueError(f"unsupported table type {suffix!r} ({name}); .xls needs xlrd which is not installed")


def locate_header(rows: Sequence[Sequence[Any]], keys: Iterable[str]) -> int:
    """Index of the first row whose folded cells contain one of ``keys``; 0 when none matches."""
    wanted = [fold_text(k) for k in keys if k]
    partial_hit: int | None = None
    for i, row in enumerate(rows[:30]):
        cells = [fold_text(c) for c in row]
        if any(c == k for c in cells for k in wanted):
            return i  # a cell equal to a header key: this is the header row
        non_empty = sum(1 for c in cells if c)
        if partial_hit is None and non_empty >= 2 and any(k in c for c in cells for k in wanted):
            partial_hit = i  # substring match only counts on rows with several cells (not a title line)
    return partial_hit if partial_hit is not None else 0


def _is_blank(row: Sequence[Any]) -> bool:
    return all(c is None or str(c).strip() == "" for c in row)


def _at(row: Sequence[Any], i: int | None) -> Any:
    return row[i] if i is not None and i < len(row) else None


def parse_date_cell(value: Any) -> date | None:
    """Best-effort date parsing for DataStore cells; None when empty or unrecognised (logged)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none", "-"}:
        return None
    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%Y%m%d", "%d.%m.%y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    log.warning("unrecognised date cell %r", value)
    return None


@dataclass
class ColumnSpec:
    """Candidate header names (folded) for one normalised column."""

    target: str
    candidates: tuple[str, ...]
    required: bool = False
    matched: str | None = field(default=None, init=False)


def map_columns(header: Sequence[Any], specs: Sequence[ColumnSpec]) -> dict[str, int]:
    """Resolve normalised columns to header positions (exact fold match first, then substring)."""
    folded = [fold_text(h) for h in header]
    used: set[int] = set()
    positions: dict[str, int] = {}
    for spec in specs:
        pos: int | None = None
        for cand in spec.candidates:
            if cand in folded and folded.index(cand) not in used:
                pos = folded.index(cand)
                break
        if pos is None:
            for cand in spec.candidates:
                for i, h in enumerate(folded):
                    if i not in used and h and cand in h:
                        pos = i
                        break
                if pos is not None:
                    break
        if pos is not None:
            positions[spec.target] = pos
            used.add(pos)
            spec.matched = header[pos]
        elif spec.required:
            raise ValueError(f"required column {spec.target!r} not found in header {list(header)}")
    return positions


# ----------------------------------------------------------------------------- constituents (exsrkYYYY.zip)
def _constituent_specs() -> list[ColumnSpec]:
    return [
        ColumnSpec("ticker", ("PAY KODU", "HISSE KODU", "KOD", "EQUITY CODE", "TICKER", "CODE"), required=True),
        ColumnSpec("security_name", ("PAY ADI", "HISSE ADI", "UNVAN", "EQUITY NAME", "NAME", "SIRKET")),
        ColumnSpec(
            "index_code",
            ("BULUNDUGU ENDEKS", "ENDEKS KODU", "ENDEKS", "INDEX CODE", "INDEX", "INDICES"),
            required=True,
        ),
        ColumnSpec("snapshot_label", ("TARIH", "DONEM", "PERIYOT", "DATE", "PERIOD", "YIL")),
    ]


def _label_from_name(member_name: str, zip_name: str) -> str:
    stem = Path(member_name).stem
    m = re.search(r"(\d{4}[_\-]?Q?\d?|\d{8})", stem, flags=re.IGNORECASE)
    if m and m.group(1) != Path(zip_name).stem.replace("exsrk", ""):
        return m.group(1)
    return stem if stem else Path(zip_name).stem


def parse_constituents_table(rows: Sequence[Sequence[Any]], zip_name: str, member_name: str) -> pd.DataFrame:
    """Turn one member's rows into ``CONSTITUENT_COLUMNS`` (one row per ticker x index)."""
    h = locate_header(rows, ["PAY KODU", "HISSE KODU", "EQUITY CODE"])
    header = rows[h]
    specs = _constituent_specs()
    positions = map_columns(header, specs)
    log.info("%s::%s column mapping: %s", zip_name, member_name, {s.target: s.matched for s in specs if s.matched})
    default_label = _label_from_name(member_name, zip_name)
    records: list[dict[str, Any]] = []
    for row in rows[h + 1 :]:
        if _is_blank(row):
            continue
        ticker = normalise_ticker(_at(row, positions.get("ticker")))
        if not ticker:
            continue
        label_cell = _at(row, positions.get("snapshot_label"))
        label = default_label
        if label_cell is not None and str(label_cell).strip():
            parsed = parse_date_cell(label_cell) if not isinstance(label_cell, int | float) else None
            label = parsed.isoformat() if parsed else str(label_cell).strip()
        name = _at(row, positions.get("security_name"))
        codes = split_index_cell(_at(row, positions.get("index_code"))) or [""]
        for code in codes:
            records.append(
                {
                    "file_name": zip_name,
                    "member_name": member_name,
                    "snapshot_label": label,
                    "ticker": ticker,
                    "security_name": None if name is None else str(name).strip(),
                    "index_code": code,
                }
            )
    df = pd.DataFrame.from_records(records, columns=list(CONSTITUENT_COLUMNS))
    for col in CONSTITUENT_COLUMNS:
        df[col] = df[col].astype("string")
    odd = df.loc[~df["ticker"].str.match(TICKER_RE.pattern), "ticker"].unique().tolist()
    if odd:
        log.warning("%s::%s: %d odd equity codes: %s", zip_name, member_name, len(odd), odd[:10])
    unmapped = sorted(set(df["index_code"].dropna()) - set(INDEX_NAME_MAP.values()))
    if unmapped:
        log.warning(
            "%s::%s: index labels not in INDEX_NAME_MAP kept verbatim: %s", zip_name, member_name, unmapped[:10]
        )
    return df


def read_constituents_zip(path: Path) -> pd.DataFrame:
    """Read one ``exsrkYYYY.zip``; columns ``CONSTITUENT_COLUMNS`` (empty frame when no member is readable)."""
    path = Path(path)
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir() or Path(info.filename).suffix.lower() not in TABLE_SUFFIXES:
                if not info.is_dir():
                    log.warning("%s: member %s skipped (unsupported type)", path.name, info.filename)
                continue
            try:
                rows = read_table_bytes(info.filename, zf.read(info))
                frames.append(parse_constituents_table(rows, path.name, info.filename))
            except Exception as exc:  # noqa: BLE001 - keep going, report the member
                log.warning("%s: member %s could not be parsed: %s", path.name, info.filename, exc)
    if not frames:
        return pd.DataFrame({c: pd.Series(dtype="string") for c in CONSTITUENT_COLUMNS})
    df = pd.concat(frames, ignore_index=True)
    log.info(
        "%s: %d rows, snapshots %s, XU100 rows per snapshot %s",
        path.name,
        len(df),
        sorted(df["snapshot_label"].unique().tolist()),
        df[df["index_code"] == "XU100"].groupby("snapshot_label").size().to_dict(),
    )
    return df


# ----------------------------------------------------------------------------- generic files
_GENERIC_SPECS: dict[str, list[tuple[str, tuple[str, ...], bool]]] = {
    "delisted": [
        ("ticker", ("PAY KODU", "HISSE KODU", "ESKI KOD", "KOD", "EQUITY CODE", "TICKER", "CODE"), True),
        ("name", ("PAY ADI", "HISSE ADI", "SIRKET UNVANI", "UNVAN", "SIRKET", "EQUITY NAME", "COMPANY", "NAME"), False),
        (
            "date",
            (
                "KOTTAN CIKARILMA TARIHI",
                "CIKARILMA TARIHI",
                "ISLEM GORMEME TARIHI",
                "SON ISLEM TARIHI",
                "TARIH",
                "DELISTING DATE",
                "LAST TRADING DATE",
                "DATE",
            ),
            False,
        ),
        ("note", ("NEDEN", "ACIKLAMA", "GEREKCE", "REASON", "NOTE", "PAZAR", "MARKET"), False),
    ],
    "code_changes": [
        ("ticker", ("ESKI PAY KODU", "ESKI KOD", "ESKI HISSE KODU", "OLD CODE", "OLD EQUITY CODE", "PAY KODU"), True),
        ("new_ticker", ("YENI PAY KODU", "YENI KOD", "YENI HISSE KODU", "NEW CODE", "NEW EQUITY CODE"), False),
        ("name", ("ESKI PAY ADI", "ESKI UNVAN", "OLD NAME", "OLD EQUITY NAME", "PAY ADI", "UNVAN"), False),
        ("new_name", ("YENI PAY ADI", "YENI UNVAN", "NEW NAME", "NEW EQUITY NAME"), False),
        ("date", ("DEGISIKLIK TARIHI", "GECERLILIK TARIHI", "TARIH", "CHANGE DATE", "EFFECTIVE DATE", "DATE"), False),
        ("note", ("ACIKLAMA", "NEDEN", "NOTE", "REASON"), False),
    ],
    "first_trading": [
        ("ticker", ("PAY KODU", "HISSE KODU", "KOD", "EQUITY CODE", "TICKER", "CODE"), True),
        ("name", ("PAY ADI", "HISSE ADI", "UNVAN", "SIRKET", "EQUITY NAME", "COMPANY", "NAME"), False),
        (
            "date",
            ("ILK ISLEM TARIHI", "ISLEM TARIHI", "HALKA ARZ TARIHI", "TARIH", "FIRST TRADING DATE", "IPO DATE", "DATE"),
            False,
        ),
        ("note", ("PAZAR", "MARKET", "ACIKLAMA", "NOTE", "HALKA ARZ FIYATI", "IPO PRICE"), False),
    ],
}


def _table_payloads(path: Path) -> list[tuple[str, bytes]]:
    """(name, bytes) of every table inside ``path`` (a zip yields its table members)."""
    path = Path(path)
    if path.suffix.lower() == ".zip":
        out: list[tuple[str, bytes]] = []
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if not info.is_dir() and Path(info.filename).suffix.lower() in TABLE_SUFFIXES:
                    out.append((info.filename, zf.read(info)))
        return out
    return [(path.name, path.read_bytes())]


def read_generic_table(path: Path, kind: str) -> pd.DataFrame:
    """Tolerant reader for the delisted / code-change / first-trading files.

    Columns ``GENERIC_COLUMNS``: ``ticker``, ``name``, ``date`` (``datetime.date`` or None),
    ``new_ticker`` and ``new_name`` (code changes only, else None), ``note`` and ``source_file``.
    Unmapped source columns are appended as ``extra_<folded header>`` so nothing is lost. The header
    mapping is logged at INFO.
    """
    if kind not in _GENERIC_SPECS:
        raise ValueError(f"unknown table kind {kind!r}; expected one of {sorted(_GENERIC_SPECS)}")
    path = Path(path)
    frames: list[pd.DataFrame] = []
    for name, data in _table_payloads(path):
        rows = read_table_bytes(name, data)
        if not rows:
            continue
        specs = [ColumnSpec(t, c, r) for t, c, r in _GENERIC_SPECS[kind]]
        h = locate_header(rows, [c for _, cands, _ in _GENERIC_SPECS[kind] for c in cands[:3]])
        header = rows[h]
        positions = map_columns(header, specs)
        mapping = {s.target: s.matched for s in specs if s.matched}
        extras = {
            f"extra_{fold_text(hdr).lower().replace(' ', '_')}": i
            for i, hdr in enumerate(header)
            if i not in positions.values() and fold_text(hdr)
        }
        log.info("%s (%s): column mapping %s; extras %s", name, kind, mapping, sorted(extras))
        records: list[dict[str, Any]] = []
        for row in rows[h + 1 :]:
            if _is_blank(row):
                continue
            ticker = normalise_ticker(_at(row, positions.get("ticker")))
            if not ticker:
                continue
            rec: dict[str, Any] = {
                "ticker": ticker,
                "name": _clean_str(_at(row, positions.get("name"))),
                "date": parse_date_cell(_at(row, positions.get("date"))),
                "new_ticker": normalise_ticker(_at(row, positions.get("new_ticker"))) or None,
                "new_name": _clean_str(_at(row, positions.get("new_name"))),
                "note": _clean_str(_at(row, positions.get("note"))),
                "source_file": f"{path.name}::{name}" if path.suffix.lower() == ".zip" else path.name,
            }
            for col, i in extras.items():
                rec[col] = _clean_str(_at(row, i))
            records.append(rec)
        frames.append(pd.DataFrame.from_records(records, columns=list(GENERIC_COLUMNS) + sorted(extras)))
    if not frames:
        return pd.DataFrame({c: pd.Series(dtype="string" if c != "date" else "object") for c in GENERIC_COLUMNS})
    df = pd.concat(frames, ignore_index=True)
    for col in df.columns:
        if col != "date":
            df[col] = df[col].astype("string")
    return df


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def read_delisted_file(path: Path) -> pd.DataFrame:
    """Companies whose equities were permanently delisted: [ticker, name, date, ..., note, source_file]."""
    return read_generic_table(path, "delisted")


def read_code_changes_file(path: Path) -> pd.DataFrame:
    """Equity code changes: [ticker (old), name (old), date, new_ticker, new_name, note, source_file]."""
    return read_generic_table(path, "code_changes")


def read_first_trading_file(path: Path) -> pd.DataFrame:
    """First trading dates (IPOs, transfers): [ticker, name, date, ..., note, source_file]."""
    return read_generic_table(path, "first_trading")


# ----------------------------------------------------------------------------- directory loader
def classify_datastore_file(name: str) -> str | None:
    """Bucket a file name into constituents / delisted / code_changes / first_trading, or None."""
    stem = fold_text(Path(name).stem).lower()
    if Path(name).suffix.lower() not in TABLE_SUFFIXES | {".zip"}:
        return None
    if re.match(r"^exsrk\d{4}", stem):
        return "constituents"
    if any(k in stem for k in ("delist", "kottan", "kotdan", "cikar", "cikan", "ihrac")):
        return "delisted"
    if ("kod" in stem or "code" in stem) and any(k in stem for k in ("degis", "chang", "change")):
        return "code_changes"
    if any(k in stem for k in ("ilk", "first", "halka")):
        return "first_trading"
    return None


def load_datastore_dir(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Read every recognised file under ``<raw_dir>/datastore``; ``{}`` when the folder is absent.

    Keys present only when files were found: ``constituents`` (all exsrk zips concatenated),
    ``delisted``, ``code_changes``, ``first_trading``. Unrecognised files are logged and ignored.
    """
    folder = Path(raw_dir) / DATASTORE_SUBDIR
    if not folder.is_dir():
        log.info("no DataStore folder at %s (download the free files there by hand)", folder)
        return {}
    buckets: dict[str, list[pd.DataFrame]] = {}
    readers = {
        "constituents": read_constituents_zip,
        "delisted": read_delisted_file,
        "code_changes": read_code_changes_file,
        "first_trading": read_first_trading_file,
    }
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        kind = classify_datastore_file(path.name)
        if kind is None:
            log.warning("%s: not recognised as a DataStore table; ignored", path.name)
            continue
        try:
            buckets.setdefault(kind, []).append(readers[kind](path))
        except Exception as exc:  # noqa: BLE001 - one bad file must not hide the others
            log.warning("%s (%s) could not be read: %s", path.name, kind, exc)
    return {kind: pd.concat(frames, ignore_index=True) for kind, frames in buckets.items() if frames}


# ----------------------------------------------------------------------------- constituents, matrix layout (real files)
# The files actually served by DataStore (product type 3184, downloaded 2026-09-24) are one workbook per year:
# row 3 = header ``PAY KODU | PAY ADI | <effective date> | <effective date> ...`` (quarter starts plus any
# mid-period change dates), one row per equity, cell = the NARROWEST index the equity belonged to from that
# date (``XU030`` implies XU050 and XU100). ``.xls`` (2018 and earlier) needs ``xlrd``.
NESTED_INDICES: dict[str, tuple[str, ...]] = {
    "XU030": ("XU030", "XU050", "XU100"),
    "XU050": ("XU050", "XU100"),
    "XU100": ("XU100",),
}
MATRIX_COLUMNS: tuple[str, ...] = ("file_name", "snapshot_date", "ticker", "security_name", "index_code", "narrowest")


def read_constituent_matrix(path: Path, expand_nested: bool = True) -> pd.DataFrame:
    """Read one real ``exsrkYYYY.zip`` (matrix layout) into one row per (snapshot date, ticker, index)."""
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        members = [i for i in zf.infolist() if Path(i.filename).suffix.lower() in {".xls", ".xlsx", ".xlsm"}]
        if not members:
            return pd.DataFrame({c: pd.Series(dtype="object") for c in MATRIX_COLUMNS})
        raw = zf.read(members[0])
        member_name = members[0].filename
    engine = "xlrd" if member_name.lower().endswith(".xls") else "openpyxl"
    frame = pd.read_excel(io.BytesIO(raw), header=None, engine=engine)
    header_row = None
    for i in range(min(10, len(frame))):
        first = str(frame.iat[i, 0]).upper()
        if "PAY KODU" in first or "CODE" in first:
            header_row = i
            break
    if header_row is None:
        raise ValueError(f"{path.name}: header row with PAY KODU not found")
    header = frame.iloc[header_row].tolist()
    date_cols: list[tuple[int, date]] = []
    for j, cell in enumerate(header[2:], start=2):
        d = parse_date_cell(cell)
        if d is not None:
            date_cols.append((j, d))
    records: list[dict[str, Any]] = []
    for _, row in frame.iloc[header_row + 1 :].iterrows():
        ticker = normalise_ticker(row.iloc[0])
        if not ticker:
            continue
        name = row.iloc[1]
        name = None if pd.isna(name) else str(name).strip()
        for j, d in date_cols:
            cell = row.iloc[j]
            if pd.isna(cell) or not str(cell).strip():
                continue
            narrowest = normalise_index_code(str(cell).strip())
            codes = NESTED_INDICES.get(narrowest, (narrowest,)) if expand_nested else (narrowest,)
            for code in codes:
                records.append(
                    {
                        "file_name": path.name,
                        "snapshot_date": d,
                        "ticker": ticker,
                        "security_name": name,
                        "index_code": code,
                        "narrowest": narrowest,
                    }
                )
    out = pd.DataFrame.from_records(records, columns=list(MATRIX_COLUMNS))
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"])
    return out


def load_constituent_matrices(directory: Path, years: Iterable[int] | None = None) -> pd.DataFrame:
    """Every ``*exsrkYYYY.zip`` under ``directory``; when a year was downloaded several times (DataStore
    re-publishes the yearly file after each change) only the newest download (highest leading sequence
    number, else lexicographically last name) is used, because later versions contain every column of the
    earlier ones."""
    directory = Path(directory)
    by_year: dict[int, Path] = {}
    for p in sorted(directory.glob("*exsrk*.zip")):
        m = re.search(r"exsrk(\d{4})", p.name)
        if not m:
            continue
        year = int(m.group(1))
        if years is not None and year not in set(years):
            continue
        seq = re.match(r"(\d+)_", p.name)
        key = (int(seq.group(1)) if seq else 0, p.name)
        cur = by_year.get(year)
        cur_seq = re.match(r"(\d+)_", cur.name) if cur else None
        cur_key = (int(cur_seq.group(1)) if cur_seq else 0, cur.name) if cur else None
        # a LOWER sequence number was listed higher in the library = newer download (list is newest-first)
        if cur is None or key < cur_key:
            by_year[year] = p
    frames = [read_constituent_matrix(p) for _, p in sorted(by_year.items())]
    if not frames:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in MATRIX_COLUMNS})
    out = pd.concat(frames, ignore_index=True)
    log.info(
        "constituent matrices: %d years, %d rows, %d snapshot dates",
        len(frames),
        len(out),
        out["snapshot_date"].nunique(),
    )
    return out
