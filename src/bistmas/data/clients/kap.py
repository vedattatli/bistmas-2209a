"""KAP (Public Disclosure Platform, kap.org.tr) client: company master, sector tree, member lookup.

KAP's website is a Next.js app. The reference pages used here render their data server-side and
ship it to the browser as React Flight chunks inside ``<script>self.__next_f.push([1,"..."])</script>``
tags. Each pushed string is a JavaScript string literal (JSON escaping plus ``\\u003c``/``\\u003e``/
``\\u0026`` for ``<``, ``>``, ``&``). Concatenating the decoded chunks gives the Flight payload, a text of
``<id>:<json>`` lines; the interesting props are plain JSON inside those lines. The extractor below
decodes the literals and then *JSON-decodes the object array* starting at the ``"data":`` key that
precedes the first record, instead of regexing over the whole page.

Pages and routes (all verified live on 2026-09-23):

* ``/tr/bist-sirketler`` — 758 listed companies grouped by initial letter:
  ``{"data":[{"code":"A","content":[{mkkMemberOid, kapMemberTitle, relatedMemberTitle (auditor),
  stockCode, cityName, relatedMemberOid, kapMemberType}, ...]}, ...]}``. ``stockCode`` may hold several
  comma-separated codes (share classes or bank codes, e.g. ``"KRDMA, KRDMB, KRDMD"``); they are exploded.
* ``/tr/Sektorler`` — 626 (company, code) rows in a two-level tree:
  ``{"data":[{title, children: {sectorOid: {title, children, content:[{sectorName, sectorOid, sectorNo,
  mkkMemberOid, stockCode, title, kapTypes}]}}, content:[{mainSectorName, mainSectorOid, mainSectorNo,
  mkkMemberOid, stockCode, title, kapTypes}]}]}``.
* ``GET /tr/api/member/filter/{text}`` — free-text member search returning
  ``[{companyCode, mkkMemberOid, title, permaLink}]`` (no stock code in the response; a ticker such as
  ``GARAN`` also matches every company whose title contains "GARANTİ").

Access policy (docs/data_governance.md): GET routes only, ≤ 2 requests/s (``PoliteSession(min_interval_s=0.5)``),
every response cached once under ``data/raw/kap`` and never re-fetched unless ``force=True``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from bistmas.data.clients.http import PoliteSession, RateLimitedError
from bistmas.shared.log import get_logger
from bistmas.shared.store import append_manifest, read_raw_gzip, write_raw_gzip

log = get_logger("data.kap")

KAP_BASE_URL = "https://www.kap.org.tr"
COMPANIES_URL = f"{KAP_BASE_URL}/tr/bist-sirketler"
SECTORS_URL = f"{KAP_BASE_URL}/tr/Sektorler"
MEMBER_FILTER_URL = f"{KAP_BASE_URL}/tr/api/member/filter/{{ticker}}"

DEFAULT_HEADERS: dict[str, str] = {
    "Referer": f"{KAP_BASE_URL}/tr/bildirim-sorgu",
    "Origin": KAP_BASE_URL,
    "Accept": "application/json,text/plain,*/*",
}

COMPANY_COLUMNS = ["mkk_member_oid", "ticker", "title", "city", "member_type", "auditor_title", "auditor_oid"]
SECTOR_COLUMNS = [
    "mkk_member_oid",
    "ticker",
    "title",
    "main_sector",
    "main_sector_no",
    "sector",
    "sector_no",
    "sector_oid",
]

# Raw-layer layout below ``raw_dir`` (gzip bodies + manifest.jsonl written by PoliteSession.get_cached).
RAW_PAGES_DIR = "pages"
RAW_MEMBER_DIR = "member"
RAW_MANIFEST = "manifest.jsonl"

_PUSH_MARKER = "self.__next_f.push("
_PUSH_HEAD_RE = re.compile(r"\[\s*(\d+)\s*(?:,\s*\"|\])")
_JS_STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"', re.S)
_JS_HEX_ESCAPE_RE = re.compile(r"\\x([0-9a-fA-F]{2})")
_JS_SINGLE_ESCAPE_RE = re.compile(r"\\(['/])")
_DECODER = json.JSONDecoder()


BYCRITERIA_URL = KAP_BASE_URL + "/{lang}/api/disclosure/members/byCriteria"
ATTACHMENT_DETAIL_URL = KAP_BASE_URL + "/tr/api/notification/attachment-detail/{index}"
FILE_DOWNLOAD_URL = KAP_BASE_URL + "/tr/api/file/download/{obj_id}"
RAW_DISCLOSURES_DIR = "disclosures"
RAW_ATTACHMENT_DIR = "attachment"
RAW_FILES_DIR = "files"
DISCLOSURE_CAP = 2000
FR_SUBJECTS = frozenset({"financial report", "finansal rapor"})


class KapParseError(ValueError):
    """The page does not contain the expected embedded JSON (layout change or truncated download)."""


# ----------------------------------------------------------------------------- Next.js Flight extraction
@dataclass(frozen=True)
class FlightChunk:
    """One ``self.__next_f.push([kind, text])`` call; ``complete`` is False when the page ended mid-string."""

    kind: int
    text: str
    complete: bool = True


def _decode_js_string(raw: str) -> str:
    """Decode the body of a JavaScript string literal (JSON escapes, plus ``\\x..``, ``\\'``, ``\\/``)."""
    try:
        return json.loads(f'"{raw}"')
    except json.JSONDecodeError:
        pass
    fixed = raw
    trailing = len(fixed) - len(fixed.rstrip("\\"))
    if trailing % 2 == 1:  # truncated download ended on a lone backslash
        fixed = fixed[:-1]
    fixed = _JS_HEX_ESCAPE_RE.sub(lambda m: f"\\u00{m.group(1)}", fixed)
    fixed = _JS_SINGLE_ESCAPE_RE.sub(r"\1", fixed)
    try:
        return json.loads(f'"{fixed}"')
    except json.JSONDecodeError as exc:
        raise KapParseError(f"cannot decode __next_f string literal: {exc}") from exc


def iter_flight_chunks(html: str) -> list[FlightChunk]:
    """Return every ``self.__next_f.push([...])`` chunk in page order.

    Pushes without a string (``push([0])``) yield an empty text. A page truncated inside a string
    literal yields the available prefix with ``complete=False`` so callers can report it.
    """
    chunks: list[FlightChunk] = []
    pos = 0
    while (start := html.find(_PUSH_MARKER, pos)) >= 0:
        head = _PUSH_HEAD_RE.match(html, start + len(_PUSH_MARKER))
        if head is None:
            pos = start + len(_PUSH_MARKER)
            continue
        kind = int(head.group(1))
        if head.group(0).endswith("]"):
            chunks.append(FlightChunk(kind, ""))
            pos = head.end()
            continue
        literal = _JS_STRING_RE.match(html, head.end() - 1)
        if literal is not None:
            chunks.append(FlightChunk(kind, _decode_js_string(literal.group(1))))
            pos = literal.end()
            continue
        # unterminated string: the download was cut off inside this chunk
        chunks.append(FlightChunk(kind, _decode_js_string(html[head.end() :]), complete=False))
        break
    return chunks


def flight_payload(html: str) -> str:
    """Concatenate the text chunks (kind 1) of a Next.js page into the React Flight payload."""
    chunks = iter_flight_chunks(html)
    if not chunks:
        raise KapParseError("no self.__next_f.push(...) chunks found; not a Next.js page?")
    if not all(c.complete for c in chunks):
        log.warning("Next.js page is truncated inside a __next_f chunk; extraction may fail")
    return "".join(c.text for c in chunks if c.kind == 1)


def extract_json_value(payload: str, anchor: str, container: str) -> Any:
    """JSON-decode the value that follows the last ``container`` key before the first ``anchor``.

    Example: ``extract_json_value(payload, '"kapMemberTitle"', '"data":')`` returns the company groups.
    The value is decoded with ``JSONDecoder.raw_decode`` so the surrounding Flight/React markup is
    never touched; a layout change or truncated page raises ``KapParseError``.
    """
    i = payload.find(anchor)
    if i < 0:
        raise KapParseError(f"anchor {anchor!r} not found in the Flight payload")
    j = payload.rfind(container, 0, i)
    if j < 0:
        raise KapParseError(f"container {container!r} not found before {anchor!r}")
    k = j + len(container)
    while k < len(payload) and payload[k].isspace():
        k += 1
    try:
        value, _ = _DECODER.raw_decode(payload, k)
    except json.JSONDecodeError as exc:
        raise KapParseError(f"cannot decode JSON after {container!r} (truncated page?): {exc}") from exc
    return value


# ----------------------------------------------------------------------------- page parsers
def _split_codes(value: Any) -> list[str]:
    """``"KRDMA, KRDMB, KRDMD"`` -> ``["KRDMA", "KRDMB", "KRDMD"]``; None/empty -> ``[]``."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [c.strip() for c in str(value).split(",") if c.strip()]


def parse_companies_page(html: str) -> list[dict[str, Any]]:
    """Return the raw company records (one per KAP member, ``stockCode`` not yet exploded)."""
    groups = extract_json_value(flight_payload(html), '"kapMemberTitle"', '"data":')
    if not isinstance(groups, list):
        raise KapParseError("company data is not a list of letter groups")
    rows: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        rows.extend(item for item in (group.get("content") or []) if isinstance(item, dict))
    if not rows:
        raise KapParseError("company data holds no records")
    return rows


def companies_frame(records: Iterable[dict[str, Any]], member_types: Sequence[str] | None = ("IGS",)) -> pd.DataFrame:
    """Build the companies DataFrame (one row per stock code) from raw company records.

    ``member_types`` keeps only those ``kapMemberType`` values (``IGS`` = listed equity issuer); ``None`` keeps all.
    """
    rows = []
    for rec in records:
        member_type = rec.get("kapMemberType")
        if member_types is not None and member_type not in member_types:
            continue
        for code in _split_codes(rec.get("stockCode")):
            rows.append(
                {
                    "mkk_member_oid": rec.get("mkkMemberOid"),
                    "ticker": code,
                    "title": rec.get("kapMemberTitle"),
                    "city": rec.get("cityName"),
                    "member_type": member_type,
                    "auditor_title": rec.get("relatedMemberTitle"),
                    "auditor_oid": rec.get("relatedMemberOid"),
                }
            )
    df = pd.DataFrame(rows, columns=COMPANY_COLUMNS).astype("str")
    df = df.drop_duplicates().sort_values(["ticker", "mkk_member_oid"], kind="stable").reset_index(drop=True)
    return df


def parse_sectors_page(html: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(main_sector_items, sub_sector_items)`` from the ``/tr/Sektorler`` tree."""
    nodes = extract_json_value(flight_payload(html), '"mainSectorName"', '"data":')
    if not isinstance(nodes, list):
        raise KapParseError("sector data is not a list of nodes")
    main_items: list[dict[str, Any]] = []
    sub_items: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        for item in node.get("content") or []:
            if not isinstance(item, dict):
                continue
            if "mainSectorName" in item:
                main_items.append(item)
            elif "sectorName" in item:
                sub_items.append(item)
        children = node.get("children")
        if isinstance(children, dict):
            for child in children.values():
                walk(child)
        elif isinstance(children, list):
            for child in children:
                walk(child)

    for node in nodes:
        walk(node)
    if not main_items:
        raise KapParseError("sector data holds no main-sector records")
    return main_items, sub_items


def sectors_frame(main_items: Iterable[dict[str, Any]], sub_items: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Join main-sector and sub-sector items on (member oid, stock code) and explode multi-code entries."""
    main = pd.DataFrame(
        [
            {
                "mkk_member_oid": x.get("mkkMemberOid"),
                "stock_code": x.get("stockCode"),
                "title": x.get("title"),
                "main_sector": x.get("mainSectorName"),
                "main_sector_no": x.get("mainSectorNo"),
            }
            for x in main_items
        ],
        columns=["mkk_member_oid", "stock_code", "title", "main_sector", "main_sector_no"],
    )
    sub = pd.DataFrame(
        [
            {
                "mkk_member_oid": x.get("mkkMemberOid"),
                "stock_code": x.get("stockCode"),
                "sub_title": x.get("title"),
                "sector": x.get("sectorName"),
                "sector_no": x.get("sectorNo"),
                "sector_oid": x.get("sectorOid"),
            }
            for x in sub_items
        ],
        columns=["mkk_member_oid", "stock_code", "sub_title", "sector", "sector_no", "sector_oid"],
    )
    df = main.merge(sub, on=["mkk_member_oid", "stock_code"], how="outer")
    df["title"] = df["title"].fillna(df["sub_title"])
    df["ticker"] = df["stock_code"].map(_split_codes)
    df = df.explode("ticker")
    df = df[SECTOR_COLUMNS].astype("str")
    df = df.dropna(subset=["ticker"]).drop_duplicates()
    return df.sort_values(["ticker", "mkk_member_oid"], kind="stable").reset_index(drop=True)


def parse_member_filter(body: bytes | str) -> list[dict[str, Any]]:
    """Decode a ``/tr/api/member/filter/{text}`` response (a JSON list, possibly empty)."""
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise KapParseError(f"member filter response is not JSON: {exc}") from exc
    if not isinstance(data, list):
        raise KapParseError("member filter response is not a list")
    return [d for d in data if isinstance(d, dict)]


# ----------------------------------------------------------------------------- client
class KapClient:
    """Cached, paced access to KAP reference data through a :class:`PoliteSession`.

    ``raw_dir`` is the KAP raw-layer directory (``data/raw/kap``). Bodies are stored gzip-compressed
    under ``pages/`` and ``member/`` with a shared ``manifest.jsonl``; cached bodies are never
    re-fetched unless ``force=True``. Use ``PoliteSession(min_interval_s=0.5)`` (≤ 2 GET/s).
    """

    def __init__(self, session: PoliteSession, raw_dir: Path, headers: dict[str, str] | None = None) -> None:
        self.session = session
        self.raw_dir = Path(raw_dir)
        self.headers = {**DEFAULT_HEADERS, **(headers or {})}

    # ------------------------------------------------------------------ raw layer
    @property
    def manifest_path(self) -> Path:
        return self.raw_dir / RAW_MANIFEST

    def _get_cached(self, url: str, raw_path: Path, force: bool = False) -> bytes:
        return self.session.get_cached(
            url, raw_path, self.manifest_path, source="kap", force=force, headers=self.headers
        )

    def _page_path(self, name: str) -> Path:
        return self.raw_dir / RAW_PAGES_DIR / f"{name}.html.gz"

    # ------------------------------------------------------------------ reference data
    def fetch_companies(self, force: bool = False, member_types: Sequence[str] | None = ("IGS",)) -> pd.DataFrame:
        """Listed companies from ``/tr/bist-sirketler``, one row per stock code.

        Columns: ``mkk_member_oid, ticker, title, city, member_type, auditor_title, auditor_oid``.
        Only ``kapMemberType == "IGS"`` (listed equity issuers) by default; ``member_types=None`` keeps all.
        """
        body = self._get_cached(COMPANIES_URL, self._page_path("bist-sirketler"), force=force)
        records = parse_companies_page(body.decode("utf-8"))
        df = companies_frame(records, member_types=member_types)
        log.info("KAP companies: %d members, %d stock codes (member_types=%s)", len(records), len(df), member_types)
        return df

    def fetch_sectors(self, force: bool = False) -> pd.DataFrame:
        """Sector classification from ``/tr/Sektorler``, one row per stock code.

        Columns: ``mkk_member_oid, ticker, title, main_sector, main_sector_no, sector, sector_no, sector_oid``.
        """
        body = self._get_cached(SECTORS_URL, self._page_path("Sektorler"), force=force)
        main_items, sub_items = parse_sectors_page(body.decode("utf-8"))
        df = sectors_frame(main_items, sub_items)
        log.info("KAP sectors: %d main / %d sub records -> %d rows", len(main_items), len(sub_items), len(df))
        return df

    def member_lookup(self, ticker: str, prefer_oid: str | None = None, force: bool = False) -> dict[str, Any] | None:
        """Look a member up by stock code via ``GET /tr/api/member/filter/{ticker}``.

        Returns the matching record (``companyCode, mkkMemberOid, title, permaLink``) or ``None`` when the
        search is empty. The route is a free-text search over titles as well, so when several members
        match, ``prefer_oid`` (e.g. from :meth:`fetch_companies`) selects the right one; otherwise the
        first record is returned with a warning.
        """
        ticker = ticker.strip().upper()
        if not re.fullmatch(r"[A-Z0-9]{2,8}", ticker):
            raise ValueError(f"invalid ticker {ticker!r}")
        raw_path = self.raw_dir / RAW_MEMBER_DIR / f"{ticker}.json.gz"
        body = self._get_cached(MEMBER_FILTER_URL.format(ticker=ticker), raw_path, force=force)
        records = parse_member_filter(body)
        if not records:
            return None
        if prefer_oid is not None:
            for rec in records:
                if rec.get("mkkMemberOid") == prefer_oid:
                    return rec
        if len(records) > 1:
            log.warning(
                "member filter %s returned %d records; taking the first (%s)",
                ticker,
                len(records),
                records[0].get("title"),
            )
        return records[0]

    # ------------------------------------------------------------------ filings and statements (M3)
    def list_disclosures(
        self,
        mkk_member_oid: str,
        from_date: date | str,
        to_date: date | str,
        lang: str = "en",
        force: bool = False,
    ) -> list[dict[str, Any]]:
        """All disclosures of one member between two dates (``POST /{lang}/api/disclosure/members/byCriteria``).

        The response is cached under ``disclosures/<oid>_<from>_<to>.<lang>.json.gz``. The route has a
        strict per-IP quota: a 429 raises :class:`RateLimitedError` immediately (no retry) so the caller
        can checkpoint and resume later. Responses are capped at :data:`DISCLOSURE_CAP` rows — callers
        must split the date range when ``len(rows) >= DISCLOSURE_CAP``.
        """
        frm, to = _iso(from_date), _iso(to_date)
        raw_path = self.raw_dir / RAW_DISCLOSURES_DIR / f"{mkk_member_oid}_{frm}_{to}.{lang}.json.gz"
        if raw_path.exists() and not force:
            return _decode_list(read_raw_gzip(raw_path), raw_path)
        self._warm_up()
        url = BYCRITERIA_URL.format(lang=lang)
        body = {"fromDate": frm, "toDate": to, "mkkMemberOidList": [mkk_member_oid], "subjectList": []}
        resp = self.session.post(
            url, data=json.dumps(body), headers={**self.headers, "Content-Type": "application/json"}
        )
        if resp.status_code == 429:
            raise RateLimitedError(f"KAP byCriteria quota hit for {mkk_member_oid} {frm}..{to}")
        if resp.status_code != 200:
            raise KapParseError(f"byCriteria {resp.status_code} for {mkk_member_oid} {frm}..{to}: {resp.text[:120]}")
        rows = _decode_list(resp.content, raw_path)
        digest = write_raw_gzip(resp.content, raw_path)
        append_manifest(
            self.manifest_path,
            {
                "source": "kap",
                "url": url,
                "oid": mkk_member_oid,
                "from": frm,
                "to": to,
                "rows": len(rows),
                "sha256": digest,
            },
        )
        return rows

    def fetch_attachment_detail(self, disclosure_index: int, force: bool = False) -> dict[str, Any]:
        """``GET /tr/api/notification/attachment-detail/{index}`` -> the first (only) entry of the list.

        Keys: ``disclosure`` (``disclosureBasic`` with publishDate 'YYYY.MM.DD HH:mm:ss', year, donem,
        stockCode, mkkMemberOid), ``disclosureBody`` (list of XBRL-tagged HTML blocks) and
        ``attachments`` (objIds for :meth:`download_attachment`). Cached under ``attachment/<index>.json.gz``.
        """
        raw_path = self.raw_dir / RAW_ATTACHMENT_DIR / f"{int(disclosure_index)}.json.gz"
        body = self._get_cached(ATTACHMENT_DETAIL_URL.format(index=int(disclosure_index)), raw_path, force=force)
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raw_path.unlink(missing_ok=True)
            raise KapParseError(f"attachment-detail {disclosure_index} is not JSON (cache removed): {exc}") from exc
        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            raw_path.unlink(missing_ok=True)
            raise KapParseError(f"attachment-detail {disclosure_index}: unexpected payload (cache removed)")
        return data[0]

    def download_attachment(self, obj_id: str, force: bool = False) -> bytes:
        """``GET /tr/api/file/download/{objId}`` with the Java-serialisation prefix stripped (PDF bytes)."""
        if not re.fullmatch(r"[0-9a-f]{32}", obj_id):
            raise ValueError(f"invalid objId {obj_id!r}")
        raw_path = self.raw_dir / RAW_FILES_DIR / f"{obj_id}.bin.gz"
        body = self._get_cached(FILE_DOWNLOAD_URL.format(obj_id=obj_id), raw_path, force=force)
        i = body.find(b"%PDF")
        if i < 0:
            raw_path.unlink(missing_ok=True)
            raise KapParseError(f"attachment {obj_id} is not a PDF (cache removed)")
        return body[i:]

    def _warm_up(self) -> None:
        """One GET on the search page per session so the POST route sees a normal browser session."""
        if getattr(self, "_warmed", False):
            return
        try:
            self.session.get(f"{KAP_BASE_URL}/tr/bildirim-sorgu", headers=self.headers)
        except Exception as exc:  # noqa: BLE001 - warm-up is best effort
            log.warning("KAP warm-up failed: %s", exc)
        self._warmed = True


def _iso(value: date | str) -> str:
    return value.isoformat() if isinstance(value, date) else str(value)[:10]


def _decode_list(body: bytes, raw_path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise KapParseError(f"{raw_path.name}: response is not JSON: {exc}") from exc
    if isinstance(data, dict) and "success" in data:
        raise KapParseError(f"{raw_path.name}: KAP error payload {data}")
    if not isinstance(data, list):
        raise KapParseError(f"{raw_path.name}: response is not a list")
    return [d for d in data if isinstance(d, dict)]


def financial_report_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the periodic financial-report disclosures (class FR, subject 'Financial Report'/'Finansal Rapor')."""
    keep = []
    for r in rows:
        if str(r.get("disclosureClass", "")).upper() != "FR":
            continue
        if str(r.get("subject", "")).strip().lower() not in FR_SUBJECTS:
            continue
        keep.append(r)
    return keep
