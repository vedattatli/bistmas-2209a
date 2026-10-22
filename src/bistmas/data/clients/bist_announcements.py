"""Borsa İstanbul index-announcement archive: client and parsers.

The public archive page https://www.borsaistanbul.com/en/indices/index-announcements lists every
index announcement since 2013 (578 rows on 2026-09-23). Quarterly BIST stock-index reviews are:

* 2013Q3 .. 2024Q2 and 2025Q3: borsaistanbul.com pages (``/en/announcement/{id}/{slug}``) with
  inline HTML tables — one table per index, headed "BIST 100 INDEX", "BIST 50 INDEX", ...; each
  data row holds three (no, ticker, company) triplets for STOCKS TO BE INCLUDED / EXCLUDED /
  SUBSTITUTE, empty cells rendered as ``&nbsp;`` (so cells must be mapped by position, never by
  dropping empties).
* 2024Q3 .. 2025Q2 and 2025Q4 onward: KAP disclosures (``https://www.kap.org.tr/en/Bildirim/{idx}``)
  whose lists are only in PDF attachments (``YYYY_Q_Periodic_Changes.pdf`` English,
  ``YYYY_Q_Donemsel_Degisiklikler.pdf`` Turkish). The download body carries a Java-serialisation
  prefix in front of ``%PDF``. Tickers are extracted from pypdf's layout-mode text by regex and
  assigned to columns by horizontal offset, because company names wrap across lines.

Conventions
-----------
* ``announcement_date`` is a ``datetime.date``; the English archive prints MM.DD.YYYY.
* Index codes: BIST 100 -> ``XU100``, BIST 50 -> ``XU050``, BIST 30 -> ``XU030``; every other
  heading is kept verbatim (e.g. ``BIST LIQUID 10 EX BANKS INDEX``); a table whose heading the page
  omitted (2016Q3 second table) gets ``UNKNOWN`` and a warning rather than a guess.
* ``action`` is ``include`` | ``exclude`` | ``substitute``.
* Raw layer: everything is cached under ``raw_dir/bist_announcements/`` through
  :class:`bistmas.data.clients.http.PoliteSession.get_cached` and never fetched twice:
  ``index-announcements.html.gz``, ``announcement_{id}.html.gz`` (borsaistanbul pages),
  ``kap_bildirim_{idx}.html.gz`` (KAP pages) and ``kap_file_{objId}.pdf.gz`` (KAP attachments,
  stored with the Java prefix as served).
* Changes apply on the first session of the quarter, never on the announcement date; this module
  only reports what was announced.
"""

from __future__ import annotations

import io
import re
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import pandas as pd
from lxml import etree
from lxml import html as lh
from pypdf import PdfReader

from bistmas.shared.codes import CORE_INDICES, EQUITY_CODE_PATTERN, EQUITY_CODE_RE
from bistmas.shared.log import get_logger
from bistmas.shared.store import append_manifest, read_raw_gzip, utc_now_iso, write_raw_gzip

if TYPE_CHECKING:
    from lxml.html import HtmlElement

    from bistmas.data.clients.http import PoliteSession

log = get_logger("data.bist_announcements")

ARCHIVE_URL = "https://www.borsaistanbul.com/en/indices/index-announcements"
KAP_FILE_DOWNLOAD_URL = "https://www.kap.org.tr/tr/api/file/download/{obj_id}"
KAP_HEADERS: dict[str, str] = {
    "Referer": "https://www.kap.org.tr/tr/bildirim-sorgu",
    "Origin": "https://www.kap.org.tr",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
RAW_SUBDIR = "bist_announcements"
ARCHIVE_RAW_NAME = "index-announcements.html.gz"

INDEX_CODES: dict[str, str] = {
    "BIST 100 INDEX": "XU100",
    "BIST 50 INDEX": "XU050",
    "BIST 30 INDEX": "XU030",
}
ACTIONS: tuple[str, ...] = ("include", "exclude", "substitute")
UNKNOWN_INDEX = "UNKNOWN"
HOSTS: dict[str, str] = {"borsaistanbul.com": "borsaistanbul", "kap.org.tr": "kap"}

#: Quarters whose announcement page is known to carry no parsable list (2017Q2: the page body is
#: empty apart from the title line). Used by :func:`build_changes_table`.
KNOWN_EMPTY_QUARTERS: frozenset[tuple[int, int]] = frozenset({(2017, 2)})
#: (year, quarter, index_code) combinations whose page is known to lack that index's table even though
#: the review happened: the English 2020Q1 page (rule-change quarter) only carries the BIST 100 table.
#: The missing BIST 30/50 lists must be supplied through configs/universe_overrides.csv.
KNOWN_MISSING_TABLES: frozenset[tuple[int, int, str]] = frozenset({(2020, 1, "XU050"), (2020, 1, "XU030")})

TICKER_RE = EQUITY_CODE_RE
HEADING_RE = re.compile(r"^BIST\s+.+?\s+(?:INDEX(?:ES)?|ENDEKS[İI]?)$", re.I)  # "BIST 30 ENDEKSİ" on the 2019Q4 page
QUARTER_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4}
_STOCK_INDICES_RE = re.compile(r"\bBIST\s+(?:Equity|Stock)\s+Indices\b", re.I)
_QUARTER_RE = re.compile(r"\b(first|second|third|fourth)\s+quarter\s+of\s+(\d{4})\b", re.I)
_PERIODIC_REVIEW_RE = re.compile(r"^BIST\s+Stock\s+Indices\s+Periodic\s+Review$", re.I)
_PAIR_RE = re.compile(rf"(?<![A-Z0-9])(\d{{1,3}})\s+({EQUITY_CODE_PATTERN})(?=\s|$)")
_KAP_ANCHOR_RE = re.compile(r"file/download/([0-9a-f]{32})\"[^>]*>\s*([^<]+?)\s*</a>")
_KAP_JSON_RE = re.compile(r"\\?\"objId\\?\":\\?\"([0-9a-f]{32})\\?\",\\?\"fileName\\?\":\\?\"([^\"\\]+)")
_KAP_ID_RE = re.compile(r"file/download/([0-9a-f]{32})")


# --------------------------------------------------------------------------- helpers
def _raw_dir(raw_dir: Path) -> Path:
    return Path(raw_dir) / RAW_SUBDIR


def _norm(text: str) -> str:
    """Collapse whitespace (incl. NBSP) and strip."""
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _as_text(html: str | bytes) -> str:
    return html.decode("utf-8", "replace") if isinstance(html, bytes) else html


def _host_of(url: str) -> str:
    if not url:
        return "other"
    netloc = urlsplit(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return HOSTS.get(netloc, "other")


def index_code_for(heading: str) -> str:
    """Map a table heading such as 'BIST 100 INDEX' to XU100/XU050/XU030; other headings verbatim."""
    key = _norm(heading).upper().rstrip(":")
    key = re.sub(r"\bENDEKS[İI]?$", "INDEX", key)
    return INDEX_CODES.get(key, key)


def strip_java_prefix(body: bytes) -> bytes:
    """Return ``body`` from the first ``%PDF`` on (KAP downloads wrap the PDF in a Java byte[])."""
    i = body.find(b"%PDF")
    if i < 0:
        raise ValueError("no %PDF header found in payload")
    return body[i:]


def quarter_from_month(year: int, month: int) -> tuple[int, int]:
    """Quarter *announced* by a review published in ``month``: Dec -> Q1 next year, Mar -> Q2, ..."""
    n = month // 3
    return (year + 1, 1) if n == 4 else (year, n + 1)


# --------------------------------------------------------------------------- archive
def fetch_archive(session: PoliteSession, raw_dir: Path, force: bool = False) -> str:
    """Return the archive page HTML, cached at ``raw_dir/bist_announcements/index-announcements.html.gz``.

    A pre-existing uncompressed ``index-announcements.html`` next to it (saved by hand) is imported
    into the gzip cache instead of re-fetching.
    """
    path = _raw_dir(raw_dir) / ARCHIVE_RAW_NAME
    legacy = path.with_suffix("")  # index-announcements.html
    if not path.exists() and not force and legacy.exists():
        payload = legacy.read_bytes()
        digest = write_raw_gzip(payload, path)
        append_manifest(
            path.parent / "manifest.jsonl",
            {
                "source": "bist_announcements",
                "url": ARCHIVE_URL,
                "path": str(path),
                "bytes": len(payload),
                "sha256": digest,
                "status": None,
                "fetched_at": None,
                "note": "imported from pre-existing uncompressed copy",
            },
        )
        log.info("imported saved archive copy %s -> %s", legacy.name, path.name)
        return payload.decode("utf-8", "replace")
    body = session.get_cached(ARCHIVE_URL, path, source="bist_announcements", force=force)
    return body.decode("utf-8", "replace")


def parse_archive(html: str | bytes) -> pd.DataFrame:
    """Parse the archive page into one row per announcement.

    Columns: ``announcement_date`` (datetime.date), ``title``, ``url`` (http -> https normalised),
    ``kind`` ('periodic' | 'non_periodic' | 'other'), ``index_groups`` (str, as printed, e.g.
    'BIST Stock Indices|Benchmark Indices, BIST Stock Indices|Liquid'), ``host``
    ('borsaistanbul' | 'kap' | 'other') and ``announcement_type`` (the raw type text).
    """
    doc = lh.fromstring(_as_text(html))
    rows = doc.xpath('//tr[contains(concat(" ", normalize-space(@class), " "), " table-tr ")]')
    records: list[dict[str, Any]] = []
    for tr in rows:
        date_txt = _norm(_cell_text(tr, "view-field-announcement-date-table-column"))
        title_td = tr.xpath('.//td[@headers="view-title-table-column"]')
        links = title_td[0].xpath(".//a[@href]") if title_td else []
        title = _norm(links[0].text_content() if links else (title_td[0].text_content() if title_td else ""))
        url = _norm(links[0].get("href", "")) if links else ""
        if url.startswith("http://") and _host_of(url) in ("borsaistanbul", "kap"):
            url = "https://" + url[len("http://") :]
        type_txt = _norm(_cell_text(tr, "view-field-announcement-type-id-table-column"))
        groups = _norm(_cell_text(tr, "view-field-ana-endeks-grubu-3-table-column"))
        low = type_txt.lower()
        if "non-periodic" in low:
            kind = "non_periodic"
        elif "periodic" in low:
            kind = "periodic"
        else:
            kind = "other"
        try:
            ann_date = datetime.strptime(date_txt, "%m.%d.%Y").date()
        except ValueError:
            log.warning("archive row with unparsable date %r skipped (%s)", date_txt, title[:60])
            continue
        records.append(
            {
                "announcement_date": ann_date,
                "title": title,
                "url": url,
                "kind": kind,
                "index_groups": groups,
                "host": _host_of(url),
                "announcement_type": type_txt,
            }
        )
    columns = ["announcement_date", "title", "url", "kind", "index_groups", "host", "announcement_type"]
    df = pd.DataFrame.from_records(records, columns=columns)
    log.info("archive: %d announcements parsed", len(df))
    return df


def _cell_text(tr: HtmlElement, header_id: str) -> str:
    cells = tr.xpath(f'.//td[@headers="{header_id}"]')
    return cells[0].text_content() if cells else ""


def _title_quarter(title: str, ann_date: date) -> tuple[int, int] | None:
    """(year, quarter) of a quarterly BIST stock/equity index review title, else None."""
    t = _norm(title).rstrip(". ")
    if _PERIODIC_REVIEW_RE.match(t):
        return quarter_from_month(ann_date.year, ann_date.month)
    if not _STOCK_INDICES_RE.search(t):
        return None
    m = _QUARTER_RE.search(t)
    if not m:
        return None
    return int(m.group(2)), QUARTER_WORDS[m.group(1).lower()]


def select_stock_index_periodic(archive_df: pd.DataFrame, start: tuple[int, int] = (2015, 4)) -> pd.DataFrame:
    """Select the quarterly BIST stock/equity index reviews (one row per quarter) from ``start`` on.

    Matches both title families: "Borsa İstanbul announces (the) constituent changes to the BIST
    (Equity|Stock) Indices for the <n> quarter of YYYY" (any host, incl. the 2015 "... have been
    changed" wording) and the KAP-hosted "BIST Stock Indices Periodic Review" (quarter derived from
    the announcement month). Other index families (BIST-KYD, Buyback, Dividend, ...) are ignored.

    Returns columns ``year, quarter, announcement_date, url, host`` sorted by (year, quarter).
    Raises ``ValueError`` when a (year, quarter) appears twice.
    """
    records: list[dict[str, Any]] = []
    for row in archive_df.itertuples(index=False):
        if row.kind != "periodic":
            continue
        yq = _title_quarter(row.title, row.announcement_date)
        if yq is None or yq < tuple(start):
            continue
        records.append(
            {
                "year": yq[0],
                "quarter": yq[1],
                "announcement_date": row.announcement_date,
                "url": row.url,
                "host": row.host,
            }
        )
    out = pd.DataFrame.from_records(records, columns=["year", "quarter", "announcement_date", "url", "host"])
    dupes = out[out.duplicated(["year", "quarter"], keep=False)]
    if not dupes.empty:
        raise ValueError(
            f"duplicate quarterly reviews in archive: {dupes[['year', 'quarter', 'url']].to_dict('records')}"
        )
    out = out.sort_values(["year", "quarter"], kind="stable").reset_index(drop=True)
    log.info("selected %d quarterly stock-index reviews (%s)", len(out), _span(out))
    return out


def _span(df: pd.DataFrame) -> str:
    if df.empty:
        return "none"
    first, last = df.iloc[0], df.iloc[-1]
    return f"{first.year}Q{first.quarter}..{last.year}Q{last.quarter}"


# --------------------------------------------------------------------------- announcement pages
def announcement_raw_name(url: str) -> str:
    """Raw-cache file name for an announcement URL, derived from the numeric id in the URL."""
    host = _host_of(url)
    if host == "borsaistanbul":
        m = re.search(r"/announcement/(\d+)", url)
        if m:
            return f"announcement_{m.group(1)}.html.gz"
    if host == "kap":
        m = re.search(r"/Bildirim/(\d+)", url, re.I)
        if m:
            return f"kap_bildirim_{m.group(1)}.html.gz"
    slug = re.sub(r"[^A-Za-z0-9]+", "_", urlsplit(url).path).strip("_")[:80]
    return f"page_{slug or 'root'}.html.gz"


def _discard_cached(path: Path, reason: str) -> None:
    """Remove a cached body that failed validation and record the discard in the manifest."""
    if path.exists():
        path.unlink()
    append_manifest(
        path.parent / "manifest.jsonl",
        {"path": str(path), "status": "discarded", "reason": reason, "at": utc_now_iso()},
    )


def validate_page_body(body: bytes, host: str) -> str | None:
    """Return a reason string when ``body`` is not a usable page for ``host`` (WAF challenge, truncation)."""
    if host == "kap":
        if b"__next_f" not in body:
            return "KAP page without Next.js payload (WAF challenge or error page)"
        if b"</html>" not in body[-4096:]:
            return "KAP page truncated (no closing </html>)"
    elif host == "borsaistanbul":
        if b"<main" not in body and b"<table" not in body and b"announcement" not in body.lower():
            return "borsaistanbul page without content (challenge or error page)"
    return None


def fetch_announcement(session: PoliteSession, url: str, raw_dir: Path) -> bytes:
    """Return the announcement page body (borsaistanbul.com or KAP), cached by its URL id.

    Bodies that do not look like a real page are removed from the raw layer and raise, so a WAF
    challenge or truncated download never poisons later runs.
    """
    host = _host_of(url)
    path = _raw_dir(raw_dir) / announcement_raw_name(url)
    kwargs: dict[str, Any] = {"headers": KAP_HEADERS} if host == "kap" else {}
    body = session.get_cached(url, path, source=host, **kwargs)
    reason = validate_page_body(body, host)
    if reason:
        _discard_cached(path, reason)
        raise RuntimeError(f"{url}: {reason}; cached copy {path} removed, re-run to fetch again")
    return body


_HEADER_WORDS = ("STOCKS", "SUBSTITUTE", "PAYLAR")  # English and Turkish column labels


def _action_from_header(text: str) -> str | None:
    t = text.upper().replace("Ç", "C").replace("İ", "I")
    if "INCLUDE" in t or "ALINACAK" in t:
        return "include"
    if "EXCLUDE" in t or "CIKARILACAK" in t:
        return "exclude"
    if "SUBSTITUTE" in t or "YEDEK" in t:
        return "substitute"
    return None


def _content_root(doc: HtmlElement) -> HtmlElement:
    for xp in ("//div[contains(@class, 'announcement-content')]", "//main", "//body"):
        found = doc.xpath(xp)
        if found:
            return found[0]
    return doc


def parse_change_tables(html: str | bytes) -> pd.DataFrame:
    """Parse the include/exclude/substitute tables of a borsaistanbul.com announcement page.

    Headings ("BIST 100 INDEX", ...) are tracked sequentially in document order, whether they sit in
    a paragraph before a table or in a spanning row inside one. Regular data rows (three cells per
    header column) are read by cell position in (no, ticker, company) triplets; the header row of
    each table gives the action of each triplet (default order include, exclude, substitute).
    Irregular rows (left-shifted substitute-only rows on the 2016 pages) are resolved from the row
    number and the cell position. Cells that are empty or not a 4-5 letter ticker are skipped.

    Returns columns ``index_code, action, ticker, company``.
    """
    doc = lh.fromstring(_as_text(html))
    root = _content_root(doc)
    records: list[dict[str, str]] = []
    state = _TableState()
    # One document-order walk: headings may sit in paragraphs, in spanning rows, inside the cells of
    # an outer layout table that wraps the real tables (2022Q1 page) or as bare text between <br>
    # tags (2017Q3 page, seen as an element *tail* at its end event); rows whose cells contain
    # nested tables are containers and are ignored as rows.
    for event, el in etree.iterwalk(root, events=("start", "end")):
        if not isinstance(el.tag, str):
            continue
        if event == "end":
            _maybe_heading(el.tail, state)
            continue
        if el.tag == "table":
            state.new_table()
        elif el.tag == "tr":
            _parse_row(el, state, records)
        else:
            _maybe_heading(el.text, state)
            if el.tag in _HEADING_TAGS and not el.xpath(".//table"):
                _maybe_heading(el.text_content(), state)
    df = pd.DataFrame.from_records(records, columns=["index_code", "action", "ticker", "company"])
    log.debug("parsed %d change records from announcement page", len(df))
    return df


class _TableState:
    """Sequential parser state: current index heading, header actions and per-action row counters."""

    def __init__(self) -> None:
        self.heading: str | None = None
        self.actions: list[str] = list(ACTIONS)
        self.counts: dict[str, int] = {}
        self.consumed = False  # data rows were already read under the current heading

    def new_table(self) -> None:
        self.actions = list(ACTIONS)
        self.counts = {}
        if self.consumed:
            # a second table under one heading: the page omitted the heading (2016Q3 BIST 50 table)
            log.warning("table without its own heading after %r; index_code set to UNKNOWN", self.heading)
            self.heading = UNKNOWN_INDEX
            self.consumed = False

    def new_heading(self, heading: str) -> None:
        self.heading = heading
        self.actions = list(ACTIONS)
        self.counts = {}
        self.consumed = False

    def add(self, records: list[dict[str, str]], action: str, number: str, ticker: str, company: str) -> None:
        if not ticker:
            return
        if not TICKER_RE.match(ticker):
            compact = re.sub(r"[.\s]", "", ticker)  # 2019Q3 prints "A.V.O.D" for AVOD
            if not TICKER_RE.match(compact):
                if not re.fullmatch(r"[-–\s]+", ticker):
                    log.debug("non-ticker cell %r ignored under %r", ticker, self.heading)
                return
            log.warning("ticker cell %r under %r read as %s", ticker, self.heading, compact)
            ticker = compact
        if self.heading is None:
            log.warning("data row before any index heading: %s %s -> index_code UNKNOWN", ticker, company)
            self.heading = UNKNOWN_INDEX
        records.append(
            {"index_code": index_code_for(self.heading), "action": action, "ticker": ticker, "company": company}
        )
        self.counts[action] = max(self.counts.get(action, 0), _row_number(number))
        self.consumed = True


def _row_number(text: str) -> int:
    digits = re.sub(r"\D", "", text)
    return int(digits) if digits else 0


_HEADING_TAGS = frozenset({"p", "h1", "h2", "h3", "h4", "h5", "h6", "div", "span", "strong", "b", "u", "em", "td"})


def _maybe_heading(text: str | None, state: _TableState) -> None:
    if text:
        norm = _norm(text)
        if HEADING_RE.match(norm):
            state.new_heading(norm)


def _parse_row(tr: HtmlElement, state: _TableState, records: list[dict[str, str]]) -> None:
    cells = [c for c in tr if isinstance(c.tag, str) and c.tag in {"td", "th"}]
    if any(c.xpath(".//tr") for c in cells):
        return  # layout container row; its nested rows are visited by the walk
    texts = [_norm(c.text_content()) for c in cells]
    non_empty = [t for t in texts if t]
    if not non_empty:
        return
    if len(non_empty) == 1 and HEADING_RE.match(non_empty[0]):
        state.new_heading(non_empty[0])
        return
    header_actions = [a for a in (_action_from_header(t) for t in non_empty) if a]
    if header_actions and any(w in t.upper() for t in non_empty for w in _HEADER_WORDS):
        state.actions = header_actions
        state.counts = {}
        return
    actions = state.actions
    while len(texts) % 3 and texts and not texts[-1]:
        texts.pop()  # trailing empty filler cells (2023Q1 sustainability tables)
    cells = cells[: len(texts)]
    if len(cells) % 3 == 0 and len(cells) // 3 >= len(actions):
        # regular row: one (no, ticker, company) triplet per header column; extra trailing triplets
        # (a 9-cell row under a 2-column header) take the default action order
        for g in range(len(cells) // 3):
            action = actions[g] if g < len(actions) else ACTIONS[min(g, len(ACTIONS) - 1)]
            state.add(records, action, texts[3 * g], texts[3 * g + 1], texts[3 * g + 2])
        return
    # Irregular row (wrong cell count / left-shifted): locate ticker cells and infer their column
    # from the row number (the action whose next expected row it is) or, failing that, from the
    # relative cell position.
    log.warning("irregular row (%d cells, %d columns) under %r: %s", len(cells), len(actions), state.heading, non_empty)
    for i, ticker in enumerate(texts):
        if not TICKER_RE.match(ticker):
            continue
        number = texts[i - 1] if i >= 1 else ""
        company = texts[i + 1] if i + 1 < len(texts) else ""
        n = _row_number(number)
        candidates = [a for a in actions if state.counts.get(a, 0) + 1 == n] if n else []
        if len(candidates) == 1:
            action = candidates[0]
        else:
            k = round(i / max(len(texts) - 1, 1) * (len(actions) - 1))
            action = actions[min(k, len(actions) - 1)]
        state.add(records, action, number, ticker, company)


# --------------------------------------------------------------------------- KAP-hosted announcements
def find_kap_attachments(html: str | bytes) -> list[dict[str, str]]:
    """Attachment ids and file names embedded in a KAP disclosure page (anchor and escaped-JSON forms)."""
    text = _as_text(html)
    found: dict[str, str] = {}
    for obj_id, name in _KAP_ANCHOR_RE.findall(text) + _KAP_JSON_RE.findall(text):
        found.setdefault(obj_id, _norm(name))
    for obj_id in _KAP_ID_RE.findall(text):
        found.setdefault(obj_id, "")
    return [{"obj_id": k, "file_name": v} for k, v in found.items()]


def choose_periodic_attachment(attachments: list[dict[str, str]], lang: str = "en") -> dict[str, str]:
    """Pick the periodic-changes PDF: ``*Periodic*`` for 'en', ``*Donemsel*`` for 'tr', else first PDF."""
    if not attachments:
        raise ValueError("no attachments to choose from")
    pdfs = [a for a in attachments if a["file_name"].lower().endswith(".pdf") or not a["file_name"]]
    pool = pdfs or attachments
    wanted = ("periodic",) if lang == "en" else ("donemsel", "dönemsel")
    for a in pool:
        if any(w in a["file_name"].lower() for w in wanted):
            return a
    log.warning("no %s periodic PDF among %s; using %s", lang, [a["file_name"] for a in pool], pool[0]["file_name"])
    return pool[0]


def kap_periodic_pdf(session: PoliteSession, kap_url: str, raw_dir: Path, lang: str = "en") -> bytes:
    """Download the periodic-changes PDF attached to a KAP-hosted review; returns clean PDF bytes."""
    page = fetch_announcement(session, kap_url, raw_dir)
    attachments = find_kap_attachments(page)
    if not attachments:
        raise RuntimeError(f"no attachments found on {kap_url}")
    chosen = choose_periodic_attachment(attachments, lang)
    obj_id = chosen["obj_id"]
    path = _raw_dir(raw_dir) / f"kap_file_{obj_id}.pdf.gz"
    body = session.get_cached(KAP_FILE_DOWNLOAD_URL.format(obj_id=obj_id), path, source="kap", headers=KAP_HEADERS)
    if body.find(b"%PDF", 0, 4096) < 0:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"attachment {obj_id} of {kap_url} is not a PDF ({len(body)} bytes); cache entry removed")
    log.info("KAP %s: using attachment %s (%s)", kap_url, chosen["file_name"] or obj_id, f"{len(body):,} bytes")
    return strip_java_prefix(body)


def parse_periodic_pdf(pdf_bytes: bytes) -> pd.DataFrame:
    """Parse a KAP ``*_Periodic_Changes.pdf`` (Java prefix tolerated) into ``index_code, action, ticker, company``."""
    reader = PdfReader(io.BytesIO(strip_java_prefix(pdf_bytes)))
    text = "\n".join((page.extract_text(extraction_mode="layout") or "") for page in reader.pages)
    return parse_periodic_text(text)


def parse_periodic_text(text: str) -> pd.DataFrame:
    """Parse pypdf *layout-mode* text of a periodic-changes PDF.

    Sections start with a "BIST ... INDEX" line followed by a header line whose column labels
    (STOCKS TO BE INCLUDED / STOCKS TO BE EXCLUDED / SUBSTITUTE STOCKS) fix the horizontal offset
    of each action; every "<no> <TICKER>" pair on the following lines is assigned to the nearest
    column. Wrapped company-name lines directly below a data line are appended to that column's
    last record.
    """
    records: list[dict[str, str]] = []
    heading: str | None = None
    columns: list[tuple[int, str]] = []
    last_by_action: dict[str, dict[str, str]] = {}
    prev_was_data = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            prev_was_data = False
            continue
        if HEADING_RE.match(stripped):
            heading, columns, last_by_action, prev_was_data = stripped, [], {}, False
            continue
        header_cols = [(m.start(), a) for a, m in _find_header_labels(line)]
        if header_cols:
            columns = sorted(header_cols)
            prev_was_data = False
            continue
        if heading is None or not columns:
            continue
        matches = list(_PAIR_RE.finditer(line))
        if matches:
            for i, m in enumerate(matches):
                end = matches[i + 1].start() if i + 1 < len(matches) else len(line)
                col = min(range(len(columns)), key=lambda k: abs(columns[k][0] - m.start()))
                action = columns[col][1]
                if col + 1 < len(columns):
                    end = min(end, max(columns[col + 1][0] - 2, m.end()))
                rec = {
                    "index_code": index_code_for(heading),
                    "action": action,
                    "ticker": m.group(2),
                    "company": _clean_company(line[m.end() : end]),
                }
                records.append(rec)
                last_by_action[action] = rec
            prev_was_data = True
            continue
        if prev_was_data:
            for j, (offset, action) in enumerate(columns):
                seg_end = columns[j + 1][0] - 2 if j + 1 < len(columns) else len(line)
                segment = _norm(line[max(offset - 2, 0) : seg_end])
                if segment and action in last_by_action:
                    rec = last_by_action[action]
                    rec["company"] = _clean_company(f"{rec['company']} {segment}")
    return pd.DataFrame.from_records(records, columns=["index_code", "action", "ticker", "company"])


def _clean_company(text: str) -> str:
    """Normalise a company-name fragment and drop '--'/'-----' placeholders of empty columns."""
    return _norm(re.sub(r"(?:(?<=\s)|^)-{2,}(?=\s|$)", " ", text))


def _find_header_labels(line: str) -> list[tuple[str, re.Match[str]]]:
    out: list[tuple[str, re.Match[str]]] = []
    for action, pattern in (
        ("include", r"STOCKS\s+TO\s+BE\s+INCLUDED"),
        ("exclude", r"STOCKS\s+TO\s+BE\s+EXCLUDED"),
        ("substitute", r"SUBSTITUTE\s+STOCKS?"),
    ):
        m = re.search(pattern, line, re.I)
        if m:
            out.append((action, m))
    return out


# --------------------------------------------------------------------------- end-to-end table
def build_changes_table(
    session: PoliteSession,
    raw_dir: Path,
    start: tuple[int, int] = (2015, 4),
    end: tuple[int, int] | None = None,
    known_empty: frozenset[tuple[int, int]] = KNOWN_EMPTY_QUARTERS,
    known_missing: frozenset[tuple[int, int, str]] = KNOWN_MISSING_TABLES,
    core_indices: tuple[str, ...] = CORE_INDICES,
) -> pd.DataFrame:
    """All announced constituent changes for every quarterly review between ``start`` and ``end``.

    Columns: ``year, quarter, announcement_date, index_code, action, ticker, company, source_url``.
    borsaistanbul.com pages go through :func:`parse_change_tables`, KAP pages through
    :func:`kap_periodic_pdf` + :func:`parse_periodic_pdf`. Raises ``RuntimeError`` when a quarter
    yields no record for one of ``core_indices`` unless the quarter is in ``known_empty`` or the
    (year, quarter, index) triple is in ``known_missing``; those gaps are listed in
    ``result.attrs["missing_tables"]`` so the universe builder can report them.
    """
    archive = parse_archive(fetch_archive(session, raw_dir))
    selected = select_stock_index_periodic(archive, start=start)
    if end is not None:
        selected = selected[[(y, q) <= tuple(end) for y, q in zip(selected.year, selected.quarter, strict=True)]]
    frames: list[pd.DataFrame] = []
    missing_tables: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        label = f"{row.year}Q{row.quarter}"
        if row.host == "kap":
            changes = parse_periodic_pdf(kap_periodic_pdf(session, row.url, raw_dir))
        elif row.host == "borsaistanbul":
            changes = parse_change_tables(fetch_announcement(session, row.url, raw_dir))
        else:
            raise RuntimeError(f"{label}: unsupported host for {row.url}")
        xu100 = changes[changes.index_code == "XU100"]
        counts = xu100.action.value_counts()
        log.info(
            "%s XU100 +%d -%d (subs %d), %d records over %d indices <- %s",
            label,
            counts.get("include", 0),
            counts.get("exclude", 0),
            counts.get("substitute", 0),
            len(changes),
            changes.index_code.nunique(),
            row.url,
        )
        for idx in core_indices:
            if changes[changes.index_code == idx].empty:
                key = (row.year, row.quarter)
                if key in known_empty or (row.year, row.quarter, idx) in known_missing:
                    log.warning("%s: no %s table on %s (known gap; supply the list via overrides)", label, idx, row.url)
                    missing_tables.append(
                        {"year": row.year, "quarter": row.quarter, "index_code": idx, "source_url": row.url}
                    )
                else:
                    raise RuntimeError(f"{label}: no {idx} records parsed from {row.url}")
        changes = changes.copy()
        changes.insert(0, "year", row.year)
        changes.insert(1, "quarter", row.quarter)
        changes.insert(2, "announcement_date", row.announcement_date)
        changes["source_url"] = row.url
        frames.append(changes)
    columns = ["year", "quarter", "announcement_date", "index_code", "action", "ticker", "company", "source_url"]
    if not frames:
        out = pd.DataFrame(columns=columns)
        out.attrs["missing_tables"] = missing_tables
        return out
    out = pd.concat(frames, ignore_index=True)[columns]
    out["year"] = out["year"].astype("int64")
    out["quarter"] = out["quarter"].astype("int64")
    out.attrs["missing_tables"] = missing_tables
    return out


# --------------------------------------------------------------------------- non-periodic rows
_TICKER_STOPWORDS = frozenset(
    {
        "BIST",
        "INDEX",
        "STOCK",
        "SHARE",
        "SHARES",
        "MAIN",
        "GMYO",
        "GIDA",
        "KABLO",
        "TRUST",
        "HOLD",
        "TARIM",
        "TURK",
        "YESIL",
        "SEKER",
        "TEKS",
        "PLAS",
        "MAK",
        "ILAC",
        "SERUM",
        "TABAN",
        "BORU",
        "DEMIR",
        "CELIK",
        "BOYA",
        "TEKST",
        "PETROL",
        "HALI",
        "AMBAL",
        "YATIRIM",
        "ENERJ",
        "CIMENTO",
        "SPOR",
        "TEKNO",
        "SIGORTA",
        "TICARET",
        "SANAYI",
        "KIMYA",
        "METAL",
        "MADEN",
        "KAGIT",
        "GALVA",
    }
)
_COMPANY_RE = re.compile(
    r"(?:Inclusion of|Exclusion of|sector change of|Transferring of|suspension of trading of|"
    r"headquarters change of|spin-off transaction of|inclusion of|relocation of)\s+(.+?)"
    r"(?:\s+(?:shares?\b|Transferring\b|to\b|in\b|from\b|for\b)|\s*$)",
    re.I,
)


def _reason_guess(title: str) -> str:
    t = title.lower()
    if "suspension" in t or "suspended" in t:
        return "suspension"
    if "watchlist" in t or "watch list" in t or "market" in t or "submarket" in t or "pre-market" in t:
        return "market_transfer"
    if "delist" in t or ("exclusion" in t and "bist stock indices" in t):
        return "delisting"
    if "sector" in t:
        return "sector"
    if "inclusion of" in t or "will be included" in t or "ipo" in t or re.search(r"\bshares in bist\b", t):
        return "ipo"
    return "other"


def _ticker_guess(title: str) -> str | None:
    m = re.search(r"\(([A-Z]{4,5})\)", title)
    if m:
        return m.group(1)
    for tok in re.findall(r"(?<![A-Za-z0-9])([A-Z]{4,5})(?![A-Za-z0-9])", title):
        if tok not in _TICKER_STOPWORDS:
            return tok
    return None


def _company_guess(title: str) -> str | None:
    m = _COMPANY_RE.search(title)
    if m:
        return _norm(m.group(1)) or None
    m = re.match(r"^(.+?)\s+(?:Shares?\s+)?(?:will be included|in BIST)", title)
    return _norm(m.group(1)) if m else None


def list_non_periodic(archive_df: pd.DataFrame, start_year: int = 2018) -> pd.DataFrame:
    """Non-periodic (mid-quarter) index-change announcements from ``start_year`` on, with heuristics.

    Columns: ``announcement_date, title, url, index_groups, ticker_guess, reason_guess,
    company_guess``. ``ticker_guess`` is the first 4-5 letter upper-case token of the title that is
    not a generic word (titles usually name the company, not the code, so it is often None or a
    coincidental match); ``reason_guess`` is one of ipo | delisting | suspension | market_transfer |
    sector | other; ``company_guess`` is the company name fragment named in the title.
    """
    df = archive_df[
        (archive_df.kind == "non_periodic") & (archive_df.announcement_date.map(lambda d: d.year) >= start_year)
    ].copy()
    df["ticker_guess"] = df.title.map(_ticker_guess)
    df["reason_guess"] = df.title.map(_reason_guess)
    df["company_guess"] = df.title.map(_company_guess)
    columns = ["announcement_date", "title", "url", "index_groups", "ticker_guess", "reason_guess", "company_guess"]
    out = df[columns].sort_values("announcement_date", kind="stable").reset_index(drop=True)
    log.info("%d non-periodic announcements from %d on", len(out), start_year)
    return out


def read_cached_pdf(raw_dir: Path, obj_id: str) -> bytes:
    """Convenience: clean PDF bytes of an already cached KAP attachment (no network)."""
    return strip_java_prefix(read_raw_gzip(_raw_dir(raw_dir) / f"kap_file_{obj_id}.pdf.gz"))
