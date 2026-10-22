"""Parser for KAP financial-report bodies (XBRL-tagged HTML) into a long table of first-reported values.

A report body (``attachment-detail`` -> ``disclosureBody``) is a list of HTML blocks, one per statement
(balance sheet role 210000, income statement 310003/310002, cash flow 520000/510000, equity 610000, ...).
Each block starts with a header table (``Sunum Para Birimi`` = unit, ``Finansal Tablo Niteliği`` =
Konsolide / Konsolide Olmayan), then a table whose ``context-header`` cells name the columns
(``Cari Dönem 01.01.2018 - 31.12.2018``, ``Önceki Dönem ...``) and whose ``data-input-row`` rows carry the
XBRL concept (``ifrs-full_Revenue|<role>``), Turkish/English labels, a footnote and one value cell per
column (``taxonomy-context-value col-order-class-K``; the raw number sits in the ``title`` attribute).

Values are scaled to TL (unit multiplier applied). Nothing is differenced or restated here: every row
is exactly what the company first reported in that filing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from lxml import html as lh

from bistmas.shared.log import get_logger

log = get_logger("data.statements_kap")

ROLE_NAMES: dict[str, str] = {
    "210000": "balance_sheet",
    "220000": "balance_sheet",
    "310003": "income_statement",
    "310002": "income_statement",
    "310001": "income_statement",
    "320000": "comprehensive_income",
    "410000": "comprehensive_income",
    "510000": "cash_flow",
    "520000": "cash_flow",
    "610000": "equity_changes",
}
ROLE_PREFIXES: dict[str, str] = {
    "21": "balance_sheet",
    "22": "balance_sheet",
    "31": "income_statement",
    "32": "comprehensive_income",
    "41": "comprehensive_income",
    "42": "comprehensive_income",
    "51": "cash_flow",
    "52": "cash_flow",
    "61": "equity_changes",
}


def statement_name(role: str) -> str:
    """Statement type from the KAP role code (taxonomy versions differ in the trailing digits)."""
    if role in ROLE_NAMES:
        return ROLE_NAMES[role]
    return ROLE_PREFIXES.get(role[:2], f"role_{role}")


UNIT_RE = re.compile(r"^\s*(?:(1(?:\.000)*)\s+)?([A-Z]{2,3})\s*$")
DATE_RE = re.compile(r"(\d{2})\.(\d{2})\.(\d{4})")
COL_ORDER_RE = re.compile(r"col-order-class-(\d+)")
ROLE_RE = re.compile(
    r"tbl_[a-z-]+_role_(\d+)"
)  # tbl_general_role_, tbl_banks_role_, tbl_par-banks_role_ (participation banks)

ROW_COLUMNS = [
    "statement_role",
    "statement",
    "concept",
    "concept_role",
    "label_tr",
    "label_en",
    "footnote",
    "col_index",
    "context_label",
    "context_slice",
    "period_start",
    "period_end",
    "value_raw",
    "value_tl",
]


@dataclass
class BlockHeader:
    statement_role: str
    statement: str
    unit_text: str
    unit_multiplier: float
    currency: str
    consolidated: bool | None
    contexts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ReportMeta:
    disclosure_index: int
    mkk_member_oid: str
    ticker: str
    company_title: str
    publish_ts: pd.Timestamp
    fiscal_year: int | None
    period: int | None
    period_code: str
    is_late: bool
    consolidated: bool | None
    unit_multiplier: float | None
    currency: str | None
    n_blocks: int


# ----------------------------------------------------------------------------- helpers
def _text(node: Any) -> str:
    return re.sub(r"\s+", " ", " ".join(node.xpath(".//text()"))).strip()


def parse_unit(text: str) -> tuple[float, str]:
    """``'1.000.000 TL' -> (1e6, 'TL')``, ``'1.000 TL' -> (1e3, 'TL')``, ``'TL' -> (1.0, 'TL')``."""
    m = UNIT_RE.match(text or "")
    if not m:
        return 1.0, (text or "").strip() or "TL"
    scale = m.group(1) or "1"
    return float(scale.replace(".", "")), m.group(2)


def parse_context(label: str) -> dict[str, Any]:
    """``'Cari Dönem 01.01.2018 - 31.12.2018'`` -> label, period_start, period_end (end only when one date)."""
    dates = [pd.Timestamp(int(y), int(mo), int(d)) for d, mo, y in DATE_RE.findall(label or "")]
    name = DATE_RE.sub("", label or "").replace("-", " ").strip()
    name = re.sub(r"\s+", " ", name)
    if len(dates) >= 2:
        return {"label": name, "period_start": dates[0], "period_end": dates[1]}
    if len(dates) == 1:
        return {"label": name, "period_start": pd.NaT, "period_end": dates[0]}
    return {"label": name, "period_start": pd.NaT, "period_end": pd.NaT}


def parse_number(title: str | None, text: str) -> float | None:
    """Prefer the raw ``title`` number; fall back to the displayed text (dots = thousands, comma = decimal)."""
    for candidate in (title, text):
        if candidate is None:
            continue
        s = str(candidate).strip().replace("\xa0", "")
        if s in ("", "-", "—"):
            continue
        neg = s.startswith("(") and s.endswith(")")
        s = s.strip("()")
        if candidate is text:
            s = s.replace(".", "").replace(",", ".")
        try:
            v = float(s)
        except ValueError:
            continue
        return -v if neg else v
    return None


def _contexts(tree: Any) -> list[dict[str, Any]]:
    """Column contexts in value-cell order; two-level headers (period x TP/YP/Toplam) are expanded."""
    header_rows = tree.xpath('//tr[.//td[contains(@class,"context-header")]]')
    if not header_rows:
        return []

    def cells(row: Any) -> list[tuple[str, int]]:
        out = []
        for td in row.xpath('./td[contains(@class,"context-header")]'):
            tr_divs = td.xpath('.//div[contains(@class,"content-tr")]')
            label = _text(tr_divs[0]) if tr_divs else _text(td)
            try:
                span = int(td.get("colspan") or 1)
            except ValueError:
                span = 1
            out.append((label, span))
        return out

    top = cells(header_rows[0])
    sub = cells(header_rows[1]) if len(header_rows) > 1 else []
    contexts: list[dict[str, Any]] = []
    if sub and all(not DATE_RE.search(lbl) for lbl, _ in sub):
        k = 0
        for label, span in top:
            base = parse_context(label)
            for _ in range(max(span, 1)):
                slice_label = sub[k][0] if k < len(sub) else ""
                k += 1
                contexts.append({**base, "label": f"{base['label']} {slice_label}".strip(), "slice": slice_label})
        return contexts
    for label, _span in top:
        contexts.append({**parse_context(label), "slice": ""})
    return contexts


# ----------------------------------------------------------------------------- block parsing
def parse_block(block_html: str) -> tuple[BlockHeader, list[dict[str, Any]]]:
    tree = lh.fromstring(block_html)
    header: dict[str, str] = {}
    for tr in tree.xpath('//table[contains(@class,"financial-header-table")]//tr'):
        tds = tr.xpath("./td")
        if len(tds) >= 2:
            header[_text(tds[0])] = _text(tds[1])
    unit_text = header.get("Sunum Para Birimi", "")
    multiplier, currency = parse_unit(unit_text)
    nature = header.get("Finansal Tablo Niteliği", "")
    consolidated: bool | None
    if not nature:
        consolidated = None
    else:
        consolidated = "olmayan" not in nature.lower()
    tables = tree.xpath('//table[contains(@class,"financial-table")]')
    role = ""
    if tables:
        m = ROLE_RE.search(tables[0].get("class") or "")
        role = m.group(1) if m else ""
    contexts = _contexts(tree)
    head = BlockHeader(role, statement_name(role), unit_text, multiplier, currency, consolidated, contexts)

    rows = tree.xpath('//tr[contains(@class,"data-input-row")]')
    # column order classes present in this block, ranked -> context index
    orders: set[int] = set()
    for tr in rows:
        for td in tr.xpath('./td[contains(@class,"taxonomy-context-value")]'):
            m = COL_ORDER_RE.search(td.get("class") or "")
            if m:
                orders.add(int(m.group(1)))
    order_rank = {k: i for i, k in enumerate(sorted(orders))}
    out: list[dict[str, Any]] = []
    for tr in rows:
        name = (
            _text(tr.xpath('.//div[contains(@class,"taxonomy-field-name")]')[0])
            if tr.xpath('.//div[contains(@class,"taxonomy-field-name")]')
            else ""
        )
        if not name:
            continue
        concept, _, concept_role = name.partition("|")
        title_cell = tr.xpath('./td[contains(@class,"taxonomy-field-title")]')
        label_tr = label_en = ""
        if title_cell:
            tr_divs = title_cell[0].xpath('.//div[contains(@class,"content-tr")]')
            en_divs = title_cell[0].xpath('.//div[contains(@class,"content-en")]')
            label_tr = _text(tr_divs[0]) if tr_divs else ""
            label_en = _text(en_divs[0]) if en_divs else ""
        fn = tr.xpath('.//div[contains(@class,"taxonomy-footnote-value")]')
        footnote = _text(fn[0]) if fn else ""
        for pos, td in enumerate(tr.xpath('./td[contains(@class,"taxonomy-context-value")]')):
            m = COL_ORDER_RE.search(td.get("class") or "")
            # reports published before ~2018-04 carry no col-order class: cells follow the header order
            col = order_rank.get(int(m.group(1)), None) if m else pos
            inner = td.xpath(".//div[@title]")
            title = inner[0].get("title") if inner else None
            value = parse_number(title, _text(td))
            if value is None:
                continue
            ctx = (
                contexts[col]
                if col is not None and col < len(contexts)
                else {"label": "", "slice": "", "period_start": pd.NaT, "period_end": pd.NaT}
            )
            out.append(
                {
                    "statement_role": role,
                    "statement": head.statement,
                    "concept": concept.strip(),
                    "concept_role": concept_role.strip(),
                    "label_tr": label_tr,
                    "label_en": label_en,
                    "footnote": footnote,
                    "col_index": col,
                    "context_label": ctx["label"],
                    "context_slice": ctx.get("slice", ""),
                    "period_start": ctx["period_start"],
                    "period_end": ctx["period_end"],
                    "value_raw": value,
                    "value_tl": value * multiplier if currency == "TL" else value,
                }
            )
    return head, out


# ----------------------------------------------------------------------------- report parsing
def report_meta(doc: dict[str, Any], headers: list[BlockHeader] | None = None) -> ReportMeta:
    basic = (doc.get("disclosure") or {}).get("disclosureBasic") or {}
    ts = pd.to_datetime(str(basic.get("publishDate", "")), format="%Y.%m.%d %H:%M:%S", errors="coerce")
    first = headers[0] if headers else None
    return ReportMeta(
        disclosure_index=int(basic.get("disclosureIndex") or 0),
        mkk_member_oid=str(basic.get("mkkMemberOid") or ""),
        ticker=str(basic.get("stockCode") or ""),
        company_title=str(basic.get("companyTitle") or ""),
        publish_ts=ts,
        fiscal_year=int(basic["year"]) if basic.get("year") not in (None, "") else None,
        period=int(basic["donem"]) if basic.get("donem") not in (None, "") else None,
        period_code=str(basic.get("period") or ""),
        is_late=bool(basic.get("isLate", False)),
        consolidated=first.consolidated if first else None,
        unit_multiplier=first.unit_multiplier if first else None,
        currency=first.currency if first else None,
        n_blocks=len(doc.get("disclosureBody") or []),
    )


def parse_report(doc: dict[str, Any]) -> tuple[ReportMeta, pd.DataFrame]:
    """Parse one ``attachment-detail`` document into ``(meta, long DataFrame with ROW_COLUMNS)``."""
    bodies = doc.get("disclosureBody") or []
    headers: list[BlockHeader] = []
    records: list[dict[str, Any]] = []
    for block in bodies:
        if not isinstance(block, str) or "<table" not in block:
            continue
        try:
            head, rows = parse_block(block)
        except Exception as exc:  # noqa: BLE001 - one bad block must not lose the report
            log.warning("block parse failed: %s", exc)
            continue
        headers.append(head)
        records.extend(rows)
    meta = report_meta(doc, headers)
    df = pd.DataFrame(records, columns=ROW_COLUMNS)
    if len(df):
        df["period_start"] = pd.to_datetime(df["period_start"]).astype("datetime64[ns]")
        df["period_end"] = pd.to_datetime(df["period_end"]).astype("datetime64[ns]")
    return meta, df


def meta_to_dict(meta: ReportMeta) -> dict[str, Any]:
    d = meta.__dict__.copy()
    d["publish_ts"] = None if pd.isna(meta.publish_ts) else meta.publish_ts
    return d
