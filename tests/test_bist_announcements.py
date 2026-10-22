"""Tests for the Borsa İstanbul index-announcement client (offline fixtures + optional network checks)."""

from __future__ import annotations

import io
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from bistmas.data.clients import bist_announcements as ba
from bistmas.shared.store import read_manifest, read_raw_gzip, write_raw_gzip

FIX = Path(__file__).parent / "fixtures" / "announcements"
JAVA_PREFIX = b"\xac\xed\x00\x05ur\x00\x02[B\xac\xf3\x17\xf8\x06\x08T\xe0\x02\x00\x00xp\x00\x02:"
URL_2020Q1 = (
    "https://www.borsaistanbul.com/en/announcement/13577/"
    "borsa-istanbul-announces-constituent-changes-bist-stock-indices-first-quarter-2020"
)
URL_2021Q3 = (
    "https://www.borsaistanbul.com/en/announcement/12664/"
    "borsa-istanbul-announces-constituent-changes-bist-stock-indices-third-quarter-2021"
)
URL_2024Q4_KAP = "https://www.kap.org.tr/en/Bildirim/1336462"
PDF_2024Q4_EN = "4028328d919f184901920f72b0e25f80"


# --------------------------------------------------------------------------- helpers
def _read(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


def _make_pdf(lines: list[tuple[float, float, str]]) -> bytes:
    """Minimal one-page PDF (Helvetica 9pt) with text placed at (x, y) points, for layout-mode tests."""
    content = "BT /F1 9 Tf\n"
    for x, y, text in lines:
        esc = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content += f"1 0 0 1 {x} {y} Tm ({esc}) Tj\n"
    stream = (content + "ET\n").encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, obj in enumerate(objs, 1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n")
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return out.getvalue()


def _synthetic_periodic_pdf() -> bytes:
    """Two sections mimicking a KAP periodic-changes PDF, incl. a wrapped name and a substitute-only row."""
    y = 800.0
    rows: list[tuple[float, float, str]] = [
        (40, y, "PERIODIC INDEX CHANGES FOR THE PERIOD OCTOBER 1, 2024-DECEMBER 31, 2024")
    ]

    def add(*cells: tuple[float, str]) -> None:
        nonlocal y
        y -= 12
        rows.extend((x, y, t) for x, t in cells)

    y -= 18
    rows.append((40, y, "BIST 100 INDEX"))
    add((40, "STOCKS TO BE INCLUDED"), (220, "STOCKS TO BE EXCLUDED"), (400, "SUBSTITUTE STOCKS"))
    add(
        (40, "1"),
        (55, "ADEL"),
        (100, "ADEL KALEMCILIK"),
        (220, "1"),
        (235, "BFREN"),
        (280, "BOSCH FREN"),
        (400, "1"),
        (415, "CVKMD"),
        (460, "CVK MADEN"),
    )
    add((280, "SISTEMLERI"))
    add((40, "2"), (55, "ALTNY"), (100, "ALTINAY SAVUNMA"), (220, "2"), (235, "ECZYT"), (280, "ECZACIBASI YATIRIM"))
    y -= 18
    rows.append((40, y, "BIST LIQUID 10 EX BANKS INDEX"))
    add((40, "STOCKS TO BE INCLUDED"), (220, "STOCKS TO BE EXCLUDED"), (400, "SUBSTITUTE STOCKS"))
    add(
        (40, "1"),
        (55, "KOZAL"),
        (100, "KOZA ALTIN"),
        (220, "1"),
        (235, "SAHOL"),
        (280, "SABANCI HOLDING"),
        (400, "1"),
        (415, "SAHOL"),
        (460, "SABANCI HOLDING"),
    )
    add((400, "2"), (415, "PETKM"), (460, "PETKIM"))
    return _make_pdf(rows)


class FakeSession:
    """Stands in for PoliteSession.get_cached: serves canned bodies and mimics the raw-layer cache."""

    def __init__(self, pages: dict[str, bytes]) -> None:
        self.pages = pages
        self.calls: list[tuple[str, Path, dict[str, Any]]] = []

    def get_cached(
        self,
        url: str,
        raw_path: Path,
        manifest_path: Path | None = None,
        source: str = "",
        force: bool = False,
        expect_status: int = 200,
        **kwargs: Any,
    ) -> bytes:
        self.calls.append((url, raw_path, kwargs))
        if raw_path.exists() and not force:
            return read_raw_gzip(raw_path)
        if url not in self.pages:
            raise AssertionError(f"unexpected fetch: {url}")
        payload = self.pages[url]
        write_raw_gzip(payload, raw_path)
        return payload


@pytest.fixture(scope="module")
def archive_df() -> pd.DataFrame:
    return ba.parse_archive(_read("index_announcements_trimmed.html"))


def _counts(df: pd.DataFrame, index_code: str) -> dict[str, int]:
    return df[df.index_code == index_code].action.value_counts().to_dict()


# --------------------------------------------------------------------------- archive
def test_parse_archive_columns_dates_hosts(archive_df: pd.DataFrame) -> None:
    assert list(archive_df.columns) == [
        "announcement_date",
        "title",
        "url",
        "kind",
        "index_groups",
        "host",
        "announcement_type",
    ]
    assert len(archive_df) == 33
    assert all(isinstance(d, date) for d in archive_df.announcement_date)
    row = archive_df[archive_df.url == "https://www.kap.org.tr/en/Bildirim/1666347"].iloc[0]
    assert row.announcement_date == date(2026, 9, 21)  # archive prints MM.DD.YYYY
    assert row.title == "BIST Stock Indices Periodic Review"
    assert row.kind == "periodic" and row.host == "kap"
    assert "BIST Stock Indices|Benchmark Indices" in row.index_groups
    assert set(archive_df.kind) == {"periodic", "non_periodic", "other"}
    assert set(archive_df.host) == {"borsaistanbul", "kap"}
    bist = archive_df[archive_df.host == "borsaistanbul"]
    assert bist.url.str.startswith("https://www.borsaistanbul.com/en/announcement/").all()
    assert archive_df.url.str.contains("/announcement/15121/").any()


def test_select_stock_index_periodic(archive_df: pd.DataFrame) -> None:
    sel = ba.select_stock_index_periodic(archive_df)
    assert list(sel.columns) == ["year", "quarter", "announcement_date", "url", "host"]
    got = set(zip(sel.year, sel.quarter, strict=True))
    assert got == {
        (2015, 4),
        (2016, 1),
        (2018, 1),
        (2018, 2),
        (2019, 4),
        (2020, 1),
        (2021, 3),
        (2024, 2),
        (2024, 3),
        (2024, 4),
        (2025, 3),
        (2026, 2),
        (2026, 3),
        (2026, 4),
    }
    assert sel.iloc[0].year == 2015 and sel.iloc[-1].quarter == 4  # sorted
    by = sel.set_index(["year", "quarter"])
    assert by.loc[(2024, 4), "host"] == "kap" and by.loc[(2024, 4), "url"] == URL_2024Q4_KAP
    assert by.loc[(2025, 3), "host"] == "borsaistanbul"
    assert by.loc[(2026, 3), "url"].endswith("/1619135")  # quarter derived from the June date
    assert by.loc[(2026, 4), "url"].endswith("/1666347")  # September -> Q4
    assert by.loc[(2018, 1), "announcement_date"] == date(2017, 12, 18)
    # other periodic families (KYD funds, Buyback, Dividend, sustainability debt) never sneak in
    assert not sel.url.str.contains("14550|1658688|1465197|15527").any()
    # 2015Q3 is before the default start but selectable
    assert (2015, 3) in set(
        zip(*[ba.select_stock_index_periodic(archive_df, start=(2015, 3))[c] for c in ("year", "quarter")], strict=True)
    )


def test_select_rejects_duplicate_quarters(archive_df: pd.DataFrame) -> None:
    dup = pd.concat([archive_df, archive_df[archive_df.url == URL_2024Q4_KAP]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        ba.select_stock_index_periodic(dup)


def test_quarter_from_month() -> None:
    assert ba.quarter_from_month(2025, 12) == (2026, 1)
    assert ba.quarter_from_month(2026, 3) == (2026, 2)
    assert ba.quarter_from_month(2026, 6) == (2026, 3)
    assert ba.quarter_from_month(2026, 9) == (2026, 4)


# --------------------------------------------------------------------------- announcement pages
def test_parse_change_tables_2020q1_rule_change() -> None:
    df = ba.parse_change_tables(_read("announcement_13577_main.html"))
    assert list(df.columns) == ["index_code", "action", "ticker", "company"]
    assert set(df.index_code) == {"XU100"}
    assert _counts(df, "XU100") == {"include": 23, "exclude": 23, "substitute": 3}
    inc = df[df.action == "include"]
    assert inc.ticker.iloc[:3].tolist() == ["ADANA", "AKGRT", "ALKIM"]
    assert inc.company.iloc[0] == "ADANA CIMENTO (A)"
    assert df[df.action == "exclude"].ticker.iloc[0] == "AFYON"
    assert df[df.action == "substitute"].ticker.tolist() == ["BOLUC", "SARKY", "VKGYO"]
    assert df.ticker.str.fullmatch(r"[A-Z]{4,5}").all()


def test_parse_change_tables_2021q3_substitute_only_row() -> None:
    df = ba.parse_change_tables(_read("announcement_12664_main.html"))
    assert len(df) == 48
    assert list(dict.fromkeys(df.index_code)) == [
        "XU100",
        "XU050",
        "XU030",
        "BIST LIQUID 10 EX BANKS INDEX",
    ]  # BIST LIQUID BANKS has a dash-only table -> no records
    assert _counts(df, "XU100") == {"include": 10, "exclude": 10, "substitute": 3}
    assert _counts(df, "XU050") == {"include": 4, "exclude": 4, "substitute": 3}
    otkar = df[df.ticker == "OTKAR"]
    assert otkar.action.tolist() == ["substitute"] and otkar.index_code.tolist() == ["XU030"]
    xu30 = df[df.index_code == "XU030"]
    assert xu30[xu30.action == "include"].ticker.tolist() == ["FROTO", "VESTL"]
    assert xu30[xu30.action == "substitute"].ticker.tolist() == ["HEKTS", "MGROS", "OTKAR"]


def test_parse_change_tables_heading_inside_table_and_left_shifted_row() -> None:
    html = """
    <html><body><main><div class="announcement-content">
    <p>All changes will be applied after the close of business.</p>
    <table><tbody>
      <tr><td colspan="3">BIST 100 INDEX</td><td></td><td></td><td></td><td colspan="2"></td></tr>
      <tr><td colspan="3">STOCKS TO BE INCLUDED</td><td colspan="3">STOCKS TO BE EXCLUDED</td>
          <td colspan="6">SUBSTITUTE STOCKS</td></tr>
      <tr><td>1.</td><td>LOGO</td><td>LOGO YAZILIM</td><td>1.</td><td>NTHOL</td><td>NET HOLDİNG</td>
          <td>1.</td><td>ASLAN</td><td>ASLAN ÇİMENTO</td></tr>
      <tr><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>2.</td><td>ERBOS</td><td>ERBOSAN</td>
          <td>2.</td><td>KRDMD</td><td>KARDEMIR (D)</td></tr>
      <tr><td></td><td></td><td></td><td></td><td>3.</td><td>BRYAT</td><td>BORUSAN YAT. PAZ.</td></tr>
      <tr><td colspan="3">BIST 30 INDEX</td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
      <tr><td colspan="3">STOCKS TO BE INCLUDED</td><td colspan="3">STOCKS TO BE EXCLUDED</td></tr>
      <tr><td>--</td><td>-----</td><td>-----</td><td>1.</td><td>MGROS</td><td>MİGROS TİCARET</td></tr>
    </tbody></table>
    </div></main></body></html>"""
    df = ba.parse_change_tables(html)
    recs = df.to_dict("records")
    assert {"index_code": "XU100", "action": "include", "ticker": "LOGO", "company": "LOGO YAZILIM"} in recs
    assert {"index_code": "XU100", "action": "exclude", "ticker": "ERBOS", "company": "ERBOSAN"} in recs
    # the left-shifted 7-cell row is a third substitute (row number 3 exceeds include/exclude rows)
    assert {"index_code": "XU100", "action": "substitute", "ticker": "BRYAT", "company": "BORUSAN YAT. PAZ."} in recs
    assert _counts(df, "XU100") == {"include": 1, "exclude": 2, "substitute": 3}
    assert _counts(df, "XU030") == {"exclude": 1}
    assert df[df.index_code == "XU030"].ticker.tolist() == ["MGROS"]


def test_parse_change_tables_2022q1_nested_layout_table() -> None:
    # the whole announcement sits inside an outer layout table: headings are <p> inside its cells
    df = ba.parse_change_tables(_read("announcement_12771_main.html"))
    assert len(df) == 44
    assert _counts(df, "XU100") == {"include": 9, "exclude": 9, "substitute": 3}
    assert _counts(df, "XU050") == {"include": 3, "exclude": 3, "substitute": 3}
    assert _counts(df, "XU030") == {"include": 1, "exclude": 1, "substitute": 3}
    assert _counts(df, "BIST SUSTAINABILITY INDEX") == {"include": 3}
    xu100 = df[df.index_code == "XU100"]
    assert xu100[xu100.action == "include"].ticker.tolist()[:3] == ["ADESE", "BASGZ", "BRYAT"]
    assert xu100[xu100.action == "substitute"].ticker.tolist() == ["GSDHO", "KONTR", "PENTA"]
    assert ba.UNKNOWN_INDEX not in set(df.index_code)


def test_parse_change_tables_page_quirks() -> None:
    html = """
    <html><body><main><div class="announcement-content">
    <span>PERIODIC INDEX CHANGES<br></span>BIST 100 INDEX<br><p>&nbsp;</p>
    <table><tbody>
      <tr><td colspan="3">STOCK TO BE INCLUDED</td><td colspan="3">STOCK TO BE EXCLUDED</td>
          <td colspan="3">SUBSTITUTE STOCKS</td></tr>
      <tr><td>1.</td><td>A.V.O.D</td><td>A.V.O.D GIDA VE TARIM</td><td>1.</td><td>KARTN</td><td>KARTONSAN</td>
          <td>1.</td><td>TEKTU</td><td>TEK-ART TURIZM</td></tr>
    </tbody></table>
    <br><br>
    <table><tbody>
      <tr><td colspan="3">STOCKS TO BE INCLUDED</td><td colspan="3">STOCKS TO BE EXCLUDED</td></tr>
      <tr><td>1</td><td>SODA</td><td>SODA SANAYİİ</td><td>1</td><td>LOGO</td><td>LOGO YAZILIM</td></tr>
    </tbody></table>
    <p><strong>BIST SÜRDÜRÜLEBİLİRLİK ENDEKSİ</strong></p>
    <table><tbody>
      <tr><td colspan="3">ALINACAK PAYLAR</td><td colspan="3">ÇIKARILACAK PAYLAR</td>
          <td colspan="3">YEDEK PAYLAR</td></tr>
      <tr><td>1</td><td>AKSEN</td><td>AKSA ENERJI</td><td>--</td><td>-----</td><td>-----</td><td></td></tr>
      <tr><td></td><td></td><td></td><td>1</td><td>CIMSA</td><td>CIMSA</td>
          <td>1</td><td>VESBE</td><td>VESTEL BEYAZ ESYA</td></tr>
    </tbody></table>
    </div></main></body></html>"""
    df = ba.parse_change_tables(html)
    recs = df.to_dict("records")
    # bare-text heading (tail of a <span>) is seen before the first table; "A.V.O.D" is read as AVOD
    assert {"index_code": "XU100", "action": "include", "ticker": "AVOD", "company": "A.V.O.D GIDA VE TARIM"} in recs
    assert {"index_code": "XU100", "action": "exclude", "ticker": "KARTN", "company": "KARTONSAN"} in recs
    # a second table without its own heading is not silently attributed to BIST 100
    unknown = df[df.index_code == ba.UNKNOWN_INDEX]
    assert unknown.ticker.tolist() == ["SODA", "LOGO"] and unknown.action.tolist() == ["include", "exclude"]
    assert _counts(df, "XU100") == {"include": 1, "exclude": 1, "substitute": 1}
    # Turkish column labels and a trailing filler cell
    sust = df[df.index_code == "BIST SÜRDÜRÜLEBİLİRLİK INDEX"]  # Turkish heading kept verbatim, suffix normalised
    assert sust[["action", "ticker"]].values.tolist() == [
        ["include", "AKSEN"],
        ["exclude", "CIMSA"],
        ["substitute", "VESBE"],
    ]


def test_parse_change_tables_empty_page() -> None:
    df = ba.parse_change_tables("<html><body><main><p>PERIODIC INDEX CHANGES FOR THE PERIOD</p></main></body></html>")
    assert df.empty and list(df.columns) == ["index_code", "action", "ticker", "company"]


def test_index_code_mapping() -> None:
    assert ba.index_code_for("BIST 100 INDEX") == "XU100"
    assert ba.index_code_for(" bist 50 index ") == "XU050"
    assert ba.index_code_for("BIST 30 INDEX:") == "XU030"
    assert ba.index_code_for("BIST LIQUID 10 EX BANKS INDEX") == "BIST LIQUID 10 EX BANKS INDEX"
    assert ba.index_code_for("BIST 30 ENDEKSİ") == "XU030"  # Turkish heading on the 2019Q4 page


# --------------------------------------------------------------------------- KAP PDFs
def test_parse_periodic_text_real_layout() -> None:
    df = ba.parse_periodic_text(_read("kap_2024_4_periodic_changes_layout.txt"))
    assert len(df) == 90
    assert _counts(df, "XU100") == {"include": 10, "exclude": 10, "substitute": 3}
    assert _counts(df, "XU030") == {"include": 3, "exclude": 3, "substitute": 3}
    assert _counts(df, "XU050") == {"include": 4, "exclude": 4, "substitute": 3}
    assert _counts(df, "BIST 500 INDEX") == {"include": 14, "exclude": 14, "substitute": 3}
    xu100 = df[df.index_code == "XU100"]
    assert xu100[xu100.action == "include"].ticker.tolist() == [
        "ADEL",
        "ALTNY",
        "ANSGR",
        "BJKAS",
        "CLEBI",
        "FENER",
        "KARSN",
        "MPARK",
        "PAPIL",
        "RGYAS",
    ]
    assert xu100[xu100.action == "exclude"].ticker.tolist() == [
        "BFREN",
        "ECZYT",
        "EUREN",
        "ISGYO",
        "IZENR",
        "KAYSE",
        "QUAGR",
        "SDTTR",
        "TKNSA",
        "YYLGD",
    ]
    assert xu100[xu100.action == "substitute"].ticker.tolist() == ["CVKMD", "ENTRA", "ANHYT"]
    liq = df[df.index_code == "BIST LIQUID 10 EX BANKS INDEX"]
    assert liq[liq.action == "substitute"].ticker.tolist() == ["SAHOL", "PETKM", "BIMAS"]  # substitute-only rows
    assert liq[liq.action == "include"].ticker.tolist() == ["KOZAL"]
    # wrapped company names are re-joined, empty-column dashes are not glued to names
    assert df[df.ticker == "BRSAN"].company.iloc[0] == "BORUSAN BORU SANAYI"
    assert df[df.ticker == "RGYAS"].company.iloc[0] == "RONESANS GAYRIMENKUL YAT."
    sust = df[df.index_code == "BIST SUSTAINABILITY INDEX"]
    assert sust.company.tolist() == ["IS GMYO", "NUH CIMENTO"] and sust.action.tolist() == ["include", "include"]
    banks = df[df.index_code == "BIST LIQUID BANKS INDEX"]
    assert banks.action.tolist() == ["exclude", "exclude"] and banks.ticker.tolist() == ["SKBNK", "TSKB"]


def test_parse_periodic_pdf_synthetic_with_java_prefix() -> None:
    body = JAVA_PREFIX + _synthetic_periodic_pdf()
    assert ba.strip_java_prefix(body).startswith(b"%PDF")
    df = ba.parse_periodic_pdf(body)
    xu100 = df[df.index_code == "XU100"]
    assert xu100[xu100.action == "include"].ticker.tolist() == ["ADEL", "ALTNY"]
    assert xu100[xu100.action == "exclude"].ticker.tolist() == ["BFREN", "ECZYT"]
    assert xu100[xu100.action == "substitute"].ticker.tolist() == ["CVKMD"]
    assert xu100[xu100.ticker == "BFREN"].company.iloc[0] == "BOSCH FREN SISTEMLERI"
    liq = df[df.index_code == "BIST LIQUID 10 EX BANKS INDEX"]
    assert liq[liq.action == "substitute"].ticker.tolist() == ["SAHOL", "PETKM"]
    assert liq[liq.action == "include"].ticker.tolist() == ["KOZAL"]
    with pytest.raises(ValueError):
        ba.strip_java_prefix(b"<html>error page</html>")


def test_find_and_choose_kap_attachments() -> None:
    atts = ba.find_kap_attachments(_read("kap_bildirim_1336462_snippet.html"))
    assert atts == [
        {"obj_id": "4028328d919f184901920f72b0d45f7f", "file_name": "2024_4_Donemsel_Degisiklikler.pdf"},
        {"obj_id": PDF_2024Q4_EN, "file_name": "2024_4_Periodic_Changes.pdf"},
    ]
    assert ba.choose_periodic_attachment(atts, "en")["obj_id"] == PDF_2024Q4_EN
    assert ba.choose_periodic_attachment(atts, "tr")["file_name"] == "2024_4_Donemsel_Degisiklikler.pdf"
    # JSON-only pages (no anchors) still work, and id-only mentions are kept with an empty name
    only_json = r"{\"objId\":\"" + "a" * 32 + r"\",\"fileName\":\"2026_3_Periodic_Changes.pdf\"}"
    assert ba.find_kap_attachments(only_json)[0]["file_name"] == "2026_3_Periodic_Changes.pdf"
    assert ba.find_kap_attachments("x/file/download/" + "b" * 32 + " y") == [{"obj_id": "b" * 32, "file_name": ""}]
    with pytest.raises(ValueError):
        ba.choose_periodic_attachment([])


# --------------------------------------------------------------------------- caching / naming
def test_announcement_raw_names() -> None:
    assert ba.announcement_raw_name(URL_2020Q1) == "announcement_13577.html.gz"
    assert ba.announcement_raw_name(URL_2024Q4_KAP) == "kap_bildirim_1336462.html.gz"
    assert ba.announcement_raw_name("https://example.org/some/page") == "page_some_page.html.gz"


def test_fetch_archive_imports_saved_copy(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    legacy = raw / "bist_announcements" / "index-announcements.html"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(_read("index_announcements_trimmed.html"), encoding="utf-8")
    session = FakeSession({})
    html = ba.fetch_archive(session, raw)
    assert "table-tr" in html and session.calls == []
    gz = raw / "bist_announcements" / "index-announcements.html.gz"
    assert gz.exists()
    manifest = read_manifest(raw / "bist_announcements" / "manifest.jsonl")
    assert manifest[-1]["url"] == ba.ARCHIVE_URL and manifest[-1]["note"].startswith("imported")
    # second call reads the gzip cache (FakeSession would raise on a real fetch of an unknown URL)
    assert ba.fetch_archive(session, raw) == html
    assert [c[0] for c in session.calls] == [ba.ARCHIVE_URL]


def test_fetch_announcement_caches_and_uses_kap_headers(tmp_path: Path) -> None:
    page_a = b"<html><main>a</main></html>"
    page_k = b"<html><script>self.__next_f.push([1,'k'])</script></html>"
    session = FakeSession({URL_2020Q1: page_a, URL_2024Q4_KAP: page_k})
    assert ba.fetch_announcement(session, URL_2020Q1, tmp_path) == page_a
    assert ba.fetch_announcement(session, URL_2024Q4_KAP, tmp_path) == page_k
    assert (tmp_path / "bist_announcements" / "announcement_13577.html.gz").exists()
    assert (tmp_path / "bist_announcements" / "kap_bildirim_1336462.html.gz").exists()
    assert "headers" not in session.calls[0][2]
    assert session.calls[1][2]["headers"]["Origin"] == "https://www.kap.org.tr"
    ba.fetch_announcement(session, URL_2020Q1, tmp_path)  # served from the raw layer
    assert len([c for c in session.calls if c[0] == URL_2020Q1]) == 2  # the FakeSession logs the cache hit too


def test_kap_periodic_pdf_rejects_non_pdf(tmp_path: Path) -> None:
    session = FakeSession(
        {
            URL_2024Q4_KAP: _read("kap_bildirim_1336462_snippet.html").encode(),
            ba.KAP_FILE_DOWNLOAD_URL.format(obj_id=PDF_2024Q4_EN): b"<html>WAF error</html>",
        }
    )
    with pytest.raises(RuntimeError, match="not a PDF"):
        ba.kap_periodic_pdf(session, URL_2024Q4_KAP, tmp_path)
    assert not (tmp_path / "bist_announcements" / f"kap_file_{PDF_2024Q4_EN}.pdf.gz").exists()


# --------------------------------------------------------------------------- end-to-end (offline)
def _offline_session() -> FakeSession:
    return FakeSession(
        {
            ba.ARCHIVE_URL: _read("index_announcements_trimmed.html").encode(),
            URL_2020Q1: _read("announcement_13577_main.html").encode(),
            URL_2021Q3: _read("announcement_12664_main.html").encode(),
            URL_2024Q4_KAP: _read("kap_bildirim_1336462_snippet.html").encode(),
            ba.KAP_FILE_DOWNLOAD_URL.format(obj_id=PDF_2024Q4_EN): JAVA_PREFIX + _synthetic_periodic_pdf(),
        }
    )


def test_build_changes_table_borsaistanbul_quarter(tmp_path: Path) -> None:
    table = ba.build_changes_table(_offline_session(), tmp_path, start=(2020, 1), end=(2020, 1))
    assert list(table.columns) == [
        "year",
        "quarter",
        "announcement_date",
        "index_code",
        "action",
        "ticker",
        "company",
        "source_url",
    ]
    assert len(table) == 49 and set(table.year) == {2020} and set(table.quarter) == {1}
    assert table.announcement_date.iloc[0] == date(2019, 12, 17)
    assert (table.source_url == URL_2020Q1).all()
    assert _counts(table, "XU100") == {"include": 23, "exclude": 23, "substitute": 3}


def test_build_changes_table_multi_quarter_and_kap(tmp_path: Path) -> None:
    session = _offline_session()
    table = ba.build_changes_table(session, tmp_path, start=(2021, 3), end=(2021, 3))
    assert len(table) == 48 and table[table.ticker == "OTKAR"].action.tolist() == ["substitute"]
    kap = ba.build_changes_table(session, tmp_path, start=(2024, 4), end=(2024, 4), core_indices=("XU100",))
    assert set(kap.index_code) == {"XU100", "BIST LIQUID 10 EX BANKS INDEX"}
    assert kap[kap.action == "include"].ticker.tolist() == ["ADEL", "ALTNY", "KOZAL"]
    assert (kap.source_url == URL_2024Q4_KAP).all() and kap.announcement_date.iloc[0] == date(2024, 9, 20)
    assert (tmp_path / "bist_announcements" / f"kap_file_{PDF_2024Q4_EN}.pdf.gz").exists()
    # quarters outside the served pages are never requested when the range excludes them
    assert not any("/13524/" in url for url, _, _ in session.calls)


def test_build_changes_table_raises_on_empty_xu100(tmp_path: Path) -> None:
    session = _offline_session()
    session.pages[URL_2020Q1] = b"<html><body><main><p>PERIODIC INDEX CHANGES</p></main></body></html>"
    with pytest.raises(RuntimeError, match="no XU100 records"):
        ba.build_changes_table(session, tmp_path, start=(2020, 1), end=(2020, 1))
    ok = ba.build_changes_table(session, tmp_path, start=(2020, 1), end=(2020, 1), known_empty=frozenset({(2020, 1)}))
    assert ok.empty and "ticker" in ok.columns


# --------------------------------------------------------------------------- non-periodic rows
def test_list_non_periodic_heuristics(archive_df: pd.DataFrame) -> None:
    df = ba.list_non_periodic(archive_df, start_year=2018)
    assert list(df.columns) == [
        "announcement_date",
        "title",
        "url",
        "index_groups",
        "ticker_guess",
        "reason_guess",
        "company_guess",
    ]
    assert len(df) == 13 and df.announcement_date.is_monotonic_increasing
    assert set(df.reason_guess) <= {"ipo", "delisting", "suspension", "market_transfer", "sector", "other"}
    by_url = df.set_index(df.url.str.extract(r"/(?:announcement|Bildirim)/(\d+)")[0])
    assert by_url.loc["1343872", "reason_guess"] == "suspension"  # UMPAS: trading suspended > 5 days
    assert by_url.loc["1343872", "company_guess"] == "UMPAS HOLDING"
    assert by_url.loc["1661014", "reason_guess"] == "market_transfer"  # ALVES to Watchlist
    assert by_url.loc["1661014", "ticker_guess"] == "ALVES"
    assert by_url.loc["1612495", "reason_guess"] == "market_transfer"  # TURK ILAC to Watchlist
    assert by_url.loc["1336226", "reason_guess"] == "market_transfer"  # T.TUBORG to Main Market
    assert by_url.loc["1361122", "reason_guess"] == "market_transfer"  # OTTO to Submarket
    assert by_url.loc["1661945", "reason_guess"] == "sector"  # DAGI sector change
    assert by_url.loc["1661945", "company_guess"] == "DAGI GIYIM"
    assert by_url.loc["1663800", "reason_guess"] == "ipo"  # inclusion of a new listing
    assert by_url.loc["1663800", "company_guess"] == "NET GLOBAL ENDUSTRIYEL YAT."
    assert by_url.loc["14581", "reason_guess"] == "ipo"  # "... Shares will be included in BIST Participation ..."
    assert by_url.loc["14581", "company_guess"] == "Hatsan Gemi"
    assert by_url.loc["1652636", "reason_guess"] == "other"  # exclusion from the Corporate Governance index
    assert by_url.loc["1370246", "reason_guess"] == "other"
    assert by_url.loc["1465197", "reason_guess"] == "other"  # Dividend index review typed non-periodic on the archive
    assert pd.isna(by_url.loc["1370246", "ticker_guess"])
    assert ba.list_non_periodic(archive_df, start_year=2025).announcement_date.map(lambda d: d.year).min() >= 2025
    assert ba.list_non_periodic(archive_df, start_year=2027).empty


# --------------------------------------------------------------------------- network
@pytest.mark.network
def test_network_full_range_2018q1_2024q2(settings: Any) -> None:
    from bistmas.data.clients.http import PoliteSession

    session = PoliteSession(min_interval_s=1.0, timeout_s=30)
    table = ba.build_changes_table(session, settings.paths.raw, start=(2018, 1), end=(2024, 2))
    xu100 = table[table.index_code == "XU100"]
    per_q = xu100.groupby(["year", "quarter", "action"]).size().unstack(fill_value=0)
    print("\nXU100 per quarter:\n", per_q.to_string())
    quarters = set(zip(table.year, table.quarter, strict=True))
    assert len(quarters) == 26
    q1_2020 = per_q.loc[(2020, 1)]
    assert q1_2020["include"] == 23 and q1_2020["exclude"] == 23
    totals = xu100.action.value_counts()
    print("XU100 totals 2018Q1-2024Q2:", totals.to_dict())
    assert 200 <= totals["include"] <= 215 and 200 <= totals["exclude"] <= 215
    assert (per_q["include"] > 0).all() and (per_q["exclude"] > 0).all()
    assert table.ticker.str.fullmatch(r"[A-Z]{4,5}").all()


@pytest.mark.network
def test_network_kap_pdf_quarters(settings: Any) -> None:
    from bistmas.data.clients.http import PoliteSession

    session = PoliteSession(min_interval_s=0.5, timeout_s=30)
    pdf = ba.kap_periodic_pdf(session, URL_2024Q4_KAP, settings.paths.raw)
    df = ba.parse_periodic_pdf(pdf)
    assert _counts(df, "XU100") == {"include": 10, "exclude": 10, "substitute": 3}
    df26 = ba.parse_periodic_pdf(
        ba.kap_periodic_pdf(session, "https://www.kap.org.tr/en/Bildirim/1619135", settings.paths.raw)
    )
    c = _counts(df26, "XU100")
    assert c["include"] == c["exclude"] > 0
