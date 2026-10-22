"""Offline tests for the KAP client (fixtures trimmed from the live pages of 2026-09-23) plus one network test."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from bistmas.data.clients.http import PoliteSession
from bistmas.data.clients.kap import (
    COMPANY_COLUMNS,
    DEFAULT_HEADERS,
    SECTOR_COLUMNS,
    FlightChunk,
    KapClient,
    KapParseError,
    _decode_js_string,
    companies_frame,
    extract_json_value,
    flight_payload,
    iter_flight_chunks,
    parse_companies_page,
    parse_member_filter,
    parse_sectors_page,
    sectors_frame,
)
from bistmas.shared.store import write_raw_gzip

FIXTURES = Path(__file__).parent / "fixtures" / "kap"
THYAO_OID = "4028e4a140f2ed720140f376bebb01a7"
GARAN_OID = "4028e4a140f2ed720140f37cb2a601b7"
KRDM_OID = "4028e4a140f2ed7201412ace3ada0707"


@pytest.fixture(scope="module")
def companies_html() -> str:
    return (FIXTURES / "bist-sirketler.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def sectors_html() -> str:
    return (FIXTURES / "Sektorler.html").read_text(encoding="utf-8")


class OfflineSession(PoliteSession):
    """PoliteSession that refuses to touch the network (cache hits only)."""

    def request(self, method: str, url: str, **kwargs: Any) -> Any:  # pragma: no cover - guard
        raise AssertionError(f"unexpected network call: {method} {url}")


def _populate_cache(raw_dir: Path) -> None:
    write_raw_gzip((FIXTURES / "bist-sirketler.html").read_bytes(), raw_dir / "pages" / "bist-sirketler.html.gz")
    write_raw_gzip((FIXTURES / "Sektorler.html").read_bytes(), raw_dir / "pages" / "Sektorler.html.gz")
    for ticker in ("THYAO", "GARAN", "ZZZZQ"):
        write_raw_gzip(
            (FIXTURES / f"member_filter_{ticker}.json").read_bytes(), raw_dir / "member" / f"{ticker}.json.gz"
        )


# ----------------------------------------------------------------------------- Flight extraction
def test_fixtures_stay_small() -> None:
    for path in FIXTURES.iterdir():
        assert path.stat().st_size < 150_000, path


def test_iter_flight_chunks_reads_every_push(companies_html: str) -> None:
    chunks = iter_flight_chunks(companies_html)
    assert [c.kind for c in chunks] == [0, 1, 1, 1]
    assert chunks[0] == FlightChunk(0, "", True)
    assert all(c.complete for c in chunks)
    assert chunks[1].text.startswith("1:")  # RSC preamble line ids
    assert '"kapMemberTitle"' in chunks[2].text


def test_decode_js_string_handles_json_and_js_escapes() -> None:
    assert _decode_js_string(r"a\"b\\c\n\u003cdiv\u003e \u0026 x") == 'a"b\\c\n<div> & x'
    assert _decode_js_string(r"\u003cscript\u003e") == "<script>"  # Next.js HTML-safe escapes
    assert _decode_js_string(r"\x41\'\/") == "A'/"
    assert _decode_js_string("abc\\") == "abc"  # truncated on a lone backslash
    with pytest.raises(KapParseError):
        _decode_js_string("\\u12")


def test_flight_payload_requires_next_page() -> None:
    with pytest.raises(KapParseError):
        flight_payload("<html><body>no next here</body></html>")


def test_extract_json_value_errors() -> None:
    payload = '0:["$","div",{"data":[{"x":1}]}]'
    assert extract_json_value(payload, '"x"', '"data":') == [{"x": 1}]
    with pytest.raises(KapParseError):
        extract_json_value(payload, '"missing"', '"data":')
    with pytest.raises(KapParseError):
        extract_json_value(payload, '"x"', '"nodata":')
    with pytest.raises(KapParseError):
        extract_json_value('0:{"data":[{"x":1', '"x"', '"data":')


def test_truncated_page_raises_and_warns(sectors_html: str, caplog: pytest.LogCaptureFixture) -> None:
    cut = sectors_html[: sectors_html.find('"mainSectorName"') + 2000]
    with caplog.at_level(logging.WARNING, logger="bistmas.data.kap"), pytest.raises(KapParseError):
        parse_sectors_page(cut)
    assert any("truncated" in rec.message for rec in caplog.records)


# ----------------------------------------------------------------------------- companies
def test_parse_companies_page(companies_html: str) -> None:
    records = parse_companies_page(companies_html)
    assert len(records) == 11
    by_code = {r["stockCode"]: r for r in records}
    assert by_code["THYAO"]["mkkMemberOid"] == THYAO_OID
    assert by_code["THYAO"]["kapMemberTitle"] == "TÜRK HAVA YOLLARI A.O."
    assert by_code["KRDMA, KRDMB, KRDMD"]["cityName"] == "KARABÜK"
    assert set(by_code["AKBNK"]) == {
        "mkkMemberOid",
        "kapMemberTitle",
        "relatedMemberTitle",
        "stockCode",
        "cityName",
        "relatedMemberOid",
        "kapMemberType",
    }


def test_companies_frame_explodes_codes_and_filters(companies_html: str) -> None:
    records = parse_companies_page(companies_html)
    df = companies_frame(records)
    assert list(df.columns) == COMPANY_COLUMNS
    assert len(df) == 14  # 11 members, KRDM* x3 and GARAN/TGB x2
    assert df["ticker"].is_monotonic_increasing
    assert set(df.loc[df["mkk_member_oid"] == KRDM_OID, "ticker"]) == {"KRDMA", "KRDMB", "KRDMD"}
    assert set(df.loc[df["mkk_member_oid"] == GARAN_OID, "ticker"]) == {"GARAN", "TGB"}
    assert (df["member_type"] == "IGS").all()
    akbnk = df.set_index("ticker").loc["AKBNK"]
    assert akbnk["title"] == "AKBANK T.A.Ş."
    assert akbnk["city"] == "İSTANBUL"
    assert akbnk["auditor_title"].startswith("DRT BAĞIMSIZ DENETİM")
    assert akbnk["auditor_oid"] == "33E5FED7244500EAE0530A4A622B2AEA"
    assert len(companies_frame(records, member_types=None)) == 14
    empty = companies_frame(records, member_types=("FON",))
    assert empty.empty and list(empty.columns) == COMPANY_COLUMNS


def test_companies_frame_handles_missing_fields() -> None:
    df = companies_frame([{"mkkMemberOid": "x", "stockCode": "ABCD", "kapMemberType": "IGS"}])
    assert len(df) == 1
    assert pd.isna(df.loc[0, "city"])
    assert companies_frame([{"mkkMemberOid": "x", "stockCode": None, "kapMemberType": "IGS"}]).empty


# ----------------------------------------------------------------------------- sectors
def test_parse_sectors_page(sectors_html: str) -> None:
    main_items, sub_items = parse_sectors_page(sectors_html)
    assert len(main_items) == len(sub_items) == 11
    assert {m["mainSectorName"] for m in main_items} >= {"MALİ KURULUŞLAR", "İMALAT", "TEKNOLOJİ"}
    assert {s["sectorName"] for s in sub_items} >= {"BANKALAR", "SİGORTA ŞİRKETLERİ", "SAVUNMA"}


def test_sectors_frame_joins_levels_and_explodes(sectors_html: str) -> None:
    df = sectors_frame(*parse_sectors_page(sectors_html))
    assert list(df.columns) == SECTOR_COLUMNS
    assert len(df) == 14
    row = df.set_index("ticker").loc["AKBNK"]
    assert row["mkk_member_oid"] == "4028e4a240e8d1830140e905edcd0006"
    assert row["main_sector"] == "MALİ KURULUŞLAR"
    assert row["main_sector_no"] == "008000."
    assert row["sector"] == "BANKALAR"
    assert row["sector_no"] == "008000.001000."
    assert row["sector_oid"] == "33E5FED8040000EAE0530A4A622B2AEA"
    assert row["title"] == "AKBANK T.A.Ş."
    assert set(df.loc[df["mkk_member_oid"] == KRDM_OID, "ticker"]) == {"KRDMA", "KRDMB", "KRDMD"}
    assert (df.loc[df["mkk_member_oid"] == KRDM_OID, "sector"] == "ANA METAL SANAYİ").all()
    assert df.set_index("ticker").loc["THYAO", "main_sector"] == "ULAŞTIRMA VE DEPOLAMA"


def test_sectors_frame_keeps_unmatched_levels() -> None:
    main = [{"mkkMemberOid": "a", "stockCode": "AAAA", "title": "A", "mainSectorName": "M", "mainSectorNo": "1."}]
    sub = [
        {"mkkMemberOid": "b", "stockCode": "BBBB", "title": "B", "sectorName": "S", "sectorNo": "2.", "sectorOid": "o"}
    ]
    df = sectors_frame(main, sub).set_index("ticker")
    assert pd.isna(df.loc["AAAA", "sector"]) and df.loc["AAAA", "main_sector"] == "M"
    assert pd.isna(df.loc["BBBB", "main_sector"]) and df.loc["BBBB", "title"] == "B"


# ----------------------------------------------------------------------------- member lookup
def test_parse_member_filter() -> None:
    records = parse_member_filter((FIXTURES / "member_filter_GARAN.json").read_bytes())
    assert len(records) == 8
    assert {"companyCode", "mkkMemberOid", "title", "permaLink"} <= set(records[0])
    assert parse_member_filter(b"[]") == []
    with pytest.raises(KapParseError):
        parse_member_filter(b"<html>blocked</html>")
    with pytest.raises(KapParseError):
        parse_member_filter(b'{"a": 1}')


# ----------------------------------------------------------------------------- client (offline, cache hits)
def test_client_reads_everything_from_the_raw_cache(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    raw_dir = tmp_path / "kap"
    _populate_cache(raw_dir)
    client = KapClient(OfflineSession(min_interval_s=0.0), raw_dir)

    companies = client.fetch_companies()
    sectors = client.fetch_sectors()
    assert len(companies) == 14 and len(sectors) == 14
    assert set(companies["ticker"]) == set(sectors["ticker"])

    assert client.member_lookup("THYAO")["mkkMemberOid"] == THYAO_OID
    assert client.member_lookup("thyao ")["mkkMemberOid"] == THYAO_OID
    assert client.member_lookup("ZZZZQ") is None
    assert client.member_lookup("GARAN", prefer_oid=GARAN_OID)["title"] == "TÜRKİYE GARANTİ BANKASI A.Ş."
    with caplog.at_level(logging.WARNING, logger="bistmas.data.kap"):
        first = client.member_lookup("GARAN")
    assert first["companyCode"] == "2113"  # first hit of the free-text search, not the bank
    assert any("returned 8 records" in rec.message for rec in caplog.records)
    with pytest.raises(ValueError):
        client.member_lookup("not a ticker")
    assert not client.manifest_path.exists()  # nothing was fetched, so nothing was appended


def test_client_sends_browser_headers_and_uses_raw_layout(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    class RecordingSession:
        def get_cached(self, url: str, raw_path: Path, manifest_path: Path, **kwargs: Any) -> bytes:
            calls.append({"url": url, "raw_path": raw_path, "manifest_path": manifest_path, **kwargs})
            if url.endswith("bist-sirketler"):
                return (FIXTURES / "bist-sirketler.html").read_bytes()
            if url.endswith("Sektorler"):
                return (FIXTURES / "Sektorler.html").read_bytes()
            return (FIXTURES / "member_filter_THYAO.json").read_bytes()

    raw_dir = tmp_path / "kap"
    client = KapClient(RecordingSession(), raw_dir, headers={"X-Extra": "1"})  # type: ignore[arg-type]
    client.fetch_companies(force=True)
    client.fetch_sectors()
    client.member_lookup("THYAO")

    assert [c["url"] for c in calls] == [
        "https://www.kap.org.tr/tr/bist-sirketler",
        "https://www.kap.org.tr/tr/Sektorler",
        "https://www.kap.org.tr/tr/api/member/filter/THYAO",
    ]
    assert [c["raw_path"] for c in calls] == [
        raw_dir / "pages" / "bist-sirketler.html.gz",
        raw_dir / "pages" / "Sektorler.html.gz",
        raw_dir / "member" / "THYAO.json.gz",
    ]
    assert all(c["manifest_path"] == raw_dir / "manifest.jsonl" for c in calls)
    assert all(c["source"] == "kap" for c in calls)
    assert [c["force"] for c in calls] == [True, False, False]
    for call in calls:
        assert call["headers"]["Referer"] == "https://www.kap.org.tr/tr/bildirim-sorgu"
        assert call["headers"]["Origin"] == "https://www.kap.org.tr"
        assert call["headers"]["Accept"] == "application/json,text/plain,*/*"
        assert call["headers"]["X-Extra"] == "1"
    assert DEFAULT_HEADERS["Referer"] == "https://www.kap.org.tr/tr/bildirim-sorgu"


# ----------------------------------------------------------------------------- live
@pytest.mark.network
def test_live_pages_parse(tmp_path: Path) -> None:
    client = KapClient(PoliteSession(min_interval_s=0.5, timeout_s=30), tmp_path / "kap")
    companies = client.fetch_companies()
    sectors = client.fetch_sectors()
    assert len(companies) >= 700
    assert companies["mkk_member_oid"].nunique() >= 700
    assert len(sectors) >= 600
    assert companies.set_index("ticker").loc["THYAO", "mkk_member_oid"] == THYAO_OID
    assert sectors.set_index("ticker").loc["AKBNK", "sector"] == "BANKALAR"
    assert client.member_lookup("THYAO")["mkkMemberOid"] == THYAO_OID
    assert (tmp_path / "kap" / "manifest.jsonl").exists()
