"""Offline tests for the security master (KAP fixtures + the curated configs)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from bistmas.data.clients.kap import companies_frame, parse_companies_page, parse_sectors_page, sectors_frame
from bistmas.data.security_master import (
    COLUMNS,
    TEMPLATES,
    SectorGroupMapper,
    build_security_master,
    load_sector_groups,
    load_security_master,
    normalise_sector,
    write_security_master,
)
from bistmas.shared.config import CONFIG_DIR

FIXTURES = Path(__file__).parent / "fixtures" / "kap"
SECTOR_GROUPS = CONFIG_DIR / "sector_groups.csv"
ALIASES = CONFIG_DIR / "ticker_aliases.csv"
AGESA_OID = "4028e4a140e95bea0140edeb7a54015d"
TURSG_OID = "4028e4a140e95be70140ee2f466d0159"


@pytest.fixture(scope="module")
def companies() -> pd.DataFrame:
    return companies_frame(parse_companies_page((FIXTURES / "bist-sirketler.html").read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def sectors() -> pd.DataFrame:
    return sectors_frame(*parse_sectors_page((FIXTURES / "Sektorler.html").read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def master(companies: pd.DataFrame, sectors: pd.DataFrame) -> pd.DataFrame:
    return build_security_master(companies, sectors, SECTOR_GROUPS, ALIASES)


# ----------------------------------------------------------------------------- normalisation and mapping
def test_normalise_sector_is_turkish_aware() -> None:
    assert normalise_sector("sigorta şirketleri") == "SIGORTA ŞIRKETLERI"
    assert normalise_sector("Bankacılık") == "BANKACILIK"
    assert normalise_sector("İMALAT") == normalise_sector("imalat") == "IMALAT"
    assert normalise_sector("  toptan  ve\tperakende ") == "TOPTAN VE PERAKENDE"
    assert normalise_sector("ğüöç") == "ĞÜÖÇ"
    assert normalise_sector("I\u0307malat") == "IMALAT"  # decomposed dotted I (NFC)
    assert normalise_sector(None) == "" and normalise_sector(float("nan")) == ""


def test_sector_group_mapper_rules() -> None:
    mapper = SectorGroupMapper(load_sector_groups(SECTOR_GROUPS))
    assert mapper.resolve("MALİ KURULUŞLAR", "BANKALAR")[:2] == ("BANKS", "bank")
    assert mapper.resolve("mali kuruluşlar", "bankalar")[:2] == ("BANKS", "bank")
    assert mapper.resolve("MALİ KURULUŞLAR", "SİGORTA ŞİRKETLERİ")[:2] == ("INSURANCE", "insurer")
    assert mapper.resolve("MALİ KURULUŞLAR", "SIGORTA SIRKETLERI")[:2] == ("INSURANCE", "insurer")
    assert mapper.resolve("MALİ KURULUŞLAR", "HOLDİNGLER VE YATIRIM ŞİRKETLERİ")[:2] == ("HOLDINGS", "industrial")
    assert mapper.resolve("MALİ KURULUŞLAR", "GAYRİMENKUL YATIRIM ORTAKLIKLARI")[0] == "CONSTRUCTION_REIT"
    assert mapper.resolve("MALİ KURULUŞLAR", "ARACI KURUMLAR")[0] == "OTHER_FINANCIAL"
    assert mapper.resolve("MALİ KURULUŞLAR", "FİNANSAL KİRALAMA VE FAKTORİNG ŞİRKETLERİ")[0] == "OTHER_FINANCIAL"
    assert mapper.resolve("İMALAT", "ANA METAL SANAYİ")[0] == "MANUFACTURING"
    assert mapper.resolve("MADENCİLİK VE TAŞ OCAKÇILIĞI", "METAL CEVHERİ MADENCİLİĞİ")[0] == "MANUFACTURING"
    assert mapper.resolve("ELEKTRİK GAZ VE SU", "ELEKTRİK GAZ VE BUHAR")[0] == "ENERGY_UTILITIES"
    assert mapper.resolve("TEKNOLOJİ", "SAVUNMA")[0] == "TECHNOLOGY"
    assert mapper.resolve("BİLGİ VE İLETİŞİM", "TELEKOMÜNİKASYON")[0] == "TECHNOLOGY"
    assert mapper.resolve("OTELLER VE LOKANTALAR", "YİYECEK VE İÇECEK HİZMETLERİ")[0] == "RETAIL"
    assert mapper.resolve("GAYRİMENKUL FAALİYETLERİ", "GAYRİMENKUL FAALİYETLERİ")[0] == "CONSTRUCTION_REIT"
    assert mapper.resolve("ULAŞTIRMA VE DEPOLAMA", "ULAŞTIRMA VE DEPOLAMA")[0] == "TRANSPORT"
    assert mapper.resolve("TARIM, ORMANCILIK VE BALIKÇILIK", "BALIKÇILIK VE SU ÜRÜNLERİ") == (
        "OTHER",
        "industrial",
        None,
    )
    assert mapper.resolve(None, None) == ("OTHER", "industrial", None)
    # the bank/insurer keyword wins over the main sector, whatever the sub sector says
    assert mapper.resolve("BANKACILIK", "")[:2] == ("BANKS", "bank")


def test_sector_groups_csv_rejects_unknown_template(tmp_path: Path) -> None:
    bad = tmp_path / "sector_groups.csv"
    bad.write_text('kap_main_sector,group,template,note\n"İMALAT",MANUFACTURING,foo,""\n', encoding="utf-8")
    with pytest.raises(ValueError, match="template"):
        SectorGroupMapper(load_sector_groups(bad))
    wrong_columns = tmp_path / "wrong.csv"
    wrong_columns.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_sector_groups(wrong_columns)


# ----------------------------------------------------------------------------- build
def test_master_shape_and_dtypes(master: pd.DataFrame) -> None:
    assert list(master.columns) == COLUMNS
    assert master["is_current"].dtype == bool
    assert set(master["template"].unique()) <= TEMPLATES
    assert master["ticker"].is_unique
    current = master[master["is_current"]]
    assert len(current) == 13  # 14 codes minus the bank instrument code TGB
    assert "TGB" not in set(master["ticker"])
    assert (current["note"] == "").all()
    # current rows first (sorted), alias rows after
    assert master["is_current"].tolist() == [True] * 13 + [False] * (len(master) - 13)


def test_master_groups_and_templates(master: pd.DataFrame) -> None:
    by = master.set_index("ticker")
    assert (by.loc["AKBNK", "sector_group"], by.loc["AKBNK", "template"]) == ("BANKS", "bank")
    assert (by.loc["GARAN", "sector_group"], by.loc["GARAN", "template"]) == ("BANKS", "bank")
    assert (by.loc["TURSG", "sector_group"], by.loc["TURSG", "template"]) == ("INSURANCE", "insurer")
    assert (by.loc["AGESA", "sector_group"], by.loc["AGESA", "template"]) == ("INSURANCE", "insurer")
    assert (by.loc["KCHOL", "sector_group"], by.loc["KCHOL", "template"]) == ("HOLDINGS", "industrial")
    assert by.loc["ISGYO", "sector_group"] == "CONSTRUCTION_REIT"
    assert by.loc["EREGL", "sector_group"] == "MANUFACTURING"
    assert by.loc["KRDMD", "sector_group"] == "MANUFACTURING"
    assert by.loc["THYAO", "sector_group"] == "TRANSPORT"
    assert by.loc["ASELS", "sector_group"] == "TECHNOLOGY"
    assert (by.loc["KNFRT", "sector_group"], by.loc["KNFRT", "template"]) == ("OTHER", "industrial")
    assert by.loc["AKBNK", "main_sector"] == "MALİ KURULUŞLAR"
    assert by.loc["AKBNK", "sector"] == "BANKALAR"
    assert by.loc["AKBNK", "sector_no"] == "008000.001000."
    assert by.loc["AKBNK", "city"] == "İSTANBUL"
    assert by.loc["AKBNK", "title"] == "AKBANK T.A.Ş."


def test_master_alias_rows(master: pd.DataFrame) -> None:
    aliases = pd.read_csv(ALIASES, dtype=str)
    alias_rows = master[~master["is_current"]].set_index("ticker")
    assert set(alias_rows.index) == set(aliases["old_ticker"])
    assert set(alias_rows["note"]) == {"rename", "merger"}
    # successor on KAP: oid + sector data are inherited
    avisa = alias_rows.loc["AVISA"]
    assert avisa["mkk_member_oid"] == AGESA_OID
    assert avisa["title"] == "AGESA HAYAT VE EMEKLİLİK A.Ş."
    assert (avisa["sector_group"], avisa["template"], avisa["note"]) == ("INSURANCE", "insurer", "rename")
    gusgr = alias_rows.loc["GUSGR"]
    assert gusgr["mkk_member_oid"] == TURSG_OID and gusgr["template"] == "insurer"
    # merger: a distinct legal entity, so no oid/sector is inherited (successor recorded separately)
    boluc = alias_rows.loc["BOLUC"]
    assert boluc["mkk_member_oid"] == "" and boluc["title"] == "" and boluc["main_sector"] == ""
    assert (boluc["sector_group"], boluc["template"], boluc["note"]) == ("OTHER", "industrial", "merger")
    assert (boluc["successor_ticker"], boluc["alias_kind"]) == ("OYAKC", "merger")
    assert avisa["successor_ticker"] == "AGESA" and avisa["successor_oid"] == AGESA_OID


def test_master_flags_codes_without_sector_row(companies: pd.DataFrame, sectors: pd.DataFrame) -> None:
    trimmed = sectors[sectors["ticker"] != "THYAO"]
    master = build_security_master(companies, trimmed, SECTOR_GROUPS, ALIASES)
    thyao = master.set_index("ticker").loc["THYAO"]
    assert thyao["main_sector"] == "" and thyao["sector"] == ""
    assert (thyao["sector_group"], thyao["template"], thyao["note"]) == ("OTHER", "industrial", "no_kap_sector")
    assert thyao["title"] == "TÜRK HAVA YOLLARI A.O."  # company data survives the missing sector row


def test_master_equity_only_switch(companies: pd.DataFrame, sectors: pd.DataFrame) -> None:
    master = build_security_master(companies, sectors, SECTOR_GROUPS, ALIASES, equity_only=False)
    assert "TGB" in set(master["ticker"])
    assert master.set_index("ticker").loc["TGB", "template"] == "bank"


def test_master_alias_skipped_when_old_code_is_current(
    companies: pd.DataFrame, sectors: pd.DataFrame, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    aliases = tmp_path / "aliases.csv"
    aliases.write_text(
        "old_ticker,new_ticker,effective_date,kind,note\nAKBNK,GARAN,2020-01-01,rename,bogus\nOLDXX,GARAN,2020-01-01,merger,ok\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="bistmas.data.security_master"):
        master = build_security_master(companies, sectors, SECTOR_GROUPS, aliases)
    assert master["ticker"].is_unique
    assert master.set_index("ticker").loc["AKBNK", "is_current"]
    oldxx = master.set_index("ticker").loc["OLDXX"]
    assert oldxx["mkk_member_oid"] == ""  # merger: not the successor's entity
    assert oldxx["successor_oid"] == companies.set_index("ticker").loc["GARAN", "mkk_member_oid"]
    assert any("skipped" in rec.message for rec in caplog.records)


def test_master_rejects_ticker_mapped_to_two_members(companies: pd.DataFrame, sectors: pd.DataFrame) -> None:
    dup = pd.concat([companies, companies.iloc[[0]].assign(mkk_member_oid="another-oid")], ignore_index=True)
    with pytest.raises(ValueError, match="several members"):
        build_security_master(dup, sectors, SECTOR_GROUPS, ALIASES)


# ----------------------------------------------------------------------------- store
def test_write_and_load_roundtrip(master: pd.DataFrame, tmp_path: Path) -> None:
    store = tmp_path / "store"
    path = write_security_master(master, store)
    assert path == store / "security_master.parquet" and path.exists()
    back = load_security_master(store)
    pd.testing.assert_frame_equal(back, master)
    with pytest.raises(FileNotFoundError):
        load_security_master(tmp_path / "missing")
    with pytest.raises(ValueError, match="missing columns"):
        write_security_master(master.drop(columns=["template"]), store)


def test_sector_overrides_fix_merged_away_codes(master: pd.DataFrame) -> None:
    rows = master.set_index("ticker")
    assert rows.loc["SODA", "sector_group"] == "MANUFACTURING" and not rows.loc["SODA", "is_current"]
    assert rows.loc["ADANA", "sector_group"] == "MANUFACTURING"
    assert rows.loc["KIPA", "sector_group"] == "RETAIL"
    # merger aliases without an override stay OTHER; renames inherit the successor's sector
    assert rows.loc["BOLUC", "sector_group"] == "OTHER"


def test_sector_overrides_custom_file_and_rejects_unmapped(
    companies: pd.DataFrame, sectors: pd.DataFrame, tmp_path: Path
) -> None:
    bad = tmp_path / "o.csv"
    bad.write_text("ticker,kap_main_sector,kap_sub_sector,source,note\nSODA,NOSUCHSECTOR,,x,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="maps to no group"):
        build_security_master(companies, sectors, SECTOR_GROUPS, ALIASES, sector_overrides_csv=bad)
    ok = tmp_path / "ok.csv"
    ok.write_text("ticker,kap_main_sector,kap_sub_sector,source,note\nSODA,TEKNOLOJİ,,x,\n", encoding="utf-8")
    m = build_security_master(companies, sectors, SECTOR_GROUPS, ALIASES, sector_overrides_csv=ok)
    assert m.set_index("ticker").loc["SODA", "sector_group"] == "TECHNOLOGY"
