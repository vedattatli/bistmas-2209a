"""Offline tests for bistmas.data.clients.datastore_files (synthetic DataStore-like fixtures)."""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from bistmas.data.clients import datastore_files as ds

FIXTURES = Path(__file__).parent / "fixtures" / "anchors"


# ----------------------------------------------------------------------------- helpers
def test_fold_text_and_index_normalisation():
    assert ds.fold_text("BULUNDUĞU ENDEKS") == "BULUNDUGU ENDEKS"
    assert ds.fold_text(" Şirket   Ünvanı ") == "SIRKET UNVANI"
    assert ds.normalise_index_code("BIST 100") == "XU100"
    assert ds.normalise_index_code("bist100") == "XU100"
    assert ds.normalise_index_code("BIST 50 ENDEKSİ") == "XU050"
    assert ds.normalise_index_code("XU030") == "XU030"
    assert ds.normalise_index_code("BIST TÜM") == "XUTUM"
    assert ds.normalise_index_code("XBANK") == "XBANK"
    assert ds.normalise_index_code("BIST BANKA") == "BIST BANKA"  # unknown label passes through folded
    assert ds.split_index_cell("BIST 30, BIST 50 VE BIST 100") == ["XU030", "XU050", "XU100"]
    assert ds.split_index_cell("BIST 50 / BIST 100") == ["XU050", "XU100"]
    assert ds.split_index_cell("") == []
    assert ds.normalise_ticker("thyao.e") == "THYAO"


def test_sniff_delimiter_and_decode():
    assert ds.sniff_delimiter("a;b;c\n1;2;3") == ";"
    assert ds.sniff_delimiter("a,b,c\n1,2,3") == ","
    assert ds.sniff_delimiter("a\tb\n1\t2") == "\t"
    assert ds.decode_text("ŞİRKET".encode("cp1254")) == "ŞİRKET"
    assert ds.decode_text("﻿ŞİRKET".encode()) == "ŞİRKET"


def test_parse_date_cell_formats():
    assert ds.parse_date_cell("03.09.2018") == date(2018, 9, 3)
    assert ds.parse_date_cell("20/10/2017") == date(2017, 10, 20)
    assert ds.parse_date_cell("2019-02-18") == date(2019, 2, 18)
    assert ds.parse_date_cell("20180102") == date(2018, 1, 2)
    assert ds.parse_date_cell(date(2020, 1, 1)) == date(2020, 1, 1)
    assert ds.parse_date_cell("") is None and ds.parse_date_cell(None) is None
    assert ds.parse_date_cell("n/a") is None


def test_locate_header_skips_title_lines():
    rows = [["Başlık"], [], ["PAY KODU", "PAY ADI"], ["THYAO", "THY"]]
    assert ds.locate_header(rows, ["PAY KODU"]) == 2
    assert ds.locate_header([["a", "b"]], ["PAY KODU"]) == 0


def test_map_columns_prefers_exact_then_substring():
    header = ["ESKİ PAY KODU", "YENİ PAY KODU", "TARİH"]
    specs = [
        ds.ColumnSpec("ticker", ("ESKI PAY KODU", "PAY KODU"), required=True),
        ds.ColumnSpec("new_ticker", ("YENI PAY KODU",)),
        ds.ColumnSpec("date", ("DEGISIKLIK TARIHI", "TARIH")),
        ds.ColumnSpec("note", ("ACIKLAMA",)),
    ]
    assert ds.map_columns(header, specs) == {"ticker": 0, "new_ticker": 1, "date": 2}
    with pytest.raises(ValueError, match="required column 'ticker'"):
        ds.map_columns(["X", "Y"], [ds.ColumnSpec("ticker", ("PAY KODU",), required=True)])


# ----------------------------------------------------------------------------- constituents zip
def test_read_constituents_zip_reads_csv_and_xlsx_members():
    df = ds.read_constituents_zip(FIXTURES / "exsrk2018.zip")
    assert list(df.columns) == list(ds.CONSTITUENT_COLUMNS)
    assert set(df["member_name"]) == {"exsrk2018_1.csv", "exsrk2018_2.xlsx"}  # readme.txt skipped
    csv_part = df[df["member_name"] == "exsrk2018_1.csv"]
    assert csv_part["snapshot_label"].unique().tolist() == ["2018_1"]
    assert csv_part[csv_part["ticker"] == "THYAO"]["index_code"].tolist() == ["XU030", "XU050", "XU100"]
    assert csv_part[csv_part["ticker"] == "TRKCM"]["index_code"].tolist() == ["XU050", "XU100"]
    assert csv_part[csv_part["ticker"] == "KIPA"]["security_name"].iloc[0] == "TESCO KİPA"  # cp1254 decoded
    assert "A1CAP" in set(csv_part["ticker"])
    xlsx_part = df[df["member_name"] == "exsrk2018_2.xlsx"]
    assert xlsx_part["snapshot_label"].unique().tolist() == ["2018-04-02"]  # TARİH column wins over the name
    assert xlsx_part[xlsx_part["ticker"] == "THYAO"]["index_code"].tolist() == ["XU030"]  # '.E' stripped
    assert xlsx_part[xlsx_part["ticker"] == "SOKM"]["index_code"].tolist() == ["XU100"]  # 'ENDEKSİ' suffix
    assert xlsx_part[xlsx_part["ticker"] == "ENJSA"]["index_code"].tolist() == ["XU050", "XU100"]
    xu100 = df[df["index_code"] == "XU100"].groupby("snapshot_label")["ticker"].apply(sorted).to_dict()
    assert xu100 == {"2018-04-02": ["ENJSA", "SOKM"], "2018_1": ["A1CAP", "ASELS", "KIPA", "THYAO", "TRKCM"]}
    assert all(str(df[c].dtype) == "string" for c in ds.CONSTITUENT_COLUMNS)


def test_read_constituents_zip_without_tables_is_empty(tmp_path):
    import zipfile

    path = tmp_path / "exsrk1999.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("notes.pdf", b"%PDF-1.4")
    df = ds.read_constituents_zip(path)
    assert df.empty and list(df.columns) == list(ds.CONSTITUENT_COLUMNS)


# ----------------------------------------------------------------------------- generic readers
def test_read_delisted_file_maps_turkish_headers(caplog):
    with caplog.at_level("INFO", logger="bistmas.data.datastore_files"):
        df = ds.read_delisted_file(FIXTURES / "delisted_companies.csv")
    assert list(df.columns) == list(ds.GENERIC_COLUMNS)
    assert df["ticker"].tolist() == ["KIPA", "NTTUR", "DGZTE", "XXXX1"]
    assert df["date"].tolist() == [date(2018, 9, 3), date(2017, 10, 20), date(2019, 2, 18), None]
    assert df["name"].iloc[0] == "TESCO KİPA KİTLE PAZARLAMA"
    assert df["note"].iloc[1] == "Birleşme (NTHOL)"
    assert df["new_ticker"].isna().all()
    assert df["source_file"].unique().tolist() == ["delisted_companies.csv"]
    assert any("column mapping" in rec.message and "KOTTAN" in rec.message for rec in caplog.records)


def test_read_code_changes_file_xlsx_with_title_row():
    df = ds.read_code_changes_file(FIXTURES / "equity_code_changes.xlsx")
    assert df["ticker"].tolist() == ["AVISA", "GUSGR", "YAZIC"]
    assert df["new_ticker"].tolist() == ["AGESA", "TURSG", "AGHOL"]  # '.E' suffix stripped
    assert df["date"].tolist() == [date(2021, 1, 4), date(2020, 9, 1), date(2018, 5, 11)]
    assert df["new_name"].iloc[0] == "AGESA HAYAT VE EMEKLİLİK"
    assert df["note"].iloc[0] == "Unvan değişikliği" and pd.isna(df["note"].iloc[1])


def test_read_first_trading_file_comma_csv_keeps_extras():
    df = ds.read_first_trading_file(FIXTURES / "first_trading_data.csv")
    assert df["ticker"].tolist() == ["ENJSA", "SOKM", "MPARK"]
    assert df["date"].tolist() == [date(2018, 2, 8), date(2018, 5, 23), date(2018, 2, 13)]
    assert df["note"].iloc[0] == "YILDIZ PAZAR"  # PAZAR mapped to note
    assert df["extra_halka_arz_fiyati"].tolist() == ["6,25", "10,50", "26,00"]


def test_read_generic_table_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown table kind"):
        ds.read_generic_table(FIXTURES / "first_trading_data.csv", "prices")


def test_read_generic_table_from_zip(tmp_path):
    import zipfile

    path = tmp_path / "delisted.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.write(FIXTURES / "delisted_companies.csv", "delisted_companies.csv")
    df = ds.read_delisted_file(path)
    assert len(df) == 4 and df["source_file"].iloc[0] == "delisted.zip::delisted_companies.csv"


# ----------------------------------------------------------------------------- directory loader
def test_classify_datastore_file():
    assert ds.classify_datastore_file("exsrk2018.zip") == "constituents"
    assert ds.classify_datastore_file("delisted_companies.csv") == "delisted"
    assert ds.classify_datastore_file("Kottan_Cikarilan_Sirketler.xlsx") == "delisted"
    assert ds.classify_datastore_file("equity_code_changes.xlsx") == "code_changes"
    assert ds.classify_datastore_file("pay_kodu_degisiklikleri.csv") == "code_changes"
    assert ds.classify_datastore_file("first_trading_data.csv") == "first_trading"
    assert ds.classify_datastore_file("ilk_islem_tarihleri.xlsx") == "first_trading"
    assert ds.classify_datastore_file("notes.pdf") is None
    assert ds.classify_datastore_file("random.csv") is None


def test_load_datastore_dir(tmp_path):
    assert ds.load_datastore_dir(tmp_path) == {}
    folder = tmp_path / "datastore"
    folder.mkdir()
    for name in ("exsrk2018.zip", "delisted_companies.csv", "equity_code_changes.xlsx", "first_trading_data.csv"):
        shutil.copy(FIXTURES / name, folder / name)
    (folder / "unrelated.csv").write_text("a;b\n1;2\n")
    (folder / "broken_delisted.csv").write_bytes(b"\xff\xfe garbage without header")
    out = ds.load_datastore_dir(tmp_path)
    assert set(out) == {"constituents", "delisted", "code_changes", "first_trading"}
    assert len(out["constituents"]) == 14 and len(out["delisted"]) == 4
    assert out["code_changes"]["new_ticker"].tolist() == ["AGESA", "TURSG", "AGHOL"]
    assert out["first_trading"]["ticker"].tolist() == ["ENJSA", "SOKM", "MPARK"]
