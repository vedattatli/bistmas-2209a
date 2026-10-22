"""Real DataStore constituent files are yearly matrices: ticker x effective-date, cell = narrowest index."""

import zipfile
from datetime import datetime

import openpyxl
import pandas as pd

from bistmas.data.clients.datastore_files import load_constituent_matrices, read_constituent_matrix


def _matrix_zip(path, year, dates, rows):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append([None, None, year] + [None] * (len(dates) - 1))
    ws.append([None, None, "1.ÇEYREK"] + [None] * (len(dates) - 1))
    ws.append([None, None, "1ST QUARTER"] + [None] * (len(dates) - 1))
    ws.append(["PAY KODU\nCODE", "PAY ADI\nEQUITY"] + [datetime.fromisoformat(d) for d in dates])
    for r in rows:
        ws.append(r)
    xlsx = path.parent / f"exsrk{year}.xlsx"
    wb.save(xlsx)
    with zipfile.ZipFile(path, "w") as z:
        z.write(xlsx, xlsx.name)
    xlsx.unlink()


def test_matrix_parse_expands_nested_indices_and_picks_newest_download(tmp_path):
    dates = ["2018-01-02", "2018-05-18", "2018-07-02"]
    _matrix_zip(tmp_path / "43_2018-01-02_301716_exsrk2018.zip", 2018, dates[:1], [["AAAA", "A", "XU030"]])
    _matrix_zip(
        tmp_path / "36_2018-10-01_485978_exsrk2018.zip",
        2018,
        dates,
        [["AAAA", "A", "XU030", "XU050", "XU100"], ["BBBB", "B", None, "XU100", None], ["ZZZZ", "Z", None, None, None]],
    )
    one = read_constituent_matrix(tmp_path / "36_2018-10-01_485978_exsrk2018.zip")
    assert set(one["ticker"]) == {"AAAA", "BBBB"}
    a1 = one[(one["ticker"] == "AAAA") & (one["snapshot_date"] == pd.Timestamp("2018-01-02"))]
    assert set(a1["index_code"]) == {"XU030", "XU050", "XU100"} and set(a1["narrowest"]) == {"XU030"}
    a2 = one[(one["ticker"] == "AAAA") & (one["snapshot_date"] == pd.Timestamp("2018-05-18"))]
    assert set(a2["index_code"]) == {"XU050", "XU100"}
    assert len(one[one["ticker"] == "BBBB"]) == 1  # only the 2018-05-18 column, XU100 only
    both = load_constituent_matrices(tmp_path)
    # the lower sequence number (listed first in the library = newest download) wins for the year
    assert set(both["file_name"]) == {"36_2018-10-01_485978_exsrk2018.zip"}
    assert both["snapshot_date"].nunique() == 3


def test_validate_against_matrices_is_rename_aware(tmp_path):
    from bistmas.data.universe.build import validate_against_constituent_matrices

    _matrix_zip(
        tmp_path / "10_2020-10-01_1_exsrk2020.zip",
        2020,
        ["2020-01-02", "2020-07-01"],
        [["TURSG", "Türkiye Sigorta", "XU100", "XU100"], ["AAAA", "A", "XU100", None], ["BBBB", "B", None, "XU100"]],
    )
    membership = pd.DataFrame(
        {
            "index_code": ["XU100"] * 3,
            "ticker": ["GUSGR", "AAAA", "BBBB"],
            "effective_start": pd.to_datetime(["2019-10-01", "2019-10-01", "2020-07-01"]),
            "effective_end": pd.to_datetime([None, "2020-07-01", None]),
        }
    )
    aliases = pd.DataFrame(
        {
            "old_ticker": ["GUSGR"],
            "new_ticker": ["TURSG"],
            "effective_date": ["2020-09-01"],
            "kind": ["rename"],
            "note": [""],
        }
    )
    out = validate_against_constituent_matrices(membership, aliases, tmp_path, index_codes=("XU100",))
    assert out["snapshots"] == 2 and out["mismatches"] == []
    membership.loc[2, "effective_start"] = pd.Timestamp("2020-10-01")  # BBBB now enters too late
    out = validate_against_constituent_matrices(membership, aliases, tmp_path, index_codes=("XU100",))
    assert out["mismatches"] == [{"index_code": "XU100", "date": "2020-07-01", "missing": ["BBBB"], "extra": []}]
