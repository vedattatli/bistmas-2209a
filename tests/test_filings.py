import json
from datetime import date
from pathlib import Path

import pandas as pd

from bistmas.data.clients.kap import financial_report_rows
from bistmas.data.filings import _split, parse_listing_rows

FIX = Path(__file__).parent / "fixtures" / "statements"


def _rows():
    return json.loads((FIX / "kap_listing_thyao_2017.json").read_text(encoding="utf-8"))


def test_financial_report_filter_and_parse():
    rows = _rows()
    fr = financial_report_rows(rows)
    assert 1 <= len(fr) < len(rows)
    parsed = parse_listing_rows(rows, "oid-1", "THYAO")
    df = pd.DataFrame(parsed)
    assert set(df["ticker"]) == {"THYAO"} and df["disclosure_index"].dtype.kind == "i"
    assert df["publish_ts"].notna().all()
    assert set(df["period"]) <= {1, 2, 3, 4}
    assert df.loc[df["disclosure_index"] == 605351, "publish_ts"].iloc[0] == pd.Timestamp("2017-05-05 20:31:50")


def test_split_windows():
    assert _split((date(2017, 1, 1), date(2019, 12, 31))) == [
        (date(2017, 1, 1), date(2017, 12, 31)),
        (date(2018, 1, 1), date(2018, 12, 31)),
        (date(2019, 1, 1), date(2019, 12, 31)),
    ]
    assert _split((date(2018, 1, 1), date(2018, 12, 31))) == [
        (date(2018, 1, 1), date(2018, 6, 30)),
        (date(2018, 7, 1), date(2018, 12, 31)),
    ]


def test_default_windows_respect_the_one_year_limit():
    from bistmas.data.filings import default_windows

    w = default_windows(date(2019, 6, 1), date(2021, 3, 31))
    assert w[0] == (date(2019, 6, 1), date(2019, 12, 31)) and w[-1] == (date(2021, 1, 1), date(2021, 3, 31))
    assert (date(2020, 1, 1), date(2020, 6, 30)) in w and (date(2020, 7, 1), date(2020, 12, 31)) in w  # leap year split
    assert all((b - a).days + 1 <= 365 for a, b in w)
