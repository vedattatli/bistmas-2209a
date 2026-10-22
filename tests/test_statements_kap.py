import json
from pathlib import Path

import pandas as pd
import pytest

from bistmas.data.statements_kap import parse_context, parse_number, parse_report, parse_unit

FIX = Path(__file__).parent / "fixtures" / "statements"


def _doc():
    return json.loads((FIX / "kap_744943_trimmed.json").read_text(encoding="utf-8"))[0]


def test_unit_and_context_parsing():
    assert parse_unit("1.000.000 TL") == (1e6, "TL")
    assert parse_unit("1.000 TL") == (1e3, "TL")
    assert parse_unit("TL") == (1.0, "TL")
    assert parse_unit("USD") == (1.0, "USD")
    c = parse_context("Cari Dönem01.01.2018 - 31.12.2018")
    assert (c["label"], c["period_start"], c["period_end"]) == (
        "Cari Dönem",
        pd.Timestamp("2018-01-01"),
        pd.Timestamp("2018-12-31"),
    )
    c = parse_context("Önceki Dönem31.12.2017")
    assert pd.isna(c["period_start"]) and c["period_end"] == pd.Timestamp("2017-12-31")


def test_number_parsing_prefers_title_attribute():
    assert parse_number("8606", "8.606") == 8606.0
    assert parse_number(None, "-49.284") == -49284.0
    assert parse_number(None, "1.234,5") == 1234.5
    assert parse_number(None, "(12)") == -12.0
    assert parse_number(None, "") is None and parse_number("", "-") is None


def test_parse_report_thyao_fy2018_first_reported_values():
    meta, df = parse_report(_doc())
    assert meta.ticker == "THYAO" and meta.fiscal_year == 2018 and meta.period == 4
    assert meta.publish_ts == pd.Timestamp("2019-03-05 08:09:36")
    assert meta.consolidated is True and meta.unit_multiplier == 1e6 and meta.currency == "TL"
    cur = df[df["col_index"] == 0].set_index("concept")
    prev = df[df["col_index"] == 1].set_index("concept")
    assert (
        cur.loc["ifrs-full_Revenue", "value_tl"] == 62_853e6 and prev.loc["ifrs-full_Revenue", "value_tl"] == 39_779e6
    )
    assert cur.loc["ifrs-full_ProfitLoss", "value_tl"] == 4_045e6
    assert cur.loc["ifrs-full_Assets", "value_tl"] == 109_076e6 and cur.loc["ifrs-full_Equity", "value_tl"] == 31_284e6
    assert cur.loc["ifrs-full_CashFlowsFromUsedInOperatingActivities", "value_tl"] == 9_613e6
    assert cur.loc["ifrs-full_CostOfSales", "value_tl"] == -49_284e6  # negated label keeps the sign as reported
    assert cur.loc["ifrs-full_Revenue", "period_start"] == pd.Timestamp("2018-01-01")
    assert cur.loc["ifrs-full_Revenue", "period_end"] == pd.Timestamp("2018-12-31")
    assert pd.isna(cur.loc["ifrs-full_Assets", "period_start"]) and cur.loc[
        "ifrs-full_Assets", "period_end"
    ] == pd.Timestamp("2018-12-31")
    assert set(df["statement"]) == {"balance_sheet", "income_statement", "cash_flow"}
    assert (
        cur.loc["ifrs-full_Revenue", "label_en"] == "Revenue" and cur.loc["ifrs-full_Revenue", "label_tr"] == "Hasılat"
    )


def test_statement_name_prefix_mapping():
    from bistmas.data.statements_kap import statement_name

    assert statement_name("210000") == "balance_sheet" and statement_name("210015") == "balance_sheet"
    assert statement_name("310003") == "income_statement" and statement_name("310000") == "income_statement"
    assert statement_name("420000") == "comprehensive_income" and statement_name("520003") == "cash_flow"
    assert statement_name("610000") == "equity_changes" and statement_name("999999") == "role_999999"


def test_parse_bank_report_two_level_header_uses_totals():
    doc = json.loads((FIX / "kap_1386666_bank_trimmed.json").read_text(encoding="utf-8"))[0]
    meta, df = parse_report(doc)
    assert meta.ticker == "AKBNK" and meta.fiscal_year == 2024 and meta.period == 4 and meta.unit_multiplier == 1e3
    bs = df[df["statement"] == "balance_sheet"]
    assert set(bs["context_slice"]) == {"TP", "YP", "Toplam"}
    tot = bs[(bs["context_slice"] == "Toplam") & (bs["context_label"].str.startswith("Cari"))].set_index("concept")
    assert tot.loc["ifrs-full_Assets", "value_tl"] == 2_515_596_654e3
    assert (
        tot.loc["ifrs-full_Equity", "value_tl"] == 240_348_000e3
        or abs(tot.loc["ifrs-full_Equity", "value_tl"] / 1e6 - 240_348) < 1
    )
    assert tot.loc["ifrs-full_Assets", "period_end"] == pd.Timestamp("2024-12-31")
    inc = df[(df["statement"] == "income_statement") & (df["col_index"] == 0)].set_index("concept")
    assert (
        inc.loc["kap-fr_InterestIncomeOrExpense", "value_tl"] == 65_045_000e3
        or abs(inc.loc["kap-fr_InterestIncomeOrExpense", "value_tl"] / 1e6 - 65_045) < 1
    )
    assert inc.loc["ifrs-full_ProfitLoss", "value_tl"] / 1e6 == pytest.approx(42_366, abs=1)
    assert (inc["context_slice"] == "").all()


def test_role_regex_accepts_participation_bank_tables():
    # Participation banks (ALBRK) publish under a hyphenated taxonomy family.
    from bistmas.data.statements_kap import ROLE_RE, statement_name

    for cls, expected in (
        ("financial-table tbl_par-banks_role_210001", "balance_sheet"),
        ("financial-table tbl_banks_role_310000", "income_statement"),
        ("financial-table tbl_general_role_520003", "cash_flow"),
    ):
        m = ROLE_RE.search(cls)
        assert m is not None, cls
        assert statement_name(m.group(1)) == expected


def test_parse_pre_2018_report_without_col_order_classes():
    # AEFES 9M-2017 (disclosure 638714): value cells carry no col-order class; they follow the header order.
    doc = json.loads((FIX / "kap_638714_2017_trimmed.json").read_text(encoding="utf-8"))[0]
    meta, df = parse_report(doc)
    assert (meta.ticker, meta.fiscal_year, meta.period, meta.unit_multiplier) == ("AEFES", 2017, 3, 1000.0)
    assert len(df) > 0
    bs = df[(df.statement == "balance_sheet") & (df.concept == "ifrs-full_CashAndCashEquivalents")].set_index(
        "col_index"
    )
    assert bs.loc[0, "value_tl"] == pytest.approx(5_056_646_000) and bs.loc[0, "period_end"] == pd.Timestamp(
        "2017-09-30"
    )
    assert bs.loc[1, "value_tl"] == pytest.approx(2_745_264_000) and bs.loc[1, "period_end"] == pd.Timestamp(
        "2016-12-31"
    )
    inc = df[(df.statement == "income_statement") & (df.concept == "ifrs-full_Revenue")].set_index("col_index")
    assert inc.loc[0, "value_tl"] == pytest.approx(10_188_580_000)
    assert (inc.loc[0, "period_start"], inc.loc[0, "period_end"]) == (
        pd.Timestamp("2017-01-01"),
        pd.Timestamp("2017-09-30"),
    )
    assert inc.loc[2, "context_label"] == "Cari Dönem 3 Aylık" and inc.loc[2, "period_start"] == pd.Timestamp(
        "2017-07-01"
    )
