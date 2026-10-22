"""statements_view / ratios corrections (prereg-v2 wp1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bistmas.scoring.ratios import compute_ratios
from bistmas.scoring.statements_view import _pick, add_ttm, filings_wide, load_concept_map

CFG = load_concept_map()


def _long(rows):
    df = pd.DataFrame(rows, columns=["disclosure_index", "concept", "start", "end", "value", "period"])
    df["ticker"] = "XXX"
    df["mkk_member_oid"] = "x"
    df["fiscal_year"] = pd.to_datetime(df["end"]).dt.year
    df["consolidated"] = True
    df["publish_ts"] = pd.to_datetime(df["end"]) + pd.Timedelta(days=30)
    df["availability_date"] = df["publish_ts"].dt.normalize() + pd.Timedelta(days=1)
    df["period_start"] = pd.to_datetime(df["start"])
    df["period_end"] = pd.to_datetime(df["end"])
    df["value_tl"] = df["value"]
    df["context_slice"] = ""
    return df


def test_pick_sum_zero_and_abs():
    v = pd.Series({"a": 10.0, "b": 5.0, "z": 0.0, "n": 7.0, "neg": -3.0})
    assert _pick(v, ["a+b"]) == 15.0 and _pick(v, ["a+missing"]) == 10.0 and np.isnan(_pick(v, ["x+y"]))
    assert _pick(v, ["z", "n"]) == 0.0  # default: the first reported value wins
    assert _pick(v, ["z", "n"], zero_as_missing=True) == 7.0
    assert _pick(v, ["z"], zero_as_missing=True) == 0.0
    assert _pick(v, ["neg"], use_abs=True) == 3.0


def test_holding_revenue_includes_finance_revenue():
    rows = [
        (1, c, "2019-01-01", "2019-12-31", x, 4)
        for c, x in [
            ("ifrs-full_Revenue", 18.0),
            ("kap-fr_RevenueFromFinanceSectorOperations", 41.0),
            ("ifrs-full_GrossProfit", 17.0),
        ]
    ]
    w = filings_wide(_long(rows), pd.Series({"XXX": "industrial"}), CFG)
    assert w.loc[0, "revenue_cur"] == 59.0
    r = compute_ratios({"revenue_ttm": 59.0, "gross_profit_ttm": 17.0}, 1000.0, "industrial")
    assert r["gross_margin"] == pytest.approx(17 / 59)


def test_bank_concepts_participation_and_personnel():
    rows = [
        (1, c, "2019-01-01", "2019-12-31", x, 4)
        for c, x in [
            ("kap-fr_ProfitShareIncomeOrExpense", 100.0),
            ("kap-fr_OtherOperatingExpenses", -30.0),
            ("kap-fr_PersonnelExpenses", -20.0),
        ]
    ]
    rows += [(1, "kap-fr_FundsCollected", None, "2019-12-31", 500.0, 4)]
    w = filings_wide(_long(rows), pd.Series({"XXX": "bank"}), CFG)
    assert w.loc[0, "net_interest_income_cur"] == 100.0
    assert w.loc[0, "operating_expenses_cur"] == -50.0 and w.loc[0, "deposits_cur"] == 500.0


def test_zero_parent_profit_falls_back_and_interest_sign():
    rows = [
        (1, c, "2019-01-01", "2019-12-31", x, 4)
        for c, x in [
            ("ifrs-full_ProfitLossAttributableToOwnersOfParent", 0.0),
            ("ifrs-full_ProfitLoss", 12.0),
            ("ifrs-full_FinanceCosts", -4.0),
        ]
    ]
    w = filings_wide(_long(rows), pd.Series({"XXX": "industrial"}), CFG)
    assert w.loc[0, "net_profit_parent_cur"] == 12.0 and w.loc[0, "interest_expense_cur"] == 4.0


def test_leap_day_prior_column_is_found():
    rows = [
        (1, "ifrs-full_Revenue", "2020-03-01", "2021-02-28", 100.0, 4),
        (1, "ifrs-full_Revenue", "2019-03-01", "2020-02-29", 80.0, 4),
    ]
    w = filings_wide(_long(rows), pd.Series({"XXX": "industrial"}), CFG)
    assert w.loc[0, "revenue_prior"] == 80.0


def _w(rows):
    recs = [{"ticker": "XXX", "consolidated": True, "template": "bank", "basis": "nominal", **r} for r in rows]
    out = pd.DataFrame(recs)
    for c in ("publish_ts", "availability_date"):
        out[c] = pd.to_datetime(out[c])
    return out


def _r(idx, fy, period, pub, cur, prior, **kw):
    return {
        "disclosure_index": idx,
        "fiscal_year": fy,
        "period": period,
        "publish_ts": pub,
        "availability_date": pub,
        "x_cur": cur,
        "x_prior": prior,
        **kw,
    }


def test_missing_comparative_uses_prior_year_filing():
    w = _w(
        [
            _r(1, 2017, 4, "2018-02-01", 100.0, 90.0),
            _r(2, 2017, 1, "2017-05-01", 20.0, np.nan),
            _r(3, 2018, 1, "2018-05-01", 30.0, np.nan),
            _r(4, 2018, 1, "2018-05-10", 30.0, 0.0),
        ]
    )
    t = add_ttm(w, ["x"]).set_index("disclosure_index")
    assert t.loc[3, "x_ttm"] == 110.0 and t.loc[3, "ttm_method"] == "fy_plus_ytd_delta_prior_filing"
    assert t.loc[4, "x_ttm"] == 110.0 and t.loc[4, "x_prior"] == 20.0  # parsed 0 comparative = missing


def test_nominal_ttm_kept_despite_date_derived_basis_label_and_missing_prior_flag():
    w = _w(
        [
            _r(1, 2023, 4, "2023-12-02", 100.0, 90.0),
            _r(2, 2024, 2, "2024-05-02", 60.0, 40.0, basis="tms29"),
            _r(3, 2030, 2, "2030-05-02", np.nan, np.nan, basis="tms29"),
        ]
    )
    t = add_ttm(w, ["x"]).set_index("disclosure_index")
    assert t.loc[2, "x_ttm"] == 100.0 + 60.0 - 40.0 and t.loc[2, "ttm_method"] == "fy_plus_ytd_delta"
    assert t.loc[3, "ttm_method"] == "missing_prior"


def test_nci_in_ev_fcf_and_roic():
    row = {
        "net_profit_parent_ttm": 100.0,
        "equity_parent_cur": 500.0,
        "equity_cur": 1500.0,
        "assets_cur": 3000.0,
        "ebit_ttm": 200.0,
        "depreciation_ttm": 0.0,
        "st_debt_cur": 400.0,
        "cash_cur": 0.0,
        "cfo_ttm": 300.0,
        "capex_ttm": -100.0,
        "revenue_ttm": 1000.0,
    }
    r = compute_ratios(row, mcap_tl=1000.0, template="industrial", tax_rate=0.2)
    ev = 1000 + 1000 + 400  # mcap + NCI + net debt
    assert r["ebitda_yield"] == pytest.approx(200 / ev) and r["fcf_yield"] == pytest.approx(200 / ev)
    assert r["roic"] == pytest.approx(200 * 0.8 / (1500 + 400))
    assert r["earnings_yield"] == pytest.approx(0.1) and r["roe"] == pytest.approx(0.2)
    row2 = {k: v for k, v in row.items() if k not in ("st_debt_cur", "cash_cur")}
    r2 = compute_ratios(row2, mcap_tl=1000.0, template="industrial", tax_rate=0.2)
    assert r2["fcf_yield"] == pytest.approx(200 / 2000) and np.isnan(r2["ebitda_yield"])
