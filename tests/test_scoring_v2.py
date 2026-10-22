"""prereg-v2 scoring corrections: pooling hierarchy, basis-split growth, flagged returns, listing age."""

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest
from test_scoring import _membership, _prices, inputs  # noqa: F401

from bistmas.scoring.features import PricePanel, listed_since_from_membership
from bistmas.scoring.normalize import load_parent_map, normalise_features
from bistmas.scoring.screen import screen_one
from bistmas.shared.config import load_settings

PARENTS = {"A": "NON_FINANCIALS", "B": "NON_FINANCIALS", "C": "FINANCIALS", "D": "FINANCIALS"}


def test_parent_map_from_csv_and_other_financial_template():
    pm = load_parent_map()
    assert pm["BANKS"] == pm["INSURANCE"] == pm["OTHER_FINANCIAL"] == "FINANCIALS"
    assert pm["MANUFACTURING"] == pm["RETAIL"] == "NON_FINANCIALS"
    csv = pd.read_csv("configs/sector_groups.csv", dtype=str)
    assert csv.loc[csv["group"] == "OTHER_FINANCIAL", "template"].iloc[0] == "other_financial"


def test_pooling_goes_to_parent_then_all_and_counts_non_null_values():
    # A has 6 names but only 3 non-null vol values -> that feature pools into the parent (A+B = 6 values)
    df = pd.DataFrame(
        {
            "sector_group": ["A"] * 6 + ["B"] * 3 + ["C"] * 2,
            "vol_6m": [0.1, 0.2, 0.3, np.nan, np.nan, np.nan, 0.4, 0.5, 0.6, 0.7, 0.8],
            "mdd_6m": [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.1, -0.2, -0.3, -0.1, -0.2],
        }
    )
    out = normalise_features(df, ["vol_6m", "mdd_6m"], min_group_size=5, parent_map=PARENTS)
    assert out["norm_group"].tolist() == ["A"] * 6 + ["NON_FINANCIALS"] * 3 + ["ALL"] * 2
    vol = df.loc[:8, "vol_6m"].dropna()
    med = vol.median()
    expected = -((vol - med) / (1.4826 * (vol - med).abs().median())).clip(-3, 3)
    assert out["z_vol_6m"].iloc[:9].dropna().tolist() == pytest.approx(expected.tolist())
    assert out["z_vol_6m"].iloc[9:].notna().all()  # FINANCIALS has 2 values -> all eligible names of the month


def test_growth_is_compared_within_basis_in_mixed_months():
    n = 12
    df = pd.DataFrame(
        {
            "sector_group": ["A"] * n,
            "basis": ["nominal"] * 6 + ["tms29"] * 6,
            "revenue_growth": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            "roe": np.arange(n, dtype=float),
        }
    )
    out = normalise_features(df, ["revenue_growth", "roe"], min_group_size=5, parent_map=PARENTS)
    assert out["z_revenue_growth"].iloc[:6].tolist() == pytest.approx(out["z_revenue_growth"].iloc[6:].tolist())
    assert out["z_roe"].iloc[0] < 0 < out["z_roe"].iloc[-1]  # non-growth features ignore the basis
    df2 = df.copy()
    df2["basis"] = ["nominal"] * 3 + ["tms29"] * 9
    out2 = normalise_features(df2, ["revenue_growth"], min_group_size=5, parent_map=PARENTS)
    assert out2["z_revenue_growth"].iloc[:3].isna().all() and out2["z_revenue_growth"].iloc[3:].notna().all()
    out3 = normalise_features(df.assign(basis="tms29"), ["revenue_growth"], min_group_size=5, parent_map=PARENTS)
    assert out3["z_revenue_growth"].notna().all()  # a single basis behaves as before


def _panel_with_flag(flag_day: int | None) -> PricePanel:
    sessions = pd.bdate_range("2017-01-02", periods=300)
    close = pd.Series(10.0 * (1.001 ** np.arange(300)), index=sessions)
    if flag_day is not None:
        close.iloc[flag_day:] *= 0.5  # a vendor misadjustment halves the price from the flagged session on
    rows = [
        {
            "ticker": "X",
            "date": d,
            "close_adj": c,
            "close_raw": c,
            "traded_value_tl": 1e7,
            "mcap_tl": 1e9,
            "ffloat_mcap_tl": 1e8,
            "flag_cancelled_session": False,
            "flag_capital_change": flag_day is not None and i == flag_day,
            "flag_adj_mismatch": False,
        }
        for i, (d, c) in enumerate(close.items())
    ]
    return PricePanel.from_long(pd.DataFrame(rows))


def test_flagged_returns_are_excluded_from_momentum_vol_and_drawdown():
    clean = _panel_with_flag(None)
    flagged = _panel_with_flag(250)
    as_of = clean.close_adj.index[-1]
    m0 = clean.metrics_at(as_of, ["X"]).loc["X"]
    m1 = flagged.metrics_at(as_of, ["X"]).loc["X"]
    # the flagged session contributes a zero return instead of its 0.1 % drift (and not the -50 % jump)
    assert 1 + m1["momentum_12_1"] == pytest.approx((1 + m0["momentum_12_1"]) / 1.001, rel=1e-9)
    assert m1["vol_6m"] == pytest.approx(m0["vol_6m"], abs=1e-9)
    assert m1["mdd_6m"] == pytest.approx(m0["mdd_6m"], abs=1e-9)
    unflagged = replace(flagged, flags=None)
    assert unflagged.metrics_at(as_of, ["X"]).loc["X", "vol_6m"] > 10 * m0["vol_6m"] + 1e-6


def test_pivots_share_the_close_index_so_windows_cannot_look_ahead():
    prices = _prices()
    prices = prices[prices["ticker"] == "T01"].copy()
    gap = prices["date"] == pd.Timestamp("2018-03-05")
    prices.loc[gap, ["close_adj", "close_raw", "traded_value_tl", "mcap_tl", "mcap_tl_aligned"]] = np.nan
    prices.loc[gap, "ffloat_mcap_tl"] = np.nan  # all-NaN session: pivot_table drops the date
    # the matrices must all be reindexed to close_adj, whatever pivot_table dropped
    prices.loc[gap, "close_adj"] = 10.0
    panel = PricePanel.from_long(prices)
    for m in (panel.traded_value, panel.mcap, panel.ffloat_mcap, panel.close_raw):
        assert m.index.equals(panel.close_adj.index) and list(m.columns) == list(panel.close_adj.columns)
    prices.loc[prices["date"] > pd.Timestamp("2018-03-06"), "ffloat_mcap_tl"] = 1e12  # future value
    out = PricePanel.from_long(prices).metrics_at(pd.Timestamp("2018-03-06"), ["T01"])
    assert out.loc["T01", "ffloat_mcap_tl"] == pytest.approx(3e8)


def test_listing_age_uses_earliest_of_vendor_price_and_membership():
    prices = _prices()
    late = prices[(prices["ticker"] == "T01") & (prices["date"] < "2019-06-01")].index
    panel = PricePanel.from_long(prices.drop(late))  # vendor data for T01 starts 2019-06
    ts = pd.Timestamp("2019-09-30")
    assert panel.metrics_at(ts, ["T01"]).loc["T01", "listed_sessions"] < 250
    ls = listed_since_from_membership(_membership())
    assert ls["T01"] == pd.Timestamp("2017-01-02")
    assert panel.metrics_at(ts, ["T01"], ls).loc["T01", "listed_sessions"] > 600


def test_listing_age_before_panel_start_counts_missing_weekdays():
    panel = PricePanel.from_long(_prices())
    ts = pd.Timestamp("2017-03-31")
    n0 = panel.metrics_at(ts, ["T01"]).loc["T01", "listed_sessions"]
    n1 = panel.metrics_at(ts, ["T01"], {"T01": pd.Timestamp("2010-01-04")}).loc["T01", "listed_sessions"]
    assert n1 > 1500 > n0


def test_unrankable_names_get_no_composite_reason(inputs):  # noqa: F811
    settings = load_settings()
    settings.universe.min_adv_tl_6m = 0.0
    f = inputs.filings.copy()
    cols = [
        c
        for c in f.columns
        if c.endswith(("_cur", "_prior")) and c.split("_")[0] in ("revenue", "gross", "operating", "ebit")
    ]
    f.loc[f["ticker"] == "T02", cols] = np.nan
    out = screen_one(date(2018, 8, 31), replace(inputs, filings=f), settings)
    assert not (out["eligible"] & out["composite"].isna()).any()
    bad = out[out["filter_reason"] == "no_composite"]
    assert not bad["eligible"].any() and bad["rank"].isna().all()


def test_filter_parameters_are_typed_config_fields():
    s = load_settings()
    assert (s.scoring.min_listed_sessions, s.scoring.adv_percentile_floor, s.scoring.max_statement_age_days) == (
        250,
        0.10,
        400,
    )


def test_listed_since_ignores_rows_not_yet_known_at_decision_date():
    m = _membership().copy()
    m["membership_known_since"] = pd.Timestamp("2021-01-01")
    assert "T01" not in listed_since_from_membership(m, as_of=pd.Timestamp("2019-01-01"))
    assert listed_since_from_membership(m, as_of=pd.Timestamp("2022-01-01"))["T01"] == pd.Timestamp("2017-01-02")
