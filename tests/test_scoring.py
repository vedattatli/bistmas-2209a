"""Scoring layer on a small synthetic world: 12 tickers, 2 sector groups, monthly filings."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from bistmas.data.universe.provider import MEMBERSHIP_COLUMNS
from bistmas.scoring.composite import ALL_COMPONENTS, composite_score, pillar_scores
from bistmas.scoring.features import FeatureInputs, PricePanel, build_features_one
from bistmas.scoring.normalize import normalise_features, pool_small_groups, robust_z
from bistmas.scoring.ratios import compute_ratios
from bistmas.scoring.statements_view import add_ttm, latest_filing_as_of
from bistmas.shared.config import load_settings

TICKERS = [f"T{i:02d}" for i in range(1, 13)]


def _prices() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    sessions = pd.bdate_range("2017-01-02", "2019-12-31")
    rows = []
    for j, t in enumerate(TICKERS):
        drift = 0.0004 * (j - 6)
        rets = rng.normal(drift, 0.02, len(sessions))
        close = 10 * np.exp(np.cumsum(rets))
        for d, c in zip(sessions, close, strict=True):
            rows.append(
                {
                    "ticker": t,
                    "date": d,
                    "close_adj": c,
                    "close_raw": c,
                    "vwap_adj": c * 1.001,
                    "traded_value_tl": 1e6 * (j + 1),
                    "mcap_tl": 1e9 * (j + 1),
                    "mcap_tl_aligned": 1e9 * (j + 1),
                    "ffloat_mcap_tl": 3e8 * (j + 1),
                    "flag_cancelled_session": False,
                }
            )
    return pd.DataFrame(rows)


def _membership() -> pd.DataFrame:
    return pd.DataFrame(
        [("XU100", t, pd.Timestamp("2017-01-02"), pd.NaT, pd.Timestamp("2016-12-20"), "anchor") for t in TICKERS],
        columns=MEMBERSHIP_COLUMNS,
    )


def _filings() -> pd.DataFrame:
    rows = []
    for j, t in enumerate(TICKERS):
        base = 1e9 * (j + 1)
        for fy, period, pend, avail in [
            (2017, 4, "2017-12-31", "2018-03-01"),
            (2018, 1, "2018-03-31", "2018-05-10"),
            (2018, 2, "2018-06-30", "2018-08-10"),
        ]:
            frac = {4: 1.0, 1: 0.25, 2: 0.5}[period]
            rows.append(
                {
                    "ticker": t,
                    "mkk_member_oid": f"oid{j}",
                    "disclosure_index": 1000 + j * 10 + period,
                    "fiscal_year": fy,
                    "period": period,
                    "period_end": pd.Timestamp(pend),
                    "publish_ts": pd.Timestamp(avail) - pd.Timedelta(days=1),
                    "availability_date": pd.Timestamp(avail),
                    "consolidated": True,
                    "template": "industrial" if j < 10 else "bank",
                    "basis": "nominal",
                    "revenue_cur": base * frac,
                    "revenue_prior": base * frac * 0.8,
                    "gross_profit_cur": base * frac * 0.3,
                    "gross_profit_prior": base * frac * 0.25,
                    "operating_profit_cur": base * frac * 0.15,
                    "operating_profit_prior": base * frac * 0.12,
                    "ebit_cur": base * frac * 0.16,
                    "ebit_prior": base * frac * 0.13,
                    "net_profit_parent_cur": base * frac * 0.1,
                    "net_profit_parent_prior": base * frac * 0.08,
                    "depreciation_cur": base * frac * 0.05,
                    "depreciation_prior": base * frac * 0.05,
                    "cfo_cur": base * frac * 0.14,
                    "cfo_prior": base * frac * 0.12,
                    "capex_cur": -base * frac * 0.06,
                    "capex_prior": -base * frac * 0.05,
                    "interest_expense_cur": base * frac * 0.02,
                    "interest_expense_prior": base * frac * 0.02,
                    "cash_cur": base * 0.2,
                    "st_investments_cur": 0.0,
                    "st_debt_cur": base * 0.1,
                    "st_debt_ltportion_cur": base * 0.05,
                    "lt_debt_cur": base * 0.3,
                    "equity_parent_cur": base * 0.6 * (1 if j != 3 else -1),  # T04 has negative equity
                    "assets_cur": base * 1.5,
                }
            )
    return pd.DataFrame(rows)


def _master() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": TICKERS,
            "sector_group": ["MANUFACTURING"] * 6 + ["RETAIL"] * 4 + ["BANKS"] * 2,
            "template": ["industrial"] * 10 + ["bank"] * 2,
        }
    ).set_index("ticker")


@pytest.fixture(scope="module")
def inputs() -> FeatureInputs:
    filings = add_ttm(
        _filings(),
        [
            "revenue",
            "gross_profit",
            "operating_profit",
            "ebit",
            "net_profit_parent",
            "depreciation",
            "cfo",
            "capex",
            "interest_expense",
        ],
    )
    return FeatureInputs(_membership(), PricePanel.from_long(_prices()), filings, _master(), {2017: 0.2, 2018: 0.22})


def test_ttm_and_latest_filing(inputs: FeatureInputs):
    f = inputs.filings
    q2 = f[(f["ticker"] == "T01") & (f["period"] == 2)].iloc[0]
    # FY2017 + YTD(H1 2018) - YTD(H1 2017): 1e9 + 0.5e9 - 0.4e9
    assert q2["revenue_ttm"] == pytest.approx(1.1e9) and q2["ttm_method"] == "fy_plus_ytd_delta"
    latest = latest_filing_as_of(f, date(2018, 6, 29))
    assert set(latest["period"]) == {1}  # H1 filing not yet available on 29 June
    latest = latest_filing_as_of(f, date(2018, 8, 31))
    assert set(latest["period"]) == {2}


def test_ratios_orientation_and_bank_subset():
    row = {
        "net_profit_parent_ttm": 100.0,
        "equity_parent_cur": 500.0,
        "assets_cur": 2000.0,
        "net_profit_parent_cur": 60.0,
        "net_profit_parent_prior": 50.0,
        "revenue_ttm": 1000.0,
        "ebit_ttm": 150.0,
        "depreciation_ttm": 50.0,
        "st_debt_cur": 100.0,
        "st_debt_ltportion_cur": 50.0,
        "lt_debt_cur": 350.0,
        "cash_cur": 100.0,
        "st_investments_cur": 0.0,
        "cfo_ttm": 140.0,
        "capex_ttm": -60.0,
        "gross_profit_ttm": 300.0,
        "operating_profit_ttm": 140.0,
        "revenue_cur": 500.0,
        "revenue_prior": 400.0,
        "interest_expense_ttm": 30.0,
    }
    r = compute_ratios(row, mcap_tl=1000.0, template="industrial", tax_rate=0.2)
    assert r["earnings_yield"] == pytest.approx(0.1) and r["book_to_market"] == pytest.approx(0.5)
    assert r["ebitda_yield"] == pytest.approx(200 / (1000 + 400))  # EV = mcap + net debt 400
    assert r["fcf_yield"] == pytest.approx(80 / 1400) and r["roic"] == pytest.approx(150 * 0.8 / (500 + 400))
    assert r["revenue_growth"] == pytest.approx(0.25) and r["profit_growth"] == pytest.approx(0.2)
    assert r["net_debt_to_ebitda"] == pytest.approx(2.0) and r["interest_coverage"] == pytest.approx(5.0)
    b = compute_ratios(row, mcap_tl=1000.0, template="bank")
    assert b["roe"] == pytest.approx(0.2) and np.isnan(b["ebitda_yield"]) and np.isnan(b["revenue_growth"])
    assert all(np.isnan(v) for v in compute_ratios(row, mcap_tl=np.nan, template="industrial").values())


def test_features_are_point_in_time(inputs: FeatureInputs):
    f = build_features_one(date(2018, 6, 29), inputs)
    assert len(f) == 12 and (f["max_source_available_date"] <= pd.Timestamp("2018-06-29")).all()
    assert (f["statement_period_end"] == pd.Timestamp("2018-03-31")).all()
    assert f["momentum_12_1"].notna().all() and f["vol_6m"].notna().all() and (f["listed_sessions"] > 250).all()
    early = build_features_one(date(2017, 6, 30), inputs)
    assert (
        early["statement_period_end"].isna().all() and early["momentum_12_1"].isna().all()
    )  # < 252 sessions, no filing


def test_normalisation_pools_small_groups_and_orients_signs():
    df = pd.DataFrame(
        {
            "sector_group": ["A"] * 6 + ["B"] * 2,
            "template": ["industrial"] * 8,
            "vol_6m": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.2, 0.9],
            "adv_6m_tl": [1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 1e6, 2e6],
        }
    )
    out = normalise_features(df, ["vol_6m", "adv_6m_tl"], min_group_size=5)
    assert set(out["norm_group"]) == {"A", "NON_FINANCIALS"}  # B has 2 rows -> its parent (unknown group default)
    a = out[out["norm_group"] == "A"]
    assert a["z_vol_6m"].iloc[0] > a["z_vol_6m"].iloc[-1]  # lower volatility scores higher
    assert a["z_adv_6m_tl"].iloc[-1] > a["z_adv_6m_tl"].iloc[0]
    assert robust_z(pd.Series([1.0, 1.0, 1.0])).tolist() == [0.0, 0.0, 0.0]
    assert pool_small_groups(pd.Series(["X", "X", "Y"]), 2).tolist() == ["X", "X", "POOLED"]


def test_screen_end_to_end(inputs: FeatureInputs):
    settings = load_settings()
    settings.universe.min_adv_tl_6m = 0.0  # the synthetic world trades a few million TL a day
    from bistmas.scoring.screen import screen_one

    out = screen_one(date(2018, 8, 31), inputs, settings)
    assert len(out) == 12
    assert out.loc[out["ticker"] == "T04", "filter_reason"].iloc[0] == "negative_equity"
    elig = out[out["eligible"]]
    assert elig["composite"].notna().sum() >= 8 and (elig["n_pillars"] == 3).all()
    assert out["top_n"].sum() == min(10, int(elig["composite"].notna().sum()))
    ranked = elig.dropna(subset=["composite"]).sort_values("rank")
    assert ranked["composite"].is_monotonic_decreasing
    # z-scores are computed only among eligible names, per pooled sector group
    assert set(elig["norm_group"]) <= {"MANUFACTURING", "NON_FINANCIALS", "FINANCIALS", "ALL"}


def test_pillars_need_two_components():
    df = pd.DataFrame(
        {
            "template": ["industrial", "industrial"],
            "z_roic": [1.0, np.nan],
            "z_operating_margin": [0.5, 0.2],
            "z_gross_margin": [np.nan, np.nan],
        }
    )
    out = pillar_scores(df)
    assert out["S_fundamental"].iloc[0] == pytest.approx(0.75) and np.isnan(out["S_fundamental"].iloc[1])
    out["S_valuation"] = [1.0, 1.0]
    out["S_momentum_risk"] = [np.nan, 1.0]
    comp = composite_score(out)
    assert np.isnan(comp["composite"].iloc[0]) and np.isnan(comp["composite"].iloc[1])  # require all pillars
    comp2 = composite_score(out, require_all=False)
    assert comp2["composite"].iloc[0] == pytest.approx((0.75 + 1.0) / 2)
    assert len(ALL_COMPONENTS) >= 12


def test_screen_rank_survives_parquet_round_trip_numerically(tmp_path):
    """A stored screen must sort 2 before 10: the rank is a nullable integer, not a string."""
    import pandas as pd

    from bistmas.shared.store import read_parquet, write_parquet

    scored = pd.DataFrame({"ticker": [f"T{i:02d}" for i in range(12)], "composite": [12 - i for i in range(12)]})
    scored["rank"] = pd.array(range(1, 13), dtype="Int64")
    rest = pd.DataFrame({"ticker": ["ZZZZ"], "composite": [pd.NA]})
    rest["rank"] = pd.array([pd.NA], dtype="Int64")
    frame = pd.concat([scored, rest], ignore_index=True)
    write_parquet(frame, tmp_path / "s.parquet")
    back = read_parquet(tmp_path / "s.parquet")
    order = back.dropna(subset=["rank"]).sort_values("rank")["ticker"].tolist()
    assert order[:11] == [f"T{i:02d}" for i in range(11)]
    assert str(back["rank"].dtype) in ("Int64", "int64", "float64")
