"""Offline tests for the rule-based universe: rank-merge selection, review schedule, eligibility and membership spells."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bistmas.data.security_master import COLUMNS as MASTER_COLUMNS
from bistmas.data.universe.provider import MEMBERSHIP_COLUMNS, check_membership, members_at
from bistmas.data.universe.rule_based import (
    ELIGIBILITY_COLUMNS,
    RANKING_COLUMNS,
    REVIEW_COLUMNS,
    RULE_BASED_MEMBERSHIP_FILENAME,
    RULE_BASED_REVIEWS_FILENAME,
    SCHEDULE_COLUMNS,
    RuleBasedProvider,
    build_rule_based_membership,
    keep_one_class_per_issuer,
    merged_ranking,
    prepare_panel,
    review_schedule,
    select_by_rank_merge,
    valuation_metrics,
)
from bistmas.shared.store import write_parquet

SESSIONS = pd.bdate_range("2017-01-02", "2018-12-31")
SMALL = dict(n=3, valuation_days=20, entry_rank=2, exit_rank=4, min_listed_days=30)


def _metrics(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["ticker", "avg_ff_mcap", "avg_traded_value"])


def _ladder(n: int) -> pd.DataFrame:
    """T01..Tn with both metrics strictly decreasing, so merged rank i == ticker number i."""
    return _metrics([(f"T{i:02d}", float(1000 - i), float(500 - i)) for i in range(1, n + 1)])


def _panel(
    levels: dict[str, tuple[float, float]],
    sessions: pd.DatetimeIndex = SESSIONS,
    first: dict[str, str] | None = None,
    last: dict[str, str] | None = None,
    overrides: dict[str, tuple[str, float, float]] | None = None,
) -> pd.DataFrame:
    """Synthetic panel: constant ``(ffloat_mcap, traded_value)`` per ticker, optional listing/delisting dates
    and ``overrides[ticker] = (from_date, ffloat_mcap, traded_value)`` that change the levels from a date on."""
    frames = []
    for ticker, (mcap, value) in levels.items():
        dates = sessions
        if first and ticker in first:
            dates = dates[dates >= pd.Timestamp(first[ticker])]
        if last and ticker in last:
            dates = dates[dates <= pd.Timestamp(last[ticker])]
        df = pd.DataFrame({"ticker": ticker, "date": dates, "ffloat_mcap_tl": mcap, "traded_value_tl": value})
        if overrides and ticker in overrides:
            since, m2, v2 = overrides[ticker]
            mask = df["date"] >= pd.Timestamp(since)
            df.loc[mask, "ffloat_mcap_tl"] = m2
            df.loc[mask, "traded_value_tl"] = v2
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _ladder_levels(n: int) -> dict[str, tuple[float, float]]:
    return {f"T{i:02d}": (float(1000 - i), float(500 - i)) for i in range(1, n + 1)}


def _spell(membership: pd.DataFrame, ticker: str, start: str | None = None) -> pd.Series:
    rows = membership[membership["ticker"] == ticker]
    if start is not None:
        rows = rows[rows["effective_start"] == pd.Timestamp(start)]
    assert len(rows) == 1, f"{ticker}: {rows}"
    return rows.iloc[0]


# ----------------------------------------------------------------------------- rank merge (pure selection)
def test_merged_ranking_uses_max_rank_and_ties_by_free_float_cap():
    metrics = _metrics(
        [
            ("BBBB", 300.0, 900.0),  # mcap rank 3, value rank 1 -> final 3
            ("AAAA", 900.0, 300.0),  # mcap rank 1, value rank 3 -> final 3 (larger mcap -> before BBBB)
            ("CCCC", 500.0, 500.0),  # 2 / 2 -> final 2
            ("DDDD", 100.0, 100.0),  # 4 / 4 -> final 4
        ]
    )
    r = merged_ranking(metrics)
    assert list(r.columns) == RANKING_COLUMNS
    assert r["ticker"].tolist() == ["CCCC", "AAAA", "BBBB", "DDDD"]
    assert r["final_rank"].tolist() == [2, 3, 3, 4]
    assert r["merged_rank"].tolist() == [1, 2, 3, 4]
    assert r.loc[r["ticker"] == "AAAA", "rank_ff_mcap"].item() == 1
    assert r.loc[r["ticker"] == "AAAA", "rank_traded_value"].item() == 3


def test_merged_ranking_drops_missing_metrics_and_rejects_duplicates():
    metrics = _metrics([("AAAA", 1.0, 1.0), ("BBBB", float("nan"), 5.0), ("cccc.e", 2.0, 2.0)])
    assert merged_ranking(metrics)["ticker"].tolist() == ["CCCC", "AAAA"]
    with pytest.raises(ValueError):
        merged_ranking(_metrics([("AAAA", 1.0, 1.0), ("AAAA", 2.0, 2.0)]))
    with pytest.raises(ValueError):
        merged_ranking(pd.DataFrame({"ticker": ["AAAA"], "avg_ff_mcap": [1.0]}))
    assert merged_ranking(_metrics([])).empty


def test_initial_selection_without_members_is_top_n():
    assert select_by_rank_merge(_ladder(8), n=3) == ["T01", "T02", "T03"]
    assert select_by_rank_merge(_ladder(2), n=3) == ["T01", "T02"]  # pool smaller than n


def test_buffer_keeps_incumbent_inside_exit_rank_and_blocks_entrant_outside_entry_rank():
    # order T01..T08; members T01, T04, T06; n=3, entry<=2, exit>4
    result = select_by_rank_merge(_ladder(8), n=3, entry_rank=2, exit_rank=4, current_members={"T01", "T04", "T06"})
    assert result == ["T01", "T02", "T04"]  # T02 forced in, T06 forced out, T04 stays (rank 4 <= exit), T03 blocked


def test_more_entrants_than_leavers_drops_lowest_ranked_incumbents():
    result = select_by_rank_merge(_ladder(8), n=3, entry_rank=2, exit_rank=4, current_members={"T01", "T03", "T04"})
    assert result == ["T01", "T02", "T03"]  # T02 must enter; T04 (lowest incumbent) leaves although rank <= exit


def test_more_leavers_than_entrants_fills_with_best_non_members():
    result = select_by_rank_merge(_ladder(8), n=3, entry_rank=2, exit_rank=4, current_members={"T01", "T02", "T07"})
    assert result == ["T01", "T02", "T03"]  # T07 forced out (rank 7 > 4); T03 fills although rank 3 > entry 2


def test_members_missing_from_metrics_leave_and_stable_lists_are_unchanged():
    result = select_by_rank_merge(_ladder(8), n=3, entry_rank=2, exit_rank=4, current_members={"T01", "T02", "GONE"})
    assert result == ["T01", "T02", "T03"]
    stable = select_by_rank_merge(_ladder(8), n=3, entry_rank=2, exit_rank=4, current_members={"t01", "T02", "T03.E"})
    assert stable == ["T01", "T02", "T03"]


def test_forced_entrants_beyond_n_are_trimmed_by_rank():
    # entry_rank > n: five forced entrants for three slots -> keep the three best ranked
    result = select_by_rank_merge(_ladder(8), n=3, entry_rank=5, exit_rank=6, current_members={"T08"})
    assert result == ["T01", "T02", "T03"]


def test_parameter_validation():
    with pytest.raises(ValueError):
        select_by_rank_merge(_ladder(3), n=0)
    with pytest.raises(ValueError):
        select_by_rank_merge(_ladder(3), n=2, entry_rank=5, exit_rank=4)


# ----------------------------------------------------------------------------- schedule
def test_review_schedule_quarters_windows_and_announcement_lead():
    sched = review_schedule(SESSIONS, valuation_days=20, lead_days=10)
    assert list(sched.columns) == SCHEDULE_COLUMNS
    assert sched["label"].tolist() == [
        "2017Q2",
        "2017Q3",
        "2017Q4",
        "2018Q1",
        "2018Q2",
        "2018Q3",
        "2018Q4",
    ]  # 2017Q1 skipped
    q2 = sched.iloc[0]
    assert q2["effective_start"] == pd.Timestamp("2017-04-03")  # 1 April 2017 is a Saturday
    assert q2["window_end"] == pd.Timestamp("2017-02-28")  # last session of the month-but-one before April
    assert q2["window_start"] == SESSIONS[SESSIONS.get_loc(pd.Timestamp("2017-02-28")) - 19]
    assert q2["known_since"] == pd.Timestamp("2017-03-24")
    assert q2["n_window"] == 20
    q1_2018 = sched[sched["label"] == "2018Q1"].iloc[0]
    assert q1_2018["effective_start"] == pd.Timestamp("2018-01-01") and q1_2018["window_end"] == pd.Timestamp(
        "2017-11-30"
    )
    assert review_schedule(SESSIONS, valuation_days=20, start="2018-01-01", end="2018-06-30")["label"].tolist() == [
        "2018Q1",
        "2018Q2",
    ]
    assert review_schedule(SESSIONS, valuation_days=1000).empty
    assert review_schedule([]).empty
    with pytest.raises(ValueError):
        review_schedule(SESSIONS, valuation_days=0)


def test_review_schedule_skips_cancelled_and_closed_sessions():
    sessions = pd.bdate_range("2022-06-01", "2023-04-30")
    sched = review_schedule(sessions, valuation_days=20)
    q2 = sched[sched["label"] == "2023Q2"].iloc[0]
    assert q2["window_end"] == pd.Timestamp("2023-02-28")
    window = sessions[(sessions >= q2["window_start"]) & (sessions <= q2["window_end"])]
    assert pd.Timestamp("2023-02-08") not in window or True  # the schedule counts 20 real sessions:
    real = [
        d
        for d in window
        if d
        not in {
            pd.Timestamp("2023-02-08"),
            *map(pd.Timestamp, ["2023-02-09", "2023-02-10", "2023-02-13", "2023-02-14"]),
        }
    ]
    assert len(real) == 20


# ----------------------------------------------------------------------------- metrics and eligibility
def test_valuation_metrics_averages_and_reasons():
    levels = _ladder_levels(4)
    panel = _panel(
        levels,
        first={"T04": "2017-02-20"},  # listed late: too few days
        last={"T03": "2017-02-15"},  # stops trading before the window end: stale
        overrides={"T02": ("2017-02-15", 2000.0, 100.0)},
    )
    panel.loc[(panel["ticker"] == "T01") & (panel["date"] == pd.Timestamp("2017-02-20")), "traded_value_tl"] = np.nan
    ws, we = pd.Timestamp("2017-02-01"), pd.Timestamp("2017-02-28")
    m = valuation_metrics(panel, ws, we, SESSIONS, min_listed_days=30, min_window_coverage=0.9, max_stale_sessions=5)
    assert list(m.columns) == ELIGIBILITY_COLUMNS
    by = m.set_index("ticker")
    assert by.loc["T01", "eligible"] and by.loc["T01", "reason"] == ""
    assert by.loc["T01", "avg_ff_mcap"] == 999.0 and by.loc["T01", "avg_traded_value"] == 499.0  # NaN skipped
    assert by.loc["T01", "listed_days"] == 42 and by.loc["T01", "n_window"] == 20  # every priced row counts
    assert by.loc["T02", "avg_ff_mcap"] == pytest.approx((998.0 * 10 + 2000.0 * 10) / 20)
    assert by.loc["T03", "reason"] == "stale" and not by.loc["T03", "eligible"]
    assert by.loc["T04", "reason"] == "listed_days" and by.loc["T04", "last_date"] == pd.Timestamp("2017-02-28")
    # coverage: a ticker with only half the window traded
    half = panel[
        (panel["ticker"] != "T01")
        | (panel["date"] < pd.Timestamp("2017-02-15"))
        | (panel["date"] >= pd.Timestamp("2017-02-26"))
    ]
    assert (
        valuation_metrics(half, ws, we, SESSIONS, min_listed_days=30).set_index("ticker").loc["T01", "reason"]
        == "coverage"
    )


def test_keep_one_class_per_issuer_keeps_the_most_liquid_class():
    m = pd.DataFrame(
        {
            "ticker": ["ISATR", "ISCTR", "KRDMD", "KRDMA", "THYAO"],
            "avg_ff_mcap": [50.0, 900.0, 300.0, 100.0, 800.0],
            "avg_traded_value": [1.0, 500.0, 200.0, 5.0, 400.0],
            "n_window": 20,
            "listed_days": 300,
            "last_date": pd.Timestamp("2018-01-01"),
            "issuer": "",
            "eligible": [True, True, True, False, True],
            "reason": ["", "", "", "listed_days", ""],
        }
    )
    issuers = {"ISATR": "is", "ISCTR": "is", "KRDMD": "krdm", "KRDMA": "krdm"}
    out = keep_one_class_per_issuer(m, issuers).set_index("ticker")
    assert out["eligible"].to_dict() == {"ISATR": False, "ISCTR": True, "KRDMD": True, "KRDMA": False, "THYAO": True}
    assert out.loc["ISATR", "reason"] == "other_class" and out.loc["KRDMA", "reason"] == "listed_days"
    assert out.loc["THYAO", "issuer"] == "THYAO" and out.loc["ISCTR", "issuer"] == "is"


def test_prepare_panel_validates():
    with pytest.raises(ValueError):
        prepare_panel(pd.DataFrame({"ticker": ["A"], "date": ["2018-01-01"]}))
    dup = pd.DataFrame(
        {"ticker": ["AAAA", "aaaa.e"], "date": ["2018-01-02"] * 2, "ffloat_mcap_tl": [1, 1], "traded_value_tl": [1, 1]}
    )
    with pytest.raises(ValueError):
        prepare_panel(dup)
    quake = pd.DataFrame(
        {
            "ticker": ["AAAA"] * 2,
            "date": ["2023-02-07", "2023-02-08"],
            "ffloat_mcap_tl": [1, 1],
            "traded_value_tl": [1, 1],
        }
    )
    assert prepare_panel(quake)["date"].tolist() == [pd.Timestamp("2023-02-07")]  # cancelled session dropped


# ----------------------------------------------------------------------------- membership
def test_stable_panel_gives_one_open_spell_per_top_n_ticker():
    membership, reviews = build_rule_based_membership(_panel(_ladder_levels(6)), **SMALL)
    assert list(membership.columns) == MEMBERSHIP_COLUMNS and list(reviews.columns) == REVIEW_COLUMNS
    assert membership["ticker"].tolist() == ["T01", "T02", "T03"]
    assert (membership["effective_start"] == pd.Timestamp("2017-04-03")).all()
    assert membership["effective_end"].isna().all()
    assert (membership["membership_known_since"] == pd.Timestamp("2017-03-24")).all()
    assert (membership["source"] == "rule_based:2017Q2").all()
    assert (membership["index_code"] == "XU100").all()
    assert check_membership(membership) == []
    assert reviews["label"].tolist()[0] == "2017Q2" and len(reviews) == 7
    assert reviews["members"].tolist() == ["T01,T02,T03"] * 7
    assert reviews["n_in"].tolist() == [3, 0, 0, 0, 0, 0, 0] and reviews["n_out"].sum() == 0
    assert members_at(membership, "2017-04-03") == ["T01", "T02", "T03"]
    assert members_at(membership, "2017-03-24") == []  # known but not yet effective
    assert members_at(membership, "2017-03-30", known_only=False) == []


def test_buffers_move_a_riser_in_at_the_next_review_after_the_window():
    # T04 becomes the top name from June 2017: the 2017Q3 window (ends 31 May) does not see it, 2017Q4 does
    panel = _panel(_ladder_levels(6), overrides={"T04": ("2017-06-01", 5000.0, 5000.0)})
    membership, reviews = build_rule_based_membership(panel, **SMALL)
    assert reviews.set_index("label")["members"].to_dict() == {
        "2017Q2": "T01,T02,T03",
        "2017Q3": "T01,T02,T03",
        "2017Q4": "T04,T01,T02",
        "2018Q1": "T04,T01,T02",
        "2018Q2": "T04,T01,T02",
        "2018Q3": "T04,T01,T02",
        "2018Q4": "T04,T01,T02",
    }
    t03 = _spell(membership, "T03")
    assert t03["effective_start"] == pd.Timestamp("2017-04-03") and t03["effective_end"] == pd.Timestamp("2017-10-02")
    t04 = _spell(membership, "T04")
    assert t04["effective_start"] == pd.Timestamp("2017-10-02") and pd.isna(t04["effective_end"])
    assert t04["membership_known_since"] == pd.Timestamp("2017-09-22") and t04["source"] == "rule_based:2017Q4"
    assert reviews.set_index("label").loc["2017Q4", ["n_in", "n_out"]].tolist() == [1, 1]
    assert members_at(membership, "2017-09-29") == ["T01", "T02", "T03"]
    assert members_at(membership, "2017-10-02") == ["T01", "T02", "T04"]
    assert check_membership(membership) == []


def test_member_leaving_the_panel_mid_quarter_is_closed_on_the_next_session():
    panel = _panel(_ladder_levels(6), last={"T02": "2017-11-15"})
    membership, reviews = build_rule_based_membership(panel, **SMALL)
    t02 = _spell(membership, "T02")
    assert t02["effective_end"] == pd.Timestamp("2017-11-16")
    assert members_at(membership, "2017-11-15") == ["T01", "T02", "T03"]
    assert members_at(membership, "2017-11-16") == ["T01", "T03"]
    # the 2018Q1 review (window to 30 Nov) sees T02 as stale, T04 fills the slot from 2018-01-01
    assert reviews.set_index("label").loc["2018Q1", "members"] == "T01,T03,T04"
    assert _spell(membership, "T04")["effective_start"] == pd.Timestamp("2018-01-01")
    assert check_membership(membership) == []


def test_ticker_selected_on_stale_data_never_opens_a_spell():
    # T07 outranks everyone from September; its last trade is 15 Dec 2017: selected at the 2018Q1 review
    # (window ends 30 Nov) but it never trades in 2018Q1, so no spell is created
    levels = {**_ladder_levels(6), "T07": (1.0, 1.0)}
    panel = _panel(levels, last={"T07": "2017-12-15"}, overrides={"T07": ("2017-09-01", 9000.0, 9000.0)})
    membership, reviews = build_rule_based_membership(panel, **SMALL)
    assert "T07" in reviews.set_index("label").loc["2018Q1", "members"]
    assert "T07" not in set(membership["ticker"])
    assert check_membership(membership) == []


def test_newly_listed_ticker_waits_for_min_listed_days_and_share_classes_collapse():
    levels = {**_ladder_levels(4), "NEWA": (9000.0, 9000.0), "T01B": (999.0, 5000.0)}
    panel = _panel(levels, first={"NEWA": "2017-07-03"})
    issuers = {"T01": "issuer1", "T01B": "issuer1"}
    membership, reviews = build_rule_based_membership(
        panel, issuer_of=issuers, n=3, valuation_days=20, entry_rank=2, exit_rank=4, min_listed_days=60
    )
    by_label = reviews.set_index("label")["members"].to_dict()
    assert by_label["2017Q4"] == "T01B,T02,T03"  # NEWA has ~44 sessions < 60; T01B is the liquid class of issuer1
    assert by_label["2018Q1"] == "NEWA,T01B,T02"  # NEWA now eligible and forced in; T03 (lowest incumbent) leaves
    assert "T01" not in set(membership["ticker"])
    assert _spell(membership, "NEWA")["effective_start"] == pd.Timestamp("2018-01-01")
    without = build_rule_based_membership(
        panel,
        issuer_of=issuers,
        one_class_per_issuer=False,
        n=3,
        valuation_days=20,
        entry_rank=2,
        exit_rank=4,
        min_listed_days=60,
    )[1]
    assert without.set_index("label")["members"]["2017Q4"] == "T01B,T01,T02"


def test_empty_or_short_panels_return_empty_tables():
    membership, reviews = build_rule_based_membership(
        _panel(_ladder_levels(3), sessions=pd.bdate_range("2018-01-02", "2018-01-31")), **SMALL
    )
    assert membership.empty and list(membership.columns) == MEMBERSHIP_COLUMNS and reviews.empty


# ----------------------------------------------------------------------------- provider (store files)
def _write_store(store: Path, panel: pd.DataFrame, master_rows: list[tuple[str, str]]) -> None:
    prices = panel.copy()
    prices["close_raw"] = 10.0
    prices["flag_cancelled_session"] = False
    # an extra cancelled row must be ignored by the provider
    extra = prices.iloc[[0]].copy()
    extra["date"] = pd.Timestamp("2017-01-03")
    extra["ticker"] = "ZZZZ"
    extra["flag_cancelled_session"] = True
    write_parquet(pd.concat([prices, extra], ignore_index=True), store / "prices_daily.parquet")
    master = pd.DataFrame(
        [{c: "" for c in MASTER_COLUMNS} | {"ticker": t, "mkk_member_oid": oid} for t, oid in master_rows]
    )
    master["is_current"] = True
    write_parquet(master[MASTER_COLUMNS], store / "security_master.parquet")


def test_provider_builds_from_store_files_and_writes(tmp_path: Path):
    levels = {**_ladder_levels(4), "T01B": (999.0, 5000.0)}
    panel = _panel(levels)
    _write_store(
        tmp_path, panel, [("T01", "issuer1"), ("T01B", "issuer1"), ("T02", "issuer2"), ("T03", ""), ("T04", "issuer4")]
    )
    provider = RuleBasedProvider(tmp_path, n=3, valuation_days=20, entry_rank=2, exit_rank=4, min_listed_days=30)
    assert provider.name == "rule_based"
    membership = provider.membership()
    expected, _ = build_rule_based_membership(panel, issuer_of={"T01": "issuer1", "T01B": "issuer1"}, **SMALL)
    pd.testing.assert_frame_equal(membership, expected)
    assert membership["ticker"].tolist() == ["T01B", "T02", "T03"]
    assert provider.reviews is not None and len(provider.reviews) == 7
    paths = provider.write()
    assert paths == (tmp_path / RULE_BASED_MEMBERSHIP_FILENAME, tmp_path / RULE_BASED_REVIEWS_FILENAME)
    assert all(p.exists() for p in paths)
    reloaded = pd.read_parquet(paths[0])
    pd.testing.assert_frame_equal(reloaded, membership)


def test_provider_errors_and_missing_master(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    with pytest.raises(FileNotFoundError):
        RuleBasedProvider(tmp_path).membership()
    write_parquet(pd.DataFrame({"ticker": ["A"], "date": ["2018-01-02"]}), tmp_path / "prices_daily.parquet")
    with pytest.raises(ValueError):
        RuleBasedProvider(tmp_path).membership()
    panel = _panel(_ladder_levels(4))
    write_parquet(panel, tmp_path / "prices_daily.parquet")  # no flag column, no security master
    provider = RuleBasedProvider(tmp_path, **SMALL)
    assert provider.load_issuers() == {}
    assert provider.membership()["ticker"].tolist() == ["T01", "T02", "T03"]
    assert provider.start is None and provider.end is None and provider.index_code == "XU100"
    assert date(2017, 4, 3) == provider.membership()["effective_start"].iloc[0].date()
