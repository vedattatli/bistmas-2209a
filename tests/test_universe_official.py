"""Offline tests for the official-universe roll-forward and the membership helpers."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from bistmas.data.universe.official import (
    MEMBERSHIP_FILENAME,
    REPORT_FILENAME,
    OfficialUniverseProvider,
    build_official_universe,
    load_aliases,
    load_overrides,
    quarter_effective_date,
    read_membership,
    read_report,
    resolve_alias,
    roll_forward,
    validate_against_anchor,
    write_membership,
    write_report,
)
from bistmas.data.universe.provider import (
    MEMBERSHIP_COLUMNS,
    UniverseProvider,
    check_membership,
    coerce_membership,
    members_at,
    membership_matrix,
)
from bistmas.shared.calendar import TradingCalendar

FIXTURES = Path(__file__).parent / "fixtures" / "universe"


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURES / name, dtype=str)


def _changes(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["year", "quarter", "announcement_date", "index_code", "action", "ticker", "company", "source_url"],
    )


def _overrides(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["ticker", "event", "effective_date", "source", "note"])


def _aliases(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["old_ticker", "new_ticker", "effective_date", "kind", "note"])


def _spell(df: pd.DataFrame, ticker: str, index_code: str = "XU100") -> pd.DataFrame:
    return df[(df["ticker"] == ticker) & (df["index_code"] == index_code)].reset_index(drop=True)


# --------------------------------------------------------------------------- quarter dates
def test_quarter_effective_date_without_calendar_uses_first_weekday():
    assert quarter_effective_date(2018, 1) == date(2018, 1, 1)  # Monday (holiday ignored without a calendar)
    assert quarter_effective_date(2022, 1) == date(2022, 1, 3)  # 1 Jan 2022 is a Saturday
    assert quarter_effective_date(2023, 3) == date(2023, 7, 3)  # 1 Jul 2023 is a Saturday
    assert quarter_effective_date(2018, 4) == date(2018, 10, 1)
    with pytest.raises(ValueError):
        quarter_effective_date(2018, 5)


def test_quarter_effective_date_with_calendar_skips_holidays():
    sessions = [d.date() for d in pd.bdate_range("2017-12-20", "2018-01-31")]
    sessions.remove(date(2018, 1, 1))  # New Year's Day
    cal = TradingCalendar(sessions)
    assert quarter_effective_date(2018, 1, cal) == date(2018, 1, 2)


# --------------------------------------------------------------------------- roll-forward
def test_two_quarter_roll_forward():
    changes = _changes(
        [
            (2018, 2, date(2018, 3, 20), "XU100", "include", "EEEE", "E", "u2"),
            (2018, 2, date(2018, 3, 20), "XU100", "exclude", "AAAA", "A", "u2"),
            (2018, 2, date(2018, 3, 20), "XU100", "substitute", "SSSS", "S", "u2"),
            (2018, 3, date(2018, 6, 20), "XU100", "include", "FFFF", "F", "u3"),
            (2018, 3, date(2018, 6, 20), "XU100", "exclude", "BBBB", "B", "u3"),
            (2018, 3, date(2018, 6, 20), "XU030", "exclude", "AAAA", "A", "u3"),  # other index: ignored
        ]
    )
    df, diag = roll_forward({"AAAA", "BBBB", "CCCC", "DDDD"}, date(2018, 1, 2), date(2017, 12, 29), changes, None, None)

    assert list(df.columns) == MEMBERSHIP_COLUMNS
    assert check_membership(df) == []
    assert diag["anchor_size"] == 4 and diag["changes_applied"] == 4 and diag["final_size"] == 4
    assert diag["final_members"] == ["CCCC", "DDDD", "EEEE", "FFFF"]
    assert [q["size_after"] for q in diag["quarters"]] == [4, 4]
    assert diag["substitutes"][0]["ticker"] == "SSSS" and diag["substitutes"][0]["effective_date"] == date(2018, 4, 2)

    a = _spell(df, "AAAA")
    assert a.loc[0, "effective_start"] == pd.Timestamp("2018-01-02")
    assert a.loc[0, "effective_end"] == pd.Timestamp("2018-04-02")
    assert a.loc[0, "membership_known_since"] == pd.Timestamp("2017-12-29")
    assert a.loc[0, "source"] == "anchor:anchor"
    e = _spell(df, "EEEE")
    assert e.loc[0, "effective_start"] == pd.Timestamp("2018-04-02") and pd.isna(e.loc[0, "effective_end"])
    assert e.loc[0, "membership_known_since"] == pd.Timestamp("2018-03-20")
    assert e.loc[0, "source"] == "announcement:2018Q2"
    f = _spell(df, "FFFF")
    assert f.loc[0, "effective_start"] == pd.Timestamp("2018-07-02")  # 1 Jul 2018 is a Sunday

    assert members_at(df, date(2018, 1, 2)) == ["AAAA", "BBBB", "CCCC", "DDDD"]
    assert members_at(df, date(2018, 3, 30)) == ["AAAA", "BBBB", "CCCC", "DDDD"]
    assert members_at(df, date(2018, 4, 2)) == ["BBBB", "CCCC", "DDDD", "EEEE"]
    assert members_at(df, date(2018, 7, 2)) == ["CCCC", "DDDD", "EEEE", "FFFF"]
    assert members_at(df, date(2018, 7, 2), index_code="XU030") == []


def test_changes_before_anchor_and_after_end_are_skipped():
    changes = _changes(
        [
            (2017, 4, date(2017, 9, 20), "XU100", "include", "ZZZZ", "Z", "u"),  # effective 2017-10-02 <= anchor
            (2018, 1, date(2017, 12, 18), "XU100", "include", "EEEE", "E", "u"),
            (2018, 3, date(2018, 6, 20), "XU100", "include", "FFFF", "F", "u"),  # after end
        ]
    )
    df, diag = roll_forward({"AAAA"}, date(2017, 10, 2), date(2017, 9, 29), changes, None, None, end=date(2018, 6, 29))
    assert diag["changes_skipped_before_anchor"] == 1 and diag["changes_skipped_after_end"] == 1
    assert diag["final_members"] == ["AAAA", "EEEE"]
    assert pd.isna(_spell(df, "EEEE").loc[0, "effective_end"])  # open spells stay open at `end`


def test_exclude_missing_and_include_duplicate_diagnostics():
    changes = _changes(
        [
            (2018, 2, date(2018, 3, 20), "XU100", "exclude", "HURGZ", "Hurriyet", "u"),
            (2018, 2, date(2018, 3, 20), "XU100", "include", "AAAA", "A", "u"),
            (2018, 2, date(2018, 3, 20), "XU100", "bogus", "QQQQ", "Q", "u"),
        ]
    )
    df, diag = roll_forward({"AAAA", "BBBB"}, date(2018, 1, 2), date(2017, 12, 29), changes, None, None)
    assert diag["final_members"] == ["AAAA", "BBBB"]
    assert diag["changes_applied"] == 0
    assert [(d["ticker"], d["year"], d["quarter"]) for d in diag["exclude_missing"]] == [("HURGZ", 2018, 2)]
    assert [d["ticker"] for d in diag["include_duplicate"]] == ["AAAA"]
    assert diag["unknown_actions"][0]["action"] == "bogus"
    assert len(df) == 2 and df["effective_end"].isna().all()


def test_override_entry_exit_rename_and_unresolved():
    changes = _changes([(2019, 1, date(2018, 12, 20), "XU100", "exclude", "TURSG", "Turkiye Sigorta", "u")])
    overrides = _overrides(
        [
            ("KIPA", "exit", "2018-09-03", "hand", "merged"),
            ("ENJSA", "entry", "2018-02-08", "hand", "IPO"),
            ("GUSGR", "rename", "2018-06-01", "hand", "renamed"),
            ("CRFSA", "exit", "", "hand", "date TBD"),
            ("YAZIC", "rename", "", "hand", "date TBD"),
            ("NOPE", "exit", "2018-05-05", "hand", "never a member"),
            ("AAAA", "entry", "2018-05-05", "hand", "already a member"),
            ("XXXX", "rename", "2018-05-05", "hand", "no alias"),
        ]
    )
    aliases = _aliases(
        [
            ("GUSGR", "TURSG", "2018-06-01", "rename", ""),
            ("YAZIC", "AGHOL", "2020-01-01", "rename", ""),
            ("KIPA", "MGROS", "2018-08-31", "merger", ""),
        ]
    )
    df, diag = roll_forward(
        {"AAAA", "KIPA", "GUSGR"}, date(2018, 1, 2), date(2017, 12, 29), changes, overrides, aliases
    )

    assert check_membership(df) == []
    kipa = _spell(df, "KIPA")
    assert kipa.loc[0, "effective_end"] == pd.Timestamp("2018-09-03")
    enjsa = _spell(df, "ENJSA")
    assert enjsa.loc[0, "effective_start"] == pd.Timestamp("2018-02-08")
    assert enjsa.loc[0, "membership_known_since"] == pd.Timestamp("2018-02-08")
    assert enjsa.loc[0, "source"] == "override:entry"
    gusgr = _spell(df, "GUSGR")
    assert gusgr.loc[0, "effective_end"] == pd.Timestamp("2018-06-01")
    tursg = _spell(df, "TURSG")
    assert tursg.loc[0, "effective_start"] == pd.Timestamp("2018-06-01")
    assert tursg.loc[0, "effective_end"] == pd.Timestamp("2019-01-01")  # periodic exclude of the new code works
    assert tursg.loc[0, "source"] == "override:rename"

    assert members_at(df, date(2018, 8, 31)) == ["AAAA", "ENJSA", "KIPA", "TURSG"]
    assert members_at(df, date(2018, 9, 3)) == ["AAAA", "ENJSA", "TURSG"]
    assert members_at(df, date(2019, 1, 2)) == ["AAAA", "ENJSA"]

    unresolved = {(d["ticker"], d["event"]) for d in diag["unresolved_overrides"]}
    assert unresolved == {("CRFSA", "exit"), ("YAZIC", "rename"), ("XXXX", "rename")}
    yazic = next(d for d in diag["unresolved_overrides"] if d["ticker"] == "YAZIC")
    assert yazic["hint_new_ticker"] == "AGHOL" and yazic["hint_alias_effective_date"] == date(2020, 1, 1)
    ignored = {(d["ticker"], d["reason"]) for d in diag["overrides_ignored"]}
    assert ignored == {("NOPE", "not a member"), ("AAAA", "already a member")}
    assert [d["ticker"] for d in diag["overrides_applied"]] == ["ENJSA", "GUSGR", "KIPA"]
    assert diag["exclude_missing"] == []


def test_override_index_code_scopes_entries():
    overrides = pd.DataFrame(
        [("ENJSA", "entry", "2018-02-08", "hand", "IPO", "XU100"), ("KIPA", "exit", "2018-09-03", "hand", "", "")],
        columns=["ticker", "event", "effective_date", "source", "note", "index_code"],
    )
    _, diag30 = roll_forward({"AAAA", "KIPA"}, date(2018, 1, 2), date(2017, 12, 29), None, overrides, None, "XU030")
    assert diag30["final_members"] == ["AAAA"]  # KIPA exit applied everywhere, ENJSA entry only to XU100
    _, diag100 = roll_forward({"AAAA", "KIPA"}, date(2018, 1, 2), date(2017, 12, 29), None, overrides, None, "XU100")
    assert diag100["final_members"] == ["AAAA", "ENJSA"]


def test_resolve_alias_prefers_rename():
    aliases = _aliases([("ADANA", "OYAKC", "2020-01-01", "merger", ""), ("GUSGR", "TURSG", "2020-01-01", "rename", "")])
    assert resolve_alias("gusgr", aliases) == "TURSG"
    assert resolve_alias("ADANA", aliases) == "OYAKC"
    assert resolve_alias("THYAO", aliases) is None


# --------------------------------------------------------------------------- provider helpers
def _synthetic_membership() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "index_code": ["XU100", "XU100", "XU100", "XU030"],
            "ticker": ["AAAA", "BBBB", "AAAA", "AAAA"],
            "effective_start": [date(2018, 1, 2), date(2018, 4, 2), date(2019, 1, 2), date(2018, 1, 2)],
            "effective_end": [date(2018, 7, 2), None, None, None],
            "membership_known_since": [date(2018, 1, 31), date(2018, 3, 20), date(2018, 12, 20), date(2018, 1, 2)],
            "source": ["anchor:x", "announcement:2018Q2", "announcement:2019Q1", "anchor:x"],
        }
    )


def test_members_at_known_only():
    df = _synthetic_membership()
    # AAAA is a member from 2018-01-02 but the anchor was only captured on 2018-01-31.
    assert members_at(df, date(2018, 1, 15), known_only=True) == []
    assert members_at(df, date(2018, 1, 15), known_only=False) == ["AAAA"]
    assert members_at(df, date(2018, 1, 31)) == ["AAAA"]
    assert members_at(df, date(2018, 4, 2)) == ["AAAA", "BBBB"]
    assert members_at(df, date(2018, 7, 2)) == ["BBBB"]  # end is exclusive
    assert members_at(df, date(2019, 1, 2)) == ["AAAA", "BBBB"]  # second spell of AAAA
    assert members_at(df, "2018-05-01", index_code="xu030") == ["AAAA"]
    assert members_at(df, pd.Timestamp("2018-05-01"), index_code="XU050") == []


def test_membership_matrix():
    df = _synthetic_membership()
    dates = [date(2018, 1, 15), date(2018, 4, 2), date(2018, 7, 2), date(2019, 1, 2)]
    m = membership_matrix(df, dates)
    assert list(m.columns) == ["AAAA", "BBBB"]
    assert m.index.equals(pd.DatetimeIndex(pd.to_datetime(dates), name="date"))
    assert m["AAAA"].tolist() == [False, True, False, True]
    assert m["BBBB"].tolist() == [False, True, True, True]
    m_all = membership_matrix(df, dates, known_only=False)
    assert m_all["AAAA"].tolist() == [True, True, False, True]
    empty = membership_matrix(df, dates, index_code="XU050")
    assert empty.shape == (4, 0)


def test_coerce_and_check_membership():
    df = coerce_membership(_synthetic_membership())
    assert str(df["effective_start"].dtype) == "datetime64[ns]"
    assert check_membership(df) == []
    with pytest.raises(ValueError):
        coerce_membership(df.drop(columns=["source"]))
    bad = _synthetic_membership()
    bad.loc[2, "effective_start"] = date(2018, 5, 1)  # overlaps AAAA's first spell
    bad.loc[3, "effective_end"] = date(2017, 1, 1)  # ends before it starts
    problems = check_membership(bad)
    assert any("overlapping spells for XU100/AAAA" in p for p in problems)
    assert any("effective_end <= effective_start" in p for p in problems)
    assert coerce_membership(None).empty and list(coerce_membership(None).columns) == MEMBERSHIP_COLUMNS


# --------------------------------------------------------------------------- validation
def test_validate_against_anchor():
    df, _ = roll_forward({"AAAA", "BBBB", "KIPA"}, date(2018, 1, 2), date(2017, 12, 29), None, None, None)
    anchor = pd.DataFrame(
        {
            "snapshot_date": [date(2018, 12, 31)] * 3,
            "composition_effective_date": [date(2019, 1, 2)] * 3,
            "index_code": ["XU100"] * 3,
            "ticker": ["AAAA.E", "BBBB", "ENJSA"],
            "weight": [0.5, 0.3, 0.2],
            "source_name": ["wayback"] * 3,
        }
    )
    result = validate_against_anchor(df, anchor, date(2019, 1, 2), "XU100")
    assert result["as_of"] == date(2019, 1, 2)
    assert result["missing"] == ["ENJSA"] and result["extra"] == ["KIPA"]
    assert result["n_anchor"] == 3 and result["n_rolled"] == 3 and result["match"] is False
    # falls back to snapshot_date when the effective date does not match
    assert validate_against_anchor(df, anchor, date(2018, 12, 31), "XU100")["missing"] == ["ENJSA"]
    with pytest.raises(ValueError):
        validate_against_anchor(df, anchor, date(2020, 1, 2), "XU100")


# --------------------------------------------------------------------------- end-to-end with fixtures
def test_build_official_universe_from_fixtures(tmp_path):
    changes = _csv("changes_synthetic.csv")
    anchors = _csv("anchors_synthetic.csv")
    overrides = _csv("overrides_synthetic.csv")
    aliases = _csv("aliases_synthetic.csv")
    membership, report = build_official_universe(
        changes, anchors, overrides, aliases, index_codes=("XU100", "XU030", "XU050")
    )

    assert check_membership(membership) == []
    assert report["index_codes"] == ["XU100", "XU030", "XU050"]
    x100 = report["indices"]["XU100"]
    assert x100["anchor"] == {
        "composition_effective_date": date(2017, 7, 3),
        "snapshot_date": date(2017, 6, 30),
        "source_name": "wayback_20170630",
        "n_members": 5,
    }
    assert [v["as_of"] for v in x100["validations"]] == [date(2019, 1, 2), date(2021, 7, 1)]
    assert x100["all_anchors_match"], x100["validations"]
    assert report["indices"]["XU030"]["all_anchors_match"], report["indices"]["XU030"]["validations"]
    assert report["indices"]["XU050"]["error"] == "no anchor rows"

    diag = x100["diagnostics"]
    assert diag["changes_skipped_before_anchor"] == 1  # 2017Q3 KIPA include is already in the anchor
    assert [d["ticker"] for d in diag["exclude_missing"]] == ["HURGZ"]
    assert [d["ticker"] for d in diag["include_duplicate"]] == ["AAAA"]
    assert [d["ticker"] for d in diag["substitutes"]] == ["SSSS"]
    assert [d["ticker"] for d in diag["unresolved_overrides"]] == ["CRFSA"]
    assert diag["final_members"] == ["AAAA", "BBBB", "EEEE", "ENJSA", "TURSG"]
    assert report["indices"]["XU030"]["diagnostics"]["final_members"] == ["AAAA", "EEEE"]

    assert members_at(membership, date(2017, 7, 3)) == ["AAAA", "BBBB", "CCCC", "GUSGR", "KIPA"]  # '.E' stripped
    assert members_at(membership, date(2018, 6, 29), "XU030") == ["AAAA", "EEEE"]
    assert members_at(membership, date(2020, 10, 19)) == ["AAAA", "BBBB", "EEEE", "ENJSA", "TURSG"]

    # persistence round-trip (Parquet + JSON) into a temporary store directory
    assert write_membership(membership, tmp_path).name == MEMBERSHIP_FILENAME
    assert write_report(report, tmp_path).name == REPORT_FILENAME
    back = read_membership(tmp_path, canonical=False)
    pd.testing.assert_frame_equal(back, membership)
    # the default (canonical) read re-keys the GUSGR spell under its successor TURSG as one continuous spell
    canon = read_membership(tmp_path)
    assert members_at(canon, date(2017, 7, 3)) == ["AAAA", "BBBB", "CCCC", "KIPA", "TURSG"]
    assert members_at(canon, date(2020, 10, 19)) == ["AAAA", "BBBB", "EEEE", "ENJSA", "TURSG"]
    assert len(canon[canon["ticker"] == "TURSG"]) == 1
    stored = read_report(tmp_path)
    assert stored["indices"]["XU100"]["anchor"]["composition_effective_date"] == "2017-07-03"
    assert stored["indices"]["XU100"]["validations"][0]["match"] is True

    provider = OfficialUniverseProvider(tmp_path)
    assert isinstance(provider, UniverseProvider) and provider.name == "official"
    pd.testing.assert_frame_equal(provider.membership(), canon)  # consumers get the canonical (successor-keyed) view
    with pytest.raises(FileNotFoundError):
        OfficialUniverseProvider(tmp_path / "missing").membership()


def test_repo_config_tables_load():
    overrides = load_overrides()
    aliases = load_aliases()
    assert set(overrides["event"]) <= {"entry", "exit", "rename"}
    assert overrides.loc[overrides["ticker"] == "KIPA", "effective_date"].iloc[0] == pd.Timestamp("2018-09-03")
    assert overrides["effective_date"].isna().sum() == 0  # every curated event is dated (sources in the CSV)
    assert set(overrides.loc[overrides["event"] == "exit", "counterpart"]) >= {"-", "SARKY", "ECZYT"}
    assert resolve_alias("GUSGR", aliases) == "TURSG"
    assert str(aliases["effective_date"].dtype) == "datetime64[ns]"


def test_anchor_snapshot_after_effective_date_is_not_clamped():
    # A list captured after its effective date (e.g. a later Wayback capture) is only "known" from the capture.
    anchors = pd.DataFrame(
        {
            "snapshot_date": [date(2017, 7, 31)] * 2,
            "composition_effective_date": [date(2017, 7, 3)] * 2,
            "index_code": ["XU100"] * 2,
            "ticker": ["AAAA", "BBBB"],
            "weight": [0.5, 0.5],
            "source_name": ["wayback_20170731"] * 2,
        }
    )
    membership, report = build_official_universe(None, anchors, None, None, index_codes=("XU100",))
    assert report["indices"]["XU100"]["anchor"]["snapshot_date"] == date(2017, 7, 31)
    assert (membership["membership_known_since"] == pd.Timestamp("2017-07-31")).all()
    assert members_at(membership, date(2017, 7, 10)) == []
    assert members_at(membership, date(2017, 7, 10), known_only=False) == ["AAAA", "BBBB"]
    assert members_at(membership, date(2017, 7, 31)) == ["AAAA", "BBBB"]


def test_exit_override_promotes_substitute_within_its_period() -> None:
    """An exit inside a review period promotes the first unused substitute of THAT period only."""
    from bistmas.data.universe.official import roll_forward
    from bistmas.data.universe.provider import members_at

    changes = pd.DataFrame(
        [
            (2018, 3, date(2018, 6, 19), "XU100", "include", "NEWA", "New A", "u"),
            (2018, 3, date(2018, 6, 19), "XU100", "exclude", "OLDA", "Old A", "u"),
            (2018, 3, date(2018, 6, 19), "XU100", "substitute", "SUB1", "Sub 1", "u"),
            (2018, 3, date(2018, 6, 19), "XU100", "substitute", "SUB2", "Sub 2", "u"),
        ],
        columns=["year", "quarter", "announcement_date", "index_code", "action", "ticker", "company", "source_url"],
    )
    overrides = pd.DataFrame(
        [
            ("KIPA", "exit", "2018-09-03", "", "hand", "merger"),
            ("LATE", "exit", "2018-10-15", "", "hand", "after the period: no 2018Q4 list -> no promotion"),
        ],
        columns=["ticker", "event", "effective_date", "index_code", "source", "note"],
    )
    membership, diag = roll_forward(
        {"OLDA", "KIPA", "LATE", "X001"}, date(2017, 7, 3), date(2017, 6, 30), changes, overrides, None, "XU100"
    )
    promoted = diag["substitutes_promoted"]
    assert [(p["ticker"], p["replaces"], p["list"]) for p in promoted] == [("SUB1", "KIPA", "2018Q3")]
    assert members_at(membership, date(2018, 9, 3), "XU100") == ["NEWA", "SUB1", "LATE", "X001"] or set(
        members_at(membership, date(2018, 9, 3), "XU100")
    ) == {"NEWA", "SUB1", "LATE", "X001"}
    assert "LATE" not in members_at(membership, date(2018, 10, 15), "XU100")
    assert "SUB2" not in members_at(membership, date(2018, 10, 15), "XU100")
    row = membership[membership["ticker"] == "SUB1"].iloc[0]
    assert row["source"] == "substitute:2018Q3" and row["effective_start"] == pd.Timestamp("2018-09-03")


def test_canonicalise_renames_merges_touching_spells_and_ignores_mergers():
    import pandas as pd

    from bistmas.data.universe.official import canonicalise_renames

    mem = pd.DataFrame(
        {
            "index_code": ["XU100"] * 5,
            "ticker": ["GUSGR", "TURSG", "ADANA", "OYAKC", "AAAA"],
            "effective_start": ["2017-07-03", "2020-09-01", "2017-07-03", "2019-01-02", "2017-07-03"],
            "effective_end": ["2020-09-01", None, "2020-05-21", None, "2018-01-02"],
            "membership_known_since": ["2017-06-20", "2020-08-31", "2017-06-20", "2018-12-20", "2017-06-20"],
            "source": ["anchor", "override", "anchor", "periodic", "anchor"],
        }
    )
    aliases = pd.DataFrame(
        {
            "old_ticker": ["GUSGR", "ADANA", "AAAA"],
            "new_ticker": ["TURSG", "OYAKC", "BBBB"],
            "effective_date": ["2020-09-01", "2020-05-21", "2018-01-02"],
            "kind": ["rename", "merger", "rename"],
            "note": ["", "", ""],
        }
    )
    out = canonicalise_renames(mem, aliases)
    t = out[out["ticker"] == "TURSG"]
    assert (
        len(t) == 1
        and t["effective_start"].iloc[0] == pd.Timestamp("2017-07-03")
        and pd.isna(t["effective_end"].iloc[0])
    )
    assert t["membership_known_since"].iloc[0] == pd.Timestamp("2017-06-20")
    assert "GUSGR" not in set(out["ticker"]) and "ADANA" in set(out["ticker"])  # merger untouched
    assert set(out["ticker"]) == {"TURSG", "ADANA", "OYAKC", "BBBB"}
    assert list(out.columns) == list(mem.columns)
