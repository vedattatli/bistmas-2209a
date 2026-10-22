"""Official BIST index universe: roll an anchor composition forward through announced changes.

Borsa İstanbul reviews its equity indices four times a year. Each review is announced at least
ten calendar days before the period and takes effect on the first session of January, April, July
and October. Membership is therefore reconstructed as follows:

1. Start from an official full-list *anchor* (constituents-and-weights file, e.g. the 2017-06-30
   CSV whose composition is effective 2017-07-03).
2. Apply every periodic announcement (``include`` / ``exclude``) at the quarter's effective date
   (:func:`quarter_effective_date`), never at the announcement date. The announcement date becomes
   ``membership_known_since`` so the backtest can refuse to use lists that were not yet public.
3. Apply hand-resolved mid-period events from ``configs/universe_overrides.csv`` (delisting, merger,
   IPO fast-track entry, ticker rename via ``configs/ticker_aliases.csv``) on their effective date.
4. Validate the rolled list against every later anchor (:func:`validate_against_anchor`).

Everything that cannot be reconciled automatically is *reported*, never silently fixed: an
``exclude`` of a non-member (a substitute or IPO that entered mid-period) lands in
``diagnostics['exclude_missing']``, an ``include`` of a current member in ``include_duplicate``,
substitutes in ``substitutes`` (they are not applied: only the first substitute enters, and only
when a member leaves mid-period), and overrides without a date in ``unresolved_overrides``.

Input contracts (produced by the announcement/anchor clients; dates may be ``datetime.date``,
``datetime64`` or ISO strings, tickers may carry a ``.E`` suffix):

* changes: ``[year, quarter, announcement_date, index_code, action ('include'|'exclude'|'substitute'),
  ticker, company, source_url]``
* anchors: ``[snapshot_date, composition_effective_date, index_code, ticker, weight, source_name]``
* overrides: ``[ticker, event ('entry'|'exit'|'rename'), effective_date, source, note]`` plus an optional
  ``index_code`` column (blank = applies to every index; use it for index-specific IPO entries)
* aliases: ``[old_ticker, new_ticker, effective_date, kind, note]``

Outputs: the membership table (``provider.MEMBERSHIP_COLUMNS``) and a JSON-serialisable report.
Dates inside diagnostics/report are ``datetime.date`` objects (``store.write_json`` renders them as
ISO strings).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from bistmas.data.universe.provider import (
    MEMBERSHIP_COLUMNS,
    check_membership,
    coerce_membership,
    members_at,
    normalise_ticker,
    normalise_ticker_series,
    to_datetime_column,
    to_timestamp,
)
from bistmas.shared.calendar import TradingCalendar
from bistmas.shared.config import CONFIG_DIR
from bistmas.shared.log import get_logger
from bistmas.shared.store import read_json, read_parquet, utc_now_iso, write_json, write_parquet

log = get_logger("data.universe.official")

MEMBERSHIP_FILENAME = "universe_membership.parquet"
REPORT_FILENAME = "universe_report.json"

CHANGE_COLUMNS: list[str] = ["year", "quarter", "announcement_date", "index_code", "action", "ticker"]
ANCHOR_COLUMNS: list[str] = ["snapshot_date", "composition_effective_date", "index_code", "ticker"]
OVERRIDE_COLUMNS: list[str] = ["ticker", "event", "effective_date"]
ALIAS_COLUMNS: list[str] = ["old_ticker", "new_ticker"]

CHANGE_ACTIONS = ("include", "exclude", "substitute")
OVERRIDE_EVENTS = ("entry", "exit", "rename")
QUARTER_START_MONTH = {1: 1, 2: 4, 3: 7, 4: 10}

# Same-day ordering: hand-resolved overrides first (explicit knowledge), then periodic excludes,
# then periodic includes, so that "exclude X / include X" on one day never yields a duplicate.
_PRIORITY = {"exit": 0, "rename": 1, "entry": 2, "exclude": 3, "include": 4}


# --------------------------------------------------------------------------- dates
def quarter_effective_date(year: int, quarter: int, calendar: TradingCalendar | None = None) -> date:
    """First session on or after 1 Jan / 1 Apr / 1 Jul / 1 Oct of the given quarter.

    With a :class:`TradingCalendar` the exact first session is returned. Without one the first
    weekday is used, which ignores public holidays (1 January is always a BIST holiday, so 2018Q1
    resolves to 2018-01-01 instead of the true 2018-01-02); pass the data-derived calendar (M2)
    for exact dates. The approximation never matters for month-end decisions.
    """
    if quarter not in QUARTER_START_MONTH:
        raise ValueError(f"quarter must be 1..4, got {quarter!r}")
    d = date(int(year), QUARTER_START_MONTH[int(quarter)], 1)
    if calendar is not None:
        return calendar.next_session(d, strict=False)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _d(ts: pd.Timestamp | None) -> date | None:
    return None if ts is None or pd.isna(ts) else pd.Timestamp(ts).date()


# --------------------------------------------------------------------------- input preparation
def _require(df: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} table is missing columns {missing}")


def _prepare_changes(changes: pd.DataFrame | None) -> pd.DataFrame:
    if changes is None or len(changes) == 0:
        return pd.DataFrame(
            {
                "year": pd.Series(dtype=int),
                "quarter": pd.Series(dtype=int),
                "announcement_date": pd.Series(dtype="datetime64[ns]"),
                "index_code": pd.Series(dtype=str),
                "action": pd.Series(dtype=str),
                "ticker": pd.Series(dtype=str),
                "company": pd.Series(dtype=str),
                "source_url": pd.Series(dtype=str),
            }
        )
    _require(changes, CHANGE_COLUMNS, "changes")
    out = changes.copy()
    out["year"] = out["year"].astype(int)
    out["quarter"] = out["quarter"].astype(int)
    out["announcement_date"] = to_datetime_column(out["announcement_date"])
    out["index_code"] = out["index_code"].astype("string").fillna("").str.strip().str.upper().astype(str)
    out["action"] = out["action"].astype("string").fillna("").str.strip().str.lower().astype(str)
    out["ticker"] = normalise_ticker_series(out["ticker"])
    for col in ("company", "source_url"):
        out[col] = out[col].astype("string").fillna("").astype(str) if col in out.columns else ""
    return out


def _prepare_anchors(anchors: pd.DataFrame | None) -> pd.DataFrame:
    if anchors is None or len(anchors) == 0:
        return pd.DataFrame(
            {
                "snapshot_date": pd.Series(dtype="datetime64[ns]"),
                "composition_effective_date": pd.Series(dtype="datetime64[ns]"),
                "index_code": pd.Series(dtype=str),
                "ticker": pd.Series(dtype=str),
                "weight": pd.Series(dtype=float),
                "source_name": pd.Series(dtype=str),
            }
        )
    _require(anchors, ANCHOR_COLUMNS, "anchors")
    out = anchors.copy()
    out["snapshot_date"] = to_datetime_column(out["snapshot_date"])
    out["composition_effective_date"] = to_datetime_column(out["composition_effective_date"])
    out["index_code"] = out["index_code"].astype("string").fillna("").str.strip().str.upper().astype(str)
    out["ticker"] = normalise_ticker_series(out["ticker"])
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce") if "weight" in out.columns else float("nan")
    out["source_name"] = (
        out["source_name"].astype("string").fillna("").astype(str) if "source_name" in out.columns else "anchor"
    )
    return out


def _prepare_overrides(overrides: pd.DataFrame | None) -> pd.DataFrame:
    if overrides is None or len(overrides) == 0:
        return pd.DataFrame(
            {
                "ticker": pd.Series(dtype=str),
                "event": pd.Series(dtype=str),
                "effective_date": pd.Series(dtype="datetime64[ns]"),
                "source": pd.Series(dtype=str),
                "note": pd.Series(dtype=str),
                "index_code": pd.Series(dtype=str),
                "counterpart": pd.Series(dtype=str),
            }
        )
    _require(overrides, OVERRIDE_COLUMNS, "overrides")
    out = overrides.copy()
    out["ticker"] = normalise_ticker_series(out["ticker"])
    out["event"] = out["event"].astype("string").fillna("").str.strip().str.lower().astype(str)
    out["effective_date"] = to_datetime_column(out["effective_date"])
    for col in ("source", "note", "index_code", "counterpart"):
        out[col] = out[col].astype("string").fillna("").str.strip().astype(str) if col in out.columns else ""
    out["index_code"] = out["index_code"].str.upper()
    out["counterpart"] = out["counterpart"].str.upper()
    return out


def _prepare_aliases(aliases: pd.DataFrame | None) -> pd.DataFrame:
    if aliases is None or len(aliases) == 0:
        return pd.DataFrame(
            {
                "old_ticker": pd.Series(dtype=str),
                "new_ticker": pd.Series(dtype=str),
                "effective_date": pd.Series(dtype="datetime64[ns]"),
                "kind": pd.Series(dtype=str),
                "note": pd.Series(dtype=str),
            }
        )
    _require(aliases, ALIAS_COLUMNS, "aliases")
    out = aliases.copy()
    out["old_ticker"] = normalise_ticker_series(out["old_ticker"])
    out["new_ticker"] = normalise_ticker_series(out["new_ticker"])
    out["effective_date"] = to_datetime_column(out["effective_date"]) if "effective_date" in out.columns else pd.NaT
    for col in ("kind", "note"):
        out[col] = (
            out[col].astype("string").fillna("").str.strip().str.lower().astype(str) if col in out.columns else ""
        )
    return out


def load_overrides(path: Path | None = None) -> pd.DataFrame:
    """Read ``configs/universe_overrides.csv`` (blank ``effective_date`` -> ``NaT``)."""
    return _prepare_overrides(pd.read_csv(path or CONFIG_DIR / "universe_overrides.csv", dtype=str))


def load_aliases(path: Path | None = None) -> pd.DataFrame:
    """Read ``configs/ticker_aliases.csv`` (blank ``effective_date`` -> ``NaT``)."""
    return _prepare_aliases(pd.read_csv(path or CONFIG_DIR / "ticker_aliases.csv", dtype=str))


def resolve_alias(ticker: str, aliases: pd.DataFrame) -> str | None:
    """New code for ``ticker`` from the alias table (``kind == 'rename'`` preferred); ``None`` if unknown."""
    al = _prepare_aliases(aliases)
    cand = al[al["old_ticker"] == normalise_ticker(ticker)]
    if cand.empty:
        return None
    renames = cand[cand["kind"] == "rename"]
    pick = renames.iloc[0] if not renames.empty else cand.iloc[0]
    return str(pick["new_ticker"]) or None


# --------------------------------------------------------------------------- roll-forward
@dataclass(order=True)
class _Event:
    when: pd.Timestamp
    priority: int
    ticker: str
    kind: str = field(compare=False)
    known_since: pd.Timestamp = field(compare=False)
    source: str = field(compare=False)
    meta: dict[str, Any] = field(compare=False, default_factory=dict)


def _new_diagnostics(index_code: str, anchor_ts: pd.Timestamp, known_ts: pd.Timestamp, n_anchor: int) -> dict[str, Any]:
    return {
        "index_code": index_code,
        "anchor_effective": _d(anchor_ts),
        "anchor_known_since": _d(known_ts),
        "anchor_size": n_anchor,
        "changes_applied": 0,
        "changes_skipped_before_anchor": 0,
        "changes_skipped_after_end": 0,
        "unknown_actions": [],
        "exclude_missing": [],
        "include_duplicate": [],
        "substitutes": [],
        "substitutes_promoted": [],
        "quarters_without_rows": [],
        "quarters": [],
        "overrides_applied": [],
        "overrides_ignored": [],
        "unresolved_overrides": [],
        "zero_length_dropped": 0,
        "final_size": 0,
        "final_members": [],
    }


def _periodic_events(
    ch: pd.DataFrame,
    code: str,
    anchor_ts: pd.Timestamp,
    end_ts: pd.Timestamp | None,
    calendar: TradingCalendar | None,
    diag: dict[str, Any],
) -> list[_Event]:
    events: list[_Event] = []
    ch = ch[ch["index_code"] == code]
    for (year, quarter), grp in ch.groupby(["year", "quarter"], sort=True):
        year, quarter = int(year), int(quarter)
        eff = to_timestamp(quarter_effective_date(year, quarter, calendar))
        if eff <= anchor_ts:
            diag["changes_skipped_before_anchor"] += int(len(grp))
            continue
        if end_ts is not None and eff > end_ts:
            diag["changes_skipped_after_end"] += int(len(grp))
            continue
        label = f"{year}Q{quarter}"
        announced = grp["announcement_date"].min()
        summary = {
            "year": year,
            "quarter": quarter,
            "effective_date": _d(eff),
            "announcement_date": _d(announced),
            "n_include": 0,
            "n_exclude": 0,
            "n_substitute": 0,
            "size_after": None,
        }
        for row in grp.itertuples(index=False):
            action = row.action
            known = eff if pd.isna(row.announcement_date) else row.announcement_date
            meta = {
                "year": year,
                "quarter": quarter,
                "index_code": code,
                "ticker": row.ticker,
                "company": row.company,
                "effective_date": _d(eff),
                "announcement_date": _d(row.announcement_date),
                "source_url": row.source_url,
            }
            if action == "substitute":
                diag["substitutes"].append(meta)
                summary["n_substitute"] += 1
                continue
            if action not in ("include", "exclude"):
                diag["unknown_actions"].append({**meta, "action": action})
                continue
            summary["n_include" if action == "include" else "n_exclude"] += 1
            events.append(_Event(eff, _PRIORITY[action], row.ticker, action, known, f"announcement:{label}", meta))
        diag["quarters"].append(summary)
    return events


def _override_events(
    ov: pd.DataFrame,
    al: pd.DataFrame,
    code: str,
    anchor_ts: pd.Timestamp,
    end_ts: pd.Timestamp | None,
    diag: dict[str, Any],
) -> list[_Event]:
    events: list[_Event] = []
    for row in ov.itertuples(index=False):
        meta = {
            "ticker": row.ticker,
            "event": row.event,
            "effective_date": _d(row.effective_date),
            "index_code": row.index_code or None,
            "source": row.source,
            "note": row.note,
            "counterpart": str(getattr(row, "counterpart", "") or ""),
        }
        if row.index_code and row.index_code != code:
            continue
        if row.event not in OVERRIDE_EVENTS:
            diag["overrides_ignored"].append({**meta, "reason": f"unknown event {row.event!r}"})
            continue
        if pd.isna(row.effective_date):
            entry = dict(meta)
            if row.event == "rename":
                hit = al[al["old_ticker"] == row.ticker]
                if not hit.empty:
                    entry["hint_new_ticker"] = str(hit.iloc[0]["new_ticker"])
                    entry["hint_alias_effective_date"] = _d(hit.iloc[0]["effective_date"])
            diag["unresolved_overrides"].append(entry)
            continue
        when = row.effective_date
        if when <= anchor_ts:
            diag["overrides_ignored"].append({**meta, "reason": "on or before the anchor date"})
            continue
        if end_ts is not None and when > end_ts:
            diag["overrides_ignored"].append({**meta, "reason": "after end"})
            continue
        if row.event == "rename":
            new_code = resolve_alias(row.ticker, al)
            if not new_code:
                diag["unresolved_overrides"].append({**meta, "reason": "no alias for old ticker"})
                continue
            meta["new_ticker"] = new_code
        events.append(_Event(when, _PRIORITY[row.event], row.ticker, row.event, when, f"override:{row.event}", meta))
    return events


def _quarters_without_rows(
    ch: pd.DataFrame,
    code: str,
    anchor_ts: pd.Timestamp,
    end_ts: pd.Timestamp | None,
    calendar: TradingCalendar | None,
    diag: dict[str, Any],
) -> None:
    """Record quarters that have rows for other core indices but none for ``code`` (missing tables)."""
    if ch.empty:
        return
    all_quarters = {(int(y), int(q)) for y, q in ch[["year", "quarter"]].drop_duplicates().itertuples(index=False)}
    mine = {
        (int(y), int(q))
        for y, q in ch.loc[ch["index_code"] == code, ["year", "quarter"]].drop_duplicates().itertuples(index=False)
    }
    for year, quarter in sorted(all_quarters - mine):
        eff = to_timestamp(quarter_effective_date(year, quarter, calendar))
        if eff <= anchor_ts or (end_ts is not None and eff > end_ts):
            continue
        diag["quarters_without_rows"].append({"year": year, "quarter": quarter, "effective_date": _d(eff)})


def _substitute_lists(
    diag: dict[str, Any], calendar: TradingCalendar | None = None
) -> list[tuple[pd.Timestamp, pd.Timestamp, str, list[str]]]:
    """Ordered substitute lists per periodic review with their validity window ``[eff, next_eff)``."""
    by_quarter: dict[tuple[int, int], tuple[pd.Timestamp, list[str]]] = {}
    for meta in diag["substitutes"]:
        key = (int(meta["year"]), int(meta["quarter"]))
        eff = to_timestamp(meta["effective_date"])
        by_quarter.setdefault(key, (eff, []))[1].append(normalise_ticker(meta["ticker"]))
    out = []
    for (y, q), (eff, tickers) in sorted(by_quarter.items()):
        ny, nq = (y + 1, 1) if q == 4 else (y, q + 1)
        next_eff = to_timestamp(quarter_effective_date(ny, nq, calendar))
        out.append((eff, next_eff, f"{y}Q{q}", tickers))
    return out


def _substitutes_in_force(
    lists: list[tuple[pd.Timestamp, pd.Timestamp, str, list[str]]], when: pd.Timestamp
) -> tuple[str, list[str]]:
    """The substitute list of the periodic review whose period contains ``when`` (empty if none)."""
    for eff, next_eff, label, tickers in lists:
        if eff <= when < next_eff:
            return label, tickers
    return "", []


def roll_forward(
    anchor_members: Iterable[str],
    anchor_effective: date | pd.Timestamp | str,
    anchor_known_since: date | pd.Timestamp | str,
    changes: pd.DataFrame | None,
    overrides: pd.DataFrame | None,
    aliases: pd.DataFrame | None,
    index_code: str = "XU100",
    calendar: TradingCalendar | None = None,
    end: date | pd.Timestamp | str | None = None,
    anchor_source: str = "anchor",
    promote_substitutes: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Roll ``anchor_members`` (effective ``anchor_effective``) forward through changes and overrides.

    When ``promote_substitutes`` is true, an ``exit`` override that closes a member mid-period promotes
    the first unused substitute of the latest periodic list in force (Borsa İstanbul ground rules):
    the promoted spell starts on the exit date with ``source = "substitute:<year>Q<quarter>"`` and is
    listed in ``diagnostics["substitutes_promoted"]``. Quarters that carry rows for other indices in
    ``changes`` but none for ``index_code`` are listed in ``diagnostics["quarters_without_rows"]``.

    Returns ``(membership, diagnostics)``. Membership has ``MEMBERSHIP_COLUMNS``; anchor spells
    carry ``membership_known_since = anchor_known_since`` and ``source = 'anchor:<anchor_source>'``.
    Changes effective on or before the anchor date are already reflected in the anchor and are
    skipped; events after ``end`` (if given) are skipped, open spells stay open (``NaT``).
    """
    code = index_code.strip().upper()
    anchor_ts = to_timestamp(anchor_effective)
    known_ts = to_timestamp(anchor_known_since)
    end_ts = None if end is None else to_timestamp(end)
    members = {normalise_ticker(t) for t in anchor_members}
    members.discard("")
    diag = _new_diagnostics(code, anchor_ts, known_ts, len(members))

    prepared_changes = _prepare_changes(changes)
    events = _periodic_events(prepared_changes, code, anchor_ts, end_ts, calendar, diag)
    _quarters_without_rows(prepared_changes, code, anchor_ts, end_ts, calendar, diag)
    substitute_lists = _substitute_lists(diag, calendar)
    promoted: set[str] = set()
    events += _override_events(_prepare_overrides(overrides), _prepare_aliases(aliases), code, anchor_ts, end_ts, diag)
    events.sort()

    open_spells: dict[str, dict[str, Any]] = {
        t: {"start": anchor_ts, "known": known_ts, "source": f"anchor:{anchor_source}"} for t in sorted(members)
    }
    rows: list[dict[str, Any]] = []
    quarter_sizes: dict[tuple[int, int], int] = {}

    def close(ticker: str, when: pd.Timestamp) -> None:
        spell = open_spells.pop(ticker)
        if spell["start"] >= when:
            diag["zero_length_dropped"] += 1
            return
        rows.append(
            {
                "index_code": code,
                "ticker": ticker,
                "effective_start": spell["start"],
                "effective_end": when,
                "membership_known_since": spell["known"],
                "source": spell["source"],
            }
        )

    def open_(ticker: str, when: pd.Timestamp, known: pd.Timestamp, source: str) -> None:
        open_spells[ticker] = {"start": when, "known": known, "source": source}

    for ev in events:
        if ev.kind == "exclude":
            if ev.ticker in open_spells:
                close(ev.ticker, ev.when)
                diag["changes_applied"] += 1
            else:
                diag["exclude_missing"].append(ev.meta)
        elif ev.kind == "include":
            if ev.ticker in open_spells:
                diag["include_duplicate"].append(ev.meta)
            else:
                open_(ev.ticker, ev.when, ev.known_since, ev.source)
                diag["changes_applied"] += 1
        elif ev.kind == "exit":
            if ev.ticker in open_spells:
                close(ev.ticker, ev.when)
                diag["overrides_applied"].append(ev.meta)
                explicit = ev.meta.get("counterpart") or ""
                if explicit in ("-", "NONE"):
                    # displacement exit: the vacancy is filled by an explicit entry override, never by a substitute
                    pass
                elif explicit:
                    if explicit in open_spells:
                        diag["overrides_ignored"].append(
                            {**ev.meta, "reason": f"counterpart {explicit} already a member"}
                        )
                    else:
                        promoted.add(explicit)
                        open_(explicit, ev.when, ev.when, "substitute:override")
                        diag["substitutes_promoted"].append(
                            {
                                "ticker": explicit,
                                "replaces": ev.ticker,
                                "effective_date": _d(ev.when),
                                "list": "override",
                            }
                        )
                elif promote_substitutes:
                    label, candidates = _substitutes_in_force(substitute_lists, ev.when)
                    for sub in candidates:
                        if sub in open_spells or sub in promoted:
                            continue
                        promoted.add(sub)
                        open_(sub, ev.when, ev.when, f"substitute:{label}")
                        diag["substitutes_promoted"].append(
                            {"ticker": sub, "replaces": ev.ticker, "effective_date": _d(ev.when), "list": label}
                        )
                        break
            else:
                diag["overrides_ignored"].append({**ev.meta, "reason": "not a member"})
        elif ev.kind == "entry":
            if ev.ticker in open_spells:
                diag["overrides_ignored"].append({**ev.meta, "reason": "already a member"})
            else:
                open_(ev.ticker, ev.when, ev.known_since, ev.source)
                diag["overrides_applied"].append(ev.meta)
        elif ev.kind == "rename":
            new_code = ev.meta["new_ticker"]
            if ev.ticker not in open_spells:
                diag["overrides_ignored"].append({**ev.meta, "reason": "not a member"})
                continue
            close(ev.ticker, ev.when)
            if new_code in open_spells:
                diag["overrides_applied"].append({**ev.meta, "note_applied": "new code already a member; old closed"})
            else:
                open_(new_code, ev.when, ev.known_since, ev.source)
                diag["overrides_applied"].append(ev.meta)
        if ev.kind in ("include", "exclude"):
            quarter_sizes[(ev.meta["year"], ev.meta["quarter"])] = len(open_spells)

    for summary in diag["quarters"]:
        summary["size_after"] = quarter_sizes.get((summary["year"], summary["quarter"]), summary["size_after"])

    for ticker in sorted(open_spells):
        spell = open_spells[ticker]
        rows.append(
            {
                "index_code": code,
                "ticker": ticker,
                "effective_start": spell["start"],
                "effective_end": pd.NaT,
                "membership_known_since": spell["known"],
                "source": spell["source"],
            }
        )
    diag["final_size"] = len(open_spells)
    diag["final_members"] = sorted(open_spells)
    membership = coerce_membership(pd.DataFrame(rows, columns=MEMBERSHIP_COLUMNS))
    log.info(
        "%s roll-forward from %s: %d anchor members, %d changes applied, %d exclude-missing, %d include-duplicate, "
        "%d unresolved overrides, final size %d",
        code,
        anchor_ts.date(),
        len(members),
        diag["changes_applied"],
        len(diag["exclude_missing"]),
        len(diag["include_duplicate"]),
        len(diag["unresolved_overrides"]),
        diag["final_size"],
    )
    return membership, diag


# --------------------------------------------------------------------------- validation
def validate_against_anchor(
    membership_df: pd.DataFrame,
    anchor_df: pd.DataFrame,
    snapshot_effective_date: date | pd.Timestamp | str,
    index_code: str = "XU100",
) -> dict[str, Any]:
    """Compare the rolled membership on ``snapshot_effective_date`` with an official anchor list.

    Anchor rows are selected by ``composition_effective_date == snapshot_effective_date``; when none
    match, rows with ``snapshot_date == snapshot_effective_date`` are used and the rolled list is
    evaluated on THEIR ``composition_effective_date`` (a period-end file holds the next period's
    composition, so comparing on the file date would be wrong). The rolled list is taken with
    ``known_only=False`` because the comparison is about composition, not about the information
    set. Returns ``{as_of (evaluation date), requested_date, index_code, n_anchor, n_rolled,
    missing (in anchor, not rolled), extra (rolled, not in anchor), match}``.
    """
    code = index_code.strip().upper()
    as_of = to_timestamp(snapshot_effective_date)
    anchors = _prepare_anchors(anchor_df)
    anchors = anchors[anchors["index_code"] == code]
    sel = anchors[anchors["composition_effective_date"] == as_of]
    eval_ts = as_of
    if sel.empty:
        sel = anchors[anchors["snapshot_date"] == as_of]
        if not sel.empty:
            eval_ts = pd.Timestamp(sel["composition_effective_date"].iloc[0])
    if sel.empty:
        raise ValueError(f"no anchor rows for {code} dated {as_of.date()}")
    anchor_set = {t for t in sel["ticker"] if t}
    rolled = set(members_at(membership_df, eval_ts, code, known_only=False))
    missing = sorted(anchor_set - rolled)
    extra = sorted(rolled - anchor_set)
    return {
        "as_of": eval_ts.date(),
        "requested_date": as_of.date(),
        "index_code": code,
        "n_anchor": len(anchor_set),
        "n_rolled": len(rolled),
        "missing": missing,
        "extra": extra,
        "match": not missing and not extra,
    }


# --------------------------------------------------------------------------- end-to-end build
def build_official_universe(
    changes: pd.DataFrame | None,
    anchors: pd.DataFrame,
    overrides: pd.DataFrame | None,
    aliases: pd.DataFrame | None,
    index_codes: Sequence[str] = ("XU100", "XU030"),
    calendar: TradingCalendar | None = None,
    end: date | pd.Timestamp | str | None = None,
    promote_substitutes: bool = True,
    missing_tables: Sequence[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Roll every index in ``index_codes`` from its earliest anchor and validate against later anchors.

    When ``calendar`` is given, period-end anchors (``composition_effective_date > snapshot_date``)
    are snapped to the first trading session on/after their effective date so that anchors and
    periodic changes use the same session. ``missing_tables`` (from
    ``build_changes_table(...).attrs``) is echoed into the report.

    The anchor's ``snapshot_date`` (file date, at or before the effective date) is used as
    ``membership_known_since`` for the anchor spells. Returns ``(membership, report)``; the report
    is JSON-serialisable and holds, per index, the anchor used, the roll-forward diagnostics and one
    validation entry per later anchor (``report['indices'][code]['validations']``).
    """
    prepared = _prepare_anchors(anchors)
    if calendar is not None and not prepared.empty:
        period_end = prepared["composition_effective_date"] > prepared["snapshot_date"]
        prepared.loc[period_end, "composition_effective_date"] = prepared.loc[
            period_end, "composition_effective_date"
        ].map(lambda ts: to_timestamp(calendar.next_session(pd.Timestamp(ts).date(), strict=False)))
    frames: list[pd.DataFrame] = []
    report: dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "index_codes": [c.strip().upper() for c in index_codes],
        "end": _d(None if end is None else to_timestamp(end)),
        "missing_tables": list(missing_tables or []),
        "indices": {},
    }
    for raw_code in index_codes:
        code = raw_code.strip().upper()
        ai = prepared[(prepared["index_code"] == code) & prepared["composition_effective_date"].notna()]
        if ai.empty:
            log.warning("no anchor rows for %s; index skipped", code)
            report["indices"][code] = {"error": "no anchor rows", "validations": [], "all_anchors_match": False}
            continue
        snapshots = sorted(ai["composition_effective_date"].unique())
        first = pd.Timestamp(snapshots[0])
        first_rows = ai[ai["composition_effective_date"] == first]
        # Conservative: the file date is when the list became public. A snapshot captured after the
        # effective date (e.g. a later Wayback capture) keeps its later date; never clamp it earlier.
        known = first_rows["snapshot_date"].min()
        known = first if pd.isna(known) else pd.Timestamp(known)
        source_name = str(first_rows["source_name"].iloc[0]) or "anchor"
        anchor_members = {t for t in first_rows["ticker"] if t}
        membership, diag = roll_forward(
            anchor_members,
            first,
            known,
            changes,
            overrides,
            aliases,
            index_code=code,
            calendar=calendar,
            end=end,
            anchor_source=source_name,
            promote_substitutes=promote_substitutes,
        )
        end_ts = None if end is None else to_timestamp(end)
        validations = [
            validate_against_anchor(membership, ai, pd.Timestamp(s), code)
            for s in snapshots[1:]
            if end_ts is None or pd.Timestamp(s) <= end_ts
        ]
        if not validations:
            log.warning("%s: no later anchor available for validation (end=%s)", code, _d(end_ts))
        problems = check_membership(membership)
        all_match = bool(validations) and all(v["match"] for v in validations)
        frames.append(membership)
        report["indices"][code] = {
            "anchor": {
                "composition_effective_date": _d(first),
                "snapshot_date": _d(known),
                "source_name": source_name,
                "n_members": len(anchor_members),
            },
            "n_intervals": int(len(membership)),
            "n_tickers": int(membership["ticker"].nunique()),
            "diagnostics": diag,
            "validations": validations,
            "n_validations": len(validations),
            "all_anchors_match": all_match,
            "invariant_problems": problems,
            "quarters_without_rows": diag["quarters_without_rows"],
            "ok": all_match and not problems and not diag["quarters_without_rows"],
        }
    membership_all = coerce_membership(pd.concat(frames, ignore_index=True) if frames else None)
    return membership_all, report


# --------------------------------------------------------------------------- persistence
def write_membership(df: pd.DataFrame, store_dir: Path) -> Path:
    """Write the membership table to ``<store_dir>/universe_membership.parquet`` (atomic)."""
    return write_parquet(coerce_membership(df), Path(store_dir) / MEMBERSHIP_FILENAME)


def read_membership(store_dir: Path, canonical: bool = True) -> pd.DataFrame:
    """Read ``<store_dir>/universe_membership.parquet`` with normalised dtypes.

    With ``canonical=True`` (the default for every consumer) renamed codes are mapped to their successor
    and touching spells are merged (:func:`canonicalise_renames`), so a renamed issuer is one continuous
    member under the code its price and statement history is keyed by. The stored file keeps the
    historical codes, which is what the official-list validation and the reports use.
    """
    df = coerce_membership(read_parquet(Path(store_dir) / MEMBERSHIP_FILENAME))
    if canonical:
        df = canonicalise_renames(df, load_aliases(CONFIG_DIR / "ticker_aliases.csv"))
    return df


def canonicalise_renames(membership: pd.DataFrame, aliases: pd.DataFrame) -> pd.DataFrame:
    """Map ``kind == "rename"`` codes to their (final) successor and merge touching or overlapping spells.

    Merger aliases are left alone: the merged-away code had its own listing, prices and filings. Renames
    (same legal entity, same KAP id) are keyed under the new code by both İş Yatırım and KAP, so the old
    code would otherwise be a member without any data.
    """
    ren = {
        str(r.old_ticker): str(r.new_ticker)
        for r in aliases.itertuples(index=False)
        if str(r.kind) == "rename" and str(r.old_ticker) and str(r.new_ticker)
    }
    if not ren or membership.empty:
        return membership

    def resolve(t: str) -> str:
        seen = set()
        while t in ren and t not in seen:
            seen.add(t)
            t = ren[t]
        return t

    df = coerce_membership(membership.copy())
    df["ticker"] = df["ticker"].map(resolve)
    df = df.sort_values(["index_code", "ticker", "effective_start"], kind="stable")
    rows: list[dict[str, Any]] = []
    for _, g in df.groupby(["index_code", "ticker"], sort=False):
        cur: dict[str, Any] | None = None
        for r in g.to_dict("records"):
            if cur is None:
                cur = dict(r)
                continue
            open_end = pd.isna(cur["effective_end"])
            touching = open_end or pd.Timestamp(r["effective_start"]) <= pd.Timestamp(cur["effective_end"])
            if touching:
                if not (open_end or pd.isna(r["effective_end"])):
                    cur["effective_end"] = max(pd.Timestamp(cur["effective_end"]), pd.Timestamp(r["effective_end"]))
                elif pd.isna(r["effective_end"]):
                    cur["effective_end"] = pd.NaT
                ks = [x for x in (cur.get("membership_known_since"), r.get("membership_known_since")) if pd.notna(x)]
                cur["membership_known_since"] = min(ks) if ks else pd.NaT
            else:
                rows.append(cur)
                cur = dict(r)
        if cur is not None:
            rows.append(cur)
    out = pd.DataFrame(rows, columns=list(df.columns))
    return (
        coerce_membership(out)
        .sort_values(["index_code", "effective_start", "ticker"], kind="stable")
        .reset_index(drop=True)
    )


def write_report(report: dict[str, Any], store_dir: Path) -> Path:
    """Write the build report to ``<store_dir>/universe_report.json`` (dates rendered as ISO strings)."""
    return write_json(report, Path(store_dir) / REPORT_FILENAME)


def read_report(store_dir: Path) -> dict[str, Any]:
    return read_json(Path(store_dir) / REPORT_FILENAME)


class OfficialUniverseProvider:
    """``UniverseProvider`` backed by the membership table built by :func:`build_official_universe`."""

    name = "official"

    def __init__(self, store_dir: Path) -> None:
        self.store_dir = Path(store_dir)

    def membership(self) -> pd.DataFrame:
        path = self.store_dir / MEMBERSHIP_FILENAME
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `bistmas fetch universe` (build_official_universe) first")
        return read_membership(self.store_dir)
