"""Rule-based replica of the BIST index selection (robustness universe only).

The official universe (``official.py``) reproduces the published BIST 100 / BIST 30 lists. This
module re-implements the *published selection rules* on our own price panel so that results can be
reported on a universe that does not depend on Borsa İstanbul's lists at all (a robustness table,
never the headline). It cannot be "BIST 100" for benchmarking because several inputs are not
public in point-in-time form.

BIST ground rules (Borsa İstanbul, *BIST Pay Endeksleri Temel Kuralları*, April 2020 edition, and the
Feb-2026 *BIST Piyasa Değeri Ağırlıklı Pay Endeksleri Kural Seti*; sections 5-8):

* Four index periods per year (Jan-Mar, Apr-Jun, Jul-Sep, Oct-Dec). Changes are announced at least
  10 calendar days before a period and take effect on the first session of the period.
* Valuation period: the six months preceding the review (older rules: three months for some
  indices). Only stocks that traded at least 60 days by the valuation day are eligible.
* Selection pool: stocks that will trade in the Yıldız Pazar (Star Market) in the next period
  (since the Oct-2019 rule change; before it, Ana Pazar stocks were eligible too, which is why
  2020Q1 had 23 in / 23 out). BIST 100 falls back to an "additional stocks" list when the pool is
  short of 108 (2020) / 113 (2026) names.
* Two rankings over the valuation period: (a) average free-float market capitalisation
  (*ortalama fiili dolaşımdaki pay piyasa değeri*) and (b) average daily traded value (only
  secondary-market ``.E`` trades; special-trade reports excluded).
* Rank merge ("both lists within the top *k*" procedure): a stock's final rank is the smallest
  *k* such that it is within the top *k* of both lists, i.e. ``max(rank_a, rank_b)``; ties are
  broken by the larger free-float market cap.
* Entry / exit buffers (section 7.2 table, per index): an incumbent stays unless its merged rank
  falls below the exit rank; a non-member enters only if its merged rank is within the entry rank.
  When entrants outnumber leavers, the lowest-ranked incumbents are removed to keep *n*; when
  leavers outnumber entrants, the best-ranked non-members are added. BIST 30 uses 25/35 (older
  rules 20/40 style buffers exist; check the table of the rule set in force for each period).
* One share class per issuer (the more liquid class); stocks on the Watchlist (Yakın İzleme) or
  suspended for more than five business days are excluded.
* Mid-period events (section 7/8): delisting, transfer to the Watchlist or a suspension longer
  than five business days remove a stock on the same day; the first stock of the announced
  substitute list takes its place. Large IPOs whose free-float market cap exceeds that of the 20th
  BIST 30 stock enter BIST 30/50/100 on the fifth trading day (rule 7.5).

Replica conventions (what :func:`build_rule_based_membership` actually does):

* Review dates are the first session on/after 1 Jan / 1 Apr / 1 Jul / 1 Oct found in the session
  list (``effective_start``). The valuation window is the last ``valuation_days`` (126 ~ six months)
  sessions ending on the **last session of the month-but-one before the period** (Nov / Feb / May /
  Aug): the exchange values a period on that day and announces the list at least ten days before the
  period starts, so ``membership_known_since = effective_start - lead_days`` (10 calendar days).
* Metrics per ticker: simple averages of ``ffloat_mcap_tl`` (vendor point-in-time free-float market
  cap, ``HAO_PD``) and ``traded_value_tl`` over the sessions the ticker traded inside the window.
* Eligibility: at least ``min_listed_days`` price rows up to the window end (250 ~ one year, stricter
  than the exchange's 60 days, matching ``universe.min_months_listed = 12``), at least
  ``min_window_coverage`` of the window sessions traded, a trade within the last ``max_stale_sessions``
  sessions of the window (proxy for "not suspended / not delisted"), both metrics present, and one
  share class per issuer (the class with the larger average traded value; issuers come from the
  security master's ``mkk_member_oid``, tickers without an oid are their own issuer).
* Selection: :func:`select_by_rank_merge` with the entry/exit buffers against the previous review's
  list (the first review is an initial top-``n`` selection).
* Spells: consecutive reviews that keep a ticker merge into one spell; a leaver's spell ends on the
  review's ``effective_start`` (exclusive). A member whose price series ends before the panel does is
  closed on the first session after its last trade (``effective_end`` exclusive); a spell that would
  start after that (selected on stale data) is dropped. Substitutes, fast-track IPO entries, the
  Yıldız Pazar constraint and watch-list transfers are not modelled (documented gap).

Everything above is a pure function of the price panel (``[ticker, date, ffloat_mcap_tl,
traded_value_tl]``), so tests run on synthetic panels; :class:`RuleBasedProvider` only loads
``prices_daily.parquet`` / ``security_master.parquet`` from the store and calls it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bistmas.data.security_master import STORE_FILENAME as SECURITY_MASTER_FILENAME
from bistmas.data.universe.provider import (
    MEMBERSHIP_COLUMNS,
    coerce_membership,
    normalise_ticker,
    normalise_ticker_series,
    to_datetime_column,
)
from bistmas.shared.calendar import CANCELLED_SESSIONS, CLOSURES
from bistmas.shared.log import get_logger
from bistmas.shared.store import read_parquet, write_parquet

log = get_logger("data.universe.rule_based")

METRIC_COLUMNS: list[str] = ["ticker", "avg_ff_mcap", "avg_traded_value"]
RANKING_COLUMNS: list[str] = [
    "ticker",
    "avg_ff_mcap",
    "avg_traded_value",
    "rank_ff_mcap",
    "rank_traded_value",
    "final_rank",
    "merged_rank",
]
PANEL_COLUMNS: list[str] = ["ticker", "date", "ffloat_mcap_tl", "traded_value_tl"]
ELIGIBILITY_COLUMNS: list[str] = [
    "ticker",
    "avg_ff_mcap",
    "avg_traded_value",
    "n_window",
    "listed_days",
    "last_date",
    "issuer",
    "eligible",
    "reason",
]
SCHEDULE_COLUMNS: list[str] = ["label", "effective_start", "window_start", "window_end", "known_since", "n_window"]
REVIEW_COLUMNS: list[str] = [*SCHEDULE_COLUMNS, "n_eligible", "n_selected", "n_in", "n_out", "members"]

PRICES_FILENAME = "prices_daily.parquet"
RULE_BASED_MEMBERSHIP_FILENAME = "universe_membership_rule_based.parquet"
RULE_BASED_REVIEWS_FILENAME = "universe_rule_based_reviews.parquet"
CANCELLED_FLAG_COLUMN = "flag_cancelled_session"

QUARTER_MONTHS: tuple[int, ...] = (1, 4, 7, 10)
DEFAULT_N = 100
DEFAULT_VALUATION_DAYS = 126
DEFAULT_ENTRY_RANK = 95
DEFAULT_EXIT_RANK = 105
DEFAULT_MIN_LISTED_DAYS = 250
DEFAULT_LEAD_DAYS = 10
DEFAULT_MIN_WINDOW_COVERAGE = 0.9
DEFAULT_MAX_STALE_SESSIONS = 5
DEFAULT_INDEX_CODE = "XU100"


# --------------------------------------------------------------------------- inputs
def _sessions_index(sessions: Iterable[date | pd.Timestamp | str] | pd.DatetimeIndex) -> pd.DatetimeIndex:
    values = to_datetime_column(list(sessions)).dropna().unique()
    idx = pd.DatetimeIndex(np.sort(values), name="date")
    drop = {pd.Timestamp(d) for d in CANCELLED_SESSIONS | CLOSURES}
    return pd.DatetimeIndex([d for d in idx if d not in drop], dtype="datetime64[ns]", name="date")


def prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise a price panel to :data:`PANEL_COLUMNS` (one row per ticker-session)."""
    missing = [c for c in PANEL_COLUMNS if c not in panel.columns]
    if missing:
        raise ValueError(f"price panel is missing columns {missing}")
    out = panel.loc[:, PANEL_COLUMNS].copy()
    out["ticker"] = normalise_ticker_series(out["ticker"])
    out["date"] = to_datetime_column(out["date"])
    for col in ("ffloat_mcap_tl", "traded_value_tl"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    out = out[(out["ticker"] != "") & out["date"].notna()]
    drop = {pd.Timestamp(d) for d in CANCELLED_SESSIONS | CLOSURES}
    if drop:
        out = out[~out["date"].isin(list(drop))]
    dup = out.duplicated(["ticker", "date"])
    if dup.any():
        raise ValueError(f"price panel has {int(dup.sum())} duplicate (ticker, date) rows")
    return out.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- schedule
def quarter_label(ts: pd.Timestamp) -> str:
    return f"{ts.year}Q{(ts.month - 1) // 3 + 1}"


def review_schedule(
    sessions: Iterable[date | pd.Timestamp | str] | pd.DatetimeIndex,
    valuation_days: int = DEFAULT_VALUATION_DAYS,
    lead_days: int = DEFAULT_LEAD_DAYS,
    start: date | pd.Timestamp | str | None = None,
    end: date | pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Quarterly reviews (:data:`SCHEDULE_COLUMNS`) that the session list supports.

    ``effective_start`` is the first session on/after 1 Jan/Apr/Jul/Oct; ``window_end`` the last session
    on/before the end of the month-but-one before it; ``window_start`` is ``valuation_days - 1``
    sessions earlier. Quarters whose window would reach before the first session are skipped (logged).
    ``start`` / ``end`` restrict ``effective_start`` (inclusive).
    """
    if valuation_days < 1:
        raise ValueError("valuation_days must be >= 1")
    idx = _sessions_index(sessions)
    rows: list[dict[str, Any]] = []
    if len(idx) == 0:
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)
    for year in range(idx[0].year, idx[-1].year + 1):
        for month in QUARTER_MONTHS:
            q_day = pd.Timestamp(year, month, 1)
            pos = idx.searchsorted(q_day, side="left")
            if pos >= len(idx):
                continue
            effective = idx[pos]
            if (effective.year, (effective.month - 1) // 3) != (year, (month - 1) // 3):
                continue  # no session inside this quarter at all
            window_end_day = q_day - pd.offsets.MonthBegin(1) - pd.Timedelta(days=1)
            wpos = idx.searchsorted(window_end_day, side="right") - 1
            if wpos < valuation_days - 1:
                log.info(
                    "review %s skipped: only %d sessions before %s",
                    quarter_label(q_day),
                    wpos + 1,
                    window_end_day.date(),
                )
                continue
            rows.append(
                {
                    "label": quarter_label(q_day),
                    "effective_start": effective,
                    "window_start": idx[wpos - valuation_days + 1],
                    "window_end": idx[wpos],
                    "known_since": effective - pd.Timedelta(days=lead_days),
                    "n_window": valuation_days,
                }
            )
    df = pd.DataFrame(rows, columns=SCHEDULE_COLUMNS)
    for col in ("effective_start", "window_start", "window_end", "known_since"):
        df[col] = pd.to_datetime(df[col]).astype("datetime64[ns]")
    if start is not None:
        df = df[df["effective_start"] >= pd.Timestamp(start).normalize()]
    if end is not None:
        df = df[df["effective_start"] <= pd.Timestamp(end).normalize()]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- metrics and eligibility
def valuation_metrics(
    panel: pd.DataFrame,
    window_start: date | pd.Timestamp | str,
    window_end: date | pd.Timestamp | str,
    sessions: pd.DatetimeIndex | None = None,
    min_listed_days: int = DEFAULT_MIN_LISTED_DAYS,
    min_window_coverage: float = DEFAULT_MIN_WINDOW_COVERAGE,
    max_stale_sessions: int = DEFAULT_MAX_STALE_SESSIONS,
) -> pd.DataFrame:
    """Per-ticker window averages and eligibility (:data:`ELIGIBILITY_COLUMNS`; ``issuer`` = ticker).

    ``reason`` names the first failed test (``listed_days``, ``stale``, ``coverage``, ``missing_metric``)
    and is empty for eligible rows. ``panel`` may be the full history: only rows up to ``window_end``
    count towards ``listed_days`` and only rows inside the window towards the averages.
    """
    p = prepare_panel(panel) if list(panel.columns) != PANEL_COLUMNS else panel
    ws, we = pd.Timestamp(window_start).normalize(), pd.Timestamp(window_end).normalize()
    hist = p[p["date"] <= we]
    win = hist[hist["date"] >= ws]
    idx = _sessions_index(sessions) if sessions is not None else pd.DatetimeIndex(np.sort(hist["date"].unique()))
    window_sessions = idx[(idx >= ws) & (idx <= we)]
    n_sessions = max(len(window_sessions), 1)
    stale_cutoff = window_sessions[-max_stale_sessions] if len(window_sessions) >= max_stale_sessions else ws

    g = win.groupby("ticker", sort=True)
    out = pd.DataFrame(
        {
            "avg_ff_mcap": g["ffloat_mcap_tl"].mean(),
            "avg_traded_value": g["traded_value_tl"].mean(),
            "n_window": g.size(),
        }
    )
    out["listed_days"] = hist.groupby("ticker").size().reindex(out.index).fillna(0).astype(int)
    out["last_date"] = hist.groupby("ticker")["date"].max().reindex(out.index)
    out = out.reset_index()
    out["issuer"] = out["ticker"]

    reasons = pd.Series("", index=out.index, dtype=object)
    need = int(np.ceil(min_window_coverage * n_sessions))
    checks = [
        ("listed_days", out["listed_days"] < min_listed_days),
        ("stale", out["last_date"] < stale_cutoff),
        ("coverage", out["n_window"] < need),
        ("missing_metric", out["avg_ff_mcap"].isna() | out["avg_traded_value"].isna()),
    ]
    for name, failed in checks:
        reasons = reasons.where(~(failed & (reasons == "")), name)
    out["reason"] = reasons.astype(str)
    out["eligible"] = out["reason"] == ""
    out["last_date"] = pd.to_datetime(out["last_date"]).astype("datetime64[ns]")
    return out.loc[:, ELIGIBILITY_COLUMNS].reset_index(drop=True)


def keep_one_class_per_issuer(metrics: pd.DataFrame, issuer_of: Mapping[str, str] | None) -> pd.DataFrame:
    """Mark all but the most liquid eligible share class of each issuer ineligible (``reason='other_class'``).

    ``issuer_of`` maps ticker -> issuer id (e.g. ``mkk_member_oid``); unmapped tickers are their own issuer.
    """
    out = metrics.copy()
    mapping = {normalise_ticker(k): str(v) for k, v in (issuer_of or {}).items() if str(v)}
    out["issuer"] = [mapping.get(t, t) for t in out["ticker"]]
    elig = out[out["eligible"]].sort_values(
        ["issuer", "avg_traded_value", "avg_ff_mcap", "ticker"], ascending=[True, False, False, True], kind="stable"
    )
    keep = set(elig.drop_duplicates("issuer", keep="first")["ticker"])
    other = out["eligible"] & ~out["ticker"].isin(keep)
    out.loc[other, "eligible"] = False
    out.loc[other, "reason"] = "other_class"
    return out


# --------------------------------------------------------------------------- membership
def _empty_membership() -> pd.DataFrame:
    return coerce_membership(None)


def build_rule_based_membership(
    panel: pd.DataFrame,
    sessions: Iterable[date | pd.Timestamp | str] | pd.DatetimeIndex | None = None,
    issuer_of: Mapping[str, str] | None = None,
    n: int = DEFAULT_N,
    valuation_days: int = DEFAULT_VALUATION_DAYS,
    entry_rank: int = DEFAULT_ENTRY_RANK,
    exit_rank: int = DEFAULT_EXIT_RANK,
    min_listed_days: int = DEFAULT_MIN_LISTED_DAYS,
    one_class_per_issuer: bool = True,
    index_code: str = DEFAULT_INDEX_CODE,
    lead_days: int = DEFAULT_LEAD_DAYS,
    start: date | pd.Timestamp | str | None = None,
    end: date | pd.Timestamp | str | None = None,
    min_window_coverage: float = DEFAULT_MIN_WINDOW_COVERAGE,
    max_stale_sessions: int = DEFAULT_MAX_STALE_SESSIONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quarterly rule-based membership from a price panel (see the module docstring for the rules).

    Returns ``(membership, reviews)``: the standard membership table (``MEMBERSHIP_COLUMNS``, sorted)
    and one row per review (:data:`REVIEW_COLUMNS`, ``members`` = comma-joined selected tickers).
    ``sessions`` defaults to the panel's distinct dates; pass the data-derived trading calendar so the
    first session after a delisting is exact.
    """
    p = prepare_panel(panel)
    idx = _sessions_index(sessions) if sessions is not None else _sessions_index(p["date"].unique())
    if len(idx) == 0:
        return _empty_membership(), pd.DataFrame(columns=REVIEW_COLUMNS)
    schedule = review_schedule(idx, valuation_days, lead_days, start, end)
    if schedule.empty:
        log.warning("rule-based universe: no review fits the panel (%s..%s)", idx[0].date(), idx[-1].date())
        return _empty_membership(), pd.DataFrame(columns=REVIEW_COLUMNS)

    previous: set[str] | None = None
    open_spells: dict[str, dict[str, Any]] = {}
    spells: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    for r in schedule.itertuples(index=False):
        metrics = valuation_metrics(
            p, r.window_start, r.window_end, idx, min_listed_days, min_window_coverage, max_stale_sessions
        )
        if one_class_per_issuer:
            metrics = keep_one_class_per_issuer(metrics, issuer_of)
        eligible = metrics.loc[metrics["eligible"], METRIC_COLUMNS]
        selected = select_by_rank_merge(
            eligible, n=n, entry_rank=entry_rank, exit_rank=exit_rank, current_members=previous
        )
        chosen = set(selected)
        n_in = len(chosen - previous) if previous is not None else len(chosen)
        n_out = len(previous - chosen) if previous is not None else 0
        for ticker in list(open_spells):
            if ticker not in chosen:
                spell = open_spells.pop(ticker)
                spell["effective_end"] = r.effective_start
                spells.append(spell)
        for ticker in selected:
            if ticker not in open_spells:
                open_spells[ticker] = {
                    "index_code": index_code,
                    "ticker": ticker,
                    "effective_start": r.effective_start,
                    "effective_end": pd.NaT,
                    "membership_known_since": r.known_since,
                    "source": f"rule_based:{r.label}",
                }
        previous = chosen
        reviews.append(
            {
                **{c: getattr(r, c) for c in SCHEDULE_COLUMNS},
                "n_eligible": int(len(eligible)),
                "n_selected": int(len(selected)),
                "n_in": int(n_in),
                "n_out": int(n_out),
                "members": ",".join(selected),
            }
        )
        log.info(
            "review %s: %d eligible, %d selected (%d in / %d out), window %s..%s",
            r.label,
            len(eligible),
            len(selected),
            n_in,
            n_out,
            r.window_start.date(),
            r.window_end.date(),
        )
    spells.extend(open_spells.values())

    # members whose series ends before the panel does leave on the first session after their last trade
    last_dates = p.groupby("ticker")["date"].max()
    panel_end = idx[-1]
    kept: list[dict[str, Any]] = []
    for spell in spells:
        last = last_dates.get(spell["ticker"])
        if last is not None and last < panel_end:
            pos = idx.searchsorted(last, side="right")
            close_at = idx[pos] if pos < len(idx) else last + timedelta(days=1)
            if close_at <= spell["effective_start"]:
                log.info(
                    "spell %s from %s dropped: last trade %s",
                    spell["ticker"],
                    spell["effective_start"].date(),
                    last.date(),
                )
                continue
            if pd.isna(spell["effective_end"]) or close_at < spell["effective_end"]:
                spell["effective_end"] = close_at
        kept.append(spell)

    membership = coerce_membership(pd.DataFrame(kept, columns=MEMBERSHIP_COLUMNS)) if kept else _empty_membership()
    reviews_df = pd.DataFrame(reviews, columns=REVIEW_COLUMNS)
    return membership, reviews_df


# --------------------------------------------------------------------------- provider
class RuleBasedProvider:
    """Rule-based BIST index replica built from ``<store_dir>/prices_daily.parquet``.

    ``membership()`` returns the standard membership table (cached after the first call) and fills
    ``reviews`` (one row per quarterly review). Issuers for the one-class-per-issuer rule come from
    ``<store_dir>/security_master.parquet`` (``ticker -> mkk_member_oid``); when that file is missing every
    ticker is its own issuer (warning). Parameters mirror :func:`build_rule_based_membership`.
    """

    name = "rule_based"

    def __init__(
        self,
        store_dir: Path,
        n: int = DEFAULT_N,
        valuation_days: int = DEFAULT_VALUATION_DAYS,
        entry_rank: int = DEFAULT_ENTRY_RANK,
        exit_rank: int = DEFAULT_EXIT_RANK,
        min_listed_days: int = DEFAULT_MIN_LISTED_DAYS,
        one_class_per_issuer: bool = True,
        index_code: str = DEFAULT_INDEX_CODE,
        lead_days: int = DEFAULT_LEAD_DAYS,
        start: date | pd.Timestamp | str | None = None,
        end: date | pd.Timestamp | str | None = None,
    ) -> None:
        self.store_dir = Path(store_dir)
        self.n = n
        self.valuation_days = valuation_days
        self.entry_rank = entry_rank
        self.exit_rank = exit_rank
        self.min_listed_days = min_listed_days
        self.one_class_per_issuer = one_class_per_issuer
        self.index_code = index_code
        self.lead_days = lead_days
        self.start = start
        self.end = end
        self.reviews: pd.DataFrame | None = None
        self._membership: pd.DataFrame | None = None

    @property
    def prices_path(self) -> Path:
        return self.store_dir / PRICES_FILENAME

    def load_panel(self) -> pd.DataFrame:
        """``[ticker, date, ffloat_mcap_tl, traded_value_tl]`` from the price store (cancelled sessions dropped)."""
        import pyarrow.parquet as pq

        if not self.prices_path.exists():
            raise FileNotFoundError(f"{self.prices_path} not found; run `bistmas fetch prices` first")
        available = pq.ParquetFile(self.prices_path).schema_arrow.names
        missing = [c for c in PANEL_COLUMNS if c not in available]
        if missing:
            raise ValueError(f"{self.prices_path} lacks columns {missing}")
        columns = PANEL_COLUMNS + ([CANCELLED_FLAG_COLUMN] if CANCELLED_FLAG_COLUMN in available else [])
        df = read_parquet(self.prices_path, columns=columns)
        if CANCELLED_FLAG_COLUMN in df.columns:
            df = df[~df[CANCELLED_FLAG_COLUMN].fillna(False).astype(bool)]
        return prepare_panel(df)

    def load_issuers(self) -> dict[str, str]:
        """``ticker -> mkk_member_oid`` from the security master (empty when the file is absent)."""
        path = self.store_dir / SECURITY_MASTER_FILENAME
        if not path.exists():
            log.warning("%s not found; one-class-per-issuer rule runs with ticker == issuer", path)
            return {}
        master = read_parquet(path, columns=["ticker", "mkk_member_oid"])
        master = master[master["mkk_member_oid"].fillna("").astype(str) != ""]
        return {normalise_ticker(t): str(o) for t, o in zip(master["ticker"], master["mkk_member_oid"], strict=True)}

    def membership(self) -> pd.DataFrame:
        if self._membership is None:
            panel = self.load_panel()
            issuers = self.load_issuers() if self.one_class_per_issuer else {}
            self._membership, self.reviews = build_rule_based_membership(
                panel,
                issuer_of=issuers,
                n=self.n,
                valuation_days=self.valuation_days,
                entry_rank=self.entry_rank,
                exit_rank=self.exit_rank,
                min_listed_days=self.min_listed_days,
                one_class_per_issuer=self.one_class_per_issuer,
                index_code=self.index_code,
                lead_days=self.lead_days,
                start=self.start,
                end=self.end,
            )
        return self._membership.copy()

    def write(self, store_dir: Path | None = None) -> tuple[Path, Path]:
        """Write the membership and review tables next to the official ones; returns both paths."""
        target = Path(store_dir) if store_dir is not None else self.store_dir
        membership = self.membership()
        reviews = self.reviews if self.reviews is not None else pd.DataFrame(columns=REVIEW_COLUMNS)
        return (
            write_parquet(membership, target / RULE_BASED_MEMBERSHIP_FILENAME),
            write_parquet(reviews, target / RULE_BASED_REVIEWS_FILENAME),
        )


# --------------------------------------------------------------------------- selection (pure)
def merged_ranking(metrics: pd.DataFrame) -> pd.DataFrame:
    """Rank tickers by BIST rank merge: ``final_rank = max(rank_ff_mcap, rank_traded_value)``.

    Both metric ranks are descending (1 = largest, ties share the minimum rank). Rows are sorted by
    ``final_rank``, then larger ``avg_ff_mcap`` first, then ticker; ``merged_rank`` is the 1-based
    position in that order and is what the entry/exit buffers compare against. Rows with a missing
    or non-numeric metric are ineligible and dropped; duplicate tickers raise ``ValueError``.
    """
    missing = [c for c in METRIC_COLUMNS if c not in metrics.columns]
    if missing:
        raise ValueError(f"metrics table is missing columns {missing}")
    df = metrics.loc[:, METRIC_COLUMNS].copy()
    df["ticker"] = normalise_ticker_series(df["ticker"])
    df["avg_ff_mcap"] = pd.to_numeric(df["avg_ff_mcap"], errors="coerce")
    df["avg_traded_value"] = pd.to_numeric(df["avg_traded_value"], errors="coerce")
    df = df[(df["ticker"] != "") & df["avg_ff_mcap"].notna() & df["avg_traded_value"].notna()]
    duplicates = sorted(df.loc[df["ticker"].duplicated(), "ticker"].unique())
    if duplicates:
        raise ValueError(f"duplicate tickers in metrics: {duplicates}")
    if df.empty:
        return pd.DataFrame({c: pd.Series(dtype=float if c != "ticker" else str) for c in RANKING_COLUMNS})
    df["rank_ff_mcap"] = df["avg_ff_mcap"].rank(ascending=False, method="min").astype(int)
    df["rank_traded_value"] = df["avg_traded_value"].rank(ascending=False, method="min").astype(int)
    df["final_rank"] = df[["rank_ff_mcap", "rank_traded_value"]].max(axis=1).astype(int)
    df = df.sort_values(
        ["final_rank", "avg_ff_mcap", "ticker"], ascending=[True, False, True], kind="stable"
    ).reset_index(drop=True)
    df["merged_rank"] = range(1, len(df) + 1)
    return df.loc[:, RANKING_COLUMNS]


def select_by_rank_merge(
    metrics: pd.DataFrame,
    n: int = 100,
    entry_rank: int = 95,
    exit_rank: int = 105,
    current_members: set[str] | None = None,
) -> list[str]:
    """Select ``n`` tickers by BIST rank merge with entry/exit buffers.

    ``metrics`` has ``[ticker, avg_ff_mcap, avg_traded_value]`` for the valuation period. Without
    ``current_members`` the top ``n`` of :func:`merged_ranking` is returned (initial selection).
    With ``current_members``:

    1. non-members with ``merged_rank <= entry_rank`` must enter;
    2. members with ``merged_rank > exit_rank`` (or missing from ``metrics``) must leave;
    3. if the result exceeds ``n``, the lowest-ranked incumbents (never a forced entrant) are
       dropped; if it is short of ``n``, the best-ranked remaining non-members (excluding the forced
       leavers) are added, so the list can be shorter than ``n`` only when the pool is too small.

    The result is ordered by ``merged_rank``.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if entry_rank > exit_rank:
        raise ValueError("entry_rank must be <= exit_rank")
    ranking = merged_ranking(metrics)
    order = ranking["ticker"].tolist()
    if current_members is None:
        return order[:n]
    position = {t: int(p) for t, p in zip(ranking["ticker"], ranking["merged_rank"], strict=True)}
    current = {normalise_ticker(t) for t in current_members} - {""}
    must_in = [t for t in order if t not in current and position[t] <= entry_rank]
    must_out = {t for t in current if t not in position or position[t] > exit_rank}
    selected = {t for t in current if t in position and t not in must_out} | set(must_in)
    if len(selected) > n:
        for t in reversed(order):
            if len(selected) <= n:
                break
            if t in selected and t not in must_in:
                selected.discard(t)
        if len(selected) > n:  # only forced entrants left (entry_rank > n): keep the best ranked
            selected = set([t for t in order if t in selected][:n])
    elif len(selected) < n:
        for t in order:
            if len(selected) >= n:
                break
            if t not in selected and t not in must_out:
                selected.add(t)
    return [t for t in order if t in selected]
