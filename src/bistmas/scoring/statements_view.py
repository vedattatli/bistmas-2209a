"""From the long statements table to one wide row per filing, then trailing-twelve-month flows.

* :func:`load_concept_map` — ``configs/concepts.toml`` (aliases -> candidate XBRL concepts per template).
* :func:`filings_wide` — per (ticker, disclosure_index): ``<alias>_cur`` / ``<alias>_prior`` for every alias.
  Flow items take the year-to-date columns (period_start = first day of the fiscal year); stock items the
  period-end columns. The prior column is the comparative of the SAME filing (same purchasing-power
  basis), which is what growth rates must use.
* :func:`add_ttm` — trailing-twelve-month flows per filing: ``FY(prev) + YTD(cur) - YTD(prior, same filing)``
  for interim periods, the annual figure for period 4; ``ttm_method`` records how each value was built.

Nothing here looks past a filing's own availability date: every row keeps ``availability_date`` and the
feature builder selects, for a decision date, the latest filing whose availability_date <= as_of.
"""

from __future__ import annotations

import tomllib
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bistmas.shared.config import CONFIG_DIR
from bistmas.shared.log import get_logger

log = get_logger("scoring.statements_view")

KEY_COLUMNS = [
    "ticker",
    "mkk_member_oid",
    "disclosure_index",
    "fiscal_year",
    "period",
    "period_end",
    "publish_ts",
    "availability_date",
    "consolidated",
    "template",
    "basis",
]


def load_concept_map(path: Path | None = None) -> dict[str, Any]:
    with open(path or CONFIG_DIR / "concepts.toml", "rb") as fh:
        return tomllib.load(fh)


def basis_for(period_end: pd.Timestamp, template: str, cfg: dict[str, Any]) -> str:
    """``'tms29'`` for inflation-adjusted statements (non-financial issuers from FY2023), else ``'nominal'``."""
    b = cfg.get("basis", {})
    if template in b.get("tms29_exempt_templates", []):
        return "nominal"
    return "tms29" if pd.Timestamp(period_end) >= pd.Timestamp(b.get("tms29_from", "2023-12-31")) else "nominal"


def _value(values: pd.Series, cand: str) -> float:
    """Value of one candidate; ``"a+b"`` sums the parts that are present (NaN when none is)."""
    parts = [c for c in cand.split("+") if c in values.index]
    vals = []
    for c in parts:
        v = values.loc[c]
        if isinstance(v, pd.Series):
            v = v.iloc[0]
        if pd.notna(v):
            vals.append(float(v))
    return float(sum(vals)) if vals else np.nan


def _pick(values: pd.Series, candidates: list[str], zero_as_missing: bool = False, use_abs: bool = False) -> float:
    """First candidate with a value in ``values`` (index = concept); ``"a+b"`` candidates are summed.

    ``zero_as_missing``: a reported 0 is skipped in favour of a later non-zero candidate (0 is returned only
    when no candidate is non-zero). ``use_abs``: return the magnitude (sign differs between concepts).
    """
    zero_seen = False
    for c in candidates:
        v = _value(values, c)
        if np.isnan(v):
            continue
        if zero_as_missing and v == 0.0:
            zero_seen = True
            continue
        return abs(v) if use_abs else v
    return 0.0 if zero_seen else np.nan


def filings_wide(long_df: pd.DataFrame, templates: pd.Series, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """One row per filing with ``<alias>_cur`` / ``<alias>_prior`` columns (see module docstring).

    ``templates`` maps ticker -> template ('industrial' | 'bank' | 'insurer' | 'other_financial').
    """
    cfg = cfg or load_concept_map()
    if long_df.empty:
        return pd.DataFrame(columns=KEY_COLUMNS)
    df = long_df.copy()
    if "context_slice" in df.columns:
        df = df[df["context_slice"].fillna("").isin(("", "Toplam"))]  # bank tables: TP/YP breakdowns are dropped
    df["period_end"] = pd.to_datetime(df["period_end"])
    df["period_start"] = pd.to_datetime(df["period_start"])
    rows: list[dict[str, Any]] = []
    for (ticker, idx), g in df.groupby(["ticker", "disclosure_index"], sort=True):
        template = str(templates.get(ticker, "industrial"))
        tmap = cfg.get(template) or cfg["industrial"]
        first = g.iloc[0]
        # current period end = the latest period_end in the filing; prior = one year earlier (same month/day)
        cur_end = g["period_end"].max()
        prior_end = _nearest(g["period_end"], cur_end - pd.DateOffset(years=1))  # leap-day tolerant
        fy_start = pd.Timestamp(year=cur_end.year, month=1, day=1)
        # fiscal years not ending in December: derive the year start from the annual filing's own start
        starts = g.loc[g["period_end"] == cur_end, "period_start"].dropna()
        if len(starts):
            fy_start = starts.min()
        flows_cur = g[(g["period_end"] == cur_end) & (g["period_start"] == fy_start)]
        prior_start = _nearest(g.loc[g["period_end"] == prior_end, "period_start"], fy_start - pd.DateOffset(years=1))
        flows_prior = g[(g["period_end"] == prior_end) & (g["period_start"] == prior_start)]
        stocks_cur = g[(g["period_end"] == cur_end) & g["period_start"].isna()]
        stocks_prior = g[(g["period_end"] == prior_end) & g["period_start"].isna()]

        def table(sub: pd.DataFrame) -> pd.DataFrame:
            return sub.drop_duplicates("concept").set_index("concept")["value_tl"]

        fc, fp, sc, sp = table(flows_cur), table(flows_prior), table(stocks_cur), table(stocks_prior)
        row: dict[str, Any] = {
            "ticker": ticker,
            "mkk_member_oid": first["mkk_member_oid"],
            "disclosure_index": int(idx),
            "fiscal_year": int(first["fiscal_year"]) if pd.notna(first["fiscal_year"]) else cur_end.year,
            "period": int(first["period"]) if pd.notna(first["period"]) else _period_from_dates(fy_start, cur_end),
            "period_end": cur_end,
            "publish_ts": first["publish_ts"],
            "availability_date": first["availability_date"],
            "consolidated": first["consolidated"],
            "template": template,
            "basis": basis_for(cur_end, template, cfg),
        }
        picks = cfg.get("picks", {})
        zero_missing, abs_aliases = set(picks.get("zero_as_missing", [])), set(picks.get("abs_aliases", []))
        for kind, cur_t, prior_t in (("flows", fc, fp), ("stocks", sc, sp)):
            for alias, cands in tmap.get(kind, {}).items():
                kw = {"zero_as_missing": alias in zero_missing, "use_abs": alias in abs_aliases}
                row[f"{alias}_cur"] = _pick(cur_t, cands, **kw)
                row[f"{alias}_prior"] = _pick(prior_t, cands, **kw)
        rows.append(row)
    out = pd.DataFrame(rows)
    for col in ("period_end", "publish_ts", "availability_date"):
        out[col] = pd.to_datetime(out[col]).astype("datetime64[ns]")
    return out.sort_values(["ticker", "period_end", "publish_ts"]).reset_index(drop=True)


def _nearest(dates: pd.Series, target: pd.Timestamp, tol_days: int = 5) -> pd.Timestamp:
    """The date in ``dates`` within ``tol_days`` of ``target`` (closest), else ``target`` (2021-02-28 vs 2020-02-29)."""
    d = pd.to_datetime(dates.dropna()).drop_duplicates()
    if d.empty:
        return target
    diff = (d - target).abs()
    return d[diff.idxmin()] if diff.min() <= pd.Timedelta(days=tol_days) else target


def _period_from_dates(start: pd.Timestamp, end: pd.Timestamp) -> int:
    months = (end.year - start.year) * 12 + end.month - start.month + 1
    return max(1, min(4, int(round(months / 3))))


def add_ttm(wide: pd.DataFrame, flow_aliases: list[str]) -> pd.DataFrame:
    """Add ``<alias>_ttm`` and ``ttm_method`` per filing.

    Interim filings: ``FY(prev fiscal year, same ticker/consolidation) + cur - prior``; the prior
    column of the SAME filing keeps the purchasing-power basis consistent. Annual filings: ``cur``.

    * Missing comparative (NaN, or a parsed 0 while ``cur`` is non-zero; 2018 bank bodies carry an empty
      'Onceki Donem' column): the YTD of the previous year is taken from the first-reported same-period
      filing of the previous fiscal year (same ticker/consolidation, available at or before this
      filing) and also fills ``<alias>_prior``; ``ttm_method = 'fy_plus_ytd_delta_prior_filing'``.
    * When no previous annual filing is available, the YTD figure is annualised (``ttm_method = 'annualised'``).
      The ``basis`` label is date-derived, so it is NOT used as a guard: interims of non-December fiscal years
      ending in early 2024 are labelled tms29 but are still nominal and difference consistently with FY nominal.
    * ``ttm_method = 'missing_prior'`` when no flow TTM could be built.
    """
    out = wide.copy()
    for a in flow_aliases:
        out[f"{a}_ttm"] = np.nan
    out["ttm_method"] = ""
    annual = out[out["period"] == 4]

    for i, row in out.iterrows():
        if int(row["period"]) == 4:
            for a in flow_aliases:
                out.at[i, f"{a}_ttm"] = row.get(f"{a}_cur", np.nan)
            out.at[i, "ttm_method"] = "annual"
            continue
        base = (out["ticker"] == row["ticker"]) & (out["consolidated"] == row["consolidated"])
        prev_fy = annual[
            base.loc[annual.index]
            & (annual["fiscal_year"] == int(row["fiscal_year"]) - 1)
            & (annual["availability_date"] <= row["availability_date"])
        ]
        fy = prev_fy.sort_values("publish_ts").iloc[-1] if len(prev_fy) else None
        if fy is not None:
            same_period = out[
                base
                & (out["period"] == row["period"])
                & (out["fiscal_year"] == int(row["fiscal_year"]) - 1)
                & (out["availability_date"] <= row["availability_date"])
            ]
            py = same_period.sort_values("publish_ts").iloc[0] if len(same_period) else None
            used_py = False
            for a in flow_aliases:
                cur, prior, fyv = row.get(f"{a}_cur", np.nan), row.get(f"{a}_prior", np.nan), fy.get(f"{a}_cur", np.nan)
                if pd.notna(cur) and (pd.isna(prior) or (prior == 0 and cur != 0)) and py is not None:
                    alt = py.get(f"{a}_cur", np.nan)
                    if pd.notna(alt):
                        prior, used_py = alt, True
                        if f"{a}_prior" in out.columns:
                            out.at[i, f"{a}_prior"] = alt
                out.at[i, f"{a}_ttm"] = (
                    fyv + cur - prior if pd.notna(cur) and pd.notna(prior) and pd.notna(fyv) else np.nan
                )
            out.at[i, "ttm_method"] = "fy_plus_ytd_delta_prior_filing" if used_py else "fy_plus_ytd_delta"
        else:
            factor = 4.0 / int(row["period"])
            for a in flow_aliases:
                cur = row.get(f"{a}_cur", np.nan)
                out.at[i, f"{a}_ttm"] = cur * factor if pd.notna(cur) else np.nan
            out.at[i, "ttm_method"] = "annualised"
        if not any(pd.notna(out.at[i, f"{a}_ttm"]) for a in flow_aliases):
            out.at[i, "ttm_method"] = "missing_prior"
    return out


def latest_filing_as_of(wide: pd.DataFrame, as_of: date | pd.Timestamp, max_age_days: int = 400) -> pd.DataFrame:
    """Latest available filing per ticker at ``as_of`` (availability_date <= as_of), one row per ticker.

    Filings older than ``max_age_days`` (period_end) are dropped: a company that stopped filing is not
    scored on stale numbers. Consolidated statements are preferred when both were filed on the same day.
    """
    ts = pd.Timestamp(as_of)
    ok = wide[(wide["availability_date"] <= ts) & (wide["period_end"] >= ts - pd.Timedelta(days=max_age_days))]
    if ok.empty:
        return ok
    # banks: solo (unconsolidated) statements by decision (Turkish bank analysts use BDDK solo figures);
    # everyone else: consolidated preferred. The preferred kind sorts last so tail(1) picks it.
    prefer_solo = (
        ok["template"].fillna("industrial").isin(["bank"])
        if "template" in ok.columns
        else pd.Series(False, index=ok.index)
    )
    ok = ok.assign(_pref=(ok["consolidated"].fillna(False).astype(bool) != prefer_solo).astype(int))
    ok = ok.sort_values(["ticker", "period_end", "_pref", "publish_ts"], ascending=[True, True, True, True])
    return ok.groupby("ticker", sort=True).tail(1).drop(columns="_pref").reset_index(drop=True)
