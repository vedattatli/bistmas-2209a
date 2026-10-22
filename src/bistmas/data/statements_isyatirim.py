"""İş Yatırım ``MaliTablo`` fallback for companies without a KAP member id (delisted / suspended names).

Only used where KAP first-reported bodies cannot be listed. Values are taken for period ends up to
``LAST_RELIABLE_PERIOD_END`` (2022-09-30): from 2022Q4 the vendor overwrites periods with TMS-29 restated
comparatives, which would leak. Publication timestamps are unknown here, so ``availability_date`` uses
the conservative p95-style lag rule (:func:`bistmas.data.pit.fallback_availability`). Output rows have the
same wide layout as :func:`bistmas.scoring.statements_view.filings_wide` (``<alias>_cur`` / ``<alias>_prior``).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bistmas.data.clients.http import PoliteSession
from bistmas.data.pit import fallback_availability, period_end_date
from bistmas.shared.log import get_logger
from bistmas.shared.store import append_manifest, read_raw_gzip, write_raw_gzip

log = get_logger("data.statements_isyatirim")

MALITABLO_URL = (
    "https://www.isyatirim.com.tr/_layouts/15/IsYatirim.Website/Common/Data.aspx/MaliTablo"
    "?companyCode={ticker}&exchange={exchange}&financialGroup={group}"
    "&year1={y}&period1=3&year2={y}&period2=6&year3={y}&period3=9&year4={y}&period4=12"
)
RAW_SUBDIR = "isyatirim_fin"
LAST_RELIABLE_PERIOD_END = date(2022, 9, 30)
#: İş Yatırım industrial (XI_29) item codes -> statements_view aliases
CODE_MAP: dict[str, str] = {
    "3C": "revenue",
    "3D": "gross_profit",
    "3DF": "operating_profit",
    "3H": "ebit",
    "3I": "pretax_profit",
    "3IA": "tax",
    "3L": "net_profit",
    "3Z": "net_profit_parent",
    "4B": "depreciation",
    "4BB": "interest_expense",
    "4C": "cfo",
    "4CAI": "capex",
    "1AA": "cash",
    "1AB": "st_investments",
    "2AA": "st_debt",
    "2BA": "lt_debt",
    "2N": "equity",
    "2O": "equity_parent",
    "1BL": "assets",
    "1A": "current_assets",
    "2A": "current_liabilities",
    "1BM": "liabilities",
}
FLOW_ALIASES = {
    "revenue",
    "gross_profit",
    "operating_profit",
    "ebit",
    "pretax_profit",
    "tax",
    "net_profit",
    "net_profit_parent",
    "depreciation",
    "interest_expense",
    "cfo",
    "capex",
}


def fetch_malitablo(
    session: PoliteSession,
    raw_dir: Path,
    ticker: str,
    year: int,
    group: str = "XI_29",
    exchange: str = "TRY",
    force: bool = False,
) -> list[dict[str, Any]]:
    """One cached request per (ticker, year, group, exchange); an empty ``value`` list is a valid result."""
    path = Path(raw_dir) / RAW_SUBDIR / ticker.upper() / f"{year}_{group}_{exchange}.json.gz"
    if path.exists() and not force:
        return json.loads(read_raw_gzip(path)).get("value", [])
    url = MALITABLO_URL.format(ticker=ticker.upper(), exchange=exchange, group=group, y=year)
    resp = session.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"MaliTablo {resp.status_code} for {ticker} {year}")
    try:
        data = resp.json()
    except ValueError as exc:
        raise RuntimeError(f"MaliTablo {ticker} {year}: not JSON ({exc})") from exc
    if not isinstance(data, dict) or "value" not in data:
        raise RuntimeError(f"MaliTablo {ticker} {year}: unexpected payload")
    digest = write_raw_gzip(resp.content, path)
    append_manifest(
        Path(raw_dir) / RAW_SUBDIR / "manifest.jsonl",
        {
            "source": "isyatirim",
            "url": url,
            "ticker": ticker,
            "year": year,
            "sha256": digest,
            "items": len(data["value"]),
        },
    )
    return data["value"]


def parse_year(items: list[dict[str, Any]], ticker: str, year: int, template: str = "industrial") -> pd.DataFrame:
    """Wide rows (one per period with any data) from one year's item list; ``*_prior`` left NaN."""
    codes = {str(it.get("itemCode")).strip(): it for it in items}
    rows = []
    for period in (1, 2, 3, 4):
        vals: dict[str, float] = {}
        for code, alias in CODE_MAP.items():
            it = codes.get(code)
            v = it.get(f"value{period}") if it else None
            vals[alias] = float(v) if v not in (None, "") else np.nan
        if all(np.isnan(v) for v in vals.values()):
            continue
        pend = period_end_date(year, period)
        rows.append(
            {
                "ticker": ticker.upper(),
                "mkk_member_oid": "",
                "disclosure_index": -int(f"{year}{period}"),  # negative synthetic id: fallback source
                "fiscal_year": year,
                "period": period,
                "period_end": pd.Timestamp(pend),
                "publish_ts": pd.NaT,
                "availability_date": pd.Timestamp(fallback_availability(pend, period, template)),
                "consolidated": True,
                "template": template,
                "basis": "nominal",
                "source": "isyatirim_fallback",
                **{f"{a}_cur": v for a, v in vals.items()},
                **{f"{a}_prior": np.nan for a in vals},
            }
        )
    return pd.DataFrame(rows)


def add_prior_from_previous_year(wide: pd.DataFrame) -> pd.DataFrame:
    """Fill ``*_prior`` with the same period of the previous fiscal year (nominal basis, so consistent)."""
    out = wide.copy()
    key = out.set_index(["ticker", "fiscal_year", "period"])
    for i, row in out.iterrows():
        prev = (row["ticker"], int(row["fiscal_year"]) - 1, int(row["period"]))
        if prev in key.index:
            p = key.loc[prev]
            for alias in CODE_MAP.values():
                out.at[i, f"{alias}_prior"] = p.get(f"{alias}_cur", np.nan)
    return out


def build_fallback_wide(
    session: PoliteSession, raw_dir: Path, plan: pd.DataFrame, last_year: int = LAST_RELIABLE_PERIOD_END.year
) -> pd.DataFrame:
    """Fetch (cached) and parse every (ticker, year) of the missing-oid companies within their membership years."""
    frames = []
    for rec in plan.itertuples(index=False):
        y_from, y_to = int(rec.year_from), min(int(rec.year_to), last_year) if int(rec.year_to) < 9999 else last_year
        for year in range(max(y_from, 2016), y_to + 1):
            try:
                items = fetch_malitablo(session, raw_dir, rec.ticker, year)
            except RuntimeError as exc:
                log.warning("%s %s: %s", rec.ticker, year, exc)
                continue
            frames.append(parse_year(items, rec.ticker, year))
    if not frames:
        return pd.DataFrame()
    wide = pd.concat([f for f in frames if len(f)], ignore_index=True)
    wide = add_prior_from_previous_year(wide)
    wide = wide[wide["period_end"] <= pd.Timestamp(LAST_RELIABLE_PERIOD_END)]
    log.info("İş Yatırım fallback: %d filing rows for %d tickers", len(wide), wide["ticker"].nunique())
    return wide.reset_index(drop=True)
