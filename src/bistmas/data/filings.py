"""Filing index: which KAP financial-report disclosures exist for every universe company, and when.

Two steps, both idempotent:

* :func:`run_filing_listing` — the quota-aware network job. One ``byCriteria`` POST per company over the
  whole date range; a capped response (2000 rows, typical for banks) is split into calendar years, and a
  capped year into half-years. Every response is cached by the KAP client, progress is persisted after
  every request, and a 429 stops the run (``stopped_reason == "blocked"``) so it can be resumed later.
* :func:`build_filing_index` — offline: reads every cached listing, keeps the periodic financial reports
  and writes ``data/store/filing_index.parquet``.

Companies without a KAP oid (delisted before the current member list) are recorded with
``oid_source == "missing"``; their statements come from the İş Yatırım fallback in the feature build.
"""

from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from bistmas.data.clients.http import PoliteSession, RateLimitedError
from bistmas.data.clients.kap import DISCLOSURE_CAP, KapClient, KapParseError, financial_report_rows
from bistmas.data.security_master import load_security_master
from bistmas.data.universe.official import load_aliases, read_membership
from bistmas.shared.config import CONFIG_DIR, Settings
from bistmas.shared.log import get_logger
from bistmas.shared.store import read_json, read_parquet, utc_now_iso, write_json, write_parquet

log = get_logger("data.filings")

FILING_INDEX_FILE = "filing_index.parquet"
FILING_META_FILE = "filing_index_meta.json"
PROGRESS_FILE = "filings_progress.json"
DEFAULT_START = date(2017, 1, 1)
INDEX_COLUMNS = [
    "mkk_member_oid",
    "ticker",
    "disclosure_index",
    "publish_ts",
    "fiscal_year",
    "period",
    "rule_type",
    "is_late",
    "stock_codes",
    "subject",
    "lang",
]


# ----------------------------------------------------------------------------- companies to cover
def plan_companies(
    settings: Settings, index_codes: tuple[str, ...] = ("XU100",), lookback_years: int = 1
) -> pd.DataFrame:
    """Universe tickers with their KAP oid: ``[ticker, mkk_member_oid, oid_source, year_from, year_to]``.

    ``year_from``/``year_to`` bound the filings a ticker can need: from ``lookback_years`` before its first
    membership spell (the previous annual report feeds the TTM) to the end of its last spell (9999 = open).

    ``oid_source`` is ``"kap"`` (current member), ``"successor"`` (renamed: the new code's oid, same
    legal entity) or ``"missing"`` (delisted before the current list; fallback source needed).
    """
    membership = read_membership(settings.paths.store)
    mem = membership[membership["index_code"].isin(index_codes)]
    tickers = sorted(mem["ticker"].unique())
    spans = mem.groupby("ticker").agg(first=("effective_start", "min"), last=("effective_end", "max"))
    open_end = mem.groupby("ticker")["effective_end"].apply(lambda x: bool(x.isna().any()))
    master = load_security_master(settings.paths.store)
    current = master[master["is_current"]].set_index("ticker")
    aliases = load_aliases(CONFIG_DIR / "ticker_aliases.csv")
    renames = {str(r.old_ticker): str(r.new_ticker) for r in aliases.itertuples(index=False) if str(r.kind) == "rename"}
    rows = []
    for t in tickers:
        y_from = int(pd.Timestamp(spans.loc[t, "first"]).year) - lookback_years
        last = spans.loc[t, "last"]
        y_to = 9999 if bool(open_end.get(t, False)) or pd.isna(last) else int(pd.Timestamp(last).year)
        base = {"ticker": t, "year_from": y_from, "year_to": y_to}
        if t in current.index:
            rows.append({**base, "mkk_member_oid": str(current.loc[t, "mkk_member_oid"]), "oid_source": "kap"})
        elif renames.get(t) in current.index:
            rows.append(
                {**base, "mkk_member_oid": str(current.loc[renames[t], "mkk_member_oid"]), "oid_source": "successor"}
            )
        else:
            rows.append({**base, "mkk_member_oid": "", "oid_source": "missing"})
    df = pd.DataFrame(rows, columns=["ticker", "mkk_member_oid", "oid_source", "year_from", "year_to"])
    log.info("filing plan: %d tickers, oid sources %s", len(df), df["oid_source"].value_counts().to_dict())
    return df


# ----------------------------------------------------------------------------- progress
def progress_path(settings: Settings) -> Path:
    return settings.paths.raw / "kap" / PROGRESS_FILE


def _load_progress(path: Path) -> dict[str, Any]:
    if path.exists():
        return read_json(path)
    return {"version": 1, "started_at": utc_now_iso(), "done": {}, "failed": {}, "requests_made": 0, "runs": []}


MAX_WINDOW_DAYS = 365  # the byCriteria route rejects longer ranges with "HTTP 400"


def default_windows(start: date, end: date) -> list[tuple[date, date]]:
    """Calendar-year windows of at most 365 days (leap years are split into two halves)."""
    out: list[tuple[date, date]] = []
    for y in range(start.year, end.year + 1):
        a, b = max(start, date(y, 1, 1)), min(end, date(y, 12, 31))
        if (b - a).days + 1 > MAX_WINDOW_DAYS:
            mid = date(y, 7, 1)
            out.extend([(a, mid - timedelta(days=1)), (mid, b)])
        else:
            out.append((a, b))
    return out


def _windows_for(oid: str, start: date, end: date, progress: dict[str, Any]) -> list[tuple[date, date]]:
    """Date windows still to fetch for one company: yearly windows, or the splits recorded after a cap."""
    splits = progress.get("splits", {}).get(oid)
    if splits:
        return [(date.fromisoformat(a), date.fromisoformat(b)) for a, b in splits]
    return default_windows(start, end)


def _split(window: tuple[date, date]) -> list[tuple[date, date]]:
    """Split a window into calendar years, or a single year into two halves."""
    a, b = window
    if a.year != b.year:
        out = []
        for y in range(a.year, b.year + 1):
            out.append((max(a, date(y, 1, 1)), min(b, date(y, 12, 31))))
        return out
    mid = date(a.year, 7, 1)
    if a < mid <= b:
        return [(a, mid - timedelta(days=1)), (mid, b)]
    raise KapParseError(f"window {a}..{b} still capped after splitting; narrow it by hand")


# ----------------------------------------------------------------------------- network job
def run_filing_listing(
    settings: Settings,
    start: date = DEFAULT_START,
    end: date | None = None,
    max_requests: int | None = None,
    index_codes: tuple[str, ...] = ("XU100",),
    client: KapClient | None = None,
) -> dict[str, Any]:
    """Fetch (cached) the disclosure listings of every planned company; resumable and quota-aware."""
    end = end or date.today()
    plan = plan_companies(settings, index_codes)
    todo = plan[plan["oid_source"] != "missing"].drop_duplicates("mkk_member_oid")
    client = client or KapClient(
        PoliteSession(min_interval_s=2.5, timeout_s=30.0, retry_on_429=False), settings.paths.raw / "kap"
    )
    path = progress_path(settings)
    progress = _load_progress(path)
    progress.setdefault("splits", {})
    summary: dict[str, Any] = {
        "companies": int(len(todo)),
        "missing_oid": sorted(plan.loc[plan["oid_source"] == "missing", "ticker"]),
        "requested": 0,
        "cached_hits": 0,
        "capped_splits": 0,
        "failed": 0,
        "stopped_reason": "complete",
        "progress_path": str(path),
    }
    progress["runs"].append({"started_at": utc_now_iso(), "max_requests": max_requests})
    t0 = time.monotonic()
    try:
        for rec in todo.itertuples(index=False):
            oid = str(rec.mkk_member_oid)
            c_start = max(start, date(int(rec.year_from), 1, 1))
            c_end = min(end, date(int(rec.year_to), 12, 31)) if int(rec.year_to) < 9999 else end
            if c_start > c_end:
                continue
            key = f"{oid}_{c_start}_{c_end}"
            if key in progress["done"]:
                continue
            windows = _windows_for(oid, c_start, c_end, progress)
            pending = list(windows)
            for k in [k for k in progress["failed"] if k.startswith(f"{oid}_")]:
                progress["failed"].pop(k, None)
            while pending:
                a, b = pending.pop(0)
                cached_lang = next(
                    (
                        lg
                        for lg in ("en", "tr")
                        if (
                            client.raw_dir / "disclosures" / f"{oid}_{a.isoformat()}_{b.isoformat()}.{lg}.json.gz"
                        ).exists()
                    ),
                    None,
                )
                cached = cached_lang is not None
                lang = cached_lang or ("en" if progress["requests_made"] % 2 == 0 else "tr")  # two quota buckets
                if not cached:
                    if max_requests is not None and summary["requested"] >= max_requests:
                        summary["stopped_reason"] = "max_requests"
                        progress["splits"][oid] = [[x.isoformat(), y.isoformat()] for x, y in [(a, b), *pending]]
                        _save(path, progress, t0)
                        return _finish(summary, t0)
                    summary["requested"] += 1
                    progress["requests_made"] += 1
                else:
                    summary["cached_hits"] += 1
                try:
                    rows = client.list_disclosures(oid, a, b, lang=lang)
                except RateLimitedError as exc:
                    log.error("%s: %s — stopping, resume later", rec.ticker, exc)
                    summary["stopped_reason"] = "blocked"
                    progress["splits"][oid] = [[x.isoformat(), y.isoformat()] for x, y in [(a, b), *pending]]
                    _save(path, progress, t0)
                    return _finish(summary, t0)
                except KapParseError as exc:
                    log.warning("%s %s..%s: %s", rec.ticker, a, b, exc)
                    summary["failed"] += 1
                    progress["failed"][f"{oid}_{a}_{b}"] = str(exc)[:200]
                    continue
                if len(rows) >= DISCLOSURE_CAP:
                    parts = _split((a, b))
                    log.warning(
                        "%s %s..%s hit the %d-row cap; splitting into %d windows",
                        rec.ticker,
                        a,
                        b,
                        DISCLOSURE_CAP,
                        len(parts),
                    )
                    summary["capped_splits"] += 1
                    pending = parts + pending
                    continue
                log.info(
                    "%s %s..%s: %d rows (%d financial reports)%s",
                    rec.ticker,
                    a,
                    b,
                    len(rows),
                    len(financial_report_rows(rows)),
                    " [cache]" if cached else "",
                )
                _save(path, progress, t0)
            failed_windows = [k for k in progress["failed"] if k.startswith(f"{oid}_")]
            if failed_windows:
                log.warning(
                    "%s: %d window(s) failed; company left pending for the next run", rec.ticker, len(failed_windows)
                )
                progress["splits"].pop(oid, None)
                _save(path, progress, t0)
                continue
            progress["done"][key] = {"ticker": rec.ticker, "windows": len(windows), "at": utc_now_iso()}
            progress["splits"].pop(oid, None)
            _save(path, progress, t0)
    except KeyboardInterrupt:
        summary["stopped_reason"] = "interrupted"
        _save(path, progress, t0)
    if summary["stopped_reason"] == "complete" and summary["failed"]:
        summary["stopped_reason"] = "failures"
    return _finish(summary, t0)


def _save(path: Path, progress: dict[str, Any], t0: float) -> None:
    progress["updated_at"] = utc_now_iso()
    progress["seconds_spent"] = progress.get("seconds_spent", 0.0) + (time.monotonic() - t0)
    write_json(progress, path)


def _finish(summary: dict[str, Any], t0: float) -> dict[str, Any]:
    summary["seconds"] = round(time.monotonic() - t0, 1)
    log.info("filing listing: %s", summary)
    return summary


# ----------------------------------------------------------------------------- offline index build
def parse_listing_rows(rows: list[dict[str, Any]], oid: str, ticker: str, lang: str = "en") -> list[dict[str, Any]]:
    out = []
    for r in financial_report_rows(rows):
        ts = pd.to_datetime(str(r.get("publishDate", "")), format="%d.%m.%Y %H:%M:%S", errors="coerce")
        out.append(
            {
                "mkk_member_oid": oid,
                "ticker": ticker,
                "disclosure_index": int(r.get("disclosureIndex")),
                "publish_ts": ts,
                "fiscal_year": int(r.get("year")) if r.get("year") not in (None, "") else None,
                "period": int(r.get("period")) if r.get("period") not in (None, "") else None,
                "rule_type": str(r.get("ruleType", "")),
                "is_late": bool(r.get("isLate", False)),
                "stock_codes": str(r.get("stockCodes", "")),
                "subject": str(r.get("subject", "")),
                "lang": lang,
            }
        )
    return out


def build_filing_index(settings: Settings, index_codes: tuple[str, ...] = ("XU100",)) -> pd.DataFrame:
    """Read every cached listing of the planned companies and write ``filing_index.parquet``."""
    plan = plan_companies(settings, index_codes)
    raw_dir = settings.paths.raw / "kap" / "disclosures"
    records: list[dict[str, Any]] = []
    covered: dict[str, int] = {}
    for rec in plan.itertuples(index=False):
        if not rec.mkk_member_oid:
            continue
        files = sorted(raw_dir.glob(f"{rec.mkk_member_oid}_*.json.gz"))
        n = 0
        for f in files:
            lang = "tr" if f.name.endswith(".tr.json.gz") else "en"
            from bistmas.shared.store import read_raw_gzip

            try:
                rows = json.loads(read_raw_gzip(f))
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("%s: unreadable listing cache (%s)", f.name, exc)
                continue
            parsed = parse_listing_rows(rows if isinstance(rows, list) else [], rec.mkk_member_oid, rec.ticker, lang)
            records.extend(parsed)
            n += len(parsed)
        covered[rec.ticker] = n
    df = pd.DataFrame(records, columns=INDEX_COLUMNS)
    if len(df):
        df = (
            df.drop_duplicates(subset=["mkk_member_oid", "disclosure_index"])
            .sort_values(["ticker", "publish_ts"])
            .reset_index(drop=True)
        )
        df["publish_ts"] = pd.to_datetime(df["publish_ts"]).astype("datetime64[ns]")
    write_parquet(df, settings.paths.store / FILING_INDEX_FILE)
    meta = {
        "generated_at": utc_now_iso(),
        "n_reports": int(len(df)),
        "n_companies_with_reports": int((pd.Series(covered) > 0).sum()),
        "companies_without_listing": sorted(t for t, n in covered.items() if n == 0),
        "missing_oid": sorted(plan.loc[plan["oid_source"] == "missing", "ticker"]),
        "by_year": {str(k): int(v) for k, v in df["fiscal_year"].value_counts().sort_index().items()}
        if len(df)
        else {},
    }
    write_json(meta, settings.paths.store / FILING_META_FILE)
    log.info(
        "filing index: %d reports, %d companies; %d without listing, %d without oid",
        len(df),
        meta["n_companies_with_reports"],
        len(meta["companies_without_listing"]),
        len(meta["missing_oid"]),
    )
    return df


def load_filing_index(store_dir: Path) -> pd.DataFrame:
    return read_parquet(Path(store_dir) / FILING_INDEX_FILE)
