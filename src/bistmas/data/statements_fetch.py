"""Fetch every financial-report body listed in ``filing_index.parquet`` and build the statements store.

* :func:`run_statement_fetch` — GET ``attachment-detail`` for each disclosure index (cached, paced,
  checkpointed in ``data/raw/kap/statements_progress.json``; ``max_requests`` slices the job).
* :func:`build_statements_store` — offline: parse every cached body with :mod:`statements_kap`, drop
  dimensional blocks (no period context), attach the point-in-time ``availability_date`` from the
  publication timestamp and the trading calendar, and write ``statements_long.parquet`` +
  ``statements_reports.parquet`` (one row per report) + ``statements_meta.json``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from bistmas.data.calendar_build import load_calendar
from bistmas.data.clients.http import PoliteSession, RateLimitedError
from bistmas.data.clients.kap import KapClient, KapParseError
from bistmas.data.filings import load_filing_index
from bistmas.data.pit import availability_from_publish
from bistmas.data.statements_kap import meta_to_dict, parse_report
from bistmas.shared.config import Settings
from bistmas.shared.log import get_logger
from bistmas.shared.store import read_json, read_parquet, utc_now_iso, write_json, write_parquet

log = get_logger("data.statements_fetch")

PROGRESS_FILE = "statements_progress.json"
LONG_FILE = "statements_long.parquet"
REPORTS_FILE = "statements_reports.parquet"
META_FILE = "statements_meta.json"
SKIP_STATEMENTS = frozenset({"equity_changes"})
LONG_COLUMNS = [
    "mkk_member_oid",
    "ticker",
    "disclosure_index",
    "fiscal_year",
    "period",
    "consolidated",
    "statement",
    "concept",
    "concept_role",
    "label_en",
    "col_index",
    "context_label",
    "context_slice",
    "period_start",
    "period_end",
    "value_tl",
    "currency",
    "publish_ts",
    "availability_date",
]


def progress_path(settings: Settings) -> Path:
    return settings.paths.raw / "kap" / PROGRESS_FILE


def run_statement_fetch(
    settings: Settings,
    max_requests: int | None = None,
    client: KapClient | None = None,
    tickers: list[str] | None = None,
) -> dict[str, Any]:
    """Download the report bodies for the filing index (cached; ~2 s pacing; resumable)."""
    index = load_filing_index(settings.paths.store)
    if tickers:
        index = index[index["ticker"].isin([t.upper() for t in tickers])]
    client = client or KapClient(
        PoliteSession(min_interval_s=2.0, timeout_s=60.0, retry_on_429=False), settings.paths.raw / "kap"
    )
    path = progress_path(settings)
    progress = (
        read_json(path) if path.exists() else {"version": 1, "done": {}, "failed": {}, "requests_made": 0, "runs": []}
    )
    progress["runs"].append({"started_at": utc_now_iso(), "max_requests": max_requests})
    summary: dict[str, Any] = {
        "reports": int(len(index)),
        "requested": 0,
        "cached_hits": 0,
        "failed": 0,
        "stopped_reason": "complete",
    }
    t0 = time.monotonic()
    try:
        for rec in index.sort_values(["ticker", "publish_ts"]).itertuples(index=False):
            idx = int(rec.disclosure_index)
            raw = client.raw_dir / "attachment" / f"{idx}.json.gz"
            if raw.exists():
                summary["cached_hits"] += 1
                progress["done"].setdefault(str(idx), {"ticker": rec.ticker, "at": "cache"})
                continue
            if max_requests is not None and summary["requested"] >= max_requests:
                summary["stopped_reason"] = "max_requests"
                break
            summary["requested"] += 1
            progress["requests_made"] += 1
            try:
                doc = client.fetch_attachment_detail(idx)
            except RateLimitedError as exc:
                log.error("%s %s: %s — stopping", rec.ticker, idx, exc)
                summary["stopped_reason"] = "blocked"
                break
            except KapParseError as exc:
                log.warning("%s %s: %s", rec.ticker, idx, exc)
                summary["failed"] += 1
                progress["failed"][str(idx)] = str(exc)[:200]
                continue
            n_blocks = len(doc.get("disclosureBody") or [])
            progress["done"][str(idx)] = {"ticker": rec.ticker, "blocks": n_blocks, "at": utc_now_iso()}
            if summary["requested"] % 10 == 0:
                progress["updated_at"] = utc_now_iso()
                write_json(progress, path)
                log.info(
                    "statements fetch: %d requested, %d cached, %d failed",
                    summary["requested"],
                    summary["cached_hits"],
                    summary["failed"],
                )
    except KeyboardInterrupt:
        summary["stopped_reason"] = "interrupted"
    progress["updated_at"] = utc_now_iso()
    progress["seconds_spent"] = progress.get("seconds_spent", 0.0) + (time.monotonic() - t0)
    write_json(progress, path)
    summary["seconds"] = round(time.monotonic() - t0, 1)
    summary["progress_path"] = str(path)
    log.info("statements fetch: %s", summary)
    return summary


def build_statements_store(settings: Settings, tickers: list[str] | None = None) -> dict[str, Any]:
    """Parse every cached report body listed in the filing index and write the statements store."""
    index = load_filing_index(settings.paths.store)
    if tickers:
        index = index[index["ticker"].isin([t.upper() for t in tickers])]
    calendar = load_calendar(settings.paths.store)
    client = KapClient(
        PoliteSession(min_interval_s=2.0, timeout_s=60.0, retry_on_429=False), settings.paths.raw / "kap"
    )
    frames: list[pd.DataFrame] = []
    reports: list[dict[str, Any]] = []
    missing: list[int] = []
    unparsed: list[int] = []
    for rec in index.itertuples(index=False):
        idx = int(rec.disclosure_index)
        raw = client.raw_dir / "attachment" / f"{idx}.json.gz"
        if not raw.exists():
            missing.append(idx)
            continue
        try:
            doc = client.fetch_attachment_detail(idx)  # served from cache
            meta, df = parse_report(doc)
        except Exception as exc:  # noqa: BLE001
            log.warning("report %s failed to parse: %s", idx, exc)
            unparsed.append(idx)
            continue
        publish_ts = meta.publish_ts if not pd.isna(meta.publish_ts) else rec.publish_ts
        try:
            avail = availability_from_publish(publish_ts, calendar)
        except ValueError:
            avail = None
        info = meta_to_dict(meta)
        info.update({"ticker_index": rec.ticker, "availability_date": avail, "n_rows": int(len(df))})
        reports.append(info)
        if df.empty:
            continue
        df = df[~df["statement"].isin(SKIP_STATEMENTS) & df["period_end"].notna()].copy()
        df["mkk_member_oid"] = meta.mkk_member_oid or rec.mkk_member_oid
        df["ticker"] = rec.ticker
        df["disclosure_index"] = idx
        df["fiscal_year"] = meta.fiscal_year if meta.fiscal_year is not None else rec.fiscal_year
        df["period"] = meta.period if meta.period is not None else rec.period
        df["consolidated"] = meta.consolidated
        df["currency"] = meta.currency
        df["publish_ts"] = pd.Timestamp(publish_ts)
        df["availability_date"] = pd.Timestamp(avail) if avail else pd.NaT
        frames.append(df[LONG_COLUMNS])
    long_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=LONG_COLUMNS)
    for col in ("publish_ts", "availability_date", "period_start", "period_end"):
        long_df[col] = pd.to_datetime(long_df[col]).astype("datetime64[ns]")
    write_parquet(long_df, settings.paths.store / LONG_FILE)
    reports_df = pd.DataFrame(reports)
    if len(reports_df):
        reports_df["publish_ts"] = pd.to_datetime(reports_df["publish_ts"]).astype("datetime64[ns]")
        reports_df["availability_date"] = pd.to_datetime(reports_df["availability_date"]).astype("datetime64[ns]")
    write_parquet(reports_df, settings.paths.store / REPORTS_FILE)
    meta = {
        "generated_at": utc_now_iso(),
        "reports_in_index": int(len(index)),
        "reports_parsed": int(len(reports)),
        "reports_missing_body": len(missing),
        "reports_unparsed": unparsed[:50],
        "rows": int(len(long_df)),
        "tickers": int(long_df["ticker"].nunique()) if len(long_df) else 0,
        "by_statement": long_df["statement"].value_counts().to_dict() if len(long_df) else {},
        "consolidated_share": float(reports_df["consolidated"].fillna(False).mean()) if len(reports_df) else None,
        "currencies": reports_df["currency"].value_counts().to_dict() if len(reports_df) else {},
    }
    write_json(meta, settings.paths.store / META_FILE)
    log.info("statements store: %s", {k: v for k, v in meta.items() if k != "reports_unparsed"})
    return meta


def load_statements(store_dir: Path) -> pd.DataFrame:
    return read_parquet(Path(store_dir) / LONG_FILE)


def load_reports(store_dir: Path) -> pd.DataFrame:
    return read_parquet(Path(store_dir) / REPORTS_FILE)
