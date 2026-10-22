"""Checkpointed İş Yatırım price pull behind ``bistmas fetch prices``.

The job iterates *ticker x calendar year* through :class:`~bistmas.data.clients.isyatirim.IsYatirimClient`,
skips every ticker-year that is already in the raw layer, persists ``data/raw/isyatirim/progress.json``
after each request so an interrupted run resumes where it stopped, logs an ETA, stops cleanly on
``KeyboardInterrupt`` / rate limiting / WAF blocks, and can be run in slices with ``max_requests``.

Ticker plan (:func:`plan_tickers`): the union of every ticker in ``data/store/universe_membership.parquet``
(all indices) plus, with ``include_all_equities=True``, every current equity code of the security master
that has a KAP sector row (bond-only issuers are skipped). Renamed companies are served by the endpoint only
under their **new** code (AVISA -> AGESA, GUSGR -> TURSG, YAZIC -> AGHOL), so ``kind == "rename"`` rows of
``configs/ticker_aliases.csv`` are mapped old -> new before fetching and the mapping is recorded in the
progress file and the summary. Merged-away codes (TRKCM, UNYEC, KIPA, ...) keep their own history under the
old code and are fetched as such; codes the endpoint does not know (MRDIN, ITTFH) answer ``{"value": []}``,
which is cached once as a valid empty result.

Volume semantics (verified on the live THYAO 2018 and ASELS 2020 payloads, 2026-09-23): ``HG_HACIM`` and
``HGDG_HACIM`` are the **TL traded value** of the session, identical to each other, with
``DOLAR_HACIM = HACIM / DD_DEGER``. Hence ``traded_value_tl = volume_raw`` and the share count traded is
``volume_raw / vwap_raw``; see :func:`bistmas.data.clients.isyatirim.volume_semantics`.

Throughput (measured): ~0.4-12 s per one-year request plus the 1.5 s floor; :data:`EST_SECONDS_PER_REQUEST`
is the planning figure until the progress file has its own average.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from bistmas.data.clients.http import PoliteSession, RateLimitedError
from bistmas.data.clients.isyatirim import (
    FIRST_YEAR,
    SOURCE_NAME,
    IsYatirimClient,
    IsYatirimError,
    make_session,
    normalise_ticker,
)
from bistmas.data.security_master import load_security_master
from bistmas.data.universe.official import load_aliases, read_membership
from bistmas.shared.codes import EQUITY_CODE_RE
from bistmas.shared.config import CONFIG_DIR, Settings
from bistmas.shared.log import get_logger
from bistmas.shared.store import read_json, utc_now_iso, write_json

log = get_logger("data.prices_fetch")

PROGRESS_FILENAME = "progress.json"
PROGRESS_VERSION = 1
DEFAULT_START_YEAR = 2017
EST_SECONDS_PER_REQUEST = 11.0  # 8-12 s measured on one-year requests (research 2026-09-23) incl. spacing
MAX_CONSECUTIVE_FAILURES = 5  # abort before hammering a blocked endpoint
LOG_EVERY = 10
FATAL_HTTP_STATUS = frozenset({401, 403, 429})


def progress_path(settings: Settings) -> Path:
    return Path(settings.paths.raw) / SOURCE_NAME / PROGRESS_FILENAME


# ----------------------------------------------------------------------------- planning
def resolve_fetch_codes(
    tickers: Iterable[str], aliases: pd.DataFrame | None = None
) -> tuple[list[str], dict[str, str]]:
    """Map renamed codes to the code the endpoint serves and return ``(sorted codes, {old: new})``.

    Only ``kind == "rename"`` aliases are applied (same legal entity, history re-keyed under the new code);
    merger aliases are left alone because the merged-away code has its own price history. Codes that are
    not equity codes are dropped with a warning. The mapping only lists renames that were actually applied.
    """
    if aliases is None:
        aliases = load_aliases(CONFIG_DIR / "ticker_aliases.csv")
    renames = {
        str(row.old_ticker): str(row.new_ticker)
        for row in aliases.itertuples(index=False)
        if str(row.kind) == "rename" and row.old_ticker and row.new_ticker
    }
    codes: set[str] = set()
    applied: dict[str, str] = {}
    for raw in tickers:
        code = str(raw).strip().upper()
        if code.endswith(".E"):
            code = code[:-2]
        if not EQUITY_CODE_RE.fullmatch(code):
            log.warning("skipping %r: not an equity code", raw)
            continue
        if code in renames:
            applied[code] = renames[code]
            code = renames[code]
        codes.add(code)
    return sorted(codes), applied


def plan_tickers(
    settings: Settings, include_all_equities: bool = False, aliases: pd.DataFrame | None = None
) -> list[str]:
    """Codes to fetch: every membership ticker (all indices) [+ current equity codes], renames resolved."""
    store = Path(settings.paths.store)
    membership = read_membership(store)
    tickers: set[str] = set(membership["ticker"].astype(str))
    log.info(
        "plan: %d tickers from universe membership (%s)",
        len(tickers),
        ", ".join(sorted(membership["index_code"].unique())),
    )
    if include_all_equities:
        master = load_security_master(store)
        current = master[master["is_current"].astype(bool) & (master["note"].astype(str) != "no_kap_sector")]
        extra = set(current["ticker"].astype(str)) - tickers
        log.info("plan: +%d current equity codes from the security master", len(extra))
        tickers |= extra
    codes, applied = resolve_fetch_codes(tickers, aliases)
    if applied:
        log.info("plan: renames applied for fetching: %s", ", ".join(f"{k}->{v}" for k, v in sorted(applied.items())))
    return codes


def year_span(
    start_year: int = DEFAULT_START_YEAR, end_year: int | None = None, today: date | None = None
) -> list[int]:
    today = today or date.today()
    end_year = end_year or today.year
    if start_year < FIRST_YEAR or end_year > today.year or start_year > end_year:
        raise ValueError(f"year span {start_year}..{end_year} invalid (allowed {FIRST_YEAR}..{today.year})")
    return list(range(int(start_year), int(end_year) + 1))


def pending_pairs(client: IsYatirimClient, tickers: Sequence[str], years: Sequence[int]) -> list[tuple[str, int]]:
    """``(ticker, year)`` pairs not yet in the raw layer, in ticker-major order."""
    return [(t, y) for t in tickers for y in years if not client.is_cached(t, y)]


def estimate(
    settings: Settings,
    tickers: Sequence[str],
    start_year: int = DEFAULT_START_YEAR,
    end_year: int | None = None,
    client: IsYatirimClient | None = None,
) -> tuple[int, float]:
    """``(n_requests, seconds)`` still needed for ``tickers`` over the year span (cached years excluded).

    The per-request figure is the running average stored in ``progress.json`` when it exists, otherwise
    :data:`EST_SECONDS_PER_REQUEST`.
    """
    client = client or IsYatirimClient(make_session(), settings.paths.raw)
    n = len(pending_pairs(client, tickers, year_span(start_year, end_year)))
    per_request = EST_SECONDS_PER_REQUEST
    progress = load_progress(progress_path(settings))
    if progress and progress.get("requests_made", 0) >= 5 and progress.get("seconds_spent", 0) > 0:
        per_request = max(1.5, progress["seconds_spent"] / progress["requests_made"])
    return n, n * per_request


# ----------------------------------------------------------------------------- progress file
def new_progress() -> dict[str, Any]:
    return {
        "version": PROGRESS_VERSION,
        "started_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "aliases": {},
        "done": {},
        "failed": {},
        "requests_made": 0,
        "seconds_spent": 0.0,
        "runs": [],
    }


def load_progress(path: Path) -> dict[str, Any]:
    """Read ``progress.json``; an unreadable or absent file yields an empty dict (a fresh job)."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = read_json(path)
    except (OSError, ValueError) as exc:
        log.warning("progress file %s unreadable (%s); starting fresh", path, exc)
        return {}
    if not isinstance(data, dict) or data.get("version") != PROGRESS_VERSION:
        log.warning("progress file %s has an unexpected layout; starting fresh", path)
        return {}
    return data


def save_progress(path: Path, progress: dict[str, Any]) -> None:
    progress["updated_at"] = utc_now_iso()
    write_json(progress, Path(path))


def _key(ticker: str, year: int) -> str:
    return f"{ticker}/{year}"


def _fmt_eta(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m" if h else f"{m:d}m{s:02d}s"


# ----------------------------------------------------------------------------- the job
def run_fetch(
    settings: Settings,
    tickers: Sequence[str],
    start_year: int = DEFAULT_START_YEAR,
    end_year: int | None = None,
    max_requests: int | None = None,
    dry_run: bool = False,
    force: bool = False,
    client: IsYatirimClient | None = None,
    session: PoliteSession | None = None,
) -> dict[str, Any]:
    """Fetch every missing ticker-year for ``tickers`` (renames resolved) and return a summary.

    * cached years are skipped (``force=True`` re-fetches everything in the plan, used for the periodic
      full re-pull because adjusted series are restated);
    * progress is written to ``data/raw/isyatirim/progress.json`` after every request;
    * ``max_requests`` counts network requests (not cache hits) so the job can run in slices;
    * ``dry_run`` only plans and estimates;
    * ``KeyboardInterrupt`` saves progress and returns (``stopped_reason == "interrupted"``); 401/403/429
      or :data:`MAX_CONSECUTIVE_FAILURES` failures in a row stop the job (``"blocked"`` / ``"failures"``).

    Summary keys: ``tickers, aliases, years, planned, cached, pending, requested, fetched, empty, failed,
    stopped_reason, seconds, eta_seconds_remaining, progress_path, estimate_seconds``.
    """
    codes, aliases = resolve_fetch_codes(tickers)
    years = year_span(start_year, end_year)
    client = client or IsYatirimClient(session or make_session(), settings.paths.raw)
    path = progress_path(settings)
    progress = load_progress(path) or new_progress()
    progress["aliases"].update(aliases)

    plan = [(t, y) for t in codes for y in years]
    pending = plan if force else [(t, y) for t, y in plan if not client.is_cached(t, y)]
    n_req_est, sec_est = estimate(settings, codes, start_year, end_year, client=client)
    summary: dict[str, Any] = {
        "tickers": len(codes),
        "aliases": dict(sorted(aliases.items())),
        "years": [years[0], years[-1]],
        "planned": len(plan),
        "cached": len(plan) - len(pending),
        "pending": len(pending),
        "requested": 0,
        "fetched": 0,
        "empty": 0,
        "failed": 0,
        "stopped_reason": "",
        "seconds": 0.0,
        "eta_seconds_remaining": sec_est,
        "estimate_seconds": sec_est,
        "progress_path": str(path),
        "dry_run": dry_run,
    }
    log.info(
        "prices fetch: %d tickers x %d years = %d ticker-years, %d cached, %d pending (~%s)%s",
        len(codes),
        len(years),
        len(plan),
        summary["cached"],
        len(pending),
        _fmt_eta(sec_est),
        " [dry run]" if dry_run else "",
    )
    if dry_run or not pending:
        summary["stopped_reason"] = "dry_run" if dry_run else "complete"
        return summary

    run_started = utc_now_iso()
    progress["runs"].append({"started_at": run_started, "pending": len(pending), "max_requests": max_requests})
    t_run = time.monotonic()
    consecutive_failures = 0
    todo = len(pending)
    try:
        for i, (ticker, year) in enumerate(pending, start=1):
            key = _key(ticker, year)
            if max_requests is not None and summary["requested"] >= max_requests:
                summary["stopped_reason"] = "max_requests"
                break
            summary["requested"] += 1
            t0 = time.monotonic()
            try:
                frame = client.fetch_year(ticker, year, force=force)
            except RateLimitedError as exc:
                _record_failure(progress, key, exc)
                summary["failed"] += 1
                summary["stopped_reason"] = "blocked"
                log.error("%s: rate limited (%s); stopping", key, exc)
                break
            except requests.HTTPError as exc:
                _record_failure(progress, key, exc)
                summary["failed"] += 1
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in FATAL_HTTP_STATUS:
                    summary["stopped_reason"] = "blocked"
                    log.error("%s: HTTP %s; stopping (blocked?)", key, status)
                    break
                consecutive_failures += 1
            except (requests.RequestException, IsYatirimError, OSError) as exc:
                _record_failure(progress, key, exc)
                summary["failed"] += 1
                consecutive_failures += 1
                log.warning("%s: %s: %s", key, exc.__class__.__name__, exc)
            else:
                elapsed = time.monotonic() - t0
                consecutive_failures = 0
                progress["done"][key] = {"rows": int(len(frame)), "empty": len(frame) == 0, "fetched_at": utc_now_iso()}
                progress["failed"].pop(key, None)
                progress["requests_made"] += 1
                progress["seconds_spent"] += elapsed
                summary["fetched"] += 1
                summary["empty"] += int(len(frame) == 0)
            finally:
                save_progress(path, progress)
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                summary["stopped_reason"] = "failures"
                log.error("%d consecutive failures; stopping so the endpoint is not hammered", consecutive_failures)
                break
            if i % LOG_EVERY == 0 or i == todo:
                spent = time.monotonic() - t_run
                per = spent / max(1, summary["requested"])
                remaining = todo - i
                log.info(
                    "progress %d/%d (%s %d) %.1fs/req, %d failed, ETA %s",
                    i,
                    todo,
                    ticker,
                    year,
                    per,
                    summary["failed"],
                    _fmt_eta(per * remaining),
                )
    except KeyboardInterrupt:
        summary["stopped_reason"] = "interrupted"
        log.warning("interrupted; progress saved to %s (re-run to resume)", path)
    finally:
        summary["seconds"] = round(time.monotonic() - t_run, 1)
        if not summary["stopped_reason"]:
            summary["stopped_reason"] = "complete" if summary["failed"] == 0 else "complete_with_failures"
        progress["runs"][-1].update(
            {
                "finished_at": utc_now_iso(),
                "requested": summary["requested"],
                "fetched": summary["fetched"],
                "failed": summary["failed"],
                "stopped_reason": summary["stopped_reason"],
            }
        )
        save_progress(path, progress)

    remaining = len(pending_pairs(client, codes, years))
    per = summary["seconds"] / max(1, summary["requested"]) if summary["requested"] else EST_SECONDS_PER_REQUEST
    summary["eta_seconds_remaining"] = remaining * per
    summary["remaining"] = remaining
    log.info(
        "prices fetch stopped (%s): %d requested, %d fetched (%d empty), %d failed in %.0fs; "
        "%d ticker-years remain (~%s)",
        summary["stopped_reason"],
        summary["requested"],
        summary["fetched"],
        summary["empty"],
        summary["failed"],
        summary["seconds"],
        remaining,
        _fmt_eta(summary["eta_seconds_remaining"]),
    )
    return summary


def _record_failure(progress: dict[str, Any], key: str, exc: BaseException) -> None:
    entry = progress["failed"].get(key, {"attempts": 0})
    entry["attempts"] = int(entry.get("attempts", 0)) + 1
    entry["error"] = f"{exc.__class__.__name__}: {str(exc)[:200]}"
    entry["last_at"] = utc_now_iso()
    progress["failed"][key] = entry


def fetch_prices(
    settings: Settings,
    include_all_equities: bool = False,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int | None = None,
    max_requests: int | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """CLI entry point: plan the tickers from the store tables and run the checkpointed pull."""
    tickers = plan_tickers(settings, include_all_equities=include_all_equities)
    return run_fetch(
        settings,
        tickers,
        start_year=start_year,
        end_year=end_year,
        max_requests=max_requests,
        dry_run=dry_run,
        force=force,
    )


__all__ = [
    "DEFAULT_START_YEAR",
    "EST_SECONDS_PER_REQUEST",
    "estimate",
    "fetch_prices",
    "load_progress",
    "normalise_ticker",
    "pending_pairs",
    "plan_tickers",
    "progress_path",
    "resolve_fetch_codes",
    "run_fetch",
    "save_progress",
    "year_span",
]
