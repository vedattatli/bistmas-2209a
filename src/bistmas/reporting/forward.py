"""Monthly forward (paper-trading) decision — the only uncontaminated test of the jury.

Semantics (red-team pitfalls #24 and the judges' forward-run notes):
* idempotent per month: the decision for ``as_of`` (last session of a COMPLETED month) is written once to
  ``data/runs/forward/<YYYY-MM>/decision.json`` and never modified; ``as_of`` is stored and re-checked;
* dry runs (``dry_run=True``) never touch that state: they write under ``data/runs/forward_dryrun/`` and use
  their own jury run, so a later real run is unaffected;
* sealed before execution: the planned execution session is the FIRST session after the actual run
  time, never a session that has already traded (looked up in a holiday-aware calendar that extends the
  data-derived one past the last stored session, see :func:`bistmas.shared.calendar.with_future_sessions`);
  ``late_run`` records when the run happened after the
  first session of the month;
* provenance: code commit, config hash, store vintages, model digest, prompt hashes and the PIT check are
  embedded (certificate) and the file's sha256 goes into the e-mail;
* catch-up: months without a decision file since the last one are produced in order, each marked as a
  late run (their execution session is still "first session after now").
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from bistmas.backtest.run import load_inputs
from bistmas.data.calendar_build import load_calendar
from bistmas.data.security_master import load_security_master
from bistmas.jury.run import run_jury
from bistmas.reporting.certificate import build_certificate
from bistmas.reporting.email import send_report
from bistmas.reporting.render import render_decision_markdown
from bistmas.scoring.screen import screen_one
from bistmas.shared.calendar import TradingCalendar, with_future_sessions
from bistmas.shared.config import Settings
from bistmas.shared.hashing import sha256_text
from bistmas.shared.log import get_logger
from bistmas.shared.store import read_parquet, utc_now_iso, write_json

log = get_logger("reporting.forward")


def forward_dir(settings: Settings) -> Path:
    return settings.paths.runs / "forward"


def dry_run_dir(settings: Settings) -> Path:
    return settings.paths.runs / "forward_dryrun"


def future_calendar(settings: Settings) -> tuple[TradingCalendar, TradingCalendar]:
    """(data-derived calendar, the same extended with holiday-aware weekday sessions after the last stored one)."""
    cal = load_calendar(settings.paths.store)
    return cal, with_future_sessions(cal)


def completed_month_ends(settings: Settings, today: date | None = None, now: datetime | None = None) -> list[date]:
    """Month-end sessions whose calendar month is over (strictly before ``today``), oldest first.

    The month end is the last session of the month in the extended calendar, so a store that stops in the
    middle of a month never yields that month; the caller checks the store actually reaches the date.
    """
    _cal, fut = future_calendar(settings)
    today = today or (now.date() if now else date.today())
    ends = fut.month_ends(date(today.year - 1, 1, 1), today)
    return [d for d in ends if d < today or (now is not None and d == today and now.time() >= time(18, 0))]


def latest_decision_date(settings: Settings, today: date | None = None) -> date:
    """Last session of the most recent completed month; raises when the store does not reach it yet."""
    cal, _fut = future_calendar(settings)
    ends = completed_month_ends(settings, today)
    if not ends:
        raise RuntimeError("no completed month yet")
    target = ends[-1]
    if not cal.is_session(target):
        raise RuntimeError(
            f"store ends at {cal.sessions[-1]} but the latest completed month ends {target}: refresh the prices first"
        )
    return target


def pending_months(settings: Settings, today: date | None = None) -> list[date]:
    """Completed-month decision dates still to produce, oldest first (catch-up), covered by the store.

    Starts after the latest sealed decision; with none sealed yet only the latest completed month is returned
    (the paper-trading record starts there, never back-filled with months that were not decided in real time).
    """
    cal, _fut = future_calendar(settings)
    ends = completed_month_ends(settings, today)[-6:]
    sealed = sorted(p.parent.name for p in forward_dir(settings).glob("*/decision.json"))
    if sealed:
        ends = [d for d in ends if d.strftime("%Y-%m") > sealed[-1]]
    else:
        ends = ends[-1:]
    out = []
    for d in ends:
        if not cal.is_session(d):
            log.warning("catch-up: store ends %s, month end %s not covered; refresh prices", cal.sessions[-1], d)
            continue
        out.append(d)
    return out


def planned_execution_session(fut: TradingCalendar, now: datetime) -> date:
    """First session that has not traded yet: today if still before 10:00, otherwise the next session."""
    if fut.is_session(now.date()) and now.time().hour >= 10:
        return fut.next_session(now.date(), strict=True)
    return fut.next_session(now.date(), strict=False)


def is_completed_month_end(fut: TradingCalendar, as_of: date, now: datetime) -> bool:
    """``as_of`` is the last session of its calendar month and that month is over (decision cut-off passed)."""
    first = date(as_of.year, as_of.month, 1)
    ends = fut.month_ends(first, date(as_of.year + (as_of.month == 12), as_of.month % 12 + 1, 1) - timedelta(days=1))
    if not ends or ends[-1] != as_of:
        return False
    return as_of < now.date() or (as_of == now.date() and now.time() >= time(18, 0))


def run_forward(
    settings: Settings,
    as_of: date,
    dry_run: bool = False,
    now: datetime | None = None,
    jury_run_name: str | None = None,
    send_mail: bool = True,
) -> dict[str, Any]:
    """Produce (once) the sealed decision for ``as_of`` and e-mail the report.

    ``as_of`` must be the last session of a month that is over and covered by the store. A dry run writes
    under ``forward_dryrun/`` with its own jury run and never seals, logs or blocks the real decision.
    """
    now = now or datetime.now()
    base = dry_run_dir(settings) if dry_run else forward_dir(settings)
    month_dir = base / pd.Timestamp(as_of).strftime("%Y-%m")
    decision_path = month_dir / "decision.json"
    if not dry_run and decision_path.exists():
        sealed = json.loads(decision_path.read_text(encoding="utf-8"))
        if sealed.get("as_of") != str(as_of):
            raise RuntimeError(f"{decision_path} is sealed for {sealed.get('as_of')}, not {as_of}")
        log.info("decision for %s already sealed at %s", as_of, decision_path)
        return sealed
    cal, fut = future_calendar(settings)
    if not is_completed_month_end(fut, as_of, now):
        raise ValueError(f"{as_of} is not the last session of a completed month (now {now:%Y-%m-%d %H:%M})")
    if not cal.is_session(as_of):
        raise RuntimeError(f"store ends at {cal.sessions[-1]}, before decision date {as_of}: refresh the prices first")
    planned_exec = planned_execution_session(fut, now)
    first_session_of_month = fut.next_session(as_of, strict=True)
    late_run = planned_exec > first_session_of_month
    inputs, _prices, provenance = load_inputs(settings)
    screen = screen_one(as_of, inputs, settings)
    if screen.empty:
        raise RuntimeError(f"empty screen for {as_of}: rebuild the store first")
    screen["as_of"] = pd.Timestamp(as_of)
    master = load_security_master(settings.paths.store)
    stamp = pd.Timestamp(as_of).strftime("%Y%m%d")
    run_name = jury_run_name or (f"forward_dryrun_{stamp}" if dry_run else f"forward_{stamp}")
    jury_summary = run_jury(settings, run_name, screen, master, decisions=[pd.Timestamp(as_of)], dry_run=dry_run)
    if not dry_run and (
        jury_summary.get("stopped_reason") != "complete" or (jury_summary.get("abstain_rate") or 0) >= 1
    ):
        raise RuntimeError(
            f"jury run {run_name} not usable (stopped_reason={jury_summary.get('stopped_reason')}, "
            f"abstain_rate={jury_summary.get('abstain_rate')}): nothing sealed; re-run to resume"
        )
    ts = pd.Timestamp(as_of)
    cons_path = settings.paths.runs / run_name / "jury_consensus.parquet"
    cons = read_parquet(cons_path) if cons_path.exists() else pd.DataFrame()
    scores_path = settings.paths.runs / run_name / "jury_scores.parquet"
    scores = read_parquet(scores_path) if scores_path.exists() else pd.DataFrame()
    if len(cons):
        cons = cons[pd.to_datetime(cons["as_of"]) == ts]
    if len(scores):
        scores = scores[pd.to_datetime(scores["as_of"]) == ts]
    top_rows = []
    if len(cons):
        c = cons.dropna(subset=["rank_J"]).sort_values("rank_J").head(settings.backtest.headline_top_n)
        sec = screen.set_index("ticker")["sector_group"].to_dict()
        top_rows = [
            {
                "rank_J": float(r.rank_J),
                "ticker": r.ticker,
                "J": float(r.J),
                "n_votes": int(r.n_votes),
                "composite_rank": float(r.composite_rank) if pd.notna(r.composite_rank) else float("nan"),
                "sector_group": sec.get(r.ticker, ""),
            }
            for r in c.itertuples(index=False)
        ]
    screen_top = screen.dropna(subset=["rank"]).sort_values("rank").head(settings.scoring.top_n_screen)
    screen_rows = screen_top[
        ["rank", "ticker", "composite", "S_fundamental", "S_valuation", "S_momentum_risk", "sector_group"]
    ].to_dict("records")
    rationales = []
    if len(scores):
        picked = {r["ticker"] for r in top_rows}
        for r in scores[scores["ticker"].isin(picked) & ~scores["abstain"].astype(bool)].itertuples(index=False):
            rationales.append(
                {"ticker": r.ticker, "agent": r.agent, "score": float(r.score), "rationale": str(r.rationale)}
            )
    certificate = build_certificate(settings, pd.Timestamp(as_of), screen, jury_summary, {"provenance": provenance})
    decision: dict[str, Any] = {
        "run_id": run_name,
        "as_of": str(as_of),
        "run_time": now.isoformat(timespec="seconds"),
        "first_session_of_month": str(first_session_of_month),
        "planned_execution_session": str(planned_exec),
        "late_run": bool(late_run),
        "dry_run": dry_run,
        "top": top_rows,
        "screen_top": screen_rows,
        "rationales": rationales,
        "jury": {k: v for k, v in jury_summary.items() if k != "prompt_hashes"},
        "certificate": certificate,
    }
    body = json.dumps(decision, ensure_ascii=False, sort_keys=True, default=str)
    decision["sha256"] = sha256_text(body)
    month_dir.mkdir(parents=True, exist_ok=True)
    write_json(decision, decision_path)
    md = render_decision_markdown(decision)
    (month_dir / "report.md").write_text(md, encoding="utf-8")
    if not dry_run:
        with open(forward_dir(settings) / "log.jsonl", "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "as_of": str(as_of),
                        "sealed_at": utc_now_iso(),
                        "sha256": decision["sha256"],
                        "planned_execution_session": str(planned_exec),
                        "late_run": late_run,
                        "top": [r["ticker"] for r in top_rows],
                    }
                )
                + "\n"
            )
    status = "skipped"
    if send_mail:
        status = send_report(
            f"bistmas decision {as_of} (exec {planned_exec})",
            md,
            settings.forward.email_to,
            settings.forward.smtp_host,
            settings.forward.smtp_port,
            dry_run=dry_run or not settings.forward.enabled,
            attachments=[decision_path],
        )
    decision["email_status"] = status
    log.info("forward decision %s sealed (%s); e-mail: %s", as_of, decision["sha256"][:12], status)
    return decision
