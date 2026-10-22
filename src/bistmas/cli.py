"""Command-line entry point: `bistmas <command>`.

Every command is idempotent and resumable; heavy fetches persist each response before moving on.
Commands other than `doctor` are filled in milestone by milestone (see the plan).
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from datetime import date
from typing import Any

import pandas as pd

from bistmas import __version__
from bistmas.shared.config import load_settings
from bistmas.shared.log import get_logger

log = get_logger("cli")

PACKAGES = [
    "pandas",
    "numpy",
    "pyarrow",
    "duckdb",
    "requests",
    "pydantic",
    "ollama",
    "scipy",
    "statsmodels",
    "pypdf",
    "lxml",
]

SMOKE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "canary": {"type": "string"},
        "sum": {"type": "integer"},
        "ok": {"type": "boolean"},
    },
    "required": ["canary", "sum", "ok"],
}


def cmd_doctor(args: argparse.Namespace) -> int:
    settings = load_settings()
    failures = 0

    def report(name: str, status: str, detail: str = "") -> None:
        print(f"{status:7s} {name:28s} {detail}")

    py = platform.python_version()
    ok = sys.version_info >= (3, 12)
    report("python", "OK" if ok else "FAIL", py)
    failures += not ok

    for pkg in PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            report(f"package {pkg}", "OK", getattr(mod, "__version__", ""))
        except Exception as exc:  # noqa: BLE001
            report(f"package {pkg}", "FAIL", str(exc))
            failures += 1

    report("config", "OK", f"root={settings.root}")
    for label, path in (("raw", settings.paths.raw), ("store", settings.paths.store), ("runs", settings.paths.runs)):
        report(f"path {label}", "OK" if path.exists() else "FAIL", str(path))
        failures += not path.exists()

    from bistmas.jury.ollama_client import OllamaClient

    client = OllamaClient(settings.jury.host, settings.jury.model, settings.jury.sampling_options())
    if not client.ping():
        report("ollama server", "FAIL", f"not reachable at {settings.jury.host} (brew services start ollama)")
        return failures + 1
    report("ollama server", "OK", settings.jury.host)

    digest = client.model_digest()
    if digest is None:
        report("ollama model", "FAIL", f"{settings.jury.model} not pulled (ollama pull {settings.jury.model})")
        return failures + 1
    pinned = settings.jury.model_digest
    if not pinned:
        report(
            "ollama model",
            "WARN",
            f"{settings.jury.model} digest {digest} — pin it in configs/base.toml [jury].model_digest",
        )
    elif pinned != digest:
        report("ollama model", "FAIL", f"digest {digest} != pinned {pinned}")
        failures += 1
    else:
        report("ollama model", "OK", f"{settings.jury.model} @ {digest[:12]}")

    if not args.skip_smoke:
        res = client.chat_json(
            system=(
                "You are a JSON-only responder. The canary word is 'bistmas-7f3a'. "
                "Always echo it in the 'canary' field."
            ),
            user="Compute 17 + 25 and put the result in 'sum'. Set 'ok' to true.",
            schema=SMOKE_SCHEMA,
        )
        if res.ok and res.parsed and res.parsed.get("canary") == "bistmas-7f3a" and res.parsed.get("sum") == 42:
            report(
                "ollama smoke json",
                "OK",
                f"prompt_tokens={res.prompt_eval_count} out_tokens={res.eval_count} {res.duration_ms:.0f} ms",
            )
        else:
            report("ollama smoke json", "FAIL", res.error or f"unexpected: {res.content[:120]}")
            failures += 1
        report(
            "ollama logprobs",
            "OK" if client.supports_logprobs() else "INFO",
            "available (LAP audit possible)"
            if client.supports_logprobs()
            else "not exposed (LAP audit will be skipped)",
        )
    return 1 if failures else 0


def cmd_fetch(args: argparse.Namespace) -> int:
    settings = load_settings()
    if args.target == "universe":
        from bistmas.data.universe.build import fetch_universe_raw

        summary = fetch_universe_raw(settings, force=args.force)
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        return 0
    if args.target == "prices":
        from bistmas.data.prices_fetch import fetch_prices

        summary = fetch_prices(
            settings,
            include_all_equities=args.all_equities,
            start_year=args.start_year,
            end_year=args.end_year,
            max_requests=args.max_requests,
            dry_run=args.dry_run,
            force=args.force,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        return 0 if summary.get("failed", 0) == 0 else 1
    if args.target == "macro":
        from bistmas.data.macro import build_macro_store, calendar_dates_from_store

        frame = build_macro_store(settings, calendar_dates_from_store(settings.paths.store), force=args.force)
        print(json.dumps({"rows": int(len(frame)), "columns": list(frame.columns)}, indent=2))
        return 0
    if args.target == "statements":
        from bistmas.data.filings import build_filing_index, run_filing_listing
        from bistmas.data.statements_fetch import run_statement_fetch

        listing = run_filing_listing(settings, max_requests=args.max_requests)
        print(json.dumps({"listing": listing}, indent=2, ensure_ascii=False, default=str))
        if listing["stopped_reason"] not in ("complete",):
            print(f"listing stopped ({listing['stopped_reason']}); re-run later to resume")
            return 3
        index = build_filing_index(settings)
        from bistmas.data.clients.http import PoliteSession
        from bistmas.data.filings import plan_companies
        from bistmas.data.statements_isyatirim import build_fallback_wide
        from bistmas.shared.store import write_parquet

        plan = plan_companies(settings)
        missing = plan[plan["oid_source"] == "missing"]
        if len(missing):
            fb = build_fallback_wide(PoliteSession(min_interval_s=2.0, timeout_s=60.0), settings.paths.raw, missing)
            if len(fb):
                write_parquet(fb, settings.paths.store / "statements_fallback_wide.parquet")
        bodies = run_statement_fetch(settings, max_requests=args.max_requests)
        print(
            json.dumps(
                {"filing_index_rows": int(len(index)), "bodies": bodies}, indent=2, ensure_ascii=False, default=str
            )
        )
        return 0 if bodies["stopped_reason"] == "complete" else 3
    print(f"fetch {args.target} is implemented in a later milestone.")
    return 2


def cmd_build(args: argparse.Namespace) -> int:
    settings = load_settings()
    if args.target in ("universe", "all"):
        from bistmas.data.universe.build import build_universe

        summary = build_universe(settings, end=args.end)
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        if args.target == "universe":
            return 0 if all(v.get("ok_within_window") for v in summary["indices"].values()) else 1
    if args.target in ("prices", "all"):
        from bistmas.data.calendar_build import build_and_write_calendar
        from bistmas.data.corporate_actions import (
            infer_corporate_actions,
            summarise_corporate_actions,
            write_corporate_actions,
        )
        from bistmas.data.prices import build_prices_store

        df = build_prices_store(settings)
        cal = build_and_write_calendar(df, settings.paths.store)
        ca = infer_corporate_actions(df)
        write_corporate_actions(ca, settings.paths.store)
        print(
            json.dumps(
                {
                    "prices_rows": int(len(df)),
                    "tickers": int(df["ticker"].nunique()),
                    "date_range": [str(df["date"].min().date()), str(df["date"].max().date())] if len(df) else None,
                    "volume_semantics": df.attrs.get("volume_semantics"),
                    "calendar_sessions": int(len(cal)),
                    "corporate_actions": summarise_corporate_actions(ca),
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
        if args.target == "prices":
            return 0
    if args.target in ("statements", "all"):
        from bistmas.data.filings import build_filing_index
        from bistmas.data.statements_fetch import build_statements_store

        build_filing_index(settings)
        meta = build_statements_store(settings)
        print(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
        if args.target == "statements":
            return 0
    if args.target in ("features", "all"):
        from bistmas.backtest.run import load_inputs, run_screens
        from bistmas.data.calendar_build import load_calendar

        inputs, _prices, provenance = load_inputs(settings)
        cal = load_calendar(settings.paths.store)
        first = date.fromisoformat(settings.rebalance.first_decision)
        last = date.fromisoformat(settings.rebalance.last_decision)
        decisions = [date.fromisoformat(args.as_of)] if args.as_of else cal.month_ends(first, last)
        screens = run_screens(settings, inputs, decisions)
        from bistmas.shared.store import write_parquet

        write_parquet(
            screens.astype({c: str for c in screens.columns if screens[c].dtype == object}),
            settings.paths.store / "features_monthly.parquet",
        )
        print(
            json.dumps(
                {"provenance": provenance, "rows": int(len(screens)), "decisions": len(decisions)},
                indent=2,
                default=str,
            )
        )
    return 0


def cmd_screen(args: argparse.Namespace) -> int:
    settings = load_settings()
    from bistmas.backtest.run import load_inputs
    from bistmas.scoring.screen import screen_one

    inputs, _prices, _prov = load_inputs(settings)
    out = screen_one(date.fromisoformat(args.as_of), inputs, settings)
    cols = [
        "rank",
        "ticker",
        "sector_group",
        "norm_group",
        "composite",
        "S_fundamental",
        "S_valuation",
        "S_momentum_risk",
        "filter_reason",
    ]
    print(out[[c for c in cols if c in out.columns]].head(25).to_string(index=False))
    return 0


def cmd_jury(args: argparse.Namespace) -> int:
    settings = load_settings()
    from bistmas.data.security_master import load_security_master
    from bistmas.jury.run import run_jury
    from bistmas.shared.store import read_parquet

    screens_path = settings.paths.store / "features_monthly.parquet"
    if not screens_path.exists():
        print("run `bistmas build features` first (features_monthly.parquet missing)")
        return 2
    screens = read_parquet(screens_path)
    master = load_security_master(settings.paths.store)
    months = [pd.Timestamp(m) for m in args.months] if args.months else None
    agent_models = None
    if args.agent_models:
        agent_models = dict(pair.split("=", 1) for pair in args.agent_models.split(",") if "=" in pair)
    summary = run_jury(
        settings,
        args.run,
        screens,
        master,
        decisions=months,
        dry_run=args.dry_run,
        max_calls=args.max_calls,
        agent_models=agent_models,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "prompt_hashes"}, indent=2, default=str))
    return 0 if summary["stopped_reason"] == "complete" else 3


def cmd_backtest(args: argparse.Namespace) -> int:
    settings = load_settings()
    if args.spec in ("deterministic", "all"):
        from bistmas.backtest.run import run_deterministic

        report = run_deterministic(settings, run_name=args.run_name or "deterministic")
        print(json.dumps({k: v for k, v in report["strategies"].items()}, indent=2, default=str))
    if args.spec in ("hybrid", "all"):
        from bistmas.experiments.hybrid import run_hybrid

        if not args.jury_run:
            print("hybrid backtest needs --jury-run <name> (a finished `bistmas jury --run <name>`).")
            return 2
        report = run_hybrid(settings, args.jury_run, run_name=args.run_name, n_random=args.n_random)
        print(
            json.dumps(
                {"strategies": report["strategies"], "random_jury_null": report.get("random_jury_null")},
                indent=2,
                default=str,
            )
        )
    return 0


def cmd_audit_rubric(args: argparse.Namespace) -> int:
    """H2 human rubric: no model calls (``--rubric-export <run>`` / ``--rubric-score <csv ...>``)."""
    from pathlib import Path

    settings = load_settings()
    if args.rubric_score:
        from bistmas.jury.audits.rubric import score_rubric

        print(json.dumps(score_rubric([Path(f) for f in args.rubric_score]), indent=2))
        return 0
    from bistmas.jury.audits.rubric import export_rubric
    from bistmas.jury.cache import JuryCache
    from bistmas.shared.store import read_parquet

    run = args.rubric_export
    scores = read_parquet(settings.paths.runs / run / "jury_scores.parquet")
    res = export_rubric(
        scores,
        JuryCache(settings.paths.runs / "jury_cache"),
        settings.audit.rubric_n,
        settings.audit.rubric_seed,
        settings.paths.runs / "audits",
        run,
    )
    print(json.dumps(res, indent=2, default=str))
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    settings = load_settings()
    if args.rubric_export or args.rubric_score:
        return cmd_audit_rubric(args)
    if not args.jury_run:
        print("audit needs --jury-run <name> (a finished `bistmas jury --run <name>`).")
        return 2
    from bistmas.jury.audits.run import ALL_AUDITS, run_audits

    which = tuple(args.which) if args.which else (() if args.robustness else ALL_AUDITS)
    out = run_audits(settings, args.jury_run, which=which, sample_size=args.sample_size, robustness=args.robustness)
    print(json.dumps(out["audits"], indent=2, default=str)[:6000])
    return 0


def cmd_forward(args: argparse.Namespace) -> int:
    settings = load_settings()
    from bistmas.reporting.forward import latest_decision_date, pending_months, run_forward

    if args.as_of:
        months = [date.fromisoformat(args.as_of)]
    elif args.catchup:
        months = pending_months(settings)
    else:
        months = [latest_decision_date(settings)]
    for m in months:
        d = run_forward(settings, m, dry_run=args.dry_run, send_mail=not args.no_mail)
        print(
            json.dumps(
                {
                    "as_of": d["as_of"],
                    "planned_execution_session": d["planned_execution_session"],
                    "late_run": d["late_run"],
                    "top": [r["ticker"] for r in d["top"]],
                    "sha256": d["sha256"],
                    "email": d.get("email_status"),
                },
                indent=2,
            )
        )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    settings = load_settings()
    from bistmas.reporting.results import write_results

    path = write_results(settings)
    if getattr(args, "figures", False):
        from bistmas.data.prices import load_prices
        from bistmas.reporting.figures import write_figures

        prices = load_prices(settings.paths.store, columns=["date", "usdtry"])
        usdtry = prices.groupby("date")["usdtry"].first().dropna()
        usdtry.index = pd.to_datetime(usdtry.index)
        from bistmas.backtest.tables import usdtry_series

        usdtry = usdtry_series(settings.paths.store, usdtry)
        figs = write_figures(settings.paths.runs, usdtry)
        print(f"{len(figs)} figure files written")
    print(path.read_text(encoding="utf-8")[:4000])
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bistmas", description="Hybrid rule + LLM stock selection for Borsa İstanbul")
    p.add_argument("--version", action="version", version=f"bistmas {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="check environment, Ollama server, pinned model and a JSON smoke call")
    d.add_argument("--skip-smoke", action="store_true", help="do not run the model smoke call")
    d.set_defaults(func=cmd_doctor)

    f = sub.add_parser("fetch", help="fetch raw data into data/raw (idempotent)")
    f.add_argument("target", choices=["universe", "prices", "statements", "macro", "kap-text"])
    f.add_argument("--force", action="store_true", help="re-fetch even if a cached copy exists")
    f.add_argument("--dry-run", action="store_true", help="prices: plan only, no requests")
    f.add_argument("--max-requests", type=int, default=None, help="prices: stop after this many requests (resumable)")
    f.add_argument("--start-year", type=int, default=2017)
    f.add_argument("--end-year", type=int, default=None)
    f.add_argument("--all-equities", action="store_true", help="prices: also fetch every current equity code")
    f.set_defaults(func=cmd_fetch)

    b = sub.add_parser("build", help="build store tables (universe, features) from raw data")
    b.add_argument("target", nargs="?", default="all", choices=["universe", "prices", "statements", "features", "all"])
    b.add_argument("--as-of", help="rebuild a single decision date (YYYY-MM-DD) for tests")
    b.add_argument("--end", help="stop the universe roll-forward at this date (YYYY-MM-DD)")
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("screen", help="deterministic screen: composites and Top-10 for one date")
    s.add_argument("--as-of", required=True)
    s.set_defaults(func=cmd_screen)

    j = sub.add_parser("jury", help="run the LLM jury over all decision dates (cached, resumable)")
    j.add_argument("--run", required=True, help="run name (e.g. prereg-v1)")
    j.add_argument("--max-calls", type=int, default=None, help="stop after this many new model calls (resumable)")
    j.add_argument("--dry-run", action="store_true", help="build and check every prompt, call no model")
    j.add_argument("--months", nargs="*", default=None, help="restrict to these decision dates (YYYY-MM-DD)")
    j.add_argument(
        "--agent-models",
        default=None,
        help="cross-family jury: agent=tag pairs, e.g. quality_growth=qwen3.5:9b,value=gemma4:12b-it-qat,"
        "risk_timing=llama3.1:8b (unlisted agents keep the configured model)",
    )
    j.set_defaults(func=cmd_jury)

    bt = sub.add_parser("backtest", help="walk-forward backtest and reports")
    bt.add_argument("--spec", choices=["deterministic", "hybrid", "all"], default="all")
    bt.add_argument("--run-name", default=None)
    bt.add_argument("--jury-run", default=None, help="hybrid: name of the jury run to use")
    bt.add_argument("--n-random", type=int, default=None, help="hybrid: random-jury seeds (default from config)")
    bt.set_defaults(func=cmd_backtest)

    a = sub.add_parser("audit", help="leakage and LLM-memorisation audits over a finished jury run")
    a.add_argument("--jury-run", default=None, help="name of the jury run to audit")
    a.add_argument(
        "--which",
        nargs="*",
        default=None,
        help="subset of: groundedness reidentification elicitation date_arms test_retest lap",
    )
    a.add_argument("--sample-size", type=int, default=None, help="default: 10 %% of the run (as_of, ticker) candidates")
    a.add_argument(
        "--robustness",
        action="store_true",
        help="H2: re-score sampled months under +-0.25 z noise and a prompt paraphrase (model calls; off by default; "
        "alone it skips the standard audits, with --which it runs both)",
    )
    a.add_argument("--rubric-export", metavar="RUN", default=None, help="H2: write the blinded 30-rationale rubric CSV")
    a.add_argument(
        "--rubric-score", nargs="+", metavar="CSV", default=None, help="H2: kappa from the filled-in rater CSV files"
    )
    a.set_defaults(func=cmd_audit)

    fw = sub.add_parser("forward", help="monthly forward paper-trading decision (sealed, e-mailed)")
    fw.add_argument("--dry-run", action="store_true", help="no model calls, no e-mail; renders the report only")
    fw.add_argument("--as-of", default=None, help="decision date (default: last session of the latest completed month)")
    fw.add_argument("--catchup", action="store_true", help="produce every missing month since the last decision")
    fw.add_argument("--no-mail", action="store_true")
    fw.set_defaults(func=cmd_forward)

    r = sub.add_parser("report", help="collect every run into results tables (Markdown + CSV)")
    r.add_argument("--figures", action="store_true", help="also write NAV / calendar-year / jury figures per run")
    r.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
