"""Results tables for the paper: one Markdown + CSV summary across every run under ``data/runs``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from bistmas.shared.config import Settings
from bistmas.shared.log import get_logger

log = get_logger("reporting.results")

METRICS = ["cagr", "vol", "sharpe", "sortino", "max_drawdown", "total_return", "avg_turnover", "hit_rate_monthly"]


def collect_runs(runs_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for report in sorted(Path(runs_dir).glob("*/report.json")):
        data = json.loads(report.read_text(encoding="utf-8"))
        for label, entry in data.get("strategies", {}).items():
            for numeraire in ("tl", "usd", "real"):
                m = entry.get(numeraire) or {}
                if numeraire == "real" and not m:
                    continue
                row = {
                    "run": data.get("run"),
                    "strategy": label,
                    "numeraire": numeraire,
                    **{k: m.get(k) for k in METRICS},
                }
                # active-return statistics are computed per numeraire; older reports only have the TL ones
                act = m.get("active") or (entry if numeraire == "tl" else {})
                nw = act.get("newey_west") or {}
                row["active_mean_monthly"] = act.get("mean", act.get("active_monthly_mean", nw.get("mean")))
                row["nw_t"] = nw.get("t")
                row["nw_p"] = nw.get("p")
                ci = act.get("bootstrap_ci_95") or {}
                row["boot_lo"], row["boot_hi"] = ci.get("lo"), ci.get("hi")
                row["information_ratio"] = act.get("information_ratio")
                dsr = act.get("deflated_sharpe") or {}
                row["psr"], row["n_trials"] = dsr.get("psr"), dsr.get("n_trials")
                vs = entry.get("vs_deterministic") or {}
                if numeraire == "tl":
                    row["vs_det_mean_monthly"] = vs.get("mean_monthly_diff")
                    row["vs_det_nw_t"] = (vs.get("newey_west") or {}).get("t")
                    row["vs_det_nw_p"] = (vs.get("newey_west") or {}).get("p")
                rows.append(row)
        null = data.get("random_jury_null")
        if null:
            rows.append(
                {
                    "run": data.get("run"),
                    "strategy": "random_jury_null",
                    "numeraire": "tl",
                    "cagr": null.get("cagr_mean"),
                    "null_cagr_p05": null.get("cagr_p05"),
                    "null_cagr_p95": null.get("cagr_p95"),
                    "null_hybrid_percentile": null.get("hybrid_percentile"),
                    "null_n": null.get("n"),
                }
            )
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_no runs yet_"
    cols = [
        "run",
        "strategy",
        "numeraire",
        "cagr",
        "vol",
        "sharpe",
        "sortino",
        "max_drawdown",
        "avg_turnover",
        "active_mean_monthly",
        "nw_t",
        "nw_p",
        "boot_lo",
        "boot_hi",
        "information_ratio",
        "psr",
        "n_trials",
        "vs_det_mean_monthly",
        "vs_det_nw_t",
        "vs_det_nw_p",
    ]
    cols = [c for c in cols if c in df.columns]
    null_rows = df[df["strategy"] == "random_jury_null"] if "strategy" in df.columns else df.iloc[0:0]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    df = df[df["strategy"] != "random_jury_null"] if "strategy" in df.columns else df
    for r in df[cols].itertuples(index=False):
        cells = []
        for v in r:
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else f"{v:.3f}")
            else:
                cells.append("" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + _null_table(null_rows)


def _null_table(null: pd.DataFrame) -> str:
    if null.empty:
        return ""
    lines = [
        "\n\n### Random-jury null (TL CAGR of the headline Top-N under random scores)\n",
        "| run | n | null CAGR mean | p05 | p95 | hybrid percentile |\n|---|---|---|---|---|---|",
    ]
    for r in null.itertuples(index=False):
        lines.append(
            f"| {r.run} | {_fmt(r.null_n, 0)} | {_fmt(r.cagr)} | {_fmt(r.null_cagr_p05)} "
            f"| {_fmt(r.null_cagr_p95)} | {_fmt(r.null_hybrid_percentile, 3)} |"
        )
    return "\n".join(lines)


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)


def secondary_sections(runs_dir: Path) -> str:
    """Markdown sections for the pre-registered secondary tables of every run that has them."""
    parts: list[str] = []
    for report in sorted(Path(runs_dir).glob("*/report.json")):
        data = json.loads(report.read_text(encoding="utf-8"))
        tables = data.get("tables") or {}
        if not tables:
            continue
        run = data.get("run")
        parts.append(f"\n## {run}: secondary tables\n")
        cy = tables.get("calendar_year") or []
        if cy:
            cols = list(cy[0].keys())
            parts.append("### Calendar-year net returns (holding year)\n")
            parts.append("| " + " | ".join(cols) + " |")
            parts.append("|" + "---|" * len(cols))
            for row in cy:
                parts.append("| " + " | ".join(_fmt(row[c]) if c != "year" else str(row[c]) for c in cols) + " |")
        mde = tables.get("mde") or {}
        if mde:
            parts.append("\n### Minimum detectable monthly active return (80 % power, two-sided 5 %)\n")
            parts.append("| strategy | n | HAC se | MDE monthly | MDE annualised |\n|---|---|---|---|---|")
            for label, m in mde.items():
                parts.append(
                    f"| {label} | {m.get('n')} | {_fmt(m.get('se'), 4)} | {_fmt(m.get('mde_monthly'), 4)} "
                    f"| {_fmt(m.get('mde_annualised'), 3)} |"
                )
        loyo = tables.get("leave_one_year_out") or {}
        if loyo:
            parts.append("\n### Leave-one-year-out (Newey-West mean of monthly active return)\n")
            parts.append("| strategy | year excluded | n | mean | t | p |\n|---|---|---|---|---|---|")
            for label, rows in loyo.items():
                for r in rows:
                    parts.append(
                        f"| {label} | {r['year_excluded']} | {r['n']} | {_fmt(r['mean'], 4)} "
                        f"| {_fmt(r['t'], 2)} | {_fmt(r['p'], 3)} |"
                    )
        cs = tables.get("cost_sensitivity") or []
        if cs:
            parts.append("\n### Cost sensitivity (one-way bps)\n")
            parts.append("| cost | strategy | CAGR TL | active monthly | NW t | NW p |\n|---|---|---|---|---|---|")
            for r in cs:
                parts.append(
                    f"| {r['cost_bps_one_way']:.0f} | {r['strategy']} | {_fmt(r.get('cagr_tl'))} "
                    f"| {_fmt(r.get('active_monthly_mean'), 4)} | {_fmt(r.get('nw_t'), 2)} | {_fmt(r.get('nw_p'), 3)} |"
                )
        rf = data.get("rf")
        if rf:
            parts.append(f"\nRisk-free for TL Sharpe/Sortino: {rf.get('source')} (USD leg uses 0).")
    return "\n".join(parts)


def write_results(settings: Settings) -> Path:
    df = collect_runs(settings.paths.runs)
    out_dir = settings.paths.runs / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results.csv", index=False)
    md = "# Results\n\n" + to_markdown(df) + "\n" + secondary_sections(settings.paths.runs) + "\n"
    (out_dir / "results.md").write_text(md, encoding="utf-8")
    log.info("results written to %s (%d rows)", out_dir, len(df))
    return out_dir / "results.md"
