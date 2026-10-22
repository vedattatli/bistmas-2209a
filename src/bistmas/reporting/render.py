"""Markdown rendering of a monthly decision (used for the e-mail body and the decision file)."""

from __future__ import annotations

from typing import Any

import pandas as pd


def render_decision_markdown(decision: dict[str, Any]) -> str:
    cert = decision["certificate"]
    lines = [
        f"# bistmas monthly decision — as of {decision['as_of']}",
        "",
        f"Planned execution session: **{decision['planned_execution_session']}**  ",
        f"Run: `{decision['run_id']}` · code `{cert['code_commit'][:10]}` · model `{cert['model']['name']}` "
        f"@ `{str(cert['model']['digest_pinned'])[:12]}`  ",
        f"Point-in-time check: {'OK' if cert['pit']['ok'] else 'FAILED'} ({cert['pit']['rows']} feature rows)",
        "",
        "## Top picks (hybrid: jury score J)",
        "",
        "| rank | ticker | J | votes | composite rank | sector |",
        "|---|---|---|---|---|---|",
    ]
    for r in decision["top"]:
        lines.append(
            f"| {r['rank_J']:.0f} | {r['ticker']} | {r['J']:.3f} | {r['n_votes']} | {r['composite_rank']:.0f} "
            f"| {r.get('sector_group', '')} |"
        )
    lines += [
        "",
        "## Deterministic Top-10 (composite)",
        "",
        "| rank | ticker | composite | fundamental | valuation | momentum/risk |",
        "|---|---|---|---|---|---|",
    ]
    for r in decision["screen_top"]:
        lines.append(
            f"| {r['rank']:.0f} | {r['ticker']} | {r['composite']:.2f} | {_f(r.get('S_fundamental'))} "
            f"| {_f(r.get('S_valuation'))} | {_f(r.get('S_momentum_risk'))} |"
        )
    lines += ["", "## Jury rationales (masked inputs; names re-attached after scoring)", ""]
    for r in decision.get("rationales", []):
        lines.append(f"**{r['ticker']} — {r['agent']}** (score {r['score']:.2f}): {r['rationale']}")
        lines.append("")
    lines += [
        "",
        f"Decision file sha256: `{decision.get('sha256', '')}`",
        "",
        "This is a research prototype (TÜBİTAK 2209-A); it is not investment advice.",
    ]
    return "\n".join(lines)


def _f(v: Any) -> str:
    return "" if v is None or pd.isna(v) else f"{float(v):.2f}"
