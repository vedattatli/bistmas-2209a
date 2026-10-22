"""Rationale groundedness and leak scan over a finished jury run (pure, no model calls).

* grounded-fact rate: every metric-like token in the rationale (``name_z`` / ``name_pct``) must exist in the
  payload the agent saw; ``cited_metrics`` were already validated at call time.
* leak-flag rate: the rationale must not contain a ticker, a company-name token, a year, a date, a month
  name or an event keyword (the model may only reason from the masked numbers).
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from bistmas.jury.cache import JuryCache
from bistmas.jury.payload import LEAK_PATTERNS

_EN_MONTHS = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
# 'may' / 'march' are ordinary English words: an English month name only counts as a date next to a day or year
EN_MONTH_DATE = re.compile(rf"\b{_EN_MONTHS}\.?\s+\d{{1,4}}\b|\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_EN_MONTHS}\b", re.I)
_PLAIN_ENGLISH_MONTH = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b", re.I
)
# the payload's pattern set minus the bare English month names (replaced by EN_MONTH_DATE)
RATIONALE_LEAK_PATTERNS = [p for p in LEAK_PATTERNS if p.pattern != _PLAIN_ENGLISH_MONTH.pattern] + [EN_MONTH_DATE]

METRIC_TOKEN = re.compile(r"\b([a-z][a-z0-9_]*_(?:z|pct))\b")


def scan_rationale(rationale: str, payload_metrics: set[str], forbidden: set[str]) -> dict[str, Any]:
    tokens = set(METRIC_TOKEN.findall(rationale or ""))
    ungrounded = sorted(t for t in tokens if t not in payload_metrics)
    leaks: list[str] = []
    for pat in RATIONALE_LEAK_PATTERNS:
        m = pat.search(rationale or "")
        if m:
            leaks.append(m.group(0))
    # tickers / title words are matched in upper case only: 'ELITE' is a code, 'elite' and 'Status' are words
    for tok in forbidden:
        if len(tok) >= 4 and re.search(rf"\b{re.escape(tok.upper())}\b", rationale or ""):
            leaks.append(tok)
    return {
        "n_metric_tokens": len(tokens),
        "ungrounded": ungrounded,
        "grounded": not ungrounded,
        "leaks": leaks,
        "leak_flag": bool(leaks),
    }


def audit_groundedness(scores: pd.DataFrame, cache: JuryCache, forbidden: set[str]) -> dict[str, Any]:
    """Per-agent grounded-fact and leak-flag rates for every non-abstained verdict of a run."""
    rows = []
    for rec in scores.itertuples(index=False):
        if bool(rec.abstain):
            continue
        cached = cache.get(str(rec.cache_key))
        metrics = set((cached or {}).get("payload", {}).get("metrics", {}).keys()) if cached else set()
        res = scan_rationale(str(rec.rationale), metrics, forbidden)
        rows.append({"as_of": rec.as_of, "ticker": rec.ticker, "agent": rec.agent, **res})
    df = pd.DataFrame(rows)
    if df.empty:
        return {"n": 0}
    by_agent = df.groupby("agent").agg(
        grounded_rate=("grounded", "mean"), leak_rate=("leak_flag", "mean"), n=("grounded", "size")
    )
    matches: dict[str, int] = {}
    for lst in df["leaks"]:
        for m in lst:
            matches[str(m).lower()] = matches.get(str(m).lower(), 0) + 1
    return {
        "n": int(len(df)),
        "leak_matches": dict(sorted(matches.items(), key=lambda kv: -kv[1])[:30]),
        "grounded_rate": float(df["grounded"].mean()),
        "leak_rate": float(df["leak_flag"].mean()),
        "by_agent": {a: {k: (float(v) if k != "n" else int(v)) for k, v in r.items()} for a, r in by_agent.iterrows()},
        "examples_ungrounded": df[~df["grounded"]]
        .head(10)[["as_of", "ticker", "agent", "ungrounded"]]
        .to_dict("records"),
        "examples_leaks": df[df["leak_flag"]].head(10)[["as_of", "ticker", "agent", "leaks"]].to_dict("records"),
    }


def load_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return value
    try:
        out = json.loads(value)
        return out if isinstance(out, list) else []
    except (TypeError, ValueError):
        return []
