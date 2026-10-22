"""Masked, partitioned jury payloads.

Anonymisation (red-team pitfall #1): the model sees a per-month pseudonym (STOCK_A..), a coarse sector
group, a size tercile and a statement-age bucket — never names, tickers, dates, fiscal periods, TMS-29
flags or absolute amounts. Numbers are within-month percentile ranks (0-100, among eligible names) and
robust z-scores (2 decimals) of the agent's OWN dimension only (information partition), never the
composite. ``assert_clean`` scans every rendered prompt for leaks.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from bistmas.scoring.composite import PILLARS
from bistmas.scoring.normalize import FEATURE_SIGN

AGENT_DIMENSION = {"quality_growth": "fundamental", "value": "valuation", "risk_timing": "momentum_risk"}
PSEUDONYMS = [f"STOCK_{c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
SIZE_LABELS = ["small", "mid", "large"]
LEAK_PATTERNS = [
    re.compile(r"\b(19|20)\d{2}\b"),  # years
    re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b"),  # dates
    re.compile(r"\b(?:ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık)\b", re.I),
    re.compile(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b", re.I),
    re.compile(r"\b(?:Q[1-4]|H[12]|FY)\b"),
    re.compile(r"\b(?:TL|TRY|USD|EUR)\b"),
    re.compile(r"\b\d{7,}\b"),  # absolute amounts
    re.compile(r"\b(?:election|pandemic|covid|earthquake|lira|inflation accounting|tms|ifrs)\b", re.I),
]


@dataclass(frozen=True)
class CandidatePayload:
    candidate_id: str
    agent: str
    fields: dict[str, Any]  # what the model sees
    ticker: str  # kept OUT of the prompt; used for the cache key and the join back

    def render(self) -> str:
        return json.dumps({"candidate_id": self.candidate_id, **self.fields}, ensure_ascii=False, indent=1)


def pseudonyms_for(as_of: pd.Timestamp, tickers: list[str], salt: str) -> dict[str, str]:
    """Per-month random mapping ticker -> STOCK_X (re-randomised every month; identities never persist)."""
    seed = int(hashlib.sha256(f"{salt}|{pd.Timestamp(as_of).date()}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    labels = list(PSEUDONYMS[: max(len(tickers), 1)])
    rng.shuffle(labels)
    return {t: labels[i] for i, t in enumerate(sorted(tickers))}


def size_terciles(mcap: pd.Series) -> pd.Series:
    ranks = mcap.rank(pct=True)
    return pd.cut(ranks, [0, 1 / 3, 2 / 3, 1.0001], labels=SIZE_LABELS, include_lowest=True).astype(str)


def percentile_ranks(values: pd.Series) -> pd.Series:
    return (values.rank(pct=True) * 100).round().astype("Int64")


def build_payloads(
    screen_df: pd.DataFrame,
    as_of: pd.Timestamp,
    candidates: list[str],
    agent: str,
    salt: str,
    template_col: str = "template",
) -> list[CandidatePayload]:
    """One payload per candidate for ``agent``: pseudonym, coarse context and the agent's own dimension block.

    Percentiles are computed among the eligible names of the SAME normalisation group (``norm_group``, the
    pooled group the screen z-scores within; ``sector_group`` for old stores without it), and that group is
    also the coarse sector label sent to the model, so percentile, z-score and label share one peer set.
    The z-scores are the ones used by the deterministic screen. Missing components are reported as null.
    """
    dim = AGENT_DIMENSION[agent]
    elig = screen_df[screen_df["eligible"]].copy()
    names = pseudonyms_for(as_of, candidates, salt)
    gcol = "norm_group" if "norm_group" in elig.columns else "sector_group"
    elig["_grp"] = elig[gcol].astype(str)
    elig["size_tercile"] = size_terciles(pd.to_numeric(elig["mcap_tl"], errors="coerce"))
    out: list[CandidatePayload] = []
    for t in candidates:
        if t not in set(elig["ticker"]):
            continue
        row = elig[elig["ticker"] == t].iloc[0]
        peers = elig[elig["_grp"] == row["_grp"]]
        template = str(row.get(template_col, "industrial"))
        comps = PILLARS[dim].get(template, PILLARS[dim]["industrial"])
        block: dict[str, Any] = {}
        for c in comps:
            zc = f"z_{c}"
            z = row.get(zc, np.nan)
            # rank on the SIGNED value so that percentile 100 is always the best/cheapest/safest name
            pct = (
                percentile_ranks(FEATURE_SIGN.get(c, 1.0) * pd.to_numeric(peers[c], errors="coerce"))
                if c in peers.columns
                else None
            )
            p = pct[peers["ticker"] == t].iloc[0] if pct is not None else pd.NA
            block[f"{c}_z"] = None if pd.isna(z) else round(float(z), 2)
            block[f"{c}_pct"] = None if pd.isna(p) else int(p)
        fields = {
            "sector_group": str(row.get(gcol, "OTHER")).lower().replace("_", " "),
            "size_tercile": str(row["size_tercile"]),
            "statement_age": "recent"
            if pd.notna(row.get("statement_age_days")) and float(row["statement_age_days"]) <= 120
            else "stale",
            "dimension": dim,
            "metrics": block,
        }
        out.append(CandidatePayload(names[t], agent, fields, t))
    return out


CONTEXT_FIELDS = ("sector_group", "size_tercile", "statement_age")


def allowed_metric_names(payload: CandidatePayload) -> set[str]:
    """Field names a verdict may cite: every metric key (with and without the _z/_pct suffix) and the
    coarse context fields (sector group, size tercile, statement age) that the payload also shows."""
    names = set(payload.fields["metrics"].keys())
    names |= {k[:-2] for k in names if k.endswith("_z")} | {k[:-4] for k in names if k.endswith("_pct")}
    names |= {f for f in CONTEXT_FIELDS if f in payload.fields}
    return names


def assert_clean(text: str, forbidden_tokens: set[str]) -> None:
    """Raise if the rendered prompt contains a ticker, a company-name token, a date/year or an amount."""
    for pat in LEAK_PATTERNS:
        m = pat.search(text)
        if m:
            raise ValueError(f"prompt leak: pattern {pat.pattern!r} matched {m.group(0)!r}")
    upper = text.upper()
    for tok in forbidden_tokens:
        if len(tok) >= 4 and re.search(rf"\b{re.escape(tok.upper())}\b", upper):
            raise ValueError(f"prompt leak: forbidden token {tok!r}")
