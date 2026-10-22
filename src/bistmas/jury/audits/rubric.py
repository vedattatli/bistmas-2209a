"""H2 human blind rubric: export 30 stratified rationales for three raters, score their agreement.

``export_rubric`` writes a blinded CSV (shuffled rows, opaque ids, pseudonymised payload summary + rationale +
score, empty rating columns 1-5) and a separate key file (id -> as_of / ticker / agent / cache key) that the raters
must not see. Each rater fills a copy of the CSV; ``score_rubric`` reads the copies and reports pairwise and mean
quadratic-weighted Cohen's kappa per criterion.

Criteria (1 = very poor, 5 = excellent):
* groundedness: every claim in the rationale is supported by the metrics shown;
* clarity: the rationale is understandable and well argued;
* consistency: the rationale is consistent with the score given (and with itself).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bistmas.jury.cache import JuryCache

CRITERIA = ("groundedness", "clarity", "consistency")
LEVELS = (1, 2, 3, 4, 5)
BLIND_COLUMNS = ["id", "payload_summary", "rationale", "score", *CRITERIA]


def summarise_payload(fields: dict[str, Any]) -> str:
    """Human-readable, pseudonymised payload: context labels and each metric as ``name: z=.. pct=..``."""
    metrics: dict[str, Any] = fields.get("metrics", {})
    names = sorted({k[:-2] if k.endswith("_z") else k[:-4] for k in metrics})

    def fmt(v: Any) -> str:
        return "n/a" if v is None else str(v)

    lines = [
        f"sector group: {fields.get('sector_group', 'n/a')} | size: {fields.get('size_tercile', 'n/a')} | "
        f"statement: {fields.get('statement_age', 'n/a')} | dimension: {fields.get('dimension', 'n/a')}"
    ]
    lines += [f"{n}: z={fmt(metrics.get(n + '_z'))} pct={fmt(metrics.get(n + '_pct'))}" for n in names]
    return "\n".join(lines)


def sample_rationales(scores: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """``n`` non-abstained verdicts spread over time: sort by (as_of, ticker, agent), cut into ``n`` bins, draw one."""
    ok = scores[~scores["abstain"].astype(bool) & (scores["rationale"].astype(str).str.len() > 0)]
    ok = ok.sort_values(["as_of", "ticker", "agent"]).reset_index(drop=True)
    if len(ok) <= n:
        return ok
    rng = np.random.default_rng(seed)
    idx = [int(rng.choice(chunk)) for chunk in np.array_split(np.arange(len(ok)), n) if len(chunk)]
    return ok.iloc[idx].reset_index(drop=True)


def export_rubric(
    scores: pd.DataFrame, cache: JuryCache, n: int, seed: int, out_dir: Path, name: str
) -> dict[str, Path | int]:
    """Write ``<name>_rubric_blind.csv`` (for the raters) and ``<name>_rubric_key.csv`` (for the analyst)."""
    picked = sample_rationales(scores, n, seed)
    rng = np.random.default_rng(seed + 1)
    picked = picked.iloc[rng.permutation(len(picked))].reset_index(drop=True)
    blind, key = [], []
    for i, r in enumerate(picked.itertuples(index=False), start=1):
        cached = cache.get(str(r.cache_key)) or {}
        rid = f"R{i:02d}"
        blind.append(
            {
                "id": rid,
                "payload_summary": summarise_payload(cached.get("payload", {})) if cached else "(payload unavailable)",
                "rationale": str(r.rationale),
                "score": float(r.score),
                **dict.fromkeys(CRITERIA, ""),
            }
        )
        key.append({"id": rid, "as_of": r.as_of, "ticker": r.ticker, "agent": r.agent, "cache_key": r.cache_key})
    out_dir.mkdir(parents=True, exist_ok=True)
    blind_path = out_dir / f"{name}_rubric_blind.csv"
    key_path = out_dir / f"{name}_rubric_key.csv"
    pd.DataFrame(blind, columns=BLIND_COLUMNS).to_csv(blind_path, index=False, encoding="utf-8")
    pd.DataFrame(key).to_csv(key_path, index=False, encoding="utf-8")
    return {"blind": blind_path, "key": key_path, "n": len(blind)}


def weighted_kappa(a: Sequence[int], b: Sequence[int], levels: Sequence[int] = LEVELS) -> float:
    """Quadratic-weighted Cohen's kappa of two integer ratings of the same items (NaN if chance disagreement is 0)."""
    k = len(levels)
    pos = {lv: i for i, lv in enumerate(levels)}
    obs = np.zeros((k, k))
    for x, y in zip(a, b, strict=True):
        obs[pos[int(x)], pos[int(y)]] += 1
    n = obs.sum()
    if n == 0:
        return float("nan")
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], dtype=float) / (k - 1) ** 2
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / n
    denom = (w * exp).sum()
    if denom == 0:
        return float("nan")
    return float(1.0 - (w * obs).sum() / denom)


def score_rubric(files: Sequence[Path]) -> dict[str, Any]:
    """Pairwise and mean quadratic-weighted kappa per criterion over the items rated by every rater."""
    raters = {Path(f).stem: pd.read_csv(f, dtype={"id": str}).set_index("id") for f in files}
    if len(raters) < 2:
        raise ValueError("need at least two rater files")
    ids = sorted(set.intersection(*(set(df.index) for df in raters.values())))
    out: dict[str, Any] = {"raters": sorted(raters), "n_items": len(ids), "criteria": {}}
    for c in CRITERIA:
        cols = {}
        for r, df in raters.items():
            col = pd.to_numeric(df.loc[ids, c], errors="coerce")
            cols[r] = col
        frame = pd.DataFrame(cols).dropna()
        bad = frame[(~frame.isin(LEVELS)).any(axis=1)]
        if len(bad):
            raise ValueError(f"criterion {c}: ratings outside 1-5 for items {sorted(bad.index)[:5]}")
        pairs = {
            f"{x}|{y}": weighted_kappa(frame[x].astype(int), frame[y].astype(int))
            for x, y in itertools.combinations(sorted(frame.columns), 2)
        }
        vals = [v for v in pairs.values() if not np.isnan(v)]
        out["criteria"][c] = {
            "n_complete": int(len(frame)),
            "pairwise_kappa": pairs,
            "mean_kappa": float(np.mean(vals)) if vals else None,
        }
    return out
