"""H2 robustness of the jury ranking to input noise and to a prompt paraphrase (docs/preregistration.md).

For a stratified sample of decision months of a finished jury run the SAME candidates are scored again by the
three agents under two perturbations (each separately, so the effects are not confounded):

* ``noise``: every z-score in the payload gets independent uniform noise in ``[-noise_z, +noise_z]`` (seeded per
  month, rounded to 2 decimals like the original). The percentile ranks are left as they are: they are computed
  from the raw feature values, not from the z-scores, so the perturbation models noise in the standardised
  inputs only. Nulls stay null.
* ``paraphrase``: the unchanged payload under a reworded prompt set (``prompts/v2_paraphrase``: same instructions,
  output schema and placeholders).

Per month the consensus J of the perturbed run is compared with the original J of the same run: Kendall tau (tau-b,
over candidates with a J in both) and Top-k overlap (share of the original Top-k that is still in the Top-k). The
summary is the mean and median over months. Calls go through ``JuryAgent`` (cache, leak check, validation).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from bistmas.jury.agents import AGENTS, AgentPrompts, JuryAgent
from bistmas.jury.consensus import consensus
from bistmas.jury.payload import CandidatePayload

CONDITIONS = ("noise", "paraphrase")


def sample_months(months: list[str], n: int, seed: int, exclude_years: tuple[int, ...] = ()) -> list[str]:
    """``n`` months, one drawn at random from each of ``n`` equal time bins (covers the whole sample period).

    Months of ``exclude_years`` (the prompt-development slice) are skipped while enough other months remain.
    """
    ms = sorted(set(months))
    pool = [m for m in ms if int(m[:4]) not in exclude_years]
    pool = pool if len(pool) >= n else ms
    if len(pool) <= n:
        return pool
    rng = np.random.default_rng(seed)
    picks = [pool[int(rng.choice(chunk))] for chunk in np.array_split(np.arange(len(pool)), n) if len(chunk)]
    return sorted(picks)


def perturb_payload(p: CandidatePayload, noise_z: float, rng: np.random.Generator) -> CandidatePayload:
    """Copy of ``p`` with uniform noise in ``[-noise_z, +noise_z]`` added to every non-null ``*_z`` metric."""
    metrics = dict(p.fields["metrics"])
    for k in sorted(metrics):  # fixed order: the draw sequence depends only on the seed
        v = metrics[k]
        if k.endswith("_z") and v is not None:
            metrics[k] = round(float(v) + float(rng.uniform(-noise_z, noise_z)), 2)
    return dataclasses.replace(p, fields={**p.fields, "metrics": metrics})


def month_seed(seed: int, as_of: str) -> int:
    return seed + int(as_of.replace("-", ""))


def compare_month(orig: pd.DataFrame, new: pd.DataFrame, top_k: int) -> dict[str, Any]:
    """Kendall tau of J and Top-k overlap between two consensus tables of ONE month (see module doc)."""
    both = orig[["ticker", "J"]].merge(new[["ticker", "J"]], on="ticker", suffixes=("_o", "_n")).dropna()
    tau = float("nan")
    if len(both) >= 2 and both["J_o"].nunique() > 1 and both["J_n"].nunique() > 1:
        tau = float(sps.kendalltau(both["J_o"], both["J_n"]).statistic)
    top_o = orig.dropna(subset=["rank_J"]).sort_values("rank_J").head(top_k)["ticker"].tolist()
    top_n = new.dropna(subset=["rank_J"]).sort_values("rank_J").head(top_k)["ticker"].tolist()
    k_eff = min(top_k, len(top_o), len(top_n))
    overlap = len(set(top_o[:k_eff]) & set(top_n[:k_eff])) / k_eff if k_eff else float("nan")
    return {
        "n_candidates": int(len(both)),
        "kendall_tau": tau,
        "top_k_overlap": overlap,
        "top_original": top_o,
        "top_perturbed": top_n,
    }


def _nanstat(values: list[float], fn: Callable[[list[float]], float]) -> float | None:
    vals = [v for v in values if not (v is None or math.isnan(v))]
    return float(fn(vals)) if vals else None


def summarise(per_month: list[dict[str, Any]]) -> dict[str, Any]:
    taus = [m["kendall_tau"] for m in per_month]
    ovl = [m["top_k_overlap"] for m in per_month]
    return {
        "n_months": len(per_month),
        "kendall_tau_mean": _nanstat(taus, np.mean),
        "kendall_tau_median": _nanstat(taus, np.median),
        "top_k_overlap_mean": _nanstat(ovl, np.mean),
        "top_k_overlap_median": _nanstat(ovl, np.median),
    }


def audit_robustness(
    scores: pd.DataFrame,
    payloads_for: Callable[[str, str], list[CandidatePayload]],
    make_agent: Callable[[str, AgentPrompts], JuryAgent],
    original_prompts: AgentPrompts,
    paraphrase_prompts: AgentPrompts,
    months: list[str],
    noise_z: float,
    top_k: int,
    seed: int,
    min_votes: int = 2,
    theta: float = 0.75,
) -> dict[str, Any]:
    """Re-score ``months`` of a finished run under both perturbations.

    ``payloads_for(as_of, agent)`` rebuilds the run's payloads (same pseudonym salt); only candidates that were
    scored in the original run are used. ``make_agent(agent, prompts)`` returns a ``JuryAgent`` bound to the
    given prompt set. Raises ``TransportError`` (from the agent) if the model host fails; nothing partial is kept.
    """
    sc = scores.assign(score=pd.to_numeric(scores["score"], errors="coerce"))
    sc["as_of"] = sc["as_of"].astype(str)
    out: dict[str, Any] = {
        "months": months,
        "noise_z": noise_z,
        "top_k": top_k,
        "seed": seed,
        "paraphrase_prompt_version": paraphrase_prompts.version,
        "paraphrase_prompt_hashes": paraphrase_prompts.hashes,
        "percentiles": "left unchanged (noise is applied to *_z only)",
        "per_month": {c: [] for c in CONDITIONS},
        "calls": 0,
    }
    for m in months:
        month_scores = sc[sc["as_of"] == m]
        tickers = set(month_scores["ticker"])
        orig = consensus(month_scores, min_votes=min_votes, theta=theta)
        rng = np.random.default_rng(month_seed(seed, m))
        new_rows: dict[str, list[dict[str, Any]]] = {c: [] for c in CONDITIONS}
        crank = month_scores.groupby("ticker")["composite_rank"].first() if "composite_rank" in sc.columns else {}
        for agent in AGENTS:
            payloads = [p for p in payloads_for(m, agent) if p.ticker in tickers]
            noisy = [perturb_payload(p, noise_z, rng) for p in payloads]
            runs = {
                "noise": (make_agent(agent, original_prompts), noisy),
                "paraphrase": (make_agent(agent, paraphrase_prompts), payloads),
            }
            for cond, (ja, ps) in runs.items():
                for p in ps:
                    rec = ja.judge(m, p)
                    if not rec.cache_hit:
                        out["calls"] += 1
                    new_rows[cond].append(
                        {
                            "as_of": m,
                            "ticker": p.ticker,
                            "agent": agent,
                            "score": np.nan if rec.abstain else rec.score,
                            "composite_rank": crank.get(p.ticker) if len(crank) else np.nan,
                        }
                    )
        for cond in CONDITIONS:
            new = consensus(pd.DataFrame(new_rows[cond]), min_votes=min_votes, theta=theta)
            out["per_month"][cond].append({"as_of": m, **compare_month(orig, new, top_k)})
    out["summary"] = {c: summarise(out["per_month"][c]) for c in CONDITIONS}
    return out
