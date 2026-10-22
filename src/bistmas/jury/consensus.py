"""Consensus rule (pre-registered): J = mean of the available agent scores; abstentions never impute.

A candidate with fewer than ``min_votes`` valid scores has J = NaN and is excluded from the Top-N, but it stays
in the table (also when all three agents abstained) so comparators see the full candidate set; the share of
such candidates is reported. J is rounded to ``J_DECIMALS`` before ranking and before the theta threshold, so
floating-point noise cannot break exact ties (scores sit on a 0.05 grid).
``rank_J`` ranks within the month (ties broken by the deterministic composite rank so the outcome is reproducible).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bistmas.jury.agents import AGENTS

J_DECIMALS = 6


def consensus(scores: pd.DataFrame, min_votes: int = 2, theta: float = 0.75) -> pd.DataFrame:
    """``scores``: one row per (as_of, ticker, agent) with ``score`` (NaN when abstained).

    Returns one row per (as_of, ticker): ``J, n_votes, buy (J >= theta), rank_J`` plus per-agent columns.
    """
    # groupby keeps all-NaN (all agents abstained) candidates; pivot_table would drop them
    piv = scores.groupby(["as_of", "ticker", "agent"])["score"].first().unstack("agent")
    for a in AGENTS:
        if a not in piv.columns:
            piv[a] = np.nan
    piv = piv[list(AGENTS)]
    out = piv.copy()
    out["n_votes"] = piv.notna().sum(axis=1)
    out["J"] = piv.mean(axis=1).round(J_DECIMALS).where(out["n_votes"] >= min_votes)
    out["buy"] = out["J"] >= theta
    out = out.reset_index()
    if "composite_rank" in scores.columns:
        cr = scores.groupby(["as_of", "ticker"])["composite_rank"].first()
        out = out.merge(cr.rename("composite_rank").reset_index(), on=["as_of", "ticker"], how="left")
    else:
        out["composite_rank"] = np.nan
    out = out.sort_values(["as_of", "J", "composite_rank"], ascending=[True, False, True], na_position="last")
    out["rank_J"] = out.groupby("as_of")["J"].rank(method="first", ascending=False)
    out.loc[out["J"].isna(), "rank_J"] = np.nan
    return out.reset_index(drop=True)
