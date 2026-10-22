"""Random-jury null: uniform(0, 1) scores rounded to the 0.05 grid through the identical consensus/portfolio code path.

Used to build the empirical null distribution of the hybrid strategy's performance (>= 500 seeds).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bistmas.jury.agents import AGENTS


def random_scores(candidates: pd.DataFrame, seed: int) -> pd.DataFrame:
    """``candidates``: rows (as_of, ticker[, composite_rank]) -> scores frame like the real jury's."""
    rng = np.random.default_rng(seed)
    rows = []
    for rec in candidates.itertuples(index=False):
        for a in AGENTS:
            rows.append(
                {
                    "as_of": rec.as_of,
                    "ticker": rec.ticker,
                    "agent": a,
                    "score": round(float(rng.uniform()) * 20)
                    / 20,  # the real jury's 0.05 grid, so ties occur as in reality
                    "composite_rank": getattr(rec, "composite_rank", np.nan),
                }
            )
    return pd.DataFrame(rows)
