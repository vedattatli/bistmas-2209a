"""Three composite scores (equal weights, pre-registered) and the overall composite.

Each pillar is the mean of its available z-scored components; a pillar needs at least ``min_components``
non-missing components, otherwise it is NaN. The overall composite is the mean of the available pillars
(all three required by default for the Top-10 ranking; the screen reports how many were available).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PILLARS: dict[str, dict[str, list[str]]] = {
    "fundamental": {
        "industrial": ["roic", "operating_margin", "gross_margin", "revenue_growth", "profit_growth"],
        "bank": ["roe", "nim", "cost_to_income", "equity_to_assets", "profit_growth"],
        "insurer": ["roe", "equity_to_assets", "profit_growth"],
        "other_financial": ["roe", "equity_to_assets", "profit_growth"],
    },
    "valuation": {
        "industrial": ["earnings_yield", "book_to_market", "ebitda_yield", "fcf_yield"],
        "bank": ["earnings_yield", "book_to_market"],
        "insurer": ["earnings_yield", "book_to_market"],
        "other_financial": ["earnings_yield", "book_to_market"],
    },
    "momentum_risk": {
        "industrial": ["momentum_12_1", "vol_6m", "mdd_6m", "adv_6m_tl"],
        "bank": ["momentum_12_1", "vol_6m", "mdd_6m", "adv_6m_tl"],
        "insurer": ["momentum_12_1", "vol_6m", "mdd_6m", "adv_6m_tl"],
        "other_financial": ["momentum_12_1", "vol_6m", "mdd_6m", "adv_6m_tl"],
    },
}
ALL_COMPONENTS = sorted({c for p in PILLARS.values() for cols in p.values() for c in cols})


def pillar_scores(df: pd.DataFrame, min_components: int = 2) -> pd.DataFrame:
    """Add ``S_fundamental, S_valuation, S_momentum_risk`` from the ``z_`` columns, per template."""
    out = df.copy()
    for pillar, by_template in PILLARS.items():
        vals = pd.Series(np.nan, index=out.index)
        for template, comps in by_template.items():
            mask = out["template"].fillna("industrial") == template
            if not mask.any():
                continue
            zcols = [f"z_{c}" for c in comps if f"z_{c}" in out.columns]
            block = out.loc[mask, zcols]
            n_ok = block.notna().sum(axis=1)
            vals.loc[mask] = block.mean(axis=1).where(n_ok >= min_components)
        out[f"S_{pillar}"] = vals
    return out


def composite_score(
    df: pd.DataFrame, weights: dict[str, float] | None = None, require_all: bool = True
) -> pd.DataFrame:
    """Add ``composite`` (weighted mean of the pillars) and ``n_pillars``."""
    w = weights or {"fundamental": 1.0, "valuation": 1.0, "momentum_risk": 1.0}
    out = df.copy()
    cols = {p: f"S_{p}" for p in w}
    block = out[list(cols.values())]
    out["n_pillars"] = block.notna().sum(axis=1)
    wsum = sum(w.values())
    weighted = sum(block[cols[p]].fillna(0) * w[p] for p in w)
    avail = sum(block[cols[p]].notna() * w[p] for p in w)
    comp = weighted / avail.replace(0, np.nan)
    if require_all:
        comp = comp.where(out["n_pillars"] == len(w))
    out["composite"] = comp * (wsum / wsum)
    return out
