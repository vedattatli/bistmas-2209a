"""Deterministic screen for one decision date: features -> filters -> z -> pillars -> composite -> Top-N."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pandas as pd

from bistmas.scoring.composite import ALL_COMPONENTS, composite_score, pillar_scores
from bistmas.scoring.features import FeatureInputs, build_features_one
from bistmas.scoring.filters import apply_filters
from bistmas.scoring.normalize import normalise_features
from bistmas.shared.config import Settings


def screen_one(as_of: date | pd.Timestamp, inputs: FeatureInputs, settings: Settings) -> pd.DataFrame:
    """Full scored table for ``as_of`` (all members), sorted by composite; ``rank`` and ``top_n`` flags added."""
    inputs = replace(inputs, max_statement_age_days=settings.scoring.max_statement_age_days)
    feats = build_features_one(as_of, inputs)
    if feats.empty:
        return feats
    feats = apply_filters(
        feats,
        min_listed_sessions=settings.scoring.min_listed_sessions,
        adv_percentile_floor=settings.scoring.adv_percentile_floor,
        min_adv_tl=float(settings.universe.min_adv_tl_6m),
        max_statement_age_days=settings.scoring.max_statement_age_days,
    )
    elig = feats[feats["eligible"]]
    scored = normalise_features(
        elig, ALL_COMPONENTS, min_group_size=settings.scoring.sector_group_min_size, clip=settings.scoring.zscore_clip
    )
    scored = pillar_scores(scored)
    scored = composite_score(scored, settings.scoring.composite_weights)
    # eligible names whose composite is missing (a pillar without enough components) cannot be ranked
    unrankable = scored["composite"].isna()
    scored.loc[unrankable, "filter_reason"] = "no_composite"
    scored.loc[unrankable, "eligible"] = False
    scored = scored.sort_values("composite", ascending=False, na_position="last")
    scored["rank"] = pd.array(range(1, len(scored) + 1), dtype="Int64")
    scored.loc[scored["composite"].isna(), "rank"] = pd.NA
    scored["top_n"] = scored["rank"].le(settings.scoring.top_n_screen)
    rest = feats[~feats["eligible"]].copy()
    rest["rank"] = pd.array([pd.NA] * len(rest), dtype="Int64")
    rest["top_n"] = False
    out = pd.concat([scored, rest], ignore_index=True)
    out.attrs["adv_floor_tl"] = feats.attrs.get("adv_floor_tl")
    return out
