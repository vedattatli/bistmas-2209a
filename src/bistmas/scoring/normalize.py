"""Cross-sectional robust normalisation within sector groups.

``robust_z`` = (x - median) / (1.4826 * MAD), clipped to ±clip. A group with fewer than ``min_group_size``
NON-NULL values of a feature is pooled, for that feature, into its PARENT group (``parent_group`` column of
``configs/sector_groups.csv``: FINANCIALS / NON_FINANCIALS); if the parent is still too small, all eligible
names of the month are used. ``norm_group`` records the group a name was pooled to by row count.

Growth features (``GROWTH_FEATURES``) are only compared within the same accounting basis (nominal vs
TMS-29) in a month where both bases coexist; a basis subgroup (after pooling) with too few values gets a
NaN z-score. Orientation: every feature is converted so that HIGHER IS BETTER before z-scoring
(``FEATURE_SIGN``); features not listed are used as they are.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from bistmas.shared.config import CONFIG_DIR

POOLED_GROUP = "POOLED"  # legacy label, only used by pool_small_groups
ALL_GROUP = "ALL"
DEFAULT_PARENT = "NON_FINANCIALS"
GROWTH_FEATURES = ("revenue_growth", "profit_growth")
_NAN_LABEL = "__NAN__"
MAD_SCALE = 1.4826
FEATURE_SIGN: dict[str, float] = {
    "vol_6m": -1.0,
    "mdd_6m": +1.0,  # drawdown is negative; less negative is better
    "net_debt_to_ebitda": -1.0,
    "cost_to_income": -1.0,
}


def pool_small_groups(groups: pd.Series, min_group_size: int) -> pd.Series:
    """Legacy single-bucket pooling (kept for callers/tests; the screen uses the parent hierarchy)."""
    counts = groups.value_counts()
    small = set(counts[counts < min_group_size].index)
    return groups.where(~groups.isin(small), POOLED_GROUP)


@lru_cache(maxsize=4)
def _load_parent_map(path: str) -> dict[str, str]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "parent_group" not in df.columns:
        return {}
    return {str(g).strip(): str(p).strip() for g, p in zip(df["group"], df["parent_group"], strict=True) if p.strip()}


def load_parent_map(path: Path | None = None) -> dict[str, str]:
    """group -> parent group from ``configs/sector_groups.csv`` (unknown groups fall under NON_FINANCIALS)."""
    return dict(_load_parent_map(str(path or CONFIG_DIR / "sector_groups.csv")))


def robust_z(values: pd.Series, clip: float = 3.0) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    med = x.median()
    mad = (x - med).abs().median()
    if pd.isna(mad) or mad == 0:
        spread = x.std()
        if pd.isna(spread) or spread == 0:
            return pd.Series(np.where(x.notna(), 0.0, np.nan), index=x.index)
        z = (x - med) / spread
    else:
        z = (x - med) / (MAD_SCALE * mad)
    return z.clip(-clip, clip)


def _pool_labels(
    group: pd.Series, parent: pd.Series, valid: pd.Series, min_size: int, split: pd.Series | None = None
) -> pd.Series:
    """Per-row pooling key for one feature: own group, else parent, else all names (counts of non-null values).

    With ``split`` (accounting basis) every level is taken within the basis, and rows whose basis has fewer
    than ``min_size`` values in total get the NaN label.
    """
    sfx = "" if split is None else "|" + split.astype(str)
    g, p = group + sfx, parent + sfx
    everyone = pd.Series(ALL_GROUP, index=group.index) + sfx
    n_g, n_p, n_all = (valid.groupby(k).transform("sum") for k in (g, p, everyone))
    label = g.where(n_g >= min_size, p.where(n_p >= min_size, everyone))
    if split is not None:
        label = label.where(~((label == everyone) & (n_all < min_size)), _NAN_LABEL)
    return label


def normalise_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    group_col: str = "sector_group",
    min_group_size: int = 5,
    clip: float = 3.0,
    parent_map: dict[str, str] | None = None,
    basis_col: str = "basis",
) -> pd.DataFrame:
    """Add ``z_<feature>`` columns, oriented (higher = better) and robust-z-scored within pooled peer groups."""
    out = df.copy()
    pm = load_parent_map() if parent_map is None else parent_map
    group = out[group_col].fillna("OTHER").astype(str)
    parent = group.map(lambda g: pm.get(g, DEFAULT_PARENT))
    rows = pd.Series(1, index=out.index)
    n_g, n_p = rows.groupby(group).transform("sum"), rows.groupby(parent).transform("sum")
    out["norm_group"] = group.where(n_g >= min_group_size, parent.where(n_p >= min_group_size, ALL_GROUP))
    basis = out[basis_col].fillna("").astype(str) if basis_col in out.columns else pd.Series("", index=out.index)
    for col in feature_cols:
        sign = FEATURE_SIGN.get(col, 1.0)
        oriented = pd.to_numeric(out[col], errors="coerce") * sign
        if col == "adv_6m_tl":
            oriented = np.log(oriented.where(oriented > 0))
        split = None
        if col in GROWTH_FEATURES and len(set(basis[oriented.notna() & (basis != "")])) > 1:
            split = basis  # nominal and TMS-29 filers coexist in this month: compare within basis only
        label = _pool_labels(group, parent, oriented.notna(), min_group_size, split)
        z = oriented.groupby(label).transform(lambda s: robust_z(s, clip))
        out[f"z_{col}"] = z.where(label != _NAN_LABEL)
    return out
