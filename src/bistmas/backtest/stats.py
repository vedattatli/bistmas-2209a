"""Inference on monthly active returns: Newey-West t-test, stationary block bootstrap, deflated Sharpe."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps


def newey_west_ttest(x: pd.Series, lags: int = 3) -> dict[str, Any]:
    """Mean of ``x`` with HAC (Bartlett kernel, ``lags``) standard error; two-sided p-value."""
    v = pd.Series(x).dropna().to_numpy(dtype=float)
    n = len(v)
    if n < 6:
        return {"mean": float(np.mean(v)) if n else np.nan, "t": np.nan, "p": np.nan, "n": n, "lags": lags}
    m = v.mean()
    e = v - m
    gamma0 = float(e @ e / n)
    s = gamma0
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        gk = float(e[k:] @ e[:-k] / n)
        s += 2 * w * gk
    se = np.sqrt(max(s, 1e-18) / n)
    t = m / se
    p = 2 * (1 - sps.t.cdf(abs(t), df=n - 1))
    return {"mean": float(m), "se": float(se), "t": float(t), "p": float(p), "n": n, "lags": lags}


def stationary_bootstrap_ci(
    x: pd.Series, n_boot: int = 2000, mean_block: float = 4.0, alpha: float = 0.05, seed: int = 20260923
) -> dict[str, Any]:
    """Politis-Romano stationary bootstrap CI for the mean (geometric block lengths, mean ``mean_block``)."""
    v = pd.Series(x).dropna().to_numpy(dtype=float)
    n = len(v)
    if n < 6:
        return {"lo": np.nan, "hi": np.nan, "n_boot": 0}
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        i = rng.integers(n)
        for j in range(n):
            if j > 0 and rng.random() >= p:
                i = (i + 1) % n
            else:
                i = rng.integers(n)
            idx[j] = i
        means[b] = v[idx].mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"lo": float(lo), "hi": float(hi), "n_boot": n_boot, "mean_block": mean_block}


def deflated_sharpe(
    sharpe_hat: float, n_obs: int, n_trials: int, skew: float = 0.0, kurt: float = 3.0, sharpe_var: float | None = None
) -> dict[str, Any]:
    """Bailey & López de Prado deflated Sharpe ratio probability (annualised inputs are converted per period)."""
    if n_trials < 1 or n_obs < 3 or np.isnan(sharpe_hat):
        return {"psr": np.nan, "sr_benchmark": np.nan}
    var = sharpe_var if sharpe_var is not None else 1.0 / max(n_obs - 1, 1)
    euler = 0.5772156649
    z = (
        (1 - euler) * sps.norm.ppf(1 - 1.0 / n_trials) + euler * sps.norm.ppf(1 - 1.0 / (n_trials * np.e))
        if n_trials > 1
        else 0.0
    )
    sr0 = np.sqrt(var) * z
    num = (sharpe_hat - sr0) * np.sqrt(n_obs - 1)
    den = np.sqrt(1 - skew * sharpe_hat + (kurt - 1) / 4 * sharpe_hat**2)
    psr = float(sps.norm.cdf(num / den)) if den > 0 else np.nan
    return {"psr": psr, "sr_benchmark": float(sr0), "n_trials": n_trials}
