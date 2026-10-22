"""Typed configuration loaded from configs/base.toml (+ optional configs/local.toml overlay).

Every parameter that can change a result lives in the TOML files so that a run is reproducible
from (code commit, config hash, data vintage). Secrets never go here; use environment variables.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"


class PathsConfig(BaseModel):
    data: Path = Path("data")
    raw: Path = Path("data/raw")
    store: Path = Path("data/store")
    runs: Path = Path("data/runs")

    def resolve(self, root: Path) -> PathsConfig:
        return PathsConfig(
            data=root / self.data,
            raw=root / self.raw,
            store=root / self.store,
            runs=root / self.runs,
        )


class UniverseConfig(BaseModel):
    provider: str = "official"
    index: str = "XU100"
    min_months_listed: int = 12
    min_adv_tl_6m: float = 5_000_000
    one_class_per_issuer: bool = True


class RebalanceConfig(BaseModel):
    frequency: str = "monthly"
    first_decision: str = "2017-12-29"
    last_decision: str = "2024-11-29"
    decision_cutoff_time: str = "18:00"
    execution: str = "next_session_vwap"


class ScoringConfig(BaseModel):
    sector_group_min_size: int = 5
    zscore_method: str = "median_mad"
    zscore_clip: float = 3.0
    composite_weights: dict[str, float] = Field(
        default_factory=lambda: {"fundamental": 1.0, "valuation": 1.0, "momentum_risk": 1.0}
    )
    top_n_screen: int = 10
    min_listed_sessions: int = 250  # listing-age filter (sessions since listed_since)
    adv_percentile_floor: float = 0.10  # liquidity floor = max(universe.min_adv_tl_6m, this quantile of members' ADV)
    max_statement_age_days: int = 400  # latest filing older than this (period_end to decision date) -> stale


class JuryConfig(BaseModel):
    model: str = "qwen3.5:9b"
    model_digest: str = ""
    host: str = "http://127.0.0.1:11434"
    num_ctx: int = 8192
    num_predict: int = 700
    temperature: float = 0.0
    seed: int = 20260923
    think: bool = False
    prompt_version: str = "v1"
    theta: float = 0.75
    max_validation_retries: int = 1
    min_votes: int = 2  # a candidate needs at least this many valid agent scores to get a J
    agents: list[str] = Field(default_factory=lambda: ["quality_growth", "value", "risk_timing"])

    def sampling_options(self) -> dict[str, Any]:
        """Options passed to Ollama on every call; part of the jury cache key."""
        return {
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "temperature": self.temperature,
            "seed": self.seed,
            "think": self.think,
        }


class BacktestConfig(BaseModel):
    top_n: list[int] = Field(default_factory=lambda: [3, 5, 10])
    headline_top_n: int = 3
    cost_bps_commission: float = 10
    cost_bps_slippage: float = 20
    cost_sensitivity_bps: list[float] = Field(default_factory=lambda: [0, 30, 60, 100])
    delisting_haircut: float = 0.0
    haircut_sensitivity: float = 0.20  # pre-registered rerun with involuntary-exit haircut
    rf_source: str = "tcmb_policy_rate"
    random_jury_seeds: int = 500
    theta_step: float = 0.05  # BUY-only sensitivity: theta +- step
    newey_west_lags: int = 3


class AuditConfig(BaseModel):
    """H2 audits that are off by default (``bistmas audit --robustness`` / ``--rubric-export``)."""

    robustness_months: int = 6  # decision months of the run that are re-scored (stratified over time)
    robustness_noise_z: float = 0.25  # uniform noise in [-x, +x] added to every z-score of the payload
    robustness_top_k: int = 3  # Top-k overlap (headline portfolio size)
    robustness_seed: int = 20260923
    robustness_paraphrase_version: str = "v2_paraphrase"
    rubric_n: int = 30  # rationales in the blind human rubric
    rubric_seed: int = 20260923


class ForwardConfig(BaseModel):
    enabled: bool = False
    email_to: str = ""
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587


class MacroConfig(BaseModel):
    """EVDS macro series. ``cpi_series`` is the headline index; when it stops publishing, ``cpi_continuation_series``
    (a re-based successor) is spliced on with the ratio estimated on the overlapping months. Empty disables."""

    cpi_series: str = "TP.FG.J0"
    cpi_continuation_series: str = "TP.TUKFIY2025.GENEL"
    cpi_stale_days: int = 60  # cpi_pit is NaN (and a warning is logged) once the latest print is older than this


class Settings(BaseModel):
    root: Path = REPO_ROOT
    paths: PathsConfig = Field(default_factory=PathsConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    rebalance: RebalanceConfig = Field(default_factory=RebalanceConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    jury: JuryConfig = Field(default_factory=JuryConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    forward: ForwardConfig = Field(default_factory=ForwardConfig)
    macro: MacroConfig = Field(default_factory=MacroConfig)
    raw_config: dict[str, Any] = Field(default_factory=dict, exclude=True)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_settings(base: Path | None = None, local: Path | None = None) -> Settings:
    """Load configs/base.toml and overlay configs/local.toml (or $BISTMAS_CONFIG) if present."""
    base = base or CONFIG_DIR / "base.toml"
    local = local or Path(os.environ.get("BISTMAS_CONFIG", CONFIG_DIR / "local.toml"))
    with open(base, "rb") as fh:
        cfg = tomllib.load(fh)
    if local.exists():
        with open(local, "rb") as fh:
            cfg = _deep_merge(cfg, tomllib.load(fh))
    settings = Settings(
        paths=PathsConfig(**cfg.get("paths", {})).resolve(REPO_ROOT),
        universe=UniverseConfig(**cfg.get("universe", {})),
        rebalance=RebalanceConfig(**cfg.get("rebalance", {})),
        scoring=ScoringConfig(**cfg.get("scoring", {})),
        jury=JuryConfig(**cfg.get("jury", {})),
        backtest=BacktestConfig(**cfg.get("backtest", {})),
        forward=ForwardConfig(**cfg.get("forward", {})),
        macro=MacroConfig(**cfg.get("macro", {})),
        raw_config=cfg,
    )
    for path in (settings.paths.raw, settings.paths.store, settings.paths.runs):
        path.mkdir(parents=True, exist_ok=True)
    return settings
