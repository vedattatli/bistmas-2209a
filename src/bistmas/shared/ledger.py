"""Append-only experiment ledger (``data/runs/ledger.jsonl``).

Every backtest, jury, audit and forward run appends one line with the code commit, the config and
prompt hashes, the model digest, the store vintages and a metric summary. The ledger is the trial
count behind the deflated Sharpe ratio and the "number of configurations examined" line of the
pre-registration: nothing that was run can disappear from it.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from bistmas.shared.config import CONFIG_DIR, REPO_ROOT, Settings
from bistmas.shared.hashing import sha256_file, sha256_json
from bistmas.shared.store import utc_now_iso

LEDGER_FILE = "ledger.jsonl"
STORE_VINTAGE_FILES = (
    "features_monthly.parquet",
    "prices_daily.parquet",
    "statements_long.parquet",
    "universe_membership.parquet",
)
# hand-curated inputs that change the canonical membership / sectors without touching Settings
CONFIG_VINTAGE_FILES = (
    "ticker_aliases.csv",
    "universe_overrides.csv",
    "index_changes_supplement.csv",
    "sector_overrides.csv",
    "sector_groups.csv",
    "adjustment_overrides.csv",
)


def git_commit(cwd: Path = REPO_ROOT) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - no git is not an error for the ledger
        return ""


def ledger_path(settings: Settings) -> Path:
    return settings.paths.runs / LEDGER_FILE


def config_hash(settings: Settings) -> str:
    """Hash of every parameter except machine-specific paths, so the same config hashes alike on any host."""
    dumped = settings.model_dump(mode="json")
    dumped.pop("root", None)
    dumped.pop("paths", None)
    return sha256_json(dumped)[:12]


def store_vintages(settings: Settings) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in STORE_VINTAGE_FILES:
        p = settings.paths.store / name
        out[name.split(".")[0]] = sha256_file(p)[:16] if p.exists() else None
    for name in CONFIG_VINTAGE_FILES:
        p = CONFIG_DIR / name
        out["config_" + name.split(".")[0]] = sha256_file(p)[:16] if p.exists() else None
    return out


def record_run(
    settings: Settings,
    kind: str,
    run: str,
    strategies: dict[str, dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one ledger line and return it. ``strategies`` maps label -> compact metric summary."""
    entry: dict[str, Any] = {
        "ts": utc_now_iso(),
        "kind": kind,
        "run": run,
        "code_commit": git_commit(),
        "config_hash": config_hash(settings),
        "config_file": sha256_file(CONFIG_DIR / "base.toml")[:12] if (CONFIG_DIR / "base.toml").exists() else None,
        "prompt_version": settings.jury.prompt_version,
        "model": settings.jury.model,
        "model_digest": settings.jury.model_digest,
        "store_vintages": store_vintages(settings),
        "strategies": strategies or {},
        **(extra or {}),
    }
    path = ledger_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    return entry


def read_ledger(settings: Settings) -> list[dict[str, Any]]:
    path = ledger_path(settings)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _trial_label(label: str, top_n_screen: int) -> str:
    """Portfolio identity of a strategy label. The jury only reorders the screen's Top-N, so the hybrid
    portfolio with N == top_n_screen is the deterministic one."""
    return f"det_top{top_n_screen}" if label == f"hybrid_top{top_n_screen}" else label


def trial_count(
    settings: Settings, kinds: tuple[str, ...] = ("deterministic", "hybrid"), extra_labels: Iterable[str] = ()
) -> int:
    """Distinct portfolio configurations evaluated so far: (strategy label, config hash) pairs.

    ``kind`` is not part of the key (``det_top3`` of a hybrid run is the ``det_top3`` of the deterministic
    run), re-running a configuration does not add a trial, and ``hybrid_top{top_n_screen}`` is the same
    portfolio as ``det_top{top_n_screen}``. The benchmarks are never trials. ``extra_labels`` are the
    strategies of the run being recorded now, counted under the current config hash, so the count a run
    reports already includes itself exactly once.
    """
    n = settings.scoring.top_n_screen
    seen: set[tuple[str, str]] = set()
    for e in read_ledger(settings):
        if e.get("kind") not in kinds:
            continue
        for label in e.get("strategies", {}):
            if not label.startswith("bench"):
                seen.add((_trial_label(str(label), n), str(e.get("config_hash"))))
    current = config_hash(settings)
    for label in extra_labels:
        if not label.startswith("bench"):
            seen.add((_trial_label(str(label), n), current))
    return len(seen)
