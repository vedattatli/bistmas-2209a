"""Run certificate: proves what a run used and that nothing dated after the decision date entered it."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from bistmas.shared.config import REPO_ROOT, Settings
from bistmas.shared.hashing import sha256_file, sha256_json
from bistmas.shared.ledger import config_hash
from bistmas.shared.store import utc_now_iso


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--", "src", "configs", "prompts"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        return bool(out.strip())
    except Exception:  # noqa: BLE001
        return True


def pit_check(features: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, Any]:
    col = (
        pd.to_datetime(features["max_source_available_date"])
        if "max_source_available_date" in features.columns
        else pd.Series(dtype="datetime64[ns]")
    )
    late = features[col > pd.Timestamp(as_of)] if len(col) else features.iloc[0:0]
    return {
        "as_of": str(pd.Timestamp(as_of).date()),
        "rows": int(len(features)),
        "rows_after_as_of": int(len(late)),
        "ok": len(late) == 0,
    }


def build_certificate(
    settings: Settings,
    as_of: pd.Timestamp,
    features: pd.DataFrame,
    jury_summary: dict[str, Any] | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    store = settings.paths.store
    vintages = {}
    for name in (
        "prices_daily.parquet",
        "statements_long.parquet",
        "universe_membership.parquet",
        "security_master.parquet",
    ):
        p = store / name
        vintages[name] = sha256_file(p)[:16] if p.exists() else None
    cert = {
        "generated_at": utc_now_iso(),
        "code_commit": git_commit(),
        "code_dirty": git_dirty(),
        # same hash as the ledger / pre-registration (typed parameters, no machine paths); the raw-file hash and
        # the store path are kept for forensics only
        "config_hash": config_hash(settings),
        "raw_config_hash": sha256_json(settings.raw_config)[:16],
        "store_path": str(settings.paths.store),
        "store_vintages": vintages,
        "pit": pit_check(features, as_of),
        "model": {
            "name": settings.jury.model,
            "digest_pinned": settings.jury.model_digest,
            "options": settings.jury.sampling_options(),
        },
        "prompt_version": settings.jury.prompt_version,
        "jury": {k: jury_summary.get(k) for k in ("model_digest", "prompt_hashes", "abstain_rate", "records")}
        if jury_summary
        else None,
    }
    if extra:
        cert.update(extra)
    cert["ok"] = bool(cert["pit"]["ok"]) and (
        not jury_summary or jury_summary.get("model_digest") in (settings.jury.model_digest, None, "unknown")
    )
    return cert


def certificate_path(run_dir: Path) -> Path:
    return Path(run_dir) / "certificate.json"
