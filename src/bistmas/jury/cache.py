"""Content-addressed jury cache: one JSON file per call under ``data/runs/jury_cache/``.

Key = sha256(payload text, system prompt hash, schema hash, model digest, sampling options). A hit is
served only when the stored request hash equals the freshly computed one, so a rebuilt feature store,
a re-pulled model or a changed prompt can never reuse a stale verdict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bistmas.shared.hashing import sha256_json, sha256_text


def cache_key(payload_text: str, prompt_hash: str, schema_hash: str, model_digest: str, options: dict[str, Any]) -> str:
    return sha256_json(
        {
            "payload": payload_text,
            "prompt": prompt_hash,
            "schema": schema_hash,
            "model": model_digest,
            "options": options,
        }
    )


class JuryCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        p = self.path(key)
        if not p.exists():
            return None
        with open(p, encoding="utf-8") as fh:
            rec = json.load(fh)
        return rec if rec.get("key") == key else None

    def put(self, key: str, record: dict[str, Any]) -> None:
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({**record, "key": key}, fh, ensure_ascii=False, indent=1, default=str)
        tmp.replace(p)


def prompt_hash(text: str) -> str:
    return sha256_text(text)
