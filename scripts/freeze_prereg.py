"""Fill the freeze record of docs/preregistration.md from the current store, config and ledger.

Run once, right before tagging, after the final deterministic backtest:
    uv run python scripts/freeze_prereg.py det-v1
    BISTMAS_CONFIG=configs/prereg_v2.toml uv run python scripts/freeze_prereg.py det-v2 preregistration_v2.md prereg-v2
"""

from __future__ import annotations

import re
import sys

from bistmas.shared.config import REPO_ROOT, load_settings
from bistmas.shared.ledger import config_hash, git_commit, read_ledger, store_vintages


def main(det_run: str, doc_name: str = "preregistration.md", tag: str = "prereg-v1") -> None:
    settings = load_settings()
    vint = store_vintages(settings)
    runs = [e for e in read_ledger(settings) if e.get("kind") == "deterministic" and e.get("run") == det_run]
    if not runs:
        raise SystemExit(f"no deterministic ledger line for run {det_run!r}; run the backtest first")
    det = runs[-1]
    doc = REPO_ROOT / "docs" / doc_name
    text = doc.read_text(encoding="utf-8")
    rows = {
        "tag / commit": f"`{tag}` @ `{git_commit()[:12]}` (tag applied to the commit that contains this table)",
        "features vintage (`features_monthly.parquet` sha256[:16])": f"`{vint['features_monthly']}`",
        "prices vintage": f"`{vint['prices_daily']}`",
        "statements vintage": f"`{vint['statements_long']}`",
        "config hash (`bistmas.shared.ledger.config_hash`)": f"`{config_hash(settings)}`",
        "deterministic run recorded before the freeze": (
            f"`{det_run}` (ledger line {det['ts']}, code `{det['code_commit'][:12]}`)"
        ),
    }
    for item, value in rows.items():
        pat = re.compile(rf"^\| {re.escape(item)} \| .*\|$", re.M)
        assert pat.search(text), item
        text = pat.sub(f"| {item} | {value} |", text)
    text = text.replace(
        "Status: **DRAFT** — becomes binding when the tag `prereg-v1` exists;",
        "Status: **FROZEN** (tag `prereg-v1`);",
    )
    doc.write_text(text, encoding="utf-8")
    print("freeze record written:", rows)


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["det-v1"]))
