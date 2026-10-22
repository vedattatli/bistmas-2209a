"""End-to-end universe pipeline used by the CLI.

``fetch_universe_raw`` pulls every source into the immutable raw layer and materialises the two
intermediate tables the builder needs (index changes, anchors) plus the KAP company/sector frames
under ``data/store/``. ``build_universe`` runs offline from those tables: security master, official
membership roll-forward with validation, and the JSON report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from bistmas.data.clients.bist_announcements import build_changes_table, fetch_archive, list_non_periodic, parse_archive
from bistmas.data.clients.datastore_files import load_datastore_dir
from bistmas.data.clients.http import PoliteSession
from bistmas.data.clients.index_anchors import load_anchor_memberships
from bistmas.data.clients.kap import KapClient
from bistmas.data.security_master import build_security_master, write_security_master
from bistmas.data.universe.official import (
    build_official_universe,
    load_aliases,
    load_overrides,
    write_membership,
    write_report,
)
from bistmas.shared.calendar import TradingCalendar
from bistmas.shared.config import CONFIG_DIR, Settings
from bistmas.shared.log import get_logger
from bistmas.shared.store import read_json, read_parquet, write_json, write_parquet

log = get_logger("data.universe.build")

CHANGES_FILE = "index_changes.parquet"
CHANGES_META_FILE = "index_changes_meta.json"
ANCHORS_FILE = "index_anchors.parquet"
NON_PERIODIC_FILE = "index_non_periodic.parquet"
COMPANIES_FILE = "kap_companies.parquet"
SECTORS_FILE = "kap_sectors.parquet"
INDEX_CODES: tuple[str, ...] = ("XU100", "XU050", "XU030")


def fetch_universe_raw(settings: Settings, force: bool = False) -> dict[str, Any]:
    """Fetch (or serve from cache) everything the universe needs and store the intermediate tables."""
    raw, store = settings.paths.raw, settings.paths.store
    web = PoliteSession(min_interval_s=1.0, timeout_s=30.0)
    kap_session = PoliteSession(min_interval_s=0.6, timeout_s=30.0)

    archive = parse_archive(fetch_archive(web, raw, force=force))
    changes = build_changes_table(web, raw)
    write_parquet(_dates_to_str(changes, ["announcement_date"]), store / CHANGES_FILE)
    write_json({"missing_tables": changes.attrs.get("missing_tables", [])}, store / CHANGES_META_FILE)
    non_periodic = list_non_periodic(archive, start_year=2018)
    write_parquet(_dates_to_str(non_periodic, ["announcement_date"]), store / NON_PERIODIC_FILE)

    anchors = load_anchor_memberships(web, raw)
    write_parquet(_dates_to_str(anchors, ["snapshot_date", "composition_effective_date"]), store / ANCHORS_FILE)

    kap = KapClient(kap_session, raw / "kap")
    companies = kap.fetch_companies(force=force)
    sectors = kap.fetch_sectors(force=force)
    write_parquet(companies, store / COMPANIES_FILE)
    write_parquet(sectors, store / SECTORS_FILE)

    datastore = load_datastore_dir(raw)
    summary = {
        "changes_rows": int(len(changes)),
        "quarters": int(changes[["year", "quarter"]].drop_duplicates().shape[0]) if len(changes) else 0,
        "missing_tables": changes.attrs.get("missing_tables", []),
        "non_periodic_rows": int(len(non_periodic)),
        "anchor_rows": int(len(anchors)),
        "anchors": sorted(anchors["source_name"].unique().tolist()) if len(anchors) else [],
        "companies": int(len(companies)),
        "sector_rows": int(len(sectors)),
        "datastore_tables": {k: int(len(v)) for k, v in datastore.items()},
    }
    log.info("universe raw fetch complete: %s", summary)
    return summary


def build_universe(
    settings: Settings, calendar: TradingCalendar | None = None, end: str | None = None
) -> dict[str, Any]:
    """Build the security master and the official membership table from the stored intermediate tables."""
    store = settings.paths.store
    for name in (CHANGES_FILE, ANCHORS_FILE, COMPANIES_FILE, SECTORS_FILE):
        if not (store / name).exists():
            raise FileNotFoundError(f"{store / name} missing; run `bistmas fetch universe` first")

    companies = read_parquet(store / COMPANIES_FILE)
    sectors = read_parquet(store / SECTORS_FILE)
    master = build_security_master(
        companies, sectors, CONFIG_DIR / "sector_groups.csv", CONFIG_DIR / "ticker_aliases.csv"
    )
    write_security_master(master, store)

    changes = read_parquet(store / CHANGES_FILE)
    changes = _apply_supplement(changes, CONFIG_DIR / "index_changes_supplement.csv")
    meta = read_json(store / CHANGES_META_FILE) if (store / CHANGES_META_FILE).exists() else {}
    anchors = read_parquet(store / ANCHORS_FILE)
    overrides = load_overrides(CONFIG_DIR / "universe_overrides.csv")
    aliases = load_aliases(CONFIG_DIR / "ticker_aliases.csv")

    membership, report = build_official_universe(
        changes,
        anchors,
        overrides,
        aliases,
        index_codes=INDEX_CODES,
        calendar=calendar,
        end=end,
        missing_tables=meta.get("missing_tables", []),
    )
    write_membership(membership, store)
    write_report(report, store)

    window_end = pd.Timestamp(settings.raw_config.get("universe", {}).get("validation_window_end", "2025-12-31"))
    summary: dict[str, Any] = {
        "security_master_rows": int(len(master)),
        "validation_window_end": str(window_end.date()),
        "indices": {},
    }
    for code, info in report["indices"].items():
        if "error" in info:
            summary["indices"][code] = {"error": info["error"]}
            continue
        diag = info["diagnostics"]
        summary["indices"][code] = {
            "ok": info["ok"],
            "ok_within_window": _ok_within_window(info, window_end),
            "all_anchors_match": info["all_anchors_match"],
            "n_validations": info["n_validations"],
            "n_tickers": info["n_tickers"],
            "final_size": diag["final_size"],
            "exclude_missing": [f"{m['ticker']}@{m['year']}Q{m['quarter']}" for m in diag["exclude_missing"]],
            "include_duplicate": [f"{m['ticker']}@{m['year']}Q{m['quarter']}" for m in diag["include_duplicate"]],
            "unresolved_overrides": [m["ticker"] for m in diag["unresolved_overrides"]],
            "substitutes_promoted": [
                f"{m['ticker']}<-{m['replaces']}@{m['effective_date']}" for m in diag["substitutes_promoted"]
            ],
            "quarters_without_rows": [f"{q['year']}Q{q['quarter']}" for q in diag["quarters_without_rows"]],
            "validations": [
                {"as_of": str(v["as_of"]), "missing": v["missing"], "extra": v["extra"]} for v in info["validations"]
            ],
            "invariant_problems": info["invariant_problems"],
        }
    matrices_dir = settings.paths.raw / "datastore" / "index_constituents"
    if matrices_dir.is_dir():
        summary["datastore_validation"] = validate_against_constituent_matrices(membership, aliases, matrices_dir)
        dv = summary["datastore_validation"]
        if dv["mismatches"]:
            log.warning(
                "DataStore snapshot validation: %d of %d snapshots differ: %s",
                len(dv["mismatches"]),
                dv["snapshots"],
                dv["mismatches"][:5],
            )
        else:
            log.info("DataStore snapshot validation: all %d official snapshots reproduced", dv["snapshots"])
    log.info("universe build complete")
    return summary


def validate_against_constituent_matrices(
    membership: pd.DataFrame, aliases: pd.DataFrame, matrices_dir: Path, index_codes: tuple[str, ...] = INDEX_CODES
) -> dict[str, Any]:
    """Compare the rolled membership with every official DataStore constituent snapshot (product 3184).

    DataStore prints the CURRENT code of a renamed issuer in every column of a yearly file, so both sides are
    mapped to the successor code (rename aliases) before the comparison. Returns the number of snapshots
    compared and a list of ``{index_code, date, missing, extra}`` for the snapshots that differ.
    """
    from bistmas.data.clients.datastore_files import load_constituent_matrices

    matrices = load_constituent_matrices(matrices_dir)
    if matrices.empty:
        return {"snapshots": 0, "mismatches": [], "note": "no matrices"}
    ren = {
        str(r.old_ticker): str(r.new_ticker)
        for r in aliases.itertuples(index=False)
        if str(getattr(r, "kind", "")) == "rename"
    }

    def canon(t: str) -> str:
        seen: set[str] = set()
        while t in ren and t not in seen:
            seen.add(t)
            t = ren[t]
        return t

    mem = membership.copy()
    mem["effective_start"] = pd.to_datetime(mem["effective_start"])
    mem["effective_end"] = pd.to_datetime(mem["effective_end"])
    first = mem["effective_start"].min()
    mismatches: list[dict[str, Any]] = []
    n = 0
    for code in index_codes:
        official = matrices[matrices["index_code"] == code]
        ours_all = mem[mem["index_code"] == code]
        for d, g in official.groupby("snapshot_date"):
            if d < first:
                continue
            n += 1
            off = {canon(t) for t in g["ticker"]}
            live = ours_all[
                (ours_all["effective_start"] <= d)
                & ((ours_all["effective_end"] > d) | ours_all["effective_end"].isna())
            ]
            ours = {canon(t) for t in live["ticker"]}
            if off != ours:
                mismatches.append(
                    {
                        "index_code": code,
                        "date": str(pd.Timestamp(d).date()),
                        "missing": sorted(off - ours),
                        "extra": sorted(ours - off),
                    }
                )
    return {"snapshots": n, "mismatches": mismatches}


def _apply_supplement(changes: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Append hand-curated change rows (e.g. the 2020Q1 BIST 30/50 tables missing from the English page).

    Rows whose (year, quarter, index_code) already carry any row of the same action in ``changes`` are
    skipped, so the supplement never duplicates a parsed table.
    """
    if not path.exists():
        return changes
    sup = pd.read_csv(path, dtype=str).fillna("")
    if sup.empty:
        return changes
    sup["year"] = sup["year"].astype("int64")
    sup["quarter"] = sup["quarter"].astype("int64")
    existing = {
        (int(y), int(q), str(i), str(a))
        for y, q, i, a in changes[["year", "quarter", "index_code", "action"]].itertuples(index=False)
    }
    keep = [(int(r.year), int(r.quarter), r.index_code, r.action) not in existing for r in sup.itertuples(index=False)]
    added = sup[keep]
    skipped = sup[[not k for k in keep]]
    if not skipped.empty:
        log.warning("supplement: %d rows skipped because the parsed table already has that action", len(skipped))
    if added.empty:
        return changes
    log.info("supplement: %d rows added from %s", len(added), path.name)
    added = added[list(changes.columns)] if set(changes.columns) <= set(added.columns) else added
    out = pd.concat([changes, added], ignore_index=True)
    out.attrs = dict(changes.attrs)
    return out


def _ok_within_window(info: dict[str, Any], window_end: pd.Timestamp) -> bool:
    """True when every anchor validation dated on/before ``window_end`` matches and the roll is sound.

    Anchors after the window (e.g. the live 2026 list) may legitimately differ until the forward-run
    events (renames, watch-list transfers, new listings) are curated; they never affect the backtest.
    """
    inside = [v for v in info["validations"] if pd.Timestamp(v["as_of"]) <= window_end]
    return (
        bool(inside)
        and all(v["match"] for v in inside)
        and not info["invariant_problems"]
        and not info["quarters_without_rows"]
    )


def _dates_to_str(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Parquet-friendly copy: ``datetime.date`` object columns become ISO strings."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].map(lambda v: "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
    return out
