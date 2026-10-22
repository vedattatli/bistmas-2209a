"""Security master: one row per stock code, keyed by MKK member oid, with sector group and statement template.

Inputs are the two KAP reference frames from :mod:`bistmas.data.clients.kap` plus two hand-curated
tables:

* ``configs/sector_groups.csv`` (``kap_main_sector, group, template, note``) collapses KAP sectors into
  ~8 groups for sector-relative normalisation. KAP names are matched after Turkish-aware upper-casing
  (``i -> İ``, ``ı -> I``, then ``İ`` folded to ``I`` so dotted/dotless spellings compare equal). Banks and
  insurers live under the KAP main sector "MALİ KURULUŞLAR", so the *sub*-sector is consulted first:
  any sector containing ``BANKA`` -> BANKS/bank, ``SİGORTA`` -> INSURANCE/insurer; then synonyms
  (:data:`KAP_SECTOR_SYNONYMS`), exact keys, and a first-word stem match; anything else -> group OTHER,
  template industrial (logged).
* ``configs/ticker_aliases.csv`` (``old_ticker, new_ticker, effective_date, kind, note``) adds one row per
  historical code with ``is_current=False``, carrying the oid (and sector data) of its current successor
  when that ticker is on KAP today, else NaN.

Output columns (``COLUMNS``): ``mkk_member_oid, ticker, title, city, main_sector, sector, sector_no,
sector_group, template, is_current, note``; ``template`` is one of :data:`TEMPLATES`. ``note`` is the alias
``kind`` for alias rows, ``"no_kap_sector"`` for current codes without a ``/tr/Sektorler`` row (bond-only
issuers without traded equity: ~130 codes on 2026-09-23) and ``""`` otherwise. Codes that are not 4-5
uppercase letters (bank instrument codes such as ``TGB``) are dropped by default (``equity_only=True``).
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from bistmas.shared.codes import EQUITY_CODE_RE
from bistmas.shared.config import CONFIG_DIR
from bistmas.shared.log import get_logger
from bistmas.shared.store import read_parquet, write_parquet

log = get_logger("data.security_master")

COLUMNS = [
    "mkk_member_oid",
    "ticker",
    "title",
    "city",
    "main_sector",
    "sector",
    "sector_no",
    "sector_group",
    "template",
    "is_current",
    "note",
    "successor_ticker",
    "successor_oid",
    "alias_kind",
    "alias_effective_date",
]
ALIAS_COLUMNS = ["successor_ticker", "successor_oid", "alias_kind", "alias_effective_date"]
TEMPLATES = frozenset({"industrial", "bank", "insurer", "other_financial"})
OTHER_GROUP = "OTHER"
DEFAULT_TEMPLATE = "industrial"
STORE_FILENAME = "security_master.parquet"
EQUITY_TICKER_RE = EQUITY_CODE_RE

# KAP sector names that differ from the keys of configs/sector_groups.csv (both sides normalised).
KAP_SECTOR_SYNONYMS: dict[str, str] = {
    "BANKALAR": "BANKACILIK",
    "SİGORTA ŞİRKETLERİ": "SİGORTA",
    "HOLDİNGLER VE YATIRIM ŞİRKETLERİ": "HOLDİNG VE YATIRIM",
    "GAYRİMENKUL YATIRIM ORTAKLIKLARI": "GAYRİMENKUL",
    "GAYRİMENKUL FAALİYETLERİ": "GAYRİMENKUL",
    "ELEKTRİK GAZ VE SU": "ENERJİ",
    "MADENCİLİK VE TAŞ OCAKÇILIĞI": "MADENCİLİK",
    "OTELLER VE LOKANTALAR": "KONAKLAMA VE YİYECEK",
}
_BANK_KEYWORD = "BANKA"
_INSURER_KEYWORD = "SİGORTA"
_STEM_MIN_LEN = 4


def normalise_sector(text: object) -> str:
    """Turkish-aware upper-case key: NFC, ``i -> İ``, ``ı -> I``, upper(), ``İ -> I``, single spaces."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = unicodedata.normalize("NFC", str(text))
    s = s.replace("i", "İ").replace("ı", "I").upper().replace("İ", "I")
    return " ".join(s.split())


class SectorGroupMapper:
    """Resolve (KAP main sector, KAP sub sector) to ``(group, template, matched_key)``."""

    def __init__(self, groups: pd.DataFrame) -> None:
        self._rows: dict[str, tuple[str, str]] = {}
        for rec in groups.to_dict("records"):
            key = normalise_sector(rec["kap_main_sector"])
            template = str(rec["template"]).strip()
            if template not in TEMPLATES:
                raise ValueError(f"sector_groups.csv: template {template!r} for {key!r} not in {sorted(TEMPLATES)}")
            self._rows[key] = (str(rec["group"]).strip(), template)
        self._synonyms = {normalise_sector(k): normalise_sector(v) for k, v in KAP_SECTOR_SYNONYMS.items()}
        self._bank = self._first_key_containing(normalise_sector(_BANK_KEYWORD)) or ("BANKS", "bank")
        self._insurer = self._first_key_containing(normalise_sector(_INSURER_KEYWORD)) or ("INSURANCE", "insurer")

    def _first_key_containing(self, needle: str) -> tuple[str, str] | None:
        for key, value in self._rows.items():
            if needle in key:
                return value
        return None

    def resolve(self, main_sector: object, sector: object) -> tuple[str, str, str | None]:
        """Return ``(group, template, matched_key)``; ``matched_key`` is None for the OTHER fallback."""
        candidates = [normalise_sector(sector), normalise_sector(main_sector)]
        bank, insurer = normalise_sector(_BANK_KEYWORD), normalise_sector(_INSURER_KEYWORD)
        for cand in candidates:
            if bank in cand:
                return (*self._bank, cand)
            if insurer in cand:
                return (*self._insurer, cand)
        for cand in candidates:
            key = self._synonyms.get(cand, cand)
            if key in self._rows:
                return (*self._rows[key], key)
        for cand in candidates:
            words = cand.split()
            for key, value in self._rows.items():
                stem = key.split()[0] if key else ""
                if len(stem) >= _STEM_MIN_LEN and any(w.startswith(stem) for w in words):
                    return (*value, key)
        return OTHER_GROUP, DEFAULT_TEMPLATE, None


def load_sector_groups(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = {"kap_main_sector", "group", "template"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return df


def load_sector_overrides(path: Path) -> pd.DataFrame:
    """Read ``configs/sector_overrides.csv`` (``ticker, kap_main_sector, kap_sub_sector, source, note``)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = {"ticker", "kap_main_sector", "kap_sub_sector"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df["ticker"] = df["ticker"].str.strip().str.upper()
    dup = df[df.duplicated(subset=["ticker"], keep=False)]
    if not dup.empty:
        raise ValueError(f"{path}: duplicate tickers {sorted(dup['ticker'].unique())}")
    return df


def load_ticker_aliases(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = {"old_ticker", "new_ticker", "kind"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    for col in ("old_ticker", "new_ticker"):
        df[col] = df[col].str.strip().str.upper()
    return df


def build_security_master(
    companies_df: pd.DataFrame,
    sectors_df: pd.DataFrame,
    sector_groups_csv: Path,
    aliases_csv: Path,
    equity_only: bool = True,
    sector_overrides_csv: Path | None = None,
) -> pd.DataFrame:
    """Combine KAP companies + sectors with the curated sector groups and ticker aliases.

    ``companies_df`` / ``sectors_df`` are the frames returned by ``KapClient.fetch_companies`` and
    ``KapClient.fetch_sectors``. Rows are joined on (``mkk_member_oid``, ``ticker``); a company without a
    sector row keeps NaN sector columns and falls into group OTHER. Returns a frame with :data:`COLUMNS`,
    current rows first (sorted by ticker) followed by alias rows.

    ``configs/sector_overrides.csv`` (or ``sector_overrides_csv``) supplies KAP sectors for codes KAP no
    longer lists (delisted or merged-away issuers, whose alias rows would otherwise fall into group OTHER and
    be normalised against the wrong peers); it is resolved through the same mapper and applied to the row of
    that ticker, current or alias.
    """
    mapper = SectorGroupMapper(load_sector_groups(sector_groups_csv))
    aliases = load_ticker_aliases(aliases_csv)

    companies = companies_df.copy()
    companies["ticker"] = companies["ticker"].astype("str").str.strip().str.upper()
    if equity_only:
        keep = companies["ticker"].str.fullmatch(EQUITY_TICKER_RE.pattern).fillna(False)
        dropped = sorted(companies.loc[~keep, "ticker"].unique())
        if dropped:
            log.info("dropping %d non-equity codes: %s", len(dropped), ", ".join(dropped))
        companies = companies[keep]

    sectors = sectors_df[["mkk_member_oid", "ticker", "main_sector", "sector", "sector_no"]].drop_duplicates(
        subset=["mkk_member_oid", "ticker"]
    )
    df = companies[["mkk_member_oid", "ticker", "title", "city"]].merge(
        sectors, on=["mkk_member_oid", "ticker"], how="left"
    )

    dup = df[df.duplicated(subset=["ticker"], keep=False)]
    if not dup.empty:
        raise ValueError(f"security master: ticker mapped to several members: {sorted(dup['ticker'].unique())}")

    resolved = [mapper.resolve(m, s) for m, s in zip(df["main_sector"], df["sector"], strict=True)]
    df["sector_group"] = [r[0] for r in resolved]
    df["template"] = [r[1] for r in resolved]
    unmapped = df.loc[[r[2] is None for r in resolved], ["ticker", "main_sector", "sector"]]
    if not unmapped.empty:
        combos = unmapped.groupby(["main_sector", "sector"], dropna=False).size()
        log.warning(
            "%d tickers fall into sector group %s (template %s): %s",
            len(unmapped),
            OTHER_GROUP,
            DEFAULT_TEMPLATE,
            "; ".join(f"{m} / {s}: {n}" for (m, s), n in combos.items()),
        )
    df["is_current"] = True
    # IGS members without a /tr/Sektorler row are bond-only issuers (banks, factoring, leasing, ...)
    # without traded equity; keep them (they are KAP members) but flag them for the universe filter.
    no_sector = df["main_sector"].isna()
    df["note"] = pd.Series(["no_kap_sector" if x else "" for x in no_sector], index=df.index, dtype="str")
    if no_sector.any():
        log.info("%d current codes have no KAP sector row (note=no_kap_sector)", int(no_sector.sum()))
    for col in ALIAS_COLUMNS:
        df[col] = ""
    df = df.sort_values("ticker", kind="stable")

    current_by_ticker = df.set_index("ticker")
    alias_rows = []
    for rec in aliases.to_dict("records"):
        old, new, kind = rec["old_ticker"], rec["new_ticker"], str(rec["kind"]).strip()
        eff = "" if pd.isna(rec.get("effective_date")) else str(rec.get("effective_date"))
        if old in current_by_ticker.index:
            log.warning("alias %s -> %s skipped: %s is a current KAP code", old, new, old)
            continue
        row: dict[str, Any] = {col: "" for col in COLUMNS}
        row.update(
            {
                "ticker": old,
                "is_current": False,
                "note": kind,
                "successor_ticker": new,
                "alias_kind": kind,
                "alias_effective_date": eff,
            }
        )
        row["sector_group"], row["template"] = OTHER_GROUP, DEFAULT_TEMPLATE
        if new in current_by_ticker.index:
            base = current_by_ticker.loc[new]
            row["successor_oid"] = str(base["mkk_member_oid"])
            if kind == "rename":
                # same legal entity: the old code shares the successor's oid, title and sector
                for col in (
                    "mkk_member_oid",
                    "title",
                    "city",
                    "main_sector",
                    "sector",
                    "sector_no",
                    "sector_group",
                    "template",
                ):
                    row[col] = base[col]
            else:
                # merger / acquisition: a distinct issuer whose own KAP oid must come from the delisted list
                row["title"] = ""
        else:
            log.warning("alias %s -> %s: successor %s not on KAP; successor_oid left empty", old, new, new)
        alias_rows.append(row)
    alias_df = pd.DataFrame(alias_rows, columns=COLUMNS).sort_values("ticker", kind="stable")

    out = pd.concat([df[COLUMNS], alias_df], ignore_index=True)
    out = _apply_sector_overrides(out, mapper, sector_overrides_csv)
    for col in COLUMNS:
        if col != "is_current":
            out[col] = out[col].fillna("").astype("str")
    out["is_current"] = out["is_current"].astype(bool)
    bad = set(out["template"].unique()) - TEMPLATES
    if bad:
        raise ValueError(f"unexpected templates {sorted(bad)}")
    log.info(
        "security master: %d current + %d alias rows; groups: %s",
        int(out["is_current"].sum()),
        int((~out["is_current"]).sum()),
        out.loc[out["is_current"], "sector_group"].value_counts().to_dict(),
    )
    return out.reset_index(drop=True)


def _apply_sector_overrides(out: pd.DataFrame, mapper: SectorGroupMapper, path: Path | None) -> pd.DataFrame:
    path = Path(path) if path is not None else CONFIG_DIR / "sector_overrides.csv"
    if not path.exists():
        return out
    for rec in load_sector_overrides(path).to_dict("records"):
        idx = out.index[out["ticker"] == rec["ticker"]]
        if len(idx) == 0:
            continue
        group, template, key = mapper.resolve(rec["kap_main_sector"], rec["kap_sub_sector"])
        if key is None:
            raise ValueError(f"{path}: sector of {rec['ticker']} ({rec['kap_main_sector']!r}) maps to no group")
        out.loc[idx, ["main_sector", "sector", "sector_group", "template"]] = [
            rec["kap_main_sector"],
            rec["kap_sub_sector"],
            group,
            template,
        ]
    return out


def write_security_master(df: pd.DataFrame, store_dir: Path) -> Path:
    """Write ``<store_dir>/security_master.parquet`` atomically and return its path."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"security master is missing columns {missing}")
    return write_parquet(df[COLUMNS], Path(store_dir) / STORE_FILENAME)


def load_security_master(store_dir: Path) -> pd.DataFrame:
    """Read ``<store_dir>/security_master.parquet`` (raises FileNotFoundError when it was never built)."""
    path = Path(store_dir) / STORE_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run the universe fetch first")
    return read_parquet(path)
