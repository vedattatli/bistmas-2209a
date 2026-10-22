"""Daily price store built from the immutable İş Yatırım ``HisseTekil`` raw layer.

Input: ``data/raw/isyatirim/<TICKER>/<YYYY>.json.gz`` (one vendor response per ticker-year, written by
the İş Yatırım client through :meth:`bistmas.data.clients.http.PoliteSession.get_cached`) plus the
``manifest.jsonl`` lines that record each file's sha256. This module never touches the network: it
parses the cached JSON with a small tolerant reader and rebuilds ``data/store/prices_daily.parquet``
from scratch every time (the vendor restates the adjusted series after every corporate action, so the
store is versioned by ``source_vintage`` instead of being appended to).

Vendor payload (verified 2026-09-23): ``{"value": [ {...31 fields...}, ... ]}``, one object per trading
day. The fields used and their store names:

==================  =================  ==========================================================
vendor field        store column       meaning
==================  =================  ==========================================================
HGDG_TARIH          date               session date (``DD-MM-YYYY`` string; epoch-ms and
                                       ``/Date(ms)/`` forms are accepted too)
HG_KAPANIS/AOF/     close_raw, vwap_raw, low_raw, high_raw, volume_raw
MIN/MAX/HACIM                          raw (unadjusted) close, VWAP, low, high, volume
HGDG_KAPANIS/AOF/   close_adj, vwap_adj, low_adj, high_adj, volume_adj
MIN/MAX/HACIM                          adjusted for rights issues, bonus issues AND cash dividends;
                                       restated retroactively after every event
SERMAYE             shares             paid-in capital in TL = share count at 1 TL nominal
PD, HAO_PD          mcap_tl, ffloat_mcap_tl   vendor market cap / free-float cap (TL)
PD_USD, HAO_PD_USD  mcap_usd, ffloat_mcap_usd
DD_DEGER            usdtry             USD/TRY of the day
END_DEGER           xu100              XU100 level as served (÷100 rescale on 2020-07-27, see
                                       ``index_levels``)
DOLAR_BAZLI_FIYAT   price_usd
==================  =================  ==========================================================

Derived columns:

* ``traded_value_tl`` — TL value traded. ``HG_HACIM`` is a **TL traded value**, not a share count
  (verified 2026-09-23 on real rows: ``DOLAR_HACIM * DD_DEGER == HG_HACIM`` exactly, ``HGDG_HACIM ==
  HG_HACIM`` on every row including across capital events, and ``HG_HACIM / HG_KAPANIS`` gives a
  plausible share turnover while ``HG_HACIM`` read as shares would exceed half the shares outstanding
  per day for THYAO). Hence ``traded_value_tl = volume_raw`` and ``volume_raw / vwap_raw`` is the share
  count. :func:`infer_volume_semantics` re-checks this on every build and logs a warning if the data
  disagree (``volume_semantics="shares"`` then uses ``volume_raw * vwap_raw``).
* ``adj_factor = close_adj / close_raw`` — piecewise constant; it steps on every ex-date.
* ``shares_aligned`` / ``mcap_tl_aligned`` — the vendor updates ``SERMAYE`` one session **before** the
  price ex-date, so ``PD = HG_KAPANIS * SERMAYE`` doubles for a day around bonus issues. A SERMAYE step
  on session *s* is shifted to *s+1* when the raw/adjusted return mismatch is on *s+1* rather than on
  *s* (see :func:`align_shares`); ``mcap_tl_aligned = close_raw * shares_aligned``. The vendor ``PD`` is
  kept untouched as ``mcap_tl``.
  The vendor may also update ``SERMAYE`` one session **after** the ex-date (ALBRK 2022-04-27, ENKAI
  2018-05-24): when the mismatch sits on *s-1* and its implied price ratio is compatible with the
  capital ratio, the new count is applied from *s-1* (:func:`align_shares`).
* ``configs/adjustment_overrides.csv`` (``ticker, ex_date, correct_factor_step``) repairs vendor
  adjustment artefacts: the adjusted price columns before ``ex_date`` are rescaled so that the
  ``adj_factor`` step on ``ex_date`` equals ``correct_factor_step`` (LRSHO 2019-09-17: the vendor step
  of 2.5616 for a 1.5x bonus issue). See :func:`apply_adjustment_overrides`.
* ``flag_capital_change`` — ``shares_aligned`` differs from the previous session (i.e. the price
  ex-date of a capital change, after the one-day-early correction).
* ``flag_adj_mismatch`` — ``|ret_raw - ret_adj| > 5 %`` on a session without a capital change
  (cash dividends above 5 % of the price, or vendor inconsistencies).
* ``flag_cancelled_session`` — 2023-02-08 (all trades cancelled after the earthquake). Rows are kept
  and flagged; :func:`load_prices` drops them by default.
* ``source_vintage`` — first 12 hex digits of the sha256 over the sorted ``(path, sha256)`` manifest
  entries of every raw file that went into the build.

Point-in-time note: raw columns (``close_raw`` ...) are what an investor saw on the day. Adjusted
columns and ``adj_factor`` embed later corporate actions by construction (the vendor restates them),
so they may only be used for *returns*, never as levels known on ``date``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bistmas.shared.calendar import CANCELLED_SESSIONS
from bistmas.shared.codes import EQUITY_CODE_RE
from bistmas.shared.config import CONFIG_DIR, Settings
from bistmas.shared.hashing import sha256_bytes, sha256_json, short
from bistmas.shared.log import get_logger
from bistmas.shared.store import read_manifest, read_parquet, read_raw_gzip, write_json, write_parquet

log = get_logger("data.prices")

RAW_SUBDIR = "isyatirim"
RAW_MANIFEST = "manifest.jsonl"
STORE_FILENAME = "prices_daily.parquet"
META_FILENAME = "prices_daily_meta.json"
RAW_FILE_RE = re.compile(r"^(?P<year>\d{4})\.json\.gz$")
VENDOR_TZ = "Europe/Istanbul"

ADJ_MISMATCH_TOL = 0.05
"""``flag_adj_mismatch`` threshold on ``|ret_raw - ret_adj|``."""
ALIGN_MISMATCH_MIN = 0.02
"""Minimum raw/adjusted return mismatch that identifies the price ex-date of a capital change."""
VOLUME_SEMANTICS_DEFAULT = "tl"
"""Verified meaning of ``HG_HACIM`` (see module docstring): a TL traded value, not a share count."""

# vendor field -> store column
FIELD_MAP: dict[str, str] = {
    "HG_KAPANIS": "close_raw",
    "HG_AOF": "vwap_raw",
    "HG_MIN": "low_raw",
    "HG_MAX": "high_raw",
    "HG_HACIM": "volume_raw",
    "HGDG_KAPANIS": "close_adj",
    "HGDG_AOF": "vwap_adj",
    "HGDG_MIN": "low_adj",
    "HGDG_MAX": "high_adj",
    "HGDG_HACIM": "volume_adj",
    "SERMAYE": "shares",
    "PD": "mcap_tl",
    "HAO_PD": "ffloat_mcap_tl",
    "PD_USD": "mcap_usd",
    "HAO_PD_USD": "ffloat_mcap_usd",
    "DD_DEGER": "usdtry",
    "END_DEGER": "xu100",
    "DOLAR_BAZLI_FIYAT": "price_usd",
    "DOLAR_HACIM": "volume_usd",
}
DATE_FIELDS: tuple[str, ...] = ("HGDG_TARIH", "HG_TARIH", "TARIH", "DD_TARIH", "END_TARIH")
VENDOR_COLUMNS: list[str] = ["ticker", "date", *FIELD_MAP.values()]

# Contract columns of data/store/prices_daily.parquet (docs/data_dictionary.md), in this order.
PRICE_COLUMNS: list[str] = [
    "ticker",
    "date",
    "close_raw",
    "vwap_raw",
    "low_raw",
    "high_raw",
    "volume_raw",
    "close_adj",
    "vwap_adj",
    "low_adj",
    "high_adj",
    "volume_adj",
    "shares",
    "mcap_tl",
    "ffloat_mcap_tl",
    "mcap_usd",
    "ffloat_mcap_usd",
    "usdtry",
    "xu100",
    "price_usd",
    "traded_value_tl",
    "adj_factor",
    "flag_capital_change",
    "flag_adj_mismatch",
    "flag_cancelled_session",
    "source_vintage",
    # extensions (ex-date-aligned share count and the market cap recomputed from it)
    "shares_aligned",
    "mcap_tl_aligned",
]
FLAG_COLUMNS: tuple[str, ...] = ("flag_capital_change", "flag_adj_mismatch", "flag_cancelled_session")
NUMERIC_COLUMNS: tuple[str, ...] = tuple(
    c for c in PRICE_COLUMNS if c not in {"ticker", "date", "source_vintage", *FLAG_COLUMNS}
)

_DATE_DMY_RE = re.compile(r"^(\d{1,2})[-./](\d{1,2})[-./](\d{4})")
_DATE_ISO_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")
_DATE_ASPNET_RE = re.compile(r"/Date\((-?\d+)")


class RawLayerError(RuntimeError):
    """Raised when the raw layer is missing or a payload cannot be parsed."""


# --------------------------------------------------------------------------- parsing
def parse_date_value(value: Any) -> pd.Timestamp:
    """Parse one vendor date (``DD-MM-YYYY``, ISO, ``/Date(ms)/`` or epoch ms/s); ``NaT`` when unknown."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    if isinstance(value, int | float) and not isinstance(value, bool):
        v = float(value)
        unit = "ms" if abs(v) > 1e11 else "s"
        # vendor epochs are Istanbul-local midnights expressed in UTC (1514840400000 = 2018-01-02 00:00 +03)
        ts = pd.Timestamp(v, unit=unit, tz="UTC").tz_convert(VENDOR_TZ)
        return ts.tz_localize(None).normalize()
    text = str(value).strip()
    if not text:
        return pd.NaT
    m = _DATE_ASPNET_RE.match(text)
    if m:
        return parse_date_value(int(m.group(1)))
    m = _DATE_DMY_RE.match(text)
    if m:
        d, mth, y = (int(x) for x in m.groups())
        try:
            return pd.Timestamp(year=y, month=mth, day=d)
        except ValueError:
            return pd.NaT
    m = _DATE_ISO_RE.match(text)
    if m:
        y, mth, d = (int(x) for x in m.groups())
        try:
            return pd.Timestamp(year=y, month=mth, day=d)
        except ValueError:
            return pd.NaT
    if text.isdigit():
        return parse_date_value(int(text))
    return pd.NaT


def _to_number(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return np.nan
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip().replace(" ", "")
    if not text:
        return np.nan
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".")  # 1.234,56
    elif "," in text:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_hissetekil(payload: bytes | str | dict[str, Any] | list[dict[str, Any]], ticker: str) -> pd.DataFrame:
    """Parse one ``HisseTekil`` response into a frame with :data:`VENDOR_COLUMNS`.

    Tolerant by design: unknown fields are ignored, missing fields become NaN, dates come from the
    first of :data:`DATE_FIELDS` that parses, rows without a date or without a raw close are dropped
    (counted in ``df.attrs['dropped_rows']``). An empty ``{"value": []}`` yields an empty frame.
    """
    if isinstance(payload, bytes | str):
        text = payload.decode("utf-8-sig") if isinstance(payload, bytes) else payload
        if not text.strip():
            raise RawLayerError(f"{ticker}: empty payload")
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RawLayerError(f"{ticker}: payload is not JSON ({exc})") from exc
    else:
        obj = payload
    rows = obj.get("value", obj.get("d", [])) if isinstance(obj, dict) else obj
    if rows is None:
        rows = []
    if not isinstance(rows, list):
        raise RawLayerError(f"{ticker}: unexpected payload shape {type(rows).__name__}")

    records: list[dict[str, Any]] = []
    dropped = 0
    for row in rows:
        if not isinstance(row, dict):
            dropped += 1
            continue
        ts = pd.NaT
        for field in DATE_FIELDS:
            if field in row:
                ts = parse_date_value(row[field])
                if not pd.isna(ts):
                    break
        close = _to_number(row.get("HG_KAPANIS"))
        if pd.isna(ts) or np.isnan(close):
            dropped += 1
            continue
        rec: dict[str, Any] = {"ticker": ticker, "date": ts}
        for field, col in FIELD_MAP.items():
            rec[col] = _to_number(row.get(field))
        records.append(rec)
    df = pd.DataFrame(records, columns=VENDOR_COLUMNS)
    df["ticker"] = df["ticker"].astype(str)
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    for col in FIELD_MAP.values():
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df.attrs["dropped_rows"] = dropped
    return df


# --------------------------------------------------------------------------- raw layer
def raw_dir_for(raw_dir: Path) -> Path:
    return Path(raw_dir) / RAW_SUBDIR


def list_raw_files(raw_dir: Path, tickers: Iterable[str] | None = None) -> list[tuple[str, int, Path]]:
    """Sorted ``(ticker, year, path)`` for every ``<TICKER>/<YYYY>.json.gz`` under ``raw_dir/isyatirim``.

    Directory names that are not equity codes (indices, scratch dirs) are skipped with a log line.
    """
    base = raw_dir_for(raw_dir)
    if not base.exists():
        return []
    wanted = {t.strip().upper() for t in tickers} if tickers is not None else None
    out: list[tuple[str, int, Path]] = []
    for tdir in sorted(p for p in base.iterdir() if p.is_dir()):
        ticker = tdir.name.upper()
        if not EQUITY_CODE_RE.fullmatch(ticker):
            log.info("skipping raw dir %s (not an equity code)", tdir.name)
            continue
        if wanted is not None and ticker not in wanted:
            continue
        for f in sorted(tdir.iterdir()):
            m = RAW_FILE_RE.match(f.name)
            if m:
                out.append((ticker, int(m.group("year")), f))
    return out


def _manifest_key(path: str | Path) -> str:
    parts = Path(str(path)).parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else str(path)


def load_manifest_index(raw_dir: Path) -> dict[str, dict[str, Any]]:
    """Latest manifest entry per ``<TICKER>/<YYYY>.json.gz`` from the layer-level and per-ticker manifests."""
    base = raw_dir_for(raw_dir)
    index: dict[str, dict[str, Any]] = {}
    if not base.exists():
        return index
    candidates = [base / RAW_MANIFEST, *sorted(base.glob(f"*/{RAW_MANIFEST}"))]
    for mpath in candidates:
        for entry in read_manifest(mpath):
            p = entry.get("path")
            if p and entry.get("sha256"):
                index[_manifest_key(p)] = entry
    return index


def read_raw_layer(raw_dir: Path, tickers: Iterable[str] | None = None) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Parse every cached ticker-year file; returns ``(vendor frame, manifest entries used)``.

    The frame has :data:`VENDOR_COLUMNS`, one row per (ticker, date) (later year files win on overlap),
    sorted by ticker and date. Each manifest entry is ``{"path": "<TICKER>/<YYYY>.json.gz", "sha256": ...,
    "fetched_at": ...}``; when a file has no manifest line its sha256 is computed from the payload.
    """
    files = list_raw_files(raw_dir, tickers)
    if not files:
        raise RawLayerError(f"no HisseTekil files under {raw_dir_for(raw_dir)}; run `bistmas fetch prices` first")
    manifest = load_manifest_index(raw_dir)
    frames: list[pd.DataFrame] = []
    used: list[dict[str, Any]] = []
    empty_files: list[str] = []
    dropped = 0
    for ticker, year, path in files:
        payload = read_raw_gzip(path)
        key = _manifest_key(path)
        entry = manifest.get(key)
        digest = entry["sha256"] if entry else sha256_bytes(payload)
        if entry is None:
            log.debug("%s has no manifest line; sha256 computed from the payload", key)
        used.append({"path": key, "sha256": digest, "fetched_at": (entry or {}).get("fetched_at", "")})
        df = parse_hissetekil(payload, ticker)
        dropped += int(df.attrs.get("dropped_rows", 0))
        if df.empty:
            empty_files.append(f"{ticker}/{year}")
            continue
        frames.append(df)
    if empty_files:
        log.info("%d empty ticker-year files (no rows): %s", len(empty_files), ", ".join(empty_files[:20]))
    if dropped:
        log.info("%d vendor rows dropped (no date or no raw close)", dropped)
    if not frames:
        out = pd.DataFrame(columns=VENDOR_COLUMNS)
        out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
        return out, used
    out = pd.concat(frames, ignore_index=True)
    before = len(out)
    out = out.drop_duplicates(subset=["ticker", "date"], keep="last")
    if len(out) != before:
        log.info("%d duplicate (ticker, date) rows collapsed across year files", before - len(out))
    out = out.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)
    return out, used


def source_vintage(manifest_entries: Sequence[dict[str, Any]]) -> str:
    """12-hex-digit sha256 prefix over the sorted ``(path, sha256)`` pairs of the raw files used."""
    pairs = sorted({(str(e["path"]), str(e["sha256"])) for e in manifest_entries})
    return short(sha256_json([{"path": p, "sha256": s} for p, s in pairs]), 12)


# --------------------------------------------------------------------------- semantics checks
def infer_volume_semantics(df: pd.DataFrame, tol: float = 0.01) -> dict[str, Any]:
    """Decide whether ``HG_HACIM`` is a share count or a TL value from the vendor's own columns.

    If volume is in shares, ``volume_raw * vwap_raw`` equals ``volume_adj * vwap_adj`` (adjustment scales
    volume up and price down by the same ratio) while ``volume_raw != volume_adj`` across capital
    events, and ``volume_usd * usdtry`` equals ``volume_raw * vwap_raw``. If volume is a TL value,
    ``volume_raw == volume_adj`` on every row and ``volume_usd * usdtry`` equals ``volume_raw`` itself.
    Returns ``{"semantics": "shares"|"tl"|"unknown", ...diagnostics}``.
    """
    d = df[(df["volume_raw"] > 0) & (df["vwap_raw"] > 0)]
    res: dict[str, Any] = {"n_rows": int(len(d))}
    if d.empty:
        res["semantics"] = "unknown"
        return res
    raw_value = d["volume_raw"] * d["vwap_raw"]
    adj_ok = d["volume_adj"] > 0
    if adj_ok.any():
        adj_value = d.loc[adj_ok, "volume_adj"] * d.loc[adj_ok, "vwap_adj"]
        rel = (adj_value / raw_value[adj_ok] - 1).abs()
        res["value_rel_diff_median"] = float(rel.median())
        res["value_rel_diff_p95"] = float(rel.quantile(0.95))
        res["share_of_rows_volume_adj_ne_raw"] = float(
            (d.loc[adj_ok, "volume_adj"] != d.loc[adj_ok, "volume_raw"]).mean()
        )
    usd_ok = (d["volume_usd"] > 0) & (d["usdtry"] > 0)
    if usd_ok.any():
        usd_tl = d.loc[usd_ok, "volume_usd"] * d.loc[usd_ok, "usdtry"]
        res["usd_vs_shares_value_rel_diff_median"] = float((usd_tl / raw_value[usd_ok] - 1).abs().median())
        res["usd_vs_raw_volume_rel_diff_median"] = float((usd_tl / d.loc[usd_ok, "volume_raw"] - 1).abs().median())
    votes_shares = 0
    votes_tl = 0
    if "value_rel_diff_median" in res:
        if res["share_of_rows_volume_adj_ne_raw"] > 0:
            if res["value_rel_diff_median"] < tol:
                votes_shares += 1
            else:
                votes_tl += 1
    if "usd_vs_shares_value_rel_diff_median" in res:
        if res["usd_vs_shares_value_rel_diff_median"] < res["usd_vs_raw_volume_rel_diff_median"]:
            votes_shares += 1
        else:
            votes_tl += 1
    if votes_shares > votes_tl:
        res["semantics"] = "shares"
    elif votes_tl > votes_shares:
        res["semantics"] = "tl"
    else:
        res["semantics"] = "unknown"
    return res


# --------------------------------------------------------------------------- transformations
def _returns(values: np.ndarray) -> np.ndarray:
    out = np.full(len(values), np.nan)
    if len(values) > 1:
        prev = values[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[1:] = np.where(prev > 0, values[1:] / prev - 1.0, np.nan)
    return out


LATE_RATIO_TOL = 0.03


def _updated_late(
    t: int, shares: np.ndarray, ret_raw: np.ndarray, ret_adj: np.ndarray, mismatch: np.ndarray, min_mismatch: float
) -> bool:
    """True when the SERMAYE step on *t* belongs to the price ex-date *t-1* (vendor one session late).

    Needs a raw/adjusted mismatch on *t-1* (no capital step there) that is larger than on *t* and whose
    implied price ratio ``(1 + ret_adj) / (1 + ret_raw)`` lies in ``(1, capital_ratio * (1 + tol)]``
    (bonus or rights signature), so that a cash dividend on *t-1* is not mistaken for a capital event.
    """
    if t < 2 or np.isnan(shares[t - 2]) or shares[t - 1] != shares[t - 2]:
        return False
    m_prev = mismatch[t - 1]
    if np.isnan(m_prev) or m_prev <= min_mismatch:
        return False
    m_here = mismatch[t] if not np.isnan(mismatch[t]) else 0.0
    if m_here >= m_prev:
        return False
    ratio = shares[t] / shares[t - 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        implied = (1 + ret_adj[t - 1]) / (1 + ret_raw[t - 1])
    return bool(ratio > 1 and 1 < implied <= ratio * (1 + LATE_RATIO_TOL))


def align_shares(
    group: pd.DataFrame, min_mismatch: float = ALIGN_MISMATCH_MIN
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Ex-date-aligned share count for one ticker (rows sorted by date).

    For every SERMAYE step on session *s* the raw/adjusted return mismatch ``|ret_raw - ret_adj|`` on *s*
    and on *s+1* is compared: when the mismatch sits on *s+1* (and exceeds ``min_mismatch``) the vendor
    updated the capital one session early and the old count is kept on *s*; when the mismatch sits on
    *s-1* with a bonus/rights signature (:func:`_updated_late`) the new count is applied from *s-1*.
    Steps without any price
    signature (private placements at market, tiny changes) keep the vendor date. Returns the aligned
    array and a list of the shifts applied (``{"vendor_date", "ex_date", "before", "after"}``).
    """
    shares = group["shares"].to_numpy(dtype=float)
    ret_raw = _returns(group["close_raw"].to_numpy(dtype=float))
    ret_adj = _returns(group["close_adj"].to_numpy(dtype=float))
    mismatch = np.abs(ret_raw - ret_adj)
    aligned = shares.copy()
    shifts: list[dict[str, Any]] = []
    n = len(shares)
    dates = group["date"].to_numpy()
    for t in range(1, n):
        if np.isnan(shares[t]) or np.isnan(shares[t - 1]) or shares[t] == shares[t - 1]:
            continue
        m_here = mismatch[t] if not np.isnan(mismatch[t]) else 0.0
        m_next = mismatch[t + 1] if t + 1 < n and not np.isnan(mismatch[t + 1]) else 0.0
        next_is_step = t + 1 < n and not np.isnan(shares[t + 1]) and shares[t + 1] != shares[t]
        if t + 1 < n and not next_is_step and m_next > max(m_here, min_mismatch):
            # vendor updated one session early: keep the old count on t, price ex-date is t+1
            aligned[t] = shares[t - 1]
            shifts.append(
                {
                    "vendor_date": pd.Timestamp(dates[t]),
                    "ex_date": pd.Timestamp(dates[t + 1]),
                    "before": float(shares[t - 1]),
                    "after": float(shares[t]),
                }
            )
            continue
        if _updated_late(t, shares, ret_raw, ret_adj, mismatch, min_mismatch):
            # vendor updated one session late: the price ex-date is t-1, so the new count starts there
            aligned[t - 1] = shares[t]
            shifts.append(
                {
                    "vendor_date": pd.Timestamp(dates[t]),
                    "ex_date": pd.Timestamp(dates[t - 1]),
                    "before": float(shares[t - 1]),
                    "after": float(shares[t]),
                }
            )
    return aligned, shifts


ADJUSTED_PRICE_COLUMNS: tuple[str, ...] = ("close_adj", "vwap_adj", "low_adj", "high_adj")
ADJUSTMENT_OVERRIDES_FILENAME = "adjustment_overrides.csv"


def load_adjustment_overrides(path: Path | None = None) -> pd.DataFrame:
    """Read ``configs/adjustment_overrides.csv`` (empty frame when the file does not exist)."""
    path = Path(path) if path is not None else CONFIG_DIR / ADJUSTMENT_OVERRIDES_FILENAME
    cols = ["ticker", "ex_date", "correct_factor_step"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df["ticker"] = df["ticker"].str.strip().str.upper()
    df["ex_date"] = pd.to_datetime(df["ex_date"])
    df["correct_factor_step"] = df["correct_factor_step"].astype(float)
    return df


def apply_adjustment_overrides(df: pd.DataFrame, overrides: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Rescale the adjusted prices before each override's ``ex_date`` so the adjustment-factor step equals
    ``correct_factor_step``. ``df`` is sorted by (ticker, date) with ``close_raw``/``close_adj``; rows after
    the ex-date are untouched. Returns the corrected frame and the applied corrections (``multiplier`` is
    observed step / correct step). Overrides whose ticker/date is absent are skipped silently.
    """
    applied: list[dict[str, Any]] = []
    if overrides is None or overrides.empty:
        return df, applied
    df = df.copy()
    for rec in overrides.to_dict("records"):
        rows = df.index[df["ticker"] == rec["ticker"]]
        ex_pos = [i for i in rows if df.at[i, "date"] == pd.Timestamp(rec["ex_date"])]
        if not ex_pos:
            continue
        i = ex_pos[0]
        prev = rows[rows.get_loc(i) - 1] if rows.get_loc(i) > 0 else None
        if prev is None:
            continue
        step = (df.at[i, "close_adj"] / df.at[i, "close_raw"]) / (df.at[prev, "close_adj"] / df.at[prev, "close_raw"])
        mult = step / float(rec["correct_factor_step"])
        if not np.isfinite(mult) or mult <= 0:
            continue
        before = rows[: rows.get_loc(i)]
        df.loc[before, list(ADJUSTED_PRICE_COLUMNS)] = df.loc[before, list(ADJUSTED_PRICE_COLUMNS)] * mult
        applied.append(
            {"ticker": rec["ticker"], "ex_date": str(pd.Timestamp(rec["ex_date"]).date()), "multiplier": float(mult)}
        )
        log.info(
            "adjustment override %s %s: observed step %.4f -> %.4f",
            rec["ticker"],
            applied[-1]["ex_date"],
            step,
            step / mult,
        )
    return df, applied


def _flag_series(values: np.ndarray) -> np.ndarray:
    out = np.zeros(len(values), dtype=bool)
    if len(values) > 1:
        a, b = values[1:], values[:-1]
        out[1:] = (~np.isnan(a)) & (~np.isnan(b)) & (a != b)
    return out


def build_prices_frame(
    vendor_df: pd.DataFrame,
    manifest_entries: Sequence[dict[str, Any]] = (),
    volume_semantics: str | None = None,
    adj_tol: float = ADJ_MISMATCH_TOL,
    adjustment_overrides: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pure transformation from the vendor frame (:data:`VENDOR_COLUMNS`) to the store contract.

    ``volume_semantics`` is ``"tl"`` (default, verified) or ``"shares"``; ``None`` uses the default but
    runs :func:`infer_volume_semantics` and logs a warning when the data disagree. The result carries
    ``attrs['volume_semantics']``, ``attrs['share_shifts']`` (per-ticker one-day-early corrections) and
    ``attrs['source_vintage']``.
    """
    missing = [c for c in VENDOR_COLUMNS if c not in vendor_df.columns]
    if missing:
        raise ValueError(f"vendor frame is missing columns {missing}")
    df = vendor_df.copy()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize().astype("datetime64[ns]")
    df = df.dropna(subset=["close_raw"])
    df = df.drop_duplicates(subset=["ticker", "date"], keep="last")
    df = df.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)

    if adjustment_overrides is None:
        adjustment_overrides = load_adjustment_overrides()
    df, applied_overrides = apply_adjustment_overrides(df, adjustment_overrides)

    inferred = infer_volume_semantics(df) if len(df) else {"semantics": "unknown"}
    if volume_semantics is None:
        volume_semantics = VOLUME_SEMANTICS_DEFAULT
        if inferred["semantics"] not in ("unknown", volume_semantics):
            log.warning(
                "volume semantics inferred as %r, default is %r: %s", inferred["semantics"], volume_semantics, inferred
            )
        else:
            log.info("volume semantics: %s (%s)", volume_semantics, inferred)
    if volume_semantics not in ("shares", "tl"):
        raise ValueError(f"volume_semantics must be 'shares' or 'tl', got {volume_semantics!r}")

    if volume_semantics == "shares":
        df["traded_value_tl"] = df["volume_raw"] * df["vwap_raw"]
    else:
        df["traded_value_tl"] = df["volume_raw"]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["adj_factor"] = np.where(df["close_raw"] > 0, df["close_adj"] / df["close_raw"], np.nan)

    aligned = np.full(len(df), np.nan)
    cap_flag = np.zeros(len(df), dtype=bool)
    mismatch_flag = np.zeros(len(df), dtype=bool)
    shifts: dict[str, list[dict[str, Any]]] = {}
    for ticker, idx in df.groupby("ticker", sort=False).indices.items():
        g = df.iloc[idx]
        a, s = align_shares(g)
        aligned[idx] = a
        if s:
            shifts[str(ticker)] = s
        cap = _flag_series(a)
        ret_raw = _returns(g["close_raw"].to_numpy(dtype=float))
        ret_adj = _returns(g["close_adj"].to_numpy(dtype=float))
        mm = np.abs(ret_raw - ret_adj)
        mismatch_flag[idx] = (~np.isnan(mm)) & (mm > adj_tol) & (~cap)
        cap_flag[idx] = cap
    df["shares_aligned"] = aligned
    df["mcap_tl_aligned"] = df["close_raw"] * df["shares_aligned"]
    df["flag_capital_change"] = cap_flag
    df["flag_adj_mismatch"] = mismatch_flag
    cancelled = pd.to_datetime(sorted(CANCELLED_SESSIONS))
    df["flag_cancelled_session"] = df["date"].isin(cancelled)
    vintage = source_vintage(manifest_entries) if manifest_entries else ""
    df["source_vintage"] = vintage

    out = df[PRICE_COLUMNS].copy()
    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    for col in FLAG_COLUMNS:
        out[col] = out[col].astype(bool)
    out["ticker"] = out["ticker"].astype(str)
    out["source_vintage"] = out["source_vintage"].astype(str)
    out = out.reset_index(drop=True)
    out.attrs["volume_semantics"] = volume_semantics
    out.attrs["volume_semantics_check"] = inferred
    out.attrs["share_shifts"] = shifts
    out.attrs["adjustment_overrides"] = applied_overrides
    out.attrs["source_vintage"] = vintage
    n_shift = sum(len(v) for v in shifts.values())
    log.info(
        "prices frame: %d rows, %d tickers, %d capital changes (%d shifted one day), %d adj mismatches, vintage %s",
        len(out),
        out["ticker"].nunique(),
        int(out["flag_capital_change"].sum()),
        n_shift,
        int(out["flag_adj_mismatch"].sum()),
        vintage or "-",
    )
    return out


def check_prices(df: pd.DataFrame, max_abs_return: float = 0.5) -> list[str]:
    """Human-readable invariant violations (empty = consistent).

    Checks the contract columns, uniqueness of (ticker, date), sorting, non-positive raw closes, and
    unexplained ``|ret_raw| > max_abs_return`` on sessions without a capital change (BIST's daily limit
    is ±10 %, so anything larger is a corporate action or a data error).
    """
    problems: list[str] = []
    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        return [f"missing columns {missing}"]
    if df.duplicated(subset=["ticker", "date"]).any():
        problems.append(f"{int(df.duplicated(subset=['ticker', 'date']).sum())} duplicate (ticker, date) rows")
    key = df[["ticker", "date"]]
    if not key.equals(key.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)):
        problems.append("rows are not sorted by (ticker, date)")
    bad_close = (df["close_raw"] <= 0) | df["close_raw"].isna()
    if bad_close.any():
        problems.append(f"{int(bad_close.sum())} rows with a non-positive or missing close_raw")
    ret = df.groupby("ticker", sort=False)["close_raw"].pct_change()
    wild = (ret.abs() > max_abs_return) & ~df["flag_capital_change"] & ~df["flag_cancelled_session"]
    if wild.any():
        ex = df.loc[wild, ["ticker", "date"]].head(10)
        problems.append(
            f"{int(wild.sum())} unexplained |ret_raw| > {max_abs_return:.0%}: "
            + ", ".join(f"{t}@{d.date()}" for t, d in ex.itertuples(index=False))
        )
    return problems


# --------------------------------------------------------------------------- store I/O
def write_prices(df: pd.DataFrame, store_dir: Path) -> Path:
    """Write ``<store_dir>/prices_daily.parquet`` atomically (contract columns only) and return its path."""
    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"prices frame is missing columns {missing}")
    return write_parquet(df[PRICE_COLUMNS], Path(store_dir) / STORE_FILENAME)


def load_prices(
    store_dir: Path,
    tickers: Iterable[str] | None = None,
    columns: Sequence[str] | None = None,
    drop_cancelled: bool = True,
) -> pd.DataFrame:
    """Read ``<store_dir>/prices_daily.parquet``, optionally filtered to ``tickers`` and ``columns``.

    ``drop_cancelled=True`` (default) removes the 2023-02-08 rows (``flag_cancelled_session``), as the
    trading calendar does; pass ``False`` to see them. ``ticker``/``date`` are always read so the
    filters work, and the result is limited to ``columns`` when given.
    """
    path = Path(store_dir) / STORE_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `bistmas fetch prices` first")
    read_cols: list[str] | None = None
    if columns is not None:
        read_cols = list(dict.fromkeys(["ticker", "date", *columns]))
        if drop_cancelled and "flag_cancelled_session" not in read_cols:
            read_cols.append("flag_cancelled_session")
    df = read_parquet(path, columns=read_cols)
    if tickers is not None:
        wanted = {t.strip().upper() for t in tickers}
        df = df[df["ticker"].isin(wanted)]
    if drop_cancelled:
        df = df[~df["flag_cancelled_session"]]
    if columns is not None:
        df = df[list(columns)]
    return df.reset_index(drop=True)


def prices_as_of(df: pd.DataFrame, as_of: date | pd.Timestamp | str) -> pd.DataFrame:
    """Rows with ``date <= as_of`` (plain filter).

    This is *not* a full point-in-time view: raw columns are exactly what was observable on each
    session, but the adjusted columns and ``adj_factor`` were restated by the vendor after later
    corporate actions. Use adjusted columns for returns inside the window only, never as levels.
    """
    ts = pd.Timestamp(as_of)
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    return df[df["date"] <= ts.normalize()]


def build_prices_store(
    settings: Settings,
    tickers: Iterable[str] | None = None,
    volume_semantics: str | None = None,
) -> pd.DataFrame:
    """Rebuild ``data/store/prices_daily.parquet`` from the raw layer and return the frame.

    Also writes ``prices_daily_meta.json`` (vintage, tickers, row counts, volume-semantics check,
    share-count shifts and invariant problems) next to it.
    """
    vendor, used = read_raw_layer(settings.paths.raw, tickers)
    df = build_prices_frame(vendor, used, volume_semantics=volume_semantics)
    problems = check_prices(df)
    for p in problems:
        log.warning("prices check: %s", p)
    store = settings.paths.store
    write_prices(df, store)
    per_ticker = df.groupby("ticker")["date"].agg(["min", "max", "size"])
    meta: dict[str, Any] = {
        "source_vintage": df.attrs.get("source_vintage", ""),
        "n_rows": int(len(df)),
        "n_tickers": int(df["ticker"].nunique()),
        "date_min": str(df["date"].min().date()) if len(df) else None,
        "date_max": str(df["date"].max().date()) if len(df) else None,
        "raw_files": len(used),
        "volume_semantics": df.attrs.get("volume_semantics"),
        "volume_semantics_check": df.attrs.get("volume_semantics_check"),
        "share_shifts": {
            t: [{k: str(v) for k, v in s.items()} for s in shifts]
            for t, shifts in df.attrs.get("share_shifts", {}).items()
        },
        "capital_changes": int(df["flag_capital_change"].sum()),
        "adj_mismatches": int(df["flag_adj_mismatch"].sum()),
        "cancelled_rows": int(df["flag_cancelled_session"].sum()),
        "problems": problems,
        "tickers": {
            t: {"first": str(r["min"].date()), "last": str(r["max"].date()), "rows": int(r["size"])}
            for t, r in per_ticker.iterrows()
        },
    }
    write_json(meta, store / META_FILENAME)
    log.info("wrote %s (%d rows, vintage %s)", store / STORE_FILENAME, len(df), meta["source_vintage"])
    return df
