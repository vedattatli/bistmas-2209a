"""Yahoo Finance cross-check of the İş Yatırım adjusted series for live names.

Yahoo (``yfinance``, ``<TICKER>.IS``) is survivorship-biased (no delisted names) and carries 30–40
phantom zero-volume rows per ticker on BIST holidays, so it is never a price source here — only a
check that İş Yatırım's ``HGDG_KAPANIS`` and Yahoo's ``Adj Close`` (``auto_adjust=False``) agree up
to a constant: their ratio must be piecewise constant, stepping only where the two vendors adjust
differently (Yahoo ignores some cash dividends; both restate after events). A ratio that drifts or
steps on ordinary days points at a bad row on one side.

``yfinance`` is an *optional* dependency (not in ``pyproject.toml``): :func:`fetch_yahoo_history`
imports it lazily and raises a clear ``ImportError`` otherwise. It talks to Yahoo with its own HTTP
client (curl_cffi), the one documented exception to "all network access through PoliteSession";
every download is cached once under ``data/raw/yfinance/`` with a manifest line so it is never
repeated. :func:`cross_check` itself is pure and works offline on yfinance-like frames.
"""

from __future__ import annotations

import io
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bistmas.shared.calendar import CANCELLED_SESSIONS
from bistmas.shared.log import get_logger
from bistmas.shared.store import append_manifest, read_raw_gzip, utc_now_iso, write_raw_gzip

log = get_logger("data.yfinance_xcheck")

RAW_SUBDIR = "yfinance"
YAHOO_SUFFIX = ".IS"
RATIO_STEP_TOL = 0.01
XCHECK_COLUMNS: list[str] = [
    "ticker",
    "n_days",
    "ratio_median",
    "ratio_step_days",
    "phantom_rows_dropped",
    "n_only_isyatirim",
    "n_only_yahoo",
    "ratio_max_abs_step",
]
YAHOO_COLUMNS: list[str] = ["date", "close", "adj_close", "volume"]


def _import_yfinance() -> Any:
    try:
        import yfinance  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "yfinance is not installed; it is an optional dependency used only by the cross-check "
            "(`uv add yfinance` or pass yahoo_frames= to cross_check)"
        ) from exc
    return yfinance


def normalise_yahoo_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten a ``yf.download``/``Ticker.history`` frame to :data:`YAHOO_COLUMNS` (tz-naive dates).

    Accepts a ``DatetimeIndex`` or a ``Date`` column, single-level or ``(field, ticker)`` MultiIndex
    columns, and ``Adj Close``/``adj_close``/``AdjClose`` spellings. Rows without ``adj_close`` fall
    back to ``close`` (yfinance omits Adj Close with ``auto_adjust=True``).
    """
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(c[0]) for c in out.columns]
    if "Date" in out.columns or "date" in out.columns:
        col = "Date" if "Date" in out.columns else "date"
        out = out.set_index(col)
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    rename = {}
    for c in out.columns:
        key = str(c).strip().lower().replace(" ", "_")
        if key in ("adj_close", "adjclose"):
            rename[c] = "adj_close"
        elif key == "close":
            rename[c] = "close"
        elif key == "volume":
            rename[c] = "volume"
    out = out.rename(columns=rename)
    if "close" not in out.columns:
        raise ValueError("Yahoo frame has no Close column")
    if "adj_close" not in out.columns:
        out["adj_close"] = out["close"]
    if "volume" not in out.columns:
        out["volume"] = np.nan
    res = pd.DataFrame(
        {
            "date": out.index.astype("datetime64[ns]"),
            "close": pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=float),
            "adj_close": pd.to_numeric(out["adj_close"], errors="coerce").to_numpy(dtype=float),
            "volume": pd.to_numeric(out["volume"], errors="coerce").to_numpy(dtype=float),
        }
    )
    # duplicates are kept here: phantom holiday rows can share a date with a real session and must be
    # counted (and removed) by clean_yahoo_frame before de-duplication
    return res.dropna(subset=["date"]).sort_values("date", kind="stable").reset_index(drop=True)


def clean_yahoo_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop phantom rows (``volume == 0`` or missing ``adj_close``); returns ``(clean, n_dropped)``."""
    norm = normalise_yahoo_frame(df) if list(df.columns) != YAHOO_COLUMNS else df
    phantom = (norm["volume"].fillna(0) <= 0) | norm["adj_close"].isna()
    clean = norm[~phantom].drop_duplicates("date").reset_index(drop=True)
    return clean, int(phantom.sum())


def _raw_path(raw_dir: Path, ticker: str, start: str, end: str) -> Path:
    return Path(raw_dir) / RAW_SUBDIR / f"{ticker}_{start}_{end}.csv.gz"


def fetch_yahoo_history(
    tickers: Iterable[str],
    start: date | str,
    end: date | str,
    raw_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Download (or read from the raw cache) ``auto_adjust=False`` daily bars for ``<ticker>.IS``.

    Returns ``{ticker: yfinance-like frame}`` (normalised columns, phantom rows still present). Empty
    responses are retried once (they are known to be transient) and then logged and skipped.
    """
    start_s, end_s = str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date())
    frames: dict[str, pd.DataFrame] = {}
    yf = None
    for ticker in (t.strip().upper() for t in tickers):
        path = _raw_path(raw_dir, ticker, start_s, end_s) if raw_dir is not None else None
        if path is not None and path.exists():
            cached = pd.read_csv(io.BytesIO(read_raw_gzip(path)), parse_dates=["date"])
            cached["date"] = pd.to_datetime(cached["date"]).astype("datetime64[ns]")
            frames[ticker] = cached
            continue
        yf = yf or _import_yfinance()
        symbol = f"{ticker}{YAHOO_SUFFIX}"
        raw = pd.DataFrame()
        for attempt in range(2):
            raw = yf.download(symbol, start=start_s, end=end_s, auto_adjust=False, progress=False, threads=False)
            if raw is not None and len(raw):
                break
            log.warning("yfinance returned no rows for %s (attempt %d)", symbol, attempt + 1)
        if raw is None or not len(raw):
            continue
        norm = normalise_yahoo_frame(raw)
        frames[ticker] = norm
        if path is not None:
            payload = norm.to_csv(index=False).encode("utf-8")
            digest = write_raw_gzip(payload, path)
            append_manifest(
                path.parent / "manifest.jsonl",
                {
                    "source": "yfinance",
                    "symbol": symbol,
                    "start": start_s,
                    "end": end_s,
                    "path": str(path),
                    "rows": int(len(norm)),
                    "sha256": digest,
                    "fetched_at": utc_now_iso(),
                },
            )
    return frames


def cross_check_ticker(
    ours: pd.DataFrame,
    yahoo: pd.DataFrame,
    ticker: str,
    step_tol: float = RATIO_STEP_TOL,
) -> dict[str, Any]:
    """One row of the cross-check table for a ticker (see :func:`cross_check`)."""
    clean, dropped = clean_yahoo_frame(yahoo)
    mine = ours[["date", "close_adj"]].dropna()
    merged = mine.merge(clean[["date", "adj_close"]], on="date", how="inner").sort_values("date")
    merged = merged[(merged["adj_close"] > 0) & (merged["close_adj"] > 0)]
    ratio = merged["close_adj"] / merged["adj_close"]
    step = (ratio / ratio.shift(1) - 1).abs()
    ours_dates, yahoo_dates = set(mine["date"]), set(clean["date"])
    return {
        "ticker": ticker,
        "n_days": int(len(merged)),
        "ratio_median": float(ratio.median()) if len(ratio) else np.nan,
        "ratio_step_days": int((step > step_tol).sum()),
        "phantom_rows_dropped": dropped,
        "n_only_isyatirim": len(ours_dates - yahoo_dates),
        "n_only_yahoo": len(yahoo_dates - ours_dates),
        "ratio_max_abs_step": float(step.max()) if step.notna().any() else np.nan,
    }


def cross_check(
    prices_df: pd.DataFrame,
    tickers: Iterable[str],
    start: date | str,
    end: date | str,
    yahoo_frames: dict[str, pd.DataFrame] | None = None,
    raw_dir: Path | None = None,
    step_tol: float = RATIO_STEP_TOL,
) -> pd.DataFrame:
    """Compare ``close_adj`` (HGDG) with Yahoo ``Adj Close`` per ticker over ``[start, end]``.

    ``yahoo_frames`` (``{ticker: frame}``, yfinance-like) makes the check offline/testable; without it
    :func:`fetch_yahoo_history` is called (needs ``yfinance``). Phantom Yahoo rows (``Volume == 0``)
    are dropped before matching. Columns (:data:`XCHECK_COLUMNS`): ``n_days`` matched sessions,
    ``ratio_median`` of HGDG / Adj Close, ``ratio_step_days`` sessions where the ratio moves by more
    than ``step_tol`` (each is a vendor disagreement to inspect), ``phantom_rows_dropped``,
    ``n_only_isyatirim`` / ``n_only_yahoo`` unmatched sessions and ``ratio_max_abs_step``.
    """
    wanted = [t.strip().upper() for t in tickers]
    ts0, ts1 = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if yahoo_frames is None:
        yahoo_frames = fetch_yahoo_history(wanted, ts0, ts1, raw_dir)
    yahoo_frames = {k.upper(): v for k, v in yahoo_frames.items()}
    panel = prices_df
    if "flag_cancelled_session" in panel.columns:
        panel = panel[~panel["flag_cancelled_session"].astype(bool)]
    else:
        panel = panel[~panel["date"].isin([pd.Timestamp(d) for d in CANCELLED_SESSIONS])]
    panel = panel[(panel["date"] >= ts0) & (panel["date"] <= ts1)]
    rows: list[dict[str, Any]] = []
    for ticker in wanted:
        ours = panel[panel["ticker"] == ticker]
        yahoo = yahoo_frames.get(ticker)
        if yahoo is None or ours.empty:
            log.warning("cross-check skipped for %s (%s)", ticker, "no Yahoo data" if yahoo is None else "no rows")
            rows.append(
                {
                    "ticker": ticker,
                    "n_days": 0,
                    "ratio_median": np.nan,
                    "ratio_step_days": 0,
                    "phantom_rows_dropped": 0,
                    "n_only_isyatirim": int(len(ours)),
                    "n_only_yahoo": 0,
                    "ratio_max_abs_step": np.nan,
                }
            )
            continue
        clean_y = normalise_yahoo_frame(yahoo)
        clean_y = clean_y[(clean_y["date"] >= ts0) & (clean_y["date"] <= ts1)]
        rows.append(cross_check_ticker(ours, clean_y, ticker, step_tol))
    out = pd.DataFrame(rows, columns=XCHECK_COLUMNS)
    for col in ("n_days", "ratio_step_days", "phantom_rows_dropped", "n_only_isyatirim", "n_only_yahoo"):
        out[col] = out[col].astype(int)
    return out
