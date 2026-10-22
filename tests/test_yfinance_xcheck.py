"""Tests for bistmas.data.clients.yfinance_xcheck (offline with a yfinance-like CSV; one optional network test)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bistmas.data.clients import yfinance_xcheck as yx
from bistmas.data.prices import build_prices_frame, parse_hissetekil
from bistmas.shared.store import append_manifest, read_manifest, write_raw_gzip

FIXTURES = Path(__file__).parent / "fixtures" / "prices"


def _prices() -> pd.DataFrame:
    return build_prices_frame(parse_hissetekil((FIXTURES / "hissetekil_SYNB_2018.json").read_bytes(), "SYNB"))


def _yahoo() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "yahoo_SYNB_2018.csv")


def test_normalise_yahoo_frame_variants():
    base = _yahoo()
    norm = yx.normalise_yahoo_frame(base)
    assert list(norm.columns) == yx.YAHOO_COLUMNS
    assert str(norm["date"].dtype) == "datetime64[ns]"
    assert norm["date"].is_monotonic_increasing and norm["date"].duplicated().sum() == 1  # phantom duplicate kept

    multi = base.set_index(pd.to_datetime(base["Date"])).drop(columns=["Date"])
    multi.columns = pd.MultiIndex.from_tuples([(c, "SYNB.IS") for c in multi.columns])
    pd.testing.assert_frame_equal(yx.normalise_yahoo_frame(multi), norm)

    tz = base.set_index(pd.to_datetime(base["Date"]).dt.tz_localize("Europe/Istanbul")).drop(columns=["Date"])
    pd.testing.assert_frame_equal(yx.normalise_yahoo_frame(tz), norm)

    no_adj = base.drop(columns=["Adj Close"])
    out = yx.normalise_yahoo_frame(no_adj)
    np.testing.assert_allclose(out["adj_close"], out["close"])
    with pytest.raises(ValueError):
        yx.normalise_yahoo_frame(base.drop(columns=["Close"]))


def test_clean_drops_phantom_zero_volume_rows():
    clean, dropped = yx.clean_yahoo_frame(_yahoo())
    assert dropped == 2
    assert pd.Timestamp("2018-01-01") not in set(clean["date"])
    assert (clean["volume"] > 0).all()


def test_cross_check_offline():
    prices = _prices()
    out = yx.cross_check(prices, ["synb"], "2018-01-01", "2018-03-31", yahoo_frames={"SYNB": _yahoo()})
    assert list(out.columns) == yx.XCHECK_COLUMNS
    row = out.iloc[0]
    assert row["ticker"] == "SYNB"
    assert row["n_days"] == 63  # 64 sessions, one missing at Yahoo
    assert row["phantom_rows_dropped"] == 2
    assert row["n_only_isyatirim"] == 1 and row["n_only_yahoo"] == 0
    assert row["ratio_step_days"] == 1  # Yahoo ignored the dividend: one step in HGDG / Adj Close
    assert row["ratio_median"] == pytest.approx(1.2 / 0.92, rel=1e-3)
    assert row["ratio_max_abs_step"] == pytest.approx(0.08, rel=0.05)
    # a looser step tolerance hides the dividend step
    loose = yx.cross_check(prices, ["SYNB"], "2018-01-01", "2018-03-31", yahoo_frames={"SYNB": _yahoo()}, step_tol=0.1)
    assert loose.iloc[0]["ratio_step_days"] == 0


def test_cross_check_window_and_missing_ticker():
    prices = _prices()
    out = yx.cross_check(prices, ["SYNB", "NOPE"], "2018-02-01", "2018-02-28", yahoo_frames={"SYNB": _yahoo()})
    assert out.iloc[0]["n_days"] == 20 and out.iloc[0]["ratio_step_days"] == 0
    assert out.iloc[1]["ticker"] == "NOPE" and out.iloc[1]["n_days"] == 0 and np.isnan(out.iloc[1]["ratio_median"])


def test_fetch_uses_raw_cache_without_yfinance(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", None)  # import yfinance -> ImportError
    norm = yx.normalise_yahoo_frame(_yahoo())
    path = tmp_path / yx.RAW_SUBDIR / "SYNB_2018-01-01_2018-03-31.csv.gz"
    write_raw_gzip(norm.to_csv(index=False).encode(), path)
    append_manifest(path.parent / "manifest.jsonl", {"source": "yfinance", "path": str(path), "sha256": "x"})
    frames = yx.fetch_yahoo_history(["SYNB"], "2018-01-01", "2018-03-31", raw_dir=tmp_path)
    assert list(frames) == ["SYNB"] and len(frames["SYNB"]) == len(norm)
    with pytest.raises(ImportError, match="yfinance is not installed"):
        yx.fetch_yahoo_history(["THYAO"], "2018-01-01", "2018-03-31", raw_dir=tmp_path)
    with pytest.raises(ImportError):
        yx.cross_check(_prices(), ["SYNB"], "2018-01-01", "2018-03-31")


def test_fetch_caches_download_once(tmp_path, monkeypatch):
    calls: list[str] = []

    class FakeYF:
        @staticmethod
        def download(symbol, start, end, auto_adjust, progress, threads):
            calls.append(symbol)
            assert auto_adjust is False
            return _yahoo().set_index("Date").rename_axis("Date")

    monkeypatch.setattr(yx, "_import_yfinance", lambda: FakeYF)
    first = yx.fetch_yahoo_history(["synb"], "2018-01-01", "2018-03-31", raw_dir=tmp_path)
    second = yx.fetch_yahoo_history(["SYNB"], "2018-01-01", "2018-03-31", raw_dir=tmp_path)
    assert calls == ["SYNB.IS"]
    pd.testing.assert_frame_equal(first["SYNB"], second["SYNB"])
    manifest = read_manifest(tmp_path / yx.RAW_SUBDIR / "manifest.jsonl")
    assert len(manifest) == 1 and manifest[0]["symbol"] == "SYNB.IS" and manifest[0]["rows"] == 65


def test_fetch_skips_empty_downloads(monkeypatch):
    class EmptyYF:
        @staticmethod
        def download(*args, **kwargs):
            return pd.DataFrame()

    monkeypatch.setattr(yx, "_import_yfinance", lambda: EmptyYF)
    assert yx.fetch_yahoo_history(["MRDIN"], "2018-01-01", "2018-03-31") == {}


@pytest.mark.network
def test_live_cross_check_thyao_asels(settings):
    """Live: HGDG / Yahoo Adj Close must be piecewise constant for two live names (needs yfinance)."""
    pytest.importorskip("yfinance")
    from bistmas.data.prices import STORE_FILENAME, load_prices

    if not (settings.paths.store / STORE_FILENAME).exists():
        pytest.skip("prices_daily.parquet not built")
    prices = load_prices(settings.paths.store, tickers=["THYAO", "ASELS"])
    if prices.empty:
        pytest.skip("THYAO/ASELS not in the price store")
    start, end = prices["date"].min(), prices["date"].max()
    out = yx.cross_check(prices, ["THYAO", "ASELS"], start, end, raw_dir=settings.paths.raw)
    assert (out["n_days"] > 200).all()
    assert (out["ratio_step_days"] <= 12).all()
    assert (out["n_only_isyatirim"] <= 5).all()
    if os.environ.get("BISTMAS_NETWORK") == "1":
        print(out.to_string(index=False))
