"""Offline tests for bistmas.data.prices (synthetic HisseTekil-shaped fixtures, invented numbers)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bistmas.data import prices
from bistmas.data.prices import (
    FIELD_MAP,
    PRICE_COLUMNS,
    VENDOR_COLUMNS,
    RawLayerError,
    build_prices_frame,
    build_prices_store,
    check_prices,
    infer_volume_semantics,
    list_raw_files,
    load_prices,
    parse_date_value,
    parse_hissetekil,
    prices_as_of,
    read_raw_layer,
    source_vintage,
    write_prices,
)
from bistmas.shared.config import PathsConfig, Settings
from bistmas.shared.hashing import sha256_bytes
from bistmas.shared.store import append_manifest, read_json, write_raw_gzip

FIXTURES = Path(__file__).parent / "fixtures" / "prices"


def _payload(name: str) -> bytes:
    return (FIXTURES / f"hissetekil_{name}.json").read_bytes()


def _raw_layer(root: Path, files: dict[tuple[str, int], str], manifest: bool = True) -> Path:
    """Write fixture payloads into ``root/raw/isyatirim/<TICKER>/<YYYY>.json.gz`` (+ manifest)."""
    raw = root / "raw"
    for (ticker, year), name in files.items():
        payload = _payload(name)
        path = raw / "isyatirim" / ticker / f"{year}.json.gz"
        digest = write_raw_gzip(payload, path)
        if manifest:
            append_manifest(
                raw / "isyatirim" / "manifest.jsonl",
                {"source": "isyatirim", "url": f"https://example/{ticker}/{year}", "path": str(path), "sha256": digest},
            )
    return raw


def _frame(*names: str) -> pd.DataFrame:
    vendor = pd.concat([parse_hissetekil(_payload(n), n.split("_")[0]) for n in names], ignore_index=True)
    return build_prices_frame(vendor, [{"path": f"{n}.json.gz", "sha256": "0" * 64} for n in names])


def _row(df: pd.DataFrame, ticker: str, day: str) -> pd.Series:
    sub = df[(df["ticker"] == ticker) & (df["date"] == pd.Timestamp(day))]
    assert len(sub) == 1, f"{ticker} {day}: {len(sub)} rows"
    return sub.iloc[0]


# --------------------------------------------------------------------------- parsing
def test_parse_date_value_accepts_vendor_forms():
    assert parse_date_value("02-01-2018") == pd.Timestamp("2018-01-02")
    assert parse_date_value("02.01.2018") == pd.Timestamp("2018-01-02")
    assert parse_date_value("2018-01-02T00:00:00") == pd.Timestamp("2018-01-02")
    # epoch ms of Istanbul midnight expressed in UTC (real END_TARIH/DD_TARIH values)
    assert parse_date_value(1514840400000) == pd.Timestamp("2018-01-02")
    assert parse_date_value("/Date(1546203600000)/") == pd.Timestamp("2018-12-31")
    assert parse_date_value(1514840400) == pd.Timestamp("2018-01-02")
    assert pd.isna(parse_date_value(None))
    assert pd.isna(parse_date_value(""))
    assert pd.isna(parse_date_value("31-02-2018"))
    assert pd.isna(parse_date_value("garbage"))


def test_parse_hissetekil_sample_is_tolerant():
    df = parse_hissetekil(_payload("sample"), "SYNP")
    assert list(df.columns) == VENDOR_COLUMNS
    assert list(df["date"]) == [pd.Timestamp(d) for d in ("2018-01-02", "2018-01-03", "2018-01-04")]
    assert df.attrs["dropped_rows"] == 2  # one row without raw close, one without a date
    assert str(df["date"].dtype) == "datetime64[ns]"
    assert (df["ticker"] == "SYNP").all()
    assert df["volume_usd"].notna().all()
    for col in FIELD_MAP.values():
        assert df[col].dtype == float


def test_parse_hissetekil_empty_and_bad_payloads():
    df = parse_hissetekil(_payload("empty"), "MRDIN")
    assert df.empty and list(df.columns) == VENDOR_COLUMNS
    assert str(df["date"].dtype) == "datetime64[ns]"
    with pytest.raises(RawLayerError):
        parse_hissetekil(b"not json", "X")
    with pytest.raises(RawLayerError):
        parse_hissetekil(b"", "X")
    with pytest.raises(RawLayerError):
        parse_hissetekil({"value": "oops"}, "X")
    assert parse_hissetekil({"value": None}, "X").empty


def test_parse_hissetekil_string_numbers():
    rows = [{"HGDG_TARIH": "02-01-2018", "HG_KAPANIS": "1.234,5", "SERMAYE": "1000", "HGDG_KAPANIS": "12,5"}]
    df = parse_hissetekil({"value": rows}, "X")
    assert df.loc[0, "close_raw"] == 1234.5
    assert df.loc[0, "shares"] == 1000.0
    assert df.loc[0, "close_adj"] == 12.5
    assert np.isnan(df.loc[0, "vwap_raw"])


# --------------------------------------------------------------------------- raw layer
def test_read_raw_layer_manifest_and_vintage(tmp_path):
    raw = _raw_layer(tmp_path, {("SYNB", 2018): "SYNB_2018", ("SYNR", 2018): "SYNR_2018"})
    (raw / "isyatirim" / "XU100").mkdir()  # not an equity code: skipped
    (raw / "isyatirim" / "SYNB" / "notes.txt").write_text("ignored")
    files = list_raw_files(raw)
    assert [(t, y) for t, y, _ in files] == [("SYNB", 2018), ("SYNR", 2018)]
    assert [(t, y) for t, y, _ in list_raw_files(raw, ["synr"])] == [("SYNR", 2018)]

    vendor, used = read_raw_layer(raw)
    assert sorted(vendor["ticker"].unique()) == ["SYNB", "SYNR"]
    assert vendor[["ticker", "date"]].duplicated().sum() == 0
    assert [u["path"] for u in used] == ["SYNB/2018.json.gz", "SYNR/2018.json.gz"]
    assert used[0]["sha256"] == sha256_bytes(_payload("SYNB_2018"))

    v1 = source_vintage(used)
    assert len(v1) == 12 and int(v1, 16) >= 0
    assert source_vintage(list(reversed(used))) == v1  # order-independent
    changed = [dict(used[0], sha256="f" * 64), used[1]]
    assert source_vintage(changed) != v1


def test_read_raw_layer_without_manifest_hashes_payload(tmp_path):
    raw = _raw_layer(tmp_path, {("SYNB", 2018): "SYNB_2018"}, manifest=False)
    _, used = read_raw_layer(raw)
    assert used == [{"path": "SYNB/2018.json.gz", "sha256": sha256_bytes(_payload("SYNB_2018")), "fetched_at": ""}]


def test_read_raw_layer_missing_raises(tmp_path):
    with pytest.raises(RawLayerError):
        read_raw_layer(tmp_path / "raw")


def test_read_raw_layer_collapses_overlapping_year_files(tmp_path):
    raw = _raw_layer(tmp_path, {("SYNB", 2018): "SYNB_2018", ("SYNB", 2019): "SYNB_2018"})
    vendor, used = read_raw_layer(raw)
    assert len(used) == 2
    assert vendor[["ticker", "date"]].duplicated().sum() == 0
    assert len(vendor) == len(parse_hissetekil(_payload("SYNB_2018"), "SYNB"))


# --------------------------------------------------------------------------- transformation
def test_build_prices_frame_contract():
    df = _frame("SYNB_2018", "SYNR_2018")
    assert list(df.columns) == PRICE_COLUMNS
    assert str(df["date"].dtype) == "datetime64[ns]"
    for col in ("flag_capital_change", "flag_adj_mismatch", "flag_cancelled_session"):
        assert df[col].dtype == bool
    assert df["source_vintage"].nunique() == 1 and len(df["source_vintage"].iloc[0]) == 12
    key = df[["ticker", "date"]]
    assert key.equals(key.sort_values(["ticker", "date"]).reset_index(drop=True))
    assert key.duplicated().sum() == 0
    assert check_prices(df) == []
    np.testing.assert_allclose(df["adj_factor"], df["close_adj"] / df["close_raw"])


def test_bonus_issue_with_one_day_early_sermaye_is_aligned():
    df = _frame("SYNB_2018")
    early = _row(df, "SYNB", "2018-02-14")  # vendor already shows the new capital, price still cum
    ex = _row(df, "SYNB", "2018-02-15")
    after = _row(df, "SYNB", "2018-02-16")
    assert early["shares"] == 200e6 and early["shares_aligned"] == 100e6
    assert ex["shares_aligned"] == 200e6 and after["shares_aligned"] == 200e6
    assert early["mcap_tl"] == pytest.approx(early["close_raw"] * 200e6)  # vendor PD doubled for a day
    assert early["mcap_tl_aligned"] == pytest.approx(early["close_raw"] * 100e6)
    assert ex["mcap_tl_aligned"] == pytest.approx(ex["mcap_tl"])
    assert ex["adj_factor"] == pytest.approx(2 * early["adj_factor"], rel=0.02)
    assert ex["close_raw"] / early["close_raw"] == pytest.approx(0.5, rel=0.05)
    cap_days = df.loc[df["flag_capital_change"], "date"].tolist()
    assert cap_days == [pd.Timestamp("2018-02-15")]
    assert not ex["flag_adj_mismatch"]
    assert df.attrs["share_shifts"] == {
        "SYNB": [
            {
                "vendor_date": pd.Timestamp("2018-02-14"),
                "ex_date": pd.Timestamp("2018-02-15"),
                "before": 100e6,
                "after": 200e6,
            }
        ]
    }


def test_dividend_flags_adj_mismatch_only():
    df = _frame("SYNB_2018")
    mismatch_days = df.loc[df["flag_adj_mismatch"], "date"].tolist()
    assert mismatch_days == [pd.Timestamp("2018-03-12")]
    div = _row(df, "SYNB", "2018-03-12")
    assert not div["flag_capital_change"]
    assert div["adj_factor"] > _row(df, "SYNB", "2018-03-09")["adj_factor"] * 1.05


def test_rights_issue_step_on_ex_date_is_kept():
    df = _frame("SYNR_2018")
    assert df.attrs["share_shifts"] == {}
    np.testing.assert_array_equal(df["shares"], df["shares_aligned"])
    assert df.loc[df["flag_capital_change"], "date"].tolist() == [pd.Timestamp("2018-02-15")]
    ex = _row(df, "SYNR", "2018-02-15")
    assert ex["close_raw"] / _row(df, "SYNR", "2018-02-14")["close_raw"] == pytest.approx(0.625, rel=0.05)
    assert not df["flag_adj_mismatch"].any()


def test_align_shares_keeps_steps_without_price_signature():
    vendor = parse_hissetekil(_payload("SYNR_2018"), "SYNR")
    # private placement at market: capital steps, prices unaffected
    vendor["close_adj"] = vendor["close_raw"]
    aligned, shifts = prices.align_shares(vendor)
    assert shifts == []
    np.testing.assert_array_equal(aligned, vendor["shares"].to_numpy())


def test_volume_semantics_is_tl_value():
    vendor = parse_hissetekil(_payload("SYNB_2018"), "SYNB")
    check = infer_volume_semantics(vendor)
    assert check["semantics"] == "tl"
    assert check["usd_vs_raw_volume_rel_diff_median"] < 1e-6
    df = build_prices_frame(vendor)
    np.testing.assert_allclose(df["traded_value_tl"], df["volume_raw"])
    assert df.attrs["volume_semantics"] == "tl"
    df2 = build_prices_frame(vendor, volume_semantics="shares")
    np.testing.assert_allclose(df2["traded_value_tl"], df2["volume_raw"] * df2["vwap_raw"])
    with pytest.raises(ValueError):
        build_prices_frame(vendor, volume_semantics="lots")


def test_volume_semantics_shares_is_detected_and_warned(caplog):
    vendor = parse_hissetekil(_payload("SYNB_2018"), "SYNB")
    # rewrite the vendor columns as if HACIM were a share count
    shares_traded = vendor["volume_raw"] / vendor["vwap_raw"]
    vendor["volume_raw"] = shares_traded
    vendor["volume_adj"] = shares_traded * vendor["vwap_raw"] / vendor["vwap_adj"]
    vendor["volume_usd"] = shares_traded * vendor["vwap_raw"] / vendor["usdtry"]
    assert infer_volume_semantics(vendor)["semantics"] == "shares"
    with caplog.at_level("WARNING", logger="bistmas.data.prices"):
        build_prices_frame(vendor)
    assert any("inferred as 'shares'" in m for m in caplog.messages)


def test_cancelled_session_is_flagged_and_dropped_on_load(tmp_path):
    df = _frame("SYNC_2023")
    assert df.loc[df["flag_cancelled_session"], "date"].tolist() == [pd.Timestamp("2023-02-08")]
    assert pd.Timestamp("2023-02-09") not in set(df["date"])
    store = tmp_path / "store"
    write_prices(df, store)
    loaded = load_prices(store)
    assert list(loaded.columns) == PRICE_COLUMNS
    assert pd.Timestamp("2023-02-08") not in set(loaded["date"])
    assert len(loaded) == len(df) - 1
    kept = load_prices(store, drop_cancelled=False)
    assert len(kept) == len(df)
    sub = load_prices(store, tickers=["sync"], columns=["close_raw"])
    assert list(sub.columns) == ["close_raw"] and len(sub) == len(df) - 1
    assert load_prices(store, tickers=["NOPE"]).empty
    with pytest.raises(FileNotFoundError):
        load_prices(tmp_path / "nowhere")


def test_prices_as_of_is_a_plain_filter():
    df = _frame("SYNB_2018")
    sub = prices_as_of(df, "2018-02-15")
    assert sub["date"].max() == pd.Timestamp("2018-02-15")
    assert len(prices_as_of(df, pd.Timestamp("2018-02-15 15:00", tz="Europe/Istanbul"))) == len(sub)
    assert prices_as_of(df, "2017-12-31").empty


def test_check_prices_reports_problems():
    df = _frame("SYNB_2018")
    dup = pd.concat([df, df.iloc[[5]]], ignore_index=True)
    problems = check_prices(dup)
    assert any("duplicate" in p for p in problems) and any("not sorted" in p for p in problems)
    wild = df.copy()
    wild.loc[10, "close_raw"] = wild.loc[9, "close_raw"] * 3
    assert any("unexplained" in p for p in check_prices(wild))
    assert check_prices(df.drop(columns=["xu100"])) == ["missing columns ['xu100']"]


def test_build_prices_frame_rejects_missing_vendor_columns():
    with pytest.raises(ValueError):
        build_prices_frame(pd.DataFrame({"ticker": ["A"], "date": [pd.Timestamp("2018-01-02")]}))


# --------------------------------------------------------------------------- end to end
def test_build_prices_store_writes_parquet_and_meta(tmp_path):
    raw = _raw_layer(tmp_path, {("SYNB", 2018): "SYNB_2018", ("SYNR", 2018): "SYNR_2018", ("SYNC", 2023): "SYNC_2023"})
    settings = Settings(root=tmp_path, paths=PathsConfig(raw=raw, store=tmp_path / "store"))
    df = build_prices_store(settings, tickers=["SYNB", "SYNC"])
    assert sorted(df["ticker"].unique()) == ["SYNB", "SYNC"]
    stored = pd.read_parquet(tmp_path / "store" / prices.STORE_FILENAME)
    assert list(stored.columns) == PRICE_COLUMNS and len(stored) == len(df)
    meta = read_json(tmp_path / "store" / prices.META_FILENAME)
    assert meta["source_vintage"] == df["source_vintage"].iloc[0]
    assert meta["volume_semantics"] == "tl"
    assert meta["raw_files"] == 2 and meta["n_tickers"] == 2
    assert meta["share_shifts"]["SYNB"][0]["ex_date"].startswith("2018-02-15")
    assert meta["cancelled_rows"] == 1 and meta["problems"] == []
    assert meta["tickers"]["SYNC"]["last"] == "2023-03-31"
    # rebuilding with the same raw files reproduces the vintage
    assert build_prices_store(settings, tickers=["SYNB", "SYNC"])["source_vintage"].iloc[0] == meta["source_vintage"]


def test_fixture_files_are_small():
    for p in FIXTURES.iterdir():
        assert p.stat().st_size < 150_000, p.name
    payload = json.loads(_payload("SYNB_2018"))
    assert set(payload) == {"ok", "errorCode", "errorDescription", "transactionId", "value"}
