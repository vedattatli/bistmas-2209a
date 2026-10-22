"""Offline tests for bistmas.data.clients.index_anchors (fixtures trimmed from the real files)."""

from __future__ import annotations

import gzip
import io
import json
import shutil
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from bistmas.data.clients import index_anchors as ia
from bistmas.shared.calendar import TradingCalendar
from bistmas.shared.store import append_manifest, write_raw_gzip

FIXTURES = Path(__file__).parent / "fixtures" / "anchors"
HAS_BSDTAR = shutil.which("bsdtar") is not None


class FakeResponse:
    def __init__(self, content: bytes, status: int = 200) -> None:
        self.content = content
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"status {self.status_code}")


class FakeSession:
    """Mimics PoliteSession.get_cached / get without touching the network."""

    def __init__(self, payloads: dict[str, bytes], fail_get: bool = False) -> None:
        self.payloads = payloads
        self.fail_get = fail_get
        self.fetches: list[str] = []

    def get_cached(self, url, raw_path, manifest_path=None, source="", force=False, expect_status=200, **kw):
        if raw_path.exists() and not force:
            with gzip.open(raw_path, "rb") as fh:
                return fh.read()
        if url not in self.payloads:
            raise RuntimeError(f"404 for {url}")
        self.fetches.append(url)
        payload = self.payloads[url]
        digest = write_raw_gzip(payload, raw_path)
        append_manifest(manifest_path or raw_path.parent / "manifest.jsonl", {"source": source, "sha256": digest})
        return payload

    def get(self, url, **kw):
        if self.fail_get:
            raise ConnectionError("archive offline")
        return FakeResponse(self.payloads[url])


def _zip_with(member: str, payload: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(member, payload)
        zf.writestr("endeks_bolen_ds_genel_20170630.csv", b"TARIH;ENDEKS BOLEN\r\n")
    return buf.getvalue()


# ----------------------------------------------------------------------------- registry
def test_registry_is_consistent():
    names = [s.name for s in ia.ANCHOR_SPECS]
    assert len(names) == len(set(names))
    assert ia.MANDATORY_ANCHOR_NAMES == {"bist_weights_20170630", "bist_weights_20231229"}
    for spec in ia.ANCHOR_SPECS:
        assert spec.kind in {"rar", "xlsx", "csv"}
        assert spec.url.startswith("https://")
        assert (spec.kind == "rar") == (spec.member is not None)
        assert spec.raw_file_name == f"{spec.name}.{spec.kind}.gz"
    by_name = {s.name: s for s in ia.ANCHOR_SPECS}
    assert by_name["bist_weights_20170630"].snapshot_date == date(2017, 6, 30)
    assert by_name["bist_weights_20231229"].snapshot_date == date(2023, 12, 29)
    assert by_name["bist_weights_live"].snapshot_date is None


# ----------------------------------------------------------------------------- dates
def test_composition_effective_date_weekday_rule():
    assert ia.composition_effective_date(date(2017, 6, 30)) == date(2017, 7, 3)  # 1 July 2017 is a Saturday
    assert ia.composition_effective_date(date(2023, 9, 29)) == date(2023, 10, 2)
    assert ia.composition_effective_date(date(2025, 3, 28)) == date(2025, 4, 1)
    assert ia.composition_effective_date(date(2025, 9, 30)) == date(2025, 10, 1)
    assert ia.composition_effective_date(date(2023, 5, 31)) == date(2023, 7, 3)
    # Without a calendar 1 January 2024 (a Monday, exchange holiday) is returned...
    assert ia.composition_effective_date(date(2023, 12, 29)) == date(2024, 1, 1)
    # ...with a trading calendar the first real session is used.
    cal = TradingCalendar([date(2023, 12, 29), date(2024, 1, 2), date(2024, 1, 3)])
    assert ia.composition_effective_date(date(2023, 12, 29), cal) == date(2024, 1, 2)


def test_quarter_helpers_and_period_end_detection():
    assert ia.quarter_start(date(2024, 2, 6)) == date(2024, 1, 1)
    assert ia.next_quarter_start(date(2024, 11, 30)) == date(2025, 1, 1)
    assert ia.is_period_end_date(date(2025, 3, 28))
    assert ia.is_period_end_date(date(2017, 6, 30))
    assert not ia.is_period_end_date(date(2024, 2, 6))
    assert not ia.is_period_end_date(date(2023, 5, 31))


def test_snapshot_valid_from():
    assert ia.snapshot_valid_from(date(2017, 6, 30), True) == date(2017, 7, 3)
    assert ia.snapshot_valid_from(date(2023, 5, 31), False) == date(2023, 5, 31)
    assert ia.snapshot_valid_from(date(2024, 2, 6)) == date(2024, 2, 6)  # auto-detected mid-period
    assert ia.snapshot_valid_from(date(2026, 6, 30)) == date(2026, 7, 1)  # auto-detected period end


# ----------------------------------------------------------------------------- parsers
def test_parse_weights_csv_16_columns():
    df = ia.parse_weights_csv((FIXTURES / "weights_20170630_snippet.csv").read_text(encoding="utf-8"))
    assert list(df.columns) == list(ia.CANONICAL_COLUMNS)
    assert len(df) == 16
    assert df["file_date"].unique().tolist() == [date(2017, 6, 30)]
    assert df["index_code"].value_counts().to_dict() == {"XU100": 6, "XUTUM": 4, "XU050": 3, "XU030": 3}
    first = df.iloc[0]
    assert first["ticker"] == "EGEEN" and first["equity_code"] == "EGEEN.E"
    assert first["security_name"] == "EGE ENDUSTRI"
    assert first["shares_total"] == 3_150_000.0 and first["free_float_ratio"] == 0.34
    assert first["weighting_factor"] == 1.0 and first["weight"] == pytest.approx(0.116)
    assert first["shares_in_index"] == pytest.approx(3_150_000 * 0.34)
    assert first["price_basis"] == "CLOSE(KAPANIS)" and first["adjustment_stage"].startswith("AFTER ADJUSTMENT")
    assert df["price"].isna().all() and df["market_value_total"].isna().all()
    assert not df["ticker"].str.contains(".", regex=False).any()


def test_parse_weights_csv_17_columns_with_price_and_digit_ticker():
    df = ia.parse_weights_csv((FIXTURES / "weights_live_17col_snippet.csv").read_text(encoding="utf-8"))
    assert list(df.columns) == list(ia.CANONICAL_COLUMNS)
    assert df["file_date"].unique().tolist() == [date(2026, 6, 30)]
    assert df["price"].notna().all()
    assert "A1CAP" in set(df["ticker"])  # digits are allowed, never dropped
    xu100 = df[df["index_code"] == "XU100"]
    assert len(xu100) == 4 and xu100["ticker"].iloc[0] == "AEFES"


def test_parse_weights_csv_dot_dates_and_plain_numbers():
    df = ia.parse_weights_csv((FIXTURES / "weights_20250328_dotdate_snippet.csv").read_text(encoding="utf-8"))
    assert df["file_date"].unique().tolist() == [date(2025, 3, 28)]
    assert df["shares_total"].iloc[0] == 592_105_263.0 and df["weighting_factor"].iloc[0] == 1.0


def test_parse_weights_csv_rejects_missing_columns():
    with pytest.raises(ia.AnchorError, match="lacks"):
        ia.parse_weights_csv("TARIH;ENDEKS KODU\r\nDATE;INDEX CODE\r\n30/06/2017;XU100\r\n")


def test_parse_weights_xlsx_16_columns_matches_csv_layout():
    df = ia.parse_weights_xlsx((FIXTURES / "weights_20231229_snippet.xlsx").read_bytes())
    assert list(df.columns) == list(ia.CANONICAL_COLUMNS)
    assert df["file_date"].unique().tolist() == [date(2023, 12, 29)]
    assert df["index_code"].value_counts().to_dict() == {"XU100": 6, "XU050": 3, "XU030": 3, "XUTUM": 3}
    aefes = df[(df["index_code"] == "XU100") & (df["ticker"] == "AEFES")].iloc[0]
    assert aefes["shares_total"] == 592_105_263.0 and aefes["free_float_ratio"] == 0.32
    assert aefes["weight"] == pytest.approx(1.2971)
    assert "A1CAP" in set(df["ticker"])
    assert df["file_date"].map(type).eq(date).all()
    assert str(df["ticker"].dtype) == "string" and df["weight"].dtype == "float64"


def test_parse_weights_xlsx_19_columns_with_market_values():
    df = ia.parse_weights_xlsx((FIXTURES / "weights_20230531_19col_snippet.xlsx").read_bytes())
    assert list(df.columns) == list(ia.CANONICAL_COLUMNS)
    assert df["file_date"].unique().tolist() == [date(2023, 5, 31)]
    row = df.iloc[0]
    assert row["price"] == pytest.approx(58.2)
    assert row["market_value_total"] == pytest.approx(34_460_526_306.6)
    assert row["market_value_weighted_free_float"] == pytest.approx(11_027_368_418.112)
    assert row["weight"] == pytest.approx(0.8863)


def test_weight_sums_groups_by_date_and_index():
    df = ia.parse_weights_csv((FIXTURES / "weights_20170630_snippet.csv").read_text(encoding="utf-8"))
    sums = ia.weight_sums(df)
    assert set(sums.index.get_level_values("index_code")) == {"XU100", "XU050", "XU030", "XUTUM"}
    assert sums.loc[(date(2017, 6, 30), "XU100")] == pytest.approx(df[df["index_code"] == "XU100"]["weight"].sum())


def test_to_float_handles_decimal_commas():
    assert ia._to_float("1,15") == 1.15
    assert ia._to_float("1.234.567,89") == 1234567.89
    assert ia._to_float("592105263") == 592105263.0
    assert pd.isna(ia._to_float(""))
    with pytest.raises(ia.AnchorError):
        ia._to_float("abc")


# ----------------------------------------------------------------------------- archives
@pytest.mark.skipif(not HAS_BSDTAR, reason="bsdtar not installed")
def test_extract_archive_member_via_bsdtar_stdin():
    csv_bytes = (FIXTURES / "weights_20170630_snippet.csv").read_bytes()
    archive = _zip_with("endeks_agirlik_ds_genel_20170630.csv", csv_bytes)
    assert ia.list_archive_members(archive) == [
        "endeks_agirlik_ds_genel_20170630.csv",
        "endeks_bolen_ds_genel_20170630.csv",
    ]
    assert ia.extract_rar_member(archive, "endeks_agirlik_ds_genel_20170630.csv") == csv_bytes
    with pytest.raises(ia.AnchorError, match="could not extract"):
        ia.extract_rar_member(archive, "missing.csv")


def test_validate_anchor_payload():
    ia.validate_anchor_payload(b"Rar!\x1a\x07\x00", "rar")
    ia.validate_anchor_payload(b"PK\x03\x04rest", "rar")
    ia.validate_anchor_payload(b"PK\x03\x04rest", "xlsx")
    ia.validate_anchor_payload("﻿TARIH;ENDEKS KODU".encode(), "csv")
    for kind in ("rar", "xlsx", "csv"):
        with pytest.raises(ia.AnchorError):
            ia.validate_anchor_payload(b"<html><body>Wayback Machine</body></html>", kind)


# ----------------------------------------------------------------------------- fetch + load
def test_fetch_anchor_caches_and_discards_bad_payloads(tmp_path):
    spec = ia.AnchorSpec("t_csv", "https://example.org/w.csv", "csv", date(2017, 6, 30))
    payload = (FIXTURES / "weights_20170630_snippet.csv").read_bytes()
    session = FakeSession({spec.url: payload})
    assert ia.fetch_anchor(session, spec, tmp_path) == payload
    assert ia.fetch_anchor(session, spec, tmp_path) == payload
    assert session.fetches == [spec.url]  # second call served from data/raw
    raw_path = tmp_path / "index_anchors" / "t_csv.csv.gz"
    assert raw_path.exists()
    manifest = [json.loads(line) for line in (tmp_path / "index_anchors" / "manifest.jsonl").read_text().splitlines()]
    assert manifest[0]["source"] == "index_anchors"

    bad = ia.AnchorSpec("t_bad", "https://example.org/bad.xlsx", "xlsx", None)
    session = FakeSession({bad.url: b"<html>stub</html>"})
    with pytest.raises(ia.AnchorError, match="not a xlsx"):
        ia.fetch_anchor(session, bad, tmp_path)
    assert not (tmp_path / "index_anchors" / "t_bad.xlsx.gz").exists()


@pytest.mark.skipif(not HAS_BSDTAR, reason="bsdtar not installed")
def test_load_anchor_memberships_over_mixed_specs(tmp_path):
    csv_2017 = (FIXTURES / "weights_20170630_snippet.csv").read_bytes()
    specs = [
        ia.AnchorSpec(
            "a_rar",
            "https://example.org/a.zip",
            "rar",
            date(2017, 6, 30),
            member="endeks_agirlik_ds_genel_20170630.csv",
            mandatory=True,
            period_end=True,
        ),
        ia.AnchorSpec("a_xlsx", "https://example.org/a.xlsx", "xlsx", date(2023, 12, 29), mandatory=True),
        ia.AnchorSpec("a_mid", "https://example.org/mid.xlsx", "xlsx", date(2023, 5, 31), period_end=False),
        ia.AnchorSpec("a_missing", "https://example.org/missing.csv", "csv", None),  # optional -> skipped
        ia.AnchorSpec("a_live", "https://example.org/live.csv", "csv", None),
    ]
    session = FakeSession(
        {
            specs[0].url: _zip_with("endeks_agirlik_ds_genel_20170630.csv", csv_2017),
            specs[1].url: (FIXTURES / "weights_20231229_snippet.xlsx").read_bytes(),
            specs[2].url: (FIXTURES / "weights_20230531_19col_snippet.xlsx").read_bytes(),
            specs[4].url: (FIXTURES / "weights_live_17col_snippet.csv").read_bytes(),
        }
    )
    df = ia.load_anchor_memberships(session, tmp_path, specs)
    assert list(df.columns) == list(ia.MEMBERSHIP_COLUMNS)
    assert set(df["source_name"]) == {"a_rar", "a_xlsx", "a_mid", "a_live"}
    per = df.groupby("source_name")[["snapshot_date", "composition_effective_date"]].first()
    assert per.loc["a_rar"].tolist() == [date(2017, 6, 30), date(2017, 7, 3)]
    assert per.loc["a_xlsx"].tolist() == [date(2023, 12, 29), date(2024, 1, 1)]
    assert per.loc["a_mid"].tolist() == [date(2023, 5, 31), date(2023, 5, 31)]
    assert per.loc["a_live"].tolist() == [date(2026, 6, 30), date(2026, 7, 1)]
    assert df[(df["source_name"] == "a_rar") & (df["index_code"] == "XU100")]["ticker"].tolist() == [
        "DOHOL",
        "EGEEN",
        "HALKB",
        "SODA",
        "TATGD",
        "TAVHL",
    ]
    assert df["weight"].dtype == "float64" and df["snapshot_date"].map(type).eq(date).all()
    assert df.equals(df.sort_values(["snapshot_date", "index_code", "ticker"], kind="stable").reset_index(drop=True))

    # A mandatory anchor that cannot be fetched is fatal.
    with pytest.raises(ia.AnchorError, match="mandatory anchor a_xlsx"):
        ia.load_anchor_memberships(FakeSession({}), tmp_path / "empty", specs[1:2])
    # Nothing loadable and nothing mandatory -> empty frame with the right columns.
    empty = ia.load_anchor_memberships(FakeSession({}), tmp_path / "empty", specs[3:4])
    assert empty.empty and list(empty.columns) == list(ia.MEMBERSHIP_COLUMNS)


def test_load_anchor_frame_warns_on_date_mismatch(tmp_path, caplog):
    spec = ia.AnchorSpec("t_csv", "https://example.org/w.csv", "csv", date(2017, 7, 1))
    session = FakeSession({spec.url: (FIXTURES / "weights_20170630_snippet.csv").read_bytes()})
    with caplog.at_level("WARNING", logger="bistmas.data.index_anchors"):
        df = ia.load_anchor_frame(session, tmp_path, spec)
    assert df["file_date"].iloc[0] == date(2017, 6, 30)
    assert any("spec expected 2017-07-01" in rec.message for rec in caplog.records)


# ----------------------------------------------------------------------------- discovery
def test_discover_wayback_captures_parses_cdx_and_drops_html_stubs():
    payload = (FIXTURES / "cdx_endeks_agirlik.json").read_bytes()
    session = FakeSession({ia.CDX_API: payload})
    df = ia.discover_wayback_captures(session)
    assert list(df.columns) == ["timestamp", "original", "mimetype", "statuscode", "digest", "length", "capture_url"]
    assert len(df) == 10 and not df["mimetype"].str.startswith("text/html").any()
    row = df[df["timestamp"] == "20240331074837"].iloc[0]
    assert row["capture_url"] == (
        "https://web.archive.org/web/20240331074837id_/https://borsaistanbul.com/files/endeks_agirlik_ds_genel_1.xlsx"
    )
    assert len(ia.discover_wayback_captures(session, drop_html=False)) == 13


def test_discover_wayback_captures_tolerates_outage():
    df = ia.discover_wayback_captures(FakeSession({}, fail_get=True))
    assert df.empty and "capture_url" in df.columns


# ----------------------------------------------------------------------------- live
@pytest.mark.network
def test_live_anchors_have_full_index_lists(settings):
    """Downloads (once, into data/raw/index_anchors) and parses every registered anchor."""
    from bistmas.data.clients.http import PoliteSession

    session = PoliteSession(min_interval_s=1.0, timeout_s=ia.ANCHOR_TIMEOUT_S)
    df = ia.load_anchor_memberships(session, settings.paths.raw)
    counts = df.groupby(["source_name", "index_code"]).size()
    for name in ia.MANDATORY_ANCHOR_NAMES:
        assert counts.loc[(name, "XU100")] == 100
        assert counts.loc[(name, "XU030")] == 30
        assert counts.loc[(name, "XU050")] == 50
    loaded = set(df["source_name"])
    print("\nanchors loaded:", sorted(loaded))
    for name in sorted(loaded):
        sub = counts.loc[name]
        print(
            name,
            df[df["source_name"] == name]["snapshot_date"].iloc[0],
            "->",
            df[df["source_name"] == name]["composition_effective_date"].iloc[0],
            {k: int(sub.get(k, 0)) for k in ("XU100", "XU050", "XU030", "XUTUM")},
            "indices:",
            len(sub),
        )
    captures = ia.discover_wayback_captures(session)
    print("cdx captures:", len(captures))
