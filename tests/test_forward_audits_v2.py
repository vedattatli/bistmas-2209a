"""Regression tests for the forward run, audits and macro/config changes of prereg-v2 (WP7)."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import bistmas.reporting.forward as fw
from bistmas.jury.audits.groundedness import scan_rationale
from bistmas.jury.audits.probes import reidentification_probe
from bistmas.jury.audits.probes import test_retest as run_test_retest
from bistmas.jury.audits.run import AliasMap, _retest_sample, _stratified_sample
from bistmas.reporting.certificate import build_certificate
from bistmas.shared.calendar import TradingCalendar, rule_sessions, with_future_sessions
from bistmas.shared.config import PathsConfig, Settings
from bistmas.shared.ledger import config_hash


def _data_cal(end: str = "2026-09-23") -> TradingCalendar:
    return TradingCalendar([d.date() for d in pd.bdate_range("2026-01-02", end)])


def _settings(tmp_path: Path) -> Settings:
    paths = PathsConfig().resolve(tmp_path)
    for p in (paths.raw, paths.store, paths.runs):
        p.mkdir(parents=True, exist_ok=True)
    return Settings(root=tmp_path, paths=paths)


# ------------------------------------------------------------------ calendar / month selection
def test_future_calendar_skips_holidays_and_extends_past_data() -> None:
    cal = with_future_sessions(_data_cal("2026-10-19"))
    assert cal.next_session(date(2026, 10, 28), strict=True) == date(2026, 10, 30)  # 29 Oct closed
    assert date(2026, 10, 30) in rule_sessions(date(2026, 10, 1), date(2026, 10, 31))
    assert cal.month_ends(date(2026, 10, 1), date(2026, 10, 31)) == [date(2026, 10, 30)]


def test_unfinished_month_is_never_a_month_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fw, "load_calendar", lambda _store: _data_cal("2026-10-19"))
    s = _settings(tmp_path)
    # store ends mid-October: the latest COMPLETED month is September, which the store does not reach (23 Sep..)
    assert fw.completed_month_ends(s, date(2026, 10, 20))[-1] == date(2026, 9, 30)
    assert fw.latest_decision_date(s, date(2026, 10, 20)) == date(2026, 9, 30)
    monkeypatch.setattr(fw, "load_calendar", lambda _store: _data_cal("2026-09-23"))
    with pytest.raises(RuntimeError, match="refresh"):
        fw.latest_decision_date(s, date(2026, 10, 20))
    assert fw.pending_months(s, date(2026, 10, 20)) == []
    assert fw.planned_execution_session(with_future_sessions(_data_cal("2026-09-23")), datetime(2026, 10, 21, 9)) == (
        date(2026, 10, 21)
    )
    assert fw.planned_execution_session(with_future_sessions(_data_cal("2026-09-23")), datetime(2026, 10, 21, 11)) == (
        date(2026, 10, 22)
    )


def _patch_run(
    monkeypatch: pytest.MonkeyPatch, cal: TradingCalendar, calls: list[str], abstain_rate: float = 0.0
) -> None:
    monkeypatch.setattr(fw, "load_calendar", lambda _store: cal)
    monkeypatch.setattr(fw, "load_inputs", lambda _s: (None, None, {}))
    monkeypatch.setattr(
        fw,
        "screen_one",
        lambda as_of, _i, _s: pd.DataFrame(
            {
                "ticker": ["AAAA"],
                "rank": [1.0],
                "sector_group": ["X"],
                "composite": [1.0],
                "S_fundamental": [1.0],
                "S_valuation": [1.0],
                "S_momentum_risk": [1.0],
            }
        ),
    )
    monkeypatch.setattr(fw, "load_security_master", lambda _store: pd.DataFrame())
    monkeypatch.setattr(fw, "build_certificate", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(fw, "render_decision_markdown", lambda d: "md")

    def fake_jury(settings, run_name, screen, master, decisions, dry_run=False, **kw):
        calls.append(run_name)
        return {"stopped_reason": "complete", "abstain_rate": abstain_rate}

    monkeypatch.setattr(fw, "run_jury", fake_jury)


def test_forward_run_seals_with_future_execution_session_and_rejects_open_months(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = _settings(tmp_path)
    cal = _data_cal("2026-10-30")
    calls: list[str] = []
    _patch_run(monkeypatch, cal, calls)
    now = datetime(2026, 11, 2, 9, 0)
    with pytest.raises(ValueError, match="completed month"):
        fw.run_forward(s, date(2026, 10, 19), now=now, send_mail=False)  # mid-month date
    assert calls == []
    with pytest.raises(ValueError, match="completed month"):
        fw.run_forward(s, date(2026, 10, 30), now=datetime(2026, 10, 30, 12), send_mail=False)  # month not over
    d = fw.run_forward(s, date(2026, 10, 30), now=now, send_mail=False)
    assert d["planned_execution_session"] == "2026-11-02" and d["late_run"] is False
    assert calls == ["forward_20261030"]
    again = fw.run_forward(s, date(2026, 10, 30), now=now, send_mail=False)
    assert again["sha256"] == d["sha256"] and calls == ["forward_20261030"]
    sealed = s.paths.runs / "forward" / "2026-10" / "decision.json"
    assert sealed.exists()


def test_store_behind_decision_date_fails_before_any_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = _settings(tmp_path)
    calls: list[str] = []
    _patch_run(monkeypatch, _data_cal("2026-09-23"), calls)
    with pytest.raises(RuntimeError, match="refresh"):
        fw.run_forward(s, date(2026, 9, 30), now=datetime(2026, 10, 21, 9), send_mail=False)
    assert calls == []


def test_dry_run_never_seals_or_blocks_the_real_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings(tmp_path)
    calls: list[str] = []
    _patch_run(monkeypatch, _data_cal("2026-10-30"), calls)
    now = datetime(2026, 11, 2, 9, 0)
    fw.run_forward(s, date(2026, 10, 30), dry_run=True, now=now, send_mail=False)
    assert calls == ["forward_dryrun_20261030"]
    assert not (s.paths.runs / "forward").exists()
    assert (s.paths.runs / "forward_dryrun" / "2026-10" / "decision.json").exists()
    real = fw.run_forward(s, date(2026, 10, 30), now=now, send_mail=False)
    assert calls[-1] == "forward_20261030" and real["dry_run"] is False
    assert (s.paths.runs / "forward" / "log.jsonl").exists()


def test_all_abstained_jury_is_not_sealed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings(tmp_path)
    _patch_run(monkeypatch, _data_cal("2026-10-30"), [], abstain_rate=1.0)
    with pytest.raises(RuntimeError, match="not usable"):
        fw.run_forward(s, date(2026, 10, 30), now=datetime(2026, 11, 2, 9), send_mail=False)
    assert not (s.paths.runs / "forward" / "2026-10" / "decision.json").exists()


# ------------------------------------------------------------------ certificate / plist / config
def test_certificate_config_hash_matches_ledger_hash(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    feats = pd.DataFrame({"max_source_available_date": pd.to_datetime(["2026-09-01"])})
    cert = build_certificate(s, pd.Timestamp("2026-09-30"), feats, None)
    assert cert["config_hash"] == config_hash(s)


def test_plist_sets_forward_config() -> None:
    text = (Path(__file__).parents[1] / "scripts" / "com.bistmas.forward.plist").read_text(encoding="utf-8")
    assert "<key>BISTMAS_CONFIG</key><string>configs/forward.toml</string>" in text


def test_macro_config_fields_in_base_toml() -> None:
    from bistmas.shared.config import load_settings

    m = load_settings(local=Path("absent.toml")).macro
    assert m.cpi_series == "TP.FG.J0" and m.cpi_stale_days == 60 and m.cpi_continuation_series


# ------------------------------------------------------------------ audits
def test_may_the_verb_and_plain_words_are_not_leaks() -> None:
    r = scan_rationale("The stock may be cheap; an elite status, roic_pct fine", {"roic_pct"}, {"ELITE", "STATUS"})
    assert not r["leak_flag"]
    assert scan_rationale("Reported in May 2019", set(), set())["leak_flag"]
    assert scan_rationale("on 31 May the price", set(), set())["leak_flag"]
    assert scan_rationale("THYAO looks cheap", set(), {"THYAO"})["leaks"] == ["THYAO"]


def _scores(n_months: int = 36, per_month: int = 10) -> pd.DataFrame:
    months = pd.date_range("2018-01-31", periods=n_months, freq="ME")
    rows = [(str(m.date()), f"T{i:02d}", float(i + 1)) for m in months for i in range(per_month)]
    return pd.DataFrame(rows, columns=["as_of", "ticker", "composite_rank"])


def test_stratified_sample_covers_every_month_and_is_not_truncated() -> None:
    sc = _scores()
    out = _stratified_sample(sc, 36, 1)
    assert len(out) == 36 and out["as_of"].nunique() == 36
    big = _stratified_sample(sc, 72, 1)  # 2 per month, drawn from different rank bins
    assert len(big) == 72 and big.groupby("as_of").size().eq(2).all()
    merged = big.merge(sc, on=["as_of", "ticker"])
    assert (merged.groupby("as_of")["composite_rank"].apply(lambda s: s.max() - s.min()) >= 1).all()
    small = _stratified_sample(sc, 10, 1)  # fewer than months: still one per month, never cut to 10
    assert small["as_of"].nunique() == 36


def test_retest_sample_excludes_development_years_and_spans_the_run() -> None:
    sample = _stratified_sample(_scores(60), 60, 1)
    r = _retest_sample(sample, 12)
    years = pd.to_datetime(r["as_of"]).dt.year
    assert len(r) == 12 and not years.isin([2018, 2019]).any() and years.nunique() >= 3


def test_alias_map_uses_code_and_title_valid_at_the_decision_date() -> None:
    aliases = pd.DataFrame(
        {
            "old_ticker": ["KOZAL", "ADANA"],
            "new_ticker": ["TRALT", "OYAKC"],
            "effective_date": ["2025-11-24", "2020-05-15"],
            "kind": ["rename", "merger"],
            "note": ["Koza Altın İşletmeleri -> Türk Altın İşletmeleri; code from 2025-11-24", ""],
        }
    )
    am = AliasMap(aliases)
    assert am.identity("TRALT", pd.Timestamp("2019-06-28"), "TÜRK ALTIN") == (
        "KOZAL",
        "Koza Altın İşletmeleri",
        {"KOZAL", "TRALT"},
    )
    assert am.identity("TRALT", pd.Timestamp("2026-01-30"), "TÜRK ALTIN") == ("TRALT", "TÜRK ALTIN", {"TRALT"})
    assert am.identity("OYAKC", pd.Timestamp("2019-01-31"), "OYAK")[0] == "OYAKC"  # merger is not a rename
    assert "KOZA" not in am.old_title_words() and "ALTIN" in am.old_title_words()


class _Res:
    def __init__(self, parsed, ok=True):
        self.parsed, self.ok = parsed, ok


class _Client:
    def __init__(self, parsed_list):
        self.parsed_list = list(parsed_list)

    def chat_json(self, *a, **k):
        return self.parsed_list.pop(0)


class _Payload:
    ticker = "TRALT"

    def render(self):
        return "{}"


def test_reidentification_accepts_either_code_and_uses_quarter_span() -> None:
    client = _Client([_Res({"ticker_guess": "KOZAL", "year_guess": 2019, "quarter_guess": 2, "confidence": 0.5})])
    out = reidentification_probe(client, [(_Payload(), {"KOZAL", "TRALT"}, 2019, 2)], 100, n_quarters=10)
    assert out["ticker_top1"] == 1.0 and out["quarter_chance"] == pytest.approx(0.1)


def test_test_retest_drops_failed_items_to_keep_groups_balanced() -> None:
    class _A:
        agent = "value"
        schema = {}

        class prompts:  # noqa: N801
            system = {"value": "s"}
            user_template = "{payload}{candidate_id}"

        class client:  # noqa: N801
            n = 0

            @classmethod
            def chat_json(cls, *a, **k):
                cls.n += 1
                if cls.n == 2:  # second call of the first item fails
                    return _Res(None, ok=False)
                return _Res({"score": 0.5 if cls.n < 4 else 0.7})

    class P:
        def __init__(self, t):
            self.ticker, self.candidate_id = t, "c"

        def render(self):
            return "x"

    out = run_test_retest(
        _A(), [P("A"), P("A"), P("B")], seeds=[1, 2], items=["2020-01-31|A", "2021-01-31|A", "2020-01-31|B"]
    )
    assert out["n_items"] == 2 and out["n_items_dropped"] == 1
    assert np.isfinite(out["share_exact"])
