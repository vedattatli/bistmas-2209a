"""prereg-v2 jury corrections: J rounding, kept candidates, transport errors, registry hash, peer-group payload."""

import json
import re

import numpy as np
import pandas as pd
import pytest
from test_jury import FakeClient, _screen

from bistmas.jury.agents import AGENTS, AgentPrompts, JuryAgent, TransportError
from bistmas.jury.cache import JuryCache
from bistmas.jury.consensus import consensus
from bistmas.jury.ollama_client import ChatResult
from bistmas.jury.payload import build_payloads
from bistmas.jury.random_jury import random_scores


def _rows(spec):
    return pd.DataFrame(
        [
            {"as_of": "2018-12-31", "ticker": t, "agent": a, "score": s, "composite_rank": cr}
            for t, cr, scores in spec
            for a, s in zip(AGENTS, scores, strict=True)
        ]
    )


def test_j_ties_use_composite_rank_not_float_noise():
    # mathematically both J = 0.8, but the raw float means differ by an ULP
    c = consensus(_rows([("EGEEN", 6, [1.0, 0.85, 0.55]), ("GSDHO", 1, [1.0, 0.95, 0.45])]), theta=0.8)
    assert c["J"].tolist() == [0.8, 0.8]
    assert c["ticker"].tolist() == ["GSDHO", "EGEEN"] and c["buy"].all()


def test_all_abstained_candidate_is_kept_with_nan_j():
    c = consensus(_rows([("AAA", 1, [np.nan] * 3), ("BBB", 2, [0.5, 0.6, 0.7]), ("CCC", 3, [0.9, np.nan, np.nan])]))
    by = c.set_index("ticker")
    assert set(by.index) == {"AAA", "BBB", "CCC"} and by.loc["AAA", "n_votes"] == 0
    assert by[["J", "rank_J"]].loc[["AAA", "CCC"]].isna().all().all() and not by["buy"].loc[["AAA", "CCC"]].any()
    assert by.loc["BBB", "rank_J"] == 1
    assert consensus(_rows([("CCC", 3, [0.9, np.nan, np.nan])]), min_votes=1)["J"].iloc[0] == 0.9


def test_random_null_scores_on_the_real_grid():
    cands = pd.DataFrame({"as_of": ["2018-12-31"] * 40, "ticker": [f"T{i}" for i in range(40)]})
    s = random_scores(cands, 3)["score"]
    assert np.allclose(s * 20, (s * 20).round())


class _Down:
    model = "fake"
    options = {"seed": 1}
    calls = 0

    def chat_json(self, system, user, schema):
        self.calls += 1
        return ChatResult("", None, None, None, None, "fake", {}, error="request failed: refused", transport_error=True)


def test_transport_error_is_not_cached_or_abstained(tmp_path):
    payload = build_payloads(_screen(), pd.Timestamp("2019-06-28"), ["EFGH"], "value", salt="r")[0]
    down = _Down()
    cache = JuryCache(tmp_path)
    a = JuryAgent("value", down, AgentPrompts.load("v2"), cache, "d", {"ABCD"}, 1)
    with pytest.raises(TransportError):
        a.judge("2019-06-28", payload)
    assert down.calls == 1 and not list(tmp_path.rglob("*.json"))


def test_run_jury_stops_with_transport_error_and_resumes(tmp_path):
    from bistmas.jury.run import run_jury
    from bistmas.shared.config import PathsConfig, Settings

    settings = Settings(root=tmp_path, paths=PathsConfig().resolve(tmp_path))
    master = pd.DataFrame({"ticker": ["ABCD", "EFGH", "IJKL", "MNOP", "QRST"], "title": ["X A.Ş."] * 5})
    down = {a: _Down() for a in AGENTS}
    s = run_jury(settings, "r", _screen(), master, top_n=3, clients=down)
    assert s["stopped_reason"] == "transport_error" and s["records"] == 0

    class Ok(FakeClient):
        def __init__(self):
            super().__init__([])

        def chat_json(self, system, user, schema):
            cid = re.search(r'candidate_id "([A-Z_]+)"', user).group(1)
            d = {
                "candidate_id": cid,
                "key_strengths": ["s"],
                "key_risks": ["r"],
                "rationale": "x",
                "cited_metrics": [],
                "score": 0.6,
                "confidence": 0.5,
            }
            return ChatResult(json.dumps(d), d, 1, 1, 1.0, "fake", self.options, None)

    s = run_jury(settings, "r", _screen(), master, top_n=3, clients={a: Ok() for a in AGENTS})
    assert s["stopped_reason"] == "complete" and s["records"] == 9 and s["abstain_rate"] == 0


def test_registry_hash_is_enforced(tmp_path):
    import shutil

    from bistmas.shared.config import REPO_ROOT

    root = tmp_path / "prompts"
    shutil.copytree(REPO_ROOT / "prompts", root)
    AgentPrompts.load("v2", root=root)
    f = root / "v2" / "system_value.md"
    f.write_text(f.read_text() + " ")
    with pytest.raises(ValueError, match="registry"):
        AgentPrompts.load("v2", root=root)
    AgentPrompts.load("v1", root=root)


def test_payload_uses_norm_group_for_label_and_percentiles():
    s = _screen()
    s["norm_group"] = ["POOLED", "POOLED", "FINANCIALS", "FINANCIALS", "FINANCIALS"]
    s["sector_group"] = ["TRANSPORT", "RETAIL", "BANKS", "BANKS", "INSURANCE"]
    s["roic"] = [0.1, 0.9, 0.1, 0.5, 0.9]
    ps = {p.ticker: p for p in build_payloads(s, pd.Timestamp("2019-06-28"), ["ABCD", "IJKL"], "quality_growth", "x")}
    assert ps["ABCD"].fields["sector_group"] == "pooled" and ps["IJKL"].fields["sector_group"] == "financials"
    assert ps["ABCD"].fields["metrics"]["roic_pct"] == 50  # lowest of 2 peers (universe-wide would be 20)
    assert ps["IJKL"].fields["metrics"]["roic_pct"] == 33
    old = s.drop(columns="norm_group")  # old store: fall back to sector_group
    p = build_payloads(old, pd.Timestamp("2019-06-28"), ["ABCD"], "value", "x")[0]
    assert p.fields["sector_group"] == "transport"


def test_run_jury_refuses_missing_or_mismatched_live_digest(tmp_path):
    from bistmas.jury.run import run_jury
    from bistmas.shared.config import PathsConfig, Settings

    settings = Settings(root=tmp_path, paths=PathsConfig().resolve(tmp_path))
    master = pd.DataFrame({"ticker": ["ABCD"], "title": ["X A.Ş."]})

    class Stub:
        model = settings.jury.model

        def __init__(self, digest):
            self._d = digest

        def model_digest(self):
            return self._d

    with pytest.raises(RuntimeError, match="no digest"):
        run_jury(settings, "d1", _screen(), master, top_n=3, client=Stub(None))
    pinned = settings.model_copy(update={"jury": settings.jury.model_copy(update={"model_digest": "aaa"})})
    with pytest.raises(RuntimeError, match="!= pinned"):
        run_jury(pinned, "d2", _screen(), master, top_n=3, client=Stub("bbb"))


def test_send_report_dry_run_never_opens_smtp(monkeypatch, tmp_path):
    import smtplib

    from bistmas.reporting.email import send_report

    def boom(*a, **k):
        raise AssertionError("SMTP opened")

    monkeypatch.setattr(smtplib, "SMTP", boom)
    monkeypatch.setenv("SMTP_USER", "u")
    monkeypatch.setenv("SMTP_PASSWORD", "p")
    assert send_report("s", "body", "a@b.c", "h", 25, dry_run=True).startswith("not sent")
