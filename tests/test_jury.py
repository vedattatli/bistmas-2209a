"""Jury layer without a model: schema, masking, cache keys, consensus, random null, agent retry/abstain."""

import json

import numpy as np
import pandas as pd
import pytest

from bistmas.jury.agents import AGENTS, AgentPrompts, JuryAgent
from bistmas.jury.cache import JuryCache, cache_key
from bistmas.jury.consensus import consensus
from bistmas.jury.ollama_client import ChatResult
from bistmas.jury.payload import assert_clean, build_payloads, pseudonyms_for
from bistmas.jury.random_jury import random_scores
from bistmas.jury.schema import json_schema, validate_verdict


def _screen(as_of="2019-06-28"):
    rows = []
    for i, t in enumerate(["ABCD", "EFGH", "IJKL", "MNOP", "QRST"]):
        rows.append(
            {
                "as_of": pd.Timestamp(as_of),
                "ticker": t,
                "eligible": True,
                "rank": i + 1,
                "template": "industrial",
                "sector_group": "MANUFACTURING",
                "mcap_tl": 1e9 * (i + 1),
                "statement_age_days": 40 + i * 50,
                "roic": 0.1 * (i + 1),
                "operating_margin": 0.05 * (i + 1),
                "gross_margin": 0.2,
                "revenue_growth": 0.1,
                "profit_growth": np.nan,
                "earnings_yield": 0.1,
                "book_to_market": 0.5,
                "ebitda_yield": 0.1,
                "fcf_yield": 0.05,
                "momentum_12_1": 0.2,
                "vol_6m": 0.3,
                "mdd_6m": -0.2,
                "adv_6m_tl": 1e7,
                "z_roic": 1.0,
                "z_operating_margin": 0.5,
                "z_gross_margin": 0.0,
                "z_revenue_growth": 0.2,
                "z_profit_growth": np.nan,
                "z_earnings_yield": 0.3,
                "z_book_to_market": 0.1,
                "z_ebitda_yield": 0.2,
                "z_fcf_yield": -0.1,
                "z_momentum_12_1": 0.4,
                "z_vol_6m": -0.5,
                "z_mdd_6m": 0.2,
                "z_adv_6m_tl": 0.9,
            }
        )
    return pd.DataFrame(rows)


def test_schema_order_and_validation():
    keys = list(json_schema()["properties"].keys())
    assert keys.index("rationale") < keys.index("score") < keys.index("confidence")
    good = {
        "candidate_id": "STOCK_A",
        "key_strengths": ["x"],
        "key_risks": [],
        "rationale": "roic_pct is high",
        "cited_metrics": ["roic_pct"],
        "score": 0.7,
        "confidence": 0.6,
    }
    v, err = validate_verdict(good, "STOCK_A", {"roic_pct", "roic_z"})
    assert v is not None and err is None and v.score == 0.7
    assert validate_verdict({**good, "candidate_id": "STOCK_B"}, "STOCK_A", {"roic_pct"})[1].startswith("candidate_id")
    assert (
        "not in the payload" in validate_verdict({**good, "cited_metrics": ["revenue_tl"]}, "STOCK_A", {"roic_pct"})[1]
    )
    assert validate_verdict({**good, "score": 1.5}, "STOCK_A", {"roic_pct"})[0] is None


def test_payload_is_masked_partitioned_and_re_randomised():
    s = _screen()
    p = build_payloads(s, pd.Timestamp("2019-06-28"), ["ABCD", "EFGH"], "value", salt="run1")
    assert {x.candidate_id for x in p} <= set(f"STOCK_{c}" for c in "AB") and len(p) == 2
    text = p[0].render()
    assert "ABCD" not in text and "2019" not in text and "composite" not in text
    assert set(p[0].fields["metrics"]) == {
        "earnings_yield_z",
        "earnings_yield_pct",
        "book_to_market_z",
        "book_to_market_pct",
        "ebitda_yield_z",
        "ebitda_yield_pct",
        "fcf_yield_z",
        "fcf_yield_pct",
    }
    assert p[0].fields["statement_age"] == "recent" and p[0].fields["size_tercile"] in {"small", "mid", "large"}
    q = build_payloads(s, pd.Timestamp("2019-06-28"), ["ABCD", "EFGH"], "quality_growth", salt="run1")
    assert "roic_pct" in q[0].fields["metrics"] and q[0].fields["metrics"]["profit_growth_z"] is None
    m1 = pseudonyms_for(pd.Timestamp("2019-06-28"), ["ABCD", "EFGH", "IJKL"], "run1")
    m2 = pseudonyms_for(pd.Timestamp("2019-07-31"), ["ABCD", "EFGH", "IJKL"], "run1")
    assert set(m1.values()) == {"STOCK_A", "STOCK_B", "STOCK_C"} and m1 != m2 or set(m1.values()) == set(m2.values())
    assert_clean(text, {"ABCD", "TURKISH AIRLINES"})
    with pytest.raises(ValueError):
        assert_clean(text + " THYAO", {"THYAO"})
    with pytest.raises(ValueError):
        assert_clean("as of 31.12.2018", set())
    with pytest.raises(ValueError):
        assert_clean("revenue 62853000000", set())


def test_cache_key_sensitivity(tmp_path):
    base = dict(payload_text="p", prompt_hash="h1", schema_hash="s1", model_digest="d1", options={"seed": 1})
    k = cache_key(**base)
    assert cache_key(**{**base, "model_digest": "d2"}) != k
    assert cache_key(**{**base, "options": {"seed": 2}}) != k
    assert cache_key(**{**base, "payload_text": "q"}) != k
    c = JuryCache(tmp_path)
    assert c.get(k) is None
    c.put(k, {"record": {"score": 0.5}})
    assert c.get(k)["record"]["score"] == 0.5
    c.put(cache_key(**{**base, "payload_text": "q"}), {"key": "bogus", "record": {}})  # key is overwritten by put()
    assert c.get(cache_key(**{**base, "payload_text": "q"})) is not None


class FakeClient:
    def __init__(self, answers):
        self.answers = list(answers)
        self.model = "fake"
        self.options = {"seed": 1, "temperature": 0}
        self.calls = 0

    def chat_json(self, system, user, schema):
        self.calls += 1
        content = self.answers.pop(0)
        try:
            parsed = json.loads(content)
            err = None
        except json.JSONDecodeError as exc:
            parsed, err = None, str(exc)
        return ChatResult(
            content, parsed if isinstance(parsed, dict) else None, 100, 50, 10.0, "fake", self.options, err
        )


def _agent(tmp_path, answers, retries=1):
    prompts = AgentPrompts.load("v1")
    return JuryAgent("value", FakeClient(answers), prompts, JuryCache(tmp_path), "digest", {"ABCD"}, retries)


def test_agent_retries_once_then_abstains(tmp_path):
    payload = build_payloads(_screen(), pd.Timestamp("2019-06-28"), ["EFGH"], "value", salt="r")[0]
    bad = json.dumps(
        {
            "candidate_id": "WRONG",
            "key_strengths": [],
            "key_risks": [],
            "rationale": "x",
            "cited_metrics": [],
            "score": 0.5,
            "confidence": 0.5,
        }
    )
    good = json.dumps(
        {
            "candidate_id": payload.candidate_id,
            "key_strengths": ["a"],
            "key_risks": ["b"],
            "rationale": "book_to_market_pct high",
            "cited_metrics": ["book_to_market_pct"],
            "score": 0.8,
            "confidence": 0.7,
        }
    )
    a = _agent(tmp_path, [bad, good])
    rec = a.judge("2019-06-28", payload)
    assert not rec.abstain and rec.score == 0.8 and rec.attempts == 2 and rec.ticker == "EFGH"
    # cache hit on the second call: no model call
    rec2 = a.judge("2019-06-28", payload)
    assert rec2.cache_hit and rec2.score == 0.8 and a.client.calls == 2
    b = _agent(tmp_path / "b", [bad, "not json"])
    rec3 = b.judge("2019-06-28", payload)
    assert rec3.abstain and rec3.score is None and rec3.attempts == 2


def test_consensus_and_random_null():
    rows = []
    for t, sc in [("ABCD", [0.9, 0.8, 0.7]), ("EFGH", [0.5, np.nan, 0.6]), ("IJKL", [np.nan, np.nan, 0.9])]:
        for a, s in zip(AGENTS, sc, strict=True):
            rows.append(
                {
                    "as_of": "2019-06-28",
                    "ticker": t,
                    "agent": a,
                    "score": s,
                    "composite_rank": {"ABCD": 1, "EFGH": 2, "IJKL": 3}[t],
                }
            )
    c = consensus(pd.DataFrame(rows), min_votes=2, theta=0.75)
    by = c.set_index("ticker")
    assert by.loc["ABCD", "J"] == pytest.approx(0.8) and by.loc["ABCD", "buy"] and by.loc["ABCD", "rank_J"] == 1
    assert by.loc["EFGH", "J"] == pytest.approx(0.55) and by.loc["EFGH", "n_votes"] == 2 and not by.loc["EFGH", "buy"]
    assert np.isnan(by.loc["IJKL", "J"]) and np.isnan(by.loc["IJKL", "rank_J"])
    cands = pd.DataFrame({"as_of": ["2019-06-28"] * 3, "ticker": ["ABCD", "EFGH", "IJKL"], "composite_rank": [1, 2, 3]})
    r1, r2 = random_scores(cands, 1), random_scores(cands, 2)
    assert len(r1) == 9 and (r1["score"].between(0, 1)).all() and not np.allclose(r1["score"], r2["score"])
    assert consensus(r1)["J"].notna().all()


def test_forbidden_lexicon_keeps_distinctive_name_words_and_drops_generic_ones():
    import pandas as pd

    from bistmas.jury.run import forbidden_tokens_from

    master = pd.DataFrame(
        {
            "ticker": ["TDGYO", "DOGTR", "GATEX", "AAAA", "BBBB", "CCCC"],
            "title": [
                "TREND GAYRİMENKUL YATIRIM ORTAKLIĞI A.Ş.",
                "DOĞAN TREND OTOMOTİV TİCARET HİZMET VE TEKNOLOJİ A.Ş.",
                "GATE GROUP TEKNOLOJİ MEDYA VE SİBER GÜVENLİK HİZMETLERİ A.Ş.",
                "FLORENCE SANAYİ A.Ş.",
                "ANADOLU SANAYİ A.Ş.",
                "ANADOLU TİCARET A.Ş.",
            ],
        }
    )
    toks = forbidden_tokens_from(master, max_title_count=1)
    assert {"TDGYO", "DOGTR", "GATEX", "FLORENCE", "SİBER", "GÜVENLİK"} <= toks
    for generic in ("TREND", "GROUP", "GATE", "SANAYİ", "TEKNOLOJİ", "ANADOLU", "A.Ş."):
        assert generic not in toks, generic


def test_payload_percentiles_follow_the_higher_is_better_orientation():
    import pandas as pd

    from bistmas.jury.payload import CONTEXT_FIELDS, allowed_metric_names, build_payloads

    as_of = pd.Timestamp("2019-06-28")
    rows = []
    for i, t in enumerate(["AAAA", "BBBB", "CCCC", "DDDD", "EEEE"]):
        rows.append(
            {
                "as_of": as_of,
                "ticker": t,
                "eligible": True,
                "template": "industrial",
                "sector_group": "MANUFACTURING",
                "mcap_tl": 1e9 * (i + 1),
                "statement_age_days": 30,
                "momentum_12_1": 0.1 * i,
                "z_momentum_12_1": 0.5 * i - 1,
                "vol_6m": 0.2 + 0.1 * i,  # EEEE is the MOST volatile
                "z_vol_6m": 1 - 0.5 * i,
                "mdd_6m": -0.1 * (i + 1),  # EEEE has the deepest drawdown
                "z_mdd_6m": 1 - 0.5 * i,
                "adv_6m_tl": 1e7 * (i + 1),
                "z_adv_6m_tl": 0.5 * i - 1,
            }
        )
    screen = pd.DataFrame(rows)
    ps = {p.ticker: p for p in build_payloads(screen, as_of, ["AAAA", "EEEE"], "risk_timing", "salt")}
    a, e = ps["AAAA"].fields["metrics"], ps["EEEE"].fields["metrics"]
    assert a["vol_6m_pct"] > e["vol_6m_pct"]  # least volatile name ranks highest
    assert a["mdd_6m_pct"] > e["mdd_6m_pct"]  # shallowest drawdown ranks highest
    assert e["momentum_12_1_pct"] > a["momentum_12_1_pct"] and e["adv_6m_tl_pct"] > a["adv_6m_tl_pct"]
    allowed = allowed_metric_names(ps["AAAA"])
    assert set(CONTEXT_FIELDS) <= allowed and "vol_6m" in allowed and "vol_6m_pct" in allowed


def test_run_jury_cross_family_records_one_model_per_agent(tmp_path):
    """A cross-family jury gives each agent its own client; the summary records the model per agent."""
    import re

    from bistmas.jury.run import run_jury
    from bistmas.shared.config import PathsConfig, Settings

    class EchoClient(FakeClient):
        def __init__(self, model):
            super().__init__([])
            self.model = model

        def chat_json(self, system, user, schema):
            self.calls += 1
            cid = re.search(r'candidate_id "([A-Z_]+)"', user).group(1)
            content = json.dumps(
                {
                    "candidate_id": cid,
                    "key_strengths": ["s"],
                    "key_risks": ["r"],
                    "rationale": "grounded on the listed metrics",
                    "cited_metrics": [],
                    "score": 0.6,
                    "confidence": 0.5,
                }
            )
            return ChatResult(content, json.loads(content), 100, 50, 10.0, self.model, self.options, None)

    settings = Settings(root=tmp_path, paths=PathsConfig().resolve(tmp_path))
    settings.jury.model_digest = ""
    clients = {"quality_growth": EchoClient("qwen"), "value": EchoClient("gemma"), "risk_timing": EchoClient("llama")}
    master = pd.DataFrame({"ticker": ["ABCD", "EFGH", "IJKL", "MNOP", "QRST"], "title": ["X A.Ş."] * 5})
    summary = run_jury(settings, "xfam", _screen(), master, top_n=3, clients=clients)
    assert summary["agent_models"] == {"quality_growth": "qwen", "value": "gemma", "risk_timing": "llama"}
    assert all(c.calls == 3 for c in clients.values())
    scores = pd.read_parquet(tmp_path / "data" / "runs" / "xfam" / "jury_scores.parquet")
    assert set(scores["model"]) == {"qwen", "gemma", "llama"} and len(scores) == 9
