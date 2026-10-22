"""H2 robustness audit and human rubric (fake LLM client, synthetic data; no Ollama)."""

import json
import re

import numpy as np
import pandas as pd
import pytest

from bistmas.cli import build_parser, main
from bistmas.jury.agents import AGENTS, AgentPrompts, JuryAgent
from bistmas.jury.audits.robustness import (
    audit_robustness,
    compare_month,
    perturb_payload,
    sample_months,
)
from bistmas.jury.audits.rubric import (
    BLIND_COLUMNS,
    export_rubric,
    sample_rationales,
    score_rubric,
    summarise_payload,
    weighted_kappa,
)
from bistmas.jury.cache import JuryCache
from bistmas.jury.consensus import consensus
from bistmas.jury.ollama_client import ChatResult
from bistmas.jury.payload import build_payloads
from bistmas.shared.config import Settings, load_settings

TICKERS = ["ABCD", "EFGH", "IJKL", "MNOP", "QRST", "UVWX"]
Z_COLS = [
    "roic", "operating_margin", "gross_margin", "revenue_growth", "profit_growth", "earnings_yield",
    "book_to_market", "ebitda_yield", "fcf_yield", "momentum_12_1", "vol_6m", "mdd_6m", "adv_6m_tl",
]  # fmt: skip


def _screen(as_of):
    rows = []
    for i, t in enumerate(TICKERS):
        row = {
            "as_of": pd.Timestamp(as_of),
            "ticker": t,
            "eligible": True,
            "rank": i + 1,
            "template": "industrial",
            "sector_group": "MANUFACTURING",
            "mcap_tl": 1e9 * (i + 1),
            "statement_age_days": 40 + i * 50,
        }
        for c in Z_COLS:
            row[c] = 0.1 * (i + 1)
            row[f"z_{c}"] = np.nan if c == "profit_growth" else 0.8 - 0.3 * i  # candidate 0 best
        rows.append(row)
    return pd.DataFrame(rows)


class ScoreByZ:
    """Fake client: score = 0.5 + 0.1 * mean(z); with ``invert`` the paraphrase prompts flip the ordering."""

    def __init__(self, invert=False):
        self.model, self.options, self.calls, self.invert = "fake", {"seed": 1, "temperature": 0}, 0, invert

    def chat_json(self, system, user, schema):
        self.calls += 1
        zs = [float(x) for x in re.findall(r'_z": (-?\d+\.?\d*)', user)]
        s = 0.5 + 0.1 * float(np.mean(zs))
        if self.invert and "Reply with the JSON object required" in system:
            s = 1.0 - s
        s = min(1.0, max(0.0, round(s, 3)))
        cid = re.search(r'"candidate_id": "(STOCK_[A-Z])"', user).group(1)
        obj = {
            "candidate_id": cid, "key_strengths": [], "key_risks": [], "rationale": "ok",
            "cited_metrics": [], "score": s, "confidence": 0.5,
        }  # fmt: skip
        return ChatResult(json.dumps(obj), obj, 10, 10, 1.0, "fake", self.options, None)


@pytest.fixture
def setup(tmp_path):
    months = ["2019-03-29", "2019-06-28", "2020-03-31", "2021-06-30"]
    screens = {m: _screen(m) for m in months}
    prompts = AgentPrompts.load("v2")
    para = AgentPrompts.load("v2_paraphrase")
    cache = JuryCache(tmp_path / "cache")
    rows = []
    base = ScoreByZ()
    for m in months:
        for a in AGENTS:
            ps = build_payloads(screens[m], pd.Timestamp(m), TICKERS, a, salt="run")
            ja = JuryAgent(a, base, prompts, cache, "d", set())
            for i, p in enumerate(ps):
                r = ja.judge(m, p)
                rows.append(
                    {"as_of": m, "ticker": p.ticker, "agent": a, "score": r.score, "abstain": r.abstain,
                     "rationale": f"{a} rationale {i}", "cache_key": r.cache_key, "composite_rank": i + 1}
                )  # fmt: skip
    scores = pd.DataFrame(rows)
    return months, screens, prompts, para, cache, scores


def _run(setup, client, noise=0.25, n_months=2):
    months, screens, prompts, para, cache, scores = setup

    def payloads_for(m, a):
        return build_payloads(screens[m], pd.Timestamp(m), TICKERS, a, salt="run")

    def make(a, pr):
        return JuryAgent(a, client, pr, cache, "d", set())

    return audit_robustness(
        scores, payloads_for, make, prompts, para, months[:n_months], noise_z=noise, top_k=3, seed=7
    )


def test_perturb_payload_bounds_and_determinism():
    p = build_payloads(_screen("2019-03-29"), pd.Timestamp("2019-03-29"), TICKERS, "quality_growth", salt="s")[0]
    a = perturb_payload(p, 0.25, np.random.default_rng(1))
    b = perturb_payload(p, 0.25, np.random.default_rng(1))
    assert a.fields == b.fields and a.fields != p.fields
    for k, v in p.fields["metrics"].items():
        if k.endswith("_z") and v is not None:
            assert abs(a.fields["metrics"][k] - v) <= 0.25 + 0.006
        else:
            assert a.fields["metrics"][k] == v  # percentiles and nulls untouched


def test_sample_months_stratified_and_dev_slice():
    ms = [f"{y}-06-28" for y in range(2018, 2025)] + [f"{y}-12-30" for y in range(2018, 2025)]
    s = sample_months(ms, 6, 1, exclude_years=(2018, 2019))
    assert len(s) == 6 and all(int(m[:4]) > 2019 for m in s) and s == sample_months(ms, 6, 1, (2018, 2019))
    assert sample_months(ms[:3], 6, 1) == sorted(ms[:3])


def test_compare_month_extremes():
    scores = pd.DataFrame(
        {
            "as_of": "m",
            "ticker": list("abcde") * 3,
            "agent": [x for x in AGENTS for _ in range(5)],
            "score": [0.9, 0.8, 0.7, 0.6, 0.5] * 3,
        }
    )
    o = consensus(scores)
    same = compare_month(o, o, 3)
    assert same["kendall_tau"] == pytest.approx(1.0) and same["top_k_overlap"] == 1.0
    rev = o.copy()
    rev["J"] = rev["J"].to_numpy()[::-1]
    rev["rank_J"] = rev["rank_J"].to_numpy()[::-1]
    r = compare_month(o, rev, 3)
    assert r["kendall_tau"] == pytest.approx(-1.0) and r["top_k_overlap"] == pytest.approx(1 / 3)


def test_audit_robustness_stable_model(setup):
    client = ScoreByZ()
    out = _run(setup, client, noise=0.0)
    assert out["summary"]["noise"]["kendall_tau_mean"] == pytest.approx(1.0)  # zero noise: identical payloads
    assert out["summary"]["paraphrase"]["top_k_overlap_mean"] == 1.0
    # noise=0 is served from the cache; only the paraphrase prompts are new calls: 2 months x 3 agents x 6
    assert client.calls == out["calls"] and 0 < out["calls"] <= 2 * 3 * len(TICKERS)
    noisy = _run(setup, ScoreByZ(), noise=0.25)
    assert noisy["summary"]["noise"]["n_months"] == 2 and noisy["noise_z"] == 0.25
    assert -1.0 <= noisy["summary"]["noise"]["kendall_tau_mean"] <= 1.0


def test_audit_robustness_detects_paraphrase_sensitivity(setup):
    out = _run(setup, ScoreByZ(invert=True), noise=0.0)
    assert out["summary"]["paraphrase"]["kendall_tau_mean"] == pytest.approx(-1.0)
    assert out["summary"]["paraphrase"]["top_k_overlap_median"] == 0.0
    assert out["summary"]["noise"]["kendall_tau_median"] == pytest.approx(1.0)


def test_h2_config_defaults_and_off_by_default():
    s = load_settings()
    assert s.audit.robustness_months == 6 and s.audit.robustness_noise_z == 0.25 and s.audit.rubric_n == 30
    assert Settings().audit.robustness_top_k == 3
    ns = build_parser().parse_args(["audit", "--jury-run", "x"])
    assert ns.robustness is False and ns.rubric_export is None and ns.rubric_score is None


def test_weighted_kappa_known_values():
    a = [1, 2, 3, 4, 5, 3, 2]
    assert weighted_kappa(a, a) == pytest.approx(1.0)
    assert weighted_kappa([1, 1, 5, 5], [5, 5, 1, 1]) == pytest.approx(-1.0)
    assert np.isnan(weighted_kappa([3, 3, 3], [3, 3, 3]))
    near = weighted_kappa([1, 2, 3, 4, 5], [2, 3, 4, 5, 5])
    far = weighted_kappa([1, 2, 3, 4, 5], [5, 4, 1, 2, 3])
    assert near > far


def test_rubric_export_and_score(setup, tmp_path):
    *_, cache, scores = setup
    res = export_rubric(scores, cache, 10, 3, tmp_path, "run")
    blind = pd.read_csv(res["blind"], dtype={"id": str})
    key = pd.read_csv(res["key"], dtype={"id": str})
    assert list(blind.columns) == BLIND_COLUMNS and len(blind) == len(key) == 10
    assert blind[["groundedness", "clarity", "consistency"]].isna().all().all()
    text = blind.to_csv()
    assert not any(t in text for t in TICKERS) and "20" not in "".join(blind["payload_summary"])
    assert ": z=" in blind["payload_summary"].iloc[0]
    assert set(key.columns) == {"id", "as_of", "ticker", "agent", "cache_key"}
    again = export_rubric(scores, cache, 10, 3, tmp_path / "b", "run")
    assert again["n"] == 10
    assert pd.read_csv(again["blind"]).equals(pd.read_csv(res["blind"]))  # seeded
    assert sample_rationales(scores, 1000, 1).shape[0] == len(scores)

    rng = np.random.default_rng(0)
    base = rng.integers(1, 6, size=10)
    files = []
    for r in range(3):
        d = blind.copy()
        for c in ("groundedness", "clarity", "consistency"):
            d[c] = np.clip(base + rng.integers(-1, 2, size=10), 1, 5)
        f = tmp_path / f"rater{r + 1}.csv"
        d.to_csv(f, index=False)
        files.append(f)
    out = score_rubric(files)
    assert out["n_items"] == 10 and out["raters"] == ["rater1", "rater2", "rater3"]
    g = out["criteria"]["groundedness"]
    assert len(g["pairwise_kappa"]) == 3 and -1 <= g["mean_kappa"] <= 1 and g["n_complete"] == 10
    bad = blind.copy()
    bad["groundedness"] = 9
    bad.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError):
        score_rubric([files[0], tmp_path / "bad.csv"])


def test_summarise_payload_is_pseudonymised():
    s = summarise_payload({"sector_group": "x", "metrics": {"roic_z": 1.0, "roic_pct": None}})
    assert "roic: z=1.0 pct=n/a" in s


def test_cli_rubric_score_only_needs_no_model(tmp_path, capsys):
    f = tmp_path / "a.csv"
    pd.DataFrame({"id": ["R01", "R02"], "groundedness": [1, 5], "clarity": [2, 4], "consistency": [3, 3]}).to_csv(
        f, index=False
    )
    h = tmp_path / "c.csv"
    h.write_text(f.read_text())
    assert main(["audit", "--rubric-score", str(f), str(h)]) == 0
    assert '"n_items": 2' in capsys.readouterr().out
