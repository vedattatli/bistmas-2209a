"""Offline end-to-end smoke test of ``run_hybrid`` on a tiny synthetic store and a synthetic finished jury run."""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bistmas.experiments import hybrid
from bistmas.jury.agents import AGENTS
from bistmas.jury.consensus import consensus
from bistmas.jury.run import CONSENSUS_FILE, SCORES_FILE
from bistmas.shared.calendar import TradingCalendar
from bistmas.shared.config import PathsConfig, Settings
from bistmas.shared.store import write_json, write_parquet

TICKERS = [f"T{i:02d}" for i in range(16)]
FIRST, LAST = date(2018, 3, 30), date(2018, 12, 28)


@pytest.fixture
def world(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    sessions = pd.bdate_range("2018-01-01", "2019-03-29")
    cal = TradingCalendar([d.date() for d in sessions])
    px = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0.0005, 0.015, (len(sessions), len(TICKERS))), axis=0),
        index=sessions,
        columns=TICKERS,
    )
    prices = px.stack().rename("vwap_adj").rename_axis(["date", "ticker"]).reset_index()
    prices["usdtry"] = 5.0 + 0.002 * prices["date"].map({d: i for i, d in enumerate(sessions)})
    membership = pd.DataFrame(
        {
            "index_code": "XU100",
            "ticker": TICKERS,
            "effective_start": pd.Timestamp("2017-01-01"),
            "effective_end": pd.NaT,
            "membership_known_since": pd.Timestamp("2017-01-01"),
            "source": "t",
        }
    )
    decisions = [pd.Timestamp(d) for d in cal.month_ends(FIRST, LAST)]
    screen_rows, score_rows = [], []
    cands_by_month = {}
    for k, m in enumerate(decisions):
        order = list(rng.permutation(TICKERS))
        for r, t in enumerate(order, start=1):
            screen_rows.append({"as_of": m, "ticker": t, "rank": r, "eligible": r <= 12})
        cands = order[:10]
        cands_by_month[m] = cands
        for r, t in enumerate(cands, start=1):
            for a in AGENTS:
                if k == 0:
                    s = 0.1  # nobody reaches any theta: BUY-only must hold cash
                elif k == 1:
                    s = 0.5  # exact ties everywhere: rank_J falls back on the composite rank
                else:
                    s = round(float(rng.uniform()) * 20) / 20
                abst = k >= 2 and rng.uniform() < 0.15
                if k == 2 and r == 1:
                    abst = True  # the best composite candidate is unanimously abstained (no J)
                score_rows.append(
                    {
                        "as_of": str(m.date()),
                        "ticker": t,
                        "candidate_id": "x",
                        "agent": a,
                        "score": None if abst else s,
                        "abstain": abst,
                        "abstain_reason": "no_data" if abst else "",
                        "composite_rank": r,
                    }
                )
    scores = pd.DataFrame(score_rows)
    settings = Settings(root=tmp_path, paths=PathsConfig().resolve(tmp_path))
    settings.paths.store.mkdir(parents=True)
    settings.rebalance.first_decision, settings.rebalance.last_decision = FIRST.isoformat(), LAST.isoformat()
    settings.backtest.top_n, settings.backtest.headline_top_n = [3, 5], 3
    settings.backtest.cost_sensitivity_bps = [0, 30]
    settings.backtest.random_jury_seeds = 3
    run_dir = settings.paths.runs / "syn"
    run_dir.mkdir(parents=True)
    write_parquet(scores, run_dir / SCORES_FILE)
    cons = consensus(
        scores.assign(score=pd.to_numeric(scores["score"], errors="coerce")),
        min_votes=settings.jury.min_votes,
        theta=settings.jury.theta,
    )
    write_parquet(cons, run_dir / CONSENSUS_FILE)
    write_json({"run": "syn"}, run_dir / "jury_summary.json")
    inputs = SimpleNamespace(membership=membership, panel=SimpleNamespace(close_adj=px))
    monkeypatch.setattr(hybrid, "load_inputs", lambda s: (inputs, prices, {"synthetic": True}))
    monkeypatch.setattr(hybrid, "load_calendar", lambda store: cal)
    monkeypatch.setattr(hybrid, "run_screens", lambda s, i, d: pd.DataFrame(screen_rows))
    return settings, cons, cands_by_month


def test_run_hybrid_end_to_end(world):
    settings, cons, cands = world
    rep = hybrid.run_hybrid(settings, "syn", run_name="syn-hybrid")
    out = settings.paths.runs / "syn-hybrid"
    assert (out / "report.json").exists() and (out / "hybrid_top3_periods.csv").exists()
    assert json.loads((out / "report.json").read_text())["run"] == "syn-hybrid"

    for lab in ("hybrid_top3", "det_top3", "hybrid_top5", "det_top5", "bench_ew_pit", "bench_ew_eligible"):
        for num in ("tl", "usd"):
            assert np.isfinite(rep["strategies"][lab][num]["cagr"])
    h1 = rep["strategies"]["hybrid_top3"]["vs_deterministic"]
    assert np.isfinite(h1["mean_monthly_diff"]) and {"newey_west", "bootstrap_ci_95"} <= set(h1)
    assert {"cost_sensitivity", "haircut_sensitivity", "mde"} <= set(rep["tables"])
    assert len(rep["tables"]["cost_sensitivity"]) > 0 and len(rep["tables"]["haircut_sensitivity"]) > 0
    null = rep["random_jury_null"]
    assert null["n"] == 3 and 0.0 <= null["hybrid_percentile"] <= 1.0 and np.isfinite(null["cagr_mean"])

    sec = rep["secondary"]
    assert {"bonferroni", "buy_only", "echo_spearman", "incremental_ic", "agreement"} <= set(sec)
    assert sec["bonferroni"]["n_configurations"] >= 1
    assert set(sec["buy_only"]) == {"0.70", "0.75", "0.80"}
    for b in sec["buy_only"].values():
        assert b["buy_count_min"] <= b["buy_count_max"]
        assert np.isfinite(b["tl"]["cagr"])
    # month 0 has no BUY at any theta: the portfolio is in cash then
    assert all(b["months_in_cash"] >= 1 for b in sec["buy_only"].values())
    assert sec["buy_only"]["0.75"]["buy_count_min"] == 0
    assert sec["echo_spearman"]["n_months"] >= 1
    assert sec["incremental_ic"]["n_months"] >= 1
    ag = sec["agreement"]
    assert 1.0 <= ag["kish_votes_mean"] <= 3.0 and len(ag["pairwise"]) == 3
    assert ag["krippendorff_alpha_interval"] is not None


def test_run_hybrid_selection_and_consensus_sanity(world):
    settings, cons, cands = world
    hybrid.run_hybrid(settings, "syn", run_name="syn-hybrid")
    assert cons["J"].dropna().between(0, 1).all()
    ties = cons[cons["as_of"] == sorted(cons["as_of"].unique())[1]].sort_values("rank_J")
    assert ties["composite_rank"].tolist() == sorted(ties["composite_rank"])  # exact J ties -> composite order
    assert cons[cons["J"].isna()]["rank_J"].isna().all()
    out = settings.paths.runs / "syn-hybrid"
    periods = pd.read_csv(out / "hybrid_top3_periods.csv")
    assert (periods["n_holdings"] <= 3).all() and len(periods) == len(cands)
    # the hybrid Top-3 of every month comes from that month's candidate set and has a J
    by_month = {pd.Timestamp(m): g for m, g in cons.groupby("as_of")}
    for m, c in cands.items():
        g = by_month[m].dropna(subset=["rank_J"]).sort_values("rank_J")
        top3 = g["ticker"].tolist()[:3]
        assert set(top3) <= set(c)
        assert g["J"].iloc[:3].is_monotonic_decreasing
