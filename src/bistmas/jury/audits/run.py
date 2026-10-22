"""Audit orchestration: builds samples from a finished jury run and writes ``data/runs/audits/<name>.json``."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from bistmas.data.security_master import load_ticker_aliases
from bistmas.jury.agents import AGENTS, AgentPrompts, JuryAgent
from bistmas.jury.audits.groundedness import audit_groundedness
from bistmas.jury.audits.probes import date_arms, direct_elicitation, lap_probe, reidentification_probe, test_retest
from bistmas.jury.audits.robustness import audit_robustness, sample_months
from bistmas.jury.cache import JuryCache
from bistmas.jury.ollama_client import OllamaClient
from bistmas.jury.payload import build_payloads
from bistmas.jury.run import GENERIC_TITLE_WORDS, forbidden_tokens_from
from bistmas.shared.config import CONFIG_DIR, Settings
from bistmas.shared.ledger import record_run
from bistmas.shared.log import get_logger
from bistmas.shared.store import read_parquet, utc_now_iso, write_json

log = get_logger("jury.audits")

ALL_AUDITS = ("groundedness", "reidentification", "elicitation", "date_arms", "test_retest", "lap")


AUDIT_SHARE = 0.10  # documented audit sample: at least 10 % of the (as_of, ticker) candidates of the run
DEV_SLICE_YEARS = (2018, 2019)  # prompt development slice (docs/preregistration.md): never used for test-retest


def _stratified_sample(scores: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """At least ``n`` (as_of, ticker) pairs, ``ceil(n / months)`` per month (so every year is covered), never truncated.

    Within a month the pairs are sorted by composite rank, cut into ``k`` equal bins and one pair is drawn
    from each bin, so the sample spans the whole rank range as well.
    """
    cols = ["as_of", "ticker"] + (["composite_rank"] if "composite_rank" in scores.columns else [])
    pairs = scores[cols].drop_duplicates(["as_of", "ticker"])
    months = sorted(pairs["as_of"].unique())
    per_month = max(1, math.ceil(n / max(len(months), 1)))
    rng = np.random.default_rng(seed)
    picks = []
    for m in months:
        g = pairs[pairs["as_of"] == m]
        if "composite_rank" in g.columns:
            g = g.assign(_r=pd.to_numeric(g["composite_rank"], errors="coerce")).sort_values(["_r", "ticker"])
        k = min(per_month, len(g))
        for chunk in np.array_split(np.arange(len(g)), k):
            if len(chunk):
                picks.append(g.iloc[[int(rng.choice(chunk))]])
    out = pd.concat(picks)[["as_of", "ticker"]]
    return out.reset_index(drop=True)


def _retest_sample(sample: pd.DataFrame, n: int) -> pd.DataFrame:
    """``n`` pairs spread evenly over the stratified sample's dates, excluding the prompt-development years."""
    years = pd.to_datetime(sample["as_of"]).dt.year
    pool = sample[~years.isin(DEV_SLICE_YEARS)]
    pool = (pool if len(pool) else sample).sort_values(["as_of", "ticker"]).reset_index(drop=True)
    if len(pool) <= n:
        return pool
    idx = np.unique(np.linspace(0, len(pool) - 1, n).round().astype(int))
    return pool.iloc[idx].reset_index(drop=True)


class AliasMap:
    """Code and title valid on a decision date for companies renamed later (``configs/ticker_aliases.csv``).

    Prices and features are keyed by the current code; the model only knows the company by the code and name it
    traded under at the time. Only ``kind == rename`` rows apply (a merger target is a different company).
    """

    def __init__(self, aliases: pd.DataFrame | None = None) -> None:
        if aliases is None:
            aliases = load_ticker_aliases(CONFIG_DIR / "ticker_aliases.csv")
        a = aliases[aliases["kind"] == "rename"]
        self._rows: dict[str, tuple[str, pd.Timestamp, str]] = {}
        for r in a.itertuples(index=False):
            note = str(getattr(r, "note", "") or "")
            old_title = note.split(" -> ")[0].strip() if " -> " in note else ""
            self._rows[str(r.new_ticker)] = (str(r.old_ticker), pd.Timestamp(r.effective_date), old_title)

    def identity(self, ticker: str, as_of: pd.Timestamp, current_title: str) -> tuple[str, str, set[str]]:
        """(code at ``as_of``, title at ``as_of``, accepted codes)."""
        row = self._rows.get(ticker)
        if row and pd.Timestamp(as_of) < row[1]:
            return row[0], (row[2] or current_title), {row[0], ticker}
        return ticker, current_title, {ticker}

    def old_title_words(self) -> set[str]:
        out: set[str] = set()
        for _old, _eff, title in self._rows.values():
            out |= {w.strip().upper() for w in title.replace(".", " ").split() if len(w.strip()) >= 5}
        return out - GENERIC_TITLE_WORDS


def _quarter_span(screens: pd.DataFrame, scores: pd.DataFrame) -> int:
    """Number of distinct fiscal quarters between the first and last decision of the run (quarter chance = 1/n)."""
    d = pd.to_datetime(scores["as_of"])
    if d.empty:
        return 28
    return max(1, (d.max().year - d.min().year) * 4 + (d.max().month - 1) // 3 - (d.min().month - 1) // 3 + 1)


def run_audits(
    settings: Settings,
    jury_run: str,
    which: tuple[str, ...] = ALL_AUDITS,
    sample_size: int | None = None,
    seed: int = 20260923,
    client: OllamaClient | None = None,
    robustness: bool = False,
) -> dict[str, Any]:
    """Run the audits in ``which``; ``robustness`` (H2, off by default) adds the noise/paraphrase re-scoring.

    The robustness result is written to ``audits/<run>_robustness.json`` so it never overwrites the standard audit file.
    """
    runs = settings.paths.runs
    scores = read_parquet(runs / jury_run / "jury_scores.parquet")
    screens = read_parquet(settings.paths.store / "features_monthly.parquet")
    screens["as_of"] = pd.to_datetime(screens["as_of"])
    from bistmas.data.prices import load_prices
    from bistmas.data.security_master import load_security_master

    master = load_security_master(settings.paths.store)
    titles = master[master["is_current"]].set_index("ticker")["title"].to_dict()
    aliases = AliasMap()
    forbidden = forbidden_tokens_from(master) | aliases.old_title_words()
    cache = JuryCache(runs / "jury_cache")
    client = client or OllamaClient(settings.jury.host, settings.jury.model, settings.jury.sampling_options())
    prompts = AgentPrompts.load(settings.jury.prompt_version)
    if sample_size is None:  # documented share: 10 % of the run's candidates
        sample_size = math.ceil(AUDIT_SHARE * len(scores[["as_of", "ticker"]].drop_duplicates()))
    out: dict[str, Any] = {
        "jury_run": jury_run,
        "generated_at": utc_now_iso(),
        "sample_size": sample_size,
        "audits": {},
    }
    sample = _stratified_sample(scores, sample_size, seed)

    def payload_for(as_of: pd.Timestamp, ticker: str, agent: str):
        s = screens[screens["as_of"] == pd.Timestamp(as_of)]
        cands = s.dropna(subset=["rank"]).sort_values("rank").head(settings.scoring.top_n_screen)["ticker"].tolist()
        ps = build_payloads(s, pd.Timestamp(as_of), cands, agent, salt=jury_run)
        return next((p for p in ps if p.ticker == ticker), None)

    def payload_list(m: str, agent: str):
        s = screens[screens["as_of"] == pd.Timestamp(m)]
        cands = s.dropna(subset=["rank"]).sort_values("rank").head(settings.scoring.top_n_screen)["ticker"].tolist()
        return build_payloads(s, pd.Timestamp(m), cands, agent, salt=jury_run)

    if "groundedness" in which:
        out["audits"]["groundedness"] = audit_groundedness(scores, cache, forbidden)
    if "reidentification" in which:
        items = []
        quarters: set[tuple[int, int]] = set()
        for rec in sample.itertuples(index=False):
            p = payload_for(rec.as_of, rec.ticker, "quality_growth")
            row = screens[(screens["as_of"] == pd.Timestamp(rec.as_of)) & (screens["ticker"] == rec.ticker)]
            if p is None or row.empty or pd.isna(row["statement_period_end"].iloc[0]):
                continue
            pe = pd.Timestamp(row["statement_period_end"].iloc[0])
            _code, _title, accepted = aliases.identity(rec.ticker, pd.Timestamp(rec.as_of), titles.get(rec.ticker, ""))
            items.append((p, accepted, pe.year, (pe.month - 1) // 3 + 1))
            quarters.add((pe.year, (pe.month - 1) // 3 + 1))
        universe = int(screens.groupby("as_of")["ticker"].nunique().median()) if len(screens) else 100
        out["audits"]["reidentification"] = reidentification_probe(
            client, items, universe, n_quarters=_quarter_span(screens, scores)
        )
    if "elicitation" in which or "lap" in which:
        prices = load_prices(settings.paths.store, columns=["ticker", "date", "close_raw", "close_adj"])
        prices["date"] = pd.to_datetime(prices["date"])
        cases = []
        for rec in sample.itertuples(index=False):
            d = pd.Timestamp(rec.as_of)
            px = prices[(prices["ticker"] == rec.ticker) & (prices["date"] <= d)].sort_values("date")
            nxt = prices[
                (prices["ticker"] == rec.ticker) & (prices["date"] <= d + pd.DateOffset(months=1))
            ].sort_values("date")
            if px.empty or nxt.empty:
                continue
            true_close = float(px["close_raw"].iloc[-1])
            direction = "up" if float(nxt["close_adj"].iloc[-1]) > float(px["close_adj"].iloc[-1]) else "down"
            code, title, _acc = aliases.identity(rec.ticker, d, titles.get(rec.ticker, rec.ticker))
            cases.append(
                {
                    "ticker": code,
                    "company": title,
                    "date": str(d.date()),
                    "true_close": true_close,
                    "true_next_direction": direction,
                    "period": "in_sample" if d.year <= 2024 else "post_cutoff",
                }
            )
        if "elicitation" in which:
            out["audits"]["elicitation"] = direct_elicitation(client, cases)
        if "lap" in which:
            out["audits"]["lap"] = (
                lap_probe(settings.jury.host, settings.jury.model, cases)
                if client.supports_logprobs()
                else {"skipped": "no token log-probs"}
            )
    if "date_arms" in which:
        items = []
        for rec in sample.itertuples(index=False):
            for agent in AGENTS:
                p = payload_for(rec.as_of, rec.ticker, agent)
                if p is not None:
                    items.append(
                        {
                            "payload": p,
                            "agent": agent,
                            "as_of": rec.as_of,
                            "ticker": rec.ticker,
                            "company": aliases.identity(
                                rec.ticker, pd.Timestamp(rec.as_of), titles.get(rec.ticker, rec.ticker)
                            )[1],
                            "display_ticker": aliases.identity(rec.ticker, pd.Timestamp(rec.as_of), "")[0],
                        }
                    )
        out["audits"]["date_arms"] = date_arms(client, prompts, items)
    if "test_retest" in which:
        res = {}
        retest = _retest_sample(sample, max(10, math.ceil(len(sample) / 5)))
        for agent in AGENTS:
            pairs = [
                (payload_for(r.as_of, r.ticker, agent), f"{pd.Timestamp(r.as_of).date()}|{r.ticker}")
                for r in retest.itertuples(index=False)
            ]
            pairs = [(p, k) for p, k in pairs if p is not None]
            ja = JuryAgent(agent, client, prompts, cache, settings.jury.model_digest, forbidden)
            res[agent] = test_retest(
                ja, [p for p, _ in pairs], seeds=[1, 2, 3, 4, 5], temperature=0.7, items=[k for _, k in pairs]
            )
        out["audits"]["test_retest"] = res
    (runs / "audits").mkdir(parents=True, exist_ok=True)
    if robustness:
        cfg = settings.audit
        months = sample_months(
            sorted(scores["as_of"].astype(str).unique()),
            cfg.robustness_months,
            cfg.robustness_seed,
            exclude_years=DEV_SLICE_YEARS,
        )
        rob = audit_robustness(
            scores,
            lambda m, agent: payload_list(m, agent),
            lambda agent, pr: JuryAgent(
                agent, client, pr, cache, settings.jury.model_digest, forbidden, settings.jury.max_validation_retries
            ),
            prompts,
            AgentPrompts.load(cfg.robustness_paraphrase_version),
            months,
            cfg.robustness_noise_z,
            cfg.robustness_top_k,
            cfg.robustness_seed,
            settings.jury.min_votes,
            settings.jury.theta,
        )
        rob = {"jury_run": jury_run, "generated_at": out["generated_at"], **rob}
        write_json(rob, runs / "audits" / f"{jury_run}_robustness.json")
        out["audits"]["robustness"] = rob["summary"]
    if not which:  # robustness only: leave the standard audit file untouched
        record_run(settings, "audit", jury_run, None, {"audits": sorted(out.get("audits", {}))})
        return out
    path = runs / "audits" / f"{jury_run}.json"
    write_json(out, path)
    record_run(settings, "audit", jury_run, None, {"audits": sorted(out.get("audits", {}))})
    log.info(
        "audits written to %s: %s",
        path,
        json.dumps(
            {k: (v if not isinstance(v, dict) else list(v.keys())) for k, v in out["audits"].items()}, default=str
        )[:400],
    )
    return out
