"""Jury run over every decision date of a screens table (Top-N candidates per month), cached and resumable."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from bistmas.jury.agents import AGENTS, AgentPrompts, JuryAgent, TransportError
from bistmas.jury.cache import JuryCache
from bistmas.jury.consensus import consensus
from bistmas.jury.ollama_client import OllamaClient
from bistmas.jury.payload import build_payloads
from bistmas.shared.config import Settings
from bistmas.shared.ledger import record_run
from bistmas.shared.log import get_logger
from bistmas.shared.store import read_parquet, utc_now_iso, write_json, write_parquet

log = get_logger("jury.run")

SCORES_FILE = "jury_scores.parquet"
CONSENSUS_FILE = "jury_consensus.parquet"


GENERIC_TITLE_WORDS = {
    # legal-form and sector words that appear in many company titles (never identifying on their own)
    "SANAYİ",
    "SANAYİİ",
    "TİCARET",
    "TİCARETİ",
    "ANONİM",
    "ŞİRKETİ",
    "ŞİRKETLERİ",
    "HOLDİNG",
    "HOLDING",
    "A.Ş.",
    "A.O.",
    "ENERJİ",
    "YATIRIM",
    "YATIRIMLAR",
    "YATIRIMLARI",
    "BANKASI",
    "SİGORTA",
    "ORTAKLIĞI",
    "GAYRİMENKUL",
    "MENKUL",
    "DEĞERLER",
    "KIYMETLER",
    "PORTFÖY",
    "FAKTORİNG",
    "KİRALAMA",
    "FİNANSAL",
    "FİNANSMAN",
    "VARLIK",
    "ÜRETİM",
    "TEKNOLOJİ",
    "TEKNOLOJİLERİ",
    "HİZMETLERİ",
    "HİZMET",
    "TURİZM",
    "ELEKTRİK",
    "İNŞAAT",
    "ÜRÜNLERİ",
    "TEKSTİL",
    "ÇELİK",
    "DEMİR",
    "AMBALAJ",
    "ÇİMENTO",
    "KİMYA",
    "PAZARLAMA",
    "YÖNETİM",
    "OTOMOTİV",
    "METAL",
    "TÜRKİYE",
    "TÜRK",
    "GİRİŞİM",
    "SERMAYESİ",
    "FİNANS",
    "TAAHHÜT",
    "GELİŞTİRME",
    "İMALAT",
    "ENDÜSTRİ",
    "MADENCİLİK",
    "PETROL",
    "PLASTİK",
    "KAĞIT",
    "GIDA",
    "BİLİŞİM",
    "LOJİSTİK",
    "ULAŞTIRMA",
    "MOTORLU",
    "ARAÇLAR",
    "HAVAYOLLARI",
    "MEDYA",
    "SPORTİF",
    "FAALİYETLER",
    "FAALİYETLERİ",
    "KATILIM",
    "İŞLETMELERİ",
    "İŞLETMECİLİĞİ",
    "SİSTEMLERİ",
    # plain English words used in the prompt templates and payload field names
    "GROUP",
    "TREND",
    "GLOBAL",
    "ENERGY",
    "CAPITAL",
    "VALUE",
    "QUALITY",
    "GROWTH",
    "RISK",
    "TIMING",
    "MEDIA",
    "TECHNOLOGY",
    "INTERNATIONAL",
    "INDUSTRY",
    "INDUSTRIES",
    "INVESTMENT",
    "TRADE",
    "FINANCE",
    "FINANCIAL",
    "REAL",
    "ESTATE",
    "SERVICE",
    "SERVICES",
    "SYSTEMS",
    "GATE",
    "EURO",
    "TURKISH",
    "TURKEY",
}


def forbidden_tokens_from(master: pd.DataFrame, max_title_count: int = 2) -> set[str]:
    """Tickers and distinctive company-name words that must never appear in a prompt.

    A title word counts as distinctive when it is at least five characters long, is not a generic legal-form,
    sector or prompt-vocabulary word (``GENERIC_TITLE_WORDS``) and occurs in at most ``max_title_count``
    different company titles — words shared by many issuers (SANAYİ, GAYRİMENKUL, ANADOLU, …) identify nobody
    and would only collide with ordinary prompt text.
    """
    toks: set[str] = set(master["ticker"].astype(str).str.upper())
    counts: dict[str, int] = {}
    for title in master["title"].astype(str).unique():
        words = {w.strip().upper() for w in title.replace(".", " ").replace(",", " ").split()}
        for w in words:
            counts[w] = counts.get(w, 0) + 1
    for w, n in counts.items():
        if len(w) >= 5 and w not in GENERIC_TITLE_WORDS and n <= max_title_count:
            toks.add(w)
    return toks


def _own_tag(agent_models: dict[str, str] | None, agent: str, settings: Settings) -> bool:
    tag = (agent_models or {}).get(agent)
    return bool(tag) and tag != settings.jury.model


def run_jury(
    settings: Settings,
    run_name: str,
    screens: pd.DataFrame,
    master: pd.DataFrame,
    decisions: list[pd.Timestamp] | None = None,
    top_n: int | None = None,
    dry_run: bool = False,
    max_calls: int | None = None,
    client: OllamaClient | None = None,
    agent_models: dict[str, str] | None = None,
    clients: dict[str, OllamaClient] | None = None,
) -> dict[str, Any]:
    """Score the Top-N candidates of every month with the three agents; write scores + consensus.

    ``agent_models`` (agent -> Ollama tag) runs a cross-family jury: each agent gets its own model whose
    live digest is recorded (the pinned digest applies only to the default model). ``clients`` injects
    per-agent clients (tests).
    """
    prompts = AgentPrompts.load(settings.jury.prompt_version)
    client = client or OllamaClient(settings.jury.host, settings.jury.model, settings.jury.sampling_options())
    pinned = settings.jury.model_digest
    uses_default = any(not (clients and a in clients) and not _own_tag(agent_models, a, settings) for a in AGENTS)
    digest = pinned or "dry-run"
    if uses_default:
        if dry_run:
            digest = pinned or "dry-run"
        else:
            # always verify the live weights behind the default client (also in a cross-family run) and key
            # the cache on the LIVE digest; a missing tag or a re-pulled model stops the run before any call
            live = client.model_digest()
            if live is None:
                raise RuntimeError(f"model {client.model} not found on the Ollama host (no digest)")
            if pinned and live != pinned:
                raise RuntimeError(f"model digest {live} != pinned {pinned}")
            digest = live
    cache = JuryCache(settings.paths.runs / "jury_cache")
    forbidden = forbidden_tokens_from(master)
    agent_clients: dict[str, OllamaClient] = {}
    agent_digests: dict[str, str] = {}
    for a in AGENTS:
        tag = (agent_models or {}).get(a)
        if clients and a in clients:
            agent_clients[a] = clients[a]
        elif tag and tag != settings.jury.model:
            agent_clients[a] = OllamaClient(settings.jury.host, tag, settings.jury.sampling_options())
        else:
            agent_clients[a] = client
        if agent_clients[a] is client:
            agent_digests[a] = digest
        else:
            live_digest = getattr(agent_clients[a], "model_digest", lambda: None)
            d = live_digest() if not dry_run else None
            if not dry_run and d is None and not (clients and a in clients):
                raise RuntimeError(f"model {tag} not found on the Ollama host (no digest)")
            agent_digests[a] = "dry-run" if dry_run else (d or "unknown")
    agents = {
        a: JuryAgent(
            a,
            agent_clients[a],
            prompts,
            cache,
            agent_digests[a],
            forbidden,
            settings.jury.max_validation_retries,
            dry_run,
        )
        for a in AGENTS
    }
    n = top_n or settings.scoring.top_n_screen
    out_dir = settings.paths.runs / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / SCORES_FILE
    existing = read_parquet(scores_path) if scores_path.exists() else pd.DataFrame()
    if len(existing) and "abstain_reason" in existing.columns:
        # dry-run placeholders must never count as finished work
        existing = existing[existing["abstain_reason"] != "dry_run"]
    done = (
        set(zip(existing["as_of"].astype(str), existing["ticker"], existing["agent"], strict=True))
        if len(existing)
        else set()
    )
    records: list[dict[str, Any]] = existing.to_dict("records") if len(existing) else []
    calls = 0
    stopped = "complete"
    months = sorted(pd.to_datetime(screens["as_of"]).unique())
    if decisions:
        wanted = {pd.Timestamp(d) for d in decisions}
        months = [m for m in months if pd.Timestamp(m) in wanted]
    for m in months:
        s = screens[pd.to_datetime(screens["as_of"]) == m].copy()
        s["rank"] = pd.to_numeric(s["rank"], errors="coerce")  # never trust the stored dtype: 10 must sort after 2
        cands = s.dropna(subset=["rank"]).sort_values("rank").head(n)["ticker"].tolist()
        crank = dict(zip(s["ticker"], s["rank"], strict=True))
        for a in AGENTS:
            payloads = build_payloads(s, pd.Timestamp(m), cands, a, salt=run_name)
            for p in payloads:
                key = (str(pd.Timestamp(m).date()), p.ticker, a)
                if key in done:
                    continue
                if max_calls is not None and calls >= max_calls:
                    stopped = "max_calls"
                    break
                try:
                    rec = agents[a].judge(str(pd.Timestamp(m).date()), p)
                except TransportError as exc:
                    log.error("jury stopped on a transport error (nothing cached, resumable): %s", exc)
                    stopped = "transport_error"
                    break
                if not rec.cache_hit:
                    calls += 1
                d = rec.__dict__.copy()
                d["composite_rank"] = crank.get(p.ticker)
                d["options"] = json.dumps(d["options"])
                for lst in ("key_strengths", "key_risks", "cited_metrics"):
                    d[lst] = json.dumps(d[lst], ensure_ascii=False)
                records.append(d)
                done.add(key)
            if stopped != "complete":
                break
        if stopped != "complete":
            break
        df = pd.DataFrame(records)
        write_parquet(df, scores_path)
        log.info("jury %s: %s done (%d calls this run)", run_name, pd.Timestamp(m).date(), calls)
    df = pd.DataFrame(records)
    if len(df):
        write_parquet(df, scores_path)
        cons = consensus(
            df.assign(score=pd.to_numeric(df["score"], errors="coerce")),
            min_votes=settings.jury.min_votes,
            theta=settings.jury.theta,
        )
        write_parquet(cons, out_dir / CONSENSUS_FILE)
    else:
        cons = pd.DataFrame()
    summary = {
        "run": run_name,
        "generated_at": utc_now_iso(),
        "months": len(months),
        "records": int(len(df)),
        "calls_this_run": calls,
        "abstain_rate": float(df["abstain"].mean()) if len(df) else None,
        "avg_prompt_tokens": float(pd.to_numeric(df["prompt_eval_count"], errors="coerce").mean()) if len(df) else None,
        "avg_output_tokens": float(pd.to_numeric(df["eval_count"], errors="coerce").mean()) if len(df) else None,
        "stopped_reason": stopped,
        "prompt_version": prompts.version,
        "prompt_hashes": prompts.hashes,
        "model_digest": digest,
        "agent_models": {a: getattr(agent_clients[a], "model", settings.jury.model) for a in AGENTS},
        "agent_digests": agent_digests,
        "sampling_options": settings.jury.sampling_options(),
        "consensus_rows": int(len(cons)),
        "no_J_share": float(cons["J"].isna().mean()) if len(cons) else None,
    }
    write_json(summary, out_dir / "jury_summary.json")
    if not dry_run:
        record_run(settings, "jury", run_name, None, {k: v for k, v in summary.items() if k != "prompt_hashes"})
    log.info("jury summary: %s", {k: v for k, v in summary.items() if k != "prompt_hashes"})
    return summary


def load_consensus(runs_dir: Path, run_name: str) -> pd.DataFrame:
    return read_parquet(Path(runs_dir) / run_name / CONSENSUS_FILE)
