"""Model-facing leakage audits (each needs the local model; sample sizes are parameters).

1. Re-identification probe — feed the production (masked) payload and ask which BIST company / period it
   is. Accuracy is compared with chance (1 / universe size; 1 / number of quarters).
2. Direct elicitation — reveal ticker + company and ask for the month-end close and the next-month
   direction; recall accuracy vs the true price panel measures how much BIST history the model memorised.
3. Date arms — the production prompt plus (a) the true as-of date, (b) the ticker + name, (c) a transplanted
   wrong date (±3 years), (d) a shifted date (−7 years); score changes vs the masked arm reveal date- or
   identity-keyed recall (HindsightBench / fake-date designs).
4. Test-retest — the same masked payload scored k times at a fixed temperature: exact-match rate and ICC.
5. Lookahead propensity (LAP) — first-token probabilities of "up" / "down" vs "unknown" for a dated,
   identity-revealed question (needs token log-probs).
"""

from __future__ import annotations

import json
import math
from collections.abc import Collection
from typing import Any

import numpy as np
import pandas as pd
import requests

from bistmas.jury.agents import AgentPrompts, JuryAgent
from bistmas.jury.ollama_client import OllamaClient
from bistmas.jury.payload import CandidatePayload

REID_SCHEMA = {
    "type": "object",
    "properties": {
        "company_guess": {"type": "string"},
        "ticker_guess": {"type": "string"},
        "year_guess": {"type": "integer"},
        "quarter_guess": {"type": "integer"},
        "confidence": {"type": "number"},
    },
    "required": ["company_guess", "ticker_guess", "year_guess", "quarter_guess", "confidence"],
}
REID_SYSTEM = (
    "You are an expert on Borsa İstanbul. You will see an anonymised metric profile of a BIST-100 company "
    "(percentile ranks and z-scores). Guess which company it is (ticker) and which fiscal year and quarter "
    "the statement belongs to. Answer with the JSON object only."
)
ELICIT_SCHEMA = {
    "type": "object",
    "properties": {
        "close_tl": {"type": "number"},
        "next_month_direction": {"type": "string", "enum": ["up", "down", "unknown"]},
        "confidence": {"type": "number"},
    },
    "required": ["close_tl", "next_month_direction", "confidence"],
}
ELICIT_SYSTEM = (
    "You are a Borsa İstanbul market historian. Answer from memory with the JSON object only; "
    "if you do not know, use 'unknown' and a low confidence."
)


def reidentification_probe(
    client: OllamaClient,
    payloads: list[tuple[CandidatePayload, str | Collection[str], int, int]],
    universe_size: int,
    n_quarters: int = 28,
) -> dict[str, Any]:
    """``payloads``: (payload, true_ticker, true_year, true_quarter); ``true_ticker`` may be a collection of
    accepted codes (the code valid at the decision date and its successor). Returns accuracies vs chance;
    the quarter chance is ``1 / n_quarters`` (number of distinct quarters the sample spans)."""
    rows = []
    for p, truth, year, quarter in payloads:
        accepted = {truth.upper()} if isinstance(truth, str) else {str(t).upper() for t in truth}
        ticker = truth if isinstance(truth, str) else sorted(accepted)[0]
        res = client.chat_json(REID_SYSTEM, p.render(), REID_SCHEMA)
        g = res.parsed or {}
        rows.append(
            {
                "ticker": ticker,
                "guess": str(g.get("ticker_guess", "")).upper(),
                "year_ok": int(g.get("year_guess", -1) or -1) == year,
                "quarter_ok": int(g.get("quarter_guess", -1) or -1) == quarter,
                "ticker_ok": str(g.get("ticker_guess", "")).upper().strip() in accepted,
                "confidence": g.get("confidence"),
                "ok": res.ok,
            }
        )
    df = pd.DataFrame(rows)
    n = int(len(df))
    return {
        "n": n,
        "ticker_top1": float(df["ticker_ok"].mean()) if n else None,
        "ticker_chance": 1.0 / max(universe_size, 1),
        "year_accuracy": float(df["year_ok"].mean()) if n else None,
        "year_and_quarter_accuracy": float((df["year_ok"] & df["quarter_ok"]).mean()) if n else None,
        "quarter_chance": 1.0 / max(n_quarters, 1),
        "valid_responses": float(df["ok"].mean()) if n else None,
        "guesses": df["guess"].value_counts().head(10).to_dict() if n else {},
    }


def direct_elicitation(client: OllamaClient, cases: list[dict[str, Any]]) -> dict[str, Any]:
    """``cases``: {ticker, company, date (YYYY-MM-DD), true_close, true_next_direction}."""
    rows = []
    for c in cases:
        user = (
            f"What was the closing price in Turkish lira of {c['company']} ({c['ticker']}) on Borsa İstanbul "
            f"on {c['date']}, and did the stock go up or down over the following month?"
        )
        res = client.chat_json(ELICIT_SYSTEM, user, ELICIT_SCHEMA)
        g = res.parsed or {}
        close = g.get("close_tl")
        ape = abs(float(close) / c["true_close"] - 1) if isinstance(close, int | float) and c["true_close"] else np.nan
        rows.append(
            {
                "ticker": c["ticker"],
                "date": c["date"],
                "ape": ape,
                "direction_ok": g.get("next_month_direction") == c["true_next_direction"],
                "answered": g.get("next_month_direction") in ("up", "down"),
                "ok": res.ok,
            }
        )
    df = pd.DataFrame(rows)
    n = int(len(df))
    return {
        "n": n,
        "mape": float(df["ape"].median()) if n else None,
        "share_within_10pct": float((df["ape"] < 0.10).mean()) if n else None,
        "direction_accuracy_when_answered": float(df.loc[df["answered"], "direction_ok"].mean())
        if n and df["answered"].any()
        else None,
        "answered_share": float(df["answered"].mean()) if n else None,
    }


ARMS = ("masked", "date_only", "revealed", "transplanted_plus3y", "shifted_minus7y")


def date_arm_context(arm: str, as_of: pd.Timestamp, ticker: str, company: str) -> str:
    if arm == "masked":
        return ""
    if arm == "date_only":
        return f"\nContext: the evaluation date is {as_of.date()}."
    if arm == "revealed":
        return f"\nContext: the company is {company} ({ticker}) and the evaluation date is {as_of.date()}."
    if arm == "transplanted_plus3y":
        return f"\nContext: the evaluation date is {(as_of + pd.DateOffset(years=3)).date()}."
    if arm == "shifted_minus7y":
        return f"\nContext: the evaluation date is {(as_of - pd.DateOffset(years=7)).date()}."
    raise ValueError(arm)


def date_arms(
    client: OllamaClient, prompts: AgentPrompts, sample: list[dict[str, Any]], arms: tuple[str, ...] = ARMS
) -> dict[str, Any]:
    """``sample``: {payload (CandidatePayload), agent, as_of, ticker, company}. Scores per arm and deltas vs masked."""
    from bistmas.jury.schema import json_schema

    schema = json_schema()
    rows = []
    for item in sample:
        p: CandidatePayload = item["payload"]
        agent = item["agent"]
        system = prompts.system[agent]
        base_user = prompts.user_template.format(payload=p.render(), candidate_id=p.candidate_id)
        for arm in arms:
            user = base_user + date_arm_context(
                arm, pd.Timestamp(item["as_of"]), item.get("display_ticker", item["ticker"]), item["company"]
            )
            res = client.chat_json(system, user, schema)
            score = (res.parsed or {}).get("score") if res.ok else None
            rows.append(
                {"as_of": str(item["as_of"]), "ticker": item["ticker"], "agent": agent, "arm": arm, "score": score}
            )
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["as_of", "ticker", "agent"], columns="arm", values="score", aggfunc="first")
    out: dict[str, Any] = {"n_items": int(len(piv)), "arms": {}}
    for arm in arms:
        if arm == "masked" or arm not in piv.columns:
            continue
        d = (piv[arm] - piv["masked"]).dropna()
        out["arms"][arm] = {
            "mean_delta": float(d.mean()) if len(d) else None,
            "mean_abs_delta": float(d.abs().mean()) if len(d) else None,
            "share_changed_gt_0_1": float((d.abs() > 0.1).mean()) if len(d) else None,
            "spearman_vs_masked": float(piv[[arm, "masked"]].dropna().corr(method="spearman").iloc[0, 1])
            if len(d) > 2
            else None,
        }
    return out


def test_retest(
    agent: JuryAgent,
    payloads: list[CandidatePayload],
    seeds: list[int],
    temperature: float = 0.7,
    items: list[str] | None = None,
) -> dict[str, Any]:
    """Score each payload once per seed at ``temperature``; exact-match rate (score within 0.005) and ICC(1).

    ``items`` (same length as ``payloads``) are the item keys, e.g. ``"2021-03-31|THYAO"`` (as_of, ticker); the
    default is the ticker. Items with a failed call are dropped so the groups are balanced (k scores each).
    """
    rows = []
    keys = items if items is not None else [p.ticker for p in payloads]
    for p, key in zip(payloads, keys, strict=True):
        for s in seeds:
            res = agent.client.chat_json(
                agent.prompts.system[agent.agent],
                agent.prompts.user_template.format(payload=p.render(), candidate_id=p.candidate_id),
                agent.schema,
                {"seed": s, "temperature": temperature},
            )
            rows.append({"item": key, "seed": s, "score": (res.parsed or {}).get("score") if res.ok else np.nan})
    df = pd.DataFrame(rows)
    n_before = int(df["item"].nunique()) if len(df) else 0
    complete = df.dropna().groupby("item")["seed"].transform("size") == len(seeds)
    df = df.dropna()[complete.reindex(df.dropna().index, fill_value=False)]
    if df.empty:
        return {"n": 0}
    groups = df.groupby("item")["score"]
    k = len(seeds)
    grand = df["score"].mean()
    ms_between = k * ((groups.mean() - grand) ** 2).sum() / max(len(groups) - 1, 1)
    ms_within = groups.var(ddof=1).fillna(0).mean()
    icc = (
        (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
        if (ms_between + (k - 1) * ms_within) > 0
        else np.nan
    )
    spread = groups.agg(lambda s: s.max() - s.min())
    return {
        "n_items": int(len(groups)),
        "n_items_dropped": n_before - int(len(groups)),
        "k": k,
        "temperature": temperature,
        "icc1": float(icc),
        "mean_range": float(spread.mean()),
        "share_exact": float((spread < 0.005).mean()),
    }


def lap_probe(
    host: str, model: str, cases: list[dict[str, Any]], keep_alive: str = "30m", timeout: float = 120
) -> dict[str, Any]:
    """Lookahead propensity: P(up)+P(down) from the first-token log-probs of a dated, revealed question."""
    rows = []
    for c in cases:
        q = (
            f"On {c['date']}, did the closing stock price of {c['company']} ({c['ticker']}) on Borsa İstanbul "
            "go up or down over the following month? Respond with exactly one word: up, down, or unknown."
        )
        body = {
            "model": model,
            "messages": [{"role": "user", "content": q}],
            "stream": False,
            "logprobs": True,
            "top_logprobs": 10,
            "options": {"num_predict": 2, "temperature": 0},
            "keep_alive": keep_alive,
            "think": False,
        }
        try:
            r = requests.post(f"{host}/api/chat", json=body, timeout=timeout)
            data = r.json()
        except Exception as exc:  # noqa: BLE001
            rows.append({"ticker": c["ticker"], "date": c["date"], "error": str(exc)[:80]})
            continue
        lp = data.get("logprobs") or data.get("message", {}).get("logprobs") or []
        first = lp[0] if lp else {}
        tops = first.get("top_logprobs") or first.get("topLogprobs") or []
        probs = {}
        for t in tops:
            tok = str(t.get("token", "")).strip().lower()
            if tok in ("up", "down", "unknown"):
                probs[tok] = probs.get(tok, 0.0) + math.exp(float(t.get("logprob", -99)))
        rows.append(
            {
                "ticker": c["ticker"],
                "date": c["date"],
                "period": c.get("period", ""),
                "p_up": probs.get("up", 0.0),
                "p_down": probs.get("down", 0.0),
                "p_unknown": probs.get("unknown", 0.0),
                "lap": probs.get("up", 0.0) + probs.get("down", 0.0),
            }
        )
    df = pd.DataFrame(rows)
    out: dict[str, Any] = {"n": int(len(df)), "errors": int(df["error"].notna().sum()) if "error" in df.columns else 0}
    if "lap" in df.columns and df["lap"].notna().any():
        out["lap_mean"] = float(df["lap"].mean())
        if "period" in df.columns:
            out["lap_by_period"] = {str(k): float(v) for k, v in df.groupby("period")["lap"].mean().items()}
    return out


def dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str, ensure_ascii=False)
