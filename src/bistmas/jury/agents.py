"""One jury agent = one persona prompt + one information partition + the pinned local model.

``judge`` renders the masked payload, checks it for leaks, looks the call up in the content-addressed
cache, calls Ollama with the JSON schema, validates the verdict (schema + id echo + cited metrics), retries
ONCE with the validation error appended, and otherwise records an abstention. Every record stores the
prompt hash, schema hash, model digest, sampling options and token counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bistmas.jury.cache import JuryCache, cache_key, prompt_hash
from bistmas.jury.ollama_client import OllamaClient
from bistmas.jury.payload import CandidatePayload, allowed_metric_names, assert_clean
from bistmas.jury.schema import json_schema, validate_verdict
from bistmas.shared.config import REPO_ROOT
from bistmas.shared.hashing import sha256_json
from bistmas.shared.log import get_logger

log = get_logger("jury.agents")

AGENTS = ("quality_growth", "value", "risk_timing")
PROMPTS_DIR = REPO_ROOT / "prompts"


class TransportError(RuntimeError):
    """Ollama could not be reached / timed out / returned an HTTP error. Never cached, never an abstention."""


def registry_hashes(registry: Path, version: str) -> dict[str, str]:
    """``{file name: sha256}`` of ``version`` as recorded in prompts/REGISTRY.md."""
    out: dict[str, str] = {}
    for line in registry.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0] == version and len(cells[2]) == 64:
            out[Path(cells[1]).name] = cells[2]
    return out


@dataclass
class AgentPrompts:
    version: str
    system: dict[str, str]
    user_template: str
    hashes: dict[str, str]

    @classmethod
    def load(cls, version: str = "v1", root: Path | None = None) -> AgentPrompts:
        base = (root or PROMPTS_DIR) / version
        system = {a: (base / f"system_{a}.md").read_text(encoding="utf-8") for a in AGENTS}
        user = (base / "user_template.md").read_text(encoding="utf-8")
        hashes = {a: prompt_hash(system[a]) for a in AGENTS}
        hashes["user_template"] = prompt_hash(user)
        registry = (root or PROMPTS_DIR) / "REGISTRY.md"
        if registry.exists():
            recorded = registry_hashes(registry, version)
            actual = {f"system_{a}.md": hashes[a] for a in AGENTS} | {"user_template.md": hashes["user_template"]}
            for name, h in actual.items():
                if recorded.get(name) != h:
                    raise ValueError(
                        f"prompt {version}/{name}: sha256 {h} != registry {recorded.get(name)} (prompts/REGISTRY.md)"
                    )
        return cls(version, system, user, hashes)


@dataclass
class JudgeRecord:
    as_of: str
    ticker: str
    candidate_id: str
    agent: str
    score: float | None
    confidence: float | None
    abstain: bool
    abstain_reason: str
    rationale: str
    key_strengths: list[str]
    key_risks: list[str]
    cited_metrics: list[str]
    cache_key: str
    cache_hit: bool
    prompt_hash: str
    schema_hash: str
    model: str
    model_digest: str
    options: dict[str, Any]
    prompt_eval_count: int | None
    eval_count: int | None
    duration_ms: float | None
    attempts: int
    raw: str


class JuryAgent:
    def __init__(
        self,
        agent: str,
        client: OllamaClient,
        prompts: AgentPrompts,
        cache: JuryCache,
        model_digest: str,
        forbidden_tokens: set[str],
        max_validation_retries: int = 1,
        dry_run: bool = False,
    ) -> None:
        if agent not in AGENTS:
            raise ValueError(f"unknown agent {agent}")
        self.agent = agent
        self.client = client
        self.prompts = prompts
        self.cache = cache
        self.model_digest = model_digest
        self.forbidden = forbidden_tokens
        self.max_retries = max_validation_retries
        self.dry_run = dry_run
        self.schema = json_schema()
        self.schema_hash = sha256_json(self.schema)

    def judge(self, as_of: str, payload: CandidatePayload) -> JudgeRecord:
        system = self.prompts.system[self.agent]
        user = self.prompts.user_template.format(payload=payload.render(), candidate_id=payload.candidate_id)
        assert_clean(system + "\n" + user, self.forbidden)
        options = dict(self.client.options)
        key = cache_key(user, self.prompts.hashes[self.agent], self.schema_hash, self.model_digest, options)
        cached = self.cache.get(key)
        if cached is not None:
            rec = JudgeRecord(**{**cached["record"], "cache_hit": True})
            rec.as_of, rec.ticker = as_of, payload.ticker
            return rec
        base = {
            "as_of": as_of,
            "ticker": payload.ticker,
            "candidate_id": payload.candidate_id,
            "agent": self.agent,
            "cache_key": key,
            "cache_hit": False,
            "prompt_hash": self.prompts.hashes[self.agent],
            "schema_hash": self.schema_hash,
            "model": self.client.model,
            "model_digest": self.model_digest,
            "options": options,
        }
        if self.dry_run:
            return JudgeRecord(
                **base,
                score=None,
                confidence=None,
                abstain=True,
                abstain_reason="dry_run",
                rationale="",
                key_strengths=[],
                key_risks=[],
                cited_metrics=[],
                prompt_eval_count=None,
                eval_count=None,
                duration_ms=None,
                attempts=0,
                raw="",
            )
        allowed = allowed_metric_names(payload)
        error: str | None = None
        attempts = 0
        last = None
        while attempts <= self.max_retries:
            attempts += 1
            user_msg = (
                user
                if error is None
                else f"{user}\n\nYour previous answer was rejected: {error}. Return a corrected JSON object."
            )
            res = self.client.chat_json(system, user_msg, self.schema)
            if getattr(res, "transport_error", False):
                raise TransportError(res.error or "transport error")  # not cached, not an abstention
            last = res
            if not res.ok or res.parsed is None:
                error = res.error or "empty response"
                continue
            verdict, error = validate_verdict(res.parsed, payload.candidate_id, allowed)
            if verdict is not None:
                rec = JudgeRecord(
                    **base,
                    score=float(verdict.score),
                    confidence=float(verdict.confidence),
                    abstain=False,
                    abstain_reason="",
                    rationale=verdict.rationale,
                    key_strengths=verdict.key_strengths,
                    key_risks=verdict.key_risks,
                    cited_metrics=verdict.cited_metrics,
                    prompt_eval_count=res.prompt_eval_count,
                    eval_count=res.eval_count,
                    duration_ms=res.duration_ms,
                    attempts=attempts,
                    raw=res.content,
                )
                self.cache.put(key, {"record": rec.__dict__, "payload": payload.fields})
                return rec
        rec = JudgeRecord(
            **base,
            score=None,
            confidence=None,
            abstain=True,
            abstain_reason=(error or "unknown")[:300],
            rationale="",
            key_strengths=[],
            key_risks=[],
            cited_metrics=[],
            prompt_eval_count=last.prompt_eval_count if last else None,
            eval_count=last.eval_count if last else None,
            duration_ms=last.duration_ms if last else None,
            attempts=attempts,
            raw=last.content if last else "",
        )
        self.cache.put(key, {"record": rec.__dict__, "payload": payload.fields})
        return rec
