"""Jury verdict schema. Property ORDER matters: reasoning fields come before the score so that a
constrained decoder produces the rationale before committing to a number."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

MAX_LIST = 3


class Verdict(BaseModel):
    candidate_id: str = Field(description="echo of the candidate id exactly as given")
    key_strengths: list[str] = Field(default_factory=list, max_length=MAX_LIST)
    key_risks: list[str] = Field(default_factory=list, max_length=MAX_LIST)
    rationale: str = Field(min_length=1, max_length=1200)
    cited_metrics: list[str] = Field(default_factory=list, max_length=8)
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("key_strengths", "key_risks", "cited_metrics", mode="before")
    @classmethod
    def _clean(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        return [str(x).strip() for x in v if str(x).strip()][:8]


def json_schema() -> dict[str, Any]:
    """JSON schema handed to Ollama's ``format`` parameter (keys in reasoning-first order)."""
    return {
        "type": "object",
        "properties": {
            "candidate_id": {"type": "string"},
            "key_strengths": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_LIST},
            "key_risks": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_LIST},
            "rationale": {"type": "string"},
            "cited_metrics": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
            "score": {"type": "number", "minimum": 0, "maximum": 1},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["candidate_id", "key_strengths", "key_risks", "rationale", "cited_metrics", "score", "confidence"],
    }


def validate_verdict(
    parsed: dict[str, Any], candidate_id: str, allowed_metrics: set[str]
) -> tuple[Verdict | None, str | None]:
    """Parse + semantic checks: id echo, cited metrics subset of the payload fields."""
    try:
        v = Verdict.model_validate(parsed)
    except Exception as exc:  # noqa: BLE001 - pydantic errors are reported as text to the retry prompt
        return None, f"schema validation failed: {exc}"
    if v.candidate_id.strip().upper() != candidate_id.upper():
        return None, f"candidate_id must echo {candidate_id!r}, got {v.candidate_id!r}"
    bad = [m for m in v.cited_metrics if m not in allowed_metrics]
    if bad:
        return None, f"cited_metrics contains fields not in the payload: {bad}"
    return v, None
