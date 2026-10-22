"""Thin, pinned Ollama client used by the jury agents and by `bistmas doctor`.

Design rules (see docs/preregistration.md and the red-team list in the plan):
* every call carries explicit sampling options (num_ctx, num_predict, temperature, seed);
* the model is pinned by digest, not only by tag — a re-pulled tag changes the digest;
* structured output is requested with a JSON schema through Ollama's `format` parameter;
* token counts are returned so callers can assert prompt tokens < num_ctx - headroom.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from bistmas.shared.log import get_logger

log = get_logger("jury.ollama")


@dataclass
class ChatResult:
    content: str
    parsed: dict[str, Any] | None
    prompt_eval_count: int | None
    eval_count: int | None
    duration_ms: float | None
    model: str
    options: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    transport_error: bool = False  # timeout / connection / HTTP failure: never a verdict, never cached

    @property
    def ok(self) -> bool:
        return self.error is None and self.parsed is not None


class OllamaClient:
    def __init__(
        self,
        host: str,
        model: str,
        options: dict[str, Any] | None = None,
        timeout: float = 600,
        keep_alive: str = "30m",
    ) -> None:
        import ollama  # local import keeps the dependency optional for data-layer tests

        self.host = host.rstrip("/")
        self.model = model
        self.options = dict(options or {})
        self.keep_alive = keep_alive
        self.timeout = timeout
        self._client = ollama.Client(host=self.host, timeout=timeout)

    # ------------------------------------------------------------------ inventory
    def ping(self) -> bool:
        try:
            self._client.list()
            return True
        except Exception as exc:  # noqa: BLE001 - any failure means "not reachable"
            log.warning("Ollama not reachable at %s: %s", self.host, exc)
            return False

    def list_models(self) -> list[dict[str, Any]]:
        resp = self._client.list()
        models = getattr(resp, "models", None) or resp.get("models", [])
        out = []
        for m in models:
            name = getattr(m, "model", None) or m.get("model") or m.get("name")
            digest = getattr(m, "digest", None) or m.get("digest")
            size = getattr(m, "size", None) or m.get("size")
            out.append({"model": name, "digest": digest, "size": size})
        return out

    def model_digest(self) -> str | None:
        wanted = {self.model, f"{self.model}:latest"} if ":" not in self.model else {self.model}
        for m in self.list_models():
            if m["model"] in wanted:
                return m["digest"]
        return None

    def show(self) -> dict[str, Any]:
        resp = self._client.show(self.model)
        details = getattr(resp, "details", None) or resp.get("details", {})
        info = getattr(resp, "modelinfo", None) or resp.get("modelinfo", {})
        return {
            "details": dict(details) if details else {},
            "modelinfo": dict(info) if info else {},
            "parameters": getattr(resp, "parameters", None) or resp.get("parameters"),
        }

    # ------------------------------------------------------------------ inference
    def chat_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        options_override: dict[str, Any] | None = None,
    ) -> ChatResult:
        opts = {**self.options, **(options_override or {})}
        think = bool(opts.pop("think", False))  # not an Ollama sampling option; passed as a top-level flag
        t0 = time.perf_counter()
        try:
            resp = self._client.chat(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                format=schema,
                options=opts,
                stream=False,
                think=think,
                keep_alive=self.keep_alive,
            )
        except Exception as exc:  # noqa: BLE001
            return ChatResult(
                "", None, None, None, None, self.model, opts, error=f"request failed: {exc}", transport_error=True
            )
        duration_ms = (time.perf_counter() - t0) * 1000
        message = getattr(resp, "message", None) or resp.get("message", {})
        content = getattr(message, "content", None) or message.get("content", "") or ""
        parsed: dict[str, Any] | None
        error: str | None = None
        try:
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                parsed, error = None, "model returned non-object JSON"
        except json.JSONDecodeError as exc:
            parsed, error = None, f"invalid JSON: {exc}"
        return ChatResult(
            content=content,
            parsed=parsed,
            prompt_eval_count=_get(resp, "prompt_eval_count"),
            eval_count=_get(resp, "eval_count"),
            duration_ms=duration_ms,
            model=self.model,
            options={**opts, "think": think},
            error=error,
        )

    def supports_logprobs(self) -> bool:
        """Probe whether this Ollama build returns token log-probabilities.

        The Lookahead-Propensity audit needs first-token probabilities; it is only run when this
        probe succeeds, and the run certificate records the outcome either way.
        """
        try:
            r = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "Reply with the single word: up"}],
                    "stream": False,
                    "logprobs": True,
                    "top_logprobs": 3,
                    "options": {"num_predict": 3, "temperature": 0},
                    "keep_alive": self.keep_alive,
                },
                timeout=self.timeout,
            )
            if r.status_code != 200:
                return False
            body = r.json()
            return bool(body.get("logprobs") or body.get("message", {}).get("logprobs"))
        except Exception:  # noqa: BLE001
            return False


def _get(resp: Any, key: str) -> Any:
    value = getattr(resp, key, None)
    if value is None and hasattr(resp, "get"):
        value = resp.get(key)
    return value
