"""Polite HTTP session shared by every data client.

Rules (docs/data_governance.md): one identifiable User-Agent, per-host minimum spacing between
requests, bounded retries with exponential backoff and jitter, short timeouts (the KAP WAF drops
connections silently), and cache-first reads from the immutable raw layer so nothing is ever
fetched twice.
"""

from __future__ import annotations

import random
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from bistmas.shared.log import get_logger
from bistmas.shared.store import append_manifest, read_raw_gzip, utc_now_iso, write_raw_gzip

log = get_logger("data.http")

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124 Safari/537.36 bistmas-research/0.1 (+vedattatli6@gmail.com)"
)


class RateLimitedError(RuntimeError):
    """Raised when the server answers 429 and retries are exhausted."""


class PoliteSession:
    """requests.Session wrapper with per-host spacing, retries and raw-layer caching."""

    def __init__(
        self,
        min_interval_s: float = 1.0,
        timeout_s: float = 30.0,
        max_retries: int = 4,
        backoff_base_s: float = 2.0,
        headers: dict[str, str] | None = None,
        retry_on_429: bool = True,
    ) -> None:
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.retry_on_429 = retry_on_429
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8"})
        if headers:
            self._session.headers.update(headers)
        self._last_call: dict[str, float] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ pacing
    def _wait_for_slot(self, host: str) -> None:
        with self._lock:
            last = self._last_call.get(host, 0.0)
            delay = self.min_interval_s - (time.monotonic() - last)
            if delay > 0:
                time.sleep(delay)
            self._last_call[host] = time.monotonic()

    # ------------------------------------------------------------------ requests
    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        host = urlsplit(url).netloc
        kwargs.setdefault("timeout", self.timeout_s)
        attempt = 0
        while True:
            self._wait_for_slot(host)
            try:
                resp = self._session.request(method, url, **kwargs)
            except (requests.ConnectionError, requests.Timeout) as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise
                sleep = self.backoff_base_s * (2 ** (attempt - 1)) + random.uniform(0, 1)
                log.warning(
                    "%s %s failed (%s); retry %d/%d in %.1fs",
                    method,
                    url,
                    exc.__class__.__name__,
                    attempt,
                    self.max_retries,
                    sleep,
                )
                time.sleep(sleep)
                continue
            if resp.status_code == 429:
                attempt += 1
                if not self.retry_on_429 or attempt > self.max_retries:
                    raise RateLimitedError(f"429 from {host} for {url}")
                sleep = self.backoff_base_s * (2 ** (attempt - 1)) * 5 + random.uniform(0, 5)
                log.warning("429 from %s; sleeping %.0fs (attempt %d/%d)", host, sleep, attempt, self.max_retries)
                time.sleep(sleep)
                continue
            if 500 <= resp.status_code < 600 and attempt < self.max_retries:
                attempt += 1
                sleep = self.backoff_base_s * (2 ** (attempt - 1)) + random.uniform(0, 1)
                log.warning(
                    "%d from %s; retry %d/%d in %.1fs", resp.status_code, host, attempt, self.max_retries, sleep
                )
                time.sleep(sleep)
                continue
            return resp

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> requests.Response:
        return self.request("POST", url, **kwargs)

    # ------------------------------------------------------------------ raw cache
    def get_cached(
        self,
        url: str,
        raw_path: Path,
        manifest_path: Path | None = None,
        source: str = "",
        force: bool = False,
        expect_status: int = 200,
        **kwargs: Any,
    ) -> bytes:
        """Return the body for url, reading from raw_path (gzip) if it exists, else fetching and storing it."""
        if raw_path.exists() and not force:
            return read_raw_gzip(raw_path)
        resp = self.get(url, **kwargs)
        if resp.status_code != expect_status:
            raise requests.HTTPError(f"{resp.status_code} for {url}", response=resp)
        payload = resp.content
        digest = write_raw_gzip(payload, raw_path)
        append_manifest(
            manifest_path or raw_path.parent / "manifest.jsonl",
            {
                "source": source or urlsplit(url).netloc,
                "url": url,
                "path": str(raw_path),
                "bytes": len(payload),
                "sha256": digest,
                "status": resp.status_code,
                "fetched_at": utc_now_iso(),
            },
        )
        return payload
