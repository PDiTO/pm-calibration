"""A polite, caching JSON client shared by both venues.

Every successful response is written to disk, gzipped, under a key derived from
the URL and its query parameters, so a second run of the pipeline makes no
network requests at all. With ``PMCAL_OFFLINE=1`` a cache miss raises instead of
fetching, which is how the tests and the "re-runnable from cache" guarantee are
checked.

Politeness: a token bucket caps the request rate per client (shared across
threads), 429 and 5xx responses are retried with exponential backoff and jitter,
``Retry-After`` is honoured, and every 429 halves the allowed rate for a while.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import logging
import os
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

log = logging.getLogger(__name__)

USER_AGENT = "pm-calibration research (public data only)"
Params = dict[str, str | int | float | bool | None]


class OfflineCacheMiss(RuntimeError):  # noqa: N818 - reads better at call sites
    """Raised in offline mode when a request is not in the cache."""


class HTTPStatusError(RuntimeError):
    def __init__(self, url: str, status: int, body: str) -> None:
        super().__init__(f"HTTP {status} for {url}: {body[:300]}")
        self.url = url
        self.status = status
        self.body = body


def offline_mode() -> bool:
    return os.environ.get("PMCAL_OFFLINE", "") not in ("", "0", "false")


def canonical_url(base_url: str, path: str, params: Params | None) -> str:
    """URL with parameters sorted and ``None`` values dropped, used as the cache key."""
    clean = {k: _fmt(v) for k, v in sorted((params or {}).items()) if v is not None}
    query = urlencode(clean)
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}" + (f"?{query}" if query else "")


def _fmt(v: str | int | float | bool) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


class RateLimiter:
    """Token bucket with adaptive slowdown after 429s. Thread-safe."""

    def __init__(self, rate_per_s: float, burst: int = 1) -> None:
        self.base_rate = rate_per_s
        self.rate = rate_per_s
        self.burst = max(1, burst)
        self._tokens = float(self.burst)
        self._last = time.monotonic()
        self._penalty_until = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                if now > self._penalty_until:
                    self.rate = self.base_rate
                self._tokens = min(self.burst, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self.rate
            time.sleep(wait)

    def penalise(self, seconds: float = 60.0) -> None:
        with self._lock:
            self.rate = max(self.base_rate / 8, self.rate / 2)
            self._penalty_until = time.monotonic() + seconds


@dataclass(frozen=True)
class CachedResponse:
    status: int
    data: Any


class CachedClient:
    """GET JSON with an on-disk cache.

    ``cache_statuses`` lists non-2xx statuses that are stable answers worth caching
    (for example 404 for a market that has no price history). Everything else that
    is not 2xx raises ``HTTPStatusError`` after retries and is not cached.
    """

    def __init__(
        self,
        base_url: str,
        cache_dir: Path,
        *,
        rate_per_s: float = 5.0,
        burst: int = 2,
        timeout: float = 60.0,
        max_retries: int = 8,
        cache_statuses: tuple[int, ...] = (404,),
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url
        self.cache_dir = Path(cache_dir)
        self.limiter = RateLimiter(rate_per_s, burst)
        self.max_retries = max_retries
        self.cache_statuses = cache_statuses
        self._client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            transport=transport,
            follow_redirects=True,
        )
        self.stats = {"hits": 0, "fetches": 0, "retries": 0, "throttled": 0}
        self._stats_lock = threading.Lock()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> CachedClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- cache ---------------------------------------------------------------

    def cache_path(self, url: str) -> Path:
        h = hashlib.sha256(url.encode()).hexdigest()
        return self.cache_dir / h[:2] / f"{h}.json.gz"

    def cached(self, path: str, params: Params | None = None) -> CachedResponse | None:
        """The cached response for a request, or None. Never touches the network."""
        url = canonical_url(self.base_url, path, params)
        return self._read(self.cache_path(url))

    @staticmethod
    def _read(p: Path) -> CachedResponse | None:
        if not p.exists():
            return None
        with gzip.open(p, "rt", encoding="utf-8") as f:
            rec = json.load(f)
        return CachedResponse(status=int(rec["status"]), data=rec["data"])

    @staticmethod
    def _write(p: Path, url: str, status: int, data: Any) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".tmp{threading.get_ident()}")
        rec = {"url": url, "status": status, "fetched_at": time.time(), "data": data}
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(rec, f, separators=(",", ":"))
        os.replace(tmp, p)

    # -- requests ------------------------------------------------------------

    def get(
        self, path: str, params: Params | None = None, *, refresh: bool = False
    ) -> CachedResponse:
        url = canonical_url(self.base_url, path, params)
        cp = self.cache_path(url)
        if not refresh:
            hit = self._read(cp)
            if hit is not None:
                self._bump("hits")
                return hit
        if offline_mode():
            raise OfflineCacheMiss(url)
        status, data = self._fetch(url)
        self._write(cp, url, status, data)
        return CachedResponse(status=status, data=data)

    def get_json(self, path: str, params: Params | None = None, *, refresh: bool = False) -> Any:
        """Like ``get`` but returns the body, or None for a cached non-2xx status."""
        r = self.get(path, params, refresh=refresh)
        return r.data if 200 <= r.status < 300 else None

    def _fetch(self, url: str) -> tuple[int, Any]:
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            self.limiter.acquire()
            try:
                resp = self._client.get(url)
            except (httpx.TransportError, httpx.TimeoutException) as e:
                if attempt == self.max_retries:
                    raise
                log.debug("transport error %s on %s, retrying", e, url)
                self._bump("retries")
                delay = self._sleep(delay, None)
                continue
            self._bump("fetches")
            if resp.status_code == 429:
                self._bump("throttled")
                self.limiter.penalise()
                if attempt == self.max_retries:
                    break
                delay = self._sleep(delay, resp.headers.get("Retry-After"))
                continue
            if resp.status_code >= 500:
                if attempt == self.max_retries:
                    break
                self._bump("retries")
                delay = self._sleep(delay, resp.headers.get("Retry-After"))
                continue
            if 200 <= resp.status_code < 300:
                return resp.status_code, resp.json()
            if resp.status_code in self.cache_statuses:
                return resp.status_code, _maybe_json(resp)
            raise HTTPStatusError(url, resp.status_code, resp.text)
        raise HTTPStatusError(url, resp.status_code, resp.text)

    @staticmethod
    def _sleep(delay: float, retry_after: str | None) -> float:
        wait = delay
        if retry_after:
            with contextlib.suppress(ValueError):
                wait = max(wait, float(retry_after))
        time.sleep(min(wait, 120.0) * (0.75 + 0.5 * random.random()))
        return min(delay * 2, 120.0)

    def _bump(self, key: str) -> None:
        with self._stats_lock:
            self.stats[key] += 1


def _maybe_json(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return resp.text
