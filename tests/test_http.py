from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from pmcal.http import CachedClient, HTTPStatusError, OfflineCacheMiss, canonical_url


@pytest.fixture(autouse=True)
def _online(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests exercise the network path against a mock transport, so they must
    not inherit PMCAL_OFFLINE from the environment (CI sets it for everything else)."""
    monkeypatch.delenv("PMCAL_OFFLINE", raising=False)


def _transport(responses: list[httpx.Response], calls: list[str]) -> httpx.MockTransport:
    it = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return next(it)

    return httpx.MockTransport(handler)


def test_canonical_url_sorts_and_drops_none() -> None:
    u = canonical_url("https://x.test/api/", "/m", {"b": 2, "a": "z", "c": None, "d": True})
    assert u == "https://x.test/api/m?a=z&b=2&d=true"


def test_cache_hit_avoids_network(tmp_path: Path) -> None:
    calls: list[str] = []
    t = _transport([httpx.Response(200, json={"v": 1})], calls)
    with CachedClient("https://x.test", tmp_path, rate_per_s=1000, transport=t) as c:
        assert c.get_json("/a", {"q": 1}) == {"v": 1}
        assert c.get_json("/a", {"q": 1}) == {"v": 1}
        assert c.stats["hits"] == 1
    assert len(calls) == 1


def test_retries_429_then_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pmcal.http.time.sleep", lambda s: None)
    calls: list[str] = []
    t = _transport(
        [
            httpx.Response(429, headers={"Retry-After": "1"}),
            httpx.Response(503),
            httpx.Response(200, json=[1, 2]),
        ],
        calls,
    )
    with CachedClient("https://x.test", tmp_path, rate_per_s=1000, transport=t) as c:
        assert c.get_json("/a") == [1, 2]
        assert c.stats["throttled"] == 1
        assert c.limiter.rate < c.limiter.base_rate
    assert len(calls) == 3


def test_404_is_cached_other_errors_raise(tmp_path: Path) -> None:
    calls: list[str] = []
    t = _transport([httpx.Response(404, json={"error": "nope"}), httpx.Response(400)], calls)
    with CachedClient("https://x.test", tmp_path, rate_per_s=1000, transport=t) as c:
        assert c.get_json("/missing") is None
        assert c.get_json("/missing") is None
        with pytest.raises(HTTPStatusError):
            c.get_json("/bad")
    assert len(calls) == 2


def test_offline_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    t = _transport([httpx.Response(200, json={"ok": True})], calls)
    with CachedClient("https://x.test", tmp_path, rate_per_s=1000, transport=t) as c:
        c.get_json("/a")
        monkeypatch.setenv("PMCAL_OFFLINE", "1")
        assert c.get_json("/a") == {"ok": True}
        with pytest.raises(OfflineCacheMiss):
            c.get_json("/b")
