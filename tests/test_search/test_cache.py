"""
tests/test_search/test_cache.py
Unit tests for mir/search/cache.py — Prompt 13.

No real Redis is required: we monkeypatch get_redis() with a fake async client.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from mir.search import cache as cache_mod
from mir.search.cache import cache_get, cache_set, cached, make_cache_key


class _FakeRedis:
    def __init__(self, *, fail: bool = False):
        self._store: dict[str, str] = {}
        self._fail = fail

    async def get(self, key):
        if self._fail:
            raise RuntimeError("redis down")
        return self._store.get(key)

    async def set(self, key, value, ex=None):
        if self._fail:
            raise RuntimeError("redis down")
        self._store[key] = value

    async def delete(self, *keys):
        if self._fail:
            raise RuntimeError("redis down")
        n = 0
        for k in keys:
            if k in self._store:
                del self._store[k]
                n += 1
        return n

    async def scan(self, cursor=0, match=None, count=100):
        keys = [k for k in self._store if match is None or k.startswith(match.rstrip("*"))]
        return 0, keys

    async def aclose(self):
        pass

    async def ping(self):
        if self._fail:
            raise RuntimeError("redis down")
        return True


@pytest.fixture(autouse=True)
def _reset_redis(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(cache_mod, "_redis_client", fake)
    monkeypatch.setattr(cache_mod, "get_redis", lambda: fake)
    yield fake


# ── Key hashing ──────────────────────────────────────────────────────────────

class TestCacheKey:
    def test_different_queries_different_keys(self):
        k1 = make_cache_key("general", "photography")
        k2 = make_cache_key("general", "cooking")
        assert k1 != k2

    def test_same_query_different_nsfw_flag_different_keys(self):
        k1 = make_cache_key("general", "q", nsfw_opt_in=False)
        k2 = make_cache_key("general", "q", nsfw_opt_in=True)
        assert k1 != k2, "nsfw_opt_in must affect the cache key"

    def test_filter_order_independence(self):
        k1 = make_cache_key("general", "q", filters={"lang": "en", "min_notes": 5})
        k2 = make_cache_key("general", "q", filters={"min_notes": 5, "lang": "en"})
        assert k1 == k2

    def test_filter_value_affects_key(self):
        k1 = make_cache_key("general", "q", filters={"lang": "en"})
        k2 = make_cache_key("general", "q", filters={"lang": "de"})
        assert k1 != k2

    def test_query_type_separates_keys(self):
        assert make_cache_key("general", "x") != make_cache_key("tags", "x")

    def test_prefix_present(self):
        k = make_cache_key("general", "q")
        assert k.startswith("mir:cache:")


# ── get/set round trip ──────────────────────────────────────────────────────

class TestCacheIO:
    @pytest.mark.asyncio
    async def test_miss_then_set_then_hit(self):
        key = make_cache_key("tags", "hello")
        assert await cache_get(key) is None
        await cache_set(key, {"a": 1})
        assert await cache_get(key) == {"a": 1}

    @pytest.mark.asyncio
    async def test_redis_failure_falls_through(self, monkeypatch):
        failing = _FakeRedis(fail=True)
        monkeypatch.setattr(cache_mod, "get_redis", lambda: failing)

        # Must NOT raise
        assert await cache_get("mir:cache:broken") is None
        assert await cache_set("mir:cache:broken", {"x": 1}) is False


# ── decorator ────────────────────────────────────────────────────────────────

class TestCachedDecorator:
    @pytest.mark.asyncio
    async def test_decorator_caches_result(self):
        calls = {"n": 0}

        @cached("general")
        async def resolver(query: str):
            calls["n"] += 1
            return {"query": query, "n": calls["n"]}

        r1 = await resolver("q1")
        r2 = await resolver("q1")
        assert r1 == r2
        assert calls["n"] == 1, "second call must hit cache"

    @pytest.mark.asyncio
    async def test_decorator_distinguishes_queries(self):
        calls = {"n": 0}

        @cached("general")
        async def resolver(query: str):
            calls["n"] += 1
            return {"query": query}

        await resolver("q1")
        await resolver("q2")
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_decorator_distinguishes_nsfw(self):
        calls = {"n": 0}

        @cached("general")
        async def resolver(query: str, *, nsfw_opt_in: bool = False):
            calls["n"] += 1
            return {"query": query, "nsfw": nsfw_opt_in}

        await resolver("q1", nsfw_opt_in=False)
        await resolver("q1", nsfw_opt_in=True)
        assert calls["n"] == 2, "nsfw_opt_in must bust cache key"
