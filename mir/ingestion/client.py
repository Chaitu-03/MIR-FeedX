from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import time
from typing import Any

import aiohttp
import redis.asyncio as aioredis

from mir.config import settings

log = logging.getLogger(__name__)

_API_BASE = "https://api.tumblr.com/v2"

# Token bucket: 1 000 req / hr / key  →  ≈ 0.2778 tok/s
_BUCKET_CAPACITY = 1000
_BUCKET_RATE = 1000 / 3600  # tokens per second (float)

# Lua script for atomic token-bucket check-and-consume.
# Returns 1 if a token was granted, 0 if the bucket is empty.
_TOKEN_BUCKET_LUA = """
local key      = KEYS[1]
local capacity = tonumber(ARGV[1])
local rate     = tonumber(ARGV[2])
local now      = tonumber(ARGV[3])

local data     = redis.call('HMGET', key, 'tokens', 'ts')
local tokens   = tonumber(data[1]) or capacity
local ts       = tonumber(data[2]) or now

local elapsed  = math.max(0, now - ts)
tokens         = math.min(capacity, tokens + elapsed * rate)

if tokens < 1 then
    redis.call('HMSET', key, 'tokens', tokens, 'ts', now)
    redis.call('EXPIRE', key, 7200)
    return 0
end

tokens = tokens - 1
redis.call('HMSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, 7200)
return 1
"""

_BACKOFF_BASE = 2.0
_BACKOFF_MAX = 64.0


class TumblrClient:
    """Async Tumblr API v2 client with key rotation and Redis-backed rate limiting."""

    def __init__(self, redis_client: aioredis.Redis | None = None) -> None:
        self._keys: list[str] = list(settings.tumblr_api_keys)
        self._key_idx = 0
        self._redis = redis_client
        self._session: aiohttp.ClientSession | None = None
        self._lua_sha: str | None = None

    async def __aenter__(self) -> "TumblrClient":
        self._session = aiohttp.ClientSession()
        if self._redis and self._keys:
            self._lua_sha = await self._redis.script_load(_TOKEN_BUCKET_LUA)
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._session:
            await self._session.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_key(self) -> str:
        if not self._keys:
            raise RuntimeError("No Tumblr API keys configured (TUMBLR_API_KEYS is empty)")
        key = self._keys[self._key_idx % len(self._keys)]
        self._key_idx += 1
        return key

    @staticmethod
    def _bucket_redis_key(api_key: str) -> str:
        h = hashlib.sha256(api_key.encode()).hexdigest()[:16]
        return f"mir:ratelimit:{h}"

    async def _acquire_token(self, api_key: str) -> None:
        """Block until the per-key token bucket grants a slot."""
        if not self._redis or not self._lua_sha:
            return
        bucket_key = self._bucket_redis_key(api_key)
        while True:
            allowed = await self._redis.evalsha(
                self._lua_sha,
                1,               # numkeys
                bucket_key,      # KEYS[1]
                str(_BUCKET_CAPACITY),
                str(_BUCKET_RATE),
                str(time.time()),
            )
            if allowed:
                return
            await asyncio.sleep(1)

    async def _request(self, path: str, params: dict[str, Any]) -> Any:
        """Execute a GET request with per-key rate limiting and 429 back-off."""
        assert self._session is not None, "Use TumblrClient as an async context manager"
        backoff = _BACKOFF_BASE

        while True:
            api_key = self._next_key()
            await self._acquire_token(api_key)
            params["api_key"] = api_key

            try:
                async with self._session.get(
                    f"{_API_BASE}{path}",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 429:
                        jitter = random.uniform(0, backoff * 0.5)
                        sleep_for = min(backoff + jitter, _BACKOFF_MAX)
                        log.warning("Rate-limited (429). Sleeping %.1fs", sleep_for)
                        await asyncio.sleep(sleep_for)
                        backoff = min(backoff * 2, _BACKOFF_MAX)
                        continue

                    resp.raise_for_status()
                    data = await resp.json()
                    return data.get("response", {})

            except aiohttp.ClientError as exc:
                log.warning("HTTP error on %s: %s", path, exc)
                raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_blog_info(self, blog_name: str) -> dict:
        return await self._request(f"/blog/{blog_name}/info", {})

    async def get_blog_posts(
        self,
        blog_name: str,
        before: int | None = None,
        limit: int = 20,
    ) -> list[dict]:
        params: dict[str, Any] = {"limit": limit}
        if before is not None:
            params["before"] = before
        result = await self._request(f"/blog/{blog_name}/posts", params)
        return result.get("posts", []) if isinstance(result, dict) else []

    async def get_tagged_posts(
        self,
        tag: str,
        before: int | None = None,
    ) -> list[dict]:
        params: dict[str, Any] = {"tag": tag}
        if before is not None:
            params["before"] = before
        result = await self._request("/tagged", params)
        return result if isinstance(result, list) else []
