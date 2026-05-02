from __future__ import annotations

import asyncio
import hashlib
import logging
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

class TumblrClient:
    """Async Tumblr API v2 client with key rotation and Redis-backed rate limiting.

    Key rotation:
      - Round-robin across all configured consumer keys.
      - On 429: current key marked with cooldown (default 5 min), next key tried.
      - On 401/403: current key marked dead until restart, next key tried.
      - If all keys are dead/cooling, the client raises after one full cycle.
    """

    _COOLDOWN_429_S = 300       # 5-minute cooldown after rate-limit
    _COOLDOWN_AUTH_S = 86_400   # effectively dead for the session

    def __init__(self, redis_client: aioredis.Redis | None = None) -> None:
        # Prefer credentials list; fall back to legacy single-key list.
        self._keys: list[str] = list(settings.all_consumer_keys)
        self._key_idx = 0
        self._cooldowns: dict[str, float] = {}  # key → epoch seconds when usable again
        self._redis = redis_client
        self._session: aiohttp.ClientSession | None = None
        self._lua_sha: str | None = None

    async def __aenter__(self) -> "TumblrClient":
        # Tumblr's edge silently drops connections that look like bots:
        #   - Default aiohttp UA ("Python/3.x aiohttp/3.x") → disconnect.
        #   - Custom UA suffix (e.g. "MIR-FeedX/0.1") → also disconnect.
        # Use a clean browser UA + explicit SSL context + force_close to dodge
        # connection reuse oddities.
        import ssl
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36"
            ),
            "Accept": "application/json",
        }
        ssl_ctx = ssl.create_default_context()
        connector = aiohttp.TCPConnector(ssl=ssl_ctx, force_close=True)
        self._session = aiohttp.ClientSession(headers=headers, connector=connector)
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
        """Return next non-cooldown key. Raises if all keys are unavailable."""
        if not self._keys:
            raise RuntimeError("No Tumblr API keys configured (TUMBLR_API_CREDENTIALS empty)")
        now = time.time()
        n = len(self._keys)
        for _ in range(n):
            key = self._keys[self._key_idx % n]
            self._key_idx += 1
            cooldown_until = self._cooldowns.get(key, 0)
            if cooldown_until <= now:
                return key
        # All keys on cooldown
        soonest = min(self._cooldowns.values()) if self._cooldowns else 0
        wait = max(0, soonest - now)
        raise RuntimeError(
            f"All {n} Tumblr API keys exhausted/cooling down. Soonest available in {wait:.0f}s."
        )

    def _mark_cooldown(self, key: str, seconds: float, reason: str) -> None:
        until = time.time() + seconds
        self._cooldowns[key] = until
        masked = key[:8] + "…" + key[-4:]
        log.warning("Key %s on cooldown for %.0fs (%s)", masked, seconds, reason)

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
        """Execute a GET request with per-key rate limiting and key rotation on failure."""
        assert self._session is not None, "Use TumblrClient as an async context manager"
        attempts = 0
        max_attempts = max(len(self._keys) * 2, 4)

        while True:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(f"Exhausted {max_attempts} attempts on {path}")

            api_key = self._next_key()  # may raise if all keys cooling
            await self._acquire_token(api_key)
            params["api_key"] = api_key

            try:
                async with self._session.get(
                    f"{_API_BASE}{path}",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 429:
                        # Rate-limited: cool down THIS key, immediately try next.
                        self._mark_cooldown(api_key, self._COOLDOWN_429_S, "429 rate-limit")
                        continue

                    if resp.status in (401, 403):
                        # Auth failure: kill key for the session, try next.
                        self._mark_cooldown(api_key, self._COOLDOWN_AUTH_S, f"HTTP {resp.status}")
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
