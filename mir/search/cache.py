"""
mir/search/cache.py
Redis caching layer for search resolvers — Prompt 13.

Cache key = prefix + SHA256(query_type | query_text | sorted(filters) | nsfw_opt_in)
Value = JSON-serialised response.
TTL = settings.cache_ttl_seconds (default 300s).

Failure mode: Redis outage must NOT break search. All exceptions from the cache
fall through with a WARNING log and the resolver is invoked normally.
"""
from __future__ import annotations

import functools
import hashlib
import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from typing import Any, Awaitable, Callable

import redis.asyncio as aioredis

from mir.config import settings

log = logging.getLogger(__name__)

# ── Shared async Redis pool ──────────────────────────────────────────────────

_redis_client: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    """Return a process-wide async Redis client with connection pooling."""
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=32,
        )
    return _redis_client


async def close_redis() -> None:
    global _redis_client
    if _redis_client is not None:
        try:
            await _redis_client.aclose()
        except Exception:
            pass
        _redis_client = None


# ── Key hashing ───────────────────────────────────────────────────────────────

def _canonical(obj: Any) -> Any:
    """Recursively convert obj to a JSON-canonical form (sorted keys)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _canonical(obj[k]) for k in sorted(obj)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(x) for x in obj]
    if is_dataclass(obj):
        return _canonical(asdict(obj))
    return str(obj)


def make_cache_key(
    query_type: str,
    query_text: str,
    filters: dict[str, Any] | None = None,
    *,
    nsfw_opt_in: bool = False,
    extra: dict[str, Any] | None = None,
) -> str:
    """Compute SHA256 cache key. Any param that affects results MUST be in `extra`."""
    payload = {
        "type": query_type,
        "q": query_text,
        "filters": _canonical(filters or {}),
        "nsfw": bool(nsfw_opt_in),
        "extra": _canonical(extra or {}),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{settings.cache_key_prefix}{query_type}:{digest}"


# ── JSON (de)serialisation for dataclass responses ────────────────────────────

def _default_encoder(o: Any) -> Any:
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    raise TypeError(f"Not JSON serialisable: {type(o).__name__}")


def dumps(value: Any) -> str:
    return json.dumps(value, default=_default_encoder)


def loads(value: str) -> Any:
    return json.loads(value)


# ── Core get/set (swallow Redis errors) ───────────────────────────────────────

async def cache_get(key: str) -> Any | None:
    try:
        client = get_redis()
        raw = await client.get(key)
        if raw is None:
            return None
        return loads(raw)
    except Exception as exc:
        log.warning("cache_get failed (%s) — falling through", exc)
        return None


async def cache_set(key: str, value: Any, ttl: int | None = None) -> bool:
    ttl = ttl if ttl is not None else settings.cache_ttl_seconds
    try:
        client = get_redis()
        await client.set(key, dumps(value), ex=ttl)
        return True
    except Exception as exc:
        log.warning("cache_set failed (%s) — continuing without cache", exc)
        return False


async def cache_clear(prefix: str | None = None) -> int:
    """Delete every key under the cache prefix. Returns number deleted."""
    pattern = (prefix or settings.cache_key_prefix) + "*"
    try:
        client = get_redis()
        cursor = 0
        deleted = 0
        while True:
            cursor, keys = await client.scan(cursor=cursor, match=pattern, count=500)
            if keys:
                deleted += await client.delete(*keys)
            if cursor == 0:
                break
        return deleted
    except Exception as exc:
        log.warning("cache_clear failed (%s)", exc)
        return 0


# ── Decorator ────────────────────────────────────────────────────────────────

def cached(
    query_type: str,
    *,
    key_builder: Callable[..., str] | None = None,
    ttl: int | None = None,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """
    Decorator for async search resolvers.

    Default key build uses the first positional `query_text` arg and any
    `filters` / `nsfw_opt_in` kwargs. Pass a custom `key_builder(*args, **kwargs)`
    for more control.
    """

    def decorator(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if key_builder is not None:
                key = key_builder(*args, **kwargs)
            else:
                # heuristic: first positional arg is the query string
                q_text = args[0] if args else kwargs.get("query", "")
                filters = kwargs.get("filters")
                if hasattr(filters, "__dict__"):
                    filters = _canonical(filters)
                nsfw_opt_in = bool(kwargs.get("nsfw_opt_in", False))
                key = make_cache_key(
                    query_type,
                    str(q_text),
                    filters if isinstance(filters, dict) else None,
                    nsfw_opt_in=nsfw_opt_in,
                )

            hit = await cache_get(key)
            if hit is not None:
                log.info("cache hit: %s", key)
                return hit

            result = await fn(*args, **kwargs)
            await cache_set(key, result, ttl=ttl)
            log.info("cache miss → stored: %s", key)
            return result

        return wrapper

    return decorator
