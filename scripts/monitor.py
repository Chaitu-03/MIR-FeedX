#!/usr/bin/env python3
"""
scripts/monitor.py
Lightweight monitoring script for MIR-FeedX.

Replaces Prometheus-based monitoring with a standalone script that gathers
the same data points from PostgreSQL, Redis, Qdrant, and the Celery broker.

Previously tracked by Prometheus → now pulled by this script:
  - mir_posts_indexed_total       → DB: COUNT(posts) WHERE nsfw = false
  - mir_nsfw_flagged_total        → DB: COUNT(posts) WHERE nsfw = true
  - mir_crawl_posts_fetched_total → DB: crawl_state rows + statuses
  - mir_search_duration_seconds   → timed GET to /api/v1/search/* endpoints
  - mir_cache_hits_total          → Redis INFO stats → keyspace_hits
  - mir_cache_misses_total        → Redis INFO stats → keyspace_misses
  - mir_queue_depth               → Redis LLEN on Celery queues

Usage:
    # Via the running API (fast, no DB deps needed locally)
    python scripts/monitor.py

    # Direct checks — hits DB, Redis, Qdrant directly (API not required)
    python scripts/monitor.py --direct

    # Continuous watch mode
    python scripts/monitor.py --watch --interval 30

    # Custom API URL
    python scripts/monitor.py --base-url http://192.168.1.10:8000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone


# ── ANSI colours ─────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


def _ok(msg: str) -> str:
    return f"{GREEN}✓{RESET} {msg}"


def _fail(msg: str) -> str:
    return f"{RED}✗{RESET} {msg}"


def _warn(msg: str) -> str:
    return f"{YELLOW}⚠{RESET} {msg}"


def _header(title: str) -> str:
    return f"\n{BOLD}{CYAN}── {title} ──{RESET}"


def _val(label: str, value, width: int = 20) -> str:
    return f"  {label:<{width}}: {value}"


# ── HTTP helpers ─────────────────────────────────────────────────────────────

def _get_json(url: str, timeout: float = 5.0) -> dict | None:
    """GET a JSON endpoint. Returns parsed dict or None on failure."""
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _timed_get_json(url: str, timeout: float = 10.0) -> tuple[dict | None, float]:
    """GET endpoint and return (data, elapsed_ms)."""
    t0 = time.perf_counter()
    data = _get_json(url, timeout)
    elapsed = (time.perf_counter() - t0) * 1000
    return data, elapsed


# ═══════════════════════════════════════════════════════════════════════════════
# API-based monitoring (hits the running FastAPI server)
# ═══════════════════════════════════════════════════════════════════════════════

def check_via_api(base_url: str) -> bool:
    all_ok = True

    # ── 1. Dependency health ─────────────────────────────────────────────
    print(_header("Dependency Health"))
    health = _get_json(f"{base_url}/api/v1/health")
    if health is None:
        print(_fail(f"API unreachable at {base_url}"))
        return False

    for dep in ("db", "qdrant", "redis"):
        info = health.get(dep, {})
        status = info.get("status", "unknown")
        latency = info.get("latency_ms")
        if status == "connected":
            lat_str = f" ({latency:.1f} ms)" if latency is not None else ""
            print(_ok(f"{dep:8s} connected{lat_str}"))
        else:
            err = info.get("error", "")
            print(_fail(f"{dep:8s} {status} — {err}"))
            all_ok = False

    overall = health.get("status", "unknown")
    color = GREEN if overall == "ok" else YELLOW
    print(f"  {color}Overall: {overall}{RESET}")

    # ── 2. Liveness + readiness ──────────────────────────────────────────
    live = _get_json(f"{base_url}/api/v1/health/live")
    if live and live.get("status") == "alive":
        print(_ok("Liveness  : alive"))
    else:
        print(_fail("Liveness  : not responding"))
        all_ok = False

    ready = _get_json(f"{base_url}/api/v1/health/ready")
    if ready and ready.get("status") == "ready":
        print(_ok("Readiness : ready"))
    else:
        print(_warn("Readiness : not ready"))

    # ── 3. Index stats (replaces posts_indexed_total + nsfw_flagged_total) ─
    print(_header("Index Stats"))
    stats = _get_json(f"{base_url}/api/v1/stats")
    if stats:
        print(_val("Posts indexed (SFW)", stats.get("total_posts", "?")))
        print(_val("NSFW flagged", stats.get("nsfw_flagged_count", "?")))
        print(_val("Accounts", stats.get("total_accounts", "?")))
        print(_val("Tags", stats.get("total_tags", "?")))
        qd = stats.get("qdrant_posts_count")
        print(_val("Qdrant vectors", qd if qd is not None else "N/A"))
        qd_depth = stats.get("queue_depth")
        print(_val("Queue depth", qd_depth if qd_depth is not None else "N/A"))
        crawl = stats.get("crawl_status")
        if crawl:
            summary = ", ".join(f"{k}={v}" for k, v in crawl.items())
            print(_val("Crawl status", summary))
        lc = stats.get("last_crawl")
        print(_val("Last crawl", lc or "never"))
    else:
        print(_warn("Stats endpoint unavailable (may require API key)"))

    # ── 4. Search latency probe (replaces search_duration_seconds) ───────
    print(_header("Search Latency Probe"))
    endpoints = {
        "accounts": f"{base_url}/api/v1/search/accounts?q=test&limit=1",
        "tags":     f"{base_url}/api/v1/search/tags?q=test&limit=1",
    }
    for name, url in endpoints.items():
        data, ms = _timed_get_json(url)
        if data is not None:
            print(_ok(f"{name:12s} {ms:7.1f} ms"))
        else:
            print(_warn(f"{name:12s} unavailable (auth required or server down)"))

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Direct monitoring (bypasses the API, connects to services directly)
# ═══════════════════════════════════════════════════════════════════════════════

def check_direct() -> bool:
    all_ok = True

    # ── Load project config ──────────────────────────────────────────────
    try:
        from mir.config import settings
    except Exception as exc:
        print(_fail(f"Cannot load mir.config: {exc}"))
        print("  Make sure you're running from the project root with deps installed.")
        return False

    # ── 1. PostgreSQL ────────────────────────────────────────────────────
    print(_header("PostgreSQL"))
    try:
        import asyncio
        import asyncpg

        async def _pg_checks():
            dsn = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
            t0 = time.perf_counter()
            conn = await asyncpg.connect(dsn, timeout=5)
            await conn.fetchval("SELECT 1")
            latency = (time.perf_counter() - t0) * 1000

            # posts_indexed_total (SFW)
            sfw = await conn.fetchval("SELECT COUNT(*) FROM posts WHERE nsfw = false")
            # nsfw_flagged_total
            nsfw = await conn.fetchval("SELECT COUNT(*) FROM posts WHERE nsfw = true")
            # accounts + tags
            accs = await conn.fetchval("SELECT COUNT(*) FROM accounts")
            tags = await conn.fetchval("SELECT COUNT(*) FROM tags")
            # crawl_state summary (replaces crawl_posts_fetched_total)
            crawl_rows = await conn.fetch(
                "SELECT status, COUNT(*) AS cnt FROM crawl_state GROUP BY status"
            )
            last_crawl = await conn.fetchval(
                "SELECT MAX(last_crawled_at) FROM crawl_state"
            )
            await conn.close()
            return latency, sfw, nsfw, accs, tags, crawl_rows, last_crawl

        lat, sfw, nsfw, accs, tags, crawl_rows, last_crawl = asyncio.run(_pg_checks())
        print(_ok(f"Connected ({lat:.1f} ms)"))
        print(_val("Posts indexed (SFW)", sfw))
        print(_val("Posts flagged NSFW", nsfw))
        print(_val("Accounts", accs))
        print(_val("Tags", tags))
        print(_val("Last crawl", last_crawl or "never"))
        if crawl_rows:
            crawl_summary = ", ".join(f"{r['status']}={r['cnt']}" for r in crawl_rows)
            print(_val("Crawl state", crawl_summary))
    except Exception as exc:
        print(_fail(f"PostgreSQL error: {exc}"))
        all_ok = False

    # ── 2. Redis — cache stats + queue depth ─────────────────────────────
    print(_header("Redis"))
    try:
        import redis as _redis

        t0 = time.perf_counter()
        r = _redis.Redis.from_url(settings.redis_url, socket_connect_timeout=3)
        r.ping()
        latency = (time.perf_counter() - t0) * 1000

        # Memory
        mem_info = r.info("memory")
        mem_mb = mem_info.get("used_memory", 0) / (1024 * 1024)

        # Cache hit/miss stats (replaces cache_hits_total + cache_misses_total)
        stats_info = r.info("stats")
        hits = stats_info.get("keyspace_hits", 0)
        misses = stats_info.get("keyspace_misses", 0)
        total_ops = hits + misses
        hit_rate = (hits / total_ops * 100) if total_ops > 0 else 0.0

        # Cache key count
        cache_prefix = settings.cache_key_prefix
        cache_keys = sum(1 for _ in r.scan_iter(match=f"{cache_prefix}*", count=500))

        # Queue depth (replaces mir_queue_depth gauge)
        queues = {"default": 0, "gpu": 0, "dead_letter": 0}
        for q_name in queues:
            queues[q_name] = r.llen(q_name) or 0

        print(_ok(f"Connected ({latency:.1f} ms, {mem_mb:.1f} MB)"))
        print(_val("Keyspace hits", f"{hits:,}"))
        print(_val("Keyspace misses", f"{misses:,}"))
        print(_val("Cache hit rate", f"{hit_rate:.1f}%"))
        print(_val("Cached responses", cache_keys))
        print(_val("Queue: default", queues["default"]))
        print(_val("Queue: gpu", queues["gpu"]))
        print(_val("Queue: dead_letter", queues["dead_letter"]))
    except Exception as exc:
        print(_fail(f"Redis error: {exc}"))
        all_ok = False

    # ── 3. Qdrant ────────────────────────────────────────────────────────
    print(_header("Qdrant"))
    try:
        from qdrant_client import QdrantClient

        t0 = time.perf_counter()
        qc = QdrantClient(
            host=settings.qdrant_host, port=settings.qdrant_port, timeout=5
        )
        collections = qc.get_collections().collections
        latency = (time.perf_counter() - t0) * 1000
        names = [c.name for c in collections]
        print(_ok(f"Connected ({latency:.1f} ms, collections: {names})"))

        for col in collections:
            info = qc.get_collection(col.name)
            print(
                _val(
                    f"  {col.name}",
                    f"{info.points_count} vectors, status={info.status.name}",
                )
            )
    except Exception as exc:
        print(_fail(f"Qdrant error: {exc}"))
        all_ok = False

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Entrypoint
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MIR-FeedX monitoring — replaces Prometheus with a lightweight script.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the running FastAPI server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Check services directly (DB/Redis/Qdrant) instead of via the API",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Run continuously at the given interval",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Seconds between checks in --watch mode (default: 60)",
    )
    args = parser.parse_args()

    check_fn = check_direct if args.direct else (lambda: check_via_api(args.base_url))

    try:
        while True:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            print(f"\n{BOLD}MIR-FeedX Monitor — {ts}{RESET}")
            ok = check_fn()
            status_msg = (
                f"{GREEN}All systems operational.{RESET}"
                if ok
                else f"{RED}Some checks failed.{RESET}"
            )
            print(f"\n{status_msg}")

            if not args.watch:
                sys.exit(0 if ok else 1)

            print(f"\n{DIM}Next check in {args.interval}s … (Ctrl+C to stop){RESET}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
