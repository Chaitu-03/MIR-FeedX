"""
mir/api/routers/health.py
Health, liveness, readiness, stats — Prompt 12 + 15.
"""
from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from mir.api.auth import require_api_key
from mir.api.deps import get_db, get_qdrant
from mir.api.schemas import DependencyHealth, HealthResponse, StatsResponse
from mir.metrics import metrics_response
from mir.search.cache import get_redis

router = APIRouter(prefix="/api/v1", tags=["health"])


async def _check_db(db: AsyncSession) -> DependencyHealth:
    t0 = time.perf_counter()
    try:
        await db.execute(text("SELECT 1"))
        return DependencyHealth(status="connected", latency_ms=(time.perf_counter() - t0) * 1000)
    except Exception as exc:
        return DependencyHealth(status="error", error=str(exc))


async def _check_qdrant(qdrant) -> DependencyHealth:
    t0 = time.perf_counter()
    try:
        await asyncio.to_thread(qdrant._client.get_collections)
        return DependencyHealth(status="connected", latency_ms=(time.perf_counter() - t0) * 1000)
    except Exception as exc:
        return DependencyHealth(status="error", error=str(exc))


async def _check_redis() -> DependencyHealth:
    t0 = time.perf_counter()
    try:
        await get_redis().ping()
        return DependencyHealth(status="connected", latency_ms=(time.perf_counter() - t0) * 1000)
    except Exception as exc:
        return DependencyHealth(status="error", error=str(exc))


@router.get("/health", response_model=HealthResponse, summary="Full dependency health")
async def health(request: Request, db: AsyncSession = Depends(get_db)) -> HealthResponse:
    qdrant = request.app.state.qdrant if hasattr(request.app.state, "qdrant") else None
    db_h, qd_h, rd_h = await asyncio.gather(
        _check_db(db),
        _check_qdrant(qdrant) if qdrant is not None else _stub_ok(),
        _check_redis(),
    )
    overall = "ok" if all(h.status == "connected" for h in (db_h, qd_h, rd_h)) else "degraded"
    return HealthResponse(status=overall, db=db_h, qdrant=qd_h, redis=rd_h)


async def _stub_ok() -> DependencyHealth:
    return DependencyHealth(status="connected", latency_ms=0.0)


@router.get("/health/live", summary="Liveness probe (always 200)")
async def liveness():
    return {"status": "alive"}


@router.get("/health/ready", summary="Readiness probe — 503 if any dep is down")
async def readiness(request: Request, db: AsyncSession = Depends(get_db)):
    qdrant = getattr(request.app.state, "qdrant", None)
    db_h = await _check_db(db)
    qd_h = await (_check_qdrant(qdrant) if qdrant is not None else _stub_ok())
    rd_h = await _check_redis()
    if all(h.status == "connected" for h in (db_h, qd_h, rd_h)):
        return {"status": "ready"}
    return Response(
        content='{"status":"not_ready"}',
        media_type="application/json",
        status_code=503,
    )


@router.get("/metrics", include_in_schema=False)
async def metrics():
    return metrics_response()


@router.get("/stats", response_model=StatsResponse, summary="Cluster stats")
async def stats(
    db: AsyncSession = Depends(get_db),
    qdrant=Depends(get_qdrant),
    _api_key=Depends(require_api_key),
) -> StatsResponse:
    posts_count = (await db.execute(text("SELECT COUNT(*) FROM posts WHERE nsfw = false"))).scalar() or 0
    acc_count = (await db.execute(text("SELECT COUNT(*) FROM accounts"))).scalar() or 0
    tag_count = (await db.execute(text("SELECT COUNT(*) FROM tags"))).scalar() or 0
    last_crawl = (await db.execute(
        text("SELECT MAX(last_crawled_at) FROM crawl_state")
    )).scalar()

    qd_posts = None
    try:
        info = await asyncio.to_thread(qdrant.get_collection_info, "posts")
        qd_posts = info.get("points_count")
    except Exception:
        pass

    return StatsResponse(
        total_posts=int(posts_count),
        total_accounts=int(acc_count),
        total_tags=int(tag_count),
        qdrant_posts_count=qd_posts,
        queue_depth=None,
        last_crawl=last_crawl,
    )
