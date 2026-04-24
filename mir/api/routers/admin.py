"""
mir/api/routers/admin.py
Admin endpoints — reindex, cache clear, community rebuild. Prompt 12.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from mir.api.auth import require_admin_key
from mir.api.schemas import CacheClearResponse, ReindexRequest, TaskIdResponse
from mir.search.cache import cache_clear

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


@router.post(
    "/reindex",
    response_model=TaskIdResponse,
    summary="Enqueue a full reindex Celery task",
)
async def reindex(body: ReindexRequest, _key=Depends(require_admin_key)) -> TaskIdResponse:
    from mir.workers.tasks import reindex_all

    async_result = reindex_all.delay(body.model)
    return TaskIdResponse(task_id=async_result.id)


@router.post(
    "/cache/clear",
    response_model=CacheClearResponse,
    summary="Flush all cached search responses (mir:cache:*)",
)
async def clear_cache(_key=Depends(require_admin_key)) -> CacheClearResponse:
    n = await cache_clear()
    return CacheClearResponse(cleared=n)


@router.post(
    "/rebuild_communities",
    response_model=TaskIdResponse,
    summary="Enqueue the nightly community rebuild",
)
async def admin_rebuild_communities(_key=Depends(require_admin_key)) -> TaskIdResponse:
    from mir.workers.tasks import rebuild_communities

    async_result = rebuild_communities.delay()
    return TaskIdResponse(task_id=async_result.id)
