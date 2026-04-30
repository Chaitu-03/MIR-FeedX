"""
mir/api/routers/search.py
Search endpoints — Prompt 12.
"""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from mir.api.auth import require_api_key
from mir.api.deps import get_db, get_qdrant, get_text_processor
from mir.api.schemas import (
    AccountNameResult,
    AccountNameSearchResponse,
    AccountResult,
    CommunityResult,
    CommunitySearchResponse,
    GeneralSearchRequest,
    GeneralSearchResponse,
    PostResult,
    TagResult,
    TagSearchResponse,
)
from mir.search.account_name import search_accounts_by_name
from mir.search.cache import cache_get, cache_set, make_cache_key
from mir.search.communities import search_communities
from mir.search.general import (
    GeneralSearchFilters as CoreFilters,
    GeneralSearchResolver,
)
from mir.search.tags import search_tags as core_search_tags

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/search", tags=["search"])


# ── General ───────────────────────────────────────────────────────────────────

@router.post(
    "/general",
    response_model=GeneralSearchResponse,
    summary="Fused keyword + vector post search",
)
async def general_search(
    body: GeneralSearchRequest,
    db: AsyncSession = Depends(get_db),
    qdrant=Depends(get_qdrant),
    text_processor=Depends(get_text_processor),
    _api_key=Depends(require_api_key),
) -> GeneralSearchResponse:
    key = make_cache_key(
        "general",
        body.query,
        filters=body.filters.model_dump(mode="json"),
        extra={
            "limit_posts": body.limit_posts,
            "limit_accounts": body.limit_accounts,
            "cursor": body.cursor,
        },
    )
    hit = await cache_get(key)
    if hit is not None:
        log.debug("cache_hit query_type=general")
        return GeneralSearchResponse.model_validate(hit)
    log.debug("cache_miss query_type=general")

    t0 = time.perf_counter()
    try:
        core_filters = CoreFilters(
            date_from=body.filters.date_from,
            date_to=body.filters.date_to,
            min_notes=body.filters.min_notes,
            tags=body.filters.tags,
            lang=body.filters.lang,
        )
        resolver = GeneralSearchResolver(db=db, qdrant_manager=qdrant, text_processor=text_processor)
        result = await resolver.search(
            query=body.query,
            filters=core_filters,
            limit_posts=body.limit_posts,
            limit_accounts=body.limit_accounts,
            cursor=body.cursor,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        elapsed = time.perf_counter() - t0
        log.info("search_duration query_type=general elapsed=%.4fs", elapsed)

    response = GeneralSearchResponse(
        posts=[PostResult(**p.__dict__) for p in result.posts],
        accounts=[AccountResult(**a.__dict__) for a in result.accounts],
        next_cursor=result.next_cursor,
        total_posts=result.total_posts,
        total_accounts=result.total_accounts,
    )
    await cache_set(key, response.model_dump(mode="json"))
    return response


# ── Account name ──────────────────────────────────────────────────────────────

@router.get(
    "/accounts",
    response_model=AccountNameSearchResponse,
    summary="Lexical + fuzzy blog-name search (pg_trgm)",
)
async def account_name_search(
    q: str = Query(..., min_length=1, max_length=100),
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    _api_key=Depends(require_api_key),
) -> AccountNameSearchResponse:
    key = make_cache_key("accounts", q, extra={"limit": limit})
    hit = await cache_get(key)
    if hit is not None:
        log.debug("cache_hit query_type=accounts")
        return AccountNameSearchResponse.model_validate(hit)
    log.debug("cache_miss query_type=accounts")

    t0 = time.perf_counter()
    results = await search_accounts_by_name(db, q, limit=limit)
    log.info("search_duration query_type=accounts elapsed=%.4fs", time.perf_counter() - t0)

    response = AccountNameSearchResponse(
        accounts=[AccountNameResult(**r.__dict__) for r in results]
    )
    await cache_set(key, response.model_dump(mode="json"))
    return response


# ── Tags ──────────────────────────────────────────────────────────────────────

@router.get(
    "/tags",
    response_model=TagSearchResponse,
    summary="Exact + prefix + semantic tag search",
)
async def tag_search(
    q: str = Query(..., min_length=1, max_length=100),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    qdrant=Depends(get_qdrant),
    text_processor=Depends(get_text_processor),
    _api_key=Depends(require_api_key),
) -> TagSearchResponse:
    key = make_cache_key("tags", q, extra={"limit": limit})
    hit = await cache_get(key)
    if hit is not None:
        log.debug("cache_hit query_type=tags")
        return TagSearchResponse.model_validate(hit)
    log.debug("cache_miss query_type=tags")

    t0 = time.perf_counter()
    results = await core_search_tags(db, qdrant, text_processor, q, limit=limit)
    log.info("search_duration query_type=tags elapsed=%.4fs", time.perf_counter() - t0)

    response = TagSearchResponse(tags=[TagResult(**r.__dict__) for r in results])
    await cache_set(key, response.model_dump(mode="json"))
    return response


# ── Communities ───────────────────────────────────────────────────────────────

@router.get(
    "/communities",
    response_model=CommunitySearchResponse,
    summary="Community discovery by query → centroid cosine + drill-down URL",
)
async def community_search(
    q: str = Query(..., min_length=1, max_length=100),
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    text_processor=Depends(get_text_processor),
    _api_key=Depends(require_api_key),
) -> CommunitySearchResponse:
    key = make_cache_key("communities", q, extra={"limit": limit})
    hit = await cache_get(key)
    if hit is not None:
        log.debug("cache_hit query_type=communities")
        return CommunitySearchResponse.model_validate(hit)
    log.debug("cache_miss query_type=communities")

    t0 = time.perf_counter()
    results = await search_communities(db, text_processor, q, limit=limit)
    log.info("search_duration query_type=communities elapsed=%.4fs", time.perf_counter() - t0)

    response = CommunitySearchResponse(
        communities=[CommunityResult(**r.__dict__) for r in results]
    )
    await cache_set(key, response.model_dump(mode="json"))
    return response
