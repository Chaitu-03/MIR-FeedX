"""
mir/api/schemas.py
Pydantic v2 request / response models — Prompt 12.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ── General Search ────────────────────────────────────────────────────────────

class GeneralSearchFilters(BaseModel):
    date_from: datetime | None = Field(None, description="ISO datetime — inclusive lower bound")
    date_to: datetime | None = Field(None, description="ISO datetime — inclusive upper bound")
    min_notes: int | None = Field(None, ge=0, description="Only return posts with note_count ≥ this")
    tags: list[str] = Field(default_factory=list, description="Must match at least one of these tags")
    lang: str | None = Field(None, min_length=2, max_length=5, description="ISO 639-1 language code")


class GeneralSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Free-text query")
    filters: GeneralSearchFilters = Field(default_factory=GeneralSearchFilters)
    limit_posts: int = Field(20, ge=1, le=100)
    limit_accounts: int = Field(10, ge=0, le=50)
    cursor: str | None = Field(None, description="Opaque cursor from previous next_cursor")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "photography landscape",
                "filters": {"lang": "en", "min_notes": 50},
                "limit_posts": 20,
                "limit_accounts": 10,
            }
        }
    }


class PostResult(BaseModel):
    post_id: int
    account_id: int
    blog_name: str
    body_clean: str | None
    image_urls: list[str] = []
    note_count: int
    published_at: datetime | None
    tags: list[str] = []
    score: float
    lang: str | None = None


class AccountResult(BaseModel):
    account_id: int
    blog_name: str
    avatar_url: str | None
    description: str | None
    score: float


class GeneralSearchResponse(BaseModel):
    posts: list[PostResult]
    accounts: list[AccountResult]
    next_cursor: str | None
    total_posts: int
    total_accounts: int


# ── Account Name Search ───────────────────────────────────────────────────────

class AccountNameResult(BaseModel):
    blog_name: str
    match_type: Literal["exact", "prefix", "fuzzy"]
    score: float
    avatar_url: str | None = None
    description_snippet: str | None = None


class AccountNameSearchResponse(BaseModel):
    accounts: list[AccountNameResult]


# ── Tag Search ────────────────────────────────────────────────────────────────

class TagResult(BaseModel):
    tag_id: int
    name: str
    match_type: Literal["exact", "prefix", "semantic"]
    usage_count: int
    cosine_sim: float
    score: float


class TagSearchResponse(BaseModel):
    tags: list[TagResult]


# ── Community Search ──────────────────────────────────────────────────────────

class CommunityResult(BaseModel):
    community_id: int
    name: str
    type: Literal["tag_cluster", "account_cluster"]
    member_count: int
    top_tags: list[str] = []
    top_accounts: list[str] = []
    score: float
    drill_down_query: str = Field(..., description="Pre-formed General Search path")


class CommunitySearchResponse(BaseModel):
    communities: list[CommunityResult]


# ── Health / Stats ────────────────────────────────────────────────────────────

class DependencyHealth(BaseModel):
    status: Literal["connected", "error"]
    latency_ms: float | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    db: DependencyHealth
    qdrant: DependencyHealth
    redis: DependencyHealth


class StatsResponse(BaseModel):
    total_posts: int
    total_accounts: int
    total_tags: int
    nsfw_flagged_count: int | None = None
    qdrant_posts_count: int | None = None
    queue_depth: int | None = None
    crawl_status: dict[str, int] | None = None
    last_crawl: datetime | None = None


# ── Admin ─────────────────────────────────────────────────────────────────────

class ReindexRequest(BaseModel):
    model: str | None = None


class TaskIdResponse(BaseModel):
    task_id: str


class CacheClearResponse(BaseModel):
    cleared: int
