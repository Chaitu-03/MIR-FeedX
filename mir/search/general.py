"""
mir/search/general.py
General Search query resolver — Prompt 8.

Fuses PostgreSQL full-text search (keyword) with Qdrant vector search
via Reciprocal Rank Fusion, then re-ranks with engagement + recency boosts.
Cursor-based pagination (NOT offset).

Note on concurrency: QdrantManager uses the sync QdrantClient.  Qdrant calls
are therefore wrapped with asyncio.to_thread so keyword retrieval (async DB)
and vector retrieval (sync Qdrant) run truly in parallel via asyncio.gather.
"""
from __future__ import annotations

import asyncio
import base64
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from mir.config import settings


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class GeneralSearchFilters:
    date_from: datetime | None = None
    date_to: datetime | None = None
    min_notes: int | None = None
    tags: list[str] = field(default_factory=list)
    lang: str | None = None


@dataclass
class PostResult:
    post_id: int
    account_id: int
    blog_name: str
    body_clean: str | None
    image_urls: list[str]
    note_count: int
    published_at: datetime | None
    tags: list[str]
    score: float
    lang: str | None = None


@dataclass
class AccountResult:
    account_id: int
    blog_name: str
    avatar_url: str | None
    description: str | None
    score: float


@dataclass
class GeneralSearchResult:
    posts: list[PostResult]
    accounts: list[AccountResult]
    next_cursor: str | None
    total_posts: int
    total_accounts: int


# ── Cursor helpers ────────────────────────────────────────────────────────────

def encode_cursor(score: float, post_id: int) -> str:
    """Serialise (score, post_id) to an opaque base64 string."""
    payload = json.dumps({"score": score, "post_id": post_id})
    return base64.urlsafe_b64encode(payload.encode()).decode()


def decode_cursor(cursor: str) -> tuple[float, int]:
    """Deserialise cursor string → (score, post_id). Raises ValueError on bad input."""
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor.encode()))
        return float(payload["score"]), int(payload["post_id"])
    except Exception as exc:
        raise ValueError(f"Invalid cursor: {cursor!r}") from exc


# ── RRF ───────────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]],
    k: int | None = None,
) -> dict[int, float]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.
    score(d) = Σ  1 / (k + rank(d))  for every list that contains d.

    Args:
        ranked_lists: each inner list is [(id, original_score), ...] best-first.
        k: RRF smoothing constant (default: settings.rrf_k = 60).
    Returns:
        dict mapping id → combined RRF score.
    """
    k = k if k is not None else settings.rrf_k
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


# ── Re-ranking ────────────────────────────────────────────────────────────────

def rerank_score(
    rrf_score: float,
    note_count: int,
    published_at: datetime | None,
    boost_coeff: float | None = None,
    half_life_days: float | None = None,
) -> float:
    """
    final_score = rrf_score
                  * (1 + boost_coeff * log2(1 + note_count))
                  * (0.5 ** (age_days / half_life_days))
    """
    boost_coeff = boost_coeff if boost_coeff is not None else settings.note_count_boost
    half_life_days = half_life_days if half_life_days is not None else settings.recency_half_life_days

    engagement_boost = 1.0 + boost_coeff * math.log2(1 + note_count)

    if published_at is not None:
        now = datetime.now(tz=timezone.utc)
        if published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (now - published_at).total_seconds() / 86400)
    else:
        age_days = half_life_days  # unknown age → neutral (half of half-life)

    recency_boost = 0.5 ** (age_days / half_life_days)
    return rrf_score * engagement_boost * recency_boost


# ── Keyword retrieval ─────────────────────────────────────────────────────────

async def _keyword_search(
    db: AsyncSession,
    query: str,
    filters: GeneralSearchFilters,
    limit: int = 100,
) -> list[tuple[int, float, int, datetime | None]]:
    """
    Full-text search using ts_body GIN index.

    Uses websearch_to_tsquery (PostgreSQL ≥ 11) which natively supports:
      - Quoted phrases: "new york" → adjacent match
      - AND:  foo bar  → both words required
      - OR:   foo OR bar
      - NOT:  foo -bar

    Returns list of (post_id, kw_score, note_count, published_at).
    """
    sql = text("""
        SELECT p.id,
               ts_rank_cd(p.ts_body, websearch_to_tsquery('english', :q)) AS kw_score,
               p.note_count,
               p.published_at
        FROM posts p
        WHERE p.ts_body @@ websearch_to_tsquery('english', :q)
          AND p.nsfw = false
          AND (CAST(:lang      AS TEXT)        IS NULL OR p.lang         = CAST(:lang      AS TEXT))
          AND (CAST(:date_from AS TIMESTAMPTZ) IS NULL OR p.published_at >= CAST(:date_from AS TIMESTAMPTZ))
          AND (CAST(:date_to   AS TIMESTAMPTZ) IS NULL OR p.published_at <= CAST(:date_to   AS TIMESTAMPTZ))
          AND (CAST(:min_notes AS INTEGER)     IS NULL OR p.note_count   >= CAST(:min_notes AS INTEGER))
        ORDER BY kw_score DESC
        LIMIT :limit
    """)
    result = await db.execute(
        sql,
        {
            "q": query,
            "lang": filters.lang,
            "date_from": filters.date_from,
            "date_to": filters.date_to,
            "min_notes": filters.min_notes,
            "limit": limit,
        },
    )
    return [
        (row.id, float(row.kw_score), row.note_count, row.published_at)
        for row in result.fetchall()
    ]


# ── Post metadata fetch ───────────────────────────────────────────────────────

async def _fetch_post_metadata(
    db: AsyncSession,
    post_ids: list[int],
) -> dict[int, dict[str, Any]]:
    """Fetch full post + account metadata for the given post IDs."""
    if not post_ids:
        return {}

    sql = text("""
        SELECT p.id,
               p.account_id,
               a.blog_name,
               p.body_clean,
               p.image_urls,
               p.note_count,
               p.published_at,
               p.lang,
               COALESCE(
                 (SELECT array_agg(t.name)
                  FROM post_tags pt
                  JOIN tags t ON t.id = pt.tag_id
                  WHERE pt.post_id = p.id),
                 ARRAY[]::text[]
               ) AS tags
        FROM posts p
        JOIN accounts a ON a.id = p.account_id
        WHERE p.id = ANY(:ids)
          AND p.nsfw = false
    """)
    result = await db.execute(sql, {"ids": post_ids})
    return {
        row.id: {
            "account_id": row.account_id,
            "blog_name": row.blog_name,
            "body_clean": row.body_clean,
            "image_urls": row.image_urls or [],
            "note_count": row.note_count,
            "published_at": row.published_at,
            "lang": row.lang,
            "tags": list(row.tags) if row.tags else [],
        }
        for row in result.fetchall()
    }


# ── Account metadata fetch ────────────────────────────────────────────────────

async def _fetch_account_metadata(
    db: AsyncSession,
    account_ids: list[int],
) -> dict[int, dict[str, Any]]:
    if not account_ids:
        return {}
    sql = text("""
        SELECT id, blog_name, avatar_url, description
        FROM accounts
        WHERE id = ANY(:ids)
    """)
    result = await db.execute(sql, {"ids": account_ids})
    return {
        row.id: {
            "blog_name": row.blog_name,
            "avatar_url": row.avatar_url,
            "description": row.description,
        }
        for row in result.fetchall()
    }


# ── Account scoring ───────────────────────────────────────────────────────────

def _normalise(values: dict[int, float]) -> dict[int, float]:
    """Min-max normalise a score dict to [0, 1]."""
    if not values:
        return {}
    min_v = min(values.values())
    max_v = max(values.values())
    span = max_v - min_v
    if span == 0:
        return {k: 1.0 for k in values}
    return {k: (v - min_v) / span for k, v in values.items()}


def _score_accounts(
    reranked_posts: list[tuple[int, float, dict[str, Any]]],
    account_vector_scores: list[tuple[int, float]],
    top_n: int = 200,
) -> dict[int, float]:
    """
    Combines two account relevance signals:
      A (post aggregation): sum(post_final_score) / sqrt(count) over top-N posts
      B (embedding cosine): from Qdrant account search
    account_score = 0.6 * norm(A) + 0.4 * norm(B)
    """
    # Signal A — post aggregation with diminishing returns
    agg_a: dict[int, list[float]] = {}
    for _post_id, final_score, meta in reranked_posts[:top_n]:
        aid = meta["account_id"]
        agg_a.setdefault(aid, []).append(final_score)
    score_a: dict[int, float] = {
        aid: sum(scores) / math.sqrt(len(scores))
        for aid, scores in agg_a.items()
    }

    # Signal B — Qdrant vector similarity
    score_b: dict[int, float] = {aid: s for aid, s in account_vector_scores}

    # Union of all account IDs
    all_ids = set(score_a) | set(score_b)
    score_a = {aid: score_a.get(aid, 0.0) for aid in all_ids}
    score_b = {aid: score_b.get(aid, 0.0) for aid in all_ids}

    norm_a = _normalise(score_a)
    norm_b = _normalise(score_b)

    return {
        aid: 0.6 * norm_a.get(aid, 0.0) + 0.4 * norm_b.get(aid, 0.0)
        for aid in all_ids
    }


# ── Main resolver ─────────────────────────────────────────────────────────────

class GeneralSearchResolver:
    """
    Orchestrates keyword + vector retrieval, RRF fusion, re-ranking,
    account scoring, and cursor-based pagination.
    """

    def __init__(self, db: AsyncSession, qdrant_manager, text_processor):
        self.db = db
        self.qdrant = qdrant_manager
        self.text_processor = text_processor

    async def search(
        self,
        query: str,
        filters: GeneralSearchFilters | None = None,
        limit_posts: int = 20,
        limit_accounts: int = 10,
        cursor: str | None = None,
    ) -> GeneralSearchResult:
        filters = filters or GeneralSearchFilters()

        # ── 1. Embed query (sync → thread so it doesn't block the loop) ───────
        q_vec: np.ndarray = await asyncio.to_thread(
            self.text_processor.embed, [query]
        )
        q_vec = q_vec[0]
        q_vec_list: list[float] = q_vec.tolist()

        # Build Qdrant filter dict from GeneralSearchFilters
        qdrant_filters: dict[str, Any] = {}
        if filters.lang:
            qdrant_filters["lang"] = filters.lang
        if filters.date_from:
            qdrant_filters["date_from"] = filters.date_from.date().isoformat()
        if filters.min_notes is not None:
            qdrant_filters["min_notes"] = filters.min_notes

        # ── 2–4. Parallel retrieval: async DB + two sync Qdrant calls ─────────
        # QdrantManager.search_posts/search_accounts are SYNC (QdrantClient).
        # Use asyncio.to_thread so they run concurrently with the async DB query.
        kw_results, vec_results, acc_vector_results = await asyncio.gather(
            _keyword_search(self.db, query, filters, limit=100),
            asyncio.to_thread(
                self.qdrant.search_posts,
                q_vec_list, 100, qdrant_filters, True,
            ),
            asyncio.to_thread(
                self.qdrant.search_accounts,
                q_vec_list, 50,
            ),
        )

        # ── 5. RRF over keyword + vector post results ──────────────────────────
        kw_ranked: list[tuple[int, float]] = [(r[0], r[1]) for r in kw_results]
        vec_ranked: list[tuple[int, float]] = list(vec_results)

        rrf_scores = reciprocal_rank_fusion([kw_ranked, vec_ranked])

        # ── 6. Fetch full post metadata for all candidates ─────────────────────
        post_meta = await _fetch_post_metadata(self.db, list(rrf_scores.keys()))

        # ── 7. Re-rank with engagement + recency boosts ────────────────────────
        reranked: list[tuple[int, float, dict[str, Any]]] = []
        for post_id, rrf_score in rrf_scores.items():
            meta = post_meta.get(post_id)
            if meta is None:
                # Post deleted or NSFW-filtered between retrieval and fetch — skip
                continue
            final = rerank_score(rrf_score, meta["note_count"], meta["published_at"])
            reranked.append((post_id, final, meta))

        # Sort descending by score; ties broken by descending post_id
        reranked.sort(key=lambda x: (-x[1], -x[0]))

        # ── 8. Account scoring ─────────────────────────────────────────────────
        account_scores = _score_accounts(reranked, list(acc_vector_results), top_n=200)
        sorted_accounts = sorted(account_scores.items(), key=lambda x: -x[1])

        # ── 9. Cursor-based pagination ─────────────────────────────────────────
        cursor_score: float | None = None
        cursor_id: int | None = None
        if cursor:
            cursor_score, cursor_id = decode_cursor(cursor)

        # Position-based paging: find where the cursor post sits in the sorted
        # list by post_id (immune to floating-point drift in time-dependent scores).
        # Falls back to score-based skip if the cursor post was evicted (deleted /
        # NSFW-filtered) between the two page fetches.
        start_idx = 0
        if cursor_id is not None:
            found = False
            for i, (pid, _, _) in enumerate(reranked):
                if pid == cursor_id:
                    start_idx = i + 1
                    found = True
                    break
            if not found and cursor_score is not None:
                # Cursor post gone: skip everything scored higher than cursor
                for i, (pid, fs, _) in enumerate(reranked):
                    if fs < cursor_score - 1e-9:
                        start_idx = i
                        break
                else:
                    start_idx = len(reranked)

        paginated: list[tuple[int, float, dict[str, Any]]] = reranked[
            start_idx : start_idx + limit_posts
        ]

        # Emit next_cursor only when there are more results to fetch
        next_cursor: str | None = None
        if paginated and (start_idx + len(paginated)) < len(reranked):
            last_id, last_score, _ = paginated[-1]
            next_cursor = encode_cursor(last_score, last_id)

        # ── 10. Build response ─────────────────────────────────────────────────
        post_results = [
            PostResult(
                post_id=pid,
                account_id=meta["account_id"],
                blog_name=meta["blog_name"],
                body_clean=meta["body_clean"],
                image_urls=meta["image_urls"],
                note_count=meta["note_count"],
                published_at=meta["published_at"],
                tags=meta["tags"],
                lang=meta["lang"],
                score=score,
            )
            for pid, score, meta in paginated
        ]

        top_account_ids = [aid for aid, _ in sorted_accounts[:limit_accounts]]
        acc_meta = await _fetch_account_metadata(self.db, top_account_ids)
        account_results = [
            AccountResult(
                account_id=aid,
                blog_name=acc_meta[aid]["blog_name"],
                avatar_url=acc_meta[aid]["avatar_url"],
                description=acc_meta[aid]["description"],
                score=account_scores[aid],
            )
            for aid in top_account_ids
            if aid in acc_meta
        ]

        return GeneralSearchResult(
            posts=post_results,
            accounts=account_results,
            next_cursor=next_cursor,
            total_posts=len(reranked),
            total_accounts=len(account_scores),
        )
