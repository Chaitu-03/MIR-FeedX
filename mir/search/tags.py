"""
mir/search/tags.py
Tag Search resolver — Prompt 10.

Combines exact match, prefix match, and semantic (Qdrant) search.
Deduplicates and scores results via a weighted formula.

Bug fixed vs. original: `await qdrant_manager.search_tags(...)` was called
directly but search_tags is synchronous (QdrantClient). Fixed by wrapping in
asyncio.to_thread so it doesn't block the event loop.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class TagResult:
    tag_id: int
    name: str
    match_type: str   # "exact" | "prefix" | "semantic"
    usage_count: int
    cosine_sim: float
    score: float


async def search_tags(
    db: AsyncSession,
    qdrant_manager,
    text_processor,
    query: str,
    limit: int = 20,
) -> list[TagResult]:
    """
    1. Exact match   — SELECT … WHERE name = lower(q)
    2. Prefix match  — SELECT … WHERE name LIKE lower(q)||'%' ORDER BY usage_count DESC LIMIT 20
    3. Semantic      — embed q → Qdrant tags collection, limit=30
    4. Merge, dedup (exact/prefix take precedence over semantic), score, sort.

    Score formula:
      tag_score = (1.0 * is_exact)
                + (0.6 * cosine_sim)
                + (0.3 * log2(1 + usage_count) / max_log_usage)
    """
    q = query.lower().strip()
    if not q:
        return []

    # ── 1. Exact match ─────────────────────────────────────────────────────────
    exact_sql = text("""
        SELECT id, name, usage_count
        FROM tags
        WHERE name = :q
    """)
    exact_rows = (await db.execute(exact_sql, {"q": q})).fetchall()

    # ── 2. Prefix match ────────────────────────────────────────────────────────
    prefix_sql = text("""
        SELECT id, name, usage_count
        FROM tags
        WHERE name LIKE :q_prefix
        ORDER BY usage_count DESC
        LIMIT 20
    """)
    prefix_rows = (await db.execute(prefix_sql, {"q_prefix": f"{q}%"})).fetchall()

    # ── 3. Semantic search via Qdrant (sync client → thread) ───────────────────
    q_vec: np.ndarray = await asyncio.to_thread(text_processor.embed, [query])
    q_vec = q_vec[0]

    semantic_hits: list[tuple[int, float]] = await asyncio.to_thread(
        qdrant_manager.search_tags, q_vec.tolist(), 30
    )

    # ── 4. Fetch DB metadata for semantic-only hits ────────────────────────────
    semantic_ids = [tag_id for tag_id, _ in semantic_hits]
    sem_scores: dict[int, float] = {tag_id: score for tag_id, score in semantic_hits}

    semantic_meta: dict[int, dict] = {}
    if semantic_ids:
        sem_sql = text("""
            SELECT id, name, usage_count FROM tags WHERE id = ANY(:ids)
        """)
        sem_rows = (await db.execute(sem_sql, {"ids": semantic_ids})).fetchall()
        semantic_meta = {
            row.id: {"name": row.name, "usage_count": row.usage_count}
            for row in sem_rows
        }

    # ── 5. Merge & deduplicate (keyed by tag name) ─────────────────────────────
    # Exact and prefix results take precedence; semantic only fills gaps.
    merged: dict[str, dict] = {}

    for row in exact_rows:
        merged[row.name] = {
            "tag_id": row.id,
            "usage_count": row.usage_count,
            "is_exact": True,
            "cosine_sim": sem_scores.get(row.id, 0.0),
            "match_type": "exact",
        }

    for row in prefix_rows:
        if row.name not in merged:
            merged[row.name] = {
                "tag_id": row.id,
                "usage_count": row.usage_count,
                "is_exact": False,
                "cosine_sim": sem_scores.get(row.id, 0.0),
                "match_type": "prefix",
            }
        else:
            # Exact match already recorded — enrich cosine_sim if higher
            merged[row.name]["cosine_sim"] = max(
                merged[row.name]["cosine_sim"],
                sem_scores.get(row.id, 0.0),
            )

    for tag_id, cosine_sim in semantic_hits:
        meta = semantic_meta.get(tag_id)
        if meta is None:
            continue
        name = meta["name"]
        if name not in merged:
            merged[name] = {
                "tag_id": tag_id,
                "usage_count": meta["usage_count"],
                "is_exact": False,
                "cosine_sim": cosine_sim,
                "match_type": "semantic",
            }
        else:
            # Already in merged — upgrade cosine_sim if semantic score is better
            merged[name]["cosine_sim"] = max(merged[name]["cosine_sim"], cosine_sim)

    # ── 6. Score ───────────────────────────────────────────────────────────────
    # max_log_usage computed in a single pass — no extra query
    log_usages = [math.log2(1 + entry["usage_count"]) for entry in merged.values()]
    max_log_usage = max(log_usages) if log_usages else 1.0
    if max_log_usage == 0:
        max_log_usage = 1.0

    results: list[TagResult] = []
    for name, entry in merged.items():
        is_exact_f = 1.0 if entry["is_exact"] else 0.0
        log_usage = math.log2(1 + entry["usage_count"])
        score = (
            (1.0 * is_exact_f)
            + (0.6 * entry["cosine_sim"])
            + (0.3 * log_usage / max_log_usage)
        )
        results.append(
            TagResult(
                tag_id=entry["tag_id"],
                name=name,
                match_type=entry["match_type"],
                usage_count=entry["usage_count"],
                cosine_sim=entry["cosine_sim"],
                score=score,
            )
        )

    results.sort(key=lambda r: -r.score)
    return results[:limit]
