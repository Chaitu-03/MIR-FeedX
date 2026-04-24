"""
mir/search/communities.py
Community Search resolver — Prompt 11, Part B.

Brute-force cosine similarity over community centroids loaded from the DB,
boosted by member count.  Scales to ~500 communities (nightly rebuild cap).
"""
from __future__ import annotations

import asyncio
import math
import urllib.parse
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class CommunityResult:
    community_id: int
    name: str
    type: str                       # "tag_cluster" | "account_cluster"
    member_count: int
    top_tags: list[str] = field(default_factory=list)
    top_accounts: list[str] = field(default_factory=list)
    score: float = 0.0
    drill_down_query: str = ""      # "/api/v1/search/general?q=…"


def community_score(
    cosine_sim: float,
    member_count: int,
) -> float:
    """
    community_score = cosine_sim * (1 + 0.1 * log2(1 + member_count))
    """
    return cosine_sim * (1 + 0.1 * math.log2(1 + member_count))


def build_drill_down_query(top_tags: list[str], fallback_query: str = "") -> str:
    """
    Build a pre-formed General Search URL from the community's top tags.

    Example: top_tags=["photography", "landscape", "nature"]
    → "/api/v1/search/general?q=photography+landscape+nature"

    Falls back to fallback_query if top_tags is empty.
    """
    terms = top_tags[:3] if top_tags else ([fallback_query] if fallback_query else [])
    q_str = urllib.parse.quote_plus(" ".join(terms))
    return f"/api/v1/search/general?q={q_str}"


async def search_communities(
    db: AsyncSession,
    text_processor,
    query: str,
    limit: int = 10,
) -> list[CommunityResult]:
    """
    1. Embed q → q_vec (384-d, MiniLM) via thread so it doesn't block the loop.
    2. Load all community centroids from communities table (brute force, ≤500 rows).
    3. community_score = cosine_sim(q_vec, centroid) * (1 + 0.1 * log2(member_count)).
    4. Return top N with drill_down_query pre-built from top tags.
    """
    q_vec_arr: np.ndarray = await asyncio.to_thread(text_processor.embed, [query])
    q_vec: np.ndarray = q_vec_arr[0].astype(np.float32)
    q_norm = float(np.linalg.norm(q_vec))
    if q_norm < 1e-10:
        return []

    sql = text("""
        SELECT id, name, type, member_ids, centroid
        FROM communities
        WHERE centroid IS NOT NULL
    """)
    rows = (await db.execute(sql)).fetchall()
    if not rows:
        return []

    scored: list[tuple[float, dict]] = []
    for row in rows:
        centroid = np.array(row.centroid, dtype=np.float32)
        c_norm = float(np.linalg.norm(centroid))
        if c_norm < 1e-10:
            continue

        cos_sim = float(np.dot(q_vec, centroid) / (q_norm * c_norm))
        member_ids: list[int] = row.member_ids or []
        score = community_score(cos_sim, len(member_ids))

        scored.append((score, {
            "id": row.id,
            "name": row.name or "",
            "type": row.type or "tag_cluster",
            "member_ids": member_ids,
        }))

    scored.sort(key=lambda x: -x[0])
    top = scored[:limit]

    results: list[CommunityResult] = []
    for score, data in top:
        community_type: str = data["type"]
        member_ids: list[int] = data["member_ids"]

        top_tags: list[str] = []
        top_accounts: list[str] = []

        if community_type == "tag_cluster" and member_ids:
            tag_rows = (await db.execute(
                text("SELECT name FROM tags WHERE id = ANY(:ids) ORDER BY usage_count DESC LIMIT 5"),
                {"ids": member_ids},
            )).fetchall()
            top_tags = [r.name for r in tag_rows]

        elif community_type == "account_cluster" and member_ids:
            acc_rows = (await db.execute(
                text("SELECT blog_name FROM accounts WHERE id = ANY(:ids) LIMIT 5"),
                {"ids": member_ids},
            )).fetchall()
            top_accounts = [r.blog_name for r in acc_rows]

        results.append(
            CommunityResult(
                community_id=data["id"],
                name=data["name"],
                type=community_type,
                member_count=len(member_ids),
                top_tags=top_tags,
                top_accounts=top_accounts,
                score=score,
                drill_down_query=build_drill_down_query(top_tags, query),
            )
        )

    return results
