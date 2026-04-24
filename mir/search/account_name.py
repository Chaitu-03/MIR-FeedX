"""
mir/search/account_name.py
Account Name Search resolver — Prompt 9.

Pure lexical / fuzzy search using pg_trgm.
No semantic embedding — matches query against accounts.blog_name.

Bug fixed vs. original: the original combined SET LOCAL + SELECT in a single
db.execute() call. asyncpg (used by SQLAlchemy's async engine) only executes
the first statement in a multi-statement string, silently ignoring the rest.
Fix: use similarity(blog_name, :q) > :threshold explicitly in WHERE clause so
SET LOCAL is not needed at all.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class AccountNameResult:
    blog_name: str
    match_type: str           # "exact" | "prefix" | "fuzzy"
    score: float
    avatar_url: str | None
    description_snippet: str | None  # first 120 chars of description


_DESCRIPTION_SNIPPET_LEN = 120
_DEFAULT_SIM_THRESHOLD = 0.3


async def search_accounts_by_name(
    db: AsyncSession,
    query: str,
    limit: int = 10,
    similarity_threshold: float = _DEFAULT_SIM_THRESHOLD,
) -> list[AccountNameResult]:
    """
    Query accounts using pg_trgm trigram similarity.

    Ranking priority:
      1. Exact match  (blog_name = query)
      2. Prefix match (blog_name LIKE query%)
      3. Fuzzy match  (trigram similarity ≥ threshold)

    Score formula:
      name_score = (10 * is_exact + 5 * is_prefix + sim)
                   * (1 + 0.05 * log2(1 + total_posts))

    Note: threshold is applied via explicit similarity() > :threshold in WHERE
    rather than SET LOCAL pg_trgm.similarity_threshold, which does not work
    with asyncpg's prepared-statement execution model.
    """
    q = query.lower().strip()
    if not q:
        return []

    # pg_trgm must be loaded (handled by init.sql / migration).
    # We use similarity(...) > threshold explicitly to avoid SET LOCAL.
    sql = text("""
        SELECT blog_name,
               total_posts,
               avatar_url,
               description,
               (blog_name = :q)::int                  AS is_exact,
               (blog_name LIKE :q_prefix)::int         AS is_prefix,
               similarity(blog_name, :q)               AS sim
        FROM accounts
        WHERE similarity(blog_name, :q) > :threshold
           OR blog_name LIKE :q_prefix
        ORDER BY is_exact DESC, is_prefix DESC, sim DESC
        LIMIT :limit
    """)

    result = await db.execute(
        sql,
        {
            "q": q,
            "q_prefix": f"{q}%",
            "threshold": similarity_threshold,
            "limit": limit,
        },
    )
    rows = result.fetchall()

    results: list[AccountNameResult] = []
    for row in rows:
        is_exact = bool(row.is_exact)
        is_prefix = bool(row.is_prefix)
        sim = float(row.sim)
        total_posts = row.total_posts or 0

        raw_score = (10 * is_exact + 5 * is_prefix + sim) * (
            1 + 0.05 * math.log2(1 + total_posts)
        )

        if is_exact:
            match_type = "exact"
        elif is_prefix:
            match_type = "prefix"
        else:
            match_type = "fuzzy"

        desc = row.description or ""
        snippet = desc[:_DESCRIPTION_SNIPPET_LEN] if desc else None

        results.append(
            AccountNameResult(
                blog_name=row.blog_name,
                match_type=match_type,
                score=raw_score,
                avatar_url=row.avatar_url,
                description_snippet=snippet,
            )
        )

    return results
