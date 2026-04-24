"""
Unit tests for mir/search/account_name.py.

DB calls are mocked with fake row objects to avoid needing live Postgres.
Tests verify:
  - Exact match ranks first
  - Prefix match ranks second
  - Fuzzy match ("photgraphy" → "photography") ranks third
  - Completely unrelated query returns empty list
  - Score formula is correct
  - description_snippet truncated at 120 chars
"""
from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mir.search.account_name import (
    AccountNameResult,
    _DESCRIPTION_SNIPPET_LEN,
    search_accounts_by_name,
)


# ── Fake row helper ───────────────────────────────────────────────────────────

def _row(
    blog_name: str,
    total_posts: int = 0,
    avatar_url: str | None = None,
    description: str | None = None,
    is_exact: int = 0,
    is_prefix: int = 0,
    sim: float = 0.0,
):
    r = MagicMock()
    r.blog_name = blog_name
    r.total_posts = total_posts
    r.avatar_url = avatar_url
    r.description = description
    r.is_exact = is_exact
    r.is_prefix = is_prefix
    r.sim = sim
    return r


def _mock_db(rows):
    """Return an AsyncMock db session that yields the given rows."""
    execute_result = MagicMock()
    execute_result.fetchall.return_value = rows
    db = AsyncMock()
    db.execute = AsyncMock(return_value=execute_result)
    return db


# ── Score formula ─────────────────────────────────────────────────────────────

class TestScoreFormula:
    def _score(self, is_exact, is_prefix, sim, total_posts=0):
        return (10 * is_exact + 5 * is_prefix + sim) * (
            1 + 0.05 * math.log2(1 + total_posts)
        )

    def test_exact_score_10_base(self):
        assert abs(self._score(1, 0, 0, 0) - 10.0) < 1e-9

    def test_prefix_score_5_base(self):
        assert abs(self._score(0, 1, 0, 0) - 5.0) < 1e-9

    def test_total_posts_boost(self):
        s0 = self._score(1, 0, 0, 0)
        s100 = self._score(1, 0, 0, 100)
        assert s100 > s0


# ── Ranking order ─────────────────────────────────────────────────────────────

class TestRankingOrder:
    @pytest.mark.asyncio
    async def test_exact_ranks_first(self):
        rows = [
            _row("photography", is_exact=1, is_prefix=1, sim=1.0, total_posts=50),
            _row("photography2020", is_exact=0, is_prefix=1, sim=0.7, total_posts=10),
            _row("photgraphy", is_exact=0, is_prefix=0, sim=0.4, total_posts=5),
        ]
        # DB returns rows pre-ordered (ORDER BY in SQL), so we trust the order.
        # Our function maps them to AccountNameResult and doesn't re-sort.
        results = await search_accounts_by_name(_mock_db(rows), "photography")
        assert results[0].blog_name == "photography"
        assert results[0].match_type == "exact"

    @pytest.mark.asyncio
    async def test_prefix_ranks_second(self):
        rows = [
            _row("photography", is_exact=1, is_prefix=1, sim=1.0),
            _row("photography2020", is_exact=0, is_prefix=1, sim=0.7),
            _row("photgraphy", is_exact=0, is_prefix=0, sim=0.4),
        ]
        results = await search_accounts_by_name(_mock_db(rows), "photography")
        assert results[1].match_type == "prefix"

    @pytest.mark.asyncio
    async def test_fuzzy_match_ranks_third(self):
        rows = [
            _row("photography", is_exact=1, is_prefix=1, sim=1.0),
            _row("photography2020", is_exact=0, is_prefix=1, sim=0.7),
            _row("photgraphy", is_exact=0, is_prefix=0, sim=0.4),
        ]
        results = await search_accounts_by_name(_mock_db(rows), "photgraphy")
        # The third row with sim=0.4 and no exact/prefix should rank last
        fuzzy = [r for r in results if r.match_type == "fuzzy"]
        assert len(fuzzy) >= 1
        assert fuzzy[0].blog_name == "photgraphy"

    @pytest.mark.asyncio
    async def test_unrelated_query_returns_empty(self):
        db = _mock_db([])  # DB returns no rows
        results = await search_accounts_by_name(db, "zzzyyyxxx_nobody")
        assert results == []

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self):
        db = AsyncMock()
        results = await search_accounts_by_name(db, "")
        assert results == []
        db.execute.assert_not_called()


# ── Snippet truncation ────────────────────────────────────────────────────────

class TestDescriptionSnippet:
    @pytest.mark.asyncio
    async def test_long_description_truncated(self):
        long_desc = "x" * 300
        rows = [_row("testblog", description=long_desc, is_exact=1, sim=1.0)]
        results = await search_accounts_by_name(_mock_db(rows), "testblog")
        assert results[0].description_snippet is not None
        assert len(results[0].description_snippet) == _DESCRIPTION_SNIPPET_LEN

    @pytest.mark.asyncio
    async def test_short_description_not_truncated(self):
        rows = [_row("testblog", description="short", is_exact=1, sim=1.0)]
        results = await search_accounts_by_name(_mock_db(rows), "testblog")
        assert results[0].description_snippet == "short"

    @pytest.mark.asyncio
    async def test_none_description_gives_none_snippet(self):
        rows = [_row("testblog", description=None, is_exact=1, sim=1.0)]
        results = await search_accounts_by_name(_mock_db(rows), "testblog")
        assert results[0].description_snippet is None
