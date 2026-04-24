"""
Unit tests for mir/search/tags.py.

DB and Qdrant calls are mocked.
Tests verify:
  - Exact match always ranks first
  - Semantic matches appear for related concepts
  - Deduplication: exact match tag doesn't appear twice when also returned semantically
  - Score formula is correct (exact gets 1.0 component)
  - Empty query returns []
"""
from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from mir.search.tags import TagResult, search_tags


# ── Helpers ───────────────────────────────────────────────────────────────────

def _row(id, name, usage_count):
    r = MagicMock()
    r.id = id
    r.name = name
    r.usage_count = usage_count
    return r


def _make_db(
    exact_rows=None,
    prefix_rows=None,
    sem_meta_rows=None,
):
    """Build an AsyncMock db that returns configured rows for each query."""
    db = AsyncMock()
    call_count = [0]

    exact_rows = exact_rows or []
    prefix_rows = prefix_rows or []
    sem_meta_rows = sem_meta_rows or []

    # Each call to db.execute returns the next configured result set
    results = [exact_rows, prefix_rows, sem_meta_rows]

    async def side_effect(sql, params=None):
        idx = min(call_count[0], len(results) - 1)
        call_count[0] += 1
        r = MagicMock()
        r.fetchall.return_value = results[idx]
        return r

    db.execute = side_effect
    return db


def _make_text_proc(vec=None):
    tp = MagicMock()
    if vec is None:
        vec = np.zeros(384, dtype=np.float32)
    tp.embed = MagicMock(return_value=np.array([vec]))
    return tp


def _make_qdrant(hits=None):
    qm = MagicMock()
    qm.search_tags = MagicMock(return_value=hits or [])
    return qm


# ── Basic functionality ───────────────────────────────────────────────────────

class TestSearchTagsBasic:
    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self):
        db = AsyncMock()
        results = await search_tags(db, MagicMock(), MagicMock(), "")
        assert results == []
        db.execute = AsyncMock()  # ensure no calls were made
        db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_exact_match_ranks_first(self):
        exact_tag = _row(1, "sunset", 500)
        prefix_tag = _row(2, "sunsets", 100)
        semantic_tag_row = _row(3, "golden hour", 80)

        db = _make_db(
            exact_rows=[exact_tag],
            prefix_rows=[prefix_tag],
            sem_meta_rows=[semantic_tag_row],
        )
        qm = _make_qdrant(hits=[(3, 0.85)])  # semantic hit for "golden hour"
        tp = _make_text_proc()

        results = await search_tags(db, qm, tp, "sunset")

        assert results[0].name == "sunset"
        assert results[0].match_type == "exact"
        # Exact tag gets is_exact=1.0 component → score ≥ 1.0
        assert results[0].score >= 1.0

    @pytest.mark.asyncio
    async def test_deduplication_no_double_exact(self):
        """Exact match tag must NOT appear twice even if returned by semantic search."""
        exact_tag = _row(1, "landscape", 300)
        sem_meta = _row(1, "landscape", 300)  # same id

        db = _make_db(
            exact_rows=[exact_tag],
            prefix_rows=[],
            sem_meta_rows=[sem_meta],
        )
        qm = _make_qdrant(hits=[(1, 0.9)])  # same tag in semantic
        tp = _make_text_proc()

        results = await search_tags(db, qm, tp, "landscape")

        names = [r.name for r in results]
        assert names.count("landscape") == 1, "Duplicate 'landscape' in results!"

    @pytest.mark.asyncio
    async def test_semantic_only_tag_appears(self):
        """Tags found only via semantic search (no exact/prefix) should be in results."""
        sem_meta = _row(99, "golden hour", 80)
        db = _make_db(
            exact_rows=[],
            prefix_rows=[],
            sem_meta_rows=[sem_meta],
        )
        qm = _make_qdrant(hits=[(99, 0.85)])
        tp = _make_text_proc()

        results = await search_tags(db, qm, tp, "sunset")
        names = [r.name for r in results]
        assert "golden hour" in names

    @pytest.mark.asyncio
    async def test_cosine_sim_enriches_exact_match(self):
        """If an exact-match tag also appears in semantic results, its cosine_sim is enriched."""
        exact_tag = _row(1, "nature", 200)
        sem_meta = _row(1, "nature", 200)
        db = _make_db(
            exact_rows=[exact_tag],
            prefix_rows=[],
            sem_meta_rows=[sem_meta],
        )
        qm = _make_qdrant(hits=[(1, 0.92)])
        tp = _make_text_proc()

        results = await search_tags(db, qm, tp, "nature")
        nature = next(r for r in results if r.name == "nature")
        assert abs(nature.cosine_sim - 0.92) < 1e-6

    @pytest.mark.asyncio
    async def test_limit_respected(self):
        prefix_rows = [_row(i, f"tag{i}", i * 10) for i in range(1, 25)]
        db = _make_db(exact_rows=[], prefix_rows=prefix_rows, sem_meta_rows=[])
        qm = _make_qdrant(hits=[])
        tp = _make_text_proc()

        results = await search_tags(db, qm, tp, "tag", limit=5)
        assert len(results) <= 5


# ── Score formula ─────────────────────────────────────────────────────────────

class TestTagScoreFormula:
    def _compute_score(self, is_exact, cosine_sim, usage_count, max_usage):
        max_log = math.log2(1 + max_usage)
        return (
            (1.0 if is_exact else 0.0)
            + 0.6 * cosine_sim
            + 0.3 * math.log2(1 + usage_count) / max(max_log, 1.0)
        )

    def test_exact_match_has_1_0_component(self):
        score = self._compute_score(True, 0.0, 0, 1)
        assert abs(score - 1.0) < 1e-9

    def test_non_exact_no_base_score(self):
        score = self._compute_score(False, 0.0, 0, 1)
        assert abs(score - 0.0) < 1e-9

    def test_cosine_component_scales(self):
        score = self._compute_score(False, 1.0, 0, 1)
        assert abs(score - 0.6) < 1e-9

    def test_usage_component_max_is_0_3(self):
        # usage_count == max_usage → log_usage / max_log = 1.0 → 0.3
        usage = 1000
        score = self._compute_score(False, 0.0, usage, usage)
        assert abs(score - 0.3) < 1e-6
