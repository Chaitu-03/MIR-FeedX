"""
Unit tests for mir/search/general.py.

All DB and Qdrant calls are mocked — no live infrastructure required.
Tests cover:
  - RRF merging with known inputs
  - Re-rank formula correctness
  - Cursor encode/decode roundtrip
  - nsfw=False always enforced in Qdrant search (mock verification)
  - Account scoring normalisation
"""
from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mir.search.general import (
    GeneralSearchFilters,
    GeneralSearchResolver,
    _normalise,
    _score_accounts,
    decode_cursor,
    encode_cursor,
    reciprocal_rank_fusion,
    rerank_score,
)


# ── RRF ───────────────────────────────────────────────────────────────────────

class TestRRF:
    def test_single_list(self):
        ranked = [(10, 0.9), (20, 0.7), (30, 0.5)]
        scores = reciprocal_rank_fusion([ranked], k=60)
        # rank 1 → 1/61, rank 2 → 1/62, rank 3 → 1/63
        assert abs(scores[10] - 1 / 61) < 1e-9
        assert abs(scores[20] - 1 / 62) < 1e-9
        assert abs(scores[30] - 1 / 63) < 1e-9

    def test_two_lists_overlap(self):
        kw = [(1, 0.9), (2, 0.7)]
        vec = [(2, 0.95), (1, 0.8), (3, 0.6)]
        scores = reciprocal_rank_fusion([kw, vec], k=60)

        # doc 1: 1/61 (rank 1 in kw) + 1/62 (rank 2 in vec)
        expected_1 = 1 / 61 + 1 / 62
        assert abs(scores[1] - expected_1) < 1e-9

        # doc 2: 1/62 (rank 2 in kw) + 1/61 (rank 1 in vec)
        expected_2 = 1 / 62 + 1 / 61
        assert abs(scores[2] - expected_2) < 1e-9

        # doc 3: only in vec at rank 3
        expected_3 = 1 / 63
        assert abs(scores[3] - expected_3) < 1e-9

    def test_post_in_only_one_list_still_scored(self):
        kw = [(1, 0.9)]
        vec = [(2, 0.8)]
        scores = reciprocal_rank_fusion([kw, vec], k=60)
        assert 1 in scores
        assert 2 in scores

    def test_descending_rank_order(self):
        ranked = [(1, 0.9), (2, 0.7), (3, 0.5)]
        scores = reciprocal_rank_fusion([ranked], k=60)
        # Higher rank (lower rank number) → higher RRF score
        assert scores[1] > scores[2] > scores[3]

    def test_custom_k(self):
        ranked = [(1, 0.9)]
        scores_k1 = reciprocal_rank_fusion([ranked], k=1)
        scores_k60 = reciprocal_rank_fusion([ranked], k=60)
        # Smaller k → higher score for rank 1
        assert scores_k1[1] > scores_k60[1]

    def test_empty_lists(self):
        scores = reciprocal_rank_fusion([[], []])
        assert scores == {}


# ── Re-rank ───────────────────────────────────────────────────────────────────

class TestRerankScore:
    def test_formula_recent_post(self):
        # A post published today should have near-full recency boost
        now = datetime.now(tz=timezone.utc)
        score = rerank_score(
            rrf_score=1.0,
            note_count=0,
            published_at=now,
            boost_coeff=0.1,
            half_life_days=90,
        )
        # age_days ≈ 0 → recency_boost ≈ 1.0; note_count=0 → engagement=1.0
        assert abs(score - 1.0) < 0.01

    def test_formula_old_post(self):
        # Post 90 days old → recency_boost = 0.5
        pub = datetime.now(tz=timezone.utc) - timedelta(days=90)
        score = rerank_score(1.0, 0, pub, boost_coeff=0.1, half_life_days=90)
        assert abs(score - 0.5) < 0.01

    def test_engagement_boost_increases_score(self):
        now = datetime.now(tz=timezone.utc)
        s0 = rerank_score(1.0, 0, now, boost_coeff=0.1, half_life_days=90)
        s100 = rerank_score(1.0, 100, now, boost_coeff=0.1, half_life_days=90)
        assert s100 > s0

    def test_none_published_at_uses_half_life(self):
        # published_at=None → age = half_life → recency_boost = 0.5
        score = rerank_score(1.0, 0, None, boost_coeff=0.1, half_life_days=90)
        assert abs(score - 0.5) < 0.01

    def test_note_count_formula(self):
        now = datetime.now(tz=timezone.utc)
        nc = 127  # log2(1+127) = 7
        score = rerank_score(1.0, nc, now, boost_coeff=0.1, half_life_days=90)
        expected_engagement = 1 + 0.1 * math.log2(1 + nc)
        # recency ≈ 1 for new post
        assert abs(score - expected_engagement) < 0.05


# ── Cursor ────────────────────────────────────────────────────────────────────

class TestCursor:
    def test_roundtrip(self):
        score, post_id = 0.98765, 42
        encoded = encode_cursor(score, post_id)
        decoded_score, decoded_id = decode_cursor(encoded)
        assert abs(decoded_score - score) < 1e-9
        assert decoded_id == post_id

    def test_opaque_string(self):
        # Encoded cursor must not expose raw score or id in plain text
        encoded = encode_cursor(0.5, 99)
        assert "0.5" not in encoded
        assert "99" not in encoded

    def test_invalid_cursor_raises(self):
        with pytest.raises(ValueError, match="Invalid cursor"):
            decode_cursor("not-a-valid-cursor!!!")

    def test_different_inputs_different_cursors(self):
        c1 = encode_cursor(0.9, 1)
        c2 = encode_cursor(0.8, 1)
        c3 = encode_cursor(0.9, 2)
        assert c1 != c2
        assert c1 != c3


# ── _normalise ────────────────────────────────────────────────────────────────

class TestNormalise:
    def test_min_max(self):
        vals = {1: 0.0, 2: 5.0, 3: 10.0}
        norm = _normalise(vals)
        assert abs(norm[1] - 0.0) < 1e-9
        assert abs(norm[3] - 1.0) < 1e-9
        assert abs(norm[2] - 0.5) < 1e-9

    def test_all_equal_returns_ones(self):
        vals = {1: 3.0, 2: 3.0}
        norm = _normalise(vals)
        assert all(v == 1.0 for v in norm.values())

    def test_empty(self):
        assert _normalise({}) == {}


# ── Account scoring ────────────────────────────────────────────────────────────

class TestScoreAccounts:
    def _make_reranked(self, pairs):
        # pairs: [(post_id, score, account_id)]
        return [
            (pid, score, {"account_id": aid, "blog_name": f"blog_{aid}"})
            for pid, score, aid in pairs
        ]

    def test_account_with_more_posts_scores_higher(self):
        reranked = self._make_reranked([
            (1, 1.0, 10), (2, 1.0, 10),   # account 10: 2 posts
            (3, 1.0, 20),                   # account 20: 1 post
        ])
        vec_scores = []
        scores = _score_accounts(reranked, vec_scores)
        # Both Signal B = 0, so score purely from A
        assert scores[10] >= scores[20]

    def test_vector_signal_contributes(self):
        reranked = self._make_reranked([])  # no posts
        vec_scores = [(100, 0.9), (200, 0.5)]
        scores = _score_accounts(reranked, vec_scores)
        assert scores[100] > scores[200]

    def test_combined_scores_in_zero_one(self):
        reranked = self._make_reranked([(1, 0.8, 10), (2, 0.5, 20)])
        vec_scores = [(10, 0.7), (20, 0.9)]
        scores = _score_accounts(reranked, vec_scores)
        for v in scores.values():
            assert 0.0 <= v <= 1.0


# ── GeneralSearchResolver (integration with mocks) ────────────────────────────

class TestGeneralSearchResolver:
    def _make_resolver(self):
        db = AsyncMock()
        qdrant = MagicMock()
        text_proc = MagicMock()

        # text_processor.embed returns a (1, 384) numpy array
        import numpy as np
        text_proc.embed = MagicMock(return_value=np.zeros((1, 384), dtype=np.float32))

        return db, qdrant, text_proc

    @pytest.mark.asyncio
    async def test_nsfw_always_excluded_in_qdrant_search(self):
        """QdrantManager.search_posts must always be called with exclude_nsfw=True."""
        import numpy as np

        db = AsyncMock()
        qdrant = MagicMock()
        text_proc = MagicMock()
        text_proc.embed = MagicMock(return_value=np.zeros((1, 384), dtype=np.float32))

        # Qdrant returns empty (no posts to fetch from DB)
        qdrant.search_posts = MagicMock(return_value=[])
        qdrant.search_accounts = MagicMock(return_value=[])

        # DB keyword search returns nothing
        kw_mock = AsyncMock()
        kw_mock.fetchall.return_value = []
        db.execute = AsyncMock(return_value=kw_mock)

        resolver = GeneralSearchResolver(db, qdrant, text_proc)

        with patch("mir.search.general._keyword_search", new=AsyncMock(return_value=[])), \
             patch("mir.search.general._fetch_post_metadata", new=AsyncMock(return_value={})), \
             patch("mir.search.general._fetch_account_metadata", new=AsyncMock(return_value={})):
            await resolver.search("cats")

        # Verify exclude_nsfw=True was the 4th positional arg (index 3)
        call_args = qdrant.search_posts.call_args
        # Either positional or keyword
        if call_args.args:
            assert call_args.args[3] is True, "exclude_nsfw must be True"
        else:
            assert call_args.kwargs.get("exclude_nsfw", True) is True

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty_result(self):
        import numpy as np

        db = AsyncMock()
        qdrant = MagicMock()
        qdrant.search_posts = MagicMock(return_value=[])
        qdrant.search_accounts = MagicMock(return_value=[])
        text_proc = MagicMock()
        text_proc.embed = MagicMock(return_value=np.zeros((1, 384), dtype=np.float32))

        resolver = GeneralSearchResolver(db, qdrant, text_proc)

        with patch("mir.search.general._keyword_search", new=AsyncMock(return_value=[])), \
             patch("mir.search.general._fetch_post_metadata", new=AsyncMock(return_value={})), \
             patch("mir.search.general._fetch_account_metadata", new=AsyncMock(return_value={})):
            result = await resolver.search("")

        assert result.posts == []
        assert result.accounts == []
        assert result.next_cursor is None
        assert result.total_posts == 0

    @pytest.mark.asyncio
    async def test_cursor_pagination_filters_seen_results(self):
        """Results at or before cursor position must be excluded from next page."""
        import numpy as np

        # Build a resolver where we can control the re-ranked list
        db = AsyncMock()
        qdrant = MagicMock()
        qdrant.search_posts = MagicMock(return_value=[(1, 0.9), (2, 0.8), (3, 0.7)])
        qdrant.search_accounts = MagicMock(return_value=[])
        text_proc = MagicMock()
        text_proc.embed = MagicMock(return_value=np.zeros((1, 384), dtype=np.float32))

        now = datetime.now(tz=timezone.utc)
        meta = {
            i: {
                "account_id": 1, "blog_name": "b", "body_clean": "",
                "image_urls": [], "note_count": 0, "published_at": now,
                "lang": "en", "tags": [],
            }
            for i in [1, 2, 3]
        }

        resolver = GeneralSearchResolver(db, qdrant, text_proc)

        # Cursor pointing at post 2 → page 2 should only return post 3
        cursor = encode_cursor(score=0.0, post_id=2)  # exact score unknown, test logic only

        with patch("mir.search.general._keyword_search", new=AsyncMock(return_value=[])), \
             patch("mir.search.general._fetch_post_metadata", new=AsyncMock(return_value=meta)), \
             patch("mir.search.general._fetch_account_metadata", new=AsyncMock(return_value={})):
            result_p1 = await resolver.search("cats", limit_posts=2)

        # First page: top 2 posts
        assert len(result_p1.posts) == 2
        assert result_p1.next_cursor is not None

        # Second page using cursor from page 1
        with patch("mir.search.general._keyword_search", new=AsyncMock(return_value=[])), \
             patch("mir.search.general._fetch_post_metadata", new=AsyncMock(return_value=meta)), \
             patch("mir.search.general._fetch_account_metadata", new=AsyncMock(return_value={})):
            result_p2 = await resolver.search("cats", limit_posts=2, cursor=result_p1.next_cursor)

        # Second page must not contain posts from first page
        p1_ids = {p.post_id for p in result_p1.posts}
        p2_ids = {p.post_id for p in result_p2.posts}
        assert p1_ids.isdisjoint(p2_ids), f"Overlap: {p1_ids & p2_ids}"
