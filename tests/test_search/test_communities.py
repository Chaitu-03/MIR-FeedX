"""
Unit tests for mir/search/communities.py and mir/processing/communities.py.

Tests cover:
  - community_score formula (known inputs → expected outputs)
  - build_drill_down_query generation (URL-encoding, join logic)
  - search_communities scoring and ordering (mocked DB)
  - _choose_k_kmeans with synthetic embeddings (3 tight clusters → 3 communities)
  - Clustering integration: 3 tight clusters of 10 points each
"""
from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from mir.search.communities import (
    CommunityResult,
    build_drill_down_query,
    community_score,
    search_communities,
)
from mir.processing.communities import _choose_k_kmeans


# ── community_score formula ───────────────────────────────────────────────────

class TestCommunityScore:
    def test_zero_members(self):
        # log2(1+0) = 0 → boost = 1.0 → score = cos_sim
        s = community_score(0.8, 0)
        assert abs(s - 0.8) < 1e-9

    def test_member_boost_increases_score(self):
        s0 = community_score(0.5, 0)
        s10 = community_score(0.5, 10)
        assert s10 > s0

    def test_formula_exact(self):
        cos_sim = 0.6
        member_count = 15
        expected = cos_sim * (1 + 0.1 * math.log2(1 + member_count))
        assert abs(community_score(cos_sim, member_count) - expected) < 1e-9

    def test_negative_cosine(self):
        # Negative cosine should produce negative score
        s = community_score(-0.3, 10)
        assert s < 0


# ── build_drill_down_query ─────────────────────────────────────────────────────

class TestBuildDrillDownQuery:
    def test_three_tags_joined_with_plus(self):
        url = build_drill_down_query(["photography", "landscape", "nature"])
        assert url == "/api/v1/search/general?q=photography+landscape+nature"

    def test_only_first_three_tags_used(self):
        url = build_drill_down_query(["a", "b", "c", "d", "e"])
        # Check only the query-param portion — the URL path itself contains letters
        q_part = url.split("?q=")[1]
        assert "d" not in q_part
        assert "e" not in q_part

    def test_spaces_in_tags_url_encoded(self):
        url = build_drill_down_query(["golden hour", "blue sky"])
        assert " " not in url
        assert "golden+hour" in url

    def test_single_tag(self):
        url = build_drill_down_query(["photography"])
        assert url == "/api/v1/search/general?q=photography"

    def test_empty_tags_uses_fallback(self):
        url = build_drill_down_query([], fallback_query="sunset vibes")
        assert "sunset" in url

    def test_empty_tags_empty_fallback(self):
        url = build_drill_down_query([], fallback_query="")
        assert url == "/api/v1/search/general?q="

    def test_special_chars_encoded(self):
        url = build_drill_down_query(["cats & dogs"])
        assert "&" not in url.split("?q=")[1]


# ── search_communities (mocked DB) ────────────────────────────────────────────

class TestSearchCommunities:
    def _centroid(self, seed: int) -> list[float]:
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(384).astype(np.float32)
        v /= np.linalg.norm(v)
        return v.tolist()

    def _make_db(self, community_rows, tag_rows=None):
        db = AsyncMock()
        call_count = [0]
        tag_rows = tag_rows or []

        results_seq = [community_rows, tag_rows]

        async def side_effect(sql, params=None):
            idx = min(call_count[0], len(results_seq) - 1)
            call_count[0] += 1
            r = MagicMock()
            r.fetchall.return_value = results_seq[idx]
            return r

        db.execute = side_effect
        return db

    def _make_tp(self, vec=None):
        tp = MagicMock()
        if vec is None:
            vec = np.zeros(384, dtype=np.float32)
        tp.embed = MagicMock(return_value=np.array([vec]))
        return tp

    def _community_row(self, id, name, type_, member_ids, centroid):
        r = MagicMock()
        r.id = id
        r.name = name
        r.type = type_
        r.member_ids = member_ids
        r.centroid = centroid
        return r

    @pytest.mark.asyncio
    async def test_empty_communities_returns_empty(self):
        db = self._make_db(community_rows=[])
        tp = self._make_tp()
        results = await search_communities(db, tp, "photography")
        assert results == []

    @pytest.mark.asyncio
    async def test_higher_cosine_community_ranks_first(self):
        # q_vec aligned with centroid_1 → community 1 should rank higher
        q = np.zeros(384, dtype=np.float32)
        q[0] = 1.0  # unit vector along dim 0

        c1 = np.zeros(384, dtype=np.float32)
        c1[0] = 1.0  # same direction → cos=1.0

        c2 = np.zeros(384, dtype=np.float32)
        c2[1] = 1.0  # orthogonal → cos=0.0

        rows = [
            self._community_row(1, "photo", "tag_cluster", [1, 2, 3], c1.tolist()),
            self._community_row(2, "music", "tag_cluster", [4, 5], c2.tolist()),
        ]

        tag_mock = MagicMock()
        tag_mock.name = "photography"
        tag_rows = [tag_mock]

        db = self._make_db(rows, tag_rows)
        tp = self._make_tp(vec=q)

        results = await search_communities(db, tp, "photography", limit=5)
        assert results[0].community_id == 1
        assert results[1].community_id == 2

    @pytest.mark.asyncio
    async def test_member_count_boosts_score(self):
        """Two communities with same cosine; more members → higher score."""
        q = np.zeros(384, dtype=np.float32)
        q[0] = 1.0

        c = np.zeros(384, dtype=np.float32)
        c[0] = 1.0

        rows = [
            self._community_row(1, "small", "tag_cluster", [1], c.tolist()),
            self._community_row(2, "large", "tag_cluster", list(range(1, 101)), c.tolist()),
        ]

        db = self._make_db(rows, [])
        tp = self._make_tp(vec=q)

        results = await search_communities(db, tp, "q")
        assert results[0].community_id == 2  # 100 members > 1 member

    @pytest.mark.asyncio
    async def test_drill_down_query_present(self):
        q = np.ones(384, dtype=np.float32)
        q /= np.linalg.norm(q)
        c = q.copy()

        rows = [self._community_row(1, "landscape", "tag_cluster", [10, 11], c.tolist())]
        tag_r = MagicMock()
        tag_r.name = "landscape"
        db = self._make_db(rows, [tag_r])
        tp = self._make_tp(vec=q)

        results = await search_communities(db, tp, "landscape")
        assert results[0].drill_down_query.startswith("/api/v1/search/general?q=")

    @pytest.mark.asyncio
    async def test_limit_respected(self):
        q = np.ones(384, dtype=np.float32)
        q /= np.linalg.norm(q)
        c = q.copy()

        rows = [
            self._community_row(i, f"c{i}", "tag_cluster", [i], c.tolist())
            for i in range(1, 11)
        ]
        db = self._make_db(rows, [])
        tp = self._make_tp(vec=q)

        results = await search_communities(db, tp, "test", limit=3)
        assert len(results) == 3


# ── _choose_k_kmeans with synthetic tight clusters ────────────────────────────

class TestChooseKKMeans:
    def _make_tight_clusters(self, n_clusters=3, points_per_cluster=10, dim=16, seed=0):
        """Generate n_clusters tight clusters of points_per_cluster points each."""
        rng = np.random.default_rng(seed)
        X_list = []
        for i in range(n_clusters):
            center = rng.standard_normal(dim).astype(np.float32)
            center /= np.linalg.norm(center)
            # Small noise around each center
            noise = rng.standard_normal((points_per_cluster, dim)).astype(np.float32) * 0.05
            cluster_pts = center + noise
            X_list.append(cluster_pts)
        return np.vstack(X_list)

    def test_three_tight_clusters_detected(self):
        X = self._make_tight_clusters(n_clusters=3, points_per_cluster=10)
        # k candidates must include 3
        best_k, best_score, labels = _choose_k_kmeans(X, k_candidates=[2, 3, 5])
        assert best_k == 3, f"Expected k=3 for 3 tight clusters, got k={best_k}"

    def test_silhouette_positive_for_tight_clusters(self):
        X = self._make_tight_clusters(n_clusters=3, points_per_cluster=15)
        _, best_score, _ = _choose_k_kmeans(X, k_candidates=[2, 3, 5])
        assert best_score > 0.5, f"Silhouette should be high for tight clusters, got {best_score:.3f}"

    def test_labels_cover_all_points(self):
        n = 30
        X = self._make_tight_clusters(n_clusters=3, points_per_cluster=10)
        _, _, labels = _choose_k_kmeans(X, k_candidates=[2, 3, 5])
        assert len(labels) == n

    def test_single_candidate_k_still_works(self):
        X = self._make_tight_clusters(n_clusters=2, points_per_cluster=10)
        best_k, _, labels = _choose_k_kmeans(X, k_candidates=[2])
        assert best_k == 2
