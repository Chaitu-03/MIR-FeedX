"""
Integration tests for mir/search/vector_store.py.

Requires Docker (testcontainers spins up a real Qdrant container).
Marked @pytest.mark.slow — skip with: pytest -m "not slow"

Tests:
  - Collection creation is idempotent (calling ensure_collections twice is safe)
  - upsert_post + search_posts with exclude_nsfw=True never returns NSFW points
  - upsert_batch roundtrip — all points retrievable, scores in valid range
  - delete_post removes the point from search results
  - search with filters: min_notes, date_from, lang
  - search_accounts and search_tags basic roundtrip
  - get_collection_info returns a dict with expected keys
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pytest
from testcontainers.qdrant import QdrantContainer

from mir.search.vector_store import (
    ACCOUNTS_COLLECTION,
    POSTS_COLLECTION,
    TAGS_COLLECTION,
    QdrantManager,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 384


def _rand_vec(seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(_DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def _post_payload(
    account_id: int = 1,
    nsfw: bool = False,
    note_count: int = 100,
    lang: str = "en",
    published_at: float | None = None,
) -> dict:
    return {
        "post_id": 0,  # overwritten by caller's id arg
        "account_id": account_id,
        "nsfw": nsfw,
        "note_count": note_count,
        "lang": lang,
        "published_at": published_at or datetime(2025, 6, 1, tzinfo=timezone.utc).timestamp(),
        "tags": [],
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qdrant_container():
    """Start a real Qdrant container for the entire test module."""
    with QdrantContainer() as container:
        # Brief wait for Qdrant to be ready (healthcheck inside container handles it,
        # but give the client a moment after the port is open)
        time.sleep(1)
        yield container


@pytest.fixture(scope="module")
def mgr(qdrant_container):
    """QdrantManager pointed at the test container via HTTP (not gRPC)."""
    host = qdrant_container.get_container_host_ip()
    http_port = int(qdrant_container.get_exposed_port(6333))
    manager = QdrantManager(host=host, port=http_port, prefer_grpc=False)
    manager.ensure_collections()
    yield manager
    manager.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestEnsureCollections:
    def test_all_collections_created(self, mgr):
        """Three collections must exist after ensure_collections."""
        info = mgr._client.get_collections()
        names = {c.name for c in info.collections}
        assert POSTS_COLLECTION in names
        assert ACCOUNTS_COLLECTION in names
        assert TAGS_COLLECTION in names

    def test_idempotent_double_call(self, mgr):
        """Calling ensure_collections a second time must not raise."""
        mgr.ensure_collections()  # second call — should be a no-op

    def test_collection_info_keys(self, mgr):
        """get_collection_info returns dict with expected keys."""
        info = mgr.get_collection_info(POSTS_COLLECTION)
        assert "name" in info
        assert "points_count" in info
        assert "status" in info


@pytest.mark.slow
class TestNSFWFiltering:
    """Safety baseline: exclude_nsfw=True must never surface NSFW posts."""

    def test_nsfw_post_excluded_from_default_search(self, mgr):
        """Insert one SFW and one NSFW post; search returns only SFW."""
        sfw_id = 9001
        nsfw_id = 9002
        query = _rand_vec(seed=42)

        # Use near-identical vectors so both would score equally without filter
        vec = _rand_vec(seed=7)

        mgr.upsert_post(sfw_id, vec, {**_post_payload(nsfw=False), "post_id": sfw_id})
        mgr.upsert_post(nsfw_id, vec, {**_post_payload(nsfw=True), "post_id": nsfw_id})

        results = mgr.search_posts(query, limit=50, exclude_nsfw=True)
        returned_ids = {pid for pid, _ in results}

        assert nsfw_id not in returned_ids, "NSFW post must never appear in safe search"

    def test_nsfw_visible_when_filter_disabled(self, mgr):
        """When exclude_nsfw=False, NSFW posts CAN appear in results."""
        nsfw_id = 9002  # inserted above
        query = _rand_vec(seed=42)

        results = mgr.search_posts(query, limit=50, exclude_nsfw=False)
        returned_ids = {pid for pid, _ in results}

        # The NSFW post has the same vector as the query, so it should appear
        assert nsfw_id in returned_ids, "NSFW post should appear when filter is disabled"

    def test_upsert_without_nsfw_field_raises(self, mgr):
        """upsert_post must enforce that payload contains 'nsfw'."""
        with pytest.raises(ValueError, match="nsfw"):
            mgr.upsert_post(99999, _rand_vec(), {"account_id": 1, "note_count": 5})


@pytest.mark.slow
class TestBatchUpsertRoundtrip:
    """Batch upsert: all points stored, all retrievable via search."""

    def test_batch_upsert_and_retrieve(self, mgr):
        from qdrant_client.models import PointStruct

        base_id = 8000
        n = 20

        # All points have the same vector direction → all should appear in search
        shared_vec = _rand_vec(seed=100)

        points = [
            PointStruct(
                id=base_id + i,
                vector=shared_vec,
                payload={**_post_payload(nsfw=False, note_count=i * 10), "post_id": base_id + i},
            )
            for i in range(n)
        ]
        mgr.upsert_batch(POSTS_COLLECTION, points)

        results = mgr.search_posts(shared_vec, limit=n + 10, exclude_nsfw=True)
        returned_ids = {pid for pid, _ in results}

        inserted_ids = {base_id + i for i in range(n)}
        assert inserted_ids.issubset(returned_ids), (
            f"Missing from results: {inserted_ids - returned_ids}"
        )

    def test_scores_in_valid_range(self, mgr):
        """All cosine scores must be in [-1, 1]."""
        results = mgr.search_posts(_rand_vec(seed=55), limit=50, exclude_nsfw=False)
        for _, score in results:
            assert -1.0 <= score <= 1.0, f"Invalid score: {score}"


@pytest.mark.slow
class TestSearchFilters:
    """Additional filter conditions: min_notes, date_from, lang."""

    def _setup_filter_posts(self, mgr) -> None:
        """Insert posts with varying attributes for filter tests."""
        from qdrant_client.models import PointStruct

        ts_2024 = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
        ts_2025 = datetime(2025, 6, 1, tzinfo=timezone.utc).timestamp()

        posts = [
            PointStruct(
                id=7001,
                vector=_rand_vec(seed=1),
                payload={
                    "post_id": 7001, "account_id": 1, "nsfw": False,
                    "note_count": 5, "lang": "en", "published_at": ts_2024, "tags": [],
                },
            ),
            PointStruct(
                id=7002,
                vector=_rand_vec(seed=2),
                payload={
                    "post_id": 7002, "account_id": 1, "nsfw": False,
                    "note_count": 500, "lang": "en", "published_at": ts_2025, "tags": [],
                },
            ),
            PointStruct(
                id=7003,
                vector=_rand_vec(seed=3),
                payload={
                    "post_id": 7003, "account_id": 2, "nsfw": False,
                    "note_count": 200, "lang": "de", "published_at": ts_2025, "tags": [],
                },
            ),
        ]
        mgr.upsert_batch(POSTS_COLLECTION, posts)

    def test_filter_min_notes(self, mgr):
        self._setup_filter_posts(mgr)
        results = mgr.search_posts(_rand_vec(seed=99), limit=50, filters={"min_notes": 100})
        returned_ids = {pid for pid, _ in results}
        # post 7001 has note_count=5, must be excluded
        assert 7001 not in returned_ids
        assert 7002 in returned_ids
        assert 7003 in returned_ids

    def test_filter_date_from(self, mgr):
        results = mgr.search_posts(
            _rand_vec(seed=99),
            limit=50,
            filters={"date_from": "2025-01-01"},
        )
        returned_ids = {pid for pid, _ in results}
        # post 7001 is from 2024, must be excluded
        assert 7001 not in returned_ids

    def test_filter_lang(self, mgr):
        results = mgr.search_posts(_rand_vec(seed=99), limit=50, filters={"lang": "de"})
        returned_ids = {pid for pid, _ in results}
        assert 7003 in returned_ids
        assert 7001 not in returned_ids
        assert 7002 not in returned_ids

    def test_combined_filters(self, mgr):
        results = mgr.search_posts(
            _rand_vec(seed=99),
            limit=50,
            filters={"min_notes": 100, "lang": "en"},
        )
        returned_ids = {pid for pid, _ in results}
        # Only post 7002: note_count=500, lang=en
        assert 7002 in returned_ids
        assert 7001 not in returned_ids  # too few notes
        assert 7003 not in returned_ids  # wrong lang


@pytest.mark.slow
class TestDeletePost:
    def test_delete_removes_from_search(self, mgr):
        post_id = 6001
        vec = _rand_vec(seed=200)
        mgr.upsert_post(post_id, vec, {**_post_payload(nsfw=False), "post_id": post_id})

        # Confirm it's there
        before = {pid for pid, _ in mgr.search_posts(vec, limit=50, exclude_nsfw=False)}
        assert post_id in before

        mgr.delete_post(post_id)

        after = {pid for pid, _ in mgr.search_posts(vec, limit=50, exclude_nsfw=False)}
        assert post_id not in after


@pytest.mark.slow
class TestAccountsAndTags:
    def test_accounts_roundtrip(self, mgr):
        acct_id = 5001
        vec = _rand_vec(seed=300)
        mgr.upsert_account(acct_id, vec, {"account_id": acct_id, "blog_name": "testblog"})

        results = mgr.search_accounts(vec, limit=10)
        ids = [aid for aid, _ in results]
        assert acct_id in ids

    def test_tags_roundtrip(self, mgr):
        tag_id = 4001
        vec = _rand_vec(seed=400)
        mgr.upsert_tag(tag_id, vec, {"tag_id": tag_id, "usage_count": 42})

        results = mgr.search_tags(vec, limit=10)
        ids = [tid for tid, _ in results]
        assert tag_id in ids
