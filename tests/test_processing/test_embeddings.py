"""
Unit tests for mir/processing/embeddings.py.

Unit tests (fast, no I/O, no ML weights):
  - build_post_embedding: all 8 None/non-None combinations
  - build_post_embedding: L2-norm of output ≈ 1.0
  - build_post_embedding: explicit weights override settings
  - build_post_embedding: all-None returns zero vector (no crash)
  - update_account_embedding: mock 150 points → only last 100 used
  - update_account_embedding: 0 points → returns new_post_embedding unchanged
  - update_account_embedding: < window points (cold start) → mean of all
  - process_and_index_post: post not found → status "not_found"
  - process_and_index_post: nsfw=True → Qdrant upsert NOT called
  - process_and_index_post: threshold met → Qdrant upsert NOT called
  - process_and_index_post: happy path → Qdrant upsert called, status "ok"
"""
from __future__ import annotations

import itertools
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from mir.processing.embeddings import (
    ACCOUNTS_COLLECTION,
    POSTS_COLLECTION,
    PostIndexer,
    build_post_embedding,
)

_DIM = 384


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(seed: int = 0) -> np.ndarray:
    """Return a deterministic 384-d unit vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


class _MockPoint:
    """Minimal stand-in for qdrant_client Record / ScoredPoint."""

    def __init__(self, vector: np.ndarray) -> None:
        self.vector = vector.tolist()


def _make_mock_post(
    post_id: int = 1,
    account_id: int = 10,
    nsfw: bool = False,
    tumblr_id: int = 99,
    body_raw: str = "Hello world",
    lang: str = "en",
    published_at=None,
) -> MagicMock:
    post = MagicMock()
    post.id = post_id
    post.account_id = account_id
    post.nsfw = nsfw
    post.tumblr_id = tumblr_id
    post.body_raw = body_raw
    post.lang = lang
    post.note_count = 5
    post.published_at = published_at
    post.account = MagicMock()
    post.account.blog_name = "testblog"
    return post


def _make_session(
    post: Any | None,
    sfw_count: int = 0,
    tags: list | None = None,
) -> AsyncMock:
    """
    Build a mock AsyncSession whose execute() returns appropriate results for:
      1st call  → scalar_one_or_none() = post
      2nd call  → scalar_one() = sfw_count
      3rd call  → scalars().all() = tags
      4th call  → scalar_one_or_none() = account (for update_account_embedding)
    """
    tags = tags or []

    # Result objects
    post_result = MagicMock()
    post_result.scalar_one_or_none.return_value = post

    count_result = MagicMock()
    count_result.scalar_one.return_value = sfw_count

    tags_result = MagicMock()
    tags_result.scalars.return_value.all.return_value = tags

    account_result = MagicMock()
    account_result.scalar_one_or_none.return_value = MagicMock()  # dummy Account

    session = AsyncMock()
    session.execute.side_effect = [
        post_result,
        count_result,
        tags_result,
        account_result,
    ]
    session.commit = AsyncMock()
    return session


def _make_qdrant(scroll_points: list | None = None) -> AsyncMock:
    """AsyncQdrantClient mock with scroll returning (points, None)."""
    scroll_points = scroll_points or []
    qdrant = AsyncMock()
    qdrant.scroll.return_value = (scroll_points, None)
    qdrant.upsert = AsyncMock()
    return qdrant


def _make_text_proc(
    text_emb: np.ndarray | None = None,
    tag_emb: np.ndarray | None = None,
) -> MagicMock:
    tp = MagicMock()
    tp.process_post.return_value = {
        "text_embedding": text_emb if text_emb is not None else _unit(10),
        "tag_embedding": tag_emb if tag_emb is not None else np.zeros(_DIM, dtype=np.float32),
    }
    tp.embed.return_value = np.stack([_unit(99)])  # (1, 384) for tag embed
    return tp


def _make_image_proc(projected: np.ndarray | None = None) -> MagicMock:
    ip = MagicMock()
    # embed_images_batch returns (N, 512); project returns (384,)
    ip.embed_images_batch.return_value = np.zeros((0, 512), dtype=np.float32)
    ip.project.return_value = projected if projected is not None else _unit(20)
    return ip


# ---------------------------------------------------------------------------
# build_post_embedding — unit tests
# ---------------------------------------------------------------------------

class TestBuildPostEmbedding:
    """Test all 8 combinations of None / non-None inputs."""

    @pytest.mark.parametrize(
        "has_text,has_image,has_tag",
        list(itertools.product([True, False], repeat=3)),
        ids=lambda b: "Y" if b else "N",
    )
    def test_output_shape(self, has_text, has_image, has_tag):
        text  = _unit(0) if has_text  else None
        image = _unit(1) if has_image else None
        tag   = _unit(2) if has_tag   else None
        out = build_post_embedding(text, image, tag)
        assert out.shape == (_DIM,), f"Expected ({_DIM},), got {out.shape}"

    @pytest.mark.parametrize(
        "has_text,has_image,has_tag",
        [combo for combo in itertools.product([True, False], repeat=3) if any(combo)],
    )
    def test_output_is_unit_norm(self, has_text, has_image, has_tag):
        """Any non-trivial combination must produce a unit vector."""
        text  = _unit(0) if has_text  else None
        image = _unit(1) if has_image else None
        tag   = _unit(2) if has_tag   else None
        out = build_post_embedding(text, image, tag)
        norm = float(np.linalg.norm(out))
        assert abs(norm - 1.0) < 1e-5, (
            f"has_text={has_text} has_image={has_image} has_tag={has_tag}: "
            f"norm={norm:.6f}"
        )

    def test_all_none_returns_zero_vector(self):
        """All-None → zero vector (no information to normalise)."""
        out = build_post_embedding(None, None, None)
        assert out.shape == (_DIM,)
        np.testing.assert_array_equal(out, np.zeros(_DIM, dtype=np.float32))

    def test_output_dtype_float32(self):
        out = build_post_embedding(_unit(0), _unit(1), _unit(2))
        assert out.dtype == np.float32

    def test_explicit_weights_respected(self):
        """Supplying explicit weights overrides settings."""
        # With weights=(1.0, 0.0, 0.0) result should equal the text input
        text = _unit(7)
        out = build_post_embedding(text, _unit(8), _unit(9), weights=(1.0, 0.0, 0.0))
        # w_text=1 → combined = text; norm(text)=1 → out == text
        np.testing.assert_allclose(out, text, atol=1e-5)

    def test_only_image_with_unit_weight(self):
        """weights=(0,1,0) → output equals image_emb."""
        image = _unit(5)
        out = build_post_embedding(None, image, None, weights=(0.0, 1.0, 0.0))
        np.testing.assert_allclose(out, image, atol=1e-5)

    def test_different_inputs_different_outputs(self):
        out_a = build_post_embedding(_unit(1), None, None)
        out_b = build_post_embedding(_unit(2), None, None)
        assert not np.allclose(out_a, out_b)

    def test_none_equivalent_to_zeros(self):
        """Passing None should produce same result as passing np.zeros(384)."""
        zeros = np.zeros(_DIM, dtype=np.float32)
        text = _unit(3)
        out_none  = build_post_embedding(text, None, None)
        out_zeros = build_post_embedding(text, zeros, zeros)
        np.testing.assert_allclose(out_none, out_zeros, atol=1e-5)


# ---------------------------------------------------------------------------
# update_account_embedding — unit tests
# ---------------------------------------------------------------------------

class TestUpdateAccountEmbedding:
    """Sliding window: verify only the most recent WINDOW points are used."""

    @pytest.mark.asyncio
    async def test_uses_only_last_window_points(self):
        """
        Qdrant returns 150 mock points; function must use only the first 100
        (order=desc so first 100 = most recent 100).
        """
        from mir.config import settings

        window = settings.account_embedding_window  # 100

        # Create 150 distinct unit vectors
        rng = np.random.default_rng(42)
        all_vecs = rng.standard_normal((150, _DIM)).astype(np.float32)
        all_vecs /= np.linalg.norm(all_vecs, axis=1, keepdims=True)

        mock_points = [_MockPoint(v) for v in all_vecs]
        qdrant = _make_qdrant(scroll_points=mock_points)

        session = AsyncMock()
        account_result = MagicMock()
        account_result.scalar_one_or_none.return_value = MagicMock()
        session.execute = AsyncMock(return_value=account_result)
        session.commit = AsyncMock()

        indexer = PostIndexer(
            qdrant=qdrant,
            session=session,
            text_proc=MagicMock(),
            image_proc=MagicMock(),
        )

        new_emb = _unit(99)
        result_emb = await indexer.update_account_embedding(1, new_emb)

        # Expected: mean of only the first 100 vectors (most recent), normalised
        expected_mean = all_vecs[:window].mean(axis=0)
        expected_norm = np.linalg.norm(expected_mean)
        expected = (expected_mean / expected_norm).astype(np.float32)

        np.testing.assert_allclose(result_emb, expected, atol=1e-5)

    @pytest.mark.asyncio
    async def test_cold_start_no_points_returns_new_embedding(self):
        """No posts indexed yet → return new_post_embedding unchanged."""
        qdrant = _make_qdrant(scroll_points=[])
        session = AsyncMock()
        indexer = PostIndexer(qdrant=qdrant, session=session, text_proc=MagicMock(), image_proc=MagicMock())

        new_emb = _unit(55)
        result = await indexer.update_account_embedding(42, new_emb)
        np.testing.assert_allclose(result, new_emb, atol=1e-6)
        # Qdrant upsert should NOT have been called (no update to make)
        qdrant.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_cold_start_few_posts_uses_all(self):
        """< window posts: use mean of all available (not just window)."""
        from mir.config import settings

        n_posts = 5  # well below window=100
        rng = np.random.default_rng(7)
        vecs = rng.standard_normal((n_posts, _DIM)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

        mock_points = [_MockPoint(v) for v in vecs]
        qdrant = _make_qdrant(scroll_points=mock_points)

        session = AsyncMock()
        account_result = MagicMock()
        account_result.scalar_one_or_none.return_value = MagicMock()
        session.execute = AsyncMock(return_value=account_result)
        session.commit = AsyncMock()

        indexer = PostIndexer(qdrant=qdrant, session=session, text_proc=MagicMock(), image_proc=MagicMock())
        result = await indexer.update_account_embedding(1, _unit(0))

        expected_mean = vecs.mean(axis=0)
        expected = (expected_mean / np.linalg.norm(expected_mean)).astype(np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    @pytest.mark.asyncio
    async def test_result_is_unit_norm(self):
        """Output of update_account_embedding must be L2-normalised."""
        rng = np.random.default_rng(13)
        vecs = rng.standard_normal((10, _DIM)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

        qdrant = _make_qdrant(scroll_points=[_MockPoint(v) for v in vecs])
        session = AsyncMock()
        account_result = MagicMock()
        account_result.scalar_one_or_none.return_value = MagicMock()
        session.execute = AsyncMock(return_value=account_result)
        session.commit = AsyncMock()

        indexer = PostIndexer(qdrant=qdrant, session=session, text_proc=MagicMock(), image_proc=MagicMock())
        result = await indexer.update_account_embedding(1, _unit(0))
        norm = float(np.linalg.norm(result))
        assert abs(norm - 1.0) < 1e-5

    @pytest.mark.asyncio
    async def test_qdrant_scroll_called_with_correct_account_id(self):
        """Scroll filter must reference the correct account_id."""
        qdrant = _make_qdrant(scroll_points=[])
        session = AsyncMock()
        indexer = PostIndexer(qdrant=qdrant, session=session, text_proc=MagicMock(), image_proc=MagicMock())

        await indexer.update_account_embedding(account_id=77, new_post_embedding=_unit(0))

        call_kwargs = qdrant.scroll.call_args[1] if qdrant.scroll.call_args.kwargs else {}
        call_args = qdrant.scroll.call_args
        # Collection name must be posts
        assert POSTS_COLLECTION in str(call_args)
        # The filter must contain account_id=77
        scroll_filter = call_args.kwargs.get("scroll_filter") or call_args.args[1] if len(call_args.args) > 1 else None
        # Check via string repr — avoids importing qdrant_client model classes in tests
        assert "77" in str(call_args)


# ---------------------------------------------------------------------------
# process_and_index_post — unit tests
# ---------------------------------------------------------------------------

class TestProcessAndIndexPost:

    @pytest.mark.asyncio
    async def test_post_not_found_returns_not_found(self):
        session = _make_session(post=None)
        qdrant = _make_qdrant()
        indexer = PostIndexer(qdrant=qdrant, session=session, text_proc=MagicMock(), image_proc=MagicMock())
        result = await indexer.process_and_index_post(post_id=999)
        assert result["status"] == "not_found"
        qdrant.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_nsfw_post_skips_indexing(self):
        """SAFETY GATE: nsfw=True → Qdrant upsert must NOT be called."""
        post = _make_mock_post(nsfw=True)

        post_result = MagicMock()
        post_result.scalar_one_or_none.return_value = post

        session = AsyncMock()
        session.execute = AsyncMock(return_value=post_result)
        session.commit = AsyncMock()

        qdrant = _make_qdrant()
        indexer = PostIndexer(qdrant=qdrant, session=session, text_proc=MagicMock(), image_proc=MagicMock())

        result = await indexer.process_and_index_post(post_id=1)

        assert result["status"] == "nsfw_skip"
        qdrant.upsert.assert_not_called()
        # execute called exactly once (only the post load — no COUNT query)
        assert session.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_threshold_met_skips_indexing(self):
        """
        THRESHOLD GATE: if SFW post count >= TARGET_POST_COUNT,
        Qdrant upsert must NOT be called.
        """
        from mir.config import settings

        post = _make_mock_post(nsfw=False)

        # Hit exactly at threshold
        session = _make_session(post=post, sfw_count=settings.target_post_count)
        qdrant = _make_qdrant()
        indexer = PostIndexer(qdrant=qdrant, session=session, text_proc=MagicMock(), image_proc=MagicMock())

        result = await indexer.process_and_index_post(post_id=1)

        assert result["status"] == "threshold_skip"
        qdrant.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_threshold_above_still_skips(self):
        """Threshold gate fires when count > TARGET_POST_COUNT too."""
        from mir.config import settings

        post = _make_mock_post(nsfw=False)
        session = _make_session(post=post, sfw_count=settings.target_post_count + 1000)
        qdrant = _make_qdrant()
        indexer = PostIndexer(qdrant=qdrant, session=session, text_proc=MagicMock(), image_proc=MagicMock())
        result = await indexer.process_and_index_post(post_id=1)
        assert result["status"] == "threshold_skip"
        qdrant.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_happy_path_calls_qdrant_upsert(self, tmp_path):
        """
        SFW post below threshold with no local images →
        Qdrant upsert is called for the posts collection.
        """
        from mir.config import settings

        post = _make_mock_post(nsfw=False)
        post.account.blog_name = "testblog"
        # Point image_dir to tmp_path — no images there → image_emb=None
        post.tumblr_id = 12345

        # Patch the image directory so glob returns nothing
        with patch("mir.processing.embeddings.Path") as mock_path_cls:
            mock_image_dir = MagicMock()
            mock_image_dir.glob.return_value = []
            mock_path_cls.return_value.__truediv__.return_value.__truediv__.return_value = mock_image_dir

            session = _make_session(
                post=post,
                sfw_count=settings.target_post_count - 1,  # one below threshold
                tags=[],
            )
            qdrant = _make_qdrant(scroll_points=[])

            text_proc = _make_text_proc()
            image_proc = _make_image_proc()

            indexer = PostIndexer(
                qdrant=qdrant,
                session=session,
                text_proc=text_proc,
                image_proc=image_proc,
            )

            result = await indexer.process_and_index_post(post_id=post.id)

        assert result["status"] == "ok"
        assert result["post_id"] == post.id
        # Qdrant upsert called at least once (posts collection)
        assert qdrant.upsert.call_count >= 1
        first_upsert_call = qdrant.upsert.call_args_list[0]
        assert POSTS_COLLECTION in str(first_upsert_call)

    @pytest.mark.asyncio
    async def test_nsfw_gate_takes_priority_over_threshold(self):
        """NSFW gate fires before the threshold query — COUNT must NOT run."""
        post = _make_mock_post(nsfw=True)

        post_result = MagicMock()
        post_result.scalar_one_or_none.return_value = post

        session = AsyncMock()
        session.execute = AsyncMock(return_value=post_result)

        qdrant = _make_qdrant()
        indexer = PostIndexer(qdrant=qdrant, session=session, text_proc=MagicMock(), image_proc=MagicMock())
        result = await indexer.process_and_index_post(post_id=1)

        assert result["status"] == "nsfw_skip"
        # Only one execute call (post load); no COUNT(*)
        assert session.execute.call_count == 1
        qdrant.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_account_embedding_updated_on_happy_path(self, tmp_path):
        """
        On a successful index, update_account_embedding is called
        (verified by Qdrant upsert on the accounts collection).
        """
        from mir.config import settings

        post = _make_mock_post(nsfw=False)

        with patch("mir.processing.embeddings.Path") as mock_path_cls:
            mock_image_dir = MagicMock()
            mock_image_dir.glob.return_value = []
            mock_path_cls.return_value.__truediv__.return_value.__truediv__.return_value = mock_image_dir

            session = _make_session(
                post=post,
                sfw_count=0,
                tags=[],
            )

            # Scroll returns one post vector for the account window
            scroll_vecs = [_MockPoint(_unit(i)) for i in range(5)]
            qdrant = _make_qdrant(scroll_points=scroll_vecs)
            # Extra execute call for update_account_embedding → account lookup
            account_result = MagicMock()
            account_result.scalar_one_or_none.return_value = MagicMock()
            session.execute.side_effect = list(session.execute.side_effect) + [account_result]

            indexer = PostIndexer(
                qdrant=qdrant,
                session=session,
                text_proc=_make_text_proc(),
                image_proc=_make_image_proc(),
            )
            result = await indexer.process_and_index_post(post_id=post.id)

        assert result["status"] == "ok"
        # Accounts upsert must have been called
        upsert_collections = [str(c) for c in qdrant.upsert.call_args_list]
        assert any(ACCOUNTS_COLLECTION in c for c in upsert_collections)
