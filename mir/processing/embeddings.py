"""
Unified post + account embedding and Qdrant indexing.

Public surface
--------------
build_post_embedding(text_emb, image_emb, tag_emb, weights=None) -> np.ndarray
    Pure function — no I/O.  Weighted average → L2-normalised 384-d vector.

PostIndexer
    Stateful orchestrator that owns DB session, Qdrant client, and ML processors.

    .process_and_index_post(post_id)
        Full pipeline: load → safety/threshold gates → embed → Qdrant upsert
        → account aggregation → tag upsert.

    .update_account_embedding(account_id, new_post_embedding)
        Sliding-window mean of the most recent WINDOW post embeddings fetched
        from Qdrant, stored back to both Qdrant and PostgreSQL.

    .ensure_collections()
        Create Qdrant collections (posts / accounts / tags) if absent.

Design notes
------------
- Sliding window (not EMA): EMA with α=0.05 takes ~60 posts to halve the
  influence of the first post.  Window mean reflects the last N posts directly
  and is easy to reason about.
- Weights are read from settings so the eval harness (Prompt 16b) can tune
  them without code changes.
- All vectors are L2-normalised before fusion and after projection.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from mir.config import settings
from mir.db.models import Account, Post, PostTag, Tag

log = logging.getLogger(__name__)

_EMBED_DIM = 384

# Qdrant collection names (centralised so callers stay in sync)
POSTS_COLLECTION = "posts"
ACCOUNTS_COLLECTION = "accounts"
TAGS_COLLECTION = "tags"


# ---------------------------------------------------------------------------
# Pure function — no I/O
# ---------------------------------------------------------------------------

def build_post_embedding(
    text_emb: np.ndarray | None,
    image_emb: np.ndarray | None,
    tag_emb: np.ndarray | None,
    weights: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """
    Weighted average of text / image / tag embeddings → L2-normalised 384-d.

    Parameters
    ----------
    text_emb  : 384-d array or None  (MiniLM, chunk-mean-pooled)
    image_emb : 384-d array or None  (CLIP projected, image-mean-pooled)
    tag_emb   : 384-d array or None  (MiniLM per-tag, then mean-pooled)
    weights   : (w_text, w_image, w_tag).  Defaults to settings values
                (post_weight_text, post_weight_image, post_weight_tag).

    Returns
    -------
    np.ndarray of shape (384,), dtype float32, L2-normalised.
    Returns zero vector only when all inputs are None or zero (no information).
    """
    if weights is None:
        weights = (
            settings.post_weight_text,
            settings.post_weight_image,
            settings.post_weight_tag,
        )
    w_t, w_i, w_g = weights

    zero = np.zeros(_EMBED_DIM, dtype=np.float32)
    e_text = text_emb if text_emb is not None else zero
    e_img  = image_emb if image_emb is not None else zero
    e_tags = tag_emb if tag_emb is not None else zero

    combined = (w_t * e_text + w_i * e_img + w_g * e_tags).astype(np.float32)
    norm = float(np.linalg.norm(combined))
    if norm < 1e-10:
        # All inputs were None/zero — no information to normalise
        return combined
    return (combined / norm).astype(np.float32)


# ---------------------------------------------------------------------------
# Stateful orchestrator
# ---------------------------------------------------------------------------

class PostIndexer:
    """
    Orchestrates embedding computation and Qdrant indexing.

    Parameters
    ----------
    qdrant     : AsyncQdrantClient  (qdrant_client.AsyncQdrantClient)
    session    : AsyncSession        (SQLAlchemy async session)
    text_proc  : TextProcessor       (mir.processing.text.TextProcessor)
    image_proc : ImageProcessor      (mir.processing.images.ImageProcessor)
    """

    def __init__(
        self,
        qdrant: Any,
        session: AsyncSession,
        text_proc: Any,
        image_proc: Any,
    ) -> None:
        self._qdrant = qdrant
        self._session = session
        self._text = text_proc
        self._image = image_proc

    # ------------------------------------------------------------------
    # Collection bootstrap
    # ------------------------------------------------------------------

    async def ensure_collections(self) -> None:
        """Create Qdrant collections (posts / accounts / tags) if absent."""
        from qdrant_client.models import Distance, VectorParams

        existing = {c.name for c in (await self._qdrant.get_collections()).collections}
        for name in (POSTS_COLLECTION, ACCOUNTS_COLLECTION, TAGS_COLLECTION):
            if name not in existing:
                await self._qdrant.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=_EMBED_DIM, distance=Distance.COSINE),
                )
                log.info("Created Qdrant collection '%s'", name)

    # ------------------------------------------------------------------
    # Account embedding — sliding window
    # ------------------------------------------------------------------

    async def update_account_embedding(
        self,
        account_id: int,
        new_post_embedding: np.ndarray,  # kept for API symmetry; post already in Qdrant
    ) -> np.ndarray:
        """
        Recompute account embedding as a sliding-window mean of the last
        ACCOUNT_EMBEDDING_WINDOW post embeddings.

        Design: sliding window over the most recent N posts, NOT EMA.
        EMA (α=0.05) needs ~60 posts to halve the influence of the first post;
        window mean reflects topic drift within a single window length.

        Steps
        -----
        1. Scroll Qdrant posts collection filtered by account_id,
           ordered by published_at desc, limit = window.
        2. Slice to at most `window` results (guard against over-fetching).
        3. Mean-pool → L2-normalise.
        4. Upsert to Qdrant accounts collection.
        5. Update PostgreSQL accounts.embedding.

        Cold start (< window posts): uses mean of all available posts.
        Returns new_post_embedding unchanged when no posts are indexed yet.
        """
        from qdrant_client.models import (
            Direction,
            FieldCondition,
            Filter,
            MatchValue,
            OrderBy,
            PointStruct,
        )

        window = settings.account_embedding_window

        points, _ = await self._qdrant.scroll(
            collection_name=POSTS_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="account_id",
                        match=MatchValue(value=account_id),
                    )
                ]
            ),
            order_by=OrderBy(key="published_at", direction=Direction.DESC),
            limit=window,
            with_vectors=True,
        )

        # Explicit slice — guard in case mock or implementation returns more
        recent = points[:window]

        if not recent:
            log.debug(
                "No indexed posts for account %d — account embedding unchanged",
                account_id,
            )
            return new_post_embedding

        vectors = np.array([p.vector for p in recent], dtype=np.float32)  # (K, 384)
        mean_vec = vectors.mean(axis=0)  # (384,)
        norm = float(np.linalg.norm(mean_vec))
        account_emb = (mean_vec / norm).astype(np.float32) if norm > 1e-10 else mean_vec

        # Persist to Qdrant accounts collection
        await self._qdrant.upsert(
            collection_name=ACCOUNTS_COLLECTION,
            points=[
                PointStruct(
                    id=account_id,
                    vector=account_emb.tolist(),
                    payload={"account_id": account_id},
                )
            ],
        )

        # Persist to PostgreSQL accounts.embedding
        result = await self._session.execute(
            select(Account).where(Account.id == account_id)
        )
        account = result.scalar_one_or_none()
        if account is not None:
            account.embedding = account_emb.tolist()
            await self._session.commit()
        else:
            log.warning("Account %d not found in PostgreSQL during embedding update", account_id)

        return account_emb

    # ------------------------------------------------------------------
    # Main orchestrator
    # ------------------------------------------------------------------

    async def process_and_index_post(self, post_id: int) -> dict:
        """
        Full pipeline: load → safety gate → threshold gate → embed → index.

        Gates
        -----
        1. NSFW gate  : if post.nsfw is True, skip immediately (no Qdrant write).
        2. Threshold gate: if SFW post count >= TARGET_POST_COUNT, skip.
           Works in tandem with the crawler's threshold guard — posts enqueued
           in Celery before the limit was reached but processed after are also
           safely skipped here.

        Pipeline (SFW posts below threshold only)
        -----------------------------------------
        3.  Load tags via PostTag join.
        4.  TextProcessor.process_post → text_embedding (384-d), tag_embedding (384-d).
        5.  ImageProcessor.embed_images_batch → mean-pool → project → image_embedding (384-d).
        6.  build_post_embedding → unified 384-d L2-normalised vector.
        7.  Upsert to Qdrant posts collection.
        8.  update_account_embedding (sliding window mean).
        9.  Upsert tags (embed unembedded tags; sync all to Qdrant tags collection).

        Returns a status dict: {"status": str, "post_id": int, ...}.
        """
        from qdrant_client.models import PointStruct

        # Load post with account (need blog_name for image path)
        result = await self._session.execute(
            select(Post)
            .options(joinedload(Post.account))
            .where(Post.id == post_id)
        )
        post = result.scalar_one_or_none()

        if post is None:
            log.warning("Post %d not found — skipping", post_id)
            return {"status": "not_found", "post_id": post_id}

        # ── Gate 1: NSFW ────────────────────────────────────────────────
        if post.nsfw:
            log.info("Post %d is NSFW — skipping indexing", post_id)
            return {"status": "nsfw_skip", "post_id": post_id}

        # ── Gate 2: Threshold ────────────────────────────────────────────
        count_result = await self._session.execute(
            select(func.count()).select_from(Post).where(Post.nsfw == False)  # noqa: E712
        )
        indexed_count: int = count_result.scalar_one()
        if indexed_count >= settings.target_post_count:
            log.info(
                "Target reached (%d >= %d) at post %d — skipping indexing",
                indexed_count,
                settings.target_post_count,
                post_id,
            )
            return {"status": "threshold_skip", "post_id": post_id}

        # ── Tags ─────────────────────────────────────────────────────────
        tags_result = await self._session.execute(
            select(Tag)
            .join(PostTag, Tag.id == PostTag.tag_id)
            .where(PostTag.post_id == post.id)
        )
        tags: list[Tag] = list(tags_result.scalars().all())
        tag_names: list[str] = [t.name for t in tags]

        # ── Text embedding ────────────────────────────────────────────────
        text_result = self._text.process_post(
            {"body_raw": post.body_raw or "", "tags": tag_names}
        )
        text_emb: np.ndarray = text_result["text_embedding"]  # (384,)
        tag_emb: np.ndarray  = text_result["tag_embedding"]   # (384,)

        # ── Image embedding ───────────────────────────────────────────────
        image_emb: np.ndarray | None = None
        if post.account is not None:
            image_dir = (
                Path("data/raw_images")
                / post.account.blog_name
                / str(post.tumblr_id)
            )
            image_paths = sorted(str(p) for p in image_dir.glob("*.jpg"))
            if image_paths:
                clip_batch = self._image.embed_images_batch(image_paths)  # (N, 512)
                mean_clip = clip_batch.mean(axis=0)                        # (512,)
                image_emb = self._image.project(mean_clip)                 # (384,)

        # ── Unified embedding ─────────────────────────────────────────────
        # Pass None for all-zero modalities so build_post_embedding can
        # reason about which modalities are absent vs. just uninformative.
        def _nonzero_or_none(v: np.ndarray) -> np.ndarray | None:
            return v if np.any(v) else None

        post_emb = build_post_embedding(
            _nonzero_or_none(text_emb),
            image_emb,
            _nonzero_or_none(tag_emb),
        )

        # ── Upsert post to Qdrant ─────────────────────────────────────────
        payload: dict = {
            "post_id": post.id,
            "account_id": post.account_id,
            "published_at": post.published_at.timestamp() if post.published_at else 0.0,
            "note_count": post.note_count or 0,
            "tags": tag_names,
            "nsfw": False,
            "lang": post.lang or "und",
        }
        await self._qdrant.upsert(
            collection_name=POSTS_COLLECTION,
            points=[PointStruct(id=post.id, vector=post_emb.tolist(), payload=payload)],
        )

        # ── Update account embedding (sliding window) ─────────────────────
        await self.update_account_embedding(post.account_id, post_emb)

        # ── Upsert tags to Qdrant ─────────────────────────────────────────
        if tags:
            await self._upsert_tags(tags, tag_names)

        log.info(
            "Indexed post %d (account=%d, tags=%d)", post.id, post.account_id, len(tags)
        )
        return {"status": "ok", "post_id": post_id}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _upsert_tags(self, tags: list[Tag], tag_names: list[str]) -> None:
        """
        Embed any tags that have no embedding yet, then upsert all to Qdrant.
        Also persists newly computed embeddings back to PostgreSQL (Tag.embedding).
        """
        from qdrant_client.models import PointStruct

        points = []
        needs_commit = False

        for tag, name in zip(tags, tag_names):
            if tag.embedding is None:
                emb = self._text.embed([name])[0].astype(np.float32)  # (384,)
                tag.embedding = emb.tolist()
                needs_commit = True
            else:
                emb = np.array(tag.embedding, dtype=np.float32)

            points.append(
                PointStruct(
                    id=tag.id,
                    vector=emb.tolist(),
                    payload={"tag_id": tag.id, "name": name},
                )
            )

        if points:
            await self._qdrant.upsert(collection_name=TAGS_COLLECTION, points=points)

        if needs_commit:
            await self._session.commit()
