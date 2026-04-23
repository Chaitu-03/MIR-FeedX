from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import numpy as np
import redis
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from mir.config import settings
from mir.db.models import Post
from mir.db.session import AsyncSessionLocal
from mir.workers.celery_app import celery_app

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared heavy objects — initialised once per worker process, not per task.
# Celery forks workers, so these are created after the fork (lazy init).
# ---------------------------------------------------------------------------

_clip_model: Any = None
_preprocess: Any = None
_nsfw_clf: Any = None
_img_processor: Any = None
_redis_client: Any = None


def _get_shared_resources():
    """Lazy-initialise ML models and Redis client once per Celery worker process."""
    global _clip_model, _preprocess, _nsfw_clf, _img_processor, _redis_client

    if _clip_model is None:
        import open_clip
        from mir.processing.images import ImageProcessor
        from mir.processing.safety import NSFWClassifier

        log.info("Loading CLIP ViT-B/32 (once per worker)…")
        _clip_model, _, _preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        _redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=False)

        # Share the same CLIP instance between classifier and processor
        _nsfw_clf = NSFWClassifier(clip_model=_clip_model, preprocess=_preprocess)
        _img_processor = ImageProcessor(
            clip_model=_clip_model,
            preprocess=_preprocess,
            redis_client=_redis_client,
        )
        log.info("Worker resources ready.")

    return _nsfw_clf, _img_processor


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, max_retries=3, default_retry_delay=5)
def process_post_images(self, post_id: int) -> dict:
    """
    Classify and embed all images for a post.

    Flow:
      1. Load Post + Account from Postgres.
      2. Discover local images under data/raw_images/{blog}/{tumblr_id}/.
      3. NSFWClassifier on every image.
         - Any NSFW → mark post.nsfw=True, post.nsfw_score=max_score, return.
      4. CLIP-embed all SFW images; mean-pool → 512-d; project → 384-d.
      5. Store mean image_embedding on the Post row.

    Returns summary dict for result backend.
    """
    try:
        return asyncio.run(_process_async(post_id))
    except Exception as exc:
        log.error("process_post_images(%d) error: %s", post_id, exc)
        raise self.retry(exc=exc)


async def _process_async(post_id: int) -> dict:
    nsfw_clf, img_processor = _get_shared_resources()

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Post)
            .options(joinedload(Post.account))
            .where(Post.id == post_id)
        )
        post = result.scalar_one_or_none()
        if post is None:
            log.warning("Post %d not found — skipping", post_id)
            return {"status": "not_found", "post_id": post_id}

        blog_name = post.account.blog_name
        image_dir = Path("data/raw_images") / blog_name / str(post.tumblr_id)
        image_paths = sorted(image_dir.glob("*.jpg"))

        if not image_paths:
            log.debug("No local images for post %d", post_id)
            return {"status": "no_images", "post_id": post_id}

        # ------------------------------------------------------------------
        # 1. NSFW classification — must run before any embedding
        # ------------------------------------------------------------------
        pil_images = [Image.open(p).convert("RGB") for p in image_paths]
        nsfw_results = [nsfw_clf.is_image_nsfw(img) for img in pil_images]

        scores = [score for _, score in nsfw_results]
        max_score = float(max(scores))
        any_nsfw = any(flag for flag, _ in nsfw_results)

        if any_nsfw:
            post.nsfw = True
            post.nsfw_score = max_score
            await session.commit()
            log.info("Post %d marked NSFW (max_score=%.3f)", post_id, max_score)
            return {"status": "nsfw", "post_id": post_id, "nsfw_score": max_score}

        # ------------------------------------------------------------------
        # 2. Embed SFW images → mean-pool → project
        # ------------------------------------------------------------------
        str_paths = [str(p) for p in image_paths]
        embeddings = img_processor.embed_images_batch(str_paths)   # (N, 512)
        mean_clip = embeddings.mean(axis=0)                         # (512,)
        projected = img_processor.project(mean_clip)                # (384,)

        post.image_embedding = projected.tolist()
        post.nsfw_score = max_score   # store the max SFW score too (for tuning)
        await session.commit()

        log.info(
            "Post %d embedded: %d image(s), mean_clip norm=%.4f",
            post_id, len(image_paths), float(np.linalg.norm(mean_clip)),
        )
        return {
            "status": "ok",
            "post_id": post_id,
            "n_images": len(image_paths),
            "nsfw_score": max_score,
        }
