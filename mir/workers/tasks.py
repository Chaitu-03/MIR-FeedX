"""
mir/workers/tasks.py
Celery task definitions — Prompts 11 + 14.

Tasks:
  - crawl_blog(blog_name)      fetch + persist posts; threshold-gated
  - crawl_active_blogs         Beat: enqueue crawl_blog for every active/pending blog
  - process_post(post_id)      text → nsfw → image → embed → Qdrant upsert
  - process_post_images(post_id) image-only pipeline (legacy entry)
  - rebuild_communities        nightly tag + account clustering
  - reindex_all(model_name)    batch re-embed + Qdrant upsert
  - nightly_optimize           Qdrant segment merge
  - cleanup_cache              drop expired / orphaned cache keys
  - record_dead_letter(...)    DLQ sink for terminally-failed tasks
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import redis
from PIL import Image
from sqlalchemy import false, func, select, update
from sqlalchemy.orm import joinedload

from mir.config import settings
from mir.db.models import CrawlState, Post
from mir.db.session import AsyncSessionLocal
from mir.workers.celery_app import celery_app

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared heavy objects — initialised once per worker process (post-fork).
# ---------------------------------------------------------------------------

_clip_model: Any = None
_preprocess: Any = None
_nsfw_clf: Any = None
_img_processor: Any = None
_redis_client: Any = None


def _get_shared_resources():
    global _clip_model, _preprocess, _nsfw_clf, _img_processor, _redis_client
    if _clip_model is None:
        import open_clip
        from mir.processing.images import ImageProcessor
        from mir.processing.safety import NSFWClassifier

        log.info("Loading CLIP ViT-B/32 once per worker…")
        _clip_model, _, _preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        _redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=False)
        _nsfw_clf = NSFWClassifier(clip_model=_clip_model, preprocess=_preprocess)
        _img_processor = ImageProcessor(
            clip_model=_clip_model,
            preprocess=_preprocess,
            redis_client=_redis_client,
        )
        log.info("Worker resources ready.")
    return _nsfw_clf, _img_processor


# ---------------------------------------------------------------------------
# crawl_blog — threshold-gated, resumable
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="mir.workers.tasks.crawl_blog",
                 max_retries=3, default_retry_delay=30)
def crawl_blog(self, blog_name: str) -> dict:
    """
    Threshold gate FIRST — if indexed SFW count >= TARGET_POST_COUNT,
    return without making a single API call. Otherwise run one crawl pass.
    """
    try:
        return asyncio.run(_crawl_blog_async(blog_name))
    except Exception as exc:
        log.error("crawl_blog(%s) error: %s", blog_name, exc)
        raise self.retry(exc=exc)


async def _crawl_blog_async(blog_name: str) -> dict:
    async with AsyncSessionLocal() as db:
        count = await db.scalar(
            select(func.count()).select_from(Post).where(Post.nsfw == false())
        ) or 0
        if count >= settings.target_post_count:
            log.info("Threshold met (%d) — crawl_blog(%s) is a no-op", count, blog_name)
            await db.execute(
                update(CrawlState)
                .where(CrawlState.blog_name == blog_name)
                .values(last_crawled_at=datetime.now(timezone.utc))
            )
            await db.commit()
            return {"status": "threshold_met", "count": count, "blog": blog_name}

        from mir.ingestion.client import TumblrClient
        from mir.ingestion.crawler import Crawler
        from mir.ingestion.images import ImageDownloader

        async with TumblrClient(api_keys=settings.tumblr_api_keys) as client:
            downloader = ImageDownloader()
            crawler = Crawler(session=db, client=client, downloader=downloader)
            await crawler.crawl(seed_blogs=[blog_name])

        new_count = await db.scalar(
            select(func.count()).select_from(Post).where(Post.nsfw == false())
        ) or 0

        values: dict[str, Any] = {"last_crawled_at": datetime.now(timezone.utc)}
        if new_count >= settings.target_post_count:
            values["status"] = "paused"
        await db.execute(
            update(CrawlState)
            .where(CrawlState.blog_name == blog_name)
            .values(**values)
        )
        await db.commit()

        return {"status": "ok", "blog": blog_name, "count": new_count}


@celery_app.task(name="mir.workers.tasks.crawl_active_blogs")
def crawl_active_blogs() -> dict:
    """Beat entry: fan out crawl_blog for every active|pending crawl state."""
    return asyncio.run(_crawl_active_blogs_async())


async def _crawl_active_blogs_async() -> dict:
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(
            select(CrawlState.blog_name).where(CrawlState.status.in_(("active", "pending")))
        )).scalars().all()
    enqueued = 0
    for blog in rows:
        crawl_blog.delay(blog)
        enqueued += 1
    return {"enqueued": enqueued}


# ---------------------------------------------------------------------------
# process_post — unified text + image + embedding pipeline
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="mir.workers.tasks.process_post",
                 max_retries=3, default_retry_delay=10)
def process_post(self, post_id: int) -> dict:
    try:
        return asyncio.run(_process_post_async(post_id))
    except Exception as exc:
        log.error("process_post(%d) error: %s", post_id, exc)
        raise self.retry(exc=exc)


async def _process_post_async(post_id: int) -> dict:
    from mir.processing.embeddings import build_post_embedding
    from mir.processing.text import TextProcessor
    from mir.search.vector_store import QdrantManager

    nsfw_clf, img_processor = _get_shared_resources()
    text_proc = TextProcessor()
    qdrant = QdrantManager()

    async with AsyncSessionLocal() as db:
        post = (await db.execute(
            select(Post).options(joinedload(Post.account)).where(Post.id == post_id)
        )).scalar_one_or_none()
        if post is None:
            return {"status": "not_found", "post_id": post_id}

        # 1. Text embedding
        text_vec = None
        if post.body_clean:
            text_vec = text_proc.embed([post.body_clean])[0]

        # 2. Image NSFW + embedding
        image_vec: np.ndarray | None = None
        image_dir = Path("data/raw_images") / post.account.blog_name / str(post.tumblr_id)
        image_paths = sorted(image_dir.glob("*.jpg"))
        if image_paths:
            pil_images = [Image.open(p).convert("RGB") for p in image_paths]
            nsfw_results = [nsfw_clf.is_image_nsfw(img) for img in pil_images]
            scores = [s for _, s in nsfw_results]
            max_score = float(max(scores))
            if any(flag for flag, _ in nsfw_results):
                post.nsfw = True
                post.nsfw_score = max_score
                await db.commit()
                return {"status": "nsfw", "post_id": post_id, "nsfw_score": max_score}

            embeddings = img_processor.embed_images_batch([str(p) for p in image_paths])
            mean_clip = embeddings.mean(axis=0)
            image_vec = img_processor.project(mean_clip)
            post.image_embedding = image_vec.tolist()
            post.nsfw_score = max_score

        # 3. Tag embedding — mean of related tag embeddings if present
        tag_vec: np.ndarray | None = None  # left to embeddings builder

        # 4. Unified post embedding → Qdrant (SFW only)
        if not post.nsfw:
            fused = build_post_embedding(
                text_emb=text_vec, image_emb=image_vec, tag_emb=tag_vec
            )
            if fused is not None and float(np.linalg.norm(fused)) > 1e-10:
                payload = {
                    "post_id": post.id,
                    "account_id": post.account_id,
                    "lang": post.lang,
                    "note_count": post.note_count,
                    "published_at": post.published_at.isoformat() if post.published_at else None,
                    "nsfw": False,
                }
                qdrant.upsert_post(post.id, fused.tolist(), payload)

        await db.commit()
        return {"status": "ok", "post_id": post_id}


# ---------------------------------------------------------------------------
# process_post_images — kept for legacy callers (Prompt 14 #3)
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="mir.workers.tasks.process_post_images",
                 max_retries=3, default_retry_delay=5)
def process_post_images(self, post_id: int) -> dict:
    try:
        return asyncio.run(_process_images_async(post_id))
    except Exception as exc:
        log.error("process_post_images(%d) error: %s", post_id, exc)
        raise self.retry(exc=exc)


async def _process_images_async(post_id: int) -> dict:
    nsfw_clf, img_processor = _get_shared_resources()
    async with AsyncSessionLocal() as db:
        post = (await db.execute(
            select(Post).options(joinedload(Post.account)).where(Post.id == post_id)
        )).scalar_one_or_none()
        if post is None:
            return {"status": "not_found", "post_id": post_id}

        image_dir = Path("data/raw_images") / post.account.blog_name / str(post.tumblr_id)
        image_paths = sorted(image_dir.glob("*.jpg"))
        if not image_paths:
            return {"status": "no_images", "post_id": post_id}

        pil_images = [Image.open(p).convert("RGB") for p in image_paths]
        nsfw_results = [nsfw_clf.is_image_nsfw(img) for img in pil_images]
        scores = [s for _, s in nsfw_results]
        max_score = float(max(scores))
        if any(flag for flag, _ in nsfw_results):
            post.nsfw = True
            post.nsfw_score = max_score
            await db.commit()
            return {"status": "nsfw", "post_id": post_id, "nsfw_score": max_score}

        embeddings = img_processor.embed_images_batch([str(p) for p in image_paths])
        mean_clip = embeddings.mean(axis=0)
        projected = img_processor.project(mean_clip)
        post.image_embedding = projected.tolist()
        post.nsfw_score = max_score
        await db.commit()
        return {
            "status": "ok",
            "post_id": post_id,
            "n_images": len(image_paths),
            "norm": float(np.linalg.norm(mean_clip)),
        }


# ---------------------------------------------------------------------------
# rebuild_communities
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="mir.workers.tasks.rebuild_communities",
                 max_retries=3, default_retry_delay=60)
def rebuild_communities(self) -> dict:
    try:
        return asyncio.run(_rebuild_communities_async())
    except Exception as exc:
        log.error("rebuild_communities error: %s", exc)
        raise self.retry(exc=exc)


async def _rebuild_communities_async() -> dict:
    from mir.processing.communities import rebuild_communities as core_rebuild
    from mir.processing.text import TextProcessor
    from mir.search.vector_store import QdrantManager

    qdrant = QdrantManager()
    text_proc = TextProcessor()
    async with AsyncSessionLocal() as db:
        await core_rebuild(db, qdrant, text_proc)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# reindex_all
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="mir.workers.tasks.reindex_all",
                 max_retries=3, default_retry_delay=60)
def reindex_all(self, model_name: str | None = None) -> dict:
    try:
        return asyncio.run(_reindex_all_async(model_name))
    except Exception as exc:
        log.error("reindex_all error: %s", exc)
        raise self.retry(exc=exc)


async def _reindex_all_async(model_name: str | None) -> dict:
    BATCH = 100
    reindexed = 0
    async with AsyncSessionLocal() as db:
        ids = (await db.execute(
            select(Post.id).where(Post.nsfw == false()).order_by(Post.id)
        )).scalars().all()
    for i in range(0, len(ids), BATCH):
        for pid in ids[i : i + BATCH]:
            process_post.delay(pid)
            reindexed += 1
    return {"enqueued": reindexed, "model": model_name}


# ---------------------------------------------------------------------------
# nightly_optimize
# ---------------------------------------------------------------------------

@celery_app.task(name="mir.workers.tasks.nightly_optimize")
def nightly_optimize() -> dict:
    from mir.search.vector_store import QdrantManager

    mgr = QdrantManager()
    mgr.nightly_optimize()
    stats = {}
    for name in ("posts", "accounts", "tags"):
        try:
            stats[name] = mgr.get_collection_info(name)
        except Exception as exc:
            stats[name] = {"error": str(exc)}
    log.info("nightly_optimize stats: %s", stats)
    return stats


# ---------------------------------------------------------------------------
# cleanup_cache — hourly orphan sweep (mostly a no-op with TTLs, but useful
# to purge well-known stale keys).
# ---------------------------------------------------------------------------

@celery_app.task(name="mir.workers.tasks.cleanup_cache")
def cleanup_cache() -> dict:
    try:
        client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        keys = list(client.scan_iter(match=f"{settings.cache_key_prefix}*", count=500))
        # Remove any key without TTL (defensive — cache_set always sets TTL)
        purged = 0
        for k in keys:
            if client.ttl(k) == -1:
                client.delete(k)
                purged += 1
        return {"scanned": len(keys), "purged": purged}
    except Exception as exc:
        log.warning("cleanup_cache failed: %s", exc)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# record_dead_letter — DLQ sink
# ---------------------------------------------------------------------------

@celery_app.task(name="mir.workers.tasks.record_dead_letter")
def record_dead_letter(task_name: str, task_id: str, exception: str,
                       args=None, kwargs=None) -> dict:
    log.error(
        "DEAD LETTER: task=%s id=%s exc=%s args=%r kwargs=%r",
        task_name, task_id, exception, args, kwargs,
    )
    return {"dead_letter": True, "task": task_name, "id": task_id}
