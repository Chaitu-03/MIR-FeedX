"""
mir/workers/celery_app.py
Celery app configuration — Prompt 14.

Queues:
  - 'gpu'          concurrency=1  ML-heavy tasks (embed / NSFW / reindex)
  - 'default'      concurrency=2  lightweight (crawl, optimize, cache cleanup)
  - 'dead_letter'  manual inspection queue for tasks that exhausted retries

Beat schedule:
  - crawl_active_blogs every settings.crawl_interval_minutes (env: CRAWL_INTERVAL_MINUTES, default 15)
  - rebuild_communities daily at 03:00 UTC
  - nightly_optimize    daily at 04:00 UTC
  - cleanup_cache       hourly
"""
from __future__ import annotations

from datetime import timedelta

from celery import Celery, signals
from celery.schedules import crontab

from mir.config import settings

celery_app = Celery(
    "mir",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["mir.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # At-least-once delivery semantics
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    # Sensible defaults; individual tasks may override
    task_default_queue="default",
    task_default_retry_delay=10,
    task_queues={
        "default": {"exchange": "default", "routing_key": "default"},
        "gpu": {"exchange": "gpu", "routing_key": "gpu"},
        "dead_letter": {"exchange": "dead_letter", "routing_key": "dead_letter"},
    },
    task_routes={
        "mir.workers.tasks.process_post": {"queue": "gpu"},
        "mir.workers.tasks.process_post_images": {"queue": "gpu"},
        "mir.workers.tasks.rebuild_communities": {"queue": "gpu"},
        "mir.workers.tasks.reindex_all": {"queue": "gpu"},
        "mir.workers.tasks.crawl_blog": {"queue": "default"},
        "mir.workers.tasks.crawl_active_blogs": {"queue": "default"},
        "mir.workers.tasks.nightly_optimize": {"queue": "default"},
        "mir.workers.tasks.cleanup_cache": {"queue": "default"},
    },
    beat_schedule={
        "crawl-active-blogs": {
            "task": "mir.workers.tasks.crawl_active_blogs",
            "schedule": timedelta(minutes=settings.crawl_interval_minutes),
        },
        "rebuild-communities": {
            "task": "mir.workers.tasks.rebuild_communities",
            "schedule": crontab(minute=0, hour=3),
        },
        "nightly-optimize": {
            "task": "mir.workers.tasks.nightly_optimize",
            "schedule": crontab(minute=0, hour=4),
        },
        "cleanup-cache": {
            "task": "mir.workers.tasks.cleanup_cache",
            "schedule": crontab(minute=0, hour="*"),
        },
    },
)


# ── Dead-letter signal handler ───────────────────────────────────────────────
# When a task permanently fails (max_retries exhausted), enqueue a marker task
# on the 'dead_letter' queue for manual inspection. Done via a signal so each
# task doesn't need to repeat DLQ boilerplate.

@signals.task_failure.connect
def _to_dead_letter(sender=None, task_id=None, exception=None, args=None, kwargs=None, einfo=None, **_):
    if sender is None:
        return
    retries = getattr(sender.request, "retries", 0) if hasattr(sender, "request") else 0
    max_retries = getattr(sender, "max_retries", 3) or 3
    if retries < max_retries:
        return  # still has retries left
    try:
        celery_app.send_task(
            "mir.workers.tasks.record_dead_letter",
            queue="dead_letter",
            kwargs={
                "task_name": sender.name,
                "task_id": task_id,
                "exception": repr(exception),
                "args": args,
                "kwargs": kwargs,
            },
        )
    except Exception:
        # Never break the worker on DLQ delivery failure
        pass
