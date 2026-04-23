from __future__ import annotations

from celery import Celery

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
    task_acks_late=True,           # re-queue on worker crash
    worker_prefetch_multiplier=1,  # one task at a time per worker (ML tasks are heavy)
)
