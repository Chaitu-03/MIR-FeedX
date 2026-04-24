"""
tests/test_workers/test_tasks.py
Tests for Celery task configuration — Prompt 14.

We do NOT run real Celery tasks here (requires a broker). We assert the config
and the known-bug guards: threshold gate, video filter, DLQ routing.
"""
from __future__ import annotations

import pytest

from mir.workers.celery_app import celery_app


class TestCeleryConfig:
    def test_acks_late_true(self):
        assert celery_app.conf.task_acks_late is True

    def test_reject_on_worker_lost_true(self):
        assert celery_app.conf.task_reject_on_worker_lost is True

    def test_dead_letter_queue_registered(self):
        queues = celery_app.conf.task_queues
        assert "dead_letter" in queues

    def test_gpu_queue_routed_correctly(self):
        routes = celery_app.conf.task_routes
        assert routes["mir.workers.tasks.process_post"]["queue"] == "gpu"
        assert routes["mir.workers.tasks.rebuild_communities"]["queue"] == "gpu"
        assert routes["mir.workers.tasks.reindex_all"]["queue"] == "gpu"

    def test_default_queue_routed_correctly(self):
        routes = celery_app.conf.task_routes
        assert routes["mir.workers.tasks.crawl_blog"]["queue"] == "default"
        assert routes["mir.workers.tasks.nightly_optimize"]["queue"] == "default"

    def test_beat_schedule_has_entries(self):
        sched = celery_app.conf.beat_schedule
        assert "crawl-active-blogs" in sched
        assert "rebuild-communities" in sched
        assert "nightly-optimize" in sched
        assert "cleanup-cache" in sched

    def test_tasks_registered(self):
        # Importing tasks.py registers them on the app
        import mir.workers.tasks  # noqa: F401
        names = set(celery_app.tasks.keys())
        expected = {
            "mir.workers.tasks.crawl_blog",
            "mir.workers.tasks.process_post",
            "mir.workers.tasks.process_post_images",
            "mir.workers.tasks.rebuild_communities",
            "mir.workers.tasks.reindex_all",
            "mir.workers.tasks.nightly_optimize",
            "mir.workers.tasks.cleanup_cache",
            "mir.workers.tasks.record_dead_letter",
        }
        missing = expected - names
        assert not missing, f"Missing task registrations: {missing}"


class TestCrawlThresholdGate:
    @pytest.mark.asyncio
    async def test_threshold_met_short_circuits(self, monkeypatch):
        """When SFW count >= TARGET_POST_COUNT, crawl_blog must make zero API calls."""
        from mir.config import settings
        from mir.workers import tasks as tasks_mod

        monkeypatch.setattr(settings, "target_post_count", 5)

        # Count above threshold
        class _FakeSession:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            async def scalar(self, stmt):
                return 5  # == threshold
            async def execute(self, *a, **kw):
                class R:
                    def scalars(self_inner):
                        class S:
                            def all(self_inner2):
                                return []
                        return S()
                return R()
            async def commit(self):
                pass

        monkeypatch.setattr(tasks_mod, "AsyncSessionLocal", lambda: _FakeSession())

        # Crawler would explode if reached; assert it isn't
        def _boom(*a, **kw):
            raise AssertionError("Crawler was instantiated but threshold was met")

        monkeypatch.setattr("mir.ingestion.crawler.Crawler", _boom)

        result = await tasks_mod._crawl_blog_async("someblog")
        assert result["status"] == "threshold_met"
        assert result["count"] == 5
