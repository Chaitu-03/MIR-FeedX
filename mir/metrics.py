"""
mir/metrics.py
Prometheus metrics — Prompt 15.

Import the counters/histograms/gauges anywhere; expose via `metrics_response()`
from the FastAPI /metrics endpoint.
"""
from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.responses import Response

registry = CollectorRegistry(auto_describe=True)

# ── Ingestion / indexing ─────────────────────────────────────────────────────
posts_indexed_total = Counter(
    "mir_posts_indexed_total",
    "Total posts indexed into Qdrant.",
    labelnames=("lang", "nsfw"),
    registry=registry,
)

nsfw_flagged_total = Counter(
    "mir_nsfw_flagged_total",
    "Posts flagged NSFW at ingestion time.",
    registry=registry,
)

crawl_posts_fetched_total = Counter(
    "mir_crawl_posts_fetched_total",
    "Posts fetched from Tumblr API by crawler.",
    labelnames=("blog_name",),
    registry=registry,
)

# ── Search ───────────────────────────────────────────────────────────────────
search_duration_seconds = Histogram(
    "mir_search_duration_seconds",
    "Search latency per query.",
    labelnames=("query_type",),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=registry,
)

cache_hits_total = Counter(
    "mir_cache_hits_total",
    "Redis cache hits.",
    labelnames=("query_type",),
    registry=registry,
)

cache_misses_total = Counter(
    "mir_cache_misses_total",
    "Redis cache misses.",
    labelnames=("query_type",),
    registry=registry,
)

# ── Queue depth ──────────────────────────────────────────────────────────────
queue_depth = Gauge(
    "mir_queue_depth",
    "Celery queue depth (pending tasks).",
    labelnames=("queue_name",),
    registry=registry,
)


def metrics_response() -> Response:
    """Return Prometheus text exposition. Wire to GET /metrics."""
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
