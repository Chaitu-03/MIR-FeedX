"""
mir/api/app.py
FastAPI app entrypoint — Prompt 12.

Lifespan initialises:
  - QdrantManager (connection check)
  - TextProcessor (MiniLM embedding model)
  - Redis pool (lazy, so nothing to do here)

CORS is permissive by default — tighten via env before shipping.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from mir.api.routers import admin as admin_router
from mir.api.routers import dashboard as dashboard_router
from mir.api.routers import health as health_router
from mir.api.routers import search as search_router
from mir.config import settings
from mir.search.cache import close_redis
from mir.search.vector_store import QdrantManager

_STATIC_DIR = Path(__file__).parent / "static"

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ──────────────────────────────────────────────────────────
    try:
        qdrant = QdrantManager()
        qdrant.ensure_collections()
        app.state.qdrant = qdrant
    except Exception as exc:
        log.warning("QdrantManager init failed: %s — /health will flag it", exc)
        app.state.qdrant = None

    try:
        from mir.processing.text import TextProcessor
        app.state.text_processor = TextProcessor()
    except Exception as exc:
        log.warning("TextProcessor init failed: %s", exc)
        app.state.text_processor = None

    try:
        # Try to configure structured logging; fall back silently if structlog absent
        from mir.logging import configure_logging
        configure_logging()
    except Exception:
        pass

    yield

    # ── shutdown ─────────────────────────────────────────────────────────
    await close_redis()


def create_app() -> FastAPI:
    app = FastAPI(
        title="MIR-FeedX Search API",
        version="0.1.0",
        description="Multimodal information retrieval over Tumblr feed data.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router.router)
    app.include_router(search_router.router)
    app.include_router(admin_router.router)
    app.include_router(dashboard_router.router)

    @app.get("/", include_in_schema=False)
    async def root():
        return FileResponse(_STATIC_DIR / "dashboard.html")

    return app


app = create_app()
