"""
tests/test_api/test_endpoints.py
Unit tests for FastAPI auth + router wiring — Prompt 12.

We do NOT spin up a real DB / Qdrant / Redis here. Instead we build a minimal
FastAPI app and override dependencies so we can assert routing + auth behavior
in isolation. The full integration flow is covered by tests/test_e2e.py.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from mir.api.auth import require_admin_key, require_api_key
from mir.api.routers import admin as admin_router
from mir.api.routers import search as search_router


class _FakeAPIKey:
    def __init__(self, is_admin: bool = False):
        self.is_admin = is_admin
        self.label = "test"


def _build_app(api_key_valid: bool = True, admin: bool = False) -> FastAPI:
    from mir.api.deps import get_db, get_qdrant, get_text_processor

    app = FastAPI()
    app.include_router(search_router.router)
    app.include_router(admin_router.router)
    # State stubs so the dep-resolver can see them if it tries
    app.state.qdrant = object()
    app.state.text_processor = object()

    async def _ok():
        return _FakeAPIKey(is_admin=admin)

    async def _reject():
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="Invalid API key")

    async def _fake_db():
        yield None

    app.dependency_overrides[get_db] = _fake_db
    app.dependency_overrides[get_qdrant] = lambda: app.state.qdrant
    app.dependency_overrides[get_text_processor] = lambda: app.state.text_processor

    if api_key_valid:
        app.dependency_overrides[require_api_key] = _ok
        app.dependency_overrides[require_admin_key] = _ok
    else:
        app.dependency_overrides[require_api_key] = _reject
        app.dependency_overrides[require_admin_key] = _reject
    return app


@pytest.mark.asyncio
async def test_missing_key_returns_401():
    app = _build_app(api_key_valid=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/api/v1/search/general", json={"query": "q"})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_admin_endpoint_requires_admin():
    app = _build_app(api_key_valid=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/api/v1/admin/cache/clear")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_request_schema_rejects_empty_query():
    """Pydantic min_length=1 on query field must yield 422."""
    app = _build_app(api_key_valid=True)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/api/v1/search/general", json={"query": ""})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_cache_clear_hits_admin_route(monkeypatch):
    """Admin route calls cache.cache_clear; patch the name the router actually uses."""
    from mir.api.routers import admin as admin_router_mod

    async def _fake_clear(*a, **kw):
        return 7

    monkeypatch.setattr(admin_router_mod, "cache_clear", _fake_clear)

    app = _build_app(api_key_valid=True, admin=True)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/api/v1/admin/cache/clear")
    assert r.status_code == 200
    assert r.json() == {"cleared": 7}
