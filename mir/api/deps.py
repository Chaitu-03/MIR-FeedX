"""
mir/api/deps.py
Shared FastAPI dependencies — DB session + app-scoped singletons.
"""
from __future__ import annotations

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from mir.db.session import AsyncSessionLocal


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


def get_qdrant(request: Request):
    return request.app.state.qdrant


def get_text_processor(request: Request):
    return request.app.state.text_processor
