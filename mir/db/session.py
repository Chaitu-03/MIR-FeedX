from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.pool import NullPool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from mir.config import settings

# NullPool: no connection reuse across asyncio.run() calls.
# Required for Celery workers — each task runs asyncio.run() which creates+closes
# an event loop; pooled connections bound to the old loop cause "Future attached
# to a different loop" on retries within the same worker process.
engine = create_async_engine(
    settings.database_url,
    echo=False,
    poolclass=NullPool,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
