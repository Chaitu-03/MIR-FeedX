"""
mir/api/auth.py
API-key authentication — Prompt 12.

Bcrypt-hashed keys live in the api_keys table. On each request we iterate all
keys and call bcrypt.checkpw because the salt is embedded in the hash (can't
hash the incoming key to a fixed value and compare). With O(10²) keys this is
fine; for larger deployments add an indexed SHA256 fingerprint column as a
pre-filter.
"""
from __future__ import annotations

from datetime import datetime, timezone

import bcrypt
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mir.db.models import APIKey
from mir.db.session import AsyncSessionLocal


async def _get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


async def _lookup_key(db: AsyncSession, raw_key: str) -> APIKey | None:
    """Iterate active keys, bcrypt.checkpw on each. Returns matched row or None."""
    rows = (await db.execute(select(APIKey))).scalars().all()
    key_bytes = raw_key.encode("utf-8")
    for row in rows:
        try:
            if bcrypt.checkpw(key_bytes, row.key_hash.encode("utf-8")):
                return row
        except ValueError:
            # malformed hash — skip
            continue
    return None


_OPEN_KEY = APIKey(id=0, key_hash="", label="open-access", is_admin=False)


async def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    db: AsyncSession = Depends(_get_db),
) -> APIKey:
    from mir.config import settings

    # When api_keys_enabled=False (dev mode), skip auth entirely.
    if not settings.api_keys_enabled:
        return _OPEN_KEY

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    row = await _lookup_key(db, x_api_key)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    row.last_used_at = datetime.now(tz=timezone.utc)
    await db.commit()
    return row


async def require_admin_key(
    api_key: APIKey = Depends(require_api_key),
) -> APIKey:
    if not api_key.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return api_key
