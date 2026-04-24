"""
mir/api/manage.py
CLI for admin operations.

Usage:
    python -m mir.api.manage create-api-key --label "my-client" [--admin]
"""
from __future__ import annotations

import argparse
import asyncio
import secrets

import bcrypt
from sqlalchemy import insert

from mir.db.models import APIKey
from mir.db.session import AsyncSessionLocal


async def _create_api_key(label: str, is_admin: bool) -> str:
    raw = secrets.token_urlsafe(32)
    hashed = bcrypt.hashpw(raw.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")
    async with AsyncSessionLocal() as db:
        await db.execute(
            insert(APIKey).values(key_hash=hashed, label=label, is_admin=is_admin)
        )
        await db.commit()
    return raw


def main() -> None:
    p = argparse.ArgumentParser(prog="mir-manage")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("create-api-key", help="Create an API key and print the plaintext once")
    c.add_argument("--label", required=True)
    c.add_argument("--admin", action="store_true")

    args = p.parse_args()
    if args.cmd == "create-api-key":
        raw = asyncio.run(_create_api_key(args.label, args.admin))
        print("API key created. Save this NOW — it will not be shown again:")
        print(raw)


if __name__ == "__main__":
    main()
