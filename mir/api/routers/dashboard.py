"""
mir/api/routers/dashboard.py
Dashboard API — unauthenticated endpoints for local dev inspection.
"""
from __future__ import annotations

import asyncio
import secrets
import subprocess
import time
from pathlib import Path

import bcrypt
from fastapi import APIRouter, Query
from pydantic import BaseModel
from sqlalchemy import insert, text
from sqlalchemy.ext.asyncio import AsyncSession

from mir.api.deps import get_db
from mir.db.models import APIKey, CrawlState
from mir.db.session import AsyncSessionLocal
from mir.ingestion.seeds import SEED_BLOGS, SEED_BLOGS_BY_GENRE
from mir.search.cache import get_redis

router = APIRouter(prefix="/dashboard/api", tags=["dashboard"])

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class SQLRequest(BaseModel):
    query: str


@router.get("/stats")
async def dashboard_stats():
    async with AsyncSessionLocal() as db:
        posts = (await db.execute(text("SELECT COUNT(*) FROM posts"))).scalar() or 0
        sfw = (await db.execute(text("SELECT COUNT(*) FROM posts WHERE nsfw = false"))).scalar() or 0
        nsfw = (await db.execute(text("SELECT COUNT(*) FROM posts WHERE nsfw = true"))).scalar() or 0
        accounts = (await db.execute(text("SELECT COUNT(*) FROM accounts"))).scalar() or 0
        tags = (await db.execute(text("SELECT COUNT(*) FROM tags"))).scalar() or 0
        communities = (await db.execute(text("SELECT COUNT(*) FROM communities"))).scalar() or 0
        last_crawl = (await db.execute(text("SELECT MAX(last_crawled_at) FROM crawl_state"))).scalar()
        crawl_rows = (await db.execute(
            text("SELECT status, COUNT(*) AS cnt FROM crawl_state GROUP BY status")
        )).fetchall()
        crawl_status = {r[0]: r[1] for r in crawl_rows} if crawl_rows else None

    return {
        "total_posts": posts,
        "sfw_posts": sfw,
        "nsfw_posts": nsfw,
        "total_accounts": accounts,
        "total_tags": tags,
        "total_communities": communities,
        "last_crawl": last_crawl.isoformat() if last_crawl else None,
        "crawl_status": crawl_status,
    }


@router.get("/crawl-state")
async def crawl_state():
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(text(
            "SELECT blog_name, status, last_crawled_at, fail_count, last_timestamp "
            "FROM crawl_state ORDER BY last_crawled_at DESC NULLS LAST"
        ))).fetchall()
    return {"rows": [dict(r._mapping) for r in rows]}


@router.post("/reset-dead")
async def reset_dead():
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            text("UPDATE crawl_state SET status = 'active', fail_count = 0 WHERE status = 'dead'")
        )
        await db.commit()
    return {"updated": result.rowcount}


@router.get("/table")
async def get_table(name: str = Query("posts"), limit: int = Query(20, ge=1, le=500)):
    allowed = {"posts", "accounts", "tags", "crawl_state", "communities", "post_tags", "api_keys"}
    if name not in allowed:
        return {"error": f"Table not allowed: {name}"}
    async with AsyncSessionLocal() as db:
        if name == "api_keys":
            rows = (await db.execute(text(
                f"SELECT id, label, is_admin, created_at, last_used_at FROM {name} LIMIT :lim"
            ), {"lim": limit})).fetchall()
        else:
            rows = (await db.execute(text(
                f"SELECT * FROM {name} LIMIT :lim"
            ), {"lim": limit})).fetchall()
    return {"rows": [dict(r._mapping) for r in rows]}


@router.post("/sql")
async def run_sql(req: SQLRequest):
    q = req.query.strip().rstrip(";")
    upper = q.upper()
    if not upper.startswith("SELECT") and not upper.startswith("WITH"):
        return {"error": "Only SELECT / WITH queries allowed"}
    t0 = time.perf_counter()
    try:
        async with AsyncSessionLocal() as db:
            rows = (await db.execute(text(q))).fetchall()
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        return {"rows": [dict(r._mapping) for r in rows], "elapsed_ms": elapsed}
    except Exception as exc:
        return {"error": str(exc)}


@router.post("/generate-key")
async def generate_key():
    raw = secrets.token_urlsafe(32)
    hashed = bcrypt.hashpw(raw.encode(), bcrypt.gensalt(rounds=12)).decode()
    async with AsyncSessionLocal() as db:
        await db.execute(
            insert(APIKey).values(key_hash=hashed, label="dashboard-auto", is_admin=True)
        )
        await db.commit()
    return {"key": raw}


@router.post("/tests")
async def run_tests(suite: str = Query("all")):
    allowed_suites = {
        "all": "tests/",
        "test_api": "tests/test_api/",
        "test_search": "tests/test_search/",
        "test_processing": "tests/test_processing/",
        "test_ingestion": "tests/test_ingestion/",
        "test_workers": "tests/test_workers/",
        "eval": "tests/eval/",
    }
    path = allowed_suites.get(suite, "tests/")
    test_path = PROJECT_ROOT / path

    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ["python", "-m", "pytest", str(test_path), "-v", "--tb=short", "-q"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
            timeout=120,
        )
        output = result.stdout + result.stderr

        passed = output.count(" PASSED")
        failed = output.count(" FAILED")
        errors = output.count(" ERROR")
        skipped = output.count(" SKIPPED")

        lines = output.strip().split("\n")
        duration = "?"
        for line in reversed(lines):
            if "second" in line or "passed" in line:
                duration = line.strip()
                break

        return {
            "exit_code": result.returncode,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "skipped": skipped,
            "duration_s": duration,
            "output": output[-5000:],
        }
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "passed": 0, "failed": 0, "errors": 1, "skipped": 0,
                "duration_s": "timeout", "output": "Test run timed out after 120s"}
    except Exception as exc:
        return {"exit_code": -1, "passed": 0, "failed": 0, "errors": 1, "skipped": 0,
                "duration_s": "error", "output": str(exc)}


@router.post("/health-tests")
async def health_tests():
    checks = []

    # DB
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(text("SELECT 1"))
        checks.append({"name": "PostgreSQL", "ok": True, "detail": "connected"})
    except Exception as e:
        checks.append({"name": "PostgreSQL", "ok": False, "detail": str(e)})

    # Redis
    try:
        r = get_redis()
        await r.ping()
        checks.append({"name": "Redis", "ok": True, "detail": "connected"})
    except Exception as e:
        checks.append({"name": "Redis", "ok": False, "detail": str(e)})

    # Qdrant
    try:
        from mir.search.vector_store import QdrantManager
        mgr = QdrantManager()
        info = mgr.get_collection_info("posts")
        checks.append({"name": "Qdrant", "ok": True, "detail": f"posts collection: {info.get('points_count', '?')} vectors"})
    except Exception as e:
        checks.append({"name": "Qdrant", "ok": False, "detail": str(e)})

    # Tables exist
    try:
        async with AsyncSessionLocal() as db:
            tables = (await db.execute(text(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            ))).scalars().all()
            expected = {"posts", "accounts", "tags", "crawl_state", "communities", "api_keys", "post_tags"}
            missing = expected - set(tables)
            if missing:
                checks.append({"name": "DB Tables", "ok": False, "detail": f"missing: {missing}"})
            else:
                checks.append({"name": "DB Tables", "ok": True, "detail": f"{len(tables)} tables present"})
    except Exception as e:
        checks.append({"name": "DB Tables", "ok": False, "detail": str(e)})

    # Crawl state
    try:
        async with AsyncSessionLocal() as db:
            count = (await db.execute(text("SELECT COUNT(*) FROM crawl_state"))).scalar()
            active = (await db.execute(text("SELECT COUNT(*) FROM crawl_state WHERE status = 'active'"))).scalar()
            checks.append({"name": "Crawl State", "ok": count > 0, "detail": f"{count} total, {active} active"})
    except Exception as e:
        checks.append({"name": "Crawl State", "ok": False, "detail": str(e)})

    # Post count
    try:
        async with AsyncSessionLocal() as db:
            count = (await db.execute(text("SELECT COUNT(*) FROM posts"))).scalar()
            checks.append({"name": "Posts", "ok": True, "detail": f"{count} posts in DB"})
    except Exception as e:
        checks.append({"name": "Posts", "ok": False, "detail": str(e)})

    passed = sum(1 for c in checks if c["ok"])
    return {"checks": checks, "passed": passed, "total": len(checks)}


@router.get("/seeds")
async def get_seeds():
    async with AsyncSessionLocal() as db:
        existing = set((await db.execute(
            text("SELECT blog_name FROM crawl_state")
        )).scalars().all())
    all_seeds = set(SEED_BLOGS)
    return {
        "seeds": SEED_BLOGS_BY_GENRE,
        "total": len(all_seeds),
        "existing_count": len(all_seeds & existing),
        "new_count": len(all_seeds - existing),
    }


@router.post("/seed-crawl")
async def seed_crawl():
    async with AsyncSessionLocal() as db:
        existing = set((await db.execute(
            text("SELECT blog_name FROM crawl_state")
        )).scalars().all())

        added = 0
        skipped = 0
        for blog in SEED_BLOGS:
            if blog in existing:
                skipped += 1
                continue
            await db.execute(
                insert(CrawlState).values(blog_name=blog, status="pending", fail_count=0)
            )
            added += 1

        await db.commit()
        total = (await db.execute(text("SELECT COUNT(*) FROM crawl_state"))).scalar()

    return {"added": added, "skipped": skipped, "total": total}


@router.post("/trigger-crawl")
async def trigger_crawl():
    from mir.workers.tasks import crawl_blog
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(text(
            "SELECT blog_name FROM crawl_state WHERE status IN ('active', 'pending')"
        ))).scalars().all()

    enqueued = 0
    for blog in rows:
        crawl_blog.delay(blog)
        enqueued += 1
    return {"enqueued": enqueued}
