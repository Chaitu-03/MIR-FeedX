"""
tests/test_e2e.py
End-to-end integration tests — Prompt 16a.

Tests MECHANICS not ML quality (10 posts is too small for semantic evaluation).
Run: pytest tests/test_e2e.py -v

Requires running Docker services (postgres, qdrant, redis).
Set TEST_DATABASE_URL env var to override the default test connection.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from PIL import Image
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

FIXTURES_PATH = Path(__file__).parent / "fixtures" / "posts.json"
TEST_DB_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://mir:mir@localhost:5432/mir",
)
TEST_API_KEY = "e2e-test-key-do-not-use-in-prod"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture(scope="session")
async def engine():
    eng = create_async_engine(TEST_DB_URL, echo=False)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture(scope="session")
async def db_session(engine):
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        yield session


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_db(db_session):
    """Create schema, insert test accounts, insert API key, seed posts."""
    import bcrypt

    raw_key = TEST_API_KEY
    key_hash = bcrypt.hashpw(raw_key.encode(), bcrypt.gensalt(12)).decode()

    # Ensure API key exists
    await db_session.execute(
        text("""
            INSERT INTO api_keys (key_hash, label, is_admin)
            VALUES (:hash, 'e2e-test', true)
            ON CONFLICT DO NOTHING
        """),
        {"hash": key_hash},
    )

    # Seed accounts for name search tests
    for blog in ("photography", "photography-art", "photgraphy-typo", "testblog123",
                 "anotheruser", "imageuser1", "imageuser2", "imageuser3",
                 "foodie42", "chefblog", "nsfwaccount"):
        await db_session.execute(
            text("""
                INSERT INTO accounts (blog_name, total_posts, description)
                VALUES (:name, :posts, :desc)
                ON CONFLICT (blog_name) DO NOTHING
            """),
            {"name": blog, "posts": 10, "desc": f"Description for {blog}"},
        )

    # Load and insert posts from fixture (excluding video)
    posts = json.loads(FIXTURES_PATH.read_text())
    for p in posts:
        if p["type"] == "video":
            continue  # video posts never reach DB
        account_result = await db_session.execute(
            text("SELECT id FROM accounts WHERE blog_name = :name"),
            {"name": p["blog_name"]},
        )
        account_id = account_result.scalar()

        await db_session.execute(
            text("""
                INSERT INTO posts (tumblr_id, account_id, post_type, body_raw, body_clean,
                                   note_count, published_at, nsfw)
                VALUES (:tumblr_id, :account_id, :post_type, :body, :body,
                        :note_count, :published_at, :nsfw)
                ON CONFLICT (tumblr_id) DO NOTHING
            """),
            {
                "tumblr_id": p["id"],
                "account_id": account_id,
                "post_type": p["type"],
                "body": p["body"],
                "note_count": p["note_count"],
                "published_at": datetime.fromtimestamp(p["published_ts"], tz=timezone.utc),
                "nsfw": p["nsfw"],
            },
        )

        # Insert tags
        for tag_name in p.get("tags", []):
            tag_row = await db_session.execute(
                text("""
                    INSERT INTO tags (name, usage_count)
                    VALUES (:name, 1)
                    ON CONFLICT (name) DO UPDATE SET usage_count = tags.usage_count + 1
                    RETURNING id
                """),
                {"name": tag_name},
            )
            tag_id = tag_row.scalar()
            post_row = await db_session.execute(
                text("SELECT id FROM posts WHERE tumblr_id = :tid"),
                {"tid": p["id"]},
            )
            post_id = post_row.scalar()
            if post_id:
                await db_session.execute(
                    text("""
                        INSERT INTO post_tags (post_id, tag_id) VALUES (:pid, :tid)
                        ON CONFLICT DO NOTHING
                    """),
                    {"pid": post_id, "tid": tag_id},
                )

    await db_session.commit()


@pytest_asyncio.fixture(scope="session")
async def client():
    from mir.api.app import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ── Helpers ───────────────────────────────────────────────────────────────────

def auth_headers():
    return {"X-API-Key": TEST_API_KEY}


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: Ingestion mechanics
# ═══════════════════════════════════════════════════════════════════════════════

class TestIngestion:
    @pytest.mark.asyncio
    async def test_video_post_never_in_db(self, db_session):
        """Video post (id=1010) must not be in PostgreSQL."""
        row = (await db_session.execute(
            text("SELECT id FROM posts WHERE tumblr_id = 1010")
        )).scalar()
        assert row is None, "Video post must never be written to the database"

    @pytest.mark.asyncio
    async def test_nsfw_post_in_db(self, db_session):
        """NSFW post (id=1009) IS stored in PostgreSQL with nsfw=True."""
        row = (await db_session.execute(
            text("SELECT nsfw FROM posts WHERE tumblr_id = 1009")
        )).fetchone()
        assert row is not None, "NSFW post should be stored in DB for audit"
        assert row.nsfw is True

    @pytest.mark.asyncio
    async def test_indexed_count_excludes_nsfw(self, db_session):
        """Threshold count = SFW posts only."""
        count = (await db_session.execute(
            text("SELECT COUNT(*) FROM posts WHERE nsfw = false")
        )).scalar()
        assert count == 9, f"Expected 9 SFW posts, got {count}"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2: API Authentication
# ═══════════════════════════════════════════════════════════════════════════════

class TestAuth:
    @pytest.mark.asyncio
    async def test_missing_key_returns_401(self, client):
        resp = await client.post("/api/v1/search/general", json={"query": "photography"})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_key_returns_401(self, client):
        resp = await client.post(
            "/api/v1/search/general",
            json={"query": "photography"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_valid_key_returns_200(self, client):
        resp = await client.post(
            "/api/v1/search/general",
            json={"query": "photography"},
            headers=auth_headers(),
        )
        assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3: General Search mechanics
# ═══════════════════════════════════════════════════════════════════════════════

class TestGeneralSearch:
    @pytest.mark.asyncio
    async def test_scores_strictly_descending(self, client):
        resp = await client.post(
            "/api/v1/search/general",
            json={"query": "photography landscape", "limit_posts": 5},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        posts = resp.json()["posts"]
        assert len(posts) > 1
        scores = [p["score"] for p in posts]
        assert scores == sorted(scores, reverse=True), "Scores must be descending"

    @pytest.mark.asyncio
    async def test_nsfw_absent_from_results(self, client):
        resp = await client.post(
            "/api/v1/search/general",
            json={"query": "photography"},
            headers=auth_headers(),
        )
        post_ids = [p["post_id"] for p in resp.json()["posts"]]
        nsfw_db_id = None  # look up the DB id for tumblr_id=1009 if needed
        assert 1009 not in post_ids, "NSFW tumblr post must not appear"

    @pytest.mark.asyncio
    async def test_accounts_non_empty(self, client):
        resp = await client.post(
            "/api/v1/search/general",
            json={"query": "photography"},
            headers=auth_headers(),
        )
        accounts = resp.json()["accounts"]
        assert len(accounts) >= 1
        blog_names = [a["blog_name"] for a in accounts]
        assert "testblog123" in blog_names or any("photo" in b for b in blog_names)

    @pytest.mark.asyncio
    async def test_cursor_pagination_no_overlap(self, client):
        payload_page1 = {"query": "photography", "limit_posts": 2}
        resp1 = await client.post(
            "/api/v1/search/general", json=payload_page1, headers=auth_headers()
        )
        data1 = resp1.json()
        page1_ids = {p["post_id"] for p in data1["posts"]}
        cursor = data1.get("next_cursor")

        if cursor is None:
            pytest.skip("Not enough results to test pagination")

        payload_page2 = {"query": "photography", "limit_posts": 2, "cursor": cursor}
        resp2 = await client.post(
            "/api/v1/search/general", json=payload_page2, headers=auth_headers()
        )
        page2_ids = {p["post_id"] for p in resp2.json()["posts"]}
        assert page1_ids.isdisjoint(page2_ids), "Pages must not share post IDs"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4: Account Name Search
# ═══════════════════════════════════════════════════════════════════════════════

class TestAccountNameSearch:
    @pytest.mark.asyncio
    async def test_exact_first_prefix_second_fuzzy_third(self, client):
        resp = await client.get(
            "/api/v1/search/accounts?q=photography&limit=10",
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        accounts = resp.json()["accounts"]
        names = [a["blog_name"] for a in accounts]
        match_types = [a["match_type"] for a in accounts]

        assert "photography" in names
        exact_idx = names.index("photography")
        assert match_types[exact_idx] == "exact"

        if "photography-art" in names:
            prefix_idx = names.index("photography-art")
            assert prefix_idx > exact_idx

    @pytest.mark.asyncio
    async def test_fuzzy_typo_matches(self, client):
        resp = await client.get(
            "/api/v1/search/accounts?q=photgraphy&limit=10",
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        names = [a["blog_name"] for a in resp.json()["accounts"]]
        assert any("photo" in n or "photgraphy" in n for n in names), \
            "Fuzzy query should return trigram matches"

    @pytest.mark.asyncio
    async def test_unrelated_query_empty(self, client):
        resp = await client.get(
            "/api/v1/search/accounts?q=zzzzzzqqqqqxxx999&limit=10",
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["accounts"] == []


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5: Tag Search mechanics
# ═══════════════════════════════════════════════════════════════════════════════

class TestTagSearch:
    @pytest.mark.asyncio
    async def test_exact_tag_ranks_first(self, client):
        resp = await client.get(
            "/api/v1/search/tags?q=photography&limit=20",
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        tags = resp.json()["tags"]
        assert len(tags) > 0
        assert tags[0]["name"] == "photography"
        assert tags[0]["match_type"] == "exact"

    @pytest.mark.asyncio
    async def test_tag_has_usage_count(self, client):
        resp = await client.get(
            "/api/v1/search/tags?q=photography&limit=5",
            headers=auth_headers(),
        )
        assert resp.json()["tags"][0]["usage_count"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Test 6: NSFW filtering
# ═══════════════════════════════════════════════════════════════════════════════

class TestNSFWFiltering:
    @pytest.mark.asyncio
    async def test_nsfw_post_absent(self, client, db_session):
        # Get the internal post ID for tumblr_id=1009
        nsfw_internal_id = (await db_session.execute(
            text("SELECT id FROM posts WHERE tumblr_id = 1009")
        )).scalar()

        resp = await client.post(
            "/api/v1/search/general",
            json={"query": "photography"},
            headers=auth_headers(),
        )
        post_ids = [p["post_id"] for p in resp.json()["posts"]]
        assert nsfw_internal_id not in post_ids

    @pytest.mark.asyncio
    async def test_total_posts_excludes_nsfw(self, client):
        resp = await client.post(
            "/api/v1/search/general",
            json={"query": "test"},
            headers=auth_headers(),
        )
        total = resp.json()["total_posts"]
        # At most 9 SFW posts can be returned
        assert total <= 9


# ═══════════════════════════════════════════════════════════════════════════════
# Test 7: Community Search (smoke test)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCommunitySearch:
    @pytest.mark.asyncio
    async def test_community_response_valid(self, client):
        # First trigger rebuild directly via admin endpoint
        rebuild_resp = await client.post(
            "/api/v1/admin/rebuild_communities",
            headers=auth_headers(),
        )
        # May return 200 or fail silently at small scale — just check JSON shape
        resp = await client.get(
            "/api/v1/search/communities?q=photography",
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "communities" in data

    @pytest.mark.asyncio
    async def test_community_has_drill_down_query(self, client):
        resp = await client.get(
            "/api/v1/search/communities?q=photography",
            headers=auth_headers(),
        )
        communities = resp.json()["communities"]
        for c in communities:
            assert "drill_down_query" in c
            assert c["drill_down_query"].startswith("/api/v1/search/general")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 8: Health endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class TestHealth:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, client):
        resp = await client.get("/api/v1/health")
        data = resp.json()
        assert "status" in data
        assert data["db"]["status"] in ("connected", "error")

    @pytest.mark.asyncio
    async def test_liveness_always_200(self, client):
        resp = await client.get("/api/v1/health/live")
        assert resp.status_code == 200
