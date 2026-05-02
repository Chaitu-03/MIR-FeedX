from __future__ import annotations

import html as _html_stdlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import bleach
from sqlalchemy import false, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from mir.config import settings
from mir.db.models import Account, CrawlState, Post, PostTag, Tag
from mir.ingestion.client import TumblrClient
from mir.ingestion.images import ImageDownloader

log = logging.getLogger(__name__)

EXCLUDED_POST_TYPES: frozenset[str] = frozenset({"video"})

_STORAGE_CAP_BYTES = 100 * 1024**3  # 100 GiB
_RAW_IMAGE_DIR = Path("data/raw_images")

_STYLE_SCRIPT_RE = re.compile(r"<(style|script)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)


def _strip_html(html: str) -> str:
    """Lightweight HTML → plain text (no ML models required)."""
    if not html:
        return ""
    no_blocks = _STYLE_SCRIPT_RE.sub("", html)
    stripped = bleach.clean(no_blocks, tags=[], strip=True)
    return _html_stdlib.unescape(stripped).strip()


def _extract_body(post: dict) -> tuple[str | None, str | None]:
    """Return (body_raw, body_clean) based on post type."""
    ptype = post.get("type", "")
    raw: str | None = None
    if ptype == "text":
        raw = post.get("body") or post.get("title") or None
    elif ptype == "photo":
        raw = post.get("caption") or None
    elif ptype == "quote":
        raw = post.get("text") or None
    elif ptype == "link":
        raw = post.get("description") or post.get("title") or None
    elif ptype == "answer":
        raw = (post.get("question") or "") + "\n" + (post.get("answer") or "")
    elif ptype == "chat":
        lines = [f"{d.get('label','')} {d.get('phrase','')}" for d in post.get("dialogue", [])]
        raw = "\n".join(lines) or None
    clean = _strip_html(raw) if raw else None
    return raw, clean


def _extract_image_urls(post: dict) -> list[str]:
    urls: list[str] = []
    for photo in post.get("photos") or []:
        orig = photo.get("original_size") or {}
        url = orig.get("url")
        if url:
            urls.append(url)
    return urls


class Crawler:
    """
    Multi-run resumable crawler.

    Each call to `crawl()` picks up exactly where the previous run left off using
    the per-blog `last_timestamp` cursor stored in CrawlState. All blogs stop being
    crawled once the indexed SFW post count reaches settings.TARGET_POST_COUNT.
    """

    def __init__(
        self,
        session: AsyncSession,
        client: TumblrClient,
        downloader: ImageDownloader,
    ) -> None:
        self._db = session
        self._client = client
        self._downloader = downloader

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _indexed_count(self) -> int:
        """Live count of SFW, non-video indexed posts."""
        result = await self._db.scalar(
            select(func.count()).select_from(Post).where(Post.nsfw == false())
        )
        return result or 0

    async def _get_or_create_crawl_state(self, blog_name: str) -> CrawlState:
        state = await self._db.scalar(
            select(CrawlState).where(CrawlState.blog_name == blog_name)
        )
        if state is None:
            state = CrawlState(blog_name=blog_name, status="pending", fail_count=0)
            self._db.add(state)
            await self._db.flush()
        return state

    async def _pause_active_blogs(self) -> None:
        """Set all currently 'active' blogs to 'paused' so the next run can resume them."""
        result = await self._db.execute(
            select(CrawlState).where(CrawlState.status == "active")
        )
        for state in result.scalars():
            state.status = "paused"
        await self._db.commit()

    async def _discover_blogs(self, post: dict) -> None:
        """Seed CrawlState for reblog sources found in a post."""
        candidates: set[str] = set()
        for field in ("reblogged_from_name", "reblogged_root_name"):
            name = post.get(field)
            if name:
                candidates.add(name)
        trail = post.get("trail") or []
        for entry in trail:
            blog = (entry.get("blog") or {}).get("name")
            if blog:
                candidates.add(blog)

        for blog_name in candidates:
            exists = await self._db.scalar(
                select(CrawlState.id).where(CrawlState.blog_name == blog_name)
            )
            if exists is None:
                self._db.add(
                    CrawlState(blog_name=blog_name, status="pending", fail_count=0)
                )

    async def _get_or_create_account(self, blog_name: str) -> int:
        """Return accounts.id for blog_name, creating the row if absent."""
        stmt = (
            pg_insert(Account)
            .values(blog_name=blog_name)
            .on_conflict_do_nothing(index_elements=["blog_name"])
            .returning(Account.id)
        )
        row = await self._db.scalar(stmt)
        if row is None:
            # Row already existed — fetch it
            row = await self._db.scalar(
                select(Account.id).where(Account.blog_name == blog_name)
            )
        return row  # type: ignore[return-value]

    async def _persist_post(self, post: dict, blog_name: str) -> int | None:
        """
        Upsert Account + Post + Tags for one Tumblr post dict.
        Returns the Post.id on success, None if tumblr_id missing.
        """
        tumblr_id = post.get("id")
        if not tumblr_id:
            return None

        account_id = await self._get_or_create_account(blog_name)

        body_raw, body_clean = _extract_body(post)
        image_urls = _extract_image_urls(post)
        published_at: datetime | None = None
        ts = post.get("timestamp")
        if ts:
            published_at = datetime.fromtimestamp(int(ts), tz=timezone.utc)

        stmt = (
            pg_insert(Post)
            .values(
                tumblr_id=int(tumblr_id),
                account_id=account_id,
                post_type=post.get("type"),
                body_raw=body_raw,
                body_clean=body_clean,
                image_urls=image_urls or None,
                note_count=int(post.get("note_count") or 0),
                reblog_key=post.get("reblog_key"),
                published_at=published_at,
            )
            .on_conflict_do_update(
                index_elements=["tumblr_id"],
                set_={
                    "note_count": int(post.get("note_count") or 0),
                    "image_urls": image_urls or None,
                },
            )
            .returning(Post.id)
        )
        post_id = await self._db.scalar(stmt)

        # Upsert tags
        tag_names: list[str] = [t for t in (post.get("tags") or []) if t]
        for tag_name in tag_names:
            tag_stmt = (
                pg_insert(Tag)
                .values(name=tag_name)
                .on_conflict_do_update(
                    index_elements=["name"],
                    set_={"usage_count": Tag.usage_count + 1},
                )
                .returning(Tag.id)
            )
            tag_id = await self._db.scalar(tag_stmt)
            if post_id and tag_id:
                pt_stmt = (
                    pg_insert(PostTag)
                    .values(post_id=post_id, tag_id=tag_id)
                    .on_conflict_do_nothing()
                )
                await self._db.execute(pt_stmt)

        await self._db.flush()
        return post_id

    async def _process_post(self, post: dict, blog_name: str) -> None:
        """Persist post metadata to DB, then download raw images."""
        post_id = await self._persist_post(post, blog_name)
        if post_id is None:
            log.warning("Skipping post with no id from %s", blog_name)
            return
        await self._downloader.download_post_images(post, blog_name)

    async def _check_storage(self) -> None:
        if not _RAW_IMAGE_DIR.exists():
            return
        total = sum(f.stat().st_size for f in _RAW_IMAGE_DIR.rglob("*") if f.is_file())
        if total > _STORAGE_CAP_BYTES:
            log.warning(
                "Raw image storage exceeds 100 GiB (current: %.1f GiB)",
                total / 1024**3,
            )

    # ------------------------------------------------------------------
    # Blog-level crawl
    # ------------------------------------------------------------------

    async def _crawl_blog(self, state: CrawlState) -> bool:
        """
        Paginate a single blog backwards in time.

        Returns True if TARGET_POST_COUNT was reached (caller should pause
        remaining blogs and exit), False otherwise.
        """
        blog_name = state.blog_name
        state.status = "active"
        state.last_crawled_at = datetime.now(timezone.utc)
        await self._db.commit()

        consecutive_failures = state.fail_count

        while True:
            # --- threshold guard (pre-fetch) ---
            if await self._indexed_count() >= settings.target_post_count:
                state.status = "paused"
                await self._db.commit()
                log.info(
                    "Threshold reached before fetching next page. Pausing %s.", blog_name
                )
                return True

            # --- fetch page ---
            try:
                posts = await self._client.get_blog_posts(
                    blog_name, before=state.last_timestamp
                )
            except Exception as exc:
                consecutive_failures += 1
                state.fail_count = consecutive_failures
                if consecutive_failures >= 3:
                    state.status = "dead"
                    await self._db.commit()
                    log.error(
                        "Blog %s marked dead after %d consecutive failures: %s",
                        blog_name,
                        consecutive_failures,
                        exc,
                    )
                else:
                    await self._db.commit()
                    log.warning(
                        "Blog %s fetch failed (attempt %d): %s",
                        blog_name,
                        consecutive_failures,
                        exc,
                    )
                return False

            # --- blog exhausted ---
            if not posts:
                state.status = "done"
                await self._db.commit()
                log.info("Blog %s exhausted.", blog_name)
                return False

            # Reset failure counter on any successful fetch
            consecutive_failures = 0
            state.fail_count = 0

            # --- video filtering at API response level ---
            filtered: list[dict] = []
            for post in posts:
                if post.get("type") in EXCLUDED_POST_TYPES:
                    log.debug("Skipping video post %s from %s", post.get("id"), blog_name)
                    continue
                filtered.append(post)

            # --- advance cursor BEFORE processing (crash-safe) ---
            timestamps = [p["timestamp"] for p in posts if p.get("timestamp") is not None]
            if timestamps:
                state.last_timestamp = min(timestamps)
                await self._db.commit()

            # --- process non-video posts ---
            for post in filtered:
                await self._process_post(post, blog_name)
                await self._discover_blogs(post)

            state.last_crawled_at = datetime.now(timezone.utc)
            await self._db.commit()
            await self._check_storage()

            # --- threshold guard (post-page) ---
            if await self._indexed_count() >= settings.target_post_count:
                state.status = "paused"
                await self._db.commit()
                log.info(
                    "Threshold reached after processing page. Pausing %s.", blog_name
                )
                return True

    # ------------------------------------------------------------------
    # Tag-level crawl
    # ------------------------------------------------------------------

    async def _crawl_tag(self, tag: str) -> bool:
        """Paginate /v2/tagged backwards. Returns True if threshold hit."""
        before: int | None = None

        while True:
            if await self._indexed_count() >= settings.target_post_count:
                log.info("Threshold reached. Stopping tag crawl for '%s'.", tag)
                return True

            try:
                posts = await self._client.get_tagged_posts(tag, before=before)
            except Exception as exc:
                log.warning("Failed to fetch tag '%s': %s", tag, exc)
                return False

            if not posts:
                return False

            # Advance cursor before processing
            timestamps = [p["timestamp"] for p in posts if p.get("timestamp") is not None]
            if timestamps:
                before = min(timestamps)

            filtered = [p for p in posts if p.get("type") not in EXCLUDED_POST_TYPES]

            for post in filtered:
                blog_name = (
                    post.get("blog_name")
                    or (post.get("blog") or {}).get("name")
                    or "unknown"
                )
                await self._process_post(post, blog_name)
                await self._discover_blogs(post)

                # Seed the blog for future runs
                exists = await self._db.scalar(
                    select(CrawlState.id).where(CrawlState.blog_name == blog_name)
                )
                if exists is None:
                    self._db.add(
                        CrawlState(blog_name=blog_name, status="pending", fail_count=0)
                    )

            await self._db.commit()

            if await self._indexed_count() >= settings.target_post_count:
                log.info("Threshold reached after tag page for '%s'.", tag)
                return True

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def crawl(
        self,
        seed_blogs: list[str] | None = None,
        seed_tags: list[str] | None = None,
    ) -> None:
        """
        Run one crawl iteration.

        Order of operations:
        1. Exit immediately (no-op) if TARGET_POST_COUNT is already reached.
        2. Seed any new blogs / tags into CrawlState.
        3. Crawl seed tags via /v2/tagged.
        4. Crawl blogs in priority order: pending → paused (resumable) → active.
        """
        # 1. Global threshold guard — must be the very first thing
        if await self._indexed_count() >= settings.target_post_count:
            log.info(
                "Threshold already reached (%d). Crawl is a no-op.",
                settings.target_post_count,
            )
            return

        # 2. Seed blogs
        for blog_name in seed_blogs or []:
            await self._get_or_create_crawl_state(blog_name)
        await self._db.commit()

        # 3. Crawl seed tags
        for tag in seed_tags or []:
            if await self._crawl_tag(tag):
                await self._pause_active_blogs()
                log.info(
                    "Crawl complete. Final indexed count: %d.", await self._indexed_count()
                )
                return

        # 4. Crawl blogs: pending first (new), then paused (resumable), then active
        for status in ("pending", "paused", "active"):
            result = await self._db.execute(
                select(CrawlState).where(CrawlState.status == status)
            )
            states = list(result.scalars())

            for state in states:
                if await self._crawl_blog(state):
                    await self._pause_active_blogs()
                    log.info(
                        "Crawl complete. Final indexed count: %d.",
                        await self._indexed_count(),
                    )
                    return

        log.info(
            "Crawl run finished. Indexed count: %d.", await self._indexed_count()
        )
