"""
Unit tests for mir/ingestion/crawler.py.

All DB, client, and downloader interactions are mocked so the tests run
without any external services.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from mir.db.models import CrawlState
from mir.ingestion.crawler import Crawler, EXCLUDED_POST_TYPES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_post(
    post_id: int,
    post_type: str = "photo",
    timestamp: int = 1000,
    photos: list | None = None,
) -> dict:
    return {
        "id": post_id,
        "type": post_type,
        "timestamp": timestamp,
        "photos": photos or [],
        "reblogged_from_name": None,
        "reblogged_root_name": None,
    }


def _make_state(
    blog_name: str = "test-blog",
    status: str = "pending",
    last_timestamp: int | None = None,
    fail_count: int = 0,
) -> CrawlState:
    state = CrawlState()
    state.blog_name = blog_name
    state.status = status
    state.last_timestamp = last_timestamp
    state.fail_count = fail_count
    return state


def _make_crawler(
    session: AsyncMock | None = None,
    client: AsyncMock | None = None,
    downloader: AsyncMock | None = None,
) -> Crawler:
    session = session or AsyncMock()
    client = client or AsyncMock()
    downloader = downloader or AsyncMock()
    downloader.download_post_images = AsyncMock(return_value=[])
    return Crawler(session=session, client=client, downloader=downloader)


# ---------------------------------------------------------------------------
# 1. Video post filtering
# ---------------------------------------------------------------------------

class TestVideoFiltering:
    """Video posts must be dropped at the API-response level — never saved or enqueued."""

    @pytest.mark.asyncio
    async def test_video_posts_never_reach_process_post(self):
        client = AsyncMock()
        downloader = AsyncMock()
        downloader.download_post_images = AsyncMock(return_value=[])

        video = _make_post(1, post_type="video", timestamp=1000)
        photo = _make_post(2, post_type="photo", timestamp=900)

        # First call returns one video + one photo; second call signals exhaustion
        client.get_blog_posts = AsyncMock(side_effect=[[video, photo], []])

        crawler = _make_crawler(client=client, downloader=downloader)
        state = _make_state()

        with patch.object(crawler, "_indexed_count", AsyncMock(return_value=0)):
            await crawler._crawl_blog(state)

        # download_post_images should be called exactly once — for the photo only
        assert downloader.download_post_images.call_count == 1
        processed_post = downloader.download_post_images.call_args[0][0]
        assert processed_post["type"] == "photo"
        assert processed_post["id"] == 2

    @pytest.mark.asyncio
    async def test_all_video_page_produces_no_downloads(self):
        client = AsyncMock()
        downloader = AsyncMock()
        downloader.download_post_images = AsyncMock(return_value=[])

        all_videos = [_make_post(i, post_type="video", timestamp=1000 - i) for i in range(5)]
        client.get_blog_posts = AsyncMock(side_effect=[all_videos, []])

        crawler = _make_crawler(client=client, downloader=downloader)
        state = _make_state()

        with patch.object(crawler, "_indexed_count", AsyncMock(return_value=0)):
            await crawler._crawl_blog(state)

        downloader.download_post_images.assert_not_called()

    @pytest.mark.asyncio
    async def test_excluded_post_types_constant_contains_video(self):
        assert "video" in EXCLUDED_POST_TYPES


# ---------------------------------------------------------------------------
# 2. Threshold check exits early without touching CrawlState
# ---------------------------------------------------------------------------

class TestThresholdGuard:
    """crawl() must exit immediately — before any CrawlState read/write — when threshold met."""

    @pytest.mark.asyncio
    async def test_crawl_is_noop_at_threshold(self):
        session = AsyncMock()
        client = AsyncMock()
        crawler = _make_crawler(session=session, client=client)

        # Threshold already met
        with patch.object(
            crawler, "_indexed_count", AsyncMock(return_value=50_000)
        ):
            await crawler.crawl(seed_blogs=["any-blog"], seed_tags=["any-tag"])

        # No API calls
        client.get_blog_posts.assert_not_called()
        client.get_tagged_posts.assert_not_called()
        # No DB mutations (execute is used for CrawlState queries)
        session.execute.assert_not_called()
        session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_crawl_noop_does_not_create_crawlstate_for_seeds(self):
        """Seed blogs must NOT be inserted into CrawlState when threshold is already met."""
        session = AsyncMock()
        crawler = _make_crawler(session=session)

        with patch.object(
            crawler, "_indexed_count", AsyncMock(return_value=999_999)
        ):
            await crawler.crawl(seed_blogs=["blog-a", "blog-b"])

        session.add.assert_not_called()
        session.flush.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Cursor written before post processing
# ---------------------------------------------------------------------------

class TestCursorBeforeProcessing:
    """last_timestamp must be committed before _process_post is called."""

    @pytest.mark.asyncio
    async def test_cursor_advanced_before_process_post_called(self):
        client = AsyncMock()
        posts = [
            _make_post(1, timestamp=1000),
            _make_post(2, timestamp=500),   # ← oldest; should become the cursor
        ]
        client.get_blog_posts = AsyncMock(return_value=posts)

        session = AsyncMock()
        crawler = _make_crawler(session=session, client=client)
        state = _make_state(last_timestamp=None)

        # Capture the cursor value at the moment _process_post is first invoked
        cursor_at_process_time: list[int | None] = []

        async def crashing_process(post: dict, blog_name: str) -> None:
            cursor_at_process_time.append(state.last_timestamp)
            raise RuntimeError("simulated crash mid-processing")

        with patch.object(crawler, "_indexed_count", AsyncMock(return_value=0)):
            with patch.object(crawler, "_process_post", side_effect=crashing_process):
                with pytest.raises(RuntimeError, match="simulated crash"):
                    await crawler._crawl_blog(state)

        # Cursor must have been set before _process_post was invoked
        assert len(cursor_at_process_time) >= 1
        assert cursor_at_process_time[0] == 500  # min(1000, 500)

    @pytest.mark.asyncio
    async def test_commit_called_before_process_post(self):
        """Verify DB commit happens before the processing loop."""
        client = AsyncMock()
        session = AsyncMock()

        posts = [_make_post(1, timestamp=800), _make_post(2, timestamp=300)]
        client.get_blog_posts = AsyncMock(return_value=posts)

        crawler = _make_crawler(session=session, client=client)
        state = _make_state()

        commit_count_at_process: list[int] = []

        async def tracking_process(post: dict, blog_name: str) -> None:
            commit_count_at_process.append(session.commit.call_count)
            raise RuntimeError("stop after first post")

        with patch.object(crawler, "_indexed_count", AsyncMock(return_value=0)):
            with patch.object(crawler, "_process_post", side_effect=tracking_process):
                with pytest.raises(RuntimeError):
                    await crawler._crawl_blog(state)

        # commit must have been called at least twice before _process_post:
        #   commit 1: state.status = "active"
        #   commit 2: state.last_timestamp = <cursor>
        assert commit_count_at_process[0] >= 2


# ---------------------------------------------------------------------------
# 4. 'paused' status set on all active blogs when threshold hit mid-crawl
# ---------------------------------------------------------------------------

class TestPausedStatusOnThreshold:
    """When threshold is hit inside _crawl_blog, the method returns True
    and the caller pauses all remaining active blogs."""

    @pytest.mark.asyncio
    async def test_current_blog_set_to_paused_on_threshold(self):
        client = AsyncMock()
        posts = [_make_post(1, timestamp=1000)]
        client.get_blog_posts = AsyncMock(return_value=posts)

        crawler = _make_crawler(client=client)
        state = _make_state(status="active")

        # Below threshold before fetch, at/above threshold after page processed
        counts = iter([0, 50_000])
        with patch.object(crawler, "_indexed_count", side_effect=lambda: next(counts, 50_000)):
            result = await crawler._crawl_blog(state)

        assert result is True
        assert state.status == "paused"

    @pytest.mark.asyncio
    async def test_pause_active_blogs_sets_status_on_all_active(self):
        """_pause_active_blogs() must flip every 'active' row to 'paused'."""
        session = AsyncMock()

        blog_b = _make_state("blog-b", status="active")
        blog_c = _make_state("blog-c", status="active")

        mock_result = MagicMock()
        mock_result.scalars.return_value = [blog_b, blog_c]
        session.execute = AsyncMock(return_value=mock_result)

        crawler = _make_crawler(session=session)
        await crawler._pause_active_blogs()

        assert blog_b.status == "paused"
        assert blog_c.status == "paused"
        session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_crawl_calls_pause_active_blogs_when_threshold_hit(self):
        """crawl() must call _pause_active_blogs() when _crawl_blog returns True."""
        session = AsyncMock()
        client = AsyncMock()

        seed_state = _make_state("blog-a", status="pending")

        # _get_or_create_crawl_state returns our state
        # DB execute for status='pending' returns [seed_state]
        pending_result = MagicMock()
        pending_result.scalars.return_value = [seed_state]

        empty_result = MagicMock()
        empty_result.scalars.return_value = []

        # First execute: pending blogs; subsequent: paused/active are empty
        session.execute = AsyncMock(
            side_effect=[pending_result, empty_result, empty_result]
        )
        session.scalar = AsyncMock(return_value=None)  # no existing crawl state

        crawler = _make_crawler(session=session, client=client)

        with patch.object(crawler, "_indexed_count", AsyncMock(return_value=0)):
            with patch.object(crawler, "_crawl_blog", AsyncMock(return_value=True)):
                with patch.object(crawler, "_pause_active_blogs", AsyncMock()) as mock_pause:
                    await crawler.crawl(seed_blogs=["blog-a"])

        mock_pause.assert_called_once()


# ---------------------------------------------------------------------------
# 5. Dead-letter promotion after 3 consecutive failures
# ---------------------------------------------------------------------------

class TestDeadLetter:
    """A blog must be marked 'dead' after >= 3 consecutive API failures."""

    @pytest.mark.asyncio
    async def test_dead_after_three_failures(self):
        client = AsyncMock()
        client.get_blog_posts = AsyncMock(side_effect=Exception("API error"))

        crawler = _make_crawler(client=client)
        state = _make_state(fail_count=2)  # 2 prior failures; this run triggers the 3rd

        with patch.object(crawler, "_indexed_count", AsyncMock(return_value=0)):
            result = await crawler._crawl_blog(state)

        assert state.status == "dead"
        assert state.fail_count == 3
        assert result is False

    @pytest.mark.asyncio
    async def test_not_dead_after_two_failures(self):
        client = AsyncMock()
        client.get_blog_posts = AsyncMock(side_effect=Exception("API error"))

        crawler = _make_crawler(client=client)
        state = _make_state(fail_count=1)  # 1 prior failure; this is the 2nd

        with patch.object(crawler, "_indexed_count", AsyncMock(return_value=0)):
            await crawler._crawl_blog(state)

        assert state.status != "dead"
        assert state.fail_count == 2

    @pytest.mark.asyncio
    async def test_fail_count_resets_on_success(self):
        """A successful fetch must reset the consecutive failure counter to 0."""
        client = AsyncMock()
        posts = [_make_post(1, timestamp=500)]
        # First call succeeds; second call returns empty (blog done)
        client.get_blog_posts = AsyncMock(side_effect=[posts, []])

        crawler = _make_crawler(client=client)
        state = _make_state(fail_count=2)  # 2 prior failures — one success resets it

        with patch.object(crawler, "_indexed_count", AsyncMock(return_value=0)):
            await crawler._crawl_blog(state)

        assert state.fail_count == 0
        assert state.status == "done"

    @pytest.mark.asyncio
    async def test_dead_blog_logged_prominently(self, caplog):
        """Marking a blog dead must emit an ERROR-level log (not just WARNING)."""
        import logging

        client = AsyncMock()
        client.get_blog_posts = AsyncMock(side_effect=Exception("timeout"))

        crawler = _make_crawler(client=client)
        state = _make_state(blog_name="dying-blog", fail_count=2)

        with patch.object(crawler, "_indexed_count", AsyncMock(return_value=0)):
            with caplog.at_level(logging.ERROR, logger="mir.ingestion.crawler"):
                await crawler._crawl_blog(state)

        assert any("dead" in record.message.lower() for record in caplog.records)
