from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path

import aiohttp
import imagehash
from PIL import Image

log = logging.getLogger(__name__)

_DEFAULT_CONCURRENCY = 20


class ImageDownloader:
    """
    Async image downloader with perceptual-hash deduplication and EXIF stripping.

    Images are saved to:
        {base_dir}/{blog_name}/{post_id}/{phash}.jpg

    Deduplication is done by phash: if a file with the same hash already exists
    in the post directory the download is skipped entirely.

    EXIF metadata is stripped by reconstructing the image from raw pixel data
    before writing, so no sensor / location information is persisted.
    """

    def __init__(
        self,
        session: aiohttp.ClientSession | None = None,
        concurrency: int = _DEFAULT_CONCURRENCY,
        base_dir: Path = Path("data/raw_images"),
    ) -> None:
        self._session = session
        self._semaphore = asyncio.Semaphore(concurrency)
        self._base_dir = base_dir

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _best_url(photo: dict) -> str | None:
        """Return the largest available image URL for a photo object."""
        alt_sizes: list[dict] = photo.get("alt_sizes") or []
        if alt_sizes:
            # Tumblr returns alt_sizes sorted largest-first
            return alt_sizes[0].get("url")
        original = photo.get("original_size") or {}
        return original.get("url")

    def _out_path(self, blog_name: str, post_id: str | int, phash: str) -> Path:
        return self._base_dir / blog_name / str(post_id) / f"{phash}.jpg"

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    async def _download_one(
        self,
        url: str,
        blog_name: str,
        post_id: str | int,
    ) -> Path | None:
        async with self._semaphore:
            # --- fetch ---
            try:
                assert self._session is not None
                async with self._session.get(
                    url, timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    resp.raise_for_status()
                    raw = await resp.read()
            except Exception as exc:
                log.warning("Failed to download %s: %s", url, exc)
                return None

            # --- open & hash ---
            try:
                img = Image.open(io.BytesIO(raw))
                img.load()  # force decode before closing the BytesIO
                ph = str(imagehash.phash(img))
            except Exception as exc:
                log.warning("Could not open/hash image from %s: %s", url, exc)
                return None

            # --- dedup ---
            out_path = self._out_path(blog_name, post_id, ph)
            if out_path.exists():
                log.debug("Duplicate phash=%s skipped for post %s", ph, post_id)
                return None

            # --- strip EXIF by rebuilding from pixel data ---
            try:
                # Convert to RGB so JPEG encoding always works (no alpha channel)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                clean = Image.new("RGB", img.size)
                clean.putdata(list(img.getdata()))

                out_path.parent.mkdir(parents=True, exist_ok=True)
                clean.save(out_path, format="JPEG", quality=95)
            except Exception as exc:
                log.warning("Could not save image (phash=%s): %s", ph, exc)
                return None

            log.debug("Saved %s", out_path)
            return out_path

    async def download_post_images(self, post: dict, blog_name: str) -> list[Path]:
        """Download all photos in a post, returning paths of newly saved files."""
        post_id = post.get("id", "unknown")
        photos: list[dict] = post.get("photos") or []

        tasks = [
            self._download_one(url, blog_name, post_id)
            for photo in photos
            if (url := self._best_url(photo))
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        paths: list[Path] = []
        for r in results:
            if isinstance(r, Path):
                paths.append(r)
            elif isinstance(r, Exception):
                log.warning("Image download task raised: %s", r)
        return paths
