"""
Qdrant vector store integration for MIR-FeedX.

QdrantManager owns three collections:
  posts    — 384-d cosine, HNSW m=16/ef=128, NSFW-safe search by default
  accounts — 384-d cosine, blog-level aggregates
  tags     — 384-d cosine, tag embeddings

Usage
-----
    mgr = QdrantManager()
    mgr.ensure_collections()           # idempotent startup hook
    mgr.upsert_post(id, vec, payload)
    results = mgr.search_posts(qvec, filters={"min_notes": 10, "lang": "en"})
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchValue,
    OptimizersConfigDiff,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    Range,
    VectorParams,
)

from mir.config import settings

log = logging.getLogger(__name__)

_EMBED_DIM = 384

POSTS_COLLECTION = "posts"
ACCOUNTS_COLLECTION = "accounts"
TAGS_COLLECTION = "tags"

_ALL_COLLECTIONS = (POSTS_COLLECTION, ACCOUNTS_COLLECTION, TAGS_COLLECTION)


class QdrantManager:
    """
    Manages Qdrant collections for MIR-FeedX.

    Connects via gRPC by default (prefer_grpc=True) for throughput.
    Pass prefer_grpc=False + port=<http_port> for test environments where
    only the HTTP port is reachable (e.g. testcontainers).

    Parameters
    ----------
    host        : Qdrant host (default: settings.qdrant_host)
    grpc_port   : Qdrant gRPC port (default: settings.qdrant_grpc_port)
    port        : Qdrant HTTP port — used when prefer_grpc=False
    prefer_grpc : Use gRPC transport (default True; set False for test containers)
    """

    def __init__(
        self,
        host: str | None = None,
        grpc_port: int | None = None,
        port: int | None = None,
        prefer_grpc: bool = True,
    ) -> None:
        kwargs: dict[str, Any] = {
            "host": host or settings.qdrant_host,
            "prefer_grpc": prefer_grpc,
        }
        if prefer_grpc:
            kwargs["grpc_port"] = grpc_port or settings.qdrant_grpc_port
        else:
            kwargs["port"] = port or settings.qdrant_port

        self._client = QdrantClient(check_compatibility=False, **kwargs)

    # ------------------------------------------------------------------
    # Bootstrap — idempotent
    # ------------------------------------------------------------------

    def ensure_collections(self) -> None:
        """Create collections and payload indexes if absent. Safe to call repeatedly."""
        existing = {c.name for c in self._client.get_collections().collections}

        if POSTS_COLLECTION not in existing:
            self._client.create_collection(
                collection_name=POSTS_COLLECTION,
                vectors_config=VectorParams(size=_EMBED_DIM, distance=Distance.COSINE),
                hnsw_config=HnswConfigDiff(m=16, ef_construct=128),
            )
            log.info("Created collection '%s'", POSTS_COLLECTION)
            self._index_posts_payload()
        else:
            log.debug("Collection '%s' already present", POSTS_COLLECTION)

        if ACCOUNTS_COLLECTION not in existing:
            self._client.create_collection(
                collection_name=ACCOUNTS_COLLECTION,
                vectors_config=VectorParams(size=_EMBED_DIM, distance=Distance.COSINE),
            )
            log.info("Created collection '%s'", ACCOUNTS_COLLECTION)
            self._index_accounts_payload()
        else:
            log.debug("Collection '%s' already present", ACCOUNTS_COLLECTION)

        if TAGS_COLLECTION not in existing:
            self._client.create_collection(
                collection_name=TAGS_COLLECTION,
                vectors_config=VectorParams(size=_EMBED_DIM, distance=Distance.COSINE),
            )
            log.info("Created collection '%s'", TAGS_COLLECTION)
            self._index_tags_payload()
        else:
            log.debug("Collection '%s' already present", TAGS_COLLECTION)

    def _index_posts_payload(self) -> None:
        schema = [
            ("account_id", PayloadSchemaType.INTEGER),
            ("published_at", PayloadSchemaType.FLOAT),
            ("note_count", PayloadSchemaType.INTEGER),
            ("lang", PayloadSchemaType.KEYWORD),
        ]
        for field, stype in schema:
            self._client.create_payload_index(
                collection_name=POSTS_COLLECTION,
                field_name=field,
                field_schema=stype,
            )
        # nsfw: bool — try BoolIndexParams (added in qdrant-client ≥1.10), fall back gracefully
        try:
            from qdrant_client.models import BoolIndexParams  # type: ignore[import]
            self._client.create_payload_index(
                collection_name=POSTS_COLLECTION,
                field_name="nsfw",
                field_schema=BoolIndexParams(),
            )
        except (ImportError, Exception) as exc:
            log.debug("BoolIndexParams unavailable (%s) — nsfw filtered without dedicated index", exc)

    def _index_accounts_payload(self) -> None:
        for field, stype in [
            ("account_id", PayloadSchemaType.INTEGER),
            ("blog_name", PayloadSchemaType.KEYWORD),
        ]:
            self._client.create_payload_index(
                collection_name=ACCOUNTS_COLLECTION,
                field_name=field,
                field_schema=stype,
            )

    def _index_tags_payload(self) -> None:
        for field, stype in [
            ("tag_id", PayloadSchemaType.INTEGER),
            ("usage_count", PayloadSchemaType.INTEGER),
        ]:
            self._client.create_payload_index(
                collection_name=TAGS_COLLECTION,
                field_name=field,
                field_schema=stype,
            )

    # ------------------------------------------------------------------
    # Single-point upserts
    # ------------------------------------------------------------------

    def upsert_post(
        self,
        post_id: int,
        vector: list[float] | np.ndarray,
        payload: dict,
    ) -> None:
        """Upsert one post. payload MUST include 'nsfw' (bool)."""
        if "nsfw" not in payload:
            raise ValueError("payload must include 'nsfw' field — safety baseline requires it")
        self.upsert_batch(
            POSTS_COLLECTION,
            [PointStruct(id=post_id, vector=_to_list(vector), payload=payload)],
        )

    def upsert_account(
        self,
        account_id: int,
        vector: list[float] | np.ndarray,
        payload: dict,
    ) -> None:
        self.upsert_batch(
            ACCOUNTS_COLLECTION,
            [PointStruct(id=account_id, vector=_to_list(vector), payload=payload)],
        )

    def upsert_tag(
        self,
        tag_id: int,
        vector: list[float] | np.ndarray,
        payload: dict,
    ) -> None:
        self.upsert_batch(
            TAGS_COLLECTION,
            [PointStruct(id=tag_id, vector=_to_list(vector), payload=payload)],
        )

    # ------------------------------------------------------------------
    # Batch upsert — preferred for bulk indexing (~10x faster)
    # ------------------------------------------------------------------

    def upsert_batch(
        self,
        collection: str,
        points: list[PointStruct],
        batch_size: int = 256,
    ) -> None:
        """
        Chunk-and-upsert list of PointStructs to any collection.

        Single-point upserts during bulk indexing are ~10x slower than batch;
        always prefer this method when indexing more than one point at a time.
        """
        for i in range(0, len(points), batch_size):
            chunk = points[i : i + batch_size]
            self._client.upsert(collection_name=collection, points=chunk)
        log.debug("Upserted %d point(s) to '%s'", len(points), collection)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_posts(
        self,
        query_vector: list[float] | np.ndarray,
        limit: int = 100,
        filters: dict | None = None,
        exclude_nsfw: bool = True,
    ) -> list[tuple[int, float]]:
        """
        Cosine search on the posts collection.

        Parameters
        ----------
        filters : optional dict — supported keys:
            min_notes  (int)  — FieldCondition note_count >= min_notes
            date_from  (str)  — ISO date "YYYY-MM-DD"; filters published_at (stored as float ts)
            lang       (str)  — exact keyword match on lang field
        exclude_nsfw : default True — adds nsfw=False condition to EVERY search.
            This is the safety baseline; only set False for admin/moderation endpoints.

        Returns
        -------
        list of (post_id, score) sorted descending by score.
        """
        must: list[FieldCondition] = []

        if exclude_nsfw:
            must.append(FieldCondition(key="nsfw", match=MatchValue(value=False)))

        if filters:
            if "min_notes" in filters:
                must.append(
                    FieldCondition(
                        key="note_count",
                        range=Range(gte=int(filters["min_notes"])),
                    )
                )
            if "date_from" in filters:
                ts = _parse_date_to_timestamp(filters["date_from"])
                must.append(
                    FieldCondition(key="published_at", range=Range(gte=ts))
                )
            if "lang" in filters:
                must.append(
                    FieldCondition(key="lang", match=MatchValue(value=str(filters["lang"])))
                )

        qdrant_filter = Filter(must=must) if must else None

        response = self._client.query_points(
            collection_name=POSTS_COLLECTION,
            query=_to_list(query_vector),
            limit=limit,
            query_filter=qdrant_filter,
        )
        return [(int(h.id), float(h.score)) for h in response.points]

    def search_accounts(
        self,
        query_vector: list[float] | np.ndarray,
        limit: int = 50,
    ) -> list[tuple[int, float]]:
        response = self._client.query_points(
            collection_name=ACCOUNTS_COLLECTION,
            query=_to_list(query_vector),
            limit=limit,
        )
        return [(int(h.id), float(h.score)) for h in response.points]

    def search_tags(
        self,
        query_vector: list[float] | np.ndarray,
        limit: int = 30,
    ) -> list[tuple[int, float]]:
        response = self._client.query_points(
            collection_name=TAGS_COLLECTION,
            query=_to_list(query_vector),
            limit=limit,
        )
        return [(int(h.id), float(h.score)) for h in response.points]

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_post(self, post_id: int) -> None:
        """Remove a single post from the index (e.g. DMCA takedown, NSFW re-classification)."""
        self._client.delete(
            collection_name=POSTS_COLLECTION,
            points_selector=PointIdsList(points=[post_id]),
        )
        log.debug("Deleted post %d from Qdrant", post_id)

    # ------------------------------------------------------------------
    # Info / admin
    # ------------------------------------------------------------------

    def get_collection_info(self, name: str) -> dict:
        """Return metadata dict for a collection (point count, index status, etc.)."""
        info = self._client.get_collection(name)
        return {
            "name": name,
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": str(info.status),
            "optimizer_status": str(info.optimizer_status),
        }

    def nightly_optimize(self) -> None:
        """
        Trigger Qdrant optimizer on all collections.

        Sets indexing_threshold=0 (force-index all unindexed vectors) on each
        collection. Safe to call from a nightly cron job; Qdrant runs optimizer
        in background and reverts threshold after completion.
        """
        for name in _ALL_COLLECTIONS:
            try:
                self._client.update_collection(
                    collection_name=name,
                    optimizer_config=OptimizersConfigDiff(indexing_threshold=0),
                )
                log.info("Triggered optimizer for collection '%s'", name)
            except Exception as exc:
                log.warning("Optimizer trigger failed for '%s': %s", name, exc)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "QdrantManager":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_list(v: list[float] | np.ndarray) -> list[float]:
    if isinstance(v, np.ndarray):
        return v.astype(np.float32).tolist()
    return list(v)


def _parse_date_to_timestamp(date_str: str) -> float:
    """Parse ISO date string 'YYYY-MM-DD' → UTC Unix timestamp (midnight)."""
    return datetime.fromisoformat(date_str).timestamp()
