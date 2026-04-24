"""
mir/processing/communities.py
Community clustering — Prompt 11, Part A.

Phase 1 — Tag clusters (k-means, k chosen by silhouette score):
  Load tag embeddings from Qdrant → k-means → store in communities table.

Phase 2 — Account clusters (HDBSCAN, min_cluster_size=5):
  Load account embeddings → HDBSCAN → store in communities table.

Entry point: rebuild_communities(db, qdrant_manager, text_processor)
Designed to run as a nightly Celery Beat task.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)

_TAG_K_CANDIDATES = [30, 50, 100, 150, 200]
_HDBSCAN_MIN_CLUSTER_SIZE = 5
_TOP_TAGS_PER_CLUSTER = 5
_NAME_TAGS = 3          # number of tags used in auto-generated community name


# ── Public entry point ────────────────────────────────────────────────────────

async def rebuild_communities(
    db: AsyncSession,
    qdrant_manager,
    text_processor,  # used to resolve tag names for account clusters
) -> None:
    """
    Full nightly rebuild: clears existing clusters, then runs Phase 1 + Phase 2.
    Safe to call repeatedly — idempotent via DELETE + INSERT.
    """
    log.info("rebuild_communities: starting")

    # Clear existing machine-generated clusters (keep any hand-curated ones)
    await db.execute(
        text("DELETE FROM communities WHERE type IN ('tag_cluster', 'account_cluster')")
    )
    await db.commit()

    await _rebuild_tag_clusters(db, qdrant_manager)
    await _rebuild_account_clusters(db, qdrant_manager)

    log.info("rebuild_communities: done")


# ── Phase 1: tag-cluster communities ─────────────────────────────────────────

async def _rebuild_tag_clusters(db: AsyncSession, qdrant_manager) -> None:
    """
    1. Load all tag embeddings from Qdrant (tags collection).
    2. Choose k via silhouette score over _TAG_K_CANDIDATES.
    3. For each cluster: centroid, top-5 tags, auto-name, store in DB.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    log.info("Phase 1: loading tag embeddings from Qdrant")
    tag_ids, tag_vecs = _scroll_all_vectors(qdrant_manager, "tags")
    if len(tag_ids) < 2:
        log.warning("Too few tags (%d) to cluster — skipping Phase 1", len(tag_ids))
        return

    X = np.array(tag_vecs, dtype=np.float32)

    # Choose k candidates that are smaller than the dataset
    candidates = [k for k in _TAG_K_CANDIDATES if k < len(tag_ids)]
    if not candidates:
        candidates = [max(2, len(tag_ids) // 2)]

    log.info("Phase 1: tuning k over %s with %d tags", candidates, len(tag_ids))
    best_k, best_score, best_labels = _choose_k_kmeans(X, candidates)
    log.info("Phase 1: best k=%d  silhouette=%.4f", best_k, best_score)

    # Compute centroids
    centroids = np.zeros((best_k, X.shape[1]), dtype=np.float32)
    for cluster_idx in range(best_k):
        mask = best_labels == cluster_idx
        if mask.any():
            centroids[cluster_idx] = X[mask].mean(axis=0)

    # Fetch tag name + usage for all tag IDs in one query
    id_to_meta: dict[int, dict] = await _fetch_tag_meta(db, tag_ids)

    now = datetime.now(tz=timezone.utc)
    for cluster_idx in range(best_k):
        mask = best_labels == cluster_idx
        member_tag_ids = [tag_ids[i] for i in range(len(tag_ids)) if mask[i]]
        if not member_tag_ids:
            continue

        centroid = centroids[cluster_idx]

        # Top-5 tags closest to centroid
        member_vecs = X[mask]
        sims = member_vecs @ centroid / (
            np.linalg.norm(member_vecs, axis=1) * np.linalg.norm(centroid) + 1e-10
        )
        top_idx = np.argsort(-sims)[:_TOP_TAGS_PER_CLUSTER]
        top_member_ids = [member_tag_ids[i] for i in top_idx]

        name_parts = [
            id_to_meta[tid]["name"]
            for tid in top_member_ids[:_NAME_TAGS]
            if tid in id_to_meta
        ]
        name = " · ".join(name_parts) if name_parts else f"cluster_{cluster_idx}"

        await db.execute(
            text("""
                INSERT INTO communities (name, centroid, type, member_ids, updated_at)
                VALUES (:name, :centroid, 'tag_cluster', :member_ids, :updated_at)
            """),
            {
                "name": name,
                "centroid": centroid.tolist(),
                "member_ids": member_tag_ids,
                "updated_at": now,
            },
        )

    await db.commit()
    log.info("Phase 1: inserted %d tag_cluster communities", best_k)


# ── Phase 2: account-cluster communities ─────────────────────────────────────

async def _rebuild_account_clusters(db: AsyncSession, qdrant_manager) -> None:
    """
    1. Load all account embeddings from Qdrant (accounts collection).
    2. HDBSCAN (min_cluster_size=5).
    3. For each cluster: centroid, member accounts, name by top tags of members.
    """
    try:
        import hdbscan as _hdbscan
    except ImportError:
        log.error("hdbscan not installed — skipping Phase 2. pip install hdbscan")
        return

    log.info("Phase 2: loading account embeddings from Qdrant")
    account_ids, account_vecs = _scroll_all_vectors(qdrant_manager, "accounts")
    if len(account_ids) < _HDBSCAN_MIN_CLUSTER_SIZE * 2:
        log.warning("Too few accounts (%d) to cluster — skipping Phase 2", len(account_ids))
        return

    X = np.array(account_vecs, dtype=np.float32)

    log.info("Phase 2: running HDBSCAN on %d accounts", len(account_ids))
    clusterer = _hdbscan.HDBSCAN(
        min_cluster_size=_HDBSCAN_MIN_CLUSTER_SIZE,
        metric="euclidean",
    )
    labels: np.ndarray = clusterer.fit_predict(X)

    unique_clusters = [c for c in np.unique(labels) if c != -1]  # -1 = noise
    log.info("Phase 2: found %d clusters (noise excluded)", len(unique_clusters))

    now = datetime.now(tz=timezone.utc)
    for cluster_idx in unique_clusters:
        mask = labels == cluster_idx
        member_account_ids = [account_ids[i] for i in range(len(account_ids)) if mask[i]]
        centroid = X[mask].mean(axis=0)

        # Name by fetching top-3 tags from members' posts
        name = await _name_account_cluster(db, member_account_ids)

        await db.execute(
            text("""
                INSERT INTO communities (name, centroid, type, member_ids, updated_at)
                VALUES (:name, :centroid, 'account_cluster', :member_ids, :updated_at)
            """),
            {
                "name": name,
                "centroid": centroid.tolist(),
                "member_ids": member_account_ids,
                "updated_at": now,
            },
        )

    await db.commit()
    log.info("Phase 2: inserted %d account_cluster communities", len(unique_clusters))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scroll_all_vectors(
    qdrant_manager,
    collection: str,
    batch_size: int = 1000,
) -> tuple[list[int], list[list[float]]]:
    """
    Scroll through an entire Qdrant collection and return (ids, vectors).
    Uses the sync QdrantClient.scroll with with_vectors=True.
    """
    client = qdrant_manager._client
    all_ids: list[int] = []
    all_vecs: list[list[float]] = []
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=offset,
            with_vectors=True,
        )
        for p in points:
            all_ids.append(int(p.id))
            vec = p.vector
            if isinstance(vec, dict):
                # Named vector — take first value
                vec = next(iter(vec.values()))
            all_vecs.append(vec)

        if next_offset is None:
            break
        offset = next_offset

    return all_ids, all_vecs


def _choose_k_kmeans(
    X: np.ndarray,
    k_candidates: list[int],
    random_state: int = 42,
) -> tuple[int, float, np.ndarray]:
    """
    Run k-means for each candidate k; return (best_k, best_silhouette, labels).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    best_k = k_candidates[0]
    best_score = -1.0
    best_labels: np.ndarray = np.zeros(len(X), dtype=int)

    for k in k_candidates:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        # silhouette requires at least 2 distinct labels
        if len(np.unique(labels)) < 2:
            continue
        score = float(silhouette_score(X, labels, sample_size=min(5000, len(X))))
        log.debug("k=%d  silhouette=%.4f", k, score)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    return best_k, best_score, best_labels


async def _fetch_tag_meta(
    db: AsyncSession,
    tag_ids: list[int],
) -> dict[int, dict]:
    """Fetch {id: {name, usage_count}} for a list of tag IDs."""
    if not tag_ids:
        return {}
    rows = (
        await db.execute(
            text("SELECT id, name, usage_count FROM tags WHERE id = ANY(:ids)"),
            {"ids": tag_ids},
        )
    ).fetchall()
    return {row.id: {"name": row.name, "usage_count": row.usage_count} for row in rows}


async def _name_account_cluster(
    db: AsyncSession,
    account_ids: list[int],
) -> str:
    """
    Derive a community name by finding the top-3 tags used by accounts in the cluster.
    Returns a " · "-joined string, or a fallback if no tags found.
    """
    if not account_ids:
        return "unnamed_cluster"

    rows = (
        await db.execute(
            text("""
                SELECT t.name, COUNT(*) AS cnt
                FROM post_tags pt
                JOIN posts p ON p.id = pt.post_id
                JOIN tags t ON t.id = pt.tag_id
                WHERE p.account_id = ANY(:ids)
                  AND p.nsfw = false
                GROUP BY t.name
                ORDER BY cnt DESC
                LIMIT :n
            """),
            {"ids": account_ids, "n": _NAME_TAGS},
        )
    ).fetchall()

    parts = [r.name for r in rows]
    return " · ".join(parts) if parts else "account_cluster"
