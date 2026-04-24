"""
tests/eval/run_eval.py
Offline evaluation harness — Prompt 16b.

Usage:
  python tests/eval/run_eval.py --qrels tests/eval/qrels.tsv --k 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from tests.eval.metrics import ndcg_at_k, reciprocal_rank, recall_at_k, average_precision


# ── qrels loader ──────────────────────────────────────────────────────────────

def load_qrels(path: str) -> dict[str, dict[int, int]]:
    """
    Load qrels.tsv: query_id TAB post_id TAB relevance
    Returns: {query_id: {post_id: relevance}}
    """
    qrels: dict[str, dict[int, int]] = defaultdict(dict)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            qid, pid, rel = parts
            qrels[qid][int(pid)] = int(rel)
    return dict(qrels)


def load_query_metadata(path: str) -> dict[str, dict[str, Any]]:
    """
    Load queries.tsv: query_id TAB query_text TAB query_type
    Returns: {query_id: {text, type}}
    """
    queries: dict[str, dict[str, Any]] = {}
    queries_path = Path(path).parent / "queries.tsv"
    if not queries_path.exists():
        return {}
    with open(queries_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            qid = parts[0]
            text = parts[1]
            qtype = parts[2] if len(parts) > 2 else "general"
            queries[qid] = {"text": text, "type": qtype}
    return queries


# ── Result fetcher ────────────────────────────────────────────────────────────

async def _fetch_general(query: str, limit: int = 100) -> list[int]:
    """Call GeneralSearchResolver directly (no HTTP). Returns ordered post_ids."""
    from mir.db.session import get_session
    from mir.search.general import GeneralSearchResolver, GeneralSearchFilters
    from mir.processing.text import TextProcessor
    from mir.search.vector_store import QdrantManager
    from mir.config import settings

    text_proc = TextProcessor()
    qdrant = QdrantManager(host=settings.QDRANT_HOST, grpc_port=settings.QDRANT_GRPC_PORT)

    async with get_session() as db:
        resolver = GeneralSearchResolver(db=db, qdrant_manager=qdrant, text_processor=text_proc)
        result = await resolver.search(
            query=query,
            filters=GeneralSearchFilters(),
            limit_posts=limit,
        )
    return [p.post_id for p in result.posts]


async def _fetch_tags(query: str, limit: int = 100) -> list[int]:
    from mir.db.session import get_session
    from mir.search.tags import search_tags
    from mir.processing.text import TextProcessor
    from mir.search.vector_store import QdrantManager
    from mir.config import settings

    text_proc = TextProcessor()
    qdrant = QdrantManager(host=settings.QDRANT_HOST, grpc_port=settings.QDRANT_GRPC_PORT)

    async with get_session() as db:
        results = await search_tags(db, qdrant, text_proc, query, limit=limit)
    return [r.tag_id for r in results]


async def _run_query(qtype: str, query: str, limit: int) -> list[int]:
    if qtype == "general":
        return await _fetch_general(query, limit)
    elif qtype in ("tag", "tags"):
        return await _fetch_tags(query, limit)
    else:
        return await _fetch_general(query, limit)


# ── Eval runner ───────────────────────────────────────────────────────────────

async def run_eval(
    qrels_path: str,
    k: int = 10,
    recall_k: int = 100,
    query_type_filter: str | None = None,
) -> dict[str, Any]:
    qrels = load_qrels(qrels_path)
    query_meta = load_query_metadata(qrels_path)

    all_ndcg: list[float] = []
    all_rr: list[float] = []
    all_recall: list[float] = []
    all_ap: list[float] = []

    per_type: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"ndcg": [], "rr": [], "recall": [], "ap": []}
    )

    for qid, doc_rels in qrels.items():
        meta = query_meta.get(qid, {})
        query_text = meta.get("text", qid)
        query_type = meta.get("type", "general")

        if query_type_filter and query_type != query_type_filter:
            continue

        try:
            ranked_ids = await _run_query(query_type, query_text, limit=recall_k)
        except Exception as exc:
            print(f"  [WARN] query {qid!r} failed: {exc}")
            continue

        gains = [doc_rels.get(pid, 0) for pid in ranked_ids]
        n_relevant = sum(1 for r in doc_rels.values() if r >= 1)

        q_ndcg = ndcg_at_k(gains, k)
        q_rr = reciprocal_rank(gains)
        q_recall = recall_at_k(gains, recall_k, n_relevant)
        q_ap = average_precision(gains)

        all_ndcg.append(q_ndcg)
        all_rr.append(q_rr)
        all_recall.append(q_recall)
        all_ap.append(q_ap)

        per_type[query_type]["ndcg"].append(q_ndcg)
        per_type[query_type]["rr"].append(q_rr)
        per_type[query_type]["recall"].append(q_recall)
        per_type[query_type]["ap"].append(q_ap)

    def _mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    summary: dict[str, Any] = {
        "n_queries": len(all_ndcg),
        f"ndcg_at_{k}": round(_mean(all_ndcg), 4),
        "mrr": round(_mean(all_rr), 4),
        f"recall_at_{recall_k}": round(_mean(all_recall), 4),
        "map": round(_mean(all_ap), 4),
        "per_type": {
            qtype: {
                f"ndcg_at_{k}": round(_mean(vals["ndcg"]), 4),
                "mrr": round(_mean(vals["rr"]), 4),
                f"recall_at_{recall_k}": round(_mean(vals["recall"]), 4),
                "n": len(vals["ndcg"]),
            }
            for qtype, vals in per_type.items()
        },
    }
    return summary


def print_summary(summary: dict[str, Any], k: int = 10, recall_k: int = 100):
    print("\n" + "=" * 60)
    print("MIR EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Queries evaluated : {summary['n_queries']}")
    print(f"  nDCG@{k:<2}           : {summary[f'ndcg_at_{k}']:.4f}  (target > 0.65)")
    print(f"  MRR               : {summary['mrr']:.4f}  (target > 0.70)")
    print(f"  Recall@{recall_k:<3}        : {summary[f'recall_at_{recall_k}']:.4f}  (target > 0.85)")
    print(f"  MAP               : {summary['map']:.4f}")
    print()
    if summary.get("per_type"):
        print("  Per-type breakdown:")
        for qtype, vals in summary["per_type"].items():
            print(f"    {qtype:<12}  nDCG@{k}={vals[f'ndcg_at_{k}']:.4f}  "
                  f"MRR={vals['mrr']:.4f}  n={vals['n']}")
    print("=" * 60 + "\n")


def save_results(summary: dict[str, Any], out_dir: str = "tests/eval"):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = Path(out_dir) / f"results_{ts}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Results saved → {out_path}")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="MIR offline eval harness")
    parser.add_argument("--qrels", default="tests/eval/qrels.tsv")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--recall-k", type=int, default=100)
    parser.add_argument("--type", dest="query_type", default=None,
                        help="Filter by query type (general|tag|account|community)")
    parser.add_argument("--save", action="store_true", default=True)
    args = parser.parse_args()

    summary = asyncio.run(
        run_eval(args.qrels, k=args.k, recall_k=args.recall_k,
                 query_type_filter=args.query_type)
    )
    print_summary(summary, k=args.k, recall_k=args.recall_k)
    if args.save:
        save_results(summary)


if __name__ == "__main__":
    main()
