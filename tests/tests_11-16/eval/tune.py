"""
tests/eval/tune.py
Grid search over ranking hyperparameters — Prompt 16b.

Validates on 80% TRAIN split, then reports TEST split once for the winner.
Saves best config to config/tuned_weights.json.

Usage:
  python tests/eval/tune.py --qrels tests/eval/qrels.tsv
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from itertools import product
from pathlib import Path
from typing import Any

from tests.eval.run_eval import load_qrels, load_query_metadata, _run_query
from tests.eval.metrics import ndcg_at_k, reciprocal_rank


# ── Grid definition ───────────────────────────────────────────────────────────

GRID = {
    "w_t":  [0.4, 0.5, 0.6],           # text weight
    "w_i":  [0.25, 0.35],              # image weight  (w_g = 1 - w_t - w_i)
    "rrf_k": [30, 60, 90],             # RRF constant
    "note_boost": [0.05, 0.1, 0.2],    # note_count boost coefficient
}

TRAIN_SPLIT = 0.8
EVAL_K = 10
RECALL_K = 100


def _build_configs() -> list[dict[str, Any]]:
    configs = []
    for w_t, w_i, rrf_k, note_boost in product(
        GRID["w_t"], GRID["w_i"], GRID["rrf_k"], GRID["note_boost"]
    ):
        w_g = round(1.0 - w_t - w_i, 4)
        if w_g <= 0:
            continue  # invalid — skip
        configs.append({
            "w_t": w_t,
            "w_i": w_i,
            "w_g": w_g,
            "rrf_k": rrf_k,
            "note_boost": note_boost,
        })
    return configs


def _split_qrels(
    qrels: dict[str, dict[int, int]],
    train_ratio: float = TRAIN_SPLIT,
    seed: int = 42,
) -> tuple[dict[str, dict[int, int]], dict[str, dict[int, int]]]:
    keys = sorted(qrels.keys())
    random.seed(seed)
    random.shuffle(keys)
    split = int(len(keys) * train_ratio)
    train_keys = set(keys[:split])
    test_keys = set(keys[split:])
    return (
        {k: qrels[k] for k in train_keys},
        {k: qrels[k] for k in test_keys},
    )


async def _eval_config(
    config: dict[str, Any],
    qrels: dict[str, dict[int, int]],
    query_meta: dict[str, dict[str, Any]],
) -> float:
    """Apply config to settings, run eval, return mean nDCG@10."""
    from mir.config import settings

    # Temporarily override settings
    original = {
        "TEXT_WEIGHT": settings.TEXT_WEIGHT,
        "IMAGE_WEIGHT": settings.IMAGE_WEIGHT,
        "TAG_WEIGHT": settings.TAG_WEIGHT,
        "RRF_K": settings.RRF_K,
        "NOTE_COUNT_BOOST": settings.NOTE_COUNT_BOOST,
    }
    settings.TEXT_WEIGHT = config["w_t"]
    settings.IMAGE_WEIGHT = config["w_i"]
    settings.TAG_WEIGHT = config["w_g"]
    settings.RRF_K = config["rrf_k"]
    settings.NOTE_COUNT_BOOST = config["note_boost"]

    ndcg_scores: list[float] = []
    for qid, doc_rels in qrels.items():
        meta = query_meta.get(qid, {})
        query_text = meta.get("text", qid)
        query_type = meta.get("type", "general")
        try:
            ranked_ids = await _run_query(query_type, query_text, limit=RECALL_K)
            gains = [doc_rels.get(pid, 0) for pid in ranked_ids]
            ndcg_scores.append(ndcg_at_k(gains, EVAL_K))
        except Exception:
            pass

    # Restore settings
    for k, v in original.items():
        setattr(settings, k, v)

    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0


async def run_grid_search(qrels_path: str) -> dict[str, Any]:
    qrels = load_qrels(qrels_path)
    query_meta = load_query_metadata(qrels_path)
    train_qrels, test_qrels = _split_qrels(qrels)
    configs = _build_configs()

    print(f"Grid search: {len(configs)} configurations × {len(train_qrels)} train queries")
    print()

    best_config: dict[str, Any] | None = None
    best_train_ndcg = -1.0
    results_log: list[dict[str, Any]] = []

    for i, cfg in enumerate(configs, start=1):
        ndcg = await _eval_config(cfg, train_qrels, query_meta)
        results_log.append({**cfg, "train_ndcg": ndcg})
        if ndcg > best_train_ndcg:
            best_train_ndcg = ndcg
            best_config = cfg.copy()
        if i % 10 == 0:
            print(f"  [{i}/{len(configs)}] best so far: nDCG@{EVAL_K}={best_train_ndcg:.4f}  config={best_config}")

    print(f"\nBest train config: {best_config}")
    print(f"  Train nDCG@{EVAL_K}: {best_train_ndcg:.4f}")

    # ── Single test evaluation ─────────────────────────────────────────────
    print(f"\nEvaluating best config on TEST split ({len(test_qrels)} queries)...")
    test_ndcg = await _eval_config(best_config, test_qrels, query_meta)
    print(f"  Test nDCG@{EVAL_K}: {test_ndcg:.4f}")

    winning = {**best_config, "train_ndcg": best_train_ndcg, "test_ndcg": test_ndcg}

    # ── Save to config/tuned_weights.json ──────────────────────────────────
    out_path = Path("config/tuned_weights.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(winning, indent=2))
    print(f"\nWinning config saved → {out_path}")

    return winning


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter grid search for MIR ranking")
    parser.add_argument("--qrels", default="tests/eval/qrels.tsv")
    args = parser.parse_args()
    result = asyncio.run(run_grid_search(args.qrels))
    print("\nFinal winning configuration:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
