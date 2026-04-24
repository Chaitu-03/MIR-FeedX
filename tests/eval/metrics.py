"""
tests/eval/metrics.py
Standard IR evaluation metrics — Prompt 16b.
"""
from __future__ import annotations

import math


def dcg_at_k(gains: list[int], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i, g in enumerate(gains[:k], start=1):
        dcg += g / math.log2(i + 1)
    return dcg


def ideal_dcg_at_k(gains: list[int], k: int) -> float:
    """Ideal DCG: gains sorted descending."""
    return dcg_at_k(sorted(gains, reverse=True), k)


def ndcg_at_k(gains: list[int], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain at k.

    Args:
        gains: list of relevance grades (0=irrelevant, 1=relevant, 2=highly relevant),
               ordered by the system's ranking.
        k: cutoff rank.

    Returns:
        nDCG@k in [0, 1].
    """
    idcg = ideal_dcg_at_k(gains, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(gains, k) / idcg


def reciprocal_rank(gains: list[int]) -> float:
    """
    Mean Reciprocal Rank (MRR) for a single query.
    Returns 1/rank of the first relevant result (grade >= 1), or 0 if none found.
    """
    for i, g in enumerate(gains, start=1):
        if g >= 1:
            return 1.0 / i
    return 0.0


def recall_at_k(gains: list[int], k: int, n_relevant: int) -> float:
    """
    Recall@k: fraction of relevant documents found in the top-k results.

    Args:
        gains: relevance grades ordered by system ranking.
        k: cutoff rank.
        n_relevant: total number of relevant documents in the corpus for this query.
    """
    if n_relevant == 0:
        return 0.0
    found = sum(1 for g in gains[:k] if g >= 1)
    return found / n_relevant


def average_precision(gains: list[int]) -> float:
    """
    Average Precision (AP) for a single query.
    Used to compute MAP across queries.
    """
    hits = 0
    sum_prec = 0.0
    for i, g in enumerate(gains, start=1):
        if g >= 1:
            hits += 1
            sum_prec += hits / i
    if hits == 0:
        return 0.0
    return sum_prec / hits
