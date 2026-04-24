"""
tests/eval/test_regression.py
Regression guard + unit tests for metrics — Prompt 16b.

Run in CI after any model or ranking logic change:
  pytest tests/eval/test_regression.py -v

Unit tests for metric functions use known inputs with expected outputs.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from tests.eval.metrics import dcg_at_k, ndcg_at_k, reciprocal_rank, recall_at_k, average_precision


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests for metric functions (deterministic, no I/O required)
# ═══════════════════════════════════════════════════════════════════════════════

class TestNDCG:
    def test_perfect_ranking(self):
        """All relevant docs at top → nDCG = 1.0"""
        gains = [2, 1, 0, 0, 0]
        assert ndcg_at_k(gains, k=5) == pytest.approx(1.0)

    def test_worst_ranking(self):
        """Relevant doc at position k+1 → nDCG@k = 0.0"""
        gains = [0, 0, 0, 0, 2]
        assert ndcg_at_k(gains, k=3) == pytest.approx(0.0)

    def test_known_value_2_0_1(self):
        """
        gains = [2, 0, 1], k=3
        DCG  = 2/log2(2) + 0/log2(3) + 1/log2(4) = 2 + 0 + 0.5 = 2.5
        IDCG = 2/log2(2) + 1/log2(3) + 0/log2(4) = 2 + 0.631 = 2.631
        nDCG = 2.5 / 2.631 ≈ 0.9502
        """
        gains = [2, 0, 1]
        expected_dcg = 2.0 / math.log2(2) + 0.0 + 1.0 / math.log2(4)
        expected_idcg = 2.0 / math.log2(2) + 1.0 / math.log2(3)
        expected = expected_dcg / expected_idcg
        assert ndcg_at_k(gains, k=3) == pytest.approx(expected, rel=1e-4)

    def test_empty_gains(self):
        assert ndcg_at_k([], k=5) == pytest.approx(0.0)

    def test_all_zeros(self):
        assert ndcg_at_k([0, 0, 0], k=3) == pytest.approx(0.0)

    def test_k_larger_than_list(self):
        """k > len(gains) should not raise — truncates at list length."""
        gains = [2, 1]
        result = ndcg_at_k(gains, k=10)
        assert 0.0 <= result <= 1.0

    def test_single_relevant_at_1(self):
        """Single relevant result at position 1 → maximum nDCG."""
        gains = [1, 0, 0]
        assert ndcg_at_k(gains, k=3) == pytest.approx(1.0)

    def test_single_relevant_at_2(self):
        """
        gains = [0, 1, 0], k=3
        DCG  = 0 + 1/log2(3) ≈ 0.631
        IDCG = 1/log2(2) = 1.0
        nDCG ≈ 0.631
        """
        gains = [0, 1, 0]
        expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
        assert ndcg_at_k(gains, k=3) == pytest.approx(expected, rel=1e-4)


class TestReciprocalRank:
    def test_first_relevant_at_1(self):
        assert reciprocal_rank([2, 0, 0]) == pytest.approx(1.0)

    def test_first_relevant_at_2(self):
        assert reciprocal_rank([0, 1, 0]) == pytest.approx(0.5)

    def test_first_relevant_at_3(self):
        assert reciprocal_rank([0, 0, 2]) == pytest.approx(1 / 3)

    def test_no_relevant(self):
        assert reciprocal_rank([0, 0, 0]) == pytest.approx(0.0)

    def test_empty(self):
        assert reciprocal_rank([]) == pytest.approx(0.0)


class TestRecallAtK:
    def test_all_found(self):
        assert recall_at_k([1, 1, 0], k=3, n_relevant=2) == pytest.approx(1.0)

    def test_partial(self):
        assert recall_at_k([1, 0, 0, 1, 0], k=2, n_relevant=2) == pytest.approx(0.5)

    def test_none_found(self):
        assert recall_at_k([0, 0, 0], k=3, n_relevant=3) == pytest.approx(0.0)

    def test_zero_relevant(self):
        assert recall_at_k([1, 1, 1], k=3, n_relevant=0) == pytest.approx(0.0)


class TestAveragePrecision:
    def test_perfect(self):
        assert average_precision([1, 1, 1]) == pytest.approx(1.0)

    def test_none_relevant(self):
        assert average_precision([0, 0, 0]) == pytest.approx(0.0)

    def test_known_value(self):
        """
        gains = [1, 0, 1]
        Hit at rank 1: prec=1/1=1.0
        Hit at rank 3: prec=2/3≈0.667
        AP = (1.0 + 0.667) / 2 = 0.833
        """
        gains = [1, 0, 1]
        assert average_precision(gains) == pytest.approx((1.0 + 2 / 3) / 2, rel=1e-4)


# ═══════════════════════════════════════════════════════════════════════════════
# Regression gate (runs only if qrels.tsv and a live DB exist)
# ═══════════════════════════════════════════════════════════════════════════════

QRELS_PATH = "tests/eval/qrels.tsv"
BASELINE_NDCG = 0.65
BASELINE_MRR = 0.70


def _load_tuned_baseline() -> float:
    """If tuned_weights.json exists, use its test_ndcg as the baseline floor."""
    config_path = Path("config/tuned_weights.json")
    if config_path.exists():
        data = json.loads(config_path.read_text())
        return data.get("test_ndcg", BASELINE_NDCG)
    return BASELINE_NDCG


@pytest.mark.skipif(
    not os.path.exists(QRELS_PATH),
    reason="qrels.tsv not found — skipping regression gate",
)
@pytest.mark.asyncio
async def test_ndcg_regression():
    """
    Regression guard: nDCG@10 must not drop below baseline.
    Skipped if qrels.tsv does not exist (e.g. in CI without labelled data).
    """
    from tests.eval.run_eval import run_eval

    summary = await run_eval(QRELS_PATH, k=10, recall_k=100)
    floor = _load_tuned_baseline()
    actual_ndcg = summary["ndcg_at_10"]
    actual_mrr = summary["mrr"]

    print(f"\n  nDCG@10: {actual_ndcg:.4f} (floor={floor:.4f})")
    print(f"  MRR:     {actual_mrr:.4f} (floor={BASELINE_MRR:.4f})")

    assert actual_ndcg >= floor, (
        f"nDCG@10 regression: got {actual_ndcg:.4f}, expected >= {floor:.4f}. "
        "Re-run tune.py or investigate ranking changes."
    )
    assert actual_mrr >= BASELINE_MRR, (
        f"MRR regression: got {actual_mrr:.4f}, expected >= {BASELINE_MRR:.4f}."
    )
