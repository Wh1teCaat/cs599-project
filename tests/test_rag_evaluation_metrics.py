"""RAG 排序评估指标公式的单元测试。

这个文件测试 ``test_vector_recall.py`` 使用的纯函数：标准文档排名查找、
Reciprocal Rank@K 和 nDCG@K。它不加载 Chroma、embedding 或真实索引，因此可以
快速发现指标公式回归。
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = PROJECT_ROOT / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from test_vector_recall import ndcg_at_k, rank_of_gold, reciprocal_rank_at_k  # noqa: E402


class RagEvaluationMetricTests(unittest.TestCase):
    def test_rank_of_gold_returns_one_based_rank(self):
        self.assertEqual(rank_of_gold(["a", "gold", "c"], "gold"), 2)
        self.assertIsNone(rank_of_gold(["a", "b"], "gold"))

    def test_reciprocal_rank_at_k_counts_only_hits_within_k(self):
        result_hashes = ["a", "gold", "c"]

        self.assertEqual(reciprocal_rank_at_k(result_hashes, "gold", 1), 0.0)
        self.assertEqual(reciprocal_rank_at_k(result_hashes, "gold", 2), 0.5)

    def test_ndcg_at_k_uses_single_relevant_document_gain(self):
        result_hashes = ["a", "gold", "c"]

        self.assertEqual(ndcg_at_k(result_hashes, "gold", 1), 0.0)
        self.assertAlmostEqual(ndcg_at_k(result_hashes, "gold", 2), 1 / math.log2(3))


if __name__ == "__main__":
    unittest.main()
