"""检查已落盘 RAG 索引的一致性。

这个文件有两个用途：
- 作为单元测试：用内存里的小样本验证一致性汇总逻辑。
- 作为脚本：读取真实 Chroma 索引、``bm25_corpus.json`` 和
  ``parent_store.json``，检查数量、chunk_hash、dataset 分组和 parent 引用
  是否一致。

运行单元测试：
    python tests/test_index_consistency.py unittest

检查真实索引：
    python tests/test_index_consistency.py
"""

from __future__ import annotations

import argparse
import json
import sys
import unittest
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "chroma_db"


@dataclass(frozen=True)
class DatasetConsistency:
    """单个数据集内 Chroma 和 BM25 的数量、chunk_hash 对比结果。"""

    dataset: str
    vector_count: int
    bm25_count: int
    vector_unique_hashes: int
    bm25_unique_hashes: int
    bm25_minus_vector: int
    vector_minus_bm25: int


@dataclass(frozen=True)
class IndexConsistencyReport:
    """Chroma、BM25 和 parent_store 的整体一致性汇总。"""

    vector_total: int
    bm25_total: int
    parent_total: int
    vector_with_parent_id: int
    vector_missing_parent_store: int
    datasets: tuple[DatasetConsistency, ...]

    @property
    def ok(self) -> bool:
        return (
            self.vector_total == self.bm25_total
            and self.vector_missing_parent_store == 0
            and all(
                item.vector_count == item.bm25_count
                and item.bm25_minus_vector == 0
                and item.vector_minus_bm25 == 0
                for item in self.datasets
            )
        )


def _dataset(metadata: dict) -> str:
    return metadata.get("dataset") or "<missing>"


def _chunk_hash(metadata: dict) -> str | None:
    return metadata.get("chunk_hash")


def summarize_index_consistency(
    vector_metadatas: list[dict],
    bm25_corpus: list[dict],
    parent_store: dict,
) -> IndexConsistencyReport:
    """比较 Chroma metadata、BM25 语料条目和 parent_store 链接。"""

    bm25_metadatas = [entry.get("metadata") or {} for entry in bm25_corpus]
    vector_counts = Counter(_dataset(metadata) for metadata in vector_metadatas)
    bm25_counts = Counter(_dataset(metadata) for metadata in bm25_metadatas)
    datasets = sorted(set(vector_counts) | set(bm25_counts))

    dataset_reports = []
    for dataset in datasets:
        vector_hashes = {
            _chunk_hash(metadata)
            for metadata in vector_metadatas
            if _dataset(metadata) == dataset and _chunk_hash(metadata)
        }
        bm25_hashes = {
            _chunk_hash(metadata)
            for metadata in bm25_metadatas
            if _dataset(metadata) == dataset and _chunk_hash(metadata)
        }
        dataset_reports.append(
            DatasetConsistency(
                dataset=dataset,
                vector_count=vector_counts[dataset],
                bm25_count=bm25_counts[dataset],
                vector_unique_hashes=len(vector_hashes),
                bm25_unique_hashes=len(bm25_hashes),
                bm25_minus_vector=len(bm25_hashes - vector_hashes),
                vector_minus_bm25=len(vector_hashes - bm25_hashes),
            )
        )

    vector_with_parent_id = sum(1 for metadata in vector_metadatas if metadata.get("parent_id"))
    vector_missing_parent_store = sum(
        1
        for metadata in vector_metadatas
        if metadata.get("parent_id") and metadata.get("parent_id") not in parent_store
    )

    return IndexConsistencyReport(
        vector_total=len(vector_metadatas),
        bm25_total=len(bm25_corpus),
        parent_total=len(parent_store),
        vector_with_parent_id=vector_with_parent_id,
        vector_missing_parent_store=vector_missing_parent_store,
        datasets=tuple(dataset_reports),
    )


def load_persisted_index(db_path: Path) -> tuple[list[dict], list[dict], dict]:
    import chromadb

    bm25_path = db_path / "bm25_corpus.json"
    parent_path = db_path / "parent_store.json"

    with bm25_path.open("r", encoding="utf-8") as f:
        bm25_corpus = json.load(f)
    parent_store = {}
    if parent_path.exists():
        with parent_path.open("r", encoding="utf-8") as f:
            parent_store = json.load(f)

    collection = chromadb.PersistentClient(path=str(db_path)).get_collection("langchain")
    data = collection.get(include=["metadatas"], limit=collection.count())
    return data.get("metadatas") or [], bm25_corpus, parent_store


def print_report(report: IndexConsistencyReport) -> None:
    print(f"vector_total: {report.vector_total}")
    print(f"bm25_total: {report.bm25_total}")
    print(f"parent_total: {report.parent_total}")
    for item in report.datasets:
        print(
            f"{item.dataset}: vector={item.vector_count}, bm25={item.bm25_count}, "
            f"vector_unique={item.vector_unique_hashes}, bm25_unique={item.bm25_unique_hashes}, "
            f"bm25_minus_vector={item.bm25_minus_vector}, "
            f"vector_minus_bm25={item.vector_minus_bm25}"
        )
    print(f"vector_with_parent_id: {report.vector_with_parent_id}")
    print(f"vector_missing_parent_store: {report.vector_missing_parent_store}")
    print(f"ok: {report.ok}")


class IndexConsistencyTests(unittest.TestCase):
    def test_summarize_index_consistency_detects_matching_indexes(self):
        report = summarize_index_consistency(
            vector_metadatas=[
                {"dataset": "d1", "chunk_hash": "a", "parent_id": "p1"},
                {"dataset": "d2", "chunk_hash": "b", "parent_id": "p2"},
            ],
            bm25_corpus=[
                {"metadata": {"dataset": "d1", "chunk_hash": "a"}},
                {"metadata": {"dataset": "d2", "chunk_hash": "b"}},
            ],
            parent_store={"p1": "parent 1", "p2": "parent 2"},
        )

        self.assertTrue(report.ok)
        self.assertEqual(report.vector_total, 2)
        self.assertEqual(report.bm25_total, 2)

    def test_summarize_index_consistency_detects_dataset_hash_and_parent_gaps(self):
        report = summarize_index_consistency(
            vector_metadatas=[
                {"dataset": "d1", "chunk_hash": "a", "parent_id": "p1"},
                {"dataset": "d1", "chunk_hash": "vector-only", "parent_id": "missing"},
            ],
            bm25_corpus=[
                {"metadata": {"dataset": "d1", "chunk_hash": "a"}},
                {"metadata": {"dataset": "d1", "chunk_hash": "bm25-only"}},
            ],
            parent_store={"p1": "parent 1"},
        )

        self.assertFalse(report.ok)
        self.assertEqual(report.vector_missing_parent_store, 1)
        self.assertEqual(report.datasets[0].bm25_minus_vector, 1)
        self.assertEqual(report.datasets[0].vector_minus_bm25, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Chroma, BM25, and parent_store consistency.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    vector_metadatas, bm25_corpus, parent_store = load_persisted_index(args.db_path)
    report = summarize_index_consistency(vector_metadatas, bm25_corpus, parent_store)
    print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "unittest":
        sys.argv.pop(1)
        unittest.main()
    raise SystemExit(main())
