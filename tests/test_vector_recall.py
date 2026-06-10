"""评估已索引数据集的向量检索排序质量。

这个脚本使用本地 ChromaDB 索引和 BM25 语料 metadata 作为评测依据。每个样本
是一条 question，标准答案文档的 hash 存在 metadata 中。脚本会检索 top-K
向量结果，并输出：

- Recall@K：标准文档是否出现在 top K 结果中。
- MRR@K：第一个标准文档排得多靠前，公式是 ``1 / rank``。
- nDCG@K：标准文档排序质量，使用对数折损 ``1 / log2(rank + 1)``。

示例：
    env EMBEDDING_DEVICE=cpu python tests/test_vector_recall.py --sample-size 50
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from retriever.vector import load_vector_retriever  # noqa: E402


DEFAULT_DATASETS = ("passage_retrieval_en", "2wikimqa_e")
DEFAULT_KS = (1, 5, 10, 20)


@dataclass(frozen=True)
class RecallCase:
    dataset: str
    question: str
    gold_hash: str


def rank_of_gold(result_hashes: list[str | None], gold_hash: str) -> int | None:
    """返回标准文档 hash 的 1-based 排名；未命中时返回 None。

    rank 表示标准文档在检索结果中的位置。rank=1 代表检索器把标准文档排在第一。
    """
    for idx, result_hash in enumerate(result_hashes, 1):
        if result_hash == gold_hash:
            return idx
    return None


def reciprocal_rank_at_k(result_hashes: list[str | None], gold_hash: str, k: int) -> float:
    """计算单个 query 的 Reciprocal Rank@K。

    MRR 是多个 query 的 Reciprocal Rank 平均值。标准文档排第 1 名得 1.0，
    第 2 名得 0.5，第 5 名得 0.2；如果 top-K 内没命中则得 0.0。这个指标
    奖励“第一个正确文档排得靠前”。
    """
    rank = rank_of_gold(result_hashes[:k], gold_hash)
    return 0.0 if rank is None else 1.0 / rank


def ndcg_at_k(result_hashes: list[str | None], gold_hash: str, k: int) -> float:
    """计算单个 query 的 nDCG@K，当前评测假设只有一个标准文档。

    nDCG 使用 ``1 / log2(rank + 1)`` 对低排名命中做折损。当前每个 query 只有
    一个标准文档，所以排第 1 名的理想得分是 1.0，排名越靠后分数越低。
    """
    rank = rank_of_gold(result_hashes[:k], gold_hash)
    return 0.0 if rank is None else 1.0 / math.log2(rank + 1)


def load_recall_cases(corpus_path: Path, datasets: tuple[str, ...]) -> list[RecallCase]:
    """Load one recall case per indexed question/document group."""
    with corpus_path.open("r", encoding="utf-8") as f:
        corpus = json.load(f)

    by_key: dict[tuple[str, str, str], RecallCase] = {}
    wanted = set(datasets)
    for entry in corpus:
        metadata = entry.get("metadata") or {}
        dataset = metadata.get("dataset")
        question = metadata.get("question")
        gold_hash = metadata.get("hash")
        if dataset not in wanted or not question or not gold_hash:
            continue
        by_key.setdefault(
            (dataset, question, gold_hash),
            RecallCase(dataset=dataset, question=question, gold_hash=gold_hash),
        )

    return list(by_key.values())


def load_indexed_hashes(db_path: Path, datasets: tuple[str, ...]) -> dict[str, set[str]]:
    """Load dataset/hash pairs that are actually present in Chroma."""
    sqlite_path = db_path / "chroma.sqlite3"
    if not sqlite_path.exists():
        raise FileNotFoundError(f"Chroma sqlite file not found: {sqlite_path}")

    wanted = set(datasets)
    indexed: dict[str, set[str]] = defaultdict(set)
    conn = sqlite3.connect(sqlite_path)
    try:
        rows = conn.execute(
            """
            select d.string_value as dataset, h.string_value as hash
            from embedding_metadata d
            join embedding_metadata h on d.id = h.id
            where d.key = 'dataset' and h.key = 'hash'
            """
        )
        for dataset, gold_hash in rows:
            if dataset in wanted and gold_hash:
                indexed[dataset].add(gold_hash)
    finally:
        conn.close()
    return indexed


def keep_indexed_cases(
    cases: list[RecallCase],
    indexed_hashes: dict[str, set[str]],
) -> tuple[list[RecallCase], dict[str, int]]:
    """Keep cases whose gold hash exists in the vector index."""
    kept: list[RecallCase] = []
    skipped: dict[str, int] = defaultdict(int)
    for case in cases:
        if case.gold_hash in indexed_hashes.get(case.dataset, set()):
            kept.append(case)
        else:
            skipped[case.dataset] += 1
    return kept, dict(skipped)


def sample_cases(
    cases: list[RecallCase],
    sample_size: int | None,
    seed: int,
) -> list[RecallCase]:
    """Sample up to sample_size cases per dataset."""
    if sample_size is None:
        return cases

    rng = random.Random(seed)
    grouped: dict[str, list[RecallCase]] = defaultdict(list)
    for case in cases:
        grouped[case.dataset].append(case)

    sampled: list[RecallCase] = []
    for dataset in sorted(grouped):
        dataset_cases = grouped[dataset]
        if len(dataset_cases) <= sample_size:
            sampled.extend(dataset_cases)
        else:
            sampled.extend(rng.sample(dataset_cases, sample_size))
    return sampled


def evaluate_recall(
    cases: list[RecallCase],
    db_path: Path,
    cache_path: Path,
    ks: tuple[int, ...],
) -> dict[str, dict[str, float]]:
    """按数据集评估 Recall@K、MRR@K 和 nDCG@K。

    Recall@K 衡量是否召回到正确上下文；MRR@K 和 nDCG@K 衡量排序质量。三者结合
    可以同时判断“有没有找到正确上下文”和“正确上下文是否足够靠前”。
    """
    if not cases:
        raise ValueError("No recall cases found. Check corpus path and dataset names.")

    max_k = max(ks)
    try:
        retriever = load_vector_retriever(str(db_path), str(cache_path), top_k=max_k)
    except ImportError as exc:
        raise RuntimeError(
            "Vector recall evaluation requires the embedding dependencies. "
            "Install the project requirements first, especially "
            "`sentence-transformers` and `langchain-huggingface`."
        ) from exc

    totals: dict[str, int] = defaultdict(int)
    hits: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    reciprocal_rank_sums: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    ndcg_sums: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))

    for idx, case in enumerate(cases, 1):
        docs = retriever.invoke(case.question)
        result_hashes = [(doc.metadata or {}).get("hash") for doc in docs[:max_k]]
        totals[case.dataset] += 1
        for k in ks:
            if case.gold_hash in result_hashes[:k]:
                hits[case.dataset][k] += 1
            reciprocal_rank_sums[case.dataset][k] += reciprocal_rank_at_k(
                result_hashes, case.gold_hash, k
            )
            ndcg_sums[case.dataset][k] += ndcg_at_k(result_hashes, case.gold_hash, k)
        print(
            f"[{idx}/{len(cases)}] {case.dataset} "
            f"hit@{max_k}={case.gold_hash in result_hashes[:max_k]}"
        )

    scores: dict[str, dict[str, float]] = {}
    for dataset in sorted(totals):
        dataset_scores = {}
        for k in ks:
            total = totals[dataset]
            dataset_scores[f"Recall@{k}"] = hits[dataset][k] / total
            dataset_scores[f"MRR@{k}"] = reciprocal_rank_sums[dataset][k] / total
            dataset_scores[f"nDCG@{k}"] = ndcg_sums[dataset][k] / total
        scores[dataset] = dataset_scores
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate vector retrieval Recall@K, MRR@K, and nDCG@K."
    )
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "chroma_db" / "bm25_corpus.json",
        help="Path to bm25_corpus.json with question/hash metadata.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "chroma_db",
        help="Path to the persisted ChromaDB directory.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "embeddings_cache.json",
        help="Path to the embedding cache JSON file.",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset names to evaluate.",
    )
    parser.add_argument(
        "--ks",
        default=",".join(str(k) for k in DEFAULT_KS),
        help="Comma-separated K values, for example: 1,5,10,20.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional number of indexed questions to sample per dataset.",
    )
    parser.add_argument(
        "--include-unindexed",
        action="store_true",
        help=(
            "Also evaluate cases whose gold hash is not present in Chroma. "
            "These cases cannot be recalled and are excluded by default."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used with --sample-size.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = tuple(d.strip() for d in args.datasets.split(",") if d.strip())
    ks = tuple(sorted({int(k.strip()) for k in args.ks.split(",") if k.strip()}))

    all_cases = load_recall_cases(args.corpus_path, datasets)
    indexed_hashes = load_indexed_hashes(args.db_path, datasets)
    if args.include_unindexed:
        cases = all_cases
        skipped = {}
    else:
        cases, skipped = keep_indexed_cases(all_cases, indexed_hashes)
    cases = sample_cases(cases, args.sample_size, args.seed)

    print(f"datasets: {', '.join(datasets)}")
    print(f"cases: {len(cases)}")
    print(
        "indexed cases: "
        + ", ".join(
            f"{dataset}={len(indexed_hashes.get(dataset, set()))}"
            for dataset in datasets
        )
    )
    if skipped:
        print(
            "skipped unindexed cases: "
            + ", ".join(f"{dataset}={count}" for dataset, count in sorted(skipped.items()))
        )
    print(f"ks: {', '.join(str(k) for k in ks)}")
    print(f"db_path: {args.db_path}")
    print(f"cache_path: {args.cache_path}")

    try:
        scores = evaluate_recall(cases, args.db_path, args.cache_path, ks)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("\nRecall results")
    for dataset, dataset_scores in scores.items():
        parts = [f"{metric}={score:.4f}" for metric, score in dataset_scores.items()]
        print(f"{dataset}: " + ", ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
