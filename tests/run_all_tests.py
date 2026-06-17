"""综合自动化测试入口。

默认执行两类检查：
- 快速单元测试：索引写入逻辑、指标公式、一致性汇总逻辑。
- 真实索引一致性检查：读取当前 Chroma、BM25 和 parent_store，确认落盘数据一致。

真实向量检索评估会加载 embedding，耗时更长；需要时用 ``--with-recall`` 开启。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_step(name: str, args: list[str], env: dict[str, str] | None = None) -> int:
    print(f"\n== {name} ==", flush=True)
    result = subprocess.run(args, cwd=PROJECT_ROOT, env=env, check=False)
    if result.returncode == 0:
        print(f"PASS: {name}", flush=True)
    else:
        print(f"FAIL: {name} (exit {result.returncode})", flush=True)
    return result.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行项目综合自动化测试。")
    parser.add_argument(
        "--with-recall",
        action="store_true",
        help="额外运行真实向量检索评估，输出 Recall/MRR/nDCG。",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1,
        help="真实检索评估每个数据集采样数量，仅在 --with-recall 时生效。",
    )
    parser.add_argument(
        "--ks",
        default="1,5",
        help="真实检索评估的 K 列表，仅在 --with-recall 时生效。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    python = sys.executable

    steps = [
        (
            "索引写入完整性单元测试",
            [python, "tests/test_indexing_integrity.py"],
            None,
        ),
        (
            "RAG 排序指标公式单元测试",
            [python, "tests/test_rag_evaluation_metrics.py"],
            None,
        ),
        (
            "索引一致性汇总逻辑单元测试",
            [python, "tests/test_index_consistency.py", "unittest"],
            None,
        ),
        (
            "展示级 HTTP API 契约测试",
            [python, "tests/test_api_contract.py"],
            None,
        ),
        (
            "真实落盘索引一致性检查",
            [python, "tests/test_index_consistency.py"],
            None,
        ),
    ]

    if args.with_recall:
        recall_env = os.environ.copy()
        recall_env.setdefault("EMBEDDING_DEVICE", "cpu")
        recall_env.setdefault("LANGCHAIN_TRACING_V2", "false")
        recall_env.setdefault("LANGSMITH_TRACING", "false")
        steps.append(
            (
                "真实向量检索 Recall/MRR/nDCG 评估",
                [
                    python,
                    "tests/test_vector_recall.py",
                    "--sample-size",
                    str(args.sample_size),
                    "--ks",
                    args.ks,
                ],
                recall_env,
            )
        )

    failures = 0
    for name, command, env in steps:
        failures += 1 if _run_step(name, command, env) != 0 else 0

    print("\n== Summary ==", flush=True)
    print(f"total: {len(steps)}, failed: {failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
