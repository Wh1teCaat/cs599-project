"""RAG 服务 CLI 入口 — LangGraph Agentic 混合检索。

用法:
  python main.py build [--sample N] [--datasets D1,D2] [--mode offline|online]
      构建 / 更新向量数据库。--sample 限定文档采样数，--datasets 限定数据集，
      --mode offline 可新建/追加，online 只读。

  python main.py query <text> [--topk N]
      单次检索。Agent 自主选择向量/BM25/双路，RRF 融合，父块回填。

  python main.py serve
      启动展示级 HTTP API 服务，提供 /health 和 /query。
"""

import argparse


def main():
    """CLI 入口：解析参数 → 分发到 build / query / serve。"""
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="RAG 知识库服务：向量化构建 + 混合检索",
    )
    sub = parser.add_subparsers(dest="command")

    # ---- build ----
    build_parser = sub.add_parser("build", help="构建/更新向量数据库")
    build_parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="采样数量（默认 None = 全量），设为 0 跳过加载仅更新已有库",
    )
    build_parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="限定加载的数据集名称，逗号分隔（如 passage_retrieval_en,2wikimqa_e），默认全部",
    )
    build_parser.add_argument(
        "--mode",
        choices=["offline", "online"],
        default="offline",
        help="运行模式：offline 可新建/追加，online 只读（默认 offline）",
    )

    # ---- query ----
    query_parser = sub.add_parser("query", help="检索知识库")
    query_parser.add_argument("text", help="查询文本")
    query_parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="返回文档数（默认 5）",
    )

    # ---- serve ----
    serve_parser = sub.add_parser("serve", help="启动展示级 HTTP API 服务")
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="监听地址（默认 127.0.0.1）",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="监听端口（默认 8000）",
    )

    args = parser.parse_args()

    if args.command == "build":
        from cli import build
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()] if args.datasets else None
        build(sample_num=args.sample, mode=args.mode, datasets=datasets)

    elif args.command == "query":
        from cli import query
        query(args.text, top_k=args.topk)

    elif args.command == "serve":
        from api import run_server
        run_server(host=args.host, port=args.port)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
