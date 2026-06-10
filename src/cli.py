"""CLI 胶水层 — 加载组件、组装管线。

职责:
  - build(): 索引构建（委托 IndexBuilder）
  - query(): 检索查询（组装工具 + 构建 GraphRetriever）

由 main.py 调用，不直接被外部使用。
"""

import json
import os
import time
from pathlib import Path

import dotenv
import yaml

dotenv.load_dotenv()

from indexer import IndexBuilder, RunMode
from retriever import BM25Retriever, ReRanker, load_vector_retriever
import tools
from graph_retriever import GraphRetriever


def _load_config() -> tuple[str, str, str]:
    """读取 config.yaml，返回 (data_path, db_path, cache_path)。"""
    root = Path(__file__).resolve().parent
    config = yaml.safe_load(open(root / "config.yaml", "r"))
    return (
        str(root / config["loader"]["data_path"]),
        str(root / config["retriever"]["db_path"]),
        str(root / config["embedding"]["cache_path"]),
    )


# ═══════════════════════════════════════════════════════════════
# build — 索引构建
# ═══════════════════════════════════════════════════════════════


def build(sample_num: int | None = None, mode: str = "offline",
          datasets: list[str] | None = None):
    """构建 / 更新向量数据库。

    Args:
        sample_num: 采样文档数，None 为全量，0 为跳过加载仅更新已有库。
        mode: "offline" 可新建+追加，"online" 只读。
        datasets: 限定数据集名称列表，None 表示全部。
    """
    data_path, db_path, cache_path = _load_config()
    run_mode = RunMode.OFFLINE if mode == "offline" else RunMode.ONLINE

    from multiloader import MultiLoader
    _original_load = MultiLoader.load

    def _patched_load(self):
        return _original_load(self, sample_num=sample_num, datasets=datasets)

    MultiLoader.load = _patched_load

    try:
        t0 = time.time()
        print(f"Mode: {run_mode.value}")
        builder = IndexBuilder(data_path, db_path, cache_path, mode=run_mode)
        builder.build()
        print(f"Done in {time.time() - t0:.0f}s")
    finally:
        MultiLoader.load = _original_load


# ═══════════════════════════════════════════════════════════════
# query — Agent 检索
# ═══════════════════════════════════════════════════════════════


def _load_retriever() -> GraphRetriever:
    """加载 ChromaDB / BM25 语料 / ReRanker，注入工具，构建 GraphRetriever。"""
    _, db_path, cache_path = _load_config()
    bm25_path = os.path.join(db_path, "bm25_corpus.json")
    parent_path = os.path.join(db_path, "parent_store.json")

    # 向量检索器
    vector_retriever = load_vector_retriever(db_path, cache_path)

    # BM25 检索器
    bm25_retriever = None
    if os.path.exists(bm25_path):
        with open(bm25_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        from langchain_core.documents import Document
        docs = [Document(page_content=e["content"], metadata=e.get("metadata", {}))
                for e in corpus]
        if docs:
            bm25_retriever = BM25Retriever(docs)

    # ReRanker
    reranker = ReRanker(model_name=os.getenv("RERANK_MODEL_NAME"))

    # 注入检索器到工具模块
    tools.init_tools(vector_retriever, reranker, bm25_retriever)

    # 工具列表
    tool_list = [tools.vector_search]
    if bm25_retriever:
        tool_list.append(tools.bm25_search)

    # parent store
    parent_store = {}
    if os.path.exists(parent_path):
        with open(parent_path, "r", encoding="utf-8") as f:
            parent_store = json.load(f)

    return GraphRetriever(
        tools=tool_list,
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        parent_store=parent_store,
    )


def query(text: str, top_k: int = 5, history: list[dict] | None = None):
    """单次检索（同步）。

    Args:
        text: 查询文本。
        top_k: 打印的文档数。
        history: 对话历史 [{"role": "user/assistant", "content": "..."}, ...]。
    """
    retriever = _load_retriever()
    docs = retriever.invoke(text, history=history)

    print(f"Query: {text}")
    print(f"Results: {len(docs)}")

    for i, doc in enumerate(docs[:top_k], 1):
        content = doc["content"]
        if len(content) > 250:
            content = content[:250] + "…"
        print(f"\n--- doc {i} ---")
        print(content)
        m = doc.get("metadata") or {}
        for key in ("source", "dataset", "parent_id", "chunk_level"):
            if key in m:
                v = m[key]
                if isinstance(v, str) and len(v) > 120:
                    v = v[:120] + "…"
                print(f"  {key}: {v}")
