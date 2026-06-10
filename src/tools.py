"""检索工具 — LangChain @tool 封装，供 Agent 调用。

使用前需调用 init_tools() 注入检索器实例。
"""

from langchain_core.documents import Document
from langchain_core.tools import tool


_vector_retriever = None
_reranker = None
_bm25_retriever = None


def init_tools(vector_retriever, reranker, bm25_retriever=None):
    """注入检索器实例 — 由 cli._load_retriever() 在构建图前调用。

    Args:
        vector_retriever: LangChain retriever (ChromaDB, top20 粗排)。
        reranker: ReRanker 实例 (CrossEncoder 精排)。
        bm25_retriever: BM25Retriever 实例，None 时跳过。
    """
    global _vector_retriever, _reranker, _bm25_retriever
    _vector_retriever = vector_retriever
    _reranker = reranker
    _bm25_retriever = bm25_retriever


def _format_docs(docs: list[Document]) -> str:
    if not docs:
        return "（无结果）"
    lines = []
    for i, doc in enumerate(docs, 1):
        content = doc.page_content[:300]
        meta = doc.metadata or {}
        source = meta.get("dataset", meta.get("source", "unknown"))
        lines.append(f"[{i}] ({source}) {content}")
    return "\n\n".join(lines)


@tool
def vector_search(query: str) -> str:
    """语义检索：向量相似度搜索。适用模糊概念、解释类、同义词丰富的查询。

    Args:
        query: 检索查询词（LLM 可自行改写以提升召回）。
    """
    docs = _vector_retriever.invoke(query)
    docs = _reranker.rerank(query, docs, top_k=4)
    return _format_docs(docs)


@tool
def bm25_search(query: str) -> str:
    """关键词检索：BM25 精确匹配。适用专有名词、技术型号、API 名称等精准查询。

    Args:
        query: 检索查询词（保留核心关键词）。
    """
    docs = _bm25_retriever.invoke(query, k=4)
    return _format_docs(docs)
