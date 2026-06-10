"""BM25 关键词检索器 — jieba 分词 + BM25Okapi。"""

from langchain_core.documents import Document

import jieba
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """基于 jieba 中文分词和 BM25Okapi 的关键词检索器。

    用法:
        bm25 = BM25Retriever(corpus_docs)
        docs = bm25.invoke("Python 多线程", k=4)
    """

    def __init__(self, corpus: list[Document]):
        """
        Args:
            corpus: 文档列表，每篇的 page_content 作为检索内容。
        """
        self._corpus = corpus
        self._tokenized_corpus = [self._tokenize(doc.page_content) for doc in corpus]
        if not self._tokenized_corpus or all(len(tokens) == 0 for tokens in self._tokenized_corpus):
            self._index = None
        else:
            self._index = BM25Okapi(self._tokenized_corpus)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """jieba 精确模式分词，过滤空白 token。"""
        if not text:
            return []
        return [t.strip() for t in jieba.cut(text) if t.strip()]

    def invoke(self, query: str, k: int = 4) -> list[Document]:
        """检索 top-k 文档。

        Args:
            query: 查询文本。
            k: 返回文档数。

        Returns:
            按 BM25 分数降序、分数 > 0 的 Document 列表。
        """
        if self._index is None:
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = self._index.get_scores(tokens)
        if not scores.size:
            return []
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._corpus[i] for i in top_indices if scores[i] > 0]

    async def ainvoke(self, query: str, k: int = 4) -> list[Document]:
        """异步接口（委托同步 invoke）。"""
        return self.invoke(query, k=k)
