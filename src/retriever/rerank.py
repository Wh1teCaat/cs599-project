"""CrossEncoder 精排器 — bge-reranker-v2-m3。"""

from langchain_core.documents import Document


class ReRanker:
    """基于 CrossEncoder 的精排器。

    对向量粗排结果重新打分排序。
    若未配置 model_name，退化为截断透传（pass-through）。

    用法:
        rr = ReRanker(model_name="/path/to/bge-reranker-v2-m3")
        docs = rr.rerank(query, coarse_docs, top_k=4)
    """

    def __init__(self, model_name: str | None = None):
        """
        Args:
            model_name: HuggingFace 模型路径，None 则透传。
        """
        self._model_name = model_name
        self._model = None

    def _lazy_load(self):
        """延迟加载 CrossEncoder，避免启动时占显存。"""
        if self._model is not None:
            return
        if self._model_name is None:
            return
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(self._model_name)

    def rerank(self, query: str, docs: list[Document], top_k: int = 4) -> list[Document]:
        """精排文档。

        Args:
            query: 查询文本。
            docs: 粗排候选文档列表。
            top_k: 返回的前 N 篇。

        Returns:
            按 CrossEncoder 相关性分数降序的 top_k 文档。
        """
        if not docs:
            return []
        self._lazy_load()
        if self._model is None:
            return docs[:top_k]
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self._model.predict(pairs)
        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]
