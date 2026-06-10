"""ChromaDB 向量检索器加载。"""

from langchain_chroma import Chroma
from cachembedding import CacheEmbedding


def load_vector_retriever(db_path: str, cache_path: str, top_k: int = 20):
    """打开 ChromaDB，返回 LangChain retriever。

    Args:
        db_path: ChromaDB 持久化目录路径。
        cache_path: Embedding 缓存 JSON 文件路径。
        top_k: 粗排召回数量（默认 20，供给精排）。

    Returns:
        LangChain VectorStoreRetriever，invoke(query) 返回 list[Document]。
    """
    db = Chroma(
        persist_directory=db_path,
        embedding_function=CacheEmbedding(cache_path),
    )
    return db.as_retriever(search_kwargs={"k": top_k})
