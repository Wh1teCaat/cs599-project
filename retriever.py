import hashlib
import json
import os
from enum import Enum

from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document

from cachembedding import CacheEmbedding
from hybridtextsplitter import HybridTextSplitter
from multiloader import MultiLoader


class RunMode(Enum):
    ONLINE = "online"
    OFFLINE = "offline"


class ParentDocumentRetriever:
    """先检索子块，再回填父块内容，提升回答上下文完整性。"""
    def __init__(self, base_retriever, parent_store=None):
        self.base_retriever = base_retriever
        self.parent_store = parent_store or {}

    def _to_parent_docs(self, docs: list[Document]) -> list[Document]:
        grouped = {}
        for doc in docs:
            metadata = dict(doc.metadata or {})
            parent_id = metadata.get("parent_id") or hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            if parent_id in grouped:
                continue

            parent_content = self.parent_store.get(parent_id)
            if parent_content:
                metadata["chunk_level"] = "parent"
                grouped[parent_id] = Document(page_content=parent_content, metadata=metadata)
            else:
                grouped[parent_id] = doc
        return list(grouped.values())

    def invoke(self, query: str):
        docs = self.base_retriever.invoke(query)
        return self._to_parent_docs(docs)

    async def ainvoke(self, query: str):
        docs = await self.base_retriever.ainvoke(query)
        return self._to_parent_docs(docs)


class RAG:
    def __init__(self, data_path, db_path, cache_path, mode=RunMode.OFFLINE):
        self.data_path = data_path
        self.db_path = db_path
        self.cache_path = cache_path
        self.parent_store_path = os.path.join(self.db_path, "parent_store.json")
        self.loader = MultiLoader(self.data_path)
        self.splitter = HybridTextSplitter(self.cache_path, enable_filter=True)
        self.embedding = self.splitter.embedding_model
        self.mode = mode
        self.parent_store = {}

    def _load_parent_store(self):
        if not os.path.exists(self.parent_store_path):
            return {}
        try:
            with open(self.parent_store_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_parent_store(self):
        os.makedirs(self.db_path, exist_ok=True)
        # 合并已存在映射，避免分批随机加载时覆盖历史父块。
        existing = self._load_parent_store()
        existing.update(self.parent_store)
        self.parent_store = existing
        with open(self.parent_store_path, "w", encoding="utf-8") as f:
            json.dump(self.parent_store, f, ensure_ascii=False)

    def _process_documents(self):
        docs = self.loader.load()
        print("文件加载完成")

        docs = self.splitter.split(docs)
        self.parent_store.update(self.splitter.parent_store)
        print("文档切分完成")
        return docs

    @staticmethod
    def make_md5(text: str):
        if not text:
            return ""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _serialize_metadata_for_chroma(docs: list[Document]) -> list[Document]:
        """将 Chroma 不支持的复杂 metadata 值序列化为 JSON 字符串。"""
        serialized_docs = []
        for doc in docs:
            metadata = dict(doc.metadata or {})
            normalized = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    normalized[key] = value
                elif isinstance(value, (list, dict)):
                    normalized[key] = json.dumps(value, ensure_ascii=False)
                else:
                    normalized[key] = str(value)
            serialized_docs.append(Document(page_content=doc.page_content, metadata=normalized))
        return serialized_docs

    def _build_db(self):
        docs = self._process_documents()
        docs = self._serialize_metadata_for_chroma(docs)
        docs = filter_complex_metadata(docs)
        db = Chroma.from_documents(
            documents=docs,
            embedding=self.embedding,
            persist_directory=self.db_path
        )
        self._save_parent_store()
        print("✅ 向量数据库构建完成")
        return db

    def _append_db(self, db):
        docs = self._process_documents()
        docs = self._serialize_metadata_for_chroma(docs)
        docs = filter_complex_metadata(docs)
        exist_docs = set(
            m.get("hash")
            for m in db.get(include=["metadatas"])["metadatas"]
            if m.get("hash")    # 不存在返回 None
        )
        docs = [d for d in docs if self.make_md5(d.page_content) not in exist_docs]

        if not docs:
            print("🟡 没有检测到新文档，数据库无需更新")
            return db

        db.add_documents(documents=docs)
        self._save_parent_store()
        return db

    def get_retriever(self):
        self.parent_store = self._load_parent_store()
        if not os.path.exists(self.db_path) or not os.listdir(self.db_path):
            print("⚠️ 未检测到持久化文件，正在重新构建数据库...")
            if self.mode == RunMode.OFFLINE:
                db = self._build_db()
            else:
                raise RuntimeError("❌ 更新无法构建新数据库，请先初始化")
        else:
            print("✅ 加载已有数据库...")
            db = Chroma(
                persist_directory=self.db_path,
                embedding_function=CacheEmbedding(self.cache_path)
            )

            if self.mode == RunMode.OFFLINE:
                db = self._append_db(db)
        child_retriever = db.as_retriever()
        return ParentDocumentRetriever(child_retriever, parent_store=self.parent_store)
