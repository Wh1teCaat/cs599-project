"""索引构建 — 加载文档、切分、ChromaDB 向量化、BM25 语料、parent_store。

离线执行，与检索逻辑完全解耦。只由 cli.build() 调用。
"""

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
    """索引构建模式。"""
    ONLINE = "online"    # 只读
    OFFLINE = "offline"  # 可新建/追加


class IndexBuilder:
    """索引构建器。

    加载文档 → HybridTextSplitter 切分 → Chroma.from_documents 向量化
    → 保存 bm25_corpus.json + parent_store.json。
    """

    def __init__(self, data_path: str, db_path: str, cache_path: str,
                 mode: RunMode = RunMode.OFFLINE):
        """
        Args:
            data_path: 数据集目录路径。
            db_path: ChromaDB 持久化目录。
            cache_path: Embedding 缓存文件路径。
            mode: 构建模式。
        """
        self.data_path = data_path
        self.db_path = db_path
        self.cache_path = cache_path
        self.parent_store_path = os.path.join(self.db_path, "parent_store.json")
        self.bm25_corpus_path = os.path.join(self.db_path, "bm25_corpus.json")
        self.loader = MultiLoader(self.data_path)
        self.splitter = HybridTextSplitter(self.cache_path, enable_filter=False)
        self.embedding = self.splitter.embedding_model
        self.mode = mode
        self.parent_store: dict = {}

    # ── parent_store 持久化 ────────────────────────────────

    def _load_parent_store(self) -> dict:
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
        existing = self._load_parent_store()
        existing.update(self.parent_store)
        self.parent_store = existing
        with open(self.parent_store_path, "w", encoding="utf-8") as f:
            json.dump(self.parent_store, f, ensure_ascii=False)

    # ── BM25 语料持久化 ────────────────────────────────────

    def _load_bm25_corpus(self) -> list[dict]:
        if not os.path.exists(self.bm25_corpus_path):
            return []
        try:
            with open(self.bm25_corpus_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save_bm25_corpus(self, corpus: list[dict]):
        os.makedirs(self.db_path, exist_ok=True)
        with open(self.bm25_corpus_path, "w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False)

    @staticmethod
    def _docs_to_corpus_entries(docs: list[Document]) -> list[dict]:
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]

    @staticmethod
    def _corpus_entries_to_docs(entries: list[dict]) -> list[Document]:
        return [Document(page_content=e["content"], metadata=e.get("metadata", {})) for e in entries]

    # ── 文档处理 ──────────────────────────────────────────

    def _process_documents(self) -> list[Document]:
        docs = self.loader.load()
        print("文件加载完成")

        docs = self.splitter.split(docs)
        self.parent_store.update(self.splitter.parent_store)
        print("文档切分完成")
        return docs

    @staticmethod
    def _make_md5(text: str) -> str:
        if not text:
            return ""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _serialize_metadata_for_chroma(docs: list[Document]) -> list[Document]:
        """将 Chroma 不支持的复杂 metadata 值序列化为 JSON 字符串。"""
        serialized = []
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
            serialized.append(Document(page_content=doc.page_content, metadata=normalized))
        return serialized

    # ── 构建 / 追加 ───────────────────────────────────────

    def _build_db(self) -> Chroma:
        docs = self._process_documents()
        # 保存 BM25 语料（在 metadata 序列化之前保存原始 docs）
        self._save_bm25_corpus(self._docs_to_corpus_entries(docs))
        docs = self._serialize_metadata_for_chroma(docs)
        docs = filter_complex_metadata(docs)
        db = Chroma.from_documents(
            documents=docs, embedding=self.embedding, persist_directory=self.db_path
        )
        self._save_parent_store()
        print("✅ 向量数据库构建完成")
        return db

    def _append_db(self, db: Chroma) -> Chroma:
        docs = self._process_documents()
        # 更新 BM25 语料：追加新文档（按 content hash 去重）
        existing_corpus = self._load_bm25_corpus()
        existing_hashes = {e.get("metadata", {}).get("hash") for e in existing_corpus}
        new_corpus = [
            entry
            for entry in self._docs_to_corpus_entries(docs)
            if self._make_md5(entry["content"]) not in existing_hashes
        ]
        existing_corpus.extend(new_corpus)
        self._save_bm25_corpus(existing_corpus)

        docs = self._serialize_metadata_for_chroma(docs)
        docs = filter_complex_metadata(docs)
        exist_ids = set(
            m.get("hash")
            for m in db.get(include=["metadatas"])["metadatas"]
            if m.get("hash")
        )
        docs = [d for d in docs if self._make_md5(d.page_content) not in exist_ids]

        if not docs:
            print("🟡 没有检测到新文档，数据库无需更新")
            return db

        db.add_documents(documents=docs)
        self._save_parent_store()
        print("✅ 向量数据库更新完成")
        return db

    # ── 主入口 ────────────────────────────────────────────

    def build(self) -> Chroma:
        """构建或更新数据库，返回 Chroma 实例。"""
        if not os.path.exists(self.db_path) or not os.listdir(self.db_path):
            print("⚠️ 未检测到持久化文件，正在重新构建数据库...")
            if self.mode == RunMode.OFFLINE:
                return self._build_db()
            else:
                raise RuntimeError("❌ 在线模式无法构建新数据库，请先初始化")
        else:
            print("✅ 加载已有数据库...")
            db = Chroma(
                persist_directory=self.db_path,
                embedding_function=CacheEmbedding(self.cache_path),
            )
            if self.mode == RunMode.OFFLINE:
                return self._append_db(db)
            return db
