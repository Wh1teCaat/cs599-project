"""索引写入完整性单元测试。

这个文件检查本地 RAG 索引的构建和追加路径：embedding 缓存写入、Chroma
分批写入、BM25 同步、parent_store 持久化，以及 Chroma metadata 规范化。
测试使用 fake 对象和临时目录，不会修改真实落盘索引。
"""

from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cachembedding import CacheEmbedding  # noqa: E402

if "multiloader" not in sys.modules:
    multiloader_stub = types.ModuleType("multiloader")
    multiloader_stub.MultiLoader = object
    sys.modules["multiloader"] = multiloader_stub

if "hybridtextsplitter" not in sys.modules:
    splitter_stub = types.ModuleType("hybridtextsplitter")
    splitter_stub.HybridTextSplitter = object
    sys.modules["hybridtextsplitter"] = splitter_stub

from indexer import IndexBuilder, RunMode  # noqa: E402


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]


class FakeDb:
    def __init__(self, *, fail_add: bool = False, fail_on_call: int | None = None):
        self.fail_add = fail_add
        self.fail_on_call = fail_on_call
        self.added_docs: list[Document] = []
        self.add_calls: list[int] = []

    def get(self, include: list[str]):
        return {"metadatas": [], "documents": []}

    def add_documents(self, documents: list[Document]):
        self.add_calls.append(len(documents))
        if self.fail_add or self.fail_on_call == len(self.add_calls):
            raise RuntimeError("add failed")
        self.added_docs.extend(documents)


class FakeChroma:
    instances: list[FakeDb] = []
    fail_on_call: int | None = None

    def __new__(cls, *args, **kwargs):
        db = FakeDb(fail_on_call=cls.fail_on_call)
        cls.instances.append(db)
        return db

    @classmethod
    def from_documents(cls, *args, **kwargs):
        raise AssertionError("build should add vector documents in batches")


class IndexingIntegrityTests(unittest.TestCase):
    def test_embed_batch_preserves_input_order_when_cache_is_partial(self):
        embedding = object.__new__(CacheEmbedding)
        embedding.cache = {
            CacheEmbedding._text_hash("cached-a"): [100.0],
            CacheEmbedding._text_hash("cached-b"): [200.0],
        }
        embedding.embeddings = FakeEmbeddings()
        embedding._save_cache = lambda: None

        result = embedding._embed_batch(["cached-a", "new-one", "cached-b", "xx"])

        self.assertEqual(result, [[100.0], [7.0], [200.0], [2.0]])

    def test_save_cache_writes_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "embeddings_cache.json"
            embedding = object.__new__(CacheEmbedding)
            embedding.cache_path = str(cache_path)
            embedding.cache = {"a": [1.0], "b": [2.0]}

            embedding._save_cache()

            with cache_path.open("r", encoding="utf-8") as f:
                self.assertEqual(json.load(f), embedding.cache)
            self.assertFalse(cache_path.with_suffix(".json.tmp").exists())

    def test_append_db_does_not_update_bm25_when_vector_add_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = object.__new__(IndexBuilder)
            builder.db_path = tmpdir
            builder.bm25_corpus_path = str(Path(tmpdir) / "bm25_corpus.json")
            builder.parent_store_path = str(Path(tmpdir) / "parent_store.json")
            builder.parent_store = {}
            builder.mode = RunMode.OFFLINE

            docs = [
                Document(
                    page_content="new chunk",
                    metadata={"hash": "source-hash", "dataset": "2wikimqa_e"},
                )
            ]

            with patch.object(builder, "_process_documents", return_value=docs):
                with self.assertRaises(RuntimeError):
                    builder._append_db(FakeDb(fail_add=True))

            self.assertFalse(Path(builder.bm25_corpus_path).exists())

    def test_append_db_persists_bm25_for_each_successful_vector_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = object.__new__(IndexBuilder)
            builder.db_path = tmpdir
            builder.bm25_corpus_path = str(Path(tmpdir) / "bm25_corpus.json")
            builder.parent_store_path = str(Path(tmpdir) / "parent_store.json")
            builder.parent_store = {"parent": {"text": "stored"}}
            builder.mode = RunMode.OFFLINE

            docs = [
                Document(page_content=f"new chunk {idx}", metadata={})
                for idx in range(2501)
            ]

            with patch.object(builder, "_process_documents", return_value=docs):
                with self.assertRaises(RuntimeError):
                    builder._append_db(FakeDb(fail_on_call=3))

            with open(builder.bm25_corpus_path, "r", encoding="utf-8") as f:
                corpus = json.load(f)
            with open(builder.parent_store_path, "r", encoding="utf-8") as f:
                parent_store = json.load(f)

            self.assertEqual(len(corpus), 2000)
            self.assertEqual(corpus[0]["content"], "new chunk 0")
            self.assertEqual(corpus[-1]["content"], "new chunk 1999")
            self.assertEqual(parent_store, {"parent": {"text": "stored"}})

    def test_append_db_adds_vector_documents_in_1000_document_batches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = object.__new__(IndexBuilder)
            builder.db_path = tmpdir
            builder.bm25_corpus_path = str(Path(tmpdir) / "bm25_corpus.json")
            builder.parent_store_path = str(Path(tmpdir) / "parent_store.json")
            builder.parent_store = {}
            builder.mode = RunMode.OFFLINE

            docs = [
                Document(page_content=f"new chunk {idx}", metadata={})
                for idx in range(2501)
            ]
            db = FakeDb()

            with patch.object(builder, "_process_documents", return_value=docs):
                builder._append_db(db)

            self.assertEqual(db.add_calls, [1000, 1000, 501])
            self.assertEqual(len(db.added_docs), 2501)

    def test_build_db_adds_vector_documents_in_1000_document_batches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = object.__new__(IndexBuilder)
            builder.db_path = tmpdir
            builder.bm25_corpus_path = str(Path(tmpdir) / "bm25_corpus.json")
            builder.parent_store_path = str(Path(tmpdir) / "parent_store.json")
            builder.parent_store = {}
            builder.embedding = object()

            docs = [
                Document(page_content=f"new chunk {idx}", metadata={})
                for idx in range(2501)
            ]

            FakeChroma.instances = []
            FakeChroma.fail_on_call = None
            with patch.object(builder, "_process_documents", return_value=docs):
                with patch("indexer.Chroma", FakeChroma):
                    db = builder._build_db()

            self.assertIs(db, FakeChroma.instances[0])
            self.assertEqual(db.add_calls, [1000, 1000, 501])
            self.assertEqual(len(db.added_docs), 2501)

    def test_build_db_persists_bm25_for_each_successful_vector_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = object.__new__(IndexBuilder)
            builder.db_path = tmpdir
            builder.bm25_corpus_path = str(Path(tmpdir) / "bm25_corpus.json")
            builder.parent_store_path = str(Path(tmpdir) / "parent_store.json")
            builder.parent_store = {"parent": "stored"}
            builder.embedding = object()

            docs = [
                Document(page_content=f"new chunk {idx}", metadata={})
                for idx in range(2501)
            ]

            FakeChroma.instances = []
            FakeChroma.fail_on_call = 3
            try:
                with patch.object(builder, "_process_documents", return_value=docs):
                    with patch("indexer.Chroma", FakeChroma):
                        with self.assertRaises(RuntimeError):
                            builder._build_db()
            finally:
                FakeChroma.fail_on_call = None

            with open(builder.bm25_corpus_path, "r", encoding="utf-8") as f:
                corpus = json.load(f)
            with open(builder.parent_store_path, "r", encoding="utf-8") as f:
                parent_store = json.load(f)

            self.assertEqual(len(corpus), 2000)
            self.assertEqual(corpus[0]["content"], "new chunk 0")
            self.assertEqual(corpus[-1]["content"], "new chunk 1999")
            self.assertEqual(parent_store, {"parent": "stored"})

    def test_serialize_metadata_for_chroma_outputs_only_supported_metadata_types(self):
        docs = [
            Document(
                page_content="chunk",
                metadata={
                    "text": "value",
                    "count": 3,
                    "score": 1.5,
                    "enabled": True,
                    "missing": None,
                    "items": ["a", "b"],
                    "payload": {"k": "v"},
                    "path": Path("source.md"),
                },
            )
        ]

        serialized = IndexBuilder._serialize_metadata_for_chroma(docs)

        self.assertEqual(
            serialized[0].metadata,
            {
                "text": "value",
                "count": 3,
                "score": 1.5,
                "enabled": True,
                "items": '["a", "b"]',
                "payload": '{"k": "v"}',
                "path": "source.md",
            },
        )


if __name__ == "__main__":
    unittest.main()
