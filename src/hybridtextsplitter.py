import os
import re
import uuid
from typing import Literal

import dotenv
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from cachembedding import CacheEmbedding

dotenv.load_dotenv()


class HybridTextSplitter:
    def __init__(
        self,
        cache_path,  # embedding 缓存文件路径
        chunk_size=500,  # 子块大小（用于向量索引）
        chunk_overlap=50,  # 子块固定重叠 token 数
        parent_chunk_size=1200,  # 父块大小（用于回答时返回更完整上下文）
        parent_chunk_overlap=120,  # 父块切分重叠
        buffer_size=4,  # 预留参数：语义切分缓冲区大小
        threshold_type: Literal["percentile", "standard_deviation", "interquartile", "gradient"] = "percentile",  # 预留参数：语义断点阈值类型
        threshold_amount=95.0,  # 预留参数：语义断点阈值大小
        similarity_threshold=0.97,  # embedding 去重相似度阈值
        enable_filter=False,  # 是否启用 embedding 去重
        filter_max_docs=1200,  # 启用 embedding 去重时的最大文档数阈值
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.embedding_model = CacheEmbedding(cache_path)
        self.buffer_size = buffer_size
        self.threshold_type = threshold_type
        self.threshold_amount = threshold_amount
        self.similarity_threshold = similarity_threshold
        self.enable_filter = enable_filter
        self.filter_max_docs = filter_max_docs
        self.parent_store = {}

        # Step 1: 父块切分 — 结构标记优先 + tiktoken 控长（三段精简为两段）
        self.parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=os.getenv("MODEL_NAME"),
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_chunk_overlap,
            separators=[
                "\n# ", "\n## ", "\n### ",
                "\n一、", "\n二、", "\n三、",
                "\n1.", "\n2.", "\n3.",
                "\n- ", "\n* ",
                "\n\n", "\n",
                "。", "！", "？", "，",
            ],
        )

        # Step 2: 子块 token 控长（固定 overlap + 自适应 overlap 兜底）
        self.child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=os.getenv("MODEL_NAME"),
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，"],
        )

        # Step 3: 可选 embedding 去重
        if self.enable_filter:
            self.filter = EmbeddingsRedundantFilter(
                embeddings=self.embedding_model,
                similarity_threshold=self.similarity_threshold,
            )

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _ends_with_sentence_boundary(text: str) -> bool:
        return text.rstrip().endswith(("。", "！", "？", "；", ".", "!", "?", ";"))

    def _adaptive_overlap(self, prev_chunk: str, current_chunk: str) -> str:
        """当上一块在句中截断时，为下一块补充尾部上下文。"""
        if not prev_chunk or not current_chunk:
            return current_chunk
        if self._ends_with_sentence_boundary(prev_chunk):
            return current_chunk

        tail = prev_chunk[-self.chunk_overlap:]
        # 优先从最近的句号/分号处分割，避免补太多噪声。
        cut_idx = max(tail.rfind("。"), tail.rfind("！"), tail.rfind("？"), tail.rfind("；"), tail.rfind("."), tail.rfind("!"), tail.rfind("?"), tail.rfind(";"))
        if cut_idx >= 0:
            tail = tail[cut_idx + 1:].strip()
        if not tail:
            return current_chunk
        return f"{tail}\n{current_chunk}"

    def _lightweight_dedup(self, docs: list[Document]) -> list[Document]:
        """文本级去重，优先去掉重复块，减少 embedding 压力。"""
        seen = set()
        unique_docs = []
        for doc in docs:
            key = self._normalize_text(doc.page_content)
            if not key or key in seen:
                continue
            seen.add(key)
            unique_docs.append(doc)
        return unique_docs

    def _build_parent_docs(self, documents: list[Document]) -> list[Document]:
        parent_docs = self.parent_splitter.split_documents(documents)
        results = []
        self.parent_store = {}
        for doc in parent_docs:
            parent_content = doc.page_content.strip()
            if not parent_content:
                continue
            metadata = dict(doc.metadata)
            metadata["parent_id"] = metadata.get("parent_id") or str(uuid.uuid4())  # 子根据 ID 召回父
            metadata["chunk_level"] = "parent"
            self.parent_store[metadata["parent_id"]] = parent_content
            results.append(Document(page_content=parent_content, metadata=metadata))
        return results

    def _build_child_docs(self, parent_docs: list[Document]) -> list[Document]:
        child_docs = []
        for parent in parent_docs:
            pieces = self.child_splitter.split_text(parent.page_content)
            prev_piece = ""
            for idx, piece in enumerate(pieces):
                piece = piece.strip()
                if not piece:
                    continue
                piece = self._adaptive_overlap(prev_piece, piece)
                metadata = dict(parent.metadata)
                metadata["chunk_level"] = "child"
                metadata["chunk_index"] = idx
                child_docs.append(Document(page_content=piece, metadata=metadata))
                prev_piece = piece
        return child_docs

    def split(self, documents: list[Document]) -> list[Document]:
        """父块切分 + 子块切分 + 轻量去重 + 可选 embedding 去重。"""
        print("Step 1️⃣ 父块切分 ...")
        parent_docs = self._build_parent_docs(documents)
        print(f"  → 父块数量: {len(parent_docs)}")

        print("Step 2️⃣ 子块切分 ...")
        child_docs = self._build_child_docs(parent_docs)
        print(f"  → 子块数量: {len(child_docs)}")

        print("Step 3️⃣ 文本级轻量去重 ...")
        child_docs = self._lightweight_dedup(child_docs)
        print(f"  → 去重后子块: {len(child_docs)}")

        if self.enable_filter:
            if len(child_docs) > self.filter_max_docs:
                print(f"Step 4️⃣ 跳过 Embedding 去重（块数 {len(child_docs)} 超过阈值 {self.filter_max_docs}）")
            else:
                print("Step 4️⃣ Embedding 去重 ...")
                child_docs = self.filter.transform_documents(child_docs)
                print(f"  → Embedding 去重后: {len(child_docs)}")

        print("✅ 切分完成")
        return child_docs
