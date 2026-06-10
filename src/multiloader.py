import hashlib
import json
import os
import random

from datasets import load_dataset
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document


class MultiLoader(BaseLoader):
    """加载指定目录下的 huggingface 数据集和本地文件"""
    def __init__(self, path: str):
        super().__init__()
        self.path = path

    @staticmethod
    def _convert_huggingface_path(dirname: str) -> str:
        """将 huggingface 缓存目录转换为 huggingface path"""
        return dirname.replace("___", "/").replace("---", "/")

    @staticmethod
    def _is_huggingface_path(filename: str) -> bool:
        return "___" in filename or "---" in filename or "/" in filename

    @staticmethod
    def make_md5(text: str):
        if not text:
            return ""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _record_to_document(self, record: dict) -> Document:
        """将不同数据集字段统一映射为 Document。"""
        # LongBench 标准字段: input/context/answers
        if "context" in record and ("input" in record or "answers" in record):
            context = record.get("context") or ""
            answers = record.get("answers") or []

            return Document(
                page_content=context,
                metadata={
                    "question": record.get("input"),
                    "answers": answers,
                    "dataset": record.get("dataset"),
                    "language": record.get("language"),
                    "hash": self.make_md5(context),
                },
            )

        # 兼容旧格式: positive_doc + question/answer
        positive_docs = record.get("positive_doc") or []
        if isinstance(positive_docs, list) and positive_docs:
            first_doc = positive_docs[0] or {}
            content = first_doc.get("content") or ""
            return Document(
                page_content=content,
                metadata={
                    "question": record.get("question"),
                    "answer": record.get("answer"),
                    "datatype": first_doc.get("datatype"),
                    "title": first_doc.get("title"),
                    "hash": self.make_md5(content),
                },
            )

        # 通用兜底字段
        content = record.get("content") or record.get("text") or ""
        return Document(
            page_content=content,
            metadata={
                "question": record.get("question") or record.get("input"),
                "answer": record.get("answer"),
                "hash": self.make_md5(content),
            },
        )

    @staticmethod
    def _parse_hf_dataset_name(filename: str):
        """支持 repo#config 语法，例如 zai-org/LongBench#multifieldqa_zh。"""
        if "#" in filename:
            dataset_name, dataset_config = filename.split("#", 1)
            return dataset_name, dataset_config
        return filename, None

    def _load_file(self, filename: str, sample_num=100, datasets: list[str] | None = None) -> list[Document]:
        """加载 huggingface 数据或本地文件"""
        path = os.path.join(self.path, filename)

        # 本地路径优先
        if os.path.isdir(path):
            docs = []
            for sub_item in os.listdir(path):
                # 按数据集过滤：在文件级匹配（去掉扩展名后和 datasets 比对）
                if datasets is not None:
                    item_stem = os.path.splitext(sub_item)[0]
                    if item_stem not in datasets:
                        continue
                docs.extend(self._load_file(os.path.join(filename, sub_item), sample_num=sample_num, datasets=datasets))
            return docs

        if os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            if ext == '.txt':
                sub_loader = TextLoader(path, encoding="utf-8")
            elif ext == '.pdf':
                sub_loader = PyPDFLoader(path)
            elif ext == '.csv':
                sub_loader = CSVLoader(path)
            elif ext == '.json':
                sub_loader = JSONLoader(
                    file_path=path,
                    jq_schema="""
                    .[] | {
                        question : .question,
                        answer : .answer,
                        content : .context
                    }
                    """,
                    metadata_func=lambda record, metadata: {
                        "question": record.get("question"),
                        "answer": record.get("answer"),
                        "source": metadata.get("source"),
                    },
                    content_key="content"
                )
            elif ext == '.jsonl':
                docs = []
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            record = json.loads(line)
                            doc = self._record_to_document(record)
                            if doc.page_content:
                                docs.append(doc)
                    return docs
                except Exception as e:
                    return [Document(page_content="", metadata={"source": path, "error": str(e)})]
            elif ext == '.html':
                sub_loader = UnstructuredHTMLLoader(path, mode="elements", strategy="fast")
            elif ext == '.md':
                sub_loader = UnstructuredMarkdownLoader(path, strategy="fast")
            else:
                return [Document(page_content="", metadata={"source": path, "error": "unsupported file type"})]
            try:
                return sub_loader.load()
            except Exception as e:
                return [Document(page_content="", metadata={"source": path, "error": str(e)})]

        # 加载 huggingface 数据
        if self._is_huggingface_path(filename):
            print(f"😀加载 HuggingFace 数据集：{filename}")
            try:
                dataset_name, dataset_config = self._parse_hf_dataset_name(filename)
                load_kwargs = {
                    "path": dataset_name,
                    "cache_dir": "./data/huggingface",
                }
                if dataset_config:
                    load_kwargs["name"] = dataset_config

                # LongBench 官方子任务主要提供 test split
                if "longbench" in dataset_name.lower():
                    dataset = load_dataset(split="test", **load_kwargs)
                else:
                    dataset = load_dataset(split="train", **load_kwargs)

                sample = dataset.shuffle().select(range(min(sample_num, len(dataset))))
            except Exception as e:
                raise RuntimeError(f"❌ 加载数据集失败: {e}")
            # 加载成 document
            docs = [self._record_to_document(record) for record in sample]
            docs = [doc for doc in docs if doc.page_content]
            return docs

        return []

    def load(self, sample_num=32, datasets: list[str] | None = None):
        """加载路径下所有文件，并随机抽样固定数量文档。

        Args:
            sample_num: 采样文档数量。
            datasets: 限定加载的数据集名称列表（如 ["passage_retrieval_en", "2wikimqa_e"]），
                      None 表示加载全部。
        """
        items = os.listdir(self.path)
        docs = []
        for item in items:
            if item == "huggingface":
                # 获取 huggingface 文件夹下所有缓存目录
                dirs = os.listdir(os.path.join(self.path, item))
                for dirname in dirs:
                    path = self._convert_huggingface_path(dirname)
                    docs.extend(self._load_file(path))
            else:
                docs.extend(self._load_file(item, datasets=datasets))

        if sample_num is not None and sample_num > 0 and len(docs) > sample_num:
            docs = random.sample(docs, sample_num)
        return docs


if __name__ == "__main__":
    loader = MultiLoader("/mnt/e/huggingface/dataset")
    documents = loader.load()
    print(f"总共加载了 {len(documents)} 条文档")
    if documents:
        first_doc = documents[0]
        print(f"首条文档长度: {len(first_doc.page_content)}")
        print(f"首条metadata字段: {list(first_doc.metadata.keys())}")
