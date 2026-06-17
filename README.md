# Agentic RAG 知识库检索服务

![Python](https://img.shields.io/badge/Python-3.10--3.12-blue)
![Framework](https://img.shields.io/badge/Framework-LangGraph-green)
![Status](https://img.shields.io/badge/Status-Final-green)

## 项目简介

一个基于 Agentic AI 的 RAG 服务：LLM 自主判断查询类型并选择检索策略（向量 / BM25 / 混合），替代传统的固定单次向量召回流程，解决 RAG 检索路径单一、难以适应多样化查询场景的问题。

## 核心特性

- **Agentic 检索决策**：基于 LangGraph 构建检索 Agent，由模型按问题类型自主选择向量检索、BM25 检索或混合检索。
- **Graph Retriever 流程**：将查询改写、工具调用、RRF 融合、充分性评估和父块回填组织成可追踪的图流程。
- **混合召回能力**：结合语义向量检索与 BM25 关键词检索，兼顾概念型问题和专有名词、型号、API 等精确查询。
- **父子块结构**：索引阶段保存子块用于召回，同时保留父块用于回答时补全上下文。
- **本地缓存与持久化**：支持 ChromaDB 向量库、BM25 语料和 embedding 缓存，减少重复构建成本。

## 项目方向

方向一：Agentic AI 原生开发 — 重点展示 LLM 如何参与检索路径规划，而不是只把 RAG 固定成单次向量召回。

## 技术栈

- AI IDE: Trae CN
- LLM: DeepSeek API
- 框架: LangGraph
- 容器: Docker

## 架构说明

详细架构见 [docs/architecture.md](docs/architecture.md)。核心流程如下：

```text
User Query
  -> rewrite_query
  -> agent decides tools
  -> vector_search / bm25_search
  -> optional RRF fusion
  -> sufficiency evaluation
  -> parent document expansion
  -> final documents
```

## 目录结构

```text
rag-service/
├── docs/                         # 项目文档
│   ├── architecture.md           # 详细架构说明
│   └── CS599_大作业报告.pdf       # 最终课程报告
├── src/                          # 项目源代码
│   ├── main.py                   # 命令入口（build / query / serve）
│   ├── cli.py                    # 命令行参数解析
│   ├── indexer.py                # 索引构建（分块、embedding、BM25 语料）
│   ├── graph_retriever.py        # LangGraph Agentic 检索图
│   ├── tools.py                  # 检索工具封装（@tool 供 Agent 调用）
│   ├── multiloader.py            # 多格式数据加载
│   ├── hybridtextsplitter.py     # 混合文本切分（结构 + 语义边界）
│   ├── cachembedding.py          # 嵌入缓存（磁盘持久化 + 去重）
│   ├── download_bge.py           # BGE 模型下载脚本
│   ├── config.yaml               # 数据集路径配置
│   └── retriever/                # 检索组件（BM25 / ReRank / Vector）
├── data/                          # 运行时数据（gitignore，构建后生成）
│   ├── chroma_db/                 # 向量库 + BM25 语料
│   └── embeddings_cache.json      # embedding 缓存
├── requirements.txt              # 项目依赖
├── .gitignore                    # 排除 .env、缓存、虚拟环境等
└── README.md
```

## 运行前检查

```bash
python --version          # 应输出 3.10.x ~ 3.12.x；暂不建议 Python 3.13
pip install -r requirements.txt
cd src && python main.py --help
```

确认 `main.py --help` 正常输出后再继续以下步骤。

## 环境搭建

### 1. 依赖安装

创建虚拟环境并安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> 如果不使用 PostgreSQL checkpoint，则无需安装可选依赖；相关 import 仅在设置了 `POSTGRES_URL` 时触发，不会影响基础检索功能。

### 2. 环境变量配置

在项目根目录创建 `.env`，至少配置模型调用所需的 Key 和模型名称。

> ⚠️ 不要把 API Key 硬编码到代码里，也不要提交 `.env`。

```env
# LLM 调用（DeepSeek API，兼容 OpenAI SDK）
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.deepseek.com/v1
MODEL_NAME=deepseek-chat

# Embedding 模型
HF_MODEL_NAME=BAAI/bge-m3
EMBEDDING_DEVICE=cuda            # 无 GPU 时改为 cpu

# Reranker 模型
RERANK_MODEL_NAME=BAAI/bge-reranker-v2-m3

# PostgreSQL checkpoint（可选，不填则使用内存存储）
POSTGRES_URL=
```

### 3. 启动步骤

```bash
# 修改 src/config.yaml 中的数据集路径、向量库路径和缓存路径
# 构建索引（小样本验证）
cd src
python main.py build --sample 100 --mode offline

# 执行查询
python main.py query "你的问题" --topk 5
```

首次配置预计 10-20 分钟；如需下载 BGE embedding / rerank 模型，额外耗时取决于网络。

## API 调用

Final 版当前提供两类稳定调用方式：命令行 API 和 Python 内部 API。`serve`
子命令作为 LangGraph 服务化入口预留，后续补齐 `langgraph.json` 和模块级
graph 导出后可扩展为 HTTP API。

### 1. 命令行 API

命令行 API 适合本地构建索引、运行 Demo、复现实验和课程展示。

```bash
cd src

# 查看全部命令
python main.py --help

# 构建或追加索引
python main.py build --sample 100 --mode offline

# 指定数据集构建
python main.py build --sample 100 --datasets passage_retrieval_en,2wikimqa_e --mode offline

# 执行一次 Agentic RAG 检索
python main.py query "对比微服务和单体架构在可观测性上的差异" --topk 5
```

`query` 调用内部会自动完成以下流程：

```text
用户问题
  -> LLM 改写查询
  -> Agent 选择 vector_search / bm25_search / 双路检索
  -> ReRanker 精排
  -> 双路时执行 RRF 融合
  -> LLM 充分性评估
  -> parent_store 父块回填
  -> 输出 final_docs
```

常用参数：

| 命令 | 参数 | 说明 |
|---|---|---|
| `build` | `--sample N` | 采样构建，便于小样本验证；不传则按配置加载 |
| `build` | `--datasets a,b` | 只加载指定数据集 |
| `build` | `--mode offline` | 可新建或追加索引 |
| `build` | `--mode online` | 只读加载已有索引 |
| `query` | `--topk N` | 控制打印的最终文档数量 |

### 2. Python 内部 API

Python API 适合在脚本、Notebook 或上层服务中直接复用检索能力。
以下示例默认在 `src/` 目录下执行；如果从项目根目录运行，需要先把 `src`
加入 `PYTHONPATH`。

```python
from cli import _load_retriever

retriever = _load_retriever()
docs = retriever.invoke(
    "BGE-M3 embedding 缓存如何减少重复构建成本？",
    history=[],
    thread_id="demo",
)

for doc in docs[:3]:
    print(doc["content"][:300])
    print(doc.get("metadata", {}))
```

核心接口：

| 接口 | 输入 | 输出 | 用途 |
|---|---|---|---|
| `IndexBuilder.build()` | 数据路径、向量库路径、缓存路径 | Chroma 实例 | 离线索引构建 |
| `GraphRetriever.invoke()` | `query`、`history`、`thread_id` | `list[dict]` | 同步 Agentic 检索 |
| `GraphRetriever.ainvoke()` | `query`、`history`、`thread_id` | `list[dict]` | 异步 Agentic 检索 |
| `BM25Retriever.invoke()` | 查询文本、`k` | `list[Document]` | 关键词召回 |
| `ReRanker.rerank()` | 查询、候选文档、`top_k` | `list[Document]` | CrossEncoder 精排 |

返回的 `final_docs` 结构如下：

```python
[
    {
        "content": "父块回填后的完整上下文文本",
        "metadata": {
            "dataset": "passage_retrieval_en",
            "parent_id": "...",
            "chunk_level": "parent"
        }
    }
]
```

### 3. 服务化 API 预留

`python main.py serve` 当前会调用 `langgraph serve`，用于后续 HTTP API
服务化扩展。由于 Final 版重点是本地 Agentic RAG 检索闭环，仓库当前未提交
`langgraph.json`，也未将 graph 以模块级变量导出，因此 HTTP 服务不作为本次
最终交付的稳定入口。

计划中的 HTTP 调用形式如下：

```bash
cd src
python main.py serve

# 默认服务地址由 langgraph serve 提供，通常为：
# http://127.0.0.1:2024
```

后续只需补充 LangGraph 服务配置，即可把当前 `GraphRetriever.invoke()` 封装为
可访问的 HTTP / MCP 工具能力。

## 环境要求

- Python 3.12
- 推荐 16 GB 内存；小样本验证可降低数据量
- embedding 默认使用 CUDA，无 GPU 时在 `.env` 中设置 `EMBEDDING_DEVICE=cpu`
- BGE-M3 embedding 模型约 2.2 GB，reranker 模型约 1.1 GB；请预留模型缓存和 ChromaDB 索引空间

## 测试

建议在 `langgraph_env` 或安装完整依赖的 Python 3.12 环境中运行测试。无 GPU
时请设置 `EMBEDDING_DEVICE=cpu`。

### 综合自动化测试

默认综合测试会执行：

- 索引构建 / 追加的分批写入与 BM25 同步单元测试
- RAG 排序指标公式测试
- 索引一致性汇总逻辑测试
- 当前真实落盘索引的一致性检查（Chroma、BM25、parent_store）

```bash
python tests/run_all_tests.py
```

如需同时跑真实向量检索评估，开启 `--with-recall`：

```bash
EMBEDDING_DEVICE=cpu python tests/run_all_tests.py --with-recall --sample-size 10 --ks 1,5,10,20
```

### 单项测试

```bash
# 索引写入完整性：分批写 Chroma、BM25 同步、parent_store 保存、metadata 规范化
python tests/test_indexing_integrity.py

# 真实落盘索引一致性：Chroma / BM25 / parent_store 数量和 chunk_hash 是否一致
python tests/test_index_consistency.py

# 指标公式：rank、MRR@K、nDCG@K
python tests/test_rag_evaluation_metrics.py

# 真实向量检索效果：Recall@K、MRR@K、nDCG@K
EMBEDDING_DEVICE=cpu python tests/test_vector_recall.py --sample-size 50 --ks 1,5,10,20
```

指标含义：

- `Recall@K`：标准文档是否出现在 top K 检索结果中。
- `MRR@K`：第一个标准文档排得多靠前，公式是 `1 / rank`。
- `nDCG@K`：标准文档排序质量，使用 `1 / log2(rank + 1)` 做排名折损。

## 项目状态

- [x] Proposal
- [x] MVP
- [x] Final（已补充最终报告、License、测试评估与索引一致性结果）
