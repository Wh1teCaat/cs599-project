# Agentic RAG 知识库检索服务

![Python](https://img.shields.io/badge/Python-3.10--3.12-blue)
![Framework](https://img.shields.io/badge/Framework-LangGraph-green)
![Status](https://img.shields.io/badge/Status-MVP-yellow)

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
│   └── agentic-graph-retriever.md
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

## API 调用（待完善）

`python main.py serve` 通过 `langgraph serve` 启动 LangGraph API 服务（默认 `http://127.0.0.1:2024`），但当前缺少 `src/langgraph.json` 配置文件，且 graph 对象未以模块级变量导出，`langgraph serve` 暂不可用。计划后续补充 `langgraph.json` 并将 `GraphRetriever` 封装为 LangGraph 可发现的服务入口。

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
- [ ] Final（待补充最终报告与完整实验结果）
