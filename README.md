# RAG Service — Agentic 混合检索系统

## 项目简介

基于 LangGraph + Tool Calling 的 Agentic 混合检索系统。LLM 自主判定查询类型，选择向量语义检索、BM25 关键词检索或双路并行，RRF 融合后经评估循环（最多 3 轮）输出父块上下文。

## 方向

方向一：Agentic AI 原生开发

## 技术栈

| 层级 | 技术 |
|------|------|
| LLM | OpenAI API (gpt-4o-mini) |
| Embedding | BGE-m3 (HuggingFace + CUDA) |
| 精排 | BGE-reranker-v2-m3 (CrossEncoder) |
| 关键词检索 | BM25Okapi + jieba 中文分词 |
| 向量数据库 | ChromaDB |
| Agent 框架 | LangGraph (Tool Calling + Postgres checkpoint) |
| 数据源 | LongBench (HuggingFace Datasets) |

## 目录结构

```
rag_service/
├── main.py                    # CLI 入口
├── cli.py                     # build() / query() 胶水层
├── config.yaml                # 路径配置
├── .env                       # 环境变量 (不提交)
│
├── indexer.py                 # 索引构建 (离线)
├── graph_retriever.py         # LangGraph Agent 检索图
├── tools.py                   # @tool 定义 (vector_search / bm25_search)
│
├── retriever/                 # 检索组件
│   ├── __init__.py
│   ├── bm25.py                # BM25Retriever
│   ├── rerank.py              # ReRanker
│   └── vector.py              # ChromaDB 加载
│
├── cachembedding.py           # Embedding 缓存
├── hybridtextsplitter.py      # 父块 + 子块切分
├── multiloader.py             # 多格式数据加载
├── download_bge.py            # BGE 模型下载
│
├── chroma_db/                 # 向量库 + BM25 语料 + parent_store
├── cache/                     # Embedding 缓存
└── docs/                      # 设计文档
```

| 模块 | 职责 |
|------|------|
| `main.py` | CLI 入口：build / query / serve |
| `cli.py` | 组件组装：加载 DB → 注入工具 → 构建 GraphRetriever |
| `indexer.py` | 离线索引：文档加载 → 切分 → ChromaDB + BM25 语料 + parent_store |
| `graph_retriever.py` | LangGraph 图 (8 结点)：rewrite → agent → tools → merge → accum → evaluate → parent_doc |
| `tools.py` | @tool 定义：vector_search / bm25_search，供 Agent Tool Calling |
| `retriever/bm25.py` | jieba + BM25Okapi 关键词检索 |
| `retriever/rerank.py` | CrossEncoder 精排 (bge-reranker-v2-m3) |
| `retriever/vector.py` | ChromaDB 向量检索器加载 |
| `cachembedding.py` | SHA256 哈希 Embedding 缓存 |
| `hybridtextsplitter.py` | 父块(1200) + 子块(500) 双层切分 |
| `multiloader.py` | LongBench 数据集 + 本地文件加载 |

## 环境搭建

### 1. 依赖安装

```bash
pip install -e .
```

核心依赖：`langchain` `langchain-chroma` `langchain-huggingface` `langchain-openai` `langgraph` `jieba` `rank-bm25` `sentence-transformers` `datasets` `psycopg`

### 2. 环境变量 (`.env`)

```bash
# LLM
MODEL_NAME=gpt-4o-mini
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai-proxy.org/v1

# Embedding & ReRanker (本地模型)
HF_MODEL_NAME=/mnt/e/huggingface/embedding_model/bge-m3
RERANK_MODEL_NAME=/mnt/e/huggingface/embedding_model/bge-reranker-v2-m3

# PostgreSQL (Agent checkpoint 持久化)
POSTGRES_URL=postgresql://postgres:020203@localhost:5432/agent_db

# Proxy
HTTP_PROXY=http://127.0.0.1:10808
HTTPS_PROXY=http://127.0.0.1:10808

# tiktoken 离线缓存
TIKTOKEN_CACHE_DIR=/tmp/data-gym-cache
```

### 3. 下载模型

```bash
python download_bge.py
```

## 使用方式

### 构建向量库

```bash
# 限定数据集构建
python main.py build --datasets passage_retrieval_en,2wikimqa_e,hotpotqa_e

# 采样构建（快速验证）
python main.py build --datasets passage_retrieval_en --sample 8

# 全量构建
python main.py build
```

### 检索查询

```bash
python main.py query "什么是注意力机制" --topk 5
```

### 启动 LangGraph 服务

```bash
python main.py serve
```

## 架构

### Agent 检索流程

```
START → rewrite_query → agent → execute_tools → decide_merge → accum → evaluate
                                                                          │
                                                     "不够" → agent (≤3 轮)
                                                     "够了" → parent_doc → END
```

### LLM 路由规则

| 查询类型 | 判定条件 | 检索策略 |
|---------|---------|---------|
| 精准查询 | 专有名词/型号/API/版本 | 只调 bm25_search |
| 模糊查询 | 概念解释/日常意图 | 只调 vector_search |
| 复杂查询 | 系统设计/多维对比 | 双路并行 → RRF 融合 |

### Agentic 特性

| 特性 | 实现 |
|------|------|
| Tool Calling | LLM 自主选择 vector_search / bm25_search |
| 并行调用 | 复杂查询时一条消息两个 tool_call，同时执行 |
| 条件 RRF | 双路才融合，单路直通 |
| 评估循环 | evaluate 结点 LLM 判断是否充分，不够改写 query 重搜 |
| Checkpoint | PostgreSQL 持久化，支持会话恢复 |

## 项目状态

- [x] Proposal
- [x] MVP
- [ ] Final
