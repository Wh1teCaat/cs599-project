# Architecture

## 概述

本项目是一个基于 LangGraph 的 Agentic RAG 检索服务。与传统 RAG（固定单次向量召回）不同，LLM 作为 Agent 自主判断查询类型，按需调度向量检索和 BM25 关键词检索，通过多轮迭代和充分性评估确保召回质量，最终以父子块回填补全上下文。

## 模块职责

| 模块 | 职责 |
|---|---|
| `main.py` | CLI 入口，解析 `build` / `query` / `serve` 子命令并分发 |
| `cli.py` | 检索器加载、GraphRetriever 实例化、命令执行逻辑 |
| `indexer.py` | 离线索引构建：文档加载 → 切分 → ChromaDB 向量化 → BM25 语料持久化 |
| `graph_retriever.py` | LangGraph 检索图定义（8 结点 + 条件路由），Agentic 检索核心 |
| `tools.py` | `@tool` 封装 `vector_search` / `bm25_search`，供 LLM Agent 调用 |
| `multiloader.py` | 多格式数据加载，支持本地 JSON/Markdown 和 HuggingFace datasets |
| `hybridtextsplitter.py` | 混合文本切分，结合结构化边界（Markdown/代码）和语义相似度断点 |
| `cachembedding.py` | BGE-M3 embedding 包装器：磁盘缓存 + SHA256 去重，减少重复推理 |
| `download_bge.py` | 离线下载 BGE embedding 和 reranker 模型 |
| `config.yaml` | 数据集路径、向量库路径、缓存路径配置 |
| `retriever/bm25.py` | BM25 关键词检索器（jieba 分词 + BM25Okapi） |
| `retriever/rerank.py` | CrossEncoder 精排器（BGE-Reranker），对粗排结果二次排序 |
| `retriever/vector.py` | ChromaDB 向量检索器加载，支持相似度阈值过滤 |

## 索引构建流程

```
数据源（本地 / HuggingFace）
  │
  └─ MultiLoader.load()
       │  支持: JSON, Markdown, HuggingFace datasets
       │  格式: dataset_name#config（如 2wikimqa_e、passage_retrieval_en）
       │
       └─ HybridTextSplitter.split()
            │  策略: 按结构化边界（标题/代码块）优先，语义断点兜底
            │  输出: 子块（chunk）+ 父块映射 → parent_store
            │
            └─ CacheEmbedding.embed_documents()
                 │  模型: BGE-M3 (HuggingFaceEmbeddings)
                 │  缓存: SHA256 → 磁盘 JSON，命中即跳过推理
                 │
                 └─ 持久化
                      ├─ ChromaDB          → data/chroma_db/chroma.sqlite3
                      ├─ BM25 语料          → data/chroma_db/bm25_corpus.json
                      └─ parent_store      → data/chroma_db/parent_store.json
```

**关键设计**：
- 所有持久化数据统一存放在 `data/chroma_db/`，embedding 缓存在 `data/embeddings_cache.json`
- 追加模式（`--mode offline`）下按 content hash 去重，避免重复入库
- BM25 语料和 ChromaDB 共用同一套文档，保证向量/关键词检索一致性

## 检索图流程

LangGraph StateGraph，8 个结点 + 条件路由，最多 3 轮 Agent 迭代：

```
                    START
                      │
              ① rewrite_query
                LLM 改写查询 + 提取实体/关键词
                      │
              ② agent
                LLM 选择工具（vector_search / bm25_search / 双路）
                      │
              ③ execute_tools
                执行工具 → 粗排(向量 top20) → ReRanker 精排 → top4
                同时调底层 retriever 写 state.vector_result/bm25_result
                      │
              ④ decide_merge
                双路 → RRF 融合 / 单路 → 直通
                      │
              ⑤ accum
                按文档 hash 去重，追加到 accumulated_docs
                      │
              ⑥ evaluate
                LLM 评估：累积文档是否足以回答原始问题？
                      │
              ┌───────┴───────┐
          sufficient       not sufficient
          or 超限          且未超限 (≤3)
              │               │
              │           → 回到 ② agent
              │           带 missing_hint 重试
              │
         ⑦ parent_doc
           子块 → 父块回填（按 parent_id 聚合去重）
              │
            END
          输出: final_docs[]
```

### LLM 路由规则

Agent 在 `_agent_node` 中根据 System Prompt 约束，按查询类型自主选择工具：

| 查询类型 | 特征 | 工具选择 |
|---|---|---|
| **精准查询** | 专有名词、技术型号、API 名、版本号、人名 | 只调 `bm25_search` |
| **模糊查询** | 概念解释、日常意图、语义宽泛 | 只调 `vector_search` |
| **复杂查询** | 系统设计、多维对比、需关键词+语义互补 | 双路并行 → RRF 融合 |

### RRF 融合

当双路同时召回时，使用 Reciprocal Rank Fusion（k=60）合并排序：

```
RRF_score(d) = Σ 1/(k + rank_i(d))
```

同名文档（按 hash）向量路和 BM25 路的分数累加，按总分降序输出。

### 充分性评估

每轮检索后，LLM 评估累积文档是否足以回答原始问题，不足时给出 `missing_hint` 指导 Agent 调整策略进入下一轮。超过 `max_iterations`（默认 3）或 LLM 判定充分后进入父块回填。

## 状态定义

```python
class RetrievalState(TypedDict):
    query: str                    # 原始用户提问
    history: list[dict]           # 对话历史
    formal_query: str             # LLM 改写后的检索查询词
    messages: list                # LangGraph messages（System + Human + AIMessage + ToolMessage）
    vector_result: list[dict]     # 向量路粗排 + ReRank 精排结果
    bm25_result: list[dict]       # BM25 路检索结果
    called_tools: list[str]       # 本轮调用的工具名列表
    new_docs: list[dict]          # 本轮去重后的新文档
    accumulated_docs: list[dict]  # 跨轮累积的全部文档
    sufficient: bool              # LLM 评估：是否足够回答
    missing_hint: str             # LLM 给出的缺失信息提示
    iteration: int                # 当前轮次
    max_iterations: int           # 最大轮次限制
    parent_store: dict            # {child_hash: parent_content}
    final_docs: list[dict]        # 父块回填后的最终文档
```

## 数据流概览

```
离线索引阶段:
  Raw Data → MultiLoader → HybridTextSplitter → CacheEmbedding
     → ChromaDB (向量) + BM25 Corpus (关键词) + parent_store (父块)

在线检索阶段:
  User Query → GraphRetriever.invoke()
     → LangGraph 图执行 (8 结点)
     → vector_retriever / bm25_retriever 粗排
     → ReRanker 精排
     → RRF 融合 (双路时)
     → LLM 充分性评估 (≤3 轮)
     → 父块回填
     → final_docs[]
```

## Checkpoint

默认使用 `MemorySaver`（内存存储，进程重启丢失）。设置 `.env` 中 `POSTGRES_URL` 后自动切换为 `PostgresSaver`，支持跨会话恢复和对话历史持久化。Postgres 相关依赖（`langgraph-checkpoint-postgres`、`psycopg`）为可选安装。

## 关键技术选型

| 组件 | 选型 | 理由 |
|---|---|---|
| 编排框架 | LangGraph | StateGraph + 条件路由，天然支持 Agent 循环和 checkpoint |
| 向量库 | ChromaDB | 轻量、本地持久化、LangChain 原生集成 |
| 关键词检索 | BM25Okapi (rank-bm25) | 经典算法，对专有名词/型号精准匹配效果好 |
| 分词 | jieba | 中文分词支持 |
| Embedding | BGE-M3 (HuggingFace) | 多语言、支持稠密+稀疏向量，1024 维 |
| Reranker | BGE-Reranker-v2-m3 (CrossEncoder) | 对粗排结果精排，显著提升召回精度 |
| LLM | DeepSeek API（兼容 OpenAI SDK） | 通过 `OPENAI_BASE_URL` 适配，ChatOpenAI 统一调用 |
| 缓存 | SHA256 → JSON 磁盘缓存 | 避免重复 embedding 推理，增量构建时直接命中 |
