# Agentic Graph Retriever 方案

## 1. 目标

- 将向量化和 BM25 文件生成抽离为独立模块，与检索逻辑解耦
- 基于 LangGraph + Tool Calling 构建真正的 Agent 混合检索
- LLM 自主决定每轮调用哪些工具、是否继续检索
- 复用现有切块策略、向量模型、ChromaDB、BM25 语料、parent_store

---

## 2. 架构拆分

### 2.1 索引层（独立，一次性离线执行）

```
main.py build → IndexBuilder
                  ├── MultiLoader            # 加载 LongBench 数据集
                  ├── HybridTextSplitter    # 父块 + 子块切分
                  ├── Chroma.from_documents # 向量库持久化
                  ├── bm25_corpus.json      # BM25 语料（子块）
                  └── parent_store.json     # 父块回填映射
```

**索引逻辑全部放在 `indexer.py`**，`retriever.py` 不再包含构建逻辑。

### 2.2 检索层（Agent + Tool Calling）

```
main.py query → GraphRetriever
                  ├── rewrite_query    (LLM: 口语化 → 正式查询)
                  │
                  ├── agent            (ReAct 循环, LLM 自主决策)
                  │     ├── tool: vector_search    # Chroma top20 → rerank top4
                  │     ├── tool: bm25_search      # BM25Okapi top4
                  │     └── finish                 # 提交结果
                  │
                  ├── rrf_fusion       (仅在 LLM 同时调用两路时触发)
                  │
                  ├── evaluate         (LLM: 判断是否需要继续检索)
                  │
                  └── parent_doc       (子块 → 父块回填)
```

---

## 3. 图结构

```
                                ┌──────────────────────────────────┐
                                │  evaluate: 不够 → 回到 agent      │
                                │  (最多 3 轮)                      │
                                └────────────┬─────────────────────┘
                                             │
START → rewrite_query → agent ───────────────╯
         (查询改写)        │
                           │ LLM 自主决定每轮调用:
                           │
              ┌────────────┼────────────┐
              ↓            ↓            ↓
         vector_search  bm25_search   finish
         (top20→top4)   (top4)        (不再搜了)
              │            │            │
              └─────┬──────┘            │
                    ↓                   │
              ┌──────────┐              │
              │ 两条路?   │              │
              │  Y: rrf  │              │
              │  N: 直通  │              │
              └────┬─────┘              │
                   ↓                    │
              accumulate               │
              (去重累积)               │
                   │                    │
                   ↓                    ↓
              evaluate ←───────────────┘
                   │
        ┌──────────┴──────────┐
        ↓                     ↓
     "够了"                "不够"
        ↓              if iteration < 3:
   parent_doc         改写提示 → 回到 agent
        ↓
     final_docs
        ↓
       END
```

---

## 4. 结点 / 工具定义

### 4.1 `rewrite_query` — 查询改写

| 项目 | 说明 |
|------|------|
| **输入** | 对话历史 + 用户原始提问 |
| **输出** | 正式化、信息完整的检索查询词 |
| **LLM** | gpt-4o-mini |
| **Prompt 要点** | 将口语化/省略的提问展开为完整的搜索 query，提取关键实体 |

### 4.2 Agent 循环 — 核心（含路由约束）

| 项目 | 说明 |
|------|------|
| **LLM** | gpt-4o-mini (with tool calling) |
| **路由机制** | Agent 内部通过 System Prompt 约束路由逻辑，不必新增 router 结点 |

#### 路由规则（System Prompt 内嵌）

LLM 在每轮选择 tool 前，**必须先判断查询类型**，按规则分派：

| 查询类型 | 判定条件 | 检索策略 |
|---------|---------|---------|
| **精准查询** | 包含专有名词、技术型号、API 名、版本号、人名、地名、缩写 | **仅调用 `bm25_search`** |
| **模糊查询** | 概念解释、日常意图、语义宽泛、同义表达丰富 | **仅调用 `vector_search`** |
| **复杂查询** | 系统设计、多维度对比、需语义+关键词互补 | **同时调用两种** → 触发 RRF 融合 |

#### System Prompt 要点

```
你是查询路由器 + 检索助手。选择 tool 前先分析查询类型：

1. 精准查询（专有名词/型号/API/版本/缩写）
   → 只调 bm25_search
   例: "BERT-base 参数"、"React 18 Suspense"、"GIL free-threading"

2. 模糊查询（概念解释/日常意图/同义表达多）
   → 只调 vector_search
   例: "什么是迁移学习"、"怎么优化 Python 性能"

3. 复杂查询（系统设计/多维对比/需互补信息）
   → 同时调用 vector_search + bm25_search（并行）
   例: "分布式缓存系统架构"、"对比微服务和单体架构"

如果检索结果充分 → 调用 finish(summary)
如果不够 → 在下一轮中调整查询词继续检索（最多 3 轮）
```

#### 工具并行说明

当 LLM 判定为复杂查询时，在一条消息中同时返回两个 `tool_call`：
```json
{
  "tool_calls": [
    {"function": {"name": "vector_search", "arguments": "{\"query\": \"分布式缓存架构设计\"}"}},
    {"function": {"name": "bm25_search", "arguments": "{\"query\": \"分布式缓存 Redis Memcached\"}"}}
  ]
}
```
LangGraph ToolNode 原生支持并行执行这两个工具。

### 4.3 `tool: vector_search`

| 项目 | 说明 |
|------|------|
| **输入** | 查询词（LLM 可自行改写） |
| **输出** | Chroma top20 粗排 + CrossEncoder 精排 top4 子块 |
| **复用** | ChromaDB + CacheEmbedding(bge-m3) + ReRanker(bge-reranker-v2-m3) |

### 4.4 `tool: bm25_search`

| 项目 | 说明 |
|------|------|
| **输入** | 查询词 |
| **输出** | BM25Okapi top4 子块 |
| **复用** | `bm25_corpus.json` + jieba 分词 |

### 4.5 `tool: finish`

| 项目 | 说明 |
|------|------|
| **参数** | `summary: str`（LLM 对检索结果的小结） |
| **作用** | 告诉系统"我不需要再搜了"，结束 agent 循环 |

### 4.6 RRF 分支（agent 循环内部）

```
if 本轮 LLM 调用了 vector_search AND bm25_search:
    → RRF 融合两路结果
else:
    → 单路结果直通
→ 去重累积到 accumulated_docs
```

> 注：LangGraph `ToolNode` 原生支持**并行 tool 调用**——LLM 一条消息里同时包含 `tool_call: vector_search` 和 `tool_call: bm25_search`，两个 tool 同时执行。

### 4.7 `evaluate` — 检索评估

| 项目 | 说明 |
|------|------|
| **输入** | 原始 query + 累积文档集 + 当前轮次 |
| **输出** | `{sufficient: bool, missing_hint: str}` |
| **LLM** | gpt-4o-mini (structured output) |
| **逻辑** | 判断累积文档是否足以完整回答；不够则给出下次检索的提示词 |

### 4.8 `parent_doc` — 父块回填

| 项目 | 说明 |
|------|------|
| **输入** | 累积的有用子块 |
| **输出** | 父块原文（1200 tokens，上下文更完整） |
| **复用** | `parent_store.json` |

---

## 5. State 定义

```python
class RetrievalState(TypedDict):
    # 输入
    query: str                     # 用户原始提问
    history: list[dict]            # 对话历史

    # 查询改写
    rewritten_query: str           # rewrite_query 输出

    # Agent 消息
    messages: list                 # LangGraph agent 消息流（tool calls 记录在这里）

    # 本轮结果
    vector_result: list[Document]  # vector_search 本轮返回
    bm25_result: list[Document]    # bm25_search 本轮返回
    need_rrf: bool                 # 本轮是否同时调了两路

    # 累积
    accumulated_docs: list[Document]  # 跨轮累积（hash 去重）

    # 评估
    sufficient: bool               # 是否足够
    missing_hint: str              # 缺什么（供 agent 下一轮参考）

    # 控制
    iteration: int                 # 当前轮次
    max_iterations: int            # 最大轮次 (default 3)

    # 输出
    final_docs: list[Document]     # 最终父块文档
```

---

## 6. Agent 循环逻辑

```
1. rewrite_query(query, history) → rewritten_query

2. while iteration < max_iterations:
     a. agent 接收: rewritten_query + 累积文档 + missing_hint
     b. LLM 自主决策，返回 tool calls（一次可以多个）
     c. ToolNode 并行执行所有 tool calls
     d. 如果调了 vector_search 且调了 bm25_search → RRF 融合
        如果只调了其中一路 → 直通
        如果只调了 finish → 跳出循环
     e. 结果去重，追加到 accumulated_docs
     f. evaluate(query, accumulated_docs) → {sufficient, missing_hint}
     g. if sufficient → break
        else → iteration += 1

3. parent_doc(accumulated_docs) → final_docs

4. return final_docs
```

### Agent 内部示例

```
Round 1:
  LLM → tool_call: vector_search("transformer attention 机制原理")
       + tool_call: bm25_search("attention 机制")
  → 两路并行执行 → RRF 融合 → 累积 6 篇
  → evaluate: 够了

Round 1:
  LLM → tool_call: bm25_search("Python GIL 全局解释器锁")
  → 单路执行 → 直通 → 累积 4 篇
  → evaluate: 不够，缺多线程优化相关内容

Round 2:
  LLM → tool_call: vector_search("Python 多线程优化 GIL 替代方案")
       + tool_call: bm25_search("GIL free threading")
  → 两路并行执行 → RRF 融合 → 新文档去重后追加
  → evaluate: 够了
```

---

## 7. 文件结构（改动后）

```
rag_service/
├── main.py                # CLI 入口（不变）
├── config.yaml            # 配置（不变）
├── .env                   # 环境变量（不变）
│
├── indexer.py             # [新增] 索引构建（从 retriever.py 抽离）
│
├── cachembedding.py       # CacheEmbedding（不变）
├── hybridtextsplitter.py  # HybridTextSplitter（不变）
├── multiloader.py         # MultiLoader（不变）
│
├── retriever.py           # [精简] 仅保留 BM25Retriever / ReRanker
├── graph_retriever.py     # [新增] LangGraph Agent 检索图
│
├── vectorize.py           # [调整] 改为调用 indexer.py
│
├── chroma_db/             # 向量库（复用）
│   ├── bm25_corpus.json
│   ├── parent_store.json
│   └── chroma.sqlite3
│
└── docs/
    └── agentic-graph-retriever.md  # 本文档
```

---

## 8. Agentic 特性总结

| 特性 | 说明 |
|------|------|
| **路由约束** | Agent 按 System Prompt 规则判定查询类型：精准→BM25、模糊→Vector、复杂→两路并行 |
| **Tool Calling** | LLM 自主选择 vector_search / bm25_search / finish |
| **并行调用** | 复杂查询时一条消息同时两个 tool call，ToolNode 并行执行 |
| **条件 RRF** | 两路都调才融合，单路直通，精准/模糊查询不经过 RRF |
| **自主停止** | LLM 通过 finish tool 主动结束检索 |
| **反思循环** | evaluate 给出 missing_hint，LLM 在下轮据此调整搜索策略 |
| **最多 3 轮** | 硬上限防止无限循环 |

---

## 9. 不改变的部分

| 组件 | 说明 |
|------|------|
| 切块策略 | 父块 1200 / 子块 500 / overlap 50 |
| 向量模型 | bge-m3 (HuggingFaceEmbeddings) |
| BM25 模型 | jieba 分词 + BM25Okapi |
| ReRanker | bge-reranker-v2-m3 CrossEncoder |
| 检索粒度 | 两路都检索子块 |
| 父块回填 | parent_store.json 映射 |
| 持久化 | chroma.sqlite3 / bm25_corpus.json 文件格式不变 |
