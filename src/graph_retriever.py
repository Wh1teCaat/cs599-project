"""LangGraph Agentic 检索图。

图结构:
    START → rewrite_query → agent → execute_tools → decide_merge → accum → evaluate
                                                                              │
                                         "不够" → agent (≤3 轮)              "够了" → parent_doc → END

LLM 按查询类型路由:
  - 精准查询 (专有名词/型号/API) → 只调 bm25_search
  - 模糊查询 (概念解释/日常意图) → 只调 vector_search
  - 复杂查询 (系统设计/多维对比) → 同时调两路 → RRF 融合
"""

import hashlib
import json
import operator
import os
from typing import Annotated, Literal, TypedDict

import dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

dotenv.load_dotenv()

# ═══════════════════════════════════════════════════════════════
# State
# ═══════════════════════════════════════════════════════════════


class RetrievalState(TypedDict):
    query: str
    history: list[dict]
    formal_query: str
    messages: Annotated[list, operator.add]
    vector_result: list[dict]
    bm25_result: list[dict]
    called_tools: list[str]
    new_docs: list[dict]
    accumulated_docs: list[dict]
    sufficient: bool
    missing_hint: str
    iteration: int
    max_iterations: int
    parent_store: dict[str, str]
    final_docs: list[dict]


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _doc_to_dict(doc: Document) -> dict:
    return {"content": doc.page_content, "metadata": dict(doc.metadata or {})}

def _dict_to_doc(d: dict) -> Document:
    return Document(page_content=d["content"], metadata=d.get("metadata", {}))

def _doc_hash(doc: dict) -> str:
    return doc.get("metadata", {}).get("hash") or hashlib.md5(
        doc["content"].encode("utf-8")
    ).hexdigest()


# ═══════════════════════════════════════════════════════════════
# GraphRetriever
# ═══════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """你是查询路由器 + 检索助手。根据用户提问类型，选择最合适的检索工具。

路由规则（必须遵守）:

1. 精准查询 → 只调 bm25_search
   特征: 包含专有名词、技术型号、API名、版本号、人名、地名、缩写
   例: "BERT-base 参数"、"React 18 Suspense"、"GIL free-threading"

2. 模糊查询 → 只调 vector_search
   特征: 概念解释、日常意图、语义宽泛、同义表达丰富
   例: "什么是迁移学习"、"怎么优化代码性能"、"如何理解注意力机制"

3. 复杂查询 → 同时调用 vector_search + bm25_search
   特征: 系统设计、多维对比、需语义+关键词互补
   例: "分布式缓存系统架构"、"对比微服务和单体架构优缺点"

检索到足够信息后停止调用工具，系统会自动结束本轮检索。"""


class GraphRetriever:
    """基于 LangGraph 的 Agentic 混合检索器。

    图结构 (8 结点):
        START → rewrite_query → agent → execute_tools → decide_merge
        → accum → evaluate → [不够→agent | 够了→parent_doc → END]

    LLM 路由规则 (System Prompt 约束):
        - 精准查询 → 只调 bm25_search
        - 模糊查询 → 只调 vector_search
        - 复杂查询 → 双路并行 → RRF 融合
    """

    def __init__(
        self,
        tools: list,
        vector_retriever=None,
        bm25_retriever=None,
        reranker=None,
        parent_store: dict[str, str] | None = None,
        rrf_k: int = 60,
        vector_top_k: int = 4,
        bm25_top_k: int = 4,
        max_iterations: int = 3,
    ):
        """
        Args:
            tools: LangChain @tool 列表 [vector_search, bm25_search]。
            vector_retriever: LangChain retriever (ChromaDB, k=20)。
            bm25_retriever: BM25Retriever 实例。
            reranker: ReRanker 实例 (CrossEncoder 精排)。
            parent_store: {child_hash: parent_content} 映射。
            rrf_k: RRF 平滑因子 (默认 60)。
            vector_top_k: 向量通路精排输出数 (默认 4)。
            bm25_top_k: BM25 通路输出数 (默认 4)。
            max_iterations: 最大 Agent 循环轮次 (默认 3)。
        """
        self.tools = tools
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.parent_store = parent_store or {}
        self.rrf_k = rrf_k
        self.max_iterations = max_iterations

        self._llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            temperature=0,
        )
        self._llm_with_tools = self._llm.bind_tools(tools)
        self._graph = self._build()

    # ── 图编译 ──────────────────────────────────────────

    def _build(self):
        builder = StateGraph(RetrievalState)

        builder.add_node("rewrite_query", self._rewrite_query_node)
        builder.add_node("agent", self._agent_node)
        builder.add_node("execute_tools", self._execute_tools_node)
        builder.add_node("decide_merge", self._decide_merge_node)
        builder.add_node("accum", self._accum_node)
        builder.add_node("evaluate", self._evaluate_node)
        builder.add_node("parent_doc", self._parent_doc_node)

        builder.add_edge(START, "rewrite_query")
        builder.add_edge("rewrite_query", "agent")
        builder.add_edge("agent", "execute_tools")
        builder.add_edge("execute_tools", "decide_merge")
        builder.add_edge("decide_merge", "accum")
        builder.add_edge("accum", "evaluate")
        builder.add_conditional_edges(
            "evaluate", self._route_after_evaluate,
            {"agent": "agent", "parent_doc": "parent_doc"},
        )
        builder.add_edge("parent_doc", END)

        pg_url = os.getenv("POSTGRES_URL", "")
        if pg_url:
            import psycopg
            from langgraph.checkpoint.postgres import PostgresSaver

            self._checkpoint_conn = psycopg.connect(pg_url)
            self._checkpointer = PostgresSaver(self._checkpoint_conn)
            self._checkpointer.setup()
        else:
            from langgraph.checkpoint.memory import MemorySaver
            self._checkpointer = MemorySaver()

        return builder.compile(checkpointer=self._checkpointer)

    # ── 结点 ────────────────────────────────────────────

    def _rewrite_query_node(self, state: RetrievalState) -> dict:
        """结点1: LLM 查询改写 → 提取实体/关键词 → 初始化 messages。

        Returns:
            formal_query, messages(含 SystemPrompt), iteration=0 等初始状态。
        """
        query = state["query"]
        history = state.get("history", [])

        history_text = ""
        if history:
            lines = []
            for h in history[-5:]:
                role = "用户" if h.get("role") == "user" else "助手"
                lines.append(f"{role}: {h.get('content', '')}")
            history_text = "\n".join(lines)

        prompt = f"""将用户的提问改写为正式、完整的检索查询词。

对话历史:
{history_text or "（无历史）"}

用户提问: {query}

请输出 JSON:
{{"formal_query": "改写后的检索查询词", "entities": ["实体1", "实体2"], "keywords": ["关键词1", "关键词2"]}}"""

        response = self._llm.invoke(prompt)
        try:
            result = json.loads(response.content)
        except json.JSONDecodeError:
            result = {"formal_query": query, "entities": [], "keywords": []}

        formal_query = result.get("formal_query", query)
        entities = result.get("entities", [])
        keywords = result.get("keywords", [])

        user_msg = f"原始问题: {query}\n改写查询: {formal_query}"
        if entities:
            user_msg += f"\n提取实体: {', '.join(entities)}"
        if keywords:
            user_msg += f"\n关键词: {', '.join(keywords)}"

        return {
            "formal_query": formal_query,
            "messages": [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_msg)],
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "accumulated_docs": [],
            "sufficient": False,
            "missing_hint": "",
        }

    def _agent_node(self, state: RetrievalState) -> dict:
        """结点2: LLM Agent — 基于累积 messages 选择 tools。

        首轮: 直接使用 rewrite_query 初始化的 messages 调用 LLM。
        重试轮: 追加 missing_hint 提示后调用 LLM。

        Returns:
            messages 增量: 可能包含 HumanMessage(提示) + AIMessage(tool_calls)。
        """
        iteration = state.get("iteration", 0)
        missing_hint = state.get("missing_hint", "")
        accumulated = state.get("accumulated_docs", [])

        if iteration > 0 and missing_hint:
            retry_msg = HumanMessage(content=(
                f"第 {iteration + 1} 轮检索。上一轮缺失: {missing_hint}\n"
                f"已累积 {len(accumulated)} 篇。请调整策略继续。"
            ))
            response = self._llm_with_tools.invoke(list(state["messages"]) + [retry_msg])
            return {"messages": [retry_msg, response]}

        if accumulated:
            prompt_msg = HumanMessage(content=f"已检索 {len(accumulated)} 篇。可继续或停止。")
            response = self._llm_with_tools.invoke(list(state["messages"]) + [prompt_msg])
            return {"messages": [prompt_msg, response]}

        response = self._llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def _execute_tools_node(self, state: RetrievalState) -> dict:
        """结点3: 执行 LLM 选择的 tools。

        每条 tool_call:
          1) 调 LD tool 函数 → 格式化字符串 → ToolMessage (LLM 可读)
          2) 调底层 retriever → 结构化 Document 列表 → state.vector_result/bm25_result

        Returns:
            messages(ToolMessage 增量), vector_result, bm25_result, called_tools。
        """
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", []) or []

        tool_map = {t.name: t for t in self.tools}
        results = {"vector_result": [], "bm25_result": [], "called_tools": []}
        tool_messages = []

        for tc in tool_calls:
            name = tc.get("name", tc.get("function", {}).get("name", ""))
            args = tc.get("args", tc.get("function", {}).get("arguments", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tc_id = tc.get("id", "")
            query = args.get("query", state.get("formal_query", state["query"]))

            # 调工具 → LLM 可读的字符串
            func = tool_map.get(name)
            if func:
                content = func.invoke({"query": query})
            else:
                content = ""

            # 调检索器 → 结构化文档写入 state
            if name == "vector_search" and self.vector_retriever:
                docs = self.vector_retriever.invoke(query)
                docs = self.reranker.rerank(query, docs, top_k=self.vector_top_k)
                results["vector_result"] = [_doc_to_dict(d) for d in docs]
            elif name == "bm25_search" and self.bm25_retriever:
                docs = self.bm25_retriever.invoke(query, k=self.bm25_top_k)
                results["bm25_result"] = [_doc_to_dict(d) for d in docs]

            results["called_tools"].append(name)
            tool_messages.append(ToolMessage(content=content, tool_call_id=tc_id))

        results["messages"] = tool_messages
        return results

    def _decide_merge_node(self, state: RetrievalState) -> dict:
        """结点4: 双路 → RRF 融合; 单路 → 直通。

        根据 called_tools 判断: 同时 vector_search + bm25_search → RRF; 其他 → 直通。
        """
        called = state.get("called_tools", [])
        vector_result = [_dict_to_doc(d) for d in state.get("vector_result", [])]
        bm25_result = [_dict_to_doc(d) for d in state.get("bm25_result", [])]

        if "vector_search" in called and "bm25_search" in called:
            docs = self._rrf_fusion(vector_result, bm25_result)
        elif "vector_search" in called:
            docs = vector_result
        elif "bm25_search" in called:
            docs = bm25_result
        else:
            return {"new_docs": []}

        return {"new_docs": [_doc_to_dict(d) for d in docs]}

    def _rrf_fusion(self, vector_docs: list[Document], bm25_docs: list[Document]) -> list[Document]:
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}
        for rank, doc in enumerate(vector_docs):
            doc_id = doc.metadata.get("hash") or doc.page_content
            doc_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (self.rrf_k + rank + 1)
        for rank, doc in enumerate(bm25_docs):
            doc_id = doc.metadata.get("hash") or doc.page_content
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (self.rrf_k + rank + 1)
        sorted_ids = sorted(scores, key=scores.get, reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_ids]

    def _accum_node(self, state: RetrievalState) -> dict:
        """结点5: 按文档 hash 去重，追加到 accumulated_docs。"""
        accumulated = list(state.get("accumulated_docs", []))
        existing_hashes = {_doc_hash(d) for d in accumulated}
        for doc in state.get("new_docs", []):
            h = _doc_hash(doc)
            if h not in existing_hashes:
                accumulated.append(doc)
                existing_hashes.add(h)
        return {"accumulated_docs": accumulated}

    def _evaluate_node(self, state: RetrievalState) -> dict:
        """结点6: LLM 评估累积文档是否足以回答原始问题。

        Returns:
            {sufficient: bool, missing_hint: str, iteration: int+1}。
        """
        query = state["query"]
        accumulated = state.get("accumulated_docs", [])
        iteration = state.get("iteration", 0)

        if not accumulated:
            return {"sufficient": False, "missing_hint": "未检索到任何文档",
                    "iteration": iteration + 1}

        docs_text = ""
        for i, doc in enumerate(accumulated, 1):
            docs_text += (
                f"\n文档{i}: [{doc.get('metadata', {}).get('dataset', 'unknown')}] "
                f"{doc['content'][:200]}\n"
            )

        prompt = f"""评估检索结果是否足以完整回答用户问题。

用户问题: {query}

已检索文档 ({len(accumulated)} 篇):
{docs_text}

请输出 JSON:
{{"sufficient": true或false, "missing_hint": "如果不充分，具体描述缺少什么信息"}}"""

        response = self._llm.invoke(prompt)
        try:
            result = json.loads(response.content)
        except json.JSONDecodeError:
            result = {"sufficient": True, "missing_hint": ""}

        return {
            "sufficient": result.get("sufficient", True),
            "missing_hint": result.get("missing_hint", ""),
            "iteration": iteration + 1,
        }

    def _route_after_evaluate(self, state: RetrievalState) -> Literal["agent", "parent_doc"]:
        """条件路由: sufficient 或超限 → parent_doc，否则 → agent 重试。"""
        sufficient = state.get("sufficient", False)
        iteration = state.get("iteration", 0)
        if sufficient or iteration >= state.get("max_iterations", self.max_iterations):
            return "parent_doc"
        return "agent"

    def _parent_doc_node(self, state: RetrievalState) -> dict:
        """结点7: 子块 → 父块回填。按 parent_id 去重，写入 final_docs。"""
        accumulated = state.get("accumulated_docs", [])
        parent_store = state.get("parent_store", self.parent_store)

        grouped: dict[str, dict] = {}
        for doc in accumulated:
            meta = dict(doc.get("metadata", {}))
            parent_id = meta.get("parent_id") or hashlib.md5(
                doc["content"].encode("utf-8")).hexdigest()
            if parent_id in grouped:
                continue
            parent_content = parent_store.get(parent_id)
            if parent_content:
                meta["chunk_level"] = "parent"
                grouped[parent_id] = {"content": parent_content, "metadata": meta}
            else:
                grouped[parent_id] = doc

        return {"final_docs": list(grouped.values())}

    # ── 检索接口 ────────────────────────────────────────

    def invoke(self, query: str, history: list[dict] | None = None,
               thread_id: str = "default") -> list[dict]:
        """同步检索入口。

        Args:
            query: 用户提问。
            history: 对话历史 [{"role": "user/assistant", "content": "..."}]。
            thread_id: 会话 ID，相同 ID 可恢复 PostgreSQL checkpoint。

        Returns:
            list[dict]: [{"content": "父块原文", "metadata": {...}}, ...]。
        """
        config = {"configurable": {"thread_id": thread_id}}
        result = self._graph.invoke(
            {"query": query, "history": history or [], "parent_store": self.parent_store},
            config=config,
        )
        return result.get("final_docs", [])

    async def ainvoke(self, query: str, history: list[dict] | None = None,
                      thread_id: str = "default") -> list[dict]:
        """异步检索入口。参数同 invoke。"""
        config = {"configurable": {"thread_id": thread_id}}
        result = await self._graph.ainvoke(
            {"query": query, "history": history or [], "parent_store": self.parent_store},
            config=config,
        )
        return result.get("final_docs", [])
