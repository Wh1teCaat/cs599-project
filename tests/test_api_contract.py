"""HTTP API contract tests.

These tests cover the API payload/response layer without loading embedding
models, Chroma, or the LLM-backed retriever.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from api import ApiRequestError, build_query_response, parse_query_payload  # noqa: E402


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def invoke(self, query, history=None, thread_id="default"):
        self.calls.append({"query": query, "history": history, "thread_id": thread_id})
        return [
            {"content": "parent context 1", "metadata": {"dataset": "demo", "rank": 1}},
            {"content": "parent context 2", "metadata": {"dataset": "demo", "rank": 2}},
            {"content": "parent context 3", "metadata": {"dataset": "demo", "rank": 3}},
        ]


class ApiContractTests(unittest.TestCase):
    def test_parse_query_payload_rejects_blank_query(self):
        with self.assertRaises(ApiRequestError) as ctx:
            parse_query_payload({"query": "   "})

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("query", str(ctx.exception))

    def test_parse_query_payload_normalizes_optional_fields(self):
        payload = parse_query_payload(
            {
                "query": "  什么是 RAG?  ",
                "top_k": "2",
                "history": [{"role": "user", "content": "上一轮"}],
                "thread_id": "demo-thread",
            }
        )

        self.assertEqual(payload.query, "什么是 RAG?")
        self.assertEqual(payload.top_k, 2)
        self.assertEqual(payload.history, [{"role": "user", "content": "上一轮"}])
        self.assertEqual(payload.thread_id, "demo-thread")

    def test_build_query_response_uses_retriever_and_limits_documents(self):
        retriever = FakeRetriever()

        response = build_query_response(
            {
                "query": "系统架构",
                "top_k": 2,
                "history": [{"role": "user", "content": "上下文"}],
                "thread_id": "api-test",
            },
            retriever,
        )

        self.assertEqual(
            retriever.calls,
            [
                {
                    "query": "系统架构",
                    "history": [{"role": "user", "content": "上下文"}],
                    "thread_id": "api-test",
                }
            ],
        )
        self.assertEqual(response["query"], "系统架构")
        self.assertEqual(response["top_k"], 2)
        self.assertEqual(response["count"], 2)
        self.assertEqual(len(response["documents"]), 2)
        self.assertEqual(response["documents"][0]["metadata"]["rank"], 1)


if __name__ == "__main__":
    unittest.main()
