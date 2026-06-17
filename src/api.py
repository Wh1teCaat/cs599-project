"""展示级 HTTP API — 标准库 JSON 服务。

提供:
  GET  /health
  POST /query

默认懒加载 GraphRetriever，避免服务启动时立即加载 embedding / reranker。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass(frozen=True)
class QueryPayload:
    query: str
    top_k: int
    history: list[dict[str, Any]]
    thread_id: str


class ApiRequestError(ValueError):
    """Client-side request error that can be returned as JSON."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


_retriever = None


def parse_query_payload(payload: dict[str, Any]) -> QueryPayload:
    """Validate and normalize a /query JSON payload."""
    query = str(payload.get("query", "")).strip()
    if not query:
        raise ApiRequestError("`query` must be a non-empty string")

    try:
        top_k = int(payload.get("top_k", 5))
    except (TypeError, ValueError) as exc:
        raise ApiRequestError("`top_k` must be an integer") from exc
    if top_k < 1 or top_k > 20:
        raise ApiRequestError("`top_k` must be between 1 and 20")

    history = payload.get("history", [])
    if history is None:
        history = []
    if not isinstance(history, list):
        raise ApiRequestError("`history` must be a list")

    thread_id = str(payload.get("thread_id", "default")).strip() or "default"
    return QueryPayload(query=query, top_k=top_k, history=history, thread_id=thread_id)


def _get_retriever():
    global _retriever
    if _retriever is None:
        from cli import _load_retriever

        _retriever = _load_retriever()
    return _retriever


def build_query_response(payload: dict[str, Any], retriever=None) -> dict[str, Any]:
    """Run retrieval and return the public API response shape."""
    parsed = parse_query_payload(payload)
    active_retriever = retriever or _get_retriever()
    docs = active_retriever.invoke(
        parsed.query,
        history=parsed.history,
        thread_id=parsed.thread_id,
    )
    docs = docs[: parsed.top_k]
    return {
        "query": parsed.query,
        "top_k": parsed.top_k,
        "count": len(docs),
        "documents": docs,
    }


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class RagApiHandler(BaseHTTPRequestHandler):
    server_version = "RagServiceHTTP/1.0"

    def log_message(self, format: str, *args):  # noqa: A002
        """Keep demo output compact; request results are returned as JSON."""

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = _json_bytes(payload)
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "service": "agentic-rag",
                    "retriever_loaded": _retriever is not None,
                },
            )
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/query":
            self._send_json(404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length)
            payload = json.loads(raw_body.decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                raise ApiRequestError("request body must be a JSON object")
            response = build_query_response(payload)
            self._send_json(200, response)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON body"})
        except ApiRequestError as exc:
            self._send_json(exc.status_code, {"error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"error": f"internal server error: {exc}"})


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), RagApiHandler)
    print(f"RAG API listening on http://{host}:{port}")
    print("Endpoints: GET /health, POST /query")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping RAG API server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
