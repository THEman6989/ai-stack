from __future__ import annotations

import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

import rag_api_client  # noqa: E402


class _FakeResponse:
    def __init__(self, status_code: int, payload, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


def test_query_sources_uses_single_file_query_endpoint(monkeypatch) -> None:
    calls = []

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, **kwargs):
            calls.append((url, kwargs))
            return _FakeResponse(
                200,
                [[{"page_content": "matching chunk", "metadata": {"file_id": "doc-1", "filename": "Doc"}} , 0.12]],
            )

    monkeypatch.setattr(rag_api_client.httpx, "AsyncClient", FakeAsyncClient)
    config = rag_api_client.RagApiConfig(base_url="http://rag", timeout_seconds=3)

    hits = asyncio.run(rag_api_client.query_sources("question", ["doc-1"], limit=4, config=config))

    assert calls[0][0] == "http://rag/query"
    assert calls[0][1]["json"] == {"query": "question", "file_id": "doc-1", "k": 4, "entity_id": "alpharavis"}
    assert hits[0]["source_key"] == "doc-1"
    assert hits[0]["distance"] == 0.12


def test_query_sources_uses_multi_file_query_endpoint(monkeypatch) -> None:
    calls = []

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, **kwargs):
            calls.append((url, kwargs))
            return _FakeResponse(404, {"detail": "none"}, "none")

    monkeypatch.setattr(rag_api_client.httpx, "AsyncClient", FakeAsyncClient)
    config = rag_api_client.RagApiConfig(base_url="http://rag", timeout_seconds=3)

    hits = asyncio.run(rag_api_client.query_sources("question", ["doc-1", "doc-2"], limit=2, config=config))

    assert calls[0][0] == "http://rag/query_multiple"
    assert calls[0][1]["json"] == {"query": "question", "file_ids": ["doc-1", "doc-2"], "k": 2}
    assert hits == []


def test_mirror_text_posts_text_file_to_embed_endpoint(monkeypatch) -> None:
    calls = []

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, **kwargs):
            calls.append((url, kwargs))
            return _FakeResponse(200, {"status": True, "file_id": "archive:a"})

    monkeypatch.setattr(rag_api_client.httpx, "AsyncClient", FakeAsyncClient)
    config = rag_api_client.RagApiConfig(base_url="http://rag", timeout_seconds=3, entity_id="thread-a")

    payload = asyncio.run(
        rag_api_client.mirror_text(file_id="archive:a", text="large archive", filename="archive_a.txt", config=config)
    )

    assert calls[0][0] == "http://rag/embed"
    assert calls[0][1]["data"] == {"file_id": "archive:a", "entity_id": "thread-a"}
    assert calls[0][1]["files"]["file"][0] == "archive_a.txt"
    assert payload["status"] is True


def test_headers_can_build_local_jwt_from_shared_secret(monkeypatch) -> None:
    monkeypatch.delenv("ALPHARAVIS_RAG_API_BEARER_TOKEN", raising=False)
    monkeypatch.setenv("JWT_SECRET", "local-secret")
    config = rag_api_client.RagApiConfig(base_url="http://rag", timeout_seconds=3, entity_id="thread-a")

    headers = rag_api_client._headers(config)

    assert headers["Authorization"].startswith("Bearer ")
    assert rag_api_client.auth_mode(config) == "local_jwt"
