from __future__ import annotations

import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

import retrieval_router  # noqa: E402


def test_normalize_source_keys_accepts_strings_lists_and_source_key() -> None:
    assert retrieval_router.normalize_source_keys("doc-a, doc-b,doc-a", source_key="archive-1") == [
        "archive-1",
        "doc-a",
        "doc-b",
    ]


def test_query_sources_with_backends_combines_pgvector_and_rag(monkeypatch) -> None:
    calls: dict[str, object] = {}

    async def fake_pgvector_search(**kwargs):
        calls["pgvector"] = kwargs
        return [
            {
                "source_type": "archive",
                "source_key": "archive-1",
                "title": "Archive One",
                "chunk_text": "pgvector chunk",
                "similarity": 0.91,
                "distance": 0.09,
            }
        ]

    async def fake_rag_query(query, file_ids, limit):
        calls["rag"] = {"query": query, "file_ids": file_ids, "limit": limit}
        return [
            {
                "source_type": "external_document",
                "source_key": "doc-1",
                "title": "Doc One",
                "chunk_text": "rag chunk",
                "preview_text": "rag chunk",
                "distance": 0.12,
                "retrieval_backend": "rag_api",
            }
        ], ""

    monkeypatch.setenv("ALPHARAVIS_PGVECTOR_SEARCH_LIMIT", "5")
    payload = asyncio.run(
        retrieval_router.query_sources_with_backends(
            query="old decision",
            source_keys=["archive-1"],
            source_type="all",
            limit=4,
            thread_id="thread-1",
            pgvector_search=fake_pgvector_search,
            pgvector_available=True,
            rag_query_func=fake_rag_query,
            rag_source_keys=["doc-1"],
        )
    )

    assert calls["pgvector"]["thread_id"] == "thread-1"
    assert calls["pgvector"]["source_keys"] == ["archive-1"]
    assert calls["rag"]["file_ids"] == ["doc-1"]
    assert payload["backend_counts"] == {"alpharavis_pgvector": 1, "rag_api": 1}
    assert [item["retrieval_backend"] for item in payload["results"]] == ["alpharavis_pgvector", "rag_api"]
    assert "filtered to the requested source_key" in payload["retrieval_policy"]


def test_query_sources_with_backends_keeps_archive_rag_passive_by_default(monkeypatch) -> None:
    calls: dict[str, object] = {}

    async def fake_pgvector_search(**kwargs):
        return []

    async def fake_rag_query(query, file_ids, limit):
        calls["rag"] = True
        return [], ""

    monkeypatch.setenv("ALPHARAVIS_ENABLE_RAG_FEDERATED_SEARCH", "true")
    payload = asyncio.run(
        retrieval_router.query_sources_with_backends(
            query="archive question",
            source_keys=["archive-1"],
            source_type="archive",
            limit=3,
            pgvector_search=fake_pgvector_search,
            pgvector_available=True,
            rag_query_func=fake_rag_query,
        )
    )

    assert calls == {}
    assert payload["backend_counts"] == {"alpharavis_pgvector": 0, "rag_api": 0}


def test_ingest_source_routes_external_document_to_rag_by_default(monkeypatch) -> None:
    calls: dict[str, object] = {}

    async def fake_pgvector_index(**kwargs):
        calls["pgvector"] = kwargs
        return "indexed"

    async def fake_rag_mirror(**kwargs):
        calls["rag"] = kwargs
        return {"status": True, "file_id": kwargs["file_id"]}

    monkeypatch.delenv("ALPHARAVIS_INGEST_INDEX_DOCUMENTS_IN_PGVECTOR", raising=False)
    result = asyncio.run(
        retrieval_router.ingest_source(
            source_type="external_document",
            source_key="doc-1",
            title="Doc One",
            content="document body",
            thread_id="thread-1",
            pgvector_index=fake_pgvector_index,
            rag_mirror_func=fake_rag_mirror,
        )
    )

    assert "pgvector" not in calls
    assert calls["rag"]["file_id"] == "doc-1"
    assert calls["rag"]["filename"] == "Doc One.txt"
    assert result["index_status"] == "indexed"
    assert result["indexed_backends"] == ["rag_api"]
    assert result["metadata"]["rag_file_id"] == "doc-1"


def test_ingest_source_indexes_archive_pgvector_without_rag_when_mirror_disabled(monkeypatch) -> None:
    calls: dict[str, object] = {}

    async def fake_pgvector_index(**kwargs):
        calls["pgvector"] = kwargs
        return "queued:job-1"

    async def fake_rag_mirror(**kwargs):
        calls["rag"] = kwargs
        return {"status": True}

    monkeypatch.setenv("ALPHARAVIS_ENABLE_RAG_ARCHIVE_MIRROR", "false")
    result = asyncio.run(
        retrieval_router.ingest_source(
            source_type="archive",
            source_key="archive-1",
            title="Archive One",
            content="archive body",
            thread_id="thread-1",
            pgvector_index=fake_pgvector_index,
            rag_mirror_func=fake_rag_mirror,
        )
    )

    assert calls["pgvector"]["source_type"] == "archive"
    assert calls["pgvector"]["metadata"]["rag_file_id"] == "archive:archive-1"
    assert "rag" not in calls
    assert result["rag_file_id"] == "archive:archive-1"
    assert result["indexed_backends"] == ["alpharavis_pgvector"]


def test_ingest_source_reports_partial_when_rag_fails_after_pgvector(monkeypatch) -> None:
    async def fake_pgvector_index(**kwargs):
        return "indexed"

    async def fake_rag_mirror(**kwargs):
        raise RuntimeError("embedding backend offline")

    result = asyncio.run(
        retrieval_router.ingest_source(
            source_type="large_paste",
            source_key="paste-1",
            title="Paste One",
            content="large paste body",
            preferred_backend="both",
            pgvector_index=fake_pgvector_index,
            rag_mirror_func=fake_rag_mirror,
        )
    )

    assert result["index_status"] == "partial"
    assert result["indexed_backends"] == ["alpharavis_pgvector"]
    assert result["errors"] == [{"stage": "rag_api", "error": "embedding backend offline"}]
    assert result["metadata"]["rag_index_status"] == "failed"


def test_ingest_source_validates_content() -> None:
    result = asyncio.run(
        retrieval_router.ingest_source(
            source_type="memory",
            source_key="memory-1",
            title="Memory One",
            content=" ",
        )
    )

    assert result["index_status"] == "failed"
    assert result["errors"][0]["stage"] == "validate"


def test_grade_retrieval_hits_and_context_packet() -> None:
    hits = [
        {
            "source_type": "archive",
            "source_key": "archive-1",
            "title": "Qwen3 VL notes",
            "chunk_text": "Qwen3-VL embedding on Ollama was discussed with llama.cpp mtmd.",
            "similarity": 0.92,
            "retrieval_backend": "alpharavis_pgvector",
        },
        {
            "source_type": "archive",
            "source_key": "archive-2",
            "title": "Unrelated",
            "chunk_text": "Coffee recipe notes.",
            "similarity": 0.01,
        },
    ]

    grade = retrieval_router.grade_retrieval_hits(query="Qwen3-VL Ollama embedding", hits=hits, min_relevance=0.2)
    packet = retrieval_router.build_grounded_context_packet(query="Qwen3-VL Ollama embedding", hits=grade["relevant_hits"])

    assert grade["decision"] == "generate_answer"
    assert grade["relevant_count"] == 1
    assert packet["chunk_count"] == 1
    assert packet["chunks"][0]["source_key"] == "archive-1"
    assert "read_archive_record" in packet["instructions"]


def test_rewrite_retrieval_query_removes_vague_archive_phrasing() -> None:
    rewritten = retrieval_router.rewrite_retrieval_query(
        "Wie war das nochmal mit Qwen3-VL und Ollama im Archiv?",
        source_keys=["archive-1"],
    )

    assert "qwen3-vl" in rewritten
    assert "ollama" in rewritten
    assert "nochmal" not in rewritten
    assert "archive-1" in rewritten


def test_agentic_rag_retrieve_rewrites_weak_query_and_returns_context(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_pgvector_search(**kwargs):
        calls.append(kwargs["query"])
        if len(calls) == 1:
            return [
                {
                    "source_type": "archive",
                    "source_key": "archive-1",
                    "title": "Weak",
                    "chunk_text": "unrelated text",
                    "similarity": 0.0,
                }
            ]
        return [
            {
                "source_type": "archive",
                "source_key": "archive-1",
                "title": "Qwen3 VL",
                "chunk_text": "Qwen3-VL Ollama embedding needs a bounded RAG lookup.",
                "similarity": 0.95,
            }
        ]

    monkeypatch.setenv("ALPHARAVIS_AGENTIC_RAG_MIN_RELEVANCE", "0.2")
    payload = asyncio.run(
        retrieval_router.agentic_rag_retrieve(
            query="Wie war das nochmal mit Qwen3-VL und Ollama?",
            source_keys=["archive-1"],
            source_type="archive",
            limit=3,
            pgvector_search=fake_pgvector_search,
            pgvector_available=True,
        )
    )

    assert len(calls) == 2
    assert payload["rewritten_query"]
    assert payload["next_action"] == "generate_answer"
    assert payload["context_packet"]["chunk_count"] == 1
    assert [step["node"] for step in payload["graph_trace"]] == [
        "retrieve",
        "grade_documents",
        "rewrite_question",
        "retrieve",
        "grade_documents",
        "generate_answer",
    ]
