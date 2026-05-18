from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))


def _install_agent_graph_import_stubs() -> None:
    class DummyMessage:
        def __init__(self, content: str = "", id: str | None = None, **kwargs):
            self.content = content
            self.id = id
            for key, value in kwargs.items():
                setattr(self, key, value)

    class RemoveMessage(DummyMessage):
        type = "remove"

    class SystemMessage(DummyMessage):
        type = "system"

    class DummyStateGraph:
        def __init__(self, *args, **kwargs):
            pass

        def add_node(self, *args, **kwargs):
            pass

        def add_edge(self, *args, **kwargs):
            pass

        def add_conditional_edges(self, *args, **kwargs):
            pass

        def compile(self, *args, **kwargs):
            return self

    def decorator_passthrough(func=None, *args, **kwargs):
        if func is not None and callable(func):
            return func

        def inner(wrapped):
            return wrapped

        return inner

    modules = {
        "deepagents": {"create_deep_agent": lambda *args, **kwargs: types.SimpleNamespace()},
        "langchain_community": {},
        "langchain_community.tools": {"DuckDuckGoSearchRun": object},
        "langchain_community.tools.tavily_search": {"TavilySearchResults": object},
        "langchain_core": {},
        "langchain_core.globals": {"set_llm_cache": lambda *args, **kwargs: None},
        "langchain_core.messages": {
            "AIMessage": DummyMessage,
            "BaseMessage": DummyMessage,
            "HumanMessage": DummyMessage,
            "RemoveMessage": RemoveMessage,
            "SystemMessage": SystemMessage,
        },
        "langchain_core.runnables": {"RunnableConfig": dict},
        "langchain_core.tools": {"tool": decorator_passthrough},
        "langchain_litellm": {"ChatLiteLLM": object},
        "langchain_redis": {"RedisCache": object},
        "langgraph": {},
        "langgraph.func": {"task": decorator_passthrough},
        "langgraph.graph": {
            "END": "__end__",
            "START": "__start__",
            "MessagesState": dict,
            "StateGraph": DummyStateGraph,
        },
        "langgraph.graph.message": {"REMOVE_ALL_MESSAGES": "__remove_all__"},
        "langgraph.types": {"interrupt": lambda *args, **kwargs: None},
        "langgraph_swarm": {
            "create_handoff_tool": lambda *args, **kwargs: object(),
            "create_swarm": lambda *args, **kwargs: types.SimpleNamespace(compile=lambda *a, **k: types.SimpleNamespace()),
        },
        "langmem": {
            "create_manage_memory_tool": lambda *args, **kwargs: object(),
            "create_search_memory_tool": lambda *args, **kwargs: object(),
        },
    }
    for name, attrs in modules.items():
        module = sys.modules.setdefault(name, types.ModuleType(name))
        for attr, value in attrs.items():
            setattr(module, attr, value)


_install_agent_graph_import_stubs()
import agent_graph  # noqa: E402
import vector_memory  # noqa: E402


def test_normalize_source_keys_accepts_string_lists_and_dedupes() -> None:
    assert agent_graph._normalize_source_keys("doc-a, doc-b,doc-a", source_key="archive-1") == [
        "archive-1",
        "doc-a",
        "doc-b",
    ]


def test_query_sources_filters_pgvector_and_rag_by_source_key(monkeypatch) -> None:
    calls: dict[str, object] = {}

    async def fake_pgvector_search(**kwargs):
        calls["pgvector"] = kwargs
        return [
            {
                "source_type": "archive",
                "source_key": "archive-1",
                "title": "Archive One",
                "chunk_text": "The relevant archived detail.",
                "similarity": 0.91,
                "distance": 0.09,
                "chunk_index": 0,
                "chunk_count": 2,
                "metadata": {},
            }
        ]

    async def fake_rag_query(query, source_keys, limit):
        calls["rag"] = {"query": query, "source_keys": source_keys, "limit": limit}
        return [
            {
                "source_type": "external_document",
                "source_key": "archive-1",
                "title": "External One",
                "preview_text": "A matching document chunk.",
                "chunk_text": "A matching document chunk.",
                "score": 0.2,
                "metadata": {"file_id": "archive-1"},
            }
        ], ""

    monkeypatch.setattr(agent_graph, "_pgvector_memory_enabled", lambda: True)
    monkeypatch.setattr(agent_graph, "_pgvector_semantic_search", fake_pgvector_search)
    monkeypatch.setattr(agent_graph, "_rag_query_sources", fake_rag_query)
    monkeypatch.setattr(agent_graph, "_state_thread_id", lambda state=None: "thread-1")

    payload = json.loads(
        asyncio.run(
            agent_graph._query_sources_impl(
                query="relevant detail",
                source_keys=["archive-1"],
                source_type="all",
                limit=4,
            )
        )
    )

    assert calls["pgvector"]["source_keys"] == ["archive-1"]
    assert calls["pgvector"]["thread_id"] == "thread-1"
    assert calls["rag"]["source_keys"] == ["archive-1"]
    assert payload["source_keys"] == ["archive-1"]
    assert [item["source_type"] for item in payload["results"]] == ["archive", "external_document"]
    assert payload["results"][0]["distance"] == 0.09
    assert "filtered to the requested source_key" in payload["retrieval_policy"]


def test_query_archive_uses_archive_source_type(monkeypatch) -> None:
    calls: dict[str, object] = {}

    async def fake_pgvector_search(**kwargs):
        calls["pgvector"] = kwargs
        return []

    async def fake_rag_query(query, source_keys, limit):
        calls["rag_called"] = True
        return [], ""

    monkeypatch.setattr(agent_graph, "_pgvector_memory_enabled", lambda: True)
    monkeypatch.setattr(agent_graph, "_pgvector_semantic_search", fake_pgvector_search)
    monkeypatch.setattr(agent_graph, "_rag_query_sources", fake_rag_query)

    payload = json.loads(
        asyncio.run(
            agent_graph.query_archive(
                query="old decision",
                archive_key="archive-2",
                limit=2,
            )
        )
    )

    assert calls["pgvector"]["source_type"] == "archive"
    assert calls["pgvector"]["source_keys"] == ["archive-2"]
    assert "rag_called" not in calls
    assert payload["source_type_filter"] == "archive"


def test_query_archive_prefers_existing_rag_mirror_and_keeps_pgvector_fallback(monkeypatch) -> None:
    calls: dict[str, object] = {}

    async def fake_pgvector_search(**kwargs):
        calls["pgvector"] = kwargs
        return [
            {
                "source_type": "archive",
                "source_key": "archive-3",
                "title": "Archive Three",
                "chunk_text": "pgvector fallback chunk",
                "similarity": 0.7,
            }
        ]

    async def fake_rag_query(query, source_keys, limit):
        calls["rag"] = {"query": query, "source_keys": source_keys, "limit": limit}
        return [
            {
                "source_type": "external_document",
                "source_key": "archive:archive-3",
                "title": "Archive Three Mirror",
                "preview_text": "rag mirror chunk",
                "chunk_text": "rag mirror chunk",
                "score": 0.1,
                "metadata": {"file_id": "archive:archive-3"},
            }
        ], ""

    async def fake_rag_file_id(archive_key):
        return f"archive:{archive_key}"

    monkeypatch.setattr(agent_graph, "_pgvector_memory_enabled", lambda: True)
    monkeypatch.setattr(agent_graph, "_pgvector_semantic_search", fake_pgvector_search)
    monkeypatch.setattr(agent_graph, "_rag_query_sources", fake_rag_query)
    monkeypatch.setattr(agent_graph, "_rag_file_id_for_archive", fake_rag_file_id)

    payload = json.loads(
        asyncio.run(
            agent_graph.query_archive(
                query="old mirrored decision",
                archive_key="archive-3",
                limit=2,
            )
        )
    )

    assert calls["pgvector"]["source_keys"] == ["archive-3"]
    assert calls["rag"]["source_keys"] == ["archive:archive-3"]
    assert [item["source_key"] for item in payload["results"]] == ["archive-3", "archive:archive-3"]


def test_agentic_rag_retrieve_tool_returns_bounded_context_packet(monkeypatch) -> None:
    calls: dict[str, object] = {}

    async def fake_router_agentic_rag_retrieve(**kwargs):
        calls.update(kwargs)
        return {
            "query": kwargs["query"],
            "final_query": "qwen ollama",
            "source_keys": kwargs["source_keys"],
            "source_type_filter": kwargs["source_type"],
            "context_packet": {
                "query": "qwen ollama",
                "chunk_count": 1,
                "chunks": [{"source_key": "archive-4", "chunk_text": "bounded chunk"}],
            },
            "next_action": "generate_answer",
            "graph_trace": [{"node": "retrieve"}],
            "warnings": [],
        }

    async def fake_rag_file_id(archive_key):
        return f"archive:{archive_key}"

    monkeypatch.setattr(agent_graph, "_router_agentic_rag_retrieve", fake_router_agentic_rag_retrieve)
    monkeypatch.setattr(agent_graph, "_rag_file_id_for_archive", fake_rag_file_id)
    monkeypatch.setattr(agent_graph, "_pgvector_memory_enabled", lambda: True)
    monkeypatch.setattr(agent_graph, "_state_thread_id", lambda state=None: "thread-1")

    payload = json.loads(
        asyncio.run(
            agent_graph.agentic_rag_retrieve(
                query="Wie war das nochmal mit Qwen und Ollama?",
                source_keys=["archive-4"],
                source_type="archive",
                limit=3,
                max_context_chars=1200,
            )
        )
    )

    assert calls["thread_id"] == "thread-1"
    assert calls["source_keys"] == ["archive-4"]
    assert calls["rag_source_keys"] == ["archive:archive-4"]
    assert calls["max_context_chars"] == 1200
    assert payload["next_action"] == "generate_answer"
    assert payload["context_packet"]["chunk_count"] == 1


def test_vector_memory_distance_threshold_uses_alpha_env(monkeypatch) -> None:
    monkeypatch.setenv("ALPHARAVIS_PGVECTOR_DISTANCE_THRESHOLD", "0.42")
    assert vector_memory._distance_threshold() == 0.42


def test_vector_memory_distance_threshold_rejects_invalid_env(monkeypatch) -> None:
    monkeypatch.setenv("ALPHARAVIS_PGVECTOR_DISTANCE_THRESHOLD", "close")
    try:
        vector_memory._distance_threshold()
    except vector_memory.VectorMemoryError as exc:
        assert "ALPHARAVIS_PGVECTOR_DISTANCE_THRESHOLD" in str(exc)
    else:
        raise AssertionError("invalid distance threshold should raise VectorMemoryError")
