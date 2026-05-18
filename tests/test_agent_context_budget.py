from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest


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
from context_compressor import CompressionResult  # noqa: E402


def test_store_compression_archive_uses_ingest_router(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}
    writes: list[tuple[tuple[str, ...], str, dict[str, object]]] = []

    async def fake_ingest_source(**kwargs):
        calls["ingest"] = kwargs
        return {
            "index_status": "indexed",
            "rag_file_id": "archive:archive-key",
            "rag_index_status": "indexed",
            "rag_indexed_at": 123456,
            "indexed_backends": ["alpharavis_pgvector", "rag_api"],
            "errors": [],
        }

    async def fake_put(store, namespace, key, value):
        writes.append((namespace, key, value))

    monkeypatch.setattr(agent_graph, "_router_ingest_source", fake_ingest_source)
    monkeypatch.setattr(agent_graph, "_maybe_put", fake_put)
    monkeypatch.setattr(agent_graph.hashlib, "sha256", lambda *_args, **_kwargs: types.SimpleNamespace(hexdigest=lambda: "archive-key-abcdef"))

    result = CompressionResult(
        mode="PRE_RUN",
        thread_id="thread-1",
        thread_key="thread:key",
        token_limit=1000,
        token_estimate_before=2000,
        token_estimate_after=800,
        head=[],
        middle=[{"role": "assistant", "content": "old detail"}],
        tail=[],
        summary="summary",
        summary_message_content="summary",
        archive_content="raw archived content",
        archive_metadata={"middle_indexes": [1]},
        pruned_middle_text="old detail",
    )

    archive_key, record = asyncio.run(
        agent_graph._store_compression_archive(
            store=object(),
            result=result,
            mode="PRE_RUN",
            thread_id="thread-1",
            thread_key="thread:key",
        )
    )

    assert archive_key == "archive-key-abcdef"[:24]
    assert calls["ingest"]["source_type"] == "archive"
    assert calls["ingest"]["source_key"] == archive_key
    assert calls["ingest"]["content"] == "raw archived content"
    assert calls["ingest"]["pgvector_index"] == agent_graph._maybe_index_vector_memory
    assert record["rag_file_id"] == "archive:archive-key"
    assert record["rag_index_status"] == "indexed"
    assert record["indexed_backends"] == ["alpharavis_pgvector", "rag_api"]
    assert record["metadata"]["ingest_status"] == "indexed"
    assert len(writes) == 2


def test_context_budget_snapshot_uses_provider_reported_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPHARAVIS_ENABLE_PERCENT_CONTEXT_LIMITS", "true")
    monkeypatch.setenv("ALPHARAVIS_ACTIVE_CONTEXT_TRIGGER_RATIO", "0.50")
    monkeypatch.setenv("ALPHARAVIS_HARD_CONTEXT_RATIO", "0.95")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_RATIO", "0.75")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_MAX_TOKENS", "0")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_CHUNK_RATIO", "0.03")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_CHUNK_MAX_TOKENS", "0")
    monkeypatch.setattr(agent_graph, "_detected_context_length", lambda: 128000)

    snapshot = agent_graph._context_budget_snapshot(
        {"provider_reported_context_limit": 32768, "messages": [{"role": "user", "content": "hi"}]}
    )

    assert snapshot["detected_context_length"] == 128000
    assert snapshot["provider_reported_context_limit"] == 32768
    assert snapshot["context_length"] == 32768
    assert snapshot["active_limit"] == 16384
    assert snapshot["hard_limit"] == 31129
    assert snapshot["compression_summary_budget"]["token_limit"] == snapshot["context_length"]
    assert snapshot["compression_summary_budget"]["active_compression_token_limit"] == snapshot["effective_active_limit"]
    assert snapshot["compression_summary_budget"]["summary_prompt_tokens"] == 24576
    assert snapshot["compression_summary_budget"]["summary_chunk_output_tokens"] == 983
    assert snapshot["compression_summary_budget"]["max_tokens_zero_means_dynamic"] is True


def test_dynamic_compression_max_passes_defaults_to_until_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPHARAVIS_DYNAMIC_COMPRESSION_UNTIL_BUDGET", "true")
    monkeypatch.setenv("ALPHARAVIS_DYNAMIC_COMPRESSION_MAX_PASSES", "7")
    monkeypatch.setenv("ALPHARAVIS_DYNAMIC_COMPRESSION_HARD_MAX_PASSES", "5")

    assert agent_graph._dynamic_compression_max_passes("PRE_RUN", "ALPHARAVIS_PRE_RUN_COMPRESSION_MAX_PASSES", "3") == 5


def test_final_budget_rescue_runs_until_budget_with_dynamic_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPHARAVIS_ENABLE_FINAL_BUDGET_RESCUE", "true")
    monkeypatch.setenv("ALPHARAVIS_DYNAMIC_COMPRESSION_UNTIL_BUDGET", "true")
    monkeypatch.setenv("ALPHARAVIS_DYNAMIC_COMPRESSION_MAX_PASSES", "6")
    monkeypatch.setenv("ALPHARAVIS_DYNAMIC_COMPRESSION_HARD_MAX_PASSES", "12")
    monkeypatch.setattr(agent_graph, "_static_context_reserve_tokens", lambda state: 0)

    calls = {"count": 0}

    def fake_budget(state, *, messages=None):
        count = calls["count"]
        return {
            "context_length": 128000,
            "detected_context_length": 128000,
            "provider_reported_context_limit": None,
            "message_tokens": 2000 - (count * 300),
            "static_context_reserve_tokens": 0,
            "static_context_reserve_detail": {},
            "request_tokens": 2000 - (count * 300),
            "active_limit": 1000,
            "effective_active_limit": 1000,
            "hard_limit": 1200,
            "effective_hard_limit": 1200,
            "compression_needed": count < 4,
            "hard_rescue_needed": False,
            "message_count": len(messages or []),
            "archived_context_count": 0,
            "archive_collection_count": 0,
        }

    async def fake_compression(*, state, runtime, mode, token_limit, force, **_kwargs):
        calls["count"] += 1
        updates = {"messages": [{"role": "system", "content": f"compressed pass {calls['count']}"}]}
        return (
            CompressionResult(
                mode=mode,
                thread_id="thread",
                thread_key="thread:key",
                token_limit=token_limit,
                token_estimate_before=2000,
                token_estimate_after=900,
                head=[],
                middle=[{"role": "assistant", "content": "old"}],
                tail=[],
                summary="summary",
                summary_message_content="summary",
                archive_content="archive",
                archive_metadata={},
                pruned_middle_text="old",
            ),
            f"archive-{calls['count']}",
            updates,
        )

    monkeypatch.setattr(agent_graph, "_context_budget_snapshot", fake_budget)
    monkeypatch.setattr(agent_graph, "_run_hermes_style_compression", fake_compression)

    updates = asyncio.run(agent_graph.final_budget_rescue_node({"messages": [{"role": "user", "content": "x"}]}))

    assert calls["count"] == 4
    assert updates["run_profile"]["final_budget_rescue_passes"] == 4
    assert updates["run_profile"]["final_budget_rescue_max_passes"] == 6
    assert updates["run_profile"]["final_budget_rescue_budget_met"] is True
