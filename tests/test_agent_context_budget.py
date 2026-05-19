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
            "rag_active": False,
            "active_rag_file_ids": [],
            "active_source_keys": [],
            "archive_rag_mode": "tool_only",
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
    assert record["rag_active"] is False
    assert record["archive_rag_mode"] == "tool_only"
    assert record["metadata"]["ingest_status"] == "indexed"
    assert record["metadata"]["archive_rag_mode"] == "tool_only"
    assert len(writes) == 2


def test_rag_state_update_from_document_ingest_merges_active_sources() -> None:
    update = agent_graph._rag_state_update_from_ingest(
        {
            "rag_active": True,
            "active_source_keys": ["doc-old"],
            "active_rag_file_ids": ["file-old"],
            "rag_activation_reason": "document_ingest",
        },
        {
            "source_key": "doc-new",
            "rag_file_id": "file-new",
            "rag_active": True,
            "active_source_keys": ["doc-new"],
            "active_rag_file_ids": ["file-new"],
            "rag_activation_reason": "large_paste",
            "archive_rag_mode": "tool_only",
        },
    )

    assert update["rag_active"] is True
    assert update["active_source_keys"] == ["doc-old", "doc-new"]
    assert update["active_rag_file_ids"] == ["file-old", "file-new"]
    assert update["rag_activation_reason"] == "large_paste"
    assert update["archive_rag_mode"] == "tool_only"


def test_rag_state_update_from_archive_ingest_stays_passive() -> None:
    update = agent_graph._rag_state_update_from_ingest(
        {},
        {
            "source_key": "archive-1",
            "rag_file_id": "archive:archive-1",
            "rag_active": False,
            "archive_rag_mode": "tool_only",
        },
    )

    assert update == {"archive_rag_mode": "tool_only"}


def test_run_profile_start_ingests_large_paste_and_replaces_active_context(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    async def fake_ingest_source(**kwargs):
        calls["ingest"] = kwargs
        return {
            "source_key": kwargs["source_key"],
            "rag_file_id": kwargs["source_key"],
            "index_status": "indexed",
            "indexed_backends": ["rag_api"],
            "rag_active": True,
            "active_source_keys": [kwargs["source_key"]],
            "active_rag_file_ids": [kwargs["source_key"]],
            "rag_activation_reason": "large_paste",
            "archive_rag_mode": "tool_only",
            "metadata": {
                "source_key": kwargs["source_key"],
                "rag_file_id": kwargs["source_key"],
                "rag_active": True,
                "active_source_keys": [kwargs["source_key"]],
                "active_rag_file_ids": [kwargs["source_key"]],
                "rag_activation_reason": "large_paste",
                "archive_rag_mode": "tool_only",
            },
        }

    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_MIN_CHARS", "20")
    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_COMPRESSION_MARGIN_TOKENS", "999999")
    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_MARKER_PREVIEW_CHARS", "8")
    monkeypatch.setattr(agent_graph, "_router_ingest_source", fake_ingest_source)
    monkeypatch.setattr(agent_graph, "_maybe_index_vector_memory", object())

    content = "0123456789" * 5
    updates = asyncio.run(
        agent_graph.run_profile_start_node(
            {
                "messages": [{"role": "human", "content": content, "id": "msg-1"}],
                "thread_id": "thread-1",
                "thread_key": "thread-key",
            }
        )
    )

    assert calls["ingest"]["source_type"] == "large_paste"
    assert calls["ingest"]["content"] == content
    assert updates["rag_active"] is True
    assert updates["rag_activation_reason"] == "large_paste"
    assert updates["active_source_keys"] == [calls["ingest"]["source_key"]]
    replacement_messages = updates["messages"]
    assert isinstance(replacement_messages[0], agent_graph.RemoveMessage)
    replacement_text = replacement_messages[1]["content"]
    assert "Large paste indexed for bounded RAG retrieval" in replacement_text
    assert content not in replacement_text
    ingest_record = updates["run_profile"]["large_paste_ingests"][0]
    assert ingest_record["index_status"] == "indexed"
    assert ingest_record["events"][0]["event"] == "large_ingest.started"
    assert ingest_record["events"][-1]["event"] == "large_ingest.completed"
    assert ingest_record["events"][-1]["status"] == "indexed"


def test_run_profile_start_skips_auto_large_paste_when_context_margin_is_large(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    async def fake_ingest_source(**kwargs):
        calls["ingest"] = kwargs
        return {}

    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_MIN_CHARS", "20")
    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_COMPRESSION_MARGIN_TOKENS", "5")
    monkeypatch.setattr(agent_graph, "_router_ingest_source", fake_ingest_source)
    monkeypatch.setattr(agent_graph, "_maybe_index_vector_memory", object())

    content = "Document:\n" + ("runtime marker with enough content " * 20)
    updates = asyncio.run(
        agent_graph.run_profile_start_node(
            {
                "messages": [{"role": "human", "content": content, "id": "msg-1"}],
                "thread_id": "thread-1",
                "thread_key": "thread-key",
            }
        )
    )

    assert calls == {}
    assert "messages" not in updates
    ingest_record = updates["run_profile"]["large_paste_ingests"][0]
    assert ingest_record["index_status"] == "skipped"
    assert ingest_record["skip_reason"] == "context_margin_above_auto_rag_threshold"
    assert ingest_record["tokens_until_compression"] > ingest_record["auto_margin_tokens"]
    assert ingest_record["events"] == [
        {
            "event": "large_ingest.skipped",
            "t": 0.0,
            "reason": "context_margin_above_auto_rag_threshold",
            "content_chars": len(content),
        }
    ]


def test_run_profile_start_manual_rag_block_forces_ingest(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    async def fake_ingest_source(**kwargs):
        calls["ingest"] = kwargs
        return {
            "source_key": kwargs["source_key"],
            "rag_file_id": kwargs["source_key"],
            "index_status": "indexed",
            "indexed_backends": ["alpharavis_pgvector"],
            "rag_active": True,
            "active_source_keys": [kwargs["source_key"]],
            "active_rag_file_ids": [],
            "rag_activation_reason": "large_paste",
            "archive_rag_mode": "tool_only",
            "metadata": {
                "source_key": kwargs["source_key"],
                "rag_file_id": kwargs["source_key"],
                "rag_active": True,
                "active_source_keys": [kwargs["source_key"]],
                "active_rag_file_ids": [],
                "rag_activation_reason": "large_paste",
                "archive_rag_mode": "tool_only",
            },
        }

    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_MIN_CHARS", "999999")
    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_COMPRESSION_MARGIN_TOKENS", "0")
    monkeypatch.setattr(agent_graph, "_router_ingest_source", fake_ingest_source)
    monkeypatch.setattr(agent_graph, "_maybe_index_vector_memory", object())

    content = "\n".join(
        [
            "Bitte merke dieses Dokument.",
            "/rag",
            "Document:",
            "Runtime marker: MANUAL_RAG_BLOCK. This block must be indexed.",
            "/rag",
            "Was steht im Marker?",
        ]
    )
    updates = asyncio.run(
        agent_graph.run_profile_start_node(
            {
                "messages": [{"role": "human", "content": content, "id": "msg-1"}],
                "thread_id": "thread-1",
                "thread_key": "thread-key",
            }
        )
    )

    assert calls["ingest"]["metadata"]["manual_rag_block"] is True
    assert "MANUAL_RAG_BLOCK" in calls["ingest"]["content"]
    assert updates["rag_active"] is True
    ingest_record = updates["run_profile"]["large_paste_ingests"][0]
    assert ingest_record["events"][0]["event"] == "large_ingest.started"
    assert ingest_record["events"][-1]["indexed_backends"] == ["alpharavis_pgvector"]
    replacement_text = updates["messages"][1]["content"]
    assert "/rag" not in replacement_text
    assert "Large paste indexed for bounded RAG retrieval" in replacement_text
    assert "Was steht im Marker?" in replacement_text


def test_large_paste_intent_classifier_detects_instruction_and_mixed() -> None:
    instruction = """
System prompt:
You are a coding agent.
Always follow the repository instructions.
Never expose hidden memory.
Output format: concise final answer.
"""
    mixed = """
Instructions:
You must extract only supported facts.

Document:
Runtime marker: ABC. This report contains deployment data and logs.
"""

    instruction_result = agent_graph._classify_large_paste_intent(instruction)
    mixed_result = agent_graph._classify_large_paste_intent(mixed)

    assert instruction_result["intent"] == "instruction"
    assert instruction_result["instruction_score"] > instruction_result["document_score"]
    assert mixed_result["intent"] == "mixed"


def test_run_profile_start_indexes_instruction_paste_without_rag_activation(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    async def fake_ingest_source(**kwargs):
        calls["ingest"] = kwargs
        return {
            "source_key": kwargs["source_key"],
            "rag_file_id": kwargs["source_key"],
            "index_status": "indexed",
            "indexed_backends": ["alpharavis_pgvector"],
            "rag_active": False,
            "active_source_keys": [],
            "active_rag_file_ids": [],
            "archive_rag_mode": "",
            "metadata": {
                "source_key": kwargs["source_key"],
                "rag_file_id": kwargs["source_key"],
                "rag_active": False,
                "active_source_keys": [],
                "active_rag_file_ids": [],
            },
        }

    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_MIN_CHARS", "20")
    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_COMPRESSION_MARGIN_TOKENS", "999999")
    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_INSTRUCTION_BRIEF_CHARS", "500")
    monkeypatch.setattr(agent_graph, "_router_ingest_source", fake_ingest_source)
    monkeypatch.setattr(agent_graph, "_maybe_index_vector_memory", object())

    content = "\n".join(
        [
            "System prompt:",
            "You are a strict local coding agent.",
            "Always follow AGENTS.md.",
            "Never expose hidden memory.",
            "Output format: concise engineering answer.",
            *[f"UNIMPORTANT TRAILING DETAIL {index}" for index in range(80)],
        ]
    )
    updates = asyncio.run(
        agent_graph.run_profile_start_node(
            {
                "messages": [{"role": "human", "content": content, "id": "msg-1"}],
                "thread_id": "thread-1",
                "thread_key": "thread-key",
            }
        )
    )

    assert calls["ingest"]["source_type"] == "large_instruction"
    assert calls["ingest"]["preferred_backend"] == "alpharavis_pgvector"
    assert calls["ingest"]["metadata"]["paste_intent"] == "instruction"
    assert updates.get("rag_active") is not True
    assert updates["run_profile"]["rag_active"] is False
    ingest_record = updates["run_profile"]["large_paste_ingests"][0]
    assert ingest_record["paste_intent"] == "instruction"
    assert ingest_record["rag_active"] is False
    replacement_text = updates["messages"][1]["content"]
    assert "classified as instruction-like" in replacement_text
    assert "Follow the condensed instruction brief" in replacement_text
    assert "UNIMPORTANT TRAILING DETAIL 79" not in replacement_text


def test_run_profile_start_keeps_mixed_paste_rag_active_with_instruction_brief(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    async def fake_ingest_source(**kwargs):
        calls["ingest"] = kwargs
        return {
            "source_key": kwargs["source_key"],
            "rag_file_id": kwargs["source_key"],
            "index_status": "indexed",
            "indexed_backends": ["alpharavis_pgvector"],
            "rag_active": True,
            "active_source_keys": [kwargs["source_key"]],
            "active_rag_file_ids": [],
            "rag_activation_reason": "large_paste",
            "archive_rag_mode": "tool_only",
            "metadata": {
                "source_key": kwargs["source_key"],
                "rag_file_id": kwargs["source_key"],
                "rag_active": True,
                "active_source_keys": [kwargs["source_key"]],
                "active_rag_file_ids": [],
                "rag_activation_reason": "large_paste",
                "archive_rag_mode": "tool_only",
            },
        }

    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_MIN_CHARS", "20")
    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_COMPRESSION_MARGIN_TOKENS", "999999")
    monkeypatch.setattr(agent_graph, "_router_ingest_source", fake_ingest_source)
    monkeypatch.setattr(agent_graph, "_maybe_index_vector_memory", object())

    content = "\n".join(
        [
            "Instructions:",
            "You must answer only from the provided document.",
            "Never invent facts.",
            "",
            "Document:",
            "Runtime marker: MIXED_NATIVE_RAG. The deployment uses AlphaRavis pgvector.",
            *[f"log line {index}: deployment data" for index in range(20)],
        ]
    )
    updates = asyncio.run(
        agent_graph.run_profile_start_node(
            {
                "messages": [{"role": "human", "content": content, "id": "msg-1"}],
                "thread_id": "thread-1",
                "thread_key": "thread-key",
            }
        )
    )

    assert calls["ingest"]["source_type"] == "large_paste"
    assert calls["ingest"]["metadata"]["paste_intent"] == "mixed"
    assert "You must answer only" not in calls["ingest"]["content"]
    assert "Runtime marker: MIXED_NATIVE_RAG" in calls["ingest"]["content"]
    assert calls["ingest"]["metadata"]["instruction_text_stripped_from_index"] is True
    assert updates["rag_active"] is True
    assert updates["active_source_keys"] == [calls["ingest"]["source_key"]]
    replacement_text = updates["messages"][1]["content"]
    assert "classified as mixed instructions plus document/data" in replacement_text
    assert "Use active RAG/query_source" in replacement_text


def test_active_rag_prefetch_injects_bounded_context(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    async def fake_agentic_rag_retrieve(**kwargs):
        calls.update(kwargs)
        return {
            "final_query": kwargs["query"],
            "graph_trace": [{"node": "retrieve"}],
            "context_packet": {
                "query": kwargs["query"],
                "chunk_count": 1,
                "chunks": [
                    {
                        "rank": 1,
                        "source_key": "doc-1",
                        "retrieval_backend": "rag_api",
                        "relevance_score": 0.9,
                        "chunk_text": "The grounded document detail.",
                    }
                ],
            },
        }

    monkeypatch.setattr(agent_graph, "_router_agentic_rag_retrieve", fake_agentic_rag_retrieve)

    updates = asyncio.run(
        agent_graph.active_rag_prefetch_node(
            {
                "messages": [{"role": "human", "content": "Welche Details stehen im Dokument?"}],
                "thread_id": "thread-1",
                "rag_active": True,
                "active_source_keys": ["doc-1"],
                "active_rag_file_ids": ["doc-1"],
                "archive_rag_mode": "tool_only",
            }
        )
    )

    assert calls["source_keys"] == ["doc-1"]
    assert calls["rag_source_keys"] == ["doc-1"]
    assert updates["run_profile"]["active_rag_prefetch_status"] == "injected"
    assert updates["messages"][0].id == agent_graph.ACTIVE_RAG_CONTEXT_MESSAGE_ID
    assert "The grounded document detail." in updates["messages"][0].content


def test_active_rag_prefetch_uses_pgvector_only_sources_without_rag_file_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    async def fake_agentic_rag_retrieve(**kwargs):
        calls.update(kwargs)
        return {"context_packet": {"query": kwargs["query"], "chunk_count": 0, "chunks": []}, "graph_trace": []}

    monkeypatch.setattr(agent_graph, "_router_agentic_rag_retrieve", fake_agentic_rag_retrieve)

    asyncio.run(
        agent_graph.active_rag_prefetch_node(
            {
                "messages": [{"role": "human", "content": "Welche Details stehen im Dokument?"}],
                "thread_id": "thread-1",
                "rag_active": True,
                "active_source_keys": ["doc-1"],
                "active_rag_file_ids": [],
                "archive_rag_mode": "tool_only",
            }
        )
    )

    assert calls["source_keys"] == ["doc-1"]
    assert calls["rag_source_keys"] is None


def test_active_rag_prefetch_stays_inactive_for_archive_only() -> None:
    updates = asyncio.run(
        agent_graph.active_rag_prefetch_node(
            {
                "messages": [{"role": "human", "content": "Was war im Archiv?"}],
                "rag_active": True,
                "active_rag_file_ids": ["archive:archive-1"],
                "archive_rag_mode": "tool_only",
            }
        )
    )

    assert "messages" not in updates
    assert updates["run_profile"]["active_rag_prefetch_status"] == "archive_tool_only"


def test_active_rag_prefetch_does_not_inject_without_grounded_context(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_agentic_rag_retrieve(**kwargs):
        return {"context_packet": {"query": kwargs["query"], "chunk_count": 0, "chunks": []}, "graph_trace": []}

    monkeypatch.setattr(agent_graph, "_router_agentic_rag_retrieve", fake_agentic_rag_retrieve)

    updates = asyncio.run(
        agent_graph.active_rag_prefetch_node(
            {
                "messages": [{"role": "human", "content": "Welche Details stehen im Dokument?"}],
                "rag_active": True,
                "active_source_keys": ["doc-1"],
                "active_rag_file_ids": ["doc-1"],
            }
        )
    )

    assert "messages" not in updates
    assert updates["run_profile"]["active_rag_prefetch_status"] == "no_grounded_context"


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
