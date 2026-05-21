from __future__ import annotations

import asyncio
import json
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
    events = record["metadata"]["events"]
    assert events[-1]["event"] == "compression.postcompact"
    assert events[-1]["archive_key"] == archive_key
    assert events[-1]["scope"] == "PRE_RUN"
    assert events[-1]["token_estimate_before"] == 2000
    assert events[-1]["token_estimate_after"] == 800
    assert result.archive_metadata["events"] == events
    assert calls["ingest"]["metadata"]["events"][-1]["event"] == "compression.postcompact"
    assert len(writes) == 2


def test_write_artifact_indexes_through_retrieval_router(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}
    writes: list[tuple[tuple[str, ...], str, dict[str, object]]] = []

    async def fake_ingest_source(**kwargs):
        calls["ingest"] = kwargs
        return {
            "index_status": "queued",
            "rag_file_id": "artifact:artifact-key",
            "indexed_backends": [],
            "queued_backends": ["alpharavis_pgvector"],
            "rag_active": True,
            "active_source_keys": [kwargs["source_key"]],
            "active_rag_file_ids": [],
            "backend_results": {"alpharavis_pgvector": "queued embedding job for artifact"},
            "warnings": [],
            "errors": [],
        }

    async def fake_put(_store, namespace, key, value):
        writes.append((namespace, key, value))

    monkeypatch.setenv("ALPHARAVIS_ENABLE_ARTIFACTS", "true")
    monkeypatch.setattr(agent_graph, "_workspace_root", lambda: str(tmp_path))
    monkeypatch.setattr(agent_graph, "_state_thread_id", lambda *_args, **_kwargs: "thread-1")
    monkeypatch.setattr(agent_graph, "_state_thread_key", lambda *_args, **_kwargs: "thread:key")
    monkeypatch.setattr(agent_graph, "get_store", lambda: object())
    monkeypatch.setattr(agent_graph, "_maybe_put", fake_put)
    monkeypatch.setattr(agent_graph, "_router_ingest_source", fake_ingest_source)

    raw = asyncio.run(
        agent_graph.write_alpha_ravis_artifact(
            title="Artifact Router Smoke",
            content="Artifact body should be indexed through retrieval_router.ingest_source.",
            artifact_type="note",
            suggested_filename="router-smoke.md",
        )
    )
    payload = json.loads(raw)

    assert calls["ingest"]["source_type"] == "artifact"
    assert calls["ingest"]["title"] == "Artifact Router Smoke"
    assert calls["ingest"]["thread_id"] == "thread-1"
    assert calls["ingest"]["thread_key"] == "thread:key"
    assert calls["ingest"]["preferred_backend"] == "auto"
    assert calls["ingest"]["pgvector_index"] == agent_graph._maybe_index_vector_memory
    assert calls["ingest"]["metadata"]["rag_activation_reason"] == "artifact"
    assert calls["ingest"]["metadata"]["filename"].endswith("router-smoke.md")
    assert payload["ingest_status"] == "queued"
    assert payload["queued_backends"] == ["alpharavis_pgvector"]
    assert payload["rag_file_id"] == "artifact:artifact-key"
    assert writes[-1][2]["ingest_status"] == "queued"


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


def test_compact_instructions_parse_chat_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPHARAVIS_COMPACT_INSTRUCTIONS_MAX_CHARS", "200")
    text = """
Bitte weiterarbeiten.
<focus_topic>RAG archive recall and source manifests</focus_topic>
/compact preserve exact file paths and commands
"""

    extracted = agent_graph._extract_compact_instructions(text)

    assert "focus_topic: RAG archive recall and source manifests" in extracted
    assert "preserve exact file paths and commands" in extracted


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
    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_AUTO_STAGE", "pre_run")
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
    assert "Source manifest:" in replacement_text
    assert content not in replacement_text
    ingest_record = updates["run_profile"]["large_paste_ingests"][0]
    assert ingest_record["index_status"] == "indexed"
    assert ingest_record["source_manifest"]["source_key"] == calls["ingest"]["source_key"]
    assert ingest_record["source_manifest"]["message_index"] == 0
    assert ingest_record["events"][0]["event"] == "large_ingest.started"
    assert ingest_record["events"][-1]["event"] == "large_ingest.completed"
    assert ingest_record["events"][-1]["status"] == "indexed"


def test_large_paste_auto_ingest_defers_until_after_pre_run_compression(monkeypatch: pytest.MonkeyPatch) -> None:
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
        }

    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_MIN_CHARS", "200")
    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_AUTO_STAGE", "post_compression")
    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_POST_COMPRESSION_TRIGGER_RATIO", "0.01")
    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_POST_RAG_COMPRESSION_ENABLED", "false")
    monkeypatch.setenv("ALPHARAVIS_ENABLE_LARGE_PASTE_SMALL_CLASSIFIER", "false")
    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_MARKER_PREVIEW_CHARS", "0")
    monkeypatch.setattr(agent_graph, "_router_ingest_source", fake_ingest_source)
    monkeypatch.setattr(agent_graph, "_maybe_index_vector_memory", object())

    content = "Document:\n" + ("post compression large paste marker " * 800)
    start_updates = asyncio.run(
        agent_graph.run_profile_start_node(
            {
                "messages": [{"role": "human", "content": content, "id": "msg-1"}],
                "thread_id": "thread-1",
                "thread_key": "thread-key",
            }
        )
    )

    assert calls == {}
    assert "messages" not in start_updates
    assert start_updates["run_profile"]["large_paste_ingests"][0]["skip_reason"] == "large_paste_deferred_until_post_compression"

    post_updates = asyncio.run(
        agent_graph.large_paste_post_compression_node(
            {
                **start_updates,
                "messages": [{"role": "human", "content": content, "id": "msg-1"}],
                "thread_id": "thread-1",
                "thread_key": "thread-key",
            }
        )
    )

    assert calls["ingest"]["content"] == content
    assert post_updates["rag_active"] is True
    assert post_updates["run_profile"]["large_paste_post_compression_ingested"] is True
    assert post_updates["run_profile"]["large_paste_ingests"][-1]["index_status"] == "indexed"
    assert isinstance(post_updates["messages"][0], agent_graph.RemoveMessage)


def test_post_compression_130k_first_message_is_chunked_and_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    async def fake_pgvector_index(**kwargs):
        calls["pgvector"] = kwargs
        progress_callback = kwargs.get("progress_callback")
        for index in range(3):
            event = {
                "event": "large_ingest.chunk_indexed",
                "source_type": kwargs["source_type"],
                "source_key": kwargs["source_key"],
                "chunk_index": index,
                "chunk_number": index + 1,
                "chunk_count": 3,
                "chunk_chars": 45000 if index < 2 else 40000,
                "chunk_digest": f"chunk-{index}",
                "source_digest": "source-digest-130k",
            }
            if progress_callback is not None:
                progress_callback(event)
        return "indexed:large_paste:3"

    async def fake_ingest_source(**kwargs):
        calls["ingest"] = kwargs
        await kwargs["pgvector_index"](
            source_type=kwargs["source_type"],
            source_key=kwargs["source_key"],
            title=kwargs["title"],
            content=kwargs["content"],
            thread_id=kwargs["thread_id"],
            thread_key=kwargs["thread_key"],
            scope=kwargs["scope"],
            metadata=kwargs["metadata"],
        )
        return {
            "source_key": kwargs["source_key"],
            "rag_file_id": kwargs["source_key"],
            "index_status": "indexed",
            "indexed_backends": ["alpharavis_pgvector"],
            "queued_backends": [],
            "rag_active": True,
            "active_source_keys": [kwargs["source_key"]],
            "active_rag_file_ids": [],
            "rag_activation_reason": "large_paste",
            "archive_rag_mode": "tool_only",
        }

    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_MIN_CHARS", "20000")
    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_AUTO_STAGE", "post_compression")
    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_POST_COMPRESSION_TRIGGER_RATIO", "0.01")
    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_POST_RAG_COMPRESSION_ENABLED", "false")
    monkeypatch.setenv("ALPHARAVIS_ENABLE_LARGE_PASTE_SMALL_CLASSIFIER", "false")
    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_MARKER_PREVIEW_CHARS", "0")
    monkeypatch.setattr(agent_graph, "_router_ingest_source", fake_ingest_source)
    monkeypatch.setattr(agent_graph, "_maybe_index_vector_memory", fake_pgvector_index)

    content = "Document:\n" + ("FIRST_MESSAGE_130K_CHUNKED source body line.\n" * 3300)
    assert len(content) > 130000

    updates = asyncio.run(
        agent_graph.large_paste_post_compression_node(
            {
                "messages": [{"role": "human", "content": content, "id": "msg-1"}],
                "thread_id": "thread-1",
                "thread_key": "thread-key",
                "run_profile": {"large_paste_ingests": []},
            }
        )
    )

    assert calls["ingest"]["content"] == content
    ingest_record = updates["run_profile"]["large_paste_ingests"][-1]
    assert ingest_record["chunk_count"] == 3
    assert ingest_record["indexed_chunk_count"] == 3
    assert ingest_record["source_manifest"]["chunk_count"] == 3
    assert ingest_record["source_manifest"]["source_digest"] == "source-digest-130k"
    assert [event["chunk_number"] for event in ingest_record["events"] if event["event"] == "large_ingest.chunk_indexed"] == [1, 2, 3]
    replacement_text = updates["messages"][1]["content"]
    assert "Large paste indexed for bounded RAG retrieval" in replacement_text
    assert "No explicit current question was detected" in replacement_text
    assert "FIRST_MESSAGE_130K_CHUNKED" not in replacement_text


def test_large_paste_post_rag_compresses_remaining_context(monkeypatch: pytest.MonkeyPatch) -> None:
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
        }

    async def fake_compress(**kwargs):
        calls["compress"] = kwargs
        result = types.SimpleNamespace(
            skipped=False,
            reason="",
            middle=[{"role": "assistant", "content": "old chatter"}],
            head=[],
            tail=[],
            summary_failed=False,
            summary_error="",
            archive_metadata={},
        )
        return result, "archive-after-rag", {
            "messages": [
                agent_graph.RemoveMessage(id=agent_graph.REMOVE_ALL_MESSAGES),
                {"role": "system", "content": "compressed remaining chatter"},
            ],
            "run_profile": {"pre_run_compression_used": True},
        }

    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_MIN_CHARS", "200")
    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_AUTO_STAGE", "post_compression")
    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_POST_COMPRESSION_TRIGGER_RATIO", "0.01")
    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_POST_RAG_COMPRESSION_ENABLED", "true")
    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_POST_RAG_COMPRESSION_TRIGGER_RATIO", "0.01")
    monkeypatch.setenv("ALPHARAVIS_ENABLE_LARGE_PASTE_SMALL_CLASSIFIER", "false")
    monkeypatch.setattr(agent_graph, "_router_ingest_source", fake_ingest_source)
    monkeypatch.setattr(agent_graph, "_maybe_index_vector_memory", object())
    monkeypatch.setattr(agent_graph, "_run_hermes_style_compression", fake_compress)

    content = "Document:\n" + ("post rag compression source " * 1000)
    updates = asyncio.run(
        agent_graph.large_paste_post_compression_node(
                {
                    "messages": [
                        {"role": "assistant", "content": "old chatter " * 4000},
                        {"role": "human", "content": content, "id": "msg-1"},
                    ],
                "thread_id": "thread-1",
                "thread_key": "thread-key",
                "run_profile": {"large_paste_ingests": []},
            }
        )
    )

    assert calls["ingest"]["content"] == content
    assert calls["compress"]["token_limit"] > 0
    assert updates["messages"][1]["content"] == "compressed remaining chatter"
    assert updates["run_profile"]["large_paste_post_rag_compression_used"] is True
    assert updates["run_profile"]["large_paste_post_rag_compression_archive_key"] == "archive-after-rag"


def test_run_profile_start_skips_auto_large_paste_when_context_margin_is_large(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    async def fake_ingest_source(**kwargs):
        calls["ingest"] = kwargs
        return {}

    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_MIN_CHARS", "20")
    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_AUTO_STAGE", "pre_run")
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


def test_run_profile_start_replaces_queued_large_paste_with_source_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_ingest_source(**kwargs):
        return {
            "source_key": kwargs["source_key"],
            "rag_file_id": kwargs["source_key"],
            "index_status": "queued",
            "indexed_backends": [],
            "queued_backends": ["alpharavis_pgvector"],
            "rag_active": True,
            "active_source_keys": [kwargs["source_key"]],
            "active_rag_file_ids": [],
            "rag_activation_reason": "large_paste",
            "archive_rag_mode": "tool_only",
        }

    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_MIN_CHARS", "20")
    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_AUTO_STAGE", "pre_run")
    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_COMPRESSION_MARGIN_TOKENS", "999999")
    monkeypatch.setattr(agent_graph, "_router_ingest_source", fake_ingest_source)
    monkeypatch.setattr(agent_graph, "_maybe_index_vector_memory", object())

    content = "Document:\n" + ("queued native pgvector source " * 10)
    updates = asyncio.run(
        agent_graph.run_profile_start_node(
            {
                "messages": [{"role": "human", "content": content, "id": "msg-1"}],
                "thread_id": "thread-1",
                "thread_key": "thread-key",
            }
        )
    )

    ingest_record = updates["run_profile"]["large_paste_ingests"][0]
    assert ingest_record["index_status"] == "queued"
    assert ingest_record["queued_backends"] == ["alpharavis_pgvector"]
    assert ingest_record["events"][-1]["status"] == "queued"
    assert ingest_record["events"][-1]["queued_backends"] == ["alpharavis_pgvector"]
    assert updates["rag_active"] is True
    replacement_text = updates["messages"][1]["content"]
    assert "Large paste queued for bounded RAG retrieval" in replacement_text
    assert "Source manifest:" in replacement_text
    assert content not in replacement_text


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


def test_big_context_tag_forces_ingest_and_keeps_outer_question(monkeypatch: pytest.MonkeyPatch) -> None:
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
        }

    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_MIN_CHARS", "999999")
    monkeypatch.setattr(agent_graph, "_router_ingest_source", fake_ingest_source)
    monkeypatch.setattr(agent_graph, "_maybe_index_vector_memory", object())

    content = "\n".join(
        [
            "Bitte benutze diese Quelle.",
            '<big-context name="ops-log">',
            "Runtime marker: BIG_CONTEXT_BLOCK. This block must be indexed.",
            "</big-context>",
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
    assert "BIG_CONTEXT_BLOCK" in calls["ingest"]["content"]
    replacement_text = updates["messages"][1]["content"]
    assert "<big-context" not in replacement_text
    assert "Was steht im Marker?" in replacement_text
    assert "Source manifest:" in replacement_text


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
    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_AUTO_STAGE", "pre_run")
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
    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_AUTO_STAGE", "pre_run")
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


def test_large_paste_small_classifier_ranges_strip_instruction_and_question(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    async def fake_classifier(text: str):
        return {
            "intent": "mixed",
            "retrieval_query": "MIXED_RANGE_MARKER AlphaRavis deployment",
            "instruction_lines": [[1, 3]],
            "document_lines": [[5, 26]],
            "question_lines": [[27, 27]],
            "confidence": 0.92,
            "model": "small",
            "base_url": "http://100.71.57.22:8001/v1",
            "elapsed_seconds": 0.1,
        }

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
        }

    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_MIN_CHARS", "20")
    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_AUTO_STAGE", "pre_run")
    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_COMPRESSION_MARGIN_TOKENS", "999999")
    monkeypatch.setenv("ALPHARAVIS_ENABLE_LARGE_PASTE_SMALL_CLASSIFIER", "true")
    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_SMALL_CLASSIFIER_MIN_CHARS", "20")
    monkeypatch.setattr(agent_graph, "_classify_prompt_for_retrieval", fake_classifier)
    monkeypatch.setattr(agent_graph, "_router_ingest_source", fake_ingest_source)
    monkeypatch.setattr(agent_graph, "_maybe_index_vector_memory", object())

    content = "\n".join(
        [
            "Instructions:",
            "You must answer only from the provided document.",
            "Never invent facts.",
            "",
            "Document:",
            "Runtime marker: MIXED_RANGE_MARKER. The deployment uses AlphaRavis pgvector.",
            *[f"log line {index}: deployment data and classifier range coverage" for index in range(20)],
            "Welche Deployment-Regel steht im Dokument?",
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

    indexed_content = calls["ingest"]["content"]
    assert "Runtime marker: MIXED_RANGE_MARKER" in indexed_content
    assert "You must answer only" not in indexed_content
    assert "Welche Deployment-Regel" not in indexed_content
    assert calls["ingest"]["metadata"]["paste_intent_classifier"] == "small_model_classifier"
    assert calls["ingest"]["metadata"]["paste_intent_question_line_ranges"] == [[27, 27]]
    ingest_record = updates["run_profile"]["large_paste_ingests"][0]
    assert ingest_record["paste_intent_classifier"] == "small_model_classifier"
    replacement_text = updates["messages"][1]["content"]
    assert "Retrieval/query focus" in replacement_text
    assert "MIXED_RANGE_MARKER AlphaRavis deployment" in replacement_text
    assert "Current question/task lines" in replacement_text


def test_classifier_json_parser_recovers_complete_fields_from_truncated_reason() -> None:
    parsed = agent_graph._parse_classifier_json(
        '{"intent":"mixed","retrieval_query":"AlphaRavis deployment",'
        '"instruction_lines":[[1,3]],"document_lines":[[5,26]],'
        '"question_lines":[[27,27]],"confidence":0.92,"reason":"unterminated'
    )

    assert parsed["intent"] == "mixed"
    assert parsed["retrieval_query"] == "AlphaRavis deployment"
    assert parsed["instruction_lines"] == [[1, 3]]
    assert parsed["question_lines"] == [[27, 27]]
    assert parsed["confidence"] == 0.92
    assert "reason" not in parsed


def test_large_paste_small_instruction_does_not_override_local_mixed_document() -> None:
    result = agent_graph._large_paste_intent_from_small_classifier(
        {
            "intent": "instruction",
            "retrieval_query": "deployment rule",
            "instruction_lines": [[1, 3]],
            "document_lines": [],
            "question_lines": [],
            "confidence": 0.91,
        },
        {
            "intent": "mixed",
            "classifier": "heuristic",
            "confidence": 0.7,
            "instruction_score": 4.0,
            "document_score": 3.0,
        },
    )

    assert result["intent"] == "mixed"
    body = agent_graph._large_paste_document_body_for_index(
        "\n".join(
            [
                "Instructions:",
                "You must use only the document.",
                "Never invent facts.",
                "",
                "Document:",
                "Runtime marker: LOCAL_MIXED_WINS.",
                *[f"log line {index}: document data" for index in range(20)],
                "Welche Regel steht im Dokument?",
            ]
        ),
        result["intent"],
        result,
    )
    assert "You must use only" not in body
    assert "Welche Regel" not in body
    assert "Runtime marker: LOCAL_MIXED_WINS" in body


def test_source_metadata_summary_labels_code_log_table_and_symbols() -> None:
    code = agent_graph._source_metadata_summary(
        "def alpha_ravis_router():\n    return 'ok'\nclass RetrievalRouter:\n    pass",
        title="router.py",
        metadata={},
    )
    log = agent_graph._source_metadata_summary(
        "2026-05-20 ERROR api-bridge failed\nTraceback boom\nINFO retry complete",
        title="api.log",
        metadata={},
    )
    table = agent_graph._source_metadata_summary(
        "name,status,count\nalpha,ok,3\nbeta,failed,1\ngamma,ok,2",
        title="results.csv",
        metadata={},
    )

    assert code["content_type"] == "code"
    assert "alpha_ravis_router" in code["source_symbols"]
    assert log["content_type"] == "log"
    assert table["content_type"] == "table"
    assert code["source_keywords"]


def test_large_paste_ingest_adds_source_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    async def fake_classifier(content: str):
        return {"intent": "document", "classifier": "heuristic", "confidence": 0.8}

    async def fake_ingest_source(**kwargs):
        calls["ingest"] = kwargs
        return {
            "index_status": "indexed",
            "rag_file_id": kwargs["source_key"],
            "indexed_backends": ["alpharavis_pgvector"],
            "queued_backends": [],
            "rag_active": True,
            "active_source_keys": [kwargs["source_key"]],
            "active_rag_file_ids": [],
            "rag_activation_reason": "large_paste",
            "metadata": kwargs["metadata"],
        }

    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_MIN_CHARS", "20")
    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_AUTO_STAGE", "pre_run")
    monkeypatch.setenv("ALPHARAVIS_LARGE_PASTE_RAG_COMPRESSION_MARGIN_TOKENS", "999999")
    monkeypatch.setattr(agent_graph, "_classify_large_paste_for_ingest", fake_classifier)
    monkeypatch.setattr(agent_graph, "_router_ingest_source", fake_ingest_source)
    monkeypatch.setattr(agent_graph, "_maybe_index_vector_memory", object())

    content = "def alpha_ravis_source_metadata():\n    return 'code'\n" * 6
    updates = asyncio.run(
        agent_graph.run_profile_start_node(
            {
                "messages": [{"role": "human", "content": content, "id": "msg-1"}],
                "thread_id": "thread-1",
                "thread_key": "thread-key",
            }
        )
    )

    metadata = calls["ingest"]["metadata"]
    assert metadata["content_type"] == "code"
    assert "alpha_ravis_source_metadata" in metadata["source_symbols"]
    ingest_record = updates["run_profile"]["large_paste_ingests"][0]
    assert ingest_record["content_type"] == "code"
    assert "source_keywords" in ingest_record


def test_long_prompt_direct_route_classifier_can_use_fast_path(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_classifier(text: str):
        return {
            "intent": "noisy_query",
            "retrieval_query": "weather in berlin",
            "instruction_lines": [],
            "document_lines": [],
            "question_lines": [[1, 1]],
            "confidence": 0.93,
        }

    monkeypatch.setenv("ALPHARAVIS_ENABLE_FAST_PATH", "true")
    monkeypatch.setenv("ALPHARAVIS_FAST_PATH_MAX_CHARS", "80")
    monkeypatch.setenv("ALPHARAVIS_ENABLE_RETRIEVAL_QUERY_CLASSIFIER", "true")
    monkeypatch.setenv("ALPHARAVIS_RETRIEVAL_QUERY_CLASSIFIER_MIN_CHARS", "500")
    monkeypatch.setenv("ALPHARAVIS_LONG_PROMPT_DIRECT_ROUTE_MAX_CHARS", "2000")
    monkeypatch.setattr(agent_graph, "_classify_prompt_for_retrieval", fake_classifier)

    long_noisy = "Viel Vorrede ohne Spezialbedarf. " * 30 + "Was ist die kurze Antwort?"
    updates = asyncio.run(agent_graph.route_decision_node({"messages": [{"role": "human", "content": long_noisy}]}))

    assert updates["fast_path_route"] == "fast_path"
    assert updates["run_profile"]["route_classifier"]["intent"] == "noisy_query"


def test_long_prompt_document_route_stays_agent_path(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_classifier(text: str):
        return {
            "intent": "document",
            "retrieval_query": "deployment document",
            "instruction_lines": [],
            "document_lines": [[1, 20]],
            "question_lines": [[21, 21]],
            "confidence": 0.95,
        }

    monkeypatch.setenv("ALPHARAVIS_ENABLE_FAST_PATH", "true")
    monkeypatch.setenv("ALPHARAVIS_FAST_PATH_MAX_CHARS", "80")
    monkeypatch.setenv("ALPHARAVIS_ENABLE_RETRIEVAL_QUERY_CLASSIFIER", "true")
    monkeypatch.setenv("ALPHARAVIS_RETRIEVAL_QUERY_CLASSIFIER_MIN_CHARS", "500")
    monkeypatch.setenv("ALPHARAVIS_LONG_PROMPT_DIRECT_ROUTE_MAX_CHARS", "2000")
    monkeypatch.setattr(agent_graph, "_classify_prompt_for_retrieval", fake_classifier)

    document_prompt = "Material:\n" + ("runtime setting retrieval enabled\n" * 30) + "Welche Regel steht drin?"
    updates = asyncio.run(agent_graph.route_decision_node({"messages": [{"role": "human", "content": document_prompt}]}))

    assert updates["fast_path_route"] == "swarm"
    assert "classified as document" in updates["run_profile"]["route_reason"]


def test_archive_recall_query_condenser_uses_topic_and_context() -> None:
    profile = agent_graph._condense_archive_recall_query_from_text(
        "Wie war das nochmal mit dem Reranker?",
        "Vorher haben wir Qwen3-Reranker GPU batch size und deterministic fallback besprochen.",
    )

    assert profile["strategy"] == "archive_recall_condenser"
    assert "Reranker" in profile["query"] or "reranker" in profile["query"]
    assert "fallback" in profile["query"].lower()


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


def test_active_rag_prefetch_caps_long_noisy_query(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    async def fake_agentic_rag_retrieve(**kwargs):
        calls.update(kwargs)
        return {
            "final_query": kwargs["query"],
            "graph_trace": [],
            "context_packet": {
                "query": kwargs["query"],
                "chunk_count": 1,
                "chunks": [
                    {
                        "rank": 1,
                        "source_key": "doc-1",
                        "retrieval_backend": "alpharavis_pgvector",
                        "relevance_score": 0.9,
                        "chunk_text": "Condensed query detail.",
                    }
                ],
            },
        }

    monkeypatch.setenv("ALPHARAVIS_ENABLE_RETRIEVAL_QUERY_CLASSIFIER", "false")
    monkeypatch.setenv("ALPHARAVIS_RETRIEVAL_DIRECT_QUERY_MAX_CHARS", "200")
    monkeypatch.setenv("ALPHARAVIS_RETRIEVAL_QUERY_MAX_CHARS", "180")
    monkeypatch.setattr(agent_graph, "_router_agentic_rag_retrieve", fake_agentic_rag_retrieve)

    noisy = "Dies ist sehr viel Vorrede. " * 80 + "\nWelche Details stehen im AlphaRavis Dokument ueber Reranking?"
    updates = asyncio.run(
        agent_graph.active_rag_prefetch_node(
            {
                "messages": [{"role": "human", "content": noisy}],
                "thread_id": "thread-1",
                "rag_active": True,
                "active_source_keys": ["doc-1"],
            }
        )
    )

    assert len(calls["query"]) <= 180
    assert "Reranking" in calls["query"]
    assert updates["run_profile"]["active_rag_prefetch_query_strategy"] == "local_condensed"
    assert updates["run_profile"]["active_rag_prefetch_original_query_chars"] > 200


def test_active_rag_prefetch_uses_small_model_classifier_query(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    async def fake_classifier(text: str):
        return {
            "intent": "mixed",
            "retrieval_query": "AlphaRavis reranker GPU embedding context",
            "instruction_lines": [[1, 3]],
            "document_lines": [[10, 40]],
            "question_lines": [[90, 91]],
            "confidence": 0.91,
            "model": "small",
            "elapsed_seconds": 0.1,
        }

    async def fake_agentic_rag_retrieve(**kwargs):
        calls.update(kwargs)
        return {
            "context_packet": {
                "query": kwargs["query"],
                "chunk_count": 1,
                "chunks": [
                    {
                        "rank": 1,
                        "source_key": "doc-1",
                        "retrieval_backend": "alpharavis_pgvector",
                        "relevance_score": 0.9,
                        "chunk_text": "Classifier query detail.",
                    }
                ],
            },
            "graph_trace": [],
        }

    monkeypatch.setenv("ALPHARAVIS_ENABLE_RETRIEVAL_QUERY_CLASSIFIER", "true")
    monkeypatch.setenv("ALPHARAVIS_RETRIEVAL_QUERY_CLASSIFIER_MIN_CHARS", "200")
    monkeypatch.setattr(agent_graph, "_classify_prompt_for_retrieval", fake_classifier)
    monkeypatch.setattr(agent_graph, "_router_agentic_rag_retrieve", fake_agentic_rag_retrieve)

    long_prompt = "Instructions: keep output short.\n" + ("Document: AlphaRavis embeddings and reranking.\n" * 20)
    updates = asyncio.run(
        agent_graph.active_rag_prefetch_node(
            {
                "messages": [{"role": "human", "content": long_prompt}],
                "thread_id": "thread-1",
                "rag_active": True,
                "active_source_keys": ["doc-1"],
            }
        )
    )

    assert calls["query"] == "AlphaRavis reranker GPU embedding context"
    assert updates["run_profile"]["active_rag_prefetch_query_strategy"] == "small_model_classifier"
    assert updates["run_profile"]["active_rag_prefetch_classifier"]["intent"] == "mixed"


def test_active_rag_prefetch_falls_back_when_classifier_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    async def failing_classifier(text: str):
        raise RuntimeError("offline")

    async def fake_agentic_rag_retrieve(**kwargs):
        calls.update(kwargs)
        return {
            "context_packet": {
                "query": kwargs["query"],
                "chunk_count": 1,
                "chunks": [
                    {
                        "rank": 1,
                        "source_key": "doc-1",
                        "retrieval_backend": "alpharavis_pgvector",
                        "relevance_score": 0.9,
                        "chunk_text": "Fallback detail.",
                    }
                ],
            },
            "graph_trace": [],
        }

    monkeypatch.setenv("ALPHARAVIS_ENABLE_RETRIEVAL_QUERY_CLASSIFIER", "true")
    monkeypatch.setenv("ALPHARAVIS_RETRIEVAL_QUERY_CLASSIFIER_MIN_CHARS", "200")
    monkeypatch.setenv("ALPHARAVIS_RETRIEVAL_QUERY_MAX_CHARS", "220")
    monkeypatch.setattr(agent_graph, "_classify_prompt_for_retrieval", failing_classifier)
    monkeypatch.setattr(agent_graph, "_router_agentic_rag_retrieve", fake_agentic_rag_retrieve)

    long_prompt = "Vorrede. " * 80 + "\nWelche Details stehen im Dokument?"
    updates = asyncio.run(
        agent_graph.active_rag_prefetch_node(
            {
                "messages": [{"role": "human", "content": long_prompt}],
                "thread_id": "thread-1",
                "rag_active": True,
                "active_source_keys": ["doc-1"],
            }
        )
    )

    assert calls["query"]
    assert updates["run_profile"]["active_rag_prefetch_query_warning"].startswith("classifier_failed")


def test_active_rag_prefetch_auto_queries_archives_on_intent(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    async def fake_archive_classifier(messages):
        return {
            "archive_recall": True,
            "query": "Qwen3 reranker deterministic fallback",
            "confidence": 0.92,
            "strategy": "small_model_archive_intent",
        }

    async def fake_agentic_rag_retrieve(**kwargs):
        calls.update(kwargs)
        return {
            "final_query": kwargs["query"],
            "context_packet": {
                "query": kwargs["query"],
                "chunk_count": 1,
                "chunks": [
                    {
                        "rank": 1,
                        "source_key": "archive-1",
                        "retrieval_backend": "alpharavis_pgvector",
                        "relevance_score": 0.9,
                        "chunk_text": "Qwen3 reranker fallback detail from archive.",
                    }
                ],
            },
            "graph_trace": [],
        }

    monkeypatch.setattr(agent_graph, "_archive_auto_intent_profile_for_messages", fake_archive_classifier)
    monkeypatch.setattr(agent_graph, "_router_agentic_rag_retrieve", fake_agentic_rag_retrieve)

    updates = asyncio.run(
        agent_graph.active_rag_prefetch_node(
            {
                "messages": [
                    {"role": "human", "content": "Wie war das nochmal mit dem Qwen3 Reranker fallback?"}
                ],
                "thread_id": "thread-1",
                "rag_active": False,
                "archive_rag_mode": "auto_on_intent",
                "archived_context_keys": ["archive-1"],
            }
        )
    )

    assert calls["source_type"] == "archive"
    assert calls["source_keys"] == ["archive-1"]
    assert calls["query"] == "Qwen3 reranker deterministic fallback"
    assert updates["run_profile"]["active_rag_prefetch_status"] == "injected"
    assert updates["run_profile"]["active_rag_prefetch_archive_auto_on_intent"] is True
    assert updates["run_profile"]["active_rag_prefetch_query_strategy"] == "small_model_archive_intent"
    assert "Qwen3 reranker fallback detail" in updates["messages"][0].content


def test_active_rag_prefetch_archive_auto_skips_when_2b_says_no(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_archive_classifier(messages):
        return {
            "archive_recall": False,
            "query": "new unrelated task",
            "confidence": 0.88,
            "strategy": "small_model_archive_intent",
            "reason": "new task",
        }

    async def fake_agentic_rag_retrieve(**kwargs):
        raise AssertionError("archive retrieval must not run when 2B rejects recall")

    monkeypatch.setattr(agent_graph, "_archive_auto_intent_profile_for_messages", fake_archive_classifier)
    monkeypatch.setattr(agent_graph, "_router_agentic_rag_retrieve", fake_agentic_rag_retrieve)

    updates = asyncio.run(
        agent_graph.active_rag_prefetch_node(
            {
                "messages": [{"role": "human", "content": "Mach bitte eine neue Aufgabe zu Reranking."}],
                "thread_id": "thread-1",
                "archive_rag_mode": "auto_on_intent",
                "archived_context_keys": ["archive-1"],
            }
        )
    )

    assert updates["run_profile"]["active_rag_prefetch_status"] == "archive_auto_no_intent"
    assert updates["run_profile"]["active_rag_prefetch_classifier"]["archive_recall"] is False


def test_archive_auto_intent_profile_falls_back_when_2b_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    async def failing_classifier(query, context):
        raise RuntimeError("offline")

    monkeypatch.setenv("ALPHARAVIS_ENABLE_ARCHIVE_AUTO_INTENT_CLASSIFIER", "true")
    monkeypatch.setattr(agent_graph, "_classify_archive_recall_with_small_model", failing_classifier)

    profile = asyncio.run(
        agent_graph._archive_auto_intent_profile_for_messages(
            [{"role": "human", "content": "Wie war das nochmal mit dem Reranker fallback?"}]
        )
    )

    assert profile["archive_recall"] is True
    assert profile["strategy"] == "archive_recall_condenser_fallback"
    assert profile["classifier_warning"].startswith("classifier_failed")
    assert "Reranker" in profile["query"] or "reranker" in profile["query"]


def test_active_rag_prefetch_auto_checks_archives_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    async def fake_archive_classifier(messages):
        return {
            "archive_recall": True,
            "query": "default archive recall query",
            "confidence": 0.91,
            "strategy": "small_model_archive_intent",
        }

    async def fake_agentic_rag_retrieve(**kwargs):
        calls.update(kwargs)
        return {
            "final_query": kwargs["query"],
            "context_packet": {
                "query": kwargs["query"],
                "chunk_count": 1,
                "chunks": [
                    {
                        "rank": 1,
                        "source_key": "archive-1",
                        "retrieval_backend": "alpharavis_pgvector",
                        "relevance_score": 0.9,
                        "chunk_text": "Default archive context.",
                    }
                ],
            },
            "graph_trace": [],
        }

    monkeypatch.setattr(agent_graph, "_archive_auto_intent_profile_for_messages", fake_archive_classifier)
    monkeypatch.setattr(agent_graph, "_router_agentic_rag_retrieve", fake_agentic_rag_retrieve)

    updates = asyncio.run(
        agent_graph.active_rag_prefetch_node(
            {
                "messages": [{"role": "human", "content": "Wie war das nochmal mit dem Reranker?"}],
                "thread_id": "thread-1",
                "archive_rag_mode": "tool_only",
                "archived_context_keys": ["archive-1"],
            }
        )
    )

    assert calls["source_type"] == "archive"
    assert calls["query"] == "default archive recall query"
    assert updates["run_profile"]["active_rag_prefetch_status"] == "injected"
    assert updates["run_profile"]["active_rag_prefetch_archive_auto_on_intent"] is True


def test_active_rag_prefetch_archive_auto_default_skips_current_source_task_when_2b_says_no(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_archive_classifier(messages):
        return {
            "archive_recall": False,
            "query": "use current video",
            "confidence": 0.93,
            "strategy": "small_model_archive_intent",
            "reason": "current explicit media task",
        }

    async def fake_agentic_rag_retrieve(**kwargs):
        raise AssertionError("archive retrieval must not run for current media/source tasks")

    monkeypatch.setattr(agent_graph, "_archive_auto_intent_profile_for_messages", fake_archive_classifier)
    monkeypatch.setattr(agent_graph, "_router_agentic_rag_retrieve", fake_agentic_rag_retrieve)

    updates = asyncio.run(
        agent_graph.active_rag_prefetch_node(
            {
                "messages": [{"role": "human", "content": "Benutze dieses Video und mache daraus einen Pixelle Prompt."}],
                "thread_id": "thread-1",
                "archive_rag_mode": "tool_only",
                "archived_context_keys": ["archive-1"],
            }
        )
    )

    assert updates["run_profile"]["active_rag_prefetch_status"] == "archive_auto_no_intent"
    assert updates["run_profile"]["active_rag_prefetch_classifier"]["archive_recall"] is False


def test_active_rag_prefetch_manual_archive_mode_stays_passive(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_agentic_rag_retrieve(**kwargs):
        raise AssertionError("archive retrieval must stay passive in manual mode")

    monkeypatch.setattr(agent_graph, "_router_agentic_rag_retrieve", fake_agentic_rag_retrieve)

    updates = asyncio.run(
        agent_graph.active_rag_prefetch_node(
            {
                "messages": [{"role": "human", "content": "Wie war das nochmal mit dem Reranker?"}],
                "thread_id": "thread-1",
                "archive_rag_mode": "manual",
                "archived_context_keys": ["archive-1"],
            }
        )
    )

    assert updates["run_profile"]["active_rag_prefetch_status"] == "disabled_or_inactive"


def test_rag_pin_tools_persist_thread_active_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStore:
        def __init__(self):
            self.records = {}

        def put(self, namespace, key, value):
            self.records[(tuple(namespace), key)] = value

        def get(self, namespace, key):
            return self.records.get((tuple(namespace), key))

    store = FakeStore()
    monkeypatch.setattr(agent_graph, "get_store", lambda: store)
    monkeypatch.setattr(agent_graph, "_thread_id_from_config", lambda: "thread-1")

    pinned = asyncio.run(agent_graph.pin_active_rag_sources(["doc-1"], ["file-1"]))
    inspected = asyncio.run(agent_graph.inspect_active_rag_sources())
    unpinned = asyncio.run(agent_graph.unpin_active_rag_sources(["doc-1"], ["file-1"]))

    assert '"status": "pinned"' in pinned
    assert '"active_source_keys": [\n    "doc-1"\n  ]' in inspected
    assert '"active_rag_file_ids": [\n    "file-1"\n  ]' in inspected
    assert '"rag_active": false' in unpinned


def test_active_rag_prefetch_uses_pinned_sources_without_state_rag_active(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStore:
        def get(self, namespace, key):
            return {
                "rag_active": True,
                "active_source_keys": ["pinned-doc"],
                "active_rag_file_ids": [],
                "archive_rag_mode": "tool_only",
            }

    calls: dict[str, object] = {}

    async def fake_agentic_rag_retrieve(**kwargs):
        calls.update(kwargs)
        return {
            "context_packet": {
                "query": kwargs["query"],
                "chunk_count": 1,
                "chunks": [
                    {
                        "rank": 1,
                        "source_key": "pinned-doc",
                        "retrieval_backend": "alpharavis_pgvector",
                        "relevance_score": 0.8,
                        "chunk_text": "Pinned source detail.",
                    }
                ],
            },
            "graph_trace": [],
        }

    monkeypatch.setattr(agent_graph, "get_store", lambda: FakeStore())
    monkeypatch.setattr(agent_graph, "_router_agentic_rag_retrieve", fake_agentic_rag_retrieve)

    updates = asyncio.run(
        agent_graph.active_rag_prefetch_node(
            {
                "messages": [{"role": "human", "content": "Welche Details stehen im gepinnten Dokument?"}],
                "thread_id": "thread-1",
                "rag_active": False,
            }
        )
    )

    assert calls["source_keys"] == ["pinned-doc"]
    assert updates["run_profile"]["active_rag_prefetch_status"] == "injected"
    assert "Pinned source detail." in updates["messages"][0].content


def test_read_source_chunks_tool_uses_current_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    async def fake_read_source_chunks(**kwargs):
        calls.update(kwargs)
        return {"source_key": kwargs["source_key"], "chunks": [{"chunk_text": "bounded"}]}

    monkeypatch.setattr(agent_graph, "_pgvector_read_source_chunks", fake_read_source_chunks)
    monkeypatch.setattr(agent_graph, "_thread_id_from_config", lambda: "thread-1")

    result = asyncio.run(agent_graph.read_source_chunks("source-1", max_chunks=2, max_chars=500))

    assert calls["source_key"] == "source-1"
    assert calls["thread_id"] == "thread-1"
    assert calls["max_chunks"] == 2
    assert calls["max_chars"] == 500
    assert '"chunk_text": "bounded"' in result


def test_read_raw_source_reads_bounded_store_slice(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStore:
        def __init__(self):
            self.values = {
                (agent_graph._thread_source_record_ns("thread-1"), "source-1"): {
                    "source_key": "source-1",
                    "source_type": "large_paste",
                    "title": "Raw Source",
                    "content": "alpha\nneedle starts here\n" + ("body " * 200),
                    "thread_id": "thread-1",
                    "thread_key": "thread-1",
                    "metadata": {"origin": "test"},
                }
            }

        def get(self, namespace, key):
            return self.values.get((namespace, key))

    monkeypatch.setattr(agent_graph, "get_store", lambda: FakeStore())
    monkeypatch.setattr(agent_graph, "_thread_id_from_config", lambda: "thread-1")

    result = asyncio.run(agent_graph.read_raw_source("source-1", source_type="large_paste", search="needle", max_chars=80))

    assert '"found": true' in result
    assert "needle starts here" in result
    assert '"total_chars"' in result
    assert '"truncated_after": true' in result


def test_store_raw_source_record_writes_thread_and_index(monkeypatch: pytest.MonkeyPatch) -> None:
    writes: list[tuple[tuple[str, ...], str, dict[str, object]]] = []

    async def fake_put(store, namespace, key, value):
        writes.append((namespace, key, value))

    monkeypatch.setattr(agent_graph, "get_store", lambda: object())
    monkeypatch.setattr(agent_graph, "_maybe_put", fake_put)

    result = asyncio.run(
        agent_graph._store_raw_source_record(
            source_type="large_paste",
            source_key="source-1",
            title="Source",
            content="raw content",
            indexed_content="indexed content",
            thread_id="thread-1",
            thread_key="thread-key",
            metadata={"origin": "test"},
        )
    )

    assert result["stored"] is True
    assert writes[0][0] == agent_graph._thread_source_record_ns("thread-1")
    assert writes[0][1] == "source-1"
    assert writes[0][2]["content"] == "raw content"
    assert writes[0][2]["indexed_content"] == "indexed content"
    assert writes[1][0] == agent_graph.SOURCE_RECORD_INDEX_NS
    assert "content" not in writes[1][2]


def test_read_archive_record_returns_bounded_slice(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStore:
        def get(self, namespace, key):
            if namespace == agent_graph._thread_archive_ns("thread-1") and key == "archive-1":
                return {
                    "archive_key": "archive-1",
                    "title": "Archive",
                    "summary": "summary",
                    "content": "prefix " + ("archive raw " * 200),
                    "thread_id": "thread-1",
                    "thread_key": "thread-1",
                    "metadata": {},
                }
            return None

    monkeypatch.setattr(agent_graph, "get_store", lambda: FakeStore())

    result = asyncio.run(agent_graph.read_archive_record("archive-1", thread_id="thread-1", start=7, max_chars=240))

    assert '"found": true' in result
    assert '"start": 7' in result
    assert '"max_chars": 240' in result
    assert '"truncated_after": true' in result


def test_ingest_document_file_loads_and_pins_allowed_file(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "doc.md"
    source.write_text("# ignored by fake loader", encoding="utf-8")
    calls: dict[str, object] = {}
    writes: list[tuple[tuple[str, ...], str, dict[str, object]]] = []

    def fake_load_document_file(path):
        assert Path(path) == source.resolve()
        return {
            "ok": True,
            "path": str(path),
            "title": "Doc Title",
            "text": "Grounded loaded document.",
            "text_chars": 25,
            "metadata": {"filename": "doc.md", "extension": ".md"},
            "error": "",
        }

    async def fake_ingest_source(**kwargs):
        calls.update(kwargs)
        return {
            "index_status": "indexed",
            "source_key": kwargs["source_key"],
            "rag_file_id": kwargs["source_key"],
            "rag_active": True,
            "active_source_keys": [kwargs["source_key"]],
            "active_rag_file_ids": [],
            "indexed_backends": ["alpharavis_pgvector"],
            "queued_backends": [],
            "warnings": [],
            "errors": [],
        }

    async def fake_put(store, namespace, key, value):
        writes.append((namespace, key, value))

    monkeypatch.setenv("ALPHARAVIS_DOCUMENT_INGEST_ROOT", str(tmp_path))
    monkeypatch.setattr(agent_graph, "_document_load_file", fake_load_document_file)
    monkeypatch.setattr(agent_graph, "_router_ingest_source", fake_ingest_source)
    monkeypatch.setattr(agent_graph, "_thread_id_from_config", lambda: "thread-1")
    monkeypatch.setattr(agent_graph, "get_store", lambda: object())
    monkeypatch.setattr(agent_graph, "_maybe_put", fake_put)

    result = asyncio.run(agent_graph.ingest_document_file(str(source), source_key="doc-1"))

    assert calls["source_type"] == "uploaded_document"
    assert calls["source_key"] == "doc-1"
    assert calls["content"] == "Grounded loaded document."
    assert calls["metadata"]["origin"] == "agent_document_file_ingest"
    assert writes[-1][2]["active_source_keys"] == ["doc-1"]
    assert '"ok": true' in result
    assert '"source_key": "doc-1"' in result


def test_ingest_document_file_blocks_outside_ingest_root(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    source = outside / "doc.md"
    source.write_text("outside", encoding="utf-8")

    monkeypatch.setenv("ALPHARAVIS_DOCUMENT_INGEST_ROOT", str(allowed))

    result = asyncio.run(agent_graph.ingest_document_file(str(source)))

    assert '"index_status": "blocked"' in result
    assert "outside the allowed root" in result


def test_pending_document_uploads_ingest_into_active_rag(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "upload.md"
    source.write_text("ignored", encoding="utf-8")
    calls: dict[str, object] = {}

    def fake_load_document_file(path):
        return {
            "ok": True,
            "path": str(path),
            "title": "Upload",
            "text": "Uploaded document content.",
            "text_chars": 26,
            "metadata": {"filename": "upload.md"},
        }

    async def fake_ingest_source(**kwargs):
        calls.update(kwargs)
        await kwargs["pgvector_index"](
            source_type=kwargs["source_type"],
            source_key=kwargs["source_key"],
            title=kwargs["title"],
            content=kwargs["content"],
            thread_id=kwargs["thread_id"],
            thread_key=kwargs["thread_key"],
            scope=kwargs["scope"],
            metadata=kwargs["metadata"],
        )
        return {
            "index_status": "indexed",
            "source_key": kwargs["source_key"],
            "rag_file_id": kwargs["source_key"],
            "rag_active": True,
            "active_source_keys": [kwargs["source_key"]],
            "active_rag_file_ids": [],
            "indexed_backends": ["alpharavis_pgvector"],
            "queued_backends": [],
        }

    async def fake_index(**kwargs):
        progress_callback = kwargs.get("progress_callback")
        if progress_callback:
            progress_callback({"event": "large_ingest.chunk_indexed", "chunk_number": 1, "chunk_count": 1})
        return "uploaded_document:librechat:file_doc:1"

    monkeypatch.setenv("ALPHARAVIS_DOCUMENT_INGEST_ROOT", str(tmp_path))
    monkeypatch.setattr(agent_graph, "_document_load_file", fake_load_document_file)
    monkeypatch.setattr(agent_graph, "_router_ingest_source", fake_ingest_source)
    monkeypatch.setattr(agent_graph, "_maybe_index_vector_memory", fake_index)

    ingests, rag_update = asyncio.run(
        agent_graph._ingest_pending_document_uploads(
            {
                "thread_id": "thread-1",
                "thread_key": "thread-key",
                "pending_document_ingests": [
                    {
                        "path": str(source),
                        "source_key": "librechat:file_doc",
                        "title": "Upload",
                    }
                ],
            }
        )
    )

    assert calls["source_key"] == "librechat:file_doc"
    assert calls["content"] == "Uploaded document content."
    assert rag_update["rag_active"] is True
    assert rag_update["active_source_keys"] == ["librechat:file_doc"]
    assert ingests[0]["events"][1]["event"] == "document_ingest.chunk_indexed"
    assert ingests[0]["events"][-1]["event"] == "document_ingest.completed"


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
