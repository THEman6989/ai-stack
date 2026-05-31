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
    result_keys = [item["source_key"] for item in payload["results"]]
    assert "archive-3" in result_keys
    assert "archive:archive-3" in result_keys
    assert len(result_keys) == 2


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


def test_vector_memory_source_declares_pgvector_search_head_contract() -> None:
    source = (ROOT / "langgraph-app" / "vector_memory.py").read_text(encoding="utf-8")

    assert "source_id TEXT NOT NULL DEFAULT ''" in source
    assert "version TEXT NOT NULL DEFAULT 'v1'" in source
    assert "raw_ref JSONB NOT NULL DEFAULT '{{}}'::jsonb" in source
    assert '("raw_ref", "JSONB NOT NULL DEFAULT \'{}\'::jsonb")' in source
    assert "source_id, version, raw_ref" in source
    assert "SELECT\n            id, scope, thread_id, thread_key, source_type, source_key," in source
    assert "source_id, version, raw_ref" in source[source.index("def _search_sync") : source.index("def _search_vision_sync")]


def test_record_curated_memory_update_reindexes_pgvector() -> None:
    source = (ROOT / "langgraph-app" / "agent_graph.py").read_text(encoding="utf-8")
    _update_path = source[source.index('action == "update"') : source.index("CREATE (default)")]
    assert "_maybe_index_vector_memory" in _update_path, (
        "record_curated_memory UPDATE must re-index PGVector — "
        "should call _maybe_index_vector_memory after updating Mongo"
    )


def test_activate_skill_candidate_reindexes_pgvector() -> None:
    source = (ROOT / "langgraph-app" / "agent_graph.py").read_text(encoding="utf-8")
    _activate_path = source[
        source.index("def activate_skill_candidate") : source.index("def deactivate_skill")
    ]
    assert "_maybe_index_vector_memory" in _activate_path, (
        "activate_skill_candidate must re-index PGVector — "
        "should call _maybe_index_vector_memory after updating status to active"
    )


def test_deactivate_skill_reindexes_pgvector() -> None:
    source = (ROOT / "langgraph-app" / "agent_graph.py").read_text(encoding="utf-8")
    _deactivate_path = source[
        source.index("def deactivate_skill") : source.index("async def _load_configured_mcp_tools")
    ]
    assert "_maybe_index_vector_memory" in _deactivate_path, (
        "deactivate_skill must re-index PGVector — "
        "should re-index or delete PGVector row after status change"
    )


def test_record_curated_memory_delete_syncs_pgvector() -> None:
    source = (ROOT / "langgraph-app" / "agent_graph.py").read_text(encoding="utf-8")
    _delete_path = source[
        source.index('action == "delete"') : source.index('action == "update"')
    ]
    assert "_pgvector_delete_memory_record" in _delete_path, (
        "record_curated_memory DELETE must sync PGVector — "
        "should call _pgvector_delete_memory_record(source_key=memory_id) after Mongo delete"
    )
    assert "deleted_pgvector" in _delete_path, (
        "record_curated_memory DELETE must track pgvector deletion outcome — "
        "deleted_pgvector flag is missing"
    )
    assert "pgvector cleanup" in _delete_path, (
        "record_curated_memory DELETE must report pgvector cleanup status in warning — "
        "pgvector cleanup skipped/failed message is missing"
    )


def test_source_metadata_summary_includes_source_digest() -> None:
    source = (ROOT / "langgraph-app" / "agent_graph.py").read_text(encoding="utf-8")
    _summary_fn = source[
        source.index("def _source_metadata_summary") : source.index("def _classify_prompt_for_retrieval")
    ]
    assert "source_digest" in _summary_fn, (
        "_source_metadata_summary must include source_digest — "
        "so PGVector version/source_digest metadata flows through all ingest paths"
    )
    assert 'hashlib.sha256' in _summary_fn, (
        "_source_metadata_summary must compute source_digest via sha256 — "
        "content hash is required for dedup and version tracking"
    )


def test_read_source_chunks_docstring_states_pgvector_chunks_contain_text() -> None:
    source = (ROOT / "langgraph-app" / "agent_graph.py").read_text(encoding="utf-8")
    _docstring = source[
        source.index("Read bounded ordered chunks") : source.index("if _pgvector_read_source_chunks is None")
    ]
    assert "chunk_text" in _docstring, (
        "read_source_chunks docstring must state PGVector chunks contain chunk_text — "
        "agents should not fetch raw source for chunk body"
    )
    assert "raw source lookup" in _docstring.lower(), (
        "read_source_chunks docstring must direct agents to raw source only when full context needed"
    )


def test_repo_skills_has_skill_entry_to_index_document() -> None:
    source = (ROOT / "langgraph-app" / "repo_skills.py").read_text(encoding="utf-8")
    assert "def skill_entry_to_index_document" in source, (
        "repo_skills.py must have skill_entry_to_index_document — "
        "converts scanned entries to PGVector index payloads"
    )


def test_reload_repo_ai_skills_has_vector_index_feature_flag() -> None:
    source = (ROOT / "langgraph-app" / "agent_graph.py").read_text(encoding="utf-8")
    _reload_fn = source[
        source.index("def reload_repo_ai_skills") : source.index("def read_repo_ai_skill")
    ]
    assert "ALPHARAVIS_ENABLE_REPO_SKILL_VECTOR_INDEX" in _reload_fn, (
        "reload_repo_ai_skills must gate PGVector indexing behind "
        "ALPHARAVIS_ENABLE_REPO_SKILL_VECTOR_INDEX feature flag"
    )
    assert "_repo_skill_to_index_document" in _reload_fn, (
        "reload_repo_ai_skills must call _repo_skill_to_index_document "
        "to build PGVector payloads from scanned entries"
    )


def test_execute_local_command_has_tool_run_indexing_flag() -> None:
    source = (ROOT / "langgraph-app" / "agent_graph.py").read_text(encoding="utf-8")
    _cmd_fn = source[
        source.index("def execute_local_command") : source.index("def storage_manager_status")
    ]
    assert "ALPHARAVIS_ENABLE_TOOL_EVENT_VECTOR_INDEX" not in _cmd_fn, (
        "execute_local_command delegates to event_indexing module — "
        "flag is checked in maybe_index_tool_run, not duplicated here"
    )
    assert "_index_tool_call" in _cmd_fn, (
        "execute_local_command must call _index_tool_call "
        "to schedule optional tool-run PGVector indexing"
    )


