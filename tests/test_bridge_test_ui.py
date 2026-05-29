from __future__ import annotations

import asyncio
import importlib.util
from importlib.machinery import ModuleSpec
import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

if importlib.util.find_spec("fastapi") is None:
    fastapi_stub = types.ModuleType("fastapi")
    fastapi_stub.__spec__ = ModuleSpec("fastapi", loader=None)

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str = "") -> None:
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

    class Request:
        headers: dict[str, str] = {}

    class FastAPI:
        openapi_version = "3.1.0"

        def __init__(self, *args, **kwargs) -> None:
            self.user_middleware = []

        def get(self, *args, **kwargs):
            return lambda fn: fn

        def post(self, *args, **kwargs):
            return lambda fn: fn

        def delete(self, *args, **kwargs):
            return lambda fn: fn

        def middleware(self, *args, **kwargs):
            return lambda fn: fn

        def mount(self, *args, **kwargs) -> None:
            return None

        def add_middleware(self, middleware_class, *args, **kwargs) -> None:
            self.user_middleware.append(types.SimpleNamespace(cls=middleware_class, options=kwargs))

    fastapi_stub.FastAPI = FastAPI
    fastapi_stub.HTTPException = HTTPException
    fastapi_stub.Request = Request
    sys.modules["fastapi"] = fastapi_stub

    responses_stub = types.ModuleType("fastapi.responses")
    responses_stub.__spec__ = ModuleSpec("fastapi.responses", loader=None)

    class HTMLResponse(str):
        def __new__(cls, content: str = "", *args, **kwargs):
            return str.__new__(cls, content)

    class RedirectResponse(str):
        status_code: int

        def __new__(cls, url: str = "", *args, **kwargs):
            obj = str.__new__(cls, url)
            obj.status_code = kwargs.get("status_code", 307)
            return obj

    class JSONResponse:
        def __init__(self, content=None, status_code: int = 200, *args, **kwargs) -> None:
            self.body = json.dumps(content or {}).encode("utf-8")
            self.status_code = status_code

    class StreamingResponse:
        def __init__(self, content, *args, **kwargs) -> None:
            self.content = content
            self.media_type = kwargs.get("media_type", "")
            self.headers = kwargs.get("headers", {})

    responses_stub.HTMLResponse = HTMLResponse
    setattr(responses_stub, "RedirectResponse", RedirectResponse)
    responses_stub.JSONResponse = JSONResponse
    responses_stub.StreamingResponse = StreamingResponse
    sys.modules["fastapi.responses"] = responses_stub

    staticfiles_stub = types.ModuleType("fastapi.staticfiles")
    staticfiles_stub.__spec__ = ModuleSpec("fastapi.staticfiles", loader=None)

    class StaticFiles:
        def __init__(self, *args, **kwargs) -> None:
            pass

    staticfiles_stub.StaticFiles = StaticFiles
    sys.modules["fastapi.staticfiles"] = staticfiles_stub

    middleware_stub = types.ModuleType("fastapi.middleware")
    middleware_stub.__spec__ = ModuleSpec("fastapi.middleware", loader=None)
    cors_stub = types.ModuleType("fastapi.middleware.cors")
    cors_stub.__spec__ = ModuleSpec("fastapi.middleware.cors", loader=None)

    class CORSMiddleware:
        def __init__(self, *args, **kwargs) -> None:
            pass

    setattr(cors_stub, "CORSMiddleware", CORSMiddleware)
    sys.modules["fastapi.middleware"] = middleware_stub
    sys.modules["fastapi.middleware.cors"] = cors_stub

if importlib.util.find_spec("pydantic") is None:
    pydantic_stub = types.ModuleType("pydantic")
    pydantic_stub.__spec__ = ModuleSpec("pydantic", loader=None)

    class BaseModel:
        def __init__(self, **kwargs) -> None:
            for name, value in self.__class__.__dict__.items():
                if not name.startswith("_") and name not in kwargs:
                    setattr(self, name, value)
            for name, value in kwargs.items():
                setattr(self, name, value)

    def Field(default=None, *, default_factory=None, **kwargs):
        return default_factory() if default_factory is not None else default

    pydantic_stub.BaseModel = BaseModel
    pydantic_stub.Field = Field
    sys.modules["pydantic"] = pydantic_stub

import test_ui_server  # noqa: E402


def test_stream_payload_uses_responses_sse() -> None:
    request = test_ui_server.ChatRequest(
        message="Hi",
        messages=[{"role": "assistant", "content": "Hallo"}],
        protocol="responses",
    )

    url, payload = test_ui_server._bridge_request_payload(
        request,
        text="Welche Tools hast du?",
        protocol="responses",
        session_id="session_test",
        trace_id="trace_test",
        stream=True,
    )

    assert url.endswith("/responses")
    assert payload["stream"] is True
    assert payload["input"] == [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hallo"}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Welche Tools hast du?"}],
        },
    ]
    assert payload["metadata"]["trace_id"] == "trace_test"


def test_stream_payload_uses_chat_sse() -> None:
    request = test_ui_server.ChatRequest(
        message="Hi",
        messages=[{"role": "assistant", "content": "Hallo"}],
        protocol="chat",
    )

    url, payload = test_ui_server._bridge_request_payload(
        request,
        text="Weiter",
        protocol="chat",
        session_id="session_test",
        trace_id="trace_test",
        stream=True,
    )

    assert url.endswith("/chat/completions")
    assert payload["stream"] is True
    assert payload["messages"] == [
        {"role": "assistant", "content": "Hallo"},
        {"role": "user", "content": "Weiter"},
    ]


def test_test_ui_event_is_valid_sse_json() -> None:
    raw = test_ui_server._test_ui_event("test_ui.started", {"trace_id": "trace_test"})

    assert raw.startswith("event: test_ui.started\n")
    data_line = next(line for line in raw.splitlines() if line.startswith("data: "))
    assert json.loads(data_line.removeprefix("data: ")) == {"trace_id": "trace_test"}


def test_html_renders_collapsed_reasoning_panel_from_stream() -> None:
    assert "response.reasoning.delta" in test_ui_server.HTML
    assert "reasoning-details" in test_ui_server.HTML
    assert "live-panels" in test_ui_server.HTML
    assert "live-status" in test_ui_server.HTML
    assert "live-reasoning" in test_ui_server.HTML
    assert "live-plan" in test_ui_server.HTML
    assert "live-context" in test_ui_server.HTML
    assert "Kontext" in test_ui_server.HTML
    assert "data-panel-toggle" in test_ui_server.HTML
    assert "panel.classList.toggle('expanded')" in test_ui_server.HTML
    assert "function renderLivePanels(msg)" in test_ui_server.HTML
    assert "context_compaction" in test_ui_server.HTML
    assert "context_hard" in test_ui_server.HTML
    assert "[COMPACT]" in test_ui_server.HTML
    assert "[HARD]" in test_ui_server.HTML
    assert "summary.textContent = 'Reasoning'" in test_ui_server.HTML
    assert "reasoningStatus" in test_ui_server.HTML
    assert "internalPlan" in test_ui_server.HTML
    assert "Interner Plan" in test_ui_server.HTML
    assert "Modell-Reasoning" in test_ui_server.HTML
    assert "function reasoningKind(data, text, msg)" in test_ui_server.HTML
    assert "reasoningOpen: false" in test_ui_server.HTML


def test_test_ui_has_observer_navigation_button() -> None:
    assert 'href="/observer"' in test_ui_server.HTML
    assert "Observer" in test_ui_server.HTML
    assert "nav-button" in test_ui_server.HTML


def test_html_sse_parser_escapes_regex_newlines() -> None:
    assert "block.split(/\\r?\\n/)" in test_ui_server.HTML
    assert "dataLines.join('\\n')" in test_ui_server.HTML
    assert "buffer.split(/\\r?\\n\\r?\\n/)" in test_ui_server.HTML
    assert "block.split(/\r?\n/)" not in test_ui_server.HTML


def test_clear_resets_backend_session_id() -> None:
    assert "let sessionId = storedSessionId();" in test_ui_server.HTML
    assert "function resetSessionId()" in test_ui_server.HTML
    assert "sessionId = resetSessionId();" in test_ui_server.HTML
    assert "neue Session bereit" in test_ui_server.HTML


def test_html_shows_route_badge_for_fast_or_agent_path() -> None:
    assert "route-badge" in test_ui_server.HTML
    assert "function routeLabel(routeName)" in test_ui_server.HTML
    assert "Fast Path" in test_ui_server.HTML
    assert "Agent Path" in test_ui_server.HTML
    assert "routeFromEvent" in test_ui_server.HTML
    assert "fast_chat" in test_ui_server.HTML
    assert "swarm" in test_ui_server.HTML


def test_html_compacts_text_delta_trace_rows_by_default() -> None:
    assert "trace-delta-details" in test_ui_server.HTML
    assert "Delta-Details" in test_ui_server.HTML
    assert "function summarizeTraceSteps(steps)" in test_ui_server.HTML
    assert "${group.name || 'Delta empfangen'} (${group.count} Deltas" in test_ui_server.HTML
    assert "Delta-Zeilen zusammengefasst" in test_ui_server.HTML
    assert "traceDeltaDetails.checked ? rawSteps : summarizeTraceSteps(rawSteps)" in test_ui_server.HTML


def test_observer_page_is_full_table_view() -> None:
    assert "AlphaRavis Bridge Observer" in test_ui_server.OBSERVER_HTML
    assert "<table>" in test_ui_server.OBSERVER_HTML
    assert "Senden" in test_ui_server.OBSERVER_HTML
    assert "Empfang" in test_ui_server.OBSERVER_HTML
    assert "Kompression" in test_ui_server.OBSERVER_HTML
    assert "Nur Kontext" in test_ui_server.OBSERVER_HTML
    assert "Vollansicht" in test_ui_server.OBSERVER_HTML
    assert "model_context_messages" in test_ui_server.OBSERVER_HTML
    assert "State Msg" in test_ui_server.OBSERVER_HTML
    assert "Context Budget" in test_ui_server.OBSERVER_HTML
    assert "budgetOf(record)" in test_ui_server.OBSERVER_HTML
    assert "context_budget" in test_ui_server.OBSERVER_HTML
    assert "Source Ingest" in test_ui_server.OBSERVER_HTML
    assert "Big Message / Source Ingest" in test_ui_server.OBSERVER_HTML
    assert "source_ingests" in test_ui_server.OBSERVER_HTML
    assert "function renderSourceIngests()" in test_ui_server.OBSERVER_HTML
    assert "Marker aktiv" in test_ui_server.OBSERVER_HTML
    assert "Embedding Queue" in test_ui_server.OBSERVER_HTML
    assert "/api/embedding-queue/status" in test_ui_server.OBSERVER_HTML
    assert "function renderEmbeddingQueue" in test_ui_server.OBSERVER_HTML
    assert "provider_reported_context_limit" in test_ui_server.OBSERVER_HTML
    assert "final_budget_rescue_budget_met" in test_ui_server.OBSERVER_HTML
    assert "Restbudget" in test_ui_server.OBSERVER_HTML
    assert "Provider Ctx" in test_ui_server.OBSERVER_HTML
    assert "Compression Shrinking" in test_ui_server.OBSERVER_HTML
    assert "function renderShrinking()" in test_ui_server.OBSERVER_HTML
    assert "function shrinkSummary(record)" in test_ui_server.OBSERVER_HTML
    assert "Workflow / Tool Events" in test_ui_server.OBSERVER_HTML
    assert "workflow_event_preview" in test_ui_server.OBSERVER_HTML
    assert "summary_chunking_used" in test_ui_server.OBSERVER_HTML
    assert "summary_prompt_pruned" in test_ui_server.OBSERVER_HTML
    assert "Chunk Count" in test_ui_server.OBSERVER_HTML
    assert "Chunk Payload" in test_ui_server.OBSERVER_HTML
    assert "Prompt Overhead" in test_ui_server.OBSERVER_HTML
    assert "Synth Payload" in test_ui_server.OBSERVER_HTML
    assert "Compact Focus" in test_ui_server.OBSERVER_HTML
    assert "Compress Limit" in test_ui_server.OBSERVER_HTML
    assert "Summary Context" in test_ui_server.OBSERVER_HTML
    assert "Head/Middle/Tail" in test_ui_server.OBSERVER_HTML
    assert "receive.compression" in test_ui_server.OBSERVER_HTML
    assert "compressionTab" in test_ui_server.OBSERVER_HTML
    assert "Chunking Lab" in test_ui_server.OBSERVER_HTML
    assert "runChunking" in test_ui_server.OBSERVER_HTML
    assert "chunkCompactInstructions" in test_ui_server.OBSERVER_HTML
    assert "compact_instructions" in test_ui_server.OBSERVER_HTML
    assert "/api/chunking/runs" in test_ui_server.OBSERVER_HTML
    assert "summary_chunk_omitted_chars_zero" in test_ui_server.OBSERVER_HTML
    assert "Tool-Spuren" in test_ui_server.OBSERVER_HTML
    assert "Prompt-Last" in test_ui_server.OBSERVER_HTML
    assert "chunkSummaryMode" in test_ui_server.OBSERVER_HTML
    assert "Real LLM" in test_ui_server.OBSERVER_HTML
    assert "Summary Mode" in test_ui_server.OBSERVER_HTML
    assert "chunkingCompare" in test_ui_server.OBSERVER_HTML
    assert "Vorher: Prepared Compression Input" in test_ui_server.OBSERVER_HTML
    assert "Nachher: Finale Summary" in test_ui_server.OBSERVER_HTML
    assert "comparison_stats" in test_ui_server.OBSERVER_HTML
    assert "langgraph_state_profile" in test_ui_server.OBSERVER_HTML
    assert "Archive RAG Smoke" in test_ui_server.OBSERVER_HTML
    assert "runArchiveRagSmoke" in test_ui_server.OBSERVER_HTML
    assert "/api/archive-rag-smoke" in test_ui_server.OBSERVER_HTML
    assert "archiveRagRaw" in test_ui_server.OBSERVER_HTML
    assert "runtime prüfen" in test_ui_server.OBSERVER_HTML
    assert "no_runtime_errors" in test_ui_server.OBSERVER_HTML
    assert "Native Document RAG Smoke" in test_ui_server.OBSERVER_HTML
    assert "runNativeRagSmoke" in test_ui_server.OBSERVER_HTML
    assert "/api/native-document-rag-smoke" in test_ui_server.OBSERVER_HTML
    assert "nativeRagRaw" in test_ui_server.OBSERVER_HTML
    assert "NATIVE_PGVECTOR_RAG_SMOKE" in test_ui_server.OBSERVER_HTML
    assert "Small Qwen Classifier Probe" in test_ui_server.OBSERVER_HTML
    assert "runRagClassifierProbe" in test_ui_server.OBSERVER_HTML
    assert "/api/rag-classifier-probe" in test_ui_server.OBSERVER_HTML
    assert "ragClassifierRaw" in test_ui_server.OBSERVER_HTML
    assert "Memory Embed Tester" in test_ui_server.OBSERVER_HTML
    assert "runMemoryEmbedProbe" in test_ui_server.OBSERVER_HTML
    assert "/api/memory-embed-probe" in test_ui_server.OBSERVER_HTML
    assert "Ollama /api/embed" in test_ui_server.OBSERVER_HTML
    assert "OpenAI /v1" in test_ui_server.OBSERVER_HTML
    assert "RAG Load Probe" in test_ui_server.OBSERVER_HTML
    assert "runRagLoadProbe" in test_ui_server.OBSERVER_HTML
    assert "/api/rag-load-probe" in test_ui_server.OBSERVER_HTML
    assert "Curated Memory Review" in test_ui_server.OBSERVER_HTML
    assert "/api/curated-memory/review/extract" in test_ui_server.OBSERVER_HTML
    assert "400,1000,4000,10000,20000,40000" in test_ui_server.OBSERVER_HTML
    assert "window.setInterval" in test_ui_server.OBSERVER_HTML


def test_curated_memory_review_extract_and_decide(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ALPHARAVIS_CURATED_MEMORY_REVIEW_PATH", str(tmp_path / "review.json"))

    result = test_ui_server.memory_review_extract_candidates(
        "Remember that I prefer concise feature updates.",
        source_key="thread-1",
        source_type="thread",
        thread_id="thread-1",
    )

    assert result["count"] >= 1
    candidate_id = result["items"][0]["candidate_id"]
    accepted = test_ui_server.memory_review_update_candidate(candidate_id, status="accepted", reviewer_note="ok")

    assert accepted["ok"] is True
    assert accepted["item"]["status"] == "accepted"


def test_chunking_diagnostic_uses_real_compressor_with_tool_and_prompt_load() -> None:
    request = test_ui_server.ChunkingRunRequest(
        approx_tokens=60000,
        token_limit=12000,
        summary_context_token_limit=16000,
        max_chunks=32,
        include_tools=True,
        variable_prompt_load=True,
        summary_mode="stub",
        compact_instructions="preserve chunk acceptance criteria",
    )

    result = asyncio.run(test_ui_server._run_chunking_diagnostic(request))
    metadata = result["archive_metadata"]

    assert result["summary_call_count"] > 1
    assert metadata["summary_chunking_used"] is True
    assert metadata["summary_chunk_count"] > 1
    assert metadata["summary_chunk_omitted_chars"] == 0
    assert metadata["summary_chunk_payload_token_limit"] < metadata["summary_chunk_prompt_token_limit"]
    assert metadata["summary_chunk_prompt_overhead_tokens"] > 0
    assert metadata["compact_instructions"] == "preserve chunk acceptance criteria"
    assert result["tool_stats"]["pruned_tool_count"] > 0
    assert result["tool_stats"]["deduped_tool_count"] > 0
    assert result["tool_stats"]["workflow_event_count"] > 0
    assert result["tool_stats"]["workflow_event_chars"] > 0
    assert result["acceptance"]["summary_failed_false"] is True
    assert result["acceptance"]["summary_chunking_used_true"] is True
    assert result["acceptance"]["summary_chunk_omitted_chars_zero"] is True
    assert result["config"]["summary_mode"] == "stub"
    assert result["config"]["summary_model"] == "stub"
    assert result["config"]["compact_instructions_chars"] > 0
    assert any(action["event"] == "compression.workflow_events.compacted" for action in result["actions"])
    assert any(action["event"] == "compression.chunk.completed" for action in result["actions"])
    assert all(call["summary_mode"] == "stub" for call in result["summary_calls"])
    comparison = result["comparison"]
    assert comparison["before_prepared_summary_input"]["text"]
    assert comparison["after_summary"]["text"]
    assert comparison["before_tokens_estimate"] > comparison["after_tokens_estimate"]
    assert comparison["active_shrink_ratio"] > 0


def test_archive_rag_smoke_mirrors_then_queries_rag_api(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    async def fake_mirror_text(**kwargs):
        calls.append(("mirror", kwargs))
        return {"status": True, "file_id": kwargs["file_id"]}

    async def fake_query_sources(query, file_ids, *, limit):
        calls.append(("query", {"query": query, "file_ids": file_ids, "limit": limit}))
        return [
            {
                "source_key": file_ids[0],
                "chunk_text": "query_archive should retrieve bounded chunks from a rag_api mirror.",
                "metadata": {"file_id": file_ids[0]},
            }
        ]

    monkeypatch.setattr(test_ui_server, "rag_api_mirror_text", fake_mirror_text)
    monkeypatch.setattr(test_ui_server, "rag_api_query_sources", fake_query_sources)

    result = asyncio.run(
        test_ui_server._run_archive_rag_smoke(
            test_ui_server.ArchiveRagSmokeRequest(
                archive_key="archive-test",
                archive_text="Decision: query_archive should retrieve bounded chunks.",
                query="Welche Entscheidung?",
                limit=3,
            )
        )
    )

    assert calls[0][0] == "mirror"
    assert calls[0][1]["file_id"] == "archive:archive-test"
    assert calls[1] == ("query", {"query": "Welche Entscheidung?", "file_ids": ["archive:archive-test"], "limit": 3})
    assert result["rag_file_id"] == "archive:archive-test"
    assert result["acceptance_ok"] is True
    assert result["acceptance"]["query_returned_hits"] is True
    assert result["acceptance"]["no_runtime_errors"] is True


def test_archive_rag_smoke_reports_query_runtime_error(monkeypatch) -> None:
    async def fake_mirror_text(**kwargs):
        return {"status": True, "file_id": kwargs["file_id"]}

    async def fake_query_sources(query, file_ids, *, limit):
        raise test_ui_server.RagApiClientError("embedding backend refused connection")

    monkeypatch.setattr(test_ui_server, "rag_api_mirror_text", fake_mirror_text)
    monkeypatch.setattr(test_ui_server, "rag_api_query_sources", fake_query_sources)

    result = asyncio.run(
        test_ui_server._run_archive_rag_smoke(
            test_ui_server.ArchiveRagSmokeRequest(archive_key="archive-test", query="Was steht drin?")
        )
    )

    assert result["status"] == "failed"
    assert result["acceptance_ok"] is False
    assert result["acceptance"]["no_runtime_errors"] is False
    assert result["errors"] == [{"stage": "query", "error": "embedding backend refused connection"}]
    assert result["actions"][-1]["event"] == "query.failed"


def test_native_document_rag_smoke_uses_alpharavis_pgvector_only(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    async def fake_upsert(**kwargs):
        return "large_paste:native-doc:1"

    async def fake_ingest_source(**kwargs):
        calls.append(("ingest", kwargs))
        return {
            "source_key": kwargs["source_key"],
            "source_type": kwargs["source_type"],
            "index_status": "indexed",
            "indexed_backends": ["alpharavis_pgvector"],
            "rag_active": True,
            "active_source_keys": [kwargs["source_key"]],
            "active_rag_file_ids": [],
            "metadata": {
                "source_key": kwargs["source_key"],
                "rag_active": True,
                "active_source_keys": [kwargs["source_key"]],
                "active_rag_file_ids": [],
            },
        }

    async def fake_agentic_rag_retrieve(**kwargs):
        calls.append(("retrieve", kwargs))
        return {
            "next_action": "generate_answer",
            "context_packet": {
                "chunk_count": 1,
                "chunks": [
                    {
                        "source_key": kwargs["source_keys"][0],
                        "retrieval_backend": "alpharavis_pgvector",
                        "chunk_text": "Runtime marker: NATIVE_PGVECTOR_RAG_SMOKE. AlphaRavis-owned pgvector is the default.",
                    }
                ],
            },
            "graph_trace": [{"node": "retrieve"}],
        }

    monkeypatch.setattr(test_ui_server, "pgvector_memory_enabled", lambda: True)
    monkeypatch.setattr(test_ui_server, "pgvector_upsert_memory_record", fake_upsert)
    monkeypatch.setattr(test_ui_server, "pgvector_semantic_search", object())
    monkeypatch.setattr(test_ui_server, "router_ingest_source", fake_ingest_source)
    monkeypatch.setattr(test_ui_server, "router_agentic_rag_retrieve", fake_agentic_rag_retrieve)

    result = asyncio.run(
        test_ui_server._run_native_document_rag_smoke(
            test_ui_server.NativeDocumentRagSmokeRequest(
                source_key="native-doc",
                document_text="Runtime marker: NATIVE_PGVECTOR_RAG_SMOKE.",
                query="Welche Regel?",
            )
        )
    )

    assert calls[0][0] == "ingest"
    assert calls[0][1]["preferred_backend"] == "pgvector"
    assert calls[0][1]["rag_mirror_func"] is None
    assert calls[1][0] == "retrieve"
    assert calls[1][1]["rag_query_func"] is None
    assert calls[1][1]["rag_source_keys"] is None
    assert result["acceptance_ok"] is True
    assert result["acceptance"]["pgvector_backend_selected"] is True
    assert result["acceptance"]["rag_api_not_used"] is True


def test_embedding_queue_status_exposes_source_progress(monkeypatch) -> None:
    async def fake_queue_stats():
        return {
            "table": "alpharavis_embedding_jobs",
            "counts": {"pending": 1, "running": 1, "done": 2},
            "recent_active": [],
            "source_progress": [
                {
                    "id": "job-1",
                    "status": "running",
                    "source_type": "large_paste",
                    "source_key": "source-1",
                    "planned_chunks": 4,
                    "completed_chunks": 2,
                    "progress": 0.5,
                }
            ],
        }

    monkeypatch.setattr(test_ui_server, "pgvector_queue_stats", fake_queue_stats)

    result = asyncio.run(test_ui_server._embedding_queue_status())

    assert result["active_count"] == 2
    assert result["source_progress"][0]["source_key"] == "source-1"
    assert result["source_progress"][0]["completed_chunks"] == 2


def test_rag_classifier_probe_covers_semantic_and_fallback_cases() -> None:
    result = asyncio.run(
        test_ui_server._run_rag_classifier_probe(
            test_ui_server.RagClassifierProbeRequest(mode="local_fallback")
        )
    )

    cases = {item["case"]: item for item in result["results"]}
    assert result["status"] == "passed"
    assert set(cases) == {
        "short_direct",
        "long_noisy",
        "instruction_only",
        "document_only",
        "mixed",
        "fallback_down",
        "fallback_invalid_json",
        "fallback_timeout",
    }
    assert cases["short_direct"]["classification"] == "short_direct"
    assert cases["long_noisy"]["classification"] == "long_noisy"
    assert cases["instruction_only"]["classification"] == "instruction_only"
    assert cases["document_only"]["classification"] == "document_only"
    assert cases["mixed"]["classification"] == "mixed"
    assert cases["fallback_down"]["fallback_used"] is True
    assert cases["fallback_invalid_json"]["fallback_used"] is True
    assert cases["fallback_timeout"]["fallback_used"] is True


def test_embedding_queue_status_reports_progress(monkeypatch) -> None:
    async def fake_queue_stats():
        return {
            "table": "alpharavis_embedding_queue",
            "counts": {"pending": 2, "running": 1, "failed": 1, "done": 6},
            "recent_active": [
                {
                    "source_type": "large_paste",
                    "source_key": "source-1",
                    "title": "Large Source",
                    "status": "pending",
                    "last_error": "",
                }
            ],
        }

    monkeypatch.setattr(test_ui_server, "pgvector_queue_stats", fake_queue_stats)

    result = asyncio.run(test_ui_server._embedding_queue_status())

    assert result["status"] == "active"
    assert result["active_count"] == 4
    assert result["done_count"] == 6
    assert result["progress"] == 0.6
    assert result["recent_active"][0]["source_key"] == "source-1"


def test_memory_embed_probe_openai_compatible(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeResponse:
        status_code = 200
        text = "{}"

        def json(self):
            return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

    class FakeClient:
        def __init__(self, *, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, endpoint, *, json, headers):
            calls.append({"endpoint": endpoint, "json": json, "headers": headers, "timeout": self.timeout})
            return FakeResponse()

    monkeypatch.setattr(test_ui_server.httpx, "AsyncClient", FakeClient)

    result = asyncio.run(
        test_ui_server._run_memory_embed_probe(
            test_ui_server.MemoryEmbedProbeRequest(
                base_url="http://embed-box:8080/v1",
                model="memory-embed",
                api_key="sk-test",
                backend="openai",
                start_chars=8,
                max_chars=16,
                max_steps=2,
            )
        )
    )

    assert result["status"] == "passed"
    assert result["max_accepted_chars"] == 16
    assert result["results"][0]["embedding_dimensions"] == 3
    assert calls[0]["endpoint"] == "http://embed-box:8080/v1/embeddings"
    assert calls[0]["headers"] == {"Authorization": "Bearer sk-test"}
    assert calls[0]["json"]["model"] == "memory-embed"


def test_memory_embed_probe_reports_rejection(monkeypatch) -> None:
    class FakeResponse:
        status_code = 413
        text = "context too large"

        def json(self):
            return {}

    class FakeClient:
        def __init__(self, *, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, endpoint, *, json, headers):
            return FakeResponse()

    monkeypatch.setattr(test_ui_server.httpx, "AsyncClient", FakeClient)

    result = asyncio.run(
        test_ui_server._run_memory_embed_probe(
            test_ui_server.MemoryEmbedProbeRequest(
                base_url="http://ollama-box:11434",
                model="bge-m3",
                backend="ollama_embed",
                start_chars=128,
                max_chars=512,
            )
        )
    )

    assert result["status"] == "failed"
    assert result["stop_reason"] == "rejected"
    assert result["results"][0]["status_code"] == 413
    assert "context too large" in result["results"][0]["error"]


def test_rag_load_probe_runs_embedding_and_reranker(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeResponse:
        def __init__(self, endpoint: str):
            self.endpoint = endpoint
            self.status_code = 200
            self.text = "{}"

        def json(self):
            if self.endpoint.endswith("/api/embed"):
                return {"embeddings": [[0.1, 0.2, 0.3, 0.4]], "prompt_eval_count": 12}
            if self.endpoint.endswith("/reranking"):
                return {
                    "usage": {"prompt_tokens": 88, "total_tokens": 88},
                    "results": [{"index": 0, "relevance_score": 0.99}],
                }
            return {"output": [{"content": [{"text": "Probe answer mentions pgvector and reranking."}]}]}

    class FakeClient:
        def __init__(self, *, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, endpoint, *, json, headers=None):
            calls.append((endpoint, json))
            return FakeResponse(endpoint)

    monkeypatch.setattr(test_ui_server.httpx, "AsyncClient", FakeClient)

    result = asyncio.run(
        test_ui_server._run_rag_load_probe(
            test_ui_server.RagLoadProbeRequest(
                embedding_base_url="http://ollama-box:11434",
                reranker_url="http://reranker-box:8000",
                token_steps="400,1000",
                reranker_doc_count=3,
                bridge_query_mode="first_last",
            )
        )
    )

    assert result["status"] == "passed"
    assert result["ok_step_count"] == 2
    assert result["results"][0]["embedding"]["embedding_dimensions"] == 4
    assert result["results"][0]["reranker"]["prompt_tokens"] == 88
    assert result["results"][0]["bridge"]["ok"] is True
    assert any(endpoint == "http://ollama-box:11434/api/embed" for endpoint, _payload in calls)
    assert any(endpoint == "http://reranker-box:8000/reranking" for endpoint, _payload in calls)
