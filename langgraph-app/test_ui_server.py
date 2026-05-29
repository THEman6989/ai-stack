from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from context_compressor import compress_messages, estimate_tokens_rough, prepare_messages_for_summary
from curated_memory_review import extract_candidates as memory_review_extract_candidates
from curated_memory_review import list_candidates as memory_review_list_candidates
from curated_memory_review import update_candidate as memory_review_update_candidate
from rag_api_client import RagApiClientError
from rag_api_client import mirror_text as rag_api_mirror_text
from rag_api_client import query_sources as rag_api_query_sources
from retrieval_router import agentic_rag_retrieve as router_agentic_rag_retrieve
from retrieval_router import archive_rag_file_id
from retrieval_router import ingest_source as router_ingest_source
from rag_pins_manager import list_pins as rag_list_pins
from rag_pins_manager import load_pins as rag_load_pins
from rag_pins_manager import update_pins as rag_update_pins
from run_state_manager import list_run_checkpoints

try:
    from vector_memory import is_enabled as pgvector_memory_enabled
    from vector_memory import queue_stats as pgvector_queue_stats
    from vector_memory import semantic_search as pgvector_semantic_search
    from vector_memory import upsert_memory_record as pgvector_upsert_memory_record
except Exception as exc:  # pragma: no cover - depends on optional runtime deps
    PGVECTOR_IMPORT_ERROR: Exception | None = exc
    pgvector_memory_enabled = None
    pgvector_queue_stats = None
    pgvector_semantic_search = None
    pgvector_upsert_memory_record = None
else:
    PGVECTOR_IMPORT_ERROR = None


BRIDGE_BASE_URL = os.getenv("TEST_UI_BRIDGE_BASE_URL", "http://api-bridge:8123/v1").rstrip("/")
BRIDGE_MODEL = os.getenv("TEST_UI_MODEL", "my-agent")
BRIDGE_TIMEOUT_SECONDS = float(os.getenv("TEST_UI_BRIDGE_TIMEOUT_SECONDS", "240"))
CHUNKING_RUNS: dict[str, dict[str, Any]] = {}
CHUNKING_RUN_RETENTION = int(os.getenv("TEST_UI_CHUNKING_RUN_RETENTION", "20"))

app = FastAPI(title="AlphaRavis Bridge Test UI")


class ChatRequest(BaseModel):
    message: str
    messages: list[dict[str, Any]] = []
    protocol: str = "responses"
    stream: bool = True
    session_id: str = ""
    trace_id: str = ""


class ChunkingRunRequest(BaseModel):
    approx_tokens: int = 300000
    token_limit: int = 64000
    summary_context_token_limit: int = 128000
    max_chunks: int = 12
    include_tools: bool = True
    variable_prompt_load: bool = True
    summary_mode: str = "stub"
    compact_instructions: str = ""
    corpus_text: str = ""


class ArchiveRagSmokeRequest(BaseModel):
    archive_key: str = ""
    archive_text: str = ""
    query: str = "Welche Retrieval-Entscheidung steht im Archiv?"
    limit: int = 4


class NativeDocumentRagSmokeRequest(BaseModel):
    source_key: str = ""
    source_type: str = "large_paste"
    title: str = "AlphaRavis Native RAG Smoke"
    document_text: str = ""
    query: str = "Welche native AlphaRavis-RAG-Regel steht im Dokument?"
    limit: int = 4


class MemoryEmbedProbeRequest(BaseModel):
    base_url: str = "http://litellm:4000/v1"
    model: str = "memory-embed"
    api_key: str = "sk-local-dev"
    backend: str = "openai"
    input_kind: str = "text"
    text: str = "AlphaRavis memory embedding smoke text."
    image_data_url: str = ""
    start_chars: int = 256
    max_chars: int = 131072
    multiplier: float = 2.0
    timeout_seconds: float = 30.0
    slow_seconds: float = 10.0
    stop_on_slow: bool = True
    max_steps: int = 8


class RagLoadProbeRequest(BaseModel):
    embedding_base_url: str = "http://192.168.178.140:11434"
    embedding_model: str = "qwen3-embedding:4b"
    embedding_api_key: str = ""
    embedding_backend: str = "ollama_embed"
    reranker_url: str = "http://192.168.178.140:8000"
    reranker_endpoint: str = "/reranking"
    reranker_model: str = "qwen3-reranker-0.6b"
    query: str = "How does AlphaRavis handle native pgvector retrieval and reranking?"
    text: str = (
        "AlphaRavis native RAG stores source-scoped chunks in pgvector, uses a "
        "durable embedding queue, and can rerank bounded retrieval hits before "
        "grounding the answer."
    )
    token_steps: str = "400,1000,4000,10000,20000,40000"
    chars_per_token: float = 4.0
    reranker_doc_count: int = 10
    reranker_doc_chars: int = 700
    timeout_seconds: float = 240.0
    bridge_query_mode: str = "none"
    bridge_protocol: str = "responses"
    stop_on_failure: bool = True


class RagClassifierProbeRequest(BaseModel):
    mode: str = "local_fallback"
    classifier_base_url: str = ""
    classifier_model: str = ""
    timeout_seconds: float = 8.0


class RagPinsRequest(BaseModel):
    thread_id: str
    add_source_keys: list[str] = []
    add_rag_file_ids: list[str] = []
    remove_source_keys: list[str] = []
    remove_rag_file_ids: list[str] = []
    clear_all: bool = False
    archive_rag_mode: str = "tool_only"


class CuratedMemoryReviewExtractRequest(BaseModel):
    text: str
    source_key: str = ""
    source_type: str = "thread"
    thread_id: str = ""
    title: str = ""
    max_candidates: int = 8


class CuratedMemoryReviewDecisionRequest(BaseModel):
    candidate_id: str
    reviewer_note: str = ""


def _extract_responses_text(payload: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        for part in item.get("content", []):
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                chunks.append(part["text"])
    return "".join(chunks)


def _extract_chat_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content", "")
    return content if isinstance(content, str) else str(content)


def _extract_trace(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    trace = metadata.get("alpha_trace")
    if not isinstance(trace, dict):
        trace = payload.get("alpharavis_trace")
    return trace if isinstance(trace, dict) else {}


def _protocol(raw: str) -> str:
    protocol = raw.strip().lower()
    return protocol if protocol in {"responses", "chat"} else "responses"


def _bridge_request_payload(
    request: ChatRequest,
    *,
    text: str,
    protocol: str,
    session_id: str,
    trace_id: str,
    stream: bool,
) -> tuple[str, dict[str, Any]]:
    metadata = {
        "conversation_id": f"bridge-test-ui-{session_id}",
        "trace_id": trace_id,
        "trace_source": "bridge-test-ui",
    }
    if protocol == "chat":
        history = [
            {"role": str(item.get("role") or "user"), "content": str(item.get("content") or "")}
            for item in request.messages
            if item.get("role") in {"user", "assistant"} and item.get("content")
        ]
        return (
            f"{BRIDGE_BASE_URL}/chat/completions",
            {
                "model": BRIDGE_MODEL,
                "messages": [*history, {"role": "user", "content": text}],
                "stream": stream,
                "max_tokens": 512,
                "metadata": metadata,
            },
        )

    input_items: list[dict[str, Any]] = []
    for item in request.messages[-20:]:
        role = str(item.get("role") or "")
        content = str(item.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            input_items.append(
                {
                    "type": "message",
                    "role": role,
                    "content": [{"type": "output_text" if role == "assistant" else "input_text", "text": content}],
                }
            )
    input_items.append(
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        }
    )
    return (
        f"{BRIDGE_BASE_URL}/responses",
        {
            "model": BRIDGE_MODEL,
            "input": input_items,
            "stream": stream,
            "max_output_tokens": 512,
            "metadata": metadata,
        },
    )


def _test_ui_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _bounded_int(value: Any, *, minimum: int, maximum: int, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _bounded_float(value: Any, *, minimum: float, maximum: float, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not parsed == parsed:
        return default
    return max(minimum, min(maximum, parsed))


def _chunking_log(run: dict[str, Any], event: str, **details: Any) -> None:
    started = float(run.get("started_monotonic") or time.perf_counter())
    run.setdefault("actions", []).append(
        {
            "t": round(time.perf_counter() - started, 3),
            "event": event,
            **details,
        }
    )


def _synthetic_web_corpus(target_tokens: int) -> str:
    target_chars = max(4000, int(target_tokens) * 4)
    paragraphs = [
        (
            "Source: local synthetic web corpus. Section {i}. AlphaRavis chunking "
            "diagnostic text mixes documentation notes, issue comments, log excerpts, "
            "search-result style snippets, and repeated facts so compression must keep "
            "stable details while ignoring already-handled noise. "
            "Decision marker AR-{marker}: keep archive lookup guidance, preserve tool "
            "failures, and distinguish old instructions from the latest request."
        ),
        (
            "Search result {i}: context compression with static prompts, variable "
            "handoff notes, MemoryKernel facts, skill hints, and oversized tool output. "
            "The active summary should mention chunk coverage, omitted characters, "
            "budget limits, and whether exact details require read_archive_record."
        ),
        (
            "Log excerpt {i}: pytest -q tests/test_context_compressor.py completed; "
            "summary_prompt_pruned may become true; summary_chunking_used should become "
            "true when the prepared middle exceeds the summary prompt budget. "
            "Repeated terminal output follows: INFO action=probe status=ok."
        ),
    ]
    parts: list[str] = []
    index = 0
    while sum(len(part) for part in parts) < target_chars:
        template = paragraphs[index % len(paragraphs)]
        parts.append(template.format(i=index, marker=f"{index % 97:02d}") + "\n")
        index += 1
    return "".join(parts)[:target_chars]


def _slice_text(text: str, *, chunk_chars: int) -> list[str]:
    return [text[index : index + chunk_chars] for index in range(0, len(text), chunk_chars) if text[index : index + chunk_chars]]


def _build_chunking_messages(corpus: str, *, include_tools: bool) -> list[dict[str, Any]]:
    slices = _slice_text(corpus, chunk_chars=3600)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "AlphaRavis chunking diagnostic policy: preserve only reference facts."},
        {"role": "user", "content": "Start a long diagnostic thread. Latest real request appears after compression."},
    ]
    duplicate_tool_output = (
        "Search results found 42 matches for AlphaRavis chunking.\n"
        + "\n".join(f"/repo/file_{i}.py:{i}: summary_chunking_used marker" for i in range(80))
    )
    tool_index = 0
    for index, segment in enumerate(slices):
        if include_tools and index % 14 == 5:
            call_id = f"call_{tool_index:04d}"
            huge_args = {
                "query": "AlphaRavis chunking static prompt variable prompt compression stats",
                "path": "/workspace/diagnostic/very-large-thread.md",
                "payload": "tool-argument-detail " * 280,
            }
            messages.append(
                {
                    "role": "assistant",
                    "content": "I will inspect the diagnostic corpus with a tool.",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "diagnostic_search",
                                "arguments": json.dumps(huge_args),
                            },
                        }
                    ],
                }
            )
            tool_content = duplicate_tool_output if tool_index % 2 == 0 else (
                "Command: rg summary_chunking_used\nExit code: 0\n"
                + segment
                + "\n"
                + "\n".join(f"line {n}: chunk payload evidence" for n in range(160))
            )
            messages.append(
                {
                    "role": "tool",
                    "name": "diagnostic_search",
                    "tool_call_id": call_id,
                    "content": tool_content,
                }
            )
            tool_index += 1
            continue
        role = "assistant" if index % 2 else "user"
        messages.append({"role": role, "content": f"Corpus segment {index}\n{segment}"})
    messages.append({"role": "user", "content": "Latest active request: report whether chunked compression stayed within budget."})
    return messages


def _variable_prompt_context(enabled: bool) -> dict[str, str]:
    if not enabled:
        return {
            "current_task_brief": "",
            "latest_handoff_packet": "",
            "memory_kernel_context": "",
            "skill_context": "",
        }
    return {
        "current_task_brief": (
            "Validate chunked summary compression with a large active thread. Preserve the latest "
            "user ask, chunk statistics, tool-pruning evidence, and whether archive lookup is needed. "
            + "Task-brief payload. " * 160
        ),
        "latest_handoff_packet": (
            "<handoff-packet>{\"owner\":\"bridge-test-ui\",\"goal\":\"chunking-soak\","
            "\"must_report\":[\"summary_chunking_used\",\"summary_chunk_omitted_chars\","
            "\"summary_failed\"]}</handoff-packet>\n"
            + "handoff detail " * 180
        ),
        "memory_kernel_context": (
            "MemoryKernel: the operator wants visible modern diagnostics and machine-readable API status. "
            + "durable preference " * 180
        ),
        "skill_context": (
            "Skill Context: use Hermes-style compression rules, preserve archive references, and do not "
            "treat old middle content as a fresh instruction. "
            + "workflow hint " * 180
        ),
    }


def _mode_from_prompt(prompt: str) -> str:
    match = re.search(r"(?m)^mode:\s*(.+)$", prompt)
    return match.group(1).strip() if match else "unknown"


def _capture_text_window(text: str, *, max_chars: int | None = None) -> dict[str, Any]:
    text = str(text or "")
    if max_chars is None:
        max_chars = _bounded_int(
            os.getenv("TEST_UI_CHUNKING_TEXT_CAPTURE_CHARS", "240000"),
            minimum=20000,
            maximum=2000000,
            default=240000,
        )
    if len(text) <= max_chars:
        return {
            "text": text,
            "chars": len(text),
            "captured_chars": len(text),
            "truncated": False,
            "omitted_chars": 0,
        }
    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars
    marker = (
        f"\n\n[AlphaRavis Chunking Lab capture truncated: {len(text) - max_chars} chars omitted "
        "from the middle for browser/API display. The compression run itself used the prepared "
        "input according to its chunking budget.]\n\n"
    )
    captured = text[:head_chars].rstrip() + marker + text[-tail_chars:].lstrip()
    return {
        "text": captured,
        "chars": len(text),
        "captured_chars": len(captured),
        "truncated": True,
        "omitted_chars": max(0, len(text) - max_chars),
    }


def _summary_mode(raw: str) -> str:
    mode = str(raw or "stub").strip().lower()
    return mode if mode in {"stub", "real_llm"} else "stub"


def _real_llm_summary_config() -> tuple[str, str, str, float]:
    base_url = os.getenv("TEST_UI_CHUNKING_SUMMARY_API_BASE", os.getenv("OPENAI_API_BASE", "http://litellm:4000/v1")).rstrip("/")
    model = os.getenv(
        "TEST_UI_CHUNKING_SUMMARY_MODEL",
        os.getenv("ALPHARAVIS_RESPONSES_MODEL", os.getenv("ALPHARAVIS_MODEL", "big-boss")),
    )
    if "litellm" in base_url and model.startswith("openai/") and not os.getenv("TEST_UI_CHUNKING_SUMMARY_MODEL"):
        model = model.removeprefix("openai/")
    api_key = os.getenv("TEST_UI_CHUNKING_SUMMARY_API_KEY", os.getenv("OPENAI_API_KEY", "sk-local-dev"))
    timeout = float(os.getenv("TEST_UI_CHUNKING_SUMMARY_TIMEOUT_SECONDS", "240"))
    return base_url, model, api_key, timeout


def _stub_summary_text(call_type: str, mode: str, prompt_tokens: int) -> str:
    return (
        "## Active Task\n"
        f"- Diagnostic {call_type} summary for {mode}; prompt_tokens={prompt_tokens}.\n\n"
        "## Goal\n- Verify chunked summary compression and Observer/API stats.\n\n"
        "## Handoff Packet\n- Preserve chunking acceptance criteria and archive lookup note.\n\n"
        "## MemoryKernel\n- Operator wants visual and API-visible running action tracking.\n\n"
        "## Skill Context\n- Use reference-only compression semantics.\n\n"
        "## Constraints / Preferences\n- Do not treat compressed middle content as a new user instruction.\n\n"
        "## Progress Done\n- Synthetic/pasted corpus processed by the diagnostic run.\n\n"
        "## Progress In Progress\n- Inspect summary_chunking_used, omitted chars, and budget stats.\n\n"
        "## Blocked / Risks\n- Positive summary_chunk_omitted_chars requires tuning max chunks or chunk size.\n\n"
        "## Resolved Questions\n- The Bridge itself does not perform this chunking; the diagnostic calls LangGraph compression helpers.\n\n"
        "## Pending User Asks\n- Report the diagnostic result from the latest active request.\n\n"
        "## Key Decisions\n- Keep raw middle retrievable from archive references in real runs.\n\n"
        "## Relevant Files\n- langgraph-app/context_compressor.py\n\n"
        "## Commands / Tools Used\n- bridge-test-ui chunking diagnostic\n\n"
        "## Critical Context\n- Chunk summaries are intermediate evidence, not final user instructions.\n\n"
        "## Remaining Work\n- Run a real live llama.cpp soak before changing defaults.\n\n"
        "## Archive References\n- source_type=archive; exact raw messages remain graph-owned in real compression runs."
    )


async def _real_llm_summary_from_prompt(prompt: str, max_tokens: int) -> str:
    base_url, model, api_key, timeout = _real_llm_summary_config()
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
        if response.status_code >= 400:
            raise RuntimeError(f"summary LLM HTTP {response.status_code}: {response.text[:2000]}")
        raw = response.json()
    choices = raw.get("choices") if isinstance(raw, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("summary LLM returned no choices")
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("summary LLM returned empty content")
    return content


async def _run_chunking_diagnostic(request: ChunkingRunRequest, run: dict[str, Any] | None = None) -> dict[str, Any]:
    run = run if run is not None else {"started_monotonic": time.perf_counter(), "actions": []}
    approx_tokens = _bounded_int(request.approx_tokens, minimum=10000, maximum=500000, default=300000)
    token_limit = _bounded_int(request.token_limit, minimum=4096, maximum=256000, default=64000)
    summary_context = _bounded_int(
        request.summary_context_token_limit,
        minimum=8192,
        maximum=512000,
        default=128000,
    )
    max_chunks = _bounded_int(request.max_chunks, minimum=1, maximum=64, default=12)
    summary_mode = _summary_mode(request.summary_mode)
    compact_instructions = str(request.compact_instructions or "").strip()[:1200]
    summary_base_url, summary_model, _summary_api_key, _summary_timeout = _real_llm_summary_config()
    _chunking_log(
        run,
        "configured",
        approx_tokens=approx_tokens,
        token_limit=token_limit,
        summary_context_token_limit=summary_context,
        max_chunks=max_chunks,
        include_tools=bool(request.include_tools),
        variable_prompt_load=bool(request.variable_prompt_load),
        summary_mode=summary_mode,
        compact_instructions_chars=len(compact_instructions),
    )
    await asyncio.sleep(0)

    corpus = str(request.corpus_text or "").strip()
    if corpus:
        source = "pasted"
        _chunking_log(run, "corpus.loaded", source=source, chars=len(corpus))
    else:
        source = "synthetic_web_like"
        corpus = _synthetic_web_corpus(approx_tokens)
        _chunking_log(run, "corpus.generated", source=source, chars=len(corpus))
    await asyncio.sleep(0)

    messages = _build_chunking_messages(corpus, include_tools=bool(request.include_tools))
    prompt_context = _variable_prompt_context(bool(request.variable_prompt_load))
    prepared = prepare_messages_for_summary(messages)
    _chunking_log(
        run,
        "messages.prepared",
        message_count=len(messages),
        estimated_tokens=estimate_tokens_rough(messages),
        prepared_summary_chars=len(prepared.text),
        pruned_tools=prepared.pruned_tool_count,
        deduped_tools=prepared.deduped_tool_count,
        truncated_tool_args=prepared.tool_args_truncated_count,
        workflow_events=prepared.workflow_event_count,
        workflow_event_chars=prepared.workflow_event_chars,
    )
    await asyncio.sleep(0)

    summary_calls: list[dict[str, Any]] = []

    async def summarize(prompt: str, max_tokens: int) -> str:
        mode = _mode_from_prompt(prompt)
        call_type = "synthesis" if "chunked_synthesis" in mode else ("chunk" if "_chunk_" in mode else "one_shot")
        call_started = time.perf_counter()
        call = {
            "index": len(summary_calls) + 1,
            "mode": mode,
            "type": call_type,
            "summary_mode": summary_mode,
            "prompt_chars": len(prompt),
            "prompt_tokens_estimate": estimate_tokens_rough(prompt),
            "max_tokens": max_tokens,
        }
        summary_calls.append(call)
        _chunking_log(run, "summary.call", **call)
        if summary_mode == "real_llm":
            _chunking_log(run, "summary.llm.request", index=call["index"], model=summary_model, base_url=summary_base_url)
            try:
                summary = await _real_llm_summary_from_prompt(prompt, max_tokens)
            except Exception as exc:
                call["elapsed_seconds"] = round(time.perf_counter() - call_started, 3)
                detail = str(exc) or repr(exc)
                _chunking_log(
                    run,
                    "summary.failed",
                    index=call["index"],
                    elapsed_seconds=call["elapsed_seconds"],
                    error_type=type(exc).__name__,
                    detail=detail[:1000],
                )
                raise RuntimeError(f"{type(exc).__name__}: {detail}") from exc
        else:
            summary = _stub_summary_text(call_type, mode, call["prompt_tokens_estimate"])
        call["elapsed_seconds"] = round(time.perf_counter() - call_started, 3)
        _chunking_log(run, "summary.completed", index=call["index"], elapsed_seconds=call["elapsed_seconds"], summary_chars=len(summary))
        return summary

    previous_max_chunks = os.environ.get("ALPHARAVIS_COMPRESSION_SUMMARY_MAX_CHUNKS")
    os.environ["ALPHARAVIS_COMPRESSION_SUMMARY_MAX_CHUNKS"] = str(max_chunks)
    try:
        result = await compress_messages(
            messages,
            mode="bridge_test_ui_chunking",
            thread_id=f"chunking_diag_{uuid.uuid4().hex[:10]}",
            thread_key="bridge-test-ui-chunking-diagnostic",
            token_limit=token_limit,
            previous_summary=None,
            summary_context_token_limit=summary_context,
            summarize_fn=summarize,
            force=True,
            enable_chunked_summary=True,
            compact_instructions=compact_instructions,
            progress_callback=lambda event: _chunking_log(run, str(event.get("event") or "compression.progress"), **{
                key: value for key, value in event.items() if key != "event"
            }),
            **prompt_context,
        )
    finally:
        if previous_max_chunks is None:
            os.environ.pop("ALPHARAVIS_COMPRESSION_SUMMARY_MAX_CHUNKS", None)
        else:
            os.environ["ALPHARAVIS_COMPRESSION_SUMMARY_MAX_CHUNKS"] = previous_max_chunks

    metadata = result.archive_metadata
    acceptance = {
        "summary_failed_false": result.summary_failed is False,
        "summary_chunking_used_true": metadata.get("summary_chunking_used") is True,
        "summary_chunk_omitted_chars_zero": int(metadata.get("summary_chunk_omitted_chars") or 0) == 0,
        "token_estimate_after_under_limit": int(result.token_estimate_after or 0) <= token_limit,
    }
    _chunking_log(
        run,
        "compression.completed",
        summary_failed=result.summary_failed,
        summary_chunking_used=metadata.get("summary_chunking_used"),
        summary_chunk_count=metadata.get("summary_chunk_count"),
        summary_chunk_omitted_chars=metadata.get("summary_chunk_omitted_chars"),
        token_estimate_before=result.token_estimate_before,
        token_estimate_after=result.token_estimate_after,
    )
    return {
        "source": source,
        "config": {
            "approx_tokens": approx_tokens,
            "token_limit": token_limit,
            "summary_context_token_limit": summary_context,
            "max_chunks": max_chunks,
            "include_tools": bool(request.include_tools),
            "variable_prompt_load": bool(request.variable_prompt_load),
            "summary_mode": summary_mode,
            "compact_instructions_chars": len(compact_instructions),
            "summary_api_base": summary_base_url if summary_mode == "real_llm" else "",
            "summary_model": summary_model if summary_mode == "real_llm" else "stub",
        },
        "input": {
            "corpus_chars": len(corpus),
            "message_count": len(messages),
            "estimated_tokens": estimate_tokens_rough(messages),
            "prepared_summary_chars": len(prepared.text),
            "prepared_summary_tokens": estimate_tokens_rough(prepared.text),
        },
        "tool_stats": {
            "pruned_tool_count": result.pruned_tool_count,
            "deduped_tool_count": result.deduped_tool_count,
            "tool_args_truncated_count": result.tool_args_truncated_count,
            "workflow_event_count": result.workflow_event_count,
            "workflow_tool_call_count": result.workflow_tool_call_count,
            "workflow_tool_result_count": result.workflow_tool_result_count,
            "workflow_event_chars": result.workflow_event_chars,
        },
        "summary_calls": summary_calls,
        "summary_call_count": len(summary_calls),
        "actions": list(run.get("actions") or []),
        "archive_metadata": metadata,
        "token_estimate_before": result.token_estimate_before,
        "token_estimate_after": result.token_estimate_after,
        "summary_preview": result.summary[:1600],
        "comparison": {
            "before_prepared_summary_input": _capture_text_window(prepared.text),
            "after_summary": _capture_text_window(result.summary, max_chars=120000),
            "before_tokens_estimate": estimate_tokens_rough(prepared.text),
            "after_tokens_estimate": estimate_tokens_rough(result.summary),
            "active_tokens_before": result.token_estimate_before,
            "active_tokens_after": result.token_estimate_after,
            "active_shrink_ratio": round(1 - (result.token_estimate_after / max(1, result.token_estimate_before)), 4),
        },
        "acceptance": acceptance,
        "acceptance_ok": all(acceptance.values()),
    }


def _default_archive_smoke_text(archive_key: str) -> str:
    return (
        f"AlphaRavis archive smoke record {archive_key}.\n\n"
        "Decision: compression archives stay in MongoDB/LangGraph Store as the "
        "source of truth, while rag_api may hold a secondary retrieval mirror.\n"
        "Important retrieval rule: query_archive should retrieve bounded chunks "
        "by question and must not inject the entire raw archive into active model "
        "context.\n"
        "Fallback: if rag_api is missing or the mirror does not exist, AlphaRavis "
        "falls back to source-key search in vector_memory.py.\n"
    )


def _default_native_document_smoke_text(source_key: str) -> str:
    return (
        f"AlphaRavis native document RAG smoke source {source_key}.\n\n"
        "Runtime marker: NATIVE_PGVECTOR_RAG_SMOKE.\n"
        "Decision: explicit documents and large pasted sources should be indexed "
        "through AlphaRavis-owned pgvector by default, not through rag_api. "
        "Important retrieval rule: active document threads keep active_source_keys "
        "and should retrieve bounded chunks from vector_memory.py before answering.\n"
        "Adapter rule: rag_api remains selectable only as an adapter or comparison "
        "backend, not as the default product implementation.\n"
    )


def _rag_smoke_payload_ok(payload: dict[str, Any]) -> bool:
    status = payload.get("status", True)
    if isinstance(status, bool):
        return status
    if isinstance(status, str):
        return status.strip().lower() not in {"false", "failed", "error"}
    return bool(payload)


async def _native_pgvector_index(**kwargs: Any) -> str | None:
    if pgvector_upsert_memory_record is None:
        raise RuntimeError(f"AlphaRavis pgvector memory unavailable: {PGVECTOR_IMPORT_ERROR}")
    return await pgvector_upsert_memory_record(**kwargs)


async def _run_native_document_rag_smoke(request: NativeDocumentRagSmokeRequest) -> dict[str, Any]:
    started = time.perf_counter()
    source_key = re.sub(r"[^a-zA-Z0-9_.:-]+", "_", str(request.source_key or "").strip())[:120]
    if not source_key:
        source_key = f"native_doc_smoke:{uuid.uuid4().hex[:10]}"
    source_type = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(request.source_type or "large_paste").strip().lower())[:80] or "large_paste"
    title = str(request.title or source_key).strip()[:200] or source_key
    text = str(request.document_text or "").strip() or _default_native_document_smoke_text(source_key)
    query = str(request.query or "").strip() or "Welche native AlphaRavis-RAG-Regel steht im Dokument?"
    limit = _bounded_int(request.limit, minimum=1, maximum=20, default=4)
    thread_id = f"bridge-test-ui-native-rag-{uuid.uuid4().hex[:10]}"
    thread_key = "bridge-test-ui-native-rag"

    actions: list[dict[str, Any]] = []

    def log(event: str, **details: Any) -> None:
        actions.append({"t": round(time.perf_counter() - started, 3), "event": event, **details})

    errors: list[dict[str, Any]] = []
    ingest: dict[str, Any] = {}
    retrieval: dict[str, Any] = {}

    if pgvector_memory_enabled is None or pgvector_upsert_memory_record is None or pgvector_semantic_search is None:
        errors.append({"stage": "pgvector", "error": f"AlphaRavis pgvector memory unavailable: {PGVECTOR_IMPORT_ERROR}"})
        log("pgvector.unavailable", error=str(PGVECTOR_IMPORT_ERROR)[:500])
    elif not pgvector_memory_enabled():
        errors.append({"stage": "pgvector", "error": "AlphaRavis pgvector memory is disabled."})
        log("pgvector.disabled")
    else:
        log("ingest.started", backend="alpharavis_pgvector", source_key=source_key, chars=len(text))
        try:
            ingest = await router_ingest_source(
                source_type=source_type,
                source_key=source_key,
                title=title,
                content=text,
                thread_id=thread_id,
                thread_key=thread_key,
                scope="thread",
                metadata={
                    "origin": "bridge_test_ui_native_rag_smoke",
                    "rag_activation_reason": "large_paste" if source_type in {"large_paste", "large_ingest"} else "document_ingest",
                    "runtime_marker": "NATIVE_PGVECTOR_RAG_SMOKE",
                },
                preferred_backend="pgvector",
                pgvector_index=_native_pgvector_index,
                rag_mirror_func=None,
            )
            log(
                "ingest.completed",
                status=ingest.get("index_status"),
                indexed_backends=ingest.get("indexed_backends", []),
            )
        except Exception as exc:
            errors.append({"stage": "ingest", "error": f"{type(exc).__name__}: {exc}"})
            log("ingest.failed", error=f"{type(exc).__name__}: {exc}"[:500])

        if ingest and not errors:
            log("retrieve.started", source_key=source_key, limit=limit)
            try:
                retrieval = await router_agentic_rag_retrieve(
                    query=query,
                    source_keys=[source_key],
                    source_type=source_type,
                    limit=limit,
                    include_other_threads=False,
                    thread_id=thread_id,
                    pgvector_search=pgvector_semantic_search,
                    pgvector_available=True,
                    pgvector_import_error=PGVECTOR_IMPORT_ERROR,
                    rag_query_func=None,
                    rag_source_keys=None,
                    allow_rewrite=True,
                    max_context_chars=4000,
                )
                packet = retrieval.get("context_packet") if isinstance(retrieval, dict) else {}
                log(
                    "retrieve.completed",
                    chunk_count=packet.get("chunk_count", 0) if isinstance(packet, dict) else 0,
                    next_action=retrieval.get("next_action", "") if isinstance(retrieval, dict) else "",
                )
            except Exception as exc:
                errors.append({"stage": "retrieve", "error": f"{type(exc).__name__}: {exc}"})
                log("retrieve.failed", error=f"{type(exc).__name__}: {exc}"[:500])

    packet = retrieval.get("context_packet") if isinstance(retrieval, dict) else {}
    chunks = packet.get("chunks") if isinstance(packet, dict) else []
    chunks = chunks if isinstance(chunks, list) else []
    context_text = "\n".join(str(chunk.get("chunk_text") or "") for chunk in chunks if isinstance(chunk, dict))
    indexed_backends = list(ingest.get("indexed_backends") or []) if isinstance(ingest, dict) else []
    active_rag_file_ids = list(ingest.get("active_rag_file_ids") or []) if isinstance(ingest, dict) else []
    active_source_keys = list(ingest.get("active_source_keys") or []) if isinstance(ingest, dict) else []
    acceptance = {
        "pgvector_backend_selected": indexed_backends == ["alpharavis_pgvector"],
        "rag_api_not_used": "rag_api" not in indexed_backends and not active_rag_file_ids,
        "active_source_key_recorded": source_key in active_source_keys,
        "retrieval_returned_context": len(chunks) > 0,
        "retrieved_expected_marker": "NATIVE_PGVECTOR_RAG_SMOKE" in context_text or "AlphaRavis-owned pgvector" in context_text,
        "no_runtime_errors": not errors,
    }
    return {
        "source_key": source_key,
        "source_type": source_type,
        "thread_id": thread_id,
        "query": query,
        "text_chars": len(text),
        "status": "passed" if all(acceptance.values()) else "failed",
        "errors": errors,
        "ingest": ingest,
        "retrieval": retrieval,
        "context_packet": packet if isinstance(packet, dict) else {},
        "hit_count": len(chunks),
        "actions": actions,
        "acceptance": acceptance,
        "acceptance_ok": all(acceptance.values()),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


async def _run_archive_rag_smoke(request: ArchiveRagSmokeRequest) -> dict[str, Any]:
    started = time.perf_counter()
    archive_key = re.sub(r"[^a-zA-Z0-9_.:-]+", "_", str(request.archive_key or "").strip())[:80]
    if not archive_key:
        archive_key = f"smoke_{uuid.uuid4().hex[:10]}"
    file_id = archive_rag_file_id(archive_key)
    text = str(request.archive_text or "").strip() or _default_archive_smoke_text(archive_key)
    query = str(request.query or "").strip() or "Welche Retrieval-Entscheidung steht im Archiv?"
    limit = _bounded_int(request.limit, minimum=1, maximum=20, default=4)

    actions: list[dict[str, Any]] = []

    def log(event: str, **details: Any) -> None:
        actions.append({"t": round(time.perf_counter() - started, 3), "event": event, **details})

    mirror: dict[str, Any] = {}
    hits: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    log("mirror.started", file_id=file_id, chars=len(text))
    try:
        mirror = await rag_api_mirror_text(
            file_id=file_id,
            text=text,
            filename=f"{file_id.replace(':', '_')}.txt",
        )
        log("mirror.completed", status=mirror.get("status"), file_id=mirror.get("file_id"))
        if not _rag_smoke_payload_ok(mirror):
            message = str(mirror.get("message") or mirror.get("error") or "rag_api mirror returned failed status")
            errors.append({"stage": "mirror", "error": message})
            log("mirror.status_failed", error=message[:500])
    except (RagApiClientError, httpx.HTTPError) as exc:
        errors.append({"stage": "mirror", "error": str(exc)})
        log("mirror.failed", error=str(exc)[:500])
    except Exception as exc:
        errors.append({"stage": "mirror", "error": f"{type(exc).__name__}: {exc}"})
        log("mirror.failed", error=f"{type(exc).__name__}: {exc}"[:500])

    if mirror and not errors:
        log("query.started", file_id=file_id, limit=limit)
        try:
            hits = await rag_api_query_sources(query, [file_id], limit=limit)
            log("query.completed", hit_count=len(hits))
        except (RagApiClientError, httpx.HTTPError) as exc:
            errors.append({"stage": "query", "error": str(exc)})
            log("query.failed", error=str(exc)[:500])
        except Exception as exc:
            errors.append({"stage": "query", "error": f"{type(exc).__name__}: {exc}"})
            log("query.failed", error=f"{type(exc).__name__}: {exc}"[:500])

    first_hit = hits[0] if hits else {}
    hit_text = str(first_hit.get("chunk_text") or first_hit.get("preview_text") or "")
    acceptance = {
        "mirror_status_ok": bool(mirror) and _rag_smoke_payload_ok(mirror),
        "rag_file_id_matches": str(mirror.get("file_id") or file_id) == file_id,
        "query_returned_hits": len(hits) > 0,
        "retrieved_expected_archive_rule": "query_archive" in hit_text or "bounded chunks" in hit_text,
        "no_runtime_errors": not errors,
    }
    return {
        "archive_key": archive_key,
        "rag_file_id": file_id,
        "query": query,
        "text_chars": len(text),
        "status": "passed" if all(acceptance.values()) else "failed",
        "errors": errors,
        "mirror": mirror,
        "hits": hits,
        "hit_count": len(hits),
        "actions": actions,
        "acceptance": acceptance,
        "acceptance_ok": all(acceptance.values()),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _embed_probe_text(seed: str, target_chars: int) -> str:
    seed = (seed or "AlphaRavis memory embedding probe.").strip()
    marker = " ALPHARAVIS_MEMORY_EMBED_PROBE "
    repeated = (seed + marker) * max(1, (target_chars // max(1, len(seed) + len(marker))) + 1)
    return repeated[:target_chars]


def _embed_probe_steps(start_chars: int, max_chars: int, multiplier: float, max_steps: int) -> list[int]:
    steps: list[int] = []
    current = max(1, start_chars)
    max_chars = max(current, max_chars)
    multiplier = max(1.1, multiplier)
    for _ in range(max_steps):
        if current not in steps:
            steps.append(current)
        if current >= max_chars:
            break
        next_value = max(current + 1, int(current * multiplier))
        current = min(max_chars, next_value)
    return steps


def _strip_data_url(value: str) -> str:
    value = str(value or "").strip()
    if "," in value and value.split(",", 1)[0].lower().startswith("data:"):
        return value.split(",", 1)[1].strip()
    return value


def _embedding_dimension(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and isinstance(first.get("embedding"), list):
            return len(first["embedding"])
    embeddings = payload.get("embeddings")
    if isinstance(embeddings, list) and embeddings:
        first = embeddings[0]
        if isinstance(first, list):
            return len(first)
    embedding = payload.get("embedding")
    if isinstance(embedding, list):
        return len(embedding)
    return None


def _embedding_endpoint(base_url: str, backend: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        base = "http://litellm:4000/v1"
    if backend == "ollama_embed":
        return f"{base.removesuffix('/v1')}/api/embed"
    if backend == "ollama_embeddings":
        return f"{base.removesuffix('/v1')}/api/embeddings"
    return f"{base}/embeddings" if base.endswith("/v1") else f"{base}/v1/embeddings"


def _embedding_payload(request: MemoryEmbedProbeRequest, text: str, backend: str) -> dict[str, Any]:
    input_kind = str(request.input_kind or "text").strip().lower()
    image_data = str(request.image_data_url or "").strip()
    if backend == "ollama_embed":
        payload: dict[str, Any] = {"model": request.model, "input": text}
        if input_kind == "vision" and image_data:
            payload["images"] = [_strip_data_url(image_data)]
        return payload
    if backend == "ollama_embeddings":
        payload = {"model": request.model, "prompt": text}
        if input_kind == "vision" and image_data:
            payload["images"] = [_strip_data_url(image_data)]
        return payload
    if input_kind == "vision" and image_data:
        return {
            "model": request.model,
            "input": [
                {"type": "input_text", "text": text},
                {"type": "input_image", "image_url": image_data},
            ],
        }
    return {"model": request.model, "input": text}


async def _run_memory_embed_probe(request: MemoryEmbedProbeRequest) -> dict[str, Any]:
    started = time.perf_counter()
    backend = str(request.backend or "openai").strip().lower()
    if backend not in {"openai", "ollama_embed", "ollama_embeddings"}:
        backend = "openai"
    timeout_seconds = _bounded_float(request.timeout_seconds, minimum=1.0, maximum=240.0, default=30.0)
    slow_seconds = _bounded_float(request.slow_seconds, minimum=0.1, maximum=240.0, default=10.0)
    start_chars = _bounded_int(request.start_chars, minimum=1, maximum=2_000_000, default=256)
    max_chars = _bounded_int(request.max_chars, minimum=start_chars, maximum=2_000_000, default=8192)
    max_steps = _bounded_int(request.max_steps, minimum=1, maximum=32, default=8)
    multiplier = _bounded_float(request.multiplier, minimum=1.1, maximum=10.0, default=2.0)
    endpoint = _embedding_endpoint(request.base_url, backend)
    headers = {"Authorization": f"Bearer {request.api_key.strip()}"} if backend == "openai" and request.api_key.strip() else {}
    steps = _embed_probe_steps(start_chars, max_chars, multiplier, max_steps)
    results: list[dict[str, Any]] = []
    stop_reason = "completed"

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        for chars in steps:
            text = _embed_probe_text(request.text, chars)
            payload = _embedding_payload(request, text, backend)
            call_started = time.perf_counter()
            result: dict[str, Any] = {
                "chars": chars,
                "approx_tokens": estimate_tokens_rough(text),
                "backend": backend,
                "endpoint": endpoint,
            }
            try:
                response = await client.post(endpoint, json=payload, headers=headers)
                elapsed = round(time.perf_counter() - call_started, 3)
                result.update({"status_code": response.status_code, "elapsed_seconds": elapsed})
                if response.status_code >= 400:
                    result.update({"ok": False, "error": response.text[:1000]})
                    stop_reason = "rejected"
                    results.append(result)
                    break
                try:
                    body = response.json()
                except Exception:
                    body = {"raw": response.text[:1000]}
                dim = _embedding_dimension(body)
                result.update(
                    {
                        "ok": dim is not None,
                        "embedding_dimensions": dim,
                        "slow": elapsed >= slow_seconds,
                        "response_keys": sorted(body.keys()) if isinstance(body, dict) else [],
                    }
                )
                if dim is None:
                    result["error"] = "Embedding response did not contain a recognized embedding vector."
                    stop_reason = "unexpected_response"
                    results.append(result)
                    break
                results.append(result)
                if result["slow"] and request.stop_on_slow:
                    stop_reason = "slow_threshold"
                    break
            except Exception as exc:
                result.update(
                    {
                        "ok": False,
                        "elapsed_seconds": round(time.perf_counter() - call_started, 3),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                results.append(result)
                stop_reason = "error"
                break

    accepted = [item for item in results if item.get("ok")]
    return {
        "status": "passed" if accepted and stop_reason == "completed" else ("partial" if accepted else "failed"),
        "backend": backend,
        "input_kind": str(request.input_kind or "text").strip().lower(),
        "base_url": str(request.base_url or "").strip(),
        "endpoint": endpoint,
        "model": request.model,
        "steps": steps,
        "results": results,
        "accepted_step_count": len(accepted),
        "max_accepted_chars": max([int(item["chars"]) for item in accepted], default=0),
        "max_accepted_approx_tokens": max([int(item["approx_tokens"]) for item in accepted], default=0),
        "stop_reason": stop_reason,
        "acceptance_ok": bool(accepted) and stop_reason == "completed",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "notes": [
            "Vision embedding support is backend/model specific; this tester reports whether the selected endpoint accepts the submitted vision payload.",
            "Use OpenAI-compatible mode for LiteLLM or llama.cpp /v1 servers; use Ollama native modes for direct Ollama /api/embed or /api/embeddings probes.",
        ],
    }


def _load_probe_token_steps(raw_steps: str, *, max_steps: int = 8) -> list[int]:
    steps: list[int] = []
    for part in str(raw_steps or "").split(","):
        try:
            tokens = int(part.strip())
        except ValueError:
            continue
        tokens = _bounded_int(tokens, minimum=1, maximum=60000, default=1)
        if tokens not in steps:
            steps.append(tokens)
        if len(steps) >= max_steps:
            break
    return steps or [400, 1000, 4000]


def _reranker_endpoint(base_url: str, endpoint: str) -> str:
    base = str(base_url or "").strip().rstrip("/") or "http://192.168.178.140:8000"
    suffix = str(endpoint or "/reranking").strip() or "/reranking"
    if not suffix.startswith("/"):
        suffix = f"/{suffix}"
    return f"{base}{suffix}"


def _rag_load_documents(text: str, *, doc_count: int, doc_chars: int) -> list[str]:
    doc_count = _bounded_int(doc_count, minimum=1, maximum=50, default=10)
    doc_chars = _bounded_int(doc_chars, minimum=80, maximum=4000, default=700)
    seed = text or "AlphaRavis RAG load probe."
    docs: list[str] = []
    stride = max(1, doc_chars // 2)
    for index in range(doc_count):
        start = (index * stride) % max(1, len(seed))
        chunk = (seed[start:] + "\n" + seed[:start]).strip()
        docs.append(f"Document {index + 1}: {chunk[:doc_chars]}")
    return docs


async def _call_embedding_step(
    client: httpx.AsyncClient,
    request: RagLoadProbeRequest,
    *,
    text: str,
) -> dict[str, Any]:
    backend = str(request.embedding_backend or "openai").strip().lower()
    if backend not in {"openai", "ollama_embed", "ollama_embeddings"}:
        backend = "openai"
    endpoint = _embedding_endpoint(request.embedding_base_url, backend)
    probe_request = MemoryEmbedProbeRequest(
        base_url=request.embedding_base_url,
        model=request.embedding_model,
        api_key=request.embedding_api_key,
        backend=backend,
        text=text,
    )
    headers = (
        {"Authorization": f"Bearer {request.embedding_api_key.strip()}"}
        if backend == "openai" and request.embedding_api_key.strip()
        else {}
    )
    started = time.perf_counter()
    result: dict[str, Any] = {"endpoint": endpoint, "model": request.embedding_model, "backend": backend}
    try:
        response = await client.post(endpoint, json=_embedding_payload(probe_request, text, backend), headers=headers)
        result.update({"status_code": response.status_code, "elapsed_seconds": round(time.perf_counter() - started, 3)})
        if response.status_code >= 400:
            result.update({"ok": False, "error": response.text[:1000]})
            return result
        body = response.json()
        dim = _embedding_dimension(body)
        result.update(
            {
                "ok": dim is not None,
                "embedding_dimensions": dim,
                "prompt_eval_count": body.get("prompt_eval_count") if isinstance(body, dict) else None,
                "total_duration_ns": body.get("total_duration") if isinstance(body, dict) else None,
                "response_keys": sorted(body.keys()) if isinstance(body, dict) else [],
            }
        )
        if dim is None:
            result["error"] = "Embedding response did not contain a recognized embedding vector."
    except Exception as exc:
        result.update({"ok": False, "elapsed_seconds": round(time.perf_counter() - started, 3), "error": f"{type(exc).__name__}: {exc}"})
    return result


async def _call_reranker_step(
    client: httpx.AsyncClient,
    request: RagLoadProbeRequest,
    *,
    documents: list[str],
) -> dict[str, Any]:
    endpoint = _reranker_endpoint(request.reranker_url, request.reranker_endpoint)
    payload = {"model": request.reranker_model, "query": request.query, "documents": documents}
    started = time.perf_counter()
    result: dict[str, Any] = {"endpoint": endpoint, "model": request.reranker_model, "document_count": len(documents)}
    try:
        response = await client.post(endpoint, json=payload)
        result.update({"status_code": response.status_code, "elapsed_seconds": round(time.perf_counter() - started, 3)})
        if response.status_code >= 400:
            result.update({"ok": False, "error": response.text[:1000]})
            return result
        body = response.json()
        items = body.get("results") if isinstance(body, dict) else []
        usage = body.get("usage") if isinstance(body, dict) and isinstance(body.get("usage"), dict) else {}
        result.update(
            {
                "ok": isinstance(items, list) and len(items) > 0,
                "prompt_tokens": usage.get("prompt_tokens") or usage.get("total_tokens"),
                "result_count": len(items) if isinstance(items, list) else 0,
                "top_index": items[0].get("index") if isinstance(items, list) and items and isinstance(items[0], dict) else None,
                "top_score": items[0].get("relevance_score") if isinstance(items, list) and items and isinstance(items[0], dict) else None,
            }
        )
        if not result["ok"]:
            result["error"] = "Reranker response did not contain results."
    except Exception as exc:
        result.update({"ok": False, "elapsed_seconds": round(time.perf_counter() - started, 3), "error": f"{type(exc).__name__}: {exc}"})
    return result


async def _call_bridge_load_query(
    client: httpx.AsyncClient,
    request: RagLoadProbeRequest,
    *,
    text: str,
    tokens: int,
    step_index: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    prompt = (
        f"AlphaRavis load probe step {step_index + 1}, roughly {tokens} input tokens.\n"
        "Answer briefly and mention whether the probe text talks about embeddings, reranking, or pgvector.\n\n"
        f"{text}"
    )
    chat_request = ChatRequest(message=prompt, protocol=request.bridge_protocol, stream=False)
    trace_id = f"rag-load-probe-{uuid.uuid4().hex[:8]}"
    session_id = f"rag-load-{uuid.uuid4().hex[:8]}"
    url, payload = _bridge_request_payload(
        chat_request,
        text=prompt,
        protocol=_protocol(request.bridge_protocol),
        session_id=session_id,
        trace_id=trace_id,
        stream=False,
    )
    result: dict[str, Any] = {"endpoint": url, "protocol": _protocol(request.bridge_protocol), "trace_id": trace_id}
    try:
        response = await client.post(url, json=payload, headers={"Authorization": "Bearer sk-local-dev"})
        result.update({"status_code": response.status_code, "elapsed_seconds": round(time.perf_counter() - started, 3)})
        if response.status_code >= 400:
            result.update({"ok": False, "error": response.text[:1000]})
            return result
        body = response.json()
        text_out = _extract_responses_text(body) if result["protocol"] == "responses" else _extract_chat_text(body)
        result.update({"ok": bool(text_out), "output_chars": len(text_out), "output_preview": text_out[:400]})
    except Exception as exc:
        result.update({"ok": False, "elapsed_seconds": round(time.perf_counter() - started, 3), "error": f"{type(exc).__name__}: {exc}"})
    return result


async def _run_rag_load_probe(request: RagLoadProbeRequest) -> dict[str, Any]:
    started = time.perf_counter()
    timeout_seconds = _bounded_float(request.timeout_seconds, minimum=5.0, maximum=900.0, default=240.0)
    chars_per_token = _bounded_float(request.chars_per_token, minimum=1.0, maximum=12.0, default=4.0)
    steps = _load_probe_token_steps(request.token_steps)
    bridge_mode = str(request.bridge_query_mode or "none").strip().lower()
    if bridge_mode not in {"none", "first_last", "all"}:
        bridge_mode = "none"
    results: list[dict[str, Any]] = []
    stop_reason = "completed"

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        for index, tokens in enumerate(steps):
            chars = max(1, int(tokens * chars_per_token))
            text = _embed_probe_text(request.text, chars)
            documents = _rag_load_documents(text, doc_count=request.reranker_doc_count, doc_chars=request.reranker_doc_chars)
            step_started = time.perf_counter()
            embedding_task = _call_embedding_step(client, request, text=text)
            reranker_task = _call_reranker_step(client, request, documents=documents)
            embedding, reranker = await asyncio.gather(embedding_task, reranker_task)
            bridge = None
            if bridge_mode == "all" or (bridge_mode == "first_last" and index in {0, len(steps) - 1}):
                bridge = await _call_bridge_load_query(client, request, text=text, tokens=tokens, step_index=index)
            step = {
                "tokens": tokens,
                "chars": chars,
                "elapsed_seconds": round(time.perf_counter() - step_started, 3),
                "embedding": embedding,
                "reranker": reranker,
                "bridge": bridge,
                "ok": bool(embedding.get("ok")) and bool(reranker.get("ok")) and (bridge is None or bool(bridge.get("ok"))),
            }
            results.append(step)
            if not step["ok"] and request.stop_on_failure:
                stop_reason = "failed_step"
                break

    ok_count = sum(1 for item in results if item.get("ok"))
    required_count = len(results)
    return {
        "status": "passed" if results and ok_count == required_count and stop_reason == "completed" else ("partial" if ok_count else "failed"),
        "steps": steps,
        "results": results,
        "ok_step_count": ok_count,
        "completed_step_count": len(results),
        "stop_reason": stop_reason,
        "embedding_model": request.embedding_model,
        "reranker_model": request.reranker_model,
        "bridge_query_mode": bridge_mode,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "notes": [
            "Each step runs the embedding request and reranker request concurrently to expose shared GPU/CPU contention.",
            "Bridge query mode sends a real AlphaRavis request after the concurrent embedding/reranker step for selected sizes.",
        ],
    }


def _classifier_base_url(request: RagClassifierProbeRequest) -> str:
    configured = (request.classifier_base_url or os.getenv("ALPHARAVIS_RAG_CLASSIFIER_API_BASE", "")).strip().rstrip("/")
    if configured:
        return configured
    boss = os.getenv("BIG_BOSS_API_BASE", "http://100.71.57.22:8033/v1").strip().rstrip("/")
    match = re.match(r"^(https?://[^:/]+)(?::\d+)?(?:/v1)?$", boss)
    if match:
        return f"{match.group(1)}:8001/v1"
    return "http://100.71.57.22:8001/v1"


def _classifier_model(request: RagClassifierProbeRequest) -> str:
    return (request.classifier_model or os.getenv("ALPHARAVIS_RAG_CLASSIFIER_MODEL", "qwen3.5-2b")).strip()


def _classifier_probe_cases() -> list[dict[str, str]]:
    noisy = " ".join(["kurz gesagt"] * 120)
    return [
        {
            "case": "short_direct",
            "text": "Was steht im RAG ueber AlphaRavis reranking?",
            "expected": "short_direct",
        },
        {
            "case": "long_noisy",
            "text": f"{noisy}\n\nEigentliche Frage: Wie funktioniert der Archive Recall Condenser?",
            "expected": "long_noisy",
        },
        {
            "case": "instruction_only",
            "text": "Bitte antworte knapp, nutze Quellen, veraendere keine FastPass Defaults.",
            "expected": "instruction_only",
        },
        {
            "case": "document_only",
            "text": "AlphaRavis RAG Policy:\n- Large Paste wird als Quelle abgelegt.\n- Reranker bleibt aktiv.",
            "expected": "document_only",
        },
        {
            "case": "mixed",
            "text": "Bitte analysiere das Dokument.\n\n```log\nERROR alpha rag queue pending\nINFO reranker active\n```\n\nWas ist die Ursache?",
            "expected": "mixed",
        },
        {
            "case": "fallback_down",
            "text": "Simulierter Endpoint-Ausfall: Was soll die lokale Kondensation tun?",
            "expected": "fallback",
        },
        {
            "case": "fallback_invalid_json",
            "text": "Simuliertes kaputtes JSON: extrahiere trotzdem eine Suchfrage.",
            "expected": "fallback",
        },
        {
            "case": "fallback_timeout",
            "text": "Simulierter Timeout: nutze lokale Fallback-Klassifikation.",
            "expected": "fallback",
        },
    ]


def _json_object_from_text(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(raw[start : end + 1])
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}
    return {}


def _local_classifier_decision(case: dict[str, str], *, reason: str = "") -> dict[str, Any]:
    text = case["text"]
    lines = text.splitlines()
    has_code_fence = "```" in text
    has_bullets = any(line.lstrip().startswith(("-", "*")) for line in lines)
    has_question = "?" in text or "Eigentliche Frage:" in text or "Was ist" in text
    word_count = len(re.findall(r"\w+", text))
    if case["case"] == "short_direct":
        kind = "short_direct"
        query = text.strip()
    elif case["case"] == "instruction_only":
        kind = "instruction_only"
        query = ""
    elif has_code_fence and has_question:
        kind = "mixed"
        query = lines[-1].strip() if lines else text.strip()
    elif has_bullets and not has_question:
        kind = "document_only"
        query = ""
    elif word_count > 120:
        kind = "long_noisy"
        match = re.search(r"Eigentliche Frage:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        query = match.group(1).strip() if match else " ".join(text.split()[-40:])
    else:
        kind = "retrieval_query"
        query = text.strip()
    document_ranges = []
    if kind in {"document_only", "mixed"} and lines:
        document_ranges.append({"start_line": 1 if kind == "document_only" else 3, "end_line": len(lines)})
    return {
        "classification": kind,
        "query": query[:500],
        "document_line_ranges": document_ranges,
        "fallback_used": True,
        "fallback_reason": reason,
    }


async def _call_qwen_classifier(
    client: httpx.AsyncClient,
    request: RagClassifierProbeRequest,
    case: dict[str, str],
) -> dict[str, Any]:
    system = (
        "Classify an AlphaRavis RAG prompt. Return strict JSON only with keys "
        "classification, query, document_line_ranges. classification must be one "
        "of short_direct,long_noisy,instruction_only,document_only,mixed,retrieval_query."
    )
    response = await client.post(
        f"{_classifier_base_url(request)}/chat/completions",
        json={
            "model": _classifier_model(request),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": case["text"]},
            ],
            "temperature": 0,
            "max_tokens": 512,
        },
        headers={"Authorization": "Bearer sk-local-dev"},
    )
    response.raise_for_status()
    payload = response.json()
    content = _extract_chat_text(payload)
    parsed = _json_object_from_text(content)
    if not parsed:
        raise ValueError("classifier returned invalid JSON")
    parsed["fallback_used"] = False
    return parsed


async def _run_rag_classifier_probe(request: RagClassifierProbeRequest) -> dict[str, Any]:
    started = time.perf_counter()
    mode = (request.mode or "local_fallback").strip().lower()
    cases = _classifier_probe_cases()
    results: list[dict[str, Any]] = []
    real_cases = {"short_direct", "long_noisy", "instruction_only", "document_only", "mixed"}
    async with httpx.AsyncClient(timeout=max(1.0, float(request.timeout_seconds))) as client:
        for case in cases:
            case_started = time.perf_counter()
            simulated = case["case"].startswith("fallback_")
            try:
                if simulated:
                    if case["case"] == "fallback_invalid_json":
                        parsed = _json_object_from_text("not-json")
                        if not parsed:
                            raise ValueError("invalid JSON")
                    if case["case"] == "fallback_timeout":
                        raise httpx.TimeoutException("simulated timeout")
                    raise httpx.ConnectError("simulated endpoint down")
                if mode == "real_qwen" and case["case"] in real_cases:
                    decision = await _call_qwen_classifier(client, request, case)
                    source = "qwen"
                else:
                    decision = _local_classifier_decision(case, reason="local probe mode")
                    source = "local_fallback"
            except Exception as exc:
                decision = _local_classifier_decision(case, reason=str(exc)[:200])
                source = "fallback"
            classification = str(decision.get("classification") or "")
            expected = case["expected"]
            if expected == "fallback":
                ok = bool(decision.get("fallback_used"))
            else:
                ok = bool(classification)
            results.append(
                {
                    "case": case["case"],
                    "expected": expected,
                    "classification": classification,
                    "query": decision.get("query", ""),
                    "document_line_ranges": decision.get("document_line_ranges", []),
                    "fallback_used": bool(decision.get("fallback_used")),
                    "fallback_reason": decision.get("fallback_reason", ""),
                    "source": source,
                    "ok": ok,
                    "elapsed_seconds": round(time.perf_counter() - case_started, 3),
                }
            )
    ok_count = sum(1 for item in results if item.get("ok"))
    return {
        "status": "passed" if ok_count == len(results) else ("partial" if ok_count else "failed"),
        "mode": mode,
        "classifier_base_url": _classifier_base_url(request),
        "classifier_model": _classifier_model(request),
        "ok_case_count": ok_count,
        "case_count": len(results),
        "results": results,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "notes": [
            "real_qwen mode calls the configured small Qwen classifier for the five semantic cases.",
            "fallback cases deliberately simulate down, invalid JSON, and timeout behavior.",
        ],
    }


async def _embedding_queue_status() -> dict[str, Any]:
    if pgvector_queue_stats is None:
        return {
            "enabled": False,
            "status": "unavailable",
            "error": str(PGVECTOR_IMPORT_ERROR or "pgvector queue module unavailable"),
        }
    try:
        stats = await pgvector_queue_stats()
    except Exception as exc:
        return {"enabled": False, "status": "error", "error": str(exc)[:500]}
    counts = dict(stats.get("counts") or {}) if isinstance(stats, dict) else {}
    pending = int(counts.get("pending") or 0)
    running = int(counts.get("running") or 0)
    failed = int(counts.get("failed") or 0)
    done = int(counts.get("done") or 0)
    active = pending + running + failed
    total = active + done
    progress = round(done / total, 4) if total > 0 else None
    return {
        "enabled": bool(stats.get("enabled", True)) if isinstance(stats, dict) else True,
        "status": "active" if active else "idle",
        "table": stats.get("table", "") if isinstance(stats, dict) else "",
        "counts": counts,
        "active_count": active,
        "done_count": done,
        "progress": progress,
        "recent_active": stats.get("recent_active", []) if isinstance(stats, dict) else [],
        "source_progress": stats.get("source_progress", []) if isinstance(stats, dict) else [],
        "raw": stats,
        "meaning": {
            "pending": "queued but not indexed yet",
            "running": "currently claimed by the embedding runner",
            "failed": "not indexed; will retry until max attempts",
            "done": "indexed or processed successfully",
        },
    }


def _awaiting_resume_dashboard(limit: int = 50) -> dict[str, Any]:
    records = list_run_checkpoints(status="awaiting_resume", limit=limit)
    items = []
    for record in records:
        run_profile = record.get("run_profile") if isinstance(record.get("run_profile"), dict) else {}
        items.append(
            {
                "thread_id": record.get("thread_id", ""),
                "thread_key": record.get("thread_key", ""),
                "phase": record.get("phase", ""),
                "status": record.get("status", ""),
                "updated_at": record.get("updated_at"),
                "current_task_brief": record.get("current_task_brief", ""),
                "planner_context": record.get("planner_context", ""),
                "active_agent": record.get("active_agent", ""),
                "selected_toolsets": record.get("selected_toolsets", []),
                "active_source_keys": run_profile.get("active_source_keys", []),
                "active_rag_file_ids": run_profile.get("active_rag_file_ids", []),
                "error": record.get("error", ""),
                "resume_text": "ja, weiter",
            }
        )
    return {"status": "ok", "count": len(items), "items": items}


def _rag_pins_dashboard(thread_id: str = "", limit: int = 50) -> dict[str, Any]:
    thread_id = str(thread_id or "").strip()
    if thread_id:
        pins = rag_load_pins(thread_id)
        return {"status": "ok", "thread_id": thread_id, "pins": pins or {"thread_id": thread_id, "rag_active": False}}
    return {"status": "ok", "items": rag_list_pins(limit=limit)}


def _trim_chunking_runs() -> None:
    if len(CHUNKING_RUNS) <= CHUNKING_RUN_RETENTION:
        return
    ordered = sorted(CHUNKING_RUNS.items(), key=lambda item: float(item[1].get("created_at") or 0), reverse=True)
    keep = {run_id for run_id, _run in ordered[:CHUNKING_RUN_RETENTION]}
    for run_id in list(CHUNKING_RUNS):
        if run_id not in keep:
            CHUNKING_RUNS.pop(run_id, None)


async def _chunking_job(run_id: str, request: ChunkingRunRequest) -> None:
    run = CHUNKING_RUNS[run_id]
    run["status"] = "running"
    try:
        run["result"] = await _run_chunking_diagnostic(request, run)
        run["status"] = "completed"
    except Exception as exc:
        _chunking_log(run, "failed", detail=str(exc))
        run["status"] = "failed"
        run["error"] = str(exc)
    finally:
        run["elapsed_seconds"] = round(time.perf_counter() - float(run.get("started_monotonic") or time.perf_counter()), 3)
        _trim_chunking_runs()


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(HTML, headers={"Cache-Control": "no-store"})


@app.get("/observer", response_class=HTMLResponse)
async def observer() -> HTMLResponse:
    return HTMLResponse(OBSERVER_HTML, headers={"Cache-Control": "no-store"})


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "bridge_base_url": BRIDGE_BASE_URL,
        "model": BRIDGE_MODEL,
    }


@app.get("/api/observer")
async def observer_records(limit: int = 80) -> JSONResponse:
    async with httpx.AsyncClient(timeout=BRIDGE_TIMEOUT_SECONDS) as client:
        response = await client.get(f"{BRIDGE_BASE_URL.removesuffix('/v1')}/_alpharavis/bridge-observer", params={"limit": limit})
        response.raise_for_status()
        return JSONResponse(response.json())


@app.delete("/api/observer")
async def clear_observer_records() -> JSONResponse:
    async with httpx.AsyncClient(timeout=BRIDGE_TIMEOUT_SECONDS) as client:
        response = await client.delete(f"{BRIDGE_BASE_URL.removesuffix('/v1')}/_alpharavis/bridge-observer")
        response.raise_for_status()
        return JSONResponse(response.json())


@app.post("/api/chunking/runs")
async def start_chunking_run(request: ChunkingRunRequest) -> JSONResponse:
    run_id = f"chunk_{uuid.uuid4().hex[:12]}"
    CHUNKING_RUNS[run_id] = {
        "id": run_id,
        "status": "queued",
        "created_at": time.time(),
        "started_monotonic": time.perf_counter(),
        "actions": [],
        "result": None,
    }
    _chunking_log(CHUNKING_RUNS[run_id], "queued")
    asyncio.create_task(_chunking_job(run_id, request))
    return JSONResponse(CHUNKING_RUNS[run_id])


@app.get("/api/chunking/runs/{run_id}")
async def get_chunking_run(run_id: str) -> JSONResponse:
    run = CHUNKING_RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="chunking run not found")
    return JSONResponse(run)


@app.post("/api/archive-rag-smoke")
async def archive_rag_smoke(request: ArchiveRagSmokeRequest) -> JSONResponse:
    return JSONResponse(await _run_archive_rag_smoke(request))


@app.post("/api/native-document-rag-smoke")
async def native_document_rag_smoke(request: NativeDocumentRagSmokeRequest) -> JSONResponse:
    return JSONResponse(await _run_native_document_rag_smoke(request))


@app.post("/api/memory-embed-probe")
async def memory_embed_probe(request: MemoryEmbedProbeRequest) -> JSONResponse:
    return JSONResponse(await _run_memory_embed_probe(request))


@app.post("/api/rag-load-probe")
async def rag_load_probe(request: RagLoadProbeRequest) -> JSONResponse:
    return JSONResponse(await _run_rag_load_probe(request))


@app.post("/api/rag-classifier-probe")
async def rag_classifier_probe(request: RagClassifierProbeRequest) -> JSONResponse:
    return JSONResponse(await _run_rag_classifier_probe(request))


class LatencyBenchRequest(BaseModel):
    embedding_base_url: str = "http://192.168.178.140:11434"
    embedding_model: str = ""  # default from ALPHARAVIS_OLLAMA_EMBED_MODEL→qwen3-embedding:0.6b
    embedding_api_key: str = ""
    embedding_backend: str = "ollama_embed"
    reranker_url: str = "http://192.168.178.140:8000"
    reranker_endpoint: str = "/reranking"
    reranker_model: str = "qwen3-reranker-0.6b"
    classifier_base_url: str = ""
    classifier_model: str = ""
    classifier_text_sizes: str = "500,5000,50000"
    bench_query: str = "How does AlphaRavis handle native pgvector retrieval and reranking?"
    bench_text: str = (
        "AlphaRavis native RAG stores source-scoped chunks in pgvector, uses a "
        "small Qwen classifier for intent detection, and reranks results through "
        "a dedicated reranker model before assembling the final context."
    )
    timeout_seconds: float = 60.0


async def _run_latency_bench(request: LatencyBenchRequest) -> dict[str, Any]:
    """Comprehensive latency benchmark: embedding, pgvector search, classifier, reranker."""
    started = time.perf_counter()
    steps: list[dict[str, Any]] = []

    timeout = max(5.0, float(request.timeout_seconds))
    bench_text = str(request.bench_text or "").strip() or "AlphaRavis latency bench probe text."
    bench_query = str(request.bench_query or "").strip() or "How does AlphaRavis handle retrieval?"
    embed_model = (request.embedding_model or os.getenv("ALPHARAVIS_OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")).strip()

    async with httpx.AsyncClient(timeout=timeout) as client:

        # ── 1. Embedding timing ──
        embed_started = time.perf_counter()
        embed_result: dict[str, Any] = {"component": "embedding", "model": embed_model}
        try:
            emb_url = f"{request.embedding_base_url.rstrip('/')}/api/embeddings"
            emb_payload = {"model": embed_model, "input": bench_text}
            emb_resp = await client.post(emb_url, json=emb_payload)
            emb_body = emb_resp.json() if emb_resp.status_code < 400 else {}
            emb_data = emb_body.get("data") if isinstance(emb_body, dict) else None
            dim = len(emb_data[0].get("embedding", [])) if isinstance(emb_data, list) and emb_data else 0
            embed_result.update({
                "ok": emb_resp.status_code < 400 and dim > 0,
                "status_code": emb_resp.status_code,
                "embedding_dim": dim,
                "input_chars": len(bench_text),
                "elapsed_seconds": round(time.perf_counter() - embed_started, 3),
            })
        except Exception as exc:
            embed_result.update({
                "ok": False, "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.perf_counter() - embed_started, 3),
            })
        steps.append(embed_result)

        # ── 2. Pgvector search timing ──
        pgv_started = time.perf_counter()
        pgv_result: dict[str, Any] = {"component": "pgvector_search", "query_chars": len(bench_query)}
        if pgvector_semantic_search is None:
            pgv_result.update({"ok": False, "error": "pgvector unavailable", "elapsed_seconds": 0})
        else:
            try:
                search_results = await pgvector_semantic_search(
                    query=bench_query, limit=5, source_keys=None,
                )
                hit_count = len(search_results) if isinstance(search_results, list) else 0
                pgv_result.update({
                    "ok": True,
                    "hit_count": hit_count,
                    "elapsed_seconds": round(time.perf_counter() - pgv_started, 3),
                })
            except Exception as exc:
                pgv_result.update({
                    "ok": False, "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": round(time.perf_counter() - pgv_started, 3),
                })
        steps.append(pgv_result)

        # ── 3. Classifier timing (multiple text sizes) ──
        cls_sizes = [int(s.strip()) for s in request.classifier_text_sizes.split(",") if s.strip().isdigit()]
        if not cls_sizes:
            cls_sizes = [500, 5000, 50000]
        cls_base = (request.classifier_base_url or os.getenv("ALPHARAVIS_RAG_CLASSIFIER_API_BASE", "")).strip().rstrip("/")
        if not cls_base:
            cls_base = "http://192.168.178.153:8001/v1"
        cls_model = (request.classifier_model or os.getenv("ALPHARAVIS_RAG_CLASSIFIER_MODEL", "unsloth/Qwen3.5-2B-GGUF:Q4_1")).strip()
        cls_url = f"{cls_base}/chat/completions"
        cls_api_key = os.getenv("LOCAL_LLM_API_KEY", "sk-local-dev")

        for size in cls_sizes:
            cls_started = time.perf_counter()
            cls_result: dict[str, Any] = {"component": "classifier", "model": cls_model, "input_chars": size}
            try:
                text = (bench_text * ((size // max(1, len(bench_text))) + 1))[:size]
                cls_payload = {
                    "model": cls_model,
                    "messages": [
                        {"role": "system", "content": "Classify this text into: question, document, instruction, or mixed. Return JSON with keys: intent, confidence."},
                        {"role": "user", "content": text},
                    ],
                    "temperature": 0,
                    "max_tokens": 128,
                    "stream": False,
                }
                cls_resp = await client.post(cls_url, json=cls_payload, headers={"Authorization": f"Bearer {cls_api_key}"})
                cls_resp.raise_for_status()
                cls_result.update({
                    "ok": True,
                    "status_code": cls_resp.status_code,
                    "elapsed_seconds": round(time.perf_counter() - cls_started, 3),
                })
            except Exception as exc:
                cls_result.update({
                    "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                    "elapsed_seconds": round(time.perf_counter() - cls_started, 3),
                })
            steps.append(cls_result)

        # ── 4. Reranker timing ──
        rerank_started = time.perf_counter()
        rerank_result: dict[str, Any] = {"component": "reranker", "model": request.reranker_model}
        try:
            rerank_url = f"{request.reranker_url.rstrip('/')}{request.reranker_endpoint}"
            rerank_payload = {
                "model": request.reranker_model,
                "query": bench_query,
                "documents": [bench_text, "AlphaRavis uses pgvector for semantic search with HNSW indexing.", "Reranker improves precision by scoring relevance."],
            }
            rerank_resp = await client.post(rerank_url, json=rerank_payload)
            rerank_body = rerank_resp.json() if rerank_resp.status_code < 400 else {}
            items = rerank_body.get("results") if isinstance(rerank_body, dict) else []
            rerank_result.update({
                "ok": isinstance(items, list) and len(items) > 0,
                "status_code": rerank_resp.status_code,
                "result_count": len(items) if isinstance(items, list) else 0,
                "elapsed_seconds": round(time.perf_counter() - rerank_started, 3),
            })
        except Exception as exc:
            rerank_result.update({
                "ok": False, "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.perf_counter() - rerank_started, 3),
            })
        steps.append(rerank_result)

    ok_count = sum(1 for s in steps if s.get("ok"))
    return {
        "status": "passed" if ok_count == len(steps) else ("partial" if ok_count else "failed"),
        "ok_step_count": ok_count,
        "total_steps": len(steps),
        "steps": steps,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "summary": {
            "embedding": steps[0] if len(steps) > 0 else None,
            "pgvector_search": steps[1] if len(steps) > 1 else None,
            "classifier_small": steps[2] if len(steps) > 2 else None,
            "classifier_medium": steps[3] if len(steps) > 3 else None,
            "classifier_large": steps[4] if len(steps) > 4 else None,
            "reranker": steps[5] if len(steps) > 5 else None,
        },
    }


@app.post("/api/latency-bench")
async def latency_bench(request: LatencyBenchRequest) -> JSONResponse:
    return JSONResponse(await _run_latency_bench(request))


@app.get("/api/embedding-queue/status")
async def embedding_queue_status() -> JSONResponse:
    return JSONResponse(await _embedding_queue_status())


@app.get("/api/resume-runs")
async def resume_runs(limit: int = 50) -> JSONResponse:
    return JSONResponse(_awaiting_resume_dashboard(limit=limit))


@app.get("/api/rag-pins")
async def rag_pins(thread_id: str = "", limit: int = 50) -> JSONResponse:
    return JSONResponse(_rag_pins_dashboard(thread_id=thread_id, limit=limit))


@app.post("/api/rag-pins")
async def update_rag_pins(request: RagPinsRequest) -> JSONResponse:
    try:
        pins = rag_update_pins(
            thread_id=request.thread_id,
            add_source_keys=request.add_source_keys,
            add_rag_file_ids=request.add_rag_file_ids,
            remove_source_keys=request.remove_source_keys,
            remove_rag_file_ids=request.remove_rag_file_ids,
            clear_all=request.clear_all,
            archive_rag_mode=request.archive_rag_mode,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"status": "ok", "pins": pins})


@app.get("/api/curated-memory/review")
async def list_curated_memory_review(status: str = "pending", limit: int = 50) -> JSONResponse:
    return JSONResponse(memory_review_list_candidates(status=status, limit=limit))


@app.post("/api/curated-memory/review/extract")
async def extract_curated_memory_review(request: CuratedMemoryReviewExtractRequest) -> JSONResponse:
    return JSONResponse(
        memory_review_extract_candidates(
            request.text,
            source_key=request.source_key,
            source_type=request.source_type,
            thread_id=request.thread_id,
            title=request.title,
            max_candidates=request.max_candidates,
        )
    )


@app.post("/api/curated-memory/review/{candidate_id}/accept")
async def accept_curated_memory_review(candidate_id: str, request: CuratedMemoryReviewDecisionRequest) -> JSONResponse:
    return JSONResponse(
        memory_review_update_candidate(
            candidate_id or request.candidate_id,
            status="accepted",
            reviewer_note=request.reviewer_note,
        )
    )


@app.post("/api/curated-memory/review/{candidate_id}/reject")
async def reject_curated_memory_review(candidate_id: str, request: CuratedMemoryReviewDecisionRequest) -> JSONResponse:
    return JSONResponse(
        memory_review_update_candidate(
            candidate_id or request.candidate_id,
            status="rejected",
            reviewer_note=request.reviewer_note,
        )
    )


@app.post("/api/send")
async def send_chat(request: ChatRequest) -> JSONResponse:
    text = request.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="message is required")

    started = time.perf_counter()
    trace_id = request.trace_id.strip() or f"trace_{uuid.uuid4().hex[:12]}"
    session_id = request.session_id.strip() or f"session_{uuid.uuid4().hex[:12]}"
    protocol = _protocol(request.protocol)

    async with httpx.AsyncClient(timeout=BRIDGE_TIMEOUT_SECONDS) as client:
        url, payload = _bridge_request_payload(
            request,
            text=text,
            protocol=protocol,
            session_id=session_id,
            trace_id=trace_id,
            stream=False,
        )
        response = await client.post(
            url,
            json=payload,
            headers={"x-alpha-trace-id": trace_id},
        )
        response.raise_for_status()
        raw = response.json()
        if protocol == "chat":
            response.raise_for_status()
            answer = _extract_chat_text(raw)
        else:
            answer = _extract_responses_text(raw)

    elapsed_seconds = round(time.perf_counter() - started, 3)
    trace = _extract_trace(raw)
    if not trace:
        trace = {"trace_id": trace_id, "protocol": protocol, "steps": []}
    trace.setdefault("trace_id", trace_id)
    trace.setdefault("protocol", protocol)
    trace.setdefault("steps", [])
    trace["test_ui_server_elapsed_seconds"] = elapsed_seconds
    trace["steps"] = [
        {"name": "test_ui.server.received", "elapsed_seconds": 0.0},
        *[step for step in trace.get("steps", []) if isinstance(step, dict)],
        {"name": "test_ui.server.completed", "elapsed_seconds": elapsed_seconds},
    ]

    return JSONResponse(
        {
            "answer": answer,
            "protocol": protocol,
            "elapsed_seconds": elapsed_seconds,
            "trace": trace,
            "raw": raw,
        }
    )


@app.post("/api/send_stream")
async def send_chat_stream(request: ChatRequest) -> StreamingResponse:
    text = request.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="message is required")

    trace_id = request.trace_id.strip() or f"trace_{uuid.uuid4().hex[:12]}"
    session_id = request.session_id.strip() or f"session_{uuid.uuid4().hex[:12]}"
    protocol = _protocol(request.protocol)
    url, payload = _bridge_request_payload(
        request,
        text=text,
        protocol=protocol,
        session_id=session_id,
        trace_id=trace_id,
        stream=True,
    )

    async def proxy_events() -> AsyncIterator[str]:
        started = time.perf_counter()
        yield _test_ui_event(
            "test_ui.started",
            {
                "protocol": protocol,
                "trace_id": trace_id,
                "model": BRIDGE_MODEL,
            },
        )
        try:
            async with httpx.AsyncClient(timeout=BRIDGE_TIMEOUT_SECONDS) as client:
                async with client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers={"x-alpha-trace-id": trace_id},
                ) as response:
                    if response.status_code >= 400:
                        body = (await response.aread()).decode(errors="replace")
                        yield _test_ui_event(
                            "test_ui.error",
                            {
                                "protocol": protocol,
                                "trace_id": trace_id,
                                "status_code": response.status_code,
                                "detail": body[:4000],
                                "elapsed_seconds": round(time.perf_counter() - started, 3),
                            },
                        )
                        return
                    async for chunk in response.aiter_text():
                        if chunk:
                            yield chunk
        except Exception as exc:
            yield _test_ui_event(
                "test_ui.error",
                {
                    "protocol": protocol,
                    "trace_id": trace_id,
                    "detail": str(exc),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                },
            )
            return

        yield _test_ui_event(
            "test_ui.completed",
            {
                "protocol": protocol,
                "trace_id": trace_id,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            },
        )

    return StreamingResponse(
        proxy_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store"},
    )


OBSERVER_HTML = """<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AlphaRavis Bridge Observer</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; background: #0f1117; color: #eef1f5; }
    main { width: 100%; min-height: 100vh; padding: 16px; display: grid; grid-template-rows: auto minmax(300px, 1fr) auto auto minmax(240px, 36vh); gap: 12px; }
    header { display: flex; align-items: center; justify-content: space-between; gap: 12px; border-bottom: 1px solid #2b3240; padding-bottom: 10px; }
    h1 { margin: 0; font-size: 20px; font-weight: 700; }
    .tools { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    button, select, input { border: 1px solid #3a4252; border-radius: 7px; background: #171b23; color: #eef1f5; padding: 8px 10px; font: inherit; }
    button.active { background: #2d6cdf; border-color: #2d6cdf; }
    .status { color: #9aa4b2; font-size: 12px; }
    .table-wrap { border: 1px solid #2b3240; border-radius: 8px; overflow: auto; background: #11151d; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; min-width: 1180px; }
    th, td { border-bottom: 1px solid #222938; padding: 7px 8px; text-align: left; vertical-align: top; }
    th { position: sticky; top: 0; z-index: 1; background: #151a24; color: #9aa4b2; font-weight: 650; }
    tr { cursor: pointer; }
    tr:hover, tr.selected { background: #1a2230; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .pill { display: inline-block; border: 1px solid #3a4252; border-radius: 999px; padding: 1px 7px; color: #cbd5e1; }
    .hard { border-color: #8f3b3b; color: #f59b9b; }
    .done { border-color: #2f8f5b; color: #7dd3a8; }
    .warn { border-color: #9a6b24; color: #f6c46d; }
    .ok { border-color: #2f8f5b; color: #7dd3a8; }
    .detail { border: 1px solid #2b3240; border-radius: 8px; background: #0c1017; display: grid; grid-template-rows: auto 1fr; min-height: 0; }
    .detail-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; border-bottom: 1px solid #222938; padding: 8px; }
    .budget { border: 1px solid #2b3240; border-radius: 8px; background: #11151d; padding: 10px; display: grid; gap: 8px; }
    .budget-title { display: flex; align-items: center; justify-content: space-between; color: #cbd5e1; font-weight: 650; font-size: 13px; }
    .budget-grid { display: grid; grid-template-columns: repeat(12, minmax(96px, 1fr)); gap: 8px; }
    .source-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 8px; }
    .source-card { border: 1px solid #252d3b; border-radius: 7px; background: #0c1017; padding: 9px; display: grid; gap: 8px; min-width: 0; }
    .source-card-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
    .source-card h3 { margin: 0; font-size: 13px; color: #eef1f5; overflow-wrap: anywhere; }
    .source-card p { margin: 0; color: #9aa4b2; font-size: 12px; overflow-wrap: anywhere; }
    .source-metrics { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; }
    .shrink { border: 1px solid #2b3240; border-radius: 8px; background: #11151d; padding: 10px; display: grid; gap: 8px; }
    .shrink-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 8px; }
    .shrink-card { border: 1px solid #252d3b; border-radius: 7px; background: #0c1017; padding: 9px; display: grid; gap: 8px; min-width: 0; }
    .shrink-card-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
    .shrink-card h3 { margin: 0; font-size: 13px; color: #eef1f5; }
    .shrink-bar { height: 8px; background: #1a2230; border-radius: 999px; overflow: hidden; }
    .shrink-bar span { display: block; height: 100%; background: #2f8f5b; border-radius: inherit; }
    .shrink-metrics { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; }
    .workflow-log { border-top: 1px solid #252d3b; padding-top: 7px; color: #c9d3e0; }
    .workflow-log summary { cursor: pointer; color: #eef1f5; font-size: 12px; }
    .workflow-log pre { margin: 7px 0 0; max-height: 180px; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; color: #d8e0ea; font-size: 11px; line-height: 1.4; }
    .chunk-lab { border: 1px solid #2b3240; border-radius: 8px; background: #11151d; padding: 10px; display: grid; gap: 10px; }
    .chunk-lab-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
    .chunk-lab h2 { margin: 0; font-size: 13px; color: #cbd5e1; }
    .chunk-form { display: grid; grid-template-columns: repeat(8, minmax(96px, 1fr)); gap: 8px; align-items: end; }
    .chunk-form label { display: grid; gap: 4px; color: #8994a5; font-size: 11px; }
    .chunk-form input[type="number"] { width: 100%; }
    .chunk-form .check { display: flex; align-items: center; gap: 7px; min-height: 35px; }
    .chunk-form .run { min-height: 35px; }
    .archive-rag-form { grid-template-columns: 180px minmax(240px, 1fr) 90px 130px; }
    .archive-rag-form textarea { min-height: 58px; resize: vertical; }
    .embed-form { grid-template-columns: minmax(220px, 1.4fr) 150px 120px 120px repeat(4, minmax(86px, 1fr)) 130px; }
    .embed-text-form { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .embed-text-form textarea { min-height: 58px; resize: vertical; }
    .rag-load-form { grid-template-columns: minmax(190px, 1.2fr) minmax(160px, 1fr) 135px minmax(180px, 1.2fr) minmax(150px, 1fr) minmax(190px, 1fr) 70px 86px 86px 105px 120px; }
    .chunk-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; }
    .chunk-actions { border: 1px solid #252d3b; border-radius: 7px; background: #0c1017; max-height: 180px; overflow: auto; }
    .chunk-actions div { display: grid; grid-template-columns: 58px minmax(120px, 1fr) 2fr; gap: 8px; padding: 6px 8px; border-bottom: 1px solid #182030; color: #cbd5e1; font-size: 12px; }
    .chunk-actions div:last-child { border-bottom: 0; }
    .chunk-actions .muted { color: #8994a5; }
    .chunk-compare { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
    .chunk-compare details { border: 1px solid #252d3b; border-radius: 7px; background: #0c1017; min-width: 0; }
    .chunk-compare summary { cursor: pointer; padding: 8px 10px; color: #cbd5e1; font-size: 12px; user-select: none; }
    .chunk-compare pre { max-height: 340px; border-top: 1px solid #182030; }
    .chunk-raw { max-height: 220px; border: 1px solid #252d3b; border-radius: 7px; background: #0c1017; }
    .metric { border: 1px solid #252d3b; border-radius: 7px; padding: 7px; background: #0c1017; min-width: 0; }
    .metric.ok { border-color: #246d4a; background: #0c1712; }
    .metric.warn { border-color: #7b5a20; background: #1a150c; }
    .metric.hard { border-color: #783232; background: #190f12; }
    .metric span { display: block; color: #8994a5; font-size: 11px; }
    .metric strong { display: block; color: #eef1f5; font-size: 13px; overflow-wrap: anywhere; }
    .tabs, .mode { display: flex; gap: 6px; align-items: center; }
    pre { margin: 0; padding: 12px; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12px; line-height: 1.45; color: #d6deeb; }
    .empty { color: #9aa4b2; }
    @media (max-width: 900px) { .chunk-form { grid-template-columns: repeat(2, minmax(0, 1fr)); } .chunk-compare { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>AlphaRavis Bridge Observer</h1>
        <div id="status" class="status">lädt...</div>
      </div>
      <div class="tools">
        <label class="status">Limit <input id="limit" type="number" min="1" max="200" value="80" style="width:72px"></label>
        <button id="refresh" type="button">Aktualisieren</button>
        <button id="clear" type="button">Leeren</button>
        <a class="status" href="/">Test UI</a>
      </div>
    </header>
    <section class="table-wrap" aria-label="Bridge Requests">
      <table>
        <thead>
          <tr>
            <th>Zeit</th>
            <th>Protokoll</th>
            <th>Status</th>
            <th>Stream</th>
            <th>Model</th>
            <th>Thread-Key</th>
            <th>Thread-ID</th>
            <th>Raw Msg</th>
            <th>Raw Tokens</th>
            <th>Model Msg</th>
            <th>Model Tokens</th>
            <th>State Msg</th>
            <th>Budget</th>
            <th>Output</th>
            <th>Reasoning</th>
          </tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
    </section>
    <section class="budget" aria-label="Context Budget">
      <div class="budget-title">
        <span>Context Budget</span>
        <span id="budgetStatus" class="status">Keine Anfrage ausgewählt.</span>
      </div>
      <div id="budgetGrid" class="budget-grid"></div>
    </section>
    <section class="budget" aria-label="Big Message / Source Ingest">
      <div class="budget-title">
        <span>Big Message / Source Ingest</span>
        <span id="sourceStatus" class="status">Keine Quellen-Events.</span>
      </div>
      <div id="sourceGrid" class="source-grid"></div>
    </section>
    <section class="budget" aria-label="Embedding Queue">
      <div class="budget-title">
        <span>Embedding Queue</span>
        <span id="embeddingQueueStatus" class="status">lade...</span>
      </div>
      <div id="embeddingQueueGrid" class="source-grid"></div>
      <div id="embeddingQueueActions" class="chunk-actions"></div>
    </section>
    <section class="budget" aria-label="Awaiting Resume">
      <div class="budget-title">
        <span>Awaiting Resume</span>
        <span id="resumeStatus" class="status">lade...</span>
      </div>
      <div id="resumeGrid" class="source-grid"></div>
    </section>
    <section class="budget" aria-label="RAG Pins">
      <div class="budget-title">
        <span>RAG Pins</span>
        <span id="ragPinsStatus" class="status">lade...</span>
      </div>
      <div class="chunk-form archive-rag-form">
        <label>Thread ID
          <input id="ragPinsThread" type="text" placeholder="thread_id">
        </label>
        <label>Source Key
          <input id="ragPinsSource" type="text" placeholder="large_paste:...">
        </label>
        <label>Mode
          <select id="ragPinsMode">
            <option value="tool_only">tool_only</option>
            <option value="auto_on_intent">auto_on_intent</option>
            <option value="manual">manual</option>
          </select>
        </label>
        <button id="ragPinsPin" type="button">Pin</button>
        <button id="ragPinsUnpin" type="button">Unpin</button>
        <button id="ragPinsClear" type="button">Clear</button>
      </div>
      <div id="ragPinsGrid" class="source-grid"></div>
    </section>
    <section class="budget" aria-label="Curated Memory Review">
      <div class="budget-title">
        <span>Curated Memory Review</span>
        <span id="memoryReviewStatus" class="status">lade...</span>
      </div>
      <div class="chunk-form archive-rag-form">
        <label>Source Key
          <input id="memoryReviewSource" type="text" placeholder="thread/archive/source">
        </label>
        <label>Type
          <input id="memoryReviewType" type="text" value="thread">
        </label>
        <label>Thread ID
          <input id="memoryReviewThread" type="text" placeholder="optional">
        </label>
        <label style="grid-column: 1 / -1;">Text
          <textarea id="memoryReviewText" placeholder="Text aus Thread/Archiv einfuegen"></textarea>
        </label>
        <button id="memoryReviewExtract" type="button">Extrahieren</button>
        <button id="memoryReviewRefresh" type="button">Aktualisieren</button>
      </div>
      <div id="memoryReviewGrid" class="source-grid"></div>
    </section>
    <section class="shrink" aria-label="Compression Shrinking">
      <div class="budget-title">
        <span>Shrinking</span>
        <span id="shrinkStatus" class="status">Keine Kompressionsdaten.</span>
      </div>
      <div id="shrinkGrid" class="shrink-grid"></div>
    </section>
    <section class="chunk-lab" aria-label="Chunking Lab">
      <div class="chunk-lab-head">
        <h2>Chunking Lab</h2>
        <span id="chunkingStatus" class="status">bereit</span>
      </div>
      <div class="chunk-form">
        <label>Tokens
          <input id="chunkTokens" type="number" min="10000" max="500000" step="10000" value="300000">
        </label>
        <label>Active Limit
          <input id="chunkActiveLimit" type="number" min="4096" max="256000" step="1024" value="64000">
        </label>
        <label>Summary Context
          <input id="chunkSummaryContext" type="number" min="8192" max="512000" step="1024" value="128000">
        </label>
        <label>Max Chunks
          <input id="chunkMaxChunks" type="number" min="1" max="64" step="1" value="12">
        </label>
        <label class="check"><input id="chunkTools" type="checkbox" checked> Tool-Spuren</label>
        <label class="check"><input id="chunkVariablePrompt" type="checkbox" checked> Prompt-Last</label>
        <label>Summary Mode
          <select id="chunkSummaryMode">
            <option value="stub">Stub schnell</option>
            <option value="real_llm">Real LLM</option>
          </select>
        </label>
        <button id="runChunking" class="run" type="button">Test Chunking</button>
      </div>
      <div class="chunk-form archive-rag-form">
        <label style="grid-column: 1 / -1;">Compact Instructions
          <textarea id="chunkCompactInstructions" placeholder="optional: preserve exact file paths, commands, unresolved decisions"></textarea>
        </label>
      </div>
      <div id="chunkingStats" class="chunk-stats"></div>
      <div id="chunkingActions" class="chunk-actions"></div>
      <div id="chunkingCompare" class="chunk-compare"></div>
      <pre id="chunkingRaw" class="chunk-raw">{}</pre>
    </section>
    <section class="chunk-lab" aria-label="Archive RAG Smoke">
      <div class="chunk-lab-head">
        <h2>Archive RAG Smoke</h2>
        <span id="archiveRagStatus" class="status">bereit</span>
      </div>
      <div class="chunk-form archive-rag-form">
        <label>Archive Key
          <input id="archiveRagKey" type="text" value="smoke_archive">
        </label>
        <label>Query
          <input id="archiveRagQuery" type="text" value="Welche Retrieval-Entscheidung steht im Archiv?">
        </label>
        <label>Limit
          <input id="archiveRagLimit" type="number" min="1" max="20" step="1" value="4">
        </label>
        <button id="runArchiveRagSmoke" class="run" type="button">Smoke</button>
      </div>
      <div class="chunk-form archive-rag-form">
        <label style="grid-column: 1 / -1;">Archive Text
          <textarea id="archiveRagText">Decision: query_archive should retrieve bounded chunks from a rag_api mirror and must not inject the whole raw archive into active context. Fallback: AlphaRavis vector_memory.py remains available when rag_api is missing.</textarea>
        </label>
      </div>
      <div id="archiveRagStats" class="chunk-stats"></div>
      <div id="archiveRagActions" class="chunk-actions"></div>
      <pre id="archiveRagRaw" class="chunk-raw">{}</pre>
    </section>
    <section class="chunk-lab" aria-label="Native Document RAG Smoke">
      <div class="chunk-lab-head">
        <h2>Native Document RAG Smoke</h2>
        <span id="nativeRagStatus" class="status">bereit</span>
      </div>
      <div class="chunk-form archive-rag-form">
        <label>Source Key
          <input id="nativeRagSourceKey" type="text" value="native_doc_smoke">
        </label>
        <label>Source Type
          <select id="nativeRagSourceType">
            <option value="large_paste">Large Paste</option>
            <option value="external_document">External Document</option>
            <option value="artifact_document">Artifact Document</option>
          </select>
        </label>
        <label>Query
          <input id="nativeRagQuery" type="text" value="Welche native AlphaRavis-RAG-Regel steht im Dokument?">
        </label>
        <label>Limit
          <input id="nativeRagLimit" type="number" min="1" max="20" step="1" value="4">
        </label>
        <button id="runNativeRagSmoke" class="run" type="button">Native Smoke</button>
      </div>
      <div class="chunk-form archive-rag-form">
        <label style="grid-column: 1 / -1;">Document Text
          <textarea id="nativeRagText">Runtime marker: NATIVE_PGVECTOR_RAG_SMOKE. Decision: explicit documents and large pasted sources should use AlphaRavis-owned pgvector by default. rag_api remains only an adapter or comparison backend.</textarea>
        </label>
      </div>
      <div id="nativeRagStats" class="chunk-stats"></div>
      <div id="nativeRagActions" class="chunk-actions"></div>
      <pre id="nativeRagRaw" class="chunk-raw">{}</pre>
    </section>
    <section class="chunk-lab" aria-label="Small Qwen Classifier Probe">
      <div class="chunk-lab-head">
        <h2>Small Qwen Classifier Probe</h2>
        <span id="ragClassifierStatus" class="status">bereit</span>
      </div>
      <div class="chunk-form archive-rag-form">
        <label>Mode
          <select id="ragClassifierMode">
            <option value="local_fallback">Local/Fallback</option>
            <option value="real_qwen">Real Qwen 2B</option>
          </select>
        </label>
        <label>Classifier URL
          <input id="ragClassifierBaseUrl" type="text" value="">
        </label>
        <label>Model
          <input id="ragClassifierModel" type="text" value="qwen3.5-2b">
        </label>
        <label>Timeout
          <input id="ragClassifierTimeout" type="number" min="1" max="60" step="1" value="8">
        </label>
        <button id="runRagClassifierProbe" class="run" type="button">Classifier Probe</button>
      </div>
      <div id="ragClassifierStats" class="chunk-stats"></div>
      <div id="ragClassifierActions" class="chunk-actions"></div>
      <pre id="ragClassifierRaw" class="chunk-raw">{}</pre>
    </section>
    <section class="chunk-lab" aria-label="Memory Embed Probe">
      <div class="chunk-lab-head">
        <h2>Memory Embed Tester</h2>
        <span id="memoryEmbedStatus" class="status">bereit</span>
      </div>
      <div class="chunk-form embed-form">
        <label>Base URL
          <input id="memoryEmbedBaseUrl" type="text" value="http://litellm:4000/v1">
        </label>
        <label>Model
          <input id="memoryEmbedModel" type="text" value="memory-embed">
        </label>
        <label>Backend
          <select id="memoryEmbedBackend">
            <option value="openai">OpenAI /v1</option>
            <option value="ollama_embed">Ollama /api/embed</option>
            <option value="ollama_embeddings">Ollama /api/embeddings</option>
          </select>
        </label>
        <label>Input
          <select id="memoryEmbedInputKind">
            <option value="text">Text</option>
            <option value="vision">Vision</option>
          </select>
        </label>
        <label>Start
          <input id="memoryEmbedStartChars" type="number" min="1" max="2000000" step="128" value="256">
        </label>
        <label>Max
          <input id="memoryEmbedMaxChars" type="number" min="1" max="2000000" step="4096" value="131072">
        </label>
        <label>Slow s
          <input id="memoryEmbedSlowSeconds" type="number" min="0.1" max="240" step="0.5" value="10">
        </label>
        <label>Timeout
          <input id="memoryEmbedTimeout" type="number" min="1" max="240" step="1" value="30">
        </label>
        <button id="runMemoryEmbedProbe" class="run" type="button">Probe</button>
      </div>
      <div class="chunk-form embed-text-form">
        <label>Probe Text
          <textarea id="memoryEmbedText">AlphaRavis memory embed probe. This text is repeated to test accepted input size, latency, and embedding dimensions.</textarea>
        </label>
        <label>Vision Data URL / Base64
          <textarea id="memoryEmbedImageData" placeholder="optional: data:image/png;base64,..."></textarea>
        </label>
      </div>
      <div id="memoryEmbedStats" class="chunk-stats"></div>
      <div id="memoryEmbedActions" class="chunk-actions"></div>
      <pre id="memoryEmbedRaw" class="chunk-raw">{}</pre>
    </section>
    <section class="chunk-lab" aria-label="RAG Load Probe">
      <div class="chunk-lab-head">
        <h2>RAG Load Probe</h2>
        <span id="ragLoadStatus" class="status">bereit</span>
      </div>
      <div class="chunk-form rag-load-form">
        <label>Embedding URL
          <input id="ragLoadEmbeddingBaseUrl" type="text" value="http://192.168.178.140:11434">
        </label>
        <label>Embedding Model
          <input id="ragLoadEmbeddingModel" type="text" value="qwen3-embedding:4b">
        </label>
        <label>Embedding Backend
          <select id="ragLoadEmbeddingBackend">
            <option value="ollama_embed">Ollama /api/embed</option>
            <option value="openai">OpenAI /v1</option>
            <option value="ollama_embeddings">Ollama /api/embeddings</option>
          </select>
        </label>
        <label>Reranker URL
          <input id="ragLoadRerankerUrl" type="text" value="http://192.168.178.140:8000">
        </label>
        <label>Reranker Model
          <input id="ragLoadRerankerModel" type="text" value="qwen3-reranker-0.6b">
        </label>
        <label>Token Steps
          <input id="ragLoadTokenSteps" type="text" value="400,1000,4000,10000,20000,40000">
        </label>
        <label>Docs
          <input id="ragLoadDocCount" type="number" min="1" max="50" step="1" value="10">
        </label>
        <label>Doc Chars
          <input id="ragLoadDocChars" type="number" min="80" max="4000" step="20" value="700">
        </label>
        <label>Timeout
          <input id="ragLoadTimeout" type="number" min="5" max="900" step="5" value="240">
        </label>
        <label>Bridge
          <select id="ragLoadBridgeMode">
            <option value="none">Keine</option>
            <option value="first_last">Erste+Letzte</option>
            <option value="all">Alle</option>
          </select>
        </label>
        <button id="runRagLoadProbe" class="run" type="button">Load Probe</button>
      </div>
      <div class="chunk-form archive-rag-form">
        <label>Query
          <input id="ragLoadQuery" type="text" value="How does AlphaRavis handle native pgvector retrieval and reranking?">
        </label>
        <label style="grid-column: 1 / -1;">Probe Text
          <textarea id="ragLoadText">AlphaRavis native RAG stores source-scoped chunks in pgvector, uses a durable embedding queue, and can rerank bounded retrieval hits before grounding the answer.</textarea>
        </label>
      </div>
      <div id="ragLoadStats" class="chunk-stats"></div>
      <div id="ragLoadActions" class="chunk-actions"></div>
      <pre id="ragLoadRaw" class="chunk-raw">{}</pre>
    </section>
    <section class="detail">
      <div class="detail-head">
        <div class="tabs">
          <button id="sendTab" class="active" type="button">Senden</button>
          <button id="receiveTab" type="button">Empfang</button>
          <button id="compressionTab" type="button">Kompression</button>
        </div>
        <div class="mode">
          <button id="contextMode" class="active" type="button">Nur Kontext</button>
          <button id="fullMode" type="button">Vollansicht</button>
        </div>
      </div>
      <pre id="detail" class="empty">Keine Anfrage ausgewählt.</pre>
    </section>
  </main>
  <script>
    const rowsEl = document.getElementById('rows');
    const detailEl = document.getElementById('detail');
    const statusEl = document.getElementById('status');
    const limitEl = document.getElementById('limit');
    const refreshBtn = document.getElementById('refresh');
    const clearBtn = document.getElementById('clear');
    const sendTab = document.getElementById('sendTab');
    const receiveTab = document.getElementById('receiveTab');
    const compressionTab = document.getElementById('compressionTab');
    const contextMode = document.getElementById('contextMode');
    const fullMode = document.getElementById('fullMode');
    const budgetGrid = document.getElementById('budgetGrid');
    const budgetStatus = document.getElementById('budgetStatus');
    const sourceGrid = document.getElementById('sourceGrid');
    const sourceStatus = document.getElementById('sourceStatus');
    const shrinkGrid = document.getElementById('shrinkGrid');
    const shrinkStatus = document.getElementById('shrinkStatus');
    const chunkingStatus = document.getElementById('chunkingStatus');
    const chunkTokens = document.getElementById('chunkTokens');
    const chunkActiveLimit = document.getElementById('chunkActiveLimit');
    const chunkSummaryContext = document.getElementById('chunkSummaryContext');
    const chunkMaxChunks = document.getElementById('chunkMaxChunks');
    const chunkTools = document.getElementById('chunkTools');
    const chunkVariablePrompt = document.getElementById('chunkVariablePrompt');
    const chunkSummaryMode = document.getElementById('chunkSummaryMode');
    const chunkCompactInstructions = document.getElementById('chunkCompactInstructions');
    const runChunking = document.getElementById('runChunking');
    const chunkingStats = document.getElementById('chunkingStats');
    const chunkingActions = document.getElementById('chunkingActions');
    const chunkingCompare = document.getElementById('chunkingCompare');
    const chunkingRaw = document.getElementById('chunkingRaw');
    const archiveRagStatus = document.getElementById('archiveRagStatus');
    const archiveRagKey = document.getElementById('archiveRagKey');
    const archiveRagQuery = document.getElementById('archiveRagQuery');
    const archiveRagLimit = document.getElementById('archiveRagLimit');
    const archiveRagText = document.getElementById('archiveRagText');
    const runArchiveRagSmoke = document.getElementById('runArchiveRagSmoke');
    const archiveRagStats = document.getElementById('archiveRagStats');
    const archiveRagActions = document.getElementById('archiveRagActions');
    const archiveRagRaw = document.getElementById('archiveRagRaw');
    const nativeRagStatus = document.getElementById('nativeRagStatus');
    const nativeRagSourceKey = document.getElementById('nativeRagSourceKey');
    const nativeRagSourceType = document.getElementById('nativeRagSourceType');
    const nativeRagQuery = document.getElementById('nativeRagQuery');
    const nativeRagLimit = document.getElementById('nativeRagLimit');
    const nativeRagText = document.getElementById('nativeRagText');
    const runNativeRagSmoke = document.getElementById('runNativeRagSmoke');
    const nativeRagStats = document.getElementById('nativeRagStats');
    const nativeRagActions = document.getElementById('nativeRagActions');
    const nativeRagRaw = document.getElementById('nativeRagRaw');
    const memoryEmbedStatus = document.getElementById('memoryEmbedStatus');
    const memoryEmbedBaseUrl = document.getElementById('memoryEmbedBaseUrl');
    const memoryEmbedModel = document.getElementById('memoryEmbedModel');
    const memoryEmbedBackend = document.getElementById('memoryEmbedBackend');
    const memoryEmbedInputKind = document.getElementById('memoryEmbedInputKind');
    const memoryEmbedStartChars = document.getElementById('memoryEmbedStartChars');
    const memoryEmbedMaxChars = document.getElementById('memoryEmbedMaxChars');
    const memoryEmbedSlowSeconds = document.getElementById('memoryEmbedSlowSeconds');
    const memoryEmbedTimeout = document.getElementById('memoryEmbedTimeout');
    const memoryEmbedText = document.getElementById('memoryEmbedText');
    const memoryEmbedImageData = document.getElementById('memoryEmbedImageData');
    const runMemoryEmbedProbe = document.getElementById('runMemoryEmbedProbe');
    const memoryEmbedStats = document.getElementById('memoryEmbedStats');
    const memoryEmbedActions = document.getElementById('memoryEmbedActions');
    const memoryEmbedRaw = document.getElementById('memoryEmbedRaw');
    const ragLoadStatus = document.getElementById('ragLoadStatus');
    const ragLoadEmbeddingBaseUrl = document.getElementById('ragLoadEmbeddingBaseUrl');
    const ragLoadEmbeddingModel = document.getElementById('ragLoadEmbeddingModel');
    const ragLoadEmbeddingBackend = document.getElementById('ragLoadEmbeddingBackend');
    const ragLoadRerankerUrl = document.getElementById('ragLoadRerankerUrl');
    const ragLoadRerankerModel = document.getElementById('ragLoadRerankerModel');
    const ragLoadTokenSteps = document.getElementById('ragLoadTokenSteps');
    const ragLoadDocCount = document.getElementById('ragLoadDocCount');
    const ragLoadDocChars = document.getElementById('ragLoadDocChars');
    const ragLoadTimeout = document.getElementById('ragLoadTimeout');
    const ragLoadBridgeMode = document.getElementById('ragLoadBridgeMode');
    const ragLoadQuery = document.getElementById('ragLoadQuery');
    const ragLoadText = document.getElementById('ragLoadText');
    const runRagLoadProbe = document.getElementById('runRagLoadProbe');
    const ragLoadStats = document.getElementById('ragLoadStats');
    const ragLoadActions = document.getElementById('ragLoadActions');
    const ragLoadRaw = document.getElementById('ragLoadRaw');
    const memoryReviewStatus = document.getElementById('memoryReviewStatus');
    const memoryReviewSource = document.getElementById('memoryReviewSource');
    const memoryReviewType = document.getElementById('memoryReviewType');
    const memoryReviewThread = document.getElementById('memoryReviewThread');
    const memoryReviewText = document.getElementById('memoryReviewText');
    const memoryReviewExtract = document.getElementById('memoryReviewExtract');
    const memoryReviewRefresh = document.getElementById('memoryReviewRefresh');
    const memoryReviewGrid = document.getElementById('memoryReviewGrid');
    let records = [];
    let selectedId = '';
    let activeTab = 'send';
    let activeMode = 'context';
    let chunkingRunId = '';
    let chunkingPollTimer = null;

    function fmtTime(value) {
      if (!value) return '';
      return new Date(value * 1000).toLocaleTimeString();
    }
    function short(value, length = 42) {
      const text = String(value || '');
      return text.length > length ? `${text.slice(0, length)}…` : text;
    }
    function pretty(value) {
      return JSON.stringify(value ?? {}, null, 2);
    }
    function budgetOf(record) {
      const receiveBudget = record?.receive?.context_budget || {};
      const stateProfile = record?.send?.langgraph_state_profile || {};
      const finalBudget = stateProfile.final_context_budget || {};
      return Object.keys(receiveBudget).length ? receiveBudget : finalBudget;
    }
    function compressionOf(record) {
      return record?.receive?.compression || {};
    }
    function sourceIngestsOf(record) {
      return record?.receive?.source_ingests || {};
    }
    function num(value) {
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : null;
    }
    function fmtNumber(value) {
      const parsed = num(value);
      return parsed === null ? (value ?? '') : parsed.toLocaleString();
    }
    function budgetHeadroom(budget) {
      const request = num(budget.request_tokens);
      const active = num(budget.effective_active_limit || budget.active_limit);
      return request !== null && active !== null ? active - request : null;
    }
    function shrinkPct(scope) {
      const before = num(scope?.tokens);
      const after = num(scope?.tokens_after);
      if (before === null || after === null || before <= 0) return null;
      return Math.max(0, Math.min(100, Math.round(((before - after) / before) * 100)));
    }
    function shrinkSummary(record) {
      const compression = compressionOf(record);
      return ['pre_run_compression', 'final_budget_rescue', 'post_run_compression', 'handoff_context']
        .map((name) => ({ name, data: compression[name] || {} }))
        .filter((scope) => Object.keys(scope.data).length)
        .map((scope) => ({
          scope: scope.name,
          tokens_before: scope.data.tokens,
          tokens_after: scope.data.tokens_after,
          shrink_pct: shrinkPct(scope.data),
          passes: scope.data.passes,
          budget_met: scope.data.budget_met,
          summary_failed: scope.data.summary_failed,
          summary_prompt_pruned: scope.data.summary_prompt_pruned,
          summary_chunking_used: scope.data.summary_chunking_used,
          summary_chunk_count: scope.data.summary_chunk_count,
          archive_key: scope.data.archive_key,
        }));
    }
    function budgetState(budget) {
      if (!budget || !Object.keys(budget).length) return { label: 'keine Daten', className: 'pill' };
      if (budget.hard_context_trim_used || budget.hard_rescue_needed) return { label: 'Hard Rescue', className: 'pill hard' };
      if (budget.final_budget_rescue_budget_met === true || budget.pre_run_compression_budget_met === true) {
        return { label: 'Rescue OK', className: 'pill ok' };
      }
      if (budget.compression_needed || budget.final_budget_rescue_used || budget.pre_run_compression_used) {
        return { label: 'Kompression', className: 'pill warn' };
      }
      const headroom = budgetHeadroom(budget);
      if (headroom !== null && headroom < 0) return { label: 'über Budget', className: 'pill hard' };
      if (headroom !== null && headroom < Math.max(1000, Number(budget.effective_active_limit || 0) * 0.10)) {
        return { label: 'knapp', className: 'pill warn' };
      }
      return { label: 'unter Budget', className: 'pill ok' };
    }
    function metric(label, value, className = '') {
      const el = document.createElement('div');
      el.className = `metric ${className}`.trim();
      const small = document.createElement('span');
      small.textContent = label;
      const strong = document.createElement('strong');
      strong.textContent = value ?? '';
      el.appendChild(small);
      el.appendChild(strong);
      return el;
    }
    function scopeLabel(name) {
      return {
        pre_run_compression: 'Pre-Run',
        final_budget_rescue: 'Final Rescue',
        post_run_compression: 'Post-Run',
        handoff_context: 'Handoff',
      }[name] || name;
    }
    function renderShrinking() {
      shrinkGrid.innerHTML = '';
      const record = selectedRecord();
      const compression = compressionOf(record);
      const scopes = ['pre_run_compression', 'final_budget_rescue', 'post_run_compression', 'handoff_context']
        .map((name) => ({ name, data: compression[name] || {} }))
        .filter((scope) => Object.keys(scope.data).length);
      if (!record || !scopes.length) {
        shrinkStatus.textContent = 'Keine Kompressionsdaten fuer diese Anfrage.';
        shrinkStatus.className = 'status';
        return;
      }
      shrinkStatus.textContent = compression.node ? `letzter Knoten: ${compression.node}` : `${scopes.length} Scope(s)`;
      shrinkStatus.className = 'status ok';
      for (const scope of scopes) {
        const data = scope.data;
        const pct = shrinkPct(data);
        const card = document.createElement('div');
        card.className = 'shrink-card';
        const head = document.createElement('div');
        head.className = 'shrink-card-head';
        const title = document.createElement('h3');
        title.textContent = scopeLabel(scope.name);
        const badge = document.createElement('span');
        badge.className = data.summary_failed ? 'pill hard' : (data.summary_chunking_used ? 'pill warn' : 'pill ok');
        badge.textContent = data.summary_failed ? 'Summary Fehler' : (data.summary_chunking_used ? 'Chunking' : 'One-shot');
        head.appendChild(title);
        head.appendChild(badge);
        const bar = document.createElement('div');
        bar.className = 'shrink-bar';
        const fill = document.createElement('span');
        fill.style.width = `${pct ?? 0}%`;
        bar.appendChild(fill);
        const grid = document.createElement('div');
        grid.className = 'shrink-metrics';
        [
          ['Before', fmtNumber(data.tokens)],
          ['After', fmtNumber(data.tokens_after)],
          ['Shrink', pct === null ? '' : `${pct}%`],
          ['Request', data.request_tokens_after ? `${fmtNumber(data.request_tokens)} -> ${fmtNumber(data.request_tokens_after)}` : ''],
          ['Passes', data.max_passes ? `${data.passes || 0}/${data.max_passes}` : (data.passes || '')],
          ['Budget OK', data.budget_met === undefined ? '' : (data.budget_met ? 'ja' : 'nein')],
          ['Compact Focus', data.compact_instructions ? short(data.compact_instructions, 80) : ''],
          ['Compress Limit', fmtNumber(data.compression_token_limit || '')],
          ['Summary Context', fmtNumber(data.summary_context_token_limit || '')],
          ['Head/Middle/Tail', `${data.head_message_count ?? ''}/${data.middle_message_count ?? ''}/${data.tail_message_count ?? ''}`],
          ['Middle Tokens', fmtNumber(data.middle_token_estimate)],
          ['Prompt Tokens', data.summary_prompt_tokens_estimate ? `${fmtNumber(data.summary_prompt_tokens_estimate)} / ${fmtNumber(data.summary_prompt_token_limit)}` : ''],
          ['Prompt Payload', data.summary_prompt_payload_token_limit ? fmtNumber(data.summary_prompt_payload_token_limit) : ''],
          ['Prompt Overhead', data.summary_prompt_overhead_tokens_estimate ? fmtNumber(data.summary_prompt_overhead_tokens_estimate) : ''],
          ['Prompt Pruned', data.summary_prompt_pruned === undefined ? '' : (data.summary_prompt_pruned ? 'ja' : 'nein')],
          ['Workflow Events', data.workflow_event_count ? fmtNumber(data.workflow_event_count) : ''],
          ['Tool Calls', data.workflow_tool_call_count ? fmtNumber(data.workflow_tool_call_count) : ''],
          ['Tool Results', data.workflow_tool_result_count ? fmtNumber(data.workflow_tool_result_count) : ''],
          ['Workflow Chars', data.workflow_event_chars ? fmtNumber(data.workflow_event_chars) : ''],
          ['Chunk Count', data.summary_chunk_count ? `${data.summary_chunk_count}` : ''],
          ['Chunk Payload', data.summary_chunk_payload_token_limit ? fmtNumber(data.summary_chunk_payload_token_limit) : ''],
          ['Chunk Overhead', data.summary_chunk_prompt_overhead_tokens ? fmtNumber(data.summary_chunk_prompt_overhead_tokens) : ''],
          ['Chunk Omitted', fmtNumber(data.summary_chunk_omitted_chars || '')],
          ['Chunk Output', fmtNumber(data.summary_chunk_output_tokens || '')],
          ['Synth Pruned', data.summary_chunk_synthesis_pruned === undefined ? '' : (data.summary_chunk_synthesis_pruned ? 'ja' : 'nein')],
          ['Synth Payload', data.summary_chunk_synthesis_payload_token_limit ? fmtNumber(data.summary_chunk_synthesis_payload_token_limit) : ''],
          ['Archive', short(data.archive_key || '', 24)],
        ].forEach(([name, value]) => {
          if (value !== '') grid.appendChild(metric(name, value));
        });
        card.appendChild(head);
        card.appendChild(bar);
        card.appendChild(grid);
        if (data.workflow_event_preview) {
          const workflow = document.createElement('details');
          workflow.className = 'workflow-log';
          const summary = document.createElement('summary');
          summary.textContent = `Workflow / Tool Events (${fmtNumber(data.workflow_event_count || 0)})`;
          const preview = document.createElement('pre');
          preview.textContent = data.workflow_event_preview;
          workflow.append(summary, preview);
          card.appendChild(workflow);
        }
        shrinkGrid.appendChild(card);
      }
    }
    function renderSourceIngests() {
      sourceGrid.innerHTML = '';
      const record = selectedRecord();
      const ingests = sourceIngestsOf(record);
      const groups = ['large_paste_ingests', 'document_ingests']
        .flatMap((field) => (Array.isArray(ingests[field]) ? ingests[field].map((item) => ({ field, item })) : []));
      if (!record || !groups.length) {
        sourceStatus.textContent = 'Keine Quellen-Events fuer diese Anfrage.';
        sourceStatus.className = 'status';
        return;
      }
      sourceStatus.textContent = ingests.node ? `${groups.length} Quelle(n); letzter Knoten: ${ingests.node}` : `${groups.length} Quelle(n)`;
      sourceStatus.className = 'status ok';
      for (const { field, item } of groups) {
        const card = document.createElement('div');
        card.className = 'source-card';
        const head = document.createElement('div');
        head.className = 'source-card-head';
        const title = document.createElement('h3');
        title.textContent = item.title || item.source_key || field;
        const badge = document.createElement('span');
        const status = String(item.index_status || item.latest_event?.event || '');
        badge.className = status === 'failed' ? 'pill hard' : (status === 'skipped' ? 'pill warn' : 'pill ok');
        badge.textContent = item.message_replaced ? 'Marker aktiv' : (status || 'Quelle');
        head.append(title, badge);
        const source = document.createElement('p');
        source.textContent = item.source_key || '';
        const backends = [...(item.indexed_backends || []), ...(item.queued_backends || [])].filter(Boolean).join(', ');
        const queued = (item.queued_backends || []).filter(Boolean).join(', ');
        const grid = document.createElement('div');
        grid.className = 'source-metrics';
        [
          ['Status', status],
          ['Typ', item.content_type || item.source_type || ''],
          ['Intent', item.paste_intent || ''],
          ['Chars', fmtNumber(item.content_chars || '')],
          ['Indexed', fmtNumber(item.indexed_content_chars || '')],
          ['Chunks', item.chunk_count ? `${fmtNumber(item.indexed_chunk_count || 0)}/${fmtNumber(item.chunk_count)}` : ''],
          ['Backends', backends],
          ['Queue', queued],
          ['RAG aktiv', item.rag_active === undefined ? '' : (item.rag_active ? 'ja' : 'nein')],
          ['Manual', item.manual_rag_block ? 'ja' : 'nein'],
          ['Elapsed', item.elapsed_seconds ? `${item.elapsed_seconds}s` : ''],
          ['Reason', item.skip_reason || ''],
        ].forEach(([name, value]) => {
          if (value !== '') grid.appendChild(metric(name, value));
        });
        card.append(head, source, grid);
        sourceGrid.appendChild(card);
      }
    }
    function renderEmbeddingQueue(result) {
      embeddingQueueGrid.innerHTML = '';
      embeddingQueueActions.innerHTML = '';
      if (!result || result.status === 'unavailable' || result.status === 'error') {
        embeddingQueueStatus.textContent = result?.error || 'Queue nicht verfuegbar.';
        embeddingQueueStatus.className = 'status warn';
        return;
      }
      const counts = result.counts || {};
      const active = Number(result.active_count || 0);
      embeddingQueueStatus.textContent = active
        ? `${fmtNumber(active)} aktive Queue-Jobs`
        : 'idle';
      embeddingQueueStatus.className = `status ${active ? 'warn' : 'ok'}`;
      [
        ['Pending', fmtNumber(counts.pending || 0), counts.pending ? 'warn' : 'ok'],
        ['Running', fmtNumber(counts.running || 0), counts.running ? 'warn' : ''],
        ['Failed', fmtNumber(counts.failed || 0), counts.failed ? 'hard' : ''],
        ['Done', fmtNumber(counts.done || 0), 'ok'],
        ['Progress', result.progress === null || result.progress === undefined ? '' : `${Math.round(Number(result.progress) * 100)}%`],
        ['Table', result.table || ''],
      ].forEach(([name, value, className]) => {
        if (value !== '') embeddingQueueGrid.appendChild(metric(name, value, className || ''));
      });
      for (const item of (result.recent_active || [])) {
        const row = document.createElement('div');
        const status = document.createElement('span');
        status.className = item.status === 'failed' ? 'pill hard' : (item.status === 'running' ? 'pill warn' : 'pill');
        status.textContent = item.status || '';
        const source = document.createElement('span');
        source.textContent = `${item.source_type || ''}:${item.source_key || ''}`;
        const detail = document.createElement('span');
        detail.className = 'muted';
        detail.textContent = [item.title || '', item.last_error || ''].filter(Boolean).join(' | ');
        row.append(status, source, detail);
        embeddingQueueActions.appendChild(row);
      }
      for (const item of (result.source_progress || [])) {
        const card = document.createElement('div');
        card.className = 'source-card';
        const head = document.createElement('div');
        head.className = 'source-card-head';
        const title = document.createElement('h3');
        title.textContent = item.title || item.source_key || item.id || 'Queue Source';
        const badge = document.createElement('span');
        badge.className = item.status === 'failed' ? 'pill hard' : (item.status === 'running' ? 'pill warn' : 'pill');
        badge.textContent = item.status || '';
        head.append(title, badge);
        const source = document.createElement('p');
        source.textContent = `${item.source_type || ''}:${item.source_key || ''}`;
        const planned = Number(item.planned_chunks || 0);
        const completed = Number(item.completed_chunks || 0);
        const percent = item.progress === null || item.progress === undefined ? '' : `${Math.round(Number(item.progress) * 100)}%`;
        const grid = document.createElement('div');
        grid.className = 'source-metrics';
        [
          ['Chunks', planned ? `${completed}/${planned}` : ''],
          ['Progress', percent],
          ['Thread', short(item.thread_id || '', 18)],
          ['Event', item.last_event || ''],
        ].forEach(([name, value]) => {
          if (value !== '') grid.appendChild(metric(name, value));
        });
        card.append(head, source, grid);
        embeddingQueueGrid.appendChild(card);
      }
    }
    async function refreshEmbeddingQueue() {
      try {
        const response = await fetch('/api/embedding-queue/status', { cache: 'no-store' });
        if (!response.ok) throw new Error(await response.text());
        renderEmbeddingQueue(await response.json());
      } catch (error) {
        embeddingQueueStatus.textContent = error.message;
        embeddingQueueStatus.className = 'status hard';
      }
    }
    function renderResumeRuns(result) {
      resumeGrid.innerHTML = '';
      const items = result?.items || [];
      resumeStatus.textContent = items.length ? `${items.length} wartende Runs` : 'keine wartenden Runs';
      resumeStatus.className = `status ${items.length ? 'warn' : 'ok'}`;
      for (const item of items) {
        const card = document.createElement('div');
        card.className = 'source-card';
        const head = document.createElement('div');
        head.className = 'source-card-head';
        const title = document.createElement('h3');
        title.textContent = item.thread_key || item.thread_id || 'Thread';
        const badge = document.createElement('span');
        badge.className = 'pill warn';
        badge.textContent = item.phase || item.status || 'awaiting_resume';
        head.append(title, badge);
        const source = document.createElement('p');
        source.textContent = item.current_task_brief || item.error || item.planner_context || '';
        const grid = document.createElement('div');
        grid.className = 'source-metrics';
        [
          ['Thread ID', short(item.thread_id || '', 24)],
          ['Agent', item.active_agent || ''],
          ['Toolsets', Array.isArray(item.selected_toolsets) ? item.selected_toolsets.join(', ') : ''],
          ['Resume Text', item.resume_text || 'ja, weiter'],
        ].forEach(([name, value]) => {
          if (value !== '') grid.appendChild(metric(name, value));
        });
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = 'Thread übernehmen';
        button.addEventListener('click', () => {
          ragPinsThread.value = item.thread_id || '';
          navigator.clipboard?.writeText(item.resume_text || 'ja, weiter').catch(() => {});
        });
        card.append(head, source, grid, button);
        resumeGrid.appendChild(card);
      }
    }
    async function refreshResumeRuns() {
      try {
        const response = await fetch('/api/resume-runs', { cache: 'no-store' });
        if (!response.ok) throw new Error(await response.text());
        renderResumeRuns(await response.json());
      } catch (error) {
        resumeStatus.textContent = error.message;
        resumeStatus.className = 'status hard';
      }
    }
    function renderRagPins(result) {
      ragPinsGrid.innerHTML = '';
      const items = result?.pins ? [result.pins] : (result?.items || []);
      ragPinsStatus.textContent = items.length ? `${items.length} Pin-Satz` : 'keine Pins';
      ragPinsStatus.className = `status ${items.some((item) => item.rag_active) ? 'ok' : ''}`;
      for (const item of items) {
        const card = document.createElement('div');
        card.className = 'source-card';
        const head = document.createElement('div');
        head.className = 'source-card-head';
        const title = document.createElement('h3');
        title.textContent = item.thread_id || 'Thread';
        const badge = document.createElement('span');
        badge.className = item.rag_active ? 'pill ok' : 'pill';
        badge.textContent = item.rag_active ? 'aktiv' : 'inaktiv';
        head.append(title, badge);
        const source = document.createElement('p');
        source.textContent = (item.active_source_keys || []).join(', ');
        const grid = document.createElement('div');
        grid.className = 'source-metrics';
        [
          ['Source Keys', (item.active_source_keys || []).length],
          ['RAG File IDs', (item.active_rag_file_ids || []).length],
          ['Archive Mode', item.archive_rag_mode || 'tool_only'],
          ['Updated', item.updated_at ? fmtTime(item.updated_at) : ''],
        ].forEach(([name, value]) => {
          if (value !== '') grid.appendChild(metric(name, value));
        });
        card.append(head, source, grid);
        ragPinsGrid.appendChild(card);
      }
    }
    async function refreshRagPins() {
      try {
        const params = ragPinsThread.value.trim() ? `?thread_id=${encodeURIComponent(ragPinsThread.value.trim())}` : '';
        const response = await fetch(`/api/rag-pins${params}`, { cache: 'no-store' });
        if (!response.ok) throw new Error(await response.text());
        renderRagPins(await response.json());
      } catch (error) {
        ragPinsStatus.textContent = error.message;
        ragPinsStatus.className = 'status hard';
      }
    }
    async function writeRagPins(action) {
      const source = ragPinsSource.value.trim();
      const body = {
        thread_id: ragPinsThread.value.trim(),
        archive_rag_mode: ragPinsMode.value,
        add_source_keys: action === 'pin' && source ? [source] : [],
        remove_source_keys: action === 'unpin' && source ? [source] : [],
        clear_all: action === 'clear',
      };
      if (!body.thread_id) {
        ragPinsStatus.textContent = 'thread_id fehlt';
        ragPinsStatus.className = 'status hard';
        return;
      }
      const response = await fetch('/api/rag-pins', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!response.ok) throw new Error(await response.text());
      renderRagPins(await response.json());
    }
    function renderMemoryReview(result) {
      memoryReviewGrid.innerHTML = '';
      const items = result?.items || [];
      memoryReviewStatus.textContent = items.length ? `${items.length} Kandidat(en)` : 'keine offenen Kandidaten';
      memoryReviewStatus.className = `status ${items.length ? 'warn' : 'ok'}`;
      for (const item of items) {
        const card = document.createElement('div');
        card.className = 'source-card';
        const head = document.createElement('div');
        head.className = 'source-card-head';
        const title = document.createElement('h3');
        title.textContent = item.memory_type || 'fact';
        const badge = document.createElement('span');
        badge.className = item.status === 'accepted' ? 'pill ok' : (item.status === 'rejected' ? 'pill hard' : 'pill warn');
        badge.textContent = item.status || 'pending';
        head.append(title, badge);
        const source = document.createElement('p');
        source.textContent = item.memory || '';
        const grid = document.createElement('div');
        grid.className = 'source-metrics';
        [
          ['Confidence', item.confidence === undefined ? '' : Math.round(Number(item.confidence) * 100) + '%'],
          ['Source', item.source_key || item.source_type || ''],
          ['Thread', short(item.thread_id || '', 24)],
          ['Updated', item.updated_at ? fmtTime(item.updated_at) : ''],
        ].forEach(([name, value]) => {
          if (value !== '') grid.appendChild(metric(name, value));
        });
        const preview = document.createElement('pre');
        preview.className = 'chunk-raw';
        preview.textContent = item.source_preview || '';
        const actions = document.createElement('div');
        actions.className = 'chunk-actions';
        const accept = document.createElement('button');
        accept.type = 'button';
        accept.textContent = 'Accept';
        accept.disabled = item.status === 'accepted';
        accept.addEventListener('click', () => decideMemoryReview(item.candidate_id, 'accept'));
        const reject = document.createElement('button');
        reject.type = 'button';
        reject.textContent = 'Reject';
        reject.disabled = item.status === 'rejected';
        reject.addEventListener('click', () => decideMemoryReview(item.candidate_id, 'reject'));
        actions.append(accept, reject);
        card.append(head, source, grid, preview, actions);
        memoryReviewGrid.appendChild(card);
      }
    }
    async function refreshMemoryReview() {
      const response = await fetch('/api/curated-memory/review?status=pending&limit=50', { cache: 'no-store' });
      if (!response.ok) throw new Error(await response.text());
      renderMemoryReview(await response.json());
    }
    async function extractMemoryReview() {
      const text = memoryReviewText.value.trim();
      if (!text) {
        memoryReviewStatus.textContent = 'Text fehlt';
        memoryReviewStatus.className = 'status hard';
        return;
      }
      const response = await fetch('/api/curated-memory/review/extract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          source_key: memoryReviewSource.value.trim(),
          source_type: memoryReviewType.value.trim() || 'thread',
          thread_id: memoryReviewThread.value.trim(),
        }),
      });
      if (!response.ok) throw new Error(await response.text());
      renderMemoryReview(await response.json());
    }
    async function decideMemoryReview(candidateId, action) {
      const response = await fetch(`/api/curated-memory/review/${encodeURIComponent(candidateId)}/${action}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ candidate_id: candidateId }),
      });
      if (!response.ok) throw new Error(await response.text());
      await refreshMemoryReview();
    }
    function chunkMetric(label, value, className = '') {
      return metric(label, value, className);
    }
    function renderChunkingRun(run) {
      const result = run?.result || {};
      const meta = result.archive_metadata || {};
      const acceptance = result.acceptance || {};
      const comparison = result.comparison || {};
      chunkingStatus.textContent = run?.status === 'completed'
        ? `fertig in ${run.elapsed_seconds || 0}s`
        : (run?.status || 'bereit');
      chunkingStatus.className = `status ${run?.status === 'failed' ? 'hard' : (run?.status === 'completed' ? 'ok' : 'warn')}`;
      chunkingStats.innerHTML = '';
      [
        ['Status', run?.status || '', run?.status === 'failed' ? 'hard' : (result.acceptance_ok ? 'ok' : 'warn')],
        ['Summary Mode', result.config?.summary_mode || ''],
        ['Compact Focus', result.config?.compact_instructions_chars ? `${fmtNumber(result.config.compact_instructions_chars)} chars` : ''],
        ['Summary Model', result.config?.summary_model || ''],
        ['Input Tokens', fmtNumber(result.input?.estimated_tokens || '')],
        ['Prepared Tokens', fmtNumber(result.input?.prepared_summary_tokens || '')],
        ['Before', fmtNumber(result.token_estimate_before || '')],
        ['After', fmtNumber(result.token_estimate_after || '')],
        ['Chunking', meta.summary_chunking_used === undefined ? '' : (meta.summary_chunking_used ? 'ja' : 'nein'), meta.summary_chunking_used ? 'ok' : 'warn'],
        ['Chunk Count', fmtNumber(meta.summary_chunk_count || '')],
        ['Chunk Omitted', fmtNumber(meta.summary_chunk_omitted_chars || 0), Number(meta.summary_chunk_omitted_chars || 0) > 0 ? 'hard' : 'ok'],
        ['Summary Failed', meta.summary_failed === undefined ? '' : (meta.summary_failed ? 'ja' : 'nein'), meta.summary_failed ? 'hard' : 'ok'],
        ['Prompt Payload', fmtNumber(meta.summary_chunk_payload_token_limit || meta.summary_prompt_payload_token_limit || '')],
        ['Prompt Overhead', fmtNumber(meta.summary_chunk_prompt_overhead_tokens || meta.summary_prompt_overhead_tokens_estimate || '')],
        ['Synth Pruned', meta.summary_chunk_synthesis_pruned === undefined ? '' : (meta.summary_chunk_synthesis_pruned ? 'ja' : 'nein'), meta.summary_chunk_synthesis_pruned ? 'warn' : 'ok'],
        ['Tool Pruned', fmtNumber(result.tool_stats?.pruned_tool_count || 0)],
        ['Tool Dedup', fmtNumber(result.tool_stats?.deduped_tool_count || 0)],
        ['Args Trunc', fmtNumber(result.tool_stats?.tool_args_truncated_count || 0)],
        ['Workflow Events', fmtNumber(result.tool_stats?.workflow_event_count || 0)],
        ['Workflow Chars', fmtNumber(result.tool_stats?.workflow_event_chars || 0)],
        ['Acceptance', result.acceptance_ok === undefined ? '' : (result.acceptance_ok ? 'ok' : 'prüfen'), result.acceptance_ok ? 'ok' : 'warn'],
      ].forEach(([label, value, className]) => chunkingStats.appendChild(chunkMetric(label, value, className || '')));
      chunkingActions.innerHTML = '';
      const actions = Array.isArray(run?.actions) ? run.actions : [];
      for (const action of actions.slice(-80)) {
        const row = document.createElement('div');
        const time = document.createElement('span');
        time.className = 'muted';
        time.textContent = `${Number(action.t || 0).toFixed(3)}s`;
        const event = document.createElement('span');
        event.textContent = action.event || '';
        const detail = document.createElement('span');
        detail.className = 'muted';
        detail.textContent = Object.entries(action)
          .filter(([key]) => !['t', 'event'].includes(key))
          .map(([key, value]) => `${key}=${typeof value === 'string' ? value : JSON.stringify(value)}`)
          .join(' ');
        row.append(time, event, detail);
        chunkingActions.appendChild(row);
      }
      chunkingCompare.innerHTML = '';
      function comparePanel(title, payload, open) {
        const details = document.createElement('details');
        details.open = Boolean(open);
        const summary = document.createElement('summary');
        const flags = [];
        if (payload?.chars !== undefined) flags.push(`${fmtNumber(payload.chars)} chars`);
        if (payload?.truncated) flags.push(`${fmtNumber(payload.omitted_chars)} omitted`);
        summary.textContent = flags.length ? `${title} (${flags.join(', ')})` : title;
        const pre = document.createElement('pre');
        pre.textContent = payload?.text || '(leer)';
        details.append(summary, pre);
        return details;
      }
      chunkingCompare.appendChild(comparePanel('Vorher: Prepared Compression Input', comparison.before_prepared_summary_input || {}, false));
      chunkingCompare.appendChild(comparePanel('Nachher: Finale Summary', comparison.after_summary || {}, true));
      chunkingRaw.textContent = pretty({
        id: run?.id,
        status: run?.status,
        error: run?.error,
        config: result.config,
        acceptance,
        archive_metadata: meta,
        comparison_stats: {
          before_tokens_estimate: comparison.before_tokens_estimate,
          after_tokens_estimate: comparison.after_tokens_estimate,
          active_tokens_before: comparison.active_tokens_before,
          active_tokens_after: comparison.active_tokens_after,
          active_shrink_ratio: comparison.active_shrink_ratio,
        },
        summary_calls: result.summary_calls,
        summary_preview: result.summary_preview,
        summary_chunk_omitted_chars_zero: acceptance.summary_chunk_omitted_chars_zero,
      });
    }
    async function pollChunkingRun(runId) {
      const response = await fetch(`/api/chunking/runs/${encodeURIComponent(runId)}`, { cache: 'no-store' });
      if (!response.ok) throw new Error(await response.text());
      const run = await response.json();
      renderChunkingRun(run);
      if (!['completed', 'failed'].includes(run.status)) {
        chunkingPollTimer = window.setTimeout(() => pollChunkingRun(runId).catch((error) => {
          chunkingStatus.textContent = error.message;
          chunkingStatus.className = 'status hard';
        }), 700);
      }
    }
    async function startChunkingRun() {
      if (chunkingPollTimer) window.clearTimeout(chunkingPollTimer);
      runChunking.disabled = true;
      chunkingStatus.textContent = 'starte...';
      chunkingStatus.className = 'status warn';
      try {
        const response = await fetch('/api/chunking/runs', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            approx_tokens: Number(chunkTokens.value || 300000),
            token_limit: Number(chunkActiveLimit.value || 64000),
            summary_context_token_limit: Number(chunkSummaryContext.value || 128000),
            max_chunks: Number(chunkMaxChunks.value || 12),
            include_tools: chunkTools.checked,
            variable_prompt_load: chunkVariablePrompt.checked,
            summary_mode: chunkSummaryMode.value || 'stub',
            compact_instructions: chunkCompactInstructions.value || '',
          }),
        });
        if (!response.ok) throw new Error(await response.text());
        const run = await response.json();
        chunkingRunId = run.id;
        renderChunkingRun(run);
        await pollChunkingRun(chunkingRunId);
      } catch (error) {
        chunkingStatus.textContent = error.message;
        chunkingStatus.className = 'status hard';
      } finally {
        runChunking.disabled = false;
      }
    }
    function renderArchiveRagSmoke(result) {
      const hasErrors = Array.isArray(result.errors) && result.errors.length > 0;
      archiveRagStatus.textContent = result.acceptance_ok ? `ok in ${result.elapsed_seconds || 0}s` : (hasErrors ? 'runtime prüfen' : 'prüfen');
      archiveRagStatus.className = `status ${result.acceptance_ok ? 'ok' : (hasErrors ? 'hard' : 'warn')}`;
      archiveRagStats.innerHTML = '';
      [
        ['RAG File ID', result.rag_file_id || ''],
        ['Mirror', result.acceptance?.mirror_status_ok ? 'ok' : 'prüfen', result.acceptance?.mirror_status_ok ? 'ok' : 'warn'],
        ['Hits', fmtNumber(result.hit_count || 0), Number(result.hit_count || 0) > 0 ? 'ok' : 'hard'],
        ['Rule Found', result.acceptance?.retrieved_expected_archive_rule ? 'ja' : 'nein', result.acceptance?.retrieved_expected_archive_rule ? 'ok' : 'warn'],
        ['Runtime', result.acceptance?.no_runtime_errors ? 'ok' : 'Fehler', result.acceptance?.no_runtime_errors ? 'ok' : 'hard'],
      ].forEach(([label, value, className]) => archiveRagStats.appendChild(chunkMetric(label, value, className || '')));
      archiveRagActions.innerHTML = '';
      for (const action of (result.actions || [])) {
        const row = document.createElement('div');
        const time = document.createElement('span');
        time.className = 'muted';
        time.textContent = `${Number(action.t || 0).toFixed(3)}s`;
        const event = document.createElement('span');
        event.textContent = action.event || '';
        const detail = document.createElement('span');
        detail.className = 'muted';
        detail.textContent = Object.entries(action)
          .filter(([key]) => !['t', 'event'].includes(key))
          .map(([key, value]) => `${key}=${typeof value === 'string' ? value : JSON.stringify(value)}`)
          .join(' ');
        row.append(time, event, detail);
        archiveRagActions.appendChild(row);
      }
      archiveRagRaw.textContent = pretty(result);
    }
    async function startArchiveRagSmoke() {
      runArchiveRagSmoke.disabled = true;
      archiveRagStatus.textContent = 'starte...';
      archiveRagStatus.className = 'status warn';
      try {
        const response = await fetch('/api/archive-rag-smoke', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            archive_key: archiveRagKey.value || '',
            archive_text: archiveRagText.value || '',
            query: archiveRagQuery.value || '',
            limit: Number(archiveRagLimit.value || 4),
          }),
        });
        if (!response.ok) throw new Error(await response.text());
        renderArchiveRagSmoke(await response.json());
      } catch (error) {
        archiveRagStatus.textContent = error.message;
        archiveRagStatus.className = 'status hard';
      } finally {
        runArchiveRagSmoke.disabled = false;
      }
    }
    function renderNativeRagSmoke(result) {
      const hasErrors = Array.isArray(result.errors) && result.errors.length > 0;
      nativeRagStatus.textContent = result.acceptance_ok ? `ok in ${result.elapsed_seconds || 0}s` : (hasErrors ? 'runtime prüfen' : 'prüfen');
      nativeRagStatus.className = `status ${result.acceptance_ok ? 'ok' : (hasErrors ? 'hard' : 'warn')}`;
      nativeRagStats.innerHTML = '';
      [
        ['Source Key', result.source_key || ''],
        ['Backend', result.acceptance?.pgvector_backend_selected ? 'pgvector' : 'prüfen', result.acceptance?.pgvector_backend_selected ? 'ok' : 'warn'],
        ['rag_api', result.acceptance?.rag_api_not_used ? 'nicht benutzt' : 'prüfen', result.acceptance?.rag_api_not_used ? 'ok' : 'warn'],
        ['Active Source', result.acceptance?.active_source_key_recorded ? 'ja' : 'nein', result.acceptance?.active_source_key_recorded ? 'ok' : 'warn'],
        ['Chunks', fmtNumber(result.hit_count || 0), Number(result.hit_count || 0) > 0 ? 'ok' : 'hard'],
        ['Marker', result.acceptance?.retrieved_expected_marker ? 'ja' : 'nein', result.acceptance?.retrieved_expected_marker ? 'ok' : 'warn'],
        ['Runtime', result.acceptance?.no_runtime_errors ? 'ok' : 'Fehler', result.acceptance?.no_runtime_errors ? 'ok' : 'hard'],
      ].forEach(([label, value, className]) => nativeRagStats.appendChild(chunkMetric(label, value, className || '')));
      nativeRagActions.innerHTML = '';
      for (const action of (result.actions || [])) {
        const row = document.createElement('div');
        const time = document.createElement('span');
        time.className = 'muted';
        time.textContent = `${Number(action.t || 0).toFixed(3)}s`;
        const event = document.createElement('span');
        event.textContent = action.event || '';
        const detail = document.createElement('span');
        detail.className = 'muted';
        detail.textContent = Object.entries(action)
          .filter(([key]) => !['t', 'event'].includes(key))
          .map(([key, value]) => `${key}=${typeof value === 'string' ? value : JSON.stringify(value)}`)
          .join(' ');
        row.append(time, event, detail);
        nativeRagActions.appendChild(row);
      }
      nativeRagRaw.textContent = pretty(result);
    }
    async function startNativeRagSmoke() {
      runNativeRagSmoke.disabled = true;
      nativeRagStatus.textContent = 'starte...';
      nativeRagStatus.className = 'status warn';
      try {
        const response = await fetch('/api/native-document-rag-smoke', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            source_key: nativeRagSourceKey.value || '',
            source_type: nativeRagSourceType.value || 'large_paste',
            document_text: nativeRagText.value || '',
            query: nativeRagQuery.value || '',
            limit: Number(nativeRagLimit.value || 4),
          }),
        });
        if (!response.ok) throw new Error(await response.text());
        renderNativeRagSmoke(await response.json());
      } catch (error) {
        nativeRagStatus.textContent = error.message;
        nativeRagStatus.className = 'status hard';
      } finally {
        runNativeRagSmoke.disabled = false;
      }
    }
    function renderRagClassifierProbe(result) {
      const passed = result.status === 'passed';
      const partial = result.status === 'partial';
      ragClassifierStatus.textContent = `${result.status || 'unknown'} in ${result.elapsed_seconds || 0}s`;
      ragClassifierStatus.className = `status ${passed ? 'ok' : (partial ? 'warn' : 'hard')}`;
      ragClassifierStats.innerHTML = '';
      [
        ['Cases', `${fmtNumber(result.ok_case_count || 0)} / ${fmtNumber(result.case_count || 0)}`, passed ? 'ok' : 'warn'],
        ['Mode', result.mode || ''],
        ['Endpoint', result.classifier_base_url || ''],
        ['Model', result.classifier_model || ''],
      ].forEach(([label, value, className]) => ragClassifierStats.appendChild(chunkMetric(label, value, className || '')));
      ragClassifierActions.innerHTML = '';
      for (const item of (result.results || [])) {
        const row = document.createElement('div');
        const badge = document.createElement('span');
        badge.className = item.ok ? 'pill ok' : 'pill hard';
        badge.textContent = item.case || '';
        const event = document.createElement('span');
        event.textContent = `${item.classification || ''}${item.fallback_used ? ' (fallback)' : ''}`;
        const detail = document.createElement('span');
        detail.className = 'muted';
        detail.textContent = [item.source || '', item.query || '', item.fallback_reason || ''].filter(Boolean).join(' | ').slice(0, 220);
        row.append(badge, event, detail);
        ragClassifierActions.appendChild(row);
      }
      ragClassifierRaw.textContent = pretty(result);
    }
    async function startRagClassifierProbe() {
      runRagClassifierProbe.disabled = true;
      ragClassifierStatus.textContent = 'starte...';
      ragClassifierStatus.className = 'status warn';
      try {
        const response = await fetch('/api/rag-classifier-probe', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            mode: ragClassifierMode.value || 'local_fallback',
            classifier_base_url: ragClassifierBaseUrl.value || '',
            classifier_model: ragClassifierModel.value || '',
            timeout_seconds: Number(ragClassifierTimeout.value || 8),
          }),
        });
        if (!response.ok) throw new Error(await response.text());
        renderRagClassifierProbe(await response.json());
      } catch (error) {
        ragClassifierStatus.textContent = error.message;
        ragClassifierStatus.className = 'status hard';
      } finally {
        runRagClassifierProbe.disabled = false;
      }
    }
    function renderMemoryEmbedProbe(result) {
      const passed = result.status === 'passed';
      const partial = result.status === 'partial';
      memoryEmbedStatus.textContent = `${result.status || 'unknown'} in ${result.elapsed_seconds || 0}s`;
      memoryEmbedStatus.className = `status ${passed ? 'ok' : (partial ? 'warn' : 'hard')}`;
      memoryEmbedStats.innerHTML = '';
      [
        ['Backend', result.backend || ''],
        ['Endpoint', result.endpoint || ''],
        ['Accepted', fmtNumber(result.accepted_step_count || 0), Number(result.accepted_step_count || 0) > 0 ? 'ok' : 'hard'],
        ['Max Chars', fmtNumber(result.max_accepted_chars || 0), result.max_accepted_chars ? 'ok' : 'hard'],
        ['Max Tokens', fmtNumber(result.max_accepted_approx_tokens || 0), result.max_accepted_approx_tokens ? 'ok' : 'hard'],
        ['Stop', result.stop_reason || '', result.stop_reason === 'completed' ? 'ok' : 'warn'],
      ].forEach(([label, value, className]) => memoryEmbedStats.appendChild(chunkMetric(label, value, className || '')));
      memoryEmbedActions.innerHTML = '';
      for (const item of (result.results || [])) {
        const row = document.createElement('div');
        const time = document.createElement('span');
        time.className = 'muted';
        time.textContent = `${Number(item.elapsed_seconds || 0).toFixed(3)}s`;
        const event = document.createElement('span');
        event.textContent = item.ok ? `${fmtNumber(item.chars)} chars ok` : `${fmtNumber(item.chars)} chars failed`;
        const detail = document.createElement('span');
        detail.className = 'muted';
        detail.textContent = item.ok
          ? `dim=${item.embedding_dimensions || ''} slow=${Boolean(item.slow)} status=${item.status_code || ''}`
          : `${item.status_code || ''} ${item.error || ''}`;
        row.append(time, event, detail);
        memoryEmbedActions.appendChild(row);
      }
      memoryEmbedRaw.textContent = pretty(result);
    }
    async function startMemoryEmbedProbe() {
      runMemoryEmbedProbe.disabled = true;
      memoryEmbedStatus.textContent = 'starte...';
      memoryEmbedStatus.className = 'status warn';
      try {
        const response = await fetch('/api/memory-embed-probe', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            base_url: memoryEmbedBaseUrl.value || '',
            model: memoryEmbedModel.value || '',
            backend: memoryEmbedBackend.value || 'openai',
            input_kind: memoryEmbedInputKind.value || 'text',
            text: memoryEmbedText.value || '',
            image_data_url: memoryEmbedImageData.value || '',
            start_chars: Number(memoryEmbedStartChars.value || 256),
            max_chars: Number(memoryEmbedMaxChars.value || 131072),
            timeout_seconds: Number(memoryEmbedTimeout.value || 30),
            slow_seconds: Number(memoryEmbedSlowSeconds.value || 10),
            multiplier: 2,
            max_steps: 8,
            stop_on_slow: true,
          }),
        });
        if (!response.ok) throw new Error(await response.text());
        renderMemoryEmbedProbe(await response.json());
      } catch (error) {
        memoryEmbedStatus.textContent = error.message;
        memoryEmbedStatus.className = 'status hard';
      } finally {
        runMemoryEmbedProbe.disabled = false;
      }
    }
    function renderRagLoadProbe(result) {
      const passed = result.status === 'passed';
      const partial = result.status === 'partial';
      ragLoadStatus.textContent = `${result.status || 'unknown'} in ${result.elapsed_seconds || 0}s`;
      ragLoadStatus.className = `status ${passed ? 'ok' : (partial ? 'warn' : 'hard')}`;
      ragLoadStats.innerHTML = '';
      [
        ['Steps', `${fmtNumber(result.ok_step_count || 0)} / ${fmtNumber(result.completed_step_count || 0)}`, passed ? 'ok' : (partial ? 'warn' : 'hard')],
        ['Embedding', result.embedding_model || ''],
        ['Reranker', result.reranker_model || ''],
        ['Bridge', result.bridge_query_mode || 'none'],
        ['Stop', result.stop_reason || '', result.stop_reason === 'completed' ? 'ok' : 'warn'],
      ].forEach(([label, value, className]) => ragLoadStats.appendChild(chunkMetric(label, value, className || '')));
      ragLoadActions.innerHTML = '';
      for (const item of (result.results || [])) {
        const row = document.createElement('div');
        const time = document.createElement('span');
        time.className = 'muted';
        time.textContent = `${Number(item.elapsed_seconds || 0).toFixed(3)}s`;
        const event = document.createElement('span');
        event.textContent = `${fmtNumber(item.tokens)} tok / ${fmtNumber(item.chars)} chars`;
        const detail = document.createElement('span');
        detail.className = 'muted';
        const embedding = item.embedding || {};
        const reranker = item.reranker || {};
        const bridge = item.bridge || null;
        const parts = [
          `emb=${embedding.ok ? 'ok' : 'fail'} ${Number(embedding.elapsed_seconds || 0).toFixed(2)}s dim=${embedding.embedding_dimensions || ''}`,
          `rank=${reranker.ok ? 'ok' : 'fail'} ${Number(reranker.elapsed_seconds || 0).toFixed(2)}s toks=${reranker.prompt_tokens || ''}`,
        ];
        if (bridge) parts.push(`llm=${bridge.ok ? 'ok' : 'fail'} ${Number(bridge.elapsed_seconds || 0).toFixed(2)}s`);
        if (!item.ok) parts.push((embedding.error || reranker.error || bridge?.error || '').slice(0, 180));
        detail.textContent = parts.join(' | ');
        row.append(time, event, detail);
        ragLoadActions.appendChild(row);
      }
      ragLoadRaw.textContent = pretty(result);
    }
    async function startRagLoadProbe() {
      runRagLoadProbe.disabled = true;
      ragLoadStatus.textContent = 'starte...';
      ragLoadStatus.className = 'status warn';
      try {
        const response = await fetch('/api/rag-load-probe', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            embedding_base_url: ragLoadEmbeddingBaseUrl.value || '',
            embedding_model: ragLoadEmbeddingModel.value || '',
            embedding_backend: ragLoadEmbeddingBackend.value || 'ollama_embed',
            reranker_url: ragLoadRerankerUrl.value || '',
            reranker_model: ragLoadRerankerModel.value || '',
            query: ragLoadQuery.value || '',
            text: ragLoadText.value || '',
            token_steps: ragLoadTokenSteps.value || '',
            reranker_doc_count: Number(ragLoadDocCount.value || 10),
            reranker_doc_chars: Number(ragLoadDocChars.value || 700),
            timeout_seconds: Number(ragLoadTimeout.value || 240),
            bridge_query_mode: ragLoadBridgeMode.value || 'none',
            stop_on_failure: true,
          }),
        });
        if (!response.ok) throw new Error(await response.text());
        renderRagLoadProbe(await response.json());
      } catch (error) {
        ragLoadStatus.textContent = error.message;
        ragLoadStatus.className = 'status hard';
      } finally {
        runRagLoadProbe.disabled = false;
      }
    }
    function selectedRecord() {
      return records.find((record) => record.id === selectedId) || records[0] || null;
    }
    function statusClass(status) {
      if (status === 'hard_cutoff') return 'pill hard';
      if (status === 'completed') return 'pill done';
      return 'pill';
    }
    function renderRows() {
      rowsEl.innerHTML = '';
      for (const record of records) {
        const tr = document.createElement('tr');
        if (record.id === selectedId) tr.className = 'selected';
        const send = record.send || {};
        const receive = record.receive || {};
        const stateProfile = send.langgraph_state_profile || {};
        const budget = budgetOf(record);
        const budgetStatus = budgetState(budget);
        const budgetText = budget.request_tokens || budget.message_tokens
          ? `${fmtNumber(budget.request_tokens || budget.message_tokens)} / ${fmtNumber(budget.effective_active_limit || budget.active_limit || '')}`
          : '';
        const cells = [
          fmtTime(record.created_at),
          record.protocol,
          record.status,
          record.stream ? 'ja' : 'nein',
          record.model,
          short(record.thread_key, 48),
          short(record.thread_id, 36),
          send.raw_message_count,
          send.raw_token_estimate,
          send.model_context_message_count,
          send.model_context_token_estimate,
          stateProfile.message_count,
          budgetText,
          receive.output_chars,
          receive.reasoning_chars,
        ];
        cells.forEach((value, index) => {
          const td = document.createElement('td');
          if (index === 2) {
            const span = document.createElement('span');
            span.className = statusClass(record.status);
            span.textContent = value || 'received';
            td.appendChild(span);
          } else {
            if (index === 12 && value) {
              const span = document.createElement('span');
              span.className = budgetStatus.className;
              span.textContent = `${budgetStatus.label}: ${value}`;
              td.appendChild(span);
            } else {
              td.textContent = value ?? '';
            }
            if ([5, 6].includes(index)) td.className = 'mono';
          }
          tr.appendChild(td);
        });
        tr.addEventListener('click', () => {
          selectedId = record.id;
          if (record.thread_id) ragPinsThread.value = record.thread_id;
          renderRows();
          renderDetail();
          renderBudget();
          renderSourceIngests();
          renderShrinking();
          refreshRagPins().catch(() => {});
        });
        rowsEl.appendChild(tr);
      }
    }
    function renderDetail() {
      const record = selectedRecord();
      if (!record) {
        detailEl.className = 'empty';
        detailEl.textContent = 'Keine Anfrage ausgewählt.';
        return;
      }
      detailEl.className = '';
      if (activeTab === 'send') {
        const send = record.send || {};
        detailEl.textContent = activeMode === 'context'
          ? pretty({
              thread_key: record.thread_key,
              thread_id: record.thread_id,
              model_context_token_estimate: send.model_context_token_estimate,
              langgraph_state_profile: send.langgraph_state_profile || {},
              context_budget: budgetOf(record),
              source_ingests: sourceIngestsOf(record),
              shrinking: shrinkSummary(record),
              model_context_messages: send.model_context_messages || [],
              model_context: send.model_context || {},
            })
          : pretty(send);
      } else {
        const receive = record.receive || {};
        if (activeTab === 'compression') {
          detailEl.textContent = pretty({
            context_budget: budgetOf(record),
            source_ingests: sourceIngestsOf(record),
            shrinking: shrinkSummary(record),
            compression: receive.compression || {},
            memory_notice: receive.output_text || '',
          });
          return;
        }
        detailEl.textContent = activeMode === 'context'
          ? pretty({
              status: receive.status,
              output_text: receive.output_text || '',
              reasoning_text: receive.reasoning_text || '',
              context_budget: receive.context_budget || {},
              source_ingests: sourceIngestsOf(record),
              shrinking: shrinkSummary(record),
              compression: receive.compression || {},
              event_counts: receive.event_counts || {},
            })
          : pretty(receive);
      }
    }
    function renderBudget() {
      budgetGrid.innerHTML = '';
      const record = selectedRecord();
      const budget = budgetOf(record);
      if (!record || !budget || !Object.keys(budget).length) {
        budgetStatus.textContent = 'Noch keine Budgetdaten.';
        budgetStatus.className = 'status';
        return;
      }
      const state = budgetState(budget);
      budgetStatus.textContent = budget.node ? `${state.label}; letzter Knoten: ${budget.node}` : state.label;
      budgetStatus.className = `status ${state.className.replace('pill', '').trim()}`.trim();
      const headroom = budgetHeadroom(budget);
      const providerRetry = budget.provider_context_overflow_retry_used ? 'ja' : 'nein';
      const providerReason = budget.provider_context_overflow_retry_classification?.reason || '';
      [
        ['Status', state.label, state.className.replace('pill', '').trim()],
        ['Restbudget', headroom === null ? '' : fmtNumber(headroom), headroom !== null && headroom < 0 ? 'hard' : (headroom !== null && headroom < 2000 ? 'warn' : 'ok')],
        ['Kontext', fmtNumber(budget.context_length)],
        ['Provider Ctx', fmtNumber(budget.provider_reported_context_limit || '')],
        ['Detected Ctx', fmtNumber(budget.detected_context_length || '')],
        ['Messages', fmtNumber(budget.message_tokens)],
        ['Reserve', fmtNumber(budget.static_context_reserve_tokens)],
        ['Request', fmtNumber(budget.request_tokens)],
        ['Eff. Active', fmtNumber(budget.effective_active_limit)],
        ['Eff. Hard', fmtNumber(budget.effective_hard_limit)],
        ['Rescue Pässe', budget.final_budget_rescue_passes || budget.pre_run_compression_passes || ''],
        ['Retry', providerReason ? `${providerRetry} (${providerReason})` : providerRetry],
      ].forEach(([name, value, className]) => budgetGrid.appendChild(metric(name, value, className || '')));
    }
    async function loadRecords() {
      const limit = Math.max(1, Math.min(200, Number(limitEl.value || 80)));
      const response = await fetch(`/api/observer?limit=${limit}`, { cache: 'no-store' });
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json();
      records = Array.isArray(payload.records) ? payload.records : [];
      if (!selectedId && records[0]) selectedId = records[0].id;
      if (selectedId && !records.some((record) => record.id === selectedId) && records[0]) selectedId = records[0].id;
      renderRows();
      renderDetail();
      renderBudget();
      renderSourceIngests();
      renderShrinking();
      statusEl.textContent = `${records.length} Requests`;
    }
    function setTab(tab) {
      activeTab = tab;
      sendTab.classList.toggle('active', tab === 'send');
      receiveTab.classList.toggle('active', tab === 'receive');
      compressionTab.classList.toggle('active', tab === 'compression');
      renderDetail();
    }
    function setMode(mode) {
      activeMode = mode;
      contextMode.classList.toggle('active', mode === 'context');
      fullMode.classList.toggle('active', mode === 'full');
      renderDetail();
    }
    refreshBtn.addEventListener('click', () => loadRecords().catch((error) => { statusEl.textContent = error.message; }));
    clearBtn.addEventListener('click', async () => {
      await fetch('/api/observer', { method: 'DELETE' });
      records = [];
      selectedId = '';
      renderRows();
      renderDetail();
      renderBudget();
      renderSourceIngests();
      renderShrinking();
      statusEl.textContent = 'geleert';
    });
    sendTab.addEventListener('click', () => setTab('send'));
    receiveTab.addEventListener('click', () => setTab('receive'));
    compressionTab.addEventListener('click', () => setTab('compression'));
    contextMode.addEventListener('click', () => setMode('context'));
    fullMode.addEventListener('click', () => setMode('full'));
    runChunking.addEventListener('click', () => startChunkingRun());
    runArchiveRagSmoke.addEventListener('click', () => startArchiveRagSmoke());
    runNativeRagSmoke.addEventListener('click', () => startNativeRagSmoke());
    runRagClassifierProbe.addEventListener('click', () => startRagClassifierProbe());
    runMemoryEmbedProbe.addEventListener('click', () => startMemoryEmbedProbe());
    runRagLoadProbe.addEventListener('click', () => startRagLoadProbe());
    ragPinsPin.addEventListener('click', () => writeRagPins('pin').catch((error) => { ragPinsStatus.textContent = error.message; ragPinsStatus.className = 'status hard'; }));
    ragPinsUnpin.addEventListener('click', () => writeRagPins('unpin').catch((error) => { ragPinsStatus.textContent = error.message; ragPinsStatus.className = 'status hard'; }));
    ragPinsClear.addEventListener('click', () => writeRagPins('clear').catch((error) => { ragPinsStatus.textContent = error.message; ragPinsStatus.className = 'status hard'; }));
    ragPinsThread.addEventListener('change', () => refreshRagPins().catch(() => {}));
    memoryReviewExtract.addEventListener('click', () => extractMemoryReview().catch((error) => { memoryReviewStatus.textContent = error.message; memoryReviewStatus.className = 'status hard'; }));
    memoryReviewRefresh.addEventListener('click', () => refreshMemoryReview().catch((error) => { memoryReviewStatus.textContent = error.message; memoryReviewStatus.className = 'status hard'; }));
    loadRecords().catch((error) => { statusEl.textContent = error.message; });
    refreshEmbeddingQueue();
    refreshResumeRuns();
    refreshRagPins();
    refreshMemoryReview();
    window.setInterval(() => loadRecords().catch(() => {}), 2500);
    window.setInterval(() => refreshEmbeddingQueue().catch(() => {}), 5000);
    window.setInterval(() => refreshResumeRuns().catch(() => {}), 10000);
  </script>
</body>
</html>
"""


HTML = """<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AlphaRavis Bridge Test UI</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #111318; color: #eef1f5; min-height: 100vh; }
    main { max-width: 960px; margin: 0 auto; padding: 24px; display: grid; gap: 16px; }
    header { display: flex; align-items: center; justify-content: space-between; gap: 12px; border-bottom: 1px solid #2d3340; padding-bottom: 12px; }
    h1 { font-size: 20px; margin: 0; font-weight: 650; }
    .header-left { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .nav-button { display: inline-flex; align-items: center; justify-content: center; border: 1px solid #3a4252; border-radius: 8px; background: #171b23; color: #eef1f5; padding: 8px 11px; font-size: 13px; text-decoration: none; }
    .nav-button:hover { background: #202633; }
    .status { color: #9aa4b2; font-size: 13px; }
    .live-panels { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
    .live-panel { border: 1px solid #2d3340; border-radius: 8px; background: #0d1016; min-height: 112px; display: grid; grid-template-rows: auto 1fr; overflow: hidden; }
    .live-panel.expanded { grid-column: 1 / -1; min-height: 280px; }
    .live-panel-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; border-bottom: 1px solid #202633; padding: 6px 8px 6px 10px; }
    .live-panel h2 { margin: 0; color: #9aa4b2; font-size: 11px; font-weight: 650; text-transform: uppercase; letter-spacing: 0; }
    .panel-toggle { border-radius: 6px; padding: 4px 7px; font-size: 11px; line-height: 1; }
    .live-panel pre { margin: 0; padding: 10px; max-height: 180px; overflow: auto; color: #cbd5e1; white-space: pre-wrap; overflow-wrap: anywhere; }
    .live-panel.expanded pre { max-height: 440px; }
    #chat { min-height: 48vh; max-height: 68vh; overflow: auto; display: flex; flex-direction: column; gap: 10px; padding: 4px 2px; }
    .msg { border: 1px solid #2d3340; background: #181c24; border-radius: 8px; padding: 10px 12px; white-space: pre-wrap; line-height: 1.45; }
    .user { align-self: flex-end; max-width: 78%; background: #16324a; border-color: #24557e; }
    .assistant { align-self: flex-start; max-width: 86%; }
    .meta { color: #9aa4b2; font-size: 12px; margin-bottom: 4px; }
    .route-badge { display: inline-block; margin-left: 8px; border: 1px solid #3a4252; border-radius: 999px; padding: 1px 7px; color: #cbd5e1; font-size: 11px; }
    .route-fast { border-color: #2f8f5b; color: #7dd3a8; }
    .route-agent { border-color: #8a6d2e; color: #f3c969; }
    .route-hard { border-color: #8f3b3b; color: #f59b9b; }
    .reasoning-details { margin-top: 8px; border-top: 1px solid #2d3340; padding-top: 8px; color: #cbd5e1; }
    .reasoning-details summary { cursor: pointer; color: #9aa4b2; font-size: 12px; user-select: none; }
    .reasoning-section { margin-top: 8px; display: grid; gap: 4px; }
    .reasoning-label { color: #9aa4b2; font-size: 11px; font-weight: 650; text-transform: uppercase; letter-spacing: 0; }
    .reasoning-body { font-size: 12px; line-height: 1.45; color: #cbd5e1; white-space: pre-wrap; }
    .reasoning-status { color: #9aa4b2; }
    .context-terminal { display: grid; gap: 6px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .context-event { border-left: 3px solid #3a4252; padding-left: 7px; white-space: pre-wrap; }
    .context-event.compaction { border-color: #d1a32c; color: #f3c969; }
    .context-event.hard { border-color: #d94b4b; color: #f59b9b; }
    #live-context.context-compaction { color: #f3c969; }
    #live-context.context-hard { color: #f59b9b; }
    form { display: grid; gap: 10px; border-top: 1px solid #2d3340; padding-top: 14px; }
    textarea { width: 100%; min-height: 96px; resize: vertical; border: 1px solid #3a4252; border-radius: 8px; background: #0d1016; color: #eef1f5; padding: 12px; font: inherit; }
    .controls { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    select, button { border: 1px solid #3a4252; border-radius: 8px; background: #171b23; color: #eef1f5; padding: 9px 12px; font: inherit; }
    button.primary { background: #2d6cdf; border-color: #2d6cdf; }
    button:disabled { opacity: .55; cursor: wait; }
    details { border: 1px solid #2d3340; border-radius: 8px; padding: 8px 10px; background: #0d1016; }
    .trace { border: 1px solid #2d3340; border-radius: 8px; background: #0d1016; padding: 10px; }
    .trace-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 8px; }
    .trace h2 { font-size: 14px; margin: 0; font-weight: 650; }
    .trace-toggle { display: inline-flex; align-items: center; gap: 6px; color: #9aa4b2; font-size: 12px; user-select: none; }
    .trace-toggle input { margin: 0; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { border-top: 1px solid #202633; padding: 6px 4px; text-align: left; vertical-align: top; }
    th { color: #9aa4b2; font-weight: 600; }
    .bar-wrap { height: 8px; background: #161b24; border-radius: 999px; overflow: hidden; min-width: 120px; }
    .bar { height: 100%; background: #2d6cdf; width: 0; }
    pre { overflow: auto; font-size: 12px; color: #cbd5e1; }
    @media (max-width: 760px) { .live-panels { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <header>
      <div class="header-left">
        <a class="nav-button" href="/observer">Observer</a>
        <h1>AlphaRavis Bridge Test UI</h1>
      </div>
      <div id="status" class="status">bereit</div>
    </header>
    <section class="live-panels" aria-label="Stream-Details">
      <div class="live-panel">
        <div class="live-panel-head"><h2>Status</h2><button class="panel-toggle" type="button" data-panel-toggle>Gross</button></div>
        <pre id="live-status">(leer)</pre>
      </div>
      <div class="live-panel">
        <div class="live-panel-head"><h2>Reasoning</h2><button class="panel-toggle" type="button" data-panel-toggle>Gross</button></div>
        <pre id="live-reasoning">(leer)</pre>
      </div>
      <div class="live-panel">
        <div class="live-panel-head"><h2>Planer</h2><button class="panel-toggle" type="button" data-panel-toggle>Gross</button></div>
        <pre id="live-plan">(leer)</pre>
      </div>
      <div class="live-panel">
        <div class="live-panel-head"><h2>Kontext</h2><button class="panel-toggle" type="button" data-panel-toggle>Gross</button></div>
        <pre id="live-context">(leer)</pre>
      </div>
    </section>
    <section id="chat" aria-live="polite"></section>
    <form id="form">
      <textarea id="input" placeholder="Nachricht eingeben..." autofocus></textarea>
      <div class="controls">
        <select id="protocol" title="Bridge-Protokoll">
          <option value="responses">Responses</option>
          <option value="chat">Chat Completions</option>
        </select>
        <button class="primary" id="send" type="submit">Senden</button>
        <button id="clear" type="button">Verlauf leeren</button>
      </div>
    </form>
    <section class="trace">
      <div class="trace-head">
        <h2>Trace</h2>
        <label class="trace-toggle"><input id="trace-delta-details" type="checkbox"> Delta-Details</label>
      </div>
      <div id="trace-summary" class="status">noch keine Anfrage</div>
      <table>
        <thead><tr><th>t</th><th>Schritt</th><th>Dauer</th><th>Details</th><th></th></tr></thead>
        <tbody id="trace-body"></tbody>
      </table>
    </section>
    <details>
      <summary>Letzte rohe Bridge-Antwort</summary>
      <pre id="raw">{}</pre>
    </details>
  </main>
  <script>
    const chat = document.getElementById('chat');
    const form = document.getElementById('form');
    const input = document.getElementById('input');
    const statusEl = document.getElementById('status');
    const liveStatusEl = document.getElementById('live-status');
    const liveReasoningEl = document.getElementById('live-reasoning');
    const livePlanEl = document.getElementById('live-plan');
    const liveContextEl = document.getElementById('live-context');
    const rawEl = document.getElementById('raw');
    const traceSummary = document.getElementById('trace-summary');
    const traceBody = document.getElementById('trace-body');
    const traceDeltaDetails = document.getElementById('trace-delta-details');
    const sendBtn = document.getElementById('send');
    const clearBtn = document.getElementById('clear');
    const protocol = document.getElementById('protocol');
    const messages = [];

    function makeId(prefix) {
      if (window.crypto && typeof window.crypto.randomUUID === 'function') {
        return `${prefix}_${window.crypto.randomUUID().split('-').join('').slice(0, 12)}`;
      }
      const random = Math.random().toString(16).slice(2, 14);
      return `${prefix}_${Date.now().toString(16)}${random}`.slice(0, prefix.length + 13);
    }

    function storedSessionId() {
      try {
        const existing = window.localStorage.getItem('alpharavis-test-ui-session');
        if (existing) return existing;
        const created = makeId('session');
        window.localStorage.setItem('alpharavis-test-ui-session', created);
        return created;
      } catch (error) {
        return makeId('session');
      }
    }

    function resetSessionId() {
      const created = makeId('session');
      try {
        window.localStorage.setItem('alpharavis-test-ui-session', created);
      } catch (error) {
        // Ignore storage failures; the in-memory session id is still reset.
      }
      return created;
    }

    let sessionId = storedSessionId();
    let lastTrace = null;
    let lastTraceBrowserSeconds = 0;

    window.addEventListener('error', (event) => {
      statusEl.textContent = `JS-Fehler: ${event.message || 'unbekannt'}`;
    });

    function isCompactDeltaStep(step) {
      if (!step || typeof step !== 'object') return false;
      if (step.event === 'response.output_text.delta') return true;
      if (step.event === 'response.reasoning.delta' && ['internal_plan', 'model'].includes(step.reasoning_kind)) return true;
      if (step.event === 'message' && ['internal_plan', 'model'].includes(step.reasoning_kind)) return true;
      return step.event === 'message' && step.text_delta === true;
    }

    function summarizeTraceSteps(steps) {
      const summarized = [];
      let group = null;
      function flushGroup() {
        if (!group) return;
        summarized.push({
          name: `${group.name || 'Delta empfangen'} (${group.count} Deltas, ${group.chars} Zeichen)`,
          elapsed_seconds: group.firstElapsed,
          duration_seconds: Math.max(0, group.lastElapsed - group.firstElapsed),
          event: group.event,
          reasoning_kind: group.reasoningKind,
          sequence_number: `${group.firstSequence ?? '?'}..${group.lastSequence ?? '?'}`,
          delta_chars: group.chars,
        });
        group = null;
      }
      for (const step of steps) {
        if (!isCompactDeltaStep(step)) {
          flushGroup();
          summarized.push(step);
          continue;
        }
        const elapsed = Number(step.elapsed_seconds || 0);
        const chars = Number(step.delta_chars || 0);
        if (!group || group.event !== step.event) {
          flushGroup();
          group = {
            event: step.event,
            name: step.name,
            reasoningKind: step.reasoning_kind,
            count: 0,
            chars: 0,
            firstElapsed: elapsed,
            lastElapsed: elapsed,
            firstSequence: step.sequence_number,
            lastSequence: step.sequence_number,
          };
        }
        group.count += 1;
        group.chars += chars;
        group.lastElapsed = elapsed;
        group.lastSequence = step.sequence_number;
      }
      flushGroup();
      return summarized;
    }

    function renderTrace(trace, browserSeconds) {
      lastTrace = trace || {};
      lastTraceBrowserSeconds = browserSeconds || 0;
      traceBody.innerHTML = '';
      trace = trace || {};
      const rawSteps = Array.isArray(trace.steps) ? trace.steps : [];
      const steps = traceDeltaDetails.checked ? rawSteps : summarizeTraceSteps(rawSteps);
      const maxElapsed = Math.max(browserSeconds || 0, ...steps.map((step) => Number(step.elapsed_seconds || 0)), 0.001);
      const hiddenSteps = rawSteps.length - steps.length;
      const compactSuffix = hiddenSteps > 0 ? ` | ${hiddenSteps} Delta-Zeilen zusammengefasst` : '';
      traceSummary.textContent = `${trace.trace_id || 'trace'} | ${steps.length} Schritte | ${browserSeconds.toFixed(2)}s Browser${compactSuffix}`;
      for (const step of steps) {
        const elapsed = Number(step.elapsed_seconds || 0);
        const duration = step.duration_seconds == null ? '' : `${Number(step.duration_seconds).toFixed(3)}s`;
        const details = Object.entries(step)
          .filter(([key]) => !['name', 'elapsed_seconds', 'duration_seconds'].includes(key))
          .map(([key, value]) => `${key}=${typeof value === 'string' ? value : JSON.stringify(value)}`)
          .join(' ');
        const tr = document.createElement('tr');
        const cells = [
          `${elapsed.toFixed(3)}s`,
          step.name || '',
          duration,
          details,
        ];
        for (const text of cells) {
          const td = document.createElement('td');
          td.textContent = text;
          tr.appendChild(td);
        }
        const barCell = document.createElement('td');
        const wrap = document.createElement('div');
        wrap.className = 'bar-wrap';
        const bar = document.createElement('div');
        bar.className = 'bar';
        bar.style.width = `${Math.min(100, (elapsed / maxElapsed) * 100)}%`;
        wrap.appendChild(bar);
        barCell.appendChild(wrap);
        tr.appendChild(barCell);
        traceBody.appendChild(tr);
      }
    }

    function render() {
      chat.innerHTML = '';
      for (const msg of messages) {
        const el = document.createElement('div');
        el.className = `msg ${msg.role}`;
        const meta = document.createElement('div');
        meta.className = 'meta';
        meta.textContent = msg.role;
        if (msg.role === 'assistant') {
          const badge = document.createElement('span');
          badge.className = `route-badge ${routeClass(msg.route)}`;
          badge.textContent = routeLabel(msg.route);
          meta.appendChild(badge);
        }
        const body = document.createElement('div');
        body.textContent = msg.content || '(leer)';
        el.append(meta, body);
        if (msg.role === 'assistant' && (msg.reasoningStatus || msg.internalPlan || msg.reasoning || (msg.contextEvents && msg.contextEvents.length))) {
          const details = document.createElement('details');
          details.className = 'reasoning-details';
          details.open = Boolean(msg.reasoningOpen);
          details.addEventListener('toggle', () => {
            msg.reasoningOpen = details.open;
          });
          const summary = document.createElement('summary');
          summary.textContent = 'Reasoning';
          details.appendChild(summary);
          if (msg.reasoningStatus) {
            const statusSection = document.createElement('div');
            statusSection.className = 'reasoning-section';
            const statusLabel = document.createElement('div');
            statusLabel.className = 'reasoning-label';
            statusLabel.textContent = 'Status';
            const statusBody = document.createElement('div');
            statusBody.className = 'reasoning-body reasoning-status';
            statusBody.textContent = msg.reasoningStatus;
            statusSection.append(statusLabel, statusBody);
            details.appendChild(statusSection);
          }
          if (msg.internalPlan) {
            const planSection = document.createElement('div');
            planSection.className = 'reasoning-section';
            const planLabel = document.createElement('div');
            planLabel.className = 'reasoning-label';
            planLabel.textContent = 'Interner Plan';
            const planBody = document.createElement('div');
            planBody.className = 'reasoning-body';
            planBody.textContent = msg.internalPlan;
            planSection.append(planLabel, planBody);
            details.appendChild(planSection);
          }
          if (msg.reasoning) {
            const reasoningSection = document.createElement('div');
            reasoningSection.className = 'reasoning-section';
            const reasoningLabel = document.createElement('div');
            reasoningLabel.className = 'reasoning-label';
            reasoningLabel.textContent = 'Modell-Reasoning';
            const reasoningBody = document.createElement('div');
            reasoningBody.className = 'reasoning-body';
            reasoningBody.textContent = msg.reasoning;
            reasoningSection.append(reasoningLabel, reasoningBody);
            details.appendChild(reasoningSection);
          }
          if (msg.contextEvents && msg.contextEvents.length) {
            const contextSection = document.createElement('div');
            contextSection.className = 'reasoning-section';
            const contextLabel = document.createElement('div');
            contextLabel.className = 'reasoning-label';
            contextLabel.textContent = 'Kontext';
            const contextBody = document.createElement('div');
            contextBody.className = 'reasoning-body context-terminal';
            for (const item of msg.contextEvents) {
              const line = document.createElement('div');
              line.className = `context-event ${item.kind === 'context_hard' ? 'hard' : 'compaction'}`;
              line.textContent = item.text;
              contextBody.appendChild(line);
            }
            contextSection.append(contextLabel, contextBody);
            details.appendChild(contextSection);
          }
          el.appendChild(details);
        }
        chat.appendChild(el);
      }
      chat.scrollTop = chat.scrollHeight;
      const currentAssistant = [...messages].reverse().find((msg) => msg.role === 'assistant');
      renderLivePanels(currentAssistant || null);
    }

    function renderLivePanels(msg) {
      const statusText = msg && msg.reasoningStatus ? msg.reasoningStatus.trim() : '';
      const reasoningText = msg && msg.reasoning ? msg.reasoning.trim() : '';
      const planText = msg && msg.internalPlan ? msg.internalPlan.trim() : '';
      const contextText = msg && msg.contextEvents && msg.contextEvents.length
        ? msg.contextEvents.map((item) => `${item.kind === 'context_hard' ? '[HARD]' : '[COMPACT]'} ${item.text}`).join('\\n')
        : '';
      liveStatusEl.textContent = statusText || '(leer)';
      liveReasoningEl.textContent = reasoningText || '(leer)';
      livePlanEl.textContent = planText || '(leer)';
      liveContextEl.textContent = contextText || '(leer)';
      liveContextEl.classList.toggle('context-hard', Boolean(msg && msg.contextEvents && msg.contextEvents.some((item) => item.kind === 'context_hard')));
      liveContextEl.classList.toggle('context-compaction', Boolean(msg && msg.contextEvents && msg.contextEvents.some((item) => item.kind === 'context_compaction')));
      liveStatusEl.scrollTop = liveStatusEl.scrollHeight;
      liveReasoningEl.scrollTop = liveReasoningEl.scrollHeight;
      livePlanEl.scrollTop = livePlanEl.scrollHeight;
      liveContextEl.scrollTop = liveContextEl.scrollHeight;
    }

    function parseSseBlock(block) {
      const lines = block.split(/\\r?\\n/);
      let eventName = 'message';
      const dataLines = [];
      for (const line of lines) {
        if (line.startsWith('event:')) {
          eventName = line.slice(6).trim();
        } else if (line.startsWith('data:')) {
          dataLines.push(line.slice(5).trimStart());
        }
      }
      if (!dataLines.length) return null;
      const dataText = dataLines.join('\\n');
      if (dataText === '[DONE]') {
        return { event: eventName, done: true, data: '[DONE]' };
      }
      try {
        return { event: eventName, data: JSON.parse(dataText) };
      } catch (error) {
        return { event: eventName, data: dataText };
      }
    }

    function responseTextDelta(eventName, data) {
      if (!data || typeof data !== 'object') return '';
      if (eventName === 'response.output_text.delta' && typeof data.delta === 'string') {
        return data.delta;
      }
      return '';
    }

    function chatTextDelta(data) {
      if (!data || typeof data !== 'object') return '';
      const choice = Array.isArray(data.choices) ? data.choices[0] : null;
      const delta = choice && choice.delta && typeof choice.delta === 'object' ? choice.delta : {};
      return typeof delta.content === 'string' ? delta.content : '';
    }

    function reasoningDelta(protocolName, eventName, data, currentReasoning) {
      if (!data || typeof data !== 'object') return '';
      if (protocolName === 'responses') {
        if (eventName === 'response.reasoning.delta' && typeof data.delta === 'string') {
          return data.delta;
        }
        if (eventName === 'response.reasoning.done' && !currentReasoning && typeof data.text === 'string') {
          return data.text;
        }
        return '';
      }
      const choice = Array.isArray(data.choices) ? data.choices[0] : null;
      const delta = choice && choice.delta && typeof choice.delta === 'object' ? choice.delta : {};
      if (typeof delta.reasoning_content === 'string') return delta.reasoning_content;
      if (typeof delta.reasoning === 'string') return delta.reasoning;
      return '';
    }

    function reasoningKind(data, text, msg) {
      if (data && typeof data === 'object' && typeof data.alpha_reasoning_kind === 'string') {
        return data.alpha_reasoning_kind;
      }
      const choice = data && Array.isArray(data.choices) ? data.choices[0] : null;
      const delta = choice && choice.delta && typeof choice.delta === 'object' ? choice.delta : {};
      if (typeof delta.alpha_reasoning_kind === 'string') return delta.alpha_reasoning_kind;
      const value = String(text || '').trimStart();
      if (value.startsWith('Status:')) return 'status';
      if (value.startsWith('Interner Plan')) return 'internal_plan';
      if (msg && msg.reasoningMode === 'internal_plan') return 'internal_plan';
      return 'model';
    }

    function cleanInternalReasoning(text) {
      return String(text || '').replace(/^Interner Plan \\([^)]*\\):\\n?/, '');
    }

    function isContextReasoningKind(kind) {
      return kind === 'context_compaction' || kind === 'context_hard';
    }

    function routeClass(routeName) {
      if (routeName === 'fast_path') return 'route-fast';
      if (routeName === 'agent_path') return 'route-agent';
      if (routeName === 'hard_stop') return 'route-hard';
      return '';
    }

    function routeLabel(routeName) {
      if (routeName === 'fast_path') return 'Fast Path';
      if (routeName === 'agent_path') return 'Agent Path';
      if (routeName === 'hard_stop') return 'Hard Stop';
      return 'Route offen';
    }

    function routeFromText(text) {
      const value = String(text || '').toLowerCase();
      if (!value) return '';
      if (value.includes('fast-path aktiv') || value.includes('fast_chat')) return 'fast_path';
      if (value.includes('hard_stop') || value.includes('hard context')) return 'hard_stop';
      if (
        value.includes('swarm') ||
        value.includes('planner') ||
        value.includes('memory_kernel') ||
        value.includes('skill_library') ||
        value.includes('handoff_context_guard') ||
        value.includes('crisis_preflight')
      ) {
        return 'agent_path';
      }
      return '';
    }

    function routeFromEvent(protocolName, eventName, data, textDelta, reasoning) {
      if (eventName === 'response.output_text.delta' || eventName === 'message') {
        const fromText = routeFromText(textDelta);
        if (fromText) return fromText;
      }
      if (eventName === 'response.reasoning.delta') {
        const fromReasoning = routeFromText(reasoning || (data && data.delta));
        if (fromReasoning) return fromReasoning;
      }
      return '';
    }

    function streamStatusText(eventName, data) {
      if (eventName === 'test_ui.started') return 'Stream gestartet';
      if (eventName === 'test_ui.completed') return 'Stream abgeschlossen';
      if (eventName === 'test_ui.error') return 'Stream-Fehler';
      if (eventName === 'response.reasoning.delta' && data && typeof data.delta === 'string') {
        if (data.alpha_reasoning_kind === 'internal_plan') return 'Interner Plan empfangen';
        if (data.alpha_reasoning_kind === 'model') return 'Modell-Reasoning empfangen';
        return data.delta.trim();
      }
      if (eventName === 'response.output_text.delta') return 'Antworttext empfangen';
      if (eventName === 'message' && data && Array.isArray(data.choices)) {
        const choice = data.choices[0] || {};
        const delta = choice.delta && typeof choice.delta === 'object' ? choice.delta : {};
        if (choice.finish_reason) return `Chat abgeschlossen: ${choice.finish_reason}`;
        if (delta.alpha_reasoning_kind === 'internal_plan') return 'Interner Plan empfangen';
        if (delta.alpha_reasoning_kind === 'model' || delta.reasoning_content || delta.reasoning) {
          return 'Modell-Reasoning empfangen';
        }
        if (chatTextDelta(data)) return 'Antworttext empfangen';
      }
      return eventName;
    }

    function traceStepForEvent(parsed, started) {
      const data = parsed.data && typeof parsed.data === 'object' ? parsed.data : {};
      const choice = data && Array.isArray(data.choices) ? data.choices[0] : null;
      const choiceDelta = choice && choice.delta && typeof choice.delta === 'object' ? choice.delta : {};
      const chatReasoning = typeof choiceDelta.reasoning_content === 'string'
        ? choiceDelta.reasoning_content
        : (typeof choiceDelta.reasoning === 'string' ? choiceDelta.reasoning : '');
      return {
        name: streamStatusText(parsed.event, data),
        elapsed_seconds: (performance.now() - started) / 1000,
        event: parsed.event,
        sequence_number: data.sequence_number,
        delta_chars: typeof data.delta === 'string' ? data.delta.length : (chatReasoning ? chatReasoning.length : undefined),
        reasoning_kind: typeof data.alpha_reasoning_kind === 'string'
          ? data.alpha_reasoning_kind
          : (typeof choiceDelta.alpha_reasoning_kind === 'string' ? choiceDelta.alpha_reasoning_kind : undefined),
        text_delta: parsed.event === 'response.output_text.delta' || Boolean(chatTextDelta(data)),
      };
    }

    async function consumeSseResponse(res, handlers) {
      if (!res.body) throw new Error('Streaming wird von diesem Browser nicht unterstützt.');
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const blocks = buffer.split(/\\r?\\n\\r?\\n/);
        buffer = blocks.pop() || '';
        for (const block of blocks) {
          const parsed = parseSseBlock(block.trim());
          if (parsed) handlers.onEvent(parsed);
        }
        if (done) break;
      }
      const tail = buffer.trim();
      if (tail) {
        const parsed = parseSseBlock(tail);
        if (parsed) handlers.onEvent(parsed);
      }
    }

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const text = input.value.trim();
      if (!text) return;
      input.value = '';
      messages.push({ role: 'user', content: text });
      const assistantMsg = {
        role: 'assistant',
        content: '',
        reasoning: '',
        reasoningStatus: '',
        internalPlan: '',
        contextEvents: [],
        reasoningMode: '',
        reasoningOpen: false,
        route: ''
      };
      messages.push(assistantMsg);
      render();
      sendBtn.disabled = true;
      statusEl.textContent = 'streamt...';
      const started = performance.now();
      const traceId = makeId('trace');
      const rawEvents = [];
      const streamSteps = [];
      try {
        const res = await fetch('/api/send_stream', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            message: text,
            messages: messages.slice(0, -2),
            protocol: protocol.value,
            stream: true,
            session_id: sessionId,
            trace_id: traceId
          })
        });
        if (!res.ok) {
          const errorText = await res.text();
          throw new Error(errorText || res.statusText);
        }
        await consumeSseResponse(res, {
          onEvent(parsed) {
            rawEvents.push(parsed);
            if (parsed.done) return;
            const data = parsed.data;
            if (parsed.event === 'test_ui.error') {
              const detail = data && typeof data === 'object' ? data.detail || data.status_code || 'unbekannt' : data;
              throw new Error(`Stream-Fehler: ${detail}`);
            }
            const reasoning = reasoningDelta(
              protocol.value,
              parsed.event,
              data,
              `${assistantMsg.reasoningStatus}${assistantMsg.internalPlan}${assistantMsg.reasoning}`
            );
            if (reasoning) {
              const kind = reasoningKind(data, reasoning, assistantMsg);
              assistantMsg.reasoningMode = kind;
              if (kind === 'status') {
                assistantMsg.reasoningStatus += reasoning;
              } else if (kind === 'internal_plan') {
                assistantMsg.internalPlan += cleanInternalReasoning(reasoning);
              } else if (isContextReasoningKind(kind)) {
                const text = String(reasoning || '').trim();
                if (text && !assistantMsg.contextEvents.some((item) => item.kind === kind && item.text === text)) {
                  assistantMsg.contextEvents.push({ kind, text });
                }
              } else {
                assistantMsg.reasoning += reasoning;
              }
            }
            const delta = protocol.value === 'chat' ? chatTextDelta(data) : responseTextDelta(parsed.event, data);
            if (delta) {
              assistantMsg.content += delta;
            }
            const inferredRoute = routeFromEvent(protocol.value, parsed.event, data, delta, reasoning);
            if (inferredRoute && !assistantMsg.route) {
              assistantMsg.route = inferredRoute;
            }
            if (reasoning || delta || inferredRoute) {
              render();
            }
            const step = traceStepForEvent(parsed, started);
            streamSteps.push(step);
            if (streamSteps.length > 160) streamSteps.splice(0, streamSteps.length - 160);
            const browserSeconds = (performance.now() - started) / 1000;
            renderTrace({ trace_id: traceId, protocol: protocol.value, steps: streamSteps }, browserSeconds);
            statusEl.textContent = `${protocol.value} stream | ${routeLabel(assistantMsg.route)} | ${streamStatusText(parsed.event, data)}`;
          }
        });
        rawEl.textContent = JSON.stringify(rawEvents, null, 2);
        const browserSeconds = (performance.now() - started) / 1000;
        renderTrace({ trace_id: traceId, protocol: protocol.value, steps: streamSteps }, browserSeconds);
        statusEl.textContent = `${protocol.value} stream | ${routeLabel(assistantMsg.route)} | ${browserSeconds.toFixed(2)}s browser`;
        if (!assistantMsg.content) assistantMsg.content = '(kein sichtbarer Antworttext gestreamt)';
      } catch (error) {
        assistantMsg.content = `FEHLER: ${error.message || error}`;
        statusEl.textContent = 'Fehler';
      } finally {
        sendBtn.disabled = false;
        render();
        input.focus();
      }
    });

    clearBtn.addEventListener('click', () => {
      messages.length = 0;
      sessionId = resetSessionId();
      rawEl.textContent = '{}';
      traceBody.innerHTML = '';
      traceSummary.textContent = 'noch keine Anfrage';
      lastTrace = null;
      lastTraceBrowserSeconds = 0;
      statusEl.textContent = 'neue Session bereit';
      render();
      input.focus();
    });

    document.querySelectorAll('[data-panel-toggle]').forEach((button) => {
      button.addEventListener('click', () => {
        const panel = button.closest('.live-panel');
        if (!panel) return;
        const expanded = panel.classList.toggle('expanded');
        button.textContent = expanded ? 'Klein' : 'Gross';
      });
    });

    traceDeltaDetails.addEventListener('change', () => {
      if (lastTrace) renderTrace(lastTrace, lastTraceBrowserSeconds);
    });

    input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
        if (typeof form.requestSubmit === 'function') {
          form.requestSubmit();
        } else {
          sendBtn.click();
        }
      }
    });

    render();
  </script>
</body>
</html>
"""
