from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langgraph_sdk import get_client

from context_references import preprocess_context_references
from error_classifier import classify_api_error, format_user_error
from internal_context import StreamingInternalContextScrubber, sanitize_internal_context


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


try:
    from operational_logging import (
        log_dependency_status as _op_log_dependency_status,
        log_event as _op_log_event,
        log_exception as _op_log_exception,
        setup_logging as _setup_operational_logging,
    )
except Exception as exc:  # pragma: no cover - logging must never block bridge startup
    _op_log_dependency_status = None
    _op_log_event = None
    _op_log_exception = None
    _setup_operational_logging = None
    OPERATIONAL_LOGGING_IMPORT_ERROR: Exception | None = exc
else:
    OPERATIONAL_LOGGING_IMPORT_ERROR = None

if _setup_operational_logging is not None:
    try:
        _setup_operational_logging(component="bridge")
    except Exception:
        pass


LANGGRAPH_API_URL = os.getenv("LANGGRAPH_API_URL", "http://langgraph-api:2024")
LANGGRAPH_ASSISTANT_ID = os.getenv("LANGGRAPH_ASSISTANT_ID", "alpha_ravis")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "my-agent")
SERVER_MODEL_MANAGER_MODEL_NAME = os.getenv("SERVER_MODEL_MANAGER_MODEL_NAME", "server-model-manager")
BRIDGE_RUN_TIMEOUT_SECONDS = float(os.getenv("BRIDGE_RUN_TIMEOUT_SECONDS", "180"))
BRIDGE_STREAM_MODE = os.getenv("BRIDGE_STREAM_MODE", "events").lower()
BRIDGE_STREAM_SUBGRAPHS = _env_bool("BRIDGE_STREAM_SUBGRAPHS", "true")
BRIDGE_MESSAGE_SYNC_MODE = os.getenv("BRIDGE_MESSAGE_SYNC_MODE", "delta").lower()
BRIDGE_SHOW_ACTIVITY_EVENTS = _env_bool("BRIDGE_SHOW_ACTIVITY_EVENTS", "false")
BRIDGE_ACTIVITY_DETAIL = os.getenv("BRIDGE_ACTIVITY_DETAIL", "summary").lower()
BRIDGE_STREAM_REASONING_EVENTS = _env_bool("BRIDGE_STREAM_REASONING_EVENTS", "false")
BRIDGE_REASONING_DELTA_FIELD = os.getenv("BRIDGE_REASONING_DELTA_FIELD", "reasoning_content")
BRIDGE_ENABLE_RESPONSES_API = _env_bool("BRIDGE_ENABLE_RESPONSES_API", "true")
BRIDGE_PREFERRED_API_MODE = os.getenv("BRIDGE_PREFERRED_API_MODE", "responses").lower()
BRIDGE_RESPONSES_STORE = _env_bool("BRIDGE_RESPONSES_STORE", "true")
BRIDGE_RESPONSES_STORE_MAX = int(os.getenv("BRIDGE_RESPONSES_STORE_MAX", "200"))
BRIDGE_RESPONSES_DONE_SENTINEL = _env_bool("BRIDGE_RESPONSES_DONE_SENTINEL", "true")
BRIDGE_RESPONSES_ALLOW_CLIENT_TOOLS = _env_bool("BRIDGE_RESPONSES_ALLOW_CLIENT_TOOLS", "false")
BRIDGE_RESPONSES_STREAM_TOOL_EVENTS = _env_bool("BRIDGE_RESPONSES_STREAM_TOOL_EVENTS", "true")
BRIDGE_RESPONSES_STREAM_ACTIVITY_EVENTS = _env_bool("BRIDGE_RESPONSES_STREAM_ACTIVITY_EVENTS", "true")
BRIDGE_RESPONSES_STREAM_REASONING_EVENTS = _env_bool("BRIDGE_RESPONSES_STREAM_REASONING_EVENTS", "true")
BRIDGE_RESPONSES_TOOL_OUTPUT_MAX_CHARS = int(os.getenv("BRIDGE_RESPONSES_TOOL_OUTPUT_MAX_CHARS", "8000"))
BRIDGE_RESPONSES_OUTPUT_DELTA_MAX_CHARS = int(os.getenv("BRIDGE_RESPONSES_OUTPUT_DELTA_MAX_CHARS", "1"))
BRIDGE_RESPONSES_REASONING_DELTA_MAX_CHARS = int(os.getenv("BRIDGE_RESPONSES_REASONING_DELTA_MAX_CHARS", "1"))
BRIDGE_HARD_INPUT_TOKEN_LIMIT = int(os.getenv("BRIDGE_HARD_INPUT_TOKEN_LIMIT", "0"))
BRIDGE_HARD_INPUT_HTTP_ERROR = _env_bool("BRIDGE_HARD_INPUT_HTTP_ERROR", "false")
BRIDGE_ALLOW_RAW_MEDIA_CONTEXT = _env_bool("BRIDGE_ALLOW_RAW_MEDIA_CONTEXT", "false")
BRIDGE_MEDIA_CONTEXT_MODE = os.getenv("BRIDGE_MEDIA_CONTEXT_MODE", "metadata").lower()
BRIDGE_MEDIA_GALLERY_AUTO_REGISTER_VIDEOS = _env_bool(
    "BRIDGE_MEDIA_GALLERY_AUTO_REGISTER_VIDEOS",
    "true",
)
BRIDGE_MEDIA_GALLERY_AUTO_REGISTER_IMAGES = _env_bool(
    "BRIDGE_MEDIA_GALLERY_AUTO_REGISTER_IMAGES",
    "true",
)
BRIDGE_DOCUMENT_RAG_AUTO_INGEST = _env_bool("BRIDGE_DOCUMENT_RAG_AUTO_INGEST", "true")
BRIDGE_DOCUMENT_RAG_INGEST_ROOT = os.getenv("BRIDGE_DOCUMENT_RAG_INGEST_ROOT", "/workspace/media-data")
BRIDGE_MEDIA_GALLERY_CONTAINER_ROOT = os.getenv("BRIDGE_MEDIA_GALLERY_CONTAINER_ROOT", "/media-data")
BRIDGE_MEDIA_GALLERY_TIMEOUT_SECONDS = float(os.getenv("BRIDGE_MEDIA_GALLERY_TIMEOUT_SECONDS", "45"))


def _is_server_model_manager_model(model: str) -> bool:
    normalized = (model or "").strip().lower()
    aliases = {
        SERVER_MODEL_MANAGER_MODEL_NAME.strip().lower(),
        "server-model-manager",
        "server_model_manager",
        "model-manager",
        "model_manager",
        "server-manager",
        "server_manager",
    }
    return normalized in aliases
ALPHARAVIS_MEDIA_GALLERY_URL = os.getenv(
    "ALPHARAVIS_MEDIA_GALLERY_URL",
    "http://media-gallery:8130",
).rstrip("/")
MEDIA_BLOCK_TYPES = {
    "image_url",
    "input_image",
    "video_url",
    "input_video",
    "audio_url",
    "input_audio",
    "file",
    "input_file",
}
VIDEO_MEDIA_BLOCK_TYPES = {"video_url", "input_video"}
IMAGE_MEDIA_BLOCK_TYPES = {"image_url", "input_image"}
BRIDGE_LLM_HEALTH_URL = os.getenv("BRIDGE_LLM_HEALTH_URL", "http://litellm:4000/v1").rstrip("/")
BRIDGE_LLM_HEALTH_API_KEY = os.getenv("BRIDGE_LLM_HEALTH_API_KEY", os.getenv("OPENAI_API_KEY", "sk-local-dev"))
BRIDGE_LLM_HEALTH_MODEL = os.getenv("BRIDGE_LLM_HEALTH_MODEL", "big-boss")
BRIDGE_LLM_HEALTH_FALLBACK_MODEL = os.getenv("BRIDGE_LLM_HEALTH_FALLBACK_MODEL", "edge-gemma")
BRIDGE_LLM_HEALTH_TIMEOUT_SECONDS = float(os.getenv("BRIDGE_LLM_HEALTH_TIMEOUT_SECONDS", "10"))
BRIDGE_LLM_HEALTH_PROMPT = os.getenv("BRIDGE_LLM_HEALTH_PROMPT", "Antworte nur mit OK.")
BRIDGE_ENABLE_LANGGRAPH_TOOL = _env_bool("BRIDGE_ENABLE_LANGGRAPH_TOOL", "false")
BRIDGE_LANGGRAPH_TOOL_API_KEY = os.getenv("BRIDGE_LANGGRAPH_TOOL_API_KEY", "")
BRIDGE_LANGGRAPH_TOOL_TIMEOUT_SECONDS = float(os.getenv("BRIDGE_LANGGRAPH_TOOL_TIMEOUT_SECONDS", "120"))
BRIDGE_SHOW_ERROR_CLASSIFICATION = _env_bool("BRIDGE_SHOW_ERROR_CLASSIFICATION", "true")
BRIDGE_SCRUB_INTERNAL_CONTEXT = _env_bool("BRIDGE_SCRUB_INTERNAL_CONTEXT", "true")
BRIDGE_ENABLE_CONTEXT_REFERENCES = _env_bool("BRIDGE_ENABLE_CONTEXT_REFERENCES", "true")
BRIDGE_ENABLE_APPROVAL_MEMORY = _env_bool("BRIDGE_ENABLE_APPROVAL_MEMORY", "true")
BRIDGE_APPROVAL_MEMORY_MAX = int(os.getenv("BRIDGE_APPROVAL_MEMORY_MAX", "200"))
BRIDGE_CONTEXT_REFERENCES_FETCH_URLS = _env_bool("BRIDGE_CONTEXT_REFERENCES_FETCH_URLS", "true")
BRIDGE_CONTEXT_REFERENCE_CONTEXT_LENGTH = int(
    os.getenv("BRIDGE_CONTEXT_REFERENCE_CONTEXT_LENGTH", str(BRIDGE_HARD_INPUT_TOKEN_LIMIT or 128000))
)
BRIDGE_CONTEXT_REFERENCE_SOFT_RATIO = float(os.getenv("BRIDGE_CONTEXT_REFERENCE_SOFT_RATIO", "0.25"))
BRIDGE_CONTEXT_REFERENCE_HARD_RATIO = float(os.getenv("BRIDGE_CONTEXT_REFERENCE_HARD_RATIO", "0.50"))
BRIDGE_CONTEXT_REFERENCE_MAX_URL_CHARS = int(os.getenv("BRIDGE_CONTEXT_REFERENCE_MAX_URL_CHARS", "12000"))
BRIDGE_CONTEXT_REFERENCE_FOLDER_LIMIT = int(os.getenv("BRIDGE_CONTEXT_REFERENCE_FOLDER_LIMIT", "200"))
BRIDGE_OBSERVER_MAX_RECORDS = int(os.getenv("BRIDGE_OBSERVER_MAX_RECORDS", "80"))
BRIDGE_OBSERVER_MAX_STRING_CHARS = int(os.getenv("BRIDGE_OBSERVER_MAX_STRING_CHARS", "240000"))
BRIDGE_OBSERVER_RECEIVE_PREVIEW_MAX_CHARS = int(os.getenv("BRIDGE_OBSERVER_RECEIVE_PREVIEW_MAX_CHARS", "240000"))
BRIDGE_ALLOW_USER_THREAD_KEY = os.getenv("BRIDGE_ALLOW_USER_THREAD_KEY", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
BRIDGE_WORKSPACE_ROOT = Path(
    os.getenv("BRIDGE_CONTEXT_REFERENCE_WORKSPACE_ROOT") or Path(__file__).resolve().parents[1]
).expanduser().resolve()
BRIDGE_CONTEXT_REFERENCE_CWD = Path(
    os.getenv("BRIDGE_CONTEXT_REFERENCE_CWD") or BRIDGE_WORKSPACE_ROOT
).expanduser().resolve()

app = FastAPI(title="AlphaRavis OpenAI Bridge", openapi_version="3.1.0")
_RESPONSES_STORE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_RESPONSES_INPUT_ITEMS: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
_APPROVAL_MEMORY: OrderedDict[str, float] = OrderedDict()
_BRIDGE_OBSERVATIONS: deque[dict[str, Any]] = deque(maxlen=max(1, BRIDGE_OBSERVER_MAX_RECORDS))


def _new_trace(protocol: str, body: dict[str, Any], request: Request) -> dict[str, Any]:
    metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
    trace_id = (
        metadata.get("trace_id")
        or metadata.get("alpha_trace_id")
        or request.headers.get("x-alpha-trace-id")
        or f"trace_{uuid.uuid4().hex[:12]}"
    )
    return {
        "trace_id": str(trace_id),
        "protocol": protocol,
        "steps": [],
    }


def _trace_step(
    trace: dict[str, Any] | None,
    name: str,
    request_started: float,
    *,
    duration_seconds: float | None = None,
    **fields: Any,
) -> None:
    if not isinstance(trace, dict):
        return
    step = {
        "name": name,
        "elapsed_seconds": round(time.perf_counter() - request_started, 3),
    }
    if duration_seconds is not None:
        step["duration_seconds"] = round(duration_seconds, 3)
    for key, value in fields.items():
        if value is not None:
            step[key] = value
    trace.setdefault("steps", []).append(step)


def _attach_trace_metadata(body: dict[str, Any], trace: dict[str, Any] | None) -> None:
    if not isinstance(trace, dict):
        return
    metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
    metadata = dict(metadata)
    metadata["trace_id"] = trace.get("trace_id")
    metadata["alpha_trace"] = trace
    body["metadata"] = metadata


def _merge_langgraph_trace(
    trace: dict[str, Any] | None,
    state: Any,
    *,
    request_started: float,
    graph_offset_seconds: float,
) -> None:
    if not isinstance(trace, dict):
        return
    values = _state_values(state)
    steps = values.get("alpha_trace_steps") if isinstance(values, dict) else None
    if not isinstance(steps, list):
        return
    for step in steps:
        if not isinstance(step, dict):
            continue
        graph_elapsed = float(step.get("elapsed_seconds") or 0.0)
        merged = dict(step)
        merged["elapsed_seconds"] = round(graph_offset_seconds + graph_elapsed, 3)
        trace.setdefault("steps", []).append(merged)


def _log_event(level: int | str, event: str, *, message: str = "", **fields: Any) -> None:
    if _op_log_event is None:
        return
    try:
        _op_log_event(level, event, component="bridge", message=message, **fields)
    except Exception:
        pass


def _log_exception(event: str, exc: BaseException, *, level: int | str = logging.ERROR, message: str = "", **fields: Any) -> None:
    if _op_log_exception is None:
        return
    try:
        _op_log_exception(event, exc, component="bridge", level=level, message=message, **fields)
    except Exception:
        pass


def _log_dependency(dependency: str, status: str, *, level: int | str = logging.INFO, message: str = "", **fields: Any) -> None:
    if _op_log_dependency_status is None:
        return
    try:
        _op_log_dependency_status(
            dependency,
            status,
            component="bridge",
            level=level,
            message=message,
            **fields,
        )
    except Exception:
        pass


@app.middleware("http")
async def _operational_request_logging(request: Request, call_next):
    started = time.perf_counter()
    request_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:12]}"
    try:
        response = await call_next(request)
    except Exception as exc:
        _log_exception(
            "bridge.request.failed",
            exc,
            path=request.url.path,
            method=request.method,
            request_id=request_id,
            client_host=getattr(request.client, "host", ""),
            elapsed_seconds=round(time.perf_counter() - started, 3),
        )
        raise
    _log_event(
        logging.INFO if response.status_code < 500 else logging.ERROR,
        "bridge.request.completed",
        path=request.url.path,
        method=request.method,
        status_code=response.status_code,
        request_id=request_id,
        client_host=getattr(request.client, "host", ""),
        elapsed_seconds=round(time.perf_counter() - started, 3),
    )
    return response


def _client():
    return get_client(url=LANGGRAPH_API_URL)


def _extract_thread_key(body: dict[str, Any], request: Request) -> str:
    metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
    candidates = [
        body.get("conversationId"),
        body.get("conversation_id"),
        metadata.get("conversationId"),
        metadata.get("conversation_id"),
        request.headers.get("x-conversation-id"),
        request.headers.get("x-thread-id"),
    ]
    for candidate in candidates:
        if candidate:
            return str(candidate)
    if BRIDGE_ALLOW_USER_THREAD_KEY and body.get("user"):
        return str(body["user"])
    user_fragment = str(body.get("user") or "anonymous")[:32]
    return f"ephemeral:{user_fragment}:{uuid.uuid4().hex}"


def _thread_id_for_key(thread_key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"alpharavis:librechat:{thread_key}"))


async def _ensure_thread(client: Any, thread_id: str, thread_key: str) -> str:
    try:
        await client.threads.create(
            thread_id=thread_id,
            if_exists="do_nothing",
            graph_id=LANGGRAPH_ASSISTANT_ID,
            metadata={"source": "librechat", "thread_key": thread_key},
        )
    except Exception as exc:
        if "409" not in str(exc) and "already" not in str(exc).lower():
            raise
    return thread_id


def _media_block_summary(part: dict[str, Any]) -> str:
    block_type = str(part.get("type") or "media")
    media_url = ""
    file_id = str(part.get("file_id") or part.get("id") or "")
    for key in ("image_url", "video_url", "audio_url", "file_url", "url"):
        value = part.get(key)
        if isinstance(value, dict):
            media_url = str(value.get("url") or "")
        elif isinstance(value, str):
            media_url = value
        if media_url:
            break
    mime_type = str(part.get("mime_type") or part.get("media_type") or "")
    title = str(part.get("filename") or part.get("name") or part.get("title") or "")
    fields = [f"type={block_type}"]
    if title:
        fields.append(f"title={title}")
    if file_id:
        fields.append(f"file_id={file_id}")
    if media_url:
        fields.append(f"url={media_url}")
    if mime_type:
        fields.append(f"mime_type={mime_type}")
    return "[Media attachment withheld from raw LLM context: " + ", ".join(fields) + "]"


def _media_part_source_url(part: dict[str, Any]) -> str:
    for key in ("video_url", "image_url", "audio_url", "file_url", "url"):
        value = part.get(key)
        if isinstance(value, dict):
            media_url = str(value.get("url") or "")
        elif isinstance(value, str):
            media_url = value
        else:
            media_url = ""
        if media_url:
            return media_url
    file_data = part.get("file_data")
    return str(file_data or "") if isinstance(file_data, str) else ""


def _media_part_title(part: dict[str, Any]) -> str:
    return str(part.get("filename") or part.get("name") or part.get("title") or "")


def _media_part_mime_type(part: dict[str, Any]) -> str:
    explicit = str(part.get("mime_type") or part.get("media_type") or "")
    if explicit.startswith(("image/", "video/", "audio/")):
        return explicit
    source_url = _media_part_source_url(part)
    if source_url.startswith("data:"):
        header = source_url.partition(",")[0]
        return header.removeprefix("data:").split(";", 1)[0]
    return explicit


def _is_video_media_part(part: dict[str, Any]) -> bool:
    block_type = str(part.get("type") or "")
    if block_type in VIDEO_MEDIA_BLOCK_TYPES:
        return True
    mime_type = _media_part_mime_type(part)
    if mime_type == "video" or mime_type.startswith("video/"):
        return True
    source_url = _media_part_source_url(part).lower().split("?", 1)[0].split("#", 1)[0]
    return source_url.endswith((".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"))


def _is_image_media_part(part: dict[str, Any]) -> bool:
    block_type = str(part.get("type") or "")
    if block_type in IMAGE_MEDIA_BLOCK_TYPES:
        return True
    mime_type = _media_part_mime_type(part)
    if mime_type == "image" or mime_type.startswith("image/"):
        return True
    source_url = _media_part_source_url(part).lower().split("?", 1)[0].split("#", 1)[0]
    return source_url.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".avif"))


def _media_gallery_type_for_part(part: dict[str, Any]) -> str:
    if _is_video_media_part(part):
        return "video"
    if _is_image_media_part(part):
        return "image"
    block_type = str(part.get("type") or "")
    mime_type = _media_part_mime_type(part).lower()
    source_url = _media_part_source_url(part).lower().split("?", 1)[0].split("#", 1)[0]
    title = _media_part_title(part).lower()
    document_ext = (
        ".pdf", ".doc", ".docx", ".html", ".htm", ".md", ".markdown", ".txt",
        ".log", ".csv", ".json", ".yaml", ".yml",
    )
    if (
        block_type in {"file", "input_file"}
        or mime_type.startswith(("application/pdf", "application/msword", "application/vnd.openxmlformats", "text/"))
        or any(source_url.endswith(ext) or title.endswith(ext) for ext in document_ext)
    ):
        return "document"
    return ""


def _replace_media_part_source_url(part: dict[str, Any], public_url: str) -> None:
    for key in ("video_url", "image_url", "audio_url", "file_url", "url"):
        value = part.get(key)
        if isinstance(value, dict):
            updated = dict(value)
            updated["url"] = public_url
            part[key] = updated
            return
        if isinstance(value, str):
            part[key] = public_url
            return
    if isinstance(part.get("file_data"), str):
        part["url"] = public_url
        return
    block_type = str(part.get("type") or "")
    part["image_url" if block_type in IMAGE_MEDIA_BLOCK_TYPES else "video_url"] = {"url": public_url}


def _bridge_media_source_key(part: dict[str, Any], source_url: str, media_type: str) -> str:
    file_id = str(part.get("file_id") or part.get("id") or "").strip()
    if file_id:
        return f"librechat:{file_id}"
    digest = hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:24]
    return f"librechat-{media_type or 'media'}-url:{digest}"


async def _mirror_media_part_to_media_gallery(
    part: dict[str, Any],
    *,
    thread_id: str,
    thread_key: str,
) -> dict[str, Any] | None:
    if part.get("alpharavis_media_gallery_url"):
        return None
    media_type = _media_gallery_type_for_part(part)
    if media_type == "video" and not BRIDGE_MEDIA_GALLERY_AUTO_REGISTER_VIDEOS:
        return None
    if media_type == "image" and not BRIDGE_MEDIA_GALLERY_AUTO_REGISTER_IMAGES:
        return None
    if media_type == "document" and not BRIDGE_DOCUMENT_RAG_AUTO_INGEST:
        return None
    if media_type not in {"image", "video", "document"}:
        return None
    source_url = _media_part_source_url(part)
    if not source_url:
        return None
    source_key = _bridge_media_source_key(part, source_url, media_type)
    title = _media_part_title(part) or source_key
    librechat_owned = bool(part.get("file_id") or part.get("id")) or source_url.startswith("data:") or any(
        marker in source_url for marker in ("librechat", "/api/files/", "/uploads/")
    )
    payload = {
        "source_url": source_url,
        "file_id": str(part.get("file_id") or part.get("id") or ""),
        "source_key": source_key,
        "thread_id": thread_id,
        "thread_key": thread_key,
        "group_id": thread_key or thread_id,
        "role": "input",
        "asset_kind": "original",
        "origin": "librechat_upload" if librechat_owned else "chat_url",
        "media_type": media_type,
        "mime_type": _media_part_mime_type(part),
        "title": title,
        "download": True,
        "metadata": {
            "bridge_auto_registered": True,
            "original_block_type": str(part.get("type") or ""),
            "original_source_key": source_key,
        },
    }
    try:
        async with httpx.AsyncClient(timeout=BRIDGE_MEDIA_GALLERY_TIMEOUT_SECONDS) as client:
            response = await client.post(f"{ALPHARAVIS_MEDIA_GALLERY_URL}/assets/register", json=payload)
        if response.status_code >= 400:
            _log_event(
                logging.WARNING,
                "bridge.media_gallery.register_failed",
                status_code=response.status_code,
                source_key=source_key,
                media_type=media_type,
                thread_id=thread_id,
            )
            return None
        record = response.json()
    except Exception as exc:
        _log_exception(
            "bridge.media_gallery.register_failed",
            exc,
            level=logging.WARNING,
            source_key=source_key,
            media_type=media_type,
            thread_id=thread_id,
        )
        return None

    public_url = str(record.get("public_url") or "")
    if not public_url or str(record.get("download_error") or ""):
        return record if isinstance(record, dict) else None
    original_url = source_url
    if media_type in {"image", "video"}:
        _replace_media_part_source_url(part, public_url)
    part["alpharavis_media_gallery_url"] = public_url
    part["alpharavis_original_media_url"] = original_url
    part["alpharavis_media_asset_id"] = str(record.get("asset_id") or "")
    if media_type == "document":
        ingest_path = _document_ingest_path_from_media_record(record)
        if ingest_path:
            part["alpharavis_document_ingest"] = {
                "path": ingest_path,
                "source_key": source_key,
                "title": title,
                "file_id": str(part.get("file_id") or part.get("id") or ""),
                "mime_type": _media_part_mime_type(part),
                "public_url": public_url,
                "asset_id": str(record.get("asset_id") or ""),
                "origin": "librechat_upload",
            }
    return record if isinstance(record, dict) else None


def _document_ingest_path_from_media_record(record: dict[str, Any]) -> str:
    relative_path = str(record.get("relative_path") or "").strip().lstrip("/")
    local_path = str(record.get("local_path") or "").strip()
    gallery_root = BRIDGE_MEDIA_GALLERY_CONTAINER_ROOT.rstrip("/")
    if local_path and gallery_root and local_path.startswith(gallery_root + "/"):
        relative_path = local_path[len(gallery_root) + 1 :]
    if relative_path:
        return str(Path(BRIDGE_DOCUMENT_RAG_INGEST_ROOT) / relative_path)
    return ""


def _collect_pending_document_ingests(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            ingest = part.get("alpharavis_document_ingest")
            if isinstance(ingest, dict) and ingest.get("path"):
                pending.append(dict(ingest))
    return pending[:20]


async def _mirror_video_part_to_media_gallery(
    part: dict[str, Any],
    *,
    thread_id: str,
    thread_key: str,
) -> dict[str, Any] | None:
    return await _mirror_media_part_to_media_gallery(part, thread_id=thread_id, thread_key=thread_key)


async def _mirror_video_parts_in_messages(
    messages: list[dict[str, Any]],
    *,
    thread_id: str,
    thread_key: str,
) -> list[dict[str, Any]]:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict):
                await _mirror_media_part_to_media_gallery(part, thread_id=thread_id, thread_key=thread_key)
    return messages


async def _mirror_video_parts_in_responses_body(
    body: dict[str, Any],
    request: Request,
) -> dict[str, Any]:
    raw_input = body.get("input")
    if not isinstance(raw_input, list):
        return body
    thread_key = _extract_thread_key(body, request)
    thread_id = _thread_id_for_key(thread_key)
    pseudo_messages: list[dict[str, Any]] = []
    for item in raw_input:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "") == "message" or item.get("role"):
            content = item.get("content")
            if isinstance(content, list):
                pseudo_messages.append({"content": content})
            continue
        if str(item.get("type") or "") in MEDIA_BLOCK_TYPES:
            pseudo_messages.append({"content": [item]})
    await _mirror_video_parts_in_messages(pseudo_messages, thread_id=thread_id, thread_key=thread_key)
    return body


def _sanitize_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    if BRIDGE_ALLOW_RAW_MEDIA_CONTEXT:
        return _responses_content_to_text(content)

    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
            continue
        if not isinstance(part, dict):
            parts.append(str(part))
            continue
        block_type = str(part.get("type") or "")
        if block_type in {"text", "input_text"} and isinstance(part.get("text"), str):
            parts.append(part["text"])
        elif block_type in MEDIA_BLOCK_TYPES:
            if BRIDGE_MEDIA_CONTEXT_MODE != "off":
                parts.append(_media_block_summary(part))
        elif isinstance(part.get("content"), str):
            parts.append(part["content"])
    return "\n".join(part for part in parts if part)


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        role = message.get("role", "user")
        if role == "assistant":
            role = "ai"
        elif role == "user":
            role = "human"
        normalized.append({"role": role, "content": _sanitize_message_content(message.get("content") or "")})
    return normalized


def _last_user_content(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "").strip()
    return ""


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _request_token_estimate(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        total += _approx_tokens(str(message.get("role", "")))
        total += _approx_tokens(_responses_content_to_text(message.get("content", "")))
    return total


def _hard_input_error(messages: list[dict[str, Any]]) -> str:
    if BRIDGE_HARD_INPUT_TOKEN_LIMIT <= 0:
        return ""
    estimate = _request_token_estimate(messages)
    if estimate <= BRIDGE_HARD_INPUT_TOKEN_LIMIT:
        return ""
    return (
        "Hard context cutoff: Diese Anfrage wird nicht an AlphaRavis gesendet, "
        f"weil sie ca. {estimate} Tokens umfasst und das Bridge-Limit "
        f"{BRIDGE_HARD_INPUT_TOKEN_LIMIT} ist. Bitte kuerze die Eingabe oder "
        "nutze Archiv-/RAG-Suche statt den ganzen Kontext direkt zu senden."
    )


def _observer_safe(value: Any, *, max_string_chars: int | None = None) -> Any:
    max_chars = max(0, max_string_chars if max_string_chars is not None else BRIDGE_OBSERVER_MAX_STRING_CHARS)
    if isinstance(value, str):
        if max_chars and len(value) > max_chars:
            return value[:max_chars] + f"\n...[truncated {len(value) - max_chars} chars]"
        return value
    if isinstance(value, dict):
        return {str(key): _observer_safe(item, max_string_chars=max_chars) for key, item in value.items()}
    if isinstance(value, list):
        return [_observer_safe(item, max_string_chars=max_chars) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _observer_headers(request: Request) -> dict[str, str]:
    names = [
        "x-conversation-id",
        "x-thread-id",
        "x-alpha-trace-id",
        "referer",
        "user-agent",
    ]
    return {name: request.headers.get(name, "") for name in names if request.headers.get(name)}


def _observer_start(
    *,
    protocol: str,
    request: Request,
    body: dict[str, Any],
    messages: list[dict[str, Any]],
) -> str:
    observation_id = f"obs_{uuid.uuid4().hex[:12]}"
    record = {
        "id": observation_id,
        "created_at": time.time(),
        "protocol": protocol,
        "path": getattr(getattr(request, "url", None), "path", ""),
        "method": getattr(request, "method", ""),
        "client_host": getattr(getattr(request, "client", None), "host", ""),
        "headers": _observer_headers(request),
        "stream": bool(body.get("stream")),
        "model": str(body.get("model") or OPENAI_MODEL_NAME),
        "status": "received",
        "send": {
            "raw_body": _observer_safe(body),
            "raw_messages": _observer_safe(messages),
            "raw_message_count": len(messages),
            "raw_token_estimate": _request_token_estimate(messages),
            "metadata": _observer_safe(body.get("metadata") if isinstance(body.get("metadata"), dict) else {}),
        },
        "receive": {
            "event_counts": {},
            "preview": "",
        },
    }
    _BRIDGE_OBSERVATIONS.appendleft(record)
    return observation_id


def _observer_record(observation_id: str, **updates: Any) -> None:
    for record in _BRIDGE_OBSERVATIONS:
        if record.get("id") == observation_id:
            record.update(_observer_safe(updates))
            return


def _observer_send_update(observation_id: str, **updates: Any) -> None:
    for record in _BRIDGE_OBSERVATIONS:
        if record.get("id") == observation_id:
            send = record.setdefault("send", {})
            if isinstance(send, dict):
                send.update(_observer_safe(updates))
            return


def _observer_receive_update(observation_id: str, **updates: Any) -> None:
    for record in _BRIDGE_OBSERVATIONS:
        if record.get("id") == observation_id:
            receive = record.setdefault("receive", {})
            if isinstance(receive, dict):
                receive.update(_observer_safe(updates))
            return


def _state_values(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    values = state.get("values")
    return values if isinstance(values, dict) else {}


def _state_message_profile(state: Any) -> dict[str, Any]:
    messages = _state_values(state).get("messages") or []
    if not isinstance(messages, list):
        messages = []
    latest = _message_content(messages[-1]) if messages else ""
    reasoning_chars = sum(len(_message_reasoning_content(message)) for message in messages)
    return {
        "message_count": len(messages),
        "visible_content_chars": sum(len(_message_content(message)) for message in messages),
        "reasoning_chars": reasoning_chars,
        "latest_message_preview": latest[:600],
    }


def _observer_prepared(observation_id: str, *, thread_key: str, thread_id: str, run_payload: dict[str, Any]) -> None:
    input_payload = run_payload.get("input") if isinstance(run_payload, dict) else None
    context_messages = []
    if isinstance(input_payload, dict) and isinstance(input_payload.get("messages"), list):
        context_messages = input_payload["messages"]
    _observer_record(
        observation_id,
        thread_key=thread_key,
        thread_id=thread_id,
        status="prepared",
    )
    _observer_send_update(
        observation_id,
        model_context=input_payload if isinstance(input_payload, dict) else {},
        model_context_messages=context_messages,
        model_context_message_count=len(context_messages),
        model_context_token_estimate=_request_token_estimate(context_messages),
        langgraph_state_profile=run_payload.get("state_profile", {}),
        direct_response=bool(run_payload.get("direct_response")),
        command=run_payload.get("command", {}),
    )


def _observer_hard_cutoff(observation_id: str, hard_error: str) -> None:
    _observer_record(observation_id, status="hard_cutoff")
    _observer_receive_update(
        observation_id,
        status="hard_cutoff",
        output_text=hard_error,
        preview=hard_error,
    )


def _observer_complete(
    observation_id: str,
    *,
    status: str = "completed",
    output_text: str = "",
    reasoning_text: str = "",
    elapsed_seconds: float | None = None,
) -> None:
    updates: dict[str, Any] = {"status": status}
    if elapsed_seconds is not None:
        updates["elapsed_seconds"] = round(elapsed_seconds, 3)
    _observer_record(observation_id, **updates)
    _observer_receive_update(
        observation_id,
        status=status,
        output_text=output_text,
        reasoning_text=reasoning_text,
        output_chars=len(output_text),
        reasoning_chars=len(reasoning_text),
        preview=(output_text or reasoning_text)[:BRIDGE_OBSERVER_RECEIVE_PREVIEW_MAX_CHARS],
    )


def _observer_note_event(observation_id: str, event_name: str) -> None:
    for record in _BRIDGE_OBSERVATIONS:
        if record.get("id") == observation_id:
            receive = record.setdefault("receive", {})
            if isinstance(receive, dict):
                counts = receive.setdefault("event_counts", {})
                if isinstance(counts, dict):
                    counts[event_name] = int(counts.get(event_name, 0)) + 1
            return


def _budget_profile_from_run_profile(profile: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(profile, dict):
        return {}
    budget = profile.get("final_context_budget") if isinstance(profile.get("final_context_budget"), dict) else {}
    result = dict(budget)
    mapping = {
        "context_length": ("pre_run_context_length", "context_length"),
        "detected_context_length": ("pre_run_detected_context_length", "detected_context_length"),
        "provider_reported_context_limit": ("pre_run_provider_reported_context_limit", "provider_reported_context_limit"),
        "message_tokens": ("pre_run_compression_tokens_after", "post_run_compression_tokens_after", "token_estimate"),
        "request_tokens": ("pre_run_request_tokens_after", "post_run_request_tokens_after", "request_token_estimate"),
        "static_context_reserve_tokens": (
            "static_context_reserve_tokens",
            "pre_run_static_context_reserve_tokens",
            "post_run_static_context_reserve_tokens",
            "handoff_static_context_reserve_tokens",
        ),
        "effective_active_limit": (
            "pre_run_effective_active_limit",
            "post_run_effective_context_limit",
            "handoff_effective_context_limit",
        ),
        "effective_hard_limit": ("pre_run_effective_hard_limit",),
        "active_limit": ("pre_run_active_limit",),
        "hard_limit": ("pre_run_hard_limit", "hard_context_limit"),
    }
    for target, keys in mapping.items():
        if result.get(target) is not None:
            continue
        for key in keys:
            if profile.get(key) is not None:
                result[target] = profile.get(key)
                break
    for key in (
        "final_budget_rescue_used",
        "final_budget_rescue_passes",
        "final_budget_rescue_max_passes",
        "final_budget_rescue_budget_met",
        "final_budget_rescue_archive_key",
        "hard_context_trim_used",
        "hard_context_rescued",
        "pre_run_compression_used",
        "pre_run_compression_passes",
        "pre_run_compression_max_passes",
        "pre_run_compression_budget_met",
        "post_run_compression_used",
        "handoff_context_guard_used",
        "provider_reported_context_limit",
        "provider_context_overflow_retry_used",
    ):
        if profile.get(key) is not None:
            result[key] = profile.get(key)
    classification = profile.get("provider_context_overflow_retry_classification")
    if isinstance(classification, dict):
        result["provider_context_overflow_retry_classification"] = classification
        if result.get("provider_reported_context_limit") is None and classification.get("provider_reported_context_limit"):
            result["provider_reported_context_limit"] = classification.get("provider_reported_context_limit")
    return result


def _compression_profile_from_run_profile(profile: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(profile, dict):
        return {}
    prefixes = (
        "pre_run_compression",
        "final_budget_rescue",
        "post_run_compression",
        "handoff_context",
    )
    result: dict[str, Any] = {}
    suffixes = (
        "used",
        "tokens",
        "tokens_after",
        "request_tokens",
        "request_tokens_after",
        "passes",
        "max_passes",
        "budget_met",
        "archive_key",
        "summary_failed",
        "summary_error",
        "compact_instructions",
        "compact_instructions_chars",
        "events",
        "middle_message_count",
        "head_message_count",
        "tail_message_count",
        "compression_token_limit",
        "summary_context_token_limit",
        "middle_token_estimate",
        "summary_prompt_pruned",
        "summary_prompt_original_tokens_estimate",
        "summary_prompt_tokens_estimate",
        "summary_prompt_token_limit",
        "summary_prompt_payload_token_limit",
        "summary_prompt_overhead_tokens_estimate",
        "summary_prompt_original_chars",
        "summary_prompt_chars",
        "summary_prompt_omitted_chars",
        "summary_chunking_used",
        "summary_chunk_count",
        "summary_chunk_chars",
        "summary_chunk_prompt_token_limit",
        "summary_chunk_payload_token_limit",
        "summary_chunk_prompt_overhead_tokens",
        "summary_chunk_overlap_chars",
        "summary_chunk_max_chunks",
        "summary_chunk_omitted_chars",
        "summary_chunk_output_tokens",
        "summary_chunk_summary_tokens_estimate",
        "summary_chunk_synthesis_pruned",
        "summary_chunk_synthesis_tokens_estimate",
        "summary_chunk_synthesis_payload_token_limit",
        "summary_chunk_synthesis_prompt_overhead_tokens",
        "pruned_tool_count",
        "deduped_tool_count",
        "tool_args_truncated_count",
        "workflow_event_count",
        "workflow_tool_call_count",
        "workflow_tool_result_count",
        "workflow_event_chars",
        "workflow_event_preview",
    )
    for prefix in prefixes:
        item = {suffix: profile.get(f"{prefix}_{suffix}") for suffix in suffixes if profile.get(f"{prefix}_{suffix}") is not None}
        if item:
            result[prefix] = item
    for key in ("compression_stats", "final_context_budget"):
        if isinstance(profile.get(key), dict):
            result[key] = profile[key]
    return result


def _large_ingest_profile_from_run_profile(profile: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(profile, dict):
        return {}
    result: dict[str, Any] = {}
    for field in ("document_ingests", "large_paste_ingests"):
        records = profile.get(field)
        if not isinstance(records, list):
            continue
        compact_records: list[dict[str, Any]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            manifest = record.get("source_manifest") if isinstance(record.get("source_manifest"), dict) else {}
            events = record.get("events") if isinstance(record.get("events"), list) else []
            latest_event = events[-1] if events and isinstance(events[-1], dict) else {}
            compact: dict[str, Any] = {
                "source_key": manifest.get("source_key") or record.get("source_key") or latest_event.get("source_key") or "",
                "source_type": manifest.get("source_type") or record.get("source_type") or "",
                "title": manifest.get("title") or record.get("source_title") or "",
                "content_type": manifest.get("content_type") or record.get("content_type") or "",
                "index_status": manifest.get("index_status") or record.get("index_status") or latest_event.get("status") or "",
                "rag_file_id": manifest.get("rag_file_id") or record.get("rag_file_id") or "",
                "rag_active": bool(manifest.get("rag_active") if "rag_active" in manifest else record.get("rag_active")),
                "message_replaced": bool(record.get("message_replaced")),
                "manual_rag_block": bool(manifest.get("manual_rag_block") if "manual_rag_block" in manifest else record.get("manual_rag_block")),
                "paste_intent": manifest.get("paste_intent") or record.get("paste_intent") or "",
                "content_chars": manifest.get("content_chars") or record.get("content_chars") or latest_event.get("content_chars") or 0,
                "indexed_content_chars": manifest.get("indexed_content_chars") or record.get("indexed_content_chars") or 0,
                "chunk_count": manifest.get("chunk_count") or record.get("chunk_count") or latest_event.get("chunk_count") or 0,
                "indexed_chunk_count": manifest.get("indexed_chunk_count") or record.get("indexed_chunk_count") or 0,
                "source_digest": manifest.get("source_digest") or record.get("source_digest") or latest_event.get("source_digest") or "",
                "indexed_backends": list(manifest.get("indexed_backends") or record.get("indexed_backends") or []),
                "queued_backends": list(manifest.get("queued_backends") or record.get("queued_backends") or []),
                "skip_reason": record.get("skip_reason") or latest_event.get("reason") or "",
                "elapsed_seconds": record.get("elapsed_seconds") or latest_event.get("t") or 0,
                "latest_event": latest_event,
            }
            compact_records.append(compact)
        if compact_records:
            result[field] = compact_records
    return result


def _observer_note_budget(observation_id: str, *, node_name: str, profile: dict[str, Any]) -> None:
    budget = _budget_profile_from_run_profile(profile)
    compression = _compression_profile_from_run_profile(profile)
    ingests = _large_ingest_profile_from_run_profile(profile)
    if not budget and not compression and not ingests:
        return
    if budget:
        budget["node"] = node_name
    if compression:
        compression["node"] = node_name
    if ingests:
        ingests["node"] = node_name
    for record in _BRIDGE_OBSERVATIONS:
        if record.get("id") == observation_id:
            receive = record.setdefault("receive", {})
            if isinstance(receive, dict):
                if budget:
                    receive["context_budget"] = _observer_safe(budget)
                if compression:
                    receive["compression"] = _observer_safe(compression)
                if ingests:
                    receive["source_ingests"] = _observer_safe(ingests)
            return


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")

    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                block_type = part.get("type")
                if block_type in {"thinking", "reasoning"}:
                    continue
                if isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
                elif isinstance(part.get("content"), str):
                    text_parts.append(part["content"])
            else:
                text_parts.append(str(part))
        return "".join(text_parts)
    return str(content)


def _message_reasoning_content(message: Any) -> str:
    if isinstance(message, dict):
        candidates = [
            message.get("reasoning_content"),
            message.get("reasoning"),
            (message.get("additional_kwargs") or {}).get("reasoning_content")
            if isinstance(message.get("additional_kwargs"), dict)
            else None,
        ]
        content = message.get("content", "")
    else:
        additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
        candidates = [
            getattr(message, "reasoning_content", None),
            getattr(message, "reasoning", None),
            additional_kwargs.get("reasoning_content") if isinstance(additional_kwargs, dict) else None,
        ]
        content = getattr(message, "content", "")

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate

    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"thinking", "reasoning"}:
                if isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
                elif isinstance(part.get("content"), str):
                    text_parts.append(part["content"])
        return "".join(text_parts)
    return ""


class _VisibleThinkingSplitter:
    OPEN_TAGS = ("<think>", "<thinking>")
    CLOSE_TAGS = ("</think>", "</thinking>")

    def __init__(self) -> None:
        self._pending = ""
        self._inside = False

    @staticmethod
    def _longest_pending_prefix(text: str, tags: tuple[str, ...]) -> int:
        max_len = 0
        max_tag_len = max(len(tag) for tag in tags)
        for size in range(1, min(len(text), max_tag_len - 1) + 1):
            suffix = text[-size:]
            if any(tag.startswith(suffix) for tag in tags):
                max_len = size
        return max_len

    @staticmethod
    def _next_tag(text: str, tags: tuple[str, ...]) -> tuple[int, str] | None:
        matches = [(index, tag) for tag in tags if (index := text.find(tag)) >= 0]
        if not matches:
            return None
        return min(matches, key=lambda item: item[0])

    def feed(self, text: str, *, emit_reasoning: bool = True) -> tuple[str, str]:
        if not text:
            return "", ""

        visible_parts: list[str] = []
        reasoning_parts: list[str] = []
        remaining = self._pending + text
        self._pending = ""

        while remaining:
            tags = self.CLOSE_TAGS if self._inside else self.OPEN_TAGS
            match = self._next_tag(remaining, tags)
            if match is not None:
                index, tag = match
                before = remaining[:index]
                if self._inside:
                    if emit_reasoning:
                        reasoning_parts.append(before)
                else:
                    visible_parts.append(before)
                remaining = remaining[index + len(tag) :]
                self._inside = not self._inside
                continue

            pending_size = self._longest_pending_prefix(remaining, tags)
            if pending_size:
                emit_part = remaining[:-pending_size]
                self._pending = remaining[-pending_size:]
            else:
                emit_part = remaining

            if self._inside:
                if emit_reasoning:
                    reasoning_parts.append(emit_part)
            else:
                visible_parts.append(emit_part)
            break

        return "".join(visible_parts), "".join(reasoning_parts)

    def flush(self, *, emit_reasoning: bool = True) -> tuple[str, str]:
        pending = self._pending
        self._pending = ""
        was_inside = self._inside
        self._inside = False
        if not pending:
            return "", ""
        if was_inside:
            if pending.startswith("</t") and any(tag.startswith(pending) for tag in self.CLOSE_TAGS):
                return "", ""
            return "", pending if emit_reasoning else ""
        if pending.startswith("<t") and any(tag.startswith(pending) for tag in self.OPEN_TAGS):
            return "", ""
        return pending, ""


def _split_visible_thinking_once(text: str, *, emit_reasoning: bool = True) -> tuple[str, str]:
    splitter = _VisibleThinkingSplitter()
    visible, reasoning = splitter.feed(text, emit_reasoning=emit_reasoning)
    visible_tail, reasoning_tail = splitter.flush(emit_reasoning=emit_reasoning)
    return visible + visible_tail, reasoning + reasoning_tail


def _message_type(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("type") or message.get("role") or "").lower()
    return str(getattr(message, "type", getattr(message, "role", ""))).lower()


def _is_ai_message(message: Any) -> bool:
    message_type = _message_type(message)
    return message_type in {"ai", "assistant"} or "aimessage" in message_type


def _is_human_message(message: Any) -> bool:
    message_type = _message_type(message)
    return message_type in {"human", "user"} or "humanmessage" in message_type


def _is_tool_message(message: Any) -> bool:
    message_type = _message_type(message)
    return message_type == "tool" or "toolmessage" in message_type


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_tool_calls_from_message(message: Any) -> list[dict[str, Any]]:
    raw = _get_value(message, "tool_calls", None)
    if raw is None:
        additional = _get_value(message, "additional_kwargs", {})
        if isinstance(additional, dict):
            raw = additional.get("tool_calls")
    if not isinstance(raw, list):
        return []

    calls: list[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        function = item.get("function") if isinstance(item.get("function"), dict) else {}
        name = item.get("name") or function.get("name") or item.get("title") or f"tool_{idx + 1}"
        args = item.get("args")
        if args is None:
            args = function.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"arguments": args}
        calls.append(
            {
                "id": str(item.get("id") or item.get("tool_call_id") or f"call_{uuid.uuid4().hex[:12]}"),
                "name": str(name),
                "args": args if isinstance(args, dict) else {"value": args},
            }
        )
    return calls


def _extract_tool_result(message: Any) -> tuple[str, str, str] | None:
    candidate = message.get("message") if isinstance(message, dict) and "message" in message else message
    if not _is_tool_message(candidate) and not (
        isinstance(candidate, dict) and ("tool_call_id" in candidate or candidate.get("type") == "tool")
    ):
        return None

    tool_call_id = str(_get_value(candidate, "tool_call_id", "") or _get_value(candidate, "id", "") or "")
    if not tool_call_id:
        tool_call_id = f"call_{uuid.uuid4().hex[:12]}"
    name = str(_get_value(candidate, "name", "") or _get_value(candidate, "tool", "") or "tool")
    return tool_call_id, name, _message_content(candidate)


def _last_ai_content(state: Any) -> str:
    state = _state_values(state)
    messages = state.get("messages", []) if isinstance(state, dict) else []
    trailing_notices: list[str] = []
    for message in reversed(messages):
        if _is_ai_message(message):
            content = _message_content(message)
            visible_content = _visible_content(content).strip()
            if not visible_content:
                continue
            stripped = content.lstrip()
            if stripped.startswith(("Memory-Notice:", "Run-Profile:", "Fast-Path-Notice:")):
                trailing_notices.append(content)
                continue
            if trailing_notices:
                return f"{content}\n\n" + "\n".join(reversed(trailing_notices))
            return content
    return ""


def _state_failure_message(state: Any) -> str:
    state = _state_values(state)
    if not isinstance(state, dict):
        return ""
    steps = state.get("alpha_trace_steps")
    if not isinstance(steps, list):
        return ""
    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        name = str(step.get("name") or "")
        if not name.endswith(".failed") and ".failed." not in name:
            continue
        error_type = str(step.get("error_type") or "unknown_error")
        classification = str(step.get("classification") or "").strip()
        suffix = f" ({classification})" if classification else ""
        return (
            "AlphaRavis hat keine sichtbare finale Antwort erzeugt, weil ein "
            f"interner Schritt fehlgeschlagen ist: {name}: {error_type}{suffix}."
        )
    return ""


def _state_values(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    values = state.get("values")
    return values if isinstance(values, dict) else state


def _state_messages(state: Any) -> list[Any]:
    values = _state_values(state)
    messages = values.get("messages", [])
    return messages if isinstance(messages, list) else []


def _last_human_content_from_state(state: Any) -> str:
    for message in reversed(_state_messages(state)):
        if _is_human_message(message):
            return _message_content(message).strip()
    return ""


def _latest_human_message(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for message in reversed(messages):
        if message.get("role") == "human":
            return message
    return None


def _messages_after_last_human(messages: list[dict[str, Any]], last_human_content: str) -> list[dict[str, Any]]:
    if not last_human_content:
        return messages

    last_seen = -1
    for index, message in enumerate(messages):
        if message.get("role") == "human" and str(message.get("content") or "").strip() == last_human_content:
            last_seen = index

    if last_seen < 0:
        latest = _latest_human_message(messages)
        return [latest] if latest else messages

    new_human_messages = [
        message for message in messages[last_seen + 1 :] if message.get("role") == "human"
    ]
    if new_human_messages:
        return new_human_messages

    latest = _latest_human_message(messages)
    return [latest] if latest else []


async def _apply_context_references_to_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not BRIDGE_ENABLE_CONTEXT_REFERENCES:
        return messages, []

    output: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "")
        content = message.get("content")
        if role not in {"human", "user"} or not isinstance(content, str) or "@" not in content:
            output.append(message)
            continue

        result = await preprocess_context_references(
            content,
            cwd=BRIDGE_CONTEXT_REFERENCE_CWD,
            allowed_root=BRIDGE_WORKSPACE_ROOT,
            context_length=BRIDGE_CONTEXT_REFERENCE_CONTEXT_LENGTH,
            soft_ratio=BRIDGE_CONTEXT_REFERENCE_SOFT_RATIO,
            hard_ratio=BRIDGE_CONTEXT_REFERENCE_HARD_RATIO,
            max_url_chars=BRIDGE_CONTEXT_REFERENCE_MAX_URL_CHARS,
            folder_limit=BRIDGE_CONTEXT_REFERENCE_FOLDER_LIMIT,
            fetch_urls=BRIDGE_CONTEXT_REFERENCES_FETCH_URLS,
        )
        if result.references:
            profiles.append(result.profile())
        if result.expanded:
            updated = dict(message)
            updated["content"] = result.message
            output.append(updated)
        else:
            output.append(message)

    return output, profiles


async def _build_input_payload(
    raw_messages: list[dict[str, Any]],
    state: Any,
    *,
    thread_id: str,
    thread_key: str,
    model: str = "",
    trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    await _mirror_video_parts_in_messages(raw_messages, thread_id=thread_id, thread_key=thread_key)
    normalized = _normalize_messages(raw_messages)
    if BRIDGE_MESSAGE_SYNC_MODE in {"full", "all"}:
        selected = normalized
    else:
        last_human = _last_human_content_from_state(state)
        selected = _messages_after_last_human(normalized, last_human)
        if not selected and normalized:
            selected = [normalized[-1]]

    selected, reference_profiles = await _apply_context_references_to_messages(selected)
    pending_document_ingests = _collect_pending_document_ingests(raw_messages)

    payload = {
        "messages": selected,
        "thread_id": thread_id,
        "thread_key": thread_key,
        "bridge_context_references": reference_profiles,
    }
    if _is_server_model_manager_model(model):
        payload.update(
            {
                "active_agent": "power_management_agent",
                "selected_toolsets": ["agent/power"],
                "fast_path_locked": True,
                "fast_path_lock_reason": "server_model_manager_endpoint",
                "server_model_manager_mode": True,
            }
        )
    if pending_document_ingests:
        payload["pending_document_ingests"] = pending_document_ingests
    if isinstance(trace, dict):
        payload["alpha_trace"] = {
            "trace_id": trace.get("trace_id"),
            "protocol": trace.get("protocol"),
        }
    return payload


def _chat_completion_response(content: str, model: str) -> dict[str, Any]:
    content = _visible_content(content)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def _chunk(
    content: str,
    model: str,
    *,
    role: str | None = None,
    finish_reason: str | None = None,
    reasoning_content: str | None = None,
    alpha_reasoning_kind: str | None = None,
    alpha_reasoning_label: str | None = None,
) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if role:
        delta["role"] = role
    if content:
        delta["content"] = content
    if reasoning_content:
        delta[BRIDGE_REASONING_DELTA_FIELD] = reasoning_content
        if alpha_reasoning_kind:
            delta["alpha_reasoning_kind"] = alpha_reasoning_kind
        if alpha_reasoning_label:
            delta["alpha_reasoning_label"] = alpha_reasoning_label

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _stream_data(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _activity_chunk(text: str, model: str) -> str:
    return _stream_data(_chunk(f"\n\nStatus: {text}\n", model))


def _openai_stream_response(content: str, model: str) -> list[str]:
    content = _visible_content(content)
    return [
        _stream_data(_chunk("", model, role="assistant")),
        _stream_data(_chunk(content, model)),
        _stream_data(_chunk("", model, finish_reason="stop")),
        "data: [DONE]\n\n",
    ]


def _clean_error_message(exc: Exception) -> str:
    return format_user_error(exc, component="AlphaRavis Bridge")


def _error_activity_text(exc: Exception) -> str:
    classified = classify_api_error(exc, provider="bridge", model=OPENAI_MODEL_NAME)
    return f"Fehler klassifiziert: {classified.reason.value}; Aktion: {classified.action}."


def _require_langgraph_tool_access(request: Request) -> None:
    if not BRIDGE_ENABLE_LANGGRAPH_TOOL:
        raise HTTPException(status_code=404, detail="LangGraph tool endpoint is disabled.")

    if not BRIDGE_LANGGRAPH_TOOL_API_KEY:
        return

    expected = f"Bearer {BRIDGE_LANGGRAPH_TOOL_API_KEY}"
    if request.headers.get("authorization") != expected:
        raise HTTPException(status_code=401, detail="Invalid LangGraph tool API key.")


def _tool_thread_id_for_key(thread_key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"alpharavis:hermes-tool:{thread_key}"))


async def _smoke_test_litellm_model(model: str) -> dict[str, Any]:
    started = time.perf_counter()
    headers = {"Content-Type": "application/json"}
    if BRIDGE_LLM_HEALTH_API_KEY:
        headers["Authorization"] = f"Bearer {BRIDGE_LLM_HEALTH_API_KEY}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": BRIDGE_LLM_HEALTH_PROMPT}],
        "max_tokens": 8,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        async with httpx.AsyncClient(timeout=BRIDGE_LLM_HEALTH_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{BRIDGE_LLM_HEALTH_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
        elapsed = round(time.perf_counter() - started, 3)
        if response.status_code >= 400:
            _log_dependency(
                "llm_health_backend",
                "error",
                level=logging.WARNING,
                model=model,
                status_code=response.status_code,
                elapsed_seconds=elapsed,
                url=BRIDGE_LLM_HEALTH_URL,
            )
            return {
                "ok": False,
                "model": model,
                "status_code": response.status_code,
                "elapsed_seconds": elapsed,
                "error": response.text[:500],
            }

        data = response.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        content = str(message.get("content") or "").strip()
        reasoning = str(message.get("reasoning_content") or "").strip()
        _log_dependency(
            "llm_health_backend",
            "ready" if (content or reasoning) else "empty_response",
            level=logging.INFO if (content or reasoning) else logging.WARNING,
            model=model,
            status_code=response.status_code,
            elapsed_seconds=elapsed,
            url=BRIDGE_LLM_HEALTH_URL,
        )
        return {
            "ok": bool(content or reasoning),
            "model": model,
            "status_code": response.status_code,
            "elapsed_seconds": elapsed,
            "content_preview": (content or reasoning)[:120],
        }
    except Exception as exc:
        _log_exception(
            "bridge.llm_health.failed",
            exc,
            level=logging.WARNING,
            dependency="llm_health_backend",
            model=model,
            url=BRIDGE_LLM_HEALTH_URL,
            elapsed_seconds=round(time.perf_counter() - started, 3),
        )
        return {
            "ok": False,
            "model": model,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "error": str(exc),
        }


def _approval_resume_from_messages(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    content = _last_user_content(messages)
    lowered = content.lower().strip()
    if not lowered:
        return None

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and parsed.get("action"):
            return parsed
    except Exception:
        pass

    approve_memory_phrases = {
        "approve always",
        "always approve",
        "allow always",
        "always allow",
        "approve this command",
        "allow this command",
        "approve for this chat",
        "allow for this chat",
        "do not ask again",
        "don't ask again",
        "dont ask again",
        "remember approval",
        "remember this command",
        "immer erlauben",
        "immer genehmigen",
        "immer freigeben",
        "diesen befehl erlauben",
        "diesen befehl genehmigen",
        "diesen befehl merken",
        "fuer diesen chat erlauben",
        "für diesen chat erlauben",
        "fuer diesen befehl erlauben",
        "für diesen befehl erlauben",
        "fuer diesen befehl nicht mehr fragen",
        "für diesen befehl nicht mehr fragen",
        "nicht mehr fragen",
        "ja immer",
        "ok immer",
    }
    if any(phrase in lowered for phrase in approve_memory_phrases):
        return {"action": "approve", "remember": "thread_command"}

    approve_words = {"approve", "approved", "yes", "ja", "ok", "go", "genehmigt", "mach", "mach es"}
    reject_words = {"reject", "rejected", "no", "nein", "stop", "abbrechen", "ablehnen"}
    if lowered in approve_words:
        return {"action": "approve"}
    if lowered in reject_words:
        return {"action": "reject"}

    replace_prefixes = ("replace:", "command:", "ersetze:", "ändere zu:", "aendere zu:")
    for prefix in replace_prefixes:
        if lowered.startswith(prefix):
            replacement = content.split(":", 1)[1].strip()
            if replacement:
                return {"action": "replace", "command": replacement}

    return None


def _approval_fingerprint(interrupt_value: dict[str, Any]) -> str:
    payload = {
        "scope": str(interrupt_value.get("scope") or ""),
        "target": str(interrupt_value.get("target") or ""),
        "command": str(interrupt_value.get("command") or "").strip(),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _approval_memory_key(thread_id: str, interrupt_value: dict[str, Any]) -> str:
    return f"{thread_id}:{_approval_fingerprint(interrupt_value)}"


def _approval_should_remember(resume: dict[str, Any]) -> bool:
    action = str(resume.get("action") or "").lower().strip()
    if action not in {"approve", "approved", "yes", "ja", "genehmigt"}:
        return False
    remember = resume.get("remember")
    if isinstance(remember, bool):
        return remember
    marker = str(remember or resume.get("scope") or resume.get("remember_scope") or "").lower().strip()
    return marker in {"thread_command", "chat_command", "this_chat", "conversation", "always", "true", "1"}


def _remember_approval(thread_id: str, interrupt_value: dict[str, Any], resume: dict[str, Any]) -> None:
    if not BRIDGE_ENABLE_APPROVAL_MEMORY or not _approval_should_remember(resume):
        return
    key = _approval_memory_key(thread_id, interrupt_value)
    _APPROVAL_MEMORY[key] = time.time()
    _APPROVAL_MEMORY.move_to_end(key)
    while len(_APPROVAL_MEMORY) > max(1, BRIDGE_APPROVAL_MEMORY_MAX):
        _APPROVAL_MEMORY.popitem(last=False)


def _approval_auto_resume(thread_id: str, interrupt_value: dict[str, Any]) -> dict[str, Any] | None:
    if not BRIDGE_ENABLE_APPROVAL_MEMORY:
        return None
    key = _approval_memory_key(thread_id, interrupt_value)
    if key not in _APPROVAL_MEMORY:
        return None
    _APPROVAL_MEMORY.move_to_end(key)
    return {"action": "approve", "remembered": "thread_command"}


def _find_command_approval_interrupt(state: Any) -> dict[str, Any] | None:
    def visit(obj: Any) -> dict[str, Any] | None:
        if isinstance(obj, dict):
            if obj.get("type") == "command_approval":
                return obj
            value = obj.get("value")
            if isinstance(value, dict) and value.get("type") == "command_approval":
                return value
            for nested in obj.values():
                found = visit(nested)
                if found:
                    return found
        elif isinstance(obj, (list, tuple)):
            for nested in obj:
                found = visit(nested)
                if found:
                    return found
        else:
            value = getattr(obj, "value", None)
            if isinstance(value, dict) and value.get("type") == "command_approval":
                return value
        return None

    return visit(state)


def _approval_prompt(interrupt_value: dict[str, Any]) -> str:
    return (
        "Ein Debugger-Befehl wartet auf Freigabe.\n\n"
        f"Ziel: {interrupt_value.get('target', 'unknown')}\n"
        f"Befehl: `{interrupt_value.get('command', '')}`\n"
        f"Risiko: {interrupt_value.get('risk', 'unknown')}\n\n"
        "Antworte mit `approve`, `reject` oder `replace: <sichererer Befehl>`.\n"
        "Mit `approve always` oder `immer erlauben` wird exakt dieser Befehl in diesem Chat gemerkt."
    )


async def _prepare_run_payload(
    client: Any,
    thread_id: str,
    thread_key: str,
    messages: list[dict[str, Any]],
    model: str = "",
    trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        state = await client.threads.get_state(thread_id)
    except Exception:
        state = None
    state_profile = _state_message_profile(state)

    interrupt_value = _find_command_approval_interrupt(state)
    if interrupt_value:
        auto_resume = _approval_auto_resume(thread_id, interrupt_value)
        if auto_resume is not None:
            return {"command": {"resume": auto_resume}, "state_profile": state_profile}
        resume = _approval_resume_from_messages(messages)
        if resume is None:
            return {"direct_response": _approval_prompt(interrupt_value), "state_profile": state_profile}
        _remember_approval(thread_id, interrupt_value, resume)
        return {"command": {"resume": resume}, "state_profile": state_profile}

    input_payload = await _build_input_payload(
        messages,
        state,
        thread_id=thread_id,
        thread_key=thread_key,
        model=model,
        trace=trace,
    )
    return {"input": input_payload, "state_profile": state_profile}


def _extract_stream_text(part: Any) -> str:
    data = getattr(part, "data", None)
    if data is None and isinstance(part, dict):
        data = part.get("data")

    if isinstance(data, tuple) and data:
        return _message_content(data[0]) if _is_ai_message(data[0]) else ""

    if isinstance(data, list) and data:
        for message in reversed(data):
            if _is_ai_message(message):
                return _message_content(message)
        return ""

    if isinstance(data, dict):
        if "chunk" in data:
            return _message_content(data["chunk"]) if _is_ai_message(data["chunk"]) else ""
        if "messages" in data and data["messages"]:
            for message in reversed(data["messages"]):
                if _is_ai_message(message):
                    return _message_content(message)

    return ""


def _extract_stream_reasoning(part: Any, *, force: bool = False) -> str:
    if not force and not BRIDGE_STREAM_REASONING_EVENTS:
        return ""

    data = getattr(part, "data", None)
    if data is None and isinstance(part, dict):
        data = part.get("data")

    if isinstance(data, tuple) and data:
        return _message_reasoning_content(data[0]) if _is_ai_message(data[0]) else ""

    if isinstance(data, list) and data:
        for message in reversed(data):
            if _is_ai_message(message):
                return _message_reasoning_content(message)
        return ""

    if isinstance(data, dict):
        if "chunk" in data:
            return _message_reasoning_content(data["chunk"]) if _is_ai_message(data["chunk"]) else ""
        if "messages" in data and data["messages"]:
            for message in reversed(data["messages"]):
                if _is_ai_message(message):
                    return _message_reasoning_content(message)

    return ""


def _stream_part_is_delta(part: Any) -> bool:
    data = getattr(part, "data", None)
    if data is None and isinstance(part, dict):
        data = part.get("data")

    if isinstance(data, tuple) and data:
        message = data[0]
        type_name = type(message).__name__.lower()
        return "chunk" in type_name or "chunk" in _message_type(message)

    if isinstance(data, dict) and "chunk" in data:
        return True

    return False


def _stream_event_name(part: Any) -> str:
    if isinstance(part, dict):
        return str(part.get("event") or "")
    return str(getattr(part, "event", ""))


def _stream_event_data(part: Any) -> Any:
    if isinstance(part, dict):
        return part.get("data")
    return getattr(part, "data", None)


def _find_langgraph_node_value(value: Any, *, depth: int = 0) -> str:
    if depth > 6:
        return ""
    if isinstance(value, dict):
        for key in ("langgraph_node", "langgraph_node_name", "node"):
            node = value.get(key)
            if isinstance(node, str) and node:
                return node
        for key in ("langgraph_path", "langgraph_triggers", "tags"):
            nested = value.get(key)
            if isinstance(nested, (list, tuple)):
                for item in nested:
                    text = str(item)
                    if text in {"planner", "alpha_ravis_swarm", "fast_chat"}:
                        return text
        for nested in value.values():
            node = _find_langgraph_node_value(nested, depth=depth + 1)
            if node:
                return node
    elif isinstance(value, (list, tuple)):
        for nested in value:
            node = _find_langgraph_node_value(nested, depth=depth + 1)
            if node:
                return node
    return ""


def _message_id(message: Any) -> str:
    return str(_get_value(message, "id", "") or "")


def _stream_message_metadata_nodes(part: Any) -> dict[str, str]:
    if _stream_event_name(part) != "messages/metadata":
        return {}
    data = _stream_event_data(part)
    if not isinstance(data, dict):
        return {}
    nodes: dict[str, str] = {}
    for message_id, metadata in data.items():
        node = _find_langgraph_node_value(metadata)
        if node:
            nodes[str(message_id)] = node
    return nodes


def _stream_message_node_from_metadata(part: Any, message_nodes: dict[str, str] | None) -> str:
    if not message_nodes:
        return ""
    for message in _stream_messages_from_part(part):
        node = message_nodes.get(_message_id(message))
        if node:
            return node
    return ""


def _stream_message_delta_key(part: Any) -> str:
    for message in _stream_messages_from_part(part):
        message_id = _message_id(message)
        if message_id:
            return message_id
    return "__default__"


def _stream_langgraph_node(part: Any, *, message_nodes: dict[str, str] | None = None) -> str:
    node = _stream_message_node_from_metadata(part, message_nodes)
    if node:
        return node
    data = _stream_event_data(part)
    metadata_candidates: list[Any] = []
    if not isinstance(part, dict):
        metadata_candidates.append(getattr(part, "metadata", None))
        metadata_candidates.append(getattr(part, "run_metadata", None))
    if isinstance(data, tuple) and len(data) > 1:
        metadata_candidates.append(data[1])
    elif isinstance(data, dict):
        metadata_candidates.extend(
            [
                data.get("metadata"),
                data.get("config"),
                data.get("run_metadata"),
                data.get("checkpoint"),
            ]
        )
        chunk = data.get("chunk")
        if isinstance(chunk, dict):
            metadata_candidates.extend([chunk.get("metadata"), chunk.get("response_metadata")])
    for candidate in metadata_candidates:
        node = _find_langgraph_node_value(candidate)
        if node:
            return node
    return ""


def _stream_text_is_internal_reasoning(part: Any, *, message_nodes: dict[str, str] | None = None) -> bool:
    return _stream_langgraph_node(part, message_nodes=message_nodes) in {"planner"}


def _clean_internal_plan_text(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    text = re.sub(r"</?execution-plan>\s*", "", text).strip()
    text = re.sub(
        r"^\[System note: compact plan for the current agent run\.[^\]]*\]\s*",
        "",
        text,
        flags=re.DOTALL,
    ).strip()
    return text


def _extract_internal_plan_update(part: Any) -> str:
    if _stream_event_name(part) != "updates":
        return ""
    data = _stream_event_data(part)
    if not isinstance(data, dict):
        return ""

    planner_data = data.get("planner")
    if not isinstance(planner_data, dict):
        return ""
    for key in ("planner_context", "plan", "planner_output", "content"):
        value = planner_data.get(key)
        if isinstance(value, str) and value.strip():
            return _clean_internal_plan_text(value)
    return ""


def _extract_activity_text(part: Any, *, force: bool = False) -> str:
    if not force and (not BRIDGE_SHOW_ACTIVITY_EVENTS or BRIDGE_ACTIVITY_DETAIL == "off"):
        return ""

    event = _stream_event_name(part)
    data = _stream_event_data(part)

    if event == "updates" and isinstance(data, dict):
        node_names = [str(name) for name in data.keys() if not str(name).startswith("__")]
        if node_names:
            joined = ", ".join(node_names[:3])
            return f"LangGraph-Schritt abgeschlossen: {joined}."

    if BRIDGE_ACTIVITY_DETAIL == "debug" and event and event not in {"messages", "metadata"}:
        return f"LangGraph-Event: {event}."

    return ""


def _extract_context_activity(part: Any) -> tuple[str, str, str]:
    if _stream_event_name(part) != "updates":
        return "", "", ""
    data = _stream_event_data(part)
    if not isinstance(data, dict):
        return "", "", ""

    for node, update in data.items():
        if str(node).startswith("__") or not isinstance(update, dict):
            continue
        profile = update.get("run_profile") if isinstance(update.get("run_profile"), dict) else {}
        notice = str(update.get("memory_notice") or "")
        node_name = str(node)
        lower_notice = notice.lower()

        ingest_text = _extract_ingest_activity(profile)
        if ingest_text:
            return "large_ingest", node_name, ingest_text
        compression_progress = _extract_compression_progress_activity(profile)
        if compression_progress:
            return "context_compaction", node_name, compression_progress

        if (
            node_name == "hard_context_stop"
            or profile.get("hard_context_stopped")
            or profile.get("hard_context_trim_used")
            or profile.get("hard_context_rescued")
            or "hard context cutoff" in lower_notice
            or "hart getrimmt" in lower_notice
        ):
            before = profile.get("hard_context_trim_tokens_before") or profile.get("token_estimate") or ""
            after = profile.get("hard_context_trim_tokens_after") or ""
            removed = profile.get("hard_context_trim_removed_messages")
            details = []
            if before:
                details.append(f"vorher~{before}")
            if after:
                details.append(f"nachher~{after}")
            if removed is not None:
                details.append(f"entfernt={removed}")
            suffix = f" ({', '.join(details)})" if details else ""
            label = "Hard-Trim aktiv" if profile.get("hard_context_trim_used") else "Hard-Cutoff aktiv"
            return "context_hard", node_name, f"{label}: {node_name}{suffix}"

        if (
            profile.get("pre_run_compression_used")
            or profile.get("post_run_compression_used")
            or profile.get("handoff_context_guard_used")
            or profile.get("context_compressed")
            or "komprimiert" in lower_notice
            or "compression" in lower_notice
        ):
            before = (
                profile.get("pre_run_compression_tokens")
                or profile.get("post_run_compression_tokens")
                or profile.get("handoff_context_tokens")
                or ""
            )
            after = (
                profile.get("pre_run_compression_tokens_after")
                or profile.get("post_run_compression_tokens_after")
                or profile.get("handoff_context_tokens_after")
                or ""
            )
            details = []
            if before:
                details.append(f"vorher~{before}")
            if after:
                details.append(f"nachher~{after}")
            suffix = f" ({', '.join(details)})" if details else ""
            return "context_compaction", node_name, f"Compaction aktiv: {node_name}{suffix}"

    return "", "", ""


def _extract_ingest_activity(profile: dict[str, Any]) -> str:
    for field, label in (("document_ingests", "Document ingest"), ("large_paste_ingests", "Large ingest")):
        records = profile.get(field)
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            events = record.get("events") if isinstance(record.get("events"), list) else []
            if not events:
                continue
            latest = events[-1] if isinstance(events[-1], dict) else {}
            event_name = str(latest.get("event") or "")
            source_key = str(record.get("source_key") or latest.get("source_key") or "")
            status = str(record.get("index_status") or latest.get("status") or "")
            chunk_number = latest.get("chunk_number")
            chunk_count = latest.get("chunk_count")
            if event_name.endswith(".chunk_indexed") and chunk_number and chunk_count:
                return f"{label}: Chunk {chunk_number}/{chunk_count} indexiert ({source_key})"
            if event_name.endswith(".deduped"):
                return f"{label}: dedupliziert ({source_key}, chunks={latest.get('chunk_count', 0)})"
            if event_name.endswith(".deduped"):
                return f"{label}: vorhandene identische Quelle wiederverwendet ({source_key})"
            if event_name.endswith(".completed"):
                return f"{label}: abgeschlossen ({source_key}, status={status})"
            if event_name.endswith(".started"):
                return f"{label}: gestartet ({source_key})"
            if event_name.endswith(".failed") or event_name.endswith(".blocked"):
                return f"{label}: fehlgeschlagen ({source_key}, status={status or event_name})"
    return ""


def _extract_compression_progress_activity(profile: dict[str, Any]) -> str:
    if not isinstance(profile, dict):
        return ""
    for prefix, label in (
        ("pre_run_compression", "Compression"),
        ("final_budget_rescue", "Final rescue"),
        ("large_paste_post_rag_compression", "Post-RAG compression"),
        ("post_run_compression", "Post-run compression"),
        ("handoff_context", "Handoff compression"),
    ):
        events = profile.get(f"{prefix}_events")
        if not isinstance(events, list) or not events:
            continue
        latest = events[-1] if isinstance(events[-1], dict) else {}
        event_name = str(latest.get("event") or "")
        chunk_number = latest.get("chunk_number")
        chunk_count = latest.get("chunk_count")
        if event_name == "compression.chunk.started" and chunk_number and chunk_count:
            return f"{label}: Chunk {chunk_number}/{chunk_count} gestartet"
        if event_name == "compression.chunk.completed" and chunk_number and chunk_count:
            return f"{label}: Chunk {chunk_number}/{chunk_count} abgeschlossen"
        if event_name == "compression.synthesis.started":
            return f"{label}: Synthese gestartet"
        if event_name == "compression.synthesis.completed":
            return f"{label}: Synthese abgeschlossen"
        if event_name == "compression.workflow_events.compacted":
            count = latest.get("workflow_event_count")
            return f"{label}: Tool-/Workflow-Events kompakt ({count or 0})"
        if event_name == "compression.precompact":
            pressure = latest.get("token_pressure")
            hmt = (
                latest.get("head_message_count"),
                latest.get("middle_message_count"),
                latest.get("tail_message_count"),
            )
            hmt_text = f", H/M/T={hmt[0]}/{hmt[1]}/{hmt[2]}" if all(value is not None for value in hmt) else ""
            pressure_text = f", Druck={pressure}" if pressure not in (None, "") else ""
            return f"{label}: PreCompact{pressure_text}{hmt_text}"
        if event_name == "compression.postcompact":
            archive_key = str(latest.get("archive_key") or "")
            archive_text = f", Archiv={archive_key}" if archive_key else ""
            return f"{label}: PostCompact abgeschlossen{archive_text}"
        if event_name == "compression.completed":
            summary = "Chunking" if latest.get("summary_chunking_used") else "One-shot"
            return f"{label}: abgeschlossen ({summary})"
        if event_name == "compression.skipped":
            return f"{label}: uebersprungen ({latest.get('reason', '')})"
        if event_name == "compression.synthesis.failed":
            return f"{label}: Summary fehlgeschlagen"
        if event_name == "compression.started":
            return f"{label}: gestartet"
    return ""


def _delta_text(text: str, emitted: str) -> str:
    if not text:
        return ""
    if emitted and text.startswith(emitted):
        return text[len(emitted) :]
    return text


async def _stream_chat_final(
    client: Any,
    thread_id: str,
    run_payload: dict[str, Any],
    model: str,
) -> AsyncIterator[str]:
    content = await _run_wait_content(client, thread_id, run_payload)
    for chunk in _openai_stream_response(content, model):
        yield chunk


async def _stream_chat_events(
    client: Any,
    thread_id: str,
    run_payload: dict[str, Any],
    model: str,
    *,
    include_activity: bool = True,
    observation_id: str = "",
) -> AsyncIterator[str]:
    yield _stream_data(_chunk("", model, role="assistant"))
    if include_activity and BRIDGE_SHOW_ACTIVITY_EVENTS and BRIDGE_ACTIVITY_DETAIL != "off":
        yield _activity_chunk("AlphaRavis startet den LangGraph-Run.", model)

    saw_token = False
    emitted = ""
    emitted_text_by_message: dict[str, str] = {}
    emitted_reasoning = ""
    emitted_activity: set[str] = set()
    emitted_context_activity: set[str] = set()
    explicit_reasoning_seen = False
    internal_reasoning_sections: set[str] = set()
    emitted_internal_updates: set[str] = set()
    message_nodes: dict[str, str] = {}
    thinking_splitter = _VisibleThinkingSplitter()
    used_state_fallback = False
    content_scrubber = StreamingInternalContextScrubber() if BRIDGE_SCRUB_INTERNAL_CONTEXT else None
    reasoning_scrubber = StreamingInternalContextScrubber() if BRIDGE_SCRUB_INTERNAL_CONTEXT else None

    stream_kwargs = {
        "stream_mode": ["messages", "updates"] if include_activity and BRIDGE_SHOW_ACTIVITY_EVENTS else "messages",
        "stream_subgraphs": BRIDGE_STREAM_SUBGRAPHS,
        "multitask_strategy": "interrupt",
    }
    if "command" in run_payload:
        stream_kwargs["command"] = run_payload["command"]
    else:
        stream_kwargs["input"] = run_payload["input"]

    try:
        async with asyncio.timeout(BRIDGE_RUN_TIMEOUT_SECONDS):
            async for part in client.runs.stream(thread_id, LANGGRAPH_ASSISTANT_ID, **stream_kwargs):
                message_nodes.update(_stream_message_metadata_nodes(part))
                activity = _extract_activity_text(part)
                if include_activity and activity and activity not in emitted_activity:
                    emitted_activity.add(activity)
                    yield _activity_chunk(activity, model)

                context_kind, context_label, context_text = _extract_context_activity(part)
                if observation_id and _stream_event_name(part) == "updates":
                    update_data = _stream_event_data(part)
                    if isinstance(update_data, dict):
                        for node_name, update in update_data.items():
                            if isinstance(update, dict) and isinstance(update.get("run_profile"), dict):
                                _observer_note_budget(observation_id, node_name=str(node_name), profile=update["run_profile"])
                if include_activity and context_text and context_text not in emitted_context_activity:
                    emitted_context_activity.add(context_text)
                    yield _stream_data(
                        _chunk(
                            "",
                            model,
                            reasoning_content=f"{context_text}\n",
                            alpha_reasoning_kind=context_kind,
                            alpha_reasoning_label=context_label,
                        )
                    )

                internal_update = _extract_internal_plan_update(part)
                if internal_update and internal_update not in emitted_internal_updates:
                    emitted_internal_updates.add(internal_update)
                    prefix = "Interner Plan (planner):\n"
                    visible_internal_update = (
                        reasoning_scrubber.feed(prefix + internal_update)
                        if reasoning_scrubber
                        else prefix + internal_update
                    )
                    if visible_internal_update:
                        yield _stream_data(
                            _chunk(
                                "",
                                model,
                                reasoning_content=visible_internal_update,
                                alpha_reasoning_kind="internal_plan",
                                alpha_reasoning_label="planner",
                            )
                        )

                reasoning = _extract_stream_reasoning(part)
                reasoning_delta = reasoning if _stream_part_is_delta(part) else _delta_text(reasoning, emitted_reasoning)
                if reasoning_delta:
                    explicit_reasoning_seen = True
                    emitted_reasoning += reasoning_delta
                    visible_reasoning_delta = (
                        reasoning_scrubber.feed(reasoning_delta) if reasoning_scrubber else reasoning_delta
                    )
                    if visible_reasoning_delta:
                        yield _stream_data(_chunk("", model, reasoning_content=visible_reasoning_delta))

                text = _extract_stream_text(part)
                message_delta_key = _stream_message_delta_key(part)
                previous_text_for_message = emitted_text_by_message.get(message_delta_key, "")
                delta = text if _stream_part_is_delta(part) else _delta_text(text, previous_text_for_message)
                if delta:
                    emitted += delta
                    emitted_text_by_message[message_delta_key] = (
                        previous_text_for_message + delta if _stream_part_is_delta(part) else text
                    )
                    if _stream_text_is_internal_reasoning(part, message_nodes=message_nodes):
                        node = _stream_langgraph_node(part, message_nodes=message_nodes) or "internal"
                        prefix = ""
                        if node not in internal_reasoning_sections:
                            internal_reasoning_sections.add(node)
                            prefix = f"Interner Plan ({node}):\n"
                        visible_internal_delta = (
                            reasoning_scrubber.feed(prefix + delta) if reasoning_scrubber else prefix + delta
                        )
                        if visible_internal_delta:
                            yield _stream_data(
                                _chunk(
                                    "",
                                    model,
                                    reasoning_content=visible_internal_delta,
                                    alpha_reasoning_kind="internal_plan",
                                    alpha_reasoning_label=node,
                                )
                            )
                        continue
                    answer_delta, thinking_delta = thinking_splitter.feed(
                        delta,
                        emit_reasoning=BRIDGE_STREAM_REASONING_EVENTS and not explicit_reasoning_seen,
                    )
                    if thinking_delta:
                        visible_thinking_delta = (
                            reasoning_scrubber.feed(thinking_delta) if reasoning_scrubber else thinking_delta
                        )
                        if visible_thinking_delta:
                            yield _stream_data(_chunk("", model, reasoning_content=visible_thinking_delta))
                    visible_delta = content_scrubber.feed(answer_delta) if content_scrubber else answer_delta
                    if visible_delta:
                        saw_token = True
                        yield _stream_data(_chunk(visible_delta, model))
    except TimeoutError as exc:
        _log_exception(
            "bridge.langgraph_stream.timeout",
            exc,
            level=logging.ERROR,
            dependency="langgraph-api",
            thread_id=thread_id,
            timeout_seconds=BRIDGE_RUN_TIMEOUT_SECONDS,
            emitted_chars=len(emitted),
            emitted_reasoning_chars=len(emitted_reasoning),
        )
        if include_activity and BRIDGE_SHOW_ERROR_CLASSIFICATION and BRIDGE_SHOW_ACTIVITY_EVENTS and BRIDGE_ACTIVITY_DETAIL != "off":
            yield _activity_chunk(_error_activity_text(exc), model)
        yield _stream_data(_chunk(_clean_error_message(exc), model))
        yield _stream_data(_chunk("", model, finish_reason="stop"))
        yield "data: [DONE]\n\n"
        return
    except Exception as exc:
        _log_exception(
            "bridge.langgraph_stream.failed",
            exc,
            level=logging.ERROR,
            dependency="langgraph-api",
            thread_id=thread_id,
            emitted_chars=len(emitted),
            emitted_reasoning_chars=len(emitted_reasoning),
        )
        if include_activity and BRIDGE_SHOW_ERROR_CLASSIFICATION and BRIDGE_SHOW_ACTIVITY_EVENTS and BRIDGE_ACTIVITY_DETAIL != "off":
            yield _activity_chunk(_error_activity_text(exc), model)
        yield _stream_data(_chunk(_clean_error_message(exc), model))
        yield _stream_data(_chunk("", model, finish_reason="stop"))
        yield "data: [DONE]\n\n"
        return

    if not saw_token:
        used_state_fallback = True
        state_values: Any = {}
        try:
            state = await client.threads.get_state(thread_id)
            state_values = state.get("values", state)
            content = _last_ai_content(state_values)
        except Exception as exc:
            content = _clean_error_message(exc)
        if not content:
            content = _state_failure_message(state_values)
        if content:
            visible, thinking = _split_visible_thinking_once(
                content,
                emit_reasoning=BRIDGE_STREAM_REASONING_EVENTS and not explicit_reasoning_seen,
            )
            visible_thinking = reasoning_scrubber.feed(thinking) if reasoning_scrubber else thinking
            if visible_thinking:
                yield _stream_data(_chunk("", model, reasoning_content=visible_thinking))
            visible = content_scrubber.feed(visible) if content_scrubber else visible
            if visible:
                yield _stream_data(_chunk(visible, model))

    answer_tail, thinking_tail = (
        ("", "")
        if used_state_fallback
        else thinking_splitter.flush(
            emit_reasoning=BRIDGE_STREAM_REASONING_EVENTS and not explicit_reasoning_seen,
        )
    )
    if thinking_tail:
        visible_thinking_tail = reasoning_scrubber.feed(thinking_tail) if reasoning_scrubber else thinking_tail
        if visible_thinking_tail:
            yield _stream_data(_chunk("", model, reasoning_content=visible_thinking_tail))
    if answer_tail:
        visible_answer_tail = content_scrubber.feed(answer_tail) if content_scrubber else answer_tail
        if visible_answer_tail:
            yield _stream_data(_chunk(visible_answer_tail, model))

    if reasoning_scrubber:
        reasoning_tail = reasoning_scrubber.flush()
        if reasoning_tail:
            yield _stream_data(_chunk("", model, reasoning_content=reasoning_tail))
    if content_scrubber:
        content_tail = content_scrubber.flush()
        if content_tail:
            yield _stream_data(_chunk(content_tail, model))

    yield _stream_data(_chunk("", model, finish_reason="stop"))
    yield "data: [DONE]\n\n"


def _observer_accumulate_chat_chunk(accumulator: dict[str, list[str]], chunk: str) -> None:
    for block in chunk.split("\n\n"):
        data_lines = [line.removeprefix("data: ").strip() for line in block.splitlines() if line.startswith("data: ")]
        if not data_lines:
            continue
        data_text = "\n".join(data_lines)
        if data_text == "[DONE]":
            continue
        try:
            payload = json.loads(data_text)
        except json.JSONDecodeError:
            continue
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices:
            continue
        delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
        if not isinstance(delta, dict):
            continue
        content = delta.get("content")
        if isinstance(content, str):
            accumulator.setdefault("output", []).append(content)
        reasoning = delta.get(BRIDGE_REASONING_DELTA_FIELD) or delta.get("reasoning_content") or delta.get("reasoning")
        if isinstance(reasoning, str):
            accumulator.setdefault("reasoning", []).append(reasoning)


async def _stream_chat(body: dict[str, Any], request: Request) -> AsyncIterator[str]:
    started = time.perf_counter()
    model = str(body.get("model") or OPENAI_MODEL_NAME)
    messages = body.get("messages", [])
    observation_id = _observer_start(protocol="chat", request=request, body=body, messages=messages)
    hard_error = _hard_input_error(messages)
    if hard_error:
        _observer_hard_cutoff(observation_id, hard_error)
        for chunk in _openai_stream_response(hard_error, model):
            yield chunk
        return

    client = _client()
    thread_key = _extract_thread_key(body, request)
    thread_id = await _ensure_thread(client, _thread_id_for_key(thread_key), thread_key)
    run_payload = await _prepare_run_payload(client, thread_id, thread_key, messages, model=model)
    _observer_prepared(observation_id, thread_key=thread_key, thread_id=thread_id, run_payload=run_payload)
    _log_event(
        logging.INFO,
        "bridge.chat.stream.started",
        thread_id=thread_id,
        thread_key=thread_key,
        model=model,
        message_count=len(body.get("messages", [])),
        input_tokens_estimate=_request_token_estimate(body.get("messages", [])),
        stream_mode=BRIDGE_STREAM_MODE,
    )

    if run_payload.get("direct_response"):
        content = str(run_payload["direct_response"])
        for chunk in _openai_stream_response(content, model):
            yield chunk
        _log_event(
            logging.INFO,
            "bridge.chat.stream.completed",
            thread_id=thread_id,
            model=model,
            direct_response=True,
            elapsed_seconds=round(time.perf_counter() - started, 3),
        )
        _observer_complete(
            observation_id,
            output_text=content,
            elapsed_seconds=time.perf_counter() - started,
        )
        return

    observed_chunks: dict[str, list[str]] = {"output": [], "reasoning": []}
    if BRIDGE_STREAM_MODE in {"final", "message", "messages"}:
        async for chunk in _stream_chat_final(client, thread_id, run_payload, model):
            _observer_accumulate_chat_chunk(observed_chunks, chunk)
            yield chunk
        _log_event(
            logging.INFO,
            "bridge.chat.stream.completed",
            thread_id=thread_id,
            model=model,
            stream_mode=BRIDGE_STREAM_MODE,
            elapsed_seconds=round(time.perf_counter() - started, 3),
        )
        _observer_complete(
            observation_id,
            output_text="".join(observed_chunks.get("output", [])),
            reasoning_text="".join(observed_chunks.get("reasoning", [])),
            elapsed_seconds=time.perf_counter() - started,
        )
        return

    async for chunk in _stream_chat_events(client, thread_id, run_payload, model, observation_id=observation_id):
        _observer_accumulate_chat_chunk(observed_chunks, chunk)
        yield chunk
    _log_event(
        logging.INFO,
        "bridge.chat.stream.completed",
        thread_id=thread_id,
        model=model,
        stream_mode=BRIDGE_STREAM_MODE,
        elapsed_seconds=round(time.perf_counter() - started, 3),
    )
    _observer_complete(
        observation_id,
        output_text="".join(observed_chunks.get("output", [])),
        reasoning_text="".join(observed_chunks.get("reasoning", [])),
        elapsed_seconds=time.perf_counter() - started,
    )


async def _run_wait_content(
    client: Any,
    thread_id: str,
    run_payload: dict[str, Any],
    *,
    trace: dict[str, Any] | None = None,
    request_started: float | None = None,
    observation_id: str | None = None,
) -> str:
    wait_kwargs = {"multitask_strategy": "interrupt"}
    if "command" in run_payload:
        wait_kwargs["command"] = run_payload["command"]
    else:
        wait_kwargs["input"] = run_payload["input"]

    trace_started = request_started if request_started is not None else time.perf_counter()
    wait_started = time.perf_counter()
    graph_offset_seconds = wait_started - trace_started
    _trace_step(trace, "bridge.langgraph.wait.started", trace_started, thread_id=thread_id)
    try:
        state = await asyncio.wait_for(
            client.runs.wait(thread_id, LANGGRAPH_ASSISTANT_ID, **wait_kwargs),
            timeout=BRIDGE_RUN_TIMEOUT_SECONDS,
        )
        wait_duration = time.perf_counter() - wait_started
        _merge_langgraph_trace(
            trace,
            state,
            request_started=trace_started,
            graph_offset_seconds=graph_offset_seconds,
        )
        values = _state_values(state)
        profile = values.get("run_profile") if isinstance(values, dict) and isinstance(values.get("run_profile"), dict) else {}
        if observation_id and profile:
            _observer_note_budget(observation_id, node_name="run_wait", profile=profile)
        _trace_step(
            trace,
            "bridge.langgraph.wait.completed",
            trace_started,
            duration_seconds=wait_duration,
            thread_id=thread_id,
        )
        return _last_ai_content(state)
    except TimeoutError as exc:
        _trace_step(
            trace,
            "bridge.langgraph.wait.timeout",
            trace_started,
            duration_seconds=time.perf_counter() - wait_started,
            thread_id=thread_id,
            timeout_seconds=BRIDGE_RUN_TIMEOUT_SECONDS,
        )
        _log_exception(
            "bridge.langgraph_run.timeout",
            exc,
            level=logging.ERROR,
            dependency="langgraph-api",
            thread_id=thread_id,
            timeout_seconds=BRIDGE_RUN_TIMEOUT_SECONDS,
        )
        return _clean_error_message(exc)
    except Exception as exc:
        _trace_step(
            trace,
            "bridge.langgraph.wait.failed",
            trace_started,
            duration_seconds=time.perf_counter() - wait_started,
            thread_id=thread_id,
            error_type=type(exc).__name__,
        )
        _log_exception(
            "bridge.langgraph_run.failed",
            exc,
            level=logging.ERROR,
            dependency="langgraph-api",
            thread_id=thread_id,
        )
        return _clean_error_message(exc)


def _responses_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        block_type = str(content.get("type") or "")
        if isinstance(content.get("text"), str):
            return content["text"]
        if isinstance(content.get("content"), str):
            return content["content"]
        if block_type in MEDIA_BLOCK_TYPES:
            return _media_block_summary(content)
        if block_type in {"function_call_output", "tool_call_output", "custom_tool_call_output"}:
            call_id = content.get("call_id") or content.get("tool_call_id") or "unknown"
            return f"[Tool output {call_id}]\n{content.get('output', '')}"
        if block_type:
            return f"[Responses input item {block_type}]"
        return str(content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = _responses_content_to_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    return str(content or "")


def _visible_content(content: str) -> str:
    if not BRIDGE_SCRUB_INTERNAL_CONTEXT:
        return content
    content = re.sub(
        r"(?im)^\s*\[(?:thinking|reasoning) content block omitted\]\s*\n?",
        "",
        content,
    )
    return sanitize_internal_context(content)


def _responses_input_to_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    instructions = _responses_content_to_text(body.get("instructions") or "").strip()
    raw_input = body.get("input")
    messages: list[dict[str, Any]] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})

    if isinstance(raw_input, str):
        messages.append({"role": "user", "content": raw_input})
        return messages

    if isinstance(raw_input, list):
        for item in raw_input:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
                continue
            if not isinstance(item, dict):
                continue

            item_type = str(item.get("type") or "")
            if item_type == "message" or item.get("role"):
                role = str(item.get("role") or "user")
                content = item.get("content", item.get("text", ""))
                messages.append({"role": role, "content": _sanitize_message_content(content)})
                continue

            if item_type in {"input_text", "text"}:
                messages.append({"role": "user", "content": _responses_content_to_text(item)})
                continue

            if item_type in MEDIA_BLOCK_TYPES:
                messages.append({"role": "user", "content": _sanitize_message_content([item])})
                continue

            if item_type in {
                "function_call_output",
                "tool_call_output",
                "custom_tool_call_output",
                "computer_call_output",
                "local_shell_call_output",
                "shell_call_output",
            }:
                messages.append({"role": "user", "content": _responses_content_to_text(item)})
                continue

    if len(messages) == (1 if instructions else 0) and isinstance(body.get("messages"), list):
        fallback = list(body["messages"])
        if instructions:
            return [messages[0], *fallback]
        return fallback
    return messages


def _response_message_item(content: str, *, item_id: str | None = None, status: str = "completed") -> dict[str, Any]:
    return {
        "id": item_id or f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [{"type": "output_text", "text": content, "annotations": [], "logprobs": []}],
    }


def _response_reasoning_item(
    text: str,
    *,
    item_id: str | None = None,
    status: str = "completed",
    summary: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": item_id or f"reason_{uuid.uuid4().hex}",
        "type": "reasoning",
        "status": status,
        "summary": summary or [],
    }
    if text:
        item["content"] = [{"type": "reasoning_text", "text": text}]
    else:
        item["content"] = []
    return item


def _messages_to_input_items(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = _responses_content_to_text(message.get("content", ""))
        content_type = "input_text" if role in {"user", "system", "developer"} else "output_text"
        items.append(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "role": role,
                "content": [{"type": content_type, "text": content}],
            }
        )
    return items


def _response_output_text(response: dict[str, Any]) -> str:
    text_parts: list[str] = []
    for item in response.get("output", []):
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                text = part.get("text")
                if isinstance(text, str) and text:
                    text_parts.append(text)
    return "\n".join(text_parts).strip()


def _responses_messages_for_body(body: dict[str, Any]) -> list[dict[str, Any]]:
    messages = _responses_input_to_messages(body)
    previous_response_id = body.get("previous_response_id")
    if not previous_response_id:
        return messages

    previous = _RESPONSES_STORE.get(str(previous_response_id))
    if not previous:
        return messages
    previous_text = _response_output_text(previous)
    if not previous_text:
        return messages

    insert_at = 0
    while insert_at < len(messages) and messages[insert_at].get("role") in {"system", "developer"}:
        insert_at += 1
    messages.insert(
        insert_at,
        {
            "role": "assistant",
            "content": f"[Previous response {previous_response_id}]\n{previous_text}",
        },
    )
    return messages


def _response_usage(messages: list[dict[str, Any]], content: str) -> dict[str, Any]:
    input_tokens = _request_token_estimate(messages)
    output_tokens = _approx_tokens(content)
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens + output_tokens,
    }


def _response_store_value(body: dict[str, Any]) -> bool:
    if "store" in body:
        return bool(body.get("store"))
    return BRIDGE_RESPONSES_STORE


def _response_base_fields(body: dict[str, Any], model: str) -> dict[str, Any]:
    reasoning = body.get("reasoning") if isinstance(body.get("reasoning"), dict) else {}
    text = body.get("text") if isinstance(body.get("text"), dict) else {"format": {"type": "text"}}
    return {
        "background": bool(body.get("background", False)),
        "conversation": None,
        "instructions": body.get("instructions"),
        "max_output_tokens": body.get("max_output_tokens", body.get("max_tokens")),
        "max_tool_calls": body.get("max_tool_calls"),
        "parallel_tool_calls": body.get("parallel_tool_calls", True),
        "previous_response_id": body.get("previous_response_id"),
        "prompt": body.get("prompt"),
        "prompt_cache_key": body.get("prompt_cache_key"),
        "prompt_cache_retention": body.get("prompt_cache_retention"),
        "reasoning": {
            "effort": reasoning.get("effort"),
            "summary": reasoning.get("summary"),
        },
        "safety_identifier": body.get("safety_identifier"),
        "service_tier": body.get("service_tier", "auto"),
        "store": _response_store_value(body),
        "temperature": body.get("temperature", 1.0),
        "text": text,
        "tool_choice": body.get("tool_choice", "auto"),
        "tools": body.get("tools", []),
        "top_p": body.get("top_p", 1.0),
        "truncation": body.get("truncation", "disabled"),
        "user": body.get("user"),
        "metadata": body.get("metadata") if isinstance(body.get("metadata"), dict) else {},
        "model": model,
    }


def _response_object(
    content: str,
    model: str,
    response_id: str | None = None,
    *,
    item_id: str | None = None,
    body: dict[str, Any] | None = None,
    messages: list[dict[str, Any]] | None = None,
    status: str = "completed",
    error: dict[str, Any] | None = None,
    incomplete_details: dict[str, Any] | None = None,
    output_items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    content = _visible_content(content)
    response_id = response_id or f"resp_{uuid.uuid4().hex}"
    body = body or {}
    messages = messages or []
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "completed_at": int(time.time()) if status == "completed" else None,
        "status": status,
        "error": error,
        "incomplete_details": incomplete_details,
        "output": output_items
        if output_items is not None
        else ([_response_message_item(content, item_id=item_id, status=status)] if status != "failed" else []),
        "usage": _response_usage(messages, content) if status == "completed" else None,
        **_response_base_fields(body, model),
    }


def _store_response_object(response: dict[str, Any], body: dict[str, Any]) -> None:
    if not _response_store_value(body):
        return
    _RESPONSES_STORE[str(response["id"])] = response
    _RESPONSES_STORE.move_to_end(str(response["id"]))
    input_items = _messages_to_input_items(_responses_input_to_messages(body))
    _RESPONSES_INPUT_ITEMS[str(response["id"])] = input_items
    _RESPONSES_INPUT_ITEMS.move_to_end(str(response["id"]))
    while len(_RESPONSES_STORE) > max(1, BRIDGE_RESPONSES_STORE_MAX):
        removed_id, _ = _RESPONSES_STORE.popitem(last=False)
        _RESPONSES_INPUT_ITEMS.pop(removed_id, None)


def _responses_error(message: str, *, status_code: int = 400, code: str = "unsupported_feature") -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": None,
                "code": code,
            }
        },
        status_code=status_code,
    )


def _validate_responses_request(body: dict[str, Any]) -> JSONResponse | None:
    previous_response_id = body.get("previous_response_id")
    if previous_response_id and str(previous_response_id) not in _RESPONSES_STORE:
        return _responses_error(
            "previous_response_id was provided, but that response is not available "
            "in this bridge process. Enable BRIDGE_RESPONSES_STORE and reference a "
            "stored response from the same bridge instance.",
            status_code=404,
            code="previous_response_not_found",
        )
    if body.get("conversation") is not None and previous_response_id:
        return _responses_error(
            "conversation and previous_response_id cannot be used together.",
            code="conversation_and_previous_response_id_conflict",
        )
    if body.get("background") is True:
        return _responses_error(
            "AlphaRavis Bridge does not support Responses background mode yet. "
            "Use stream=true or a normal foreground response.",
            code="background_not_supported",
        )
    if body.get("conversation") is not None:
        return _responses_error(
            "AlphaRavis Bridge uses LangGraph thread IDs instead of OpenAI Conversations. "
            "Pass conversationId, conversation_id, x-conversation-id, or x-thread-id.",
            code="conversation_not_supported",
        )
    if body.get("prompt") is not None:
        return _responses_error(
            "Prompt-template references are not supported by the AlphaRavis Bridge. "
            "Send concrete instructions/input instead.",
            code="prompt_template_not_supported",
        )
    text_format = (body.get("text") or {}).get("format") if isinstance(body.get("text"), dict) else None
    text_format_type = text_format.get("type") if isinstance(text_format, dict) else None
    if text_format_type and text_format_type != "text":
        return _responses_error(
            "AlphaRavis Bridge currently supports text output only. Structured "
            "Responses output formats should be handled inside AlphaRavis tools or "
            "through a dedicated endpoint.",
            code="text_format_not_supported",
        )
    modalities = body.get("modalities")
    if isinstance(modalities, list) and any(str(modality) != "text" for modality in modalities):
        return _responses_error(
            "AlphaRavis Bridge currently supports text output modality only.",
            code="output_modality_not_supported",
        )
    if body.get("tools") and not BRIDGE_RESPONSES_ALLOW_CLIENT_TOOLS:
        return _responses_error(
            "Client-supplied OpenAI Responses tools are not executed by AlphaRavis Bridge. "
            "AlphaRavis uses its internal LangGraph tools instead. Set "
            "BRIDGE_RESPONSES_ALLOW_CLIENT_TOOLS=true only if you intentionally want "
            "to accept tool metadata without executing those client tools.",
            code="client_tools_not_supported",
        )
    return None


def _responses_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _done_sentinel() -> str:
    return "data: [DONE]\n\n" if BRIDGE_RESPONSES_DONE_SENTINEL else ""


def _response_created_payload(response_id: str, model: str, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "completed_at": None,
        "status": "in_progress",
        "error": None,
        "incomplete_details": None,
        "output": [],
        "usage": None,
        **_response_base_fields(body, model),
    }


def _output_text_part(text: str) -> dict[str, Any]:
    return {"type": "output_text", "text": text, "annotations": [], "logprobs": []}


def _json_tool_arguments(args: Any) -> str:
    try:
        return json.dumps(args if args is not None else {}, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(args or "")


def _truncate_tool_output(text: str) -> str:
    max_chars = max(0, BRIDGE_RESPONSES_TOOL_OUTPUT_MAX_CHARS)
    if not max_chars or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[tool output truncated to {max_chars} chars]"


def _stringify_tool_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if _is_tool_message(value) or _is_ai_message(value):
        return _message_content(value)
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    return str(value)


def _stream_messages_from_part(part: Any) -> list[Any]:
    data = _stream_event_data(part)
    if isinstance(data, tuple) and data:
        return [data[0]]
    if isinstance(data, list):
        return list(data)
    if isinstance(data, dict):
        messages = data.get("messages")
        if isinstance(messages, list):
            return messages
        if "chunk" in data:
            return [data["chunk"]]
        if "message" in data:
            return [data["message"]]
    return []


def _tool_events_from_part(
    part: Any,
    seen_tool_calls: set[str],
    seen_tool_updates: set[str],
    tool_inputs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not BRIDGE_RESPONSES_STREAM_TOOL_EVENTS:
        return []

    notifications: list[dict[str, Any]] = []
    event = _stream_event_name(part)
    data = _stream_event_data(part)

    if event in {"on_tool_start", "tool_start"} and isinstance(data, dict):
        raw_input = data.get("input") if isinstance(data.get("input"), dict) else {}
        tool_name = str(data.get("name") or data.get("tool") or data.get("run_name") or "tool")
        call_id = str(data.get("run_id") or data.get("tool_call_id") or f"call_{uuid.uuid4().hex[:12]}")
        if call_id not in seen_tool_calls:
            seen_tool_calls.add(call_id)
            tool_inputs[call_id] = {"tool": tool_name, "args": raw_input}
            notifications.append({"type": "call", "call_id": call_id, "name": tool_name, "args": raw_input})
        return notifications

    if event in {"on_tool_end", "tool_end", "on_tool_error", "tool_error"} and isinstance(data, dict):
        call_id = str(data.get("run_id") or data.get("tool_call_id") or f"call_{uuid.uuid4().hex[:12]}")
        raw = tool_inputs.get(call_id, {})
        tool_name = str(raw.get("tool") or data.get("name") or data.get("tool") or "tool")
        if call_id not in seen_tool_calls:
            seen_tool_calls.add(call_id)
            args = raw.get("args", {}) if isinstance(raw, dict) else {}
            notifications.append({"type": "call", "call_id": call_id, "name": tool_name, "args": args})
        if call_id not in seen_tool_updates:
            seen_tool_updates.add(call_id)
            output = data.get("output") if "output" in data else data.get("error", data.get("content", ""))
            status = "failed" if "error" in event else "completed"
            notifications.append(
                {
                    "type": "result",
                    "call_id": call_id,
                    "name": tool_name,
                    "output": _stringify_tool_output(output),
                    "status": status,
                }
            )
        return notifications

    for message in _stream_messages_from_part(part):
        if _is_ai_message(message):
            for call in _extract_tool_calls_from_message(message):
                call_id = call["id"]
                if call_id in seen_tool_calls:
                    continue
                seen_tool_calls.add(call_id)
                tool_inputs[call_id] = {"tool": call["name"], "args": call.get("args", {})}
                notifications.append(
                    {
                        "type": "call",
                        "call_id": call_id,
                        "name": call["name"],
                        "args": call.get("args", {}),
                    }
                )
        tool_result = _extract_tool_result(message)
        if not tool_result:
            continue
        call_id, name, output = tool_result
        if call_id not in seen_tool_calls:
            seen_tool_calls.add(call_id)
            raw = tool_inputs.get(call_id, {"args": {}})
            notifications.append(
                {
                    "type": "call",
                    "call_id": call_id,
                    "name": str(raw.get("tool") or name),
                    "args": raw.get("args", {}),
                }
            )
        if call_id not in seen_tool_updates:
            seen_tool_updates.add(call_id)
            raw = tool_inputs.get(call_id, {})
            notifications.append(
                {
                    "type": "result",
                    "call_id": call_id,
                    "name": str(raw.get("tool") or name),
                    "output": output,
                    "status": "completed",
                }
            )
    return notifications


class _ResponsesStreamBuilder:
    def __init__(self, *, response_id: str, model: str, body: dict[str, Any], messages: list[dict[str, Any]]) -> None:
        self.response_id = response_id
        self.model = model
        self.body = body
        self.messages = messages
        self.sequence_number = 0
        self.output_items: list[dict[str, Any]] = []
        self.full_content = ""
        self.full_reasoning = ""
        self.message_item: dict[str, Any] | None = None
        self.message_content_index = 0
        self.reasoning_item: dict[str, Any] | None = None
        self.reasoning_content_index = 0

    def event(self, event_type: str, **payload: Any) -> str:
        data = {"type": event_type, "sequence_number": self.sequence_number, **payload}
        self.sequence_number += 1
        return _responses_event(event_type, data)

    def start(self) -> list[str]:
        created = _response_created_payload(self.response_id, self.model, self.body)
        return [
            self.event("response.created", response=created),
            self.event("response.in_progress", response=created),
        ]

    def start_message(self, item_id: str) -> list[str]:
        item = {
            "id": item_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        self.message_item = item
        self.output_items.append(item)
        output_index = self.output_items.index(item)
        return [
            self.event("response.output_item.added", output_index=output_index, item=item),
            self.event(
                "response.content_part.added",
                item_id=item_id,
                output_index=output_index,
                content_index=self.message_content_index,
                part=_output_text_part(""),
            ),
        ]

    def text_delta(self, delta: str) -> str:
        self.full_content += delta
        output_index = self.output_items.index(self.message_item) if self.message_item in self.output_items else 0
        return self.event(
            "response.output_text.delta",
            item_id=self.message_item["id"] if self.message_item else "",
            output_index=output_index,
            content_index=self.message_content_index,
            delta=delta,
            logprobs=[],
        )

    def finish_message(self) -> list[str]:
        if self.message_item is None:
            return []
        output_index = self.output_items.index(self.message_item)
        part = _output_text_part(self.full_content)
        self.message_item["status"] = "completed"
        self.message_item["content"] = [part]
        return [
            self.event(
                "response.output_text.done",
                item_id=self.message_item["id"],
                output_index=output_index,
                content_index=self.message_content_index,
                text=self.full_content,
                logprobs=[],
            ),
            self.event(
                "response.content_part.done",
                item_id=self.message_item["id"],
                output_index=output_index,
                content_index=self.message_content_index,
                part=part,
            ),
            self.event("response.output_item.done", output_index=output_index, item=self.message_item),
        ]

    def _ensure_reasoning(self) -> list[str]:
        if self.reasoning_item is not None:
            return []
        self.reasoning_item = _response_reasoning_item("", status="in_progress")
        self.output_items.append(self.reasoning_item)
        output_index = self.output_items.index(self.reasoning_item)
        return [
            self.event("response.output_item.added", output_index=output_index, item=self.reasoning_item),
            self.event(
                "response.content_part.added",
                item_id=self.reasoning_item["id"],
                output_index=output_index,
                content_index=self.reasoning_content_index,
                part={"type": "reasoning_text", "text": ""},
            ),
        ]

    def reasoning_delta(self, delta: str, *, alpha_kind: str = "model", alpha_label: str = "") -> list[str]:
        events = self._ensure_reasoning()
        if self.reasoning_item is None:
            return events
        self.full_reasoning += delta
        output_index = self.output_items.index(self.reasoning_item)
        metadata = {"alpha_reasoning_kind": alpha_kind}
        if alpha_label:
            metadata["alpha_reasoning_label"] = alpha_label
        chunks = (
            [delta]
            if alpha_kind == "status"
            else _split_response_delta(delta, BRIDGE_RESPONSES_REASONING_DELTA_MAX_CHARS)
        )
        for chunk in chunks:
            events.append(
                self.event(
                    "response.reasoning.delta",
                    item_id=self.reasoning_item["id"],
                    output_index=output_index,
                    content_index=self.reasoning_content_index,
                    delta=chunk,
                    **metadata,
                )
            )
        return events

    def finish_reasoning(self) -> list[str]:
        if self.reasoning_item is None:
            return []
        output_index = self.output_items.index(self.reasoning_item)
        part = {"type": "reasoning_text", "text": self.full_reasoning}
        self.reasoning_item["status"] = "completed"
        self.reasoning_item["content"] = [part]
        return [
            self.event(
                "response.reasoning.done",
                item_id=self.reasoning_item["id"],
                output_index=output_index,
                content_index=self.reasoning_content_index,
                text=self.full_reasoning,
            ),
            self.event(
                "response.content_part.done",
                item_id=self.reasoning_item["id"],
                output_index=output_index,
                content_index=self.reasoning_content_index,
                part=part,
            ),
            self.event("response.output_item.done", output_index=output_index, item=self.reasoning_item),
        ]

    def tool_call(self, call_id: str, name: str, args: Any) -> list[str]:
        arguments = _json_tool_arguments(args)
        item = {
            "id": f"fc_{uuid.uuid4().hex}",
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": "",
            "status": "in_progress",
        }
        self.output_items.append(item)
        output_index = self.output_items.index(item)
        events = [self.event("response.output_item.added", output_index=output_index, item=item)]
        if arguments:
            item["arguments"] = arguments
            events.append(
                self.event(
                    "response.function_call_arguments.delta",
                    item_id=item["id"],
                    output_index=output_index,
                    call_id=call_id,
                    delta=arguments,
                )
            )
        events.append(
            self.event(
                "response.function_call_arguments.done",
                item_id=item["id"],
                output_index=output_index,
                call_id=call_id,
                arguments=arguments,
            )
        )
        item["status"] = "completed"
        events.append(self.event("response.output_item.done", output_index=output_index, item=item))
        return events

    def tool_result(self, call_id: str, output: str, status: str = "completed") -> list[str]:
        item = {
            "id": f"fco_{uuid.uuid4().hex}",
            "type": "function_call_output",
            "call_id": call_id,
            "output": _truncate_tool_output(_visible_content(output)),
            "status": status,
        }
        self.output_items.append(item)
        output_index = self.output_items.index(item)
        return [
            self.event("response.output_item.added", output_index=output_index, item=item),
            self.event("response.output_item.done", output_index=output_index, item=item),
        ]

    def response_object(self, *, status: str = "completed") -> dict[str, Any]:
        return _response_object(
            self.full_content,
            self.model,
            self.response_id,
            body=self.body,
            messages=self.messages,
            status=status,
            output_items=self.output_items,
        )


def _split_response_delta(delta: str, max_chars: int) -> list[str]:
    max_chars = max(0, max_chars)
    if not delta or max_chars <= 0 or len(delta) <= max_chars:
        return [delta] if delta else []

    chunks: list[str] = []
    rest = delta
    while len(rest) > max_chars:
        if max_chars <= 1:
            cut = 1
        else:
            space_cut = rest.rfind(" ", 0, max_chars + 1)
            newline_cut = rest.rfind("\n", 0, max_chars + 1)
            cut = max(space_cut, newline_cut)
            if cut < max_chars // 2:
                cut = max_chars
            else:
                cut += 1
        chunks.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        chunks.append(rest)
    return chunks


def _split_response_output_delta(delta: str) -> list[str]:
    return _split_response_delta(delta, BRIDGE_RESPONSES_OUTPUT_DELTA_MAX_CHARS)


async def _response_output_delta_events(
    builder: _ResponsesStreamBuilder,
    delta: str,
) -> AsyncIterator[str]:
    chunks = _split_response_output_delta(delta)
    split = len(chunks) > 1
    for chunk in chunks:
        yield builder.text_delta(chunk)
        if split:
            await asyncio.sleep(0)


async def _stream_responses(body: dict[str, Any], request: Request) -> AsyncIterator[str]:
    started = time.perf_counter()
    model = str(body.get("model") or OPENAI_MODEL_NAME)
    messages = _responses_messages_for_body(body)
    response_id = f"resp_{uuid.uuid4().hex}"
    item_id = f"msg_{uuid.uuid4().hex}"
    builder = _ResponsesStreamBuilder(response_id=response_id, model=model, body=body, messages=messages)
    hard_error = _hard_input_error(messages)
    observation_id = _observer_start(protocol="responses", request=request, body=body, messages=messages)

    for event in builder.start():
        yield event
    for event in builder.start_message(item_id):
        yield event

    if hard_error:
        _observer_hard_cutoff(observation_id, hard_error)
        async for event in _response_output_delta_events(builder, hard_error):
            yield event
        for event in builder.finish_message():
            yield event
        response = builder.response_object()
        _store_response_object(response, body)
        yield builder.event("response.completed", response=response)
        if done := _done_sentinel():
            yield done
        return

    chat_body = dict(body)
    chat_body["messages"] = messages
    client = _client()
    thread_key = _extract_thread_key(chat_body, request)
    thread_id = await _ensure_thread(client, _thread_id_for_key(thread_key), thread_key)
    run_payload = await _prepare_run_payload(client, thread_id, thread_key, chat_body.get("messages", []), model=model)
    _observer_prepared(observation_id, thread_key=thread_key, thread_id=thread_id, run_payload=run_payload)
    _log_event(
        logging.INFO,
        "bridge.responses.stream.started",
        response_id=response_id,
        thread_id=thread_id,
        thread_key=thread_key,
        model=model,
        message_count=len(messages),
        input_tokens_estimate=_request_token_estimate(messages),
    )

    if run_payload.get("direct_response"):
        content = _visible_content(str(run_payload["direct_response"]))
        if content:
            async for event in _response_output_delta_events(builder, content):
                yield event
        for event in builder.finish_message():
            yield event
        response = builder.response_object()
        _store_response_object(response, body)
        yield builder.event("response.completed", response=response)
        if done := _done_sentinel():
            yield done
        _log_event(
            logging.INFO,
            "bridge.responses.stream.completed",
            response_id=response_id,
            thread_id=thread_id,
            model=model,
            direct_response=True,
            output_chars=len(content),
            elapsed_seconds=round(time.perf_counter() - started, 3),
        )
        _observer_complete(
            observation_id,
            output_text=content,
            elapsed_seconds=time.perf_counter() - started,
        )
        return

    saw_token = False
    emitted = ""
    emitted_text_by_message: dict[str, str] = {}
    emitted_reasoning = ""
    emitted_activity: set[str] = set()
    emitted_context_activity: set[str] = set()
    seen_tool_calls: set[str] = set()
    seen_tool_updates: set[str] = set()
    tool_inputs: dict[str, dict[str, Any]] = {}
    explicit_reasoning_seen = False
    internal_reasoning_sections: set[str] = set()
    emitted_internal_updates: set[str] = set()
    message_nodes: dict[str, str] = {}
    thinking_splitter = _VisibleThinkingSplitter()
    used_state_fallback = False
    content_scrubber = StreamingInternalContextScrubber() if BRIDGE_SCRUB_INTERNAL_CONTEXT else None
    reasoning_scrubber = StreamingInternalContextScrubber() if BRIDGE_SCRUB_INTERNAL_CONTEXT else None

    stream_kwargs: dict[str, Any] = {
        "stream_mode": ["messages", "updates"],
        "stream_subgraphs": BRIDGE_STREAM_SUBGRAPHS,
        "multitask_strategy": "interrupt",
    }
    if "command" in run_payload:
        stream_kwargs["command"] = run_payload["command"]
    else:
        stream_kwargs["input"] = run_payload["input"]

    try:
        async with asyncio.timeout(BRIDGE_RUN_TIMEOUT_SECONDS):
            async for part in client.runs.stream(thread_id, LANGGRAPH_ASSISTANT_ID, **stream_kwargs):
                message_nodes.update(_stream_message_metadata_nodes(part))
                activity = _extract_activity_text(part, force=BRIDGE_RESPONSES_STREAM_ACTIVITY_EVENTS)
                if BRIDGE_RESPONSES_STREAM_ACTIVITY_EVENTS and activity and activity not in emitted_activity:
                    emitted_activity.add(activity)
                    for event in builder.reasoning_delta(f"Status: {activity}\n", alpha_kind="status"):
                        yield event

                context_kind, context_label, context_text = _extract_context_activity(part)
                if BRIDGE_RESPONSES_STREAM_ACTIVITY_EVENTS and context_text and context_text not in emitted_context_activity:
                    emitted_context_activity.add(context_text)
                    for event in builder.reasoning_delta(
                        f"{context_text}\n",
                        alpha_kind=context_kind,
                        alpha_label=context_label,
                    ):
                        yield event

                internal_update = _extract_internal_plan_update(part)
                if internal_update and internal_update not in emitted_internal_updates:
                    emitted_internal_updates.add(internal_update)
                    prefix = "Interner Plan (planner):\n"
                    visible_internal_update = (
                        reasoning_scrubber.feed(prefix + internal_update)
                        if reasoning_scrubber
                        else prefix + internal_update
                    )
                    if visible_internal_update:
                        for event in builder.reasoning_delta(
                            visible_internal_update,
                            alpha_kind="internal_plan",
                            alpha_label="planner",
                        ):
                            yield event

                for notification in _tool_events_from_part(
                    part,
                    seen_tool_calls,
                    seen_tool_updates,
                    tool_inputs,
                ):
                    if notification["type"] == "call":
                        for event in builder.tool_call(
                            str(notification["call_id"]),
                            str(notification["name"]),
                            notification.get("args", {}),
                        ):
                            yield event
                    elif notification["type"] == "result":
                        for event in builder.tool_result(
                            str(notification["call_id"]),
                            str(notification.get("output", "")),
                            status=str(notification.get("status") or "completed"),
                        ):
                            yield event

                reasoning = _extract_stream_reasoning(part, force=BRIDGE_RESPONSES_STREAM_REASONING_EVENTS)
                reasoning_delta = reasoning if _stream_part_is_delta(part) else _delta_text(reasoning, emitted_reasoning)
                if reasoning_delta:
                    explicit_reasoning_seen = True
                    emitted_reasoning += reasoning_delta
                    visible_reasoning_delta = (
                        reasoning_scrubber.feed(reasoning_delta) if reasoning_scrubber else reasoning_delta
                    )
                    if visible_reasoning_delta:
                        for event in builder.reasoning_delta(visible_reasoning_delta):
                            yield event

                text = _extract_stream_text(part)
                message_delta_key = _stream_message_delta_key(part)
                previous_text_for_message = emitted_text_by_message.get(message_delta_key, "")
                delta = text if _stream_part_is_delta(part) else _delta_text(text, previous_text_for_message)
                if delta:
                    emitted += delta
                    emitted_text_by_message[message_delta_key] = (
                        previous_text_for_message + delta if _stream_part_is_delta(part) else text
                    )
                    if _stream_text_is_internal_reasoning(part, message_nodes=message_nodes):
                        node = _stream_langgraph_node(part, message_nodes=message_nodes) or "internal"
                        prefix = ""
                        if node not in internal_reasoning_sections:
                            internal_reasoning_sections.add(node)
                            prefix = f"Interner Plan ({node}):\n"
                        visible_internal_delta = (
                            reasoning_scrubber.feed(prefix + delta) if reasoning_scrubber else prefix + delta
                        )
                        if visible_internal_delta:
                            for event in builder.reasoning_delta(
                                visible_internal_delta,
                                alpha_kind="internal_plan",
                                alpha_label=node,
                            ):
                                yield event
                        continue
                    answer_delta, thinking_delta = thinking_splitter.feed(
                        delta,
                        emit_reasoning=BRIDGE_RESPONSES_STREAM_REASONING_EVENTS and not explicit_reasoning_seen,
                    )
                    if thinking_delta:
                        visible_thinking_delta = (
                            reasoning_scrubber.feed(thinking_delta) if reasoning_scrubber else thinking_delta
                        )
                        if visible_thinking_delta:
                            for event in builder.reasoning_delta(visible_thinking_delta):
                                yield event
                    visible_delta = content_scrubber.feed(answer_delta) if content_scrubber else answer_delta
                    if visible_delta:
                        saw_token = True
                        async for event in _response_output_delta_events(builder, visible_delta):
                            yield event
    except TimeoutError as exc:
        error_message = _clean_error_message(exc)
        _log_exception(
            "bridge.responses_stream.timeout",
            exc,
            level=logging.ERROR,
            dependency="langgraph-api",
            response_id=response_id,
            thread_id=thread_id,
            timeout_seconds=BRIDGE_RUN_TIMEOUT_SECONDS,
            emitted_chars=len(emitted),
            emitted_reasoning_chars=len(emitted_reasoning),
        )
        async for event in _response_output_delta_events(builder, error_message):
            yield event
        saw_token = True
    except Exception as exc:
        error_message = _clean_error_message(exc)
        _log_exception(
            "bridge.responses_stream.failed",
            exc,
            level=logging.ERROR,
            dependency="langgraph-api",
            response_id=response_id,
            thread_id=thread_id,
            emitted_chars=len(emitted),
            emitted_reasoning_chars=len(emitted_reasoning),
        )
        async for event in _response_output_delta_events(builder, error_message):
            yield event
        saw_token = True

    if not saw_token:
        used_state_fallback = True
        state_values: Any = {}
        try:
            state = await client.threads.get_state(thread_id)
            state_values = state.get("values", state)
            content = _last_ai_content(state_values)
        except Exception as exc:
            content = _clean_error_message(exc)
        if not content:
            content = _state_failure_message(state_values)
        if content:
            visible, thinking = _split_visible_thinking_once(
                content,
                emit_reasoning=BRIDGE_RESPONSES_STREAM_REASONING_EVENTS and not explicit_reasoning_seen,
            )
            visible_thinking = reasoning_scrubber.feed(thinking) if reasoning_scrubber else thinking
            if visible_thinking:
                for event in builder.reasoning_delta(visible_thinking):
                    yield event
            visible = content_scrubber.feed(visible) if content_scrubber else visible
            if visible:
                async for event in _response_output_delta_events(builder, visible):
                    yield event

    answer_tail, thinking_tail = (
        ("", "")
        if used_state_fallback
        else thinking_splitter.flush(
            emit_reasoning=BRIDGE_RESPONSES_STREAM_REASONING_EVENTS and not explicit_reasoning_seen,
        )
    )
    if thinking_tail:
        visible_thinking_tail = reasoning_scrubber.feed(thinking_tail) if reasoning_scrubber else thinking_tail
        if visible_thinking_tail:
            for event in builder.reasoning_delta(visible_thinking_tail):
                yield event
    if answer_tail:
        visible_answer_tail = content_scrubber.feed(answer_tail) if content_scrubber else answer_tail
        if visible_answer_tail:
            async for event in _response_output_delta_events(builder, visible_answer_tail):
                yield event

    if reasoning_scrubber:
        reasoning_tail = reasoning_scrubber.flush()
        if reasoning_tail:
            for event in builder.reasoning_delta(reasoning_tail):
                yield event
    if content_scrubber:
        content_tail = content_scrubber.flush()
        if content_tail:
            async for event in _response_output_delta_events(builder, content_tail):
                yield event

    for event in builder.finish_reasoning():
        yield event
    for event in builder.finish_message():
        yield event
    response = builder.response_object()
    _store_response_object(response, body)
    yield builder.event("response.completed", response=response)
    if done := _done_sentinel():
        yield done
    _observer_complete(
        observation_id,
        output_text=builder.full_content,
        reasoning_text=builder.full_reasoning,
        elapsed_seconds=time.perf_counter() - started,
    )
    _log_event(
        logging.INFO,
        "bridge.responses.stream.completed",
        response_id=response_id,
        thread_id=thread_id,
        model=model,
        direct_response=False,
        output_chars=len(builder.full_content),
        reasoning_chars=len(builder.full_reasoning),
        elapsed_seconds=round(time.perf_counter() - started, 3),
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "langgraph_api_url": LANGGRAPH_API_URL,
        "preferred_api_mode": BRIDGE_PREFERRED_API_MODE,
        "openapi_version": app.openapi_version,
        "responses_api_enabled": BRIDGE_ENABLE_RESPONSES_API,
        "responses_store_enabled": BRIDGE_RESPONSES_STORE,
        "responses_store_size": len(_RESPONSES_STORE),
    }


@app.get("/health/llm-generation")
async def llm_generation_health():
    primary = await _smoke_test_litellm_model(BRIDGE_LLM_HEALTH_MODEL)
    fallback = None
    if BRIDGE_LLM_HEALTH_FALLBACK_MODEL and BRIDGE_LLM_HEALTH_FALLBACK_MODEL != BRIDGE_LLM_HEALTH_MODEL:
        fallback = await _smoke_test_litellm_model(BRIDGE_LLM_HEALTH_FALLBACK_MODEL)

    status = "ok"
    http_status = 200
    if not primary["ok"]:
        if fallback and fallback["ok"]:
            status = "degraded"
        else:
            status = "error"
            http_status = 503

    return JSONResponse(
        {
            "status": status,
            "litellm_url": BRIDGE_LLM_HEALTH_URL,
            "primary": primary,
            "fallback": fallback,
            "note": (
                "This checks real token generation, not only process health. "
                "Power actions remain manual/debugger-approved."
            ),
        },
        status_code=http_status,
    )


@app.get("/_alpharavis/bridge-observer")
async def bridge_observer(limit: int = 80):
    safe_limit = max(1, min(int(limit), BRIDGE_OBSERVER_MAX_RECORDS))
    return JSONResponse(
        {
            "records": list(_BRIDGE_OBSERVATIONS)[:safe_limit],
            "count": min(len(_BRIDGE_OBSERVATIONS), safe_limit),
            "max_records": BRIDGE_OBSERVER_MAX_RECORDS,
        }
    )


@app.delete("/_alpharavis/bridge-observer")
async def clear_bridge_observer():
    _BRIDGE_OBSERVATIONS.clear()
    return JSONResponse({"ok": True, "records": []})


@app.post("/tools/langgraph/run")
async def langgraph_tool_run(request: Request):
    _require_langgraph_tool_access(request)
    body = await request.json()
    if body.get("explicit_user_request") is not True:
        raise HTTPException(
            status_code=400,
            detail=(
                "Hermes may call this tool only when the user explicitly asked "
                "to use LangGraph or AlphaRavis custom-agent flow."
            ),
        )

    message = str(body.get("message") or body.get("input") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    thread_key = str(body.get("thread_key") or body.get("session_id") or "hermes-langgraph-tool")
    timeout = min(
        max(float(body.get("timeout_seconds") or BRIDGE_LANGGRAPH_TOOL_TIMEOUT_SECONDS), 5.0),
        BRIDGE_LANGGRAPH_TOOL_TIMEOUT_SECONDS,
    )

    client = _client()
    thread_id = await _ensure_thread(
        client,
        _tool_thread_id_for_key(thread_key),
        f"hermes-tool:{thread_key}",
    )
    payload = {
        "input": {
            "messages": [
                {
                    "role": "human",
                    "content": (
                        "Hermes explicitly asked AlphaRavis/LangGraph to run this "
                        "bounded subflow. Do not call Hermes back from this run.\n\n"
                        f"{message}"
                    ),
                }
            ],
            "thread_id": thread_id,
            "thread_key": f"hermes-tool:{thread_key}",
        }
    }

    try:
        content = await asyncio.wait_for(_run_wait_content(client, thread_id, payload), timeout=timeout)
    except TimeoutError:
        content = (
            "LangGraph tool run timed out. Hermes should summarize the timeout "
            "and ask the user whether to continue in AlphaRavis directly."
        )

    return JSONResponse(
        {
            "result": _visible_content(content),
            "thread_id": thread_id,
            "thread_key": f"hermes-tool:{thread_key}",
            "next_action": "Return this result to the user. Do not recursively call Hermes or LangGraph.",
        }
    )


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {"id": OPENAI_MODEL_NAME, "object": "model", "created": 0, "owned_by": "alpharavis"},
            {
                "id": SERVER_MODEL_MANAGER_MODEL_NAME,
                "object": "model",
                "created": 0,
                "owned_by": "alpharavis",
            },
        ],
    }


@app.post("/v1/responses")
async def responses(request: Request):
    started = time.perf_counter()
    if not BRIDGE_ENABLE_RESPONSES_API:
        raise HTTPException(status_code=404, detail="Responses API bridge is disabled.")

    body = await request.json()
    trace = _new_trace("responses", body, request)
    _trace_step(trace, "bridge.responses.received", started)
    validation_error = _validate_responses_request(body)
    if validation_error is not None:
        return validation_error
    await _mirror_video_parts_in_responses_body(body, request)
    messages = _responses_messages_for_body(body)
    if not messages:
        raise HTTPException(status_code=400, detail="input is required")
    hard_error = _hard_input_error(messages)
    if hard_error:
        observation_id = _observer_start(protocol="responses", request=request, body=body, messages=messages)
        _observer_hard_cutoff(observation_id, hard_error)
        if BRIDGE_HARD_INPUT_HTTP_ERROR:
            raise HTTPException(status_code=413, detail=hard_error)
        model = str(body.get("model") or OPENAI_MODEL_NAME)
        response = _response_object(hard_error, model, body=body, messages=messages)
        _store_response_object(response, body)
        return JSONResponse(response)

    if body.get("stream") is True:
        return StreamingResponse(_stream_responses(body, request), media_type="text/event-stream")

    observation_id = _observer_start(protocol="responses", request=request, body=body, messages=messages)
    model = str(body.get("model") or OPENAI_MODEL_NAME)
    chat_body = dict(body)
    chat_body["messages"] = messages
    client = _client()
    thread_key = _extract_thread_key(chat_body, request)
    thread_id = await _ensure_thread(client, _thread_id_for_key(thread_key), thread_key)
    _trace_step(trace, "bridge.thread.ready", started, thread_id=thread_id, thread_key=thread_key)
    run_payload = await _prepare_run_payload(client, thread_id, thread_key, messages, model=model, trace=trace)
    _observer_prepared(observation_id, thread_key=thread_key, thread_id=thread_id, run_payload=run_payload)
    _trace_step(trace, "bridge.run_payload.prepared", started)
    _log_event(
        logging.INFO,
        "bridge.responses.request.started",
        thread_id=thread_id,
        thread_key=thread_key,
        model=model,
        message_count=len(messages),
        input_tokens_estimate=_request_token_estimate(messages),
        stream=False,
    )

    if run_payload.get("direct_response"):
        content = str(run_payload["direct_response"])
        _trace_step(trace, "bridge.direct_response.used", started)
    else:
        content = await _run_wait_content(
            client,
            thread_id,
            run_payload,
            trace=trace,
            request_started=started,
            observation_id=observation_id,
        )

    _trace_step(trace, "bridge.response_object.created", started, output_chars=len(content))
    _attach_trace_metadata(body, trace)
    response = _response_object(content, model, body=body, messages=messages)
    _store_response_object(response, body)
    _observer_complete(
        observation_id,
        output_text=content,
        elapsed_seconds=time.perf_counter() - started,
    )
    _log_event(
        logging.INFO,
        "bridge.responses.request.completed",
        response_id=response.get("id", ""),
        thread_id=thread_id,
        model=model,
        output_chars=len(content),
        elapsed_seconds=round(time.perf_counter() - started, 3),
    )
    return JSONResponse(response)


@app.post("/v1/responses/compact")
async def compact_response(request: Request):
    _ = await request.json()
    return _responses_error(
        "OpenAI encrypted Responses compaction is proprietary and is not emulated by AlphaRavis Bridge. "
        "Use AlphaRavis active compression and archive retrieval instead.",
        status_code=501,
        code="compact_not_supported",
    )


@app.post("/v1/responses/input_tokens")
async def response_input_tokens(request: Request):
    if not BRIDGE_ENABLE_RESPONSES_API:
        raise HTTPException(status_code=404, detail="Responses API bridge is disabled.")

    body = await request.json()
    validation_error = _validate_responses_request(body)
    if validation_error is not None:
        return validation_error
    messages = _responses_messages_for_body(body)
    if not messages:
        raise HTTPException(status_code=400, detail="input is required")
    input_tokens = _request_token_estimate(messages)
    return JSONResponse(
        {
            "object": "response.input_tokens",
            "input_tokens": input_tokens,
        }
    )


@app.get("/v1/responses/{response_id}")
async def retrieve_response(response_id: str, stream: bool = False):
    if stream:
        return _responses_error(
            "Streaming retrieval for stored Responses is not implemented by AlphaRavis Bridge. "
            "Create a new streamed response with POST /v1/responses and stream=true.",
            status_code=501,
            code="retrieve_stream_not_supported",
        )
    response = _RESPONSES_STORE.get(response_id)
    if response is None:
        raise HTTPException(status_code=404, detail="Response not found or not stored by this bridge process.")
    return JSONResponse(response)


@app.get("/v1/responses/{response_id}/input_items")
async def list_response_input_items(response_id: str, limit: int = 20, order: str = "desc", after: str = ""):
    items = list(_RESPONSES_INPUT_ITEMS.get(response_id, []))
    if not items and response_id not in _RESPONSES_STORE:
        raise HTTPException(status_code=404, detail="Response not found or not stored by this bridge process.")

    if after:
        indexes = [index for index, item in enumerate(items) if item.get("id") == after]
        if indexes:
            items = items[indexes[0] + 1 :]
    if order == "desc":
        items = list(reversed(items))
    safe_limit = max(1, min(int(limit), 100))
    data = items[:safe_limit]
    return JSONResponse(
        {
            "object": "list",
            "data": data,
            "first_id": data[0]["id"] if data else None,
            "last_id": data[-1]["id"] if data else None,
            "has_more": len(items) > safe_limit,
        }
    )


@app.post("/v1/responses/{response_id}/cancel")
async def cancel_response(response_id: str):
    response = _RESPONSES_STORE.get(response_id)
    if response is None:
        raise HTTPException(status_code=404, detail="Response not found or not stored by this bridge process.")
    if response.get("status") == "completed":
        return _responses_error(
            "Only background/in-progress responses can be cancelled. This bridge currently completes foreground runs synchronously.",
            status_code=400,
            code="response_not_cancellable",
        )
    cancelled = dict(response)
    cancelled["status"] = "cancelled"
    _RESPONSES_STORE[response_id] = cancelled
    return JSONResponse(cancelled)


@app.delete("/v1/responses/{response_id}")
async def delete_response(response_id: str):
    existed = response_id in _RESPONSES_STORE
    if not existed:
        raise HTTPException(status_code=404, detail="Response not found or not stored by this bridge process.")
    del _RESPONSES_STORE[response_id]
    _RESPONSES_INPUT_ITEMS.pop(response_id, None)
    return JSONResponse({"id": response_id, "object": "response", "deleted": True})


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    started = time.perf_counter()
    body = await request.json()
    trace = _new_trace("chat", body, request)
    _trace_step(trace, "bridge.chat.received", started)
    if not body.get("messages"):
        raise HTTPException(status_code=400, detail="messages is required")
    preflight_thread_key = _extract_thread_key(body, request)
    await _mirror_video_parts_in_messages(
        body.get("messages", []),
        thread_id=_thread_id_for_key(preflight_thread_key),
        thread_key=preflight_thread_key,
    )
    hard_error = _hard_input_error(body.get("messages", []))
    if hard_error:
        observation_id = _observer_start(protocol="chat", request=request, body=body, messages=body.get("messages", []))
        _observer_hard_cutoff(observation_id, hard_error)
        if BRIDGE_HARD_INPUT_HTTP_ERROR:
            raise HTTPException(status_code=413, detail=hard_error)
        return JSONResponse(_chat_completion_response(hard_error, str(body.get("model") or OPENAI_MODEL_NAME)))

    if body.get("stream") is True:
        return StreamingResponse(_stream_chat(body, request), media_type="text/event-stream")

    observation_id = _observer_start(protocol="chat", request=request, body=body, messages=body.get("messages", []))
    model = str(body.get("model") or OPENAI_MODEL_NAME)
    client = _client()
    thread_key = _extract_thread_key(body, request)
    thread_id = await _ensure_thread(client, _thread_id_for_key(thread_key), thread_key)
    _trace_step(trace, "bridge.thread.ready", started, thread_id=thread_id, thread_key=thread_key)
    run_payload = await _prepare_run_payload(client, thread_id, thread_key, body.get("messages", []), model=model, trace=trace)
    _observer_prepared(observation_id, thread_key=thread_key, thread_id=thread_id, run_payload=run_payload)
    _trace_step(trace, "bridge.run_payload.prepared", started)
    _log_event(
        logging.INFO,
        "bridge.chat.request.started",
        thread_id=thread_id,
        thread_key=thread_key,
        model=model,
        message_count=len(body.get("messages", [])),
        input_tokens_estimate=_request_token_estimate(body.get("messages", [])),
        stream=False,
    )

    if run_payload.get("direct_response"):
        content = str(run_payload["direct_response"])
        _trace_step(trace, "bridge.direct_response.used", started)
    else:
        content = await _run_wait_content(
            client,
            thread_id,
            run_payload,
            trace=trace,
            request_started=started,
            observation_id=observation_id,
        )

    _trace_step(trace, "bridge.chat.response.created", started, output_chars=len(content))
    _log_event(
        logging.INFO,
        "bridge.chat.request.completed",
        thread_id=thread_id,
        model=model,
        output_chars=len(content),
        elapsed_seconds=round(time.perf_counter() - started, 3),
    )
    response = _chat_completion_response(content, model)
    response["alpharavis_trace"] = trace
    _observer_complete(
        observation_id,
        output_text=content,
        elapsed_seconds=time.perf_counter() - started,
    )
    return JSONResponse(response)
