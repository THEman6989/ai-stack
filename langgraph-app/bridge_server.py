from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langgraph_sdk import get_client

from context_references import preprocess_context_references
from error_classifier import classify_api_error, format_user_error
from internal_context import StreamingInternalContextScrubber, sanitize_internal_context


LANGGRAPH_API_URL = os.getenv("LANGGRAPH_API_URL", "http://langgraph-api:2024")
LANGGRAPH_ASSISTANT_ID = os.getenv("LANGGRAPH_ASSISTANT_ID", "alpha_ravis")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "my-agent")
BRIDGE_RUN_TIMEOUT_SECONDS = float(os.getenv("BRIDGE_RUN_TIMEOUT_SECONDS", "180"))
BRIDGE_STREAM_MODE = os.getenv("BRIDGE_STREAM_MODE", "events").lower()
BRIDGE_MESSAGE_SYNC_MODE = os.getenv("BRIDGE_MESSAGE_SYNC_MODE", "delta").lower()
BRIDGE_SHOW_ACTIVITY_EVENTS = os.getenv("BRIDGE_SHOW_ACTIVITY_EVENTS", "false").lower() in {
    "1",
    "true",
    "yes",
}
BRIDGE_ACTIVITY_DETAIL = os.getenv("BRIDGE_ACTIVITY_DETAIL", "summary").lower()
BRIDGE_STREAM_REASONING_EVENTS = os.getenv("BRIDGE_STREAM_REASONING_EVENTS", "false").lower() in {
    "1",
    "true",
    "yes",
}
BRIDGE_REASONING_DELTA_FIELD = os.getenv("BRIDGE_REASONING_DELTA_FIELD", "reasoning_content")
BRIDGE_ENABLE_RESPONSES_API = os.getenv("BRIDGE_ENABLE_RESPONSES_API", "true").lower() in {
    "1",
    "true",
    "yes",
}
BRIDGE_PREFERRED_API_MODE = os.getenv("BRIDGE_PREFERRED_API_MODE", "responses").lower()
BRIDGE_RESPONSES_STORE = os.getenv("BRIDGE_RESPONSES_STORE", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
BRIDGE_RESPONSES_STORE_MAX = int(os.getenv("BRIDGE_RESPONSES_STORE_MAX", "200"))
BRIDGE_RESPONSES_DONE_SENTINEL = os.getenv("BRIDGE_RESPONSES_DONE_SENTINEL", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
BRIDGE_RESPONSES_ALLOW_CLIENT_TOOLS = os.getenv("BRIDGE_RESPONSES_ALLOW_CLIENT_TOOLS", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
BRIDGE_HARD_INPUT_TOKEN_LIMIT = int(os.getenv("BRIDGE_HARD_INPUT_TOKEN_LIMIT", "128000"))
BRIDGE_HARD_INPUT_HTTP_ERROR = os.getenv("BRIDGE_HARD_INPUT_HTTP_ERROR", "false").lower() in {
    "1",
    "true",
    "yes",
}
BRIDGE_ALLOW_RAW_MEDIA_CONTEXT = os.getenv("BRIDGE_ALLOW_RAW_MEDIA_CONTEXT", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
BRIDGE_MEDIA_CONTEXT_MODE = os.getenv("BRIDGE_MEDIA_CONTEXT_MODE", "metadata").lower()
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
BRIDGE_LLM_HEALTH_URL = os.getenv("BRIDGE_LLM_HEALTH_URL", "http://litellm:4000/v1").rstrip("/")
BRIDGE_LLM_HEALTH_API_KEY = os.getenv("BRIDGE_LLM_HEALTH_API_KEY", os.getenv("OPENAI_API_KEY", "sk-local-dev"))
BRIDGE_LLM_HEALTH_MODEL = os.getenv("BRIDGE_LLM_HEALTH_MODEL", "big-boss")
BRIDGE_LLM_HEALTH_FALLBACK_MODEL = os.getenv("BRIDGE_LLM_HEALTH_FALLBACK_MODEL", "edge-gemma")
BRIDGE_LLM_HEALTH_TIMEOUT_SECONDS = float(os.getenv("BRIDGE_LLM_HEALTH_TIMEOUT_SECONDS", "10"))
BRIDGE_LLM_HEALTH_PROMPT = os.getenv("BRIDGE_LLM_HEALTH_PROMPT", "Antworte nur mit OK.")
BRIDGE_ENABLE_LANGGRAPH_TOOL = os.getenv("BRIDGE_ENABLE_LANGGRAPH_TOOL", "false").lower() in {
    "1",
    "true",
    "yes",
}
BRIDGE_LANGGRAPH_TOOL_API_KEY = os.getenv("BRIDGE_LANGGRAPH_TOOL_API_KEY", "")
BRIDGE_LANGGRAPH_TOOL_TIMEOUT_SECONDS = float(os.getenv("BRIDGE_LANGGRAPH_TOOL_TIMEOUT_SECONDS", "120"))
BRIDGE_SHOW_ERROR_CLASSIFICATION = os.getenv("BRIDGE_SHOW_ERROR_CLASSIFICATION", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
BRIDGE_SCRUB_INTERNAL_CONTEXT = os.getenv("BRIDGE_SCRUB_INTERNAL_CONTEXT", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
BRIDGE_ENABLE_CONTEXT_REFERENCES = os.getenv("BRIDGE_ENABLE_CONTEXT_REFERENCES", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
BRIDGE_CONTEXT_REFERENCES_FETCH_URLS = os.getenv("BRIDGE_CONTEXT_REFERENCES_FETCH_URLS", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
BRIDGE_CONTEXT_REFERENCE_CONTEXT_LENGTH = int(
    os.getenv("BRIDGE_CONTEXT_REFERENCE_CONTEXT_LENGTH", str(BRIDGE_HARD_INPUT_TOKEN_LIMIT or 128000))
)
BRIDGE_CONTEXT_REFERENCE_SOFT_RATIO = float(os.getenv("BRIDGE_CONTEXT_REFERENCE_SOFT_RATIO", "0.25"))
BRIDGE_CONTEXT_REFERENCE_HARD_RATIO = float(os.getenv("BRIDGE_CONTEXT_REFERENCE_HARD_RATIO", "0.50"))
BRIDGE_CONTEXT_REFERENCE_MAX_URL_CHARS = int(os.getenv("BRIDGE_CONTEXT_REFERENCE_MAX_URL_CHARS", "12000"))
BRIDGE_CONTEXT_REFERENCE_FOLDER_LIMIT = int(os.getenv("BRIDGE_CONTEXT_REFERENCE_FOLDER_LIMIT", "200"))
BRIDGE_WORKSPACE_ROOT = Path(
    os.getenv("BRIDGE_CONTEXT_REFERENCE_WORKSPACE_ROOT") or Path(__file__).resolve().parents[1]
).expanduser().resolve()
BRIDGE_CONTEXT_REFERENCE_CWD = Path(
    os.getenv("BRIDGE_CONTEXT_REFERENCE_CWD") or BRIDGE_WORKSPACE_ROOT
).expanduser().resolve()

app = FastAPI(title="AlphaRavis OpenAI Bridge", openapi_version="3.1.0")
_RESPONSES_STORE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_RESPONSES_INPUT_ITEMS: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()


def _client():
    return get_client(url=LANGGRAPH_API_URL)


def _extract_thread_key(body: dict[str, Any], request: Request) -> str:
    metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
    candidates = [
        body.get("conversationId"),
        body.get("conversation_id"),
        metadata.get("conversationId"),
        metadata.get("conversation_id"),
        body.get("user"),
        request.headers.get("x-conversation-id"),
        request.headers.get("x-thread-id"),
    ]
    for candidate in candidates:
        if candidate:
            return str(candidate)
    first_user = next(
        (message.get("content") for message in body.get("messages", []) if message.get("role") == "user"),
        "default",
    )
    return first_user[:120]


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


def _last_ai_content(state: Any) -> str:
    messages = state.get("messages", []) if isinstance(state, dict) else []
    trailing_notices: list[str] = []
    for message in reversed(messages):
        if _is_ai_message(message):
            content = _message_content(message)
            stripped = content.lstrip()
            if stripped.startswith(("Memory-Notice:", "Run-Profile:", "Fast-Path-Notice:")):
                trailing_notices.append(content)
                continue
            if trailing_notices:
                return f"{content}\n\n" + "\n".join(reversed(trailing_notices))
            return content
    if messages:
        return _message_content(messages[-1])
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
) -> dict[str, Any]:
    normalized = _normalize_messages(raw_messages)
    if BRIDGE_MESSAGE_SYNC_MODE in {"full", "all"}:
        selected = normalized
    else:
        last_human = _last_human_content_from_state(state)
        selected = _messages_after_last_human(normalized, last_human)
        if not selected and normalized:
            selected = [normalized[-1]]

    selected, reference_profiles = await _apply_context_references_to_messages(selected)

    payload = {
        "messages": selected,
        "thread_id": thread_id,
        "thread_key": thread_key,
        "bridge_context_references": reference_profiles,
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
) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if role:
        delta["role"] = role
    if content:
        delta["content"] = content
    if reasoning_content:
        delta[BRIDGE_REASONING_DELTA_FIELD] = reasoning_content

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
        "max_tokens": 4,
        "temperature": 0,
        "stream": False,
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
        return {
            "ok": bool(content or reasoning),
            "model": model,
            "status_code": response.status_code,
            "elapsed_seconds": elapsed,
            "content_preview": (content or reasoning)[:120],
        }
    except Exception as exc:
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
        "Antworte mit `approve`, `reject` oder `replace: <sichererer Befehl>`."
    )


async def _prepare_run_payload(
    client: Any,
    thread_id: str,
    thread_key: str,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        state = await client.threads.get_state(thread_id)
    except Exception:
        state = None

    interrupt_value = _find_command_approval_interrupt(state)
    if interrupt_value:
        resume = _approval_resume_from_messages(messages)
        if resume is None:
            return {"direct_response": _approval_prompt(interrupt_value)}
        return {"command": {"resume": resume}}

    input_payload = await _build_input_payload(
        messages,
        state,
        thread_id=thread_id,
        thread_key=thread_key,
    )
    return {"input": input_payload}


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


def _extract_stream_reasoning(part: Any) -> str:
    if not BRIDGE_STREAM_REASONING_EVENTS:
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


def _stream_event_name(part: Any) -> str:
    if isinstance(part, dict):
        return str(part.get("event") or "")
    return str(getattr(part, "event", ""))


def _stream_event_data(part: Any) -> Any:
    if isinstance(part, dict):
        return part.get("data")
    return getattr(part, "data", None)


def _extract_activity_text(part: Any) -> str:
    if not BRIDGE_SHOW_ACTIVITY_EVENTS or BRIDGE_ACTIVITY_DETAIL == "off":
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
) -> AsyncIterator[str]:
    yield _stream_data(_chunk("", model, role="assistant"))
    if include_activity and BRIDGE_SHOW_ACTIVITY_EVENTS and BRIDGE_ACTIVITY_DETAIL != "off":
        yield _activity_chunk("AlphaRavis startet den LangGraph-Run.", model)

    saw_token = False
    emitted = ""
    emitted_reasoning = ""
    emitted_activity: set[str] = set()
    content_scrubber = StreamingInternalContextScrubber() if BRIDGE_SCRUB_INTERNAL_CONTEXT else None
    reasoning_scrubber = StreamingInternalContextScrubber() if BRIDGE_SCRUB_INTERNAL_CONTEXT else None

    stream_kwargs = {
        "stream_mode": ["messages", "updates"] if include_activity and BRIDGE_SHOW_ACTIVITY_EVENTS else "messages",
        "multitask_strategy": "interrupt",
    }
    if "command" in run_payload:
        stream_kwargs["command"] = run_payload["command"]
    else:
        stream_kwargs["input"] = run_payload["input"]

    try:
        async with asyncio.timeout(BRIDGE_RUN_TIMEOUT_SECONDS):
            async for part in client.runs.stream(thread_id, LANGGRAPH_ASSISTANT_ID, **stream_kwargs):
                activity = _extract_activity_text(part)
                if include_activity and activity and activity not in emitted_activity:
                    emitted_activity.add(activity)
                    yield _activity_chunk(activity, model)

                reasoning = _extract_stream_reasoning(part)
                reasoning_delta = _delta_text(reasoning, emitted_reasoning)
                if reasoning_delta:
                    emitted_reasoning += reasoning_delta
                    visible_reasoning_delta = (
                        reasoning_scrubber.feed(reasoning_delta) if reasoning_scrubber else reasoning_delta
                    )
                    if visible_reasoning_delta:
                        yield _stream_data(_chunk("", model, reasoning_content=visible_reasoning_delta))

                text = _extract_stream_text(part)
                delta = _delta_text(text, emitted)
                if delta:
                    emitted += delta
                    visible_delta = content_scrubber.feed(delta) if content_scrubber else delta
                    if visible_delta:
                        saw_token = True
                        yield _stream_data(_chunk(visible_delta, model))
    except TimeoutError as exc:
        if include_activity and BRIDGE_SHOW_ERROR_CLASSIFICATION and BRIDGE_SHOW_ACTIVITY_EVENTS and BRIDGE_ACTIVITY_DETAIL != "off":
            yield _activity_chunk(_error_activity_text(exc), model)
        yield _stream_data(_chunk(_clean_error_message(exc), model))
        yield _stream_data(_chunk("", model, finish_reason="stop"))
        yield "data: [DONE]\n\n"
        return
    except Exception as exc:
        if include_activity and BRIDGE_SHOW_ERROR_CLASSIFICATION and BRIDGE_SHOW_ACTIVITY_EVENTS and BRIDGE_ACTIVITY_DETAIL != "off":
            yield _activity_chunk(_error_activity_text(exc), model)
        yield _stream_data(_chunk(_clean_error_message(exc), model))
        yield _stream_data(_chunk("", model, finish_reason="stop"))
        yield "data: [DONE]\n\n"
        return

    if not saw_token:
        try:
            state = await client.threads.get_state(thread_id)
            content = _last_ai_content(state.get("values", state))
        except Exception as exc:
            content = _clean_error_message(exc)
        if content:
            visible = _visible_content(content)
            if visible:
                yield _stream_data(_chunk(visible, model))

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


async def _stream_chat(body: dict[str, Any], request: Request) -> AsyncIterator[str]:
    model = str(body.get("model") or OPENAI_MODEL_NAME)
    hard_error = _hard_input_error(body.get("messages", []))
    if hard_error:
        for chunk in _openai_stream_response(hard_error, model):
            yield chunk
        return

    client = _client()
    thread_key = _extract_thread_key(body, request)
    thread_id = await _ensure_thread(client, _thread_id_for_key(thread_key), thread_key)
    run_payload = await _prepare_run_payload(client, thread_id, thread_key, body.get("messages", []))

    if run_payload.get("direct_response"):
        for chunk in _openai_stream_response(str(run_payload["direct_response"]), model):
            yield chunk
        return

    if BRIDGE_STREAM_MODE in {"final", "message", "messages"}:
        async for chunk in _stream_chat_final(client, thread_id, run_payload, model):
            yield chunk
        return

    async for chunk in _stream_chat_events(client, thread_id, run_payload, model):
        yield chunk


async def _run_wait_content(client: Any, thread_id: str, run_payload: dict[str, Any]) -> str:
    wait_kwargs = {"multitask_strategy": "interrupt"}
    if "command" in run_payload:
        wait_kwargs["command"] = run_payload["command"]
    else:
        wait_kwargs["input"] = run_payload["input"]

    try:
        state = await asyncio.wait_for(
            client.runs.wait(thread_id, LANGGRAPH_ASSISTANT_ID, **wait_kwargs),
            timeout=BRIDGE_RUN_TIMEOUT_SECONDS,
        )
        return _last_ai_content(state)
    except TimeoutError as exc:
        return _clean_error_message(exc)
    except Exception as exc:
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
        "content": [{"type": "output_text", "text": content, "annotations": []}],
    }


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
        "output": [_response_message_item(content, item_id=item_id, status=status)] if status != "failed" else [],
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
            "Pass conversationId, conversation_id, user, x-conversation-id, or x-thread-id.",
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


async def _stream_responses(body: dict[str, Any], request: Request) -> AsyncIterator[str]:
    model = str(body.get("model") or OPENAI_MODEL_NAME)
    messages = _responses_messages_for_body(body)
    response_id = f"resp_{uuid.uuid4().hex}"
    item_id = f"msg_{uuid.uuid4().hex}"
    hard_error = _hard_input_error(messages)
    if hard_error:
        yield _responses_event(
            "response.created",
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": _response_created_payload(response_id, model, body),
            },
        )
        yield _responses_event(
            "response.in_progress",
            {
                "type": "response.in_progress",
                "sequence_number": 1,
                "response": _response_created_payload(response_id, model, body),
            },
        )
        yield _responses_event(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "sequence_number": 2,
                "output_index": 0,
                "item": {
                    "id": item_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                },
            },
        )
        yield _responses_event(
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "sequence_number": 3,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
        )
        yield _responses_event(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "sequence_number": 4,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": hard_error,
            },
        )
        yield _responses_event(
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "sequence_number": 5,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "text": hard_error,
            },
        )
        yield _responses_event(
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "sequence_number": 6,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": hard_error, "annotations": []},
            },
        )
        response = _response_object(hard_error, model, response_id, item_id=item_id, body=body, messages=messages)
        _store_response_object(response, body)
        yield _responses_event(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "sequence_number": 7,
                "output_index": 0,
                "item": response["output"][0],
            },
        )
        yield _responses_event(
            "response.completed",
            {"type": "response.completed", "sequence_number": 8, "response": response},
        )
        if done := _done_sentinel():
            yield done
        return

    sequence_number = 0
    yield _responses_event(
        "response.created",
        {
            "type": "response.created",
            "sequence_number": sequence_number,
            "response": _response_created_payload(response_id, model, body),
        },
    )
    sequence_number += 1
    yield _responses_event(
        "response.in_progress",
        {
            "type": "response.in_progress",
            "sequence_number": sequence_number,
            "response": _response_created_payload(response_id, model, body),
        },
    )
    sequence_number += 1
    yield _responses_event(
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "sequence_number": sequence_number,
            "output_index": 0,
            "item": {
                "id": item_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        },
    )
    sequence_number += 1
    yield _responses_event(
        "response.content_part.added",
        {
            "type": "response.content_part.added",
            "sequence_number": sequence_number,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
    )
    sequence_number += 1

    chat_body = dict(body)
    chat_body["messages"] = messages
    client = _client()
    thread_key = _extract_thread_key(chat_body, request)
    thread_id = await _ensure_thread(client, _thread_id_for_key(thread_key), thread_key)
    run_payload = await _prepare_run_payload(client, thread_id, thread_key, chat_body.get("messages", []))

    if run_payload.get("direct_response"):
        content = str(run_payload["direct_response"])
        yield _responses_event(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "sequence_number": sequence_number,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": content,
            },
        )
        sequence_number += 1
        yield _responses_event(
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "sequence_number": sequence_number,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "text": content,
            },
        )
        sequence_number += 1
        yield _responses_event(
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "sequence_number": sequence_number,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": content, "annotations": []},
            },
        )
        sequence_number += 1
        response = _response_object(content, model, response_id, item_id=item_id, body=body, messages=messages)
        yield _responses_event(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "sequence_number": sequence_number,
                "output_index": 0,
                "item": response["output"][0],
            },
        )
        sequence_number += 1
        _store_response_object(response, body)
        yield _responses_event(
            "response.completed",
            {"type": "response.completed", "sequence_number": sequence_number, "response": response},
        )
        if done := _done_sentinel():
            yield done
        return

    full_content = ""
    full_reasoning = ""
    async for raw in _stream_chat_events(client, thread_id, run_payload, model, include_activity=False):
        if raw.strip() == "data: [DONE]":
            break
        if not raw.startswith("data: "):
            continue
        try:
            payload = json.loads(raw.removeprefix("data: ").strip())
        except Exception:
            continue
        choice = payload.get("choices", [{}])[0]
        delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
        text_delta = str(delta.get("content") or "")
        reasoning_delta = str(delta.get(BRIDGE_REASONING_DELTA_FIELD) or "")
        if reasoning_delta:
            full_reasoning += reasoning_delta
            yield _responses_event(
                "response.reasoning_text.delta",
                {
                    "type": "response.reasoning_text.delta",
                    "sequence_number": sequence_number,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": reasoning_delta,
                },
            )
            sequence_number += 1
        if text_delta:
            full_content += text_delta
            yield _responses_event(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "sequence_number": sequence_number,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": text_delta,
                },
            )
            sequence_number += 1

    if full_reasoning:
        yield _responses_event(
            "response.reasoning_text.done",
            {
                "type": "response.reasoning_text.done",
                "sequence_number": sequence_number,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "text": full_reasoning,
            },
        )
        sequence_number += 1
    yield _responses_event(
        "response.output_text.done",
        {
            "type": "response.output_text.done",
            "sequence_number": sequence_number,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "text": full_content,
        },
    )
    sequence_number += 1
    yield _responses_event(
        "response.content_part.done",
        {
            "type": "response.content_part.done",
            "sequence_number": sequence_number,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": full_content, "annotations": []},
        },
    )
    sequence_number += 1
    response = _response_object(full_content, model, response_id, item_id=item_id, body=body, messages=messages)
    yield _responses_event(
        "response.output_item.done",
        {
            "type": "response.output_item.done",
            "sequence_number": sequence_number,
            "output_index": 0,
            "item": response["output"][0],
        },
    )
    sequence_number += 1
    _store_response_object(response, body)
    yield _responses_event(
        "response.completed",
        {"type": "response.completed", "sequence_number": sequence_number, "response": response},
    )
    if done := _done_sentinel():
        yield done


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
        "data": [{"id": OPENAI_MODEL_NAME, "object": "model", "created": 0, "owned_by": "alpharavis"}],
    }


@app.post("/v1/responses")
async def responses(request: Request):
    if not BRIDGE_ENABLE_RESPONSES_API:
        raise HTTPException(status_code=404, detail="Responses API bridge is disabled.")

    body = await request.json()
    validation_error = _validate_responses_request(body)
    if validation_error is not None:
        return validation_error
    messages = _responses_messages_for_body(body)
    if not messages:
        raise HTTPException(status_code=400, detail="input is required")
    hard_error = _hard_input_error(messages)
    if hard_error:
        if BRIDGE_HARD_INPUT_HTTP_ERROR:
            raise HTTPException(status_code=413, detail=hard_error)
        model = str(body.get("model") or OPENAI_MODEL_NAME)
        response = _response_object(hard_error, model, body=body, messages=messages)
        _store_response_object(response, body)
        return JSONResponse(response)

    if body.get("stream") is True:
        return StreamingResponse(_stream_responses(body, request), media_type="text/event-stream")

    model = str(body.get("model") or OPENAI_MODEL_NAME)
    chat_body = dict(body)
    chat_body["messages"] = messages
    client = _client()
    thread_key = _extract_thread_key(chat_body, request)
    thread_id = await _ensure_thread(client, _thread_id_for_key(thread_key), thread_key)
    run_payload = await _prepare_run_payload(client, thread_id, thread_key, messages)

    if run_payload.get("direct_response"):
        content = str(run_payload["direct_response"])
    else:
        content = await _run_wait_content(client, thread_id, run_payload)

    response = _response_object(content, model, body=body, messages=messages)
    _store_response_object(response, body)
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
    body = await request.json()
    if not body.get("messages"):
        raise HTTPException(status_code=400, detail="messages is required")
    hard_error = _hard_input_error(body.get("messages", []))
    if hard_error:
        if BRIDGE_HARD_INPUT_HTTP_ERROR:
            raise HTTPException(status_code=413, detail=hard_error)
        return JSONResponse(_chat_completion_response(hard_error, str(body.get("model") or OPENAI_MODEL_NAME)))

    if body.get("stream") is True:
        return StreamingResponse(_stream_chat(body, request), media_type="text/event-stream")

    model = str(body.get("model") or OPENAI_MODEL_NAME)
    client = _client()
    thread_key = _extract_thread_key(body, request)
    thread_id = await _ensure_thread(client, _thread_id_for_key(thread_key), thread_key)
    run_payload = await _prepare_run_payload(client, thread_id, thread_key, body.get("messages", []))

    if run_payload.get("direct_response"):
        content = str(run_payload["direct_response"])
    else:
        content = await _run_wait_content(client, thread_id, run_payload)

    return JSONResponse(_chat_completion_response(content, model))
