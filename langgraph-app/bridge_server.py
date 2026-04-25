from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langgraph_sdk import get_client


LANGGRAPH_API_URL = os.getenv("LANGGRAPH_API_URL", "http://langgraph-api:2024")
LANGGRAPH_ASSISTANT_ID = os.getenv("LANGGRAPH_ASSISTANT_ID", "alpha_ravis")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "my-agent")
BRIDGE_RUN_TIMEOUT_SECONDS = float(os.getenv("BRIDGE_RUN_TIMEOUT_SECONDS", "180"))
BRIDGE_STREAM_MODE = os.getenv("BRIDGE_STREAM_MODE", "events").lower()

app = FastAPI(title="AlphaRavis OpenAI Bridge")


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


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        role = message.get("role", "user")
        if role == "assistant":
            role = "ai"
        elif role == "user":
            role = "human"
        normalized.append({"role": role, "content": message.get("content") or ""})
    return normalized


def _last_user_content(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "").strip()
    return ""


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


def _message_type(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("type") or message.get("role") or "").lower()
    return str(getattr(message, "type", getattr(message, "role", ""))).lower()


def _is_ai_message(message: Any) -> bool:
    message_type = _message_type(message)
    return message_type in {"ai", "assistant"} or "aimessage" in message_type


def _last_ai_content(state: Any) -> str:
    messages = state.get("messages", []) if isinstance(state, dict) else []
    memory_notices: list[str] = []
    for message in reversed(messages):
        if _is_ai_message(message):
            content = _message_content(message)
            if content.lstrip().startswith("Memory-Notice:"):
                memory_notices.append(content)
                continue
            if memory_notices:
                return f"{content}\n\n" + "\n".join(reversed(memory_notices))
            return content
    if messages:
        return _message_content(messages[-1])
    return ""


def _chat_completion_response(content: str, model: str) -> dict[str, Any]:
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


def _chunk(content: str, model: str, *, role: str | None = None, finish_reason: str | None = None) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if role:
        delta["role"] = role
    if content:
        delta["content"] = content

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _stream_data(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _openai_stream_response(content: str, model: str) -> list[str]:
    return [
        _stream_data(_chunk("", model, role="assistant")),
        _stream_data(_chunk(content, model)),
        _stream_data(_chunk("", model, finish_reason="stop")),
        "data: [DONE]\n\n",
    ]


def _clean_error_message(exc: Exception) -> str:
    if isinstance(exc, TimeoutError):
        return (
            "AlphaRavis ist erreichbar, aber der LangGraph-Run hat das Bridge-Timeout erreicht. "
            "Sehr wahrscheinlich antwortet das konfigurierte LLM-Backend gerade nicht."
        )

    text = str(exc).strip() or exc.__class__.__name__
    if "Cannot connect to host" in text or "APIConnectionError" in text:
        return (
            "AlphaRavis ist erreichbar, aber das konfigurierte LLM-Backend ist gerade nicht erreichbar. "
            "Pruefe LiteLLM und den in `litellm-config/config.yaml` eingetragenen Modellserver."
        )
    return f"AlphaRavis konnte den LangGraph-Run nicht abschliessen: {text}"


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
    messages: list[dict[str, Any]],
    input_payload: dict[str, Any],
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
) -> AsyncIterator[str]:
    yield _stream_data(_chunk("", model, role="assistant"))
    saw_token = False
    emitted = ""

    stream_kwargs = {
        "stream_mode": "messages",
        "multitask_strategy": "interrupt",
    }
    if "command" in run_payload:
        stream_kwargs["command"] = run_payload["command"]
    else:
        stream_kwargs["input"] = run_payload["input"]

    try:
        async with asyncio.timeout(BRIDGE_RUN_TIMEOUT_SECONDS):
            async for part in client.runs.stream(thread_id, LANGGRAPH_ASSISTANT_ID, **stream_kwargs):
                text = _extract_stream_text(part)
                delta = _delta_text(text, emitted)
                if delta:
                    saw_token = True
                    emitted += delta
                    yield _stream_data(_chunk(delta, model))
    except TimeoutError as exc:
        yield _stream_data(_chunk(_clean_error_message(exc), model))
        yield _stream_data(_chunk("", model, finish_reason="stop"))
        yield "data: [DONE]\n\n"
        return
    except Exception as exc:
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
            yield _stream_data(_chunk(content, model))

    yield _stream_data(_chunk("", model, finish_reason="stop"))
    yield "data: [DONE]\n\n"


async def _stream_chat(body: dict[str, Any], request: Request) -> AsyncIterator[str]:
    model = str(body.get("model") or OPENAI_MODEL_NAME)
    client = _client()
    thread_key = _extract_thread_key(body, request)
    thread_id = await _ensure_thread(client, _thread_id_for_key(thread_key), thread_key)
    input_payload = {"messages": _normalize_messages(body.get("messages", []))}
    run_payload = await _prepare_run_payload(client, thread_id, body.get("messages", []), input_payload)

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


@app.get("/health")
async def health():
    return {"status": "ok", "langgraph_api_url": LANGGRAPH_API_URL}


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [{"id": OPENAI_MODEL_NAME, "object": "model", "created": 0, "owned_by": "alpharavis"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    if not body.get("messages"):
        raise HTTPException(status_code=400, detail="messages is required")

    if body.get("stream") is True:
        return StreamingResponse(_stream_chat(body, request), media_type="text/event-stream")

    model = str(body.get("model") or OPENAI_MODEL_NAME)
    client = _client()
    thread_key = _extract_thread_key(body, request)
    thread_id = await _ensure_thread(client, _thread_id_for_key(thread_key), thread_key)
    input_payload = {"messages": _normalize_messages(body.get("messages", []))}
    run_payload = await _prepare_run_payload(client, thread_id, body.get("messages", []), input_payload)

    if run_payload.get("direct_response"):
        content = str(run_payload["direct_response"])
    else:
        content = await _run_wait_content(client, thread_id, run_payload)

    return JSONResponse(_chat_completion_response(content, model))
