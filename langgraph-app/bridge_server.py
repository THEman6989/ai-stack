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
BRIDGE_RUN_TIMEOUT_SECONDS = float(os.getenv("BRIDGE_RUN_TIMEOUT_SECONDS", "35"))

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


def _last_ai_content(state: Any) -> str:
    messages = state.get("messages", []) if isinstance(state, dict) else []
    for message in reversed(messages):
        message_type = message.get("type") or message.get("role") if isinstance(message, dict) else getattr(message, "type", "")
        if message_type in {"ai", "assistant"}:
            return _message_content(message)
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


def _extract_stream_text(part: Any) -> str:
    data = getattr(part, "data", None)
    if data is None and isinstance(part, dict):
        data = part.get("data")

    if isinstance(data, tuple) and data:
        return _message_content(data[0])

    if isinstance(data, list) and data:
        return _message_content(data[0])

    if isinstance(data, dict):
        if "chunk" in data:
            return _message_content(data["chunk"])
        if "messages" in data and data["messages"]:
            return _message_content(data["messages"][-1])

    return ""


async def _stream_chat(body: dict[str, Any], request: Request) -> AsyncIterator[str]:
    model = str(body.get("model") or OPENAI_MODEL_NAME)
    client = _client()
    thread_key = _extract_thread_key(body, request)
    thread_id = await _ensure_thread(client, _thread_id_for_key(thread_key), thread_key)
    input_payload = {"messages": _normalize_messages(body.get("messages", []))}

    yield _stream_data(_chunk("", model, role="assistant"))
    saw_token = False

    try:
        async with asyncio.timeout(BRIDGE_RUN_TIMEOUT_SECONDS):
            async for part in client.runs.stream(
                thread_id,
                LANGGRAPH_ASSISTANT_ID,
                input=input_payload,
                stream_mode="messages",
                multitask_strategy="interrupt",
            ):
                text = _extract_stream_text(part)
                if text:
                    saw_token = True
                    yield _stream_data(_chunk(text, model))
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

    try:
        state = await asyncio.wait_for(
            client.runs.wait(
                thread_id,
                LANGGRAPH_ASSISTANT_ID,
                input=input_payload,
                multitask_strategy="interrupt",
            ),
            timeout=BRIDGE_RUN_TIMEOUT_SECONDS,
        )
        content = _last_ai_content(state)
    except TimeoutError as exc:
        content = _clean_error_message(exc)
    except Exception as exc:
        content = _clean_error_message(exc)

    return JSONResponse(_chat_completion_response(content, model))
