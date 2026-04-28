from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class ResponsesResult:
    content: str
    reasoning: str = ""
    model: str = ""
    raw: dict[str, Any] | None = None
    elapsed_seconds: float = 0.0


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def responses_enabled() -> bool:
    return os.getenv("ALPHARAVIS_LLM_API_MODE", "chat_completions").strip().lower() in {
        "responses",
        "response",
        "native_responses",
    }


def _responses_base_url() -> str:
    return os.getenv(
        "ALPHARAVIS_RESPONSES_API_BASE",
        os.getenv("OPENAI_API_BASE", "http://litellm:4000/v1"),
    ).rstrip("/")


def _responses_api_key() -> str:
    return os.getenv("ALPHARAVIS_RESPONSES_API_KEY", os.getenv("OPENAI_API_KEY", "sk-local-dev"))


def _responses_model(model_name: str | None = None) -> str:
    configured = os.getenv("ALPHARAVIS_RESPONSES_MODEL", "").strip()
    if configured:
        return configured
    model = model_name or os.getenv("ALPHARAVIS_MODEL", "openai/big-boss")
    if _env_bool("ALPHARAVIS_RESPONSES_STRIP_OPENAI_PREFIX", "true") and model.startswith("openai/"):
        return model.removeprefix("openai/")
    return model


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        role = str(message.get("role") or message.get("type") or "user").lower()
    else:
        role = str(getattr(message, "role", getattr(message, "type", "user"))).lower()
    if role in {"human", "user"} or "human" in role:
        return "user"
    if role in {"ai", "assistant"} or "ai" in role:
        return "assistant"
    if role in {"system", "developer"} or "system" in role:
        return "system"
    return role or "user"


def _message_content(message: Any) -> str:
    content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") in {"thinking", "reasoning"}:
                    continue
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _messages_to_responses_payload(messages: list[Any]) -> tuple[str, str]:
    instructions: list[str] = []
    input_lines: list[str] = []
    for message in messages:
        role = _message_role(message)
        content = _message_content(message).strip()
        if not content:
            continue
        if role == "system":
            instructions.append(content)
        else:
            input_lines.append(f"{role}: {content}")
    return "\n\n".join(instructions), "\n\n".join(input_lines)


def _extract_output_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]

    output = data.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") in {"output_text", "text"}:
                        if isinstance(block.get("text"), str):
                            parts.append(block["text"])
            elif isinstance(content, str):
                parts.append(content)
        if parts:
            return "".join(parts)

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        if isinstance(message.get("content"), str):
            return message["content"]
    return ""


def _extract_reasoning_text(data: dict[str, Any]) -> str:
    parts: list[str] = []
    output = data.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "reasoning":
                summary = item.get("summary")
                if isinstance(summary, str):
                    parts.append(summary)
                elif isinstance(summary, list):
                    for block in summary:
                        if isinstance(block, dict) and isinstance(block.get("text"), str):
                            parts.append(block["text"])
            content = item.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") in {"reasoning", "thinking"}:
                        text = block.get("text") or block.get("content")
                        if isinstance(text, str):
                            parts.append(text)

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        for key in ("reasoning_content", "reasoning"):
            if isinstance(message.get(key), str):
                parts.append(message[key])
    return "\n".join(part for part in parts if part)


def _responses_extra_kwargs(model_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    model_kwargs = dict(model_kwargs or {})
    extra: dict[str, Any] = {}
    if "max_tokens" in model_kwargs:
        extra["max_output_tokens"] = model_kwargs.pop("max_tokens")
    if "temperature" in model_kwargs:
        extra["temperature"] = model_kwargs.pop("temperature")
    if "top_p" in model_kwargs:
        extra["top_p"] = model_kwargs.pop("top_p")
    if _env_bool("ALPHARAVIS_RESPONSES_INCLUDE_CHAT_TEMPLATE_KWARGS", "true"):
        chat_template_kwargs = model_kwargs.get("chat_template_kwargs")
        if isinstance(chat_template_kwargs, dict):
            extra["chat_template_kwargs"] = chat_template_kwargs
    return extra


async def invoke_responses(
    messages: list[Any],
    *,
    model_name: str | None = None,
    timeout_seconds: float | None = None,
    model_kwargs: dict[str, Any] | None = None,
    purpose: str = "langgraph",
) -> ResponsesResult:
    instructions, input_text = _messages_to_responses_payload(messages)
    if not input_text:
        input_text = "Continue."

    payload: dict[str, Any] = {
        "model": _responses_model(model_name),
        "input": input_text,
        "store": _env_bool("ALPHARAVIS_RESPONSES_STORE", "false"),
        "parallel_tool_calls": _env_bool("ALPHARAVIS_RESPONSES_PARALLEL_TOOL_CALLS", "true"),
        "truncation": os.getenv("ALPHARAVIS_RESPONSES_TRUNCATION", "disabled"),
        "metadata": {"source": "alpharavis-langgraph", "purpose": purpose},
    }
    if instructions:
        payload["instructions"] = instructions
    payload.update(_responses_extra_kwargs(model_kwargs))

    headers = {"Content-Type": "application/json"}
    api_key = _responses_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    timeout = timeout_seconds or float(os.getenv("ALPHARAVIS_LLM_TIMEOUT_SECONDS", "120"))
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{_responses_base_url()}/responses", headers=headers, json=payload)
    if response.status_code >= 400:
        raise RuntimeError(f"Responses API HTTP {response.status_code}: {response.text[:800]}")
    data = response.json()
    content = _extract_output_text(data)
    return ResponsesResult(
        content=content,
        reasoning=_extract_reasoning_text(data),
        model=str(data.get("model") or payload["model"]),
        raw=data,
        elapsed_seconds=round(time.perf_counter() - started, 3),
    )
