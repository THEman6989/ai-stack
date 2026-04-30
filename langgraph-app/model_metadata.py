from __future__ import annotations

import math
import os
import re
from typing import Any


CHARS_PER_TOKEN = 4
MINIMUM_CONTEXT_LENGTH = 4096
DEFAULT_CONTEXT_LENGTH = 128000
CONTEXT_LENGTH_KEYS = (
    "context_length",
    "context_window",
    "max_context_length",
    "max_position_embeddings",
    "max_model_len",
    "max_input_tokens",
    "max_sequence_length",
    "max_seq_len",
    "n_ctx_train",
    "n_ctx",
    "ctx_size",
)

DEFAULT_CONTEXT_LENGTHS = {
    "qwen": 128000,
    "llama": 128000,
    "deepseek": 128000,
    "mistral": 128000,
    "gemma": 32768,
    "claude": 200000,
    "gpt": 128000,
}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _content_to_text(content: Any) -> str:
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block.get("text") or ""))
                elif "content" in block:
                    parts.append(str(block.get("content") or ""))
                elif "input_text" in block:
                    parts.append(str(block.get("input_text") or ""))
                else:
                    parts.append(str(block))
            else:
                parts.append(str(block))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def estimate_tokens_rough_text(text: str) -> int:
    return max(1, math.ceil(len(str(text or "")) / CHARS_PER_TOKEN))


def image_token_estimate() -> int:
    return max(1, _env_int("ALPHARAVIS_COMPRESSION_IMAGE_TOKEN_ESTIMATE", 1600))


def _is_image_block(block: dict[str, Any]) -> bool:
    block_type = str(block.get("type") or "").lower()
    if block_type in {"image", "image_url", "input_image"}:
        return True
    return any(key in block for key in ("image_url", "input_image", "image", "file_id"))


def content_token_estimate(content: Any) -> int:
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict):
                if _is_image_block(block):
                    total += image_token_estimate()
                total += estimate_tokens_rough_text(_content_to_text(block))
            else:
                total += estimate_tokens_rough_text(str(block))
        return max(1, total)
    return estimate_tokens_rough_text(str(content or ""))


def _message_mapping(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return message
    mapping = {
        "type": getattr(message, "type", None),
        "role": getattr(message, "role", None),
        "name": getattr(message, "name", None),
        "content": getattr(message, "content", ""),
        "tool_calls": getattr(message, "tool_calls", None),
        "additional_kwargs": getattr(message, "additional_kwargs", {}) or {},
        "response_metadata": getattr(message, "response_metadata", {}) or {},
        "usage_metadata": getattr(message, "usage_metadata", {}) or {},
    }
    return mapping


def _tool_calls_from_message(message: Any) -> list[Any]:
    mapping = _message_mapping(message)
    calls = mapping.get("tool_calls") or []
    additional = mapping.get("additional_kwargs") or {}
    if additional.get("tool_calls"):
        calls = [*calls, *additional.get("tool_calls")]
    return calls if isinstance(calls, list) else []


def estimate_tool_call_tokens(message: Any) -> int:
    total = 0
    for call in _tool_calls_from_message(message):
        if isinstance(call, dict):
            total += estimate_tokens_rough_text(str(call.get("name") or call.get("id") or ""))
            args = call.get("args")
            if args is None and isinstance(call.get("function"), dict):
                args = call["function"].get("arguments")
                total += estimate_tokens_rough_text(str(call["function"].get("name") or ""))
            total += estimate_tokens_rough_text(str(args or ""))
        else:
            total += estimate_tokens_rough_text(str(call))
    return total


def estimate_message_tokens_rough(message: Any) -> int:
    mapping = _message_mapping(message)
    role_name_tokens = estimate_tokens_rough_text(
        " ".join(str(mapping.get(key) or "") for key in ("role", "type", "name"))
    )
    return max(
        1,
        role_name_tokens
        + content_token_estimate(mapping.get("content", ""))
        + estimate_tool_call_tokens(message)
        + 4,
    )


def estimate_messages_tokens_rough(messages: list[Any]) -> int:
    if not messages:
        return 1
    return max(1, sum(estimate_message_tokens_rough(message) for message in messages))


def extract_usage_tokens(value: Any) -> int | None:
    candidates: list[Any] = []
    if isinstance(value, dict):
        candidates.append(value)
        for key in ("usage", "token_usage", "usage_metadata", "response_metadata"):
            if isinstance(value.get(key), dict):
                candidates.append(value[key])
    else:
        for key in ("usage_metadata", "response_metadata"):
            item = getattr(value, key, None)
            if isinstance(item, dict):
                candidates.append(item)

    for candidate in list(candidates):
        for key in ("usage", "token_usage", "usage_metadata"):
            nested = candidate.get(key) if isinstance(candidate, dict) else None
            if isinstance(nested, dict):
                candidates.append(nested)

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for key in ("total_tokens", "totalTokenCount"):
            value = candidate.get(key)
            if isinstance(value, int):
                return value
        input_tokens = candidate.get("input_tokens", candidate.get("prompt_tokens"))
        output_tokens = candidate.get("output_tokens", candidate.get("completion_tokens"))
        if isinstance(input_tokens, int) or isinstance(output_tokens, int):
            return int(input_tokens or 0) + int(output_tokens or 0)
    return None


def _sanitize_model_env_key(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_").upper()


def get_model_context_length(model: str | None = None, provider: str | None = None, default: int | None = None) -> int:
    if model:
        model_key = _sanitize_model_env_key(model)
        specific = _env_int(f"ALPHARAVIS_CONTEXT_LENGTH_{model_key}", 0)
        if specific > 0:
            return max(MINIMUM_CONTEXT_LENGTH, specific)

    configured = _env_int("ALPHARAVIS_MODEL_CONTEXT_LENGTH", 0)
    if configured > 0:
        return max(MINIMUM_CONTEXT_LENGTH, configured)

    configured_default = _env_int("ALPHARAVIS_DEFAULT_CONTEXT_LENGTH", 0)
    if configured_default > 0:
        return max(MINIMUM_CONTEXT_LENGTH, configured_default)

    haystack = f"{provider or ''} {model or ''}".lower()
    for name, length in DEFAULT_CONTEXT_LENGTHS.items():
        if name in haystack:
            return max(MINIMUM_CONTEXT_LENGTH, length)

    return max(MINIMUM_CONTEXT_LENGTH, int(default or DEFAULT_CONTEXT_LENGTH))
