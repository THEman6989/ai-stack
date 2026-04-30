from __future__ import annotations

import math
import os
import re
import time
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

_ENDPOINT_MODEL_METADATA_CACHE: dict[str, dict[str, dict[str, Any]]] = {}
_ENDPOINT_MODEL_METADATA_CACHE_TIME: dict[str, float] = {}
_ENDPOINT_MODEL_CACHE_TTL = 300


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
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


def _normalize_base_url(base_url: str | None) -> str:
    normalized = str(base_url or "").strip().rstrip("/")
    return normalized


def _auth_headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _iter_nested_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _iter_nested_dicts(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_nested_dicts(item)


def _coerce_reasonable_int(value: Any, minimum: int = 1024, maximum: int = 10_000_000) -> int | None:
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, str):
            value = value.strip().replace(",", "")
        result = int(value)
    except (TypeError, ValueError):
        return None
    if minimum <= result <= maximum:
        return result
    return None


def _extract_context_length(payload: dict[str, Any]) -> int | None:
    keyset = {key.lower() for key in CONTEXT_LENGTH_KEYS}
    for mapping in _iter_nested_dicts(payload):
        for key, value in mapping.items():
            if str(key).lower() not in keyset:
                continue
            coerced = _coerce_reasonable_int(value)
            if coerced is not None:
                return coerced
    return None


def _add_model_aliases(cache: dict[str, dict[str, Any]], model_id: str, entry: dict[str, Any]) -> None:
    cache[model_id] = entry
    if "/" in model_id:
        cache.setdefault(model_id.split("/", 1)[1], entry)


def _base_without_v1(base_url: str) -> str:
    return base_url[:-3].rstrip("/") if base_url.endswith("/v1") else base_url


def _query_llamacpp_props(base_url: str, api_key: str = "") -> tuple[int | None, str]:
    try:
        import httpx
    except Exception:
        return None, ""

    base = _base_without_v1(_normalize_base_url(base_url))
    headers = _auth_headers(api_key)
    timeout = max(0.5, _env_float("ALPHARAVIS_CONTEXT_DISCOVERY_TIMEOUT_SECONDS", 2.0))
    for suffix in ("/v1/props", "/props"):
        try:
            response = httpx.get(base + suffix, headers=headers, timeout=timeout)
            if response.status_code != 200:
                continue
            payload = response.json()
        except Exception:
            continue
        context_length = _extract_context_length(payload)
        if context_length:
            return context_length, str(payload.get("model_alias") or payload.get("model") or "")
    return None, ""


def fetch_endpoint_model_metadata(
    base_url: str,
    api_key: str = "",
    *,
    force_refresh: bool = False,
) -> dict[str, dict[str, Any]]:
    normalized = _normalize_base_url(base_url)
    if not normalized:
        return {}

    if not force_refresh:
        cached = _ENDPOINT_MODEL_METADATA_CACHE.get(normalized)
        cached_at = _ENDPOINT_MODEL_METADATA_CACHE_TIME.get(normalized, 0)
        if cached is not None and (time.time() - cached_at) < _ENDPOINT_MODEL_CACHE_TTL:
            return cached

    try:
        import httpx
    except Exception:
        return {}

    candidates = [normalized]
    alternate = normalized[:-3].rstrip("/") if normalized.endswith("/v1") else normalized + "/v1"
    if alternate and alternate not in candidates:
        candidates.append(alternate)

    headers = _auth_headers(api_key)
    timeout = max(0.5, _env_float("ALPHARAVIS_CONTEXT_DISCOVERY_TIMEOUT_SECONDS", 2.0))
    cache: dict[str, dict[str, Any]] = {}

    for candidate in candidates:
        try:
            response = httpx.get(candidate.rstrip("/") + "/models", headers=headers, timeout=timeout)
            if response.status_code != 200:
                continue
            payload = response.json()
        except Exception:
            continue

        for model in payload.get("data", []) if isinstance(payload, dict) else []:
            if not isinstance(model, dict):
                continue
            model_id = str(model.get("id") or "")
            if not model_id:
                continue
            entry: dict[str, Any] = {"name": model.get("name", model_id)}
            context_length = _extract_context_length(model)
            if context_length is not None:
                entry["context_length"] = context_length
            _add_model_aliases(cache, model_id, entry)

        is_llamacpp = any(
            isinstance(model, dict) and str(model.get("owned_by") or "").lower() == "llamacpp"
            for model in payload.get("data", []) if isinstance(payload, dict)
        )
        props_context, model_alias = _query_llamacpp_props(candidate, api_key=api_key) if is_llamacpp or not cache else (None, "")
        if props_context:
            if model_alias and model_alias in cache:
                cache[model_alias]["context_length"] = props_context
            elif len(cache) == 1:
                next(iter(cache.values()))["context_length"] = props_context
            elif model_alias:
                _add_model_aliases(cache, model_alias, {"name": model_alias, "context_length": props_context})

        if cache:
            _ENDPOINT_MODEL_METADATA_CACHE[normalized] = cache
            _ENDPOINT_MODEL_METADATA_CACHE_TIME[normalized] = time.time()
            return cache

    props_context, model_alias = _query_llamacpp_props(normalized, api_key=api_key)
    if props_context:
        alias = model_alias or "default"
        cache = {alias: {"name": alias, "context_length": props_context}}

    _ENDPOINT_MODEL_METADATA_CACHE[normalized] = cache
    _ENDPOINT_MODEL_METADATA_CACHE_TIME[normalized] = time.time()
    return cache


def resolve_endpoint_context_length(model: str, base_url: str, api_key: str = "") -> int | None:
    metadata = fetch_endpoint_model_metadata(base_url, api_key=api_key)
    if not metadata:
        return None
    matched = metadata.get(model)
    if not matched and "/" in model:
        matched = metadata.get(model.split("/", 1)[1])
    if not matched:
        if len(metadata) == 1:
            matched = next(iter(metadata.values()))
        else:
            model_lower = model.lower()
            for key, entry in metadata.items():
                key_lower = key.lower()
                if model_lower in key_lower or key_lower in model_lower:
                    matched = entry
                    break
    context_length = matched.get("context_length") if isinstance(matched, dict) else None
    return context_length if isinstance(context_length, int) and context_length > 0 else None


def context_limit_from_ratio(
    context_length: int,
    ratio: float,
    *,
    minimum: int = MINIMUM_CONTEXT_LENGTH,
) -> int:
    ratio = max(0.01, min(float(ratio), 1.0))
    return max(int(minimum), int(context_length * ratio))


def get_model_context_length(
    model: str | None = None,
    provider: str | None = None,
    default: int | None = None,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    config_context_length: int | None = None,
) -> int:
    if config_context_length is not None and config_context_length > 0:
        return max(MINIMUM_CONTEXT_LENGTH, int(config_context_length))

    if model:
        model_key = _sanitize_model_env_key(model)
        specific = _env_int(f"ALPHARAVIS_CONTEXT_LENGTH_{model_key}", 0)
        if specific > 0:
            return max(MINIMUM_CONTEXT_LENGTH, specific)

    if base_url and _env_bool("ALPHARAVIS_AUTO_DISCOVER_CONTEXT_LENGTH", "true"):
        discovered = resolve_endpoint_context_length(model or "", base_url, api_key=api_key or "")
        if discovered:
            return max(MINIMUM_CONTEXT_LENGTH, discovered)

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
