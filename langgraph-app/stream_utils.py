"""Shared streaming utilities for bridge and ACP adapter."""

import json
import uuid
from typing import Any


def get_value(obj: Any, key: str, default: Any = None) -> Any:
    """Read *key* from a dict or object, returning *default* when missing."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def stream_event_name(part: Any) -> str:
    """Return the SSE 'event' field from a stream part."""
    if isinstance(part, dict):
        return str(part.get("event") or "")
    return str(getattr(part, "event", ""))


def stream_event_data(part: Any) -> Any:
    """Return the SSE 'data' payload from a stream part."""
    if isinstance(part, dict):
        return part.get("data")
    return getattr(part, "data", None)


def delta_text(text: str, emitted: str) -> str:
    """Return the portion of *text* that has not yet been emitted."""
    if not text:
        return ""
    if emitted and text.startswith(emitted):
        return text[len(emitted):]
    return text


def extract_tool_calls_from_message(message: Any) -> list[dict[str, Any]]:
    """Pull tool-call dicts from a LangChain-style message (dict or object)."""
    raw = get_value(message, "tool_calls", None)
    if raw is None:
        additional = get_value(message, "additional_kwargs", {})
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
