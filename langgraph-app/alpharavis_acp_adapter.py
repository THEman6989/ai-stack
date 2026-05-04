from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

try:
    from langgraph_sdk import get_client
except Exception:  # pragma: no cover - optional in pure unit tests
    get_client = None  # type: ignore[assignment]

try:
    from internal_context import StreamingInternalContextScrubber, sanitize_internal_context
except Exception:  # pragma: no cover - adapter can still run in stripped envs
    StreamingInternalContextScrubber = None  # type: ignore[assignment]

    def sanitize_internal_context(text: str) -> str:
        return text

try:
    from operational_logging import log_event, log_exception, redact_for_logs
except Exception:  # pragma: no cover - keep ACP stdout pure even if logging import fails

    def log_event(*_args: Any, **_kwargs: Any) -> None:
        return None

    def log_exception(*_args: Any, **_kwargs: Any) -> None:
        return None

    def redact_for_logs(value: Any, *, max_field_chars: int = 4000) -> Any:
        if isinstance(value, str) and len(value) > max_field_chars:
            return value[:max_field_chars] + f"... [truncated {len(value) - max_field_chars} chars]"
        return value


JSONRPC_VERSION = "2.0"
ADAPTER_VERSION = "0.1.0"
DEFAULT_LANGGRAPH_API_URL = "http://langgraph-api:2024"
DEFAULT_ASSISTANT_ID = "alpha_ravis"
DEFAULT_WORKSPACE = "/workspace"

SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|password|passwd|secret|authorization|cookie|credential)\s*[:=]\s*([^\s,;]+)"
)
BEARER_RE = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+")
OPENAI_KEY_RE = re.compile(r"\bsk-[a-zA-Z0-9][a-zA-Z0-9._-]{12,}\b")
PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL,
)


JsonWriter = Callable[[dict[str, Any]], Awaitable[None] | None]
LangGraphStreamFactory = Callable[["AcpSessionState", str, dict[str, Any] | None], AsyncIterator[Any]]


@dataclass
class AcpSessionState:
    session_id: str
    thread_id: str
    cwd: str
    created_at: float = field(default_factory=time.time)
    cancelled: bool = False
    last_prompt: str = ""
    active_run_id: str | None = None


class JsonRpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


def env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int, *, minimum: int = 0, maximum: int = 1_000_000) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except Exception:
        return default
    return max(minimum, min(maximum, value))


def env_float(name: str, default: float, *, minimum: float = 0.0, maximum: float = 1_000_000.0) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except Exception:
        return default
    return max(minimum, min(maximum, value))


def config() -> dict[str, Any]:
    trace_detail = os.getenv("ALPHARAVIS_ACP_TRACE_DETAIL", "summary").strip().lower()
    if trace_detail not in {"summary", "debug", "off"}:
        trace_detail = "summary"
    return {
        "langgraph_api_url": os.getenv("LANGGRAPH_API_URL", DEFAULT_LANGGRAPH_API_URL).strip()
        or DEFAULT_LANGGRAPH_API_URL,
        "assistant_id": os.getenv("LANGGRAPH_ASSISTANT_ID", DEFAULT_ASSISTANT_ID).strip()
        or DEFAULT_ASSISTANT_ID,
        "workspace": os.getenv("ALPHARAVIS_ACP_WORKSPACE", DEFAULT_WORKSPACE).strip()
        or DEFAULT_WORKSPACE,
        "tool_output_max_chars": env_int("ALPHARAVIS_ACP_TOOL_OUTPUT_MAX_CHARS", 8000, minimum=500),
        "trace_detail": trace_detail,
        "scrub_internal_context": env_bool("ALPHARAVIS_ACP_SCRUB_INTERNAL_CONTEXT", "true"),
        "run_timeout_seconds": env_int("ALPHARAVIS_ACP_RUN_TIMEOUT_SECONDS", 300, minimum=5),
        "debug_io": env_bool("ALPHARAVIS_ACP_DEBUG_IO", "false"),
        "allow_file_writes": env_bool("ALPHARAVIS_ACP_ALLOW_FILE_WRITES", "false"),
        "send_available_commands": env_bool("ALPHARAVIS_ACP_SEND_AVAILABLE_COMMANDS", "true"),
        "stream_heartbeat_seconds": env_float("ALPHARAVIS_ACP_STREAM_HEARTBEAT_SECONDS", 5.0, minimum=0.25),
        "debug_event_payload_chars": env_int("ALPHARAVIS_ACP_DEBUG_EVENT_PAYLOAD_CHARS", 12000, minimum=1000),
        "debug_status_to_aion": env_bool("ALPHARAVIS_ACP_DEBUG_STATUS_TO_AION", "true"),
    }


def stable_thread_id(session_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"alpharavis:aionui-acp:{session_id}"))


def scrub_text(text: Any, *, max_chars: int | None = None) -> str:
    value = "" if text is None else str(text)
    value = sanitize_internal_context(value)
    value = PRIVATE_KEY_RE.sub("[redacted private key]", value)
    value = SECRET_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}=[redacted]", value)
    value = BEARER_RE.sub("Bearer [redacted]", value)
    value = OPENAI_KEY_RE.sub("sk-[redacted]", value)
    if max_chars is not None and len(value) > max_chars:
        omitted = len(value) - max_chars
        value = value[:max_chars] + f"\n[output truncated: original_length={len(str(text))}, omitted_chars={omitted}]"
    return value


def scrub_json(value: Any, *, max_field_chars: int = 4000) -> Any:
    return redact_for_logs(value, max_field_chars=max_field_chars)


def debug_io(direction: str, payload: dict[str, Any], *, enabled: bool) -> None:
    if not enabled:
        return
    safe_payload = scrub_json(payload, max_field_chars=2000)
    print(
        f"[alpharavis-acp:{direction}] "
        f"{json.dumps(safe_payload, ensure_ascii=False, default=str, separators=(',', ':'))}",
        file=sys.stderr,
        flush=True,
    )


def debug_trace(event: str, *, enabled: bool, max_chars: int = 12000, **fields: Any) -> None:
    if not enabled:
        return
    payload = {
        "event": event,
        "ts": time.time(),
        **fields,
    }
    safe_payload = scrub_json(payload, max_field_chars=max_chars)
    print(
        f"[alpharavis-acp:trace] "
        f"{json.dumps(safe_payload, ensure_ascii=False, default=str, separators=(',', ':'))}",
        file=sys.stderr,
        flush=True,
    )


def _message_type(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("type") or message.get("role") or "").lower()
    msg_type = getattr(message, "type", None) or getattr(message, "role", None)
    if msg_type:
        return str(msg_type).lower()
    return type(message).__name__.lower()


def _is_ai_message(message: Any) -> bool:
    message_type = _message_type(message)
    return message_type in {"ai", "assistant"} or "aimessage" in message_type or "aichunk" in message_type


def _is_tool_message(message: Any) -> bool:
    message_type = _message_type(message)
    return message_type in {"tool", "toolmessage"} or "toolmessage" in message_type


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _message_content(message: Any, *, include_reasoning: bool = False) -> str:
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")

    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            if not isinstance(part, dict):
                parts.append(str(part))
                continue
            block_type = str(part.get("type") or "")
            if block_type in {"thinking", "reasoning"} and not include_reasoning:
                continue
            if isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(part.get("content"), str):
                parts.append(part["content"])
        return "".join(parts)
    return str(content or "")


def _message_reasoning_content(message: Any) -> str:
    candidates: list[Any]
    if isinstance(message, dict):
        additional = message.get("additional_kwargs") if isinstance(message.get("additional_kwargs"), dict) else {}
        candidates = [message.get("reasoning_content"), message.get("reasoning"), additional.get("reasoning_content")]
    else:
        additional = getattr(message, "additional_kwargs", {}) or {}
        candidates = [
            getattr(message, "reasoning_content", None),
            getattr(message, "reasoning", None),
            additional.get("reasoning_content") if isinstance(additional, dict) else None,
        ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return _message_content(message, include_reasoning=True) if _message_content(message, include_reasoning=True) else ""


def _stream_event_name(part: Any) -> str:
    if isinstance(part, dict):
        return str(part.get("event") or "")
    return str(getattr(part, "event", ""))


def _stream_event_data(part: Any) -> Any:
    if isinstance(part, dict):
        return part.get("data")
    return getattr(part, "data", None)


def _extract_run_id(part: Any) -> str:
    for key in ("run_id", "runId", "id"):
        value = _get_value(part, key, "")
        if value:
            return str(value)
    data = _stream_event_data(part)
    if isinstance(data, dict):
        for key in ("run_id", "runId", "id"):
            if data.get(key):
                return str(data[key])
    return ""


def _extract_stream_text(part: Any) -> str:
    data = _stream_event_data(part)
    if isinstance(data, tuple) and data:
        return _message_content(data[0]) if _is_ai_message(data[0]) else ""
    if isinstance(data, list) and data:
        for message in reversed(data):
            if _is_ai_message(message):
                return _message_content(message)
        return ""
    if isinstance(data, dict):
        if "chunk" in data:
            chunk = data["chunk"]
            return _message_content(chunk) if _is_ai_message(chunk) else ""
        messages = data.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                if _is_ai_message(message):
                    return _message_content(message)
        if "content" in data and data.get("type") in {"ai", "assistant"}:
            return str(data.get("content") or "")
    return ""


def _extract_stream_reasoning(part: Any) -> str:
    data = _stream_event_data(part)
    if isinstance(data, tuple) and data:
        return _message_reasoning_content(data[0]) if _is_ai_message(data[0]) else ""
    if isinstance(data, list) and data:
        for message in reversed(data):
            if _is_ai_message(message):
                return _message_reasoning_content(message)
        return ""
    if isinstance(data, dict):
        if "chunk" in data:
            chunk = data["chunk"]
            return _message_reasoning_content(chunk) if _is_ai_message(chunk) else ""
        messages = data.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                if _is_ai_message(message):
                    return _message_reasoning_content(message)
    return ""


def _delta_text(text: str, emitted: str) -> str:
    if not text:
        return ""
    if emitted and text.startswith(emitted):
        return text[len(emitted) :]
    return text


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


def _extract_tool_result(data: Any) -> tuple[str, str, str] | None:
    candidate = data
    if isinstance(data, dict) and "message" in data:
        candidate = data["message"]
    if not _is_tool_message(candidate) and not (
        isinstance(candidate, dict) and ("tool_call_id" in candidate or candidate.get("type") == "tool")
    ):
        return None
    tool_call_id = str(_get_value(candidate, "tool_call_id", "") or _get_value(candidate, "id", "") or "")
    if not tool_call_id:
        tool_call_id = f"call_{uuid.uuid4().hex[:12]}"
    name = str(_get_value(candidate, "name", "") or _get_value(candidate, "tool", "") or "tool")
    return tool_call_id, name, _message_content(candidate)


READ_TOOL_RE = re.compile(r"(read|list|search|grep|cat|lookup|find|fetch|open|inspect|scan)", re.IGNORECASE)
EDIT_TOOL_RE = re.compile(
    r"(write|edit|patch|apply_patch|create|delete|rename|move|replace|update_file|insert|append)",
    re.IGNORECASE,
)
EXEC_TOOL_RE = re.compile(
    r"(shell|ssh|docker|pm2|curl|npm|pnpm|yarn|python|bash|powershell|command|execute|run|start|restart|shutdown)",
    re.IGNORECASE,
)
FILE_LOCATION_KEYS = {
    "path",
    "file",
    "filepath",
    "file_path",
    "filename",
    "target",
    "relative_path",
    "files",
    "paths",
}
UNIFIED_DIFF_RE = re.compile(r"(?m)^(diff --git |--- .+\n\+\+\+ .+\n@@ |@@ .+ @@)")
DIFF_FILE_RE = re.compile(r"(?m)^(?:\+\+\+ b/|--- a/|diff --git a/)([^\s]+)")


def is_unified_diff(text: Any) -> bool:
    return isinstance(text, str) and bool(UNIFIED_DIFF_RE.search(text))


def classify_tool_kind(tool_name: str, args: Any = None, output: Any = None) -> str:
    haystack = f"{tool_name} {json.dumps(args, default=str, ensure_ascii=False) if args is not None else ''}"
    if EDIT_TOOL_RE.search(haystack) or is_unified_diff(output):
        return "edit"
    if EXEC_TOOL_RE.search(haystack):
        return "execute"
    if READ_TOOL_RE.search(haystack):
        return "read"
    return "execute"


def _walk_location_values(value: Any) -> list[str]:
    locations: list[str] = []
    if isinstance(value, str):
        if value.strip():
            locations.append(value.strip())
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            locations.extend(_walk_location_values(item))
    return locations


def extract_locations(args: Any = None, output: Any = None) -> list[dict[str, str]]:
    values: list[str] = []
    if isinstance(args, dict):
        for key, value in args.items():
            if str(key) in FILE_LOCATION_KEYS:
                values.extend(_walk_location_values(value))
    if isinstance(output, str) and is_unified_diff(output):
        values.extend(match.group(1) for match in DIFF_FILE_RE.finditer(output))

    seen: set[str] = set()
    locations: list[dict[str, str]] = []
    for value in values:
        path = value.strip().strip("\"'")
        if not path or path in seen:
            continue
        seen.add(path)
        locations.append({"path": path})
    return locations


def build_tool_content_items(text: str, *, max_chars: int, prefer_diff: bool = False) -> list[dict[str, Any]]:
    safe_text = scrub_text(text, max_chars=max_chars)
    if prefer_diff and is_unified_diff(safe_text):
        path = extract_locations(output=safe_text)
        return [
            {
                "type": "diff",
                "path": path[0]["path"] if path else "diff.patch",
                "oldText": None,
                "newText": safe_text,
            }
        ]
    if is_unified_diff(safe_text):
        safe_text = f"```diff\n{safe_text}\n```"
    return [{"type": "content", "content": {"type": "text", "text": safe_text}}]


def _find_command_approval_interrupt(obj: Any) -> dict[str, Any] | None:
    if isinstance(obj, dict):
        if obj.get("type") == "command_approval":
            return obj
        value = obj.get("value")
        if isinstance(value, dict) and value.get("type") == "command_approval":
            return value
        for nested in obj.values():
            found = _find_command_approval_interrupt(nested)
            if found:
                return found
    elif isinstance(obj, (list, tuple)):
        for nested in obj:
            found = _find_command_approval_interrupt(nested)
            if found:
                return found
    else:
        value = getattr(obj, "value", None)
        if isinstance(value, dict) and value.get("type") == "command_approval":
            return value
    return None


def _node_names_from_update(part: Any) -> list[str]:
    if _stream_event_name(part) != "updates":
        return []
    data = _stream_event_data(part)
    if not isinstance(data, dict):
        return []
    return [str(key) for key in data.keys() if not str(key).startswith("__")]


def _plan_entries_from_update(part: Any) -> list[dict[str, str]]:
    if _stream_event_name(part) != "updates":
        return []
    data = _stream_event_data(part)
    if not isinstance(data, dict):
        return []
    candidates: list[Any] = []
    for value in data.values():
        if isinstance(value, dict):
            for key in ("plan", "planner_plan", "run_plan", "tasks"):
                if isinstance(value.get(key), list):
                    candidates = value[key]
                    break
        if candidates:
            break
    entries: list[dict[str, str]] = []
    for item in candidates[:12]:
        if isinstance(item, dict):
            content = str(item.get("content") or item.get("step") or item.get("task") or "").strip()
            status = str(item.get("status") or "pending").strip()
        else:
            content = str(item).strip()
            status = "pending"
        if content:
            entries.append({"content": content[:500], "status": status})
    return entries


def extract_prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if not isinstance(prompt, list):
        return str(prompt or "")
    parts: list[str] = []
    for item in prompt:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        if item_type in {"text", "input_text"} and isinstance(item.get("text"), str):
            parts.append(item["text"])
        elif isinstance(item.get("content"), str):
            parts.append(item["content"])
        elif item_type:
            label = item.get("uri") or item.get("path") or item.get("mimeType") or "attachment"
            parts.append(f"[ACP attachment metadata: type={item_type}, ref={label}]")
    return "\n".join(part for part in parts if part)


def build_agent_message_chunk(session_id: str, text: str) -> dict[str, Any]:
    return {
        "jsonrpc": JSONRPC_VERSION,
        "method": "session/update",
        "params": {
            "sessionId": session_id,
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": text},
            },
        },
    }


def build_agent_thought_chunk(session_id: str, text: str) -> dict[str, Any]:
    return {
        "jsonrpc": JSONRPC_VERSION,
        "method": "session/update",
        "params": {
            "sessionId": session_id,
            "update": {
                "sessionUpdate": "agent_thought_chunk",
                "content": {"type": "text", "text": text},
            },
        },
    }


def build_tool_call(
    session_id: str,
    tool_call_id: str,
    title: str,
    *,
    raw_input: dict[str, Any] | None = None,
    kind: str = "execute",
    status: str = "in_progress",
    content: list[dict[str, Any]] | None = None,
    locations: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    update: dict[str, Any] = {
        "sessionUpdate": "tool_call",
        "toolCallId": tool_call_id,
        "status": status,
        "title": title,
        "kind": kind,
        "rawInput": raw_input or {"tool": title},
    }
    if content:
        update["content"] = content
    if locations:
        update["locations"] = locations
    return {
        "jsonrpc": JSONRPC_VERSION,
        "method": "session/update",
        "params": {"sessionId": session_id, "update": update},
    }


def build_tool_call_update(
    session_id: str,
    tool_call_id: str,
    *,
    status: str,
    text: str,
    raw_input: dict[str, Any] | None = None,
    max_chars: int = 8000,
) -> dict[str, Any]:
    update: dict[str, Any] = {
        "sessionUpdate": "tool_call_update",
        "toolCallId": tool_call_id,
        "status": status,
        "content": build_tool_content_items(text, max_chars=max_chars, prefer_diff=False),
    }
    if raw_input is not None:
        update["rawInput"] = raw_input
    return {
        "jsonrpc": JSONRPC_VERSION,
        "method": "session/update",
        "params": {"sessionId": session_id, "update": update},
    }


def build_plan_update(session_id: str, entries: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "jsonrpc": JSONRPC_VERSION,
        "method": "session/update",
        "params": {
            "sessionId": session_id,
            "update": {"sessionUpdate": "plan", "entries": entries},
        },
    }


def build_usage_update(session_id: str, *, used: int, size: int) -> dict[str, Any]:
    return {
        "jsonrpc": JSONRPC_VERSION,
        "method": "session/update",
        "params": {
            "sessionId": session_id,
            "update": {"sessionUpdate": "usage_update", "used": used, "size": size},
        },
    }


def build_available_commands_update(session_id: str) -> dict[str, Any]:
    commands = [
        ("debug", "Diagnose an error, logs, services, or a failing workflow.", "debug <problem>"),
        ("research", "Research a topic or document context.", "research <topic>"),
        ("fix", "Plan and implement a bounded fix.", "fix <issue>"),
        ("inspect", "Inspect repo, files, memory, or service state.", "inspect <target>"),
        ("run", "Run a safe AlphaRavis workflow.", "run <task>"),
        ("search", "Search memory, RAG, repo, or web context.", "search <query>"),
        ("open", "Open/read a specific file or known source.", "open <path-or-source>"),
        ("reload-context", "Refresh tool, skill, or architecture context.", "reload-context"),
    ]
    return {
        "jsonrpc": JSONRPC_VERSION,
        "method": "session/update",
        "params": {
            "sessionId": session_id,
            "update": {
                "sessionUpdate": "available_commands_update",
                "availableCommands": [
                    {"name": name, "description": description, "input": {"hint": hint}}
                    for name, description, hint in commands
                ],
            },
        },
    }


def build_permission_request(
    request_id: int,
    session_id: str,
    *,
    tool_call_id: str,
    title: str,
    command: str,
    description: str,
    target: str = "",
) -> dict[str, Any]:
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "method": "session/request_permission",
        "params": {
            "sessionId": session_id,
            "options": [
                {"optionId": "allow_once", "name": "Einmal erlauben", "kind": "allow_once"},
                {"optionId": "reject_once", "name": "Ablehnen", "kind": "reject_once"},
            ],
            "toolCall": {
                "toolCallId": tool_call_id,
                "title": title,
                "kind": "execute",
                "rawInput": {
                    "command": command,
                    "description": description,
                    "target": target,
                },
            },
        },
    }


def permission_result_to_resume(result: Any) -> dict[str, Any]:
    option_id = ""
    replacement = ""
    if isinstance(result, dict):
        outcome = result.get("outcome")
        if isinstance(outcome, dict):
            option_id = str(outcome.get("optionId") or "")
            replacement = str(outcome.get("command") or outcome.get("replacement") or "")
        option_id = option_id or str(result.get("optionId") or "")
        replacement = replacement or str(result.get("command") or result.get("replacement") or "")
    option_id = option_id.lower()
    if "allow" in option_id or "approve" in option_id:
        return {"action": "approve"}
    if "replace" in option_id and replacement.strip():
        return {"action": "replace", "command": replacement.strip()}
    return {"action": "reject"}


SENSITIVE_PATH_RE = re.compile(
    r"(^|[\\/])(\.env($|[.\\/])|\.ssh($|[\\/])|id_rsa$|id_ed25519$|.*private.*key.*|.*token.*|.*secret.*)",
    re.IGNORECASE,
)


def resolve_workspace_path(workspace: str, requested_path: str) -> Path:
    root = Path(workspace).expanduser().resolve()
    candidate = Path(requested_path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise JsonRpcError(-32040, f"Path outside ALPHARAVIS_ACP_WORKSPACE is blocked: {requested_path}") from exc
    return resolved


def ensure_file_read_allowed(path: Path) -> None:
    normalized = str(path).replace("\\", "/")
    if SENSITIVE_PATH_RE.search(normalized):
        raise JsonRpcError(-32041, f"Sensitive file read is blocked: {path.name}")


def ensure_file_write_allowed(path: Path, *, allow_writes: bool) -> None:
    if not allow_writes:
        raise JsonRpcError(-32042, "fs/write_text_file is disabled. Set ALPHARAVIS_ACP_ALLOW_FILE_WRITES=true to enable.")
    normalized = str(path).replace("\\", "/")
    if SENSITIVE_PATH_RE.search(normalized):
        raise JsonRpcError(-32043, f"Sensitive file write is blocked: {path.name}")


class AlphaRavisAcpAdapter:
    def __init__(
        self,
        *,
        writer: JsonWriter | None = None,
        langgraph_stream: LangGraphStreamFactory | None = None,
        adapter_config: dict[str, Any] | None = None,
    ) -> None:
        self.config = adapter_config or config()
        self.sessions: dict[str, AcpSessionState] = {}
        self._writer = writer
        self._write_lock = asyncio.Lock()
        self._next_request_id = 10_000
        self._pending_responses: dict[int, asyncio.Future[Any]] = {}
        self._langgraph_stream = langgraph_stream
        self._active_prompt_tasks: dict[str, asyncio.Task[Any]] = {}

    def _debug_enabled(self) -> bool:
        return bool(self.config.get("debug_io")) or str(self.config.get("trace_detail")) == "debug"

    def _trace(self, event: str, **fields: Any) -> None:
        debug_trace(
            event,
            enabled=self._debug_enabled(),
            max_chars=int(self.config.get("debug_event_payload_chars") or 12000),
            **fields,
        )

    async def _visible_debug(self, session_id: str, text: str) -> None:
        if self.config.get("trace_detail") == "debug" and self.config.get("debug_status_to_aion"):
            await self.send_json(build_agent_thought_chunk(session_id, text))

    async def send_json(self, payload: dict[str, Any]) -> None:
        async with self._write_lock:
            started = time.perf_counter()
            debug_io("out", payload, enabled=bool(self.config.get("debug_io")))
            if self._writer is not None:
                result = self._writer(payload)
                if asyncio.iscoroutine(result):
                    await result
                self._trace(
                    "jsonrpc.out.written",
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                    method=payload.get("method"),
                    id=payload.get("id"),
                    bytes=len(json.dumps(payload, ensure_ascii=False, default=str)),
                    writer="test",
                )
                return
            sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            sys.stdout.flush()
            self._trace(
                "jsonrpc.out.written",
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                method=payload.get("method"),
                id=payload.get("id"),
                bytes=len(json.dumps(payload, ensure_ascii=False, default=str)),
                writer="stdout",
            )

    async def send_response(self, request_id: int | str, result: Any = None) -> None:
        await self.send_json({"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result})

    async def send_error(self, request_id: int | str | None, code: int, message: str) -> None:
        payload: dict[str, Any] = {
            "jsonrpc": JSONRPC_VERSION,
            "error": {"code": code, "message": message},
        }
        if request_id is not None:
            payload["id"] = request_id
        await self.send_json(payload)

    async def handle_message(self, message: dict[str, Any]) -> None:
        received_at = time.perf_counter()
        debug_io("in", message, enabled=bool(self.config.get("debug_io")))
        self._trace(
            "jsonrpc.in.received",
            method=message.get("method"),
            id=message.get("id"),
            bytes=len(json.dumps(message, ensure_ascii=False, default=str)),
        )
        if "method" in message:
            task = asyncio.create_task(self._handle_request_or_notification(message))
            self._trace(
                "jsonrpc.in.dispatched",
                method=message.get("method"),
                id=message.get("id"),
                dispatch_elapsed_ms=round((time.perf_counter() - received_at) * 1000, 2),
            )
            return
        request_id = message.get("id")
        if isinstance(request_id, int) and request_id in self._pending_responses:
            future = self._pending_responses.pop(request_id)
            if "error" in message:
                future.set_exception(RuntimeError(str(message["error"])))
            else:
                future.set_result(message.get("result"))
            self._trace("jsonrpc.response.matched_pending", id=request_id)

    async def _handle_request_or_notification(self, message: dict[str, Any]) -> None:
        started = time.perf_counter()
        request_id = message.get("id")
        method = str(message.get("method") or "")
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        session_started_id: str | None = None
        self._trace("request.start", method=method, id=request_id, params=params)
        try:
            if method == "initialize":
                result = self.initialize(params)
            elif method == "session/new":
                result = self.new_session(params)
                session_started_id = str(result.get("sessionId") or "")
            elif method in {"session/prompt", "session/send_message"}:
                result = await self._run_prompt_request(params)
            elif method == "session/cancel":
                result = await self.cancel(params)
            elif method == "session/close":
                result = self.close(params)
            elif method == "session/load":
                result = self.load_session(params)
                session_started_id = str(result.get("sessionId") or "")
            elif method == "session/set_config_option":
                result = {}
            elif method in {"session/response", "session/permission_response"}:
                result = self.permission_response(params)
            elif method == "fs/read_text_file":
                result = self.read_text_file(params)
            elif method == "fs/write_text_file":
                result = self.write_text_file(params)
            else:
                if request_id is not None:
                    await self.send_error(request_id, -32601, f"Unsupported ACP method: {method}")
                return
            if request_id is not None:
                await self.send_response(request_id, result)
            if session_started_id:
                await self.emit_session_started_updates(session_started_id)
            self._trace(
                "request.end",
                method=method,
                id=request_id,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        except JsonRpcError as exc:
            if request_id is not None:
                await self.send_error(request_id, exc.code, exc.message)
            self._trace(
                "request.error",
                method=method,
                id=request_id,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                error=exc.message,
                code=exc.code,
            )
        except asyncio.CancelledError:
            self._trace(
                "request.cancelled",
                method=method,
                id=request_id,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            if request_id is not None and method in {"session/prompt", "session/send_message"}:
                await self.send_response(request_id, {"stopReason": "cancelled"})
        except Exception as exc:
            log_exception("acp_adapter.request.failed", exc, component="alpharavis-acp", method=method)
            if request_id is not None:
                await self.send_error(request_id, -32603, str(exc))
            self._trace(
                "request.failed",
                method=method,
                id=request_id,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                error=str(exc),
            )

    def initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        protocol_version = int(params.get("protocolVersion") or 1)
        return {
            "protocolVersion": protocol_version,
            "serverInfo": {
                "name": "alpharavis-langgraph",
                "version": ADAPTER_VERSION,
                "title": "AlphaRavis LangGraph",
            },
            "agentInfo": {
                "name": "alpharavis-langgraph",
                "version": ADAPTER_VERSION,
                "title": "AlphaRavis LangGraph",
            },
            "agentCapabilities": {
                "loadSession": True,
                "promptCapabilities": {"image": False, "audio": False, "embeddedContext": True},
                "sessionCapabilities": {"close": {}, "list": {}, "resume": {}},
                "_meta": {
                    "adapter": "alpharavis_acp_adapter",
                    "langgraphApiUrl": self.config["langgraph_api_url"],
                    "assistantId": self.config["assistant_id"],
                },
            },
            "authMethods": [],
            "modes": {
                "currentModeId": "agent",
                "availableModes": [
                    {
                        "id": "agent",
                        "name": "Agent",
                        "description": "AlphaRavis LangGraph multi-agent workflow",
                    }
                ],
            },
        }

    def _session_models(self) -> dict[str, Any]:
        model_id = str(self.config["assistant_id"])
        return {
            "currentModelId": model_id,
            "availableModels": [{"id": model_id, "name": "AlphaRavis / LangGraph"}],
        }

    def _session_modes(self) -> dict[str, Any]:
        return {
            "currentModeId": "agent",
            "availableModes": [{"id": "agent", "name": "Agent", "description": "AlphaRavis LangGraph"}],
        }

    def new_session(self, params: dict[str, Any]) -> dict[str, Any]:
        requested_session = str(params.get("sessionId") or params.get("resumeSessionId") or "").strip()
        session_id = requested_session or f"alpharavis-acp-{uuid.uuid4().hex}"
        cwd = str(params.get("cwd") or self.config["workspace"])
        state = AcpSessionState(session_id=session_id, thread_id=stable_thread_id(session_id), cwd=cwd)
        self.sessions[session_id] = state
        log_event(
            "INFO",
            "acp_adapter.session.created",
            component="alpharavis-acp",
            session_id=session_id,
            thread_id=state.thread_id,
            cwd=cwd,
        )
        return {
            "sessionId": session_id,
            "models": self._session_models(),
            "modes": self._session_modes(),
            "configOptions": [
                {
                    "id": "trace_detail",
                    "name": "Trace Detail",
                    "type": "select",
                    "value": self.config["trace_detail"],
                    "options": [
                        {"value": "summary", "label": "summary"},
                        {"value": "debug", "label": "debug"},
                        {"value": "off", "label": "off"},
                    ],
                }
            ],
        }

    async def emit_session_started_updates(self, session_id: str) -> None:
        if self.config.get("send_available_commands"):
            await self.send_json(build_available_commands_update(session_id))

    def load_session(self, params: dict[str, Any]) -> dict[str, Any]:
        return self.new_session(params)

    def close(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = str(params.get("sessionId") or "")
        self.sessions.pop(session_id, None)
        return {}

    async def cancel(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = str(params.get("sessionId") or "")
        session = self.sessions.get(session_id)
        if session is None:
            return {"cancelled": False, "reason": "unknown_session"}
        session.cancelled = True
        cancelled_remote = await self._cancel_langgraph_run(session)
        return {
            "cancelled": True,
            "langgraphRunCancel": "attempted" if cancelled_remote else "local_only",
        }

    def permission_response(self, params: dict[str, Any]) -> dict[str, Any]:
        request_id = params.get("requestId") or params.get("id")
        if not isinstance(request_id, int):
            pending_ids = list(self._pending_responses.keys())
            request_id = pending_ids[-1] if pending_ids else None
        if isinstance(request_id, int) and request_id in self._pending_responses:
            self._pending_responses.pop(request_id).set_result(params)
        return {}

    def read_text_file(self, params: dict[str, Any]) -> dict[str, Any]:
        path = resolve_workspace_path(str(self.config["workspace"]), str(params.get("path") or ""))
        ensure_file_read_allowed(path)
        if not path.is_file():
            raise JsonRpcError(-32044, f"File not found: {path}")
        return {"content": scrub_text(path.read_text(encoding="utf-8", errors="replace"))}

    def write_text_file(self, params: dict[str, Any]) -> dict[str, Any]:
        path = resolve_workspace_path(str(self.config["workspace"]), str(params.get("path") or ""))
        ensure_file_write_allowed(path, allow_writes=bool(self.config.get("allow_file_writes")))
        content = str(params.get("content") or "")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {}

    async def _run_prompt_request(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = str(params.get("sessionId") or "")
        existing = self._active_prompt_tasks.get(session_id)
        if existing and not existing.done():
            self._trace("prompt.interrupt_previous.start", session_id=session_id)
            session = self.sessions.get(session_id)
            if session is not None:
                session.cancelled = True
                await self._cancel_langgraph_run(session)
            existing.cancel()
            self._trace("prompt.interrupt_previous.done", session_id=session_id)

        current = asyncio.current_task()
        if current is not None and session_id:
            self._active_prompt_tasks[session_id] = current
        try:
            return await self.prompt(params)
        finally:
            if session_id and self._active_prompt_tasks.get(session_id) is current:
                self._active_prompt_tasks.pop(session_id, None)

    async def prompt(self, params: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        session_id = str(params.get("sessionId") or "")
        if not session_id:
            raise ValueError("sessionId is required")
        session = self.sessions.get(session_id)
        if session is None:
            session = AcpSessionState(session_id=session_id, thread_id=stable_thread_id(session_id), cwd=self.config["workspace"])
            self.sessions[session_id] = session
        session.cancelled = False
        session.active_run_id = None
        prompt = extract_prompt_text(params.get("prompt") or params.get("message") or params.get("input"))
        session.last_prompt = prompt
        self._trace(
            "prompt.start",
            session_id=session_id,
            thread_id=session.thread_id,
            prompt_chars=len(prompt),
            prompt_preview=prompt[:500],
        )
        if not prompt.strip():
            self._trace("prompt.empty", session_id=session_id)
            return {"stopReason": "end_turn"}

        await self.send_json(
            build_agent_thought_chunk(
                session_id,
                "AlphaRavis verbindet AionUi mit dem LangGraph-Run.",
            )
        )
        try:
            await self._stream_prompt(session, prompt)
        except Exception as exc:
            log_exception(
                "acp_adapter.langgraph.failed",
                exc,
                component="alpharavis-acp",
                session_id=session.session_id,
                thread_id=session.thread_id,
            )
            message = scrub_text(f"AlphaRavis-ACP konnte LangGraph nicht erreichen: {exc}", max_chars=2000)
            await self.send_json(build_agent_message_chunk(session.session_id, message))
        result = {"stopReason": "cancelled" if session.cancelled else "end_turn"}
        self._trace(
            "prompt.end",
            session_id=session_id,
            thread_id=session.thread_id,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            stop_reason=result["stopReason"],
        )
        return result

    async def _stream_prompt(self, session: AcpSessionState, prompt: str) -> None:
        command: dict[str, Any] | None = None
        for attempt in range(4):
            if session.cancelled:
                return
            self._trace(
                "prompt.run_attempt.start",
                session_id=session.session_id,
                thread_id=session.thread_id,
                attempt=attempt + 1,
                command=command,
            )
            interrupt_value = await self._stream_one_run(session, prompt, command)
            if not interrupt_value:
                self._trace(
                    "prompt.run_attempt.end",
                    session_id=session.session_id,
                    thread_id=session.thread_id,
                    attempt=attempt + 1,
                    interrupted=False,
                )
                return
            self._trace(
                "prompt.run_attempt.interrupted",
                session_id=session.session_id,
                thread_id=session.thread_id,
                attempt=attempt + 1,
                interrupt=interrupt_value,
            )
            resume = await self._handle_permission_interrupt(session, interrupt_value)
            command = {"resume": resume}
        await self.send_json(
            build_agent_message_chunk(
                session.session_id,
                "\n\nAlphaRavis hat mehrere Freigabe-Unterbrechungen erreicht. Bitte starte die Anfrage erneut, falls noch etwas offen ist.",
            )
        )

    async def _stream_one_run(
        self,
        session: AcpSessionState,
        prompt: str,
        command: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        started = time.perf_counter()
        last_event_at = started
        event_count = 0
        token_count = 0
        emitted_text = ""
        emitted_reasoning = ""
        seen_tool_calls: set[str] = set()
        seen_tool_updates: set[str] = set()
        tool_inputs: dict[str, dict[str, Any]] = {}
        content_scrubber = StreamingInternalContextScrubber() if self.config["scrub_internal_context"] and StreamingInternalContextScrubber else None
        thought_scrubber = StreamingInternalContextScrubber() if self.config["scrub_internal_context"] and StreamingInternalContextScrubber else None

        stream = self._langgraph_stream(session, prompt, command) if self._langgraph_stream else self._default_langgraph_stream(session, prompt, command)
        async with asyncio.timeout(float(self.config["run_timeout_seconds"])):
            iterator = stream.__aiter__()
            heartbeat_seconds = float(self.config.get("stream_heartbeat_seconds") or 5.0)
            next_event_task = asyncio.create_task(iterator.__anext__())
            try:
                while True:
                    done, _pending = await asyncio.wait({next_event_task}, timeout=heartbeat_seconds)
                    if not done:
                        waited = time.perf_counter() - last_event_at
                        total = time.perf_counter() - started
                        self._trace(
                            "langgraph.stream.waiting",
                            session_id=session.session_id,
                            thread_id=session.thread_id,
                            waited_since_last_event_seconds=round(waited, 3),
                            total_wait_seconds=round(total, 3),
                            emitted_text_chars=len(emitted_text),
                            emitted_reasoning_chars=len(emitted_reasoning),
                        )
                        await self._visible_debug(
                            session.session_id,
                            (
                                "ACP wartet weiter auf LangGraph/LLM-Stream "
                                f"({round(total, 1)}s seit Run-Start, {round(waited, 1)}s seit letztem Event)."
                            ),
                        )
                        continue

                    try:
                        part = next_event_task.result()
                    except StopAsyncIteration:
                        self._trace(
                            "langgraph.stream.completed",
                            session_id=session.session_id,
                            thread_id=session.thread_id,
                            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                            event_count=event_count,
                            emitted_text_chars=len(emitted_text),
                            emitted_reasoning_chars=len(emitted_reasoning),
                        )
                        break
                    next_event_task = asyncio.create_task(iterator.__anext__())

                    event_count += 1
                    now = time.perf_counter()
                    gap_ms = round((now - last_event_at) * 1000, 2)
                    last_event_at = now
                    run_id = _extract_run_id(part)
                    if run_id:
                        session.active_run_id = run_id
                    self._trace(
                        "langgraph.stream.event",
                        session_id=session.session_id,
                        thread_id=session.thread_id,
                        run_id=run_id or session.active_run_id,
                        event_index=event_count,
                        stream_event=_stream_event_name(part),
                        gap_ms=gap_ms,
                        data_summary=self._summarize_stream_part(part),
                        raw_part=part,
                    )
                    if session.cancelled:
                        self._trace("langgraph.stream.cancelled_local", session_id=session.session_id)
                        return None
                    interrupt_value = _find_command_approval_interrupt(part)
                    if interrupt_value:
                        self._trace(
                            "langgraph.stream.interrupt",
                            session_id=session.session_id,
                            thread_id=session.thread_id,
                            interrupt=interrupt_value,
                        )
                        return interrupt_value

                    await self._emit_activity_or_plan(session.session_id, part)

                    for notification in self._tool_notifications_from_part(
                        session.session_id,
                        part,
                        seen_tool_calls,
                        seen_tool_updates,
                        tool_inputs,
                    ):
                        await self.send_json(notification)

                    reasoning = _extract_stream_reasoning(part)
                    reasoning_delta = _delta_text(reasoning, emitted_reasoning)
                    if reasoning_delta and self.config["trace_detail"] == "debug":
                        emitted_reasoning += reasoning_delta
                        clean = thought_scrubber.feed(reasoning_delta) if thought_scrubber else reasoning_delta
                        clean = scrub_text(clean, max_chars=1200)
                        if clean.strip():
                            await self.send_json(build_agent_thought_chunk(session.session_id, clean))

                    text = _extract_stream_text(part)
                    delta = _delta_text(text, emitted_text)
                    if delta:
                        token_count += 1
                        if not emitted_text:
                            self._trace(
                                "langgraph.stream.first_text_delta",
                                session_id=session.session_id,
                                thread_id=session.thread_id,
                                first_text_elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                                event_index=event_count,
                            )
                        emitted_text += delta
                        visible = content_scrubber.feed(delta) if content_scrubber else delta
                        visible = scrub_text(visible)
                        if visible:
                            await self.send_json(build_agent_message_chunk(session.session_id, visible))
            finally:
                if not next_event_task.done():
                    next_event_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await next_event_task

        if content_scrubber:
            tail = scrub_text(content_scrubber.flush())
            if tail:
                await self.send_json(build_agent_message_chunk(session.session_id, tail))
        if thought_scrubber and self.config["trace_detail"] == "debug":
            tail = scrub_text(thought_scrubber.flush(), max_chars=1200)
            if tail:
                await self.send_json(build_agent_thought_chunk(session.session_id, tail))
        self._trace(
            "stream_one_run.end",
            session_id=session.session_id,
            thread_id=session.thread_id,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            event_count=event_count,
            text_delta_count=token_count,
            emitted_text_chars=len(emitted_text),
        )
        return None

    def _summarize_stream_part(self, part: Any) -> dict[str, Any]:
        data = _stream_event_data(part)
        summary: dict[str, Any] = {
            "event": _stream_event_name(part),
            "run_id": _extract_run_id(part),
            "data_type": type(data).__name__,
        }
        if isinstance(data, dict):
            summary["data_keys"] = list(data.keys())[:20]
            summary["node_names"] = [str(key) for key in data.keys() if not str(key).startswith("__")][:10]
            if "chunk" in data:
                summary["chunk_type"] = type(data["chunk"]).__name__
                summary["chunk_text_chars"] = len(_message_content(data["chunk"]))
            if "messages" in data and isinstance(data["messages"], list):
                summary["messages_count"] = len(data["messages"])
        elif isinstance(data, tuple) and data:
            summary["tuple_len"] = len(data)
            summary["message_type"] = _message_type(data[0])
            summary["message_text_chars"] = len(_message_content(data[0]))
        elif isinstance(data, list):
            summary["list_len"] = len(data)
        return summary

    async def _emit_activity_or_plan(self, session_id: str, part: Any) -> None:
        entries = _plan_entries_from_update(part)
        if entries:
            await self.send_json(build_plan_update(session_id, entries))
            return
        if self.config["trace_detail"] == "off":
            return
        node_names = _node_names_from_update(part)
        if node_names:
            joined = ", ".join(node_names[:4])
            await self.send_json(build_agent_thought_chunk(session_id, f"LangGraph-Schritt abgeschlossen: {joined}."))
        elif self.config["trace_detail"] == "debug":
            event = _stream_event_name(part)
            if event and event not in {"messages", "metadata"}:
                await self.send_json(build_agent_thought_chunk(session_id, f"LangGraph-Event: {event}." ))

    def _tool_notifications_from_part(
        self,
        session_id: str,
        part: Any,
        seen_tool_calls: set[str],
        seen_tool_updates: set[str],
        tool_inputs: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        notifications: list[dict[str, Any]] = []
        event = _stream_event_name(part)
        data = _stream_event_data(part)

        if event in {"on_tool_start", "tool_start"} and isinstance(data, dict):
            raw_input = data.get("input") if isinstance(data.get("input"), dict) else {}
            tool_name = str(data.get("name") or data.get("tool") or data.get("run_name") or "tool")
            call_id = str(data.get("run_id") or data.get("tool_call_id") or f"call_{uuid.uuid4().hex[:12]}")
            if call_id not in seen_tool_calls:
                self._trace(
                    "tool.start",
                    session_id=session_id,
                    tool_call_id=call_id,
                    tool_name=tool_name,
                    raw_input=raw_input,
                )
                seen_tool_calls.add(call_id)
                raw = {"tool": tool_name, "args": raw_input}
                tool_inputs[call_id] = raw
                notifications.append(
                    build_tool_call(
                        session_id,
                        call_id,
                        tool_name,
                        raw_input=raw,
                        kind=classify_tool_kind(tool_name, raw_input),
                        locations=extract_locations(raw_input),
                    )
                )
            return notifications

        if event in {"on_tool_end", "tool_end", "on_tool_error", "tool_error"} and isinstance(data, dict):
            call_id = str(data.get("run_id") or data.get("tool_call_id") or f"call_{uuid.uuid4().hex[:12]}")
            output = data.get("output") or data.get("error") or data.get("content") or ""
            raw = tool_inputs.get(call_id, {})
            tool_name = str(raw.get("tool") or data.get("name") or data.get("tool") or "tool")
            status = "failed" if "error" in event else "completed"
            if call_id not in seen_tool_updates:
                self._trace(
                    "tool.end",
                    session_id=session_id,
                    tool_call_id=call_id,
                    tool_name=tool_name,
                    status=status,
                    output_chars=len(str(output)),
                )
                seen_tool_updates.add(call_id)
                if call_id not in seen_tool_calls:
                    seen_tool_calls.add(call_id)
                    notifications.append(
                        build_tool_call(
                            session_id,
                            call_id,
                            tool_name,
                            raw_input=raw or {"tool": tool_name},
                            kind=classify_tool_kind(tool_name, raw.get("args", {}), output),
                            locations=extract_locations(raw.get("args", {}), output),
                        )
                    )
                notifications.append(
                    build_tool_call_update(
                        session_id,
                        call_id,
                        status=status,
                        text=str(output),
                        max_chars=int(self.config["tool_output_max_chars"]),
                    )
                )
            return notifications

        if isinstance(data, tuple) and data:
            message = data[0]
            if _is_ai_message(message):
                for call in _extract_tool_calls_from_message(message):
                    call_id = call["id"]
                    if call_id in seen_tool_calls:
                        continue
                    self._trace(
                        "tool.call_from_message",
                        session_id=session_id,
                        tool_call_id=call_id,
                        tool_name=call["name"],
                        args=call.get("args", {}),
                    )
                    seen_tool_calls.add(call_id)
                    raw = {"tool": call["name"], "args": redact_for_logs(call.get("args", {}), max_field_chars=2000)}
                    tool_inputs[call_id] = raw
                    notifications.append(
                        build_tool_call(
                            session_id,
                            call_id,
                            call["name"],
                            raw_input=raw,
                            kind=classify_tool_kind(call["name"], call.get("args", {})),
                            locations=extract_locations(call.get("args", {})),
                        )
                    )
            tool_result = _extract_tool_result(message)
            if tool_result:
                call_id, _name, output = tool_result
                if call_id not in seen_tool_updates:
                    self._trace(
                        "tool.result_from_message",
                        session_id=session_id,
                        tool_call_id=call_id,
                        tool_name=_name,
                        output_chars=len(str(output)),
                    )
                    seen_tool_updates.add(call_id)
                    raw = tool_inputs.get(call_id, {"tool": _name})
                    if call_id not in seen_tool_calls:
                        seen_tool_calls.add(call_id)
                        notifications.append(
                            build_tool_call(
                                session_id,
                                call_id,
                                str(raw.get("tool") or _name),
                                raw_input=raw,
                                kind=classify_tool_kind(str(raw.get("tool") or _name), raw.get("args", {}), output),
                                locations=extract_locations(raw.get("args", {}), output),
                            )
                        )
                    notifications.append(
                        build_tool_call_update(
                            session_id,
                            call_id,
                            status="completed",
                            text=output,
                            max_chars=int(self.config["tool_output_max_chars"]),
                        )
                    )

        if isinstance(data, dict):
            messages = data.get("messages")
            if isinstance(messages, list):
                for message in messages:
                    if _is_ai_message(message):
                        for call in _extract_tool_calls_from_message(message):
                            call_id = call["id"]
                            if call_id in seen_tool_calls:
                                continue
                            self._trace(
                                "tool.call_from_update_messages",
                                session_id=session_id,
                                tool_call_id=call_id,
                                tool_name=call["name"],
                                args=call.get("args", {}),
                            )
                            seen_tool_calls.add(call_id)
                            raw = {"tool": call["name"], "args": redact_for_logs(call.get("args", {}), max_field_chars=2000)}
                            tool_inputs[call_id] = raw
                            notifications.append(
                                build_tool_call(
                                    session_id,
                                    call_id,
                                    call["name"],
                                    raw_input=raw,
                                    kind=classify_tool_kind(call["name"], call.get("args", {})),
                                    locations=extract_locations(call.get("args", {})),
                                )
                            )
                    tool_result = _extract_tool_result(message)
                    if tool_result:
                        call_id, _name, output = tool_result
                        if call_id in seen_tool_updates:
                            continue
                        self._trace(
                            "tool.result_from_update_messages",
                            session_id=session_id,
                            tool_call_id=call_id,
                            tool_name=_name,
                            output_chars=len(str(output)),
                        )
                        seen_tool_updates.add(call_id)
                        raw = tool_inputs.get(call_id, {"tool": _name})
                        if call_id not in seen_tool_calls:
                            seen_tool_calls.add(call_id)
                            notifications.append(
                                build_tool_call(
                                    session_id,
                                    call_id,
                                    str(raw.get("tool") or _name),
                                    raw_input=raw,
                                    kind=classify_tool_kind(str(raw.get("tool") or _name), raw.get("args", {}), output),
                                    locations=extract_locations(raw.get("args", {}), output),
                                )
                            )
                        notifications.append(
                            build_tool_call_update(
                                session_id,
                                call_id,
                                status="completed",
                                text=output,
                                max_chars=int(self.config["tool_output_max_chars"]),
                            )
                        )
        return notifications

    async def _handle_permission_interrupt(self, session: AcpSessionState, interrupt_value: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        command = str(interrupt_value.get("command") or "")
        target = str(interrupt_value.get("target") or "")
        risk = str(interrupt_value.get("risk") or "This command requires approval.")
        call_id = f"approval_{uuid.uuid4().hex[:12]}"
        self._trace(
            "permission.interrupt.start",
            session_id=session.session_id,
            thread_id=session.thread_id,
            tool_call_id=call_id,
            command=command,
            target=target,
            risk=risk,
        )
        await self.send_json(
            build_tool_call(
                session.session_id,
                call_id,
                "Befehl benötigt Freigabe",
                raw_input={"command": command, "target": target, "risk": risk},
                kind="execute",
                status="pending",
            )
        )
        result = await self.request_permission(
            session.session_id,
            tool_call_id=call_id,
            title="Befehl benötigt Freigabe",
            command=command,
            description=risk,
            target=target,
        )
        resume = permission_result_to_resume(result)
        self._trace(
            "permission.interrupt.result",
            session_id=session.session_id,
            thread_id=session.thread_id,
            tool_call_id=call_id,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            resume=resume,
        )
        await self.send_json(
            build_tool_call_update(
                session.session_id,
                call_id,
                status="completed" if resume["action"] == "approve" else "failed",
                text=f"Permission result: {resume['action']}",
                max_chars=2000,
            )
        )
        return resume

    async def request_permission(
        self,
        session_id: str,
        *,
        tool_call_id: str,
        title: str,
        command: str,
        description: str,
        target: str = "",
    ) -> Any:
        request_id = self._next_request_id
        self._next_request_id += 1
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending_responses[request_id] = future
        self._trace(
            "permission.request.send",
            session_id=session_id,
            request_id=request_id,
            tool_call_id=tool_call_id,
            title=title,
            command=command,
        )
        await self.send_json(
            build_permission_request(
                request_id,
                session_id,
                tool_call_id=tool_call_id,
                title=title,
                command=scrub_text(command, max_chars=2000),
                description=scrub_text(description, max_chars=2000),
                target=target,
            )
        )
        try:
            result = await asyncio.wait_for(future, timeout=3600)
            self._trace("permission.request.response", session_id=session_id, request_id=request_id, result=result)
            return result
        except asyncio.TimeoutError:
            self._pending_responses.pop(request_id, None)
            self._trace("permission.request.timeout", session_id=session_id, request_id=request_id)
            return {"outcome": {"outcome": "rejected", "optionId": "reject_once"}}

    async def _cancel_langgraph_run(self, session: AcpSessionState) -> bool:
        if not session.active_run_id or get_client is None:
            self._trace(
                "langgraph.cancel.skipped",
                session_id=session.session_id,
                thread_id=session.thread_id,
                run_id=session.active_run_id,
                reason="missing_run_id_or_sdk",
            )
            return False
        started = time.perf_counter()
        self._trace(
            "langgraph.cancel.start",
            session_id=session.session_id,
            thread_id=session.thread_id,
            run_id=session.active_run_id,
        )
        try:
            client = get_client(url=self.config["langgraph_api_url"])
            cancel = getattr(client.runs, "cancel", None) or getattr(client.runs, "cancel_run", None)
            if cancel is None:
                self._trace("langgraph.cancel.unavailable", session_id=session.session_id, run_id=session.active_run_id)
                return False
            try:
                result = cancel(session.thread_id, session.active_run_id)
            except TypeError:
                result = cancel(thread_id=session.thread_id, run_id=session.active_run_id)
            if asyncio.iscoroutine(result):
                await result
            self._trace(
                "langgraph.cancel.end",
                session_id=session.session_id,
                thread_id=session.thread_id,
                run_id=session.active_run_id,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            return True
        except Exception as exc:
            log_exception(
                "acp_adapter.langgraph_cancel.failed",
                exc,
                component="alpharavis-acp",
                session_id=session.session_id,
                thread_id=session.thread_id,
                run_id=session.active_run_id,
            )
            self._trace(
                "langgraph.cancel.failed",
                session_id=session.session_id,
                thread_id=session.thread_id,
                run_id=session.active_run_id,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                error=str(exc),
            )
            return False

    async def _default_langgraph_stream(
        self,
        session: AcpSessionState,
        prompt: str,
        command: dict[str, Any] | None,
    ) -> AsyncIterator[Any]:
        if get_client is None:
            raise RuntimeError("langgraph_sdk is not installed; cannot reach LangGraph API")
        started = time.perf_counter()
        self._trace(
            "langgraph.default_stream.start",
            session_id=session.session_id,
            thread_id=session.thread_id,
            api_url=self.config["langgraph_api_url"],
            assistant_id=self.config["assistant_id"],
            command=command,
            prompt_chars=len(prompt),
        )
        client_started = time.perf_counter()
        client = get_client(url=self.config["langgraph_api_url"])
        self._trace(
            "langgraph.client.created",
            session_id=session.session_id,
            elapsed_ms=round((time.perf_counter() - client_started) * 1000, 2),
        )
        try:
            thread_started = time.perf_counter()
            await client.threads.create(
                thread_id=session.thread_id,
                if_exists="do_nothing",
                graph_id=self.config["assistant_id"],
                metadata={
                    "source": "aionui-acp",
                    "session_id": session.session_id,
                    "cwd": session.cwd,
                },
            )
            self._trace(
                "langgraph.thread.ensure.end",
                session_id=session.session_id,
                thread_id=session.thread_id,
                elapsed_ms=round((time.perf_counter() - thread_started) * 1000, 2),
                status="created_or_existing",
            )
        except Exception as exc:
            if "409" not in str(exc) and "already" not in str(exc).lower():
                self._trace(
                    "langgraph.thread.ensure.failed",
                    session_id=session.session_id,
                    thread_id=session.thread_id,
                    elapsed_ms=round((time.perf_counter() - thread_started) * 1000, 2),
                    error=str(exc),
                )
                raise
            self._trace(
                "langgraph.thread.ensure.end",
                session_id=session.session_id,
                thread_id=session.thread_id,
                elapsed_ms=round((time.perf_counter() - thread_started) * 1000, 2),
                status="already_exists",
            )

        stream_kwargs: dict[str, Any] = {
            "stream_mode": ["messages", "updates"],
            "multitask_strategy": "interrupt",
        }
        if command is not None:
            stream_kwargs["command"] = command
        else:
            stream_kwargs["input"] = {
                "messages": [{"role": "human", "content": prompt}],
                "thread_id": session.thread_id,
                "thread_key": session.session_id,
                "client": "aionui-acp",
                "workspace": session.cwd,
            }
        self._trace(
            "langgraph.runs.stream.open",
            session_id=session.session_id,
            thread_id=session.thread_id,
            stream_kwargs={key: value for key, value in stream_kwargs.items() if key != "input"},
            has_input="input" in stream_kwargs,
        )
        try:
            async for part in client.runs.stream(session.thread_id, self.config["assistant_id"], **stream_kwargs):
                yield part
        finally:
            self._trace(
                "langgraph.default_stream.end",
                session_id=session.session_id,
                thread_id=session.thread_id,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            )

    async def run_stdio(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if line == "":
                break
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                await self.send_error(None, -32700, f"Invalid JSON: {exc}")
                continue
            if isinstance(message, dict):
                await self.handle_message(message)


async def amain() -> None:
    adapter = AlphaRavisAcpAdapter()
    await adapter.run_stdio()


def main() -> None:
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
