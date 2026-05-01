from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
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


def env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int, *, minimum: int = 0, maximum: int = 1_000_000) -> int:
    try:
        value = int(os.getenv(name, str(default)))
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
) -> dict[str, Any]:
    return {
        "jsonrpc": JSONRPC_VERSION,
        "method": "session/update",
        "params": {
            "sessionId": session_id,
            "update": {
                "sessionUpdate": "tool_call",
                "toolCallId": tool_call_id,
                "status": status,
                "title": title,
                "kind": kind,
                "rawInput": raw_input or {"tool": title},
            },
        },
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
    safe_text = scrub_text(text, max_chars=max_chars)
    update: dict[str, Any] = {
        "sessionUpdate": "tool_call_update",
        "toolCallId": tool_call_id,
        "status": status,
        "content": [{"type": "content", "content": {"type": "text", "text": safe_text}}],
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
    if isinstance(result, dict):
        outcome = result.get("outcome")
        if isinstance(outcome, dict):
            option_id = str(outcome.get("optionId") or "")
        option_id = option_id or str(result.get("optionId") or "")
    option_id = option_id.lower()
    if "allow" in option_id or "approve" in option_id:
        return {"action": "approve"}
    return {"action": "reject"}


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

    async def send_json(self, payload: dict[str, Any]) -> None:
        async with self._write_lock:
            if self._writer is not None:
                result = self._writer(payload)
                if asyncio.iscoroutine(result):
                    await result
                return
            sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            sys.stdout.flush()

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
        if "method" in message:
            asyncio.create_task(self._handle_request_or_notification(message))
            return
        request_id = message.get("id")
        if isinstance(request_id, int) and request_id in self._pending_responses:
            future = self._pending_responses.pop(request_id)
            if "error" in message:
                future.set_exception(RuntimeError(str(message["error"])))
            else:
                future.set_result(message.get("result"))

    async def _handle_request_or_notification(self, message: dict[str, Any]) -> None:
        request_id = message.get("id")
        method = str(message.get("method") or "")
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        try:
            if method == "initialize":
                result = self.initialize(params)
            elif method == "session/new":
                result = self.new_session(params)
            elif method in {"session/prompt", "session/send_message"}:
                result = await self.prompt(params)
            elif method == "session/cancel":
                result = self.cancel(params)
            elif method == "session/close":
                result = self.close(params)
            elif method == "session/load":
                result = self.load_session(params)
            elif method == "session/set_config_option":
                result = {}
            elif method in {"session/response", "session/permission_response"}:
                result = self.permission_response(params)
            else:
                if request_id is not None:
                    await self.send_error(request_id, -32601, f"Unsupported ACP method: {method}")
                return
            if request_id is not None:
                await self.send_response(request_id, result)
        except Exception as exc:
            log_exception("acp_adapter.request.failed", exc, component="alpharavis-acp", method=method)
            if request_id is not None:
                await self.send_error(request_id, -32603, str(exc))

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

    def load_session(self, params: dict[str, Any]) -> dict[str, Any]:
        return self.new_session(params)

    def close(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = str(params.get("sessionId") or "")
        self.sessions.pop(session_id, None)
        return {}

    def cancel(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = str(params.get("sessionId") or "")
        if session_id in self.sessions:
            self.sessions[session_id].cancelled = True
        return {}

    def permission_response(self, params: dict[str, Any]) -> dict[str, Any]:
        request_id = params.get("requestId")
        if isinstance(request_id, int) and request_id in self._pending_responses:
            self._pending_responses.pop(request_id).set_result(params)
        return {}

    async def prompt(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = str(params.get("sessionId") or "")
        if not session_id:
            raise ValueError("sessionId is required")
        session = self.sessions.get(session_id)
        if session is None:
            session = AcpSessionState(session_id=session_id, thread_id=stable_thread_id(session_id), cwd=self.config["workspace"])
            self.sessions[session_id] = session
        session.cancelled = False
        prompt = extract_prompt_text(params.get("prompt") or params.get("message") or params.get("input"))
        session.last_prompt = prompt
        if not prompt.strip():
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
        return {"stopReason": "cancelled" if session.cancelled else "end_turn"}

    async def _stream_prompt(self, session: AcpSessionState, prompt: str) -> None:
        command: dict[str, Any] | None = None
        for _attempt in range(4):
            if session.cancelled:
                return
            interrupt_value = await self._stream_one_run(session, prompt, command)
            if not interrupt_value:
                return
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
        emitted_text = ""
        emitted_reasoning = ""
        seen_tool_calls: set[str] = set()
        seen_tool_updates: set[str] = set()
        content_scrubber = StreamingInternalContextScrubber() if self.config["scrub_internal_context"] and StreamingInternalContextScrubber else None
        thought_scrubber = StreamingInternalContextScrubber() if self.config["scrub_internal_context"] and StreamingInternalContextScrubber else None

        stream = self._langgraph_stream(session, prompt, command) if self._langgraph_stream else self._default_langgraph_stream(session, prompt, command)
        async with asyncio.timeout(float(self.config["run_timeout_seconds"])):
            async for part in stream:
                if session.cancelled:
                    return None
                interrupt_value = _find_command_approval_interrupt(part)
                if interrupt_value:
                    return interrupt_value

                await self._emit_activity_or_plan(session.session_id, part)

                for notification in self._tool_notifications_from_part(session.session_id, part, seen_tool_calls, seen_tool_updates):
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
                    emitted_text += delta
                    visible = content_scrubber.feed(delta) if content_scrubber else delta
                    visible = scrub_text(visible)
                    if visible:
                        await self.send_json(build_agent_message_chunk(session.session_id, visible))

        if content_scrubber:
            tail = scrub_text(content_scrubber.flush())
            if tail:
                await self.send_json(build_agent_message_chunk(session.session_id, tail))
        if thought_scrubber and self.config["trace_detail"] == "debug":
            tail = scrub_text(thought_scrubber.flush(), max_chars=1200)
            if tail:
                await self.send_json(build_agent_thought_chunk(session.session_id, tail))
        return None

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
    ) -> list[dict[str, Any]]:
        notifications: list[dict[str, Any]] = []
        event = _stream_event_name(part)
        data = _stream_event_data(part)

        if event in {"on_tool_start", "tool_start"} and isinstance(data, dict):
            raw_input = data.get("input") if isinstance(data.get("input"), dict) else {}
            tool_name = str(data.get("name") or data.get("tool") or data.get("run_name") or "tool")
            call_id = str(data.get("run_id") or data.get("tool_call_id") or f"call_{uuid.uuid4().hex[:12]}")
            if call_id not in seen_tool_calls:
                seen_tool_calls.add(call_id)
                notifications.append(build_tool_call(session_id, call_id, tool_name, raw_input={"tool": tool_name, "args": raw_input}))
            return notifications

        if event in {"on_tool_end", "tool_end", "on_tool_error", "tool_error"} and isinstance(data, dict):
            call_id = str(data.get("run_id") or data.get("tool_call_id") or f"call_{uuid.uuid4().hex[:12]}")
            output = data.get("output") or data.get("error") or data.get("content") or ""
            status = "failed" if "error" in event else "completed"
            if call_id not in seen_tool_updates:
                seen_tool_updates.add(call_id)
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
                    seen_tool_calls.add(call_id)
                    notifications.append(
                        build_tool_call(
                            session_id,
                            call_id,
                            call["name"],
                            raw_input={"tool": call["name"], "args": redact_for_logs(call.get("args", {}), max_field_chars=2000)},
                        )
                    )
            tool_result = _extract_tool_result(message)
            if tool_result:
                call_id, _name, output = tool_result
                if call_id not in seen_tool_updates:
                    seen_tool_updates.add(call_id)
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
                            seen_tool_calls.add(call_id)
                            notifications.append(
                                build_tool_call(
                                    session_id,
                                    call_id,
                                    call["name"],
                                    raw_input={"tool": call["name"], "args": redact_for_logs(call.get("args", {}), max_field_chars=2000)},
                                )
                            )
                    tool_result = _extract_tool_result(message)
                    if tool_result:
                        call_id, _name, output = tool_result
                        if call_id in seen_tool_updates:
                            continue
                        seen_tool_updates.add(call_id)
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
        command = str(interrupt_value.get("command") or "")
        target = str(interrupt_value.get("target") or "")
        risk = str(interrupt_value.get("risk") or "This command requires approval.")
        call_id = f"approval_{uuid.uuid4().hex[:12]}"
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
            return await asyncio.wait_for(future, timeout=3600)
        except asyncio.TimeoutError:
            self._pending_responses.pop(request_id, None)
            return {"outcome": {"outcome": "rejected", "optionId": "reject_once"}}

    async def _default_langgraph_stream(
        self,
        session: AcpSessionState,
        prompt: str,
        command: dict[str, Any] | None,
    ) -> AsyncIterator[Any]:
        if get_client is None:
            raise RuntimeError("langgraph_sdk is not installed; cannot reach LangGraph API")
        client = get_client(url=self.config["langgraph_api_url"])
        try:
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
        except Exception as exc:
            if "409" not in str(exc) and "already" not in str(exc).lower():
                raise

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
        async for part in client.runs.stream(session.thread_id, self.config["assistant_id"], **stream_kwargs):
            yield part

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
