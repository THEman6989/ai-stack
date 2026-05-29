from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from compression_redact import redact_sensitive_text
from model_metadata import (
    estimate_message_tokens_rough,
    estimate_tokens_rough_text,
    extract_usage_tokens,
)


SummaryFn = Callable[[str, int], Awaitable[str]]
ProgressFn = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class CompressionSelection:
    head: list[Any]
    middle: list[Any]
    tail: list[Any]
    head_indexes: list[int] = field(default_factory=list)
    middle_indexes: list[int] = field(default_factory=list)
    tail_indexes: list[int] = field(default_factory=list)
    oversized_tail_rebalanced: bool = False
    oversized_tail_tokens_before: int = 0
    oversized_tail_token_target: int = 0
    oversized_tail_moved_indexes: list[int] = field(default_factory=list)
    oversized_tail_force_latest_user_to_middle: bool = False
    oversized_tail_force_middle_target: int = 0


@dataclass(frozen=True)
class SummaryPrep:
    text: str
    pruned_tool_count: int = 0
    deduped_tool_count: int = 0
    tool_args_truncated_count: int = 0
    workflow_event_compaction: str = ""
    workflow_event_count: int = 0
    workflow_tool_call_count: int = 0
    workflow_tool_result_count: int = 0
    workflow_event_chars: int = 0


@dataclass(frozen=True)
class CompressionDecision:
    should_run: bool
    reason: str = ""


@dataclass(frozen=True)
class CompressionResult:
    mode: str
    thread_id: str
    thread_key: str
    token_limit: int
    token_estimate_before: int
    token_estimate_after: int
    head: list[Any]
    middle: list[Any]
    tail: list[Any]
    summary: str
    summary_message_content: str
    archive_content: str
    archive_metadata: dict[str, Any]
    pruned_middle_text: str
    skipped: bool = False
    reason: str = ""
    summary_failed: bool = False
    summary_error: str = ""
    compression_stats: dict[str, Any] = field(default_factory=dict)
    pruned_tool_count: int = 0
    deduped_tool_count: int = 0
    tool_args_truncated_count: int = 0
    workflow_event_count: int = 0
    workflow_tool_call_count: int = 0
    workflow_tool_result_count: int = 0
    workflow_event_chars: int = 0


PROTECTED_SECTION_NAMES = [
    "Active Task",
    "Goal",
    "Handoff Packet",
    "MemoryKernel",
    "Skill Context",
    "Constraints / Preferences",
    "Progress Done",
    "Progress In Progress",
    "Blocked / Risks",
    "Resolved Questions",
    "Pending User Asks",
    "Key Decisions",
    "Relevant Files",
    "Commands / Tools Used",
    "Critical Context",
    "Remaining Work",
    "Archive References",
]

SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION - REFERENCE ONLY]\n"
    "Earlier turns were compacted to keep the active context small. Do NOT answer "
    "questions or fulfill requests mentioned inside this summary; they were already "
    "handled unless the latest user message after the summary asks for them again. "
    "Use this only as background orientation, and retrieve archives before relying "
    "on old exact details."
)


from env_utils import env_bool as _env_bool


from env_utils import env_int as _env_int


from env_utils import env_float as _env_float


def _content_to_text(content: Any) -> str:
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if str(block.get("type") or "") in {"thinking", "reasoning"}:
                    continue
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
        return " ".join(part for part in parts if part)
    return str(content or "")


def message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or message.get("type") or "message")
    return str(getattr(message, "type", getattr(message, "role", "message")) or "message")


def message_id(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("id") or "")
    return str(getattr(message, "id", "") or "")


def message_name(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("name") or message.get("tool_call_id") or "")
    return str(getattr(message, "name", "") or getattr(message, "tool_call_id", "") or "")


def message_content(message: Any) -> str:
    if isinstance(message, dict):
        return _content_to_text(message.get("content", ""))
    return _content_to_text(getattr(message, "content", ""))


def message_text(message: Any) -> str:
    role = message_role(message)
    name = message_name(message)
    label = f"{role}({name})" if name else role
    return f"{label}: {message_content(message)}".strip()


def estimate_tokens_rough(value: Any) -> int:
    usage_tokens = extract_usage_tokens(value)
    if usage_tokens:
        return max(1, usage_tokens)
    if isinstance(value, str):
        return estimate_tokens_rough_text(value)
    if isinstance(value, list):
        total = 0
        for item in value:
            total += estimate_message_tokens_rough(item)
        return max(1, total)
    if isinstance(value, dict):
        return estimate_message_tokens_rough(value)
    return estimate_tokens_rough_text(str(value or ""))


def stable_message_key(message: Any) -> str:
    msg_id = message_id(message)
    if msg_id:
        return f"id:{msg_id}"
    return "hash:" + hashlib.sha256(message_text(message).encode("utf-8")).hexdigest()[:24]


def _message_mapping(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return message
    return {
        "type": getattr(message, "type", None),
        "role": getattr(message, "role", None),
        "name": getattr(message, "name", None),
        "id": getattr(message, "id", None),
        "content": getattr(message, "content", ""),
        "tool_call_id": getattr(message, "tool_call_id", None),
        "tool_calls": getattr(message, "tool_calls", None),
        "additional_kwargs": getattr(message, "additional_kwargs", {}) or {},
        "response_metadata": getattr(message, "response_metadata", {}) or {},
    }


def is_tool_message(message: Any) -> bool:
    role = message_role(message).lower()
    class_name = type(message).__name__.lower()
    if role in {"tool", "function"} or "toolmessage" in class_name:
        return True
    mapping = _message_mapping(message)
    return bool(mapping.get("tool_call_id"))


def has_tool_call_request(message: Any) -> bool:
    mapping = _message_mapping(message)
    additional = mapping.get("additional_kwargs") or {}
    return bool(mapping.get("tool_calls") or additional.get("tool_calls"))


def redact_secrets(text: str) -> str:
    return redact_sensitive_text(text)


def _truncate_middle(text: str, *, max_chars: int, head_chars: int, tail_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = text[:head_chars].rstrip()
    tail = text[-tail_chars:].lstrip() if tail_chars > 0 else ""
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n[... {omitted} chars omitted for compression prompt ...]\n{tail}".strip()


def _truncate_value(value: Any, *, max_chars: int, head_chars: int, tail_chars: int) -> Any:
    if isinstance(value, str):
        if len(value) <= max_chars:
            return value
        head = value[:head_chars].rstrip()
        tail = value[-tail_chars:].lstrip() if tail_chars > 0 else ""
        omitted = len(value) - len(head) - len(tail)
        return f"{head}\n[... {omitted} chars omitted from tool arguments ...]\n{tail}".strip()
    if isinstance(value, list):
        return [_truncate_value(item, max_chars=max_chars, head_chars=head_chars, tail_chars=tail_chars) for item in value]
    if isinstance(value, dict):
        return {
            key: _truncate_value(item, max_chars=max_chars, head_chars=head_chars, tail_chars=tail_chars)
            for key, item in value.items()
        }
    return value


def _truncate_tool_call_args_json(args: Any, head_chars: int | None = None) -> str:
    if args is None:
        return ""
    if not isinstance(args, str):
        try:
            args = json.dumps(args, ensure_ascii=False)
        except Exception:
            return str(args)

    raw_args = args
    try:
        parsed = json.loads(raw_args)
    except Exception:
        return raw_args

    max_chars = max(128, _env_int("ALPHARAVIS_COMPRESSION_TOOL_ARGS_MAX_CHARS", 1500))
    head = max(64, int(head_chars if head_chars is not None else _env_int("ALPHARAVIS_COMPRESSION_TOOL_ARGS_HEAD_CHARS", 1000)))
    tail = max(0, min(_env_int("ALPHARAVIS_COMPRESSION_TOOL_ARGS_TAIL_CHARS", 300), max_chars // 2))
    truncated = _truncate_value(parsed, max_chars=max_chars, head_chars=head, tail_chars=tail)
    return json.dumps(truncated, ensure_ascii=False)


def _iter_tool_calls(message: Any) -> list[dict[str, str]]:
    mapping = _message_mapping(message)
    calls: list[Any] = []
    raw_calls = mapping.get("tool_calls")
    if isinstance(raw_calls, list):
        calls.extend(raw_calls)
    additional = mapping.get("additional_kwargs") or {}
    if isinstance(additional.get("tool_calls"), list):
        calls.extend(additional["tool_calls"])

    parsed: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for call in calls:
        call_id = ""
        name = ""
        args: Any = ""
        if isinstance(call, dict):
            call_id = str(call.get("id") or "")
            name = str(call.get("name") or "")
            args = call.get("args")
            function = call.get("function")
            if isinstance(function, dict):
                name = str(function.get("name") or name)
                args = function.get("arguments", args)
        else:
            call_id = str(getattr(call, "id", "") or "")
            name = str(getattr(call, "name", "") or "")
            args = getattr(call, "args", "")

        args_text = _truncate_tool_call_args_json(args)
        key = (call_id, name, args_text)
        if key in seen:
            continue
        seen.add(key)
        parsed.append({"id": call_id, "name": name or "tool", "args": args_text})
    return parsed


def _safe_json_loads(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def _arg_value(args: Any, *names: str) -> str:
    parsed = _safe_json_loads(args)
    if isinstance(parsed, dict):
        for name in names:
            if parsed.get(name) is not None:
                return str(parsed[name])
    return ""


def _find_exit_code(text: str) -> str:
    for pattern in [
        r"(?i)\bexit(?:\s+code)?\s*[:=]\s*(-?\d+)",
        r"(?i)\breturncode\s*[:=]\s*(-?\d+)",
        r"(?i)\bExit code:\s*(-?\d+)",
    ]:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return "unknown"


def _count_search_matches(text: str) -> str:
    patterns = [
        r"(?i)\b(\d+)\s+matches?\b",
        r"(?i)\b(\d+)\s+results?\b",
        r"(?i)\bfound\s+(\d+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    nonempty = [line for line in text.splitlines() if line.strip()]
    return str(len(nonempty)) if nonempty else "0"


def summarize_tool_result(tool_name: str, tool_args: Any, tool_content: str) -> str:
    name = str(tool_name or "tool").strip() or "tool"
    lowered = name.lower()
    content = str(tool_content or "")
    line_count = len(content.splitlines())
    char_count = len(content)

    if any(part in lowered for part in ("terminal", "shell", "execute", "command", "powershell", "bash")):
        command = _arg_value(tool_args, "command", "cmd", "script", "code") or _first_command_like_line(content)
        exit_code = _find_exit_code(content)
        return f"[{name}] ran `{command or 'command unavailable'}` -> exit {exit_code}, {line_count} lines, {char_count} chars"

    if any(part in lowered for part in ("read_file", "open_file", "fetch_file", "view_file")):
        path = _arg_value(tool_args, "path", "file", "filename") or _first_path_like_line(content)
        offset = _arg_value(tool_args, "offset", "line", "start_line", "lineno")
        offset_part = f" from {offset}" if offset else ""
        return f"[{name}] read {path or 'file'}{offset_part} -> {char_count} chars, {line_count} lines"

    if any(part in lowered for part in ("write_file", "update_file", "create_file", "patch", "apply_patch")):
        path = _arg_value(tool_args, "path", "file", "filename")
        return f"[{name}] wrote/updated {path or 'file'} -> {line_count} lines, {char_count} chars"

    if any(part in lowered for part in ("search", "grep", "ripgrep", "find")):
        query = _arg_value(tool_args, "query", "q", "pattern", "search_query") or _first_search_like_line(content)
        return f"[{name}] search `{query or 'query unavailable'}` -> {_count_search_matches(content)} matches, {char_count} chars"

    if any(part in lowered for part in ("browser", "web", "url", "duckduckgo", "tavily", "searx")):
        url = _arg_value(tool_args, "url", "href") or _first_url_like_line(content)
        query = _arg_value(tool_args, "query", "q", "search_query")
        target = url or query or "web target"
        return f"[{name}] web/browser access `{target}` -> {line_count} lines, {char_count} chars"

    args_preview = _truncate_middle(str(tool_args or ""), max_chars=500, head_chars=360, tail_chars=120)
    return f"[{name}] tool result -> {line_count} lines, {char_count} chars, args={args_preview or '{}'}"


def _first_command_like_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if re.search(r"\b(docker|git|python|curl|npm|uv|pytest|powershell|Get-ChildItem|Select-String)\b", stripped):
            return stripped[:220]
    return ""


def _first_path_like_line(text: str) -> str:
    for line in text.splitlines():
        match = re.search(r"([A-Za-z]:\\[^\s:]+|(?:\.{0,2}/)?[\w./-]+\.[A-Za-z0-9]{1,8})", line)
        if match:
            return match.group(1)[:220]
    return ""


def _first_url_like_line(text: str) -> str:
    match = re.search(r"https?://[^\s'\"<>]+", text)
    return match.group(0)[:220] if match else ""


def _first_search_like_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:160]
    return ""


def _tool_label(message: Any, call_map: dict[str, dict[str, str]] | None = None) -> str:
    name = message_name(message)
    if name:
        return name
    call_id = str(_message_mapping(message).get("tool_call_id") or "")
    if call_map and call_id in call_map:
        return call_map[call_id].get("name", "tool")
    first_line = next((line.strip() for line in message_content(message).splitlines() if line.strip()), "")
    return first_line[:160] or "tool output"


def _build_tool_call_map(messages: list[Any]) -> dict[str, dict[str, str]]:
    call_map: dict[str, dict[str, str]] = {}
    for message in messages:
        for call in _iter_tool_calls(message):
            call_id = call.get("id") or ""
            if call_id:
                call_map[call_id] = call
    return call_map


def _summarize_assistant_tool_calls(message: Any) -> tuple[list[str], int]:
    lines: list[str] = []
    truncated_count = 0
    for call in _iter_tool_calls(message):
        raw_args = call.get("args", "")
        original_len = len(str(raw_args))
        args_text = _truncate_tool_call_args_json(raw_args)
        if len(args_text) < original_len:
            truncated_count += 1
        lines.append(f"- {call.get('name', 'tool')} id={call.get('id') or 'unknown'} args={args_text}")
    return lines, truncated_count


def prepare_messages_for_summary(messages: list[Any]) -> SummaryPrep:
    max_chars = _env_int("ALPHARAVIS_COMPRESSION_TOOL_OUTPUT_MAX_CHARS", 6000)
    head_chars = _env_int("ALPHARAVIS_COMPRESSION_TOOL_OUTPUT_HEAD_CHARS", 4000)
    tail_chars = _env_int("ALPHARAVIS_COMPRESSION_TOOL_OUTPUT_TAIL_CHARS", 1500)
    dedup_min_chars = _env_int("ALPHARAVIS_COMPRESSION_DEDUP_MIN_CHARS", 200)
    workflow_max_chars = _env_int("ALPHARAVIS_WORKFLOW_EVENT_OUTPUT_MAX_CHARS", 900)
    workflow_head_chars = _env_int("ALPHARAVIS_WORKFLOW_EVENT_OUTPUT_HEAD_CHARS", 620)
    workflow_tail_chars = _env_int("ALPHARAVIS_WORKFLOW_EVENT_OUTPUT_TAIL_CHARS", 180)
    call_map = _build_tool_call_map(messages)

    newest_by_hash: dict[str, int] = {}
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not is_tool_message(message):
            continue
        content = message_content(message)
        if len(content) < dedup_min_chars:
            continue
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        newest_by_hash.setdefault(digest, index)

    sections: list[str] = []
    pruned_tool_count = 0
    deduped_tool_count = 0
    tool_args_truncated_count = 0
    workflow_events: list[str] = []
    workflow_tool_call_count = 0
    workflow_tool_result_count = 0

    for index, message in enumerate(messages, start=1):
        role = message_role(message)
        content = message_content(message)
        if is_tool_message(message):
            call_id = str(_message_mapping(message).get("tool_call_id") or message_name(message) or "")
            call_info = call_map.get(call_id, {"name": _tool_label(message, call_map), "args": ""})
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            newest_index = newest_by_hash.get(digest)
            if newest_index is not None and newest_index != index - 1:
                deduped_tool_count += 1
                workflow_tool_result_count += 1
                workflow_events.append(
                    f"- message {index}: [tool-output duplicate] {call_info.get('name', 'tool')} "
                    f"same content as newer tool output at message {newest_index + 1}; hash={digest[:12]}."
                )
                sections.append(
                    f"[{index}] [workflow-event compacted] duplicate tool output for "
                    f"{call_info.get('name', 'tool')}; see Workflow / Tool Event Compact Log."
                )
                continue

            summary = summarize_tool_result(call_info.get("name", "tool"), call_info.get("args", ""), content)
            if len(content) > max_chars:
                pruned_tool_count += 1
            workflow_tool_result_count += 1
            workflow_preview = _truncate_middle(
                content,
                max_chars=workflow_max_chars,
                head_chars=workflow_head_chars,
                tail_chars=workflow_tail_chars,
            )
            workflow_events.append(f"- message {index}: {summary}\n  preview: {workflow_preview}".strip())
            sections.append(
                f"[{index}] [workflow-event compacted] tool result for "
                f"{call_info.get('name', 'tool')}: {summary}; see Workflow / Tool Event Compact Log."
            )
            continue

        if has_tool_call_request(message):
            lines, truncated = _summarize_assistant_tool_calls(message)
            tool_args_truncated_count += truncated
            text = message_text(message)
            if len(text) > max_chars:
                text = _truncate_middle(text, max_chars=max_chars, head_chars=head_chars, tail_chars=tail_chars)
            workflow_tool_call_count += len(lines)
            workflow_events.extend(f"- message {index}: tool_call {line.lstrip('- ').strip()}" for line in lines)
            sections.append(
                f"[{index}] {text}\n"
                f"[workflow-event compacted] assistant requested {len(lines)} tool call(s); "
                "details are in Workflow / Tool Event Compact Log."
            )
            continue

        text = message_text(message)
        if len(text) > max_chars:
            sections.append(
                "\n".join(
                    [
                        f"[{index}] [{role}] long message -> {len(text)} chars.",
                        _truncate_middle(text, max_chars=max_chars, head_chars=head_chars, tail_chars=tail_chars),
                    ]
                ).strip()
            )
            continue

        sections.append(f"[{index}] {text}".strip())

    workflow_text = redact_secrets("\n".join(workflow_events).strip())
    if workflow_text:
        sections.append(
            "\n".join(
                [
                    "## Workflow / Tool Event Compact Log",
                    "Tool calls, tool outputs, and action history are compacted here separately from chat messages.",
                    workflow_text,
                ]
            )
        )

    return SummaryPrep(
        text=redact_secrets("\n\n".join(sections)),
        pruned_tool_count=pruned_tool_count,
        deduped_tool_count=deduped_tool_count,
        tool_args_truncated_count=tool_args_truncated_count,
        workflow_event_compaction=workflow_text,
        workflow_event_count=len(workflow_events),
        workflow_tool_call_count=workflow_tool_call_count,
        workflow_tool_result_count=workflow_tool_result_count,
        workflow_event_chars=len(workflow_text),
    )


def prune_tool_outputs(messages: list[Any]) -> str:
    return prepare_messages_for_summary(messages).text


def _is_protected_message(message: Any, protected_message_ids: set[str]) -> bool:
    msg_id = message_id(message)
    if msg_id in protected_message_ids:
        return True
    content = message_content(message)
    return '"report_type": "handoff_packet"' in content or "<handoff-packet>" in content


def _expand_for_tool_pairs(indexes: set[int], messages: list[Any], head_indexes: set[int]) -> set[int]:
    expanded = set(indexes)
    for index in list(indexes):
        if is_tool_message(messages[index]) and index > 0 and (index - 1) not in head_indexes:
            expanded.add(index - 1)
        if has_tool_call_request(messages[index]) and index + 1 < len(messages) and (index + 1) not in head_indexes:
            expanded.add(index + 1)
    return expanded


def _latest_user_message_index(messages: list[Any], head_indexes: set[int]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if index in head_indexes:
            continue
        if message_role(messages[index]).lower() in {"user", "human"}:
            return index
    return -1


def _tail_token_estimate(messages: list[Any], indexes: set[int]) -> int:
    if not indexes:
        return 0
    return estimate_tokens_rough([messages[index] for index in sorted(indexes)])


def _rebalance_oversized_tail(
    *,
    messages: list[Any],
    tail_indexes: set[int],
    head_indexes: set[int],
    token_limit: int,
    latest_user_index: int,
) -> tuple[set[int], dict[str, Any]]:
    if not _env_bool("ALPHARAVIS_COMPRESSION_REBALANCE_OVERSIZED_TAIL", "true"):
        return tail_indexes, {}
    if not tail_indexes:
        return tail_indexes, {}

    ratio = max(0.10, min(_env_float("ALPHARAVIS_COMPRESSION_OVERSIZED_TAIL_RATIO", 0.60), 0.95))
    target = max(1, int(token_limit * ratio))
    tokens_before = _tail_token_estimate(messages, tail_indexes)
    if tokens_before <= target:
        return tail_indexes, {}

    force_ratio = max(
        ratio,
        min(_env_float("ALPHARAVIS_COMPRESSION_OVERSIZED_TAIL_FORCE_MIDDLE_RATIO", 0.80), 0.99),
    )
    force_target = max(1, int(token_limit * force_ratio))
    force_latest_user_to_middle = tokens_before > force_target
    latest_large_paste_deferred = False
    if (
        force_latest_user_to_middle
        and latest_user_index >= 0
        and _env_bool("ALPHARAVIS_DEFER_LARGE_PASTE_RAG_UNTIL_AFTER_COMPRESSION", "true")
    ):
        latest_text = message_content(messages[latest_user_index])
        large_paste_min_chars = max(1, _env_int("ALPHARAVIS_LARGE_PASTE_RAG_MIN_CHARS", 20000))
        if len(latest_text) >= large_paste_min_chars:
            force_latest_user_to_middle = False
            latest_large_paste_deferred = True
    keep_latest_user = (
        _env_bool("ALPHARAVIS_COMPRESSION_KEEP_LATEST_USER_WHEN_REBALANCING_TAIL", "true")
        and not force_latest_user_to_middle
    )
    keep_indexes: set[int] = set()
    if keep_latest_user and latest_user_index >= 0 and latest_user_index not in head_indexes:
        keep_indexes.add(latest_user_index)

    min_tail_messages = max(1, _env_int("ALPHARAVIS_COMPRESSION_OVERSIZED_TAIL_MIN_MESSAGES", 1))
    if force_latest_user_to_middle:
        min_tail_messages = 0
    rebalanced = set(tail_indexes)
    moved: list[int] = []
    candidates = [index for index in sorted(tail_indexes) if index not in keep_indexes]
    for index in candidates:
        if len(rebalanced) <= len(keep_indexes) + min_tail_messages:
            break
        if _tail_token_estimate(messages, rebalanced) <= target:
            break
        rebalanced.remove(index)
        moved.append(index)

    if not moved:
        return tail_indexes, {
            "oversized_tail_rebalanced": False,
            "oversized_tail_tokens_before": tokens_before,
            "oversized_tail_token_target": target,
            "oversized_tail_moved_indexes": [],
            "oversized_tail_force_latest_user_to_middle": force_latest_user_to_middle,
            "oversized_tail_latest_large_paste_deferred": latest_large_paste_deferred,
            "oversized_tail_force_middle_target": force_target,
        }
    return rebalanced, {
        "oversized_tail_rebalanced": True,
        "oversized_tail_tokens_before": tokens_before,
        "oversized_tail_token_target": target,
        "oversized_tail_moved_indexes": moved,
        "oversized_tail_force_latest_user_to_middle": force_latest_user_to_middle,
        "oversized_tail_latest_large_paste_deferred": latest_large_paste_deferred,
        "oversized_tail_force_middle_target": force_target,
    }


def select_head_middle_tail(
    messages: list[Any],
    *,
    token_limit: int,
    protected_message_ids: set[str] | None = None,
) -> CompressionSelection:
    protected_message_ids = protected_message_ids or set()
    protect_first = max(0, _env_int("ALPHARAVIS_COMPRESSION_PROTECT_FIRST_MESSAGES", 3))
    protect_last = max(1, _env_int("ALPHARAVIS_COMPRESSION_PROTECT_LAST_MESSAGES", 3))
    tail_ratio = max(0.05, min(_env_float("ALPHARAVIS_COMPRESSION_TAIL_TOKEN_RATIO", 0.20), 0.75))
    tail_budget = max(1, int(token_limit * tail_ratio))
    soft_ceiling_ratio = max(1.0, _env_float("ALPHARAVIS_COMPRESSION_TAIL_SOFT_CEILING_RATIO", 1.5))
    tail_soft_ceiling = max(tail_budget, int(tail_budget * soft_ceiling_ratio))

    head_indexes: set[int] = set(range(min(protect_first, len(messages))))
    for index, message in enumerate(messages):
        if _is_protected_message(message, protected_message_ids):
            head_indexes.add(index)

    tail_indexes: set[int] = set()
    tail_tokens = 0
    for index in range(len(messages) - 1, -1, -1):
        if index in head_indexes:
            continue
        message_tokens = estimate_tokens_rough([messages[index]])
        if len(tail_indexes) >= protect_last and tail_tokens >= tail_budget:
            break
        if len(tail_indexes) >= protect_last and tail_tokens + message_tokens > tail_soft_ceiling:
            break
        tail_indexes.add(index)
        tail_tokens += message_tokens

    latest_user_index = _latest_user_message_index(messages, head_indexes)
    if latest_user_index >= 0:
        tail_indexes.update(index for index in range(latest_user_index, len(messages)) if index not in head_indexes)

    tail_indexes = _expand_for_tool_pairs(tail_indexes, messages, head_indexes)
    tail_indexes -= head_indexes
    tail_indexes, oversized_tail_stats = _rebalance_oversized_tail(
        messages=messages,
        tail_indexes=tail_indexes,
        head_indexes=head_indexes,
        token_limit=token_limit,
        latest_user_index=latest_user_index,
    )

    middle_indexes = set(range(len(messages))) - head_indexes - tail_indexes
    return CompressionSelection(
        head=[messages[index] for index in sorted(head_indexes)],
        middle=[messages[index] for index in sorted(middle_indexes)],
        tail=[messages[index] for index in sorted(tail_indexes)],
        head_indexes=sorted(head_indexes),
        middle_indexes=sorted(middle_indexes),
        tail_indexes=sorted(tail_indexes),
        oversized_tail_rebalanced=bool(oversized_tail_stats.get("oversized_tail_rebalanced")),
        oversized_tail_tokens_before=int(oversized_tail_stats.get("oversized_tail_tokens_before") or 0),
        oversized_tail_token_target=int(oversized_tail_stats.get("oversized_tail_token_target") or 0),
        oversized_tail_moved_indexes=list(oversized_tail_stats.get("oversized_tail_moved_indexes") or []),
        oversized_tail_force_latest_user_to_middle=bool(
            oversized_tail_stats.get("oversized_tail_force_latest_user_to_middle")
        ),
        oversized_tail_force_middle_target=int(oversized_tail_stats.get("oversized_tail_force_middle_target") or 0),
    )


def _summary_token_limit(token_limit: int) -> int:
    ratio = max(0.05, min(_env_float("ALPHARAVIS_COMPRESSION_SUMMARY_RATIO", 0.20), 0.80))
    minimum = max(200, _env_int("ALPHARAVIS_COMPRESSION_SUMMARY_MIN_TOKENS", 1200))
    target = max(minimum, int(token_limit * ratio))
    maximum = _env_int("ALPHARAVIS_COMPRESSION_SUMMARY_MAX_TOKENS", 0)
    if maximum <= 0:
        return target
    return max(minimum, min(max(minimum, maximum), target))


def _summary_prompt_token_limit(token_limit: int) -> int:
    ratio = max(0.10, min(_env_float("ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_RATIO", 0.75), 0.95))
    minimum = max(2048, _env_int("ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_MIN_TOKENS", 8192))
    target = max(minimum, int(token_limit * ratio))
    maximum = _env_int("ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_MAX_TOKENS", 0)
    if maximum <= 0:
        return target
    return max(minimum, min(max(minimum, maximum), target))


def _summary_chunk_output_token_limit(token_limit: int) -> int:
    ratio = max(0.005, min(_env_float("ALPHARAVIS_COMPRESSION_SUMMARY_CHUNK_RATIO", 0.03), 0.20))
    minimum = max(300, _env_int("ALPHARAVIS_COMPRESSION_SUMMARY_CHUNK_MIN_TOKENS", 300))
    target = max(minimum, int(token_limit * ratio))
    maximum = _env_int("ALPHARAVIS_COMPRESSION_SUMMARY_CHUNK_MAX_TOKENS", 0)
    if maximum <= 0:
        return target
    return max(minimum, min(max(minimum, maximum), target))


def summary_budget_snapshot(token_limit: int) -> dict[str, Any]:
    """Return the dynamic summary budgets derived from the active compression limit."""

    token_limit = max(1, int(token_limit or 1))
    return {
        "token_limit": token_limit,
        "summary_output_tokens": _summary_token_limit(token_limit),
        "summary_output_ratio": max(0.05, min(_env_float("ALPHARAVIS_COMPRESSION_SUMMARY_RATIO", 0.20), 0.80)),
        "summary_output_max_tokens": _env_int("ALPHARAVIS_COMPRESSION_SUMMARY_MAX_TOKENS", 0),
        "summary_prompt_tokens": _summary_prompt_token_limit(token_limit),
        "summary_prompt_ratio": max(0.10, min(_env_float("ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_RATIO", 0.75), 0.95)),
        "summary_prompt_max_tokens": _env_int("ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_MAX_TOKENS", 0),
        "summary_chunk_output_tokens": _summary_chunk_output_token_limit(token_limit),
        "summary_chunk_ratio": max(0.005, min(_env_float("ALPHARAVIS_COMPRESSION_SUMMARY_CHUNK_RATIO", 0.03), 0.20)),
        "summary_chunk_max_tokens": _env_int("ALPHARAVIS_COMPRESSION_SUMMARY_CHUNK_MAX_TOKENS", 0),
        "max_tokens_zero_means_dynamic": True,
    }


def _summary_payload_char_budget(
    *,
    token_limit: int,
    prompt_overhead_tokens: int = 0,
) -> tuple[int, int, int]:
    prompt_token_limit = _summary_prompt_token_limit(token_limit)
    payload_token_limit = prompt_token_limit
    if prompt_overhead_tokens > 0:
        reserve = max(0, _env_int("ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_OVERHEAD_RESERVE_TOKENS", 512))
        payload_token_limit = max(512, prompt_token_limit - int(prompt_overhead_tokens) - reserve)
    chars_per_token = max(1.0, _env_float("ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_CHARS_PER_TOKEN", 2.0))
    return prompt_token_limit, payload_token_limit, max(1000, int(payload_token_limit * chars_per_token))


def _summary_prompt_overhead_tokens(
    *,
    mode: str,
    thread_id: str,
    thread_key: str,
    previous_summary: str | None,
    current_task_brief: str,
    latest_handoff_packet: str,
    memory_kernel_context: str,
    skill_context: str,
    compact_instructions: str = "",
) -> int:
    prompt = build_summary_prompt(
        mode=mode,
        thread_id=thread_id,
        thread_key=thread_key,
        previous_summary=previous_summary,
        current_task_brief=current_task_brief,
        latest_handoff_packet=latest_handoff_packet,
        memory_kernel_context=memory_kernel_context,
        skill_context=skill_context,
        compact_instructions=compact_instructions,
        pruned_middle_text="",
    )
    return estimate_tokens_rough_text(prompt)


def _truncate_summary_input_to_budget(
    text: str,
    *,
    token_limit: int,
    prompt_overhead_tokens: int = 0,
) -> tuple[str, dict[str, Any]]:
    text = str(text or "")
    original_tokens = estimate_tokens_rough_text(text)
    prompt_token_limit, payload_token_limit, max_chars = _summary_payload_char_budget(
        token_limit=token_limit,
        prompt_overhead_tokens=prompt_overhead_tokens,
    )
    if len(text) <= max_chars:
        return text, {
            "summary_prompt_pruned": False,
            "summary_prompt_original_tokens_estimate": original_tokens,
            "summary_prompt_tokens_estimate": original_tokens,
            "summary_prompt_token_limit": prompt_token_limit,
            "summary_prompt_payload_token_limit": payload_token_limit,
            "summary_prompt_overhead_tokens_estimate": prompt_overhead_tokens,
            "summary_prompt_original_chars": len(text),
            "summary_prompt_chars": len(text),
        }

    head_ratio = max(0.10, min(_env_float("ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_HEAD_RATIO", 0.55), 0.85))
    marker = (
        "\n\n[AlphaRavis summary input pruned: middle context was larger than the summary-model "
        "prompt budget. Exact raw messages are preserved in the archive; summarize from the kept "
        "head/tail evidence and mention that omitted middle details require archive lookup.]\n\n"
    )
    available = max(1000, max_chars - len(marker))
    head_chars = max(500, int(available * head_ratio))
    tail_chars = max(500, available - head_chars)
    truncated = text[:head_chars].rstrip() + marker + text[-tail_chars:].lstrip()
    return truncated, {
        "summary_prompt_pruned": True,
        "summary_prompt_original_tokens_estimate": original_tokens,
        "summary_prompt_tokens_estimate": estimate_tokens_rough_text(truncated),
        "summary_prompt_token_limit": prompt_token_limit,
        "summary_prompt_payload_token_limit": payload_token_limit,
        "summary_prompt_overhead_tokens_estimate": prompt_overhead_tokens,
        "summary_prompt_original_chars": len(text),
        "summary_prompt_chars": len(truncated),
        "summary_prompt_omitted_chars": max(0, len(text) - len(truncated)),
    }


def _chunk_summary_input(
    text: str,
    *,
    token_limit: int,
    prompt_overhead_tokens: int = 0,
) -> tuple[list[str], dict[str, Any]]:
    text = str(text or "")
    prompt_token_limit, payload_token_limit, chunk_chars = _summary_payload_char_budget(
        token_limit=token_limit,
        prompt_overhead_tokens=prompt_overhead_tokens,
    )
    overlap_chars = min(max(0, _env_int("ALPHARAVIS_COMPRESSION_SUMMARY_CHUNK_OVERLAP_CHARS", 1000)), chunk_chars - 1)
    max_chunks = max(1, _env_int("ALPHARAVIS_COMPRESSION_SUMMARY_MAX_CHUNKS", 12))
    chunks: list[str] = []
    start = 0
    covered_end = 0
    while start < len(text) and len(chunks) < max_chunks:
        end = min(len(text), start + chunk_chars)
        chunks.append(text[start:end])
        covered_end = max(covered_end, end)
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)
    omitted_chars = max(0, len(text) - covered_end)
    return chunks, {
        "summary_chunking_used": True,
        "summary_chunk_count": len(chunks),
        "summary_chunk_chars": chunk_chars,
        "summary_chunk_prompt_token_limit": prompt_token_limit,
        "summary_chunk_payload_token_limit": payload_token_limit,
        "summary_chunk_prompt_overhead_tokens": prompt_overhead_tokens,
        "summary_chunk_overlap_chars": overlap_chars,
        "summary_chunk_max_chunks": max_chunks,
        "summary_chunk_omitted_chars": omitted_chars,
    }


async def _summarize_chunked_input(
    *,
    summarize_fn: SummaryFn,
    mode: str,
    thread_id: str,
    thread_key: str,
    previous_summary: str | None,
    current_task_brief: str,
    latest_handoff_packet: str,
    memory_kernel_context: str,
    skill_context: str,
    pruned_middle_text: str,
    token_limit: int,
    compact_instructions: str = "",
    progress_callback: ProgressFn | None = None,
) -> tuple[str, dict[str, Any]]:
    chunk_prompt_overhead_tokens = _summary_prompt_overhead_tokens(
        mode=f"{mode}_chunk",
        thread_id=thread_id,
        thread_key=thread_key,
        previous_summary=previous_summary,
        current_task_brief=current_task_brief,
        latest_handoff_packet=latest_handoff_packet,
        memory_kernel_context=memory_kernel_context,
        skill_context=skill_context,
        compact_instructions=compact_instructions,
    )
    chunks, chunk_stats = _chunk_summary_input(
        pruned_middle_text,
        token_limit=token_limit,
        prompt_overhead_tokens=chunk_prompt_overhead_tokens,
    )
    if len(chunks) <= 1:
        prompt = build_summary_prompt(
            mode=mode,
            thread_id=thread_id,
            thread_key=thread_key,
            previous_summary=previous_summary,
            current_task_brief=current_task_brief,
            latest_handoff_packet=latest_handoff_packet,
            memory_kernel_context=memory_kernel_context,
            skill_context=skill_context,
            compact_instructions=compact_instructions,
            pruned_middle_text=chunks[0] if chunks else pruned_middle_text,
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "compression.synthesis.started",
                    "mode": mode,
                    "chunk_count": len(chunks),
                    "prompt_chars": len(prompt),
                    "max_tokens": _summary_token_limit(token_limit),
                }
            )
        synthesis_started = time.perf_counter()
        summary = await summarize_fn(prompt, _summary_token_limit(token_limit))
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "compression.synthesis.completed",
                    "mode": mode,
                    "chunk_count": len(chunks),
                    "elapsed_seconds": round(time.perf_counter() - synthesis_started, 3),
                    "summary_chars": len(str(summary or "")),
                }
            )
        return summary, {**chunk_stats, "summary_chunking_used": False}

    chunk_output_tokens = _summary_chunk_output_token_limit(token_limit)
    chunk_summaries: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_prompt = build_summary_prompt(
            mode=f"{mode}_chunk_{index}_of_{len(chunks)}",
            thread_id=thread_id,
            thread_key=thread_key,
            previous_summary=previous_summary if index == 1 else None,
            current_task_brief=current_task_brief,
            latest_handoff_packet=latest_handoff_packet,
            memory_kernel_context=memory_kernel_context,
            skill_context=skill_context,
            compact_instructions=compact_instructions,
            pruned_middle_text=(
                f"Summary chunk {index}/{len(chunks)}. Preserve concrete details from this chunk; "
                "a final synthesis pass will merge all chunk summaries.\n\n"
                f"{chunk}"
            ),
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "compression.chunk.started",
                    "mode": mode,
                    "chunk_number": index,
                    "chunk_count": len(chunks),
                    "source_chars": len(chunk),
                    "prompt_chars": len(chunk_prompt),
                    "max_tokens": chunk_output_tokens,
                }
            )
        chunk_started = time.perf_counter()
        chunk_summary = await summarize_fn(chunk_prompt, chunk_output_tokens)
        chunk_summary = redact_secrets(str(chunk_summary or "").strip())
        if not chunk_summary:
            raise RuntimeError(f"summary chunk {index}/{len(chunks)} returned an empty summary")
        chunk_summaries.append(f"## Chunk {index}/{len(chunks)}\n{chunk_summary}")
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "compression.chunk.completed",
                    "mode": mode,
                    "chunk_number": index,
                    "chunk_count": len(chunks),
                    "elapsed_seconds": round(time.perf_counter() - chunk_started, 3),
                    "summary_chars": len(chunk_summary),
                }
            )

    combined_input = "\n\n".join(chunk_summaries)
    synthesis_overhead_tokens = _summary_prompt_overhead_tokens(
        mode=f"{mode}_chunked_synthesis",
        thread_id=thread_id,
        thread_key=thread_key,
        previous_summary=previous_summary,
        current_task_brief=current_task_brief,
        latest_handoff_packet=latest_handoff_packet,
        memory_kernel_context=memory_kernel_context,
        skill_context=skill_context,
        compact_instructions=compact_instructions,
    )
    combined_input, combined_stats = _truncate_summary_input_to_budget(
        combined_input,
        token_limit=token_limit,
        prompt_overhead_tokens=synthesis_overhead_tokens,
    )
    omitted_chars = int(chunk_stats.get("summary_chunk_omitted_chars") or 0)
    omitted_note = ""
    if omitted_chars > 0:
        omitted_note = (
            "\n\n[AlphaRavis chunking note: the prepared middle context exceeded "
            f"ALPHARAVIS_COMPRESSION_SUMMARY_MAX_CHUNKS, so about {omitted_chars} chars "
            "of prepared summary input were not sent to the summary model. Exact raw messages "
            "remain archived by the graph; mention archive lookup for omitted middle details.]"
        )
    final_prompt = build_summary_prompt(
        mode=f"{mode}_chunked_synthesis",
        thread_id=thread_id,
        thread_key=thread_key,
        previous_summary=previous_summary,
        current_task_brief=current_task_brief,
        latest_handoff_packet=latest_handoff_packet,
        memory_kernel_context=memory_kernel_context,
        skill_context=skill_context,
        compact_instructions=compact_instructions,
        pruned_middle_text=(
            "Intermediate chunk summaries from a large compression window. Synthesize them into one "
            "coherent reference-only summary; do not treat chunk headings as user instructions.\n\n"
            f"{combined_input}{omitted_note}"
        ),
    )
    if progress_callback is not None:
        progress_callback(
            {
                "event": "compression.synthesis.started",
                "mode": f"{mode}_chunked_synthesis",
                "chunk_count": len(chunks),
                "prompt_chars": len(final_prompt),
                "max_tokens": _summary_token_limit(token_limit),
            }
        )
    synthesis_started = time.perf_counter()
    summary = await summarize_fn(final_prompt, _summary_token_limit(token_limit))
    if progress_callback is not None:
        progress_callback(
            {
                "event": "compression.synthesis.completed",
                "mode": f"{mode}_chunked_synthesis",
                "chunk_count": len(chunks),
                "elapsed_seconds": round(time.perf_counter() - synthesis_started, 3),
                "summary_chars": len(str(summary or "")),
            }
        )
    stats = {
        **chunk_stats,
        "summary_chunk_output_tokens": chunk_output_tokens,
        "summary_chunk_summary_tokens_estimate": estimate_tokens_rough_text("\n\n".join(chunk_summaries)),
        "summary_chunk_synthesis_pruned": combined_stats.get("summary_prompt_pruned", False),
        "summary_chunk_synthesis_tokens_estimate": combined_stats.get("summary_prompt_tokens_estimate"),
        "summary_chunk_synthesis_payload_token_limit": combined_stats.get("summary_prompt_payload_token_limit"),
        "summary_chunk_synthesis_prompt_overhead_tokens": synthesis_overhead_tokens,
    }
    return summary, stats


def default_compression_stats() -> dict[str, Any]:
    return {
        "last_compression_savings_pct": None,
        "ineffective_compression_count": 0,
        "summary_failure_cooldown_until": 0.0,
        "last_summary_error": "",
        "last_summary_failed_at": 0.0,
        "last_summary_fallback_used": False,
    }


def _normalized_stats(stats: dict[str, Any] | None) -> dict[str, Any]:
    normalized = default_compression_stats()
    if isinstance(stats, dict):
        normalized.update(stats)
    return normalized


def should_compress(
    *,
    token_estimate: int,
    token_limit: int,
    compression_stats: dict[str, Any] | None = None,
    force: bool = False,
    now: float | None = None,
) -> CompressionDecision:
    if token_estimate <= token_limit and not force:
        return CompressionDecision(False, "under_token_limit")
    if force:
        return CompressionDecision(True, "forced")

    stats = _normalized_stats(compression_stats)
    now = time.time() if now is None else now
    cooldown_until = float(stats.get("summary_failure_cooldown_until") or 0)
    if cooldown_until > now:
        return CompressionDecision(False, "summary_failure_cooldown")

    if _env_bool("ALPHARAVIS_COMPRESSION_ANTI_THRASHING_ENABLED", "true"):
        ineffective_count = int(stats.get("ineffective_compression_count") or 0)
        if ineffective_count >= 2:
            return CompressionDecision(False, "anti_thrashing")

    return CompressionDecision(True, "over_token_limit")


def _update_compression_stats(
    *,
    previous: dict[str, Any] | None,
    token_estimate_before: int,
    token_estimate_after: int,
    summary_failed: bool,
    summary_error: str = "",
    now: float | None = None,
) -> dict[str, Any]:
    now = time.time() if now is None else now
    stats = _normalized_stats(previous)
    if summary_failed:
        cooldown = max(0, _env_int("ALPHARAVIS_COMPRESSION_FAILURE_COOLDOWN_SECONDS", 600))
        stats.update(
            {
                "summary_failure_cooldown_until": now + cooldown,
                "last_summary_error": summary_error[:500],
                "last_summary_failed_at": now,
                "last_summary_fallback_used": True,
            }
        )
        return stats

    savings_ratio = 0.0
    if token_estimate_before > 0:
        savings_ratio = max(0.0, (token_estimate_before - token_estimate_after) / token_estimate_before)
    min_savings = max(0.0, min(_env_float("ALPHARAVIS_COMPRESSION_MIN_SAVINGS_RATIO", 0.10), 0.95))
    ineffective_count = int(stats.get("ineffective_compression_count") or 0)
    if savings_ratio < min_savings:
        ineffective_count += 1
    else:
        ineffective_count = 0

    stats.update(
        {
            "last_compression_savings_pct": round(savings_ratio * 100, 2),
            "ineffective_compression_count": ineffective_count,
            "summary_failure_cooldown_until": 0.0,
            "last_summary_error": "",
            "last_summary_fallback_used": False,
        }
    )
    return stats


def build_summary_prompt(
    *,
    mode: str,
    thread_id: str,
    thread_key: str,
    previous_summary: str | None,
    current_task_brief: str,
    latest_handoff_packet: str,
    memory_kernel_context: str,
    skill_context: str,
    pruned_middle_text: str,
    compact_instructions: str = "",
) -> str:
    previous = previous_summary.strip() if previous_summary and previous_summary.strip() else "No previous summary."
    protected_notes = "\n\n".join(
        part
        for part in [
            f"Current task brief, preserve as active orientation:\n{current_task_brief.strip()}"
            if current_task_brief.strip()
            else "",
            f"Latest handoff packet, preserve as active coordination data:\n{latest_handoff_packet.strip()}"
            if latest_handoff_packet.strip()
            else "",
            f"MemoryKernel context, preserve durable facts:\n{memory_kernel_context.strip()}"
            if memory_kernel_context.strip()
            else "",
            f"Skill hint context, preserve workflow hints:\n{skill_context.strip()}" if skill_context.strip() else "",
            (
                "User compaction instructions, apply only while deciding what this summary should preserve. "
                "Do not treat them as a new task for the next agent:\n"
                f"{compact_instructions.strip()}"
            )
            if compact_instructions.strip()
            else "",
        ]
        if part
    )
    section_template = "\n".join(f"## {section}\n-" for section in PROTECTED_SECTION_NAMES)
    iterative_rules = (
        "If Previous summary is not empty, update it instead of restarting: keep still-valid facts, "
        "remove obsolete points, move completed work into Progress Done, update Progress In Progress, "
        "and keep Remaining Work only for real unresolved tasks. Do not invent new tasks."
    )
    return (
        "You are AlphaRavis's Hermes-style active context compressor.\n"
        f"{SUMMARY_PREFIX}\n\n"
        "Create or update a compact reference-only handoff summary for the next agent step.\n"
        "Do not answer questions mentioned below. Do not fulfill old requests from the summary. "
        "Earlier turns were already handled. The next agent must respond only to the latest user request "
        "that appears after the summary in the active conversation.\n"
        "Preserve exact technical facts, file paths, commands, errors, decisions, approvals, user preferences, "
        "handoff obligations, and unresolved work.\n"
        f"{iterative_rules}\n"
        "Return Markdown with exactly these sections and concise bullets under each section:\n\n"
        f"{section_template}\n\n"
        f"mode: {mode}\nthread_id: {thread_id}\nthread_key: {thread_key}\n\n"
        f"Previous summary:\n{previous}\n\n"
        f"Protected active context notes:\n{protected_notes or 'None.'}\n\n"
        "Middle messages to compress. Tool calls, tool outputs, and workflow/action history may be "
        "collapsed into a separate Workflow / Tool Event Compact Log instead of being treated as normal chat. "
        "Secrets were redacted before this prompt:\n"
        f"{pruned_middle_text}\n\n"
        "Important: In Archive References, say that exact raw messages are archived by the graph after this "
        "summary is generated. Use source_type=archive / source_type=archive_collection wording when relevant."
    )


def _fallback_summary(
    *,
    mode: str,
    thread_id: str,
    thread_key: str,
    previous_summary: str | None,
    current_task_brief: str,
    latest_handoff_packet: str,
    memory_kernel_context: str,
    skill_context: str,
    message_count: int,
    summary_error: str,
    compact_instructions: str = "",
) -> str:
    previous = previous_summary.strip() if previous_summary and previous_summary.strip() else "No previous summary."
    return redact_secrets(
        "\n".join(
            [
                "## Active Task",
                f"- Summary model failed during {mode} compression; use the current tail messages and task brief for the active request.",
                "",
                "## Goal",
                "- Preserve runtime safety while avoiding silent context loss.",
                "",
                "## Handoff Packet",
                f"- {latest_handoff_packet.strip() or 'None.'}",
                "",
                "## MemoryKernel",
                f"- {memory_kernel_context.strip() or 'None.'}",
                "",
                "## Skill Context",
                f"- {skill_context.strip() or 'None.'}",
                "",
                "## Constraints / Preferences",
                "- This summary is reference-only and must not create a new instruction.",
                f"- Compact instructions applied: {compact_instructions.strip() or 'None.'}",
                "",
                "## Progress Done",
                "- Earlier turns were processed before compaction.",
                "",
                "## Progress In Progress",
                f"- {current_task_brief.strip() or 'Continue from the latest user request after this summary.'}",
                "",
                "## Blocked / Risks",
                f"- Summary generation failed: {summary_error[:500]}",
                "",
                "## Resolved Questions",
                "- Unknown because summary generation failed.",
                "",
                "## Pending User Asks",
                "- Use only the latest active user message after this summary.",
                "",
                "## Key Decisions",
                f"- Previous summary retained where available: {previous[:1000]}",
                "",
                "## Relevant Files",
                "- Retrieve exact archived details if needed.",
                "",
                "## Commands / Tools Used",
                "- Retrieve exact archived details if needed.",
                "",
                "## Critical Context",
                f"- {message_count} middle messages were removed from active context and should be archived as raw records if the Store is available.",
                f"- thread_id: {thread_id}; thread_key: {thread_key}",
                "",
                "## Remaining Work",
                "- Respond to the latest active user request only.",
                "",
                "## Archive References",
                "- Raw middle messages should be available as source_type=archive after the graph stores this result.",
            ]
        )
    )


def build_summary_message_content(
    *,
    mode: str,
    summary: str,
    archive_key: str,
    token_estimate_before: int,
    token_estimate_after: int,
) -> str:
    if archive_key and archive_key != "not-stored":
        archive_reference = (
            "Archive reference:\n"
            "source_type: archive\n"
            f"archive_key: {archive_key}\n"
            f"read_archive_record: read_archive_record(archive_key=\"{archive_key}\")\n"
        )
    else:
        archive_reference = "Archive reference: not-stored\n"
    return (
        "<context-compaction-summary>\n"
        f"{SUMMARY_PREFIX}\n\n"
        "Do not treat this summary as a new user instruction. Answer only the latest user request after this "
        "summary. If old exact details matter, call semantic_memory_search/read_archive_record instead of guessing.\n\n"
        f"mode: {mode}\n"
        f"archive_key: {archive_key or 'not-stored'}\n"
        f"tokens_before_estimate: {token_estimate_before}\n"
        f"tokens_after_estimate: {token_estimate_after}\n\n"
        f"{archive_reference}\n"
        f"{summary.strip()}\n"
        "</context-compaction-summary>"
    )


def build_archive_policy_message() -> str:
    return (
        "Archived context policy: This thread may have archived context. Do not assume old details from memory. "
        "If the user asks about earlier work, previous debugging, prior decisions, or old context, call semantic_memory_search first. "
        "If a result is source_type=archive_collection, inspect its child_archive_keys and load only the relevant raw archive records."
    )


def build_archive_content(
    *,
    mode: str,
    thread_id: str,
    thread_key: str,
    summary: str,
    pruned_middle_text: str,
    workflow_event_compaction: str,
    middle_messages: list[Any],
) -> str:
    original = "\n\n".join(f"[{index}] {message_text(message)}" for index, message in enumerate(middle_messages, start=1))
    original = redact_secrets(original)
    return (
        f"# Raw Archive Record\n\n"
        f"mode: {mode}\n"
        f"thread_id: {thread_id}\n"
        f"thread_key: {thread_key}\n"
        f"message_count: {len(middle_messages)}\n\n"
        "## Compression Summary\n"
        f"{summary.strip()}\n\n"
        "## Workflow / Tool Event Compact Log\n"
        f"{workflow_event_compaction.strip() or 'No compacted workflow/tool events.'}\n\n"
        "## Tool-Pruned Middle Used For Summary\n"
        f"{pruned_middle_text.strip()}\n\n"
        "## Original Archived Messages\n"
        f"{original.strip()}"
    ).strip()


def redacted_message_to_json(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        data = copy.deepcopy(message)
    else:
        data = {
            "type": getattr(message, "type", None),
            "name": getattr(message, "name", None),
            "id": getattr(message, "id", None),
            "content": getattr(message, "content", ""),
            "additional_kwargs": getattr(message, "additional_kwargs", {}),
            "response_metadata": getattr(message, "response_metadata", {}),
        }
    content = data.get("content")
    if isinstance(content, str):
        data["content"] = redact_secrets(content)
    elif isinstance(content, list):
        redacted_blocks = []
        for block in content:
            if isinstance(block, dict):
                block_copy = copy.deepcopy(block)
                for key in ("text", "content", "input_text"):
                    if isinstance(block_copy.get(key), str):
                        block_copy[key] = redact_secrets(block_copy[key])
                redacted_blocks.append(block_copy)
            else:
                redacted_blocks.append(redact_secrets(str(block)))
        data["content"] = redacted_blocks
    data["archive_redacted"] = True
    return data


def _skipped_result(
    *,
    mode: str,
    thread_id: str,
    thread_key: str,
    token_limit: int,
    token_estimate_before: int,
    previous_summary: str | None,
    reason: str,
    compression_stats: dict[str, Any] | None,
) -> CompressionResult:
    return CompressionResult(
        mode=mode,
        thread_id=thread_id,
        thread_key=thread_key,
        token_limit=token_limit,
        token_estimate_before=token_estimate_before,
        token_estimate_after=token_estimate_before,
        head=[],
        middle=[],
        tail=[],
        summary=previous_summary or "",
        summary_message_content="",
        archive_content="",
        archive_metadata={},
        pruned_middle_text="",
        skipped=True,
        reason=reason,
        compression_stats=_normalized_stats(compression_stats),
    )


async def compress_messages(
    messages: list[Any],
    *,
    mode: str,
    thread_id: str,
    thread_key: str,
    token_limit: int,
    previous_summary: str | None,
    summary_context_token_limit: int | None = None,
    current_task_brief: str = "",
    latest_handoff_packet: str = "",
    memory_kernel_context: str = "",
    skill_context: str = "",
    compact_instructions: str = "",
    protected_message_ids: set[str] | None = None,
    summarize_fn: SummaryFn,
    force: bool = False,
    compression_stats: dict[str, Any] | None = None,
    enable_chunked_summary: bool | None = None,
    progress_callback: ProgressFn | None = None,
) -> CompressionResult:
    token_estimate_before = estimate_tokens_rough(messages)
    progress_events: list[dict[str, Any]] = []
    compression_started = time.perf_counter()

    def emit_progress(event: dict[str, Any]) -> None:
        item = dict(event)
        item.setdefault("mode", mode)
        item.setdefault("t", round(time.perf_counter() - compression_started, 3))
        progress_events.append(item)
        if progress_callback is not None:
            progress_callback(item)

    emit_progress(
        {
            "event": "compression.started",
            "token_estimate_before": token_estimate_before,
            "token_limit": token_limit,
            "summary_context_token_limit": int(summary_context_token_limit or token_limit),
            "force": bool(force),
        }
    )
    decision = should_compress(
        token_estimate=token_estimate_before,
        token_limit=token_limit,
        compression_stats=compression_stats,
        force=force,
    )
    if not decision.should_run:
        emit_progress({"event": "compression.skipped", "reason": decision.reason})
        return _skipped_result(
            mode=mode,
            thread_id=thread_id,
            thread_key=thread_key,
            token_limit=token_limit,
            token_estimate_before=token_estimate_before,
            previous_summary=previous_summary,
            reason=decision.reason,
            compression_stats=compression_stats,
        )

    selection = select_head_middle_tail(
        messages,
        token_limit=token_limit,
        protected_message_ids=protected_message_ids or set(),
    )
    if not selection.middle:
        emit_progress({"event": "compression.skipped", "reason": "no_middle_messages"})
        return _skipped_result(
            mode=mode,
            thread_id=thread_id,
            thread_key=thread_key,
            token_limit=token_limit,
            token_estimate_before=token_estimate_before,
            previous_summary=previous_summary,
            reason="no_middle_messages",
            compression_stats=compression_stats,
        )

    prep = prepare_messages_for_summary(selection.middle)
    if prep.workflow_event_count:
        emit_progress(
            {
                "event": "compression.workflow_events.compacted",
                "scope": mode,
                "workflow_event_count": prep.workflow_event_count,
                "workflow_tool_call_count": prep.workflow_tool_call_count,
                "workflow_tool_result_count": prep.workflow_tool_result_count,
                "workflow_event_chars": prep.workflow_event_chars,
                "pruned_tool_count": prep.pruned_tool_count,
                "deduped_tool_count": prep.deduped_tool_count,
                "tool_args_truncated_count": prep.tool_args_truncated_count,
            }
        )
    summary_token_limit = max(1, int(summary_context_token_limit or token_limit))
    summary_input_text, summary_prompt_stats = _truncate_summary_input_to_budget(
        prep.text,
        token_limit=summary_token_limit,
    )
    chunking_enabled = (
        _env_bool("ALPHARAVIS_COMPRESSION_ENABLE_CHUNKED_SUMMARY", "false")
        if enable_chunked_summary is None
        else bool(enable_chunked_summary)
    )
    chunking_forced_by_oversized_tail = bool(selection.oversized_tail_force_latest_user_to_middle)
    chunking_enabled = chunking_enabled or chunking_forced_by_oversized_tail
    middle_token_estimate = estimate_tokens_rough(selection.middle)
    chunking_will_run = bool(chunking_enabled and summary_prompt_stats.get("summary_prompt_pruned"))
    emit_progress(
        {
            "event": "compression.precompact",
            "scope": mode,
            "reason": decision.reason,
            "token_estimate_before": token_estimate_before,
            "token_limit": token_limit,
            "token_pressure": round(token_estimate_before / max(1, token_limit), 4),
            "summary_context_token_limit": summary_token_limit,
            "head_message_count": len(selection.head),
            "middle_message_count": len(selection.middle),
            "tail_message_count": len(selection.tail),
            "head_indexes": selection.head_indexes,
            "middle_indexes": selection.middle_indexes,
            "tail_indexes": selection.tail_indexes,
            "middle_token_estimate": middle_token_estimate,
            "summary_prompt_pruned": bool(summary_prompt_stats.get("summary_prompt_pruned")),
            "chunking_enabled": bool(chunking_enabled),
            "chunking_will_run": chunking_will_run,
            "chunking_forced_by_oversized_tail": chunking_forced_by_oversized_tail,
            "oversized_tail_rebalanced": bool(selection.oversized_tail_rebalanced),
        }
    )

    summary_failed = False
    summary_error = ""
    try:
        if chunking_enabled and bool(summary_prompt_stats.get("summary_prompt_pruned")):
            summary, chunk_stats = await _summarize_chunked_input(
                summarize_fn=summarize_fn,
                mode=mode,
                thread_id=thread_id,
                thread_key=thread_key,
                previous_summary=previous_summary,
                current_task_brief=current_task_brief,
                latest_handoff_packet=latest_handoff_packet,
                memory_kernel_context=memory_kernel_context,
                skill_context=skill_context,
                pruned_middle_text=prep.text,
                token_limit=summary_token_limit,
                compact_instructions=compact_instructions,
                progress_callback=emit_progress,
            )
            summary_prompt_stats.update(chunk_stats)
        else:
            prompt = build_summary_prompt(
                mode=mode,
                thread_id=thread_id,
                thread_key=thread_key,
                previous_summary=previous_summary,
                current_task_brief=current_task_brief,
                latest_handoff_packet=latest_handoff_packet,
                memory_kernel_context=memory_kernel_context,
                skill_context=skill_context,
                pruned_middle_text=summary_input_text,
                compact_instructions=compact_instructions,
            )
            emit_progress(
                {
                    "event": "compression.synthesis.started",
                    "summary_mode": "one_shot",
                    "prompt_chars": len(prompt),
                    "max_tokens": _summary_token_limit(summary_token_limit),
                }
            )
            synthesis_started = time.perf_counter()
            summary = await summarize_fn(prompt, _summary_token_limit(summary_token_limit))
            emit_progress(
                {
                    "event": "compression.synthesis.completed",
                    "summary_mode": "one_shot",
                    "elapsed_seconds": round(time.perf_counter() - synthesis_started, 3),
                    "summary_chars": len(str(summary or "")),
                }
            )
            summary_prompt_stats.setdefault("summary_chunking_used", False)
        summary = redact_secrets(str(summary or "").strip())
        if not summary:
            raise RuntimeError("summary model returned an empty summary")
    except Exception as exc:
        summary_failed = True
        summary_error = str(exc)
        emit_progress({"event": "compression.synthesis.failed", "error": summary_error[:500]})
        summary = _fallback_summary(
            mode=mode,
            thread_id=thread_id,
            thread_key=thread_key,
            previous_summary=previous_summary,
            current_task_brief=current_task_brief,
            latest_handoff_packet=latest_handoff_packet,
            memory_kernel_context=memory_kernel_context,
            skill_context=skill_context,
            message_count=len(selection.middle),
            summary_error=summary_error,
            compact_instructions=compact_instructions,
        )

    summary_shell = build_summary_message_content(
        mode=mode,
        summary=summary,
        archive_key="pending",
        token_estimate_before=token_estimate_before,
        token_estimate_after=estimate_tokens_rough([*selection.head, *selection.tail]) + estimate_tokens_rough(summary),
    )
    token_estimate_after = estimate_tokens_rough([*selection.head, *selection.tail]) + estimate_tokens_rough(summary_shell)
    stats = _update_compression_stats(
        previous=compression_stats,
        token_estimate_before=token_estimate_before,
        token_estimate_after=token_estimate_after,
        summary_failed=summary_failed,
        summary_error=summary_error,
    )
    archive_content = build_archive_content(
        mode=mode,
        thread_id=thread_id,
        thread_key=thread_key,
        summary=summary,
        pruned_middle_text=prep.text,
        workflow_event_compaction=prep.workflow_event_compaction,
        middle_messages=selection.middle,
    )
    archive_metadata = {
        "thread_id": thread_id,
        "thread_key": thread_key,
        "compression_mode": mode,
        "compression_token_limit": token_limit,
        "summary_context_token_limit": summary_token_limit,
        "message_count": len(selection.middle),
        "token_estimate_before": token_estimate_before,
        "token_estimate_after": token_estimate_after,
        "middle_token_estimate": middle_token_estimate,
        "head_message_count": len(selection.head),
        "tail_message_count": len(selection.tail),
        "head_indexes": selection.head_indexes,
        "middle_indexes": selection.middle_indexes,
        "tail_indexes": selection.tail_indexes,
        "oversized_tail_rebalanced": selection.oversized_tail_rebalanced,
        "oversized_tail_tokens_before": selection.oversized_tail_tokens_before,
        "oversized_tail_token_target": selection.oversized_tail_token_target,
        "oversized_tail_moved_indexes": selection.oversized_tail_moved_indexes,
        "oversized_tail_force_latest_user_to_middle": selection.oversized_tail_force_latest_user_to_middle,
        "oversized_tail_force_middle_target": selection.oversized_tail_force_middle_target,
        "summary_failed": summary_failed,
        "summary_error": summary_error[:500],
        "compact_instructions": compact_instructions.strip(),
        "compact_instructions_chars": len(compact_instructions.strip()),
        **summary_prompt_stats,
        "summary_chunking_forced_by_oversized_tail": chunking_forced_by_oversized_tail,
        "pruned_tool_count": prep.pruned_tool_count,
        "deduped_tool_count": prep.deduped_tool_count,
        "tool_args_truncated_count": prep.tool_args_truncated_count,
        "workflow_event_count": prep.workflow_event_count,
        "workflow_tool_call_count": prep.workflow_tool_call_count,
        "workflow_tool_result_count": prep.workflow_tool_result_count,
        "workflow_event_chars": prep.workflow_event_chars,
        "workflow_event_compaction": prep.workflow_event_compaction,
        "compression_stats": stats,
        "events": progress_events,
    }
    emit_progress(
        {
            "event": "compression.completed",
            "summary_failed": summary_failed,
            "summary_chunking_used": summary_prompt_stats.get("summary_chunking_used", False),
            "summary_chunk_count": summary_prompt_stats.get("summary_chunk_count", 0),
            "token_estimate_after": token_estimate_after,
        }
    )
    archive_metadata["events"] = progress_events

    return CompressionResult(
        mode=mode,
        thread_id=thread_id,
        thread_key=thread_key,
        token_limit=token_limit,
        token_estimate_before=token_estimate_before,
        token_estimate_after=token_estimate_after,
        head=selection.head,
        middle=selection.middle,
        tail=selection.tail,
        summary=summary,
        summary_message_content=summary_shell,
        archive_content=archive_content,
        archive_metadata=archive_metadata,
        pruned_middle_text=prep.text,
        summary_failed=summary_failed,
        summary_error=summary_error,
        compression_stats=stats,
        pruned_tool_count=prep.pruned_tool_count,
        deduped_tool_count=prep.deduped_tool_count,
        tool_args_truncated_count=prep.tool_args_truncated_count,
        workflow_event_count=prep.workflow_event_count,
        workflow_tool_call_count=prep.workflow_tool_call_count,
        workflow_tool_result_count=prep.workflow_tool_result_count,
        workflow_event_chars=prep.workflow_event_chars,
    )


# ---------- ratio token limit utilities (used by agent_graph) ----------


def ratio_token_limit(
    context_length: int,
    *,
    ratio_env: str,
    fixed_env: str,
    default_fixed: str = "30000",
    default_ratio: float = 0.50,
) -> int:
    """Return a token limit as min(fixed_value, ratio * context_length)."""
    fixed_value = int(os.getenv(fixed_env, default_fixed))
    if fixed_value:
        ratio_value = int(context_length * _env_float(ratio_env, default_ratio))
        return min(fixed_value, ratio_value)
    return int(context_length * _env_float(ratio_env, default_ratio))


def ratio_token_limit_for_context(
    context_length: int,
    *,
    ratio_env: str,
    fixed_env: str,
    fixed_default: str = "30000",
    default_ratio: float = 0.50,
) -> int:
    """Return a context-scoped token limit. Uses fixed value only if explicitly set."""
    if os.getenv(fixed_env):
        return ratio_token_limit(
            context_length,
            ratio_env=ratio_env,
            fixed_env=fixed_env,
            default_fixed=fixed_default,
            default_ratio=default_ratio,
        )
    return ratio_token_limit(
        context_length,
        ratio_env=ratio_env,
        fixed_env="",
        default_ratio=default_ratio,
    )


def effective_context_limit(limit: int, reserve_tokens: int) -> int:
    return max(4096, limit - reserve_tokens)


def message_for_context_estimate(message: Any) -> dict[str, Any]:
    """Strip usage/reasoning metadata from a message for token estimation."""
    data = _message_mapping(message)
    for key in ("usage", "token_usage", "usage_metadata", "response_metadata"):
        data.pop(key, None)
    additional = data.get("additional_kwargs")
    if isinstance(additional, dict):
        additional = dict(additional)
        for key in ("reasoning", "reasoning_content", "usage", "token_usage", "usage_metadata", "response_metadata"):
            additional.pop(key, None)
        data["additional_kwargs"] = additional
    return data
