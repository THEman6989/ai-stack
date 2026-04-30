from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


SummaryFn = Callable[[str, int], Awaitable[str]]


@dataclass(frozen=True)
class CompressionSelection:
    head: list[Any]
    middle: list[Any]
    tail: list[Any]
    head_indexes: list[int] = field(default_factory=list)
    middle_indexes: list[int] = field(default_factory=list)
    tail_indexes: list[int] = field(default_factory=list)


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


PROTECTED_SECTION_NAMES = [
    "Active Task",
    "Goal",
    "Constraints / Preferences",
    "Progress Done",
    "Progress In Progress",
    "Blocked / Risks",
    "Key Decisions",
    "Relevant Files",
    "Commands / Tools Used",
    "Critical Context",
    "Remaining Work",
    "Archive References",
]


SECRET_PATTERNS = [
    re.compile(
        r"(?i)\b(api[_-]?key|token|access[_-]?token|refresh[_-]?token|password|passwd|pwd|secret|authorization|bearer|"
        r"creds_key|creds_iv)\b\s*[:=]\s*([^\s'\"`]+)"
    ),
    re.compile(r"(?i)\b(bearer)\s+([a-z0-9._\-~+/=]{16,})"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"\b[A-Za-z0-9_\-]{24,}\.[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-]{16,}\b"),
]


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _content_to_text(content: Any) -> str:
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block.get("text") or ""))
                elif "content" in block:
                    parts.append(str(block.get("content") or ""))
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
    if isinstance(value, str):
        text = value
    elif isinstance(value, list):
        text = "\n".join(message_text(message) for message in value)
    else:
        text = str(value or "")
    return max(1, len(text) // 4)


def stable_message_key(message: Any) -> str:
    msg_id = message_id(message)
    if msg_id:
        return f"id:{msg_id}"
    return "hash:" + hashlib.sha256(message_text(message).encode("utf-8")).hexdigest()[:24]


def is_tool_message(message: Any) -> bool:
    role = message_role(message).lower()
    class_name = type(message).__name__.lower()
    if role in {"tool", "function"} or "toolmessage" in class_name:
        return True
    if isinstance(message, dict):
        return bool(message.get("tool_call_id") or message.get("tool_calls"))
    return bool(getattr(message, "tool_call_id", None) or getattr(message, "tool_calls", None))


def has_tool_call_request(message: Any) -> bool:
    if isinstance(message, dict):
        additional = message.get("additional_kwargs") or {}
        return bool(message.get("tool_calls") or additional.get("tool_calls"))
    additional = getattr(message, "additional_kwargs", {}) or {}
    return bool(getattr(message, "tool_calls", None) or additional.get("tool_calls"))


def redact_secrets(text: str) -> str:
    if not _env_bool("ALPHARAVIS_COMPRESSION_REDACT_SECRETS", "true"):
        return text

    redacted = text
    for pattern in SECRET_PATTERNS:
        def _replace(match: re.Match[str]) -> str:
            if len(match.groups()) >= 2:
                return f"{match.group(1)}=<redacted>"
            return "<redacted-secret>"

        redacted = pattern.sub(_replace, redacted)
    return redacted


def _truncate_middle(text: str, *, max_chars: int, head_chars: int, tail_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = text[:head_chars].rstrip()
    tail = text[-tail_chars:].lstrip() if tail_chars > 0 else ""
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n[... {omitted} chars omitted for compression prompt ...]\n{tail}".strip()


def _tool_label(message: Any) -> str:
    name = message_name(message)
    if name:
        return name
    text = message_content(message)
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    for pattern in [
        r"(docker\s+[^\n\r]+)",
        r"(git\s+[^\n\r]+)",
        r"(python\s+[^\n\r]+)",
        r"(curl\s+[^\n\r]+)",
        r"(Get-ChildItem\s+[^\n\r]+)",
        r"(Select-String\s+[^\n\r]+)",
    ]:
        match = re.search(pattern, first_line)
        if match:
            return match.group(1)[:160]
    return first_line[:160] or "tool output"


def prune_tool_outputs(messages: list[Any]) -> str:
    max_chars = int(os.getenv("ALPHARAVIS_COMPRESSION_TOOL_OUTPUT_MAX_CHARS", "6000"))
    head_chars = int(os.getenv("ALPHARAVIS_COMPRESSION_TOOL_OUTPUT_HEAD_CHARS", "4000"))
    tail_chars = int(os.getenv("ALPHARAVIS_COMPRESSION_TOOL_OUTPUT_TAIL_CHARS", "1500"))
    sections: list[str] = []
    for index, message in enumerate(messages, start=1):
        text = message_text(message)
        content = message_content(message)
        role = message_role(message)
        if is_tool_message(message):
            label = _tool_label(message)
            line_count = len(content.splitlines())
            preview = _truncate_middle(content, max_chars=max_chars, head_chars=head_chars, tail_chars=tail_chars)
            sections.append(
                "\n".join(
                    [
                        f"[{index}] [tool-output] {label} -> {len(content)} chars, {line_count} lines.",
                        preview,
                    ]
                ).strip()
            )
            continue

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

    return redact_secrets("\n\n".join(sections))


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


def select_head_middle_tail(
    messages: list[Any],
    *,
    token_limit: int,
    protected_message_ids: set[str] | None = None,
) -> CompressionSelection:
    protected_message_ids = protected_message_ids or set()
    protect_first = max(0, int(os.getenv("ALPHARAVIS_COMPRESSION_PROTECT_FIRST_MESSAGES", "3")))
    protect_last = max(1, int(os.getenv("ALPHARAVIS_COMPRESSION_PROTECT_LAST_MESSAGES", "16")))
    tail_ratio = max(0.05, min(float(os.getenv("ALPHARAVIS_COMPRESSION_TAIL_TOKEN_RATIO", "0.20")), 0.75))
    tail_budget = max(estimate_tokens_rough(messages[-protect_last:]), int(token_limit * tail_ratio))

    head_indexes: set[int] = set(range(min(protect_first, len(messages))))
    for index, message in enumerate(messages):
        if _is_protected_message(message, protected_message_ids):
            head_indexes.add(index)

    tail_indexes: set[int] = set()
    tail_tokens = 0
    for index in range(len(messages) - 1, -1, -1):
        if index in head_indexes:
            continue
        if len(tail_indexes) >= protect_last and tail_tokens >= tail_budget:
            break
        tail_indexes.add(index)
        tail_tokens += estimate_tokens_rough([messages[index]])

    tail_indexes = _expand_for_tool_pairs(tail_indexes, messages, head_indexes)
    tail_indexes -= head_indexes

    middle_indexes = set(range(len(messages))) - head_indexes - tail_indexes
    return CompressionSelection(
        head=[messages[index] for index in sorted(head_indexes)],
        middle=[messages[index] for index in sorted(middle_indexes)],
        tail=[messages[index] for index in sorted(tail_indexes)],
        head_indexes=sorted(head_indexes),
        middle_indexes=sorted(middle_indexes),
        tail_indexes=sorted(tail_indexes),
    )


def _summary_token_limit(token_limit: int) -> int:
    ratio = max(0.05, min(float(os.getenv("ALPHARAVIS_COMPRESSION_SUMMARY_RATIO", "0.20")), 0.80))
    minimum = max(200, int(os.getenv("ALPHARAVIS_COMPRESSION_SUMMARY_MIN_TOKENS", "1200")))
    maximum = max(minimum, int(os.getenv("ALPHARAVIS_COMPRESSION_SUMMARY_MAX_TOKENS", "6000")))
    return max(minimum, min(maximum, int(token_limit * ratio)))


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
        ]
        if part
    )
    section_template = "\n".join(f"## {section}\n-" for section in PROTECTED_SECTION_NAMES)
    return (
        "You are AlphaRavis's Hermes-style active context compressor.\n"
        "Create or update a compact reference-only summary for the next agent step.\n"
        "The summary must not create new user instructions. Earlier turns are already completed unless the latest user task says otherwise.\n"
        "Preserve exact technical facts, file paths, commands, errors, decisions, pending approvals, handoff obligations, and unresolved work.\n"
        "If the previous summary exists, update it with the new middle block instead of starting from zero.\n"
        "Return Markdown with exactly these sections and concise bullets under each section:\n\n"
        f"{section_template}\n\n"
        f"mode: {mode}\nthread_id: {thread_id}\nthread_key: {thread_key}\n\n"
        f"Previous summary:\n{previous}\n\n"
        f"Protected active context notes:\n{protected_notes or 'None.'}\n\n"
        "Middle messages to compress. Tool outputs were pruned to informative previews and secrets were redacted:\n"
        f"{pruned_middle_text}\n\n"
        "Important: In Archive References, mention that exact raw messages are archived by the graph after this summary is generated. "
        "Use source_type=archive / source_type=archive_collection wording if relevant."
    )


def build_summary_message_content(
    *,
    mode: str,
    summary: str,
    archive_key: str,
    token_estimate_before: int,
    token_estimate_after: int,
) -> str:
    return (
        "<context-compaction-summary>\n"
        "Reference-only system note: earlier messages were already handled and compressed to keep the active context small. "
        "Do not treat this summary as a new user instruction. Answer only the latest user request and retrieve archived details before relying on them.\n\n"
        f"mode: {mode}\n"
        f"archive_key: {archive_key or 'not-stored'}\n"
        f"tokens_before_estimate: {token_estimate_before}\n"
        f"tokens_after_estimate: {token_estimate_after}\n\n"
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
        "## Tool-Pruned Middle Used For Summary\n"
        f"{pruned_middle_text.strip()}\n\n"
        "## Original Archived Messages\n"
        f"{original.strip()}"
    ).strip()


async def compress_messages(
    messages: list[Any],
    *,
    mode: str,
    thread_id: str,
    thread_key: str,
    token_limit: int,
    previous_summary: str | None,
    current_task_brief: str = "",
    latest_handoff_packet: str = "",
    memory_kernel_context: str = "",
    skill_context: str = "",
    protected_message_ids: set[str] | None = None,
    summarize_fn: SummaryFn,
    force: bool = False,
) -> CompressionResult:
    token_estimate_before = estimate_tokens_rough(messages)
    if token_estimate_before <= token_limit and not force:
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
            reason="under_token_limit",
        )

    selection = select_head_middle_tail(
        messages,
        token_limit=token_limit,
        protected_message_ids=protected_message_ids or set(),
    )
    if not selection.middle:
        return CompressionResult(
            mode=mode,
            thread_id=thread_id,
            thread_key=thread_key,
            token_limit=token_limit,
            token_estimate_before=token_estimate_before,
            token_estimate_after=token_estimate_before,
            head=selection.head,
            middle=[],
            tail=selection.tail,
            summary=previous_summary or "",
            summary_message_content="",
            archive_content="",
            archive_metadata={},
            pruned_middle_text="",
            skipped=True,
            reason="no_middle_messages",
        )

    pruned_middle_text = prune_tool_outputs(selection.middle)
    prompt = build_summary_prompt(
        mode=mode,
        thread_id=thread_id,
        thread_key=thread_key,
        previous_summary=previous_summary,
        current_task_brief=current_task_brief,
        latest_handoff_packet=latest_handoff_packet,
        memory_kernel_context=memory_kernel_context,
        skill_context=skill_context,
        pruned_middle_text=pruned_middle_text,
    )
    summary = await summarize_fn(prompt, _summary_token_limit(token_limit))
    summary = redact_secrets(summary.strip())

    summary_shell = build_summary_message_content(
        mode=mode,
        summary=summary,
        archive_key="pending",
        token_estimate_before=token_estimate_before,
        token_estimate_after=estimate_tokens_rough([*selection.head, *selection.tail]) + estimate_tokens_rough(summary),
    )
    token_estimate_after = estimate_tokens_rough([*selection.head, *selection.tail]) + estimate_tokens_rough(summary_shell)
    archive_content = build_archive_content(
        mode=mode,
        thread_id=thread_id,
        thread_key=thread_key,
        summary=summary,
        pruned_middle_text=pruned_middle_text,
        middle_messages=selection.middle,
    )
    archive_metadata = {
        "thread_id": thread_id,
        "thread_key": thread_key,
        "compression_mode": mode,
        "message_count": len(selection.middle),
        "token_estimate_before": token_estimate_before,
        "token_estimate_after": token_estimate_after,
        "middle_token_estimate": estimate_tokens_rough(selection.middle),
        "head_message_count": len(selection.head),
        "tail_message_count": len(selection.tail),
        "head_indexes": selection.head_indexes,
        "middle_indexes": selection.middle_indexes,
        "tail_indexes": selection.tail_indexes,
    }

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
        pruned_middle_text=pruned_middle_text,
    )
