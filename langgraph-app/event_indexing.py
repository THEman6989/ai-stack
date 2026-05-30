"""Optional tool-run / event surrogate indexing for PGVector.

Feature-flagged: ALPHARAVIS_ENABLE_TOOL_EVENT_VECTOR_INDEX (default OFF).

When enabled, compact textual surrogates of tool results are indexed into
PGVector so semantic search can recall past tool executions. Raw full output
always stays in the LangGraph Store / MongoDB; PGVector only stores bounded
summaries with metadata (tool name, exit code, timestamp, thread).

Surrogate format (example):

    Tool run: execute_local_command
    Status: success (exit 0)
    Command: pytest -q tests/test_retrieval_router.py
    Summary: 12 passed
    Output preview: ...first 500 chars of stdout...

Secrets are never included — only bounded stdout previews and structured
metadata go into PGVector.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

# Secret-scrubbing patterns — never index these
_SECRET_PATTERNS = [
    re.compile(r"(?:password|passwd|pass|token|secret|key|api_key|auth)\s*[=:]\s*\S+", re.IGNORECASE),
    re.compile(r"-----BEGIN\s+(?:RSA|EC|DSA|OPENSSH)\s+PRIVATE KEY-----[\s\S]*?-----END\s+(?:RSA|EC|DSA|OPENSSH)\s+PRIVATE KEY-----"),
    re.compile(r"ghp_[a-zA-Z0-9]{36}"),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"xox[bpras]-\d{10,}-\d{10,}-[a-zA-Z0-9]+"),
    re.compile(r"eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*"),  # JWT
]


def _scrub_secrets(text: str) -> str:
    """Remove common secret patterns from text before indexing."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def _bounded_excerpt(text: str, max_chars: int = 500) -> str:
    """Return a bounded preview of output."""
    text = text.strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[Output truncated.]"


def build_tool_run_surrogate(
    tool_name: str,
    result: str,
    *,
    exit_code: int | None = None,
    max_chars: int = 500,
) -> str:
    """Build a compact textual surrogate from a tool execution result.

    Args:
        tool_name: Name of the tool (e.g. 'execute_local_command').
        result: The full result string returned by the tool.
        exit_code: Optional exit code for shell commands.
        max_chars: Max characters for the output preview.

    Returns:
        A compact surrogate string suitable for PGVector chunk_text.
    """
    tool_name = str(tool_name or "unknown_tool").strip()[:80]
    result = str(result or "")

    # Determine status
    if exit_code is not None:
        status = "success" if exit_code == 0 else f"failed (exit {exit_code})"
    elif result.lower().startswith("error") or result.lower().startswith("failed"):
        status = "failed"
    else:
        status = "completed"

    # Build summary
    clean_result = _scrub_secrets(result)
    preview = _bounded_excerpt(clean_result, max_chars)

    lines = [
        f"Tool run: {tool_name}",
        f"Status: {status}",
    ]
    if exit_code is not None:
        lines.append(f"Exit code: {exit_code}")
    lines.append(f"Output: {preview}")

    return "\n".join(lines)


def maybe_index_tool_run(
    tool_name: str,
    result: str,
    *,
    exit_code: int | None = None,
    thread_id: str = "",
    thread_key: str = "",
    max_chars: int = 500,
) -> dict[str, Any] | None:
    """Schedule PGVector indexing of a tool-run surrogate if the feature flag is on.

    Returns a dict with status info, or None if indexing is disabled.
    The actual async indexing is scheduled on the running event loop.
    """
    enabled = os.getenv("ALPHARAVIS_ENABLE_TOOL_EVENT_VECTOR_INDEX", "false").lower() in {
        "1", "true", "yes", "on",
    }
    if not enabled:
        return None

    surrogate = build_tool_run_surrogate(tool_name, result, exit_code=exit_code, max_chars=max_chars)
    if not surrogate.strip():
        return None

    # Try to schedule async indexing on the running event loop
    try:
        import asyncio as _asyncio

        loop = _asyncio.get_running_loop()
    except RuntimeError:
        return {"scheduled": False, "warning": "No running event loop — tool-run not indexed."}

    # Build a source key from tool_name + timestamp
    now_ts = int(time.time())
    source_key = f"tool_run:{tool_name}:{now_ts}"

    # We need _maybe_index_vector_memory, but it lives in agent_graph.py.
    # To avoid circular imports, we return a dict and let the caller schedule.
    return {
        "scheduled": True,
        "source_type": "tool_run",
        "source_key": source_key,
        "title": f"Tool: {tool_name}",
        "content": surrogate,
        "thread_id": thread_id,
        "thread_key": thread_key,
        "scope": "thread",
        "metadata": {
            "tool_name": tool_name,
            "exit_code": exit_code,
            "status": "success" if (exit_code is not None and exit_code == 0) else "completed",
            "content_chars": len(result),
            "surrogate_chars": len(surrogate),
        },
    }


def surrogate_summary(surrogate: str, max_chars: int = 300) -> str:
    """Extract a short one-line summary from a surrogate for listing."""
    tool = "unknown"
    for line in surrogate.splitlines():
        line = line.strip()
        if line.startswith("Tool run:"):
            tool = line.replace("Tool run:", "").strip()
        elif line.startswith("Status:"):
            status = line.replace("Status:", "").strip()
            return f"{tool}: {status}"
    return surrogate[:max_chars]
