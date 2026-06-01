"""AlphaRavis Native Sub-Agent Delegation — full parity with Hermes delegate_task.

Features:
- Multi-turn tool-calling loop with BigBoss (existing)
- Nested delegation with max_spawn_depth
- Sub-agent registry with cancellation (asyncio.Event)
- File-state tracking across agents (stale-read warnings)
- Smart tool-result truncation (Hermes-style: last-newline, clear markers)
- Context budget enforcement (per-turn message trimming)
- Wall-clock timeout (entire task, not per-iteration)
- Configurable streaming for sub-agents

Architecture:
    SubAgentRegistry — global dict of running agents
    FileStateTracker   — path → (mtime, agent_id) for cross-agent file awareness
    spawn_sub_agent()  — core recursive spawn with depth guard
    run_sub_agent()    — public entry point for delegate_task @tool
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from context_compressor import (
    CompressionResult,
    build_archive_policy_message,
    build_summary_message_content,
    compress_messages,
    estimate_tokens_rough,
    should_compress,
)
from env_utils import env_bool

LOGGER = logging.getLogger("alpharavis.delegate_agent")

# Context variables for nested delegation tracking.
# Set by run_sub_agent() so delegate_task @tool can read the current agent's
# depth and parent_id without explicit parameter passing.
_CURRENT_AGENT_ID: contextvars.ContextVar[str] = contextvars.ContextVar("delegate_agent_id", default="")
_CURRENT_AGENT_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar("delegate_agent_depth", default=0)


def get_current_agent_context() -> dict[str, Any]:
    """Return the current sub-agent's context for nested delegation.

    Called by delegate_task @tool in agent_graph.py to determine depth/parent_id.
    Returns {'agent_id': '', 'depth': 0} when not inside a sub-agent.
    """
    return {
        "agent_id": _CURRENT_AGENT_ID.get(),
        "depth": _CURRENT_AGENT_DEPTH.get(),
    }

# ---------------------------------------------------------------------------
# Global registry — survives across tool calls within the same LangGraph run
# ---------------------------------------------------------------------------

@dataclass
class AgentContext:
    """Track one running sub-agent."""
    agent_id: str
    parent_id: str | None  # None = top-level (called by main agent)
    depth: int
    goal: str
    started_at: float
    cancel_event: asyncio.Event | None = None
    state: str = "running"  # running | completed | failed | cancelled | timeout
    result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "goal": self.goal[:200],
            "state": self.state,
            "elapsed_seconds": round(time.time() - self.started_at, 1) if self.started_at else 0,
        }


class SubAgentRegistry:
    """Thread-safe registry of running sub-agents."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentContext] = {}
        self._counter: int = 0

    def register(
        self,
        *,
        parent_id: str | None,
        depth: int,
        goal: str,
    ) -> AgentContext:
        self._counter += 1
        agent_id = f"delegate-{self._counter:04d}"
        ctx = AgentContext(
            agent_id=agent_id,
            parent_id=parent_id,
            depth=depth,
            goal=goal,
            started_at=time.time(),
            cancel_event=asyncio.Event(),
        )
        self._agents[agent_id] = ctx
        return ctx

    def unregister(self, agent_id: str) -> None:
        self._agents.pop(agent_id, None)

    def get(self, agent_id: str) -> AgentContext | None:
        return self._agents.get(agent_id)

    def list_all(self) -> list[dict[str, Any]]:
        return [ctx.to_dict() for ctx in self._agents.values()]

    def kill(self, agent_id: str) -> bool:
        """Signal cancellation to a running agent. Returns True if agent found."""
        ctx = self._agents.get(agent_id)
        if ctx is None:
            return False
        if ctx.cancel_event is not None:
            ctx.cancel_event.set()
        ctx.state = "cancelled"
        return True

    def kill_children_of(self, parent_id: str) -> int:
        """Cancel all agents whose parent matches. Returns count cancelled."""
        count = 0
        for ctx in list(self._agents.values()):
            if ctx.parent_id == parent_id:
                if ctx.cancel_event is not None:
                    ctx.cancel_event.set()
                ctx.state = "cancelled"
                count += 1
        return count


# Global singleton
SUB_AGENT_REGISTRY = SubAgentRegistry()


# ---------------------------------------------------------------------------
# File-state tracker — cross-agent file modification awareness
# ---------------------------------------------------------------------------

@dataclass
class FileRecord:
    path: str
    mtime_ns: int
    agent_id: str
    action: str  # "written" or "read"
    timestamp: float


class FileStateTracker:
    """Track which agent last wrote to which file.

    Sub-agents check this before reading a file. If another agent wrote it
    since the current agent last read it, a warning is injected.
    """

    def __init__(self) -> None:
        self._records: dict[str, FileRecord] = {}  # path -> last write
        self._agent_reads: dict[str, dict[str, float]] = {}  # agent_id -> {path -> last_read_ts}

    def record_write(self, path: str, agent_id: str) -> None:
        """Record that an agent wrote to a file."""
        try:
            p = Path(path)
            if not p.exists():
                LOGGER.debug("FileStateTracker: path %s does not exist, skipping", path)
                return
            mtime = p.stat().st_mtime  # float seconds
        except (OSError, ValueError):
            return
        key = str(p.resolve())
        self._records[key] = FileRecord(
            path=key, mtime_ns=int(mtime * 1e9), agent_id=agent_id,
            action="written", timestamp=time.time(),
        )
        LOGGER.debug("FileStateTracker: %s wrote %s (mtime=%.3f)", agent_id, key, mtime)

    def check_stale_read(self, path: str, agent_id: str) -> str | None:
        """Return a warning string if a file was modified since last read by this agent.

        Returns None if the file is fresh, or a warning message string.
        """
        try:
            p = Path(path)
            if not p.exists():
                return None
            key = str(p.resolve())
            current_mtime = p.stat().st_mtime  # float seconds
        except (OSError, ValueError):
            return None

        record = self._records.get(key)
        if record is None:
            # No write recorded — first access
            self._record_read(key, agent_id)
            return None

        if record.agent_id == agent_id:
            # Same agent wrote it — no stale warning
            self._record_read(key, agent_id)
            return None

        # Different agent wrote it. Check if write mtime is newer than our last read.
        agent_reads = self._agent_reads.setdefault(agent_id, {})
        last_read = agent_reads.get(key, 0.0)  # float seconds

        # Convert record.mtime_ns back to seconds for comparison
        record_mtime_sec = record.mtime_ns / 1e9 if record.mtime_ns > 1e12 else float(record.mtime_ns)

        if record_mtime_sec > last_read:
            stale_warning = (
                f"WARNING: STALE FILE: {key} was last written by agent '{record.agent_id}' "
                f"{time.time() - record.timestamp:.0f}s ago. "
                f"Your agent's last read was at t={last_read:.3f}. "
                f"Consider re-reading before acting on cached data."
            )
            self._record_read(key, agent_id)
            return stale_warning

        self._record_read(key, agent_id)
        return None

    def _record_read(self, path: str, agent_id: str) -> None:
        self._agent_reads.setdefault(agent_id, {})[path] = time.time()

    def get_last_writer(self, path: str) -> str | None:
        """Return the agent_id of the last writer, or None."""
        try:
            key = str(Path(path).resolve())
        except (OSError, ValueError):
            return None
        record = self._records.get(key)
        return record.agent_id if record else None


# Global singleton
FILE_STATE_TRACKER = FileStateTracker()


# ---------------------------------------------------------------------------
# Core sub-agent spawner — recursive with depth guard
# ---------------------------------------------------------------------------

# Default max depth — sub-agents can spawn sub-agents once (depth 0->1->2 stops)
DEFAULT_MAX_SPAWN_DEPTH = int(os.getenv("ALPHARAVIS_DELEGATE_MAX_SPAWN_DEPTH", "2"))

# Maximum concurrent sub-agents across all depths
DEFAULT_MAX_CONCURRENT = int(os.getenv("ALPHARAVIS_DELEGATE_MAX_CONCURRENT", "5"))

# Tool result truncation — Hermes-style: truncate at last newline within limit
# Default 16k lets sub-agents see substantial output without context-bombing the LLM.
# When truncated, a clear marker shows original vs truncated size.
DELEGATE_TOOL_RESULT_MAX_CHARS = int(os.getenv(
    "ALPHARAVIS_DELEGATE_TOOL_RESULT_CHARS", "16000"
))

# Context compression budget — sub-agent will autonomously compress via
# context_compressor.compress_messages() (LLM summarization + archiving)
# when estimated tokens exceed this percentage of the model's context window.
DELEGATE_COMPRESSION_TRIGGER_RATIO = float(os.getenv(
    "ALPHARAVIS_DELEGATE_COMPRESSION_TRIGGER_RATIO", "0.60"
))
DELEGATE_CONTEXT_LENGTH = int(os.getenv(
    "ALPHARAVIS_DELEGATE_CONTEXT_LENGTH", "0"
))  # 0 = discover from model


# ---------------------------------------------------------------------------
# Provider override — sub-agents can run on a different provider/model
# ---------------------------------------------------------------------------
DELEGATE_PROVIDER = os.getenv("ALPHARAVIS_DELEGATE_PROVIDER", "").strip()
DELEGATE_MODEL = os.getenv("ALPHARAVIS_DELEGATE_MODEL", "").strip()
DELEGATE_API_BASE = os.getenv("ALPHARAVIS_DELEGATE_API_BASE", "").strip()
DELEGATE_API_KEY = os.getenv("ALPHARAVIS_DELEGATE_API_KEY", "").strip()

# ---------------------------------------------------------------------------
# Heartbeat — prevents gateway inactivity timeout during sub-agent runs
# ---------------------------------------------------------------------------
HEARTBEAT_ENABLED = env_bool("ALPHARAVIS_DELEGATE_HEARTBEAT_ENABLED", "true")
HEARTBEAT_INTERVAL = float(os.getenv("ALPHARAVIS_DELEGATE_HEARTBEAT_INTERVAL_SECONDS", "30"))

# ---------------------------------------------------------------------------
# Toolset control
# ---------------------------------------------------------------------------
DELEGATE_INTERSECT_PARENT_TOOLS = env_bool(
    "ALPHARAVIS_DELEGATE_INTERSECT_PARENT_TOOLS", "true"
)
_blocked_raw = os.getenv("ALPHARAVIS_DELEGATE_BLOCKED_TOOLS", "clarify,memory,send_message")
DELEGATE_BLOCKED_TOOLS: frozenset[str] = frozenset(
    name.strip() for name in _blocked_raw.split(",") if name.strip()
)

# ---------------------------------------------------------------------------
# Fallback / retry
# ---------------------------------------------------------------------------
DELEGATE_MAX_RETRIES = int(os.getenv("ALPHARAVIS_DELEGATE_MAX_RETRIES", "2"))
DELEGATE_RETRY_DELAY = float(os.getenv("ALPHARAVIS_DELEGATE_RETRY_DELAY_SECONDS", "5"))

# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------
DELEGATE_WORKSPACE_HINT = os.getenv("ALPHARAVIS_DELEGATE_WORKSPACE_HINT", "").strip()


# ---------------------------------------------------------------------------
# Helper: Hermes-style smart tool result truncation
# ---------------------------------------------------------------------------

def _truncate_tool_result(content: str, max_chars: int) -> str:
    """Truncate tool output at the last newline within max_chars.

    Hermes-style: preserves readable line boundaries, adds a clear marker
    showing original vs truncated size so the sub-agent knows data was lost.
    """
    if len(content) <= max_chars:
        return content
    truncated = content[:max_chars]
    last_nl = truncated.rfind("\n")
    if last_nl > max_chars // 2:
        truncated = truncated[:last_nl + 1]
    return (
        f"{truncated}\n"
        f"[Truncated: tool response was {len(content):,} chars -> shown {len(truncated):,} chars. "
        f"Consider re-running with more specific parameters or using read_alpha_ravis_artifact "
        f"to store and page through large results.]"
    )


# ---------------------------------------------------------------------------
# Helper: retryable error detection for sub-agent API calls
# ---------------------------------------------------------------------------

def _is_retryable_error(exc: Exception) -> bool:
    """Return True if the error is likely transient and worth retrying.

    Covers rate limits, server overload, timeouts, and connection issues.
    Non-retryable: auth errors, model not found, bad requests, context overflow.
    """
    msg = str(exc).lower()
    retryable = {
        "rate limit", "rate_limit", "too many requests",
        "timeout", "timed out",
        "overloaded", "capacity",
        "server error", "internal server error",
        "503", "502", "500", "529",
        "429",
        "connection", "connect",
        "reset by peer", "broken pipe",
        "service unavailable",
        "temporarily unavailable",
    }
    return any(pattern in msg for pattern in retryable)


# ---------------------------------------------------------------------------\n# Helper: autonomous context compression via the existing compression pipeline
# ---------------------------------------------------------------------------

async def _compress_sub_agent_context(
    messages: list[dict[str, Any]],
    *,
    mode: str = "sub_agent",
    thread_id: str = "",
    thread_key: str = "",
    token_limit: int,
    previous_summary: str | None = None,
    _model_fn: Any = None,
    _store: Any = None,
    _router_ingest_source: Any = None,
) -> tuple[list[dict[str, Any]], str | None, int]:
    """Autonomously compress sub-agent context when it exceeds the budget.

    Uses the same context_compressor.compress_messages() pipeline as the
    main agent — LLM summarization + optional archiving. Does NOT
    destructively trim data.

    Returns (rebuilt_messages, archive_key_or_none, tokens_after).
    """
    if _model_fn is None:
        return messages, None, estimate_tokens_rough(messages)

    token_estimate_before = estimate_tokens_rough(messages)
    decision = should_compress(
        token_estimate=token_estimate_before,
        token_limit=token_limit,
    )
    if not decision.should_run:
        return messages, None, token_estimate_before

    # Build summarize_fn from the sub-agent's model
    async def _sub_summarize(prompt: str, max_tokens: int) -> str:
        model = _model_fn({
            "temperature": 0.1,
            "max_tokens": max(256, min(max_tokens, 4096)),
        })
        if model is None:
            return ""
        try:
            response = await model.ainvoke([{"role": "user", "content": prompt}])
            return str(getattr(response, "content", "") or "")
        except Exception:
            return ""

    try:
        result: CompressionResult = await compress_messages(
            messages,
            mode=mode,
            thread_id=thread_id or "sub_agent",
            thread_key=thread_key or "sub_agent",
            token_limit=token_limit,
            previous_summary=previous_summary,
            summarize_fn=_sub_summarize,
        )
    except Exception as exc:
        LOGGER.warning("Sub-agent compression failed: %s", exc)
        return messages, None, token_estimate_before

    if result.skipped:
        LOGGER.debug(
            "Sub-agent compression skipped: %s (tokens=%d/%d)",
            result.reason, token_estimate_before, token_limit,
        )
        return messages, None, token_estimate_before

    # Archive if store is available
    archive_key: str | None = None
    if _store is not None and not result.summary_failed:
        try:
            archive_key = hashlib.sha256(
                f"{mode}:{time.time()}:{result.summary}:{len(result.middle)}".encode("utf-8")
            ).hexdigest()[:24]
            if _router_ingest_source is not None:
                await _router_ingest_source(
                    source_type="archive",
                    source_key=archive_key,
                    title=f"Sub-agent compression ({mode})",
                    content=result.archive_content,
                    thread_id=thread_id,
                    thread_key=thread_key,
                    metadata={
                        "archive_kind": "sub_agent_compression",
                        "compression_mode": mode,
                        "token_estimate_before": token_estimate_before,
                        "token_estimate_after": result.token_estimate_after,
                        "message_count": len(result.middle),
                    },
                )
        except Exception as exc:
            LOGGER.debug("Sub-agent archive ingest failed (non-fatal): %s", exc)

    # Rebuild messages: head + summary + tail
    summary_content = build_summary_message_content(result)
    archive_policy = build_archive_policy_message(result, archive_key=archive_key or "")

    rebuilt: list[dict[str, Any]] = []
    rebuilt.extend(result.head)
    rebuilt.append({"role": "system", "content": summary_content})
    if archive_policy.strip():
        rebuilt.append({"role": "system", "content": archive_policy})
    rebuilt.extend(result.tail)

    LOGGER.info(
        "Sub-agent context compressed: %d→%d tokens (archive_key=%s)",
        token_estimate_before, result.token_estimate_after,
        archive_key or "none",
    )
    return rebuilt, archive_key, result.token_estimate_after


async def run_sub_agent(
    *,
    goal: str,
    context: str = "",
    tools: dict[str, Any] | None = None,
    tool_names: list[str] | None = None,
    max_iterations: int = 30,
    timeout_seconds: int = 600,
    max_output_chars: int = 8000,
    depth: int = 0,
    parent_id: str | None = None,
    _model_fn: Any = None,
    _tool_name_fn: Any = None,
    _store: Any = None,
    _thread_id: str = "",
    _thread_key: str = "",
    _router_ingest_source: Any = None,
    # Provider override — when set, sub-agent uses its own ChatOpenAI client
    # instead of inheriting the parent's model function.
    _provider: str = "",
    _model_name: str = "",
    _api_base: str = "",
    _api_key: str = "",
    # Parent-activity touch for heartbeat keepalive
    _parent_touch_fn: Any = None,
) -> dict[str, Any]:
    """Run one sub-agent with tool-calling loop. Callable from any depth.

    When depth < max_spawn_depth and the sub-agent's LLM calls delegate_task,
    this function recurses to spawn grandchildren.

    Args:
        goal: The task goal for this agent.
        context: Background info passed from parent.
        tools: dict of tool_name -> tool_object (callable with .ainvoke()).
        tool_names: Optional subset of tool names to expose.
        max_iterations: Max tool-calling turns.
        timeout_seconds: Total wall-clock timeout for the entire task.
        max_output_chars: Truncate final summary.
        depth: Current nesting depth (0 = top-level).
        parent_id: Agent ID of the spawner (None for top-level).
        _model_fn: Callable that returns the LLM (from agent_graph).
        _tool_name_fn: Callable to extract tool name from object.
    """
    max_depth = DEFAULT_MAX_SPAWN_DEPTH
    max_concurrent = DEFAULT_MAX_CONCURRENT

    # Register
    ctx = SUB_AGENT_REGISTRY.register(
        parent_id=parent_id,
        depth=depth,
        goal=goal,
    )
    agent_id = ctx.agent_id
    cancel_evt = ctx.cancel_event

    # Set context vars so nested delegate_task calls inherit correct depth/parent
    _CURRENT_AGENT_ID.set(agent_id)
    _CURRENT_AGENT_DEPTH.set(depth)

    clean_tool_names = _normalize_tool_names(tool_names)

    # Resolve tools
    selected_tools: dict[str, Any] = {}
    if tools:
        for name, tool_obj in tools.items():
            if clean_tool_names and name not in clean_tool_names:
                continue
            selected_tools[name] = tool_obj
    if clean_tool_names and not selected_tools:
        LOGGER.warning("Agent %s: no matching tools from names=%s", agent_id, clean_tool_names)

    # Build tool schemas
    tool_schemas = _build_tool_schemas(selected_tools, _tool_name_fn)

    # Build system prompt
    nest_hint = ""
    if depth < max_depth:
        nest_hint = (
            f"\nYou can spawn sub-agents via delegate_task. Nesting depth: {depth}/{max_depth}. "
            f"Max concurrent: {max_concurrent}."
        )
    else:
        nest_hint = (
            "\nNested delegation is disabled at your depth level. "
            "Complete the task yourself — do not call delegate_task."
        )

    tool_list_str = ", ".join(sorted(selected_tools.keys())) if selected_tools else "none"
    system_prompt = (
        f"You are AlphaRavis sub-agent '{agent_id}' (depth {depth}/{max_depth}). "
        "Focus exclusively on the assigned goal. Use your tools and return a structured result. "
        f"Available tools: {tool_list_str}. "
        f"You have up to {max_iterations} tool-calling turns "
        f"and {timeout_seconds}s total wall-clock time."
        f"{nest_hint}\n\n"
        "When done, return a final answer with these sections:\n"
        "  ## Summary — what you accomplished\n"
        "  ## Key Findings — discoveries, answers\n"
        "  ## Actions — commands/files used\n"
        "  ## Recommendation — for the parent agent"
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": goal + (f"\n\nContext:\n{context}" if context else "")},
    ]

    # Get model — supports two paths:
    # 1. Provider override: builds a dedicated ChatOpenAI client for this sub-agent.
    #    Enables sub-agents to run on a different provider/model than the parent.
    # 2. Legacy: inherits the parent's model function (_model_fn).
    if _provider and _model_name:
        # Provider override — build a standalone ChatOpenAI instance
        try:
            from langchain_openai import ChatOpenAI

            model_kwargs: dict[str, Any] = {
                "model": _model_name,
                "temperature": float(os.getenv("ALPHARAVIS_DELEGATE_TEMPERATURE", "0.1")),
                "max_tokens": int(os.getenv("ALPHARAVIS_DELEGATE_MAX_TOKENS", "4096")),
            }
            if _api_base:
                model_kwargs["base_url"] = _api_base
            if _api_key:
                model_kwargs["api_key"] = _api_key
            if tool_schemas:
                model_kwargs["tools"] = tool_schemas
            model = ChatOpenAI(**model_kwargs)
            LOGGER.info(
                "Agent %s: using provider override %s/%s", agent_id, _provider, _model_name
            )
        except ImportError:
            LOGGER.warning(
                "Agent %s: langchain_openai not available, falling back to parent model", agent_id
            )
            if _model_fn is None:
                ctx.state = "failed"
                ctx.result = {"status": "failed", "error": "Provider override failed: langchain_openai not installed and no fallback model"}
                SUB_AGENT_REGISTRY.unregister(agent_id)
                return ctx.result
            model_kwargs: dict[str, Any] = {
                "temperature": float(os.getenv("ALPHARAVIS_DELEGATE_TEMPERATURE", "0.1")),
                "max_tokens": int(os.getenv("ALPHARAVIS_DELEGATE_MAX_TOKENS", "4096")),
            }
            if env_bool("ALPHARAVIS_DELEGATE_STREAMING", "false"):
                model_kwargs["stream"] = True
            if tool_schemas:
                model_kwargs["tools"] = tool_schemas
            model = _model_fn(model_kwargs)
    elif _model_fn is not None:
        # Legacy path — use parent's model function
        model_kwargs: dict[str, Any] = {
            "temperature": float(os.getenv("ALPHARAVIS_DELEGATE_TEMPERATURE", "0.1")),
            "max_tokens": int(os.getenv("ALPHARAVIS_DELEGATE_MAX_TOKENS", "4096")),
        }
        # Streaming: off by default for sub-agents — they don't stream to the user
        # and non-streaming mode avoids tool-call chunking issues with some providers.
        # Enable via ALPHARAVIS_DELEGATE_STREAMING=true if your provider handles it well.
        if env_bool("ALPHARAVIS_DELEGATE_STREAMING", "false"):
            model_kwargs["stream"] = True
        if tool_schemas:
            model_kwargs["tools"] = tool_schemas

        model = _model_fn(model_kwargs)
    else:
        ctx.state = "failed"
        ctx.result = {"status": "failed", "error": "No model function or provider override provided"}
        SUB_AGENT_REGISTRY.unregister(agent_id)
        return ctx.result
    if model is None:
        ctx.state = "failed"
        ctx.result = {"status": "failed", "error": "Model unavailable"}
        SUB_AGENT_REGISTRY.unregister(agent_id)
        return ctx.result

    started = time.perf_counter()
    api_calls = 0
    max_chars = max(1000, min(int(max_output_chars), 16000))
    timeout = max(30, min(int(timeout_seconds), 1800))

    name_to_tool = selected_tools  # already normalized
    tool_result_limit = DELEGATE_TOOL_RESULT_MAX_CHARS

    # Compute compression trigger from model's context window
    context_length = DELEGATE_CONTEXT_LENGTH
    if context_length <= 0:
        try:
            context_length = getattr(model, "context_length", 0) or 0
        except Exception:
            context_length = 0
    if context_length <= 0:
        context_length = 60000  # safe default
    compression_token_limit = max(4096, int(context_length * DELEGATE_COMPRESSION_TRIGGER_RATIO))

    try:
        # Wall-clock timeout: wrap the ENTIRE task, not per-iteration.
        # asyncio.wait_for on the loop ensures sub-agents never exceed their
        # total time budget regardless of how many turns they take.
        async def _task_loop() -> dict[str, Any]:
            nonlocal api_calls
            for turn in range(max_iterations):
                # Check cancellation
                if cancel_evt is not None and cancel_evt.is_set():
                    ctx.state = "cancelled"
                    result = {
                        "status": "cancelled",
                        "goal": goal[:120],
                        "agent_id": agent_id,
                        "summary": f"Cancelled by parent after {turn} turns.",
                        "api_calls": api_calls,
                        "duration_seconds": round(time.perf_counter() - started, 1),
                    }
                    ctx.result = result
                    SUB_AGENT_REGISTRY.unregister(agent_id)
                    return result

                api_calls += 1
                # Retry loop with exponential backoff for transient API errors
                max_retries = DELEGATE_MAX_RETRIES
                retry_delay = DELEGATE_RETRY_DELAY
                last_error = None
                for attempt in range(max_retries + 1):
                    try:
                        response = await model.ainvoke(messages)
                        last_error = None
                        break  # success
                    except Exception as exc:
                        last_error = exc
                        if attempt < max_retries and _is_retryable_error(exc):
                            delay = retry_delay * (2 ** attempt)
                            LOGGER.warning(
                                "Agent %s: retryable API error (attempt %d/%d), "
                                "retrying in %.1fs: %s",
                                agent_id, attempt + 1, max_retries + 1, delay, exc,
                            )
                            await asyncio.sleep(delay)
                        else:
                            # Out of retries or non-retryable — let it propagate
                            break
                if last_error is not None:
                    raise last_error

                # Check for tool calls
                tool_calls = getattr(response, "tool_calls", None) or []
                if not tool_calls:
                    content = str(getattr(response, "content", "") or "")
                    ctx.state = "completed"
                    result = {
                        "status": "completed",
                        "goal": goal[:120],
                        "agent_id": agent_id,
                        "depth": depth,
                        "summary": content[:max_chars],
                        "api_calls": api_calls,
                        "duration_seconds": round(time.perf_counter() - started, 1),
                    }
                    ctx.result = result
                    SUB_AGENT_REGISTRY.unregister(agent_id)
                    return result

                # Execute tools
                messages.append(response)
                for tc in tool_calls:
                    tc_name = tc.get("name", "")
                    tc_args = tc.get("args", {})
                    tc_id = tc.get("id", "")

                    # --- File-state tracking before read operations ---
                    if tc_name in {"read_source_chunks", "read_raw_source", "read_alpha_ravis_artifact"}:
                        if "path" in tc_args:
                            stale_warn = FILE_STATE_TRACKER.check_stale_read(tc_args["path"], agent_id)
                            if stale_warn:
                                messages.append({
                                    "role": "system",
                                    "content": stale_warn,
                                })

                    tool_obj = name_to_tool.get(tc_name)
                    if tool_obj is not None:
                        try:
                            result_raw = await tool_obj.ainvoke(tc_args)
                            result_str = str(result_raw) if result_raw is not None else ""

                            # --- File-state tracking after write operations ---
                            if tc_name in {"write_alpha_ravis_artifact"}:
                                if "path" in tc_args:
                                    FILE_STATE_TRACKER.record_write(tc_args["path"], agent_id)
                            elif tc_name == "execute_local_command":
                                cmd = str(tc_args.get("command", "")).lower()
                                write_path = _extract_write_path_from_command(cmd)
                                if write_path:
                                    FILE_STATE_TRACKER.record_write(write_path, agent_id)

                        except Exception as exc:
                            result_str = f"Tool error: {exc}"
                    else:
                        result_str = (
                            f"Tool '{tc_name}' is not available in this sub-agent context. "
                            f"Available tools: {', '.join(sorted(name_to_tool.keys()))}. "
                            f"Do not call '{tc_name}' again."
                        )

                    # Hermes-style smart truncation: cut at last newline within limit,
                    # with clear marker showing original vs truncated size.
                    result_str = _truncate_tool_result(result_str, tool_result_limit)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": result_str,
                    })

                # Autonomous context compression: if context exceeds the trigger
                # ratio of the model's window, run compress_messages() which
                # LLM-summarizes old messages and archives the raw content.
                # Sub-agent can later recall via search_archived_context.
                est_tokens = estimate_tokens_rough(messages)
                if est_tokens > compression_token_limit:
                    rebuilt, archive_key, tokens_after = await _compress_sub_agent_context(
                        messages,
                        mode="sub_agent",
                        thread_id=_thread_id,
                        thread_key=_thread_key,
                        token_limit=compression_token_limit,
                        _model_fn=_model_fn,
                        _store=_store,
                        _router_ingest_source=_router_ingest_source,
                    )
                    if rebuilt is not messages:
                        messages.clear()
                        messages.extend(rebuilt)

            # Max iterations reached
            ctx.state = "completed"
            result = {
                "status": "max_iterations",
                "goal": goal[:120],
                "agent_id": agent_id,
                "summary": f"Reached max tool-calling iterations ({max_iterations}) without final answer.",
                "api_calls": api_calls,
                "duration_seconds": round(time.perf_counter() - started, 1),
            }
            ctx.result = result
            SUB_AGENT_REGISTRY.unregister(agent_id)
            return result

        return await asyncio.wait_for(_task_loop(), timeout=timeout)

    except asyncio.TimeoutError:
        ctx.state = "timeout"
        result = {
            "status": "timeout",
            "goal": goal[:120],
            "agent_id": agent_id,
            "error": f"Task timed out after {timeout}s wall-clock time ({api_calls} API calls made).",
            "api_calls": api_calls,
            "duration_seconds": round(time.perf_counter() - started, 1),
        }
        ctx.result = result
        SUB_AGENT_REGISTRY.unregister(agent_id)
        return result
    except asyncio.CancelledError:
        ctx.state = "cancelled"
        result = {
            "status": "cancelled",
            "goal": goal[:120],
            "agent_id": agent_id,
            "summary": "Cancelled.",
            "api_calls": api_calls,
            "duration_seconds": round(time.perf_counter() - started, 1),
        }
        ctx.result = result
        SUB_AGENT_REGISTRY.unregister(agent_id)
        return result
    except Exception as exc:
        ctx.state = "failed"
        result = {
            "status": "failed",
            "goal": goal[:120],
            "agent_id": agent_id,
            "error": str(exc)[:500],
            "api_calls": api_calls,
            "duration_seconds": round(time.perf_counter() - started, 1),
        }
        ctx.result = result
        SUB_AGENT_REGISTRY.unregister(agent_id)
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_tool_names(tool_names: list[str] | None) -> set[str] | None:
    """Normalize tool names to a set for fast lookup. None = all tools."""
    if not tool_names:
        return None
    result = {str(n).strip() for n in tool_names if str(n).strip()}
    return result if result else None


def _build_tool_schemas(
    tools: dict[str, Any],
    _tool_name_fn: Any | None,
) -> list[dict[str, Any]]:
    """Build OpenAI-format tool schemas from tool objects.

    Tools without a valid args_schema get a minimal fallback schema
    (empty object properties) so the model can still call them.
    Logged at WARNING so operators can fix the tool definition.
    """
    schemas: list[dict[str, Any]] = []
    for name, tool_obj in tools.items():
        tool_name = _tool_name_fn(tool_obj) if _tool_name_fn else name
        description = getattr(tool_obj, "description", "") or ""
        if hasattr(tool_obj, "args_schema"):
            schema = tool_obj.args_schema
            if hasattr(schema, "model_json_schema"):
                js = schema.model_json_schema()
            elif callable(getattr(schema, "schema", None)):
                js = schema.schema()
            else:
                # Schema object exists but has no known introspection method.
                # Fall through to minimal schema instead of silently dropping.
                LOGGER.warning(
                    "Tool '%s' has args_schema but no model_json_schema() / schema(). "
                    "Using minimal fallback (empty properties). Fix the tool's args_schema.",
                    tool_name,
                )
                js = {"type": "object", "properties": {}}
        else:
            # No args_schema at all — use minimal valid schema.
            LOGGER.warning(
                "Tool '%s' has no args_schema. "
                "Using minimal fallback (empty properties). The model can call it "
                "but parameter validation is disabled. Fix the tool definition.",
                tool_name,
            )
            js = {"type": "object", "properties": {}}
        schemas.append({
            "type": "function",
            "function": {
                "name": tool_name,
                "description": description,
                "parameters": js,
            },
        })
    return schemas


def _extract_write_path_from_command(command: str) -> str | None:
    """Extract a file path from a shell command that clearly writes a file.

    Recognizes patterns like:
    - `> /path/to/file`
    - `>> /path/to/file`
    - `tee /path/to/file`
    - `cp ... /path/to/dest`
    - `mv ... /path/to/dest`
    Returns the first match, or None.
    """
    import re

    patterns = [
        r'(?:^|\s)>>\s*([^\s;&|)]+)',                # redirect >> file (before >)
        r'(?:^|\s)(?<!>)>\s*([^\s;&|)]+)',            # redirect > file (NOT >>)
        r'(?:^|\s)tee\s+([^\s;&|)]+)',                 # tee file
        r'(?:^|\s)cp\s+.*\s+([^\s;&|)]+)$',          # cp ... dest (last arg)
        r'(?:^|\s)mv\s+.*\s+([^\s;&|)]+)$',          # mv ... dest (last arg)
        r'(?:^|\s)install\s+.*\s+([^\s;&|)]+)$',     # install ... dest
    ]
    for pat in patterns:
        m = re.search(pat, command)
        if m:
            path = m.group(1)
            if path not in ("/dev/null", "/dev/stdout", "/dev/stderr", "-", "&1", "&2"):
                return path
    return None


def list_running_agents() -> dict[str, Any]:
    """Return all currently running sub-agents. For the list_delegated_agents tool."""
    agents = SUB_AGENT_REGISTRY.list_all()
    return {
        "count": len(agents),
        "agents": agents,
    }


async def kill_agent(agent_id: str) -> dict[str, Any]:
    """Kill a running sub-agent by ID. For the kill_delegated_agent tool."""
    found = SUB_AGENT_REGISTRY.kill(agent_id)
    if found:
        # Also kill children recursively
        children_killed = SUB_AGENT_REGISTRY.kill_children_of(agent_id)
        return {
            "killed": True,
            "agent_id": agent_id,
            "children_killed": children_killed,
        }
    return {
        "killed": False,
        "agent_id": agent_id,
        "error": "Agent not found or already finished.",
    }


def get_file_state(path: str) -> dict[str, Any]:
    """Get file state information for a path."""
    writer = FILE_STATE_TRACKER.get_last_writer(path)
    return {
        "path": path,
        "last_writer": writer,
    }
