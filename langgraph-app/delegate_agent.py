"""AlphaRavis Native Sub-Agent Delegation — full parity with Hermes delegate_task.

Features:
- Multi-turn tool-calling loop with BigBoss (existing)
- Nested delegation with max_spawn_depth
- Sub-agent registry with cancellation (asyncio.Event)
- File-state tracking across agents (stale-read warnings)
- All four missing features in one module.

Architecture:
    SubAgentRegistry — global dict of running agents
    FileStateTracker   — path → (mtime, agent_id) for cross-agent file awareness
    spawn_sub_agent()  — core recursive spawn with depth guard
    run_sub_agent()    — public entry point for delegate_task @tool
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("alpharavis.delegate_agent")

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
        self._records: dict[str, FileRecord] = {}  # path → last write
        self._agent_reads: dict[str, dict[str, float]] = {}  # agent_id → {path → last_read_ts}

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
                f"⚠️ STALE FILE: {key} was last written by agent '{record.agent_id}' "
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

# Default max depth — sub-agents can spawn sub-agents once (depth 0→1→2 stops)
DEFAULT_MAX_SPAWN_DEPTH = int(os.getenv("ALPHARAVIS_DELEGATE_MAX_SPAWN_DEPTH", "2"))

# Maximum concurrent sub-agents across all depths
DEFAULT_MAX_CONCURRENT = int(os.getenv("ALPHARAVIS_DELEGATE_MAX_CONCURRENT", "5"))


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
) -> dict[str, Any]:
    """Run one sub-agent with tool-calling loop. Callable from any depth.

    When depth < max_spawn_depth and the sub-agent's LLM calls delegate_task,
    this function recurses to spawn grandchildren.

    Args:
        goal: The task goal for this agent.
        context: Background info passed from parent.
        tools: dict of tool_name → tool_object (callable with .ainvoke()).
        tool_names: Optional subset of tool names to expose.
        max_iterations: Max tool-calling turns.
        timeout_seconds: Per-call timeout (wraps LLM invoke).
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
        f"You have up to {max_iterations} tool-calling turns."
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

    # Get model
    if _model_fn is None:
        ctx.state = "failed"
        ctx.result = {"status": "failed", "error": "No model function provided"}
        return ctx.result

    model_kwargs: dict[str, Any] = {
        "temperature": float(os.getenv("ALPHARAVIS_DELEGATE_TEMPERATURE", "0.1")),
        "max_tokens": int(os.getenv("ALPHARAVIS_DELEGATE_MAX_TOKENS", "4096")),
    }
    if tool_schemas:
        model_kwargs["tools"] = tool_schemas

    model = _model_fn(model_kwargs)
    if model is None:
        ctx.state = "failed"
        ctx.result = {"status": "failed", "error": "Model unavailable"}
        return ctx.result

    started = time.perf_counter()
    api_calls = 0
    max_chars = max(1000, min(int(max_output_chars), 16000))
    timeout = max(30, min(int(timeout_seconds), 1800))

    name_to_tool = selected_tools  # already normalized

    try:
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
                return result

            api_calls += 1
            response = await asyncio.wait_for(
                model.ainvoke(messages),
                timeout=timeout,
            )

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
                            # Track writes from commands that clearly write files
                            cmd = str(tc_args.get("command", "")).lower()
                            write_path = _extract_write_path_from_command(cmd)
                            if write_path:
                                FILE_STATE_TRACKER.record_write(write_path, agent_id)

                    except Exception as exc:
                        result_str = f"Tool error: {exc}"
                else:
                    result_str = f"Tool '{tc_name}' not available in this sub-agent context."

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result_str[:4000],
                })

        # Max iterations reached
        ctx.state = "completed"  # still completed, but with max_iterations flag
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

    except asyncio.TimeoutError:
        ctx.state = "timeout"
        result = {
            "status": "timeout",
            "goal": goal[:120],
            "agent_id": agent_id,
            "error": f"Timed out after {timeout}s",
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
    """Build OpenAI-format tool schemas from tool objects."""
    schemas: list[dict[str, Any]] = []
    for name, tool_obj in tools.items():
        if hasattr(tool_obj, "args_schema"):
            schema = tool_obj.args_schema
            if hasattr(schema, "model_json_schema"):
                js = schema.model_json_schema()
            elif callable(getattr(schema, "schema", None)):
                js = schema.schema()
            else:
                continue
            schemas.append({
                "type": "function",
                "function": {
                    "name": _tool_name_fn(tool_obj) if _tool_name_fn else name,
                    "description": getattr(tool_obj, "description", "") or "",
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
        r'(?:^|\s)>>\s*([^\s;&|]+)',                # redirect >> file (MUST come before >)
        r'(?:^|\s)(?<!>)>\s*([^\s;&|]+)',            # redirect > file (NOT >>)
        r'(?:^|\s)tee\s+([^\s;&|]+)',                 # tee file
        r'(?:^|\s)cp\s+.*\s+([^\s;&|]+)$',          # cp ... dest (last arg)
        r'(?:^|\s)mv\s+.*\s+([^\s;&|]+)$',          # mv ... dest (last arg)
        r'(?:^|\s)install\s+.*\s+([^\s;&|]+)$',     # install ... dest
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
