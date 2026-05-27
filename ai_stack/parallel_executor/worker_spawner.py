"""Worker spawner for parallel task execution.

Provides an adapter interface for spawning workers (Hermes, direct LLM)
in isolated worktrees. Ships with a dry-run / mock worker for testing.

Design:
- Abstract base class defines the spawner contract.
- DryRunWorker logs what it would do without executing.
- DirectLLMWorker calls BigBoss directly via a callable.
- HermesWorker wraps the existing call_hermes_agent path.
- No Codex adapter — explicitly excluded.
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_stack.parallel_executor.task_graph import ModelClass, PlannedTask
from ai_stack.parallel_executor.worktree_manager import WorktreeInfo

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker result
# ---------------------------------------------------------------------------


@dataclass
class WorkerResult:
    task_id: str
    status: str  # "completed", "failed", "dry_run", "skipped", "queued"
    output: str = ""
    error: str = ""
    changed_files: list[str] = field(default_factory=list)
    worktree: WorktreeInfo | None = None
    commit_sha: str = ""
    elapsed_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in {"completed", "dry_run"}


# ---------------------------------------------------------------------------
# Abstract worker spawner
# ---------------------------------------------------------------------------


class WorkerSpawner(abc.ABC):
    """Interface for spawning a worker to execute one PlannedTask."""

    @abc.abstractmethod
    async def spawn(
        self,
        task: PlannedTask,
        *,
        worktree: WorktreeInfo | None = None,
        task_brief: str = "",
        **kwargs: Any,
    ) -> WorkerResult:
        """Spawn a worker for the given task. Must be implemented by subclasses."""
        ...


# ---------------------------------------------------------------------------
# Dry-run / mock worker (for testing)
# ---------------------------------------------------------------------------


class DryRunWorker(WorkerSpawner):
    """Logs what it would do without actually executing code.

    Useful for testing the parallel executor end-to-end without
    real Hermes/DirectLLM calls.
    """

    def __init__(self, *, simulate_success: bool = True) -> None:
        self.simulate_success = simulate_success
        self._spawned: list[tuple[PlannedTask, WorkerResult]] = []

    async def spawn(
        self,
        task: PlannedTask,
        *,
        worktree: WorktreeInfo | None = None,
        task_brief: str = "",
        **kwargs: Any,
    ) -> WorkerResult:
        LOGGER.info(
            "dry_run: would spawn worker for %s (type=%s, model=%s, worktree=%s)",
            task.task_id,
            task.task_type.value,
            task.required_model_class.value,
            worktree.path if worktree else "none",
        )

        result = WorkerResult(
            task_id=task.task_id,
            status="dry_run",
            output=f"Dry-run output for {task.task_id}: {task.title}",
            changed_files=list(task.affected_file_globs),
            worktree=worktree,
            metadata={
                "task_type": task.task_type.value,
                "model_class": task.required_model_class.value,
                "parallel_group": task.parallel_group_id,
                "worktree_path": worktree.path if worktree else "",
            },
        )

        self._spawned.append((task, result))
        return result

    @property
    def spawned_count(self) -> int:
        return len(self._spawned)


# ---------------------------------------------------------------------------
# Direct LLM worker (calls into agent_graph's _ainvoke_direct_text)
# ---------------------------------------------------------------------------


DirectLLMFn = Any  # async callable: (prompt: str, max_tokens: int) -> str


class DirectLLMWorker(WorkerSpawner):
    """Spawns a direct LLM call for a task.

    Uses a callable (typically _ainvoke_direct_text from agent_graph)
    to avoid circular imports.
    """

    def __init__(self, llm_fn: DirectLLMFn | None = None) -> None:
        self._llm_fn = llm_fn

    def set_llm_fn(self, fn: DirectLLMFn) -> None:
        self._llm_fn = fn

    async def spawn(
        self,
        task: PlannedTask,
        *,
        worktree: WorktreeInfo | None = None,
        task_brief: str = "",
        **kwargs: Any,
    ) -> WorkerResult:
        if self._llm_fn is None:
            # Fall back to dry-run behavior
            dry = DryRunWorker()
            return await dry.spawn(task, worktree=worktree, task_brief=task_brief)

        context_budget = kwargs.get("context_budget", 0)
        max_tokens = kwargs.get(
            "max_tokens",
            4096 if task.required_model_class == ModelClass.BIG_MODEL else 1024,
        )

        # Build budget-aware prompt
        budget_line = ""
        if context_budget > 0:
            budget_line = (
                f"\nMAX_CONTEXT_BUDGET: {context_budget} tokens. "
                f"Your total context (prompt + output + tool results) MUST NOT exceed this. "
                f"If you approach the limit, summarize intermediate results. "
                f"If you need more, signal NEED_MORE_CONTEXT.\n"
            )

        prompt = (
            f"Task: {task.title}\n"
            f"Type: {task.task_type.value}\n"
            f"Context: {task_brief}\n"
            f"{budget_line}"
        )

        try:
            start = time.time()
            output = await self._llm_fn(prompt, max_tokens)
            elapsed = time.time() - start
            return WorkerResult(
                task_id=task.task_id,
                status="completed",
                output=str(output),
                worktree=worktree,
                elapsed_seconds=round(elapsed, 3),
            )
        except Exception as exc:
            return WorkerResult(
                task_id=task.task_id,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                worktree=worktree,
            )


# ---------------------------------------------------------------------------
# Hermes worker (wraps the existing call_hermes_agent path)
# ---------------------------------------------------------------------------


HermesAgentFn = Any  # async callable: (task: str, context: str, max_output_chars: int) -> str


class HermesWorker(WorkerSpawner):
    """Worker that delegates to the external Hermes Agent via its OpenAI API.

    Uses the existing call_hermes_agent tool path. A callable is injected
    at construction to avoid circular imports with agent_graph.

    Hermes is a controlled worker — it gets a bounded task with context,
    allowed tools, and a context budget. It does NOT:
    - Call LangGraph or AlphaRavis back
    - Spawn further workers autonomously
    - Make destructive decisions without approval
    """

    def __init__(
        self,
        hermes_fn: HermesAgentFn | None = None,
        *,
        model_override: str = "",
    ) -> None:
        self._hermes_fn = hermes_fn
        self.model_override = model_override

    def set_hermes_fn(self, fn: HermesAgentFn) -> None:
        """Inject the Hermes callable (call_hermes_agent from agent_graph)."""
        self._hermes_fn = fn

    async def spawn(
        self,
        task: PlannedTask,
        *,
        worktree: WorktreeInfo | None = None,
        task_brief: str = "",
        **kwargs: Any,
    ) -> WorkerResult:
        if self._hermes_fn is None:
            dry = DryRunWorker()
            return await dry.spawn(task, worktree=worktree, task_brief=task_brief)

        context_budget = kwargs.get("context_budget", 0)
        extra_context_parts: list[str] = []

        if task_brief:
            extra_context_parts.append(f"Task Brief: {task_brief}")

        if task.affected_file_globs:
            extra_context_parts.append(
                f"Affected files: {', '.join(task.affected_file_globs[:10])}"
            )

        if task.dependencies:
            extra_context_parts.append(
                f"Dependencies: {', '.join(task.dependencies)}"
            )

        if context_budget > 0:
            extra_context_parts.append(
                f"MAX_CONTEXT_BUDGET: {context_budget} tokens. "
                f"Your total context MUST NOT exceed this. "
                f"Summarize intermediate results if approaching limit. "
                f"Signal NEED_MORE_CONTEXT if you need more."
            )

        if task.write_enabled:
            extra_context_parts.append(
                "You MAY write files within the approved workspace. "
                "Respect AlphaRavis file safety: no credentials, caches, "
                "shell profiles, or OS/system paths."
            )
        else:
            extra_context_parts.append(
                "READ-ONLY mode: inspect, analyze, report. Do NOT write files."
            )

        extra_context = "\n".join(extra_context_parts)
        max_output_chars = min(
            kwargs.get("max_output_chars", 6000),
            context_budget if context_budget > 0 else 6000,
        )
        max_output_chars = max(1000, max_output_chars)

        try:
            start = time.time()
            output = await self._hermes_fn(
                task=task.title,
                context=extra_context,
                max_output_chars=max_output_chars,
            )
            elapsed = time.time() - start
            return WorkerResult(
                task_id=task.task_id,
                status="completed",
                output=str(output),
                worktree=worktree,
                elapsed_seconds=round(elapsed, 3),
                metadata={
                    "worker_type": "hermes",
                    "max_output_chars": max_output_chars,
                    "context_budget": context_budget,
                },
            )
        except Exception as exc:
            return WorkerResult(
                task_id=task.task_id,
                status="failed",
                error=f"HermesWorker: {type(exc).__name__}: {exc}",
                worktree=worktree,
            )


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------


class WorkerAdapterRegistry:
    """Registry of worker spawners keyed by worker type name."""

    def __init__(self) -> None:
        self._spawners: dict[str, WorkerSpawner] = {}

    def register(self, name: str, spawner: WorkerSpawner) -> None:
        self._spawners[name] = spawner

    def get(self, name: str) -> WorkerSpawner | None:
        return self._spawners.get(name)

    def get_for_task(self, task: PlannedTask) -> WorkerSpawner:
        """Select a spawner based on task type and model class."""
        # Hermes worker selection is controlled by feature flag,
        # not derived from task metadata. The caller (executor_node)
        # decides which worker type to use.
        if task.write_enabled:
            return self._spawners.get("hermes", DryRunWorker())
        if task.task_type.value in {"summarization", "classification"}:
            return self._spawners.get("direct_llm", DryRunWorker())
        return self._spawners.get("default", DryRunWorker())

    @property
    def available(self) -> list[str]:
        return list(self._spawners.keys())


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

GLOBAL_WORKER_REGISTRY = WorkerAdapterRegistry()
