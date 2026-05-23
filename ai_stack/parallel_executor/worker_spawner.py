"""Worker spawner for parallel task execution.

Provides an adapter interface for spawning workers (Codex, Hermes, direct LLM)
in isolated worktrees. Ships with a dry-run / mock worker for testing.
Real Codex/Hermes adapters can be plugged in later.

Design:
- Abstract base class defines the spawner contract.
- DryRunWorker logs what it would do without executing.
- Real adapters (CodexSpawner, HermesSpawner) can be added later.
"""

from __future__ import annotations

import abc
import logging
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
    real Codex/Hermes calls.
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
# Adapter registry (for future real adapters)
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
        # For now, always return dry-run unless a real spawner is registered
        if task.write_enabled:
            return self._spawners.get("codex", DryRunWorker())
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
