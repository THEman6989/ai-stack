"""Parallel execution engine for Stage 2 task execution.

Takes a TaskDAG, spawns workers in parallel groups (concurrently via asyncio),
collects results, and runs a merge/review step.

Design:
- Parallel groups run concurrently via asyncio.gather().
- Write-enabled tasks get isolated git worktrees.
- Read-only tasks run in-process.
- Serial chain tasks run sequentially after parallel groups.
- Merge/review runs after all workers complete.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ai_stack.parallel_executor.task_graph import (
    ModelClass,
    PlannedTask,
    TaskDAG,
    TaskType,
)
from ai_stack.parallel_executor.worktree_manager import WorktreeInfo, WorktreeManager
from ai_stack.parallel_executor.worker_spawner import (
    DryRunWorker,
    WorkerResult,
    WorkerSpawner,
)
from ai_stack.parallel_executor.file_lock import FileLock, FileLockManager, GLOBAL_FILE_LOCK_MANAGER

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Execution plan
# ---------------------------------------------------------------------------


@dataclass
class ExecutionPlan:
    """Ordered execution plan derived from a TaskDAG."""

    parallel_groups: list[list[PlannedTask]]  # each inner list runs concurrently
    serial_chain: list[PlannedTask]           # runs sequentially after groups
    merge_task: PlannedTask | None = None      # final merge/review


@dataclass
class ExecutionReport:
    """Result of a parallel execution run."""

    results: list[WorkerResult] = field(default_factory=list)
    merge_result: WorkerResult | None = None
    total_tasks: int = 0
    completed: int = 0
    failed: int = 0
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.failed == 0 and not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [r for r in self.results],
            "merge_result": self.merge_result,
            "total_tasks": self.total_tasks,
            "completed": self.completed,
            "failed": self.failed,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Plan builder
# ---------------------------------------------------------------------------


def build_execution_plan(dag: TaskDAG) -> ExecutionPlan:
    """Convert a TaskDAG into an ordered ExecutionPlan.

    Parallel groups run first (each group concurrently). Then serial chain
    tasks run in order. A final merge/review task is appended.
    """
    parallel_groups: list[list[PlannedTask]] = []

    # Group tasks by parallel_group_id
    seen: set[str] = set()
    for task in dag.tasks:
        if task.can_parallelize and task.parallel_group_id:
            if task.task_id not in seen:
                group = [t for t in dag.tasks if t.parallel_group_id == task.parallel_group_id]
                parallel_groups.append(group)
                seen.update(t.task_id for t in group)

    # Serial chain: tasks that cannot parallelize
    serial_chain: list[PlannedTask] = []
    for task_id in dag.serial_chain:
        task = dag.get_task(task_id)
        if task and task.task_id not in seen:
            serial_chain.append(task)
            seen.add(task.task_id)

    # Any remaining unassigned tasks go to serial chain
    for task in dag.tasks:
        if task.task_id not in seen:
            serial_chain.append(task)
            seen.add(task.task_id)

    # Create a merge task if there are multiple workers
    merge_task = None
    if len(dag.tasks) > 1:
        merge_task = PlannedTask(
            task_id="merge_review",
            title="Merge and review results from parallel workers",
            task_type=TaskType.MERGE_REVIEW,
            read_only=False,
            write_enabled=True,
            required_model_class=ModelClass.BIG_MODEL,
            dependencies=[t.task_id for t in dag.tasks],
            reason_for_parallelization_decision="merge/review must be serialized",
        )

    return ExecutionPlan(
        parallel_groups=parallel_groups,
        serial_chain=serial_chain,
        merge_task=merge_task,
    )


# ---------------------------------------------------------------------------
# Parallel executor
# ---------------------------------------------------------------------------


class ParallelExecutor:
    """Executes a TaskDAG: parallel groups concurrently, then serial chain."""

    def __init__(
        self,
        *,
        spawner: WorkerSpawner | None = None,
        worktree_manager: WorktreeManager | None = None,
        merge_spawner: WorkerSpawner | None = None,
        file_lock_manager: FileLockManager | None = None,
    ) -> None:
        self.spawner = spawner or DryRunWorker()
        self.worktrees = worktree_manager or WorktreeManager()
        self.merge_spawner = merge_spawner or self.spawner
        self.file_locks = file_lock_manager or GLOBAL_FILE_LOCK_MANAGER

    async def execute(
        self,
        dag: TaskDAG,
        *,
        task_brief: str = "",
    ) -> ExecutionReport:
        """Execute a TaskDAG and return an ExecutionReport."""
        started = time.perf_counter()
        plan = build_execution_plan(dag)
        report = ExecutionReport(total_tasks=dag.task_count)

        LOGGER.info(
            "parallel_executor: starting %d tasks (%d groups, %d serial)",
            dag.task_count,
            len(plan.parallel_groups),
            len(plan.serial_chain),
        )

        # ---- Phase 1: Run parallel groups ----
        all_results: list[WorkerResult] = []

        for group in plan.parallel_groups:
            group_tasks = [t for t in group]
            LOGGER.info(
                "parallel_executor: running group %s (%d tasks)",
                group_tasks[0].parallel_group_id if group_tasks else "?",
                len(group_tasks),
            )
            group_results = await asyncio.gather(
                *[self._run_task(task, task_brief=task_brief) for task in group_tasks]
            )
            all_results.extend(group_results)

        # ---- Phase 2: Run serial chain ----
        for task in plan.serial_chain:
            result = await self._run_task(task, task_brief=task_brief)
            all_results.append(result)

        # ---- Phase 3: Merge/review ----
        if plan.merge_task:
            LOGGER.info("parallel_executor: running merge/review")
            completed = [r for r in all_results if r.ok]
            merge_input = self._build_merge_prompt(plan.merge_task, completed, task_brief)
            merge_result = await self.merge_spawner.spawn(
                plan.merge_task,
                task_brief=merge_input,
            )
            report.merge_result = merge_result

        # ---- Finalize ----
        report.results = all_results
        report.completed = sum(1 for r in all_results if r.ok)
        report.failed = sum(1 for r in all_results if r.status == "failed")
        report.elapsed_seconds = time.perf_counter() - started

        for r in all_results:
            if r.status == "failed":
                report.errors.append(f"{r.task_id}: {r.error}")

        LOGGER.info(
            "parallel_executor: done — %d/%d OK, %d failed, %.1fs",
            report.completed,
            report.total_tasks,
            report.failed,
            report.elapsed_seconds,
        )

        return report

    async def _run_task(
        self,
        task: PlannedTask,
        *,
        task_brief: str = "",
    ) -> WorkerResult:
        """Run a single task, with worktree if write-enabled and file-lock safety."""
        worktree: WorktreeInfo | None = None
        file_lock: FileLock | None = None

        # Acquire file lock for write tasks with file globs
        if task.write_enabled and task.affected_file_globs:
            file_lock = await self.file_locks.try_acquire(
                task.task_id,
                task.affected_file_globs,
            )
            if file_lock is None:
                conflicting = await self.file_locks.check_conflict(
                    task.affected_file_globs,
                    exclude_task_id=task.task_id,
                )
                return WorkerResult(
                    task_id=task.task_id,
                    status="skipped",
                    error=f"file lock conflict with: {', '.join(conflicting)}",
                )

        # Create worktree for write tasks
        if task.write_enabled and self.worktrees.is_git_repo:
            worktree = self.worktrees.create(task.task_id)

        try:
            result = await self.spawner.spawn(
                task,
                worktree=worktree,
                task_brief=task_brief,
            )
        except Exception as exc:
            result = WorkerResult(
                task_id=task.task_id,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                worktree=worktree,
            )

        # Cleanup worktree if task completed successfully
        if worktree and result.ok and not self.worktrees.check_uncommitted(task.task_id):
            self.worktrees.remove(task.task_id, force=True)

        # Release file lock
        if file_lock:
            await self.file_locks.release(task.task_id)

        return result

    @staticmethod
    def _build_merge_prompt(
        merge_task: PlannedTask,
        completed_results: list[WorkerResult],
        task_brief: str,
    ) -> str:
        """Build a merge/review prompt from completed worker results."""
        parts = [
            "Merge and review the following parallel worker results.",
            f"Original task: {task_brief}" if task_brief else "",
            "",
            "Worker outputs:",
        ]
        for r in completed_results:
            parts.append(f"\n--- {r.task_id} ({r.status}) ---")
            parts.append(r.output[:3000] if r.output else "(no output)")
            if r.changed_files:
                parts.append(f"Changed files: {', '.join(r.changed_files[:20])}")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Convenience: run a DAG if parallel execution is enabled
# ---------------------------------------------------------------------------


_GLOBAL_EXECUTOR: ParallelExecutor | None = None


def get_executor() -> ParallelExecutor:
    global _GLOBAL_EXECUTOR
    if _GLOBAL_EXECUTOR is None:
        _GLOBAL_EXECUTOR = ParallelExecutor()
    return _GLOBAL_EXECUTOR
