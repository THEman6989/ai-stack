"""Parallel task execution for AlphaRavis.

When ALPHARAVIS_PARALLEL_TASK_EXECUTION=true, the planner's output is
parsed into a structured task DAG. Independent tasks run in parallel
via git worktrees; conflicting tasks are serialized.

When disabled (default), the existing sequential swarm path is used.
"""

from ai_stack.parallel_executor.task_graph import (
    PlannedTask,
    TaskDAG,
    TaskType,
    ModelClass,
    RiskLevel,
    analyze_parallelization,
    log_parallelization_decision,
    parallel_execution_enabled,
    parallel_planner_instruction_block,
    parse_planner_text_into_tasks,
)

from ai_stack.parallel_executor.worktree_manager import (
    WorktreeInfo,
    WorktreeManager,
    repo_has_uncommitted_changes,
    repo_current_branch,
)

from ai_stack.parallel_executor.worker_spawner import (
    WorkerResult,
    WorkerSpawner,
    DryRunWorker,
    DirectLLMWorker,
    DirectLLMFn,
    WorkerAdapterRegistry,
    GLOBAL_WORKER_REGISTRY,
)

from ai_stack.parallel_executor.file_lock import (
    FileLock,
    FileLockManager,
    GLOBAL_FILE_LOCK_MANAGER,
)

from ai_stack.parallel_executor.executor import (
    ExecutionPlan,
    ExecutionReport,
    ParallelExecutor,
    build_execution_plan,
    get_executor,
)

__all__ = [
    # Task graph
    "PlannedTask",
    "TaskDAG",
    "TaskType",
    "ModelClass",
    "RiskLevel",
    "analyze_parallelization",
    "log_parallelization_decision",
    "parallel_execution_enabled",
    "parallel_planner_instruction_block",
    "parse_planner_text_into_tasks",
    # Worktree
    "WorktreeInfo",
    "WorktreeManager",
    "repo_has_uncommitted_changes",
    "repo_current_branch",
    # Worker spawner
    "WorkerResult",
    "WorkerSpawner",
    "DryRunWorker",
    "DirectLLMWorker",
    "DirectLLMFn",
    "WorkerAdapterRegistry",
    "GLOBAL_WORKER_REGISTRY",
    # File lock
    "FileLock",
    "FileLockManager",
    "GLOBAL_FILE_LOCK_MANAGER",
    # Executor
    "ExecutionPlan",
    "ExecutionReport",
    "ParallelExecutor",
    "build_execution_plan",
    "get_executor",
]
