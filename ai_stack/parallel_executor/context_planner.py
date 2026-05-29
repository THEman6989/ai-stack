"""Conservative context planner for parallel task execution.

Pre-estimates token budgets per worker via llama.cpp /tokenize API,
manages KV-unified pool admission with asymmetric slot distribution,
and enforces global safety reserves.

Key design:
- Material is tokenized via llama.cpp /tokenize — NOT loaded into own context.
- 20% overhead on estimates + 8% global pool reserve.
- Asymmetric budgets: Worker A 120k, Worker B 30k, as needed.
- Admission control: no worker starts if available < requested.
- All feature-flagged via ALPHARAVIS_PARALLEL_CONTEXT_PLANNER (default OFF).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_stack.llama_runtime.client import LlamaCppRuntimeClient

from ai_stack.parallel_executor.task_graph import PlannedTask

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker context estimate
# ---------------------------------------------------------------------------


@dataclass
class WorkerContextEstimate:
    """Token budget estimate for one worker, computed BEFORE the worker starts."""

    task_id: str
    prompt_tokens: int = 0
    rag_tokens: int = 0
    tool_output_tokens: int = 0
    file_snippet_tokens: int = 0
    total_estimated: int = 0
    safety_overhead: int = 0  # 20% of total_estimated
    recommended_budget: int = 0  # total_estimated + safety_overhead
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt_tokens": self.prompt_tokens,
            "rag_tokens": self.rag_tokens,
            "tool_output_tokens": self.tool_output_tokens,
            "file_snippet_tokens": self.file_snippet_tokens,
            "total_estimated": self.total_estimated,
            "safety_overhead": self.safety_overhead,
            "recommended_budget": self.recommended_budget,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# Slot budget tracker for KV-unified pools
# ---------------------------------------------------------------------------


@dataclass
class SlotBudget:
    """Tracks the KV-unified context pool across parallel workers.

    For KV-unified mode, all slots share one pool. This tracker ensures
    the global pool is never exhausted and maintains a safety reserve.

    Asymmetric distribution: workers get different budgets based on need,
    not an equal ctx_total/parallel split.
    """

    pool_total: int  # e.g. 320000
    parallel_slots: int  # e.g. 4 (np=4)
    kv_unified: bool = True
    safety_reserve_pct: float = 0.08  # 8% global reserve

    active_budgets: dict[str, int] = field(default_factory=dict)

    @property
    def safety_reserve(self) -> int:
        return int(self.pool_total * self.safety_reserve_pct)

    @property
    def usable_pool(self) -> int:
        """Total pool minus safety reserve — the budget available for distribution."""
        return max(0, self.pool_total - self.safety_reserve)

    @property
    def allocated(self) -> int:
        return sum(self.active_budgets.values())

    @property
    def available(self) -> int:
        """Remaining budget that can be assigned to new workers."""
        return max(0, self.usable_pool - self.allocated)

    @property
    def active_count(self) -> int:
        return len(self.active_budgets)

    def can_admit(self, requested: int) -> bool:
        """Check if a worker with the given budget can be admitted."""
        if self.active_count >= self.parallel_slots:
            return False
        return self.available >= requested

    def admit(self, task_id: str, budget: int) -> bool:
        """Reserve budget for a worker. Returns False if admission fails."""
        if not self.can_admit(budget):
            return False
        self.active_budgets[task_id] = budget
        return True

    def release(self, task_id: str) -> int | None:
        """Release a worker's budget back to the pool."""
        return self.active_budgets.pop(task_id, None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pool_total": self.pool_total,
            "parallel_slots": self.parallel_slots,
            "kv_unified": self.kv_unified,
            "safety_reserve": self.safety_reserve,
            "usable_pool": self.usable_pool,
            "allocated": self.allocated,
            "available": self.available,
            "active_count": self.active_count,
            "active_budgets": dict(self.active_budgets),
        }


# ---------------------------------------------------------------------------
# Context pre-estimator
# ---------------------------------------------------------------------------


class ContextPreEstimator:
    """Tokenizes worker material via llama.cpp /tokenize API.

    Material includes: task prompt, RAG results, tool descriptions,
    affected file snippets. None of this enters the estimator's own
    context — it's sent directly to /tokenize.

    When no runtime client is available (e.g. manager not configured),
    falls back to heuristic estimates based on character counts.
    """

    DEFAULT_CHARS_PER_TOKEN: float = 3.5
    DEFAULT_RAG_OVERHEAD: int = 4096
    DEFAULT_TOOL_OVERHEAD: int = 2048
    SAFETY_OVERHEAD_PCT: float = 0.20

    def __init__(self, runtime_client: LlamaCppRuntimeClient | None = None) -> None:
        """runtime_client: LlamaCppRuntimeClient or None for heuristic fallback."""
        self.runtime = runtime_client
        self.chars_per_token = float(
            os.getenv("ALPHARAVIS_TOKEN_ESTIMATE_CHARS_PER_TOKEN", "3.5")
        )

    @property
    def has_runtime(self) -> bool:
        return self.runtime is not None

    async def _estimate_tokens(self, text: str) -> int:
        """Count tokens via /tokenize if runtime available, else heuristic."""
        if not text:
            return 0
        if self.has_runtime:
            try:
                return await self.runtime.count_tokens_text(text)
            except Exception as exc:
                LOGGER.debug("context_planner: /tokenize failed, falling back: %s", exc)
        return max(1, int(len(text) / self.chars_per_token))

    async def estimate_worker(
        self,
        task: PlannedTask,
        *,
        task_brief: str = "",
        rag_material: list[str] | None = None,
        affected_file_contents: dict[str, str] | None = None,
        tool_descriptions: str = "",
    ) -> WorkerContextEstimate:
        """Pre-estimate token budget for one worker.

        Tokenizes:
        1. Task prompt (title + brief + instructions)
        2. RAG material
        3. File snippets (first ~200 lines each)
        4. Tool descriptions

        Budget = sum of all + 20% safety overhead.
        """
        # 1. Prompt tokens
        prompt_text = (
            f"Task: {task.title}\n"
            f"Type: {task.task_type.value}\n"
            f"Brief: {task_brief}\n"
        )
        prompt_tokens = await self._estimate_tokens(prompt_text)

        # 2. RAG material
        rag_tokens = 0
        if rag_material:
            rag_text = "\n---\n".join(rag_material)
            rag_tokens = await self._estimate_tokens(rag_text)

        # 3. File snippets
        file_tokens = 0
        if affected_file_contents:
            for content in affected_file_contents.values():
                if content:
                    file_tokens += await self._estimate_tokens(content)

        # 4. Tool overhead (use heuristic — tool descriptions are known size)
        tool_tokens = 0
        if tool_descriptions:
            tool_tokens = await self._estimate_tokens(tool_descriptions)
        else:
            tool_tokens = self.DEFAULT_TOOL_OVERHEAD

        total = prompt_tokens + rag_tokens + tool_tokens + file_tokens
        overhead = max(512, int(total * self.SAFETY_OVERHEAD_PCT))
        budget = total + overhead

        return WorkerContextEstimate(
            task_id=task.task_id,
            prompt_tokens=prompt_tokens,
            rag_tokens=rag_tokens,
            tool_output_tokens=tool_tokens,
            file_snippet_tokens=file_tokens,
            total_estimated=total,
            safety_overhead=overhead,
            recommended_budget=budget,
        )


# ---------------------------------------------------------------------------
# Parallel context planner (orchestrator)
# ---------------------------------------------------------------------------


@dataclass
class AdmissionResult:
    """Result of attempting to admit a set of workers."""

    ok: bool
    slot_budget: SlotBudget
    estimates: dict[str, WorkerContextEstimate] = field(default_factory=dict)
    admitted: list[str] = field(default_factory=list)  # task_ids that were admitted
    refused: list[str] = field(default_factory=list)    # task_ids that were refused
    reason: str = ""

    def budget_for(self, task_id: str) -> int:
        est = self.estimates.get(task_id)
        if est:
            return est.recommended_budget
        return 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "slot_budget": self.slot_budget.to_dict(),
            "admitted": self.admitted,
            "refused": self.refused,
            "reason": self.reason,
            "estimates": {
                tid: est.to_dict() for tid, est in self.estimates.items()
            },
        }


class ParallelContextPlanner:
    """Orchestrates pre-estimation and admission for parallel workers.

    Flow:
    1. estimate_all() — pre-estimate token budgets for all tasks
    2. admit_all() — admit workers in priority order, refuse if pool full
    3. release() — release budget when worker completes
    """

    def __init__(
        self,
        *,
        estimator: ContextPreEstimator | None = None,
        pool_total: int = 0,
        parallel_slots: int = 1,
        kv_unified: bool = True,
        safety_reserve_pct: float | None = None,
    ) -> None:
        self.estimator = estimator or ContextPreEstimator()
        self.safety_reserve_pct = (
            safety_reserve_pct
            if safety_reserve_pct is not None
            else float(
                os.getenv("ALPHARAVIS_PARALLEL_CONTEXT_SAFETY_RESERVE", "0.08")
            )
        )
        self.slot_budget = SlotBudget(
            pool_total=pool_total or _env_int("ALPHARAVIS_CONTEXT_POOL_TOTAL", 320000),
            parallel_slots=parallel_slots or _env_int("ALPHARAVIS_CONTEXT_PARALLEL_SLOTS", 4),
            kv_unified=kv_unified,
            safety_reserve_pct=self.safety_reserve_pct,
        )
        self.max_ratio = float(
            os.getenv("ALPHARAVIS_PARALLEL_WORKER_MAX_CONTEXT_RATIO", "0.85")
        )

    async def estimate_all(
        self,
        tasks: list[PlannedTask],
        *,
        task_brief: str = "",
        rag_material: list[str] | None = None,
        affected_files: dict[str, str] | None = None,
    ) -> dict[str, WorkerContextEstimate]:
        """Pre-estimate budgets for all tasks in parallel."""
        estimates: dict[str, WorkerContextEstimate] = {}
        for task in tasks:
            est = await self.estimator.estimate_worker(
                task,
                task_brief=task_brief,
                rag_material=rag_material,
                affected_file_contents=affected_files,
            )
            estimates[task.task_id] = est
        return estimates

    def admit_all(
        self,
        estimates: dict[str, WorkerContextEstimate],
        *,
        priority_order: list[str] | None = None,
    ) -> AdmissionResult:
        """Admit workers to the slot budget in priority order.

        Workers that can't fit are refused. The caller can then:
        - Wait for running workers to complete and retry
        - Reduce RAG material and re-estimate
        - Run refused tasks serially
        """
        tasks_in_order = priority_order or list(estimates.keys())
        admitted: list[str] = []
        refused: list[str] = []

        for task_id in tasks_in_order:
            est = estimates.get(task_id)
            if est is None:
                continue

            # Cap individual budget at max_ratio * pool_total
            capped_budget = min(
                est.recommended_budget,
                int(self.slot_budget.pool_total * self.max_ratio),
            )

            if self.slot_budget.admit(task_id, capped_budget):
                admitted.append(task_id)
            else:
                refused.append(task_id)

        result = AdmissionResult(
            ok=len(admitted) > 0,
            slot_budget=self.slot_budget,
            estimates=estimates,
            admitted=admitted,
            refused=refused,
            reason=""
            if not refused
            else f"{len(refused)} worker(s) refused: pool full ("
            f"allocated={self.slot_budget.allocated}/{self.slot_budget.usable_pool}, "
            f"available={self.slot_budget.available})",
        )

        LOGGER.info(
            "parallel_context_planner: admitted=%d refused=%d "
            "pool=%d/%d reserve=%d",
            len(admitted),
            len(refused),
            self.slot_budget.allocated,
            self.slot_budget.pool_total,
            self.slot_budget.safety_reserve,
        )

        return result

    def release(self, task_id: str) -> None:
        """Release a worker's budget back to the pool."""
        self.slot_budget.release(task_id)

    def is_full(self) -> bool:
        """Check if the pool can accept no more workers."""
        return (
            self.slot_budget.active_count >= self.slot_budget.parallel_slots
            or self.slot_budget.available <= 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_budget": self.slot_budget.to_dict(),
            "safety_reserve_pct": self.safety_reserve_pct,
            "max_ratio": self.max_ratio,
            "is_full": self.is_full(),
        }


# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------


def parallel_context_planner_enabled() -> bool:
    """Check if the parallel context planner feature is enabled."""
    return _env_bool("ALPHARAVIS_PARALLEL_CONTEXT_PLANNER", "false")


def parallel_hermes_worker_enabled() -> bool:
    """Check if Hermes worker mode is enabled."""
    return _env_bool("ALPHARAVIS_PARALLEL_HERMES_WORKER", "false")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: str = "false") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "y", "on"}
