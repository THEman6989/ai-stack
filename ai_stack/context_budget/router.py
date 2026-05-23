"""Percentage-based dynamic context budget router.

This module replaces fixed-token policies with percentage-derived decisions.
Every budget is calculated dynamically from the detected context pool size,
active slot usage, and configurable safety reserve percentages.

Design principles:
- Never hardcode a fixed token budget as permanent policy.
- Detect context pool size from the running llama-server, not from static config.
- Compute safety reserves as percentages of detected context.
- Compute output budgets from free context, task priority, and system load.
- Route requests based on dynamic state, not static task-type buckets.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ai_stack.llama_runtime.client import LlamaCppRuntimeClient
from ai_stack.ubuntu_llama_manager.schemas import UbuntuLlamaInstance


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Priority / route decision enums
# ---------------------------------------------------------------------------


class TaskPriority(str, Enum):
    CRITICAL_MAIN_AGENT = "critical_main_agent"
    CODING_AGENT = "coding_agent"
    LONG_ANALYSIS = "long_analysis"
    NORMAL_CHAT = "normal_chat"
    SUMMARIZATION = "summarization"
    BACKGROUND_TASK = "background_task"
    LOW_PRIORITY = "low_priority"

    @classmethod
    def _missing_(cls, value: object) -> "TaskPriority":
        if isinstance(value, str):
            lowered = value.strip().lower().replace(" ", "_")
            mapping: dict[str, TaskPriority] = {
                "critical": cls.CRITICAL_MAIN_AGENT,
                "high": cls.CRITICAL_MAIN_AGENT,
                "main_agent": cls.CRITICAL_MAIN_AGENT,
                "coding": cls.CODING_AGENT,
                "code": cls.CODING_AGENT,
                "analysis": cls.LONG_ANALYSIS,
                "long": cls.LONG_ANALYSIS,
                "normal": cls.NORMAL_CHAT,
                "chat": cls.NORMAL_CHAT,
                "medium": cls.NORMAL_CHAT,
                "summarize": cls.SUMMARIZATION,
                "summary": cls.SUMMARIZATION,
                "compression": cls.SUMMARIZATION,
                "background": cls.BACKGROUND_TASK,
                "speculative": cls.BACKGROUND_TASK,
                "low": cls.LOW_PRIORITY,
            }
            if lowered in mapping:
                return mapping[lowered]
        return cls.NORMAL_CHAT


class RouteAction(str, Enum):
    RUN_ON_BIG_MODEL = "run_on_big_model"
    RUN_ON_SMALL_MODEL = "run_on_small_model"
    QUEUE_DEFER = "queue_defer"
    SUMMARIZE_FIRST = "summarize_first"
    CHUNK_TASK = "chunk_task"
    REJECT_RETRY_LATER = "reject_retry_later"
    REJECT_NO_CAPACITY = "reject_no_capacity"


# ---------------------------------------------------------------------------
# Dynamic server state
# ---------------------------------------------------------------------------


@dataclass
class DynamicServerState:
    """Live state queried from a running llama-server."""

    instance_id: str
    base_url: str = ""
    model: str = ""

    # From /props or config
    context_pool_size: int = 0
    parallel_slots: int = 1
    kv_unified: bool = False

    # From /slots
    active_slots: int = 0
    idle_slots: int = 0
    estimated_kv_used: int = 0
    slots_detail: list[dict[str, Any]] = field(default_factory=list)

    # Cache / source markers
    source: str = "unknown"  # "props", "slots", "manager_config", "env_fallback"
    queried_at: float = 0.0
    error: str = ""

    @property
    def free_context(self) -> int:
        """Estimated remaining context budget."""
        if self.kv_unified:
            return max(0, self.context_pool_size - self.estimated_kv_used)
        # Conservative per-slot: track worst-case slot usage
        per_slot = max(1, self.context_pool_size // max(1, self.parallel_slots))
        return max(0, per_slot * self.idle_slots)

    @property
    def free_context_percent(self) -> float:
        if self.context_pool_size <= 0:
            return 0.0
        return self.free_context / self.context_pool_size

    @property
    def is_idle(self) -> bool:
        return self.active_slots == 0 and self.estimated_kv_used == 0

    @property
    def is_busy(self) -> bool:
        return self.free_context_percent < 0.20

    @property
    def is_critical_busy(self) -> bool:
        return self.free_context_percent < 0.05


class ServerStateProber:
    """Queries llama.cpp for /props, /slots, and extracts live context info."""

    @staticmethod
    async def probe(
        runtime: LlamaCppRuntimeClient,
        *,
        instance_id: str = "",
        model: str = "",
        kv_unified: bool = False,
        parallel_slots: int = 1,
    ) -> DynamicServerState:
        state = DynamicServerState(
            instance_id=instance_id,
            base_url=runtime.base_url,
            model=model,
            parallel_slots=parallel_slots,
            kv_unified=kv_unified,
        )

        # 1. Try /slots first — most informative
        try:
            slots_data = await runtime.get_slots()
            state.source = "slots"
            state.queried_at = time.time()
            state = ServerStateProber._parse_slots(state, slots_data)
        except Exception as exc:
            state.error = f"/slots failed: {exc}"
            LOGGER.debug("router /slots probe failed for %s: %s", instance_id, exc)

        # 2. Try /props for context size if /slots didn't give it
        if state.context_pool_size <= 0:
            try:
                props_data = await runtime.get_props()
                if state.source == "unknown":
                    state.source = "props"
                state = ServerStateProber._parse_props(state, props_data)
            except Exception as exc:
                # Don't overwrite slots error if slots gave partial data
                if state.source == "unknown":
                    state.error += f"; /props failed: {exc}"
                LOGGER.debug("router /props probe failed for %s: %s", instance_id, exc)

        # 3. Try /metrics for Prometheus-style data
        if state.estimated_kv_used == 0 and state.active_slots == 0:
            try:
                metrics_data = await runtime.get_metrics()
                state = ServerStateProber._parse_metrics(state, metrics_data)
            except Exception as exc:
                LOGGER.debug("router /metrics probe failed for %s: %s", instance_id, exc)

        return state

    @staticmethod
    def _parse_slots(state: DynamicServerState, data: Any) -> DynamicServerState:
        if not isinstance(data, list):
            return state

        active = 0
        idle = 0
        estimated_used = 0
        details: list[dict[str, Any]] = []

        for slot in data:
            if not isinstance(slot, dict):
                continue
            slot_id = slot.get("id", -1)
            slot_state = slot.get("state", 0)

            detail = {"id": slot_id, "state": slot_state}
            details.append(detail)

            if slot_state == 1:  # PROCESSING
                active += 1
                n_past = slot.get("n_past", 0)
                n_ctx = slot.get("n_ctx", state.context_pool_size)
                detail["n_past"] = n_past
                detail["n_ctx"] = n_ctx
                estimated_used += n_past

                # Extract context size from slot if not yet known
                if state.context_pool_size <= 0 and n_ctx > 0:
                    state.context_pool_size = n_ctx
            elif slot_state == 0:  # IDLE
                idle += 1

            # Extract context pool size from first slot n_ctx
            if state.context_pool_size <= 0:
                n_ctx = slot.get("n_ctx", 0)
                if n_ctx > 0:
                    state.context_pool_size = n_ctx

        state.active_slots = active
        state.idle_slots = idle
        state.estimated_kv_used = estimated_used
        state.slots_detail = details

        # If no active slots but we got slot data, context is likely free
        if active == 0 and state.estimated_kv_used == 0:
            state.estimated_kv_used = 0

        return state

    @staticmethod
    def _parse_props(state: DynamicServerState, data: Any) -> DynamicServerState:
        if isinstance(data, dict):
            total = data.get("total_slots") or data.get("n_ctx") or data.get("context_size")
            if total is not None:
                try:
                    state.context_pool_size = int(total)
                except (TypeError, ValueError):
                    pass

            parallel = data.get("n_parallel") or data.get("parallel_slots")
            if parallel is not None:
                try:
                    state.parallel_slots = int(parallel)
                except (TypeError, ValueError):
                    pass

            model_name = data.get("model_path") or data.get("model") or data.get("default_generation_settings", {}).get("model")
            if model_name:
                state.model = str(model_name)
        return state

    @staticmethod
    def _parse_metrics(state: DynamicServerState, data: Any) -> DynamicServerState:
        if not isinstance(data, str):
            return state

        for line in data.splitlines():
            line = line.strip()
            if line.startswith("#"):
                continue
            if "n_slots" in line or "slots_processing" in line:
                try:
                    val = float(line.rsplit(" ", 1)[-1])
                    if val > 0 and state.active_slots == 0:
                        state.active_slots = int(val)
                except (ValueError, IndexError):
                    pass
        return state


# ---------------------------------------------------------------------------
# Percentage-based budget policy
# ---------------------------------------------------------------------------


@dataclass
class PercentageBudgetPolicy:
    """All budget numbers are percentages of the detected context pool.

    Every value is configurable via env. No fixed token budgets.
    """

    # ---- Which priorities are primary / uncapped ----
    # Primary agents (main, coding, analysis, normal chat) get the full
    # usable free context as their output budget — no percentage cap.
    # Secondary agents (summarization, background, low-priority) use
    # their configured percentage caps.
    # Comma-separated list in env: ALPHARAVIS_BUDGET_UNCAPPED_PRIORITIES
    uncapped_priorities: tuple[TaskPriority, ...] = (
        TaskPriority.CRITICAL_MAIN_AGENT,
        TaskPriority.CODING_AGENT,
        TaskPriority.LONG_ANALYSIS,
        TaskPriority.NORMAL_CHAT,
    )

    # ---- Safety reserves (percent of context pool) ----
    safety_reserve_pct: float = 0.08          # e.g. 8% reserve for stability
    safety_reserve_multi_slot_pct: float = 0.15  # higher when multiple slots active
    safety_reserve_critical_pct: float = 0.12     # reserve for critical main-agent tasks

    # ---- Per-priority output budget caps (percent of free context) ----
    critical_output_pct: float = 0.50        # main agent gets up to 50% of free context
    coding_output_pct: float = 0.40          # coding gets up to 40%
    analysis_output_pct: float = 0.45        # long analysis gets up to 45%
    normal_chat_output_pct: float = 0.20     # normal chat gets up to 20%
    summarization_output_pct: float = 0.15   # summarization gets up to 15%
    background_output_pct: float = 0.10      # background gets up to 10%
    low_priority_output_pct: float = 0.05    # low priority gets up to 5%

    # ---- Admission thresholds (must have at least this fraction free) ----
    min_free_pct_critical: float = 0.05      # critical tasks need only 5% free
    min_free_pct_coding: float = 0.15        # coding needs 15%
    min_free_pct_analysis: float = 0.10      # analysis needs 10%
    min_free_pct_normal: float = 0.10        # normal needs 10%
    min_free_pct_background: float = 0.05    # background can run in tight spaces

    # ---- Queue / defer thresholds ----
    queue_when_free_below_pct: float = 0.10  # defer non-critical below 10% free
    route_to_small_when_free_below_pct: float = 0.15  # try small model below 15%

    # ---- Small model context ----
    small_model_context: int = 16384           # small model context (detected or configured)
    small_model_max_output_pct: float = 0.30   # small model output cap

    # ---- Output clamping ----
    min_output_tokens: int = 64
    max_output_tokens: int = 131072  # absolute ceiling regardless of percentage

    @classmethod
    def from_env(cls) -> "PercentageBudgetPolicy":
        def pct(key: str, default: float) -> float:
            return float(os.getenv(key, str(default)))

        # Parse uncapped priorities from env: comma-separated list of priority names
        uncapped_raw = os.getenv("ALPHARAVIS_BUDGET_UNCAPPED_PRIORITIES", "")
        if uncapped_raw.strip():
            uncapped = tuple(
                TaskPriority(p.strip())
                for p in uncapped_raw.split(",")
                if p.strip()
            )
        else:
            uncapped = cls.uncapped_priorities  # use class default

        return cls(
            uncapped_priorities=uncapped,
            safety_reserve_pct=pct("ALPHARAVIS_BUDGET_SAFETY_RESERVE_PCT", 0.08),
            safety_reserve_multi_slot_pct=pct("ALPHARAVIS_BUDGET_SAFETY_MULTI_SLOT_PCT", 0.15),
            safety_reserve_critical_pct=pct("ALPHARAVIS_BUDGET_SAFETY_CRITICAL_PCT", 0.12),
            critical_output_pct=pct("ALPHARAVIS_BUDGET_CRITICAL_OUTPUT_PCT", 0.50),
            coding_output_pct=pct("ALPHARAVIS_BUDGET_CODING_OUTPUT_PCT", 0.40),
            analysis_output_pct=pct("ALPHARAVIS_BUDGET_ANALYSIS_OUTPUT_PCT", 0.45),
            normal_chat_output_pct=pct("ALPHARAVIS_BUDGET_NORMAL_CHAT_OUTPUT_PCT", 0.20),
            summarization_output_pct=pct("ALPHARAVIS_BUDGET_SUMMARIZATION_OUTPUT_PCT", 0.15),
            background_output_pct=pct("ALPHARAVIS_BUDGET_BACKGROUND_OUTPUT_PCT", 0.10),
            low_priority_output_pct=pct("ALPHARAVIS_BUDGET_LOW_PRIORITY_OUTPUT_PCT", 0.05),
            min_free_pct_critical=pct("ALPHARAVIS_BUDGET_MIN_FREE_CRITICAL_PCT", 0.05),
            min_free_pct_coding=pct("ALPHARAVIS_BUDGET_MIN_FREE_CODING_PCT", 0.15),
            min_free_pct_analysis=pct("ALPHARAVIS_BUDGET_MIN_FREE_ANALYSIS_PCT", 0.10),
            min_free_pct_normal=pct("ALPHARAVIS_BUDGET_MIN_FREE_NORMAL_PCT", 0.10),
            min_free_pct_background=pct("ALPHARAVIS_BUDGET_MIN_FREE_BACKGROUND_PCT", 0.05),
            queue_when_free_below_pct=pct("ALPHARAVIS_BUDGET_QUEUE_FREE_BELOW_PCT", 0.10),
            route_to_small_when_free_below_pct=pct("ALPHARAVIS_BUDGET_ROUTE_SMALL_FREE_BELOW_PCT", 0.15),
            small_model_context=int(os.getenv("ALPHARAVIS_SMALL_MODEL_CONTEXT", "16384")),
            small_model_max_output_pct=pct("ALPHARAVIS_BUDGET_SMALL_MODEL_OUTPUT_PCT", 0.30),
            min_output_tokens=int(os.getenv("ALPHARAVIS_BUDGET_MIN_OUTPUT_TOKENS", "64")),
            max_output_tokens=int(os.getenv("ALPHARAVIS_BUDGET_MAX_OUTPUT_TOKENS", "131072")),
        )

    def compute_safety_reserve(self, state: DynamicServerState) -> int:
        """Safety reserve as a percentage of the detected context pool."""
        if state.active_slots > 1:
            pct = self.safety_reserve_multi_slot_pct
        else:
            pct = self.safety_reserve_pct
        return max(1, int(state.context_pool_size * pct))

    def compute_output_budget(
        self,
        state: DynamicServerState,
        priority: TaskPriority,
        *,
        free_context: int | None = None,
    ) -> int:
        """Compute max_tokens from free context and task priority.

        Primary agents (critical, coding, analysis, normal chat) get the
        full usable free context — no artificial percentage cap.
        Secondary agents (summarization, background, low-priority) use
        their configured percentage caps.
        """
        if free_context is None:
            free_context = state.free_context

        reserve = self.compute_safety_reserve(state)
        usable = max(0, free_context - reserve)

        # Primary agents: no cap, use full usable context
        if priority in self.uncapped_priorities:
            budget = max(self.min_output_tokens, usable)
            return min(budget, self.max_output_tokens)

        # Secondary agents: apply percentage cap
        pct_map = {
            TaskPriority.SUMMARIZATION: self.summarization_output_pct,
            TaskPriority.BACKGROUND_TASK: self.background_output_pct,
            TaskPriority.LOW_PRIORITY: self.low_priority_output_pct,
        }
        pct = pct_map.get(priority, self.normal_chat_output_pct)

        budget = max(self.min_output_tokens, int(usable * pct))
        return min(budget, self.max_output_tokens)

    def compute_small_model_output_budget(self) -> int:
        return max(self.min_output_tokens, int(self.small_model_context * self.small_model_max_output_pct))

    def can_admit(self, state: DynamicServerState, priority: TaskPriority, prompt_tokens: int) -> bool:
        """Check if the request can be admitted given free context and priority."""
        free = state.free_context
        reserve = self.compute_safety_reserve(state)
        usable = max(0, free - reserve)

        min_free_map = {
            TaskPriority.CRITICAL_MAIN_AGENT: self.min_free_pct_critical,
            TaskPriority.CODING_AGENT: self.min_free_pct_coding,
            TaskPriority.LONG_ANALYSIS: self.analysis_output_pct,
            TaskPriority.NORMAL_CHAT: self.min_free_pct_normal,
            TaskPriority.SUMMARIZATION: self.min_free_pct_background,
            TaskPriority.BACKGROUND_TASK: self.min_free_pct_background,
            TaskPriority.LOW_PRIORITY: self.min_free_pct_background,
        }
        min_free_pct = min_free_map.get(priority, self.min_free_pct_normal)
        min_free_tokens = max(1, int(state.context_pool_size * min_free_pct))

        return usable >= max(min_free_tokens, prompt_tokens + self.min_output_tokens)


# ---------------------------------------------------------------------------
# Route decision
# ---------------------------------------------------------------------------


@dataclass
class RouteDecision:
    action: RouteAction
    target_instance_id: str = ""
    max_output_tokens: int = 0
    reason: str = ""
    free_context: int = 0
    free_context_percent: float = 0.0
    safety_reserve_percent: float = 0.0
    safety_reserve_tokens: int = 0
    context_pool_size: int = 0
    priority: str = ""
    queued: bool = False
    can_use_small: bool = False

    diagnostic: dict[str, Any] = field(default_factory=dict)
    budget_notice: str = ""

    @property
    def ok(self) -> bool:
        return self.action in {RouteAction.RUN_ON_BIG_MODEL, RouteAction.RUN_ON_SMALL_MODEL}


# ---------------------------------------------------------------------------
# Budget notice injection
# ---------------------------------------------------------------------------


BUDGET_NOTICE_TIGHT = (
    "You are close to the available context budget. "
    "Give the shortest complete answer, avoid exploration, "
    "and stop after the essential result."
)

BUDGET_NOTICE_MODERATE = (
    "You have a limited output budget for this request. "
    "Prioritize the actionable result, avoid unnecessary explanation, "
    "and finish cleanly."
)

BUDGET_NOTICE_AMPLE = (
    "You have enough budget for a detailed solution, "
    "but still avoid unnecessary repetition."
)


def generate_budget_notice(output_budget: int, free_context: int) -> str:
    """Tell the model about its dynamic output limits in natural language."""
    if free_context <= 0:
        return "You have no free context budget left. Return the shortest possible answer."

    ratio = output_budget / max(1, free_context)

    if ratio > 0.80 or output_budget < 512:
        return BUDGET_NOTICE_TIGHT
    if ratio > 0.40 or output_budget < 4096:
        return BUDGET_NOTICE_MODERATE
    return BUDGET_NOTICE_AMPLE


# ---------------------------------------------------------------------------
# Priority-aware router
# ---------------------------------------------------------------------------


@dataclass
class _QueuedTask:
    """In-process deferred task entry."""

    task_id: str
    priority: TaskPriority
    prompt_tokens: int
    preferred_instance_id: str
    can_use_small: bool
    can_chunk: bool
    can_summarize_first: bool
    deadline: float = 0.0
    enqueued_at: float = 0.0


class PriorityAwareRouter:
    """Routes requests based on dynamic server state and percentage budgets."""

    def __init__(
        self,
        *,
        policy: PercentageBudgetPolicy | None = None,
    ) -> None:
        self.policy = policy or PercentageBudgetPolicy.from_env()
        self._queue: list[_QueuedTask] = []
        self._server_states: dict[str, DynamicServerState] = {}
        self._small_instance_id: str = "secondary"
        self._prober = ServerStateProber()
        self._lock = asyncio.Lock()

    # ---- Server state management ----

    async def refresh_state(
        self,
        runtime: LlamaCppRuntimeClient,
        *,
        instance: UbuntuLlamaInstance,
    ) -> DynamicServerState:
        """Probe live server state or build from manager config."""
        try:
            state = await self._prober.probe(
                runtime,
                instance_id=instance.id,
                model=instance.model,
                kv_unified=instance.kv_unified,
                parallel_slots=instance.parallel,
            )
        except Exception:
            state = DynamicServerState(
                instance_id=instance.id,
                base_url=instance.base_url,
                model=instance.model,
                context_pool_size=instance.ctx_total,
                parallel_slots=instance.parallel,
                kv_unified=instance.kv_unified,
                source="manager_config",
                error="probe failed, using manager config",
            )

        # Fallback: if probe gave no context size, use manager config
        if state.context_pool_size <= 0:
            state.context_pool_size = instance.ctx_total
            if state.source == "unknown":
                state.source = "manager_config_fallback"
            LOGGER.info(
                "router: no live context size for %s, using manager config=%d",
                instance.id,
                instance.ctx_total,
            )

        state.queried_at = time.time()
        async with self._lock:
            self._server_states[instance.id] = state
        return state

    def get_cached_state(self, instance_id: str) -> DynamicServerState | None:
        return self._server_states.get(instance_id)

    # ---- Routing ----

    async def route(
        self,
        *,
        instance: UbuntuLlamaInstance,
        runtime: LlamaCppRuntimeClient,
        priority: TaskPriority,
        prompt_tokens: int,
        estimated_output_need: int = 0,
        can_use_small: bool = True,
        can_defer: bool = True,
        can_chunk: bool = False,
        can_summarize_first: bool = False,
        preferred_instance_id: str = "",
        small_instance_id: str = "",
    ) -> RouteDecision:
        """Make a routing decision based on dynamic state."""

        if small_instance_id:
            self._small_instance_id = small_instance_id

        state = await self.refresh_state(runtime, instance=instance)

        reserve = self.policy.compute_safety_reserve(state)
        free = state.free_context

        diag: dict[str, Any] = {
            "context_pool_size": state.context_pool_size,
            "parallel_slots": state.parallel_slots,
            "kv_unified": state.kv_unified,
            "active_slots": state.active_slots,
            "idle_slots": state.idle_slots,
            "estimated_kv_used": state.estimated_kv_used,
            "free_context": free,
            "free_context_percent": round(state.free_context_percent, 4),
            "safety_reserve_pct": round(self.policy.safety_reserve_pct, 4),
            "safety_reserve_tokens": reserve,
            "prompt_tokens": prompt_tokens,
            "estimated_output_need": estimated_output_need,
            "source": state.source,
            "priority": priority.value,
        }

        # ---- Core routing logic ----

        # 1. Always try big model first if critical
        if priority == TaskPriority.CRITICAL_MAIN_AGENT:
            if self.policy.can_admit(state, priority, prompt_tokens):
                output = self.policy.compute_output_budget(state, priority)
                return RouteDecision(
                    action=RouteAction.RUN_ON_BIG_MODEL,
                    target_instance_id=instance.id,
                    max_output_tokens=output,
                    reason="critical task admitted on big model",
                    free_context=free,
                    free_context_percent=state.free_context_percent,
                    safety_reserve_percent=self.policy.safety_reserve_pct,
                    safety_reserve_tokens=reserve,
                    context_pool_size=state.context_pool_size,
                    priority=priority.value,
                    diagnostic=diag,
                    budget_notice=generate_budget_notice(output, free),
                )
            # Critical task but big model full — must try small or queue
            if can_use_small:
                return self._route_to_small(priority, instance, state, free, reserve, diag)
            return RouteDecision(
                action=RouteAction.QUEUE_DEFER,
                reason="critical task cannot be admitted, big model full, small not available",
                free_context=free,
                free_context_percent=state.free_context_percent,
                safety_reserve_percent=self.policy.safety_reserve_pct,
                safety_reserve_tokens=reserve,
                context_pool_size=state.context_pool_size,
                priority=priority.value,
                diagnostic=diag,
                queued=True,
            )

        # 2. Summarization/compression — prefer small model
        if priority in {TaskPriority.SUMMARIZATION, TaskPriority.BACKGROUND_TASK} and can_use_small:
            return self._route_to_small(priority, instance, state, free, reserve, diag)

        # 3. Check if big model has enough free context
        if self.policy.can_admit(state, priority, prompt_tokens):
            output = self.policy.compute_output_budget(state, priority)
            return RouteDecision(
                action=RouteAction.RUN_ON_BIG_MODEL,
                target_instance_id=instance.id,
                max_output_tokens=output,
                reason=f"admitted on big model, free={free} tokens",
                free_context=free,
                free_context_percent=state.free_context_percent,
                safety_reserve_percent=self.policy.safety_reserve_pct,
                safety_reserve_tokens=reserve,
                context_pool_size=state.context_pool_size,
                priority=priority.value,
                diagnostic=diag,
                budget_notice=generate_budget_notice(output, free),
            )

        # 4. Big model is busy — route to small or defer
        reason = f"big model busy (free_pct={state.free_context_percent:.2%})"

        if can_use_small and state.free_context_percent < self.policy.route_to_small_when_free_below_pct:
            return self._route_to_small(priority, instance, state, free, reserve, diag)

        # 5. Can we queue?
        if can_defer and priority not in {TaskPriority.CRITICAL_MAIN_AGENT, TaskPriority.CODING_AGENT}:
            return RouteDecision(
                action=RouteAction.QUEUE_DEFER,
                reason=f"{reason}; deferred",
                free_context=free,
                free_context_percent=state.free_context_percent,
                safety_reserve_percent=self.policy.safety_reserve_pct,
                safety_reserve_tokens=reserve,
                context_pool_size=state.context_pool_size,
                priority=priority.value,
                diagnostic=diag,
                queued=True,
            )

        # 6. Can we chunk?
        if can_chunk:
            return RouteDecision(
                action=RouteAction.CHUNK_TASK,
                reason=f"{reason}; chunk requested",
                free_context=free,
                free_context_percent=state.free_context_percent,
                safety_reserve_percent=self.policy.safety_reserve_pct,
                safety_reserve_tokens=reserve,
                context_pool_size=state.context_pool_size,
                priority=priority.value,
                diagnostic=diag,
            )

        # 7. Can we summarize first?
        if can_summarize_first:
            return RouteDecision(
                action=RouteAction.SUMMARIZE_FIRST,
                reason=f"{reason}; summarize-first requested",
                free_context=free,
                free_context_percent=state.free_context_percent,
                safety_reserve_percent=self.policy.safety_reserve_pct,
                safety_reserve_tokens=reserve,
                context_pool_size=state.context_pool_size,
                priority=priority.value,
                diagnostic=diag,
            )

        # 8. Last resort
        return RouteDecision(
            action=RouteAction.REJECT_NO_CAPACITY,
            reason=f"{reason}; no safe option available",
            free_context=free,
            free_context_percent=state.free_context_percent,
            safety_reserve_percent=self.policy.safety_reserve_pct,
            safety_reserve_tokens=reserve,
            context_pool_size=state.context_pool_size,
            priority=priority.value,
            diagnostic=diag,
        )

    def _route_to_small(
        self,
        priority: TaskPriority,
        instance: UbuntuLlamaInstance,
        state: DynamicServerState,
        free: int,
        reserve: int,
        diag: dict[str, Any],
    ) -> RouteDecision:
        output = self.policy.compute_small_model_output_budget()
        return RouteDecision(
            action=RouteAction.RUN_ON_SMALL_MODEL,
            target_instance_id=self._small_instance_id,
            max_output_tokens=output,
            reason=f"routed to small model (big free_pct={state.free_context_percent:.2%})",
            free_context=free,
            free_context_percent=state.free_context_percent,
            safety_reserve_percent=self.policy.safety_reserve_pct,
            safety_reserve_tokens=reserve,
            context_pool_size=state.context_pool_size,
            priority=priority.value,
            diagnostic=diag,
            can_use_small=True,
            budget_notice=generate_budget_notice(output, self.policy.small_model_context),
        )

    # ---- Observability ----

    def log_decision(self, decision: RouteDecision) -> None:
        """Log routing decision for observability."""
        extra = {
            "route_action": decision.action.value,
            "target_instance": decision.target_instance_id or "none",
            "priority": decision.priority,
            "free_context": decision.free_context,
            "free_context_pct": round(decision.free_context_percent, 4),
            "max_output_tokens": decision.max_output_tokens,
            "safety_reserve_pct": round(decision.safety_reserve_percent, 4),
            "safety_reserve_tokens": decision.safety_reserve_tokens,
            "context_pool_size": decision.context_pool_size,
            "reason": decision.reason,
            "queued": decision.queued,
        }
        LOGGER.info("context_budget_route %s", extra.get("route_action"), extra=extra)

    # ---- Defer queue ----

    async def enqueue(self, task: _QueuedTask) -> None:
        async with self._lock:
            self._queue.append(task)
            # Sort: critical first, then by priority, then by enqueue time
            self._queue.sort(key=lambda t: (
                0 if t.priority == TaskPriority.CRITICAL_MAIN_AGENT else 1,
                0 if t.priority == TaskPriority.CODING_AGENT else 1,
                t.enqueued_at,
            ))

    async def dequeue_next(self) -> _QueuedTask | None:
        async with self._lock:
            if not self._queue:
                return None
            return self._queue.pop(0)

    @property
    def queue_size(self) -> int:
        return len(self._queue)


# ---------------------------------------------------------------------------
# Global router singleton
# ---------------------------------------------------------------------------


_ROUTER: PriorityAwareRouter | None = None
_ROUTER_LOCK = asyncio.Lock()


async def get_priority_router() -> PriorityAwareRouter:
    global _ROUTER
    if _ROUTER is not None:
        return _ROUTER
    async with _ROUTER_LOCK:
        if _ROUTER is None:
            _ROUTER = PriorityAwareRouter()
        return _ROUTER
