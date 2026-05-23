from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

from ai_stack.context_budget.leases import GLOBAL_LEASE_STORE, ContextLease, LeaseStore
from ai_stack.context_budget.policies import RuntimeConfig, capacity_limit, parse_runtime_config_from_command
from ai_stack.context_budget.router import (
    DynamicServerState,
    PercentageBudgetPolicy,
    PriorityAwareRouter,
    RouteAction,
    RouteDecision,
    TaskPriority,
    generate_budget_notice,
    get_priority_router,
)
from ai_stack.llama_runtime.client import LlamaCppRuntimeClient
from ai_stack.ubuntu_llama_manager.client import UbuntuLlamaManagerClient
from ai_stack.ubuntu_llama_manager.schemas import UbuntuLlamaInstance


LOGGER = logging.getLogger(__name__)


class ContextScheduler:
    def __init__(
        self,
        *,
        manager_client: UbuntuLlamaManagerClient | None = None,
        lease_store: LeaseStore | None = None,
        safety_factor: float = 0.92,
        background_safety_factor: float | None = None,
        router: PriorityAwareRouter | None = None,
        budget_policy: PercentageBudgetPolicy | None = None,
    ) -> None:
        self.manager_client = manager_client
        self.lease_store = lease_store or GLOBAL_LEASE_STORE
        self.safety_factor = safety_factor
        self.background_safety_factor = (
            background_safety_factor
            if background_safety_factor is not None
            else float(os.getenv("ALPHARAVIS_BACKGROUND_CONTEXT_MAX_UTILIZATION", "0.70"))
        )
        self.instances: dict[str, UbuntuLlamaInstance] = {}
        self.router = router
        self.budget_policy = budget_policy or PercentageBudgetPolicy.from_env()

    @classmethod
    def from_env(cls) -> "ContextScheduler | None":
        manager = UbuntuLlamaManagerClient.from_env()
        if manager is None:
            return None
        return cls(
            manager_client=manager,
            safety_factor=float(os.getenv("ALPHARAVIS_CONTEXT_SAFETY_FACTOR", "0.92")),
            background_safety_factor=float(os.getenv("ALPHARAVIS_BACKGROUND_CONTEXT_MAX_UTILIZATION", "0.70")),
        )

    async def _get_router(self) -> PriorityAwareRouter:
        if self.router is not None:
            return self.router
        return await get_priority_router()

    async def refresh_instances_from_manager(self) -> list[UbuntuLlamaInstance]:
        if self.manager_client is None:
            return []
        instances = await self.manager_client.get_instances()
        self.instances = {instance.id: instance for instance in instances if instance.id and instance.base_url}
        return list(self.instances.values())

    def parse_runtime_config_from_command(
        self,
        command: str | None,
        *,
        ctx_total: int | str | None = None,
        parallel: int | str | None = None,
        kv_unified: bool | None = None,
    ) -> RuntimeConfig:
        return parse_runtime_config_from_command(command, ctx_total=ctx_total, parallel=parallel, kv_unified=kv_unified)

    async def choose_instance(
        self,
        *,
        required_tokens: int = 0,
        preferred_instance_id: str = "",
        priority: str = "medium",
        background: bool = False,
        speculative: bool = False,
    ) -> UbuntuLlamaInstance | None:
        if not self.instances:
            await self.refresh_instances_from_manager()
        candidates = list(self.instances.values())
        if preferred_instance_id:
            candidates.sort(key=lambda item: 0 if item.id == preferred_instance_id else 1)
        if priority == "high":
            candidates.sort(key=lambda item: 0 if item.kv_unified else 1)
        if background:
            candidates.sort(key=lambda item: 0 if item.id == "secondary" else 1)
        for instance in candidates:
            limit = self._instance_capacity(instance, background=background, speculative=speculative)
            active = await self.lease_store.active_required_tokens(instance.id)
            if active + max(0, required_tokens) <= limit:
                return instance
        return candidates[0] if candidates else None

    async def estimate_and_reserve(
        self,
        *,
        messages: list[Any] | None = None,
        text: str | None = None,
        max_output_tokens: int = 0,  # 0 = use dynamic budget from router
        tool_context_tokens: int = 0,
        safety_margin: int = 0,  # 0 = use dynamic reserve from router
        graph_run_id: str = "",
        request_id: str = "",
        agent_name: str = "agent",
        priority: str = "medium",
        preferred_instance_id: str = "",
        background: bool = False,
        speculative: bool = False,
        can_use_small: bool = True,
        can_defer: bool = True,
        can_chunk: bool = False,
        can_summarize_first: bool = False,
    ) -> tuple[ContextLease | None, dict[str, Any]]:
        task_priority = TaskPriority(priority)

        instance = await self.choose_instance(
            preferred_instance_id=preferred_instance_id,
            priority=priority,
            background=background,
            speculative=speculative,
        )
        if instance is None:
            return None, {"ok": False, "reason": "no_llama_instances"}

        runtime = LlamaCppRuntimeClient.from_instance(
            instance,
            timeout_seconds=float(os.getenv("ALPHARAVIS_LLAMA_RUNTIME_TIMEOUT_SECONDS", "30")),
        )
        prompt_tokens = await self._count_prompt_tokens(runtime, messages=messages, text=text)

        # ---- Dynamic routing via PriorityAwareRouter ----
        router = await self._get_router()
        route = await router.route(
            instance=instance,
            runtime=runtime,
            priority=task_priority,
            prompt_tokens=prompt_tokens,
            estimated_output_need=max_output_tokens if max_output_tokens > 0 else 0,
            can_use_small=can_use_small,
            can_defer=can_defer,
            can_chunk=can_chunk,
            can_summarize_first=can_summarize_first,
            preferred_instance_id=preferred_instance_id,
            small_instance_id="secondary",
        )
        router.log_decision(route)

        # ---- Compute dynamic budget values ----
        route_state = router.get_cached_state(instance.id)
        if route_state is None:
            route_state = DynamicServerState(
                instance_id=instance.id,
                context_pool_size=instance.ctx_total,
                parallel_slots=instance.parallel,
                kv_unified=instance.kv_unified,
            )

        dynamic_max_output = (
            max_output_tokens if max_output_tokens > 0
            else route.max_output_tokens if route.max_output_tokens > 0
            else self.budget_policy.compute_output_budget(route_state, task_priority)
        )

        dynamic_safety_margin = (
            safety_margin if safety_margin > 0
            else self.budget_policy.compute_safety_reserve(route_state)
        )

        # ---- Handle non-immediate route decisions ----
        if route.action not in {RouteAction.RUN_ON_BIG_MODEL, RouteAction.RUN_ON_SMALL_MODEL}:
            return None, {
                "ok": False,
                "reason": "route_decision",
                "route_action": route.action.value,
                "route_reason": route.reason,
                "diagnostic": route.diagnostic,
                "budget_notice": route.budget_notice,
                "free_context": route.free_context,
                "free_context_percent": route.free_context_percent,
            }

        # ---- Build lease with dynamic values ----
        lease = ContextLease.create(
            graph_run_id=graph_run_id,
            request_id=request_id or str(uuid.uuid4()),
            agent_name=agent_name,
            instance_id=instance.id,
            llama_base_url=instance.base_url,
            priority=priority,
            prompt_tokens=prompt_tokens,
            max_output_tokens=dynamic_max_output,
            tool_context_tokens=tool_context_tokens,
            safety_margin=dynamic_safety_margin,
            metadata={
                "ctx_total": instance.ctx_total,
                "parallel": instance.parallel,
                "kv_unified": instance.kv_unified,
                "conservative_ctx_per_slot": instance.runtime_config.conservative_ctx_per_slot,
                "background": background,
                "speculative": speculative,
                "route_action": route.action.value,
                "route_reason": route.reason,
                "budget_notice": route.budget_notice,
                "dynamic_max_output": dynamic_max_output,
                "dynamic_safety_margin": dynamic_safety_margin,
            },
        )
        admitted = await self.admit_request(lease, instance=instance, background=background, speculative=speculative)
        if admitted["ok"]:
            return lease, admitted
        return None, admitted

    async def admit_request(
        self,
        lease: ContextLease,
        *,
        instance: UbuntuLlamaInstance | None = None,
        background: bool | None = None,
        speculative: bool | None = None,
    ) -> dict[str, Any]:
        instance = instance or self.instances.get(lease.instance_id)
        if instance is None:
            return {"ok": False, "reason": "instance_not_found", "lease": lease.to_dict()}
        if background is None:
            background = bool(lease.metadata.get("background"))
        if speculative is None:
            speculative = bool(lease.metadata.get("speculative"))
        capacity = self._instance_capacity(instance, background=background, speculative=speculative)
        ok, active_tokens = await self.lease_store.try_add(lease, capacity_tokens=capacity)
        if not ok:
            return {
                "ok": False,
                "reason": "insufficient_context",
                "instance_id": instance.id,
                "active_tokens": active_tokens,
                "required_tokens": lease.required_tokens,
                "capacity_tokens": capacity,
                "background": background,
                "speculative": speculative,
                "decision": await self.wait_or_compress_or_route(lease, instance=instance, active_tokens=active_tokens),
            }
        return {
            "ok": True,
            "lease_id": lease.lease_id,
            "instance_id": instance.id,
            "prompt_tokens": lease.prompt_tokens,
            "required_tokens": lease.required_tokens,
            "capacity_tokens": capacity,
            "kv_unified": instance.kv_unified,
            "background": background,
            "speculative": speculative,
            "budget_notice": lease.metadata.get("budget_notice", ""),
        }

    async def release_lease(self, lease_id: str, *, status: str = "released") -> ContextLease | None:
        return await self.lease_store.release(lease_id, status=status)

    async def handle_truncated_response(self, lease: ContextLease | None, response: Any) -> None:
        if not _response_truncated(response):
            return
        LOGGER.warning(
            "llama context scheduler saw truncated=true",
            extra={"lease": lease.to_dict() if lease else None},
        )
        if lease is not None:
            await self.release_lease(lease.lease_id, status="truncated")

    async def wait_or_compress_or_route(
        self,
        lease: ContextLease,
        *,
        instance: UbuntuLlamaInstance,
        active_tokens: int,
    ) -> dict[str, Any]:
        free_tokens = max(0, self._instance_capacity(instance) - active_tokens)
        return {
            "action": "wait_or_compress_or_route",
            "free_tokens": free_tokens,
            "can_retry_with_lower_max_output_tokens": lease.max_output_tokens > 256,
            "can_reduce_rag_chunks": lease.tool_context_tokens > 0,
            "can_route_other_instance": len(self.instances) > 1,
            "can_request_manager_context_resize": self.manager_client is not None,
        }

    async def context_pressure(self, instance_id: str) -> dict[str, Any]:
        instance = self.instances.get(instance_id)
        if instance is None:
            return {"ok": False, "reason": "instance_not_found", "instance_id": instance_id}
        active_tokens = await self.lease_store.active_required_tokens(instance.id)
        normal_capacity = self._instance_capacity(instance)
        background_capacity = self._instance_capacity(instance, background=True)
        return {
            "ok": True,
            "instance_id": instance.id,
            "active_tokens": active_tokens,
            "capacity_tokens": normal_capacity,
            "background_capacity_tokens": background_capacity,
            "background_allowed": active_tokens < background_capacity,
        }

    def _instance_capacity(
        self,
        instance: UbuntuLlamaInstance,
        *,
        background: bool = False,
        speculative: bool = False,
    ) -> int:
        factor = self.safety_factor
        if background or speculative:
            factor = min(self.safety_factor, max(0.05, self.background_safety_factor))
        if instance.kv_unified:
            return capacity_limit(instance.ctx_total, factor)
        return capacity_limit(instance.runtime_config.conservative_ctx_per_slot, factor)

    async def _count_prompt_tokens(
        self,
        runtime: LlamaCppRuntimeClient,
        *,
        messages: list[Any] | None,
        text: str | None,
    ) -> int:
        if messages is not None:
            return await runtime.count_tokens_chat(messages)
        return await runtime.count_tokens_text(text or "")


def _response_truncated(response: Any) -> bool:
    if isinstance(response, dict):
        return bool(response.get("truncated"))
    additional = getattr(response, "additional_kwargs", None)
    if isinstance(additional, dict) and additional.get("truncated"):
        return True
    metadata = getattr(response, "response_metadata", None)
    return isinstance(metadata, dict) and bool(metadata.get("truncated"))


_SCHEDULER: ContextScheduler | None = None
_SCHEDULER_LOCK = asyncio.Lock()


async def get_context_scheduler() -> ContextScheduler | None:
    global _SCHEDULER
    if _SCHEDULER is not None:
        return _SCHEDULER
    async with _SCHEDULER_LOCK:
        if _SCHEDULER is None:
            _SCHEDULER = ContextScheduler.from_env()
        return _SCHEDULER
