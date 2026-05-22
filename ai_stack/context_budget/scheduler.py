from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

from ai_stack.context_budget.leases import GLOBAL_LEASE_STORE, ContextLease, LeaseStore
from ai_stack.context_budget.policies import RuntimeConfig, capacity_limit, parse_runtime_config_from_command
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
    ) -> None:
        self.manager_client = manager_client
        self.lease_store = lease_store or GLOBAL_LEASE_STORE
        self.safety_factor = safety_factor
        self.instances: dict[str, UbuntuLlamaInstance] = {}

    @classmethod
    def from_env(cls) -> "ContextScheduler | None":
        manager = UbuntuLlamaManagerClient.from_env()
        if manager is None:
            return None
        return cls(
            manager_client=manager,
            safety_factor=float(os.getenv("ALPHARAVIS_CONTEXT_SAFETY_FACTOR", "0.92")),
        )

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
    ) -> UbuntuLlamaInstance | None:
        if not self.instances:
            await self.refresh_instances_from_manager()
        candidates = list(self.instances.values())
        if preferred_instance_id:
            candidates.sort(key=lambda item: 0 if item.id == preferred_instance_id else 1)
        if priority == "high":
            candidates.sort(key=lambda item: 0 if item.kv_unified else 1)
        for instance in candidates:
            limit = self._instance_capacity(instance)
            active = await self.lease_store.active_required_tokens(instance.id)
            if active + max(0, required_tokens) <= limit:
                return instance
        return candidates[0] if candidates else None

    async def estimate_and_reserve(
        self,
        *,
        messages: list[Any] | None = None,
        text: str | None = None,
        max_output_tokens: int = 1024,
        tool_context_tokens: int = 0,
        safety_margin: int = 1024,
        graph_run_id: str = "",
        request_id: str = "",
        agent_name: str = "agent",
        priority: str = "medium",
        preferred_instance_id: str = "",
    ) -> tuple[ContextLease | None, dict[str, Any]]:
        instance = await self.choose_instance(preferred_instance_id=preferred_instance_id, priority=priority)
        if instance is None:
            return None, {"ok": False, "reason": "no_llama_instances"}

        runtime = LlamaCppRuntimeClient.from_instance(
            instance,
            timeout_seconds=float(os.getenv("ALPHARAVIS_LLAMA_RUNTIME_TIMEOUT_SECONDS", "30")),
        )
        prompt_tokens = await self._count_prompt_tokens(runtime, messages=messages, text=text)
        lease = ContextLease.create(
            graph_run_id=graph_run_id,
            request_id=request_id or str(uuid.uuid4()),
            agent_name=agent_name,
            instance_id=instance.id,
            llama_base_url=instance.base_url,
            priority=priority,
            prompt_tokens=prompt_tokens,
            max_output_tokens=max_output_tokens,
            tool_context_tokens=tool_context_tokens,
            safety_margin=safety_margin,
            metadata={
                "ctx_total": instance.ctx_total,
                "parallel": instance.parallel,
                "kv_unified": instance.kv_unified,
                "conservative_ctx_per_slot": instance.runtime_config.conservative_ctx_per_slot,
            },
        )
        admitted = await self.admit_request(lease, instance=instance)
        if admitted["ok"]:
            return lease, admitted
        return None, admitted

    async def admit_request(self, lease: ContextLease, *, instance: UbuntuLlamaInstance | None = None) -> dict[str, Any]:
        instance = instance or self.instances.get(lease.instance_id)
        if instance is None:
            return {"ok": False, "reason": "instance_not_found", "lease": lease.to_dict()}
        capacity = self._instance_capacity(instance)
        ok, active_tokens = await self.lease_store.try_add(lease, capacity_tokens=capacity)
        if not ok:
            return {
                "ok": False,
                "reason": "insufficient_context",
                "instance_id": instance.id,
                "active_tokens": active_tokens,
                "required_tokens": lease.required_tokens,
                "capacity_tokens": capacity,
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

    def _instance_capacity(self, instance: UbuntuLlamaInstance) -> int:
        if instance.kv_unified:
            return capacity_limit(instance.ctx_total, self.safety_factor)
        return capacity_limit(instance.runtime_config.conservative_ctx_per_slot, self.safety_factor)

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

