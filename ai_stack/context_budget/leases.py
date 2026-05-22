from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ContextLease:
    lease_id: str
    graph_run_id: str
    request_id: str
    agent_name: str
    instance_id: str
    llama_base_url: str
    priority: str
    prompt_tokens: int
    max_output_tokens: int
    tool_context_tokens: int
    safety_margin: int
    required_tokens: int
    started_at: float
    status: str = "active"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        graph_run_id: str,
        request_id: str,
        agent_name: str,
        instance_id: str,
        llama_base_url: str,
        priority: str,
        prompt_tokens: int,
        max_output_tokens: int,
        tool_context_tokens: int,
        safety_margin: int,
        metadata: dict[str, Any] | None = None,
    ) -> "ContextLease":
        required_tokens = int(prompt_tokens) + int(max_output_tokens) + int(tool_context_tokens) + int(safety_margin)
        return cls(
            lease_id=str(uuid.uuid4()),
            graph_run_id=graph_run_id,
            request_id=request_id,
            agent_name=agent_name,
            instance_id=instance_id,
            llama_base_url=llama_base_url,
            priority=priority,
            prompt_tokens=int(prompt_tokens),
            max_output_tokens=int(max_output_tokens),
            tool_context_tokens=int(tool_context_tokens),
            safety_margin=int(safety_margin),
            required_tokens=required_tokens,
            started_at=time.time(),
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LeaseStore:
    """Process-local async lease store.

    This protects concurrent calls inside one langgraph-api process. Multi-process
    lease coordination should use Redis/Postgres later.
    """

    def __init__(self) -> None:
        self._leases: dict[str, ContextLease] = {}
        self._lock = asyncio.Lock()

    async def active_for_instance(self, instance_id: str) -> list[ContextLease]:
        async with self._lock:
            return [
                lease
                for lease in self._leases.values()
                if lease.instance_id == instance_id and lease.status == "active"
            ]

    async def active_required_tokens(self, instance_id: str) -> int:
        leases = await self.active_for_instance(instance_id)
        return sum(max(0, lease.required_tokens) for lease in leases)

    async def try_add(self, lease: ContextLease, *, capacity_tokens: int) -> tuple[bool, int]:
        async with self._lock:
            active = sum(
                max(0, item.required_tokens)
                for item in self._leases.values()
                if item.instance_id == lease.instance_id and item.status == "active"
            )
            if active + lease.required_tokens > capacity_tokens:
                return False, active
            self._leases[lease.lease_id] = lease
            return True, active

    async def release(self, lease_id: str, *, status: str = "released") -> ContextLease | None:
        async with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                return None
            lease.status = status
            return lease


GLOBAL_LEASE_STORE = LeaseStore()

