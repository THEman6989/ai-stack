from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

LOGGER = logging.getLogger(__name__)


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextLease":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class LeaseStore(ABC):
    """Abstract lease store for context admission.

    Single-worker setups use ``LocalLeaseStore`` (process-local dict).
    Multi-worker deployments can switch to ``RedisLeaseStore`` for shared
    coordination across ``langgraph-api`` replicas.
    """

    @abstractmethod
    async def active_for_instance(self, instance_id: str) -> list[ContextLease]:
        ...

    @abstractmethod
    async def active_required_tokens(self, instance_id: str) -> int:
        ...

    @abstractmethod
    async def try_add(self, lease: ContextLease, *, capacity_tokens: int) -> tuple[bool, int]:
        ...

    @abstractmethod
    async def release(self, lease_id: str, *, status: str = "released") -> ContextLease | None:
        ...


class LocalLeaseStore(LeaseStore):
    """Process-local async lease store.

    Protects concurrent calls inside one langgraph-api process.
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


_REDIS_LEASE_LUA = """
-- KEYS[1] = lease hash key  (e.g. alpharavis:leases)
-- ARGV[1] = instance_id
-- ARGV[2] = lease_id
-- ARGV[3] = lease JSON
-- ARGV[4] = required_tokens
-- ARGV[5] = capacity_tokens
-- ARGV[6] = ttl seconds

local hash_key = KEYS[1]
local instance_id = ARGV[1]
local lease_id = ARGV[2]
local required = tonumber(ARGV[4])
local capacity = tonumber(ARGV[5])
local ttl = tonumber(ARGV[6])

-- Sum active required tokens for this instance
local active = 0
local all_fields = redis.call('HGETALL', hash_key)
for i = 1, #all_fields, 2 do
    local field = all_fields[i]
    local val = all_fields[i+1]
    local ok, lease_data = pcall(cjson.decode, val)
    if ok and type(lease_data) == 'table' then
        if lease_data.instance_id == instance_id and lease_data.status == 'active' then
            active = active + math.max(0, tonumber(lease_data.required_tokens) or 0)
        end
    end
end

-- Check capacity
if active + required > capacity then
    return {0, active}  -- rejected
end

-- Admit
local lease_json = ARGV[3]
redis.call('HSET', hash_key, lease_id, lease_json)
if ttl > 0 then
    redis.call('EXPIRE', hash_key, ttl)
end
return {1, active}  -- admitted
"""


class RedisLeaseStore(LeaseStore):
    """Redis-backed lease store for multi-worker coordination.

    Uses an atomic Lua script for admission to prevent race conditions
    between concurrent workers. Leases auto-expire via TTL so crashed
    workers don't leak context reservations.

    Requires ``redis`` package (``pip install redis``). Import is lazy —
    only triggered when this class is constructed.
    """

    def __init__(
        self,
        redis_url: str = "redis://redis:6379",
        *,
        prefix: str = "alpharavis:lease:",
        ttl: int = 600,
    ) -> None:
        self.redis_url = redis_url
        self.prefix = prefix
        self.ttl = max(10, int(ttl))
        self._hash_key = f"{prefix}leases"
        self._redis: Any = None
        self._lua_sha: str | None = None
        self._lock = asyncio.Lock()

    async def _ensure_client(self) -> Any:
        if self._redis is not None:
            return self._redis
        async with self._lock:
            if self._redis is not None:
                return self._redis
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(self.redis_url, decode_responses=False)
            # Pre-load Lua script
            self._lua_sha = await self._redis.script_load(_REDIS_LEASE_LUA)
            return self._redis

    async def active_for_instance(self, instance_id: str) -> list[ContextLease]:
        redis = await self._ensure_client()
        all_fields = await redis.hgetall(self._hash_key)
        leases: list[ContextLease] = []
        for field_name, raw in all_fields.items():
            try:
                data = json.loads(raw)
                lease = ContextLease.from_dict(data)
                if lease.instance_id == instance_id and lease.status == "active":
                    leases.append(lease)
            except Exception:
                LOGGER.debug("RedisLeaseStore: skipping unparseable lease field %s", field_name)
        return leases

    async def active_required_tokens(self, instance_id: str) -> int:
        redis = await self._ensure_client()
        all_fields = await redis.hgetall(self._hash_key)
        total = 0
        for _field_name, raw in all_fields.items():
            try:
                data = json.loads(raw)
                if data.get("instance_id") == instance_id and data.get("status") == "active":
                    total += max(0, int(data.get("required_tokens", 0)))
            except Exception:
                pass
        return total

    async def try_add(self, lease: ContextLease, *, capacity_tokens: int) -> tuple[bool, int]:
        redis = await self._ensure_client()
        lease_json = json.dumps(lease.to_dict(), default=str)
        result = await redis.evalsha(
            self._lua_sha,
            1,
            self._hash_key,
            lease.instance_id,
            lease.lease_id,
            lease_json,
            str(lease.required_tokens),
            str(capacity_tokens),
            str(self.ttl),
        )
        admitted, active = int(result[0]), int(result[1])
        if not admitted:
            return False, active
        return True, active

    async def release(self, lease_id: str, *, status: str = "released") -> ContextLease | None:
        redis = await self._ensure_client()
        raw = await redis.hget(self._hash_key, lease_id)
        if raw is None:
            return None
        try:
            data = json.loads(raw)
        except Exception:
            return None
        data["status"] = status
        updated = ContextLease.from_dict(data)
        await redis.hset(self._hash_key, lease_id, json.dumps(data, default=str))
        return updated


def _make_lease_store_from_env() -> LeaseStore:
    """Create the appropriate lease store from environment configuration.

    Controlled by ``ALPHARAVIS_CONTEXT_LEASE_BACKEND``:

    - ``local`` (default): process-local dict, no external dependencies.
    - ``redis``: shared Redis store for multi-worker coordination.

    Falls back to ``LocalLeaseStore`` if Redis is unreachable or the
    ``redis`` package is missing.
    """
    backend = os.getenv("ALPHARAVIS_CONTEXT_LEASE_BACKEND", "local").strip().lower()
    if backend == "redis":
        redis_url = os.getenv("ALPHARAVIS_REDIS_URL", "redis://redis:6379")
        ttl = int(os.getenv("ALPHARAVIS_CONTEXT_LEASE_TTL", "600"))
        try:
            store = RedisLeaseStore(redis_url=redis_url, ttl=ttl)
            LOGGER.info(
                "RedisLeaseStore configured (url=%s, ttl=%ds)",
                redis_url,
                ttl,
            )
            return store
        except Exception as exc:
            LOGGER.warning(
                "RedisLeaseStore init failed (%s), falling back to LocalLeaseStore",
                exc,
            )
            return LocalLeaseStore()
    return LocalLeaseStore()


# Kept for backward compatibility — modules can still import this.
GLOBAL_LEASE_STORE: LeaseStore = LocalLeaseStore()
