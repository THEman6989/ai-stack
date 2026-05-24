from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os

from ai_stack.context_budget.leases import (
    ContextLease,
    LeaseStore,
    LocalLeaseStore,
    RedisLeaseStore,
    _make_lease_store_from_env,
)
from ai_stack.context_budget.policies import ensure_kv_unified_in_command, parse_runtime_config_from_command
from ai_stack.context_budget.scheduler import ContextScheduler
from ai_stack.llama_runtime.client import LlamaCppRuntimeClient
from ai_stack.ubuntu_llama_manager.client import UbuntuLlamaManagerClient
from ai_stack.ubuntu_llama_manager.schemas import UbuntuLlamaInstance


def test_parse_runtime_config_from_llama_command():
    config = parse_runtime_config_from_command(
        "./llama-server -hf model --ctx-size 262144 --parallel 2 --kv-unified"
    )

    assert config.ctx_total == 262144
    assert config.parallel == 2
    assert config.kv_unified is True
    assert config.conservative_ctx_per_slot == 131072


def test_ensure_kv_unified_in_command_is_idempotent():
    command = "./llama-server -c 8192 --no-kv-unified"

    patched = ensure_kv_unified_in_command(command)

    assert "--kv-unified" in patched
    assert "--no-kv-unified" not in patched
    assert ensure_kv_unified_in_command(patched) == patched


def test_instance_derives_runtime_base_url_from_manager_host_for_localhost():
    instance = UbuntuLlamaInstance.from_api(
        {
            "id": "primary",
            "host": "127.0.0.1",
            "port": 8033,
            "command": "./llama-server -c 4096 -np 2",
        },
        manager_base_url="http://llama-box:8099",
    )

    assert instance.base_url == "http://llama-box:8033"
    assert instance.ctx_total == 4096
    assert instance.parallel == 2
    assert instance.kv_unified is False


def test_manager_client_maps_stop_restart_to_documented_endpoints():
    assert UbuntuLlamaManagerClient._control_path("primary", "restart") == "/llama/restart"
    assert UbuntuLlamaManagerClient._control_path("secondary", "stop") == "/llama-secondary/stop"


def test_runtime_chat_count_uses_apply_template_then_tokenize(monkeypatch):
    calls = []

    async def fake_request(self, method, path, *, json_payload=None):
        calls.append((method, path, json_payload))
        if path == "/apply-template":
            return {"prompt": "<s>hello</s>"}
        if path == "/tokenize":
            return {"tokens": [1, 2, 3]}
        raise AssertionError(path)

    monkeypatch.setattr(LlamaCppRuntimeClient, "_request", fake_request)

    count = asyncio.run(LlamaCppRuntimeClient("http://llama:8033").count_tokens_chat([{"role": "user", "content": "hello"}]))

    assert count == 3
    assert calls[0][1] == "/apply-template"
    assert calls[1][1] == "/tokenize"
    assert calls[1][2] == {"content": "<s>hello</s>"}


def test_scheduler_rejects_when_active_leases_exceed_capacity(monkeypatch):
    scheduler = ContextScheduler(lease_store=LocalLeaseStore(), safety_factor=0.9)
    instance = UbuntuLlamaInstance.from_api(
        {
            "id": "primary",
            "host": "llama-box",
            "port": 8033,
            "command": "./llama-server -c 1000 --parallel 1 --kv-unified",
        }
    )
    scheduler.instances = {"primary": instance}

    async def fake_count(self, messages):
        return 400

    monkeypatch.setattr(LlamaCppRuntimeClient, "count_tokens_chat", fake_count)

    first, first_admission = asyncio.run(
        scheduler.estimate_and_reserve(
            messages=[{"role": "user", "content": "one"}],
            max_output_tokens=300,
            safety_margin=50,
            preferred_instance_id="primary",
        )
    )
    second, second_admission = asyncio.run(
        scheduler.estimate_and_reserve(
            messages=[{"role": "user", "content": "two"}],
            max_output_tokens=300,
            safety_margin=50,
            preferred_instance_id="primary",
        )
    )

    assert first is not None
    assert first_admission["ok"] is True
    assert second is None
    assert second_admission["reason"] == "insufficient_context"


def test_scheduler_without_kv_unified_uses_conservative_slot_capacity(monkeypatch):
    scheduler = ContextScheduler(lease_store=LocalLeaseStore(), safety_factor=1.0)
    scheduler.instances = {
        "primary": UbuntuLlamaInstance.from_api(
            {
                "id": "primary",
                "host": "llama-box",
                "port": 8033,
                "command": "./llama-server -c 1000 --parallel 2",
            }
        )
    }

    async def fake_count(self, messages):
        return 400

    monkeypatch.setattr(LlamaCppRuntimeClient, "count_tokens_chat", fake_count)

    lease, admission = asyncio.run(
        scheduler.estimate_and_reserve(
            messages=[{"role": "user", "content": "one"}],
            max_output_tokens=80,
            safety_margin=30,
            preferred_instance_id="primary",
        )
    )

    assert lease is None
    assert admission["reason"] == "insufficient_context"
    assert admission["capacity_tokens"] == 500


def test_scheduler_background_leases_use_smaller_context_lane(monkeypatch):
    scheduler = ContextScheduler(lease_store=LocalLeaseStore(), safety_factor=0.9, background_safety_factor=0.5)
    scheduler.instances = {
        "primary": UbuntuLlamaInstance.from_api(
            {
                "id": "primary",
                "host": "llama-box",
                "port": 8033,
                "command": "./llama-server -c 1000 --parallel 1 --kv-unified",
            }
        )
    }

    async def fake_count(self, messages):
        return 400

    monkeypatch.setattr(LlamaCppRuntimeClient, "count_tokens_chat", fake_count)

    lease, admission = asyncio.run(
        scheduler.estimate_and_reserve(
            messages=[{"role": "user", "content": "background"}],
            max_output_tokens=80,
            safety_margin=30,
            preferred_instance_id="primary",
            background=True,
            speculative=True,
            priority="low",
        )
    )

    assert lease is None
    assert admission["reason"] == "insufficient_context"
    assert admission["capacity_tokens"] == 500
    assert admission["background"] is True
    assert admission["speculative"] is True


# ---------------------------------------------------------------------------
# Lease store backend tests
# ---------------------------------------------------------------------------


def test_local_lease_store_is_lease_store_subclass():
    """LocalLeaseStore is a concrete LeaseStore."""
    store = LocalLeaseStore()
    assert isinstance(store, LeaseStore)


def test_local_lease_store_active_for_instance():
    store = LocalLeaseStore()
    assert asyncio.run(store.active_for_instance("primary")) == []


def test_local_lease_store_try_add_and_release():
    store = LocalLeaseStore()
    lease = ContextLease.create(
        graph_run_id="r1",
        request_id="req1",
        agent_name="test",
        instance_id="primary",
        llama_base_url="http://llama:8033",
        priority="medium",
        prompt_tokens=200,
        max_output_tokens=100,
        tool_context_tokens=0,
        safety_margin=50,
    )
    ok, active = asyncio.run(store.try_add(lease, capacity_tokens=500))
    assert ok is True
    assert active == 0

    active_tokens = asyncio.run(store.active_required_tokens("primary"))
    assert active_tokens == 350  # 200+100+0+50

    # Release
    released = asyncio.run(store.release(lease.lease_id, status="completed"))
    assert released is not None
    assert released.status == "completed"
    active_tokens_after = asyncio.run(store.active_required_tokens("primary"))
    assert active_tokens_after == 0


def test_make_lease_store_from_env_defaults_to_local(monkeypatch):
    """When ALPHARAVIS_CONTEXT_LEASE_BACKEND is unset, returns LocalLeaseStore."""
    monkeypatch.delenv("ALPHARAVIS_CONTEXT_LEASE_BACKEND", raising=False)
    store = _make_lease_store_from_env()
    assert isinstance(store, LocalLeaseStore)


def test_make_lease_store_from_env_local_explicit(monkeypatch):
    """When ALPHARAVIS_CONTEXT_LEASE_BACKEND=local, returns LocalLeaseStore."""
    monkeypatch.setenv("ALPHARAVIS_CONTEXT_LEASE_BACKEND", "local")
    store = _make_lease_store_from_env()
    assert isinstance(store, LocalLeaseStore)


def test_make_lease_store_from_env_redis_creates_redis_store(monkeypatch):
    """When ALPHARAVIS_CONTEXT_LEASE_BACKEND=redis, returns RedisLeaseStore.

    RedisLeaseStore uses lazy connection — the actual redis.asyncio import
    only happens in _ensure_client() on first use. This means _make_lease_store_from_env
    succeeds even without the redis package installed, and the import error
    surfaces later when a lease operation is attempted.

    This is intentional: the store is created at startup, and if redis is
    unreachable at runtime, the caller should handle the OperationalError.
    """
    monkeypatch.setenv("ALPHARAVIS_CONTEXT_LEASE_BACKEND", "redis")
    monkeypatch.setenv("ALPHARAVIS_REDIS_URL", "redis://fake:6379")
    monkeypatch.setenv("ALPHARAVIS_CONTEXT_LEASE_TTL", "60")
    store = _make_lease_store_from_env()
    assert isinstance(store, RedisLeaseStore)
    assert store.redis_url == "redis://fake:6379"
    assert store.ttl == 60


def test_context_lease_from_dict_roundtrip():
    original = ContextLease.create(
        graph_run_id="r1",
        request_id="req1",
        agent_name="agent",
        instance_id="primary",
        llama_base_url="http://llama:8033",
        priority="high",
        prompt_tokens=1000,
        max_output_tokens=500,
        tool_context_tokens=100,
        safety_margin=50,
        metadata={"key": "value"},
    )
    data = original.to_dict()
    restored = ContextLease.from_dict(data)
    assert restored.lease_id == original.lease_id
    assert restored.required_tokens == original.required_tokens
    assert restored.status == original.status
    assert restored.metadata == original.metadata


# ---------------------------------------------------------------------------
# RedisLeaseStore integration tests (require redis or fakeredis)
# ---------------------------------------------------------------------------


def _fake_redis_client():
    """Build a fake async Redis client with in-memory dict backing."""
    from unittest.mock import AsyncMock, MagicMock

    storage: dict[str, dict[str, str]] = {}
    ttl_map: dict[str, int] = {}

    async def hgetall(key):
        return storage.get(key, {}).copy()

    async def hset(key, field, value):
        storage.setdefault(key, {})[field] = value
        return 1

    async def hget(key, field):
        inner = storage.get(key, {})
        return inner.get(field)

    async def expire(key, seconds):
        ttl_map[key] = seconds
        return 1

    async def evalsha(sha, numkeys, *args):
        # Simulate the Lua script logic
        hash_key = args[0]
        instance_id = args[1]
        lease_id = args[2]
        lease_json = args[3]
        required = int(args[4])
        capacity = int(args[5])
        ttl = int(args[6])

        import json

        active = 0
        inner = storage.get(hash_key, {})
        for _field, raw in inner.items():
            try:
                data = json.loads(raw)
                if data.get("instance_id") == instance_id and data.get("status") == "active":
                    active += max(0, int(data.get("required_tokens", 0)))
            except Exception:
                pass

        if active + required > capacity:
            return [0, active]  # rejected

        storage.setdefault(hash_key, {})[lease_id] = lease_json
        if ttl > 0:
            ttl_map[hash_key] = ttl
        return [1, active]  # admitted

    async def script_load(script):
        return "fake-sha"

    fake = MagicMock()
    fake.hgetall = hgetall
    fake.hset = hset
    fake.hget = hget
    fake.expire = expire
    fake.evalsha = evalsha
    fake.script_load = script_load
    return fake


class TestRedisLeaseStoreIntegration:
    def test_atomic_try_add_rejects_when_over_capacity(self, monkeypatch):
        """Two leases each needing 600 tokens against 1000 capacity → second rejected."""
        import json

        store = RedisLeaseStore(redis_url="redis://fake:6379", ttl=60)
        fake_client = _fake_redis_client()
        monkeypatch.setattr(store, "_redis", fake_client)
        monkeypatch.setattr(store, "_lua_sha", "fake-sha")

        lease1 = ContextLease.create(
            graph_run_id="r1",
            request_id="req1",
            agent_name="w1",
            instance_id="primary",
            llama_base_url="http://llama:8033",
            priority="high",
            prompt_tokens=400,
            max_output_tokens=100,
            tool_context_tokens=50,
            safety_margin=50,
        )
        lease2 = ContextLease.create(
            graph_run_id="r2",
            request_id="req2",
            agent_name="w2",
            instance_id="primary",
            llama_base_url="http://llama:8033",
            priority="high",
            prompt_tokens=400,
            max_output_tokens=100,
            tool_context_tokens=50,
            safety_margin=50,
        )

        ok1, _active1 = asyncio.run(store.try_add(lease1, capacity_tokens=1000))
        assert ok1 is True

        ok2, active2 = asyncio.run(store.try_add(lease2, capacity_tokens=1000))
        assert ok2 is False
        assert active2 == 600  # First lease occupies 600

    def test_release_frees_tokens_for_new_lease(self, monkeypatch):
        """After releasing a lease, a new lease can be admitted."""
        store = RedisLeaseStore(redis_url="redis://fake:6379", ttl=60)
        fake_client = _fake_redis_client()
        monkeypatch.setattr(store, "_redis", fake_client)
        monkeypatch.setattr(store, "_lua_sha", "fake-sha")

        lease1 = ContextLease.create(
            graph_run_id="r1",
            request_id="req1",
            agent_name="w1",
            instance_id="primary",
            llama_base_url="http://llama:8033",
            priority="high",
            prompt_tokens=500,
            max_output_tokens=100,
            tool_context_tokens=0,
            safety_margin=50,
        )  # 650 required

        lease2 = ContextLease.create(
            graph_run_id="r2",
            request_id="req2",
            agent_name="w2",
            instance_id="primary",
            llama_base_url="http://llama:8033",
            priority="high",
            prompt_tokens=500,
            max_output_tokens=100,
            tool_context_tokens=0,
            safety_margin=50,
        )  # 650 required

        ok1, _ = asyncio.run(store.try_add(lease1, capacity_tokens=1000))
        assert ok1 is True

        ok2, _ = asyncio.run(store.try_add(lease2, capacity_tokens=1000))
        assert ok2 is False  # 650+650 > 1000

        # Release first lease
        asyncio.run(store.release(lease1.lease_id, status="completed"))

        # Now second lease should be admitted
        ok3, active3 = asyncio.run(store.try_add(lease2, capacity_tokens=1000))
        assert ok3 is True
        # Only released lease counts as inactive, so active=0 before admission
        assert active3 == 0

    def test_active_required_tokens_counts_only_active_leases(self, monkeypatch):
        """Only leases with status=active contribute to token count."""
        store = RedisLeaseStore(redis_url="redis://fake:6379", ttl=60)
        fake_client = _fake_redis_client()
        monkeypatch.setattr(store, "_redis", fake_client)
        monkeypatch.setattr(store, "_lua_sha", "fake-sha")

        lease = ContextLease.create(
            graph_run_id="r1",
            request_id="req1",
            agent_name="w1",
            instance_id="primary",
            llama_base_url="http://llama:8033",
            priority="high",
            prompt_tokens=300,
            max_output_tokens=100,
            tool_context_tokens=0,
            safety_margin=50,
        )  # 450 required

        ok, _ = asyncio.run(store.try_add(lease, capacity_tokens=1000))
        assert ok is True

        tokens_before = asyncio.run(store.active_required_tokens("primary"))
        assert tokens_before == 450

        asyncio.run(store.release(lease.lease_id, status="completed"))
        tokens_after = asyncio.run(store.active_required_tokens("primary"))
        assert tokens_after == 0
