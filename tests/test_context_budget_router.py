"""Tests for the percentage-based dynamic context budget router.

Covers all required scenarios:
- Different context pool sizes (128k, 200k, 300k)
- Idle vs busy vs critical-busy server
- Priority-based routing decisions
- Small model fallback
- Queue/defer behavior
- Server restart with different context size
- /slots unavailable fallback
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_stack.context_budget.leases import LeaseStore
from ai_stack.context_budget.policies import parse_runtime_config_from_command
from ai_stack.context_budget.router import (
    DynamicServerState,
    PercentageBudgetPolicy,
    PriorityAwareRouter,
    RouteAction,
    RouteDecision,
    ServerStateProber,
    TaskPriority,
    generate_budget_notice,
)
from ai_stack.context_budget.scheduler import ContextScheduler
from ai_stack.llama_runtime.client import LlamaCppRuntimeClient
from ai_stack.ubuntu_llama_manager.schemas import UbuntuLlamaInstance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_instance(
    instance_id: str = "primary",
    ctx_total: int = 200000,
    parallel: int = 2,
    kv_unified: bool = True,
    host: str = "llama-box",
    port: int = 8033,
    base_url: str = "",
) -> UbuntuLlamaInstance:
    return UbuntuLlamaInstance.from_api(
        {
            "id": instance_id,
            "host": host,
            "port": port,
            "base_url": base_url or f"http://{host}:{port}",
            "command": f"./llama-server -c {ctx_total} --parallel {parallel}"
            + (" --kv-unified" if kv_unified else ""),
        }
    )


def _make_runtime(base_url: str = "http://llama-box:8033") -> LlamaCppRuntimeClient:
    return LlamaCppRuntimeClient(base_url, timeout_seconds=5)


# ---------------------------------------------------------------------------
# PercentageBudgetPolicy
# ---------------------------------------------------------------------------

class TestPercentageBudgetPolicy:
    def test_compute_safety_reserve_is_percentage_of_pool(self):
        policy = PercentageBudgetPolicy(safety_reserve_pct=0.10)
        state = DynamicServerState(
            instance_id="t", context_pool_size=200000, kv_unified=True
        )
        assert policy.compute_safety_reserve(state) == 20000

        state.context_pool_size = 128000
        assert policy.compute_safety_reserve(state) == 12800

    def test_safety_reserve_higher_with_multi_slot(self):
        policy = PercentageBudgetPolicy(
            safety_reserve_pct=0.08, safety_reserve_multi_slot_pct=0.15
        )
        idle = DynamicServerState(
            instance_id="t", context_pool_size=200000, active_slots=0
        )
        busy = DynamicServerState(
            instance_id="t", context_pool_size=200000, active_slots=4
        )
        assert policy.compute_safety_reserve(idle) == 16000
        assert policy.compute_safety_reserve(busy) == 30000

    def test_output_budget_scales_with_free_context_and_priority(self):
        policy = PercentageBudgetPolicy(summarization_output_pct=0.15)
        state = DynamicServerState(
            instance_id="t", context_pool_size=200000, kv_unified=True
        )
        # Critical: uncapped — gets full usable context (free - reserve)
        budget_critical = policy.compute_output_budget(
            state, TaskPriority.CRITICAL_MAIN_AGENT, free_context=100000
        )
        # Summarization: capped at 15% of usable
        budget_summary = policy.compute_output_budget(
            state, TaskPriority.SUMMARIZATION, free_context=100000
        )
        assert budget_critical > budget_summary
        # Critical should get close to full usable (free - reserve)
        reserve = policy.compute_safety_reserve(state)
        usable = 100000 - reserve
        assert budget_critical == usable  # full usable context, no cap

    def test_output_budget_respects_absolute_ceiling(self):
        policy = PercentageBudgetPolicy(
            max_output_tokens=50000
        )
        state = DynamicServerState(
            instance_id="t", context_pool_size=200000, kv_unified=True
        )
        # Critical is uncapped but respects absolute ceiling
        budget = policy.compute_output_budget(
            state, TaskPriority.CRITICAL_MAIN_AGENT, free_context=200000
        )
        assert budget <= 50000

    def test_output_budget_respects_minimum(self):
        policy = PercentageBudgetPolicy(
            normal_chat_output_pct=0.001, min_output_tokens=64
        )
        state = DynamicServerState(
            instance_id="t", context_pool_size=200000, kv_unified=True
        )
        budget = policy.compute_output_budget(
            state, TaskPriority.NORMAL_CHAT, free_context=1000
        )
        assert budget >= 64

    def test_from_env_reads_percentage_values(self, monkeypatch):
        monkeypatch.setenv("ALPHARAVIS_BUDGET_SAFETY_RESERVE_PCT", "0.12")
        monkeypatch.setenv("ALPHARAVIS_BUDGET_SUMMARIZATION_OUTPUT_PCT", "0.35")
        policy = PercentageBudgetPolicy.from_env()
        assert policy.safety_reserve_pct == 0.12
        assert policy.summarization_output_pct == 0.35

    def test_primary_agents_are_uncapped_default(self):
        policy = PercentageBudgetPolicy()
        assert TaskPriority.CRITICAL_MAIN_AGENT in policy.uncapped_priorities
        assert TaskPriority.CODING_AGENT in policy.uncapped_priorities

    def test_secondary_agents_are_capped_default(self):
        policy = PercentageBudgetPolicy()
        assert TaskPriority.SUMMARIZATION not in policy.uncapped_priorities
        assert TaskPriority.BACKGROUND_TASK not in policy.uncapped_priorities
        assert TaskPriority.LOW_PRIORITY not in policy.uncapped_priorities
        assert TaskPriority.NORMAL_CHAT not in policy.uncapped_priorities
        assert TaskPriority.LONG_ANALYSIS not in policy.uncapped_priorities

    def test_uncapped_priorities_from_env(self, monkeypatch):
        monkeypatch.setenv("ALPHARAVIS_BUDGET_UNCAPPED_PRIORITIES", "critical_main_agent, summarization")
        policy = PercentageBudgetPolicy.from_env()
        assert TaskPriority.CRITICAL_MAIN_AGENT in policy.uncapped_priorities
        assert TaskPriority.SUMMARIZATION in policy.uncapped_priorities
        assert TaskPriority.CODING_AGENT not in policy.uncapped_priorities


# ---------------------------------------------------------------------------
# DynamicServerState
# ---------------------------------------------------------------------------

class TestDynamicServerState:
    def test_free_context_with_kv_unified(self):
        state = DynamicServerState(
            instance_id="t",
            context_pool_size=200000,
            kv_unified=True,
            estimated_kv_used=150000,
        )
        assert state.free_context == 50000

    def test_free_context_without_kv_unified_uses_idle_slots(self):
        state = DynamicServerState(
            instance_id="t",
            context_pool_size=200000,  # 100k per slot with 2 slots
            parallel_slots=2,
            kv_unified=False,
            idle_slots=1,
            active_slots=1,
        )
        assert state.free_context == 100000  # 1 idle slot = 100k

    def test_free_context_percent(self):
        state = DynamicServerState(
            instance_id="t",
            context_pool_size=200000,
            kv_unified=True,
            estimated_kv_used=50000,
        )
        assert state.free_context_percent == 0.75

    def test_is_idle(self):
        idle = DynamicServerState(instance_id="t", context_pool_size=200000)
        assert idle.is_idle is True

        busy = DynamicServerState(
            instance_id="t", context_pool_size=200000, active_slots=1
        )
        assert busy.is_idle is False

    def test_is_busy(self):
        state = DynamicServerState(
            instance_id="t",
            context_pool_size=200000,
            kv_unified=True,
            estimated_kv_used=180000,
        )
        assert state.is_busy is True
        assert state.is_critical_busy is False

    def test_is_critical_busy(self):
        state = DynamicServerState(
            instance_id="t",
            context_pool_size=200000,
            kv_unified=True,
            estimated_kv_used=195000,
        )
        assert state.is_critical_busy is True


# ---------------------------------------------------------------------------
# ServerStateProber - /slots parsing
# ---------------------------------------------------------------------------

class TestServerStateProber:
    def test_parse_slots_active_and_idle(self):
        state = DynamicServerState(instance_id="t", context_pool_size=0)
        slots_data = [
            {"id": 0, "state": 1, "n_past": 50000, "n_ctx": 200000},
            {"id": 1, "state": 0, "n_ctx": 200000},
        ]
        result = ServerStateProber._parse_slots(state, slots_data)
        assert result.active_slots == 1
        assert result.idle_slots == 1
        assert result.estimated_kv_used == 50000
        assert result.context_pool_size == 200000

    def test_parse_slots_all_idle(self):
        state = DynamicServerState(instance_id="t", context_pool_size=128000)
        slots_data = [
            {"id": 0, "state": 0, "n_ctx": 128000},
        ]
        result = ServerStateProber._parse_slots(state, slots_data)
        assert result.active_slots == 0
        assert result.estimated_kv_used == 0
        assert result.context_pool_size == 128000

    def test_parse_slots_extracts_context_from_first_slot(self):
        state = DynamicServerState(instance_id="t", context_pool_size=0)
        slots_data = [
            {"id": 0, "state": 0, "n_ctx": 300000},
        ]
        result = ServerStateProber._parse_slots(state, slots_data)
        assert result.context_pool_size == 300000

    def test_parse_props_extracts_context_and_parallel(self):
        state = DynamicServerState(instance_id="t")
        props = {"total_slots": 131072, "n_parallel": 4}
        result = ServerStateProber._parse_props(state, props)
        assert result.context_pool_size == 131072
        assert result.parallel_slots == 4


# ---------------------------------------------------------------------------
# generate_budget_notice
# ---------------------------------------------------------------------------

class TestBudgetNotice:
    def test_tight_budget_notice(self):
        notice = generate_budget_notice(400, 10000)
        assert "shortest complete answer" in notice

    def test_moderate_budget_notice(self):
        notice = generate_budget_notice(5000, 10000)
        assert "limited output budget" in notice

    def test_ample_budget_notice(self):
        notice = generate_budget_notice(20000, 100000)
        assert "enough budget" in notice

    def test_zero_free_context(self):
        notice = generate_budget_notice(100, 0)
        assert "no free context" in notice.lower() or "no free context" in notice


# ---------------------------------------------------------------------------
# TaskPriority
# ---------------------------------------------------------------------------

class TestTaskPriority:
    def test_parses_string_variants(self):
        assert TaskPriority("critical") == TaskPriority.CRITICAL_MAIN_AGENT
        assert TaskPriority("high") == TaskPriority.CRITICAL_MAIN_AGENT
        assert TaskPriority("coding") == TaskPriority.CODING_AGENT
        assert TaskPriority("chat") == TaskPriority.NORMAL_CHAT
        assert TaskPriority("background") == TaskPriority.BACKGROUND_TASK
        assert TaskPriority("low") == TaskPriority.LOW_PRIORITY

    def test_unknown_falls_back_to_normal(self):
        assert TaskPriority("something_weird") == TaskPriority.NORMAL_CHAT


# ---------------------------------------------------------------------------
# Router integration tests
# ---------------------------------------------------------------------------

class TestRouterIdleServer:
    """Context pool=200k, idle server."""

    def test_critical_task_runs_on_big_when_idle(self):
        router = PriorityAwareRouter()
        instance = _make_instance(ctx_total=200000)
        runtime = _make_runtime()

        # Mock /slots to return idle
        with patch.object(
            LlamaCppRuntimeClient, "_request", new_callable=AsyncMock
        ) as mock_req:
            mock_req.return_value = [
                {"id": 0, "state": 0, "n_ctx": 200000},
                {"id": 1, "state": 0, "n_ctx": 200000},
            ]

            decision = asyncio.run(
                router.route(
                    instance=instance,
                    runtime=runtime,
                    priority=TaskPriority.CRITICAL_MAIN_AGENT,
                    prompt_tokens=5000,
                )
            )

        assert decision.action == RouteAction.RUN_ON_BIG_MODEL
        assert decision.target_instance_id == "primary"

    def test_coding_task_runs_on_big_when_idle(self):
        router = PriorityAwareRouter()
        instance = _make_instance(ctx_total=200000)
        runtime = _make_runtime()

        with patch.object(
            LlamaCppRuntimeClient, "_request", new_callable=AsyncMock
        ) as mock_req:
            mock_req.return_value = [
                {"id": 0, "state": 0, "n_ctx": 200000},
            ]

            decision = asyncio.run(
                router.route(
                    instance=instance,
                    runtime=runtime,
                    priority=TaskPriority.CODING_AGENT,
                    prompt_tokens=15000,
                )
            )

        assert decision.action == RouteAction.RUN_ON_BIG_MODEL


class TestRouter128kContext:
    """Context pool=128k (smaller than default 200k assumption)."""

    def test_context_detected_as_128k_not_200k(self):
        instance = _make_instance(ctx_total=128000)
        runtime = _make_runtime()

        with patch.object(
            LlamaCppRuntimeClient, "_request", new_callable=AsyncMock
        ) as mock_req:
            mock_req.return_value = [
                {"id": 0, "state": 0, "n_ctx": 128000},
            ]

            router = PriorityAwareRouter()
            state = asyncio.run(
                router.refresh_state(runtime, instance=instance)
            )

        assert state.context_pool_size == 128000
        assert state.source == "slots"


class TestRouter300kContext:
    """Context pool=300k (larger than default)."""

    def test_context_detected_as_300k(self):
        instance = _make_instance(ctx_total=300000)
        runtime = _make_runtime()

        with patch.object(
            LlamaCppRuntimeClient, "_request", new_callable=AsyncMock
        ) as mock_req:
            mock_req.return_value = [
                {"id": 0, "state": 0, "n_ctx": 300000},
            ]

            router = PriorityAwareRouter()
            state = asyncio.run(
                router.refresh_state(runtime, instance=instance)
            )

        assert state.context_pool_size == 300000


class TestRouterDuringCriticalJob:
    """One critical main-agent request is using most of the pool."""

    def test_normal_chat_gets_queued_during_critical_job(self):
        router = PriorityAwareRouter()
        instance = _make_instance(ctx_total=200000)
        runtime = _make_runtime()

        # Simulate a critical job using 90% of context
        with patch.object(
            LlamaCppRuntimeClient, "_request", new_callable=AsyncMock
        ) as mock_req:
            mock_req.return_value = [
                {"id": 0, "state": 1, "n_past": 180000, "n_ctx": 200000},
                {"id": 1, "state": 0, "n_ctx": 200000},
            ]

            decision = asyncio.run(
                router.route(
                    instance=instance,
                    runtime=runtime,
                    priority=TaskPriority.NORMAL_CHAT,
                    prompt_tokens=2000,
                    can_use_small=True,
                )
            )

        assert decision.action in {
            RouteAction.RUN_ON_SMALL_MODEL,
            RouteAction.QUEUE_DEFER,
        }


class TestRouterSummarizationRoutesToSmall:
    """Summarization tasks prefer small model."""

    def test_summarization_routes_to_small(self):
        router = PriorityAwareRouter()
        instance = _make_instance(ctx_total=200000)
        runtime = _make_runtime()

        with patch.object(
            LlamaCppRuntimeClient, "_request", new_callable=AsyncMock
        ) as mock_req:
            mock_req.return_value = [
                {"id": 0, "state": 0, "n_ctx": 200000},
            ]

            decision = asyncio.run(
                router.route(
                    instance=instance,
                    runtime=runtime,
                    priority=TaskPriority.SUMMARIZATION,
                    prompt_tokens=2000,
                    can_use_small=True,
                )
            )

        assert decision.action == RouteAction.RUN_ON_SMALL_MODEL
        assert decision.can_use_small is True


class TestRouterCodingWithLowFreeContext:
    """Coding task arrives with low free context."""

    def test_coding_task_queued_when_context_low(self):
        router = PriorityAwareRouter()
        instance = _make_instance(ctx_total=200000)
        runtime = _make_runtime()

        with patch.object(
            LlamaCppRuntimeClient, "_request", new_callable=AsyncMock
        ) as mock_req:
            # Only 5% free
            mock_req.return_value = [
                {"id": 0, "state": 1, "n_past": 190000, "n_ctx": 200000},
            ]

            decision = asyncio.run(
                router.route(
                    instance=instance,
                    runtime=runtime,
                    priority=TaskPriority.CODING_AGENT,
                    prompt_tokens=30000,
                    can_use_small=False,
                    can_defer=True,
                )
            )

        assert decision.action in {
            RouteAction.QUEUE_DEFER,
            RouteAction.REJECT_NO_CAPACITY,
        }


class TestRouterSlotsUnavailable:
    """Fallback when /slots is not available."""

    def test_falls_back_to_manager_config_when_slots_unavailable(self):
        router = PriorityAwareRouter()
        instance = _make_instance(ctx_total=128000)
        runtime = _make_runtime()

        with patch.object(
            LlamaCppRuntimeClient, "_request", new_callable=AsyncMock
        ) as mock_req:
            # /slots fails, /props fails too
            mock_req.side_effect = RuntimeError("connection refused")

            state = asyncio.run(
                router.refresh_state(runtime, instance=instance)
            )

        assert state.context_pool_size == 128000
        assert state.source in {"manager_config", "manager_config_fallback"}


class TestRouterServerRestart:
    """Server restarts with different context size."""

    def test_policy_recalculates_after_context_change(self):
        policy = PercentageBudgetPolicy(safety_reserve_pct=0.08)

        # First run: 128k context
        state_128k = DynamicServerState(
            instance_id="t", context_pool_size=128000
        )
        reserve_128k = policy.compute_safety_reserve(state_128k)
        assert reserve_128k == 10240

        # After restart: 200k context
        state_200k = DynamicServerState(
            instance_id="t", context_pool_size=200000
        )
        reserve_200k = policy.compute_safety_reserve(state_200k)
        assert reserve_200k == 16000

        # Different!
        assert reserve_200k != reserve_128k


class TestRouteDecision:
    def test_decision_has_diagnostic_data(self):
        decision = RouteDecision(
            action=RouteAction.RUN_ON_BIG_MODEL,
            target_instance_id="primary",
            max_output_tokens=80000,
            reason="admitted",
            free_context=100000,
            free_context_percent=0.5,
            context_pool_size=200000,
            priority="critical_main_agent",
            diagnostic={"context_pool_size": 200000},
        )
        assert decision.ok is True
        assert "context_pool_size" in decision.diagnostic


class TestRouterObservability:
    def test_log_decision_does_not_crash(self):
        router = PriorityAwareRouter()
        decision = RouteDecision(
            action=RouteAction.RUN_ON_BIG_MODEL,
            target_instance_id="primary",
            max_output_tokens=50000,
            reason="test",
            free_context=100000,
            free_context_percent=0.5,
            context_pool_size=200000,
            priority="normal_chat",
        )
        # Should not raise
        router.log_decision(decision)


# ---------------------------------------------------------------------------
# Scheduler with router integration
# ---------------------------------------------------------------------------

class TestSchedulerWithRouter:
    def test_scheduler_uses_dynamic_max_tokens_when_not_provided(self, monkeypatch):
        router = PriorityAwareRouter()
        policy = PercentageBudgetPolicy.from_env()
        scheduler = ContextScheduler(
            lease_store=LeaseStore(), router=router, budget_policy=policy
        )
        instance = _make_instance(ctx_total=200000)
        scheduler.instances = {"primary": instance}

        async def fake_count(self, messages):
            return 5000

        monkeypatch.setattr(LlamaCppRuntimeClient, "count_tokens_chat", fake_count)

        with patch.object(
            LlamaCppRuntimeClient, "_request", new_callable=AsyncMock
        ) as mock_req:
            mock_req.return_value = [
                {"id": 0, "state": 0, "n_ctx": 200000},
            ]

            lease, admission = asyncio.run(
                scheduler.estimate_and_reserve(
                    messages=[{"role": "user", "content": "hello"}],
                    max_output_tokens=0,  # let router decide
                    safety_margin=0,       # let router decide
                    preferred_instance_id="primary",
                    priority="normal_chat",
                )
            )

        assert lease is not None
        # Dynamic budget should have been computed (not 0)
        assert lease.max_output_tokens > 0
        # Budget notice should be present
        assert "budget_notice" in lease.metadata

    def test_scheduler_respects_explicit_max_tokens(self, monkeypatch):
        router = PriorityAwareRouter()
        policy = PercentageBudgetPolicy.from_env()
        scheduler = ContextScheduler(
            lease_store=LeaseStore(), router=router, budget_policy=policy
        )
        instance = _make_instance(ctx_total=200000)
        scheduler.instances = {"primary": instance}

        async def fake_count(self, messages):
            return 5000

        monkeypatch.setattr(LlamaCppRuntimeClient, "count_tokens_chat", fake_count)

        with patch.object(
            LlamaCppRuntimeClient, "_request", new_callable=AsyncMock
        ) as mock_req:
            mock_req.return_value = [
                {"id": 0, "state": 0, "n_ctx": 200000},
            ]

            lease, admission = asyncio.run(
                scheduler.estimate_and_reserve(
                    messages=[{"role": "user", "content": "hello"}],
                    max_output_tokens=4096,  # explicit
                    safety_margin=1024,       # explicit
                    preferred_instance_id="primary",
                    priority="normal_chat",
                )
            )

        assert lease is not None
        assert lease.max_output_tokens == 4096
        assert lease.safety_margin == 1024
