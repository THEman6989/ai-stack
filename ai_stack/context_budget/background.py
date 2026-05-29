from __future__ import annotations

import asyncio
import contextlib
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ai_stack.context_budget.scheduler import ContextScheduler, get_context_scheduler


TaskFactory = Callable[[], Awaitable[Any]]


@dataclass(frozen=True)
class BackgroundTaskPolicy:
    name: str
    kind: str = "read_only_tool"
    priority: str = "low"
    requires_llm: bool = False
    read_only: bool = True
    speculative: bool = False
    timeout_seconds: float | None = None
    max_output_tokens: int = 512
    tool_context_tokens: int = 0
    safety_margin: int = 512
    preferred_instance_id: str = ""
    model_name: str = ""


@dataclass
class BackgroundTaskResult:
    name: str
    status: str
    value: Any = None
    error: str = ""
    lease_id: str = ""
    admission: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.status == "completed"


class BackgroundTaskRunner:
    """Runs safe latency-hiding work without bypassing LLM context leases."""

    def __init__(
        self,
        *,
        scheduler: ContextScheduler | None = None,
        max_read_only: int | None = None,
        max_small_llm: int | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.enabled = _env_bool("ALPHARAVIS_BACKGROUND_TASKS_ENABLED", "true") if enabled is None else enabled
        self._read_only_sem = asyncio.Semaphore(
            max_read_only
            if max_read_only is not None
            else max(1, int(os.getenv("ALPHARAVIS_BACKGROUND_READ_ONLY_MAX_CONCURRENCY", "4")))
        )
        self._small_llm_sem = asyncio.Semaphore(
            max_small_llm
            if max_small_llm is not None
            else max(1, int(os.getenv("ALPHARAVIS_BACKGROUND_SMALL_LLM_MAX_CONCURRENCY", "2")))
        )
        self._tasks: dict[str, asyncio.Task[BackgroundTaskResult]] = {}
        self._policies: dict[str, BackgroundTaskPolicy] = {}
        self._lock = asyncio.Lock()

    async def submit_read_only(
        self,
        name: str,
        coro_factory: TaskFactory,
        *,
        priority: str = "low",
        timeout_seconds: float | None = None,
        speculative: bool = False,
    ) -> asyncio.Task[BackgroundTaskResult]:
        policy = BackgroundTaskPolicy(
            name=name,
            kind="read_only_tool",
            priority=priority,
            read_only=True,
            speculative=speculative,
            timeout_seconds=timeout_seconds,
        )
        return await self.submit(policy, coro_factory)

    async def submit_small_llm(
        self,
        name: str,
        coro_factory: TaskFactory,
        *,
        messages: list[Any] | None = None,
        text: str | None = None,
        max_output_tokens: int = 512,
        priority: str = "low",
        timeout_seconds: float | None = None,
        graph_run_id: str = "",
        request_id: str = "",
        preferred_instance_id: str = "",
        speculative: bool = True,
    ) -> asyncio.Task[BackgroundTaskResult]:
        policy = BackgroundTaskPolicy(
            name=name,
            kind="small_llm",
            priority=priority,
            requires_llm=True,
            read_only=True,
            speculative=speculative,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
            preferred_instance_id=preferred_instance_id,
        )

        async def leased_factory() -> Any:
            scheduler = self.scheduler or await get_context_scheduler()
            if scheduler is None:
                raise RuntimeError("context scheduler unavailable for background LLM task")
            lease, admission = await scheduler.estimate_and_reserve(
                messages=messages,
                text=text,
                max_output_tokens=max_output_tokens,
                safety_margin=policy.safety_margin,
                tool_context_tokens=policy.tool_context_tokens,
                graph_run_id=graph_run_id,
                request_id=request_id,
                agent_name=name,
                priority=priority,
                preferred_instance_id=preferred_instance_id,
                background=True,
                speculative=speculative,
            )
            if lease is None:
                if speculative or priority == "low":
                    return BackgroundTaskResult(name=name, status="skipped", admission=admission)
                raise RuntimeError(f"background LLM lease denied: {admission}")
            try:
                value = await coro_factory()
                return BackgroundTaskResult(name=name, status="completed", value=value, lease_id=lease.lease_id, admission=admission)
            finally:
                await scheduler.release_lease(lease.lease_id)

        return await self.submit(policy, leased_factory, factory_returns_result=True)

    async def submit(
        self,
        policy: BackgroundTaskPolicy,
        coro_factory: TaskFactory,
        *,
        factory_returns_result: bool = False,
    ) -> asyncio.Task[BackgroundTaskResult]:
        if not self.enabled:
            return _completed_task(BackgroundTaskResult(name=policy.name, status="skipped", error="disabled"))
        if not policy.read_only:
            return _completed_task(BackgroundTaskResult(name=policy.name, status="skipped", error="non_read_only_not_allowed"))
        semaphore = self._small_llm_sem if policy.requires_llm else self._read_only_sem

        async def run_inner() -> BackgroundTaskResult:
            try:
                async with semaphore:
                    timeout = policy.timeout_seconds
                    if timeout is None:
                        timeout = float(os.getenv("ALPHARAVIS_BACKGROUND_TASK_TIMEOUT_SECONDS", "30"))
                    value = await asyncio.wait_for(coro_factory(), timeout=max(0.1, timeout))
                    if factory_returns_result and isinstance(value, BackgroundTaskResult):
                        return value
                    return BackgroundTaskResult(name=policy.name, status="completed", value=value)
            except asyncio.CancelledError:
                return BackgroundTaskResult(name=policy.name, status="cancelled")
            except TimeoutError:
                return BackgroundTaskResult(name=policy.name, status="timeout")
            except Exception as exc:
                return BackgroundTaskResult(name=policy.name, status="failed", error=f"{type(exc).__name__}: {exc}")

        async def run() -> BackgroundTaskResult:
            try:
                return await run_inner()
            except asyncio.CancelledError:
                return BackgroundTaskResult(name=policy.name, status="cancelled")

        task = asyncio.create_task(run(), name=f"alpharavis-bg:{policy.name}")
        async with self._lock:
            self._tasks[policy.name] = task
            self._policies[policy.name] = policy
        task.add_done_callback(lambda _task: self._forget(policy.name))
        await asyncio.sleep(0)
        return task

    async def cancel_low_priority(self, reason: str = "context_pressure") -> list[str]:
        cancelled: list[str] = []
        async with self._lock:
            tasks = list(self._tasks.items())
            policies = dict(self._policies)
        for name, task in tasks:
            if task.done():
                continue
            policy = policies.get(name)
            if policy and policy.priority not in {"low", "speculative"} and not policy.speculative:
                continue
            task.cancel(msg=reason)
            cancelled.append(name)
        return cancelled

    def _forget(self, name: str) -> None:
        self._tasks.pop(name, None)
        self._policies.pop(name, None)


def _completed_task(result: BackgroundTaskResult) -> asyncio.Task[BackgroundTaskResult]:
    async def done() -> BackgroundTaskResult:
        return result

    return asyncio.create_task(done())


def _env_bool(name: str, default: str = "false") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "y", "on"}


GLOBAL_BACKGROUND_TASK_RUNNER: BackgroundTaskRunner | None = None
_RUNNER_LOCK = asyncio.Lock()


async def get_background_task_runner() -> BackgroundTaskRunner:
    global GLOBAL_BACKGROUND_TASK_RUNNER
    if GLOBAL_BACKGROUND_TASK_RUNNER is not None:
        return GLOBAL_BACKGROUND_TASK_RUNNER
    async with _RUNNER_LOCK:
        if GLOBAL_BACKGROUND_TASK_RUNNER is None:
            GLOBAL_BACKGROUND_TASK_RUNNER = BackgroundTaskRunner()
        return GLOBAL_BACKGROUND_TASK_RUNNER


@contextlib.asynccontextmanager
async def background_task_scope() -> Any:
    runner = await get_background_task_runner()
    try:
        yield runner
    finally:
        if _env_bool("ALPHARAVIS_BACKGROUND_CANCEL_ON_CONTEXT_PRESSURE", "true"):
            await runner.cancel_low_priority("scope_closed")
