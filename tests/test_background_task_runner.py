from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_stack.context_budget.background import BackgroundTaskRunner


def test_read_only_background_tasks_run_concurrently():
    async def run() -> tuple[float, list[int]]:
        runner = BackgroundTaskRunner(enabled=True, max_read_only=2)

        async def slow(value: int) -> int:
            await asyncio.sleep(0.05)
            return value

        started = time.perf_counter()
        first = await runner.submit_read_only("first", lambda: slow(1))
        second = await runner.submit_read_only("second", lambda: slow(2))
        results = await asyncio.gather(first, second)
        return time.perf_counter() - started, [result.value for result in results]

    elapsed, values = asyncio.run(run())

    assert values == [1, 2]
    assert elapsed < 0.09


def test_small_llm_background_task_requires_lease_and_releases_it():
    class FakeScheduler:
        def __init__(self) -> None:
            self.released: list[str] = []
            self.admission: dict[str, object] | None = None

        async def estimate_and_reserve(self, **kwargs):
            self.admission = kwargs
            return SimpleNamespace(lease_id="lease-1"), {"ok": True, "lease_id": "lease-1"}

        async def release_lease(self, lease_id: str, *, status: str = "released"):
            self.released.append(f"{lease_id}:{status}")

    async def run() -> tuple[object, FakeScheduler]:
        scheduler = FakeScheduler()
        runner = BackgroundTaskRunner(enabled=True, scheduler=scheduler, max_small_llm=1)
        task = await runner.submit_small_llm(
            "judge",
            lambda: asyncio.sleep(0, result="done"),
            messages=[{"role": "user", "content": "x"}],
            max_output_tokens=64,
            priority="low",
            speculative=True,
        )
        return await task, scheduler

    result, scheduler = asyncio.run(run())

    assert result.ok
    assert result.value == "done"
    assert result.lease_id == "lease-1"
    assert scheduler.released == ["lease-1:released"]
    assert scheduler.admission is not None
    assert scheduler.admission["background"] is True
    assert scheduler.admission["speculative"] is True


def test_speculative_small_llm_task_skips_when_lease_is_denied():
    class DenyingScheduler:
        async def estimate_and_reserve(self, **kwargs):
            return None, {"ok": False, "reason": "insufficient_context"}

    async def run():
        runner = BackgroundTaskRunner(enabled=True, scheduler=DenyingScheduler(), max_small_llm=1)
        task = await runner.submit_small_llm(
            "router",
            lambda: asyncio.sleep(0, result="should-not-run"),
            messages=[{"role": "user", "content": "x"}],
            speculative=True,
        )
        return await task

    result = asyncio.run(run())

    assert result.status == "skipped"
    assert result.admission == {"ok": False, "reason": "insufficient_context"}


def test_cancel_low_priority_cancels_speculative_tasks_only():
    async def run() -> tuple[list[str], str, str]:
        runner = BackgroundTaskRunner(enabled=True, max_read_only=2)
        low = await runner.submit_read_only(
            "low",
            lambda: asyncio.sleep(1, result="low"),
            speculative=True,
        )
        medium = await runner.submit_read_only(
            "medium",
            lambda: asyncio.sleep(0.01, result="medium"),
            priority="medium",
            speculative=False,
        )
        cancelled = await runner.cancel_low_priority()
        low_result = await low
        medium_result = await medium
        return cancelled, low_result.status, medium_result.status

    cancelled, low_status, medium_status = asyncio.run(run())

    assert cancelled == ["low"]
    assert low_status == "cancelled"
    assert medium_status == "completed"
