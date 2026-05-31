#!/usr/bin/env python3
"""Verify fire-and-forget fix: _schedule_background_task helper."""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

# Import just the helper functions (doesn't pull in full agent_graph deps)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "agent_graph", ROOT / "langgraph-app" / "agent_graph.py"
)
# We can't import the whole module (pulls in deepagents), so read the helper source directly
source = (ROOT / "langgraph-app" / "agent_graph.py").read_text()

# Extract the helper function source
fn_start = source.index("def _schedule_background_task(")
fn_end = source.index("\ndef _print_if_exception(")
helper_source = source[fn_start:fn_end]

fn2_start = source.index("def _print_if_exception(")
fn2_end = source.index("\n\n@tool\n", fn2_start)
helper2_source = source[fn2_start:fn2_end]

# Execute the helper definitions in a namespace
namespace = {"asyncio": asyncio}
exec(helper_source, namespace)
exec(helper2_source, namespace)
_schedule_background_task = namespace["_schedule_background_task"]
_print_if_exception = namespace["_print_if_exception"]


async def _run_tests():
    print("=== _schedule_background_task Verification ===\n")

    # Test 1: Successful task — no error output
    print("Test 1: Successful task completes silently")
    success_flag = False

    async def _success():
        nonlocal success_flag
        success_flag = True
        return "ok"

    _schedule_background_task(_success(), label="test_success")
    await asyncio.sleep(0.05)  # Let the task complete
    assert success_flag, "Task should have completed"
    print("  ✓ Successful task runs without errors")

    # Test 2: Failed task — error is logged (via print)
    print("Test 2: Failed task logs error via done_callback")
    import io
    import contextlib

    async def _fail():
        raise ValueError("simulated indexing failure")

    f = io.StringIO()
    with contextlib.redirect_stderr(f):
        _schedule_background_task(_fail(), label="test_fail")
        await asyncio.sleep(0.05)

    # The error should appear in stderr (print goes to stderr by default? No, stdout)
    # Let's capture stdout instead

    import io as io2
    g = io2.StringIO()
    with contextlib.redirect_stdout(g):
        _schedule_background_task(_fail(), label="test_fail2")
        await asyncio.sleep(0.05)

    output = g.getvalue()
    assert "background task failed" in output, f"Expected error log, got: {output}"
    assert "simulated indexing failure" in output, f"Expected exception message, got: {output}"
    assert "test_fail2" in output, f"Expected label in output, got: {output}"
    print("  ✓ Failed task produces error log with label and exception message")

    # Test 3: No event loop — silent skip
    print("Test 3: No event loop → silent skip")
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as pool:
        def _call_without_loop():
            _schedule_background_task(_success(), label="no_loop")
            return "no crash"

        result = pool.submit(_call_without_loop).result(timeout=2)
        assert result == "no crash", "Should not crash when no event loop"
    print("  ✓ No event loop → silent skip, no crash")

    # Test 4: _print_if_exception — non-exception task is silent
    print("Test 4: _print_if_exception with no exception is silent")
    async def _no_error():
        return "ok"

    task = asyncio.create_task(_no_error())
    await task

    h = io2.StringIO()
    with contextlib.redirect_stdout(h):
        _print_if_exception(task, label="test_no_error")

    assert h.getvalue() == "", f"Should produce no output for non-exception: {h.getvalue()}"
    print("  ✓ _print_if_exception silent for successful tasks")

    print(f"\n=== All tests passed ===")


if __name__ == "__main__":
    asyncio.run(_run_tests())
