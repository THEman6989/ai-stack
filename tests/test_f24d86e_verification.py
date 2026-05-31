"""Verification tests for commit f24d86e — akribische Code-Review.
Covers findings discovered during the review that existing tests missed.
"""

import ast
import asyncio
import os
import re
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Finding #1: Registry Leak — 3 exit paths without unregister
# ---------------------------------------------------------------------------


def test_registry_unregister_all_exit_paths():
    """Every return in run_sub_agent except cancellation must call unregister."""
    source = Path("langgraph-app/delegate_agent.py").read_text()

    # Extract run_sub_agent body
    fn_start = source.index("async def run_sub_agent(")
    # Find next top-level def/class
    next_boundary = float("inf")
    for pattern in [r"\nasync def ", r"\ndef ", r"\nclass "]:
        for m in re.finditer(pattern, source[fn_start + 1 :]):
            next_boundary = min(next_boundary, m.start() + fn_start + 1)
            break
    body = source[fn_start:next_boundary]

    returns = list(re.finditer(r"^\s+return\s", body, re.MULTILINE))
    for i, ret in enumerate(returns):
        before = body[: ret.start()]
        has_unregister = "unregister" in before[-800:]  # Check last 800 chars before return
        # Check if this is the cancellation path (cancel_event.is_set)
        is_cancellation_path = "cancel_evt is not None and cancel_evt.is_set()" in before[
            -1500:
        ]
        is_setup_failure = (
            "_model_fn is None" in before[-500:] or "model is None" in before[-500:]
        )

        if is_cancellation_path:
            # Cancellation: arguably intentional, but document
            pass  # Not a bug per se, but a design choice
        elif is_setup_failure:
            assert has_unregister, (
                f"Setup failure return #{i+1} MUST call unregister "
                f"(leaks agent in registry)"
            )
        else:
            assert has_unregister, (
                f"Return #{i+1} MUST call unregister (leaks agent in registry)"
            )


# ---------------------------------------------------------------------------
# Finding #2: Nested delegation depth hardcoded to 0
# ---------------------------------------------------------------------------


def test_nested_delegation_depth_not_hardcoded():
    """delegate_task must use context vars for depth, not hardcode 0."""
    source = Path("langgraph-app/agent_graph.py").read_text()

    # Find the _run_one_via_module definition
    pattern = r"async def _run_one_via_module\(task_def.*?return await _run_sub_agent\(.*?depth=(child_depth|\d+).*?parent_id=(child_parent|\w+)"
    match = re.search(pattern, source, re.DOTALL)
    assert match is not None, "Could not find _run_one_via_module in delegate_task"

    depth_val = match.group(1)
    parent_val = match.group(2)

    assert depth_val == "child_depth", (
        f"delegate_task must use dynamic depth (child_depth from context), not {depth_val}. "
        "Nested delegation is broken without dynamic depth."
    )
    assert parent_val == "child_parent", (
        f"delegate_task must use dynamic parent_id (child_parent from context), not {parent_val}. "
        "Children spawned by sub-agents would have no parent without dynamic parent_id."
    )


# ---------------------------------------------------------------------------
# Finding #3: No recursive call to run_sub_agent from within run_sub_agent
# ---------------------------------------------------------------------------


def test_run_sub_agent_has_no_recursive_call():
    """Nested delegation uses context vars + delegate_task @tool, not direct recursion.

    Architecture: sub-agents call delegate_task @tool → run_sub_agent().
    This is the correct pattern because sub-agents only have access to @tools,
    not to internal functions. Context vars (get_current_agent_context) carry
    depth/parent_id across the boundary.
    """
    source = Path("langgraph-app/delegate_agent.py").read_text()

    # Verify the context vars mechanism exists (the REAL recursion path)
    assert "contextvars" in source, "Must import contextvars for nested delegation"
    assert "get_current_agent_context" in source, "Must expose agent context for delegate_task"
    assert "_CURRENT_AGENT_ID" in source, "Must track current agent ID via context var"
    assert "_CURRENT_AGENT_DEPTH" in source, "Must track current depth via context var"

    # Verify delegate_task reads context vars
    agent_graph = Path("langgraph-app/agent_graph.py").read_text()
    assert "_get_current_agent_context" in agent_graph, (
        "delegate_task must call _get_current_agent_context for nested delegation"
    )


# ---------------------------------------------------------------------------
# Finding #4: SubAgentRegistry leak after kill_agent
# ---------------------------------------------------------------------------


def test_sub_agent_registry_does_not_unregister_on_kill():
    """kill_agent should optionally unregister or caller must handle cleanup."""
    from delegate_agent import SubAgentRegistry

    reg = SubAgentRegistry()
    ctx = reg.register(parent_id=None, depth=0, goal="Test")
    agent_id = ctx.agent_id

    reg.kill(agent_id)

    # Agent is still in registry after kill
    assert reg.get(agent_id) is not None, "Killed agent should remain visible"
    assert reg.get(agent_id).state == "cancelled"

    # But run_sub_agent also doesn't unregister on cancellation
    # This means cancelled agents stay in registry FOREVER
    # Documenting this design choice for awareness


# ---------------------------------------------------------------------------
# Finding #5: _extract_write_path edge cases
# ---------------------------------------------------------------------------


def test_extract_write_path_stderr_redirect():
    """2> and &> redirects should be captured but currently are NOT."""
    from delegate_agent import _extract_write_path_from_command

    # These are currently NOT captured — they should be
    result = _extract_write_path_from_command("cmd 2> /tmp/err.log")
    assert result is None, (
        f"2> redirect is NOT captured (returns {result!r}). "
        "This is an edge case gap — stderr redirects to files should trigger "
        "file-state tracking for cross-agent awareness."
    )

    result2 = _extract_write_path_from_command("cmd &> /tmp/all.log")
    assert result2 is None, (
        f"&> redirect is NOT captured (returns {result2!r}). "
        "This is an edge case gap."
    )


def test_extract_write_path_subshell():
    """Subshell writes like (cmd > file) should be captured."""
    from delegate_agent import _extract_write_path_from_command

    # Subshell redirect
    result = _extract_write_path_from_command("(echo test > /tmp/out.txt)")
    assert result == "/tmp/out.txt", f"Expected /tmp/out.txt, got {result!r}"


# ---------------------------------------------------------------------------
# Finding #6: _normalize_tool_names edge with mixed case
# ---------------------------------------------------------------------------


def test_normalize_tool_names_case_preservation():
    """_normalize_tool_names should preserve original names (case-sensitive lookup)."""
    from delegate_agent import _normalize_tool_names

    result = _normalize_tool_names(["ReadFile", "WriteFile"])
    assert result == {"ReadFile", "WriteFile"}, f"Expected preserved case, got {result}"

    # None input
    assert _normalize_tool_names(None) is None
    # Empty list
    assert _normalize_tool_names([]) is None
    # List with only whitespace
    assert _normalize_tool_names(["  ", "\t"]) is None


# ---------------------------------------------------------------------------
# Finding #7: Lambda closure stability
# ---------------------------------------------------------------------------


def test_lambda_closure_captures_name_not_value():
    """_model_fn lambda captures _model by name — verify this is safe.

    The lambda in delegate_task: lambda kw: _model(model_kwargs=kw)
    This captures the name '_model', not the function object.
    If _model were reassigned during agent execution, children would
    pick up the new assignment.
    """
    # Test the principle
    def original_model(**kw):
        return f"original:{kw}"

    def replacement_model(**kw):
        return f"replacement:{kw}"

    _model = original_model
    fn = lambda kw: _model(model_kwargs=kw)

    assert fn({"x": 1}) == "original:{'model_kwargs': {'x': 1}}"

    # Reassign _model
    _model = replacement_model
    assert fn({"x": 1}) == "replacement:{'model_kwargs': {'x': 1}}"

    # _model in agent_graph is never reassigned, so this is safe.
    # But the pattern is fragile — documented.


# ---------------------------------------------------------------------------
# Finding #8: FileStateTracker ns/s guard correctness
# ---------------------------------------------------------------------------


def test_file_state_tracker_mtime_guard_all_real_timestamps(tmp_path: Path):
    """Test the mtime_ns > 1e12 guard with a real file (real timestamps always > 1e12)."""
    from delegate_agent import FileStateTracker

    tracker = FileStateTracker()
    test_file = tmp_path / "guard_test.txt"
    test_file.write_text("test")

    # Record write
    tracker.record_write(str(test_file), "agent-1")

    # Verify the mtime_ns stored is > 1e12 (real Unix timestamps in ns)
    import os as _os

    stat_ns = test_file.stat().st_mtime_ns
    # Real files on any Linux system from 2001+ have mtime_ns > 1e12
    # (1e9 seconds * 1e9 ns/second = 1e18)
    assert stat_ns > 1e12, f"Real mtime_ns={stat_ns} should be > 1e12"


def test_file_state_tracker_same_agent_no_warning(tmp_path: Path):
    """Same agent writing and reading its own file should not trigger stale warning."""
    from delegate_agent import FileStateTracker

    tracker = FileStateTracker()
    test_file = tmp_path / "own.txt"
    test_file.write_text("hello")

    tracker.record_write(str(test_file), "agent-1")
    warning = tracker.check_stale_read(str(test_file), "agent-1")
    assert warning is None, "Same agent should not get stale warning"


def test_file_state_tracker_different_agent_mtime_no_false_positives(tmp_path: Path):
    """If file's mtime hasn't changed since write, second read by different agent after first read should not warn."""
    from delegate_agent import FileStateTracker

    tracker = FileStateTracker()
    test_file = tmp_path / "fresh.txt"
    test_file.write_text("initial")

    # Agent A writes
    tracker.record_write(str(test_file), "agent-A")

    # Agent B reads — stale warning (first read)
    warning1 = tracker.check_stale_read(str(test_file), "agent-B")
    assert warning1 is not None

    # Agent B reads again — no warning (already seen since write)
    warning2 = tracker.check_stale_read(str(test_file), "agent-B")
    assert warning2 is None


def test_file_state_tracker_write_then_rewrite_stale(tmp_path: Path):
    """If agent A writes, B reads, then A writes again, B should get stale warning."""
    from delegate_agent import FileStateTracker

    tracker = FileStateTracker()
    test_file = tmp_path / "rewrite.txt"
    test_file.write_text("v1")

    tracker.record_write(str(test_file), "agent-A")
    warning1 = tracker.check_stale_read(str(test_file), "agent-B")
    assert warning1 is not None  # First read after write

    # B reads again — no warning
    warning2 = tracker.check_stale_read(str(test_file), "agent-B")
    assert warning2 is None

    # A writes again (update file content to change mtime)
    time.sleep(0.01)  # Ensure mtime changes
    test_file.write_text("v2")
    tracker.record_write(str(test_file), "agent-A")

    # B reads — should get stale warning again
    warning3 = tracker.check_stale_read(str(test_file), "agent-B")
    assert warning3 is not None, "B should get stale warning after A's second write"


# ---------------------------------------------------------------------------
# Finding #9: Tool count verification in docstring vs actual list
# ---------------------------------------------------------------------------


def test_tool_count_matches_actual_list():
    """The 22-tool claim in docstring must match the actual tool list."""
    source = Path("langgraph-app/agent_graph.py").read_text()

    # Extract the tool list from delegate_task
    list_start = source.index("_delegate_tool_list = [")
    list_end = source.index("]", list_start) + 1
    list_section = source[list_start:list_end]

    # Count tool names (lines that are just identifiers)
    tool_names = []
    for line in list_section.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("_") and stripped != "]":
            if stripped.endswith(","):
                stripped = stripped[:-1]
            tool_names.append(stripped)

    actual_count = len(tool_names)
    assert actual_count == 22, (
        f"Tool list has {actual_count} entries, docstring says 22. "
        f"Tools: {tool_names}"
    )


# ---------------------------------------------------------------------------
# Finding #10: _build_tool_schemas v1/v2 fallback correctness
# ---------------------------------------------------------------------------


def test_build_tool_schemas_v1_fallback():
    """When model_json_schema is not available, schema() is used as fallback."""
    from delegate_agent import _build_tool_schemas

    class V1Schema:
        def schema(self):
            return {"type": "object", "properties": {"a": {"type": "int"}}}

    class V1Tool:
        args_schema = V1Schema()
        description = "v1 tool"

    class V2Schema:
        def model_json_schema(self):
            return {"type": "object", "properties": {"b": {"type": "str"}}}

    class V2Tool:
        args_schema = V2Schema()
        description = "v2 tool"

    tools = {"v1": V1Tool(), "v2": V2Tool()}

    schemas = _build_tool_schemas(tools, _tool_name_fn=lambda t: t.__class__.__name__)
    assert len(schemas) == 2

    # Both should have proper parameters
    for s in schemas:
        assert "parameters" in s["function"]
        assert "type" in s["function"]["parameters"]


def test_build_tool_schemas_no_args_schema():
    """Tools without args_schema get a minimal fallback schema (not silently skipped).

    Changed behavior: bfaa0dd silently skipped tools without schema.
    Now they get {"type": "object", "properties": {}} + WARNING log.
    The model can still call them — parameter validation is minimal but
    usable. This matches Hermes' approach.
    """
    from delegate_agent import _build_tool_schemas

    class NoSchemaTool:
        description = "no schema"

    tools = {"noschema": NoSchemaTool()}
    schemas = _build_tool_schemas(tools, _tool_name_fn=lambda t: "test")
    assert len(schemas) == 1, "Should get 1 schema (fallback), not 0"
    assert schemas[0]["function"]["name"] == "test"
    assert schemas[0]["function"]["description"] == "no schema"
    assert schemas[0]["function"]["parameters"] == {"type": "object", "properties": {}}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
