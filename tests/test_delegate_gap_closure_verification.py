"""Verification tests for delegate gap-closure commits (c7bfc50..050a467).

Tests: provider override, retry logic, heartbeat, toolset blocklist,
workspace hint, improved prompts, constant definitions.
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "langgraph-app"))

import pytest
from delegate_agent import (
    DELEGATE_PROVIDER, DELEGATE_MODEL, DELEGATE_API_BASE, DELEGATE_API_KEY,
    HEARTBEAT_ENABLED, HEARTBEAT_INTERVAL,
    DELEGATE_INTERSECT_PARENT_TOOLS, DELEGATE_BLOCKED_TOOLS,
    DELEGATE_MAX_RETRIES, DELEGATE_RETRY_DELAY,
    DELEGATE_WORKSPACE_HINT,
    _is_retryable_error, _heartbeat_loop,
    run_sub_agent,
)


# ── Constants ────────────────────────────────────────────────────────
def test_provider_constants_are_strings():
    assert isinstance(DELEGATE_PROVIDER, str)
    assert isinstance(DELEGATE_MODEL, str)
    assert isinstance(DELEGATE_API_BASE, str)
    assert isinstance(DELEGATE_API_KEY, str)
    # Default: empty (provider override off)
    assert DELEGATE_PROVIDER == ""
    assert DELEGATE_MODEL == ""


def test_heartbeat_constants():
    assert HEARTBEAT_ENABLED is True  # default on
    assert HEARTBEAT_INTERVAL == 30.0


def test_retry_constants():
    assert DELEGATE_MAX_RETRIES == 2
    assert DELEGATE_RETRY_DELAY == 5.0


def test_blocked_tools_is_frozenset_with_defaults():
    assert isinstance(DELEGATE_BLOCKED_TOOLS, frozenset)
    assert "clarify" in DELEGATE_BLOCKED_TOOLS
    assert "memory" in DELEGATE_BLOCKED_TOOLS
    assert "send_message" in DELEGATE_BLOCKED_TOOLS


def test_workspace_constant():
    assert isinstance(DELEGATE_WORKSPACE_HINT, str)


# ── _is_retryable_error ──────────────────────────────────────────────
@pytest.mark.parametrize("message,expected", [
    ("rate limit exceeded", True),
    ("rate_limit hit", True),
    ("too many requests", True),
    ("timeout", True),
    ("timed out waiting", True),
    ("overloaded", True),
    ("capacity exceeded", True),
    ("server error", True),
    ("internal server error", True),
    ("503", True),
    ("502 bad gateway", True),
    ("500 internal error", True),
    ("529", True),
    ("429", True),
    ("connection refused", True),
    ("connect error", True),
    ("reset by peer", True),
    ("broken pipe", True),
    ("service unavailable", True),
    ("temporarily unavailable", True),
])
def test_is_retryable_true(message, expected):
    assert _is_retryable_error(Exception(message)) == expected


@pytest.mark.parametrize("message,expected", [
    ("invalid api key", False),
    ("model not found", False),
    ("context length exceeded", False),
    ("bad request", False),
    ("authentication failed", False),
    ("unauthorized", False),
    ("", False),
])
def test_is_retryable_false(message, expected):
    assert _is_retryable_error(Exception(message)) == expected


# ── Signature checks ─────────────────────────────────────────────────
def test_run_sub_agent_signature():
    sig = inspect.signature(run_sub_agent)
    params = set(sig.parameters.keys())

    # New params must exist
    assert "_provider" in params
    assert "_model_name" in params
    assert "_api_base" in params
    assert "_api_key" in params
    assert "_parent_touch_fn" in params

    # Defaults must be correct
    assert sig.parameters["_provider"].default == ""
    assert sig.parameters["_model_name"].default == ""
    assert sig.parameters["_api_base"].default == ""
    assert sig.parameters["_api_key"].default == ""
    assert sig.parameters["_parent_touch_fn"].default is None

    # Core params unchanged
    assert "goal" in params
    assert "max_iterations" in params
    assert "timeout_seconds" in params


# ── Heartbeat ────────────────────────────────────────────────────────
def test_heartbeat_loop_is_coroutine():
    assert inspect.iscoroutinefunction(_heartbeat_loop)


def test_heartbeat_loop_signature():
    sig = inspect.signature(_heartbeat_loop)
    params = set(sig.parameters.keys())
    assert params == {"agent_id", "cancel_evt", "parent_touch_fn", "started"}


# ── Source code structure checks ─────────────────────────────────────

def _load_source(name):
    path = os.path.join(os.path.dirname(__file__), "..", "langgraph-app", name)
    return open(path).read()


def test_retry_loop_uses_exponential_backoff():
    """Verify retry uses exponential formula: delay * (2 ** attempt)."""
    source = _load_source("delegate_agent.py")
    assert "retry_delay * (2 ** attempt)" in source
    assert "max_retries = DELEGATE_MAX_RETRIES" in source
    assert "retry_delay = DELEGATE_RETRY_DELAY" in source


def test_retry_gated_by_is_retryable():
    """Only retryable errors trigger backoff; non-retryable break immediately."""
    source = _load_source("delegate_agent.py")
    assert "if attempt < max_retries and _is_retryable_error(exc):" in source
    # On non-retryable: break out (no retry)
    assert "# Out of retries or non-retryable" in source


def test_heartbeat_cleanup_in_finally():
    """The finally block must cancel and await heartbeat_task."""
    source = _load_source("delegate_agent.py")
    # Find the finally block near heartbeat cleanup
    finally_idx = source.index("# Clean up heartbeat task")
    end_idx = source.index("\ndef ", finally_idx + 10)
    finally_block = source[finally_idx:end_idx]
    assert "heartbeat_task.cancel()" in finally_block
    assert "await heartbeat_task" in finally_block
    assert "asyncio.CancelledError" in finally_block


def test_blocklist_applied_in_tool_resolution():
    """DELEGATE_BLOCKED_TOOLS is checked during tool selection."""
    source = _load_source("delegate_agent.py")
    assert "if name in DELEGATE_BLOCKED_TOOLS:" in source
    assert "continue" in source  # skip blocked tools


def test_output_format_has_issues_section():
    """The output format includes 5 sections (was 4), adding Issues."""
    source = _load_source("delegate_agent.py")
    assert "## Issues" in source
    assert "## Summary" in source
    assert "## Key Findings" in source
    assert "## Actions" in source
    assert "## Recommendation" in source


def test_orchestrator_guidance_exists():
    """Orchestrator sub-agents get guidance on when to delegate."""
    source = _load_source("delegate_agent.py")
    assert "WHEN to delegate:" in source
    assert "WHEN NOT to delegate:" in source
    assert "Subagent Spawning (Orchestrator)" in source


def test_workspace_hint_in_prompt():
    """The system prompt includes workspace path and anti-/workspace/ rule."""
    source = _load_source("delegate_agent.py")
    assert "DELEGATE_WORKSPACE_HINT or os.getenv" in source
    assert "Never assume /workspace/" in source
    assert "Use this exact path for file operations" in source


def test_no_destructive_trimming():
    """Sub-agents use compress_messages() pipeline, not trim."""
    source = _load_source("delegate_agent.py")
    assert "_compress_sub_agent_context" in source
    assert "destructively trim" in source.lower()


# ── Agent-graph integration ──────────────────────────────────────────
def test_agent_graph_passes_provider_params():
    """agent_graph.py delegate_task passes _provider, _model_name, etc."""
    ag_source = _load_source("agent_graph.py")
    call_site = ag_source[ag_source.index("return await _run_sub_agent("):ag_source.index("\n        )", ag_source.index("return await _run_sub_agent("))]
    assert "_provider=" in call_site
    assert "_model_name=" in call_site
    assert "_api_base=" in call_site
    assert "_api_key=" in call_site


def test_agent_graph_missing_parent_touch_fn():
    """KNOWN ISSUE: agent_graph.py does NOT pass _parent_touch_fn,
    so the heartbeat is effectively disabled. This test documents the gap."""
    ag_source = _load_source("agent_graph.py")
    call_site = ag_source[ag_source.index("return await _run_sub_agent("):ag_source.index("\n        )", ag_source.index("return await _run_sub_agent("))]
    # This asserts the KNOWN GAP — when the gap is closed, flip this test
    assert "_parent_touch_fn" not in call_site, (
        "GAP: _parent_touch_fn is not passed from agent_graph.py. "
        "Heartbeat is wired in delegate_agent.py but never activated. "
        "If this assertion fails, the gap has been closed — update this test."
    )


def test_intersect_parent_tools_is_unused():
    """KNOWN ISSUE: DELEGATE_INTERSECT_PARENT_TOOLS is defined but never checked.
    Tool intersection runs unconditionally regardless of this ENV var."""
    source = _load_source("delegate_agent.py")
    # Find tool resolution block
    tool_start = source.index("# Resolve tools with blocklist")
    tool_end = source.index("# Build tool schemas")
    tool_block = source[tool_start:tool_end]
    assert "DELEGATE_INTERSECT_PARENT_TOOLS" not in tool_block, (
        "GAP: DELEGATE_INTERSECT_PARENT_TOOLS is never checked in tool resolution. "
        "Intersection always runs. This ENV var is dead code."
    )
