"""Tests for langgraph-app/event_indexing.py — tool-run surrogate builder."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))


def test_import_event_indexing() -> None:
    from event_indexing import build_tool_run_surrogate, maybe_index_tool_run, surrogate_summary

    assert callable(build_tool_run_surrogate)
    assert callable(maybe_index_tool_run)
    assert callable(surrogate_summary)


def test_build_tool_run_surrogate_success() -> None:
    from event_indexing import build_tool_run_surrogate

    surrogate = build_tool_run_surrogate(
        "execute_local_command",
        "Exit Code 0\nSTDOUT: 12 passed\nSTDERR: ",
        exit_code=0,
    )
    assert "Tool run: execute_local_command" in surrogate
    assert "Status: success" in surrogate
    assert "Exit code: 0" in surrogate
    assert "12 passed" in surrogate


def test_build_tool_run_surrogate_failure() -> None:
    from event_indexing import build_tool_run_surrogate

    surrogate = build_tool_run_surrogate(
        "execute_local_command",
        "Exit Code 1\nSTDOUT: \nSTDERR: command not found",
        exit_code=1,
    )
    assert "Status: failed (exit 1)" in surrogate
    assert "command not found" in surrogate


def test_build_tool_run_surrogate_scrubs_secrets() -> None:
    from event_indexing import build_tool_run_surrogate

    surrogate = build_tool_run_surrogate(
        "execute_local_command",
        "Exit Code 0\nSTDOUT: export TOKEN=ghp_abc123def456ghi789jkl012mno345pqr678\nSTDERR: ",
        exit_code=0,
    )
    assert "ghp_" not in surrogate, "GitHub token must be redacted"
    assert "[REDACTED]" in surrogate, "Secret redaction placeholder must appear"


def test_build_tool_run_surrogate_bounds_output() -> None:
    from event_indexing import build_tool_run_surrogate

    long_output = "A" * 2000
    surrogate = build_tool_run_surrogate(
        "list_files",
        f"Exit Code 0\nSTDOUT: {long_output}\nSTDERR: ",
        exit_code=0,
        max_chars=200,
    )
    assert "[Output truncated.]" in surrogate
    assert len(surrogate) < len(long_output)


def test_maybe_index_tool_run_disabled_by_default() -> None:
    from event_indexing import maybe_index_tool_run

    # With env var absent/false, returns None
    import os
    os.environ.pop("ALPHARAVIS_ENABLE_TOOL_EVENT_VECTOR_INDEX", None)
    result = maybe_index_tool_run("test_tool", "some output", exit_code=0)
    assert result is None, "Feature flag OFF should return None"


def test_surrogate_summary_extracts_tool_and_status() -> None:
    from event_indexing import build_tool_run_surrogate, surrogate_summary

    surrogate = build_tool_run_surrogate("execute_local_command", "done", exit_code=0)
    summary = surrogate_summary(surrogate)
    assert "execute_local_command" in summary
    assert "success" in summary
