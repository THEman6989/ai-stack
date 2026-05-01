from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

from prompt_assembly import (  # noqa: E402
    build_environment_hints,
    build_stable_prompt_context,
    truncate_context_content,
)


def test_truncate_context_content_keeps_head_tail_and_marker():
    text = "A" * 100 + "MIDDLE" + "Z" * 100
    truncated = truncate_context_content(text, "sample.txt", max_chars=80)

    assert truncated.startswith("A")
    assert truncated.endswith("Z" * 16)
    assert "truncated sample.txt" in truncated
    assert "MIDDLE" not in truncated


def test_build_environment_hints_detects_windows_path():
    hints = build_environment_hints(cwd="C:\\experi\\ai\\ai-stack")

    assert "Windows" in hints
    assert "PowerShell" in hints


def test_stable_prompt_context_separates_stable_from_ephemeral():
    context = build_stable_prompt_context(cwd="C:\\experi\\ai\\ai-stack")

    assert context.startswith("<stable-runtime-context>")
    assert "Stable prompt policy" in context
    assert "Tool policy" in context
    assert "archive" in context.lower()


def _run_all() -> None:
    tests = [
        test_truncate_context_content_keeps_head_tail_and_marker,
        test_build_environment_hints_detects_windows_path,
        test_stable_prompt_context_separates_stable_from_ephemeral,
    ]
    for test in tests:
        test()


if __name__ == "__main__":
    _run_all()
