from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

from context_compressor import (  # noqa: E402
    _truncate_tool_call_args_json,
    build_summary_message_content,
    compress_messages,
    estimate_tokens_rough,
    prepare_messages_for_summary,
    redacted_message_to_json,
    select_head_middle_tail,
    should_compress,
)
from model_metadata import context_limit_from_ratio, get_model_context_length  # noqa: E402


def test_estimate_tokens_counts_images_and_tool_args() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look at this"},
                {"type": "image_url", "image_url": {"url": "file://image.png"}},
            ],
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "read_file",
                    "args": {"path": "langgraph-app/agent_graph.py", "why": "x" * 1000},
                }
            ],
        },
    ]

    assert estimate_tokens_rough(messages) > 1700


def test_json_safe_tool_args_truncation_preserves_json() -> None:
    os.environ["ALPHARAVIS_COMPRESSION_TOOL_ARGS_MAX_CHARS"] = "120"
    os.environ["ALPHARAVIS_COMPRESSION_TOOL_ARGS_HEAD_CHARS"] = "40"
    truncated = _truncate_tool_call_args_json(json.dumps({"nested": {"text": "x" * 400}}))

    parsed = json.loads(truncated)
    assert "omitted from tool arguments" in parsed["nested"]["text"]


def test_invalid_tool_args_are_left_unchanged() -> None:
    raw = "{not valid json"
    assert _truncate_tool_call_args_json(raw) == raw


def test_tool_output_deduplication_for_summary_prompt() -> None:
    repeated = "same output\n" * 80
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "call_1", "name": "read_file", "args": {"path": "a.py"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": repeated},
        {"role": "assistant", "tool_calls": [{"id": "call_2", "name": "read_file", "args": {"path": "a.py"}}]},
        {"role": "tool", "tool_call_id": "call_2", "content": repeated},
    ]

    prep = prepare_messages_for_summary(messages)
    assert prep.deduped_tool_count == 1
    assert "same content as newer tool output" in prep.text


def test_informative_tool_result_summary() -> None:
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "call_1", "name": "shell_command", "args": {"command": "docker ps"}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "Exit code: 0\nline 1\nline 2"},
    ]

    prep = prepare_messages_for_summary(messages)
    assert "ran `docker ps` -> exit 0" in prep.text


def test_head_middle_tail_keeps_handoff_packet_protected() -> None:
    messages = [{"id": f"m{i}", "role": "user", "content": f"message {i} " + ("x" * 80)} for i in range(25)]
    messages[10]["content"] = "<handoff-packet>{\"report_type\": \"handoff_packet\"}</handoff-packet>"

    selection = select_head_middle_tail(messages, token_limit=100, protected_message_ids=set())
    assert 10 in selection.head_indexes
    assert messages[10] in selection.head


def test_anti_thrashing_blocks_auto_and_force_ignores_it() -> None:
    stats = {"ineffective_compression_count": 2}
    blocked = should_compress(token_estimate=500, token_limit=100, compression_stats=stats)
    forced = should_compress(token_estimate=500, token_limit=100, compression_stats=stats, force=True)

    assert not blocked.should_run
    assert blocked.reason == "anti_thrashing"
    assert forced.should_run


def test_percent_context_limit_helper_and_env_override() -> None:
    old_values = {
        key: os.environ.get(key)
        for key in (
            "ALPHARAVIS_AUTO_DISCOVER_CONTEXT_LENGTH",
            "ALPHARAVIS_MODEL_CONTEXT_LENGTH",
            "ALPHARAVIS_DEFAULT_CONTEXT_LENGTH",
            "ALPHARAVIS_CONTEXT_LENGTH_BIG_BOSS",
        )
    }
    try:
        os.environ["ALPHARAVIS_AUTO_DISCOVER_CONTEXT_LENGTH"] = "false"
        os.environ["ALPHARAVIS_MODEL_CONTEXT_LENGTH"] = "0"
        os.environ["ALPHARAVIS_DEFAULT_CONTEXT_LENGTH"] = "0"
        os.environ["ALPHARAVIS_CONTEXT_LENGTH_BIG_BOSS"] = "64000"

        assert get_model_context_length("big-boss") == 64000
        assert context_limit_from_ratio(64000, 0.50, minimum=4096) == 32000
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_summary_failure_returns_visible_fallback_and_archive_content() -> None:
    async def failing_summary(_prompt: str, _max_tokens: int) -> str:
        raise RuntimeError("summary backend down")

    messages = [{"role": "user", "content": f"message {i}"} for i in range(26)]
    messages[6]["content"] = "api_key=sk-secret1234567890abcdef should redact"

    result = asyncio.run(
        compress_messages(
            messages,
            mode="post_run",
            thread_id="thread",
            thread_key="thread",
            token_limit=10,
            previous_summary="Old summary.",
            summarize_fn=failing_summary,
            force=True,
        )
    )

    assert result.summary_failed
    assert "Summary generation failed" in result.summary
    assert "<redacted" in result.archive_content
    assert result.middle
    assert result.compression_stats["summary_failure_cooldown_until"] > 0


def test_iterative_prompt_keeps_previous_summary() -> None:
    async def echo_summary(prompt: str, _max_tokens: int) -> str:
        assert "Previous summary:" in prompt
        assert "Old summary still relevant" in prompt
        return "## Active Task\n- updated\n\n## Remaining Work\n- none"

    messages = [{"role": "user", "content": f"message {i}"} for i in range(26)]
    result = asyncio.run(
        compress_messages(
            messages,
            mode="handoff",
            thread_id="thread",
            thread_key="thread",
            token_limit=10,
            previous_summary="Old summary still relevant",
            summarize_fn=echo_summary,
            force=True,
        )
    )
    assert "updated" in result.summary


def test_reference_only_summary_message() -> None:
    content = build_summary_message_content(
        mode="post_run",
        summary="## Active Task\n- x",
        archive_key="archive_1",
        token_estimate_before=100,
        token_estimate_after=50,
    )

    assert "REFERENCE ONLY" in content
    assert "Do NOT answer questions" in content
    assert "Answer only the latest user request" in content


def test_redacted_archive_json_is_meaningful() -> None:
    data = redacted_message_to_json({"role": "user", "content": "password=supersecretvalue and useful context"})

    assert data["archive_redacted"] is True
    assert "useful context" in data["content"]
    assert "supersecretvalue" not in data["content"]


def _run_all() -> None:
    tests = [
        test_estimate_tokens_counts_images_and_tool_args,
        test_json_safe_tool_args_truncation_preserves_json,
        test_invalid_tool_args_are_left_unchanged,
        test_tool_output_deduplication_for_summary_prompt,
        test_informative_tool_result_summary,
        test_head_middle_tail_keeps_handoff_packet_protected,
        test_anti_thrashing_blocks_auto_and_force_ignores_it,
        test_percent_context_limit_helper_and_env_override,
        test_summary_failure_returns_visible_fallback_and_archive_content,
        test_iterative_prompt_keeps_previous_summary,
        test_reference_only_summary_message,
        test_redacted_archive_json_is_meaningful,
    ]
    for test in tests:
        test()


if __name__ == "__main__":
    _run_all()
