from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

from context_compressor import (  # noqa: E402
    _summary_chunk_output_token_limit,
    _truncate_tool_call_args_json,
    _truncate_summary_input_to_budget,
    _summary_prompt_token_limit,
    _summary_token_limit,
    build_summary_message_content,
    compress_messages,
    estimate_tokens_rough,
    prepare_messages_for_summary,
    redacted_message_to_json,
    select_head_middle_tail,
    should_compress,
)
from model_metadata import context_limit_from_ratio, get_model_context_length  # noqa: E402
from model_metadata import estimate_message_tokens_rough  # noqa: E402
from model_metadata import parse_context_limit_from_error  # noqa: E402


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


def test_parse_context_limit_from_provider_errors() -> None:
    assert parse_context_limit_from_error("maximum context length is 32768 tokens") == 32768
    assert parse_context_limit_from_error("llama.cpp n_ctx_slot = 131072, prompt is too long") == 131072
    assert parse_context_limit_from_error("input 250000 tokens > 200000 maximum") == 200000


def test_summary_prompt_input_is_pruned_to_own_budget(monkeypatch) -> None:
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_MAX_TOKENS", "1200")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_MIN_TOKENS", "1000")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_CHARS_PER_TOKEN", "2")
    text = "middle evidence " * 1000

    pruned, stats = _truncate_summary_input_to_budget(text, token_limit=8000)

    assert stats["summary_prompt_pruned"] is True
    assert len(pruned) < len(text)
    assert stats["summary_prompt_token_limit"] == 2048
    assert stats["summary_prompt_chars"] <= 4096
    assert "summary input pruned" in pruned


def test_summary_limits_can_be_context_ratio_without_static_cap(monkeypatch) -> None:
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_RATIO", "0.20")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_MAX_TOKENS", "0")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_RATIO", "0.75")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_MAX_TOKENS", "0")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_CHUNK_RATIO", "0.03")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_CHUNK_MAX_TOKENS", "0")

    assert _summary_token_limit(128000) == 25600
    assert _summary_prompt_token_limit(128000) == 96000
    assert _summary_chunk_output_token_limit(128000) == 3840


def test_compression_target_and_summary_model_context_are_separate(monkeypatch) -> None:
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_RATIO", "0.20")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_MAX_TOKENS", "0")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_RATIO", "0.75")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_MAX_TOKENS", "0")

    max_tokens_seen: list[int] = []

    async def summary(prompt: str, max_tokens: int) -> str:
        max_tokens_seen.append(max_tokens)
        assert "middle evidence" in prompt
        return "## Active Task\n- compact\n\n## Archive References\n- source_type=archive"

    messages = [{"role": "user", "content": f"head {index}"} for index in range(3)]
    messages.extend({"role": "assistant", "content": f"middle evidence {index}"} for index in range(8))
    messages.extend({"role": "assistant", "content": f"tail {index}"} for index in range(3))

    result = asyncio.run(
        compress_messages(
            messages,
            mode="post_run",
            thread_id="thread",
            thread_key="thread",
            token_limit=100,
            summary_context_token_limit=128000,
            previous_summary=None,
            summarize_fn=summary,
            force=True,
        )
    )

    assert result.archive_metadata["compression_token_limit"] == 100
    assert result.archive_metadata["summary_context_token_limit"] == 128000
    assert result.archive_metadata["summary_prompt_token_limit"] == 96000
    assert max_tokens_seen == [25600]


def test_chunked_summary_is_opt_in_for_pruned_summary_prompt(monkeypatch) -> None:
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_MAX_TOKENS", "1200")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_MIN_TOKENS", "1000")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_CHARS_PER_TOKEN", "1")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_CHUNK_OVERLAP_CHARS", "0")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_MAX_CHUNKS", "20")

    calls: list[str] = []

    async def chunk_summary(prompt: str, _max_tokens: int) -> str:
        calls.append(prompt)
        if "mode: post_run_chunked_synthesis" in prompt:
            assert "Intermediate chunk summaries" in prompt
            return "## Active Task\n- final synthesis\n\n## Archive References\n- source_type=archive"
        assert "mode: post_run_chunk_" in prompt
        return "## Active Task\n- chunk evidence\n\n## Archive References\n- source_type=archive"

    messages = [
        {"role": "user", "content": f"protected head {index}"}
        for index in range(3)
    ]
    messages.extend(
        {"role": "assistant", "content": f"middle evidence {index} " + ("x " * 1500)}
        for index in range(6)
    )
    messages.extend({"role": "assistant", "content": f"tail {index}"} for index in range(3))

    result = asyncio.run(
        compress_messages(
            messages,
            mode="post_run",
            thread_id="thread",
            thread_key="thread",
            token_limit=100,
            previous_summary=None,
            summarize_fn=chunk_summary,
            force=True,
            enable_chunked_summary=True,
        )
    )

    assert not result.summary_failed
    assert "final synthesis" in result.summary
    assert result.archive_metadata["summary_prompt_pruned"] is True
    assert result.archive_metadata["summary_chunking_used"] is True
    assert result.archive_metadata["summary_chunk_count"] > 1
    assert len(calls) == result.archive_metadata["summary_chunk_count"] + 1


def test_chunked_summary_reports_and_prompts_for_max_chunk_omissions(monkeypatch) -> None:
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_MAX_TOKENS", "1200")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_MIN_TOKENS", "1000")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_CHARS_PER_TOKEN", "1")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_CHUNK_OVERLAP_CHARS", "0")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_MAX_CHUNKS", "2")

    synthesis_prompts: list[str] = []

    async def chunk_summary(prompt: str, _max_tokens: int) -> str:
        if "mode: post_run_chunked_synthesis" in prompt:
            synthesis_prompts.append(prompt)
            return "## Active Task\n- final synthesis\n\n## Archive References\n- source_type=archive"
        return "## Active Task\n- chunk evidence\n\n## Archive References\n- source_type=archive"

    messages = [{"role": "user", "content": f"protected head {index}"} for index in range(3)]
    messages.extend(
        {"role": "assistant", "content": f"middle evidence {index} " + ("x " * 2500)}
        for index in range(12)
    )
    messages.extend({"role": "assistant", "content": f"tail {index}"} for index in range(3))

    result = asyncio.run(
        compress_messages(
            messages,
            mode="post_run",
            thread_id="thread",
            thread_key="thread",
            token_limit=100,
            previous_summary=None,
            summarize_fn=chunk_summary,
            force=True,
            enable_chunked_summary=True,
        )
    )

    assert not result.summary_failed
    assert result.archive_metadata["summary_chunking_used"] is True
    assert result.archive_metadata["summary_chunk_count"] == 2
    assert result.archive_metadata["summary_chunk_omitted_chars"] > 0
    assert result.archive_metadata["summary_chunk_payload_token_limit"] < result.archive_metadata["summary_chunk_prompt_token_limit"]
    assert synthesis_prompts
    assert "AlphaRavis chunking note" in synthesis_prompts[0]
    assert "archive lookup" in synthesis_prompts[0]


def test_oversized_tail_forces_chunked_summary_when_prompt_is_pruned(monkeypatch) -> None:
    monkeypatch.setenv("ALPHARAVIS_DEFER_LARGE_PASTE_RAG_UNTIL_AFTER_COMPRESSION", "false")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_ENABLE_CHUNKED_SUMMARY", "false")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_MAX_TOKENS", "1200")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_MIN_TOKENS", "1000")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_CHARS_PER_TOKEN", "1")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_CHUNK_OVERLAP_CHARS", "0")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_SUMMARY_MAX_CHUNKS", "20")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_OVERSIZED_TAIL_RATIO", "0.60")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_OVERSIZED_TAIL_FORCE_MIDDLE_RATIO", "0.80")

    calls: list[str] = []

    async def chunk_summary(prompt: str, _max_tokens: int) -> str:
        calls.append(prompt)
        if "mode: pre_run_chunked_synthesis" in prompt:
            return "## Active Task\n- synthesized oversized latest user paste\n\n## Archive References\n- source_type=archive"
        assert "mode: pre_run_chunk_" in prompt
        return "## Active Task\n- oversized latest user paste chunk\n\n## Archive References\n- source_type=archive"

    messages = [{"role": "system", "content": f"head {index}"} for index in range(3)]
    messages.extend({"role": "assistant", "content": f"old middle {index}"} for index in range(2))
    messages.append({"role": "user", "content": "latest pasted file " + ("x " * 12000)})

    result = asyncio.run(
        compress_messages(
            messages,
            mode="pre_run",
            thread_id="thread",
            thread_key="thread",
            token_limit=100,
            previous_summary=None,
            summarize_fn=chunk_summary,
            force=True,
            enable_chunked_summary=False,
        )
    )

    assert not result.summary_failed
    assert result.archive_metadata["oversized_tail_force_latest_user_to_middle"] is True
    assert result.archive_metadata["summary_prompt_pruned"] is True
    assert result.archive_metadata["summary_chunking_forced_by_oversized_tail"] is True
    assert result.archive_metadata["summary_chunking_used"] is True
    assert result.archive_metadata["summary_chunk_count"] > 1
    assert len(calls) == result.archive_metadata["summary_chunk_count"] + 1


def test_reasoning_blocks_do_not_count_as_active_context() -> None:
    message = {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "hidden " * 4000},
            {"type": "reasoning", "content": "internal " * 4000},
            {"type": "text", "text": "visible answer"},
        ],
        "usage_metadata": {"total_tokens": 50000},
    }

    assert estimate_tokens_rough([message]) < 20
    assert estimate_message_tokens_rough(message) < 20
    prep = prepare_messages_for_summary([message])
    assert "visible answer" in prep.text
    assert "hidden" not in prep.text
    assert "internal" not in prep.text


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


def test_tail_protection_uses_hermes_style_token_budget_not_fixed_sixteen_messages() -> None:
    old_values = {
        key: os.environ.get(key)
        for key in (
            "ALPHARAVIS_COMPRESSION_PROTECT_LAST_MESSAGES",
            "ALPHARAVIS_COMPRESSION_TAIL_TOKEN_RATIO",
            "ALPHARAVIS_COMPRESSION_TAIL_SOFT_CEILING_RATIO",
        )
    }
    try:
        os.environ.pop("ALPHARAVIS_COMPRESSION_PROTECT_LAST_MESSAGES", None)
        os.environ["ALPHARAVIS_COMPRESSION_TAIL_TOKEN_RATIO"] = "0.10"
        os.environ["ALPHARAVIS_COMPRESSION_TAIL_SOFT_CEILING_RATIO"] = "1.5"
        messages = [{"id": f"m{i}", "role": "user", "content": f"message {i} " + ("x" * 80)} for i in range(30)]

        selection = select_head_middle_tail(messages, token_limit=400, protected_message_ids=set())

        assert len(selection.tail_indexes) < 16
        assert selection.tail_indexes[-3:] == [27, 28, 29]
        assert selection.middle
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_tail_protection_anchors_latest_user_message_like_hermes() -> None:
    old_values = {
        key: os.environ.get(key)
        for key in (
            "ALPHARAVIS_COMPRESSION_PROTECT_LAST_MESSAGES",
            "ALPHARAVIS_COMPRESSION_TAIL_TOKEN_RATIO",
            "ALPHARAVIS_COMPRESSION_TAIL_SOFT_CEILING_RATIO",
            "ALPHARAVIS_COMPRESSION_REBALANCE_OVERSIZED_TAIL",
        )
    }
    try:
        os.environ["ALPHARAVIS_COMPRESSION_PROTECT_LAST_MESSAGES"] = "3"
        os.environ["ALPHARAVIS_COMPRESSION_TAIL_TOKEN_RATIO"] = "0.05"
        os.environ["ALPHARAVIS_COMPRESSION_TAIL_SOFT_CEILING_RATIO"] = "1.0"
        os.environ["ALPHARAVIS_COMPRESSION_REBALANCE_OVERSIZED_TAIL"] = "false"
        messages = [{"id": f"m{i}", "role": "assistant", "content": f"assistant {i} " + ("x" * 80)} for i in range(30)]
        messages[20] = {"id": "latest-user", "role": "user", "content": "current active request"}

        selection = select_head_middle_tail(messages, token_limit=120, protected_message_ids=set())

        assert 20 in selection.tail_indexes
        assert selection.tail_indexes == list(range(20, 30))
        assert all(index not in selection.middle_indexes for index in range(20, 30))
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_oversized_tail_rebalance_moves_old_tail_messages_to_middle_but_keeps_latest_user(monkeypatch) -> None:
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_REBALANCE_OVERSIZED_TAIL", "true")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_OVERSIZED_TAIL_RATIO", "0.60")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_OVERSIZED_TAIL_FORCE_MIDDLE_RATIO", "0.99")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_PROTECT_LAST_MESSAGES", "3")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_TAIL_TOKEN_RATIO", "0.05")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_TAIL_SOFT_CEILING_RATIO", "1.0")
    messages = [{"id": f"m{i}", "role": "assistant", "content": f"assistant {i} " + ("x " * 60)} for i in range(30)]
    messages[20] = {"id": "latest-user", "role": "user", "content": "current active request"}

    selection = select_head_middle_tail(messages, token_limit=500, protected_message_ids=set())

    assert selection.oversized_tail_rebalanced is True
    assert selection.oversized_tail_tokens_before > selection.oversized_tail_token_target
    assert selection.oversized_tail_moved_indexes
    assert 20 in selection.tail_indexes
    assert any(index in selection.middle_indexes for index in range(21, 30))


def test_critical_oversized_tail_can_move_latest_user_to_middle(monkeypatch) -> None:
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_REBALANCE_OVERSIZED_TAIL", "true")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_OVERSIZED_TAIL_RATIO", "0.60")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_OVERSIZED_TAIL_FORCE_MIDDLE_RATIO", "0.80")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_PROTECT_LAST_MESSAGES", "3")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_TAIL_TOKEN_RATIO", "0.05")
    monkeypatch.setenv("ALPHARAVIS_COMPRESSION_TAIL_SOFT_CEILING_RATIO", "1.0")
    messages = [{"id": f"m{i}", "role": "assistant", "content": f"assistant {i}"} for i in range(6)]
    messages.append({"id": "huge-latest-user", "role": "user", "content": "huge active request " + ("x " * 500)})

    selection = select_head_middle_tail(messages, token_limit=100, protected_message_ids=set())

    assert selection.oversized_tail_rebalanced is True
    assert selection.oversized_tail_force_latest_user_to_middle is True
    assert selection.oversized_tail_force_middle_target == 80
    assert 6 in selection.middle_indexes
    assert 6 not in selection.tail_indexes


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
    assert "source_type: archive" in content
    assert 'read_archive_record(archive_key="archive_1")' in content


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
        test_tail_protection_uses_hermes_style_token_budget_not_fixed_sixteen_messages,
        test_tail_protection_anchors_latest_user_message_like_hermes,
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
