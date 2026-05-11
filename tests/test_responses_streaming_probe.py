from __future__ import annotations

import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import probe_responses_tool_streaming as probe  # noqa: E402

import importlib.util  # noqa: E402


PATCH_PATH = ROOT / "langgraph-app" / "patches" / "patch_langchain_openai_responses_tool_streaming.py"
PATCH_SPEC = importlib.util.spec_from_file_location("patch_responses_tool_streaming", PATCH_PATH)
patch_responses_tool_streaming = importlib.util.module_from_spec(PATCH_SPEC)
assert PATCH_SPEC and PATCH_SPEC.loader
PATCH_SPEC.loader.exec_module(patch_responses_tool_streaming)


def test_low_level_payload_includes_streaming_tool_schema() -> None:
    payload = probe.low_level_payload("big-boss", "Call the tool.", force_tool_choice=True)

    assert payload["stream"] is True
    assert payload["tools"][0]["type"] == "function"
    assert payload["tools"][0]["name"] == "marker_tool"
    assert payload["tool_choice"] == {"type": "function", "name": "marker_tool"}


def test_to_jsonable_serializes_model_like_objects() -> None:
    class Obj:
        def model_dump(self):
            return {"text": "x" * 13000}

    data = probe.to_jsonable({"obj": Obj()})

    assert data["obj"]["text"].startswith("x")
    assert "truncated" in data["obj"]["text"]


def test_content_text_chars_counts_responses_text_blocks() -> None:
    content = [{"type": "text", "text": "NO"}, {"type": "text", "text": "_TOOL"}]

    assert probe.content_text_chars(content) == len("NO_TOOL")


def test_experimental_patch_source_updates_conversion_sites() -> None:
    source = (
        patch_responses_tool_streaming.FUNCTION_CALL_NEEDLE
        + "\n"
        + patch_responses_tool_streaming.REASONING_DONE_NEEDLE
        + "\n"
        + patch_responses_tool_streaming.FUNCTION_CALL_DELTA_NEEDLE
        + "\n"
        + patch_responses_tool_streaming.FUNCTION_CALL_DONE_NEEDLE
    )

    patched = patch_responses_tool_streaming.patch_source(source)

    assert patch_responses_tool_streaming.FUNCTION_CALL_PATCH_MARKER in patched
    assert patch_responses_tool_streaming.REASONING_DONE_PATCH_MARKER in patched
    assert patch_responses_tool_streaming.FUNCTION_CALL_DELTA_PATCH_MARKER in patched
    assert patch_responses_tool_streaming.FUNCTION_CALL_DONE_PATCH_MARKER in patched
    assert patch_responses_tool_streaming.patch_source(patched) == patched


def test_experimental_patch_source_upgrades_old_function_call_patch() -> None:
    source = (
        patch_responses_tool_streaming.OLD_FUNCTION_CALL_REPLACEMENT
        + "\n"
        + patch_responses_tool_streaming.REASONING_DONE_NEEDLE
        + "\n"
        + patch_responses_tool_streaming.FUNCTION_CALL_DELTA_NEEDLE
        + "\n"
        + patch_responses_tool_streaming.FUNCTION_CALL_DONE_NEEDLE
    )

    patched = patch_responses_tool_streaming.patch_source(source)

    assert patch_responses_tool_streaming.FUNCTION_CALL_PATCH_MARKER in patched
    assert f"        {patch_responses_tool_streaming.OLD_FUNCTION_CALL_PATCH_MARKER}\n" not in patched
    assert patch_responses_tool_streaming.patch_source(patched) == patched


def test_classify_low_level_failure_keeps_hybrid_recommendation() -> None:
    result = probe.ProbeResult("low_level_responses_sse", ok=False, error="HTTP 400")

    classified = probe.classify_probe_results([result])

    assert classified["bucket"] == "provider_litellm_or_openai_sdk"
    assert "Keep hybrid mode" in classified["recommendation"]


def test_classify_invalid_tool_calls() -> None:
    results = [
        probe.ProbeResult("low_level_responses_sse", ok=True, details={"function_call_items": 1}),
        probe.ProbeResult("langchain_no_tool_astream", ok=True),
        probe.ProbeResult("langchain_react_agent_astream_events", ok=False, error="invalid_tool_calls=1"),
    ]

    classified = probe.classify_probe_results(results)

    assert classified["bucket"] == "ai_message_chunk_aggregation"
    assert any("raw Responses stream" in item for item in classified["evidence"])


def test_classify_low_level_no_tool_does_not_mask_agent_success() -> None:
    results = [
        probe.ProbeResult(
            "low_level_responses_sse",
            ok=True,
            details={"function_call_items": 0, "function_arg_delta_chars": 0},
        ),
        probe.ProbeResult("langchain_no_tool_astream", ok=True),
        probe.ProbeResult(
            "langchain_react_agent_astream_events",
            ok=True,
            details={"tool_call_chunks": 1, "marker_tool_ends": 1, "invalid_tool_calls": 0},
        ),
    ]

    classified = probe.classify_probe_results(results)

    assert classified["bucket"] == "not_reproduced"
    assert "raw Responses stream completed without function-call events" in classified["evidence"]


def test_classify_escaped_empty_content_error_as_conversion() -> None:
    results = [
        probe.ProbeResult("low_level_responses_sse", ok=True, details={"function_call_items": 1}),
        probe.ProbeResult("langchain_no_tool_astream", ok=True),
        probe.ProbeResult(
            "langchain_react_agent_astream_events",
            ok=False,
            error='BadRequestError: item[\\\'content\\\'] is empty',
        ),
    ]

    classified = probe.classify_probe_results(results)

    assert classified["bucket"] == "langchain_openai_conversion"


def test_iter_sse_response_records_semantic_event(tmp_path: Path) -> None:
    class Response:
        async def aiter_lines(self):
            lines = [
                "event: response.output_text.delta",
                'data: {"type":"response.output_text.delta","delta":"Hi"}',
                "",
                "data: [DONE]",
                "",
            ]
            for line in lines:
                yield line

    writer = probe.JsonlWriter(tmp_path / "events.jsonl")
    try:
        records = asyncio.run(_collect_sse(Response(), writer))
    finally:
        writer.close()

    assert records[0]["event"] == "response.output_text.delta"
    assert records[0]["data"]["delta"] == "Hi"
    assert records[1]["raw_data"] == "[DONE]"


async def _collect_sse(response, writer):
    return [record async for record in probe.iter_sse_response(response, writer)]
