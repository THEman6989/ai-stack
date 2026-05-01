from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

from alpharavis_acp_adapter import (  # noqa: E402
    AcpSessionState,
    AlphaRavisAcpAdapter,
    build_permission_request,
    build_tool_call_update,
    extract_prompt_text,
    permission_result_to_resume,
    scrub_text,
)


@dataclass
class _FakeMessage:
    type: str
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


async def _text_stream(_session: AcpSessionState, _prompt: str, _command: dict[str, Any] | None) -> AsyncIterator[Any]:
    yield {"event": "updates", "data": {"planner": {"plan": [{"content": "Run AlphaRavis", "status": "in_progress"}]}}}
    yield {"event": "messages", "data": (_FakeMessage("ai", "Hallo"), {})}
    yield {"event": "messages", "data": (_FakeMessage("ai", "Hallo Welt"), {})}


async def _tool_stream(_session: AcpSessionState, _prompt: str, _command: dict[str, Any] | None) -> AsyncIterator[Any]:
    yield {
        "event": "messages",
        "data": (
            _FakeMessage(
                "ai",
                "",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "execute_ssh_command",
                        "args": {"command": "docker ps"},
                    }
                ],
            ),
            {},
        ),
    }
    yield {
        "event": "messages",
        "data": (_FakeMessage("tool", "exit 0\ncontainer up", tool_call_id="call_1", name="execute_ssh_command"), {}),
    }
    yield {"event": "messages", "data": (_FakeMessage("ai", "Fertig"), {})}


async def _permission_stream(
    _session: AcpSessionState,
    _prompt: str,
    command: dict[str, Any] | None,
) -> AsyncIterator[Any]:
    if command:
        yield {"event": "messages", "data": (_FakeMessage("ai", "Genehmigt weitergemacht"), {})}
        return
    yield {
        "event": "updates",
        "data": {
            "__interrupt__": [
                {
                    "value": {
                        "type": "command_approval",
                        "command": "shutdown now",
                        "target": "llama_server",
                        "risk": "destructive",
                    }
                }
            ]
        },
    }


def _adapter_with_collector(stream_factory):
    messages: list[dict[str, Any]] = []

    async def writer(payload: dict[str, Any]) -> None:
        messages.append(payload)

    adapter = AlphaRavisAcpAdapter(
        writer=writer,
        langgraph_stream=stream_factory,
        adapter_config={
            "langgraph_api_url": "http://langgraph-api:2024",
            "assistant_id": "alpha_ravis",
            "workspace": "/workspace",
            "tool_output_max_chars": 120,
            "trace_detail": "summary",
            "scrub_internal_context": True,
            "run_timeout_seconds": 10,
        },
    )
    return adapter, messages


def test_initialize_is_aionui_compatible() -> None:
    adapter, _messages = _adapter_with_collector(_text_stream)

    result = adapter.initialize({"protocolVersion": 1})

    assert result["protocolVersion"] == 1
    assert result["serverInfo"]["name"] == "alpharavis-langgraph"
    assert result["agentInfo"]["title"] == "AlphaRavis LangGraph"
    assert result["agentCapabilities"]["promptCapabilities"]["embeddedContext"] is True


def test_session_new_creates_stable_thread() -> None:
    adapter, _messages = _adapter_with_collector(_text_stream)

    result = adapter.new_session({"sessionId": "aion-1", "cwd": "C:/repo"})

    assert result["sessionId"] == "aion-1"
    assert "aion-1" in adapter.sessions
    assert adapter.sessions["aion-1"].thread_id == adapter.sessions["aion-1"].thread_id
    assert result["models"]["currentModelId"] == "alpha_ravis"


def test_prompt_streams_text_and_plan_events() -> None:
    adapter, messages = _adapter_with_collector(_text_stream)
    adapter.new_session({"sessionId": "aion-text"})

    result = asyncio.run(adapter.prompt({"sessionId": "aion-text", "prompt": [{"type": "text", "text": "Hi"}]}))

    assert result["stopReason"] == "end_turn"
    updates = [m["params"]["update"] for m in messages if m.get("method") == "session/update"]
    assert any(update["sessionUpdate"] == "plan" for update in updates)
    assert any(update["sessionUpdate"] == "agent_message_chunk" and update["content"]["text"] == "Hallo" for update in updates)
    assert any(update["sessionUpdate"] == "agent_message_chunk" and update["content"]["text"] == " Welt" for update in updates)


def test_tool_calls_are_toolcards_not_thoughts() -> None:
    adapter, messages = _adapter_with_collector(_tool_stream)
    adapter.new_session({"sessionId": "aion-tool"})

    asyncio.run(adapter.prompt({"sessionId": "aion-tool", "prompt": [{"type": "text", "text": "Check"}]}))

    updates = [m["params"]["update"] for m in messages if m.get("method") == "session/update"]
    assert any(update["sessionUpdate"] == "tool_call" and update["toolCallId"] == "call_1" for update in updates)
    assert any(update["sessionUpdate"] == "tool_call_update" and update["toolCallId"] == "call_1" for update in updates)
    thought_text = "\n".join(
        update.get("content", {}).get("text", "")
        for update in updates
        if update["sessionUpdate"] == "agent_thought_chunk"
    )
    assert "container up" not in thought_text


def test_permission_request_and_resume_mapping() -> None:
    request = build_permission_request(
        123,
        "aion-perm",
        tool_call_id="call_perm",
        title="Befehl benötigt Freigabe",
        command="shutdown now",
        description="destructive",
        target="llama_server",
    )

    assert request["method"] == "session/request_permission"
    assert request["params"]["options"][0]["optionId"] == "allow_once"
    assert permission_result_to_resume({"outcome": {"optionId": "allow_once"}}) == {"action": "approve"}
    assert permission_result_to_resume({"outcome": {"optionId": "reject_once"}}) == {"action": "reject"}


def test_permission_interrupt_roundtrip() -> None:
    adapter, messages = _adapter_with_collector(_permission_stream)
    adapter.new_session({"sessionId": "aion-perm"})

    async def run_prompt_and_reply() -> dict[str, Any]:
        prompt_task = asyncio.create_task(
            adapter.prompt({"sessionId": "aion-perm", "prompt": [{"type": "text", "text": "Run command"}]})
        )
        for _ in range(100):
            permission = next((m for m in messages if m.get("method") == "session/request_permission"), None)
            if permission:
                await adapter.handle_message(
                    {
                        "jsonrpc": "2.0",
                        "id": permission["id"],
                        "result": {"outcome": {"optionId": "allow_once", "outcome": "selected"}},
                    }
                )
                break
            await asyncio.sleep(0.01)
        return await prompt_task

    result = asyncio.run(run_prompt_and_reply())

    assert result["stopReason"] == "end_turn"
    assert any(m.get("method") == "session/request_permission" for m in messages)
    assert any(
        m.get("params", {}).get("update", {}).get("sessionUpdate") == "agent_message_chunk"
        and "Genehmigt" in m["params"]["update"]["content"]["text"]
        for m in messages
    )


def test_secret_scrubbing_and_output_truncation() -> None:
    text = "api_key=sk-test-secret-value Bearer abcdefghijklmnopqrstuvwxyz"
    assert "secret-value" not in scrub_text(text)
    update = build_tool_call_update("s", "call", status="completed", text="x" * 200, max_chars=50)
    visible = update["params"]["update"]["content"][0]["content"]["text"]
    assert len(visible) < 140
    assert "output truncated" in visible


def test_extract_prompt_text_handles_attachments_as_metadata() -> None:
    prompt = [
        {"type": "text", "text": "Bitte prüfen"},
        {"type": "image", "uri": "file://image.png", "mimeType": "image/png"},
    ]

    text = extract_prompt_text(prompt)

    assert "Bitte prüfen" in text
    assert "ACP attachment metadata" in text
