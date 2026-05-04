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
    classify_tool_kind,
    extract_prompt_text,
    extract_locations,
    JsonRpcError,
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


async def _slow_then_text_stream(
    _session: AcpSessionState,
    _prompt: str,
    _command: dict[str, Any] | None,
) -> AsyncIterator[Any]:
    await asyncio.sleep(0.03)
    yield {"event": "messages", "data": (_FakeMessage("ai", "Spat"), {})}


async def _interruptible_stream(
    _session: AcpSessionState,
    prompt: str,
    _command: dict[str, Any] | None,
) -> AsyncIterator[Any]:
    if "first" in prompt:
        await asyncio.sleep(30)
        yield {"event": "messages", "data": (_FakeMessage("ai", "first answer"), {})}
        return
    yield {"event": "messages", "data": (_FakeMessage("ai", "second answer"), {})}


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


async def _diff_tool_stream(_session: AcpSessionState, _prompt: str, _command: dict[str, Any] | None) -> AsyncIterator[Any]:
    yield {
        "event": "messages",
        "data": (
            _FakeMessage(
                "ai",
                "",
                tool_calls=[
                    {
                        "id": "call_diff",
                        "name": "apply_patch",
                        "args": {"path": "src/app.py"},
                    }
                ],
            ),
            {},
        ),
    }
    yield {
        "event": "messages",
        "data": (
            _FakeMessage(
                "tool",
                "--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-old\n+new",
                tool_call_id="call_diff",
                name="apply_patch",
            ),
            {},
        ),
    }


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
            "debug_io": False,
            "allow_file_writes": False,
            "send_available_commands": True,
            "stream_heartbeat_seconds": 1.0,
            "debug_event_payload_chars": 12000,
            "debug_status_to_aion": True,
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


def test_tool_kind_classification_and_locations() -> None:
    assert classify_tool_kind("read_file", {"path": "a.py"}) == "read"
    assert classify_tool_kind("apply_patch", {"path": "a.py"}) == "edit"
    assert classify_tool_kind("execute_ssh_command", {"command": "docker ps"}) == "execute"
    assert extract_locations({"path": "a.py", "files": ["b.py", "c.py"]}) == [
        {"path": "a.py"},
        {"path": "b.py"},
        {"path": "c.py"},
    ]


def test_diff_output_mapping_and_edit_locations() -> None:
    adapter, messages = _adapter_with_collector(_diff_tool_stream)
    adapter.new_session({"sessionId": "aion-diff"})

    asyncio.run(adapter.prompt({"sessionId": "aion-diff", "prompt": [{"type": "text", "text": "Patch"}]}))

    updates = [m["params"]["update"] for m in messages if m.get("method") == "session/update"]
    tool_call = next(update for update in updates if update["sessionUpdate"] == "tool_call")
    assert tool_call["kind"] == "edit"
    assert tool_call["locations"] == [{"path": "src/app.py"}]
    tool_update = next(update for update in updates if update["sessionUpdate"] == "tool_call_update")
    text = tool_update["content"][0]["content"]["text"]
    assert text.startswith("```diff")
    assert "+++ b/src/app.py" in text


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
    assert permission_result_to_resume({"outcome": {"optionId": "allow_always"}}) == {"action": "approve"}
    assert permission_result_to_resume({"outcome": {"optionId": "reject_once"}}) == {"action": "reject"}
    assert permission_result_to_resume({"outcome": {"optionId": "reject_always"}}) == {"action": "reject"}


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


def test_permission_response_method_variant() -> None:
    adapter, _messages = _adapter_with_collector(_text_stream)

    async def resolve_with_method() -> Any:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        adapter._pending_responses[77] = future
        await adapter.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/permission_response",
                "params": {"requestId": 77, "outcome": {"optionId": "reject_once"}},
            }
        )
        return await future

    result = asyncio.run(resolve_with_method())

    assert result["outcome"]["optionId"] == "reject_once"


def test_secret_scrubbing_and_output_truncation() -> None:
    text = "api_key=sk-test-secret-value Bearer abcdefghijklmnopqrstuvwxyz"
    assert "secret-value" not in scrub_text(text)
    update = build_tool_call_update("s", "call", status="completed", text="x" * 200, max_chars=50)
    visible = update["params"]["update"]["content"][0]["content"]["text"]
    assert len(visible) < 140
    assert "output truncated" in visible


def test_fs_read_text_file_inside_workspace(tmp_path: Path) -> None:
    adapter, _messages = _adapter_with_collector(_text_stream)
    adapter.config["workspace"] = str(tmp_path)
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")

    result = adapter.read_text_file({"path": "note.txt"})

    assert result == {"content": "hello"}


def test_fs_read_text_file_outside_workspace_blocked(tmp_path: Path) -> None:
    adapter, _messages = _adapter_with_collector(_text_stream)
    adapter.config["workspace"] = str(tmp_path / "workspace")
    (tmp_path / "workspace").mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("nope", encoding="utf-8")

    try:
        adapter.read_text_file({"path": str(outside)})
    except JsonRpcError as exc:
        assert exc.code == -32040
    else:
        raise AssertionError("outside workspace read was not blocked")


def test_fs_write_text_file_default_blocked(tmp_path: Path) -> None:
    adapter, _messages = _adapter_with_collector(_text_stream)
    adapter.config["workspace"] = str(tmp_path)

    try:
        adapter.write_text_file({"path": "new.txt", "content": "hello"})
    except JsonRpcError as exc:
        assert exc.code == -32042
    else:
        raise AssertionError("write should be disabled by default")


def test_debug_io_logs_to_stderr_not_stdout(capsys) -> None:
    adapter, messages = _adapter_with_collector(_text_stream)
    adapter.config["debug_io"] = True

    asyncio.run(adapter.send_json({"jsonrpc": "2.0", "result": {"api_key": "sk-secret-value"}}))

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[alpharavis-acp:out]" in captured.err
    assert "secret-value" not in captured.err
    assert messages


def test_debug_trace_logs_request_timings_to_stderr(capsys) -> None:
    adapter, messages = _adapter_with_collector(_text_stream)
    adapter.config["debug_io"] = True
    adapter.config["trace_detail"] = "debug"

    async def send_initialize() -> None:
        await adapter.handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        for _ in range(50):
            if any(message.get("id") == 1 for message in messages):
                return
            await asyncio.sleep(0.01)

    asyncio.run(send_initialize())

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[alpharavis-acp:in]" in captured.err
    assert '"event":"request.start"' in captured.err
    assert '"event":"request.end"' in captured.err


def test_stream_heartbeat_does_not_cancel_slow_first_event() -> None:
    adapter, messages = _adapter_with_collector(_slow_then_text_stream)
    adapter.config["trace_detail"] = "debug"
    adapter.config["stream_heartbeat_seconds"] = 0.01
    adapter.new_session({"sessionId": "aion-slow"})

    result = asyncio.run(adapter.prompt({"sessionId": "aion-slow", "prompt": [{"type": "text", "text": "Wait"}]}))

    assert result["stopReason"] == "end_turn"
    updates = [m["params"]["update"] for m in messages if m.get("method") == "session/update"]
    assert any(
        update["sessionUpdate"] == "agent_thought_chunk" and "ACP wartet weiter" in update["content"]["text"]
        for update in updates
    )
    assert any(
        update["sessionUpdate"] == "agent_message_chunk" and "Spat" in update["content"]["text"]
        for update in updates
    )


def test_second_prompt_interrupts_previous_prompt() -> None:
    adapter, messages = _adapter_with_collector(_interruptible_stream)
    adapter.new_session({"sessionId": "aion-repeat"})

    async def run_two_prompts() -> list[dict[str, Any]]:
        await adapter.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/prompt",
                "params": {"sessionId": "aion-repeat", "prompt": [{"type": "text", "text": "first"}]},
            }
        )
        await asyncio.sleep(0.02)
        await adapter.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/prompt",
                "params": {"sessionId": "aion-repeat", "prompt": [{"type": "text", "text": "second"}]},
            }
        )
        for _ in range(200):
            response_ids = {message.get("id") for message in messages if "result" in message}
            if {1, 2}.issubset(response_ids):
                return messages
            await asyncio.sleep(0.01)
        return messages

    asyncio.run(run_two_prompts())

    responses = {m["id"]: m["result"] for m in messages if "result" in m}
    assert responses[1]["stopReason"] == "cancelled"
    assert responses[2]["stopReason"] == "end_turn"
    updates = [m["params"]["update"] for m in messages if m.get("method") == "session/update"]
    assert any(
        update["sessionUpdate"] == "agent_message_chunk" and "second answer" in update["content"]["text"]
        for update in updates
    )


def test_extract_prompt_text_handles_attachments_as_metadata() -> None:
    prompt = [
        {"type": "text", "text": "Bitte prüfen"},
        {"type": "image", "uri": "file://image.png", "mimeType": "image/png"},
    ]

    text = extract_prompt_text(prompt)

    assert "Bitte prüfen" in text
    assert "ACP attachment metadata" in text
