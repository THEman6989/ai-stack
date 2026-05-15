from __future__ import annotations

import asyncio
import json
import importlib.util
from importlib.machinery import ModuleSpec
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

if importlib.util.find_spec("fastapi") is None:
    fastapi_stub = types.ModuleType("fastapi")
    fastapi_stub.__spec__ = ModuleSpec("fastapi", loader=None)

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str = "") -> None:
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

    class Request:
        headers: dict[str, str] = {}

    class FastAPI:
        openapi_version = "3.1.0"

        def __init__(self, *args, openapi_version: str = "3.1.0", **kwargs) -> None:
            self.openapi_version = openapi_version

        def get(self, *args, **kwargs):
            return lambda fn: fn

        def post(self, *args, **kwargs):
            return lambda fn: fn

        def delete(self, *args, **kwargs):
            return lambda fn: fn

        def middleware(self, *args, **kwargs):
            return lambda fn: fn

        def mount(self, *args, **kwargs) -> None:
            return None

    fastapi_stub.FastAPI = FastAPI
    fastapi_stub.HTTPException = HTTPException
    fastapi_stub.Request = Request
    sys.modules["fastapi"] = fastapi_stub

    responses_stub = types.ModuleType("fastapi.responses")
    responses_stub.__spec__ = ModuleSpec("fastapi.responses", loader=None)

    class JSONResponse(dict):
        def __init__(self, content=None, status_code: int = 200, *args, **kwargs) -> None:
            super().__init__(content or {})
            self.status_code = status_code

    class StreamingResponse:
        def __init__(self, content, media_type: str = "") -> None:
            self.content = content
            self.media_type = media_type

    class HTMLResponse(str):
        def __new__(cls, content: str = "", *args, **kwargs):
            return str.__new__(cls, content)

    responses_stub.JSONResponse = JSONResponse
    responses_stub.StreamingResponse = StreamingResponse
    responses_stub.HTMLResponse = HTMLResponse
    sys.modules["fastapi.responses"] = responses_stub

    staticfiles_stub = types.ModuleType("fastapi.staticfiles")
    staticfiles_stub.__spec__ = ModuleSpec("fastapi.staticfiles", loader=None)

    class StaticFiles:
        def __init__(self, *args, **kwargs) -> None:
            pass

    staticfiles_stub.StaticFiles = StaticFiles
    sys.modules["fastapi.staticfiles"] = staticfiles_stub

if importlib.util.find_spec("langgraph_sdk") is None:
    sdk_stub = types.ModuleType("langgraph_sdk")
    sdk_stub.__spec__ = ModuleSpec("langgraph_sdk", loader=None)
    sdk_stub.get_client = lambda *args, **kwargs: None
    sys.modules["langgraph_sdk"] = sdk_stub

import bridge_server  # noqa: E402


class _StubRequest:
    headers: dict[str, str] = {}

    def __init__(self, body: dict) -> None:
        self._body = body

    async def json(self) -> dict:
        return self._body


class _FakeMessage:
    def __init__(
        self,
        message_type: str,
        content: str,
        *,
        reasoning_content: str = "",
        tool_calls: list[dict] | None = None,
        tool_call_id: str = "",
        name: str = "",
    ) -> None:
        self.type = message_type
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id
        self.name = name


class _FakeThreads:
    def __init__(self, state: dict | None = None) -> None:
        self.state = state or {"values": {"messages": []}}

    async def create(self, *args, **kwargs) -> dict:
        return {}

    async def get_state(self, *args, **kwargs) -> dict:
        return self.state


class _FakeRuns:
    def __init__(self, parts: list[dict], *, wait_state: dict | None = None) -> None:
        self.parts = parts
        self.wait_state = wait_state or {"values": {"messages": []}}
        self.last_stream_kwargs: dict | None = None

    async def stream(self, *args, **kwargs):
        self.last_stream_kwargs = dict(kwargs)
        for part in self.parts:
            yield part

    async def wait(self, *args, **kwargs) -> dict:
        return self.wait_state


class _FakeClient:
    def __init__(
        self,
        parts: list[dict],
        *,
        state: dict | None = None,
        wait_state: dict | None = None,
    ) -> None:
        self.threads = _FakeThreads(state)
        self.runs = _FakeRuns(parts, wait_state=wait_state)


def _parse_sse_events(chunks: list[str]) -> list[dict]:
    events: list[dict] = []
    for raw in chunks:
        if raw.strip() == "data: [DONE]" or not raw.startswith("event: "):
            continue
        event = ""
        data = None
        for line in raw.splitlines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = json.loads(line.removeprefix("data: "))
        if event and data:
            events.append({"event": event, "data": data})
    return events


def _parse_chat_chunks(chunks: list[str]) -> list[dict]:
    parsed: list[dict] = []
    for raw in chunks:
        if raw.strip() == "data: [DONE]" or not raw.startswith("data: "):
            continue
        parsed.append(json.loads(raw.split("data: ", 1)[1]))
    return parsed


async def _collect_response_stream(body: dict, parts: list[dict]) -> list[dict]:
    chunks = [chunk async for chunk in bridge_server._stream_responses(body, _StubRequest(body))]
    return _parse_sse_events(chunks)


async def _collect_chat_event_stream(parts: list[dict]) -> list[dict]:
    chunks = [
        chunk
        async for chunk in bridge_server._stream_chat_events(
            _FakeClient(parts),
            "thread_test",
            {"input": {"messages": [{"role": "human", "content": "Hi"}]}},
            "my-agent",
        )
    ]
    return _parse_chat_chunks(chunks)


def _patch_stream_env(monkeypatch, parts: list[dict]) -> None:
    monkeypatch.setattr(bridge_server, "_client", lambda: _FakeClient(parts))
    monkeypatch.setattr(bridge_server, "BRIDGE_STREAM_REASONING_EVENTS", True)
    monkeypatch.setattr(bridge_server, "BRIDGE_RESPONSES_STREAM_ACTIVITY_EVENTS", True)
    monkeypatch.setattr(bridge_server, "BRIDGE_RESPONSES_STREAM_TOOL_EVENTS", True)
    monkeypatch.setattr(bridge_server, "BRIDGE_RESPONSES_DONE_SENTINEL", False)


def test_responses_input_supports_instructions_and_content_parts() -> None:
    body = {
        "instructions": "Du bist AlphaRavis.",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Analysiere das bitte."},
                    {"type": "input_image", "image_url": {"url": "https://example.test/a.png"}},
                    {"type": "input_video", "video_url": {"url": "https://example.test/v.mp4"}},
                ],
            }
        ],
    }

    messages = bridge_server._responses_input_to_messages(body)

    assert messages[0] == {"role": "system", "content": "Du bist AlphaRavis."}
    assert messages[1]["role"] == "user"
    assert "Analysiere das bitte." in messages[1]["content"]
    assert "Media attachment withheld" in messages[1]["content"]
    assert "https://example.test/v.mp4" in messages[1]["content"]


def test_bridge_mirrors_chat_video_parts_to_media_gallery(monkeypatch) -> None:
    requests: list[dict] = []

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json() -> dict:
            return {
                "asset_id": "asset_video",
                "public_url": "http://localhost:8130/media/chat/clip.mp4",
                "download_error": "",
            }

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict) -> FakeResponse:
            requests.append({"url": url, "json": json})
            return FakeResponse()

    monkeypatch.setattr(bridge_server.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(bridge_server, "BRIDGE_MEDIA_GALLERY_AUTO_REGISTER_VIDEOS", True)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_video",
                    "file_id": "file_clip",
                    "filename": "clip.mp4",
                    "video_url": {"url": "http://librechat:3080/api/files/download/user/file_clip"},
                }
            ],
        }
    ]

    asyncio.run(
        bridge_server._mirror_video_parts_in_messages(
            messages,
            thread_id="thread_video",
            thread_key="conversation_video",
        )
    )

    part = messages[0]["content"][0]
    assert requests[0]["url"].endswith("/assets/register")
    assert requests[0]["json"]["source_key"] == "librechat:file_clip"
    assert requests[0]["json"]["origin"] == "librechat_upload"
    assert part["video_url"]["url"] == "http://localhost:8130/media/chat/clip.mp4"
    assert part["alpharavis_original_media_url"].startswith("http://librechat:3080/")


def test_responses_body_uses_gallery_url_after_video_mirroring(monkeypatch) -> None:
    class FakeResponse:
        status_code = 200

        @staticmethod
        def json() -> dict:
            return {
                "asset_id": "asset_response_video",
                "public_url": "http://localhost:8130/media/responses/clip.mp4",
                "download_error": "",
            }

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(bridge_server.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(bridge_server, "BRIDGE_MEDIA_GALLERY_AUTO_REGISTER_VIDEOS", True)
    body = {
        "conversationId": "conversation_response_video",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_video",
                        "file_id": "file_response_clip",
                        "video_url": {"url": "data:video/mp4;base64,QUJD"},
                    }
                ],
            }
        ],
    }

    asyncio.run(bridge_server._mirror_video_parts_in_responses_body(body, _StubRequest(body)))
    messages = bridge_server._responses_input_to_messages(body)

    assert "http://localhost:8130/media/responses/clip.mp4" in messages[0]["content"]
    assert "data:video/mp4;base64,QUJD" not in messages[0]["content"]


def test_response_object_has_stable_ids_and_usage() -> None:
    response = bridge_server._response_object(
        "Hallo",
        "my-agent",
        "resp_test",
        item_id="msg_test",
        body={"store": True, "metadata": {"thread": "x"}},
        messages=[{"role": "user", "content": "Hi"}],
    )

    assert response["id"] == "resp_test"
    assert response["object"] == "response"
    assert response["output"][0]["id"] == "msg_test"
    assert response["output"][0]["content"][0]["type"] == "output_text"
    assert response["usage"]["input_tokens_details"] == {"cached_tokens": 0}
    assert response["usage"]["total_tokens"] >= response["usage"]["output_tokens"]
    assert response["metadata"] == {"thread": "x"}


def test_user_field_does_not_become_thread_key_by_default(monkeypatch) -> None:
    monkeypatch.setattr(bridge_server, "BRIDGE_ALLOW_USER_THREAD_KEY", False)
    body = {"user": "same_librechat_user", "messages": [{"role": "user", "content": "Hi"}]}

    first = bridge_server._extract_thread_key(body, _StubRequest(body))
    second = bridge_server._extract_thread_key(body, _StubRequest(body))

    assert first.startswith("ephemeral:same_librechat_user:")
    assert second.startswith("ephemeral:same_librechat_user:")
    assert first != second


def test_explicit_conversation_id_stays_thread_key(monkeypatch) -> None:
    monkeypatch.setattr(bridge_server, "BRIDGE_ALLOW_USER_THREAD_KEY", False)
    body = {
        "user": "same_librechat_user",
        "conversationId": "chat_123",
        "messages": [{"role": "user", "content": "Hi"}],
    }

    assert bridge_server._extract_thread_key(body, _StubRequest(body)) == "chat_123"


def test_response_store_honors_store_flag() -> None:
    bridge_server._RESPONSES_STORE.clear()
    stored = bridge_server._response_object("stored", "my-agent", "resp_store", body={"store": True})
    skipped = bridge_server._response_object("skip", "my-agent", "resp_skip", body={"store": False})

    bridge_server._store_response_object(stored, {"store": True})
    bridge_server._store_response_object(skipped, {"store": False})

    assert "resp_store" in bridge_server._RESPONSES_STORE
    assert "resp_skip" not in bridge_server._RESPONSES_STORE


def test_run_wait_content_reads_nested_langgraph_values_state() -> None:
    client = _FakeClient(
        [],
        wait_state={"values": {"messages": [_FakeMessage("ai", "RESPONSES_AGENT_OK")]}},
    )

    content = asyncio.run(
        bridge_server._run_wait_content(
            client,
            "thread_nested_values",
            {"input": {"messages": [{"role": "user", "content": "Hi"}]}},
        )
    )

    assert content == "RESPONSES_AGENT_OK"


def test_previous_response_id_adds_stored_output_context() -> None:
    bridge_server._RESPONSES_STORE.clear()
    previous = bridge_server._response_object(
        "Vorherige Antwort",
        "my-agent",
        "resp_prev",
        body={"store": True},
    )
    bridge_server._store_response_object(previous, {"store": True})

    messages = bridge_server._responses_messages_for_body(
        {
            "instructions": "Du bist AlphaRavis.",
            "previous_response_id": "resp_prev",
            "input": "Mach weiter.",
        }
    )

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "assistant"
    assert "Vorherige Antwort" in messages[1]["content"]
    assert messages[2] == {"role": "user", "content": "Mach weiter."}


def test_responses_validation_rejects_unsupported_hosted_features() -> None:
    bridge_server._RESPONSES_STORE.clear()

    background = bridge_server._validate_responses_request({"background": True})
    tools = bridge_server._validate_responses_request({"tools": [{"type": "web_search_preview"}]})
    structured = bridge_server._validate_responses_request({"text": {"format": {"type": "json_schema"}}})
    missing_previous = bridge_server._validate_responses_request({"previous_response_id": "resp_missing"})

    assert background and background["error"]["code"] == "background_not_supported"
    assert tools and tools["error"]["code"] == "client_tools_not_supported"
    assert structured and structured["error"]["code"] == "text_format_not_supported"
    assert missing_previous and missing_previous["error"]["code"] == "previous_response_not_found"


def test_approval_resume_supports_thread_command_memory() -> None:
    resume = bridge_server._approval_resume_from_messages([{"role": "user", "content": "immer erlauben"}])

    assert resume == {"action": "approve", "remember": "thread_command"}


def test_prepare_run_payload_remembers_exact_command_approval() -> None:
    bridge_server._APPROVAL_MEMORY.clear()
    interrupt = {
        "type": "command_approval",
        "scope": "local",
        "target": "langgraph-api",
        "command": "docker compose up -d api-bridge",
        "risk": "state-changing",
    }
    state = {"values": {"__interrupt__": [{"value": interrupt}]}}
    client = _FakeClient([], state=state)

    first = asyncio.run(
        bridge_server._prepare_run_payload(
            client,
            "thread_approval",
            "conversation-a",
            [{"role": "user", "content": "approve always"}],
        )
    )
    second = asyncio.run(
        bridge_server._prepare_run_payload(
            client,
            "thread_approval",
            "conversation-a",
            [{"role": "user", "content": "weiter"}],
        )
    )
    changed = asyncio.run(
        bridge_server._prepare_run_payload(
            _FakeClient(
                [],
                state={"values": {"__interrupt__": [{"value": {**interrupt, "command": "docker compose down"}}]}},
            ),
            "thread_approval",
            "conversation-a",
            [{"role": "user", "content": "weiter"}],
        )
    )

    assert first["command"] == {"resume": {"action": "approve", "remember": "thread_command"}}
    assert second["command"] == {"resume": {"action": "approve", "remembered": "thread_command"}}
    assert first["state_profile"]["message_count"] == 0
    assert second["state_profile"]["message_count"] == 0
    assert "direct_response" in changed
    assert "docker compose down" in changed["direct_response"]


def test_input_tokens_endpoint_returns_count_object() -> None:
    result = asyncio.run(
        bridge_server.response_input_tokens(
            _StubRequest({"input": [{"role": "user", "content": "Hallo AlphaRavis"}]})
        )
    )

    assert result["object"] == "response.input_tokens"
    assert result["input_tokens"] > 0
    assert "input_tokens_details" not in result


def test_bridge_observer_records_raw_and_model_context() -> None:
    bridge_server._BRIDGE_OBSERVATIONS.clear()

    class _Request:
        method = "POST"
        headers = {"x-conversation-id": "conv_test", "x-alpha-trace-id": "trace_test"}

        class _Url:
            path = "/v1/responses"

        class _Client:
            host = "127.0.0.1"

        url = _Url()
        client = _Client()

    messages = [{"role": "user", "content": "Hi"}]
    observation_id = bridge_server._observer_start(
        protocol="responses",
        request=_Request(),
        body={"model": "my-agent", "stream": True, "metadata": {"conversation_id": "conv_test"}},
        messages=messages,
    )
    bridge_server._observer_prepared(
        observation_id,
        thread_key="conv_test",
        thread_id="thread_test",
        run_payload={
            "input": {"messages": [{"role": "human", "content": "Hi"}]},
            "state_profile": {"message_count": 7},
        },
    )
    bridge_server._observer_complete(observation_id, output_text="Hallo")

    record = bridge_server._BRIDGE_OBSERVATIONS[0]
    assert record["thread_key"] == "conv_test"
    assert record["send"]["raw_messages"] == messages
    assert record["send"]["model_context_messages"] == [{"role": "human", "content": "Hi"}]
    assert record["send"]["langgraph_state_profile"]["message_count"] == 7
    assert record["receive"]["output_text"] == "Hallo"


def test_context_activity_extracts_compaction_and_hard_trim() -> None:
    compaction = bridge_server._extract_context_activity(
        {
            "event": "updates",
            "data": {
                "pre_run_context_guard": {
                    "run_profile": {
                        "pre_run_compression_used": True,
                        "pre_run_compression_tokens": 120000,
                        "pre_run_compression_tokens_after": 64000,
                    }
                }
            },
        }
    )
    hard = bridge_server._extract_context_activity(
        {
            "event": "updates",
            "data": {
                "pre_run_context_guard": {
                    "run_profile": {
                        "hard_context_trim_used": True,
                        "hard_context_trim_tokens_before": 130000,
                        "hard_context_trim_tokens_after": 90000,
                        "hard_context_trim_removed_messages": 12,
                    }
                }
            },
        }
    )

    assert compaction[0] == "context_compaction"
    assert "Compaction aktiv" in compaction[2]
    assert hard[0] == "context_hard"
    assert "Hard-Trim aktiv" in hard[2]
    assert "entfernt=12" in hard[2]


def test_bridge_observer_records_context_budget_updates() -> None:
    bridge_server._BRIDGE_OBSERVATIONS.clear()
    bridge_server._BRIDGE_OBSERVATIONS.appendleft({"id": "obs_budget", "receive": {}})

    bridge_server._observer_note_budget(
        "obs_budget",
        node_name="final_budget_rescue",
        profile={
            "final_context_budget": {
                "context_length": 128000,
                "message_tokens": 70000,
                "static_context_reserve_tokens": 4557,
                "request_tokens": 74557,
                "effective_active_limit": 59443,
            },
            "final_budget_rescue_used": True,
            "final_budget_rescue_passes": 2,
        },
    )

    budget = bridge_server._BRIDGE_OBSERVATIONS[0]["receive"]["context_budget"]
    assert budget["node"] == "final_budget_rescue"
    assert budget["request_tokens"] == 74557
    assert budget["final_budget_rescue_used"] is True
    assert budget["final_budget_rescue_passes"] == 2


def test_input_items_and_delete_routes_use_stored_response() -> None:
    bridge_server._RESPONSES_STORE.clear()
    bridge_server._RESPONSES_INPUT_ITEMS.clear()
    response = bridge_server._response_object(
        "Antwort",
        "my-agent",
        "resp_items",
        body={"store": True, "input": "Hallo"},
    )
    bridge_server._store_response_object(response, {"store": True, "input": "Hallo"})

    items = asyncio.run(bridge_server.list_response_input_items("resp_items", limit=10, order="asc"))
    assert items["object"] == "list"
    assert items["data"][0]["content"][0]["text"] == "Hallo"

    deleted = asyncio.run(bridge_server.delete_response("resp_items"))
    assert deleted == {"id": "resp_items", "object": "response", "deleted": True}
    assert "resp_items" not in bridge_server._RESPONSES_STORE


def test_retrieve_stream_query_returns_explicit_unsupported_error() -> None:
    result = asyncio.run(bridge_server.retrieve_response("resp_any", stream=True))

    assert result.status_code == 501
    assert result["error"]["code"] == "retrieve_stream_not_supported"


def test_responses_event_is_sse_with_semantic_type() -> None:
    raw = bridge_server._responses_event(
        "response.output_text.delta",
        {"type": "response.output_text.delta", "delta": "Hi"},
    )

    assert raw.startswith("event: response.output_text.delta\n")
    assert "\ndata: " in raw
    parsed = json.loads(raw.split("data: ", 1)[1])
    assert parsed["type"] == "response.output_text.delta"


def test_stream_responses_uses_librechat_reasoning_events(monkeypatch) -> None:
    parts = [
        {"event": "updates", "data": {"general_assistant": {"status": "running"}}},
        {"event": "messages", "data": (_FakeMessage("ai", "", reasoning_content="Reasoning delta."), {})},
        {"event": "messages", "data": (_FakeMessage("ai", "Hallo"), {})},
    ]
    _patch_stream_env(monkeypatch, parts)

    events = asyncio.run(_collect_response_stream({"model": "my-agent", "input": "Hi", "stream": True}, parts))
    event_types = [event["event"] for event in events]

    assert "response.reasoning.delta" in event_types
    assert "response.reasoning.done" in event_types
    assert "response.reasoning_text.delta" not in event_types
    assert "response.reasoning_text.done" not in event_types

    text_events = [event["data"] for event in events if event["event"] in {"response.output_text.delta", "response.output_text.done"}]
    assert text_events
    assert all(event.get("logprobs") == [] for event in text_events)

    completed = next(event["data"]["response"] for event in events if event["event"] == "response.completed")
    output_types = [item["type"] for item in completed["output"]]
    assert "message" in output_types
    assert "reasoning" in output_types


def test_stream_responses_maps_internal_tools_to_function_items(monkeypatch) -> None:
    parts = [
        {
            "event": "messages",
            "data": (
                _FakeMessage(
                    "ai",
                    "",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "name": "write_file",
                            "args": {"path": "app.py", "content": "print('ok')"},
                        }
                    ],
                ),
                {},
            ),
        },
        {"event": "messages", "data": (_FakeMessage("tool", "wrote app.py", tool_call_id="call_1", name="write_file"), {})},
        {"event": "messages", "data": (_FakeMessage("ai", "Fertig"), {})},
    ]
    _patch_stream_env(monkeypatch, parts)

    events = asyncio.run(_collect_response_stream({"model": "my-agent", "input": "Write", "stream": True}, parts))
    added_items = [event["data"]["item"] for event in events if event["event"] == "response.output_item.added"]

    function_call = next(item for item in added_items if item["type"] == "function_call")
    function_output = next(item for item in added_items if item["type"] == "function_call_output")

    assert function_call["call_id"] == "call_1"
    assert function_call["name"] == "write_file"
    assert function_output["call_id"] == "call_1"
    assert "wrote app.py" in function_output["output"]
    assert any(event["event"] == "response.function_call_arguments.delta" for event in events)
    assert any(event["event"] == "response.function_call_arguments.done" for event in events)


def test_stream_responses_splits_visible_think_blocks(monkeypatch) -> None:
    parts = [
        {"event": "messages", "data": (_FakeMessage("ai", "<think>plan</think>Answer"), {})},
    ]
    _patch_stream_env(monkeypatch, parts)

    events = asyncio.run(_collect_response_stream({"model": "my-agent", "input": "Hi", "stream": True}, parts))
    reasoning_text = "".join(
        event["data"].get("delta", "") for event in events if event["event"] == "response.reasoning.delta"
    )
    output_text = "".join(
        event["data"].get("delta", "") for event in events if event["event"] == "response.output_text.delta"
    )
    output_done = next(event["data"]["text"] for event in events if event["event"] == "response.output_text.done")
    completed = next(event["data"]["response"] for event in events if event["event"] == "response.completed")
    assistant_item = next(item for item in completed["output"] if item["type"] == "message")

    assert reasoning_text == "plan"
    assert output_text == "Answer"
    assert output_done == "Answer"
    assert assistant_item["content"][0]["text"] == "Answer"
    assert "<think>" not in output_done
    assert "</think>" not in output_done


def test_stream_responses_routes_planner_text_to_reasoning(monkeypatch) -> None:
    parts = [
        {"event": "messages", "data": (_FakeMessage("AIMessageChunk", "Plan intern."), {"langgraph_node": "planner"})},
        {"event": "messages", "data": (_FakeMessage("ai", "Finale Antwort."), {"langgraph_node": "alpha_ravis_swarm"})},
    ]
    monkeypatch.setattr(bridge_server, "_client", lambda: _FakeClient(parts))
    monkeypatch.setattr(bridge_server, "BRIDGE_STREAM_REASONING_EVENTS", False)
    monkeypatch.setattr(bridge_server, "BRIDGE_RESPONSES_STREAM_ACTIVITY_EVENTS", False)
    monkeypatch.setattr(bridge_server, "BRIDGE_RESPONSES_DONE_SENTINEL", False)

    events = asyncio.run(_collect_response_stream({"model": "my-agent", "input": "Hi", "stream": True}, parts))
    reasoning_events = [event["data"] for event in events if event["event"] == "response.reasoning.delta"]
    output_text = "".join(
        event["data"].get("delta", "") for event in events if event["event"] == "response.output_text.delta"
    )

    assert output_text == "Finale Antwort."
    assert "Plan intern." in "".join(event.get("delta", "") for event in reasoning_events)
    assert all(event.get("alpha_reasoning_kind") == "internal_plan" for event in reasoning_events)
    assert all(event.get("alpha_reasoning_label") == "planner" for event in reasoning_events)


def test_stream_responses_routes_nested_planner_metadata_to_reasoning(monkeypatch) -> None:
    parts = [
        {
            "event": "messages",
            "data": (
                _FakeMessage("AIMessageChunk", "Plan intern."),
                {"metadata": {"langgraph_node": "planner"}},
            ),
        },
        {"event": "messages", "data": (_FakeMessage("ai", "Finale Antwort."), {"langgraph_node": "alpha_ravis_swarm"})},
    ]
    monkeypatch.setattr(bridge_server, "_client", lambda: _FakeClient(parts))
    monkeypatch.setattr(bridge_server, "BRIDGE_STREAM_REASONING_EVENTS", False)
    monkeypatch.setattr(bridge_server, "BRIDGE_RESPONSES_STREAM_ACTIVITY_EVENTS", False)
    monkeypatch.setattr(bridge_server, "BRIDGE_RESPONSES_DONE_SENTINEL", False)

    events = asyncio.run(_collect_response_stream({"model": "my-agent", "input": "Hi", "stream": True}, parts))
    reasoning_text = "".join(
        event["data"].get("delta", "") for event in events if event["event"] == "response.reasoning.delta"
    )
    output_text = "".join(
        event["data"].get("delta", "") for event in events if event["event"] == "response.output_text.delta"
    )

    assert "Plan intern." in reasoning_text
    assert output_text == "Finale Antwort."


def test_stream_responses_routes_messages_partial_planner_to_reasoning(monkeypatch) -> None:
    planner_id = "lc_run--planner"
    swarm_id = "lc_run--swarm"
    parts = [
        {
            "event": "messages/metadata",
            "data": {
                planner_id: {"metadata": {"langgraph_node": "planner"}},
                swarm_id: {"metadata": {"langgraph_node": "alpha_ravis_swarm"}},
            },
        },
        {"event": "messages/partial", "data": [_FakeMessage("ai", "Plan", name="", tool_calls=None)]},
        {"event": "messages/partial", "data": [_FakeMessage("ai", "Plan intern.", name="", tool_calls=None)]},
        {"event": "messages/partial", "data": [_FakeMessage("ai", "Finale", name="", tool_calls=None)]},
        {"event": "messages/partial", "data": [_FakeMessage("ai", "Finale Antwort.", name="", tool_calls=None)]},
    ]
    parts[1]["data"][0].id = planner_id
    parts[2]["data"][0].id = planner_id
    parts[3]["data"][0].id = swarm_id
    parts[4]["data"][0].id = swarm_id
    monkeypatch.setattr(bridge_server, "_client", lambda: _FakeClient(parts))
    monkeypatch.setattr(bridge_server, "BRIDGE_STREAM_REASONING_EVENTS", False)
    monkeypatch.setattr(bridge_server, "BRIDGE_RESPONSES_STREAM_ACTIVITY_EVENTS", False)
    monkeypatch.setattr(bridge_server, "BRIDGE_RESPONSES_DONE_SENTINEL", False)

    events = asyncio.run(_collect_response_stream({"model": "my-agent", "input": "Hi", "stream": True}, parts))
    reasoning_text = "".join(
        event["data"].get("delta", "") for event in events if event["event"] == "response.reasoning.delta"
    )
    output_text = "".join(
        event["data"].get("delta", "") for event in events if event["event"] == "response.output_text.delta"
    )

    assert "Plan intern." in reasoning_text
    assert "Plan intern." not in output_text
    assert output_text == "Finale Antwort."


def test_stream_responses_requests_langgraph_subgraph_streaming(monkeypatch) -> None:
    parts = [
        {"event": "messages", "data": (_FakeMessage("ai", "Finale Antwort."), {"langgraph_node": "alpha_ravis_swarm"})},
    ]
    fake_client = _FakeClient(parts)
    monkeypatch.setattr(bridge_server, "_client", lambda: fake_client)
    monkeypatch.setattr(bridge_server, "BRIDGE_STREAM_SUBGRAPHS", True)
    monkeypatch.setattr(bridge_server, "BRIDGE_RESPONSES_STREAM_ACTIVITY_EVENTS", False)
    monkeypatch.setattr(bridge_server, "BRIDGE_RESPONSES_DONE_SENTINEL", False)

    asyncio.run(_collect_response_stream({"model": "my-agent", "input": "Hi", "stream": True}, parts))

    assert fake_client.runs.last_stream_kwargs is not None
    assert fake_client.runs.last_stream_kwargs["stream_subgraphs"] is True


def test_stream_responses_splits_large_visible_output_deltas(monkeypatch) -> None:
    answer = "Alpha beta gamma delta epsilon zeta."
    parts = [
        {"event": "messages", "data": (_FakeMessage("ai", answer), {"langgraph_node": "alpha_ravis_swarm"})},
    ]
    monkeypatch.setattr(bridge_server, "_client", lambda: _FakeClient(parts))
    monkeypatch.setattr(bridge_server, "BRIDGE_RESPONSES_OUTPUT_DELTA_MAX_CHARS", 8)
    monkeypatch.setattr(bridge_server, "BRIDGE_RESPONSES_STREAM_ACTIVITY_EVENTS", False)
    monkeypatch.setattr(bridge_server, "BRIDGE_RESPONSES_DONE_SENTINEL", False)

    events = asyncio.run(_collect_response_stream({"model": "my-agent", "input": "Hi", "stream": True}, parts))
    deltas = [
        event["data"].get("delta", "") for event in events if event["event"] == "response.output_text.delta"
    ]

    assert len(deltas) > 1
    assert "".join(deltas) == answer


def test_stream_responses_can_emit_visible_output_character_deltas(monkeypatch) -> None:
    answer = "ABC"
    parts = [
        {"event": "messages", "data": (_FakeMessage("ai", answer), {"langgraph_node": "alpha_ravis_swarm"})},
    ]
    monkeypatch.setattr(bridge_server, "_client", lambda: _FakeClient(parts))
    monkeypatch.setattr(bridge_server, "BRIDGE_RESPONSES_OUTPUT_DELTA_MAX_CHARS", 1)
    monkeypatch.setattr(bridge_server, "BRIDGE_RESPONSES_STREAM_ACTIVITY_EVENTS", False)
    monkeypatch.setattr(bridge_server, "BRIDGE_RESPONSES_DONE_SENTINEL", False)

    events = asyncio.run(_collect_response_stream({"model": "my-agent", "input": "Hi", "stream": True}, parts))
    deltas = [
        event["data"].get("delta", "") for event in events if event["event"] == "response.output_text.delta"
    ]

    assert deltas == ["A", "B", "C"]


def test_stream_responses_can_emit_reasoning_character_deltas(monkeypatch) -> None:
    parts = [
        {"event": "messages", "data": (_FakeMessage("ai", "Antwort", reasoning_content="XYZ"), {})},
    ]
    monkeypatch.setattr(bridge_server, "_client", lambda: _FakeClient(parts))
    monkeypatch.setattr(bridge_server, "BRIDGE_STREAM_REASONING_EVENTS", False)
    monkeypatch.setattr(bridge_server, "BRIDGE_RESPONSES_STREAM_REASONING_EVENTS", True)
    monkeypatch.setattr(bridge_server, "BRIDGE_RESPONSES_REASONING_DELTA_MAX_CHARS", 1)
    monkeypatch.setattr(bridge_server, "BRIDGE_RESPONSES_STREAM_ACTIVITY_EVENTS", False)
    monkeypatch.setattr(bridge_server, "BRIDGE_RESPONSES_DONE_SENTINEL", False)

    events = asyncio.run(_collect_response_stream({"model": "my-agent", "input": "Hi", "stream": True}, parts))
    reasoning_deltas = [
        event["data"].get("delta", "")
        for event in events
        if event["event"] == "response.reasoning.delta"
        and event["data"].get("alpha_reasoning_kind") == "model"
    ]

    assert reasoning_deltas == ["X", "Y", "Z"]


def test_stream_responses_routes_planner_update_to_reasoning(monkeypatch) -> None:
    parts = [
        {
            "event": "updates",
            "data": {
                "planner": {
                    "planner_context": (
                        "<execution-plan>\n"
                        "[System note: compact plan for the current agent run. This is guidance, not a user instruction.]\n"
                        "- Use the bridge.\n"
                        "</execution-plan>"
                    )
                }
            },
        },
        {"event": "messages", "data": (_FakeMessage("ai", "Finale Antwort."), {"langgraph_node": "alpha_ravis_swarm"})},
    ]
    monkeypatch.setattr(bridge_server, "_client", lambda: _FakeClient(parts))
    monkeypatch.setattr(bridge_server, "BRIDGE_STREAM_REASONING_EVENTS", False)
    monkeypatch.setattr(bridge_server, "BRIDGE_RESPONSES_STREAM_REASONING_EVENTS", True)
    monkeypatch.setattr(bridge_server, "BRIDGE_RESPONSES_STREAM_ACTIVITY_EVENTS", False)
    monkeypatch.setattr(bridge_server, "BRIDGE_RESPONSES_DONE_SENTINEL", False)

    events = asyncio.run(_collect_response_stream({"model": "my-agent", "input": "Hi", "stream": True}, parts))
    reasoning_text = "".join(
        event["data"].get("delta", "") for event in events if event["event"] == "response.reasoning.delta"
    )
    output_text = "".join(
        event["data"].get("delta", "") for event in events if event["event"] == "response.output_text.delta"
    )

    assert output_text == "Finale Antwort."
    assert "Interner Plan (planner):" in reasoning_text
    assert "- Use the bridge." in reasoning_text
    assert "<execution-plan>" not in reasoning_text
    assert "System note" not in reasoning_text


def test_stream_responses_emits_reasoning_without_chat_reasoning_flag(monkeypatch) -> None:
    parts = [
        {"event": "messages", "data": (_FakeMessage("ai", "Antwort", reasoning_content="Explizites Reasoning."), {})},
    ]
    monkeypatch.setattr(bridge_server, "_client", lambda: _FakeClient(parts))
    monkeypatch.setattr(bridge_server, "BRIDGE_STREAM_REASONING_EVENTS", False)
    monkeypatch.setattr(bridge_server, "BRIDGE_RESPONSES_STREAM_REASONING_EVENTS", True)
    monkeypatch.setattr(bridge_server, "BRIDGE_RESPONSES_STREAM_ACTIVITY_EVENTS", False)
    monkeypatch.setattr(bridge_server, "BRIDGE_RESPONSES_DONE_SENTINEL", False)

    events = asyncio.run(_collect_response_stream({"model": "my-agent", "input": "Hi", "stream": True}, parts))
    reasoning_text = "".join(
        event["data"].get("delta", "") for event in events if event["event"] == "response.reasoning.delta"
    )
    output_text = "".join(
        event["data"].get("delta", "") for event in events if event["event"] == "response.output_text.delta"
    )

    assert reasoning_text == "Explizites Reasoning."
    assert output_text == "Antwort"


def test_visible_content_strips_omitted_thinking_placeholders(monkeypatch) -> None:
    monkeypatch.setattr(bridge_server, "BRIDGE_SCRUB_INTERNAL_CONTEXT", True)

    visible = bridge_server._visible_content(
        "[thinking content block omitted]\n"
        "[reasoning content block omitted]\n"
        "Kurzfassung"
    )

    assert visible == "Kurzfassung"


def test_stream_responses_splits_think_markers_across_chunks(monkeypatch) -> None:
    parts = [
        {"event": "messages", "data": (_FakeMessage("AIMessageChunk", "<thi"), {})},
        {"event": "messages", "data": (_FakeMessage("AIMessageChunk", "nk>plan</thi"), {})},
        {"event": "messages", "data": (_FakeMessage("AIMessageChunk", "nk>Answer"), {})},
    ]
    _patch_stream_env(monkeypatch, parts)

    events = asyncio.run(_collect_response_stream({"model": "my-agent", "input": "Hi", "stream": True}, parts))
    reasoning_text = "".join(
        event["data"].get("delta", "") for event in events if event["event"] == "response.reasoning.delta"
    )
    output_done = next(event["data"]["text"] for event in events if event["event"] == "response.output_text.done")

    assert reasoning_text == "plan"
    assert output_done == "Answer"
    assert "<think>" not in output_done
    assert "</think>" not in output_done


def test_stream_responses_scrubs_current_task_brief(monkeypatch) -> None:
    parts = [
        {
            "event": "messages",
            "data": (
                _FakeMessage(
                    "ai",
                    "<current-task-brief>\nUser request:\nhuman: welche Tools hast du\n</current-task-brief>Visible",
                ),
                {},
            ),
        },
    ]
    _patch_stream_env(monkeypatch, parts)

    events = asyncio.run(_collect_response_stream({"model": "my-agent", "input": "Hi", "stream": True}, parts))
    output_done = next(event["data"]["text"] for event in events if event["event"] == "response.output_text.done")

    assert output_done == "Visible"
    assert "current-task-brief" not in output_done
    assert "User request" not in output_done


def test_last_ai_content_does_not_fallback_to_user_prompt() -> None:
    state = {"values": {"messages": [_FakeMessage("human", "Chat history:\nuser: hi")]}}

    assert bridge_server._last_ai_content(state) == ""


def test_state_failure_message_uses_failed_trace_step() -> None:
    state = {
        "values": {
            "alpha_trace_steps": [
                {"name": "langgraph.memory_kernel.semantic.timeout", "error_type": "TimeoutError"},
                {
                    "name": "langgraph.planner.failed",
                    "error_type": "InternalServerError",
                    "classification": "server_error",
                },
            ]
        }
    }

    message = bridge_server._state_failure_message(state)

    assert "langgraph.planner.failed" in message
    assert "InternalServerError" in message
    assert "server_error" in message


def test_stream_chat_splits_visible_think_blocks(monkeypatch) -> None:
    parts = [
        {"event": "messages", "data": (_FakeMessage("AIMessageChunk", "<think>plan</think>Answer"), {})},
    ]
    _patch_stream_env(monkeypatch, parts)

    chunks = asyncio.run(_collect_chat_event_stream(parts))
    deltas = [chunk["choices"][0]["delta"] for chunk in chunks]
    reasoning_text = "".join(delta.get("reasoning_content", "") for delta in deltas)
    output_text = "".join(delta.get("content", "") for delta in deltas)

    assert reasoning_text == "plan"
    assert output_text == "Answer"
    assert "<think>" not in output_text
    assert "</think>" not in output_text


def test_stream_chat_routes_planner_text_to_reasoning(monkeypatch) -> None:
    parts = [
        {"event": "messages", "data": (_FakeMessage("AIMessageChunk", "Plan intern."), {"langgraph_node": "planner"})},
        {"event": "messages", "data": (_FakeMessage("ai", "Finale Antwort."), {"langgraph_node": "alpha_ravis_swarm"})},
    ]
    monkeypatch.setattr(bridge_server, "BRIDGE_STREAM_REASONING_EVENTS", False)

    chunks = asyncio.run(_collect_chat_event_stream(parts))
    deltas = [chunk["choices"][0]["delta"] for chunk in chunks]
    reasoning_deltas = [delta for delta in deltas if delta.get("reasoning_content")]
    output_text = "".join(delta.get("content", "") for delta in deltas)

    assert output_text == "Finale Antwort."
    assert any("Plan intern." in delta.get("reasoning_content", "") for delta in reasoning_deltas)
    assert all(delta.get("alpha_reasoning_kind") == "internal_plan" for delta in reasoning_deltas)
    assert all(delta.get("alpha_reasoning_label") == "planner" for delta in reasoning_deltas)


def test_stream_responses_splits_think_blocks_from_state_fallback(monkeypatch) -> None:
    state = {"values": {"messages": [_FakeMessage("ai", "<think>plan</think>Answer")]}}
    monkeypatch.setattr(bridge_server, "_client", lambda: _FakeClient([], state=state))
    monkeypatch.setattr(bridge_server, "BRIDGE_STREAM_REASONING_EVENTS", True)
    monkeypatch.setattr(bridge_server, "BRIDGE_RESPONSES_STREAM_ACTIVITY_EVENTS", True)
    monkeypatch.setattr(bridge_server, "BRIDGE_RESPONSES_STREAM_TOOL_EVENTS", True)
    monkeypatch.setattr(bridge_server, "BRIDGE_RESPONSES_DONE_SENTINEL", False)

    events = asyncio.run(_collect_response_stream({"model": "my-agent", "input": "Hi", "stream": True}, []))
    reasoning_text = "".join(
        event["data"].get("delta", "") for event in events if event["event"] == "response.reasoning.delta"
    )
    output_done = next(event["data"]["text"] for event in events if event["event"] == "response.output_text.done")

    assert reasoning_text == "plan"
    assert output_done == "Answer"


def test_explicit_reasoning_wins_over_string_think_blocks(monkeypatch) -> None:
    parts = [
        {
            "event": "messages",
            "data": (_FakeMessage("ai", "<think>duplicate</think>Answer", reasoning_content="Explicit."), {}),
        },
    ]
    _patch_stream_env(monkeypatch, parts)

    events = asyncio.run(_collect_response_stream({"model": "my-agent", "input": "Hi", "stream": True}, parts))
    reasoning_text = "".join(
        event["data"].get("delta", "") for event in events if event["event"] == "response.reasoning.delta"
    )
    output_done = next(event["data"]["text"] for event in events if event["event"] == "response.output_text.done")

    assert reasoning_text == "Explicit."
    assert "duplicate" not in reasoning_text
    assert output_done == "Answer"


def _run_all() -> None:
    tests = [
        test_responses_input_supports_instructions_and_content_parts,
        test_response_object_has_stable_ids_and_usage,
        test_response_store_honors_store_flag,
        test_previous_response_id_adds_stored_output_context,
        test_responses_validation_rejects_unsupported_hosted_features,
        test_input_tokens_endpoint_returns_count_object,
        test_input_items_and_delete_routes_use_stored_response,
        test_retrieve_stream_query_returns_explicit_unsupported_error,
        test_responses_event_is_sse_with_semantic_type,
    ]
    for test in tests:
        test()


if __name__ == "__main__":
    _run_all()
