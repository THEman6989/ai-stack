from __future__ import annotations

import asyncio
import json
import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

if importlib.util.find_spec("fastapi") is None:
    fastapi_stub = types.ModuleType("fastapi")

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

    fastapi_stub.FastAPI = FastAPI
    fastapi_stub.HTTPException = HTTPException
    fastapi_stub.Request = Request
    sys.modules["fastapi"] = fastapi_stub

    responses_stub = types.ModuleType("fastapi.responses")

    class JSONResponse(dict):
        def __init__(self, content=None, status_code: int = 200, *args, **kwargs) -> None:
            super().__init__(content or {})
            self.status_code = status_code

    class StreamingResponse:
        def __init__(self, content, media_type: str = "") -> None:
            self.content = content
            self.media_type = media_type

    responses_stub.JSONResponse = JSONResponse
    responses_stub.StreamingResponse = StreamingResponse
    sys.modules["fastapi.responses"] = responses_stub

if importlib.util.find_spec("langgraph_sdk") is None:
    sdk_stub = types.ModuleType("langgraph_sdk")
    sdk_stub.get_client = lambda *args, **kwargs: None
    sys.modules["langgraph_sdk"] = sdk_stub

import bridge_server  # noqa: E402


class _StubRequest:
    headers: dict[str, str] = {}

    def __init__(self, body: dict) -> None:
        self._body = body

    async def json(self) -> dict:
        return self._body


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
                ],
            }
        ],
    }

    messages = bridge_server._responses_input_to_messages(body)

    assert messages[0] == {"role": "system", "content": "Du bist AlphaRavis."}
    assert messages[1]["role"] == "user"
    assert "Analysiere das bitte." in messages[1]["content"]
    assert "Media attachment withheld" in messages[1]["content"]


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


def test_response_store_honors_store_flag() -> None:
    bridge_server._RESPONSES_STORE.clear()
    stored = bridge_server._response_object("stored", "my-agent", "resp_store", body={"store": True})
    skipped = bridge_server._response_object("skip", "my-agent", "resp_skip", body={"store": False})

    bridge_server._store_response_object(stored, {"store": True})
    bridge_server._store_response_object(skipped, {"store": False})

    assert "resp_store" in bridge_server._RESPONSES_STORE
    assert "resp_skip" not in bridge_server._RESPONSES_STORE


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


def test_input_tokens_endpoint_returns_count_object() -> None:
    result = asyncio.run(
        bridge_server.response_input_tokens(
            _StubRequest({"input": [{"role": "user", "content": "Hallo AlphaRavis"}]})
        )
    )

    assert result["object"] == "response.input_tokens"
    assert result["input_tokens"] > 0
    assert "input_tokens_details" not in result


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
