from __future__ import annotations

import importlib.util
from importlib.machinery import ModuleSpec
import json
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

        def __init__(self, *args, **kwargs) -> None:
            pass

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

    class HTMLResponse(str):
        def __new__(cls, content: str = "", *args, **kwargs):
            return str.__new__(cls, content)

    class JSONResponse(dict):
        def __init__(self, content=None, status_code: int = 200, *args, **kwargs) -> None:
            super().__init__(content or {})
            self.status_code = status_code

    class StreamingResponse:
        def __init__(self, content, *args, **kwargs) -> None:
            self.content = content
            self.media_type = kwargs.get("media_type", "")
            self.headers = kwargs.get("headers", {})

    responses_stub.HTMLResponse = HTMLResponse
    responses_stub.JSONResponse = JSONResponse
    responses_stub.StreamingResponse = StreamingResponse
    sys.modules["fastapi.responses"] = responses_stub

if importlib.util.find_spec("pydantic") is None:
    pydantic_stub = types.ModuleType("pydantic")
    pydantic_stub.__spec__ = ModuleSpec("pydantic", loader=None)

    class BaseModel:
        def __init__(self, **kwargs) -> None:
            for name, value in self.__class__.__dict__.items():
                if not name.startswith("_") and name not in kwargs:
                    setattr(self, name, value)
            for name, value in kwargs.items():
                setattr(self, name, value)

    pydantic_stub.BaseModel = BaseModel
    sys.modules["pydantic"] = pydantic_stub

import test_ui_server  # noqa: E402


def test_stream_payload_uses_responses_sse() -> None:
    request = test_ui_server.ChatRequest(
        message="Hi",
        messages=[{"role": "assistant", "content": "Hallo"}],
        protocol="responses",
    )

    url, payload = test_ui_server._bridge_request_payload(
        request,
        text="Welche Tools hast du?",
        protocol="responses",
        session_id="session_test",
        trace_id="trace_test",
        stream=True,
    )

    assert url.endswith("/responses")
    assert payload["stream"] is True
    assert payload["input"] == [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hallo"}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Welche Tools hast du?"}],
        },
    ]
    assert payload["metadata"]["trace_id"] == "trace_test"


def test_stream_payload_uses_chat_sse() -> None:
    request = test_ui_server.ChatRequest(
        message="Hi",
        messages=[{"role": "assistant", "content": "Hallo"}],
        protocol="chat",
    )

    url, payload = test_ui_server._bridge_request_payload(
        request,
        text="Weiter",
        protocol="chat",
        session_id="session_test",
        trace_id="trace_test",
        stream=True,
    )

    assert url.endswith("/chat/completions")
    assert payload["stream"] is True
    assert payload["messages"] == [
        {"role": "assistant", "content": "Hallo"},
        {"role": "user", "content": "Weiter"},
    ]


def test_test_ui_event_is_valid_sse_json() -> None:
    raw = test_ui_server._test_ui_event("test_ui.started", {"trace_id": "trace_test"})

    assert raw.startswith("event: test_ui.started\n")
    data_line = next(line for line in raw.splitlines() if line.startswith("data: "))
    assert json.loads(data_line.removeprefix("data: ")) == {"trace_id": "trace_test"}


def test_html_renders_collapsed_reasoning_panel_from_stream() -> None:
    assert "response.reasoning.delta" in test_ui_server.HTML
    assert "reasoning-details" in test_ui_server.HTML
    assert "live-panels" in test_ui_server.HTML
    assert "live-status" in test_ui_server.HTML
    assert "live-reasoning" in test_ui_server.HTML
    assert "live-plan" in test_ui_server.HTML
    assert "data-panel-toggle" in test_ui_server.HTML
    assert "panel.classList.toggle('expanded')" in test_ui_server.HTML
    assert "function renderLivePanels(msg)" in test_ui_server.HTML
    assert "summary.textContent = 'Reasoning'" in test_ui_server.HTML
    assert "reasoningStatus" in test_ui_server.HTML
    assert "internalPlan" in test_ui_server.HTML
    assert "Interner Plan" in test_ui_server.HTML
    assert "Modell-Reasoning" in test_ui_server.HTML
    assert "function reasoningKind(data, text, msg)" in test_ui_server.HTML
    assert "reasoningOpen: false" in test_ui_server.HTML


def test_html_sse_parser_escapes_regex_newlines() -> None:
    assert "block.split(/\\r?\\n/)" in test_ui_server.HTML
    assert "dataLines.join('\\n')" in test_ui_server.HTML
    assert "buffer.split(/\\r?\\n\\r?\\n/)" in test_ui_server.HTML
    assert "block.split(/\r?\n/)" not in test_ui_server.HTML


def test_clear_resets_backend_session_id() -> None:
    assert "let sessionId = storedSessionId();" in test_ui_server.HTML
    assert "function resetSessionId()" in test_ui_server.HTML
    assert "sessionId = resetSessionId();" in test_ui_server.HTML
    assert "neue Session bereit" in test_ui_server.HTML


def test_html_shows_route_badge_for_fast_or_agent_path() -> None:
    assert "route-badge" in test_ui_server.HTML
    assert "function routeLabel(routeName)" in test_ui_server.HTML
    assert "Fast Path" in test_ui_server.HTML
    assert "Agent Path" in test_ui_server.HTML
    assert "routeFromEvent" in test_ui_server.HTML
    assert "fast_chat" in test_ui_server.HTML
    assert "swarm" in test_ui_server.HTML


def test_html_compacts_text_delta_trace_rows_by_default() -> None:
    assert "trace-delta-details" in test_ui_server.HTML
    assert "Delta-Details" in test_ui_server.HTML
    assert "function summarizeTraceSteps(steps)" in test_ui_server.HTML
    assert "${group.name || 'Delta empfangen'} (${group.count} Deltas" in test_ui_server.HTML
    assert "Delta-Zeilen zusammengefasst" in test_ui_server.HTML
    assert "traceDeltaDetails.checked ? rawSteps : summarizeTraceSteps(rawSteps)" in test_ui_server.HTML
