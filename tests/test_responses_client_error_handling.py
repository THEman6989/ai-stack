from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

import responses_client  # noqa: E402
from error_classifier import AlphaRavisAPIError  # noqa: E402


class _Response:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text

    def json(self) -> dict:
        return {"output_text": "unused"}


class _FakeAsyncClient:
    calls: list[dict] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, url: str, headers: dict, json: dict):
        self.calls.append(json)
        if len(self.calls) == 1:
            return _Response(400, "Unsupported parameter: max_output_tokens")
        return _Response(400, "retry endpoint still rejected max_tokens")


def test_responses_retry_failure_keeps_original_and_retry_error():
    old_client = responses_client.httpx.AsyncClient
    old_retry = os.environ.get("ALPHARAVIS_RESPONSES_UNSUPPORTED_PARAM_RETRY")
    os.environ["ALPHARAVIS_RESPONSES_UNSUPPORTED_PARAM_RETRY"] = "true"
    _FakeAsyncClient.calls = []
    responses_client.httpx.AsyncClient = _FakeAsyncClient
    try:
        try:
            asyncio.run(
                responses_client.invoke_responses(
                    [{"role": "user", "content": "hello"}],
                    model_name="big-boss",
                    model_kwargs={"max_tokens": 12},
                )
            )
        except AlphaRavisAPIError as exc:
            message = str(exc)
        else:  # pragma: no cover - defensive assertion
            raise AssertionError("expected AlphaRavisAPIError")
    finally:
        responses_client.httpx.AsyncClient = old_client
        if old_retry is None:
            os.environ.pop("ALPHARAVIS_RESPONSES_UNSUPPORTED_PARAM_RETRY", None)
        else:
            os.environ["ALPHARAVIS_RESPONSES_UNSUPPORTED_PARAM_RETRY"] = old_retry

    assert len(_FakeAsyncClient.calls) == 2
    assert "max_output_tokens" in _FakeAsyncClient.calls[0]
    assert "max_tokens" in _FakeAsyncClient.calls[1]
    assert "Unsupported parameter: max_output_tokens" in message
    assert "Compatibility retry" in message
    assert "retry endpoint still rejected max_tokens" in message


if __name__ == "__main__":
    test_responses_retry_failure_keeps_original_and_retry_error()
