from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

from error_classifier import ErrorReason, classify_api_error, format_user_error  # noqa: E402


def test_context_overflow_beats_generic_400() -> None:
    err = RuntimeError("Responses API HTTP 400: context length exceeded; prompt is too long")

    classified = classify_api_error(err, provider="responses", model="big-boss", approx_tokens=140000)

    assert classified.reason == ErrorReason.context_overflow
    assert classified.should_compress
    assert classified.action == "compress_context"


def test_timeout_is_crisis_candidate() -> None:
    classified = classify_api_error(TimeoutError("read timed out"), provider="bridge", model="big-boss")

    assert classified.reason == ErrorReason.timeout
    assert classified.should_use_crisis_manager
    assert classified.should_retry


def test_502_is_server_error_and_crisis_candidate() -> None:
    classified = classify_api_error(RuntimeError("LangGraph HTTP 502: bad gateway"), provider="bridge")

    assert classified.reason == ErrorReason.server_error
    assert classified.should_use_crisis_manager
    assert classified.should_fallback


def test_rate_limit_classification() -> None:
    classified = classify_api_error(RuntimeError("HTTP 429: rate limit exceeded"), provider="litellm")

    assert classified.reason == ErrorReason.rate_limit
    assert classified.should_retry
    assert classified.should_fallback


def test_format_error_recommends_strip_or_fallback() -> None:
    classified = classify_api_error(RuntimeError("HTTP 400: unsupported parameter response_format"), provider="responses")

    assert classified.reason == ErrorReason.format_error
    assert classified.should_strip_params
    assert classified.action == "strip_params_or_fallback"


def test_user_error_message_is_actionable() -> None:
    message = format_user_error(RuntimeError("HTTP 503: overloaded"), component="AlphaRavis Bridge")

    assert "Fehlerklasse: overloaded" in message
    assert "Crisis Manager" in message


def _run_all() -> None:
    tests = [
        test_context_overflow_beats_generic_400,
        test_timeout_is_crisis_candidate,
        test_502_is_server_error_and_crisis_candidate,
        test_rate_limit_classification,
        test_format_error_recommends_strip_or_fallback,
        test_user_error_message_is_actionable,
    ]
    for test in tests:
        test()


if __name__ == "__main__":
    _run_all()
