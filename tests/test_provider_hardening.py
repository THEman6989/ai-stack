from __future__ import annotations

import sys
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

from provider_hardening import (  # noqa: E402
    harden_responses_payload,
    retry_responses_payload_for_error,
    unsupported_parameter_from_error,
)


class _EnvGuard:
    def __enter__(self):
        self.old_values = {
            "ALPHARAVIS_RESPONSES_UNSUPPORTED_PARAM_RETRY": os.environ.get(
                "ALPHARAVIS_RESPONSES_UNSUPPORTED_PARAM_RETRY"
            )
        }
        return self

    def __exit__(self, exc_type, exc, tb):
        for key, value in self.old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_harden_responses_payload_omits_kimi_temperature():
    payload = {"model": "kimi-k2", "input": "hi", "temperature": 0.2}

    hardened = harden_responses_payload(payload, base_url="https://api.kimi.com/v1")

    assert "temperature" not in hardened
    assert payload["temperature"] == 0.2


def test_unsupported_parameter_detection_from_provider_message():
    error = '{"error":{"message":"Unsupported parameter: parallel_tool_calls"}}'

    assert unsupported_parameter_from_error(error) == "parallel_tool_calls"


def test_retry_payload_removes_unsupported_parameter():
    payload = {"model": "x", "input": "hi", "parallel_tool_calls": True}

    with _EnvGuard():
        os.environ["ALPHARAVIS_RESPONSES_UNSUPPORTED_PARAM_RETRY"] = "true"
        retry, reason = retry_responses_payload_for_error(payload, "unknown parameter: parallel_tool_calls")

    assert retry is not None
    assert "parallel_tool_calls" not in retry
    assert "parallel_tool_calls" in reason


def test_retry_payload_maps_max_output_tokens_for_compat_endpoint():
    payload = {"model": "x", "input": "hi", "max_output_tokens": 512}

    with _EnvGuard():
        os.environ["ALPHARAVIS_RESPONSES_UNSUPPORTED_PARAM_RETRY"] = "true"
        retry, reason = retry_responses_payload_for_error(payload, "Unsupported parameter: max_output_tokens")

    assert retry == {"model": "x", "input": "hi", "max_tokens": 512}
    assert "mapped" in reason


def _run_all() -> None:
    tests = [
        test_harden_responses_payload_omits_kimi_temperature,
        test_unsupported_parameter_detection_from_provider_message,
        test_retry_payload_removes_unsupported_parameter,
        test_retry_payload_maps_max_output_tokens_for_compat_endpoint,
    ]
    for test in tests:
        test()


if __name__ == "__main__":
    _run_all()
