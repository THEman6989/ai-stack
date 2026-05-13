from __future__ import annotations

import sys
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

from provider_hardening import (  # noqa: E402
    chat_fallback_allowed,
    direct_non_openai_adapter_policy,
    harden_chat_model_kwargs,
    harden_responses_payload,
    provider_profile_for,
    retry_responses_payload_for_error,
    unsupported_parameter_from_error,
)


class _EnvGuard:
    def __enter__(self):
        self.old_values = {
            "ALPHARAVIS_RESPONSES_UNSUPPORTED_PARAM_RETRY": os.environ.get(
                "ALPHARAVIS_RESPONSES_UNSUPPORTED_PARAM_RETRY"
            ),
            "ALPHARAVIS_RESPONSES_OMIT_TEMPERATURE_MODE": os.environ.get(
                "ALPHARAVIS_RESPONSES_OMIT_TEMPERATURE_MODE"
            ),
            "ALPHARAVIS_CHAT_OMIT_TEMPERATURE_MODE": os.environ.get(
                "ALPHARAVIS_CHAT_OMIT_TEMPERATURE_MODE"
            ),
            "ALPHARAVIS_CHAT_TOKEN_LIMIT_PARAM_MODE": os.environ.get(
                "ALPHARAVIS_CHAT_TOKEN_LIMIT_PARAM_MODE"
            ),
            "ALPHARAVIS_PROVIDER_PROFILE": os.environ.get("ALPHARAVIS_PROVIDER_PROFILE"),
            "ALPHARAVIS_PROVIDER_REQUIRE_RESPONSES_MODE": os.environ.get(
                "ALPHARAVIS_PROVIDER_REQUIRE_RESPONSES_MODE"
            ),
            "ALPHARAVIS_CHAT_FALLBACK_MODE": os.environ.get("ALPHARAVIS_CHAT_FALLBACK_MODE"),
            "ALPHARAVIS_RESPONSES_TOKEN_LIMIT_PARAM_MODE": os.environ.get(
                "ALPHARAVIS_RESPONSES_TOKEN_LIMIT_PARAM_MODE"
            ),
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

    with _EnvGuard():
        os.environ["ALPHARAVIS_RESPONSES_OMIT_TEMPERATURE_MODE"] = "auto"
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


def test_harden_chat_model_kwargs_omits_server_managed_temperature():
    kwargs = {"temperature": 0, "max_tokens": 64}

    with _EnvGuard():
        os.environ["ALPHARAVIS_CHAT_OMIT_TEMPERATURE_MODE"] = "auto"
        hardened = harden_chat_model_kwargs(
            kwargs,
            model="moonshot/kimi-k2",
            base_url="https://api.kimi.com/v1",
        )

    assert hardened == {"max_tokens": 64}
    assert kwargs == {"temperature": 0, "max_tokens": 64}


def test_harden_chat_model_kwargs_maps_direct_openai_token_limit():
    kwargs = {"max_tokens": 64, "temperature": None}

    with _EnvGuard():
        os.environ["ALPHARAVIS_CHAT_TOKEN_LIMIT_PARAM_MODE"] = "auto"
        hardened = harden_chat_model_kwargs(
            kwargs,
            model="gpt-5.4",
            base_url="https://api.openai.com/v1",
        )

    assert hardened == {"max_completion_tokens": 64}


def test_openai_reasoning_profile_omits_sampling_and_maps_chat_token_limit():
    kwargs = {"max_tokens": 64, "temperature": 0, "top_p": 0.8}

    with _EnvGuard():
        os.environ["ALPHARAVIS_PROVIDER_PROFILE"] = "openai_reasoning"
        os.environ["ALPHARAVIS_CHAT_OMIT_TEMPERATURE_MODE"] = "auto"
        hardened = harden_chat_model_kwargs(
            kwargs,
            model="gpt-5.4",
            base_url="https://api.openai.com/v1",
        )

    assert hardened == {"max_completion_tokens": 64}


def test_responses_token_limit_mode_can_map_to_provider_required_spelling():
    payload = {"model": "x", "input": "hi", "max_output_tokens": 32}

    with _EnvGuard():
        os.environ["ALPHARAVIS_RESPONSES_TOKEN_LIMIT_PARAM_MODE"] = "max_completion_tokens"
        hardened = harden_responses_payload(payload, base_url="https://example.invalid/v1")

    assert hardened == {"model": "x", "input": "hi", "max_completion_tokens": 32}
    assert payload == {"model": "x", "input": "hi", "max_output_tokens": 32}


def test_responses_required_profile_blocks_chat_fallback():
    with _EnvGuard():
        os.environ["ALPHARAVIS_PROVIDER_PROFILE"] = "responses_required"
        profile = provider_profile_for("some-model", "https://example.invalid/v1")
        allowed = chat_fallback_allowed("some-model", "https://example.invalid/v1")

    assert profile.requires_responses is True
    assert allowed is False


def test_direct_non_openai_adapters_stay_disabled_by_policy():
    policy = direct_non_openai_adapter_policy()

    assert policy["direct_non_openai_adapter"] == "disabled"


def _run_all() -> None:
    tests = [
        test_harden_responses_payload_omits_kimi_temperature,
        test_unsupported_parameter_detection_from_provider_message,
        test_retry_payload_removes_unsupported_parameter,
        test_retry_payload_maps_max_output_tokens_for_compat_endpoint,
        test_harden_chat_model_kwargs_omits_server_managed_temperature,
        test_harden_chat_model_kwargs_maps_direct_openai_token_limit,
        test_openai_reasoning_profile_omits_sampling_and_maps_chat_token_limit,
        test_responses_token_limit_mode_can_map_to_provider_required_spelling,
        test_responses_required_profile_blocks_chat_fallback,
        test_direct_non_openai_adapters_stay_disabled_by_policy,
    ]
    for test in tests:
        test()


if __name__ == "__main__":
    _run_all()
