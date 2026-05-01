from __future__ import annotations

import copy
import os
import re
from typing import Any


UNSUPPORTED_PARAM_MARKERS = (
    "unsupported parameter",
    "unsupported_parameter",
    "not supported",
    "does not support",
    "unknown parameter",
    "unrecognized request argument",
    "unrecognized parameter",
    "invalid parameter",
)

RETRYABLE_RESPONSE_PARAMS = (
    "temperature",
    "top_p",
    "parallel_tool_calls",
    "truncation",
    "store",
    "metadata",
    "chat_template_kwargs",
    "max_output_tokens",
    "max_tokens",
    "max_completion_tokens",
)


def env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _bare_model(model: str | None) -> str:
    return str(model or "").strip().lower().rsplit("/", 1)[-1]


def model_manages_temperature(model: str | None, base_url: str | None = None) -> bool:
    bare = _bare_model(model)
    base = str(base_url or "").lower()
    if bare.startswith("kimi-") or bare == "kimi" or "moonshot" in bare:
        return True
    return "api.kimi.com" in base or "api.moonshot" in base


def harden_responses_payload(payload: dict[str, Any], *, base_url: str = "") -> dict[str, Any]:
    hardened = copy.deepcopy(payload)
    model = str(hardened.get("model") or "")
    omit_mode = os.getenv("ALPHARAVIS_RESPONSES_OMIT_TEMPERATURE_MODE", "auto").strip().lower()
    if omit_mode in {"always", "true", "1", "yes"} or (
        omit_mode in {"auto", ""} and model_manages_temperature(model, base_url)
    ):
        hardened.pop("temperature", None)
    if env_bool("ALPHARAVIS_RESPONSES_DROP_NONE_VALUES", "true"):
        hardened = {key: value for key, value in hardened.items() if value is not None}
    return hardened


def is_unsupported_parameter_error(error_text: str, param: str) -> bool:
    lowered = str(error_text or "").lower()
    param_lower = str(param or "").lower()
    if not lowered or not param_lower or param_lower not in lowered:
        return False
    return any(marker in lowered for marker in UNSUPPORTED_PARAM_MARKERS)


def unsupported_parameter_from_error(error_text: str) -> str:
    text = str(error_text or "")
    lowered = text.lower()
    for param in RETRYABLE_RESPONSE_PARAMS:
        if is_unsupported_parameter_error(text, param):
            return param

    patterns = [
        r"unsupported[_ ]parameter[:\s'\"]+(?P<param>[a-zA-Z0-9_]+)",
        r"unknown parameter[:\s'\"]+(?P<param>[a-zA-Z0-9_]+)",
        r"unrecognized request argument[:\s'\"]+(?P<param>[a-zA-Z0-9_]+)",
        r"unrecognized parameter[:\s'\"]+(?P<param>[a-zA-Z0-9_]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            param = match.group("param")
            if param in RETRYABLE_RESPONSE_PARAMS:
                return param
    return ""


def retry_responses_payload_for_error(payload: dict[str, Any], error_text: str) -> tuple[dict[str, Any] | None, str]:
    if not env_bool("ALPHARAVIS_RESPONSES_UNSUPPORTED_PARAM_RETRY", "true"):
        return None, ""

    param = unsupported_parameter_from_error(error_text)
    if not param:
        return None, ""

    retry_payload = copy.deepcopy(payload)
    reason = f"removed unsupported Responses parameter `{param}`"
    if param == "max_output_tokens" and "max_output_tokens" in retry_payload:
        value = retry_payload.pop("max_output_tokens", None)
        retry_payload.setdefault("max_tokens", value)
        reason = "mapped unsupported `max_output_tokens` to `max_tokens`"
    elif param == "max_tokens" and "max_tokens" in retry_payload:
        value = retry_payload.pop("max_tokens", None)
        retry_payload.setdefault("max_completion_tokens", value)
        reason = "mapped unsupported `max_tokens` to `max_completion_tokens`"
    elif param == "max_completion_tokens" and "max_completion_tokens" in retry_payload:
        value = retry_payload.pop("max_completion_tokens", None)
        retry_payload.setdefault("max_output_tokens", value)
        reason = "mapped unsupported `max_completion_tokens` to `max_output_tokens`"
    else:
        retry_payload.pop(param, None)

    if retry_payload == payload:
        return None, ""
    return retry_payload, reason
