from __future__ import annotations

import copy
import os
import re
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class ProviderProfile:
    name: str
    provider_family: str
    evidence: str
    omit_sampling: bool = False
    chat_token_limit_param: str = "max_tokens"
    responses_token_limit_param: str = "max_output_tokens"
    requires_responses: bool = False
    chat_fallback_allowed: bool = True
    direct_non_openai_adapter: str = "disabled"


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


def model_rejects_sampling_knobs(model: str | None, base_url: str | None = None) -> bool:
    bare = _bare_model(model)
    base = str(base_url or "").lower()
    direct_openai_like = "api.openai.com" in base or "api.githubcopilot.com" in base
    if not direct_openai_like:
        return False
    return bare.startswith(("gpt-5", "o1", "o3", "o4"))


def model_uses_max_completion_tokens(model: str | None, base_url: str | None = None) -> bool:
    bare = _bare_model(model)
    base = str(base_url or "").lower()
    direct_openai_like = "api.openai.com" in base or "api.githubcopilot.com" in base
    if not direct_openai_like:
        return False
    return bare.startswith(("gpt-5", "gpt-4o", "o1", "o3", "o4"))


def _provider_profile_override() -> str:
    return os.getenv("ALPHARAVIS_PROVIDER_PROFILE", "auto").strip().lower()


def provider_profile_for(model: str | None, base_url: str | None = None) -> ProviderProfile:
    override = _provider_profile_override()
    if override in {"local", "local_litellm", "litellm", "llama_cpp"}:
        return ProviderProfile(
            name="local_litellm",
            provider_family="local_openai_compatible",
            evidence="local LiteLLM/llama.cpp smoke path keeps Chat max_tokens and allows Chat fallback",
        )
    if override in {"kimi", "kimi_moonshot", "moonshot"}:
        return ProviderProfile(
            name="kimi_moonshot",
            provider_family="moonshot",
            evidence="Kimi/Moonshot endpoints manage sampling server-side",
            omit_sampling=True,
        )
    if override in {"openai_reasoning", "reasoning", "gpt5", "gpt_5"}:
        return ProviderProfile(
            name="openai_reasoning",
            provider_family="openai_compatible_reasoning",
            evidence="direct OpenAI/GitHub reasoning-style endpoints may reject sampling knobs and prefer max_completion_tokens on Chat",
            omit_sampling=True,
            chat_token_limit_param="max_completion_tokens",
        )
    if override in {"responses_required", "responses_only", "require_responses"}:
        return ProviderProfile(
            name="responses_required",
            provider_family="openai_compatible_responses",
            evidence="operator-selected profile for providers where Chat Completions fallback is known broken",
            requires_responses=True,
            chat_fallback_allowed=False,
        )

    if model_manages_temperature(model, base_url):
        return ProviderProfile(
            name="kimi_moonshot",
            provider_family="moonshot",
            evidence="Kimi/Moonshot model or base URL detected",
            omit_sampling=True,
        )
    if model_uses_max_completion_tokens(model, base_url):
        return ProviderProfile(
            name="openai_reasoning" if model_rejects_sampling_knobs(model, base_url) else "openai_chat_token_compat",
            provider_family="openai_compatible_reasoning",
            evidence="direct OpenAI/GitHub model family detected",
            omit_sampling=model_rejects_sampling_knobs(model, base_url),
            chat_token_limit_param="max_completion_tokens",
        )
    return ProviderProfile(
        name="local_litellm",
        provider_family="local_openai_compatible",
        evidence="default AlphaRavis local LiteLLM/OpenAI-compatible gateway profile",
    )


def provider_profile_metadata(model: str | None, base_url: str | None = None) -> dict[str, Any]:
    return asdict(provider_profile_for(model, base_url))


def direct_non_openai_adapter_policy() -> dict[str, str]:
    return {
        "direct_non_openai_adapter": "disabled",
        "reason": "AlphaRavis keeps LiteLLM/LangChain as the provider route; add direct adapters only with explicit evidence and docs.",
    }


def _temperature_policy(env_name: str, *, default: str = "auto") -> str:
    return os.getenv(env_name, default).strip().lower()


def _should_omit_temperature(
    *,
    model: str | None,
    base_url: str | None,
    env_name: str,
    default: str = "auto",
) -> bool:
    mode = _temperature_policy(env_name, default=default)
    return mode in {"always", "true", "1", "yes"} or (
        mode in {"auto", ""} and provider_profile_for(model, base_url).omit_sampling
    )


def _should_omit_sampling(
    *,
    model: str | None,
    base_url: str | None,
    env_name: str,
    default: str = "auto",
) -> bool:
    return _should_omit_temperature(model=model, base_url=base_url, env_name=env_name, default=default)


def _drop_none_values(payload: dict[str, Any]) -> dict[str, Any]:
    if env_bool("ALPHARAVIS_RESPONSES_DROP_NONE_VALUES", "true"):
        return {key: value for key, value in payload.items() if value is not None}
    return payload


def _apply_responses_token_limit_mode(
    payload: dict[str, Any],
    *,
    model: str | None,
    base_url: str | None,
) -> dict[str, Any]:
    token_mode = os.getenv("ALPHARAVIS_RESPONSES_TOKEN_LIMIT_PARAM_MODE", "auto").strip().lower()
    if token_mode in {"auto", ""}:
        token_mode = provider_profile_for(model, base_url).responses_token_limit_param
    if token_mode == "none":
        for key in ("max_output_tokens", "max_tokens", "max_completion_tokens"):
            payload.pop(key, None)
        return payload
    if token_mode not in {"max_output_tokens", "max_tokens", "max_completion_tokens"}:
        return payload

    current_key = next(
        (key for key in ("max_output_tokens", "max_tokens", "max_completion_tokens") if key in payload),
        "",
    )
    if current_key and current_key != token_mode and token_mode not in payload:
        payload[token_mode] = payload.pop(current_key)
    return payload


def harden_responses_payload(payload: dict[str, Any], *, base_url: str = "") -> dict[str, Any]:
    hardened = copy.deepcopy(payload)
    model = str(hardened.get("model") or "")
    if _should_omit_sampling(
        model=model,
        base_url=base_url,
        env_name="ALPHARAVIS_RESPONSES_OMIT_TEMPERATURE_MODE",
    ):
        hardened.pop("temperature", None)
        hardened.pop("top_p", None)
    hardened = _apply_responses_token_limit_mode(hardened, model=model, base_url=base_url)
    return _drop_none_values(hardened)


def harden_chat_model_kwargs(
    model_kwargs: dict[str, Any] | None,
    *,
    model: str = "",
    base_url: str = "",
) -> dict[str, Any]:
    hardened = copy.deepcopy(model_kwargs or {})
    if not hardened:
        return {}

    if _should_omit_sampling(
        model=model,
        base_url=base_url,
        env_name="ALPHARAVIS_CHAT_OMIT_TEMPERATURE_MODE",
        default=os.getenv("ALPHARAVIS_RESPONSES_OMIT_TEMPERATURE_MODE", "auto"),
    ):
        hardened.pop("temperature", None)
        hardened.pop("top_p", None)

    token_mode = os.getenv("ALPHARAVIS_CHAT_TOKEN_LIMIT_PARAM_MODE", "auto").strip().lower()
    should_map_to_completion = token_mode in {"max_completion_tokens", "completion"} or (
        token_mode in {"auto", ""} and provider_profile_for(model, base_url).chat_token_limit_param == "max_completion_tokens"
    )
    if should_map_to_completion and "max_tokens" in hardened and "max_completion_tokens" not in hardened:
        hardened["max_completion_tokens"] = hardened.pop("max_tokens")
    elif token_mode == "none":
        hardened.pop("max_tokens", None)
        hardened.pop("max_completion_tokens", None)

    return _drop_none_values(hardened)


def chat_fallback_allowed(model: str | None, base_url: str | None = None) -> bool:
    mode = os.getenv("ALPHARAVIS_CHAT_FALLBACK_MODE", "auto").strip().lower()
    if mode in {"always", "allow", "true", "1", "yes"}:
        return True
    if mode in {"never", "deny", "false", "0", "no", "responses_required"}:
        return False
    profile = provider_profile_for(model, base_url)
    require_mode = os.getenv("ALPHARAVIS_PROVIDER_REQUIRE_RESPONSES_MODE", "auto").strip().lower()
    if require_mode in {"always", "true", "1", "yes"}:
        return False
    if require_mode in {"never", "false", "0", "no"}:
        return True
    return profile.chat_fallback_allowed and not profile.requires_responses


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
