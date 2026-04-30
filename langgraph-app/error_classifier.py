from __future__ import annotations

import re
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorReason(str, Enum):
    auth = "auth"
    rate_limit = "rate_limit"
    overloaded = "overloaded"
    server_error = "server_error"
    timeout = "timeout"
    context_overflow = "context_overflow"
    payload_too_large = "payload_too_large"
    image_too_large = "image_too_large"
    model_not_found = "model_not_found"
    format_error = "format_error"
    unknown = "unknown"


@dataclass(frozen=True)
class ClassifiedError:
    reason: ErrorReason
    status_code: int | None = None
    message: str = ""
    provider: str = ""
    model: str = ""
    retryable: bool = True
    should_compress: bool = False
    should_retry: bool = False
    should_fallback: bool = False
    should_use_crisis_manager: bool = False
    should_strip_params: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def action(self) -> str:
        if self.should_compress:
            return "compress_context"
        if self.should_use_crisis_manager:
            return "crisis_recovery"
        if self.should_strip_params:
            return "strip_params_or_fallback"
        if self.should_retry:
            return "retry_or_backoff"
        if self.should_fallback:
            return "fallback_model_or_api_mode"
        return "surface_error"

    def to_profile(self) -> dict[str, Any]:
        return {
            "reason": self.reason.value,
            "status_code": self.status_code,
            "action": self.action,
            "retryable": self.retryable,
            "should_compress": self.should_compress,
            "should_retry": self.should_retry,
            "should_fallback": self.should_fallback,
            "should_use_crisis_manager": self.should_use_crisis_manager,
            "should_strip_params": self.should_strip_params,
            "provider": self.provider,
            "model": self.model,
            "message": self.message[:500],
            **({"metadata": self.metadata} if self.metadata else {}),
        }


class AlphaRavisAPIError(RuntimeError):
    def __init__(self, classification: ClassifiedError, *, original: Exception | None = None) -> None:
        self.classification = classification
        self.original = original
        super().__init__(classification.message or classification.reason.value)


_CONTEXT_PATTERNS = (
    "context length",
    "context size",
    "maximum context",
    "context window",
    "too many tokens",
    "token limit",
    "prompt is too long",
    "prompt exceeds",
    "reduce the length",
    "exceeds the max_model_len",
    "max_model_len",
    "maximum model length",
    "context length exceeded",
    "n_ctx_slot",
    "slot context",
    "input is too long",
    "max input token",
)
_RATE_LIMIT_PATTERNS = (
    "rate limit",
    "rate_limit",
    "too many requests",
    "throttled",
    "requests per minute",
    "tokens per minute",
    "resource_exhausted",
    "retry after",
)
_AUTH_PATTERNS = (
    "invalid api key",
    "invalid_api_key",
    "unauthorized",
    "forbidden",
    "authentication",
    "invalid token",
    "token expired",
)
_MODEL_NOT_FOUND_PATTERNS = (
    "model not found",
    "model_not_found",
    "invalid model",
    "no such model",
    "unknown model",
    "does not exist",
    "not a valid model",
)
_FORMAT_PATTERNS = (
    "unsupported parameter",
    "unknown parameter",
    "extra fields not permitted",
    "invalid request",
    "invalid tool",
    "schema",
    "json",
    "bad request",
)
_PAYLOAD_PATTERNS = (
    "payload too large",
    "request entity too large",
    "content too large",
)
_IMAGE_PATTERNS = (
    "image too large",
    "image exceeds",
    "image_too_large",
)
_TIMEOUT_PATTERNS = (
    "timeout",
    "timed out",
    "readtimeout",
    "connecttimeout",
    "cannot connect",
    "connection refused",
    "connection reset",
    "server disconnected",
    "remote protocol",
    "broken pipe",
    "eof",
)


def classify_api_error(
    error: Exception,
    *,
    provider: str = "",
    model: str = "",
    approx_tokens: int = 0,
    context_length: int = 0,
    num_messages: int = 0,
) -> ClassifiedError:
    if isinstance(error, AlphaRavisAPIError):
        return error.classification
    if os.getenv("ALPHARAVIS_ENABLE_ERROR_CLASSIFIER", "true").strip().lower() not in {"1", "true", "yes", "on"}:
        message = _extract_message(error)
        return ClassifiedError(
            reason=ErrorReason.unknown,
            status_code=_extract_status_code(error),
            message=message,
            provider=provider,
            model=model,
            retryable=True,
            should_retry=True,
        )

    status_code = _extract_status_code(error)
    message = _extract_message(error)
    lowered = message.lower()

    def result(reason: ErrorReason, **overrides: Any) -> ClassifiedError:
        kwargs: dict[str, Any] = {
            "reason": reason,
            "status_code": status_code,
            "message": message,
            "provider": provider,
            "model": model,
            "retryable": True,
        }
        kwargs.update(overrides)
        return ClassifiedError(**kwargs)

    if _has_any(lowered, _CONTEXT_PATTERNS):
        return result(
            ErrorReason.context_overflow,
            should_compress=True,
            should_retry=True,
            should_fallback=False,
            metadata=_context_metadata(approx_tokens, context_length, num_messages),
        )

    if status_code == 413 or _has_any(lowered, _PAYLOAD_PATTERNS):
        return result(ErrorReason.payload_too_large, should_compress=True, should_retry=True)

    if _has_any(lowered, _IMAGE_PATTERNS):
        return result(ErrorReason.image_too_large, retryable=False, should_fallback=False)

    if status_code in {401, 403} or _has_any(lowered, _AUTH_PATTERNS):
        return result(ErrorReason.auth, retryable=False, should_fallback=True)

    if status_code == 429 or _has_any(lowered, _RATE_LIMIT_PATTERNS):
        return result(ErrorReason.rate_limit, should_retry=True, should_fallback=True)

    if status_code in {408, 504} or _is_transport_error(error) or _has_any(lowered, _TIMEOUT_PATTERNS):
        return result(
            ErrorReason.timeout,
            should_retry=True,
            should_fallback=True,
            should_use_crisis_manager=True,
            metadata=_context_metadata(approx_tokens, context_length, num_messages),
        )

    if status_code in {503, 529}:
        return result(
            ErrorReason.overloaded,
            should_retry=True,
            should_fallback=True,
            should_use_crisis_manager=True,
        )

    if status_code in {500, 502}:
        return result(
            ErrorReason.server_error,
            should_retry=True,
            should_fallback=True,
            should_use_crisis_manager=True,
        )

    if status_code == 404 or _has_any(lowered, _MODEL_NOT_FOUND_PATTERNS):
        return result(ErrorReason.model_not_found, retryable=False, should_fallback=True)

    if status_code == 400 or _has_any(lowered, _FORMAT_PATTERNS):
        return result(
            ErrorReason.format_error,
            retryable=False,
            should_fallback=True,
            should_strip_params=True,
        )

    return result(ErrorReason.unknown, should_retry=True)


def format_user_error(error: Exception, *, component: str = "AlphaRavis") -> str:
    classified = classify_api_error(error, provider=component)
    reason = classified.reason.value
    details = classified.message[:500]

    if classified.reason == ErrorReason.context_overflow:
        return (
            f"{component}: Der Modellkontext ist zu gross fuer das Backend "
            f"(Fehlerklasse: {reason}). Ich sollte den aktiven Kontext komprimieren "
            "oder gezielt Archiv/RAG suchen, statt alles direkt zu senden."
        )
    if classified.reason in {ErrorReason.timeout, ErrorReason.server_error, ErrorReason.overloaded}:
        return (
            f"{component}: Das LLM-Backend antwortet gerade nicht sauber "
            f"(Fehlerklasse: {reason}). Wenn Advanced Model Management aktiv ist, "
            "ist das ein Kandidat fuer den Crisis Manager/Backend-Check. "
            f"Details: {details}"
        )
    if classified.reason == ErrorReason.rate_limit:
        return f"{component}: Das Backend meldet Rate-Limit/Busy-Zustand. Bitte kurz warten oder Fallback nutzen. Details: {details}"
    if classified.reason == ErrorReason.format_error:
        return (
            f"{component}: Das Backend hat das Request-Format abgelehnt "
            f"(Fehlerklasse: {reason}). Responses/Chat-Fallback oder Parameter-Strip ist sinnvoll. Details: {details}"
        )
    if classified.reason == ErrorReason.auth:
        return f"{component}: Authentifizierung/API-Key wurde abgelehnt. Details: {details}"
    if classified.reason == ErrorReason.model_not_found:
        return f"{component}: Das konfigurierte Modell wurde nicht gefunden. Details: {details}"
    if classified.reason == ErrorReason.payload_too_large:
        return f"{component}: Die Anfrage ist zu gross fuer das HTTP/API-Limit. Bitte kuerzen oder archivieren. Details: {details}"

    return f"{component}: Fehlerklasse {reason}. Details: {details}"


def _extract_status_code(error: Exception) -> int | None:
    direct = getattr(error, "status_code", None)
    if isinstance(direct, int):
        return direct

    response = getattr(error, "response", None)
    response_code = getattr(response, "status_code", None)
    if isinstance(response_code, int):
        return response_code

    match = re.search(r"\b(?:HTTP|status(?: code)?)[^\d]{0,12}([1-5]\d\d)\b", str(error), re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _extract_message(error: Exception) -> str:
    parts = [str(error).strip() or error.__class__.__name__]
    response = getattr(error, "response", None)
    if response is not None:
        text = getattr(response, "text", "")
        if text and text not in parts[0]:
            parts.append(str(text))
    return " ".join(part for part in parts if part)


def _is_transport_error(error: Exception) -> bool:
    error_type = type(error).__name__.lower()
    if isinstance(error, (TimeoutError, ConnectionError, OSError)):
        return True
    return any(token in error_type for token in ("timeout", "connect", "protocol", "network", "readerror"))


def _has_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


def _context_metadata(approx_tokens: int, context_length: int, num_messages: int) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if approx_tokens:
        metadata["approx_tokens"] = approx_tokens
    if context_length:
        metadata["context_length"] = context_length
    if num_messages:
        metadata["num_messages"] = num_messages
    return metadata
