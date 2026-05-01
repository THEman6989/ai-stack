from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any


OPERATIONAL_LOGGER_NAME = "alpharavis"
DEBUG_LOGGER_NAME = "alpharavis.debug_all"
DEFAULT_MAX_FIELD_CHARS = 4000
DEFAULT_DEBUG_MAX_FIELD_CHARS = 12000

SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|token|password|passwd|secret|authorization|cookie|credential|private[_-]?key)",
    re.IGNORECASE,
)
SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|password|passwd|secret|authorization|cookie)\s*[:=]\s*([^\s,;]+)"
)
BEARER_RE = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+")

_SETUP_DONE = False


def env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def setup_logging(component: str = "alpharavis") -> dict[str, Any]:
    """Configure AlphaRavis operational and optional all-debug file logs."""

    global _SETUP_DONE
    if _SETUP_DONE:
        return logging_status()

    operational_enabled = env_bool("ALPHARAVIS_OPERATIONAL_LOGGING", "true")
    debug_enabled = env_bool("ALPHARAVIS_DEBUG_ALL_LOGGING", "false")
    console_enabled = env_bool("ALPHARAVIS_LOG_CONSOLE", "true")
    jsonl_enabled = env_bool("ALPHARAVIS_LOG_JSONL", "true")
    retention_days = _positive_int_env("ALPHARAVIS_LOG_RETENTION_DAYS", 4, minimum=1, maximum=90)

    log_dir = resolve_log_dir()
    if operational_enabled or debug_enabled:
        log_dir.mkdir(parents=True, exist_ok=True)

    operational_logger = logging.getLogger(OPERATIONAL_LOGGER_NAME)
    operational_logger.setLevel(_level_from_env("ALPHARAVIS_LOG_LEVEL", logging.INFO))
    operational_logger.propagate = False
    _remove_managed_handlers(operational_logger)

    if operational_enabled:
        operational_dir = log_dir / "operational"
        operational_dir.mkdir(parents=True, exist_ok=True)
        _attach_text_handler(
            operational_logger,
            operational_dir / "alpharavis.log",
            retention_days=retention_days,
            level=operational_logger.level,
        )
        if jsonl_enabled:
            _attach_jsonl_handler(
                operational_logger,
                operational_dir / "alpharavis.jsonl",
                retention_days=retention_days,
                level=operational_logger.level,
            )
        if console_enabled:
            _attach_console_handler(operational_logger, level=operational_logger.level)
    else:
        operational_logger.addHandler(logging.NullHandler())

    debug_logger = logging.getLogger(DEBUG_LOGGER_NAME)
    debug_logger.setLevel(_level_from_env("ALPHARAVIS_DEBUG_LOG_LEVEL", logging.DEBUG))
    debug_logger.propagate = False
    _remove_managed_handlers(debug_logger)

    if debug_enabled:
        debug_dir = log_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        _attach_text_handler(
            debug_logger,
            debug_dir / "alpharavis-debug.log",
            retention_days=retention_days,
            level=debug_logger.level,
        )
        if jsonl_enabled:
            _attach_jsonl_handler(
                debug_logger,
                debug_dir / "alpharavis-debug.jsonl",
                retention_days=retention_days,
                level=debug_logger.level,
            )
    else:
        debug_logger.addHandler(logging.NullHandler())

    _SETUP_DONE = True
    log_event(
        logging.INFO,
        "logging.configured",
        component=component,
        message="AlphaRavis operational logging configured.",
        log_dir=str(log_dir),
        operational_enabled=operational_enabled,
        debug_all_enabled=debug_enabled,
        retention_days=retention_days,
    )
    return logging_status()


def logging_status() -> dict[str, Any]:
    return {
        "operational_enabled": env_bool("ALPHARAVIS_OPERATIONAL_LOGGING", "true"),
        "debug_all_enabled": env_bool("ALPHARAVIS_DEBUG_ALL_LOGGING", "false"),
        "console_enabled": env_bool("ALPHARAVIS_LOG_CONSOLE", "true"),
        "jsonl_enabled": env_bool("ALPHARAVIS_LOG_JSONL", "true"),
        "retention_days": _positive_int_env("ALPHARAVIS_LOG_RETENTION_DAYS", 4, minimum=1, maximum=90),
        "log_dir": str(resolve_log_dir()),
    }


def resolve_log_dir() -> Path:
    configured = os.getenv("ALPHARAVIS_LOG_DIR", "logs").strip() or "logs"
    path = Path(os.path.expandvars(os.path.expanduser(configured)))
    if path.is_absolute():
        return path

    workspace = os.getenv("ALPHARAVIS_WORKSPACE_DIR", "").strip()
    if workspace:
        base = Path(workspace)
    elif Path("/workspace").exists():
        base = Path("/workspace")
    else:
        base = Path(__file__).resolve().parents[1]
    return (base / path).resolve()


def get_logger(component: str = "alpharavis") -> logging.Logger:
    setup_logging(component=component)
    return logging.getLogger(OPERATIONAL_LOGGER_NAME)


def log_event(
    level: int | str,
    event: str,
    *,
    component: str = "alpharavis",
    message: str = "",
    debug: bool = False,
    **fields: Any,
) -> None:
    if not _SETUP_DONE:
        setup_logging(component=component)

    levelno = _coerce_level(level)
    base_payload = {
        "event": str(event),
        "component": str(component),
        "timestamp": _utc_now(),
        **fields,
    }
    operational_max_chars = _positive_int_env(
        "ALPHARAVIS_LOG_MAX_FIELD_CHARS",
        DEFAULT_MAX_FIELD_CHARS,
        minimum=256,
        maximum=100000,
    )
    operational_payload = redact_for_logs(
        {key: value for key, value in base_payload.items() if key != "traceback"},
        max_field_chars=operational_max_chars,
    )
    record_message = str(redact_for_logs(message or str(event), max_field_chars=operational_max_chars))

    if env_bool("ALPHARAVIS_OPERATIONAL_LOGGING", "true"):
        logging.getLogger(OPERATIONAL_LOGGER_NAME).log(
            levelno,
            record_message,
            extra={"alpharavis_event": operational_payload},
        )

    if env_bool("ALPHARAVIS_DEBUG_ALL_LOGGING", "false"):
        debug_payload = redact_for_logs(
            {**base_payload, "debug_event": bool(debug)},
            max_field_chars=_positive_int_env(
                "ALPHARAVIS_DEBUG_LOG_MAX_FIELD_CHARS",
                DEFAULT_DEBUG_MAX_FIELD_CHARS,
                minimum=512,
                maximum=250000,
            ),
        )
        logging.getLogger(DEBUG_LOGGER_NAME).log(
            levelno,
            record_message,
            extra={"alpharavis_event": debug_payload},
        )


def log_debug_event(event: str, *, component: str = "alpharavis", message: str = "", **fields: Any) -> None:
    if env_bool("ALPHARAVIS_DEBUG_ALL_LOGGING", "false"):
        log_event(logging.DEBUG, event, component=component, message=message, debug=True, **fields)


def log_dependency_status(
    dependency: str,
    status: str,
    *,
    component: str = "alpharavis",
    level: int | str = logging.INFO,
    message: str = "",
    **fields: Any,
) -> None:
    log_event(
        level,
        "dependency.status",
        component=component,
        message=message or f"{dependency}: {status}",
        dependency=dependency,
        status=status,
        **fields,
    )


def log_exception(
    event: str,
    exc: BaseException,
    *,
    component: str = "alpharavis",
    level: int | str = logging.ERROR,
    message: str = "",
    **fields: Any,
) -> None:
    payload = {
        "error_type": type(exc).__name__,
        "error": str(exc),
        **fields,
    }
    if env_bool("ALPHARAVIS_DEBUG_ALL_LOGGING", "false"):
        payload["traceback"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    log_event(level, event, component=component, message=message or str(exc), debug=True, **payload)


def redact_for_logs(value: Any, *, max_field_chars: int = DEFAULT_MAX_FIELD_CHARS) -> Any:
    return _redact(value, max_field_chars=max_field_chars, depth=0)


def reset_logging_for_tests() -> None:
    global _SETUP_DONE
    for name in (OPERATIONAL_LOGGER_NAME, DEBUG_LOGGER_NAME):
        logger = logging.getLogger(name)
        _remove_managed_handlers(logger)
        logger.handlers.clear()
        logger.propagate = False
    _SETUP_DONE = False


def _attach_text_handler(logger: logging.Logger, path: Path, *, retention_days: int, level: int) -> None:
    handler = TimedRotatingFileHandler(
        path,
        when="midnight",
        interval=1,
        backupCount=retention_days,
        encoding="utf-8",
        utc=True,
    )
    handler.setLevel(level)
    handler.setFormatter(_TextEventFormatter())
    _mark_managed(handler)
    logger.addHandler(handler)


def _attach_jsonl_handler(logger: logging.Logger, path: Path, *, retention_days: int, level: int) -> None:
    handler = TimedRotatingFileHandler(
        path,
        when="midnight",
        interval=1,
        backupCount=retention_days,
        encoding="utf-8",
        utc=True,
    )
    handler.setLevel(level)
    handler.setFormatter(_JsonEventFormatter())
    _mark_managed(handler)
    logger.addHandler(handler)


def _attach_console_handler(logger: logging.Logger, *, level: int) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(_ColorConsoleFormatter())
    _mark_managed(handler)
    logger.addHandler(handler)


def _mark_managed(handler: logging.Handler) -> None:
    setattr(handler, "_alpharavis_managed", True)


def _remove_managed_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if getattr(handler, "_alpharavis_managed", False):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass


class _TextEventFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, "alpharavis_event", {}) or {}
        component = event.get("component") or record.name
        event_name = event.get("event") or ""
        extra = {
            key: value
            for key, value in event.items()
            if key not in {"timestamp", "component", "event"} and value is not None and value != ""
        }
        details = " ".join(f"{key}={_compact(value)}" for key, value in sorted(extra.items()))
        base = f"{event.get('timestamp') or _utc_now()} | {record.levelname:<8} | {component} | {event_name}"
        if record.getMessage():
            base += f" | {record.getMessage()}"
        if details:
            base += f" | {details}"
        return base


class _JsonEventFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, "alpharavis_event", {}) or {}
        payload = {
            "ts": event.get("timestamp") or _utc_now(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **event,
        }
        return json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True)


class _ColorConsoleFormatter(_TextEventFormatter):
    COLORS = {
        logging.DEBUG: "\033[90m",
        logging.INFO: "\033[0m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[1;31m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        color = self.COLORS.get(record.levelno, self.RESET)
        return f"{color}{text}{self.RESET}"


def _redact(value: Any, *, max_field_chars: int, depth: int) -> Any:
    if depth > 8:
        return "[max-depth]"
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            key_str = str(key)
            if SECRET_KEY_RE.search(key_str):
                output[key_str] = "[redacted]"
            else:
                output[key_str] = _redact(item, max_field_chars=max_field_chars, depth=depth + 1)
        return output
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        max_items = 50 if max_field_chars <= DEFAULT_MAX_FIELD_CHARS else 200
        output = [_redact(item, max_field_chars=max_field_chars, depth=depth + 1) for item in items[:max_items]]
        if len(items) > max_items:
            output.append(f"[{len(items) - max_items} more items omitted]")
        return output
    if isinstance(value, (str, bytes)):
        text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
        text = SECRET_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}=[redacted]", text)
        text = BEARER_RE.sub("Bearer [redacted]", text)
        if len(text) > max_field_chars:
            omitted = len(text) - max_field_chars
            return text[:max_field_chars].rstrip() + f"... [{omitted} chars omitted]"
        return text
    return value


def _compact(value: Any) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    else:
        text = str(value)
    return text.replace("\n", "\\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _positive_int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


def _level_from_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip().upper()
    if not raw:
        return default
    return getattr(logging, raw, default)


def _coerce_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).strip().upper(), logging.INFO)
