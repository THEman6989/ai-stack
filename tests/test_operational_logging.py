from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

from operational_logging import (  # noqa: E402
    log_debug_event,
    log_event,
    log_exception,
    logging_status,
    redact_for_logs,
    reset_logging_for_tests,
    setup_logging,
)


LOG_ENV_KEYS = [
    "ALPHARAVIS_LOG_DIR",
    "ALPHARAVIS_OPERATIONAL_LOGGING",
    "ALPHARAVIS_DEBUG_ALL_LOGGING",
    "ALPHARAVIS_LOG_CONSOLE",
    "ALPHARAVIS_LOG_JSONL",
    "ALPHARAVIS_LOG_RETENTION_DAYS",
    "ALPHARAVIS_LOG_MAX_FIELD_CHARS",
    "ALPHARAVIS_DEBUG_LOG_MAX_FIELD_CHARS",
]


class _EnvGuard:
    def __enter__(self):
        self.old_values = {key: os.environ.get(key) for key in LOG_ENV_KEYS}
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        for key, value in self.old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        reset_logging_for_tests()


def _configure_tmp_logging(tmp: Path, *, debug: bool = True, max_chars: int = 4000) -> None:
    os.environ["ALPHARAVIS_LOG_DIR"] = str(tmp)
    os.environ["ALPHARAVIS_OPERATIONAL_LOGGING"] = "true"
    os.environ["ALPHARAVIS_DEBUG_ALL_LOGGING"] = "true" if debug else "false"
    os.environ["ALPHARAVIS_LOG_CONSOLE"] = "false"
    os.environ["ALPHARAVIS_LOG_JSONL"] = "true"
    os.environ["ALPHARAVIS_LOG_RETENTION_DAYS"] = "4"
    os.environ["ALPHARAVIS_LOG_MAX_FIELD_CHARS"] = str(max_chars)
    os.environ["ALPHARAVIS_DEBUG_LOG_MAX_FIELD_CHARS"] = str(max(max_chars, 512))
    reset_logging_for_tests()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_operational_logging_writes_text_and_jsonl_with_redaction() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir, _EnvGuard():
        tmp = Path(tmp_dir)
        _configure_tmp_logging(tmp)

        status = setup_logging(component="test")
        log_event(
            logging.INFO,
            "test.event",
            component="test",
            api_key="sk-secret-value",
            nested={"authorization": "Bearer secret-token", "safe": "kept"},
        )

        text_path = tmp / "operational" / "alpharavis.log"
        jsonl_path = tmp / "operational" / "alpharavis.jsonl"
        text = text_path.read_text(encoding="utf-8")
        records = _read_jsonl(jsonl_path)

    assert status["retention_days"] == 4
    assert "test.event" in text
    assert "sk-secret-value" not in text
    assert "secret-token" not in text
    assert records[-1]["event"] == "test.event"
    assert records[-1]["api_key"] == "[redacted]"
    assert records[-1]["nested"]["authorization"] == "[redacted]"
    assert records[-1]["nested"]["safe"] == "kept"


def test_debug_all_logging_is_separate_and_optional() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir, _EnvGuard():
        tmp = Path(tmp_dir)
        _configure_tmp_logging(tmp, debug=True)

        setup_logging(component="test")
        log_debug_event("debug.event", component="test", details={"x": 1})

        debug_jsonl = tmp / "debug" / "alpharavis-debug.jsonl"
        records = _read_jsonl(debug_jsonl)

    assert records[-1]["event"] == "debug.event"
    assert records[-1]["debug_event"] is True


def test_debug_all_logging_can_be_disabled() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir, _EnvGuard():
        tmp = Path(tmp_dir)
        _configure_tmp_logging(tmp, debug=False)

        setup_logging(component="test")
        log_debug_event("debug.disabled", component="test")

        debug_dir = tmp / "debug"

    assert not debug_dir.exists()


def test_log_exception_includes_traceback_only_in_debug_log() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir, _EnvGuard():
        tmp = Path(tmp_dir)
        _configure_tmp_logging(tmp, debug=True)

        setup_logging(component="test")
        try:
            raise RuntimeError("backend token=secret failed")
        except RuntimeError as exc:
            log_exception("test.exception", exc, component="test")

        operational = _read_jsonl(tmp / "operational" / "alpharavis.jsonl")
        debug = _read_jsonl(tmp / "debug" / "alpharavis-debug.jsonl")

    assert operational[-1]["event"] == "test.exception"
    assert "traceback" not in operational[-1]
    assert "traceback" in debug[-1]
    assert "token=secret" not in json.dumps(debug[-1], ensure_ascii=False)


def test_log_field_truncation_and_direct_redaction() -> None:
    redacted = redact_for_logs(
        {
            "safe": "x" * 80,
            "message": "password=supersecret and useful",
            "items": [{"token": "abc"}, "Bearer rawsecret"],
        },
        max_field_chars=32,
    )

    assert "chars omitted" in redacted["safe"]
    assert "supersecret" not in redacted["message"]
    assert redacted["items"][0]["token"] == "[redacted]"
    assert redacted["items"][1] == "Bearer [redacted]"


def _run_all() -> None:
    tests = [
        test_operational_logging_writes_text_and_jsonl_with_redaction,
        test_debug_all_logging_is_separate_and_optional,
        test_debug_all_logging_can_be_disabled,
        test_log_exception_includes_traceback_only_in_debug_log,
        test_log_field_truncation_and_direct_redaction,
    ]
    for test in tests:
        test()


if __name__ == "__main__":
    _run_all()
