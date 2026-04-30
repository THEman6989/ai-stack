from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

from file_safety import (  # noqa: E402
    FileSafetyError,
    check_path_safety,
    ensure_list_allowed,
    ensure_read_allowed,
    ensure_write_allowed,
)


def test_sensitive_env_read_is_blocked() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        secret = root / ".env"
        secret.write_text("TOKEN=secret", encoding="utf-8")

        decision = check_path_safety(secret, operation="read", allowed_root=root)

    assert not decision.allowed
    assert decision.category == "sensitive_file"
    assert "secret" in decision.reason


def test_sensitive_ssh_write_is_blocked() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        target = root / ".ssh" / "config"

        try:
            ensure_write_allowed(target, allowed_root=root)
        except FileSafetyError as exc:
            message = str(exc)
        else:
            raise AssertionError("expected FileSafetyError")

    assert "credential" in message or ".ssh" in message


def test_safe_write_root_blocks_outside_path() -> None:
    old_value = os.environ.get("ALPHARAVIS_WRITE_SAFE_ROOT")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            safe = root / "safe"
            outside = root / "outside" / "note.md"
            os.environ["ALPHARAVIS_WRITE_SAFE_ROOT"] = str(safe)

            decision = check_path_safety(outside, operation="write", allowed_root=root)
    finally:
        if old_value is None:
            os.environ.pop("ALPHARAVIS_WRITE_SAFE_ROOT", None)
        else:
            os.environ["ALPHARAVIS_WRITE_SAFE_ROOT"] = old_value

    assert not decision.allowed
    assert decision.category == "safe_write_root"


def test_safe_write_root_allows_inside_path() -> None:
    old_value = os.environ.get("ALPHARAVIS_WRITE_SAFE_ROOT")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            safe = root / "safe"
            target = safe / "note.md"
            os.environ["ALPHARAVIS_WRITE_SAFE_ROOT"] = str(safe)

            decision = ensure_write_allowed(target, allowed_root=root)
    finally:
        if old_value is None:
            os.environ.pop("ALPHARAVIS_WRITE_SAFE_ROOT", None)
        else:
            os.environ["ALPHARAVIS_WRITE_SAFE_ROOT"] = old_value

    assert decision.allowed


def test_internal_cache_listing_is_blocked() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        git_dir = root / ".git"
        git_dir.mkdir()

        try:
            ensure_list_allowed(git_dir, allowed_root=root)
        except FileSafetyError as exc:
            message = str(exc)
        else:
            raise AssertionError("expected FileSafetyError")

    assert "internal" in message


def test_normal_workspace_read_is_allowed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "src" / "app.py"
        src.parent.mkdir()
        src.write_text("print('ok')", encoding="utf-8")

        decision = ensure_read_allowed(src, allowed_root=root)

    assert decision.allowed


def _run_all() -> None:
    tests = [
        test_sensitive_env_read_is_blocked,
        test_sensitive_ssh_write_is_blocked,
        test_safe_write_root_blocks_outside_path,
        test_safe_write_root_allows_inside_path,
        test_internal_cache_listing_is_blocked,
        test_normal_workspace_read_is_allowed,
    ]
    for test in tests:
        test()


if __name__ == "__main__":
    _run_all()
