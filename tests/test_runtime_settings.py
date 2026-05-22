from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

import runtime_settings  # noqa: E402


def test_apply_runtime_overrides_updates_environment(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "runtime_settings.json"
    path.write_text(json.dumps({"values": {"ALPHARAVIS_TEST_RUNTIME_FLAG": "enabled"}}), encoding="utf-8")
    monkeypatch.delenv("ALPHARAVIS_TEST_RUNTIME_FLAG", raising=False)

    result = runtime_settings.apply_runtime_overrides(path)

    assert result["applied"] == 1
    assert os.environ["ALPHARAVIS_TEST_RUNTIME_FLAG"] == "enabled"


def test_load_runtime_overrides_tolerates_missing_or_invalid(tmp_path: Path) -> None:
    assert runtime_settings.load_runtime_overrides(tmp_path / "missing.json") == {}
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert runtime_settings.load_runtime_overrides(invalid) == {}
