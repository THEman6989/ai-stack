from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_RUNTIME_SETTINGS_PATH = "/workspace/service-dashboard-data/runtime_settings.json"


def runtime_settings_path() -> Path:
    return Path(os.getenv("ALPHARAVIS_RUNTIME_SETTINGS_FILE", DEFAULT_RUNTIME_SETTINGS_PATH))


def load_runtime_overrides(path: Path | None = None) -> dict[str, str]:
    target = path or runtime_settings_path()
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}
    values = payload.get("values") if isinstance(payload, dict) else {}
    if not isinstance(values, dict):
        return {}
    return {
        str(key): "" if value is None else str(value)
        for key, value in values.items()
        if str(key).strip()
    }


def apply_runtime_overrides(path: Path | None = None) -> dict[str, Any]:
    values = load_runtime_overrides(path)
    for key, value in values.items():
        os.environ[key] = value
    return {"applied": len(values), "keys": sorted(values)}
