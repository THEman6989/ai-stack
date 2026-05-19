#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml


def _resolve_model_ref(value: Any) -> str:
    text = str(value or "").strip()
    prefix = "os.environ/"
    if text.startswith(prefix):
        return os.getenv(text[len(prefix) :], "").strip()
    return text


def _uses_ollama(model_ref: Any) -> bool:
    return _resolve_model_ref(model_ref).lower().startswith("ollama/")


def render_config(input_path: Path, output_path: Path) -> None:
    config = yaml.safe_load(input_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{input_path} did not contain a YAML object")

    settings = config.get("litellm_settings")
    if isinstance(settings, dict):
        settings.pop("drop_params", None)

    for route in config.get("model_list") or []:
        if not isinstance(route, dict):
            continue
        params = route.get("litellm_params")
        if not isinstance(params, dict):
            continue
        if _uses_ollama(params.get("model")):
            params["drop_params"] = True
        else:
            params.pop("drop_params", None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: render_litellm_config.py INPUT_YAML OUTPUT_YAML", file=sys.stderr)
        return 2
    render_config(Path(sys.argv[1]), Path(sys.argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
