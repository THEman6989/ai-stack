#!/usr/bin/env bash
set -euo pipefail

HERMES_PATCH_TARGET_DIR=/opt/hermes /workspace/scripts/apply_hermes_agent_patches.sh

# AlphaRavis runs Hermes as a Compose-managed sidecar. Hermes keeps a persistent
# config.yaml in HERMES_HOME; if that file was created by upstream defaults it
# can override the Compose env and route smoke tests to OpenRouter. Keep only
# the model provider fields aligned with the operator's .env values.
python3 - <<'PY'
import json
import os
import re
from pathlib import Path


home = Path(os.environ.get("HERMES_HOME") or "/opt/data")
config_path = home / "config.yaml"
if not config_path.exists():
    raise SystemExit(0)

updates = {
    "default": os.environ.get("HERMES_INFERENCE_MODEL", "").strip(),
    "provider": os.environ.get("HERMES_INFERENCE_PROVIDER", "").strip(),
    "base_url": os.environ.get("OPENAI_BASE_URL", "").strip(),
    # Hermes reads custom-provider credentials from the persisted model config.
    # Keep it aligned with the Compose-provided LiteLLM key so old configs with
    # placeholder values (for example "no-key-required") do not override env.
    "api_key": (
        os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("HERMES_OPENAI_API_KEY", "").strip()
    ),
}
updates = {key: value for key, value in updates.items() if value}
if not updates:
    raise SystemExit(0)

text = config_path.read_text(encoding="utf-8")
original = text


def yaml_string(value: str) -> str:
    return json.dumps(value)


def replace_top_level_model_key(source: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(^model:\n(?:^[ \t].*\n)*?)(^[ \t]*{re.escape(key)}:\s*.*$)", re.MULTILINE)
    match = pattern.search(source)
    if match:
        indent = re.match(r"^[ \t]*", match.group(2)).group(0)
        return source[: match.start(2)] + f"{indent}{key}: {yaml_string(value)}" + source[match.end(2) :]

    model_match = re.search(r"^model:\s*$", source, flags=re.MULTILINE)
    if not model_match:
        return "model:\n" + f"  {key}: {yaml_string(value)}\n" + source
    insert_at = source.find("\n", model_match.end())
    if insert_at == -1:
        return source.rstrip() + f"\n  {key}: {yaml_string(value)}\n"
    return source[: insert_at + 1] + f"  {key}: {yaml_string(value)}\n" + source[insert_at + 1 :]


for key, value in updates.items():
    text = replace_top_level_model_key(text, key, value)

if text != original:
    config_path.write_text(text, encoding="utf-8")
    print(
        "Synced Hermes model config from AlphaRavis env: "
        + ", ".join(f"{key}={value}" for key, value in updates.items())
    )
PY

exec /opt/hermes/docker/entrypoint.sh "$@"
