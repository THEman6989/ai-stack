#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
FQBN="${FQBN:-esp8266:esp8266:nodemcuv2}"
ESP8266_INDEX_URL="${ESP8266_INDEX_URL:-https://arduino.esp8266.com/stable/package_esp8266com_index.json}"
PATH="$HOME/.local/bin:$PATH"

if [[ -f "$SCRIPT_DIR/.env" ]]; then
  "$SCRIPT_DIR/generate-config.sh"
elif [[ ! -f "$SCRIPT_DIR/config.generated.h" ]]; then
  printf 'Missing %s/config.generated.h and %s/.env\n' "$SCRIPT_DIR" "$SCRIPT_DIR" >&2
  printf 'Create local env first:\n  cp %s/.env.example %s/.env\n  nano %s/.env\n' "$SCRIPT_DIR" "$SCRIPT_DIR" "$SCRIPT_DIR" >&2
  exit 1
fi

if ! command -v arduino-cli >/dev/null 2>&1; then
  printf 'arduino-cli not found.\n' >&2
  printf 'Install it, then re-run this script. See docs/esp-nodemcu.md.\n' >&2
  exit 1
fi

arduino-cli core update-index --additional-urls "$ESP8266_INDEX_URL"
arduino-cli core install esp8266:esp8266 --additional-urls "$ESP8266_INDEX_URL"
arduino-cli compile --fqbn "$FQBN" "$SCRIPT_DIR"
