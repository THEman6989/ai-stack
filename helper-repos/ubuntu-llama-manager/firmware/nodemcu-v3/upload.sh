#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PORT="${PORT:-/dev/ttyUSB0}"
FQBN="${FQBN:-esp8266:esp8266:nodemcuv2}"
PATH="$HOME/.local/bin:$PATH"

"$SCRIPT_DIR/build.sh"
arduino-cli upload -p "$PORT" --fqbn "$FQBN" "$SCRIPT_DIR"
