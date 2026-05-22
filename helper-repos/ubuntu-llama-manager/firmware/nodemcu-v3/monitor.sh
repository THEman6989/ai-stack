#!/usr/bin/env bash

set -Eeuo pipefail

PORT="${PORT:-/dev/ttyUSB0}"
BAUD="${BAUD:-9600}"
DTR="${DTR:-off}"
RTS="${RTS:-off}"
PATH="$HOME/.local/bin:$PATH"

if command -v arduino-cli >/dev/null 2>&1; then
  exec arduino-cli monitor -p "$PORT" -c "baudrate=$BAUD" -c "dtr=$DTR" -c "rts=$RTS"
fi

if command -v screen >/dev/null 2>&1; then
  exec screen "$PORT" "$BAUD"
fi

printf 'Need arduino-cli or screen for serial monitor.\n' >&2
exit 1
