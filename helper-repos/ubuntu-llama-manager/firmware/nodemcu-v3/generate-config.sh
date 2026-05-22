#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env}"
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/config.generated.h}"

if [[ ! -r "$ENV_FILE" ]]; then
  printf 'Missing %s\n' "$ENV_FILE" >&2
  printf 'Create it first:\n  cp %s/.env.example %s/.env\n' "$SCRIPT_DIR" "$SCRIPT_DIR" >&2
  exit 1
fi

set -a
# shellcheck source=/dev/null
. "$ENV_FILE"
set +a

required() {
  local name="$1"
  local value="${!name:-}"
  if [[ -z "$value" ]]; then
    printf 'Missing required value in %s: %s\n' "$ENV_FILE" "$name" >&2
    exit 1
  fi
}

c_string() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//$'\n'/\\n}"
  printf '"%s"' "$value"
}

c_literal() {
  local value="$1"
  printf '%s' "$value"
}

required WIFI_SSID
required WIFI_PASSWORD
required MANAGER_BASE_URL
required MANAGER_API_TOKEN
required ESP_AUTH_TOKEN

cat > "$CONFIG_FILE" <<EOF
#pragma once

// Generated from firmware/nodemcu-v3/.env by generate-config.sh.
// Do not edit manually. Do not commit this file.

#define WIFI_SSID $(c_string "$WIFI_SSID")
#define WIFI_PASSWORD $(c_string "$WIFI_PASSWORD")

#define MANAGER_BASE_URL $(c_string "$MANAGER_BASE_URL")
#define MANAGER_API_TOKEN $(c_string "$MANAGER_API_TOKEN")

#define ESP_AUTH_TOKEN $(c_string "$ESP_AUTH_TOKEN")

#define ESP_DEVICE_ID $(c_string "${ESP_DEVICE_ID:-nodemcu-v3-main}")

#define POWER_BUTTON_PIN $(c_literal "${POWER_BUTTON_PIN:-D1}")
#define RESET_BUTTON_PIN $(c_literal "${RESET_BUTTON_PIN:-D2}")
#define STATUS_LED_PIN $(c_literal "${STATUS_LED_PIN:-LED_BUILTIN}")

#define OUTPUT_ACTIVE_HIGH $(c_literal "${OUTPUT_ACTIVE_HIGH:-true}")
#define GPIO_IDLE_FLOAT $(c_literal "${GPIO_IDLE_FLOAT:-true}")

#define DEFAULT_SHORT_PRESS_SECONDS $(c_literal "${DEFAULT_SHORT_PRESS_SECONDS:-1}")
#define DEFAULT_FORCE_OFF_HOLD_SECONDS $(c_literal "${DEFAULT_FORCE_OFF_HOLD_SECONDS:-8}")
#define DEFAULT_WAIT_AFTER_OFF_SECONDS $(c_literal "${DEFAULT_WAIT_AFTER_OFF_SECONDS:-20}")
#define DEFAULT_DELAY_BEFORE_ACTION_SECONDS $(c_literal "${DEFAULT_DELAY_BEFORE_ACTION_SECONDS:-30}")
#define HEARTBEAT_INTERVAL_SECONDS $(c_literal "${HEARTBEAT_INTERVAL_SECONDS:-30}")
EOF

printf 'Generated %s from %s\n' "$CONFIG_FILE" "$ENV_FILE"
