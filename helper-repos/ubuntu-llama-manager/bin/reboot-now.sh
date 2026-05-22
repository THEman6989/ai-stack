#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=bin/common.sh
. "$SCRIPT_DIR/common.sh"

load_config

REBOOT_LOG="$UBUNTU_MANAGER_DIR/logs/reboot.log"
ENABLE_AUTO_REBOOT="${ENABLE_AUTO_REBOOT:-${ENABLE_REBOOT_TIMER:-true}}"
REBOOT_COMMAND="${REBOOT_COMMAND:-/usr/bin/systemctl reboot}"
REBOOT_GRACE_SECONDS="$(config_int REBOOT_GRACE_SECONDS 10)"
REBOOT_USE_ESP_POWER_CYCLE="${REBOOT_USE_ESP_POWER_CYCLE:-false}"
REBOOT_REQUIRE_ESP_WEBHOOK="${REBOOT_REQUIRE_ESP_WEBHOOK:-false}"
REBOOT_LOCAL_SHUTDOWN_AFTER_ESP="${REBOOT_LOCAL_SHUTDOWN_AFTER_ESP:-true}"
GPU_FAULT_LOCAL_SHUTDOWN_AFTER_ESP="${GPU_FAULT_LOCAL_SHUTDOWN_AFTER_ESP:-true}"
ESP_REBOOT_ACTION="${ESP_POWER_ACTION_ON_REBOOT:-power-on}"
ESP_GPU_FAULT_ACTION="${ESP_POWER_ACTION_ON_GPU_FAULT:-power-cycle}"
ESP_REBOOT_SHUTDOWN_COMMAND="${REBOOT_ESP_SHUTDOWN_COMMAND:-/usr/bin/systemctl poweroff}"
GPU_FAULT_SHUTDOWN_COMMAND="${GPU_FAULT_SHUTDOWN_COMMAND:-/usr/bin/systemctl poweroff}"
ESP_NOTIFY_SETTLE_SECONDS="$(config_int ESP_NOTIFY_SETTLE_SECONDS 2)"
trigger_reason="${1:-manual-or-timer}"

case "$trigger_reason" in
  systemd-timer|watchdog)
    if ! bool_true "$ENABLE_AUTO_REBOOT"; then
      log_to_file "$REBOOT_LOG" "Auto reboot disabled; automatic reboot request skipped."
      exit 0
    fi
    ;;
esac

json_escape() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//$'\n'/\\n}"
  printf '%s' "$value"
}

write_esp_request() {
  local action="$1"
  local reason="$2"
  local request_file="$UBUNTU_STATE_DIR/esp-request.json"
  ensure_state_dir
  cat > "$request_file" <<EOF
{
  "action": "$(json_escape "$action")",
  "ok": true,
  "payload": {
    "delay_before_action_seconds": $(config_int ESP_POWER_DELAY_BEFORE_ACTION_SECONDS 30),
    "hold_seconds": $(config_int ESP_POWER_HOLD_SECONDS 12),
    "reason": "$(json_escape "$reason")",
    "requested_by": "reboot-now",
    "wait_seconds": $(config_int ESP_POWER_WAIT_SECONDS 20)
  },
  "requested_at": "$(date --iso-8601=seconds)",
  "status": "queued",
  "note": "ESP should power-cycle the host after the configured delay."
}
EOF
  log_to_file "$REBOOT_LOG" "Queued ESP request: $request_file"
}

send_esp_webhook() {
  local action="$1"
  local reason="$2"
  local url="${ESP_WEBHOOK_URL:-}"
  local token="${ESP_WEBHOOK_TOKEN:-}"
  local data
  local curl_args

  if [[ -z "$url" ]]; then
    log_to_file "$REBOOT_LOG" "ESP_WEBHOOK_URL empty; only local ESP request was queued."
    return 2
  fi

  if ! command -v curl >/dev/null 2>&1; then
    log_to_file "$REBOOT_LOG" "curl not found; cannot send ESP webhook."
    return 3
  fi

  data="$(printf '{"action":"%s","reason":"%s","requested_by":"reboot-now","hold_seconds":%s,"wait_seconds":%s,"delay_before_action_seconds":%s}\n' \
    "$(json_escape "$action")" \
    "$(json_escape "$reason")" \
    "$(config_int ESP_POWER_HOLD_SECONDS 12)" \
    "$(config_int ESP_POWER_WAIT_SECONDS 20)" \
    "$(config_int ESP_POWER_DELAY_BEFORE_ACTION_SECONDS 30)")"

  curl_args=(-fsS --max-time 5 -X POST -H "Content-Type: application/json" -d "$data")
  if [[ -n "$token" ]]; then
    curl_args+=(-H "Authorization: Bearer $token")
  fi

  if curl "${curl_args[@]}" "$url" >> "$REBOOT_LOG" 2>&1; then
    log_to_file "$REBOOT_LOG" "ESP webhook sent successfully: $url"
    return 0
  fi

  log_to_file "$REBOOT_LOG" "ESP webhook failed: $url"
  return 1
}

notify_esp_for_reboot() {
  local reason="$1"
  local webhook_status=0

  write_esp_request "$ESP_REBOOT_ACTION" "$reason"
  send_esp_webhook "$ESP_REBOOT_ACTION" "$reason" || webhook_status=$?

  if bool_true "$REBOOT_REQUIRE_ESP_WEBHOOK" && (( webhook_status != 0 )); then
    log_to_file "$REBOOT_LOG" "REBOOT_REQUIRE_ESP_WEBHOOK=true and ESP webhook was not confirmed. Aborting shutdown."
    exit 1
  fi

  if (( ESP_NOTIFY_SETTLE_SECONDS > 0 )); then
    log_to_file "$REBOOT_LOG" "Waiting ${ESP_NOTIFY_SETTLE_SECONDS}s after ESP notification."
    sleep "$ESP_NOTIFY_SETTLE_SECONDS"
  fi
}

log_to_file "$REBOOT_LOG" "Controlled reboot requested by ${trigger_reason}."

if bool_true "${STOP_LLAMA_BEFORE_REBOOT:-false}"; then
  log_to_file "$REBOOT_LOG" "Stopping ubuntu-llama.service before reboot."
  systemctl stop ubuntu-llama.service >/dev/null 2>&1 || true
fi

if bool_true "$REBOOT_USE_ESP_POWER_CYCLE"; then
  case "$trigger_reason" in
    gpu-health)
      ESP_REBOOT_ACTION="$ESP_GPU_FAULT_ACTION"
      REBOOT_COMMAND="$GPU_FAULT_SHUTDOWN_COMMAND"
      local_shutdown_after_esp="$GPU_FAULT_LOCAL_SHUTDOWN_AFTER_ESP"
      ;;
    *)
      REBOOT_COMMAND="$ESP_REBOOT_SHUTDOWN_COMMAND"
      local_shutdown_after_esp="$REBOOT_LOCAL_SHUTDOWN_AFTER_ESP"
      ;;
  esac

  notify_esp_for_reboot "$trigger_reason"
  if ! bool_true "$local_shutdown_after_esp"; then
    log_to_file "$REBOOT_LOG" "Local shutdown after ESP notification disabled; ESP is responsible for the next power action."
    exit 0
  fi
fi

sync || true
if (( REBOOT_GRACE_SECONDS > 0 )); then
  log_to_file "$REBOOT_LOG" "Waiting ${REBOOT_GRACE_SECONDS}s before reboot."
  sleep "$REBOOT_GRACE_SECONDS"
fi

log_to_file "$REBOOT_LOG" "Executing reboot command: $REBOOT_COMMAND"
exec bash -lc "$REBOOT_COMMAND"
