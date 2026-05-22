#!/usr/bin/env bash

set -Eeuo pipefail

COMMON_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
UBUNTU_MANAGER_DIR="$(cd -- "$COMMON_DIR/.." && pwd -P)"
UBUNTU_CONFIG="${UBUNTU_CONFIG:-${RAKAM_CONFIG:-$UBUNTU_MANAGER_DIR/ubuntu-llama.conf}}"
UBUNTU_STATE_DIR="${UBUNTU_STATE_DIR:-${RAKAM_STATE_DIR:-$UBUNTU_MANAGER_DIR/state}}"

load_config() {
  if [[ ! -r "$UBUNTU_CONFIG" ]]; then
    printf 'Config not readable: %s\n' "$UBUNTU_CONFIG" >&2
    printf 'Create one from .env.example, for example:\n  cp %s/.env.example %s\n' "$UBUNTU_MANAGER_DIR" "$UBUNTU_CONFIG" >&2
    exit 1
  fi

  set -a
  # shellcheck source=/dev/null
  . "$UBUNTU_CONFIG"
  set +a

  local esp_env_file="${ESP_ENV_FILE:-$UBUNTU_MANAGER_DIR/firmware/nodemcu-v3/.env}"
  if [[ -r "$esp_env_file" ]]; then
    set -a
    # shellcheck source=/dev/null
    . "$esp_env_file"
    set +a
  fi

  if [[ -n "${MANAGER_API_TOKEN:-}" && ( -z "${API_TOKEN:-}" || "${API_TOKEN:-}" == "change-me" ) ]]; then
    API_TOKEN="$MANAGER_API_TOKEN"
  fi

  if [[ -n "${ESP_AUTH_TOKEN:-}" && ( -z "${ESP_WEBHOOK_TOKEN:-}" || "${ESP_WEBHOOK_TOKEN:-}" == "change-me-esp-token" ) ]]; then
    ESP_WEBHOOK_TOKEN="$ESP_AUTH_TOKEN"
  fi
}

bool_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y|on|ON|enabled|ENABLED) return 0 ;;
    *) return 1 ;;
  esac
}

config_bool() {
  local name="$1"
  local fallback="${2:-false}"
  local value="${!name:-$fallback}"
  if bool_true "$value"; then
    printf 'true\n'
  else
    printf 'false\n'
  fi
}

config_int() {
  local name="$1"
  local fallback="$2"
  local value="${!name:-$fallback}"
  if [[ "$value" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "$value"
  else
    printf '%s\n' "$fallback"
  fi
}

reboot_interval_seconds() {
  if [[ "${REBOOT_INTERVAL_SECONDS:-}" =~ ^[0-9]+$ ]] && (( REBOOT_INTERVAL_SECONDS >= 60 )); then
    printf '%s\n' "$REBOOT_INTERVAL_SECONDS"
    return
  fi

  if [[ "${REBOOT_INTERVAL_HOURS:-}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    awk -v hours="$REBOOT_INTERVAL_HOURS" 'BEGIN { seconds = int(hours * 3600); if (seconds < 60) seconds = 60; print seconds }'
    return
  fi

  if [[ "${REBOOT_AFTER_SECONDS:-}" =~ ^[0-9]+$ ]] && (( REBOOT_AFTER_SECONDS >= 60 )); then
    printf '%s\n' "$REBOOT_AFTER_SECONDS"
    return
  fi

  printf '10800\n'
}

ensure_state_dir() {
  mkdir -p -- "$UBUNTU_STATE_DIR"
}

require_value() {
  local name="$1"
  local value="${2:-}"
  if [[ -z "$value" ]]; then
    printf 'Missing required config value: %s\n' "$name" >&2
    exit 1
  fi
}

ensure_log_parent() {
  local log_file="$1"
  local log_dir
  log_dir="$(dirname -- "$log_file")"
  mkdir -p -- "$log_dir"
}

log_to_file() {
  local log_file="$1"
  shift
  ensure_log_parent "$log_file"
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" >> "$log_file"
}
