#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=bin/common.sh
. "$SCRIPT_DIR/common.sh"

load_config

LOG_FILE="${LLAMA_SECONDARY_LOG_FILE:-$UBUNTU_MANAGER_DIR/logs/llama-8001.log}"
ensure_log_parent "$LOG_FILE"

if ! bool_true "${ENABLE_LLAMA_SECONDARY_SERVICE:-false}"; then
  log_to_file "$LOG_FILE" "ENABLE_LLAMA_SECONDARY_SERVICE=false, service is idling."
  exec sleep infinity
fi

require_value "LLAMA_SECONDARY_WORKDIR" "${LLAMA_SECONDARY_WORKDIR:-}"
require_value "LLAMA_SECONDARY_COMMAND" "${LLAMA_SECONDARY_COMMAND:-}"

if [[ -n "${LLAMA_SECONDARY_ENV_FILE:-}" ]]; then
  if [[ -r "$LLAMA_SECONDARY_ENV_FILE" ]]; then
    log_to_file "$LOG_FILE" "Loading LLAMA_SECONDARY_ENV_FILE: $LLAMA_SECONDARY_ENV_FILE"
    set -a
    # shellcheck source=/dev/null
    . "$LLAMA_SECONDARY_ENV_FILE"
    set +a
  else
    log_to_file "$LOG_FILE" "ERROR: LLAMA_SECONDARY_ENV_FILE is not readable: $LLAMA_SECONDARY_ENV_FILE"
    exit 1
  fi
fi

if [[ ! -d "$LLAMA_SECONDARY_WORKDIR" ]]; then
  log_to_file "$LOG_FILE" "ERROR: LLAMA_SECONDARY_WORKDIR does not exist: $LLAMA_SECONDARY_WORKDIR"
  exit 1
fi

port_busy() {
  local port="$1"
  if [[ -z "$port" ]]; then
    return 1
  fi

  if command -v ss >/dev/null 2>&1; then
    ss -H -ltn "sport = :$port" 2>/dev/null | grep -q .
    return
  fi

  return 1
}

log_port_holder() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    log_to_file "$LOG_FILE" "Port holder for $port:"
    ss -ltnp "sport = :$port" >> "$LOG_FILE" 2>&1 || true
  fi
}

wait_for_port() {
  local port="$1"
  local wait_seconds="$2"
  local waited=0

  while port_busy "$port"; do
    if (( waited == 0 )); then
      log_to_file "$LOG_FILE" "ERROR: LLAMA_SECONDARY_PORT=$port is already in use before starting llama.cpp."
      log_port_holder "$port"
    fi

    if (( wait_seconds <= 0 || waited >= wait_seconds )); then
      return 1
    fi

    sleep 2
    waited=$(( waited + 2 ))
  done

  return 0
}

case "$LLAMA_SECONDARY_COMMAND" in
  *nohup*|*disown*|*"&"*)
    log_to_file "$LOG_FILE" "WARNING: LLAMA_SECONDARY_COMMAND looks like a background command. Use a foreground command for systemd."
    ;;
esac

cd -- "$LLAMA_SECONDARY_WORKDIR"

pre_start_sleep="$(config_int LLAMA_SECONDARY_PRE_START_SLEEP_SECONDS 0)"
if (( pre_start_sleep > 0 )); then
  log_to_file "$LOG_FILE" "Waiting ${pre_start_sleep}s before starting secondary llama.cpp."
  sleep "$pre_start_sleep"
fi

llama_port="${LLAMA_SECONDARY_PORT:-}"
port_wait_seconds="$(config_int LLAMA_SECONDARY_PORT_BUSY_WAIT_SECONDS 30)"
if ! wait_for_port "$llama_port" "$port_wait_seconds"; then
  log_to_file "$LOG_FILE" "ERROR: Port $llama_port stayed busy for ${port_wait_seconds}s. Refusing to load the secondary model into a broken start."
  exit 98
fi

log_to_file "$LOG_FILE" "Starting secondary llama.cpp"
log_to_file "$LOG_FILE" "Workdir: $LLAMA_SECONDARY_WORKDIR"
log_to_file "$LOG_FILE" "Command: $LLAMA_SECONDARY_COMMAND"
log_to_file "$LOG_FILE" "User: $(id -un), PATH: ${PATH:-}"

exec bash -lc "exec $LLAMA_SECONDARY_COMMAND" >> "$LOG_FILE" 2>&1
