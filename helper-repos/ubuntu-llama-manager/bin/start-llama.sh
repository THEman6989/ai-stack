#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=bin/common.sh
. "$SCRIPT_DIR/common.sh"

load_config

LOG_FILE="${LLAMA_LOG_FILE:-$UBUNTU_MANAGER_DIR/logs/llama.log}"
ensure_log_parent "$LOG_FILE"

if ! bool_true "${ENABLE_LLAMA_SERVICE:-true}"; then
  log_to_file "$LOG_FILE" "ENABLE_LLAMA_SERVICE=false, service is idling."
  exec sleep infinity
fi

require_value "LLAMA_WORKDIR" "${LLAMA_WORKDIR:-}"
require_value "LLAMA_COMMAND" "${LLAMA_COMMAND:-}"

if [[ -n "${LLAMA_ENV_FILE:-}" ]]; then
  if [[ -r "$LLAMA_ENV_FILE" ]]; then
    log_to_file "$LOG_FILE" "Loading LLAMA_ENV_FILE: $LLAMA_ENV_FILE"
    set -a
    # shellcheck source=/dev/null
    . "$LLAMA_ENV_FILE"
    set +a
  else
    log_to_file "$LOG_FILE" "ERROR: LLAMA_ENV_FILE is not readable: $LLAMA_ENV_FILE"
    exit 1
  fi
fi

if [[ ! -d "$LLAMA_WORKDIR" ]]; then
  log_to_file "$LOG_FILE" "ERROR: LLAMA_WORKDIR does not exist: $LLAMA_WORKDIR"
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
      log_to_file "$LOG_FILE" "ERROR: LLAMA_PORT=$port is already in use before starting llama.cpp."
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

case "$LLAMA_COMMAND" in
  *nohup*|*disown*|*"&"*)
    log_to_file "$LOG_FILE" "WARNING: LLAMA_COMMAND looks like a background command. Use a foreground command for systemd."
    ;;
esac

if [[ "$LLAMA_COMMAND" =~ (^|[[:space:]])(-ngl|--gpu-layers|--n-gpu-layers)[[:space:]]+(999|all)([[:space:]]|$) ]]; then
  log_to_file "$LOG_FILE" "WARNING: LLAMA_COMMAND forces all layers onto visible GPUs. If startup fails with ROCm OOM/cudaMalloc, check that all expected GPUs/VRAM are visible."
fi

cd -- "$LLAMA_WORKDIR"

pre_start_sleep="$(config_int LLAMA_PRE_START_SLEEP_SECONDS 0)"
if (( pre_start_sleep > 0 )); then
  log_to_file "$LOG_FILE" "Waiting ${pre_start_sleep}s before starting llama.cpp."
  sleep "$pre_start_sleep"
fi

llama_port="${LLAMA_PORT:-}"
port_wait_seconds="$(config_int LLAMA_PORT_BUSY_WAIT_SECONDS 30)"
if ! wait_for_port "$llama_port" "$port_wait_seconds"; then
  log_to_file "$LOG_FILE" "ERROR: Port $llama_port stayed busy for ${port_wait_seconds}s. Refusing to load the model into a broken start."
  log_to_file "$LOG_FILE" "Hint: check old services with: systemctl is-enabled rakam-llama.service ubuntu-llama.service"
  exit 98
fi

log_to_file "$LOG_FILE" "Starting llama.cpp"
log_to_file "$LOG_FILE" "Workdir: $LLAMA_WORKDIR"
log_to_file "$LOG_FILE" "Command: $LLAMA_COMMAND"
log_to_file "$LOG_FILE" "User: $(id -un), PATH: ${PATH:-}"

exec bash -lc "exec $LLAMA_COMMAND" >> "$LOG_FILE" 2>&1
