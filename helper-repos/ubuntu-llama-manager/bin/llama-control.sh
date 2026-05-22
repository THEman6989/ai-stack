#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=bin/common.sh
. "$SCRIPT_DIR/common.sh"

load_config

cmd="${1:-status}"
service="ubuntu-llama.service"
port="${LLAMA_PORT:-8033}"
host="${LLAMA_HOST:-127.0.0.1}"

case "$cmd" in
  start|stop|restart)
    systemctl "$cmd" "$service"
    ;;
  status)
    systemctl --no-pager --full status "$service" || true
    ;;
  is-active)
    systemctl is-active "$service"
    ;;
  port)
    if command -v ss >/dev/null 2>&1; then
      ss -ltnp "sport = :$port" || true
    else
      timeout 2 bash -lc "</dev/tcp/$host/$port" >/dev/null 2>&1 && printf 'open\n' || printf 'closed\n'
    fi
    ;;
  process)
    pgrep -af -- "${LLAMA_PROCESS_PATTERN:-llama-server}" || true
    ;;
  logs)
    if [[ -n "${LLAMA_LOG_FILE:-}" && -r "$LLAMA_LOG_FILE" ]]; then
      tail -n "${2:-120}" "$LLAMA_LOG_FILE"
    else
      journalctl --no-pager -u "$service" -n "${2:-120}" || true
    fi
    ;;
  log-path)
    printf '%s\n' "${LLAMA_LOG_FILE:-$UBUNTU_MANAGER_DIR/logs/llama.log}"
    ;;
  *)
    printf 'Usage: %s [start|stop|restart|status|is-active|port|process|logs|log-path]\n' "$0" >&2
    exit 2
    ;;
esac
