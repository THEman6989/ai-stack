#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=bin/common.sh
. "$SCRIPT_DIR/common.sh"

WATCH_LOG="$UBUNTU_MANAGER_DIR/logs/reboot-watch.log"

watch_log() {
  log_to_file "$WATCH_LOG" "$*"
}

validate_seconds() {
  local value="$1"
  if [[ ! "$value" =~ ^[0-9]+$ ]] || (( value < 60 )); then
    watch_log "ERROR: REBOOT_AFTER_SECONDS must be a number >= 60, got: $value"
    exit 1
  fi
}

llama_is_active() {
  if systemctl is-active --quiet ubuntu-llama.service 2>/dev/null; then
    return 0
  fi

  if bool_true "${WATCH_EXTERNAL_LLAMA_PROCESS:-true}" && [[ -n "${LLAMA_PROCESS_PATTERN:-}" ]]; then
    if command -v pgrep >/dev/null 2>&1; then
      if [[ -n "${RUN_AS_USER:-}" ]]; then
        pgrep -u "$RUN_AS_USER" -f -- "$LLAMA_PROCESS_PATTERN" >/dev/null 2>&1 && return 0
      else
        pgrep -f -- "$LLAMA_PROCESS_PATTERN" >/dev/null 2>&1 && return 0
      fi
    fi
  fi

  return 1
}

load_config

ENABLE_AUTO_REBOOT="${ENABLE_AUTO_REBOOT:-${ENABLE_REBOOT_TIMER:-true}}"

if ! bool_true "$ENABLE_AUTO_REBOOT"; then
  watch_log "ENABLE_REBOOT_TIMER=false, watchdog exits."
  exit 0
fi

REBOOT_AFTER_SECONDS="$(reboot_interval_seconds)"
WATCH_POLL_SECONDS="${WATCH_POLL_SECONDS:-15}"
REBOOT_TIMER_MODE="${REBOOT_TIMER_MODE:-boot}"

validate_seconds "$REBOOT_AFTER_SECONDS"

case "$WATCH_POLL_SECONDS" in
  ''|*[!0-9]*)
    WATCH_POLL_SECONDS="15"
    ;;
esac

if (( WATCH_POLL_SECONDS < 5 )); then
  WATCH_POLL_SECONDS="5"
fi

case "$REBOOT_TIMER_MODE" in
  boot)
    watch_log "Countdown starts now from boot/service start: ${REBOOT_AFTER_SECONDS}s."
    ;;
  llama-start)
    watch_log "Waiting for llama before countdown starts."
    while true; do
      load_config
      ENABLE_AUTO_REBOOT="${ENABLE_AUTO_REBOOT:-${ENABLE_REBOOT_TIMER:-true}}"
      if ! bool_true "$ENABLE_AUTO_REBOOT"; then
        watch_log "Timer disabled while waiting for llama; watchdog exits."
        exit 0
      fi

      if llama_is_active; then
        watch_log "Llama detected; countdown starts: ${REBOOT_AFTER_SECONDS}s."
        break
      fi

      sleep "$WATCH_POLL_SECONDS"
    done
    ;;
  *)
    watch_log "ERROR: REBOOT_TIMER_MODE must be boot or llama-start, got: $REBOOT_TIMER_MODE"
    exit 1
    ;;
esac

start_epoch="$(date +%s)"
deadline_epoch=$((start_epoch + REBOOT_AFTER_SECONDS))

while true; do
  now_epoch="$(date +%s)"
  remaining=$((deadline_epoch - now_epoch))
  if (( remaining <= 0 )); then
    break
  fi

  sleep_for="$WATCH_POLL_SECONDS"
  if (( sleep_for > remaining )); then
    sleep_for="$remaining"
  fi
  sleep "$sleep_for"

  load_config
  ENABLE_AUTO_REBOOT="${ENABLE_AUTO_REBOOT:-${ENABLE_REBOOT_TIMER:-true}}"
  if ! bool_true "$ENABLE_AUTO_REBOOT"; then
    watch_log "Timer disabled during countdown; watchdog exits."
    exit 0
  fi
done

load_config
ENABLE_AUTO_REBOOT="${ENABLE_AUTO_REBOOT:-${ENABLE_REBOOT_TIMER:-true}}"
if ! bool_true "$ENABLE_AUTO_REBOOT"; then
  watch_log "Timer disabled at deadline; reboot skipped."
  exit 0
fi

watch_log "Deadline reached; delegating to reboot-now.sh."
exec "$UBUNTU_MANAGER_DIR/bin/reboot-now.sh" watchdog
