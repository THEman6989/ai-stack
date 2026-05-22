#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=bin/common.sh
. "$SCRIPT_DIR/common.sh"

load_config
ensure_state_dir

MARKER_FILE="$UBUNTU_STATE_DIR/reboot-test-marker.env"
REPORT_FILE="$UBUNTU_STATE_DIR/reboot-test-report.txt"
WAIT_SECONDS="$(config_int REBOOT_TEST_WAIT_SECONDS 600)"
POLL_SECONDS="$(config_int REBOOT_TEST_POLL_SECONDS 10)"

failures=0

usage() {
  cat <<'USAGE'
Usage:
  ./bin/test-reboot.sh arm       Save current boot marker before a manual reboot
  sudo ./bin/test-reboot.sh reboot
                                Save marker, then trigger a controlled reboot
  ./bin/test-reboot.sh after     Check services, timer, port, API after reboot
  ./bin/test-reboot.sh status    Same as after, but does not require a marker

Optional config:
  REBOOT_TEST_WAIT_SECONDS="600"
  REBOOT_TEST_POLL_SECONDS="10"
USAGE
}

now_iso() {
  date --iso-8601=seconds
}

boot_id() {
  cat /proc/sys/kernel/random/boot_id
}

uptime_seconds() {
  awk '{print int($1)}' /proc/uptime
}

write_report() {
  printf '%s\n' "$*" | tee -a "$REPORT_FILE"
}

pass() {
  write_report "PASS $*"
}

warn() {
  write_report "WARN $*"
}

fail() {
  failures=$(( failures + 1 ))
  write_report "FAIL $*"
}

systemctl_value() {
  local subcmd="$1"
  local unit="$2"
  systemctl "$subcmd" "$unit" 2>/dev/null || true
}

check_active() {
  local unit="$1"
  if systemctl is-active --quiet "$unit"; then
    pass "$unit is active"
  else
    fail "$unit is not active (state: $(systemctl_value is-active "$unit"))"
  fi
}

check_inactive() {
  local unit="$1"
  if systemctl is-active --quiet "$unit"; then
    fail "$unit is still active"
  else
    pass "$unit is inactive"
  fi
}

check_enabled() {
  local unit="$1"
  if systemctl is-enabled --quiet "$unit"; then
    pass "$unit is enabled"
  else
    fail "$unit is not enabled (state: $(systemctl_value is-enabled "$unit"))"
  fi
}

check_disabled_or_missing() {
  local unit="$1"
  local state
  state="$(systemctl_value is-enabled "$unit")"

  case "$state" in
    ""|disabled|masked|not-found|static)
      pass "$unit is not enabled (state: ${state:-missing})"
      ;;
    *)
      fail "$unit is unexpectedly enabled (state: $state)"
      ;;
  esac

  check_inactive "$unit"
}

port_open() {
  local host="$1"
  local port="$2"
  timeout 2 bash -lc "</dev/tcp/$host/$port" >/dev/null 2>&1
}

check_llama_port() {
  local host="${LLAMA_TEST_HOST:-127.0.0.1}"
  local port="${LLAMA_PORT:-8033}"

  if port_open "$host" "$port"; then
    pass "llama port is reachable at $host:$port"
  else
    fail "llama port is not reachable at $host:$port"
    if command -v ss >/dev/null 2>&1; then
      write_report "INFO port holder:"
      ss -ltnp "sport = :$port" 2>&1 | tee -a "$REPORT_FILE" || true
    fi
  fi
}

check_api_health() {
  local port="${API_PORT:-8099}"

  if ! bool_true "${ENABLE_API_SERVICE:-true}"; then
    warn "API service disabled by config"
    return
  fi

  if ! command -v curl >/dev/null 2>&1; then
    warn "curl missing, skipping API health check"
    return
  fi

  if curl -fsS --max-time 3 "http://127.0.0.1:$port/health" >/tmp/ubuntu-manager-reboot-test-health.json; then
    pass "API /health reachable on 127.0.0.1:$port"
    sed 's/^/INFO api health: /' /tmp/ubuntu-manager-reboot-test-health.json | tee -a "$REPORT_FILE"
  else
    fail "API /health not reachable on 127.0.0.1:$port"
  fi
}

wait_for_post_boot() {
  local waited=0
  local port="${LLAMA_PORT:-8033}"
  local host="${LLAMA_TEST_HOST:-127.0.0.1}"

  if ! bool_true "${ENABLE_LLAMA_SERVICE:-true}" || ! bool_true "${START_LLAMA_ON_BOOT:-true}"; then
    return
  fi

  write_report "INFO waiting up to ${WAIT_SECONDS}s for ubuntu-llama.service and $host:$port"
  while (( waited <= WAIT_SECONDS )); do
    if systemctl is-active --quiet ubuntu-llama.service && port_open "$host" "$port"; then
      pass "llama became reachable after ${waited}s"
      return
    fi
    sleep "$POLL_SECONDS"
    waited=$(( waited + POLL_SECONDS ))
  done

  fail "llama did not become reachable within ${WAIT_SECONDS}s"
}

arm_test() {
  cat > "$MARKER_FILE" <<EOF
MARKER_CREATED_AT="$(now_iso)"
MARKER_BOOT_ID="$(boot_id)"
MARKER_UPTIME_SECONDS="$(uptime_seconds)"
EOF
  printf 'Reboot test armed: %s\n' "$MARKER_FILE"
  printf 'Current boot id: %s\n' "$(boot_id)"
}

run_after_checks() {
  : > "$REPORT_FILE"
  write_report "Ubuntu Llama Manager reboot test"
  write_report "Started: $(now_iso)"
  write_report "Current boot id: $(boot_id)"
  write_report "Current uptime seconds: $(uptime_seconds)"

  if [[ -r "$MARKER_FILE" ]]; then
    # shellcheck source=/dev/null
    . "$MARKER_FILE"
    write_report "Marker created: ${MARKER_CREATED_AT:-unknown}"
    write_report "Marker boot id: ${MARKER_BOOT_ID:-unknown}"
    if [[ "${MARKER_BOOT_ID:-}" == "$(boot_id)" ]]; then
      fail "boot id did not change since arm; no reboot detected"
    else
      pass "boot id changed; reboot detected"
    fi
  else
    warn "no marker file found; run './bin/test-reboot.sh arm' before reboot for boot-id comparison"
  fi

  check_disabled_or_missing rakam-llama.service
  check_disabled_or_missing rakam-manager-api.service
  check_disabled_or_missing rakam-gpu-power.service
  check_disabled_or_missing rakam-gpu-health.service
  check_disabled_or_missing rakam-reboot-watch.service

  if bool_true "${ENABLE_LLAMA_SERVICE:-true}" && bool_true "${START_LLAMA_ON_BOOT:-true}"; then
    wait_for_post_boot
    check_enabled ubuntu-llama.service
    check_active ubuntu-llama.service
    check_llama_port
  else
    warn "ubuntu-llama.service boot start disabled by config"
  fi

  if bool_true "${ENABLE_API_SERVICE:-true}"; then
    check_enabled ubuntu-manager-api.service
    check_active ubuntu-manager-api.service
    check_api_health
  fi

  if bool_true "${ENABLE_GPU_HEALTH_MONITOR:-false}"; then
    check_enabled ubuntu-gpu-health.service
    check_active ubuntu-gpu-health.service
  fi

  if bool_true "${ENABLE_GPU_POWER_LIMIT:-false}" || bool_true "${ENABLE_GPU_CLOCK_TUNING:-false}"; then
    check_enabled ubuntu-gpu-power.service
    check_active ubuntu-gpu-power.service
  fi

  if bool_true "${ENABLE_AUTO_REBOOT:-${ENABLE_REBOOT_TIMER:-true}}" && [[ "${REBOOT_BACKEND:-timer}" == "timer" && "${REBOOT_TIMER_MODE:-boot}" != "llama-start" ]]; then
    check_enabled llama-reboot.timer
    check_active llama-reboot.timer
    systemctl list-timers --no-pager llama-reboot.timer 2>&1 | sed 's/^/INFO timer: /' | tee -a "$REPORT_FILE" || true
  fi

  write_report "Finished: $(now_iso)"
  write_report "Failures: $failures"
  write_report "Report: $REPORT_FILE"

  if (( failures > 0 )); then
    return 1
  fi
}

trigger_reboot() {
  if (( EUID != 0 )); then
    printf 'Please run reboot mode with sudo.\n' >&2
    exit 1
  fi

  arm_test
  printf 'Triggering controlled reboot in 5 seconds. Press Ctrl+C to cancel.\n'
  sleep 5
  bash -lc "exec ${REBOOT_COMMAND:-/usr/bin/systemctl reboot}"
}

cmd="${1:-status}"
case "$cmd" in
  arm) arm_test ;;
  reboot) trigger_reboot ;;
  after|status) run_after_checks ;;
  -h|--help|help) usage ;;
  *)
    usage >&2
    exit 2
    ;;
esac
