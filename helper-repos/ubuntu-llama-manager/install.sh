#!/usr/bin/env bash

set -Eeuo pipefail

BASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
UBUNTU_CONFIG="${UBUNTU_CONFIG:-${RAKAM_CONFIG:-$BASE_DIR/ubuntu-llama.conf}}"

# shellcheck source=bin/common.sh
. "$BASE_DIR/bin/common.sh"

usage() {
  cat <<'USAGE'
Usage:
  sudo ./install.sh apply      Install/update units and start configured services
  sudo ./install.sh disable    Stop and disable all managed services/timers
  sudo ./install.sh enable     Same as apply
  sudo ./install.sh restart    Restart configured services/timers
  sudo ./install.sh uninstall  Remove systemd units, keep this folder
  sudo ./install.sh remove-legacy
                              Stop and remove old rakam-* systemd units
  sudo ./install.sh migrate-path [target-dir]
                              Rename old install folder to ubuntu-llama-manager
  ./install.sh check           Validate config and generated unit files
  sudo ./install.sh firewall-allow
                              Allow the manager API port through UFW
  ./install.sh gpu-show        Show current GPU power info
  ./install.sh status          Show status
USAGE
}

require_root() {
  if (( EUID != 0 )); then
    printf 'Please run this command with sudo.\n' >&2
    exit 1
  fi
}

unit_paths() {
  GPU_POWER_UNIT_NAME="ubuntu-gpu-power.service"
  GPU_HEALTH_UNIT_NAME="ubuntu-gpu-health.service"
  LLAMA_UNIT_NAME="ubuntu-llama.service"
  LLAMA_SECONDARY_UNIT_NAME="ubuntu-llama-8001.service"
  API_UNIT_NAME="ubuntu-manager-api.service"
  REBOOT_WATCH_UNIT_NAME="ubuntu-reboot-watch.service"
  REBOOT_SERVICE_NAME="llama-reboot.service"
  REBOOT_TIMER_NAME="llama-reboot.timer"

  GPU_POWER_UNIT_PATH="/etc/systemd/system/$GPU_POWER_UNIT_NAME"
  GPU_HEALTH_UNIT_PATH="/etc/systemd/system/$GPU_HEALTH_UNIT_NAME"
  LLAMA_UNIT_PATH="/etc/systemd/system/$LLAMA_UNIT_NAME"
  LLAMA_SECONDARY_UNIT_PATH="/etc/systemd/system/$LLAMA_SECONDARY_UNIT_NAME"
  API_UNIT_PATH="/etc/systemd/system/$API_UNIT_NAME"
  REBOOT_WATCH_UNIT_PATH="/etc/systemd/system/$REBOOT_WATCH_UNIT_NAME"
  REBOOT_SERVICE_PATH="/etc/systemd/system/$REBOOT_SERVICE_NAME"
  REBOOT_TIMER_PATH="/etc/systemd/system/$REBOOT_TIMER_NAME"
}

validate_config() {
  RUN_AS_USER="${RUN_AS_USER:-amin}"
  RUN_AS_GROUP="${RUN_AS_GROUP:-$RUN_AS_USER}"
  LLAMA_WORKDIR="${LLAMA_WORKDIR:-}"
  LLAMA_COMMAND="${LLAMA_COMMAND:-}"
  LLAMA_PORT="${LLAMA_PORT:-8033}"
  LLAMA_SECONDARY_WORKDIR="${LLAMA_SECONDARY_WORKDIR:-}"
  LLAMA_SECONDARY_COMMAND="${LLAMA_SECONDARY_COMMAND:-}"
  LLAMA_SECONDARY_PORT="${LLAMA_SECONDARY_PORT:-8001}"

  if ! id "$RUN_AS_USER" >/dev/null 2>&1; then
    printf 'Configured RUN_AS_USER does not exist: %s\n' "$RUN_AS_USER" >&2
    exit 1
  fi

  if ! getent group "$RUN_AS_GROUP" >/dev/null 2>&1; then
    printf 'Configured RUN_AS_GROUP does not exist: %s\n' "$RUN_AS_GROUP" >&2
    exit 1
  fi

  if [[ -n "$LLAMA_WORKDIR" && ! -d "$LLAMA_WORKDIR" ]]; then
    printf 'Warning: LLAMA_WORKDIR does not exist yet: %s\n' "$LLAMA_WORKDIR" >&2
  fi

  if [[ "$LLAMA_COMMAND" == *nohup* || "$LLAMA_COMMAND" == *disown* || "$LLAMA_COMMAND" == *"&"* ]]; then
    printf 'Warning: LLAMA_COMMAND looks backgrounded. For systemd use no nohup, &, or disown.\n' >&2
  fi

  if [[ "$LLAMA_COMMAND" =~ (^|[[:space:]])(-ngl|--gpu-layers|--n-gpu-layers)[[:space:]]+(999|all)([[:space:]]|$) ]]; then
    printf 'Warning: LLAMA_COMMAND forces all layers onto visible GPUs. If startup fails with OOM, check that all expected GPUs/VRAM are visible.\n' >&2
  fi

  if bool_true "${ENABLE_LLAMA_SECONDARY_SERVICE:-false}"; then
    if [[ -n "$LLAMA_SECONDARY_WORKDIR" && ! -d "$LLAMA_SECONDARY_WORKDIR" ]]; then
      printf 'Warning: LLAMA_SECONDARY_WORKDIR does not exist yet: %s\n' "$LLAMA_SECONDARY_WORKDIR" >&2
    fi

    if [[ -z "$LLAMA_SECONDARY_COMMAND" ]]; then
      printf 'LLAMA_SECONDARY_COMMAND is required when ENABLE_LLAMA_SECONDARY_SERVICE=true.\n' >&2
      exit 1
    fi

    if [[ "$LLAMA_SECONDARY_COMMAND" == *nohup* || "$LLAMA_SECONDARY_COMMAND" == *disown* || "$LLAMA_SECONDARY_COMMAND" == *"&"* ]]; then
      printf 'Warning: LLAMA_SECONDARY_COMMAND looks backgrounded. For systemd use no nohup, &, or disown.\n' >&2
    fi

    if [[ "$LLAMA_SECONDARY_PORT" == "$LLAMA_PORT" ]]; then
      printf 'LLAMA_SECONDARY_PORT must differ from LLAMA_PORT. Both are set to %s.\n' "$LLAMA_PORT" >&2
      exit 1
    fi
  fi

  if bool_true "${ENABLE_GPU_POWER_LIMIT:-false}"; then
    POWER_LIMIT_WATTS="${POWER_LIMIT_WATTS:-}"
    POWER_LIMIT_GPU_IDS="${POWER_LIMIT_GPU_IDS:-all}"
    GPU_POWER_TOOL="${GPU_POWER_TOOL:-auto}"

    if [[ ! "$POWER_LIMIT_WATTS" =~ ^[0-9]+$ ]] || (( POWER_LIMIT_WATTS < 1 )); then
      printf 'POWER_LIMIT_WATTS must be a positive number when ENABLE_GPU_POWER_LIMIT=true.\n' >&2
      exit 1
    fi

    if [[ "$POWER_LIMIT_GPU_IDS" != "all" ]]; then
      local gpu
      for gpu in $POWER_LIMIT_GPU_IDS; do
        if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
          printf 'POWER_LIMIT_GPU_IDS must be all or GPU numbers, got: %s\n' "$gpu" >&2
          exit 1
        fi
      done
    fi

    case "$GPU_POWER_TOOL" in
      auto)
        if ! command -v rocm-smi >/dev/null 2>&1 && ! command -v amd-smi >/dev/null 2>&1; then
          printf 'ENABLE_GPU_POWER_LIMIT=true but neither rocm-smi nor amd-smi was found.\n' >&2
          exit 1
        fi
        ;;
      rocm-smi|amd-smi)
        if ! command -v "$GPU_POWER_TOOL" >/dev/null 2>&1; then
          printf 'Configured GPU_POWER_TOOL not found: %s\n' "$GPU_POWER_TOOL" >&2
          exit 1
        fi
        ;;
      *)
        printf 'GPU_POWER_TOOL must be auto, rocm-smi, or amd-smi, got: %s\n' "$GPU_POWER_TOOL" >&2
        exit 1
        ;;
    esac
  fi

  if bool_true "${ENABLE_GPU_CLOCK_TUNING:-false}"; then
    GPU_POWER_TOOL="${GPU_POWER_TOOL:-auto}"
    case "$GPU_POWER_TOOL" in
      auto|rocm-smi) ;;
      *)
        printf 'GPU clock tuning currently requires rocm-smi or auto, got: %s\n' "$GPU_POWER_TOOL" >&2
        exit 1
        ;;
    esac

    case "${GPU_PERF_LEVEL:-manual}" in
      auto|low|high|manual|profile_peak|profile_standard|profile_min_sclk|profile_min_mclk|perf_determinism) ;;
      *)
        printf 'Warning: unusual GPU_PERF_LEVEL value: %s\n' "${GPU_PERF_LEVEL:-}" >&2
        ;;
    esac
  fi
}

write_unit_file() {
  local path="$1"
  local tmp
  tmp="$(mktemp)"
  cat > "$tmp"
  install -m 0644 "$tmp" "$path"
  rm -f "$tmp"
}

write_units() {
  local interval_seconds
  interval_seconds="$(reboot_interval_seconds)"

  write_unit_file "$GPU_POWER_UNIT_PATH" <<EOF
[Unit]
Description=Ubuntu AMD GPU power limit
Conflicts=rakam-gpu-power.service
Before=ubuntu-llama.service ubuntu-gpu-health.service

[Service]
Type=oneshot
RemainAfterExit=yes
Environment=UBUNTU_CONFIG=$UBUNTU_CONFIG
WorkingDirectory=$BASE_DIR
ExecStart=$BASE_DIR/bin/set-gpu-power.sh apply

[Install]
WantedBy=multi-user.target
EOF

  write_unit_file "$LLAMA_UNIT_PATH" <<EOF
[Unit]
Description=Ubuntu llama.cpp service
Conflicts=rakam-llama.service
Wants=network-online.target $GPU_POWER_UNIT_NAME
After=network-online.target $GPU_POWER_UNIT_NAME rakam-llama.service

[Service]
Type=simple
User=$RUN_AS_USER
Group=$RUN_AS_GROUP
Environment=UBUNTU_CONFIG=$UBUNTU_CONFIG
WorkingDirectory=$BASE_DIR
ExecStart=$BASE_DIR/bin/start-llama.sh
Restart=always
RestartSec=20
KillSignal=SIGINT
TimeoutStartSec=0
TimeoutStopSec=120

[Install]
WantedBy=multi-user.target
EOF

  write_unit_file "$LLAMA_SECONDARY_UNIT_PATH" <<EOF
[Unit]
Description=Ubuntu llama.cpp secondary service on port 8001
Wants=network-online.target $GPU_POWER_UNIT_NAME $LLAMA_UNIT_NAME
After=network-online.target $GPU_POWER_UNIT_NAME $LLAMA_UNIT_NAME

[Service]
Type=simple
User=$RUN_AS_USER
Group=$RUN_AS_GROUP
Environment=UBUNTU_CONFIG=$UBUNTU_CONFIG
WorkingDirectory=$BASE_DIR
ExecStart=$BASE_DIR/bin/start-llama-secondary.sh
Restart=always
RestartSec=20
KillSignal=SIGINT
TimeoutStartSec=0
TimeoutStopSec=120

[Install]
WantedBy=multi-user.target
EOF

  write_unit_file "$GPU_HEALTH_UNIT_PATH" <<EOF
[Unit]
Description=Ubuntu ROCm GPU health monitor
Conflicts=rakam-gpu-health.service
Wants=network-online.target
After=network-online.target $GPU_POWER_UNIT_NAME rakam-gpu-health.service

[Service]
Type=simple
Environment=UBUNTU_CONFIG=$UBUNTU_CONFIG
Environment=PYTHONDONTWRITEBYTECODE=1
WorkingDirectory=$BASE_DIR
ExecStart=$BASE_DIR/bin/gpu-health-monitor.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

  write_unit_file "$API_UNIT_PATH" <<EOF
[Unit]
Description=Ubuntu llama manager API
Conflicts=rakam-manager-api.service
Wants=network-online.target
After=network-online.target rakam-manager-api.service

[Service]
Type=simple
Environment=UBUNTU_CONFIG=$UBUNTU_CONFIG
Environment=PYTHONDONTWRITEBYTECODE=1
WorkingDirectory=$BASE_DIR
ExecStart=$BASE_DIR/bin/api-server.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

  write_unit_file "$REBOOT_SERVICE_PATH" <<EOF
[Unit]
Description=Controlled interval reboot for llama/GPU stability
Documentation=file://$BASE_DIR/docs/systemd.md

[Service]
Type=oneshot
Environment=UBUNTU_CONFIG=$UBUNTU_CONFIG
WorkingDirectory=$BASE_DIR
ExecStart=$BASE_DIR/bin/reboot-now.sh systemd-timer
EOF

  write_unit_file "$REBOOT_TIMER_PATH" <<EOF
[Unit]
Description=Run controlled llama/GPU stability reboot on interval

[Timer]
OnBootSec=${interval_seconds}s
OnUnitActiveSec=${interval_seconds}s
AccuracySec=1min
Persistent=false
Unit=$REBOOT_SERVICE_NAME

[Install]
WantedBy=timers.target
EOF

  write_unit_file "$REBOOT_WATCH_UNIT_PATH" <<EOF
[Unit]
Description=Ubuntu reboot watchdog fallback
Conflicts=rakam-reboot-watch.service
After=ubuntu-llama.service rakam-reboot-watch.service

[Service]
Type=simple
Environment=UBUNTU_CONFIG=$UBUNTU_CONFIG
WorkingDirectory=$BASE_DIR
ExecStart=$BASE_DIR/bin/reboot-watch.sh
Restart=on-failure
RestartSec=15

[Install]
WantedBy=multi-user.target
EOF
}

prepare_files() {
  chmod 0755 "$BASE_DIR"/bin/*.sh "$BASE_DIR/install.sh" "$BASE_DIR/status.sh" 2>/dev/null || true
  install -d -m 0755 -o "$RUN_AS_USER" -g "$RUN_AS_GROUP" "$BASE_DIR/logs" "$BASE_DIR/state"
  touch "${LLAMA_LOG_FILE:-$BASE_DIR/logs/llama.log}" "${LLAMA_SECONDARY_LOG_FILE:-$BASE_DIR/logs/llama-8001.log}" "$BASE_DIR/logs/reboot-watch.log" "$BASE_DIR/logs/reboot.log" "$BASE_DIR/logs/gpu-power.log" "$BASE_DIR/logs/gpu-health.log"
  chown "$RUN_AS_USER:$RUN_AS_GROUP" "${LLAMA_LOG_FILE:-$BASE_DIR/logs/llama.log}" "${LLAMA_SECONDARY_LOG_FILE:-$BASE_DIR/logs/llama-8001.log}" "$BASE_DIR/logs"/*.log "$BASE_DIR/state" 2>/dev/null || true
}

daemon_reload() {
  systemctl daemon-reload
}

legacy_units() {
  printf '%s\n' \
    rakam-gpu-power.service \
    rakam-gpu-health.service \
    rakam-llama.service \
    rakam-manager-api.service \
    rakam-reboot-watch.service
}

remove_legacy_symlinks() {
  local unit
  while IFS= read -r unit; do
    rm -f \
      "/etc/systemd/system/$unit" \
      "/etc/systemd/system/multi-user.target.wants/$unit" \
      "/etc/systemd/system/timers.target.wants/$unit" \
      "/etc/systemd/system/default.target.wants/$unit" \
      "/etc/systemd/system/graphical.target.wants/$unit"
    find /etc/systemd/system -path "*/$unit" -type l -delete 2>/dev/null || true
  done < <(legacy_units)
}

disable_legacy_units() {
  systemctl disable --now $(legacy_units) >/dev/null 2>&1 || true
  remove_legacy_symlinks
  systemctl reset-failed $(legacy_units) >/dev/null 2>&1 || true
}

remove_legacy_units() {
  require_root

  systemctl disable --now $(legacy_units) >/dev/null 2>&1 || true
  remove_legacy_symlinks

  daemon_reload

  systemctl reset-failed $(legacy_units) >/dev/null 2>&1 || true

  printf 'Removed old rakam-* systemd units and stale wants symlinks.\n'
}

stop_managed_units_for_move() {
  systemctl stop \
    "$LLAMA_UNIT_NAME" \
    "$LLAMA_SECONDARY_UNIT_NAME" \
    "$API_UNIT_NAME" \
    "$GPU_HEALTH_UNIT_NAME" \
    "$REBOOT_WATCH_UNIT_NAME" \
    "$REBOOT_TIMER_NAME" \
    rakam-llama.service \
    rakam-manager-api.service \
    rakam-gpu-power.service \
    rakam-gpu-health.service \
    rakam-reboot-watch.service >/dev/null 2>&1 || true
}

migrate_install_path() {
  require_root
  unit_paths

  local target="${1:-/home/amin/experi/ubuntu-llama-manager}"
  local current="$BASE_DIR"
  local parent
  local link_target

  if command -v realpath >/dev/null 2>&1; then
    target="$(realpath -sm "$target")"
  elif [[ "$target" != /* ]]; then
    target="$(pwd -L)/$target"
  fi
  parent="$(dirname -- "$target")"

  if [[ "$current" == "$target" && ! -L "$target" ]]; then
    printf 'Install folder already uses target path: %s\n' "$target"
    return
  fi

  if [[ -e "$target" && ! -L "$target" ]]; then
    printf 'Target already exists and is not a symlink: %s\n' "$target" >&2
    printf 'Move it away first or choose another target path.\n' >&2
    exit 1
  fi

  if [[ -L "$target" ]]; then
    link_target="$(readlink -f "$target")"
    if [[ "$link_target" != "$current" ]]; then
      printf 'Target symlink points somewhere else: %s -> %s\n' "$target" "$link_target" >&2
      exit 1
    fi
  fi

  install -d -m 0755 "$parent"
  stop_managed_units_for_move
  disable_legacy_units

  if [[ -L "$target" ]]; then
    rm -f "$target"
  fi

  mv "$current" "$target"
  printf 'Moved install folder:\n  from: %s\n  to:   %s\n' "$current" "$target"

  UBUNTU_CONFIG="$target/ubuntu-llama.conf" "$target/install.sh" apply
}

enable_reboot_backend() {
  local auto_enabled="${ENABLE_AUTO_REBOOT:-${ENABLE_REBOOT_TIMER:-true}}"
  local backend="${REBOOT_BACKEND:-timer}"
  local mode="${REBOOT_TIMER_MODE:-boot}"

  if ! bool_true "$auto_enabled"; then
    systemctl disable --now "$REBOOT_TIMER_NAME" "$REBOOT_WATCH_UNIT_NAME" >/dev/null 2>&1 || true
    return
  fi

  if [[ "$backend" == "watchdog" || "$mode" == "llama-start" ]]; then
    systemctl disable --now "$REBOOT_TIMER_NAME" >/dev/null 2>&1 || true
    systemctl enable --now "$REBOOT_WATCH_UNIT_NAME"
  else
    systemctl disable --now "$REBOOT_WATCH_UNIT_NAME" >/dev/null 2>&1 || true
    systemctl enable --now "$REBOOT_TIMER_NAME"
  fi
}

apply_units() {
  require_root
  load_config
  unit_paths
  validate_config
  prepare_files
  write_units
  daemon_reload
  disable_legacy_units
  daemon_reload

  if bool_true "${ENABLE_GPU_POWER_LIMIT:-false}" || bool_true "${ENABLE_GPU_CLOCK_TUNING:-false}"; then
    systemctl enable "$GPU_POWER_UNIT_NAME"
    systemctl restart "$GPU_POWER_UNIT_NAME"
  else
    systemctl disable --now "$GPU_POWER_UNIT_NAME" >/dev/null 2>&1 || true
    "$BASE_DIR/bin/set-gpu-power.sh" reset || true
  fi

  if bool_true "${ENABLE_LLAMA_SERVICE:-true}"; then
    if bool_true "${START_LLAMA_ON_BOOT:-true}"; then
      systemctl enable "$LLAMA_UNIT_NAME"
    else
      systemctl disable "$LLAMA_UNIT_NAME" >/dev/null 2>&1 || true
    fi
    systemctl restart "$LLAMA_UNIT_NAME"
  else
    systemctl disable --now "$LLAMA_UNIT_NAME" >/dev/null 2>&1 || true
  fi

  if bool_true "${ENABLE_LLAMA_SECONDARY_SERVICE:-false}"; then
    if bool_true "${START_LLAMA_SECONDARY_ON_BOOT:-true}"; then
      systemctl enable "$LLAMA_SECONDARY_UNIT_NAME"
    else
      systemctl disable "$LLAMA_SECONDARY_UNIT_NAME" >/dev/null 2>&1 || true
    fi
    systemctl restart "$LLAMA_SECONDARY_UNIT_NAME"
  else
    systemctl disable --now "$LLAMA_SECONDARY_UNIT_NAME" >/dev/null 2>&1 || true
  fi

  if bool_true "${ENABLE_GPU_HEALTH_MONITOR:-false}"; then
    systemctl enable "$GPU_HEALTH_UNIT_NAME"
    systemctl restart "$GPU_HEALTH_UNIT_NAME"
  else
    systemctl disable --now "$GPU_HEALTH_UNIT_NAME" >/dev/null 2>&1 || true
  fi

  if bool_true "${ENABLE_API_SERVICE:-true}"; then
    systemctl enable "$API_UNIT_NAME"
    systemctl restart "$API_UNIT_NAME"
  else
    systemctl disable --now "$API_UNIT_NAME" >/dev/null 2>&1 || true
  fi

  enable_reboot_backend
  status_units
}

disable_units() {
  require_root
  load_config || true
  unit_paths
  systemctl disable --now "$LLAMA_UNIT_NAME" "$LLAMA_SECONDARY_UNIT_NAME" "$API_UNIT_NAME" "$REBOOT_TIMER_NAME" "$REBOOT_WATCH_UNIT_NAME" "$GPU_POWER_UNIT_NAME" "$GPU_HEALTH_UNIT_NAME" >/dev/null 2>&1 || true
  "$BASE_DIR/bin/set-gpu-power.sh" reset || true
  daemon_reload
  printf 'Disabled managed services and timers.\n'
}

restart_units() {
  require_root
  load_config
  unit_paths
  daemon_reload

  if bool_true "${ENABLE_GPU_POWER_LIMIT:-false}" || bool_true "${ENABLE_GPU_CLOCK_TUNING:-false}"; then
    systemctl restart "$GPU_POWER_UNIT_NAME"
  fi
  if bool_true "${ENABLE_LLAMA_SERVICE:-true}"; then
    systemctl restart "$LLAMA_UNIT_NAME"
  fi
  if bool_true "${ENABLE_LLAMA_SECONDARY_SERVICE:-false}"; then
    systemctl restart "$LLAMA_SECONDARY_UNIT_NAME"
  fi
  if bool_true "${ENABLE_GPU_HEALTH_MONITOR:-false}"; then
    systemctl restart "$GPU_HEALTH_UNIT_NAME"
  fi
  if bool_true "${ENABLE_API_SERVICE:-true}"; then
    systemctl restart "$API_UNIT_NAME"
  fi
  enable_reboot_backend
  status_units
}

uninstall_units() {
  require_root
  unit_paths
  systemctl disable --now "$LLAMA_UNIT_NAME" "$LLAMA_SECONDARY_UNIT_NAME" "$API_UNIT_NAME" "$REBOOT_TIMER_NAME" "$REBOOT_WATCH_UNIT_NAME" "$GPU_POWER_UNIT_NAME" "$GPU_HEALTH_UNIT_NAME" >/dev/null 2>&1 || true
  disable_legacy_units
  rm -f "$GPU_POWER_UNIT_PATH" "$GPU_HEALTH_UNIT_PATH" "$LLAMA_UNIT_PATH" "$LLAMA_SECONDARY_UNIT_PATH" "$API_UNIT_PATH" "$REBOOT_SERVICE_PATH" "$REBOOT_TIMER_PATH" "$REBOOT_WATCH_UNIT_PATH"
  daemon_reload
  systemctl reset-failed "$GPU_POWER_UNIT_NAME" "$GPU_HEALTH_UNIT_NAME" "$LLAMA_UNIT_NAME" "$LLAMA_SECONDARY_UNIT_NAME" "$API_UNIT_NAME" "$REBOOT_SERVICE_NAME" "$REBOOT_TIMER_NAME" "$REBOOT_WATCH_UNIT_NAME" >/dev/null 2>&1 || true
  printf 'Removed systemd units. Config folder remains: %s\n' "$BASE_DIR"
}

status_units() {
  unit_paths
  systemctl --no-pager --full status "$GPU_POWER_UNIT_NAME" "$GPU_HEALTH_UNIT_NAME" "$LLAMA_UNIT_NAME" "$LLAMA_SECONDARY_UNIT_NAME" "$API_UNIT_NAME" "$REBOOT_TIMER_NAME" "$REBOOT_WATCH_UNIT_NAME" || true
  systemctl list-timers --no-pager "$REBOOT_TIMER_NAME" || true
}

check_units() {
  load_config
  unit_paths
  validate_config

  local tmp_dir
  tmp_dir="$(mktemp -d)"
  GPU_POWER_UNIT_PATH="$tmp_dir/$GPU_POWER_UNIT_NAME"
  GPU_HEALTH_UNIT_PATH="$tmp_dir/$GPU_HEALTH_UNIT_NAME"
  LLAMA_UNIT_PATH="$tmp_dir/$LLAMA_UNIT_NAME"
  LLAMA_SECONDARY_UNIT_PATH="$tmp_dir/$LLAMA_SECONDARY_UNIT_NAME"
  API_UNIT_PATH="$tmp_dir/$API_UNIT_NAME"
  REBOOT_SERVICE_PATH="$tmp_dir/$REBOOT_SERVICE_NAME"
  REBOOT_TIMER_PATH="$tmp_dir/$REBOOT_TIMER_NAME"
  REBOOT_WATCH_UNIT_PATH="$tmp_dir/$REBOOT_WATCH_UNIT_NAME"
  write_units

  if command -v systemd-analyze >/dev/null 2>&1; then
    systemd-analyze verify "$GPU_POWER_UNIT_PATH" "$GPU_HEALTH_UNIT_PATH" "$LLAMA_UNIT_PATH" "$LLAMA_SECONDARY_UNIT_PATH" "$API_UNIT_PATH" "$REBOOT_SERVICE_PATH" "$REBOOT_TIMER_PATH" "$REBOOT_WATCH_UNIT_PATH"
  fi

  rm -rf "$tmp_dir"
  printf 'Config and generated unit files look OK. Reboot interval: %ss.\n' "$(reboot_interval_seconds)"
}

gpu_show() {
  "$BASE_DIR/bin/set-gpu-power.sh" show
}

firewall_allow() {
  require_root
  load_config

  local port="${API_PORT:-8099}"
  local source="${API_FIREWALL_ALLOW_FROM:-}"

  if ! command -v ufw >/dev/null 2>&1; then
    printf 'ufw is not installed; nothing to configure.\n'
    return 0
  fi

  if [[ -n "$source" ]]; then
    ufw allow from "$source" to any port "$port" proto tcp comment "ubuntu-llama-manager API"
    printf 'Allowed TCP %s from %s through UFW.\n' "$port" "$source"
  else
    ufw allow "$port/tcp" comment "ubuntu-llama-manager API"
    printf 'Allowed TCP %s through UFW.\n' "$port"
  fi
}

cmd="${1:-apply}"
case "$cmd" in
  apply|install|enable) apply_units ;;
  disable) disable_units ;;
  restart) restart_units ;;
  uninstall|remove) uninstall_units ;;
  remove-legacy|purge-legacy) remove_legacy_units ;;
  migrate-path|rename-install) migrate_install_path "${2:-}" ;;
  check) check_units ;;
  firewall-allow|ufw-allow) firewall_allow ;;
  gpu-show) gpu_show ;;
  status) status_units ;;
  -h|--help|help) usage ;;
  *)
    usage >&2
    exit 2
    ;;
esac
