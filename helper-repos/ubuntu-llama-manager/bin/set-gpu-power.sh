#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=bin/common.sh
. "$SCRIPT_DIR/common.sh"

POWER_LOG="$UBUNTU_MANAGER_DIR/logs/gpu-power.log"

power_log() {
  log_to_file "$POWER_LOG" "$*"
}

load_config

action="${1:-apply}"
tool="${GPU_POWER_TOOL:-auto}"
gpu_ids="${POWER_LIMIT_GPU_IDS:-all}"
watts="${POWER_LIMIT_WATTS:-}"

choose_tool() {
  case "$tool" in
    auto)
      if command -v rocm-smi >/dev/null 2>&1; then
        tool="rocm-smi"
      elif command -v amd-smi >/dev/null 2>&1; then
        tool="amd-smi"
      else
        power_log "ERROR: neither rocm-smi nor amd-smi found."
        exit 1
      fi
      ;;
    rocm-smi|amd-smi)
      if ! command -v "$tool" >/dev/null 2>&1; then
        power_log "ERROR: configured GPU_POWER_TOOL not found: $tool"
        exit 1
      fi
      ;;
    *)
      power_log "ERROR: GPU_POWER_TOOL must be auto, rocm-smi, or amd-smi, got: $tool"
      exit 1
      ;;
  esac
}

validate_watts() {
  if [[ ! "$watts" =~ ^[0-9]+$ ]] || (( watts < 1 )); then
    power_log "ERROR: POWER_LIMIT_WATTS must be a positive number, got: ${watts:-empty}"
    exit 1
  fi
}

rocm_args_for_gpu() {
  local gpu="$1"
  if [[ "$gpu" == "all" ]]; then
    printf '%s\n' "--alldevices"
  else
    printf '%s\n%s\n' "-d" "$gpu"
  fi
}

run_rocm_set() {
  local gpu="$1"
  mapfile -t args < <(rocm_args_for_gpu "$gpu")
  if bool_true "${ENABLE_GPU_CLOCK_TUNING:-false}"; then
    if [[ -n "${GPU_PERF_LEVEL:-}" ]]; then
      power_log "Setting GPU ${gpu} performance level to ${GPU_PERF_LEVEL} with rocm-smi."
      rocm-smi "${args[@]}" --setperflevel "$GPU_PERF_LEVEL" >> "$POWER_LOG" 2>&1
    fi
    if [[ -n "${GPU_SCLK_LEVELS:-}" ]]; then
      power_log "Setting GPU ${gpu} SCLK levels to ${GPU_SCLK_LEVELS} with rocm-smi."
      # shellcheck disable=SC2086
      rocm-smi "${args[@]}" --setsclk $GPU_SCLK_LEVELS >> "$POWER_LOG" 2>&1
    fi
    if [[ -n "${GPU_MCLK_LEVELS:-}" ]]; then
      power_log "Setting GPU ${gpu} MCLK levels to ${GPU_MCLK_LEVELS} with rocm-smi."
      # shellcheck disable=SC2086
      rocm-smi "${args[@]}" --setmclk $GPU_MCLK_LEVELS >> "$POWER_LOG" 2>&1
    fi
    if [[ -n "${GPU_PCIE_LEVELS:-}" ]]; then
      power_log "Setting GPU ${gpu} PCIe levels to ${GPU_PCIE_LEVELS} with rocm-smi."
      # shellcheck disable=SC2086
      rocm-smi "${args[@]}" --setpcie $GPU_PCIE_LEVELS >> "$POWER_LOG" 2>&1
    fi
  elif bool_true "${RESET_CLOCKS_ON_DISABLE:-false}"; then
    power_log "Clock tuning disabled; resetting GPU ${gpu} clocks/profile to auto with rocm-smi."
    rocm-smi "${args[@]}" --resetclocks >> "$POWER_LOG" 2>&1 || true
    rocm-smi "${args[@]}" --resetprofile >> "$POWER_LOG" 2>&1 || true
  fi

  if bool_true "${ENABLE_GPU_POWER_LIMIT:-false}"; then
    power_log "Setting GPU ${gpu} power cap to ${watts} W with rocm-smi."
    rocm-smi "${args[@]}" --setpoweroverdrive "$watts" >> "$POWER_LOG" 2>&1
    if ! bool_true "${ENABLE_GPU_CLOCK_TUNING:-false}" && [[ "${GPU_PERF_LEVEL:-auto}" == "auto" ]]; then
      power_log "Power cap applied; setting GPU ${gpu} performance level back to auto with rocm-smi."
      rocm-smi "${args[@]}" --setperflevel auto >> "$POWER_LOG" 2>&1 || true
    fi
  fi
}

run_rocm_reset() {
  local gpu="$1"
  mapfile -t args < <(rocm_args_for_gpu "$gpu")
  if bool_true "${RESET_CLOCKS_ON_DISABLE:-false}"; then
    power_log "Resetting GPU ${gpu} clocks with rocm-smi."
    rocm-smi "${args[@]}" --resetclocks >> "$POWER_LOG" 2>&1
    rocm-smi "${args[@]}" --resetprofile >> "$POWER_LOG" 2>&1 || true
  fi
  if bool_true "${RESET_POWER_LIMIT_ON_DISABLE:-false}"; then
    power_log "Resetting GPU ${gpu} power cap with rocm-smi."
    rocm-smi "${args[@]}" --resetpoweroverdrive >> "$POWER_LOG" 2>&1
  fi
}

run_amd_set() {
  local gpu="$1"
  if bool_true "${ENABLE_GPU_CLOCK_TUNING:-false}"; then
    power_log "WARNING: GPU clock tuning is only implemented for rocm-smi; skipping clocks for amd-smi."
  fi
  if bool_true "${ENABLE_GPU_POWER_LIMIT:-false}"; then
    power_log "Setting GPU ${gpu} power cap to ${watts} W with amd-smi."
    if [[ "$gpu" == "all" ]]; then
      amd-smi set --power-cap ppt0 "$watts" >> "$POWER_LOG" 2>&1
    else
      amd-smi set --gpu "$gpu" --power-cap ppt0 "$watts" >> "$POWER_LOG" 2>&1
    fi
  fi
}

run_amd_reset() {
  power_log "amd-smi reset is not used here; use rocm-smi for resetpoweroverdrive."
}

run_for_configured_gpus() {
  local fn="$1"
  if [[ "$gpu_ids" == "all" ]]; then
    "$fn" "all"
    return
  fi

  local gpu
  for gpu in $gpu_ids; do
    if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
      power_log "ERROR: POWER_LIMIT_GPU_IDS must be all or GPU numbers, got: $gpu"
      exit 1
    fi
    "$fn" "$gpu"
  done
}

choose_tool

case "$action" in
  apply)
    if ! bool_true "${ENABLE_GPU_POWER_LIMIT:-false}" && ! bool_true "${ENABLE_GPU_CLOCK_TUNING:-false}"; then
      power_log "ENABLE_GPU_POWER_LIMIT=false and ENABLE_GPU_CLOCK_TUNING=false, nothing to apply."
      exit 0
    fi
    if bool_true "${ENABLE_GPU_POWER_LIMIT:-false}"; then
      validate_watts
    fi
    case "$tool" in
      rocm-smi) run_for_configured_gpus run_rocm_set ;;
      amd-smi) run_for_configured_gpus run_amd_set ;;
    esac
    ;;
  reset)
    if ! bool_true "${RESET_POWER_LIMIT_ON_DISABLE:-false}" && ! bool_true "${RESET_CLOCKS_ON_DISABLE:-false}"; then
      power_log "RESET_POWER_LIMIT_ON_DISABLE=false and RESET_CLOCKS_ON_DISABLE=false, reset skipped."
      exit 0
    fi
    case "$tool" in
      rocm-smi) run_for_configured_gpus run_rocm_reset ;;
      amd-smi) run_for_configured_gpus run_amd_reset ;;
    esac
    ;;
  show)
    case "$tool" in
      rocm-smi) rocm-smi --showid --showpower --showmaxpower --showperflevel --showclocks --showclkfrq ;;
      amd-smi) amd-smi monitor --power ;;
    esac
    ;;
  *)
    printf 'Usage: %s [apply|reset|show]\n' "$0" >&2
    exit 2
    ;;
esac
