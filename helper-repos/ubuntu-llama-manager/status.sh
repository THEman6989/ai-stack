#!/usr/bin/env bash

set -Eeuo pipefail

BASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

systemctl --no-pager --full status ubuntu-gpu-power.service ubuntu-gpu-health.service ubuntu-llama.service ubuntu-manager-api.service llama-reboot.timer ubuntu-reboot-watch.service || true
printf '\nTimers:\n'
systemctl list-timers --no-pager llama-reboot.timer || true
printf '\nCurrent GPU power info:\n'
"$BASE_DIR/bin/set-gpu-power.sh" show || true
printf '\nLast GPU power service lines:\n'
journalctl --no-pager -u ubuntu-gpu-power.service -n 60 || true
printf '\nLast GPU health service lines:\n'
journalctl --no-pager -u ubuntu-gpu-health.service -n 80 || true
printf '\nLast API service lines:\n'
journalctl --no-pager -u ubuntu-manager-api.service -n 60 || true
printf '\nLast controlled reboot service lines:\n'
journalctl --no-pager -u llama-reboot.service -n 60 || true
printf '\nLast llama service lines:\n'
journalctl --no-pager -u ubuntu-llama.service -n 60 || true
printf '\nLast reboot watchdog lines:\n'
journalctl --no-pager -u ubuntu-reboot-watch.service -n 60 || true
