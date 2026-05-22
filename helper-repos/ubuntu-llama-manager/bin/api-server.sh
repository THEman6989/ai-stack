#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
BASE_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

cd -- "$BASE_DIR"
exec /usr/bin/python3 -m ubuntu_manager.api --config "${UBUNTU_CONFIG:-$BASE_DIR/ubuntu-llama.conf}"
