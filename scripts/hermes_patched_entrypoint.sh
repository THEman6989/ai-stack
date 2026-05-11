#!/usr/bin/env bash
set -euo pipefail

HERMES_PATCH_TARGET_DIR=/opt/hermes /workspace/scripts/apply_hermes_agent_patches.sh

exec /opt/hermes/docker/entrypoint.sh "$@"
