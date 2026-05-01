#!/usr/bin/env sh
set -eu

cd "${ALPHARAVIS_ACP_WORKSPACE:-/workspace}"
exec python /workspace/langgraph-app/alpharavis_acp_adapter.py
