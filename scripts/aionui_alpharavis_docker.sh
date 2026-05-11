#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

exec docker compose exec -T \
  -e LANGGRAPH_API_URL=http://langgraph-api:2024 \
  -e LANGGRAPH_ASSISTANT_ID=alpha_ravis \
  -e ALPHARAVIS_ACP_WORKSPACE=/workspace \
  -e ALPHARAVIS_ACP_TRACE_DETAIL=summary \
  -e ALPHARAVIS_ACP_DEBUG_IO=true \
  -e ALPHARAVIS_ACP_ALLOW_FILE_WRITES=false \
  -e ALPHARAVIS_OPERATIONAL_LOGGING=true \
  -e ALPHARAVIS_DEBUG_ALL_LOGGING=true \
  -e ALPHARAVIS_LOG_DIR=/workspace/logs \
  langgraph-api \
  python /workspace/langgraph-app/alpharavis_acp_adapter.py
