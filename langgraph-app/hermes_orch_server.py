"""
hermes_orch_server.py — Dedicated Hermes Orchestration Service

Lightweight FastAPI server that wraps the AlphaRavis → Hermes streaming path.
Handles only one endpoint: POST /hermes/stream

Architecture:
  Browser (Coding Tab: +AlphaRavis mode)
    → POST :8650/hermes/stream
    → Pre-loads Memory/RAG/Skills/Sessions
    → Hermes Agent (:8642) SSE relay
    → Auto-saves output as AlphaRavis artifact

Not part of the LangGraph graph, Bridge, or Media Gallery.
Runs standalone via: uvicorn hermes_orch_server:app --host 0.0.0.0 --port 8650
"""

from __future__ import annotations

import json
import logging
import os
import time

import httpx
from agent_graph import (
    _call_hermes_streaming_sse,
)
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger("hermes-orch")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Hermes Orchestrator",
    description="Streaming relay: AlphaRavis context + Hermes Agent",
    version="0.1.0",
)


@app.get("/health")
async def health():
    """Readiness probe."""
    return {"status": "ok", "service": "hermes-orch"}


@app.post("/hermes/stream")
async def hermes_stream(request: Request):
    """Stream Hermes with pre-loaded AlphaRavis context.

    Request:  {"message": "...", "system_prompt": "..."}
    Response: SSE stream (text/event-stream)

    Flow:
      1. Read message + optional system_prompt
      2. Call _call_hermes_streaming_sse() which:
         a. Pre-loads Memory/RAG/Skills/Sessions (parallel, best-effort)
         b. Calls Hermes Agent (:8642) with stream=true
         c. Relays Hermes SSE events directly to caller
         d. Saves full output as AlphaRavis artifact
         e. Records memory entry
      3. Return SSE stream to browser
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    message = str(body.get("message", "")).strip()
    if not message:
        raise HTTPException(status_code=400, detail="Missing 'message' field")

    system_prompt = str(body.get("system_prompt", ""))
    max_chars = int(body.get("max_output_chars", 24000))

    return StreamingResponse(
        _call_hermes_streaming_sse(
            message=message,
            system_prompt=system_prompt,
            max_output_chars=max_chars,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
