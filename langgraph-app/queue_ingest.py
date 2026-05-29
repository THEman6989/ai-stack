"""Queue Ingest — API endpoint for DeepAgent ARM Gateway offline message flush.

Receives queued messages from the gateway, checks idempotency, and submits
each new message as a user prompt to the LangGraph agent.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any
from urllib.parse import urljoin

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from langgraph_sdk import get_client

logger = logging.getLogger(__name__)

# --- Configuration ---
from env_utils import LANGGRAPH_API_URL, LANGGRAPH_ASSISTANT_ID
QUEUE_INGEST_TOKEN = os.getenv("AI_STACK_QUEUE_INGEST_TOKEN", "") or os.getenv(
    "GATEWAY_ADMIN_TOKEN", ""
)
QUEUE_INGEST_TIMEOUT_SECONDS = float(
    os.getenv("QUEUE_INGEST_TIMEOUT_SECONDS", "300")
)

# In-memory idempotency store. Survives until bridge restart.
# Key: message_id → timestamp of first processing
_PROCESSED_IDS: dict[str, float] = {}
_MAX_PROCESSED_IDS = int(os.getenv("QUEUE_INGEST_MAX_PROCESSED_IDS", "50000"))


def _client():
    return get_client(url=LANGGRAPH_API_URL)


def _mark_processed(message_id: str) -> None:
    """Mark a message_id as processed. Evicts oldest entries if over limit."""
    if len(_PROCESSED_IDS) >= _MAX_PROCESSED_IDS:
        # Evict oldest 10%
        evict_count = max(1, int(_MAX_PROCESSED_IDS * 0.1))
        sorted_ids = sorted(_PROCESSED_IDS.items(), key=lambda x: x[1])
        for old_id, _ in sorted_ids[:evict_count]:
            _PROCESSED_IDS.pop(old_id, None)
    _PROCESSED_IDS[message_id] = time.time()


def _is_processed(message_id: str) -> bool:
    return message_id in _PROCESSED_IDS


# --- Router ---
router = APIRouter(prefix="/api/queue")


@router.post("/ingest")
async def queue_ingest(request: Request):
    """Accept queued messages from the DeepAgent ARM Gateway.

    Request body:
    {
        "messages": [
            {
                "id": "uuid",
                "session_id": "uuid",
                "role": "user",
                "content": "...",
                "created_at": "ISO8601",
                "status": "queued",
                "media": [
                    {"media_id": "...", "filename": "...", "mime_type": "...",
                     "file_path": "..."}
                ]
            }
        ]
    }

    Response:
    {
        "accepted": ["uuid1"],
        "duplicates": ["uuid2"],
        "failed": []
    }
    """
    # Auth check
    if QUEUE_INGEST_TOKEN:
        auth = request.headers.get("Authorization", "")
        expected = f"Bearer {QUEUE_INGEST_TOKEN}"
        if auth != expected:
            raise HTTPException(status_code=403, detail="Invalid token")

    body = await request.json()
    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="messages is required")

    accepted: list[str] = []
    duplicates: list[str] = []
    failed: list[str] = []

    for msg in messages:
        msg_id = msg.get("id", "")
        if not msg_id:
            failed.append("missing-id")
            continue

        if _is_processed(msg_id):
            duplicates.append(msg_id)
            continue

        try:
            await _submit_to_langgraph(msg)
            _mark_processed(msg_id)
            accepted.append(msg_id)
            logger.info("Queue ingest: accepted %s", msg_id)
        except Exception as exc:
            logger.error("Queue ingest: failed %s: %s", msg_id, exc)
            failed.append(msg_id)

    return JSONResponse(
        {"accepted": accepted, "duplicates": duplicates, "failed": failed}
    )


async def _submit_to_langgraph(msg: dict[str, Any]) -> None:
    """Submit a single queued message as a user prompt to LangGraph.

    Uses the assistant's thread per session_id for session continuity.
    """
    client = _client()
    session_id = msg.get("session_id", "default")
    thread_id = f"queue-{session_id}"

    # Build content: message text + media download + descriptions
    content = msg.get("content", "")
    media_list = msg.get("media", [])
    media_contents: list[dict[str, Any]] = []
    if media_list:
        media_contents = await _download_media_files(media_list)

    if not content.strip() and not media_contents:
        return  # Empty message, skip

    # Build user message with media
    user_message: dict[str, Any] = {"role": "user"}
    if media_contents:
        # Multimodal: content is array of text + image blocks
        parts: list[dict[str, Any]] = []
        if content.strip():
            parts.append({"type": "text", "text": content})
        for mc in media_contents:
            if mc.get("mime_type", "").startswith("image/"):
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": mc.get("data_url", mc.get("media_url", ""))},
                })
            else:
                parts.append({"type": "text", "text": f"[Datei: {mc['filename']}]"})
        user_message["content"] = parts
    else:
        user_message["content"] = content

    # Submit as a run
    run_payload = {
        "assistant_id": LANGGRAPH_ASSISTANT_ID,
        "input": {
            "messages": [user_message]
        },
    }

    # Use LangGraph runs.create_and_wait or runs.create + join
    # For simplicity, use create + wait
    run = await client.runs.create(
        thread_id=thread_id,
        **run_payload,
    )

    # Wait for completion
    try:
        await client.runs.join(
            thread_id=thread_id,
            run_id=run["run_id"],
            timeout=QUEUE_INGEST_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        logger.warning(
            "Queue ingest run %s for %s did not complete cleanly: %s",
            run.get("run_id", "unknown"),
            msg.get("id", "unknown"),
            exc,
        )


async def _download_media_files(
    media_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Download media files from the gateway and return them with data_urls."""
    result: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for m in media_list:
            media_url = m.get("media_url", "")
            if not media_url:
                continue
            try:
                resp = await client.get(media_url)
                resp.raise_for_status()
                data = resp.content
                mime_type = m.get("mime_type", resp.headers.get("content-type", "application/octet-stream"))
                # Build data URL for LangGraph
                import base64
                b64 = base64.b64encode(data).decode("ascii")
                data_url = f"data:{mime_type};base64,{b64}"
                result.append({
                    "filename": m.get("filename", "file"),
                    "mime_type": mime_type,
                    "media_url": media_url,
                    "data_url": data_url,
                    "size_bytes": len(data),
                })
            except Exception as exc:
                logger.warning(
                    "Failed to download media %s for message: %s",
                    media_url, exc,
                )
    return result
