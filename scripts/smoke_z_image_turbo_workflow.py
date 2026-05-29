#!/usr/bin/env python3
"""Submit the saved z_image_turbo ComfyUI workflow and register outputs.

Intended to run inside the langgraph-api container after enabling:
ALPHARAVIS_ENABLE_COMFYUI_WORKFLOW_SUBMIT=true

    docker compose exec langgraph-api python /workspace/scripts/smoke_z_image_turbo_workflow.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

for candidate in ("/app", "/workspace/langgraph-app"):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from comfyui_client import ComfyUIClient
from comfyui_workflow_library import get_comfyui_workflow_record, submit_saved_comfyui_workflow_record


WORKFLOW_NAME = os.getenv("ALPHARAVIS_COMFYUI_SMOKE_WORKFLOW", "z_image_turbo")
TIMEOUT_SECONDS = float(os.getenv("ALPHARAVIS_COMFYUI_SMOKE_TIMEOUT_SECONDS", "300"))
POLL_SECONDS = float(os.getenv("ALPHARAVIS_COMFYUI_SMOKE_POLL_SECONDS", "3"))
MEDIA_GALLERY_URL = os.getenv("ALPHARAVIS_MEDIA_GALLERY_URL", "http://media-gallery:8130").rstrip("/")


def _post_json(url: str, payload: dict[str, Any], timeout: float = 30) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "status_code": exc.code, "error": raw[:2000]}
    return json.loads(raw) if raw else {}


async def main() -> int:
    loaded = get_comfyui_workflow_record(WORKFLOW_NAME, include_workflow=False)
    print("=== SAVED WORKFLOW ===")
    print(json.dumps(loaded, indent=2, default=str)[:4000])
    if not loaded.get("ok") or not loaded.get("found"):
        print("Workflow not found; aborting.")
        return 2

    client = ComfyUIClient()
    prompt = "AlphaRavis ComfyUI smoke test: a small friendly robot holding a sign that says AlphaRavis"
    parameters = {
        "prompt": prompt,
        "seed": int(time.time()) % 1_000_000_000,
        "steps": 4,
        "cfg": 1,
        "width": 1024,
        "height": 1024,
    }

    print("\n=== SUBMIT ===")
    submit = await submit_saved_comfyui_workflow_record(
        WORKFLOW_NAME,
        parameters,
        client=client,
        client_id="alpharavis-live-smoke",
    )
    print(json.dumps(submit, indent=2, default=str)[:8000])
    submit_result = submit.get("submit_result") if isinstance(submit.get("submit_result"), dict) else {}
    prompt_id = str(submit_result.get("prompt_id") or "")
    if not submit.get("ok") or not prompt_id:
        print("Submit did not return a prompt_id; aborting before history/register.")
        return 3

    print(f"\n=== POLL HISTORY {prompt_id} ===")
    deadline = time.time() + TIMEOUT_SECONDS
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = await client.history_outputs(prompt_id)
        outputs = last.get("outputs") or []
        if outputs:
            print(json.dumps({"prompt_id": prompt_id, "outputs": outputs}, indent=2, default=str)[:8000])
            break
        queue = await client.queue()
        print(json.dumps({"waiting": True, "prompt_id": prompt_id, "queue_running": queue.get("queue_running"), "queue_pending": queue.get("queue_pending")}, default=str)[:2000])
        await asyncio.sleep(POLL_SECONDS)
    else:
        print(json.dumps({"ok": False, "prompt_id": prompt_id, "error": "Timed out waiting for ComfyUI history outputs", "last_history": last}, indent=2, default=str)[:8000])
        return 4

    outputs = last.get("outputs") or []
    print("\n=== REGISTER OUTPUTS ===")
    registration = _post_json(
        f"{MEDIA_GALLERY_URL}/comfyui/outputs/register",
        {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "outputs": outputs,
            "source_base_url": getattr(client, "public_base_url", ""),
            "download": False,
        },
    )
    print(json.dumps(registration, indent=2, default=str)[:8000])
    if not registration.get("ok"):
        return 5

    print("\n=== SUMMARY ===")
    print(json.dumps({"ok": True, "workflow": WORKFLOW_NAME, "prompt_id": prompt_id, "output_count": len(outputs), "registration": registration}, indent=2, default=str)[:8000])
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
