from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


DEFAULT_REVIEW_PATH = "/tmp/alpharavis_run_reviews.json"


def review_store_path() -> Path:
    return Path(os.getenv("ALPHARAVIS_ASYNC_REVIEW_STORE_PATH", DEFAULT_REVIEW_PATH))


def _read_store() -> dict[str, Any]:
    path = review_store_path()
    if not path.exists():
        return {"reviews": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"reviews": {}}
    if not isinstance(payload, dict):
        return {"reviews": {}}
    reviews = payload.get("reviews")
    if not isinstance(reviews, dict):
        payload["reviews"] = {}
    return payload


def _write_store(payload: dict[str, Any]) -> None:
    path = review_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def save_run_review(
    thread_id: str,
    *,
    thread_key: str = "",
    task_brief: str = "",
    review_text: str = "",
    status: str = "pending",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    thread_id = str(thread_id or "").strip()
    if not thread_id:
        raise ValueError("thread_id is required")
    now = time.time()
    record = {
        "thread_id": thread_id,
        "thread_key": str(thread_key or ""),
        "task_brief": str(task_brief or "")[:4000],
        "review_text": str(review_text or "")[:12000],
        "status": str(status or "pending"),
        "created_at": now,
        "updated_at": now,
        "metadata": metadata or {},
    }
    payload = _read_store()
    payload.setdefault("reviews", {})[thread_id] = record
    _write_store(payload)
    return record


def load_pending_run_review(thread_id: str) -> dict[str, Any] | None:
    payload = _read_store()
    record = payload.get("reviews", {}).get(str(thread_id or ""))
    if not isinstance(record, dict):
        return None
    if str(record.get("status") or "") != "pending":
        return None
    return dict(record)


def mark_run_review_delivered(thread_id: str) -> None:
    payload = _read_store()
    record = payload.get("reviews", {}).get(str(thread_id or ""))
    if not isinstance(record, dict):
        return
    record["status"] = "delivered"
    record["updated_at"] = time.time()
    _write_store(payload)
