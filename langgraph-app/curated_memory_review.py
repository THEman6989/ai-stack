from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from maintenance_helpers import extract_review_insight_candidates


def _review_path() -> Path:
    raw = os.getenv("ALPHARAVIS_CURATED_MEMORY_REVIEW_PATH", "/tmp/alpharavis_curated_memory_review.json")
    return Path(raw).expanduser()


def _empty_state() -> dict[str, Any]:
    return {"version": 1, "items": []}


def _load_state() -> dict[str, Any]:
    path = _review_path()
    if not path.exists():
        return _empty_state()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _empty_state()
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        return _empty_state()
    return data


def _save_state(state: dict[str, Any]) -> None:
    path = _review_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _candidate_key(candidate: dict[str, Any], *, source_key: str = "", thread_id: str = "") -> str:
    payload = {
        "candidate": str(candidate.get("candidate") or candidate.get("memory") or ""),
        "kind": str(candidate.get("kind") or candidate.get("memory_type") or ""),
        "source_key": source_key,
        "thread_id": thread_id,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def extract_candidates(
    text: str,
    *,
    source_key: str = "",
    source_type: str = "thread",
    thread_id: str = "",
    title: str = "",
    max_candidates: int = 8,
) -> dict[str, Any]:
    now = int(time.time())
    extracted = extract_review_insight_candidates(text, max_candidates=max_candidates)
    state = _load_state()
    existing = {str(item.get("candidate_id")): item for item in state.get("items", []) if isinstance(item, dict)}
    items: list[dict[str, Any]] = []
    auto_accept_threshold = os.getenv("ALPHARAVIS_CURATED_MEMORY_AUTO_ACCEPT_THRESHOLD", "").strip()
    threshold = float(auto_accept_threshold) if auto_accept_threshold else 2.0  # Default 2.0 means never auto-accept

    for raw in extracted:
        candidate_id = _candidate_key(raw, source_key=source_key, thread_id=thread_id)
        current = dict(existing.get(candidate_id) or {})
        if current.get("status") in {"accepted", "rejected"}:
            items.append(current)
            continue
        
        confidence = float(raw.get("confidence") or 0.0)
        status = "pending"
        reviewer_note = ""
        
        if confidence >= threshold:
            status = "accepted"
            reviewer_note = f"Auto-accepted (confidence {confidence} >= threshold {threshold})"

        item = {
            **current,
            "candidate_id": candidate_id,
            "status": status,
            "memory": str(raw.get("candidate") or "")[:1200],
            "memory_type": str(raw.get("kind") or "fact")[:80],
            "confidence": confidence,
            "review_required": status == "pending",
            "reviewer_note": current.get("reviewer_note") or reviewer_note,
            "source_preview": str(raw.get("source_preview") or "")[:1200],
            "source_key": source_key,
            "source_type": source_type,
            "thread_id": thread_id,
            "title": title,
            "created_at": current.get("created_at") or now,
            "updated_at": now,
        }
        existing[candidate_id] = item
        items.append(item)
    state["items"] = list(existing.values())
    _save_state(state)
    return {"ok": True, "count": len(items), "items": items}


def list_candidates(status: str = "pending", limit: int = 50) -> dict[str, Any]:
    status = str(status or "pending").strip().lower()
    limit = max(1, min(int(limit or 50), 200))
    items = [item for item in _load_state().get("items", []) if isinstance(item, dict)]
    if status not in {"all", "*"}:
        items = [item for item in items if str(item.get("status") or "pending") == status]
    items.sort(key=lambda item: int(item.get("updated_at") or item.get("created_at") or 0), reverse=True)
    return {"ok": True, "status": status, "items": items[:limit]}


def update_candidate(candidate_id: str, *, status: str, reviewer_note: str = "", memory_key: str = "") -> dict[str, Any]:
    candidate_id = str(candidate_id or "").strip()
    normalized = str(status or "").strip().lower()
    if normalized not in {"accepted", "rejected", "pending"}:
        return {"ok": False, "error": "status must be accepted, rejected, or pending"}
    state = _load_state()
    for item in state.get("items", []):
        if not isinstance(item, dict) or str(item.get("candidate_id")) != candidate_id:
            continue
        item["status"] = normalized
        item["reviewer_note"] = str(reviewer_note or "")[:1200]
        if memory_key:
            item["memory_key"] = memory_key
        item["updated_at"] = int(time.time())
        _save_state(state)
        return {"ok": True, "item": item}
    return {"ok": False, "error": "candidate not found"}

