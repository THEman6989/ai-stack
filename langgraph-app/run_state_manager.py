from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

try:
    from pymongo import MongoClient
except Exception as exc:  # pragma: no cover - optional at import time
    MongoClient = None  # type: ignore[assignment]
    PYMONGO_IMPORT_ERROR: Exception | None = exc
else:
    PYMONGO_IMPORT_ERROR = None


DEFAULT_CHECKPOINT_ID = "current"


def _enabled() -> bool:
    return os.getenv("ALPHARAVIS_RUN_STATE_MANAGER_ENABLED", "true").lower() in {"1", "true", "yes", "on"}


def _mongodb_uri() -> str:
    return os.getenv("LS_MONGODB_URI") or os.getenv("MONGODB_URI", "mongodb://mongodb:27017")


def _collection():
    if MongoClient is None:
        raise RuntimeError(f"pymongo unavailable: {PYMONGO_IMPORT_ERROR}")
    client = MongoClient(_mongodb_uri(), serverSelectionTimeoutMS=5000)
    return client[
        os.getenv("ALPHARAVIS_RUN_STATE_DB", "alpharavis_state")
    ][
        os.getenv("ALPHARAVIS_RUN_STATE_COLLECTION", "run_checkpoints")
    ]


def _workflow_collection():
    if MongoClient is None:
        raise RuntimeError(f"pymongo unavailable: {PYMONGO_IMPORT_ERROR}")
    client = MongoClient(_mongodb_uri(), serverSelectionTimeoutMS=5000)
    return client[
        os.getenv("ALPHARAVIS_RUN_STATE_DB", "alpharavis_state")
    ][
        os.getenv("ALPHARAVIS_WORKFLOW_STATE_COLLECTION", "workflow_records")
    ]


def _json_safe(value: Any, *, max_chars: int = 20000) -> Any:
    try:
        encoded = json.dumps(value, ensure_ascii=False, default=str)
        if len(encoded) > max_chars:
            return str(value)[:max_chars]
        return json.loads(encoded)
    except Exception:
        return str(value)[:max_chars]


def _compact_run_profile(profile: Any) -> dict[str, Any]:
    if not isinstance(profile, dict):
        return {}
    keep = [
        "started_at",
        "finished_at",
        "route",
        "route_reason",
        "planner_used",
        "planner_error",
        "planner_error_classification",
        "selected_toolsets",
        "active_source_keys",
        "active_rag_file_ids",
        "crisis_manager_used",
        "provider_hardening_last_retry",
        "provider_hardening_profile",
    ]
    return {key: _json_safe(profile.get(key), max_chars=4000) for key in keep if key in profile}


def build_run_checkpoint(
    *,
    thread_id: str,
    thread_key: str = "",
    phase: str,
    status: str,
    state: dict[str, Any] | None = None,
    error: str = "",
    error_classification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = state if isinstance(state, dict) else {}
    now = time.time()
    profile = state.get("run_profile") if isinstance(state.get("run_profile"), dict) else {}
    return {
        "_id": f"{thread_id}:{DEFAULT_CHECKPOINT_ID}",
        "checkpoint_id": DEFAULT_CHECKPOINT_ID,
        "thread_id": str(thread_id or "global"),
        "thread_key": str(thread_key or thread_id or "global"),
        "phase": str(phase or "unknown"),
        "status": str(status or "running"),
        "current_task_brief": str(state.get("current_task_brief") or "")[:12000],
        "planner_context": str(state.get("planner_context") or "")[:12000],
        "planner_last_key": str(state.get("planner_last_key") or ""),
        "active_agent": str(state.get("active_agent") or ""),
        "selected_toolsets": _json_safe(state.get("selected_toolsets") or [], max_chars=4000),
        "run_profile": _compact_run_profile(profile),
        "error": str(error or "")[:2000],
        "error_classification": _json_safe(error_classification or {}, max_chars=4000),
        "updated_at": now,
    }


def save_run_checkpoint(
    *,
    thread_id: str,
    thread_key: str = "",
    phase: str,
    status: str,
    state: dict[str, Any] | None = None,
    error: str = "",
    error_classification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not _enabled():
        return {"saved": False, "disabled": True}
    record = build_run_checkpoint(
        thread_id=thread_id,
        thread_key=thread_key,
        phase=phase,
        status=status,
        state=state,
        error=error,
        error_classification=error_classification,
    )
    previous = load_run_checkpoint(thread_id)
    if previous and "created_at" in previous:
        record["created_at"] = previous["created_at"]
    else:
        record["created_at"] = record["updated_at"]
    _collection().replace_one({"_id": record["_id"]}, record, upsert=True)
    return {"saved": True, "record": record}


def load_run_checkpoint(thread_id: str) -> dict[str, Any] | None:
    if not _enabled() or not thread_id:
        return None
    try:
        record = _collection().find_one({"_id": f"{thread_id}:{DEFAULT_CHECKPOINT_ID}"})
    except Exception:
        return None
    if not record:
        return None
    record["_id"] = str(record.get("_id") or "")
    return record


def list_run_checkpoints(*, status: str = "awaiting_resume", limit: int = 50) -> list[dict[str, Any]]:
    if not _enabled():
        return []
    query: dict[str, Any] = {}
    if status:
        query["status"] = status
    try:
        rows = _collection().find(query).sort("updated_at", -1).limit(max(1, min(int(limit), 200)))
    except Exception:
        return []
    records: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        row["_id"] = str(row.get("_id") or "")
        records.append(row)
    return records


def resume_updates_from_checkpoint(checkpoint: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        return {}
    status = str(checkpoint.get("status") or "")
    if status == "completed":
        return {}
    updates: dict[str, Any] = {
        "run_resume_checkpoint": {
            "thread_id": checkpoint.get("thread_id"),
            "phase": checkpoint.get("phase"),
            "status": status,
            "updated_at": checkpoint.get("updated_at"),
            "error": checkpoint.get("error", ""),
            "error_classification": checkpoint.get("error_classification") or {},
        }
    }
    for key in ["current_task_brief", "planner_context", "planner_last_key", "selected_toolsets"]:
        value = checkpoint.get(key)
        if value:
            updates[key] = value
    return updates


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON through a temp file and os.replace for non-corrupt local snapshots."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)


def save_workflow_record(*, namespace: str, workflow_id: str, record: dict[str, Any] | None = None) -> dict[str, Any]:
    """Persist a named workflow record through the existing AlphaRavis state manager.

    This is intentionally generic so feature-specific workflows (Office, RAG,
    UI jobs, etc.) do not grow separate state-manager modules.
    """

    if not _enabled():
        return {"saved": False, "disabled": True}
    ns = str(namespace or "workflow").strip() or "workflow"
    wid = str(workflow_id or DEFAULT_CHECKPOINT_ID).strip() or DEFAULT_CHECKPOINT_ID
    now = time.time()
    payload = _json_safe(record or {}, max_chars=50000)
    if not isinstance(payload, dict):
        payload = {"value": payload}
    previous = load_workflow_record(ns, wid)
    stored: dict[str, Any] = {
        **payload,
        "_id": f"{ns}:{wid}",
        "namespace": ns,
        "workflow_id": wid,
        "created_at": previous.get("created_at") if previous else now,
        "updated_at": now,
    }
    try:
        _workflow_collection().replace_one({"_id": stored["_id"]}, stored, upsert=True)
    except Exception as exc:
        return {"saved": False, "error": str(exc)[:2000], "record": stored}
    return {"saved": True, "record": stored}


def load_workflow_record(namespace: str, workflow_id: str) -> dict[str, Any] | None:
    if not _enabled() or not namespace or not workflow_id:
        return None
    try:
        record = _workflow_collection().find_one({"_id": f"{namespace}:{workflow_id}"})
    except Exception:
        return None
    if not isinstance(record, dict):
        return None
    record["_id"] = str(record.get("_id") or "")
    return record


def list_workflow_records(
    *,
    namespace: str,
    status: str = "",
    file: str = "",
    limit: int = 50,
) -> list[dict[str, Any]]:
    if not _enabled() or not namespace:
        return []
    query: dict[str, Any] = {"namespace": namespace}
    if status:
        query["status"] = status
    if file:
        query["file"] = file
    try:
        rows = _workflow_collection().find(query).sort("updated_at", -1).limit(max(1, min(int(limit), 500)))
    except Exception:
        return []
    records: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        row["_id"] = str(row.get("_id") or "")
        records.append(row)
    return records
