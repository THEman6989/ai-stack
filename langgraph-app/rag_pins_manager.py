from __future__ import annotations

import os
import time
from typing import Any

try:
    from pymongo import MongoClient
except Exception as exc:  # pragma: no cover - optional at import time
    MongoClient = None  # type: ignore[assignment]
    PYMONGO_IMPORT_ERROR: Exception | None = exc
else:
    PYMONGO_IMPORT_ERROR = None


def _enabled() -> bool:
    return os.getenv("ALPHARAVIS_RAG_PINS_MANAGER_ENABLED", "true").lower() in {"1", "true", "yes", "on"}


def _mongodb_uri() -> str:
    return os.getenv("LS_MONGODB_URI") or os.getenv("MONGODB_URI", "mongodb://mongodb:27017")


def _collection():
    if MongoClient is None:
        raise RuntimeError(f"pymongo unavailable: {PYMONGO_IMPORT_ERROR}")
    client = MongoClient(_mongodb_uri(), serverSelectionTimeoutMS=5000)
    return client[
        os.getenv("ALPHARAVIS_RAG_PINS_DB", os.getenv("ALPHARAVIS_RUN_STATE_DB", "alpharavis_state"))
    ][
        os.getenv("ALPHARAVIS_RAG_PINS_COLLECTION", "rag_thread_pins")
    ]


def _normalize(values: Any) -> list[str]:
    if isinstance(values, str):
        raw = [part.strip() for part in values.split(",")]
    elif isinstance(values, (list, tuple, set)):
        raw = list(values)
    elif values:
        raw = [values]
    else:
        raw = []
    output: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = str(item or "").strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def load_pins(thread_id: str) -> dict[str, Any]:
    if not _enabled() or not str(thread_id or "").strip():
        return {}
    try:
        record = _collection().find_one({"_id": str(thread_id)})
    except Exception:
        return {}
    if not isinstance(record, dict):
        return {}
    return {
        "thread_id": str(record.get("thread_id") or record.get("_id") or thread_id),
        "rag_active": bool(record.get("rag_active")),
        "active_source_keys": _normalize(record.get("active_source_keys")),
        "active_rag_file_ids": _normalize(record.get("active_rag_file_ids")),
        "archive_rag_mode": str(record.get("archive_rag_mode") or "tool_only"),
        "updated_at": record.get("updated_at"),
    }


def save_pins(
    *,
    thread_id: str,
    active_source_keys: list[str] | None = None,
    active_rag_file_ids: list[str] | None = None,
    archive_rag_mode: str = "tool_only",
) -> dict[str, Any]:
    if not _enabled():
        raise RuntimeError("RAG pins manager is disabled.")
    thread_id = str(thread_id or "").strip()
    if not thread_id:
        raise ValueError("thread_id is required.")
    now = int(time.time())
    record = {
        "_id": thread_id,
        "thread_id": thread_id,
        "rag_active": bool(active_source_keys or active_rag_file_ids),
        "active_source_keys": _normalize(active_source_keys),
        "active_rag_file_ids": _normalize(active_rag_file_ids),
        "archive_rag_mode": str(archive_rag_mode or "tool_only"),
        "updated_at": now,
    }
    _collection().replace_one({"_id": thread_id}, record, upsert=True)
    record.pop("_id", None)
    return record


def update_pins(
    *,
    thread_id: str,
    add_source_keys: list[str] | None = None,
    add_rag_file_ids: list[str] | None = None,
    remove_source_keys: list[str] | None = None,
    remove_rag_file_ids: list[str] | None = None,
    clear_all: bool = False,
    archive_rag_mode: str = "",
) -> dict[str, Any]:
    existing = load_pins(thread_id)
    if clear_all:
        source_keys: list[str] = []
        rag_file_ids: list[str] = []
    else:
        remove_sources = set(_normalize(remove_source_keys))
        remove_files = set(_normalize(remove_rag_file_ids))
        source_keys = [item for item in _normalize(existing.get("active_source_keys")) if item not in remove_sources]
        rag_file_ids = [item for item in _normalize(existing.get("active_rag_file_ids")) if item not in remove_files]
        source_keys = _normalize([*source_keys, *_normalize(add_source_keys)])
        rag_file_ids = _normalize([*rag_file_ids, *_normalize(add_rag_file_ids)])
    mode = archive_rag_mode or str(existing.get("archive_rag_mode") or "tool_only")
    return save_pins(
        thread_id=thread_id,
        active_source_keys=source_keys,
        active_rag_file_ids=rag_file_ids,
        archive_rag_mode=mode,
    )


def list_pins(limit: int = 50) -> list[dict[str, Any]]:
    if not _enabled():
        return []
    try:
        rows = _collection().find({}).sort("updated_at", -1).limit(max(1, min(int(limit), 200)))
    except Exception:
        return []
    return [load_pins(str(row.get("thread_id") or row.get("_id") or "")) for row in rows if isinstance(row, dict)]
