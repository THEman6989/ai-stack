from __future__ import annotations

import base64
import binascii
import hashlib
import html
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote_to_bytes, urlparse

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from file_safety import ensure_write_allowed

try:
    from pymongo import MongoClient
except Exception as exc:  # pragma: no cover - optional at import time
    MongoClient = None  # type: ignore[assignment]
    PYMONGO_IMPORT_ERROR: Exception | None = exc
else:
    PYMONGO_IMPORT_ERROR = None


MEDIA_ROOT = Path(os.getenv("ALPHARAVIS_MEDIA_ROOT", "/media-data")).expanduser().resolve()
PUBLIC_BASE_URL = os.getenv("ALPHARAVIS_MEDIA_PUBLIC_BASE_URL", "http://localhost:8130").rstrip("/")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://mongodb:27017")
MONGO_DB = os.getenv("ALPHARAVIS_MEDIA_MONGO_DB", "alpharavis_media")
MONGO_COLLECTION = os.getenv("ALPHARAVIS_MEDIA_MONGO_COLLECTION", "assets")
MONGO_REFERENCES_COLLECTION = os.getenv("ALPHARAVIS_MEDIA_MONGO_REFERENCES_COLLECTION", "references")
DOWNLOAD_ENABLED = os.getenv("ALPHARAVIS_MEDIA_DOWNLOAD_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
MAX_DOWNLOAD_BYTES = int(os.getenv("ALPHARAVIS_MEDIA_MAX_DOWNLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))

app = FastAPI(title="AlphaRavis Media Gallery", openapi_version="3.1.0")
ensure_write_allowed(MEDIA_ROOT, allowed_root=MEDIA_ROOT)
MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=str(MEDIA_ROOT)), name="media")


class MediaRegisterRequest(BaseModel):
    source_url: str = ""
    file_id: str = ""
    source_key: str = ""
    thread_id: str = ""
    thread_key: str = ""
    group_id: str = ""
    role: str = "output"
    asset_kind: str = ""
    origin: str = ""
    parent_asset_id: str = ""
    root_asset_id: str = ""
    derivation_group_id: str = ""
    source_message_id: str = ""
    result_message_id: str = ""
    tool_call_id: str = ""
    processing_provider: str = ""
    processing_prompt: str = ""
    media_type: str = "unknown"
    mime_type: str = ""
    title: str = ""
    prompt: str = ""
    caption: str = ""
    download: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


def _collection():
    if MongoClient is None:
        raise RuntimeError(f"pymongo unavailable: {PYMONGO_IMPORT_ERROR}")
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    return client[MONGO_DB][MONGO_COLLECTION]


def _references_collection():
    if MongoClient is None:
        raise RuntimeError(f"pymongo unavailable: {PYMONGO_IMPORT_ERROR}")
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    return client[MONGO_DB][MONGO_REFERENCES_COLLECTION]


def _safe_segment(value: str, default: str = "asset") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", (value or "").strip().lower()).strip("-._")
    return cleaned[:96] or default


def _asset_id(request: MediaRegisterRequest) -> str:
    raw = "|".join(
        [
            request.source_url,
            request.file_id,
            request.source_key,
            request.thread_id,
            request.role,
            request.media_type,
            request.asset_kind,
            request.derivation_group_id,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _reference_id(asset_id: str, request: MediaRegisterRequest) -> str:
    raw = "|".join(
        [
            asset_id,
            request.thread_id,
            request.thread_key,
            request.source_message_id,
            request.result_message_id,
            request.tool_call_id,
            request.role,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _extension_from_url(url: str, media_type: str) -> str:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix and len(suffix) <= 12:
        return suffix
    return {"image": ".png", "video": ".mp4", "audio": ".wav", "document": ".bin"}.get(media_type, ".bin")


def _stored_source_url(source_url: str) -> str:
    if source_url.lower().startswith("data:"):
        header = source_url.partition(",")[0]
        return f"{header},[inline-data-omitted]"
    return source_url


async def _download_asset(source_url: str, target: Path) -> dict[str, Any]:
    size = 0
    ensure_write_allowed(target, allowed_root=MEDIA_ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    if source_url.lower().startswith("data:"):
        header, separator, payload = source_url.partition(",")
        if not separator:
            raise RuntimeError("invalid data URL")
        try:
            data = (
                base64.b64decode(payload, validate=True)
                if header.lower().endswith(";base64")
                else unquote_to_bytes(payload)
            )
        except (ValueError, binascii.Error) as exc:
            raise RuntimeError(f"invalid data URL payload: {exc}") from exc
        size = len(data)
        if size > MAX_DOWNLOAD_BYTES:
            raise RuntimeError(f"download exceeds limit {MAX_DOWNLOAD_BYTES} bytes")
        with tmp.open("wb") as fh:
            fh.write(data)
        os.replace(tmp, target)
        return {"bytes": size, "path": str(target)}

    async with httpx.AsyncClient(timeout=float(os.getenv("ALPHARAVIS_MEDIA_DOWNLOAD_TIMEOUT_SECONDS", "120"))) as client:
        async with client.stream("GET", source_url) as response:
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}")
            with tmp.open("wb") as fh:
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > MAX_DOWNLOAD_BYTES:
                        raise RuntimeError(f"download exceeds limit {MAX_DOWNLOAD_BYTES} bytes")
                    fh.write(chunk)
    os.replace(tmp, target)
    return {"bytes": size, "path": str(target)}


def _public_url(relative_path: str) -> str:
    return f"{PUBLIC_BASE_URL}/media/{relative_path.replace(os.sep, '/')}"


def _asset_kind_from_request(request: MediaRegisterRequest) -> str:
    value = (request.asset_kind or "").strip().lower()
    if value in {"original", "processed", "reference", "unknown"}:
        return value
    role = (request.role or "").strip().lower()
    origin = (request.origin or "").strip().lower()
    provider = (request.processing_provider or "").strip().lower()
    if role in {"input", "reference"} or origin in {"librechat_upload", "chat_url", "meet_server"}:
        return "original" if role == "input" or origin == "librechat_upload" else "reference"
    if role == "output" or provider or origin in {"pixelle_output", "processed"}:
        return "processed"
    return "unknown"


def _derivation_group(request: MediaRegisterRequest, asset_id: str) -> str:
    group = (
        request.derivation_group_id
        or request.root_asset_id
        or request.parent_asset_id
        or request.group_id
        or request.thread_key
        or request.thread_id
        or request.source_key
        or asset_id
    )
    return _safe_segment(group, "ungrouped")


@app.get("/health")
async def health():
    mongo_ok = False
    mongo_error = ""
    try:
        _collection().database.client.admin.command("ping")
        mongo_ok = True
    except Exception as exc:
        mongo_error = str(exc)
    return {
        "status": "ok" if mongo_ok else "degraded",
        "media_root": str(MEDIA_ROOT),
        "public_base_url": PUBLIC_BASE_URL,
        "mongo_ok": mongo_ok,
        "mongo_error": mongo_error,
    }


@app.post("/assets/register")
async def register_asset(request: MediaRegisterRequest):
    if not request.source_url and not request.file_id:
        raise HTTPException(status_code=400, detail="source_url or file_id is required")

    request.role = request.role if request.role in {"input", "output", "reference", "unknown"} else "unknown"
    request.media_type = request.media_type if request.media_type in {"image", "video", "audio", "document", "unknown"} else "unknown"

    asset_id = _asset_id(request)
    asset_kind = _asset_kind_from_request(request)
    day = time.strftime("%Y-%m-%d")
    group_id = _derivation_group(request, asset_id)
    relative_path = ""
    local_path = ""
    public_url = request.source_url
    download_error = ""

    if request.source_url and request.download and DOWNLOAD_ENABLED:
        filename = f"{asset_id}-{_safe_segment(request.title or request.source_key or request.media_type)}{_extension_from_url(request.source_url, request.media_type)}"
        target = MEDIA_ROOT / day / group_id / request.role / filename
        try:
            if not target.exists():
                await _download_asset(request.source_url, target)
            relative_path = str(target.relative_to(MEDIA_ROOT))
            local_path = str(target)
            public_url = _public_url(relative_path)
        except Exception as exc:
            download_error = str(exc)

    record = {
        "_id": asset_id,
        "asset_id": asset_id,
        "source_url": _stored_source_url(request.source_url),
        "file_id": request.file_id,
        "source_key": request.source_key or asset_id,
        "thread_id": request.thread_id,
        "thread_key": request.thread_key,
        "group_id": group_id,
        "asset_kind": asset_kind,
        "origin": request.origin,
        "parent_asset_id": request.parent_asset_id,
        "root_asset_id": request.root_asset_id or request.parent_asset_id or asset_id,
        "derivation_group_id": request.derivation_group_id or group_id,
        "source_message_id": request.source_message_id,
        "result_message_id": request.result_message_id,
        "tool_call_id": request.tool_call_id,
        "processing_provider": request.processing_provider,
        "processing_prompt": request.processing_prompt,
        "role": request.role,
        "media_type": request.media_type,
        "mime_type": request.mime_type,
        "title": request.title or request.source_key or asset_id,
        "prompt": request.prompt,
        "caption": request.caption,
        "metadata": request.metadata,
        "relative_path": relative_path,
        "local_path": local_path,
        "public_url": public_url,
        "download_url": public_url,
        "thumbnail_path": str(request.metadata.get("thumbnail_path", "")) if isinstance(request.metadata, dict) else "",
        "preview_path": str(request.metadata.get("preview_path", "")) if isinstance(request.metadata, dict) else "",
        "download_error": download_error,
        "created_at": int(time.time()),
    }
    _collection().replace_one({"_id": asset_id}, record, upsert=True)
    if request.thread_id or request.thread_key or request.source_message_id or request.result_message_id:
        reference_id = _reference_id(asset_id, request)
        reference = {
            "_id": reference_id,
            "reference_id": reference_id,
            "asset_id": asset_id,
            "source_key": record["source_key"],
            "media_type": request.media_type,
            "thread_id": request.thread_id,
            "thread_key": request.thread_key,
            "source_message_id": request.source_message_id,
            "result_message_id": request.result_message_id,
            "tool_call_id": request.tool_call_id,
            "role": request.role,
            "reason": str(request.metadata.get("reference_reason", "registered_media")) if isinstance(request.metadata, dict) else "registered_media",
            "created_at": int(time.time()),
        }
        _references_collection().replace_one({"_id": reference_id}, reference, upsert=True)
        record["reference_id"] = reference_id
    return record


@app.get("/assets")
async def list_assets(limit: int = 200, media_type: str = "all", thread_id: str = "", asset_kind: str = "all"):
    query: dict[str, Any] = {}
    if media_type and media_type != "all":
        query["media_type"] = media_type
    if thread_id:
        query["thread_id"] = thread_id
    if asset_kind and asset_kind != "all":
        query["asset_kind"] = asset_kind
    rows = list(_collection().find(query).sort("created_at", -1).limit(max(1, min(limit, 1000))))
    for row in rows:
        row["_id"] = str(row["_id"])
    return {"assets": rows}


@app.get("/assets/resolve")
async def resolve_asset(url: str = "", source_key: str = "", asset_id: str = ""):
    query: dict[str, Any] = {}
    if asset_id:
        query["_id"] = asset_id
    elif source_key:
        query["source_key"] = source_key
    elif url:
        query["$or"] = [{"public_url": url}, {"download_url": url}, {"source_url": url}]
    else:
        raise HTTPException(status_code=400, detail="url, source_key, or asset_id is required")
    row = _collection().find_one(query)
    if not row:
        raise HTTPException(status_code=404, detail="asset not found")
    row["_id"] = str(row["_id"])
    references = list(_references_collection().find({"asset_id": row.get("asset_id")}).sort("created_at", -1).limit(50))
    for ref in references:
        ref["_id"] = str(ref["_id"])
    row["references"] = references
    return row


@app.get("/gallery", response_class=HTMLResponse)
async def gallery(limit: int = 300, view: str = "all", media_type: str = "all"):
    query: dict[str, Any] = {}
    view = view if view in {"all", "original", "processed"} else "all"
    if view != "all":
        query["asset_kind"] = view
    if media_type and media_type != "all":
        query["media_type"] = media_type
    rows = list(_collection().find(query).sort("created_at", -1).limit(max(1, min(limit, 1000))))
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        day = time.strftime("%Y-%m-%d", time.localtime(int(row.get("created_at") or 0)))
        key = f"{day} / {row.get('derivation_group_id') or row.get('group_id', 'ungrouped')}"
        groups.setdefault(key, []).append(row)

    body = [
        "<!doctype html><html><head><meta charset='utf-8'><title>AlphaRavis Media Gallery</title>",
        "<style>"
        "body{font-family:system-ui;margin:0;background:#101214;color:#eee}"
        "header{position:sticky;top:0;background:#101214;padding:18px 24px;border-bottom:1px solid #2b3036;z-index:2}"
        "main{padding:18px 24px}.tabs{display:flex;gap:8px;flex-wrap:wrap}.tabs a{color:#ddd;text-decoration:none;border:1px solid #343a42;border-radius:7px;padding:7px 11px}.tabs a.active{background:#e8eef8;color:#111;border-color:#e8eef8}"
        "details{margin:16px 0;border-top:1px solid #333;padding-top:12px}summary{cursor:pointer;font-weight:700}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}.card{position:relative;background:#181b1f;border:1px solid #2b3036;border-radius:8px;padding:10px;min-height:180px}img,video{width:100%;aspect-ratio:16/9;object-fit:contain;border-radius:6px;background:#000}.meta{font-size:12px;color:#aaa;word-break:break-word}.badge{display:inline-block;font-size:11px;text-transform:uppercase;letter-spacing:.04em;border:1px solid #46515d;border-radius:6px;padding:2px 6px;color:#cdd7e1}.actions{position:absolute;right:10px;bottom:10px;display:flex;gap:6px}.actions button,.actions a{font-size:12px;background:#222831;color:#eee;border:1px solid #3a434d;border-radius:6px;padding:5px 7px;text-decoration:none;cursor:pointer}h3{font-size:15px;margin:8px 0;line-height:1.25}"
        "</style>",
        "<script>function copyLink(url){navigator.clipboard&&navigator.clipboard.writeText(url)}</script>",
        "</head><body><header><h1>AlphaRavis Media Gallery</h1><nav class='tabs'>",
        _tab_link("All", "all", view, media_type, limit),
        _tab_link("Original", "original", view, media_type, limit),
        _tab_link("Processed", "processed", view, media_type, limit),
        "</nav></header><main>",
    ]
    for group, assets in groups.items():
        body.append(f"<details open><summary>{html.escape(group)} ({len(assets)})</summary><div class='grid'>")
        for asset in assets:
            title = html.escape(str(asset.get("title") or asset.get("asset_id")))
            url = html.escape(str(asset.get("public_url") or asset.get("source_url") or ""))
            media_type = asset.get("media_type")
            asset_kind = html.escape(str(asset.get("asset_kind") or asset.get("role") or "unknown"))
            metadata = asset.get("metadata") if isinstance(asset.get("metadata"), dict) else {}
            provider = html.escape(str(asset.get("processing_provider") or metadata.get("provider", "")))
            body.append("<div class='card'>")
            body.append(f"<span class='badge'>{asset_kind}</span>")
            body.append(f"<h3>{title}</h3>")
            if url and media_type == "image":
                body.append(f"<a href='{url}'><img src='{url}' loading='lazy'></a>")
            elif url and media_type == "video":
                body.append(f"<video src='{url}' controls preload='metadata'></video>")
            elif url:
                body.append(f"<a href='{url}'>{url}</a>")
            body.append(
                "<div class='meta'>"
                f"role={html.escape(str(asset.get('role')))}<br>"
                f"type={html.escape(str(media_type))}<br>"
                f"source_key={html.escape(str(asset.get('source_key')))}<br>"
                f"provider={provider}<br>"
                f"thread={html.escape(str(asset.get('thread_key') or asset.get('thread_id')))}"
                "</div>"
            )
            if url:
                body.append(
                    "<div class='actions'>"
                    f"<button data-url='{url}' onclick='copyLink(this.dataset.url)'>Copy Link</button>"
                    f"<a href='{url}' target='_blank' rel='noreferrer'>Open</a>"
                    "</div>"
                )
            body.append("</div>")
        body.append("</div></details>")
    if not groups:
        body.append("<p>No media matched the selected filters.</p>")
    body.append("</main></body></html>")
    return "\n".join(body)


def _tab_link(label: str, value: str, current: str, media_type: str, limit: int) -> str:
    active = " active" if value == current else ""
    href = f"/gallery?view={value}&media_type={html.escape(media_type)}&limit={int(limit)}"
    return f"<a class='{active}' href='{href}'>{html.escape(label)}</a>"
