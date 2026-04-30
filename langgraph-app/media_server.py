from __future__ import annotations

import hashlib
import html
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

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
DOWNLOAD_ENABLED = os.getenv("ALPHARAVIS_MEDIA_DOWNLOAD_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
MAX_DOWNLOAD_BYTES = int(os.getenv("ALPHARAVIS_MEDIA_MAX_DOWNLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))

app = FastAPI(title="AlphaRavis Media Gallery", openapi_version="3.1.0")
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
            str(time.time_ns()),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _extension_from_url(url: str, media_type: str) -> str:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix and len(suffix) <= 12:
        return suffix
    return {"image": ".png", "video": ".mp4", "audio": ".wav", "document": ".bin"}.get(media_type, ".bin")


async def _download_asset(source_url: str, target: Path) -> dict[str, Any]:
    size = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
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
    day = time.strftime("%Y-%m-%d")
    group_id = _safe_segment(request.group_id or request.thread_key or request.thread_id or "ungrouped", "ungrouped")
    relative_path = ""
    local_path = ""
    public_url = request.source_url
    download_error = ""

    if request.source_url and request.download and DOWNLOAD_ENABLED:
        filename = f"{asset_id}-{_safe_segment(request.title or request.source_key or request.media_type)}{_extension_from_url(request.source_url, request.media_type)}"
        target = MEDIA_ROOT / day / group_id / request.role / filename
        try:
            await _download_asset(request.source_url, target)
            relative_path = str(target.relative_to(MEDIA_ROOT))
            local_path = str(target)
            public_url = _public_url(relative_path)
        except Exception as exc:
            download_error = str(exc)

    record = {
        "_id": asset_id,
        "asset_id": asset_id,
        "source_url": request.source_url,
        "file_id": request.file_id,
        "source_key": request.source_key or asset_id,
        "thread_id": request.thread_id,
        "thread_key": request.thread_key,
        "group_id": group_id,
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
        "download_error": download_error,
        "created_at": int(time.time()),
    }
    _collection().replace_one({"_id": asset_id}, record, upsert=True)
    return record


@app.get("/assets")
async def list_assets(limit: int = 200, media_type: str = "all", thread_id: str = ""):
    query: dict[str, Any] = {}
    if media_type and media_type != "all":
        query["media_type"] = media_type
    if thread_id:
        query["thread_id"] = thread_id
    rows = list(_collection().find(query).sort("created_at", -1).limit(max(1, min(limit, 1000))))
    for row in rows:
        row["_id"] = str(row["_id"])
    return {"assets": rows}


@app.get("/gallery", response_class=HTMLResponse)
async def gallery(limit: int = 300):
    rows = list(_collection().find({}).sort("created_at", -1).limit(max(1, min(limit, 1000))))
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        day = time.strftime("%Y-%m-%d", time.localtime(int(row.get("created_at") or 0)))
        key = f"{day} / {row.get('group_id', 'ungrouped')}"
        groups.setdefault(key, []).append(row)

    body = [
        "<!doctype html><html><head><meta charset='utf-8'><title>AlphaRavis Media Gallery</title>",
        "<style>body{font-family:system-ui;margin:24px;background:#101214;color:#eee}details{margin:16px 0;border:1px solid #333;padding:12px;border-radius:8px}summary{cursor:pointer;font-weight:700}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}.card{background:#181b1f;border:1px solid #2b3036;border-radius:8px;padding:10px}img,video{max-width:100%;border-radius:6px;background:#000}.meta{font-size:12px;color:#aaa;word-break:break-word}</style>",
        "</head><body><h1>AlphaRavis Media Gallery</h1>",
    ]
    for group, assets in groups.items():
        body.append(f"<details open><summary>{html.escape(group)} ({len(assets)})</summary><div class='grid'>")
        for asset in assets:
            title = html.escape(str(asset.get("title") or asset.get("asset_id")))
            url = html.escape(str(asset.get("public_url") or asset.get("source_url") or ""))
            media_type = asset.get("media_type")
            body.append("<div class='card'>")
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
                f"thread={html.escape(str(asset.get('thread_key') or asset.get('thread_id')))}"
                "</div>"
            )
            body.append("</div>")
        body.append("</div></details>")
    body.append("</body></html>")
    return "\n".join(body)
