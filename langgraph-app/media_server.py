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
from urllib.parse import urlencode, unquote_to_bytes, urlparse

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
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
OFFICE_OUTPUT_ROOT = Path(os.getenv("ALPHARAVIS_OFFICE_OUTPUT_ROOT", "/workspace/office-output")).expanduser().resolve()
OFFICE_OUTPUT_PUBLIC_BASE_URL = os.getenv(
    "ALPHARAVIS_OFFICE_OUTPUT_PUBLIC_BASE_URL",
    f"{PUBLIC_BASE_URL}/office-output",
).rstrip("/")
OFFICE_OUTPUT_EXTENSIONS = {".docx", ".pptx", ".xlsx", ".pdf", ".html", ".png", ".jpg", ".jpeg"}
OFFICE_UPLOAD_EXTENSIONS = {".docx", ".pptx", ".xlsx"}
DEFAULT_CORS_ALLOW_ORIGINS = (
    "http://localhost:3000,"
    "http://127.0.0.1:3000,"
    "http://localhost:3080,"
    "http://127.0.0.1:3080"
)
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://mongodb:27017")
MONGO_DB = os.getenv("ALPHARAVIS_MEDIA_MONGO_DB", "alpharavis_media")
MONGO_COLLECTION = os.getenv("ALPHARAVIS_MEDIA_MONGO_COLLECTION", "assets")
MONGO_REFERENCES_COLLECTION = os.getenv("ALPHARAVIS_MEDIA_MONGO_REFERENCES_COLLECTION", "references")
DOWNLOAD_ENABLED = os.getenv("ALPHARAVIS_MEDIA_DOWNLOAD_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
MAX_DOWNLOAD_BYTES = int(os.getenv("ALPHARAVIS_MEDIA_MAX_DOWNLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))
ASSET_SORT_FIELDS = {"created_at", "title", "media_type", "asset_kind", "thread_key", "group_id"}
GALLERY_GROUP_BY = {"day", "none", "thread", "group", "media_type"}
FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="14" fill="#0b1018"/><path d="M12 44V20h40v24H12Z" fill="#ff8a58"/><path d="M18 38l8-9 6 6 5-7 9 10H18Z" fill="#101620"/><circle cx="42" cy="25" r="4" fill="#101620"/></svg>"""
UPLOAD_MIME_TYPES = {
    "image/jpeg": "image",
    "image/png": "image",
    "image/gif": "image",
    "image/webp": "image",
    "image/heic": "image",
    "image/heif": "image",
    "video/mp4": "video",
    "video/quicktime": "video",
    "video/webm": "video",
    "audio/mpeg": "audio",
    "audio/mp4": "audio",
    "audio/wav": "audio",
    "audio/x-wav": "audio",
    "application/pdf": "document",
    "text/plain": "document",
    "text/markdown": "document",
}

app = FastAPI(title="AlphaRavis Media Gallery", openapi_version="3.1.0")


def _cors_allow_origins() -> list[str]:
    raw = os.getenv("ALPHARAVIS_MEDIA_CORS_ALLOW_ORIGINS", DEFAULT_CORS_ALLOW_ORIGINS).strip()
    if raw == "*":
        return ["*"]
    origins = [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]
    return origins or ["http://localhost:3000", "http://127.0.0.1:3000"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins(),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/gallery")


@app.get("/favicon.svg", include_in_schema=False)
async def favicon():
    from fastapi.responses import Response
    return Response(content=FAVICON_SVG, media_type="image/svg+xml")

ensure_write_allowed(MEDIA_ROOT, allowed_root=MEDIA_ROOT)
MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=str(MEDIA_ROOT)), name="media")
OFFICE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/office-output", StaticFiles(directory=str(OFFICE_OUTPUT_ROOT)), name="office-output")


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


def _media_type_from_upload(filename: str, mime_type: str) -> str:
    lowered_mime = (mime_type or "").split(";", 1)[0].strip().lower()
    if lowered_mime in UPLOAD_MIME_TYPES:
        return UPLOAD_MIME_TYPES[lowered_mime]
    suffix = Path(filename or "").suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".heif"}:
        return "image"
    if suffix in {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}:
        return "video"
    if suffix in {".mp3", ".m4a", ".wav", ".ogg", ".flac"}:
        return "audio"
    if suffix:
        return "document"
    return "unknown"


def _extension_from_upload(filename: str, media_type: str) -> str:
    suffix = Path(filename or "").suffix.lower()
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


def _parse_content_disposition(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for part in value.split(";"):
        key, separator, raw = part.strip().partition("=")
        if not separator:
            continue
        cleaned = raw.strip().strip('"').replace('\\"', '"')
        result[key.lower()] = cleaned
    return result


def _parse_gallery_upload_multipart(content_type: str, body: bytes) -> tuple[dict[str, str], dict[str, Any]]:
    if "multipart/form-data" not in content_type.lower():
        raise HTTPException(status_code=400, detail="multipart/form-data is required")
    match = re.search(r"boundary=(?P<boundary>[^;]+)", content_type, flags=re.IGNORECASE)
    if not match:
        raise HTTPException(status_code=400, detail="multipart boundary is missing")
    boundary = match.group("boundary").strip().strip('"').encode("utf-8")
    fields: dict[str, str] = {}
    uploaded: dict[str, Any] = {}
    for raw_part in body.split(b"--" + boundary):
        part = raw_part.strip(b"\r\n")
        if not part or part == b"--" or b"\r\n\r\n" not in part:
            continue
        raw_headers, payload = part.split(b"\r\n\r\n", 1)
        if payload.endswith(b"\r\n"):
            payload = payload[:-2]
        headers: dict[str, str] = {}
        for line in raw_headers.decode("utf-8", "replace").split("\r\n"):
            key, separator, value = line.partition(":")
            if separator:
                headers[key.strip().lower()] = value.strip()
        disposition = _parse_content_disposition(headers.get("content-disposition", ""))
        name = disposition.get("name", "")
        if not name:
            continue
        filename = disposition.get("filename", "")
        if filename:
            uploaded = {
                "field": name,
                "filename": Path(filename).name,
                "content_type": headers.get("content-type", "application/octet-stream"),
                "content": payload,
            }
        else:
            fields[name] = payload.decode("utf-8", "replace").strip()
    if not uploaded.get("content"):
        raise HTTPException(status_code=400, detail="file upload is required")
    return fields, uploaded


def _store_uploaded_asset(*, filename: str, content_type: str, content: bytes, title: str = "") -> dict[str, Any]:
    if not content:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    if len(content) > MAX_DOWNLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"upload exceeds limit {MAX_DOWNLOAD_BYTES} bytes")
    media_type = _media_type_from_upload(filename, content_type)
    digest = hashlib.sha256(content).hexdigest()
    asset_id = hashlib.sha256(f"gallery-upload|{digest}|{filename}".encode("utf-8")).hexdigest()[:24]
    created_at = int(time.time())
    day = time.strftime("%Y-%m-%d", time.localtime(created_at))
    safe_title = _safe_segment(title or filename or media_type, "upload")
    extension = _extension_from_upload(filename, media_type)
    target = MEDIA_ROOT / day / "gallery-upload" / "input" / f"{asset_id}-{safe_title}{extension}"
    ensure_write_allowed(target, allowed_root=MEDIA_ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    with tmp.open("wb") as fh:
        fh.write(content)
    os.replace(tmp, target)
    relative_path = str(target.relative_to(MEDIA_ROOT))
    public_url = _public_url(relative_path)
    record = {
        "_id": asset_id,
        "asset_id": asset_id,
        "source_url": public_url,
        "file_id": "",
        "source_key": f"gallery-upload-{asset_id}",
        "thread_id": "",
        "thread_key": "",
        "group_id": "gallery-upload",
        "asset_kind": "original",
        "origin": "gallery_upload",
        "parent_asset_id": "",
        "root_asset_id": asset_id,
        "derivation_group_id": "gallery-upload",
        "source_message_id": "",
        "result_message_id": "",
        "tool_call_id": "",
        "processing_provider": "",
        "processing_prompt": "",
        "role": "input",
        "media_type": media_type,
        "mime_type": content_type,
        "title": title or filename or asset_id,
        "prompt": "",
        "caption": "",
        "metadata": {"original_filename": filename, "bytes": len(content), "sha256": digest},
        "relative_path": relative_path,
        "local_path": str(target),
        "public_url": public_url,
        "download_url": public_url,
        "thumbnail_path": "",
        "preview_path": "",
        "download_error": "",
        "created_at": created_at,
    }
    _collection().replace_one({"_id": asset_id}, record, upsert=True)
    return record


def _public_url(relative_path: str) -> str:
    return f"{PUBLIC_BASE_URL}/media/{relative_path.replace(os.sep, '/')}"


def _office_output_url(relative_path: str) -> str:
    return f"{OFFICE_OUTPUT_PUBLIC_BASE_URL}/{relative_path.replace(os.sep, '/')}"


def _office_output_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    relative_path = str(resolved.relative_to(OFFICE_OUTPUT_ROOT)).replace(os.sep, "/")
    stat = resolved.stat()
    return {
        "filename": resolved.name,
        "relative_path": relative_path,
        "extension": resolved.suffix.lower(),
        "size": stat.st_size,
        "modified_at": int(stat.st_mtime),
        "public_url": _office_output_url(relative_path),
        "download_url": _office_output_url(relative_path),
    }


def _store_office_uploaded_file(*, filename: str, content: bytes) -> dict[str, Any]:
    if not content:
        raise HTTPException(status_code=400, detail="uploaded office file is empty")
    if len(content) > MAX_DOWNLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"upload exceeds limit {MAX_DOWNLOAD_BYTES} bytes")
    suffix = Path(filename or "").suffix.lower()
    if suffix not in OFFICE_UPLOAD_EXTENSIONS:
        supported = ", ".join(sorted(OFFICE_UPLOAD_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"unsupported office file type: expected {supported}")
    digest = hashlib.sha256(content).hexdigest()[:8]
    stem = _safe_segment(Path(filename).stem or "office-upload", "office-upload")
    OFFICE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    target = OFFICE_OUTPUT_ROOT / f"{stem}{suffix}"
    if target.exists():
        target = OFFICE_OUTPUT_ROOT / f"{stem}-{digest}{suffix}"
    ensure_write_allowed(target, allowed_root=OFFICE_OUTPUT_ROOT)
    tmp = target.with_name(f".{target.name}.tmp")
    with tmp.open("wb") as fh:
        fh.write(content)
    os.replace(tmp, target)
    return _office_output_record(target)


def _list_office_files_under(root: Path, limit: int = 200, extensions: set[str] | None = None) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    safe_limit = max(1, min(int(limit or 200), 1000))
    allowed_extensions = extensions or OFFICE_OUTPUT_EXTENSIONS
    files = [
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed_extensions
    ]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [_office_output_record(path) for path in files[:safe_limit]]


def _list_office_output_files(limit: int = 200) -> list[dict[str, Any]]:
    return _list_office_files_under(OFFICE_OUTPUT_ROOT, limit, OFFICE_OUTPUT_EXTENSIONS)


def _list_office_template_files(limit: int = 200) -> list[dict[str, Any]]:
    return _list_office_files_under(OFFICE_OUTPUT_ROOT / "templates", limit, OFFICE_UPLOAD_EXTENSIONS)


@app.get("/office/files")
async def list_office_output_files(limit: int = 200):
    return {"root": str(OFFICE_OUTPUT_ROOT), "files": _list_office_output_files(limit)}


@app.get("/office/templates")
async def list_office_templates(limit: int = 200):
    template_root = (OFFICE_OUTPUT_ROOT / "templates").resolve()
    return {"root": str(template_root), "files": _list_office_template_files(limit)}


@app.post("/office/upload")
async def upload_office_file(request: Request):
    fields, uploaded = _parse_gallery_upload_multipart(
        request.headers.get("content-type", ""),
        await request.body(),
    )
    record = _store_office_uploaded_file(
        filename=str(uploaded.get("filename") or fields.get("filename") or "office-upload"),
        content=uploaded.get("content") or b"",
    )
    return {"file": record}


def _sort_spec(sort: str, order: str) -> tuple[str, int]:
    field = sort if sort in ASSET_SORT_FIELDS else "created_at"
    direction = 1 if (order or "").lower() == "asc" else -1
    return field, direction


def _asset_query(
    *,
    media_type: str = "all",
    thread_id: str = "",
    thread_key: str = "",
    group_id: str = "",
    asset_kind: str = "all",
) -> dict[str, Any]:
    query: dict[str, Any] = {}
    if media_type and media_type != "all":
        query["media_type"] = media_type
    if thread_id:
        query["thread_id"] = thread_id
    if thread_key:
        query["thread_key"] = thread_key
    if group_id:
        query["$or"] = [{"group_id": group_id}, {"derivation_group_id": group_id}]
    if asset_kind and asset_kind != "all":
        query["asset_kind"] = asset_kind
    return query


def _gallery_group_key(row: dict[str, Any], group_by: str) -> str:
    day, _ = _gallery_display_date(row.get("created_at"))
    thread = str(row.get("thread_key") or row.get("thread_id") or "no-thread")
    group = str(row.get("derivation_group_id") or row.get("group_id") or "ungrouped")
    media_type = str(row.get("media_type") or "unknown")
    if group_by == "thread":
        return thread
    if group_by == "group":
        return group
    if group_by == "media_type":
        return media_type
    if group_by == "none":
        return "Alle Medien"
    return day


def _gallery_display_date(timestamp: Any) -> tuple[str, str]:
    try:
        created_at = int(timestamp or 0)
    except (TypeError, ValueError):
        created_at = 0
    if created_at <= 0:
        return "Ohne Datum", ""
    return (
        time.strftime("%d.%m.%Y", time.localtime(created_at)),
        time.strftime("%H:%M", time.localtime(created_at)),
    )


def _gallery_media_label(media_type: Any) -> str:
    return {
        "image": "Bild",
        "video": "Video",
        "audio": "Audio",
        "document": "Dokument",
    }.get(str(media_type or "").lower(), "Media")


def _gallery_kind_label(asset_kind: Any, role: Any) -> str:
    value = str(asset_kind or role or "unknown").lower()
    return {
        "original": "Original",
        "processed": "Bearbeitet",
        "reference": "Referenz",
        "input": "Original",
        "output": "Bearbeitet",
    }.get(value, "Media")


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


@app.post("/assets/upload", include_in_schema=False)
async def upload_asset(request: Request):
    content_length = int(request.headers.get("content-length") or "0")
    if content_length and content_length > MAX_DOWNLOAD_BYTES + 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"upload exceeds limit {MAX_DOWNLOAD_BYTES} bytes")
    fields, uploaded = _parse_gallery_upload_multipart(request.headers.get("content-type", ""), await request.body())
    _store_uploaded_asset(
        filename=str(uploaded.get("filename") or "upload.bin"),
        content_type=str(uploaded.get("content_type") or "application/octet-stream"),
        content=uploaded.get("content") or b"",
        title=fields.get("title", ""),
    )
    return RedirectResponse(url="/gallery?view=all&group_by=day&sort=created_at&order=desc", status_code=303)


@app.get("/assets")
async def list_assets(
    limit: int = 200,
    media_type: str = "all",
    thread_id: str = "",
    thread_key: str = "",
    group_id: str = "",
    asset_kind: str = "all",
    sort: str = "created_at",
    order: str = "desc",
):
    query = _asset_query(
        media_type=media_type,
        thread_id=thread_id,
        thread_key=thread_key,
        group_id=group_id,
        asset_kind=asset_kind,
    )
    sort_field, sort_direction = _sort_spec(sort, order)
    rows = list(_collection().find(query).sort(sort_field, sort_direction).limit(max(1, min(limit, 1000))))
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
async def gallery(
    limit: int = 300,
    view: str = "all",
    media_type: str = "all",
    thread_id: str = "",
    thread_key: str = "",
    group_id: str = "",
    group_by: str = "day",
    sort: str = "created_at",
    order: str = "desc",
):
    view = view if view in {"all", "original", "processed"} else "all"
    group_by = group_by if group_by in GALLERY_GROUP_BY else "day"
    query = _asset_query(
        media_type=media_type,
        thread_id=thread_id,
        thread_key=thread_key,
        group_id=group_id,
        asset_kind=view,
    )
    sort_field, sort_direction = _sort_spec(sort, order)
    rows = list(_collection().find(query).sort(sort_field, sort_direction).limit(max(1, min(limit, 1000))))
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = _gallery_group_key(row, group_by)
        groups.setdefault(key, []).append(row)
    common_params = {
        "media_type": media_type,
        "limit": str(limit),
        "thread_id": thread_id,
        "thread_key": thread_key,
        "group_id": group_id,
        "group_by": group_by,
        "sort": sort_field,
        "order": "asc" if sort_direction == 1 else "desc",
    }

    body = [
        "<!doctype html><html lang='de'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>AlphaRavis Media Gallery</title><link rel='icon' href='/favicon.svg' type='image/svg+xml'>",
        "<style>"
        ":root{color-scheme:dark;--bg:#0a0d0f;--surface:#11171d;--surface2:#161d24;--line:#26313b;--text:#f3f7f2;--muted:#9aa8a6;--soft:#d2ddd8;--accent:#7dd3c7;--accent2:#f4b95f;--danger:#ef8b8b}"
        "*{box-sizing:border-box}body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;background:radial-gradient(circle at 18% 0%,rgba(125,211,199,.16),transparent 30%),linear-gradient(160deg,#0a0d0f 0%,#12171b 48%,#0d1113 100%);color:var(--text);min-height:100vh}"
        "header{position:sticky;top:0;background:rgba(10,13,15,.88);backdrop-filter:blur(18px);border-bottom:1px solid rgba(255,255,255,.08);z-index:2}"
        ".shell{width:min(1480px,calc(100vw - 40px));margin:0 auto}.topbar{display:flex;align-items:center;justify-content:space-between;gap:14px;padding:18px 0 12px}.brand{display:flex;align-items:center;gap:12px;min-width:0}.mark{display:grid;place-items:center;width:42px;height:42px;border-radius:8px;background:linear-gradient(135deg,var(--accent),#d7f7ee);color:#08100f;font-weight:900;letter-spacing:0}.eyebrow{color:var(--accent);font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.08em}.titleline{display:grid;gap:2px;min-width:0}h1{font-size:28px;line-height:1;margin:0;letter-spacing:0}.count-pill{border:1px solid rgba(255,255,255,.1);border-radius:8px;padding:8px 10px;background:rgba(255,255,255,.04);color:var(--soft);font-size:13px;white-space:nowrap}"
        ".controls{display:grid;grid-template-columns:auto auto 1fr;gap:12px;align-items:start;padding:0 0 16px}.tabs{display:flex;gap:6px;flex-wrap:wrap}.tabs a{color:var(--soft);text-decoration:none;border:1px solid rgba(255,255,255,.1);border-radius:8px;padding:9px 11px;background:rgba(255,255,255,.035);font-size:13px;min-height:38px}.tabs a.active{background:var(--text);color:#11171d;border-color:var(--text);font-weight:800}"
        ".upload-form label{display:flex;align-items:center;justify-content:center;min-height:38px;border-radius:8px;border:1px solid rgba(125,211,199,.4);background:rgba(125,211,199,.1);color:#d7f7ee;padding:9px 12px;font-size:13px;font-weight:850;cursor:pointer;white-space:nowrap}.upload-form input{position:absolute;inline-size:1px;block-size:1px;opacity:0;pointer-events:none}.filters{display:grid;grid-template-columns:repeat(6,minmax(0,1fr)) auto;gap:8px}.filters input,.filters select{width:100%;background:var(--surface);color:var(--text);border:1px solid rgba(255,255,255,.1);border-radius:8px;padding:9px 10px;min-height:38px;font:inherit;font-size:13px}.filters button{background:var(--accent);color:#08100f;border:1px solid var(--accent);border-radius:8px;padding:9px 14px;cursor:pointer;font-weight:900;min-height:38px;white-space:nowrap}"
        "main{padding:22px 0 50px}.day{margin:0 0 28px}.day summary{cursor:pointer;list-style:none;display:flex;align-items:center;justify-content:space-between;gap:10px;padding:0 0 12px;border-bottom:1px solid rgba(255,255,255,.08)}.day summary::-webkit-details-marker{display:none}.day-title{display:flex;align-items:baseline;gap:10px;min-width:0}.day-title strong{font-size:22px;letter-spacing:0}.day-title span,.section-meta{color:var(--muted);font-size:13px;white-space:nowrap}.section-meta:after{content:'Hide';color:var(--soft)}.day:not([open]) .section-meta:after{content:'Show'}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:14px;margin-top:14px}.card{position:relative;background:rgba(17,23,29,.84);border:1px solid rgba(255,255,255,.09);border-radius:8px;overflow:hidden;min-height:250px;display:flex;flex-direction:column;box-shadow:0 18px 38px rgba(0,0,0,.22)}.thumb{display:block;background:#050708;overflow:hidden}img,video{width:100%;aspect-ratio:4/5;object-fit:contain;display:block;background:#050708}.file-link{display:grid;place-items:center;aspect-ratio:4/5;padding:20px;color:var(--soft);text-decoration:none;background:linear-gradient(160deg,#1b242b,#10161b)}.file-link strong{font-size:18px}.asset-info{display:grid;gap:9px;padding:11px}.asset-top{display:flex;align-items:center;justify-content:space-between;gap:8px}.asset-title{margin:0;font-size:14px;line-height:1.2;letter-spacing:0;color:var(--soft);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.badge{display:inline-flex;width:max-content;font-size:11px;border:1px solid rgba(125,211,199,.35);border-radius:7px;padding:4px 7px;color:#d7f7ee;background:rgba(125,211,199,.1);white-space:nowrap}.meta{display:flex;align-items:center;justify-content:space-between;gap:10px;color:var(--muted);font-size:12px}.actions{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:1px}.actions button,.actions a{font-size:12px;background:#202a31;color:var(--text);border:1px solid rgba(255,255,255,.11);border-radius:7px;padding:8px 9px;text-decoration:none;cursor:pointer;text-align:center;font-weight:750}.empty{color:var(--muted);border:1px dashed rgba(255,255,255,.14);border-radius:8px;padding:22px;background:rgba(255,255,255,.03)}"
        "@media(max-width:1100px){.controls{grid-template-columns:1fr}.filters{grid-template-columns:repeat(2,minmax(0,1fr))}.filters button{grid-column:span 2}.upload-form label{width:100%}}@media(max-width:980px){header{position:static}.grid{grid-template-columns:repeat(auto-fill,minmax(165px,1fr));gap:10px}.shell{width:calc(100% - 28px)}}@media(max-width:540px){.shell{width:calc(100% - 20px)}.topbar{align-items:flex-start}.count-pill{display:none}h1{font-size:24px}.mark{width:38px;height:38px}.tabs{display:grid;grid-template-columns:repeat(3,1fr)}.tabs a{text-align:center}.filters{grid-template-columns:1fr 1fr}.filters input[name='thread_key'],.filters input[name='group_id']{grid-column:span 2}.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.card{min-height:0}img,video,.file-link{aspect-ratio:1}.asset-info{padding:9px}.asset-title{font-size:13px}.actions{grid-template-columns:1fr}.day-title strong{font-size:19px}}"
        "</style>",
        "<script>function copyLink(url){navigator.clipboard&&navigator.clipboard.writeText(url)}</script>",
        f"</head><body><header><div class='shell'><div class='topbar'><div class='brand'><div class='mark'>MG</div><div class='titleline'><div class='eyebrow'>AlphaRavis Media</div><h1>Media Gallery</h1></div></div><div class='count-pill'>{len(rows)} Assets</div></div><div class='controls'><nav class='tabs'>",
        _tab_link("All", "all", view, common_params),
        _tab_link("Original", "original", view, common_params),
        _tab_link("Processed", "processed", view, common_params),
        "</nav>",
        _upload_form(),
        _filter_form(view, common_params),
        "</div></div></header><main><div class='shell'>",
    ]
    for group, assets in groups.items():
        body.append(
            "<details class='day' open>"
            f"<summary><span class='day-title'><strong>{html.escape(group)}</strong>"
            f"<span>{len(assets)} Assets</span></span><span class='section-meta'></span></summary><div class='grid'>"
        )
        for asset in assets:
            url = html.escape(str(asset.get("public_url") or asset.get("source_url") or ""))
            media_type = asset.get("media_type")
            media_label = html.escape(_gallery_media_label(media_type))
            kind_label = html.escape(_gallery_kind_label(asset.get("asset_kind"), asset.get("role")))
            day_label, time_label = _gallery_display_date(asset.get("created_at"))
            body.append("<div class='card'>")
            if url and media_type == "image":
                body.append(f"<a class='thumb' href='{url}'><img src='{url}' loading='lazy' alt=''></a>")
            elif url and media_type == "video":
                body.append(f"<div class='thumb'><video src='{url}' controls preload='metadata'></video></div>")
            elif url:
                body.append(f"<a class='file-link' href='{url}'><strong>{media_label}</strong></a>")
            else:
                body.append(f"<div class='file-link'><strong>{media_label}</strong></div>")
            body.append("<div class='asset-info'>")
            body.append(f"<div class='asset-top'><h3 class='asset-title'>{media_label}</h3><span class='badge'>{kind_label}</span></div>")
            body.append(f"<div class='meta'><span>{html.escape(day_label)}</span><span>{html.escape(time_label)}</span></div>")
            if url:
                body.append(
                    "<div class='actions'>"
                    f"<button data-url='{url}' onclick='copyLink(this.dataset.url)'>Copy</button>"
                    f"<a href='{url}' target='_blank' rel='noreferrer'>Open</a>"
                    "</div>"
                )
            body.append("</div>")
            body.append("</div>")
        body.append("</div></details>")
    if not groups:
        body.append("<p class='empty'>No media matched the selected filters.</p>")
    body.append("</div></main></body></html>")
    return "\n".join(body)


def _tab_link(label: str, value: str, current: str, params: dict[str, str]) -> str:
    active = " active" if value == current else ""
    query = dict(params)
    query["view"] = value
    href = f"/gallery?{urlencode({key: value for key, value in query.items() if value})}"
    return f"<a class='{active}' href='{href}'>{html.escape(label)}</a>"


def _select(name: str, current: str, values: list[tuple[str, str]]) -> str:
    options = []
    for value, label in values:
        selected = " selected" if value == current else ""
        options.append(f"<option value='{html.escape(value)}'{selected}>{html.escape(label)}</option>")
    return f"<select name='{html.escape(name)}'>{''.join(options)}</select>"


def _upload_form() -> str:
    return (
        "<form class='upload-form' method='post' action='/assets/upload' enctype='multipart/form-data'>"
        "<label><span>Upload</span>"
        "<input type='file' name='file' accept='image/*,video/*,audio/*,.pdf,.txt,.md,.json,.csv' onchange='this.form.submit()'>"
        "</label></form>"
    )


def _filter_form(view: str, params: dict[str, str]) -> str:
    media_type = params.get("media_type", "all")
    group_by = params.get("group_by", "day")
    sort = params.get("sort", "created_at")
    order = params.get("order", "desc")
    return (
        "<form class='filters' method='get' action='/gallery'>"
        f"<input type='hidden' name='view' value='{html.escape(view)}'>"
        f"{_select('media_type', media_type, [('all', 'All media'), ('image', 'Images'), ('video', 'Videos'), ('audio', 'Audio'), ('document', 'Documents')])}"
        f"{_select('group_by', group_by, [('day', 'Date sections'), ('none', 'No sections'), ('media_type', 'Type sections'), ('thread', 'Thread sections'), ('group', 'Group sections')])}"
        f"{_select('sort', sort, [('created_at', 'Date'), ('title', 'Name'), ('media_type', 'Type'), ('asset_kind', 'Kind'), ('thread_key', 'Thread'), ('group_id', 'Group')])}"
        f"{_select('order', order, [('desc', 'Descending'), ('asc', 'Ascending')])}"
        f"<input name='thread_key' placeholder='Thread key' value='{html.escape(params.get('thread_key', ''))}'>"
        f"<input name='group_id' placeholder='Group id' value='{html.escape(params.get('group_id', ''))}'>"
        f"<input name='limit' inputmode='numeric' value='{html.escape(params.get('limit', '300'))}'>"
        "<button type='submit'>Apply</button>"
        "</form>"
    )
