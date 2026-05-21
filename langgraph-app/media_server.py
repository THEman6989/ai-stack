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
ASSET_SORT_FIELDS = {"created_at", "title", "media_type", "asset_kind", "thread_key", "group_id"}
GALLERY_GROUP_BY = {"day_group", "thread", "group", "day", "media_type"}
FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="14" fill="#0b1018"/><path d="M12 44V20h40v24H12Z" fill="#ff8a58"/><path d="M18 38l8-9 6 6 5-7 9 10H18Z" fill="#101620"/><circle cx="42" cy="25" r="4" fill="#101620"/></svg>"""

app = FastAPI(title="AlphaRavis Media Gallery", openapi_version="3.1.0")

@app.get("/", include_in_schema=False)
async def root_redirect():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/gallery")


@app.get("/favicon.svg", include_in_schema=False)
async def favicon():
    from fastapi.responses import Response
    return Response(content=FAVICON_SVG, media_type="image/svg+xml")

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
    day = time.strftime("%Y-%m-%d", time.localtime(int(row.get("created_at") or 0)))
    thread = str(row.get("thread_key") or row.get("thread_id") or "no-thread")
    group = str(row.get("derivation_group_id") or row.get("group_id") or "ungrouped")
    media_type = str(row.get("media_type") or "unknown")
    if group_by == "thread":
        return thread
    if group_by == "group":
        return group
    if group_by == "day":
        return day
    if group_by == "media_type":
        return media_type
    return f"{day} / {group}"


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
    group_by: str = "day_group",
    sort: str = "created_at",
    order: str = "desc",
):
    view = view if view in {"all", "original", "processed"} else "all"
    group_by = group_by if group_by in GALLERY_GROUP_BY else "day_group"
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
        ":root{color-scheme:dark;--bg:#080b10;--panel:#101620;--panel2:#151d29;--line:#253144;--text:#eef4ff;--muted:#94a3b8;--soft:#c7d2e4;--accent:#ff8a58}"
        "*{box-sizing:border-box}body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;background:linear-gradient(145deg,#080b10 0%,#0d1118 52%,#10151d 100%);color:var(--text);min-height:100vh}"
        "header{position:sticky;top:0;background:rgba(8,11,16,.94);backdrop-filter:blur(12px);border-bottom:1px solid var(--line);z-index:2}"
        ".shell{width:min(1440px,calc(100vw - 40px));margin:0 auto}.hero{display:grid;gap:12px;padding:22px 0 16px}.brand{display:flex;align-items:center;gap:12px}.mark{display:grid;place-items:center;width:44px;height:44px;border-radius:8px;background:var(--accent);color:#101620;font-weight:900;letter-spacing:0}.eyebrow{color:var(--accent);font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:.08em}.titleline{display:grid;gap:2px}h1{font-size:clamp(26px,5vw,54px);line-height:1;margin:0;letter-spacing:0}.subhead{max-width:860px;margin:0;color:var(--soft);font-size:15px;line-height:1.4}"
        ".tabs{display:flex;gap:8px;flex-wrap:wrap}.tabs a{color:var(--soft);text-decoration:none;border:1px solid var(--line);border-radius:8px;padding:8px 11px;background:rgba(255,255,255,.03);font-size:13px}.tabs a.active{background:var(--accent);color:#101620;border-color:var(--accent);font-weight:800}"
        ".filters{display:grid;grid-template-columns:repeat(auto-fit,minmax(135px,1fr));gap:8px;padding:0 0 18px;max-width:1080px}.filters input,.filters select{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:8px;padding:9px;min-height:40px;font:inherit;font-size:14px}.filters button{background:#e8eef8;color:#101620;border:1px solid #e8eef8;border-radius:8px;padding:9px 12px;cursor:pointer;font-weight:800}"
        "main{padding:20px 0 44px}details{margin:0 0 18px;border-top:1px solid var(--line);padding-top:14px}summary{cursor:pointer;font-weight:800;font-size:17px;list-style:none;display:flex;align-items:center;justify-content:space-between;gap:10px}summary::-webkit-details-marker{display:none}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:12px;margin-top:12px}"
        ".card{position:relative;background:linear-gradient(180deg,rgba(21,29,41,.96),rgba(12,17,25,.98));border:1px solid var(--line);border-radius:8px;padding:12px;min-height:240px;display:flex;flex-direction:column;gap:10px;overflow:hidden}.thumb{display:block;border-radius:7px;background:#05070a;border:1px solid rgba(255,255,255,.06);overflow:hidden}img,video{width:100%;aspect-ratio:16/9;object-fit:contain;display:block;background:#000}.file-link{display:flex;align-items:center;min-height:120px;padding:12px;color:var(--soft);overflow-wrap:anywhere;text-decoration:none}.meta{font-size:12px;color:var(--muted);word-break:break-word;display:grid;gap:3px;padding-bottom:36px}.badge{display:inline-flex;width:max-content;font-size:11px;text-transform:uppercase;letter-spacing:.04em;border:1px solid rgba(255,255,255,.14);border-radius:6px;padding:3px 7px;color:var(--soft);background:rgba(255,255,255,.04)}.asset-head{display:flex;align-items:start;justify-content:space-between;gap:8px}h3{font-size:15px;margin:0;line-height:1.25;letter-spacing:0}.actions{position:absolute;right:12px;bottom:12px;display:flex;gap:6px}.actions button,.actions a{font-size:12px;background:#202838;color:var(--text);border:1px solid #344055;border-radius:7px;padding:6px 8px;text-decoration:none;cursor:pointer}.empty{color:var(--muted);border:1px dashed var(--line);border-radius:8px;padding:22px;background:rgba(255,255,255,.03)}"
        "@media(max-width:760px){.shell{width:calc(100% - 28px)}header{position:static}.hero{padding-top:18px}.grid{grid-template-columns:1fr}.filters{grid-template-columns:1fr 1fr}.subhead{font-size:14px}.card{min-height:auto}}@media(max-width:460px){.shell{width:calc(100% - 20px)}.filters{grid-template-columns:1fr}.brand{align-items:flex-start}.mark{width:38px;height:38px}}"
        "</style>",
        "<script>function copyLink(url){navigator.clipboard&&navigator.clipboard.writeText(url)}</script>",
        "</head><body><header><div class='shell'><div class='hero'><div class='brand'><div class='mark'>MG</div><div class='titleline'><div class='eyebrow'>AlphaRavis Media</div><h1>Media Gallery</h1></div></div><p class='subhead'>Gespeicherte Uploads, Pixelle-Ergebnisse, Referenzen und Analyse-Assets mit stabilen Media-URLs fuer Chat, RAG und Operator-Checks.</p></div><nav class='tabs'>",
        _tab_link("All", "all", view, common_params),
        _tab_link("Original", "original", view, common_params),
        _tab_link("Processed", "processed", view, common_params),
        "</nav>",
        _filter_form(view, common_params),
        "</div></header><main><div class='shell'>",
    ]
    for group, assets in groups.items():
        body.append(f"<details open><summary><span>{html.escape(group)} ({len(assets)})</span><span>{html.escape(str(group_by))}</span></summary><div class='grid'>")
        for asset in assets:
            title = html.escape(str(asset.get("title") or asset.get("asset_id")))
            url = html.escape(str(asset.get("public_url") or asset.get("source_url") or ""))
            media_type = asset.get("media_type")
            asset_kind = html.escape(str(asset.get("asset_kind") or asset.get("role") or "unknown"))
            metadata = asset.get("metadata") if isinstance(asset.get("metadata"), dict) else {}
            provider = html.escape(str(asset.get("processing_provider") or metadata.get("provider", "")))
            body.append("<div class='card'>")
            body.append(f"<div class='asset-head'><h3>{title}</h3><span class='badge'>{asset_kind}</span></div>")
            if url and media_type == "image":
                body.append(f"<a class='thumb' href='{url}'><img src='{url}' loading='lazy' alt=''></a>")
            elif url and media_type == "video":
                body.append(f"<div class='thumb'><video src='{url}' controls preload='metadata'></video></div>")
            elif url:
                body.append(f"<a class='file-link' href='{url}'>{url}</a>")
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


def _filter_form(view: str, params: dict[str, str]) -> str:
    media_type = params.get("media_type", "all")
    group_by = params.get("group_by", "day_group")
    sort = params.get("sort", "created_at")
    order = params.get("order", "desc")
    return (
        "<form class='filters' method='get' action='/gallery'>"
        f"<input type='hidden' name='view' value='{html.escape(view)}'>"
        f"{_select('media_type', media_type, [('all', 'All media'), ('image', 'Images'), ('video', 'Videos'), ('audio', 'Audio'), ('document', 'Documents')])}"
        f"{_select('group_by', group_by, [('day_group', 'Day + group'), ('thread', 'Thread'), ('group', 'Group'), ('day', 'Date'), ('media_type', 'Type')])}"
        f"{_select('sort', sort, [('created_at', 'Date'), ('title', 'Name'), ('media_type', 'Type'), ('asset_kind', 'Kind'), ('thread_key', 'Thread'), ('group_id', 'Group')])}"
        f"{_select('order', order, [('desc', 'Descending'), ('asc', 'Ascending')])}"
        f"<input name='thread_key' placeholder='Thread key' value='{html.escape(params.get('thread_key', ''))}'>"
        f"<input name='group_id' placeholder='Group id' value='{html.escape(params.get('group_id', ''))}'>"
        f"<input name='limit' inputmode='numeric' value='{html.escape(params.get('limit', '300'))}'>"
        "<button type='submit'>Apply</button>"
        "</form>"
    )
