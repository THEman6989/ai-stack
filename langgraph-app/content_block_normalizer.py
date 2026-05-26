"""Normalize file-type ContentBlocks into readable text for the model.

ContentBlocks like {type:"file", mimeType:"video/mp4", data:"", metadata:{url:"http://..."}}
are opaque to the model. This module extracts the URL + filename and rewrites them as
plain text so the model immediately knows what was uploaded and which tools to use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Extensions reused from agent_graph for media type detection
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".avif"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}


def _media_type_from_suffix(value: str) -> str:
    cleaned = (value or "").split("?", 1)[0].split("#", 1)[0].lower()
    suffix = Path(cleaned).suffix
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    if suffix in {".pdf", ".doc", ".docx", ".txt", ".md", ".csv", ".json"}:
        return "document"
    return "other"


def _format_bytes(size: int | float) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    if size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    return f"{size / (1024 * 1024 * 1024):.1f} GB"


def normalize_file_content_blocks(messages: list[Any]) -> list[Any]:
    """Replace file-type ContentBlocks that carry a Media Gallery URL with a readable text notice."""
    updated: list[Any] = []
    for message in messages:
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            updated.append(message)
            continue

        role = getattr(message, "type", "")
        if role not in ("human",):
            updated.append(message)
            continue

        new_content: list[Any] = []
        changed = False
        for block in content:
            if not isinstance(block, dict):
                new_content.append(block)
                continue
            block_type = str(block.get("type") or "")
            if block_type != "file":
                new_content.append(block)
                continue
            metadata = block.get("metadata") or {}
            if not isinstance(metadata, dict):
                new_content.append(block)
                continue
            url = str(metadata.get("url") or "")
            if not url:
                new_content.append(block)
                continue

            # We have a file block with a Gallery URL — replace with text notice
            mime_type = str(block.get("mimeType") or "application/octet-stream")
            filename = str(metadata.get("filename") or "uploaded-file")
            size = metadata.get("size")
            media_type = str(metadata.get("media_type") or _media_type_from_suffix(url))
            size_str = f", {_format_bytes(size)}" if isinstance(size, (int, float)) else ""

            emoji = {"video": "🎥", "audio": "🎵", "image": "🖼️"}.get(media_type, "📄")
            label = {"video": "Video", "audio": "Audio", "image": "Bild"}.get(media_type, "Datei")

            text_notice = (
                f"[{emoji} {label} uploaded: {filename}{size_str}]\n"
                f"URL: {url}\n"
                f"MIME: {mime_type}\n"
                f"Use register_media_asset to index it, or vision_analyze/plan_media_analysis to inspect it."
            )
            new_content.append({"type": "text", "text": text_notice})
            changed = True

        if changed:
            if hasattr(message, "model_copy"):
                updated.append(message.model_copy(update={"content": new_content}))
            elif isinstance(message, dict):
                updated.append({**message, "content": new_content})
            else:
                updated.append(message)
        else:
            updated.append(message)
    return updated
