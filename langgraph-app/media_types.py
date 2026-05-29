"""Shared media-type constants and helpers — single source of truth."""

import os
from pathlib import Path

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".avif"}
)
VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}
)
AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}
)


def media_type_from_suffix(path_or_url: str) -> str:
    """Return 'image', 'video', 'audio', or 'unknown' from a filename/URL suffix."""
    suffix = Path(str(path_or_url or "")).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    return "unknown"


def extension_from_url(media_url: str) -> str:
    """Extract the file extension (with dot) from a URL path."""
    url_path = (media_url or "").split("?")[0].split("#")[0]
    return Path(url_path).suffix.lower()
