from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Settings


MODEL_EXTENSIONS = {".gguf", ".safetensors", ".bin", ".json", ".model", ".tiktoken", ".txt"}
SIGNAL_FILENAMES = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "special_tokens_map.json",
}


def iso_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace("+00:00", "Z")


def display_name_for(path: Path, scan_root: Path) -> str:
    parts = path.parts
    for part in parts:
        if part.startswith("models--"):
            return part.removeprefix("models--").replace("--", "/")
    try:
        return str(path.relative_to(scan_root))
    except ValueError:
        return path.name


def model_id(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
    return f"{path.name}-{digest}"


def find_model_root(file_path: Path, scan_root: Path) -> Path:
    parts = file_path.parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 2 < len(parts):
            return Path(*parts[: index + 2])
    if file_path.name in SIGNAL_FILENAMES:
        return file_path.parent
    return file_path.parent


def scan_models(settings: Settings) -> list[dict[str, Any]]:
    max_files = settings.int("MODEL_SCAN_MAX_FILES", 20000)
    seen_files = 0
    grouped: dict[Path, dict[str, Any]] = {}

    for scan_root in settings.model_scan_dirs:
        if not scan_root.exists():
            continue

        for file_path in scan_root.rglob("*"):
            if seen_files >= max_files:
                break
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in MODEL_EXTENSIONS and file_path.name not in SIGNAL_FILENAMES:
                continue

            seen_files += 1
            root = find_model_root(file_path, scan_root)
            stat = file_path.stat()
            item = grouped.setdefault(
                root,
                {
                    "id": model_id(root),
                    "name": display_name_for(root, scan_root),
                    "path": str(root),
                    "files": [],
                    "size_bytes": 0,
                    "modified_at": "1970-01-01T00:00:00Z",
                },
            )
            try:
                relative = str(file_path.relative_to(root))
            except ValueError:
                relative = file_path.name
            item["files"].append(relative)
            item["size_bytes"] += stat.st_size
            modified = iso_from_timestamp(stat.st_mtime)
            if modified > item["modified_at"]:
                item["modified_at"] = modified

    models = list(grouped.values())
    for item in models:
        item["files"] = sorted(item["files"])
    return sorted(models, key=lambda model: model["modified_at"], reverse=True)


def get_model(settings: Settings, model_id_value: str) -> dict[str, Any] | None:
    for model in scan_models(settings):
        if model["id"] == model_id_value:
            return model
    return None
