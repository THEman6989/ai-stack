from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx


from media_types import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS


from env_utils import env_bool as _env_bool


from slug_utils import safe_segment as _safe_segment


def _default_model_cards() -> dict[str, dict[str, Any]]:
    return {
        "Qwen/Qwen3.6-35B-A3B": {
            "supports_images": True,
            "supports_video": True,
            "native_context_tokens": 262144,
            "image_longest_edge": 16777216,
            "image_shortest_edge": 65536,
            "video_longest_edge": 25165824,
            "video_shortest_edge": 4096,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "merge_size": 2,
            "preferred_video_fps": 1,
            "max_video_fps": 1,
            "max_frames": 100,
            "provider_payload": {"mm_processor_kwargs": {"fps": 1, "do_sample_frames": True}},
        },
        "big-boss": {
            "alias_of": "Qwen/Qwen3.6-35B-A3B",
        },
        "openai/big-boss": {
            "alias_of": "Qwen/Qwen3.6-35B-A3B",
        },
    }


def load_model_cards(path: str = "") -> dict[str, dict[str, Any]]:
    cards = _default_model_cards()
    configured = path or os.getenv("ALPHARAVIS_VIDEO_ANALYSIS_MODEL_CARD_PATH", "").strip()
    if configured:
        card_path = Path(configured).expanduser()
        if card_path.exists():
            with card_path.open("r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                for key, value in loaded.items():
                    if isinstance(value, dict):
                        cards[str(key)] = value
    return cards


def resolve_model_card(model_id: str = "") -> dict[str, Any]:
    cards = load_model_cards()
    model = (
        model_id
        or os.getenv("ALPHARAVIS_VIDEO_ANALYSIS_MODEL_ID")
        or os.getenv("ALPHARAVIS_VISION_EMBEDDING_MODEL")
        or os.getenv("ALPHARAVIS_RESPONSES_MODEL")
        or os.getenv("ALPHARAVIS_MODEL")
        or "big-boss"
    ).strip()
    card = dict(cards.get(model) or cards.get(model.removeprefix("openai/")) or {})
    seen = {model}
    while card.get("alias_of") and str(card["alias_of"]) not in seen:
        alias = str(card["alias_of"])
        seen.add(alias)
        parent = dict(cards.get(alias) or {})
        parent.update({key: value for key, value in card.items() if key != "alias_of"})
        card = parent
    card.setdefault("model_id", model)
    card.setdefault("supports_images", False)
    card.setdefault("supports_video", False)
    card.setdefault("preferred_video_fps", float(os.getenv("ALPHARAVIS_VIDEO_ANALYSIS_FPS", "1")))
    card.setdefault("max_video_fps", float(os.getenv("ALPHARAVIS_VIDEO_ANALYSIS_MAX_FPS", "1")))
    card.setdefault("max_frames", int(os.getenv("ALPHARAVIS_VIDEO_ANALYSIS_MAX_FRAMES", "100")))
    return card


def decide_media_mode(user_goal: str = "", requested_mode: str = "auto") -> str:
    mode = (requested_mode or "auto").strip().lower()
    aliases = {
        "passthrough": "pass_through",
        "pass-through": "pass_through",
        "register": "register_only",
        "metadata": "register_only",
        "analyse": "analyze",
        "analysis": "analyze",
        "embed": "index",
        "embedding": "index",
    }
    mode = aliases.get(mode, mode)
    if mode in {"pass_through", "register_only", "analyze", "index"}:
        return mode
    text = (user_goal or "").lower()
    analyze_terms = [
        "analys",
        "describe",
        "beschreib",
        "summar",
        "zusammenfass",
        "transkrib",
        "transcrib",
        "inspect",
        "untersuch",
        "erkenne",
        "was ist in",
        "was passiert",
    ]
    index_terms = ["index", "embedding", "einbetten", "durchsuchbar", "suchbar machen"]
    pass_terms = ["pixelle", "comfy", "mach daraus", "neues video", "use as input", "als input", "weitergeben"]
    if any(term in text for term in index_terms):
        return "index"
    if any(term in text for term in analyze_terms):
        return "analyze"
    if any(term in text for term in pass_terms):
        return "pass_through"
    return "register_only"


def _media_type_from_url(media_url: str, fallback: str = "unknown") -> str:
    suffix = Path(urlparse(media_url).path).suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    return fallback if fallback in {"image", "video", "audio", "document", "unknown"} else "unknown"


def _cache_roots() -> tuple[Path, Path]:
    media_root = Path(os.getenv("ALPHARAVIS_VIDEO_ANALYSIS_PUBLIC_MEDIA_ROOT", "/workspace/media-data")).expanduser()
    cache_root = Path(os.getenv("ALPHARAVIS_VIDEO_ANALYSIS_CACHE_ROOT", str(media_root / "analysis-cache"))).expanduser()
    return media_root.resolve(), cache_root.resolve()


def _public_media_url(path: Path) -> str:
    media_root, _ = _cache_roots()
    base = os.getenv("ALPHARAVIS_MEDIA_PUBLIC_BASE_URL", "http://localhost:8130").rstrip("/")
    try:
        relative = path.resolve().relative_to(media_root)
    except ValueError:
        return ""
    return f"{base}/media/{relative.as_posix()}"


def _extension_from_url(media_url: str, media_type: str) -> str:
    suffix = Path(urlparse(media_url).path).suffix.lower()
    if suffix and len(suffix) <= 12:
        return suffix
    return ".mp4" if media_type == "video" else ".bin"


def _ensure_under_root(path: Path, root: Path) -> None:
    path.resolve().relative_to(root.resolve())


async def _download_media(media_url: str, target: Path, max_bytes: int, *, honor_limit: bool = False) -> dict[str, Any]:
    parsed = urlparse(media_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http/https media URLs are downloadable by default.")
    target.parent.mkdir(parents=True, exist_ok=True)
    _, cache_root = _cache_roots()
    _ensure_under_root(target, cache_root)
    tmp = target.with_name(f".{target.name}.tmp")
    size = 0
    timeout = float(os.getenv("ALPHARAVIS_VIDEO_ANALYSIS_DOWNLOAD_TIMEOUT_SECONDS", "180"))
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        async with client.stream("GET", media_url) as response:
            if response.status_code >= 400:
                raise RuntimeError(f"download returned HTTP {response.status_code}")
            content_type = response.headers.get("content-type", "")
            with tmp.open("wb") as fh:
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > max_bytes:
                        if honor_limit:
                            raise RuntimeError(f"download exceeds limit {max_bytes} bytes")
                        logging.warning(
                            "Download exceeds recommended limit of %d bytes (actual: %d). Proceeding anyway (ALPHARAVIS_VIDEO_ANALYSIS_HONOR_SIZE_LIMIT is false).",
                            max_bytes, size,
                        )
                    fh.write(chunk)
    tmp.replace(target)
    return {"bytes": size, "content_type": content_type, "local_path": str(target)}


def _run_json(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    return json.loads(completed.stdout or "{}")


def _probe_video(path: Path) -> dict[str, Any]:
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe is not installed in this container.")
    data = _run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    video_stream = next((stream for stream in data.get("streams", []) if stream.get("codec_type") == "video"), {})
    duration = float(data.get("format", {}).get("duration") or video_stream.get("duration") or 0)
    return {
        "duration_seconds": duration,
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "codec": video_stream.get("codec_name", ""),
        "format_name": data.get("format", {}).get("format_name", ""),
    }


def _max_frame_side(card: dict[str, Any]) -> int:
    override = os.getenv("ALPHARAVIS_VIDEO_ANALYSIS_MAX_FRAME_SIDE", "").strip()
    if override:
        return max(64, int(override))
    longest_edge = int(card.get("video_longest_edge") or card.get("image_longest_edge") or 1048576)
    return max(64, int(math.sqrt(max(1, longest_edge))))


def _sampling_plan(duration_seconds: float, card: dict[str, Any]) -> dict[str, Any]:
    max_frames = max(1, int(os.getenv("ALPHARAVIS_VIDEO_ANALYSIS_MAX_FRAMES", str(card.get("max_frames") or 100))))
    max_fps = max(0.01, float(os.getenv("ALPHARAVIS_VIDEO_ANALYSIS_MAX_FPS", str(card.get("max_video_fps") or 1))))
    preferred_fps = max(0.01, float(os.getenv("ALPHARAVIS_VIDEO_ANALYSIS_FPS", str(card.get("preferred_video_fps") or 1))))
    fps = min(preferred_fps, max_fps)
    if duration_seconds > 0:
        fps = min(fps, max_frames / duration_seconds)
    fps = max(0.001, fps)
    estimated_frames = min(max_frames, max(1, int(math.ceil((duration_seconds or max_frames) * fps))))
    return {"fps": fps, "max_frames": max_frames, "estimated_frames": estimated_frames}


def _extract_video_frames(path: Path, output_dir: Path, fps: float, max_frames: int, max_side: int) -> list[Path]:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is not installed in this container.")
    output_dir.mkdir(parents=True, exist_ok=True)
    _, cache_root = _cache_roots()
    _ensure_under_root(output_dir, cache_root)
    pattern = output_dir / "frame-%05d.jpg"
    vf = (
        f"fps={fps:.6f},"
        f"scale=trunc(iw*min(1\\,{max_side}/iw\\,{max_side}/ih)/2)*2:"
        f"trunc(ih*min(1\\,{max_side}/iw\\,{max_side}/ih)/2)*2"
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vf",
        vf,
        "-frames:v",
        str(max_frames),
        "-q:v",
        "3",
        str(pattern),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=600)
    return sorted(output_dir.glob("frame-*.jpg"))


async def prepare_media_for_model(
    *,
    media_url: str,
    user_goal: str = "",
    mode: str = "auto",
    media_type: str = "unknown",
    source_key: str = "",
    title: str = "",
    model_id: str = "",
    thread_id: str = "",
) -> dict[str, Any]:
    media_url = (media_url or "").strip()
    if not media_url:
        return {"ok": False, "error": "media_url is required"}
    resolved_type = _media_type_from_url(media_url, media_type)
    resolved_mode = decide_media_mode(user_goal=user_goal, requested_mode=mode)
    source_key = source_key or hashlib.sha256(media_url.encode("utf-8")).hexdigest()[:16]
    result: dict[str, Any] = {
        "ok": True,
        "mode": resolved_mode,
        "media_url": media_url,
        "media_type": resolved_type,
        "source_key": source_key,
        "title": title or source_key,
        "thread_id": thread_id,
        "downloaded": False,
        "frames": [],
        "frame_count": 0,
        "indexing_requested": resolved_mode == "index",
    }
    if resolved_mode in {"pass_through", "register_only"}:
        result["decision"] = "metadata_only"
        return result
    if not _env_bool("ALPHARAVIS_VIDEO_ANALYSIS_ENABLED", "true"):
        result.update(
            {
                "ok": False,
                "disabled": True,
                "decision": "analysis_not_run",
                "error": "ALPHARAVIS_VIDEO_ANALYSIS_ENABLED=false",
            }
        )
        return result
    if resolved_type != "video":
        result.update({"ok": False, "error": f"media_type `{resolved_type}` is not supported by video analysis yet"})
        return result

    card = resolve_model_card(model_id)
    if not card.get("supports_video"):
        result.update({"ok": False, "error": f"model `{model_id or card.get('model_id')}` has no video support in model cards"})
        return result

    media_root, cache_root = _cache_roots()
    cache_root.mkdir(parents=True, exist_ok=True)
    run_id = f"{int(time.time())}-{_safe_segment(source_key)}-{hashlib.sha256(media_url.encode('utf-8')).hexdigest()[:8]}"
    run_dir = cache_root / _safe_segment(thread_id or "global", "global") / run_id
    local_media = run_dir / f"source{_extension_from_url(media_url, resolved_type)}"
    max_bytes = int(os.getenv("ALPHARAVIS_VIDEO_ANALYSIS_MAX_DOWNLOAD_BYTES", "2147483648"))
    honor_limit = _env_bool("ALPHARAVIS_VIDEO_ANALYSIS_HONOR_SIZE_LIMIT", "false")
    download_info = await _download_media(media_url, local_media, max_bytes=max_bytes, honor_limit=honor_limit)
    result["downloaded"] = True
    result["download"] = download_info

    probe = await asyncio.to_thread(_probe_video, local_media)
    plan = _sampling_plan(float(probe.get("duration_seconds") or 0), card)
    max_side = _max_frame_side(card)
    frame_paths = await asyncio.to_thread(
        _extract_video_frames,
        local_media,
        run_dir / "frames",
        float(plan["fps"]),
        int(plan["max_frames"]),
        max_side,
    )
    duration = float(probe.get("duration_seconds") or 0)
    frames = []
    for index, frame_path in enumerate(frame_paths):
        timestamp = index / float(plan["fps"]) if plan["fps"] else 0.0
        if duration:
            timestamp = min(timestamp, duration)
        frames.append(
            {
                "frame_index": index,
                "time_seconds": round(timestamp, 3),
                "timecode": _format_timecode(timestamp),
                "local_path": str(frame_path),
                "public_url": _public_media_url(frame_path),
            }
        )

    manifest = {
        "media_url": media_url,
        "source_key": source_key,
        "title": title or source_key,
        "thread_id": thread_id,
        "mode": resolved_mode,
        "model_card": card,
        "probe": probe,
        "sampling": plan,
        "max_frame_side": max_side,
        "frames": frames,
        "created_at": int(time.time()),
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    result.update(
        {
            "decision": "prepared_for_model",
            "model_card": card,
            "probe": probe,
            "sampling": plan,
            "max_frame_side": max_side,
            "frame_count": len(frames),
            "frames": frames,
            "manifest_path": str(manifest_path),
            "manifest_url": _public_media_url(manifest_path),
            "media_root": str(media_root),
        }
    )
    return result


def _format_timecode(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    whole = int(seconds)
    millis = int(round((seconds - whole) * 1000))
    hours = whole // 3600
    minutes = (whole % 3600) // 60
    secs = whole % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
