from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
import shutil
import socket
import subprocess
import time
import urllib.request
import wave
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any
from urllib.parse import urlparse


@dataclass(slots=True)
class VideoProbe:
    duration_seconds: float
    fps: float
    width: int = 0
    height: int = 0
    codec: str = ""
    has_audio: bool = False


@dataclass(slots=True)
class DropCandidate:
    time_seconds: float
    frame_index: int
    confidence: float
    reason: str


@dataclass(slots=True)
class FrameChangeScore:
    frame_index: int
    time_seconds: float
    local_path: str
    diff_score: float
    is_black: bool = False


def _safe_segment(value: str, fallback: str = "video") -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in (value or "").strip())
    return safe.strip(".-_")[:80] or fallback


def _env_bool(name: str, default: str = "false") -> bool:
    return (os.getenv(name, default) or "").strip().lower() in {"1", "true", "yes", "on"}


def _configured_media_roots() -> list[Path]:
    roots = [
        os.getenv("ALPHARAVIS_VIDEO_DROP_PLANNER_MEDIA_ROOT", ""),
        os.getenv("ALPHARAVIS_VIDEO_ANALYSIS_PUBLIC_MEDIA_ROOT", ""),
        os.getenv("ALPHARAVIS_MEDIA_ROOT", ""),
    ]
    return [Path(root).expanduser().resolve() for root in roots if root.strip()]


def _ensure_allowed_local_video_path(path: Path) -> Path:
    if not _env_bool("ALPHARAVIS_VIDEO_DROP_PLANNER_ALLOW_LOCAL_PATHS", "false"):
        raise ValueError("Local video paths are disabled for video drop planning. Use an approved media URL or set ALPHARAVIS_VIDEO_DROP_PLANNER_ALLOW_LOCAL_PATHS=true.")
    resolved = path.expanduser().resolve()
    roots = _configured_media_roots()
    if roots:
        for root in roots:
            try:
                resolved.relative_to(root)
                return resolved
            except ValueError:
                continue
        raise ValueError("Local video path must be under an approved media root for video drop planning.")
    return resolved


def _hostname_is_private(hostname: str) -> bool:
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"Cannot resolve media URL host `{hostname}`: {exc}") from exc
    addresses = {info[4][0] for info in infos if info and info[4]}
    if not addresses:
        raise ValueError(f"Cannot resolve media URL host `{hostname}`")
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
            return True
    return False


def _validate_remote_media_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http/https media URLs are downloadable by default.")
    if parsed.username or parsed.password:
        raise ValueError("Media URLs with embedded credentials are not allowed.")
    hostname = parsed.hostname or ""
    if not hostname:
        raise ValueError("Media URL must include a hostname.")
    if _hostname_is_private(hostname) and not _env_bool("ALPHARAVIS_VIDEO_DROP_PLANNER_ALLOW_PRIVATE_URLS", "false"):
        raise ValueError("Private, loopback, link-local, or reserved media URL hosts are disabled for video drop planning.")


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        raise ValueError(f"Redirects are disabled for video drop planning downloads (HTTP {code}).")


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirectHandler)


def _parse_ratio(value: str | None, default: float = 0.0) -> float:
    raw = (value or "").strip()
    if not raw or raw == "0/0":
        return default
    if "/" in raw:
        num, denom = raw.split("/", 1)
        try:
            denominator = float(denom)
            return float(num) / denominator if denominator else default
        except Exception:
            return default
    try:
        return float(raw)
    except Exception:
        return default


def _run_json(command: list[str], *, timeout: int = 60) -> dict[str, Any]:
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)
    return json.loads(completed.stdout or "{}")


def probe_video(path: str | Path) -> VideoProbe:
    """Probe duration/fps/audio with ffprobe; no frame extraction or model calls."""

    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe is required for video drop planning.")
    video_path = Path(path).expanduser()
    data = _run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]
    )
    streams = data.get("streams") or []
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
    duration = float(data.get("format", {}).get("duration") or video_stream.get("duration") or 0.0)
    fps = _parse_ratio(video_stream.get("avg_frame_rate"), 0.0) or _parse_ratio(video_stream.get("r_frame_rate"), 0.0) or 30.0
    return VideoProbe(
        duration_seconds=duration,
        fps=fps,
        width=int(video_stream.get("width") or 0),
        height=int(video_stream.get("height") or 0),
        codec=str(video_stream.get("codec_name") or ""),
        has_audio=has_audio,
    )


def extract_audio_to_wav(video_path: str | Path, out_dir: str | Path, *, sample_rate: int = 22050) -> Path:
    """Extract mono PCM WAV for deterministic beat/drop heuristics."""

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for audio extraction.")
    target_dir = Path(out_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    wav_path = target_dir / "audio-mono.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(Path(video_path).expanduser()),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "wav",
            str(wav_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return wav_path


def _pcm16_samples(wav_path: str | Path) -> tuple[list[float], int]:
    with wave.open(str(wav_path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if sample_width != 2:
        raise RuntimeError(f"Only 16-bit PCM WAV is supported, got sample width {sample_width}.")
    samples: list[float] = []
    step = sample_width * channels
    for index in range(0, len(frames), step):
        value = int.from_bytes(frames[index : index + sample_width], byteorder="little", signed=True)
        samples.append(value / 32768.0)
    return samples, sample_rate


def _rms_envelope(samples: list[float], sample_rate: int, *, window_ms: int = 80, hop_ms: int = 20) -> list[tuple[float, float]]:
    if not samples:
        return []
    window = max(1, int(sample_rate * window_ms / 1000))
    hop = max(1, int(sample_rate * hop_ms / 1000))
    envelope: list[tuple[float, float]] = []
    for start in range(0, max(1, len(samples) - window + 1), hop):
        chunk = samples[start : start + window]
        if not chunk:
            continue
        rms = math.sqrt(sum(value * value for value in chunk) / len(chunk))
        center = (start + len(chunk) / 2) / sample_rate
        envelope.append((center, rms))
    return envelope


def _local_median(values: list[float], start: int, end: int, fallback: float = 1e-9) -> float:
    subset = [value for value in values[max(0, start) : max(start + 1, end)] if value > 0]
    if not subset:
        return fallback
    return max(fallback, float(median(subset)))


def detect_audio_drop_candidates(
    wav_path: str | Path,
    *,
    fps: float = 30.0,
    max_candidates: int = 8,
    merge_seconds: float = 0.5,
    min_confidence: float = 0.35,
) -> list[DropCandidate]:
    """Find likely beat/drop onsets from RMS jumps using only stdlib WAV parsing.

    This is intentionally a first-pass heuristic. It narrows the expensive visual/LLM
    frame search window; it is not the final semantic judge.
    """

    samples, sample_rate = _pcm16_samples(wav_path)
    envelope = _rms_envelope(samples, sample_rate)
    if len(envelope) < 4:
        return []
    times = [item[0] for item in envelope]
    rms_values = [item[1] for item in envelope]
    max_rms = max(rms_values) or 1e-9
    raw_candidates: list[DropCandidate] = []
    for index in range(2, len(envelope)):
        current = rms_values[index]
        previous = _local_median(rms_values, index - 12, index - 1)
        after = _local_median(rms_values, index, min(len(rms_values), index + 6))
        jump = max(current, after) / previous
        normalized_energy = current / max_rms
        confidence = min(1.0, (jump - 1.0) / 3.0 * 0.7 + normalized_energy * 0.3)
        if jump >= 1.65 and normalized_energy >= 0.12 and confidence >= min_confidence:
            time_seconds = times[index]
            raw_candidates.append(
                DropCandidate(
                    time_seconds=round(time_seconds, 3),
                    frame_index=max(0, int(round(time_seconds * fps))),
                    confidence=round(confidence, 3),
                    reason=f"rms jump {jump:.2f}x with normalized energy {normalized_energy:.2f}",
                )
            )
    return merge_drop_candidates(raw_candidates, merge_seconds=merge_seconds, max_candidates=max_candidates)


def merge_drop_candidates(candidates: Iterable[DropCandidate], *, merge_seconds: float = 0.5, max_candidates: int = 8) -> list[DropCandidate]:
    sorted_candidates = sorted(candidates, key=lambda item: (item.time_seconds, -item.confidence))
    merged: list[DropCandidate] = []
    for candidate in sorted_candidates:
        if merged and abs(candidate.time_seconds - merged[-1].time_seconds) <= merge_seconds:
            if candidate.confidence > merged[-1].confidence:
                merged[-1] = candidate
            continue
        merged.append(candidate)
    return sorted(merged, key=lambda item: item.confidence, reverse=True)[:max_candidates]


def extract_candidate_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    start_seconds: float,
    end_seconds: float,
    fps: float = 30.0,
    max_side: int = 512,
) -> list[Path]:
    """Extract high-FPS frames only inside a narrowed candidate window."""

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for candidate frame extraction.")
    target_dir = Path(output_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    start = max(0.0, float(start_seconds))
    duration = max(0.05, float(end_seconds) - start)
    pattern = target_dir / "frame-%06d.jpg"
    vf = (
        f"fps={max(0.1, float(fps)):.6f},"
        f"scale=trunc(iw*min(1\\,{max_side}/iw\\,{max_side}/ih)/2)*2:"
        f"trunc(ih*min(1\\,{max_side}/iw\\,{max_side}/ih)/2)*2"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(Path(video_path).expanduser()),
            "-vf",
            vf,
            "-q:v",
            "3",
            str(pattern),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return sorted(target_dir.glob("frame-*.jpg"))


def _image_signature(path: Path, *, size: int = 32) -> tuple[list[float], float]:
    try:
        from PIL import Image, ImageStat  # type: ignore

        with Image.open(path) as image:
            gray = image.convert("L").resize((size, size))
            pixels = []
            for value in gray.getdata():
                scalar = value[0] if isinstance(value, tuple) else value
                pixels.append(float(scalar) / 255.0)
            mean_value = ImageStat.Stat(gray).mean[0]
            if mean_value is None:
                mean = 0.0
            else:
                mean = float(mean_value)
            return pixels, mean
    except Exception:
        data = path.read_bytes()
        if not data:
            return [0.0], 0.0
        buckets = [0] * size
        for index, byte in enumerate(data):
            buckets[index % size] += byte
        scale = max(1, max(buckets))
        mean = sum(data) / len(data)
        return [bucket / scale for bucket in buckets], mean


def compute_frame_change_scores(
    frame_paths: Iterable[str | Path],
    *,
    window_start_seconds: float = 0.0,
    fps: float = 30.0,
    extraction_fps: float | None = None,
    source_fps: float | None = None,
) -> list[FrameChangeScore]:
    """Score visual deltas between adjacent extracted frames.

    The current MVP uses whole-frame luminance delta as the deterministic baseline.
    Later phases can replace the signature with person/outfit crops or CLIP/DINO.
    """

    paths = [Path(path) for path in frame_paths]
    signatures = [_image_signature(path) for path in paths]
    scores: list[FrameChangeScore] = []
    previous: list[float] | None = None
    extract_rate = max(0.001, float(extraction_fps if extraction_fps is not None else fps))
    source_rate = max(0.001, float(source_fps if source_fps is not None else fps))
    for index, (path, (signature, mean_luma)) in enumerate(zip(paths, signatures, strict=False)):
        if previous is None:
            diff = 0.0
        else:
            width = min(len(previous), len(signature)) or 1
            diff = sum(abs(signature[i] - previous[i]) for i in range(width)) / width
        time_seconds = window_start_seconds + index / extract_rate
        frame_index = int(round(time_seconds * source_rate))
        scores.append(
            FrameChangeScore(
                frame_index=frame_index,
                time_seconds=round(time_seconds, 3),
                local_path=str(path),
                diff_score=round(float(diff), 6),
                is_black=mean_luma <= 10.0,
            )
        )
        previous = signature
    return scores


def suggest_black_frame_count(fps: float, black_frame_ms: int = 60) -> int:
    return max(1, min(3, int(round(max(1.0, fps) * max(1, black_frame_ms) / 1000))))


def detect_visual_transition(
    scores: list[FrameChangeScore],
    *,
    beat_frame: int,
    fps: float,
    black_frame_ms: int = 60,
) -> dict[str, Any]:
    if not scores:
        black_count = suggest_black_frame_count(fps, black_frame_ms)
        return {
            "visual_change_frame": beat_frame,
            "last_old_outfit_frame": max(0, beat_frame - black_count - 1),
            "first_new_outfit_frame": beat_frame,
            "insert_black_frame": True,
            "black_frame_start": max(0, beat_frame - black_count),
            "black_frame_count": black_count,
            "visual_confidence": 0.0,
            "reason": "no extracted candidate frames; falling back to beat frame",
        }
    best = max(scores[1:] or scores, key=lambda item: item.diff_score)
    black_count = suggest_black_frame_count(fps, black_frame_ms)
    black_frames = [item.frame_index for item in scores if item.is_black]
    score_by_frame = {item.frame_index: item for item in scores}
    first_new = int(best.frame_index)
    if best.is_black:
        following_non_black = next((item for item in scores if item.frame_index > best.frame_index and not item.is_black), None)
        if following_non_black is not None:
            first_new = int(following_non_black.frame_index)
    last_old = max(0, first_new - black_count - 1)
    if black_frames:
        nearest_black = min(black_frames, key=lambda frame: abs(frame - first_new))
        if abs(nearest_black - first_new) <= max(2, black_count + 1):
            black_run = sorted(frame for frame in black_frames if abs(frame - nearest_black) <= max(2, black_count + 1))
            black_start = min(black_run) if black_run else nearest_black
            last_old = max(0, black_start - 1)
            if score_by_frame.get(first_new, None) and score_by_frame[first_new].is_black:
                following_non_black = next((item for item in scores if item.frame_index > first_new and not item.is_black), None)
                if following_non_black is not None:
                    first_new = int(following_non_black.frame_index)
    return {
        "visual_change_frame": first_new,
        "last_old_outfit_frame": last_old,
        "first_new_outfit_frame": first_new,
        "insert_black_frame": True,
        "black_frame_start": max(0, first_new - black_count),
        "black_frame_count": black_count,
        "visual_confidence": round(min(1.0, best.diff_score * 2.0), 3),
        "reason": f"max adjacent frame diff at frame {first_new} ({best.diff_score:.3f})",
    }


def build_drop_plan(
    *,
    source_video: str,
    probe: VideoProbe,
    outfit_images: list[dict[str, Any]] | list[str] | None,
    workflow_name: str,
    audio_candidates: list[DropCandidate],
    visual_transitions: dict[int, dict[str, Any]] | None = None,
    window_seconds: float = 1.0,
    black_frame_ms: int = 60,
) -> dict[str, Any]:
    visual_transitions = visual_transitions or {}
    normalized_outfits: list[dict[str, Any]] = []
    for index, image in enumerate(outfit_images or []):
        if isinstance(image, dict):
            normalized_outfits.append({**image, "id": str(image.get("id") or f"outfit_{index + 1}")})
        else:
            normalized_outfits.append({"id": f"outfit_{index + 1}", "url": str(image)})
    drops: list[dict[str, Any]] = []
    for index, candidate in enumerate(sorted(audio_candidates, key=lambda item: item.time_seconds), start=1):
        transition = visual_transitions.get(candidate.frame_index) or detect_visual_transition([], beat_frame=candidate.frame_index, fps=probe.fps, black_frame_ms=black_frame_ms)
        selected_outfit = normalized_outfits[(index - 1) % len(normalized_outfits)]["id"] if normalized_outfits else ""
        start_frame = max(0, int(round((candidate.time_seconds - window_seconds) * probe.fps)))
        end_frame = int(round((candidate.time_seconds + window_seconds) * probe.fps))
        drops.append(
            {
                "drop_id": f"drop_{index:03d}",
                "drop_type": "audio_visual" if transition.get("visual_confidence", 0) else "audio_candidate",
                "beat_time_seconds": candidate.time_seconds,
                "beat_frame": candidate.frame_index,
                "window_start_frame": start_frame,
                "window_end_frame": end_frame,
                "window_start_seconds": round(max(0.0, candidate.time_seconds - window_seconds), 3),
                "window_end_seconds": round(min(probe.duration_seconds or candidate.time_seconds + window_seconds, candidate.time_seconds + window_seconds), 3),
                "selected_outfit_image": selected_outfit,
                "confidence": candidate.confidence,
                "audio_reason": candidate.reason,
                **transition,
            }
        )
    return {
        "ok": True,
        "plan_type": "video_outfit_drop_plan",
        "version": 1,
        "source_video": source_video,
        "workflow_name": workflow_name,
        "fps": round(probe.fps, 3),
        "duration_seconds": round(probe.duration_seconds, 3),
        "width": probe.width,
        "height": probe.height,
        "has_audio": probe.has_audio,
        "outfit_images": normalized_outfits,
        "settings": {
            "window_seconds": window_seconds,
            "black_frame_ms": black_frame_ms,
            "black_frame_count_default": suggest_black_frame_count(probe.fps, black_frame_ms),
        },
        "drops": drops,
        "drop_count": len(drops),
        "created_at": int(time.time()),
    }


def _planner_cache_root() -> Path:
    root = os.getenv("ALPHARAVIS_VIDEO_DROP_PLANNER_CACHE_ROOT") or os.getenv("ALPHARAVIS_VIDEO_ANALYSIS_CACHE_ROOT") or "/workspace/media-data/drop-plans"
    return Path(root).expanduser()


def resolve_video_input(media_url: str, run_dir: str | Path, *, max_bytes: int = 2_147_483_648) -> Path:
    raw = (media_url or "").strip()
    if not raw:
        raise ValueError("media_url is required")
    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https"}:
        _validate_remote_media_url(raw)
        suffix = Path(parsed.path).suffix or ".mp4"
        target = Path(run_dir) / f"source{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.tmp")
        request = urllib.request.Request(raw, headers={"User-Agent": "AlphaRavisVideoDropPlanner/1.0"})
        try:
            with _NO_REDIRECT_OPENER.open(request, timeout=180) as response, tmp.open("wb") as fh:  # noqa: S310 - URL is validated above and redirects are disabled
                final_url = response.geturl()
                if final_url != raw:
                    _validate_remote_media_url(final_url)
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > max_bytes:
                    raise RuntimeError(f"download exceeds limit {max_bytes} bytes")
                size = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        raise RuntimeError(f"download exceeds limit {max_bytes} bytes")
                    fh.write(chunk)
            tmp.replace(target)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        return target
    if parsed.scheme == "file":
        return _ensure_allowed_local_video_path(Path(parsed.path))
    return _ensure_allowed_local_video_path(Path(raw))


def plan_video_outfit_drops_from_file(
    video_path: str | Path,
    *,
    output_dir: str | Path,
    outfit_images: list[dict[str, Any]] | list[str] | None = None,
    workflow_name: str = "outfit_change_beatdrop",
    max_drops: int = 8,
    black_frame_ms: int = 60,
    window_seconds: float = 1.0,
    candidate_fps: float = 30.0,
    extract_frames: bool = True,
) -> dict[str, Any]:
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    source = Path(video_path).expanduser()
    probe = probe_video(source)
    audio_candidates: list[DropCandidate] = []
    if probe.has_audio:
        wav_path = extract_audio_to_wav(source, out_dir)
        audio_candidates = detect_audio_drop_candidates(wav_path, fps=probe.fps, max_candidates=max_drops)
    visual_transitions: dict[int, dict[str, Any]] = {}
    frame_windows: list[dict[str, Any]] = []
    if extract_frames:
        for candidate in audio_candidates:
            start = max(0.0, candidate.time_seconds - window_seconds)
            end = min(probe.duration_seconds or candidate.time_seconds + window_seconds, candidate.time_seconds + window_seconds)
            frame_dir = out_dir / "frames" / f"drop-{candidate.frame_index}"
            frame_paths = extract_candidate_frames(source, frame_dir, start_seconds=start, end_seconds=end, fps=candidate_fps)
            scores = compute_frame_change_scores(
                frame_paths,
                window_start_seconds=start,
                fps=candidate_fps,
                extraction_fps=candidate_fps,
                source_fps=probe.fps,
            )
            visual_transitions[candidate.frame_index] = detect_visual_transition(scores, beat_frame=candidate.frame_index, fps=probe.fps, black_frame_ms=black_frame_ms)
            frame_windows.append(
                {
                    "beat_frame": candidate.frame_index,
                    "start_seconds": round(start, 3),
                    "end_seconds": round(end, 3),
                    "frame_dir": str(frame_dir),
                    "frame_count": len(frame_paths),
                    "scores": [asdict(score) for score in scores[:120]],
                }
            )
    plan = build_drop_plan(
        source_video=str(source),
        probe=probe,
        outfit_images=outfit_images,
        workflow_name=workflow_name,
        audio_candidates=audio_candidates,
        visual_transitions=visual_transitions,
        window_seconds=window_seconds,
        black_frame_ms=black_frame_ms,
    )
    plan["analysis_artifacts"] = {"run_dir": str(out_dir), "frame_windows": frame_windows}
    plan_path = out_dir / "drop-plan.json"
    plan["plan_path"] = str(plan_path)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return plan


def plan_video_outfit_drops(
    *,
    media_url: str,
    outfit_images: list[dict[str, Any]] | list[str] | None = None,
    workflow_name: str = "outfit_change_beatdrop",
    source_key: str = "",
    max_drops: int = 8,
    black_frame_ms: int = 60,
    window_seconds: float = 1.0,
    candidate_fps: float = 30.0,
    extract_frames: bool = True,
) -> dict[str, Any]:
    cache_root = _planner_cache_root()
    digest = hashlib.sha256((media_url or "").encode("utf-8")).hexdigest()[:12]
    run_name = f"{int(time.time())}-{_safe_segment(source_key or Path(urlparse(media_url).path).stem or 'video')}-{digest}"
    run_dir = cache_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    local_video = resolve_video_input(media_url, run_dir)
    return plan_video_outfit_drops_from_file(
        local_video,
        output_dir=run_dir,
        outfit_images=outfit_images,
        workflow_name=workflow_name,
        max_drops=max_drops,
        black_frame_ms=black_frame_ms,
        window_seconds=window_seconds,
        candidate_fps=candidate_fps,
        extract_frames=extract_frames,
    )
