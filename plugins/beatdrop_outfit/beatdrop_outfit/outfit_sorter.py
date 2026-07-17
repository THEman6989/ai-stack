from __future__ import annotations

import base64
import csv
import hashlib
import io
import ipaddress
import json
import mimetypes
import os
import re
import socket
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

try:
    from PIL import Image
except Exception:  # pragma: no cover - ComfyUI usually has PIL/Pillow
    Image = None

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _env_bool(name: str, default: str = "false") -> bool:
    return (os.getenv(name, default) or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _configured_allowed_roots() -> List[Path]:
    raw = os.getenv("ALPHARAVIS_BEATDROP_OUTFIT_ALLOWED_ROOTS", "")
    return [
        Path(value.strip()).expanduser().resolve()
        for value in raw.split(os.pathsep)
        if value.strip()
    ]


def _ensure_allowed_local_path(path: Path, *, field: str) -> Path:
    if not _env_bool("ALPHARAVIS_BEATDROP_OUTFIT_ALLOW_LOCAL_PATHS", "false"):
        raise ValueError(
            "Local outfit paths are disabled. Set "
            "ALPHARAVIS_BEATDROP_OUTFIT_ALLOW_LOCAL_PATHS=true and configure an allowed root."
        )
    roots = _configured_allowed_roots()
    if not roots:
        raise ValueError(
            "No allowed root is configured in ALPHARAVIS_BEATDROP_OUTFIT_ALLOWED_ROOTS."
        )
    resolved = path.expanduser().resolve()
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
        raise ValueError(f"{field} must be under a configured allowed root")
    return resolved

FRAMING_BUCKETS = {
    "10_upper_body_head_chest",
    "20_waist_hips_crop",
    "30_almost_full_or_fullbody",
    "40_pose_or_style_variant",
    "90_unclear_or_bad_quality",
}
CLOTHING_BUCKETS = {
    "10_covered",
    "20_lighter",
    "30_underwear",
    "40_nude_near_nude",
}

DEFAULT_PROMPT = """Classify this outfit/reference image for a video outfit matching library.
Return strict JSON only with these keys:
{
  "framing_category": one of ["10_upper_body_head_chest", "20_waist_hips_crop", "30_almost_full_or_fullbody", "40_pose_or_style_variant", "90_unclear_or_bad_quality"],
  "clothing_category": one of ["10_covered", "20_lighter", "30_underwear", "40_nude_near_nude"],
  "confidence": number 0..1,
  "notes": short string
}

Definitions:
- 10_upper_body_head_chest: mainly head/shoulders/chest/top area; no clear waist/hips. Useful for videos cropped to bust/chest.
- 20_waist_hips_crop: crop shows waist, belly, hips, pants/shorts/skirt area, but not full legs/full body. Useful for videos cropped to hips.
- 30_almost_full_or_fullbody: almost full body or full body; enough legs/body visible for full-body matching.
- 40_pose_or_style_variant: unusual pose, extreme angle, lying down, sitting, overhead, or otherwise non-standard framing that does not fit the above categories. Useful as optional alternative to standard vertical/standing poses.
- 90_unclear_or_bad_quality: genuinely unusable/unclear/corrupt, not just cropped.

Clothing:
- 10_covered: fully/mostly clothed, lots of fabric, covered outfit (e.g. full suit, long dress, jacket+jeans).
- 20_lighter: lighter/more skin visible but still clearly an outfit (crop top, shorts, bikini top with jeans, sports bra + leggings, lingerie as outfit layer).
- 30_underwear: underwear, swimwear, lingerie only, very revealing outfit where clothing coverage is minimal.
- 40_nude_near_nude: bare skin dominant, implied or actual nudity, no meaningful clothing present.

Important: Do NOT mark useful crops as bad quality. Head/chest/waist/hips crops are useful if the video has similar framing.
"""


@dataclass
class SortRecord:
    basename: str
    original_path: str
    review_path: str
    destination: str
    framing_category: str
    clothing_category: str
    confidence: float
    notes: str
    classifier: str
    copied_from_original: bool


@dataclass
class VerificationRecord:
    basename: str
    destination: str
    original_path: str
    byte_match: bool
    source_from_cache: bool
    sha256_original: str
    sha256_destination: str


def safe_component(value: str, fallback: str = "item") -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip(".-_/\\")
    while ".." in text:
        text = text.replace("..", "_")
    return text or fallback


def _open_binary_nofollow(path: Path):
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        return os.fdopen(fd, "rb")
    except Exception:
        os.close(fd)
        raise


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with _open_binary_nofollow(path) as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_images(path: Path) -> List[Path]:
    if not path.exists():
        return []
    images: List[Path] = []
    for candidate in path.iterdir():
        if candidate.suffix.lower() not in IMAGE_EXTS:
            continue
        if candidate.is_symlink():
            raise ValueError(f"symlinked source images are not allowed: {candidate}")
        if candidate.is_file():
            images.append(candidate)
    return sorted(images)


def find_review_image(original: Path, cache_dir: Optional[Path]) -> Path:
    if cache_dir:
        p = cache_dir / original.name
        if p.is_symlink():
            raise ValueError(f"symlinked cache images are not allowed: {p}")
        if p.exists():
            if not p.is_file():
                raise ValueError(f"cache image must be a regular file: {p}")
            return p
    return original


def ensure_review_image(src: Path, review_dir: Path, max_px: int = 768) -> Path:
    """Create a resized review image for model input; never changes the original."""
    review_dir.mkdir(parents=True, exist_ok=True)
    out = review_dir / f"{src.stem}.jpg"
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return out
    if Image is None:
        # Fallback: use source directly if Pillow unavailable.
        return src
    with _open_binary_nofollow(src) as source_file:
        with Image.open(source_file) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = min(1.0, float(max_px) / float(max(w, h)))
            if scale < 1.0:
                im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
            im.save(out, "JPEG", quality=88, optimize=True)
    return out


def image_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    with _open_binary_nofollow(path) as image_file:
        data = base64.b64encode(image_file.read()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _validate_vision_endpoint(endpoint: str) -> str:
    parsed = urlparse(str(endpoint or "").strip())
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Vision endpoint must use http/https.")
    if parsed.username or parsed.password:
        raise ValueError("Vision endpoint credentials are not allowed.")
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if not hostname:
        raise ValueError("Vision endpoint must include a hostname.")
    allowed_hosts = {
        value.strip().lower().rstrip(".")
        for value in os.getenv(
            "ALPHARAVIS_BEATDROP_OUTFIT_VISION_ALLOWED_HOSTS", ""
        ).split(",")
        if value.strip()
    }
    if hostname not in allowed_hosts:
        raise ValueError("Vision endpoint host is not in the explicit allowlist.")
    try:
        addresses = {
            info[4][0]
            for info in socket.getaddrinfo(hostname, parsed.port, type=socket.SOCK_STREAM)
            if info and info[4]
        }
    except (socket.gaierror, ValueError) as exc:
        raise ValueError(f"Cannot resolve vision endpoint host `{hostname}`: {exc}") from exc
    if not addresses:
        raise ValueError(f"Cannot resolve vision endpoint host `{hostname}`")
    private_endpoint = False
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if ip.is_link_local or ip.is_multicast or ip.is_unspecified:
            raise ValueError("Link-local, multicast, and unspecified vision endpoints are blocked.")
        private_endpoint = private_endpoint or ip.is_private or ip.is_loopback or ip.is_reserved
    if private_endpoint and not _env_bool(
        "ALPHARAVIS_BEATDROP_OUTFIT_ALLOW_PRIVATE_URLS", "false"
    ):
        raise ValueError(
            "Allowlisted private vision endpoints require "
            "ALPHARAVIS_BEATDROP_OUTFIT_ALLOW_PRIVATE_URLS=true."
        )
    return str(endpoint).strip().rstrip("/")


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def classify_with_openai_vision(
    image_path: Path,
    endpoint: str,
    model: str,
    prompt: str = DEFAULT_PROMPT,
    timeout: int = 90,
) -> Dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests is not installed")
    endpoint = _validate_vision_endpoint(endpoint)
    url = endpoint
    if not url.endswith("/chat/completions"):
        url = url + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 700,
    }
    r = requests.post(url, json=payload, timeout=timeout, allow_redirects=False)
    r.raise_for_status()
    data = r.json()
    msg = data.get("choices", [{}])[0].get("message", {})
    content = msg.get("content") or msg.get("reasoning_content") or ""
    if isinstance(content, list):
        content = "\n".join(str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in content)
    parsed = extract_json(str(content))
    return parsed


def classify_by_existing_path(path: Path) -> Dict[str, Any]:
    parts = set(path.parts)
    framing = None
    clothing = None
    for b in FRAMING_BUCKETS:
        if b in parts:
            framing = b
            break
    # Detailed v3/v2 compatibility
    if framing is None:
        if any(x in parts for x in ("10_head_shoulders", "20_head_to_chest")):
            framing = "10_upper_body_head_chest"
        elif any(x in parts for x in ("30_head_to_waist", "40_head_to_hips")):
            framing = "20_waist_hips_crop"
        elif any(x in parts for x in ("50_almost_fullbody", "60_fullbody")):
            framing = "30_almost_full_or_fullbody"
        elif "40_pose_or_style_variant" in parts:
            framing = "40_pose_or_style_variant"
    for b in CLOTHING_BUCKETS:
        if b in parts:
            clothing = b
            break
    # Legacy v2/v3 clothing path mapping
    if clothing is None:
        legacy_clothing = {"30_revealing": "30_underwear", "30_underwear_revealing_fullbody": "30_underwear"}
        for legacy, mapped in legacy_clothing.items():
            if legacy in parts:
                clothing = mapped
                break
    if framing and clothing:
        return {
            "framing_category": framing,
            "clothing_category": clothing,
            "confidence": 0.82,
            "notes": "classified from existing folder path",
            "classifier": "existing_folder",
        }
    return {}


def heuristic_classify(image_path: Path) -> Dict[str, Any]:
    """Safe fallback when no vision endpoint is available.

    This is intentionally conservative; it is not a semantic replacement for a VLM.
    """
    confidence = 0.35
    framing = "90_unclear_or_bad_quality"
    clothing = "20_lighter"
    notes = "fallback heuristic; review recommended"
    if Image is not None:
        try:
            with _open_binary_nofollow(image_path) as image_file:
                with Image.open(image_file) as im:
                    w, h = im.size
            aspect = h / max(1, w)
            if aspect >= 1.35:
                framing = "30_almost_full_or_fullbody"
            elif aspect >= 0.85:
                framing = "20_waist_hips_crop"
            else:
                framing = "10_upper_body_head_chest"
            confidence = 0.45
            notes = f"fallback heuristic from aspect={aspect:.2f}; VLM review recommended"
        except Exception:
            pass
    return {
        "framing_category": framing,
        "clothing_category": clothing,
        "confidence": confidence,
        "notes": notes,
        "classifier": "heuristic",
    }


def normalize_classification(raw: Dict[str, Any], fallback_name: str = "heuristic") -> Dict[str, Any]:
    framing = str(raw.get("framing_category") or raw.get("framing") or "").strip()
    clothing = str(raw.get("clothing_category") or raw.get("clothing") or "").strip()
    if framing not in FRAMING_BUCKETS:
        framing = "90_unclear_or_bad_quality"
    if clothing not in CLOTHING_BUCKETS:
        # Map legacy v2/v3 names
        clothing_map = {"30_revealing": "30_underwear", "30_underwear_revealing_fullbody": "30_underwear"}
        clothing = clothing_map.get(clothing, "20_lighter")
    try:
        confidence = float(raw.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    return {
        "framing_category": framing,
        "clothing_category": clothing,
        "confidence": max(0.0, min(1.0, confidence)),
        "notes": str(raw.get("notes") or raw.get("reason") or ""),
        "classifier": str(raw.get("classifier") or fallback_name),
    }


def load_existing_index(existing_sorted_dir: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if not existing_sorted_dir or not existing_sorted_dir.exists():
        return {}
    index: Dict[str, Dict[str, Any]] = {}
    for p in existing_sorted_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        cls = classify_by_existing_path(p)
        if cls:
            index[p.name] = cls
    return index


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _copy_without_overwrite(source: Path, destination: Path) -> None:
    nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
    source_fd = os.open(source, os.O_RDONLY | nofollow_flag)
    try:
        source_stat = os.fstat(source_fd)
        destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow_flag
        fd = os.open(destination, destination_flags, source_stat.st_mode & 0o777)
        try:
            with os.fdopen(source_fd, "rb", closefd=False) as source_file, os.fdopen(
                fd, "wb", closefd=False
            ) as destination_file:
                shutil.copyfileobj(source_file, destination_file)
                destination_file.flush()
            os.fchmod(fd, source_stat.st_mode & 0o777)
            os.utime(fd, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns))
        finally:
            os.close(fd)
    finally:
        os.close(source_fd)


def sort_outfits(
    original_dir: str,
    output_dir: str,
    cache_dir: str = "",
    existing_sorted_dir: str = "",
    vision_endpoint: str = "",
    vision_model: str = "Qwen3.6-35B-A3B-MTP",
    mode: str = "copy",
    max_px: int = 768,
    dry_run: bool = True,
    prompt: str = DEFAULT_PROMPT,
) -> Dict[str, Any]:
    original_root = _ensure_allowed_local_path(Path(original_dir), field="original_dir")
    output_root = _ensure_allowed_local_path(Path(output_dir), field="output_dir")
    cache_root = (
        _ensure_allowed_local_path(Path(cache_dir), field="cache_dir")
        if cache_dir
        else None
    )
    existing_root = (
        _ensure_allowed_local_path(Path(existing_sorted_dir), field="existing_sorted_dir")
        if existing_sorted_dir
        else None
    )
    if not original_root.exists():
        raise FileNotFoundError(f"original_dir not found: {original_root}")
    if mode not in {"copy", "symlink"}:
        raise ValueError("mode must be copy or symlink")

    originals = iter_images(original_root)
    review_dir = output_root / "_model_review_768px"
    records: List[SortRecord] = []
    verification: List[VerificationRecord] = []
    existing_index = load_existing_index(existing_root)
    used_vision = 0
    used_existing = 0
    used_heuristic = 0
    errors: List[Dict[str, str]] = []

    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        for top in ["10_upper_body_head_chest", "20_waist_hips_crop", "30_almost_full_or_fullbody", "40_pose_or_style_variant"]:
            for cl in sorted(CLOTHING_BUCKETS):
                (output_root / top / cl).mkdir(parents=True, exist_ok=True)
        (output_root / "90_unclear_or_bad_quality").mkdir(exist_ok=True)

    for original in originals:
        review_src = find_review_image(original, cache_root)
        review_image = (
            review_src
            if dry_run
            else ensure_review_image(review_src, review_dir, max_px=max_px)
        )
        raw: Dict[str, Any]
        if vision_endpoint:
            try:
                raw = classify_with_openai_vision(review_image, vision_endpoint, vision_model, prompt=prompt)
                raw["classifier"] = "vision"
                used_vision += 1
            except Exception as e:
                # Fallback to existing index, then heuristic
                if original.name in existing_index:
                    raw = existing_index[original.name]
                    raw["notes"] = f"vision failed: {e}; reused existing_folder label. " + str(raw.get("notes", ""))
                    used_existing += 1
                else:
                    raw = heuristic_classify(review_image)
                    raw["notes"] = f"vision failed: {e}; " + raw.get("notes", "")
                    errors.append({"basename": original.name, "error": str(e)})
                    used_heuristic += 1
        elif original.name in existing_index:
            raw = existing_index[original.name]
            used_existing += 1
        else:
            raw = heuristic_classify(review_image)
            used_heuristic += 1
        cls = normalize_classification(raw, fallback_name=raw.get("classifier", "heuristic"))
        if cls["framing_category"] == "90_unclear_or_bad_quality":
            dest_dir = output_root / "90_unclear_or_bad_quality"
        else:
            dest_dir = output_root / cls["framing_category"] / cls["clothing_category"]
        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / original.name
        if not dry_run:
            if mode == "copy":
                _copy_without_overwrite(original, dest)
            else:
                if dest.exists() or dest.is_symlink():
                    if dest.is_symlink() and dest.resolve() == original:
                        pass
                    else:
                        raise FileExistsError(
                            f"refusing to replace existing symlink destination: {dest}"
                        )
                else:
                    dest.symlink_to(original)
        records.append(
            SortRecord(
                basename=original.name,
                original_path=str(original),
                review_path=str(review_image),
                destination=str(dest),
                framing_category=cls["framing_category"],
                clothing_category=cls["clothing_category"],
                confidence=cls["confidence"],
                notes=cls["notes"],
                classifier=cls["classifier"],
                copied_from_original=True,
            )
        )
        if not dry_run and dest.exists() and not dest.is_symlink():
            verification.append(
                VerificationRecord(
                    basename=original.name,
                    destination=str(dest),
                    original_path=str(original),
                    byte_match=sha256_file(original) == sha256_file(dest),
                    source_from_cache=bool(cache_root and str(original).startswith(str(cache_root))),
                    sha256_original=sha256_file(original),
                    sha256_destination=sha256_file(dest),
                )
            )

    # Optional simple pair preservation from existing pair manifests can be added later;
    # the sorter intentionally keeps pair inference separate from category sorting.
    counts: Dict[str, int] = {}
    for r in records:
        key = r.framing_category if r.framing_category == "90_unclear_or_bad_quality" else f"{r.framing_category}/{r.clothing_category}"
        counts[key] = counts.get(key, 0) + 1

    manifest = [asdict(r) for r in records]
    verify_data = [asdict(v) for v in verification]
    summary = {
        "output_root": str(output_root),
        "original_dir": str(original_root),
        "cache_dir": str(cache_root) if cache_root else "",
        "existing_sorted_dir": str(existing_root) if existing_root else "",
        "mode": mode,
        "max_model_image_px": max_px,
        "dry_run": dry_run,
        "total_original_images": len(originals),
        "records": len(records),
        "counts": counts,
        "used_existing_folder_labels": used_existing,
        "used_vision": used_vision,
        "used_heuristic": used_heuristic,
        "errors": errors,
        "verification_success": bool(dry_run or all(v.byte_match and not v.source_from_cache for v in verification)),
    }

    if not dry_run:
        write_json(output_root / "manifest.json", manifest)
        write_json(output_root / "verification.json", verify_data)
        write_json(output_root / "summary.json", summary)
        with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            if manifest:
                writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
                writer.writeheader()
                writer.writerows(manifest)
    return {"summary": summary, "manifest": manifest, "verification": verify_data}
