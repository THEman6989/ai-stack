from __future__ import annotations

import re
from typing import Any


_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "because",
    "before",
    "could",
    "from",
    "have",
    "into",
    "just",
    "make",
    "more",
    "need",
    "that",
    "the",
    "their",
    "then",
    "there",
    "this",
    "with",
    "would",
}

_INSIGHT_PATTERNS = (
    (re.compile(r"\bI prefer\b(.{8,180})", re.IGNORECASE), "user_preference", 0.72),
    (re.compile(r"\bI usually\b(.{8,180})", re.IGNORECASE), "user_habit", 0.68),
    (re.compile(r"\bI always\b(.{8,180})", re.IGNORECASE), "user_preference", 0.72),
    (re.compile(r"\bI never\b(.{8,180})", re.IGNORECASE), "user_preference", 0.72),
    (re.compile(r"\bremember that\b(.{8,180})", re.IGNORECASE), "explicit_memory_request", 0.82),
    (re.compile(r"\bdefault(?:s)? to\b(.{8,180})", re.IGNORECASE), "runtime_default", 0.66),
    (re.compile(r"\bkeep\b(.{8,180})\bby default\b", re.IGNORECASE), "runtime_default", 0.66),
)


def generate_thread_title(text: str, *, max_words: int = 8) -> str:
    cleaned = _normalize_space(text)
    if not cleaned:
        return "Untitled Thread"

    first_line = next((line.strip() for line in cleaned.splitlines() if line.strip()), cleaned)
    first_line = re.sub(r"^(user|assistant|system)\s*:\s*", "", first_line, flags=re.IGNORECASE)
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9._/-]*", first_line)
    selected = [word.strip("._/-") for word in words if word.lower() not in _STOPWORDS and word.strip("._/-")]
    if not selected:
        selected = [word.strip("._/-") for word in words if word.strip("._/-")]
    if not selected:
        return "Untitled Thread"

    max_words = max(3, min(int(max_words), 12))
    title = " ".join(selected[:max_words]).strip()
    return title[:80].rstrip(" .,:;") or "Untitled Thread"


def extract_review_insight_candidates(text: str, *, max_candidates: int = 8) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sentence in _sentences(text):
        for pattern, kind, confidence in _INSIGHT_PATTERNS:
            match = pattern.search(sentence)
            if not match:
                continue
            candidate = _normalize_space(match.group(0)).strip(" .")
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "kind": kind,
                    "candidate": candidate[:240],
                    "confidence": confidence,
                    "review_required": True,
                    "source_preview": sentence[:300],
                }
            )
            if len(candidates) >= max(1, min(int(max_candidates), 20)):
                return candidates
    return candidates


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _sentences(text: str) -> list[str]:
    normalized = str(text or "").replace("\r\n", "\n")
    pieces = re.split(r"(?<=[.!?])\s+|\n+", normalized)
    return [_normalize_space(piece) for piece in pieces if _normalize_space(piece)]
