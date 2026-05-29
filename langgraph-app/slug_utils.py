"""Shared slug/safe-segment helpers — single source of truth."""

import re


def safe_segment(value: str, default: str = "asset", max_len: int = 96) -> str:
    """Turn an arbitrary string into a filesystem-safe segment.

    Keeps [a-zA-Z0-9._-], replaces everything else with '-'.
    Truncates to *max_len* characters, falls back to *default* when empty.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", (value or "").strip().lower()).strip("-._")
    return cleaned[:max_len] or default
