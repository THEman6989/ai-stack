"""Shared environment helpers — single source of truth for env_bool."""
from __future__ import annotations

import os

# Canonical truthy values: union of all existing definitions across the codebase.
# Includes "y" from retrieval_router.py for backward compatibility.
_ENV_TRUTHY = frozenset({"1", "true", "yes", "y", "on"})


def env_bool(name: str, default: str = "false") -> bool:
    """Read an environment variable as a boolean.

    Truthy: 1, true, yes, y, on (case-insensitive).
    Uses str() wrapper for safety against non-string os.environ values.
    """
    return str(os.getenv(name, default)).strip().lower() in _ENV_TRUTHY
