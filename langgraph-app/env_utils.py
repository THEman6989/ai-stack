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


# Shared LangGraph API defaults — single source for bridge + queue_ingest
LANGGRAPH_API_URL = os.getenv("LANGGRAPH_API_URL", "http://langgraph-api:2024")
LANGGRAPH_ASSISTANT_ID = os.getenv("LANGGRAPH_ASSISTANT_ID", "alpha_ravis")


def env_int(name: str, default: int) -> int:
    """Read an environment variable as an integer, returning *default* on failure."""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    """Read an environment variable as a float, returning *default* on failure."""
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
