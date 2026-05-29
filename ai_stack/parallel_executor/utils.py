"""Shared utilities for the parallel executor package."""


def has_glob_chars(value: str) -> bool:
    """Return True when *value* contains glob wildcard characters."""
    return any(ch in value for ch in "*?[")
