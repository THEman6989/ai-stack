"""File/glob lock manager for concurrent write task safety.

Prevents race conditions when multiple parallel write tasks need
to touch the same files, even in separate git worktrees.

Design:
- Process-local lock store (no Redis/Postgres dependency).
- Each lock tracks task_id, claimed files, and expiry.
- Before spawning a write worker, the executor checks for conflicts.
- Cleaned up on task completion/failure.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass
class FileLock:
    """A claim on one or more file globs by a task."""

    lock_id: str
    task_id: str
    file_globs: list[str] = field(default_factory=list)
    acquired_at: float = 0.0
    expires_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def expired(self) -> bool:
        return self.expires_at > 0 and time.time() > self.expires_at


class FileLockManager:
    """Process-local lock manager for file/glob-level safety."""

    def __init__(self, *, default_timeout_seconds: float = 300) -> None:
        self._locks: dict[str, FileLock] = {}
        self._default_timeout = default_timeout_seconds
        self._lock = asyncio.Lock()

    async def try_acquire(
        self,
        task_id: str,
        file_globs: list[str],
        *,
        timeout_seconds: float | None = None,
    ) -> FileLock | None:
        """Try to acquire locks on all given file globs.

        Returns a FileLock if all globs are free, None if any conflict.
        """
        if not file_globs:
            return FileLock(
                lock_id=f"lock_{task_id}_empty",
                task_id=task_id,
                acquired_at=time.time(),
            )

        timeout = timeout_seconds or self._default_timeout
        async with self._lock:
            # Purge expired locks
            self._purge_expired()

            # Check conflicts
            for existing in self._locks.values():
                if existing.task_id == task_id:
                    continue
                if _globs_overlap(file_globs, existing.file_globs):
                    LOGGER.debug(
                        "file_lock: conflict — %s wants %s, but %s holds %s",
                        task_id, file_globs, existing.task_id, existing.file_globs,
                    )
                    return None

            # Acquire
            lock = FileLock(
                lock_id=f"lock_{task_id}_{int(time.time())}",
                task_id=task_id,
                file_globs=list(file_globs),
                acquired_at=time.time(),
                expires_at=time.time() + timeout,
            )
            self._locks[lock.lock_id] = lock
            LOGGER.info("file_lock: acquired %s for %s on %s", lock.lock_id, task_id, file_globs)
            return lock

    async def release(self, task_id: str) -> int:
        """Release all locks held by a task. Returns count of released locks."""
        async with self._lock:
            to_remove = [
                lid for lid, lock in self._locks.items()
                if lock.task_id == task_id
            ]
            for lid in to_remove:
                del self._locks[lid]
            if to_remove:
                LOGGER.info("file_lock: released %d locks for %s", len(to_remove), task_id)
            return len(to_remove)

    async def release_all(self) -> int:
        """Release all locks. Returns count."""
        async with self._lock:
            count = len(self._locks)
            self._locks.clear()
            return count

    async def check_conflict(self, file_globs: list[str], *, exclude_task_id: str = "") -> list[str]:
        """Return list of task_ids that conflict with given globs."""
        async with self._lock:
            self._purge_expired()
            conflicts: list[str] = []
            for lock in self._locks.values():
                if lock.task_id == exclude_task_id:
                    continue
                if _globs_overlap(file_globs, lock.file_globs):
                    conflicts.append(lock.task_id)
            return conflicts

    @property
    def active_locks(self) -> int:
        return len(self._locks)

    def _purge_expired(self) -> None:
        expired = [lid for lid, lock in self._locks.items() if lock.expired]
        for lid in expired:
            LOGGER.debug("file_lock: purging expired %s", lid)
            del self._locks[lid]


def _has_glob_chars(value: str) -> bool:
    return any(ch in value for ch in "*?[")


def _globs_overlap(globs_a: list[str], globs_b: list[str]) -> bool:
    """Check if any file/glob claims may overlap.

    Mirrors the planner-side conflict check and is intentionally conservative
    for concrete path vs wildcard glob pairs, e.g. ``src/api.py`` vs
    ``src/*.py``.
    """
    import fnmatch
    import os

    if not globs_a or not globs_b:
        return False
    for glob_a in globs_a:
        for glob_b in globs_b:
            if glob_a == glob_b:
                return True
            if fnmatch.fnmatch(glob_a, glob_b) or fnmatch.fnmatch(glob_b, glob_a):
                return True
            base_a = os.path.basename(glob_a)
            base_b = os.path.basename(glob_b)
            if base_a in {"", "*", "**"} or base_b in {"", "*", "**"}:
                continue
            if base_a == base_b and not (_has_glob_chars(base_a) or _has_glob_chars(base_b)):
                return True
            if not _has_glob_chars(base_a) and _has_glob_chars(base_b) and fnmatch.fnmatch(base_a, base_b):
                return True
            if _has_glob_chars(base_a) and not _has_glob_chars(base_b) and fnmatch.fnmatch(base_b, base_a):
                return True
    return False


# Global instance
GLOBAL_FILE_LOCK_MANAGER = FileLockManager()
