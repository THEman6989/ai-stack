"""Git worktree isolation for parallel task execution.

Adapted from Hermes Agent's CLI worktree support (cli.py).
Provides safe worktree creation/cleanup for write-enabled parallel tasks.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass
class WorktreeInfo:
    path: str
    branch: str
    repo_root: str
    name: str = ""
    task_id: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "branch": self.branch,
            "repo_root": self.repo_root,
            "name": self.name,
            "task_id": self.task_id,
        }


class WorktreeManager:
    """Manages git worktrees for isolated parallel task execution."""

    def __init__(self, repo_root: str | None = None) -> None:
        self.repo_root = repo_root or self._detect_repo_root()
        self._active: dict[str, WorktreeInfo] = {}

    @staticmethod
    def _detect_repo_root() -> str:
        """Find the git repo root from cwd, avoiding os.getcwd() for blockbuster."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        # Avoid os.getcwd() — blockbuster catches it in async handlers.
        # Default to the standard container workspace path.
        if Path("/workspace").exists():
            return "/workspace"
        return str(Path(__file__).resolve().parents[3])

    @property
    def is_git_repo(self) -> bool:
        git_dir = Path(self.repo_root) / ".git"
        return git_dir.exists()

    # ---- Worktree creation ----

    def create(self, task_id: str) -> WorktreeInfo | None:
        """Create an isolated git worktree for a write-enabled task.

        Returns WorktreeInfo on success, None on failure.
        Side effect: ensures .worktrees/ is in .gitignore.
        """
        if not self.is_git_repo:
            LOGGER.warning("worktree: not a git repository, cannot create worktree for %s", task_id)
            return None

        short_id = uuid.uuid4().hex[:8]
        name = f"alpharavis-{short_id}"
        branch = f"alpharavis/parallel/{task_id}-{short_id}"

        worktrees_dir = Path(self.repo_root) / ".worktrees"
        worktrees_dir.mkdir(parents=True, exist_ok=True)

        wt_path = worktrees_dir / name

        # Ensure .worktrees/ is in .gitignore
        self._ensure_gitignore()

        # Create worktree
        try:
            result = subprocess.run(
                ["git", "worktree", "add", str(wt_path), "-b", branch, "HEAD"],
                capture_output=True, text=True, timeout=30,
                cwd=self.repo_root,
            )
            if result.returncode != 0:
                LOGGER.error("worktree: failed to create: %s", result.stderr.strip())
                return None
        except Exception as exc:
            LOGGER.error("worktree: creation failed: %s", exc)
            return None

        # Copy .worktreeinclude files if present
        self._copy_include_files(wt_path)

        info = WorktreeInfo(
            path=str(wt_path),
            branch=branch,
            repo_root=self.repo_root,
            name=name,
            task_id=task_id,
        )
        self._active[task_id] = info
        LOGGER.info("worktree: created %s for task %s at %s", name, task_id, wt_path)
        return info

    # ---- Worktree cleanup ----

    def remove(self, task_id: str, *, force: bool = False) -> bool:
        """Remove a worktree and its branch.

        By default, only removes if the worktree has no uncommitted changes
        (i.e. the task was completed and committed). Use force=True to
        remove regardless.
        """
        info = self._active.pop(task_id, None)
        if info is None:
            LOGGER.debug("worktree: no active worktree for task %s", task_id)
            return False

        wt_path = Path(info.path)
        if not wt_path.exists():
            # Already gone
            self._delete_branch(info.branch)
            return True

        try:
            cmd = ["git", "worktree", "remove", str(wt_path)]
            if force:
                cmd.append("--force")
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=30,
                cwd=self.repo_root,
            )
            if result.returncode != 0:
                LOGGER.warning("worktree: remove failed for %s: %s", task_id, result.stderr.strip())
                return False
        except Exception as exc:
            LOGGER.error("worktree: remove failed for %s: %s", task_id, exc)
            return False

        # Delete the branch
        self._delete_branch(info.branch)
        LOGGER.info("worktree: removed %s for task %s", info.name, task_id)
        return True

    def remove_all(self, *, force: bool = False) -> int:
        """Remove all active worktrees. Returns count removed."""
        task_ids = list(self._active.keys())
        removed = 0
        for task_id in task_ids:
            if self.remove(task_id, force=force):
                removed += 1
        return removed

    # ---- Status ----

    def get_info(self, task_id: str) -> WorktreeInfo | None:
        return self._active.get(task_id)

    @property
    def active_count(self) -> int:
        return len(self._active)

    def check_uncommitted(self, task_id: str) -> bool:
        """Check if a task's worktree has uncommitted changes."""
        info = self._active.get(task_id)
        if info is None:
            return False
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=10,
                cwd=info.path,
            )
            return bool(result.stdout.strip())
        except Exception:
            return True  # assume dirty on error

    # ---- Internal helpers ----

    def _ensure_gitignore(self) -> None:
        gitignore = Path(self.repo_root) / ".gitignore"
        entry = ".worktrees/"
        try:
            existing = gitignore.read_text() if gitignore.exists() else ""
            if entry not in existing.splitlines():
                with open(gitignore, "a", encoding="utf-8") as f:
                    if existing and not existing.endswith("\n"):
                        f.write("\n")
                    f.write(f"{entry}\n")
        except Exception as exc:
            LOGGER.debug("worktree: could not update .gitignore: %s", exc)

    def _copy_include_files(self, wt_path: Path) -> None:
        include_file = Path(self.repo_root) / ".worktreeinclude"
        if not include_file.exists():
            return
        repo_root_resolved = Path(self.repo_root).resolve()
        wt_path_resolved = wt_path.resolve()
        try:
            for line in include_file.read_text().splitlines():
                entry = line.strip()
                if not entry or entry.startswith("#"):
                    continue
                src = Path(self.repo_root) / entry
                dst = wt_path / entry
                try:
                    src_resolved = src.resolve(strict=False)
                    dst_resolved = dst.resolve(strict=False)
                except (OSError, ValueError):
                    LOGGER.debug("worktree: skipping invalid include entry: %s", entry)
                    continue
                if not str(src_resolved).startswith(str(repo_root_resolved)):
                    continue
                if not str(dst_resolved).startswith(str(wt_path_resolved)):
                    continue
                if src.is_file():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(src), str(dst))
                elif src.is_dir():
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            os.symlink(str(src_resolved), str(dst))
                        except (OSError, NotImplementedError):
                            try:
                                shutil.copytree(str(src_resolved), str(dst), symlinks=True, dirs_exist_ok=False)
                            except Exception:
                                LOGGER.debug("worktree: copy fallback failed for %s", entry)
        except Exception as exc:
            LOGGER.debug("worktree: include copy error: %s", exc)

    def _delete_branch(self, branch: str) -> None:
        try:
            subprocess.run(
                ["git", "branch", "-D", branch],
                capture_output=True, text=True, timeout=10,
                cwd=self.repo_root,
            )
        except Exception as exc:
            LOGGER.debug("worktree: branch delete failed for %s: %s", branch, exc)


# ---------------------------------------------------------------------------
# Convenience: check if repo has uncommitted work before creating worktrees
# ---------------------------------------------------------------------------


def repo_has_uncommitted_changes(repo_root: str | None = None) -> bool:
    """Check if the main repo has uncommitted changes.

    Parallel execution should never overwrite dirty user work.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
            cwd=repo_root or ".",
        )
        return bool(result.stdout.strip())
    except Exception:
        return True  # assume dirty on error


def repo_current_branch(repo_root: str | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, timeout=5,
            cwd=repo_root or ".",
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"
