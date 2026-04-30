from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


FileOperation = Literal["read", "list", "write", "delete"]


class FileSafetyError(ValueError):
    """Raised when a file operation targets a sensitive or disallowed path."""


@dataclass(frozen=True)
class FileSafetyDecision:
    allowed: bool
    operation: FileOperation
    path: str
    reason: str = ""
    category: str = ""
    safe_root: str = ""
    allowed_root: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    def refusal(self) -> str:
        detail = f" ({self.category})" if self.category else ""
        return f"Access denied for {self.operation} on `{self.path}`{detail}: {self.reason}"


SENSITIVE_EXACT_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    ".env.test",
    ".netrc",
    ".pgpass",
    ".npmrc",
    ".pypirc",
    "id_rsa",
    "id_ed25519",
    "id_dsa",
    "id_ecdsa",
    "authorized_keys",
    "known_hosts",
    "credentials",
    "credentials.json",
}

SHELL_PROFILE_NAMES = {
    ".bashrc",
    ".zshrc",
    ".profile",
    ".bash_profile",
    ".zprofile",
    ".fishrc",
    "profile.ps1",
    "microsoft.powershell_profile.ps1",
}

SENSITIVE_PATH_PARTS = {
    ".ssh",
    ".aws",
    ".gnupg",
    ".kube",
    ".docker",
    ".azure",
    ".config/gh",
    ".config/gcloud",
    ".config/hub",
    ".hermes/skills/.hub",
}

INTERNAL_CACHE_PARTS = {
    ".git",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    ".codex",
}

SYSTEM_PREFIXES_POSIX = {
    "/bin",
    "/boot",
    "/dev",
    "/etc",
    "/lib",
    "/lib64",
    "/proc",
    "/root",
    "/run",
    "/sbin",
    "/sys",
    "/usr/bin",
    "/usr/lib",
    "/usr/local/bin",
    "/var/lib",
    "/var/run",
}

SYSTEM_PREFIXES_WINDOWS = {
    "c:/windows",
    "c:/program files",
    "c:/program files (x86)",
    "c:/programdata",
}


def check_path_safety(
    path: str | Path,
    *,
    operation: FileOperation,
    allowed_root: str | Path | None = None,
    safe_root_env: str = "ALPHARAVIS_WRITE_SAFE_ROOT",
) -> FileSafetyDecision:
    resolved = _resolve_path(path)
    normalized = _normalize_path(resolved)
    allowed_root_path = _resolve_path(allowed_root) if allowed_root is not None else None
    allowed_root_normalized = _normalize_path(allowed_root_path) if allowed_root_path is not None else ""

    if allowed_root_path is not None and not _is_relative_to(resolved, allowed_root_path):
        return FileSafetyDecision(
            allowed=False,
            operation=operation,
            path=str(resolved),
            reason="path is outside the allowed root",
            category="allowed_root",
            allowed_root=str(allowed_root_path),
        )

    category = _sensitive_category(resolved)
    if category:
        return FileSafetyDecision(
            allowed=False,
            operation=operation,
            path=str(resolved),
            reason="path may contain secrets, credentials, or internal cache data",
            category=category,
            allowed_root=allowed_root_normalized,
        )

    if operation in {"write", "delete"}:
        system_category = _system_path_category(normalized)
        if system_category:
            return FileSafetyDecision(
                allowed=False,
                operation=operation,
                path=str(resolved),
                reason="system paths cannot be modified by AlphaRavis tools",
                category=system_category,
                allowed_root=allowed_root_normalized,
            )

        safe_root = get_safe_write_root(env_name=safe_root_env)
        if safe_root is not None and not _is_relative_to(resolved, safe_root):
            return FileSafetyDecision(
                allowed=False,
                operation=operation,
                path=str(resolved),
                reason=f"path is outside {safe_root_env}",
                category="safe_write_root",
                safe_root=str(safe_root),
                allowed_root=allowed_root_normalized,
            )

    return FileSafetyDecision(
        allowed=True,
        operation=operation,
        path=str(resolved),
        safe_root=str(get_safe_write_root(env_name=safe_root_env) or ""),
        allowed_root=allowed_root_normalized,
    )


def ensure_path_allowed(
    path: str | Path,
    *,
    operation: FileOperation,
    allowed_root: str | Path | None = None,
    safe_root_env: str = "ALPHARAVIS_WRITE_SAFE_ROOT",
) -> FileSafetyDecision:
    decision = check_path_safety(
        path,
        operation=operation,
        allowed_root=allowed_root,
        safe_root_env=safe_root_env,
    )
    if not decision.allowed:
        raise FileSafetyError(decision.refusal())
    return decision


def ensure_read_allowed(path: str | Path, *, allowed_root: str | Path | None = None) -> FileSafetyDecision:
    return ensure_path_allowed(path, operation="read", allowed_root=allowed_root)


def ensure_list_allowed(path: str | Path, *, allowed_root: str | Path | None = None) -> FileSafetyDecision:
    return ensure_path_allowed(path, operation="list", allowed_root=allowed_root)


def ensure_write_allowed(
    path: str | Path,
    *,
    allowed_root: str | Path | None = None,
    safe_root_env: str = "ALPHARAVIS_WRITE_SAFE_ROOT",
) -> FileSafetyDecision:
    return ensure_path_allowed(path, operation="write", allowed_root=allowed_root, safe_root_env=safe_root_env)


def ensure_delete_allowed(
    path: str | Path,
    *,
    allowed_root: str | Path | None = None,
    safe_root_env: str = "ALPHARAVIS_WRITE_SAFE_ROOT",
) -> FileSafetyDecision:
    return ensure_path_allowed(path, operation="delete", allowed_root=allowed_root, safe_root_env=safe_root_env)


def is_path_allowed(
    path: str | Path,
    *,
    operation: FileOperation,
    allowed_root: str | Path | None = None,
    safe_root_env: str = "ALPHARAVIS_WRITE_SAFE_ROOT",
) -> bool:
    return check_path_safety(
        path,
        operation=operation,
        allowed_root=allowed_root,
        safe_root_env=safe_root_env,
    ).allowed


def get_safe_write_root(*, env_name: str = "ALPHARAVIS_WRITE_SAFE_ROOT") -> Path | None:
    value = os.getenv(env_name, "").strip()
    if not value:
        return None
    return _resolve_path(value)


def _resolve_path(path: str | Path | None) -> Path:
    if path is None:
        return Path.cwd().resolve()
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve(strict=False)


def _normalize_path(path: Path) -> str:
    return str(path).replace("\\", "/").rstrip("/").lower()


def _parts_key(path: Path) -> str:
    return "/".join(part.lower() for part in path.parts)


def _sensitive_category(path: Path) -> str:
    name = path.name.lower()
    if name in SENSITIVE_EXACT_NAMES or (name.startswith(".env.") and name != ".env.example"):
        return "sensitive_file"
    if name in SHELL_PROFILE_NAMES:
        return "shell_profile"

    parts = {part.lower() for part in path.parts}
    joined = _parts_key(path)
    for part in SENSITIVE_PATH_PARTS:
        if "/" in part:
            if part in joined:
                return "credential_directory"
        elif part in parts:
            return "credential_directory"

    for part in INTERNAL_CACHE_PARTS:
        if part in parts:
            return "internal_cache"

    return ""


def _system_path_category(normalized_path: str) -> str:
    for prefix in SYSTEM_PREFIXES_POSIX:
        normalized = prefix.lower().rstrip("/")
        if normalized_path == normalized or normalized_path.startswith(normalized + "/"):
            return "system_path"
    for prefix in SYSTEM_PREFIXES_WINDOWS:
        normalized = prefix.rstrip("/")
        if normalized_path == normalized or normalized_path.startswith(normalized + "/"):
            return "system_path"
    return ""


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
