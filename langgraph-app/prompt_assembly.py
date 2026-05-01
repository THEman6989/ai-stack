from __future__ import annotations

import os
import platform
import re
from pathlib import Path
from typing import Iterable


DEFAULT_CONTEXT_MAX_CHARS = 20_000
DEFAULT_HEAD_RATIO = 0.70
DEFAULT_TAIL_RATIO = 0.20

WSL_ENVIRONMENT_HINT = (
    "Runtime hint: AlphaRavis may run inside WSL. Windows drives are mounted "
    "under /mnt, for example /mnt/c maps to C:. Translate Windows paths only "
    "when the current runtime is WSL or the path clearly uses /mnt/<drive>."
)

WINDOWS_ENVIRONMENT_HINT = (
    "Runtime hint: AlphaRavis is running on Windows. Prefer PowerShell-safe "
    "commands and Windows paths. Do not assume Linux-only tools are available "
    "unless the task explicitly targets Docker, WSL, or a Linux host."
)

DOCKER_ENVIRONMENT_HINT = (
    "Runtime hint: AlphaRavis may run inside Docker. Host services can require "
    "container DNS names such as langgraph-api, api-bridge, litellm, postgres, "
    "mongo, or host.docker.internal depending on deployment."
)

STABLE_CONTEXT_POLICY = (
    "Stable prompt policy: keep identity, platform, safety, archive policy, "
    "toolset registry, and architecture hints separate from ephemeral user "
    "tasks, MemoryKernel snippets, skill context, and handoff packets. Treat "
    "ephemeral blocks as current-run evidence, not durable global rules."
)


def env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _is_wsl_runtime() -> bool:
    if env_bool("ALPHARAVIS_FORCE_WSL_HINT", "false"):
        return True
    if "microsoft" in platform.release().lower():
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return False


def _looks_like_windows_path(value: str) -> bool:
    return bool(re.match(r"^[a-zA-Z]:[\\/]", value or ""))


def _looks_like_wsl_path(value: str) -> bool:
    return bool(re.match(r"^/mnt/[a-zA-Z]/", value or ""))


def _looks_like_docker_runtime(cwd: str | Path | None) -> bool:
    if env_bool("ALPHARAVIS_FORCE_DOCKER_HINT", "false"):
        return True
    cwd_text = str(cwd or "")
    if cwd_text.startswith("/workspace") or cwd_text.startswith("/app"):
        return True
    return Path("/.dockerenv").exists()


def build_environment_hints(*, cwd: str | Path | None = None) -> str:
    hints: list[str] = []
    cwd_text = str(cwd or os.getcwd())
    if _is_wsl_runtime() or _looks_like_wsl_path(cwd_text):
        hints.append(WSL_ENVIRONMENT_HINT)
    elif _looks_like_windows_path(cwd_text) or os.name == "nt":
        hints.append(WINDOWS_ENVIRONMENT_HINT)
    if _looks_like_docker_runtime(cwd):
        hints.append(DOCKER_ENVIRONMENT_HINT)
    return "\n\n".join(dict.fromkeys(hints))


def truncate_context_content(
    content: str,
    filename: str = "context",
    *,
    max_chars: int | None = None,
    head_ratio: float | None = None,
    tail_ratio: float | None = None,
) -> str:
    text = str(content or "")
    limit = int(max_chars or DEFAULT_CONTEXT_MAX_CHARS)
    if limit <= 0 or len(text) <= limit:
        return text

    head = float(head_ratio if head_ratio is not None else DEFAULT_HEAD_RATIO)
    tail = float(tail_ratio if tail_ratio is not None else DEFAULT_TAIL_RATIO)
    head = min(max(head, 0.10), 0.90)
    tail = min(max(tail, 0.05), 0.80)
    if head + tail > 0.95:
        scale = 0.95 / (head + tail)
        head *= scale
        tail *= scale

    head_chars = max(1, int(limit * head))
    tail_chars = max(1, int(limit * tail))
    if head_chars + tail_chars >= limit:
        tail_chars = max(1, limit - head_chars - 1)

    marker = (
        f"\n\n[...truncated {filename}: kept {head_chars}+{tail_chars} "
        f"of {len(text)} chars. Use exact file/archive tools to read the full source.]\n\n"
    )
    return text[:head_chars] + marker + text[-tail_chars:]


def stable_prompt_sections(*, cwd: str | Path | None = None) -> list[str]:
    sections = [STABLE_CONTEXT_POLICY]
    env_hints = build_environment_hints(cwd=cwd)
    if env_hints:
        sections.append(env_hints)
    sections.append(
        "Archive policy: old details are retrieved through semantic_memory_search "
        "and raw archive loaders when needed. Do not inject all archive collections "
        "into every prompt."
    )
    sections.append(
        "Tool policy: start from toolset categories and bind/call concrete tools "
        "only when the task requires that capability."
    )
    return sections


def build_stable_prompt_context(*, cwd: str | Path | None = None, extra_sections: Iterable[str] = ()) -> str:
    sections = [*stable_prompt_sections(cwd=cwd), *[str(item).strip() for item in extra_sections if str(item).strip()]]
    return "<stable-runtime-context>\n" + "\n\n".join(sections) + "\n</stable-runtime-context>"
