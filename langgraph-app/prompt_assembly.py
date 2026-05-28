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

OFFICECLI_POLICY_PROMPT = (
    "OfficeCLI policy: when ALPHARAVIS_ENABLE_OFFICECLI=true and the user asks "
    "for Office documents, use the office/documents toolset and OfficeCLI via "
    "bounded terminal commands. Prefer L1 read/inspect commands before edits: "
    "officecli create <file>.docx|.xlsx|.pptx; officecli view <file> "
    "outline|text|issues|html|screenshot --json when supported; officecli get "
    "or query for stable paths; officecli add/set for edits; officecli validate "
    "before delivery. Write generated files under /workspace/office-output/ "
    "unless the user gives another path. Use watch/live preview only when "
    "explicitly needed."
)

STABLE_CONTEXT_POLICY = (
    "Stable prompt policy: keep identity, platform, safety, archive policy, "
    "toolset registry, and architecture hints separate from ephemeral user "
    "tasks, MemoryKernel snippets, skill context, and handoff packets. Treat "
    "ephemeral blocks as current-run evidence, not durable global rules."
)


def env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def officecli_prompt_enabled() -> bool:
    return env_bool("ALPHARAVIS_ENABLE_OFFICECLI", "false")


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
    # Use the passed cwd, or default to /workspace (standard container path)
    # — avoiding os.getcwd() to prevent blockbuster BlockingError in async handlers.
    cwd_text = str(cwd or "/workspace")
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
    if officecli_prompt_enabled():
        sections.append(OFFICECLI_POLICY_PROMPT)
    return sections


def build_stable_prompt_context(*, cwd: str | Path | None = None, extra_sections: Iterable[str] = ()) -> str:
    sections = [*stable_prompt_sections(cwd=cwd), *[str(item).strip() for item in extra_sections if str(item).strip()]]
    return "<stable-runtime-context>\n" + "\n\n".join(sections) + "\n</stable-runtime-context>"


# ---------- agent policy prompts ----------

HANDOFF_POLICY_PROMPT = (
    "Handoff policy: before you transfer to another AlphaRavis agent, create a "
    "handoff packet with build_specialist_report. The packet must state what is "
    "done, what remains open, evidence/source keys, files/commands/tools used, "
    "verification status, risks, and the exact next-agent instruction. Do not "
    "put long logs in the packet; store them as artifacts and cite the artifact key."
)

ARCHIVE_RETRIEVAL_POLICY_PROMPT = (
    "Archived context policy: archived context is not automatically loaded into "
    "the active prompt. If the user asks about earlier work, old debugging, "
    "previous decisions, 'damals', 'vorhin', 'letztes Mal', or if a summary says "
    "details are archived, use semantic_memory_search first. Archive collections "
    "are tables of contents; inspect child_archive_keys and load only relevant "
    "raw archives before relying on exact old details. Cross-thread retrieval "
    "requires an explicit user request."
)

CODE_WINDOW_POLICY_PROMPT = (
    "Code-window policy: when showing code, patches, shell snippets, logs, JSON, "
    "YAML, or config, use normal Markdown fenced code blocks with a language tag "
    "when known. Do not wrap code in HTML or proprietary canvas markers unless "
    "the user explicitly asks for an artifact file."
)

SPECIALIST_LOCAL_PLAN_PROMPT = (
    "Specialist planning policy: when you receive an execution plan or current "
    "task brief, first adapt it into your own short specialist plan before "
    "doing substantive work. Keep this internal plan concise: objective, needed "
    "tools/retrieval, safety gates, success criteria, and handoff target if one "
    "is likely. Do not replace the planner's task contract; refine only the "
    "part your specialist role owns."
)

# ---------- fast path routing patterns ----------

FAST_PATH_DENY_PATTERNS: list[str] = [
    "agent",
    "alpha ravis",
    "alpharavis",
    "archiv",
    "architecture",
    "code",
    "comfy",
    "context",
    "debug",
    "deepagents",
    "docker",
    "dokument",
    "datei",
    "embedding",
    "fehl",
    "git",
    "hermes",
    "image",
    "install",
    "kompression",
    "log",
    "memory",
    "mcp",
    "model management",
    "ollama",
    "pc",
    "power",
    "pdf",
    "pixelle",
    "python",
    "recherche",
    "research",
    "server",
    "shell",
    "ssh",
    "starte",
    "starten",
    "shutdown",
    "suche",
    "terminal",
    "tool",
    "wake",
    "was kannst du",
    "wer bist",
    "wol",
]

FAST_PATH_FORCE_PATTERNS: list[str] = [
    "fast path",
    "ohne tools",
    "nur chat",
    "simple chat",
]
