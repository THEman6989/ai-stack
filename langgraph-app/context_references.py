from __future__ import annotations

import asyncio
import inspect
import mimetypes
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable
from urllib.parse import urlparse

import httpx

from file_safety import ensure_list_allowed, ensure_read_allowed, is_path_allowed


_QUOTED_REFERENCE_VALUE = r'(?:`[^`\n]+`|"[^"\n]+"|\'[^\'\n]+\')'
REFERENCE_PATTERN = re.compile(
    rf"(?<![\w/])@(?:(?P<simple>diff|staged)\b|"
    rf"(?P<kind>file|folder|git|url):(?P<value>{_QUOTED_REFERENCE_VALUE}(?::\d+(?:-\d+)?)?|\S+))"
)
TRAILING_PUNCTUATION = ",.;!?"
DEFAULT_MAX_URL_CHARS = 12000
DEFAULT_FOLDER_LIMIT = 200

@dataclass(frozen=True)
class ContextReference:
    raw: str
    kind: str
    target: str
    start: int
    end: int
    line_start: int | None = None
    line_end: int | None = None


@dataclass
class ContextReferenceResult:
    message: str
    original_message: str
    references: list[ContextReference] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    injected_tokens: int = 0
    expanded: bool = False
    blocked: bool = False

    def profile(self) -> dict[str, object]:
        return {
            "reference_count": len(self.references),
            "expanded": self.expanded,
            "blocked": self.blocked,
            "injected_tokens": self.injected_tokens,
            "warnings": self.warnings[:10],
            "kinds": sorted({ref.kind for ref in self.references}),
        }


def estimate_tokens_rough(text: object) -> int:
    return max(1, len(str(text or "")) // 4)


def parse_context_references(message: str) -> list[ContextReference]:
    refs: list[ContextReference] = []
    if not message:
        return refs

    for match in REFERENCE_PATTERN.finditer(message):
        simple = match.group("simple")
        if simple:
            refs.append(
                ContextReference(
                    raw=match.group(0),
                    kind=simple,
                    target="",
                    start=match.start(),
                    end=match.end(),
                )
            )
            continue

        kind = str(match.group("kind") or "")
        value = _strip_trailing_punctuation(str(match.group("value") or ""))
        target = _strip_reference_wrappers(value)
        line_start = None
        line_end = None

        if kind == "file":
            target, line_start, line_end = _parse_file_reference_value(value)

        refs.append(
            ContextReference(
                raw=match.group(0),
                kind=kind,
                target=target,
                start=match.start(),
                end=match.end(),
                line_start=line_start,
                line_end=line_end,
            )
        )

    return refs


async def preprocess_context_references(
    message: str,
    *,
    cwd: str | Path,
    context_length: int,
    allowed_root: str | Path | None = None,
    soft_ratio: float = 0.25,
    hard_ratio: float = 0.50,
    max_url_chars: int = DEFAULT_MAX_URL_CHARS,
    folder_limit: int = DEFAULT_FOLDER_LIMIT,
    fetch_urls: bool = True,
    url_fetcher: Callable[[str], str | Awaitable[str]] | None = None,
) -> ContextReferenceResult:
    refs = parse_context_references(message)
    if not refs:
        return ContextReferenceResult(message=message, original_message=message)

    cwd_path = Path(cwd).expanduser().resolve()
    allowed_root_path = Path(allowed_root).expanduser().resolve() if allowed_root is not None else cwd_path
    warnings: list[str] = []
    blocks: list[str] = []
    injected_tokens = 0

    for ref in refs:
        warning, block = await _expand_reference(
            ref,
            cwd_path,
            allowed_root=allowed_root_path,
            max_url_chars=max_url_chars,
            folder_limit=folder_limit,
            fetch_urls=fetch_urls,
            url_fetcher=url_fetcher,
        )
        if warning:
            warnings.append(warning)
        if block:
            blocks.append(block)
            injected_tokens += estimate_tokens_rough(block)

    context_length = max(1, int(context_length or 1))
    hard_limit = max(1, int(context_length * hard_ratio))
    soft_limit = max(1, int(context_length * soft_ratio))
    stripped = _remove_reference_tokens(message, refs)

    if injected_tokens > hard_limit:
        warnings.append(
            f"Context reference injection refused: {injected_tokens} estimated tokens exceeds "
            f"the hard limit of {hard_limit}."
        )
        return ContextReferenceResult(
            message=_with_warnings(stripped or message, warnings),
            original_message=message,
            references=refs,
            warnings=warnings,
            injected_tokens=injected_tokens,
            expanded=bool(warnings),
            blocked=True,
        )

    if injected_tokens > soft_limit:
        warnings.append(
            f"Context reference warning: {injected_tokens} estimated tokens exceeds "
            f"the soft limit of {soft_limit}."
        )

    final = stripped
    if warnings:
        final = _with_warnings(final or message, warnings)
    if blocks:
        final = f"{final}\n\n--- Attached Context ---\n\n" + "\n\n".join(blocks)

    return ContextReferenceResult(
        message=final.strip(),
        original_message=message,
        references=refs,
        warnings=warnings,
        injected_tokens=injected_tokens,
        expanded=bool(blocks or warnings),
        blocked=False,
    )


def preprocess_context_references_sync(
    message: str,
    *,
    cwd: str | Path,
    context_length: int,
    allowed_root: str | Path | None = None,
    soft_ratio: float = 0.25,
    hard_ratio: float = 0.50,
    max_url_chars: int = DEFAULT_MAX_URL_CHARS,
    folder_limit: int = DEFAULT_FOLDER_LIMIT,
    fetch_urls: bool = True,
    url_fetcher: Callable[[str], str | Awaitable[str]] | None = None,
) -> ContextReferenceResult:
    coro = preprocess_context_references(
        message,
        cwd=cwd,
        context_length=context_length,
        allowed_root=allowed_root,
        soft_ratio=soft_ratio,
        hard_ratio=hard_ratio,
        max_url_chars=max_url_chars,
        folder_limit=folder_limit,
        fetch_urls=fetch_urls,
        url_fetcher=url_fetcher,
    )
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        raise RuntimeError("Use preprocess_context_references from async code.")
    return asyncio.run(coro)


async def _expand_reference(
    ref: ContextReference,
    cwd: Path,
    *,
    allowed_root: Path,
    max_url_chars: int,
    folder_limit: int,
    fetch_urls: bool,
    url_fetcher: Callable[[str], str | Awaitable[str]] | None,
) -> tuple[str | None, str | None]:
    try:
        if ref.kind == "file":
            return _expand_file_reference(ref, cwd, allowed_root=allowed_root)
        if ref.kind == "folder":
            return _expand_folder_reference(ref, cwd, allowed_root=allowed_root, folder_limit=folder_limit)
        if ref.kind == "diff":
            return _expand_git_reference(ref, cwd, ["diff"], "git diff")
        if ref.kind == "staged":
            return _expand_git_reference(ref, cwd, ["diff", "--staged"], "git diff --staged")
        if ref.kind == "git":
            count = max(1, min(int(ref.target or "1"), 10))
            return _expand_git_reference(ref, cwd, ["log", f"-{count}", "-p"], f"git log -{count} -p")
        if ref.kind == "url":
            if not fetch_urls:
                return f"{ref.raw}: URL fetching is disabled by BRIDGE_CONTEXT_REFERENCES_FETCH_URLS", None
            content = await _fetch_url_content(ref.target, max_chars=max_url_chars, url_fetcher=url_fetcher)
            if not content:
                return f"{ref.raw}: no text content extracted", None
            return None, f"URL reference {ref.raw} ({estimate_tokens_rough(content)} estimated tokens)\n{content}"
    except Exception as exc:
        return f"{ref.raw}: {exc}", None

    return f"{ref.raw}: unsupported reference type", None


def _expand_file_reference(
    ref: ContextReference,
    cwd: Path,
    *,
    allowed_root: Path,
) -> tuple[str | None, str | None]:
    path = _resolve_path(cwd, ref.target, allowed_root=allowed_root)
    _ensure_reference_path_allowed(path, allowed_root=allowed_root, operation="read")
    if not path.exists():
        return f"{ref.raw}: file not found", None
    if not path.is_file():
        return f"{ref.raw}: path is not a file", None
    if _is_binary_file(path):
        return f"{ref.raw}: binary files are not supported", None

    text = path.read_text(encoding="utf-8", errors="replace")
    if ref.line_start is not None:
        lines = text.splitlines()
        start_idx = max(ref.line_start - 1, 0)
        end_idx = min(ref.line_end or ref.line_start, len(lines))
        text = "\n".join(lines[start_idx:end_idx])

    lang = _code_fence_language(path)
    rel = _display_path(path, allowed_root)
    return None, f"File reference {ref.raw} -> {rel} ({estimate_tokens_rough(text)} estimated tokens)\n```{lang}\n{text}\n```"


def _expand_folder_reference(
    ref: ContextReference,
    cwd: Path,
    *,
    allowed_root: Path,
    folder_limit: int,
) -> tuple[str | None, str | None]:
    path = _resolve_path(cwd, ref.target, allowed_root=allowed_root)
    _ensure_reference_path_allowed(path, allowed_root=allowed_root, operation="list")
    if not path.exists():
        return f"{ref.raw}: folder not found", None
    if not path.is_dir():
        return f"{ref.raw}: path is not a folder", None

    listing = _build_folder_listing(path, allowed_root, limit=folder_limit)
    return None, f"Folder reference {ref.raw} ({estimate_tokens_rough(listing)} estimated tokens)\n{listing}"


def _expand_git_reference(
    ref: ContextReference,
    cwd: Path,
    args: list[str],
    label: str,
) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return f"{ref.raw}: git command timed out after 30 seconds", None
    except FileNotFoundError:
        return f"{ref.raw}: git is not available", None

    if result.returncode != 0:
        stderr = (result.stderr or "").strip() or "git command failed"
        return f"{ref.raw}: {stderr}", None
    content = result.stdout.strip() or "(no output)"
    return None, f"{label} reference {ref.raw} ({estimate_tokens_rough(content)} estimated tokens)\n```diff\n{content}\n```"


async def _fetch_url_content(
    url: str,
    *,
    max_chars: int,
    url_fetcher: Callable[[str], str | Awaitable[str]] | None = None,
) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("only http and https URLs are supported")

    if url_fetcher is not None:
        content = url_fetcher(url)
        if inspect.isawaitable(content):
            content = await content
        return str(content or "")[:max_chars].strip()

    async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        text = response.text
        if "html" in content_type.lower():
            text = re.sub(r"<script[\s\S]*?</script>", "", text, flags=re.IGNORECASE)
            text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text)
        return text[:max_chars].strip()


def _resolve_path(cwd: Path, target: str, *, allowed_root: Path) -> Path:
    path = Path(os.path.expanduser(target))
    if not path.is_absolute():
        path = cwd / path
    resolved = path.resolve()
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError("path is outside the allowed workspace") from exc
    return resolved


def _ensure_reference_path_allowed(path: Path, *, allowed_root: Path, operation: str) -> None:
    if operation == "list":
        ensure_list_allowed(path, allowed_root=allowed_root)
    else:
        ensure_read_allowed(path, allowed_root=allowed_root)


def _strip_trailing_punctuation(value: str) -> str:
    stripped = value.rstrip(TRAILING_PUNCTUATION)
    while stripped.endswith((")", "]", "}")):
        closer = stripped[-1]
        opener = {")": "(", "]": "[", "}": "{"}[closer]
        if stripped.count(closer) > stripped.count(opener):
            stripped = stripped[:-1]
            continue
        break
    return stripped


def _strip_reference_wrappers(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "`\"'":
        return value[1:-1]
    return value


def _parse_file_reference_value(value: str) -> tuple[str, int | None, int | None]:
    quoted_match = re.match(
        r'^(?P<quote>`|"|\')(?P<path>.+?)(?P=quote)(?::(?P<start>\d+)(?:-(?P<end>\d+))?)?$',
        value,
    )
    if quoted_match:
        line_start = quoted_match.group("start")
        line_end = quoted_match.group("end")
        return (
            quoted_match.group("path"),
            int(line_start) if line_start is not None else None,
            int(line_end or line_start) if line_start is not None else None,
        )

    range_match = re.match(r"^(?P<path>.+?):(?P<start>\d+)(?:-(?P<end>\d+))?$", value)
    if range_match:
        line_start = int(range_match.group("start"))
        return (
            range_match.group("path"),
            line_start,
            int(range_match.group("end") or range_match.group("start")),
        )

    return _strip_reference_wrappers(value), None, None


def _remove_reference_tokens(message: str, refs: list[ContextReference]) -> str:
    pieces: list[str] = []
    cursor = 0
    for ref in refs:
        pieces.append(message[cursor : ref.start])
        cursor = ref.end
    pieces.append(message[cursor:])
    text = "".join(pieces)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def _with_warnings(message: str, warnings: list[str]) -> str:
    return f"{message.strip()}\n\n--- Context Warnings ---\n" + "\n".join(f"- {warning}" for warning in warnings)


def _is_binary_file(path: Path) -> bool:
    mime, _ = mimetypes.guess_type(path.name)
    text_like_suffixes = {
        ".py",
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".css",
        ".html",
        ".xml",
        ".csv",
        ".env.example",
    }
    if mime and not mime.startswith("text/") and path.suffix.lower() not in text_like_suffixes:
        return True
    return b"\x00" in path.read_bytes()[:4096]


def _build_folder_listing(path: Path, allowed_root: Path, *, limit: int) -> str:
    root_label = _display_path(path, allowed_root)
    lines = [f"{root_label}/"]
    entries = _iter_visible_entries(path, allowed_root, limit=limit)
    for entry in entries:
        rel = _display_path(entry, allowed_root)
        if entry.is_dir():
            lines.append(f"- {rel}/")
        else:
            lines.append(f"- {rel} ({_file_metadata(entry)})")
    if len(entries) >= limit:
        lines.append("- ...")
    return "\n".join(lines)


def _iter_visible_entries(path: Path, allowed_root: Path, *, limit: int) -> list[Path]:
    rg_entries = _rg_files(path, allowed_root, limit=limit)
    if rg_entries is not None:
        entries: set[Path] = set()
        for rel in rg_entries:
            full = allowed_root / rel
            if not full.exists():
                continue
            if not is_path_allowed(full, operation="read", allowed_root=allowed_root):
                continue
            for parent in full.parents:
                if parent == allowed_root or parent == path:
                    continue
                try:
                    parent.relative_to(path)
                except ValueError:
                    continue
                entries.add(parent)
            entries.add(full)
        return sorted(entries, key=lambda item: (not item.is_dir(), str(item)))[:limit]

    output: list[Path] = []
    for root, dirs, files in os.walk(path):
        dirs[:] = sorted(
            d
            for d in dirs
            if not d.startswith(".")
            and d != "__pycache__"
            and is_path_allowed(Path(root) / d, operation="list", allowed_root=allowed_root)
        )
        files = sorted(
            f
            for f in files
            if not f.startswith(".") and is_path_allowed(Path(root) / f, operation="read", allowed_root=allowed_root)
        )
        root_path = Path(root)
        for dirname in dirs:
            output.append(root_path / dirname)
            if len(output) >= limit:
                return output
        for filename in files:
            output.append(root_path / filename)
            if len(output) >= limit:
                return output
    return output


def _rg_files(path: Path, allowed_root: Path, *, limit: int) -> list[Path] | None:
    try:
        rel = path.relative_to(allowed_root)
        result = subprocess.run(
            ["rg", "--files", str(rel)],
            cwd=allowed_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired, ValueError):
        return None
    if result.returncode != 0:
        return None
    return [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()][:limit]


def _file_metadata(path: Path) -> str:
    try:
        if _is_binary_file(path):
            return f"{path.stat().st_size} bytes"
        return f"{path.read_text(encoding='utf-8', errors='replace').count(chr(10)) + 1} lines"
    except Exception:
        return f"{path.stat().st_size} bytes"


def _display_path(path: Path, allowed_root: Path) -> str:
    try:
        return str(path.relative_to(allowed_root)).replace("\\", "/") or "."
    except ValueError:
        return str(path)


def _code_fence_language(path: Path) -> str:
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "jsx",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".json": "json",
        ".md": "markdown",
        ".sh": "bash",
        ".ps1": "powershell",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".toml": "toml",
        ".css": "css",
        ".html": "html",
        ".xml": "xml",
    }
    return mapping.get(path.suffix.lower(), "")
