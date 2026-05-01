from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


SNAPSHOT_VERSION = 1
SUPPORTING_DIRS = ("references", "templates", "scripts", "assets")
EXCLUDED_SKILL_DIRS = {
    ".git",
    ".github",
    ".hub",
    ".archive",
    ".cache",
    "__pycache__",
}

_SKILL_INVALID_CHARS = re.compile(r"[^a-z0-9-]+")
_SKILL_MULTI_HYPHEN = re.compile(r"-{2,}")


def slugify_skill_name(name: str, fallback: str = "skill") -> str:
    raw = (name or "").strip().lower().replace("_", "-").replace(" ", "-")
    slug = _SKILL_INVALID_CHARS.sub("-", raw)
    slug = _SKILL_MULTI_HYPHEN.sub("-", slug).strip("-")
    return slug or fallback


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    if not content.startswith("---"):
        return {}, content

    match = re.search(r"\n---\s*(?:\n|$)", content[3:])
    if not match:
        return {}, content

    yaml_text = content[3 : match.start() + 3]
    body = content[match.end() + 3 :]

    try:
        import yaml  # type: ignore

        parsed = yaml.safe_load(yaml_text) or {}
        if isinstance(parsed, dict):
            return parsed, body
    except Exception:
        pass

    frontmatter: dict[str, Any] = {}
    for line in yaml_text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if not key:
            continue
        frontmatter[key] = _parse_simple_frontmatter_value(value.strip())
    return frontmatter, body


def extract_skill_description(frontmatter: dict[str, Any], body: str, max_chars: int = 500) -> str:
    description = str(frontmatter.get("description") or "").strip().strip("'\"")
    if not description:
        for line in body.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                description = stripped
                break
    description = " ".join(description.split())
    if len(description) > max_chars:
        return description[: max(0, max_chars - 3)].rstrip() + "..."
    return description


def extract_skill_conditions(frontmatter: dict[str, Any]) -> dict[str, Any]:
    metadata = frontmatter.get("metadata") if isinstance(frontmatter, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    hermes = metadata.get("hermes") if isinstance(metadata.get("hermes"), dict) else {}

    def _first_list(*keys: str) -> list[str]:
        for key in keys:
            if key in frontmatter:
                return _as_list(frontmatter.get(key))
            if isinstance(hermes, dict) and key in hermes:
                return _as_list(hermes.get(key))
        return []

    return {
        "platforms": _as_list(frontmatter.get("platforms")),
        "status": str(frontmatter.get("status") or "active").strip().lower() or "active",
        "fallback_only": _as_bool(frontmatter.get("fallback_only"), default=False),
        "requires_tools": _first_list("requires_tools", "required_tools"),
        "requires_tool_categories": _first_list(
            "requires_tool_categories",
            "required_tool_categories",
            "requires_toolsets",
        ),
        "fallback_for_tools": _first_list("fallback_for_tools"),
        "fallback_for_tool_categories": _first_list(
            "fallback_for_tool_categories",
            "fallback_for_toolsets",
        ),
    }


def skill_matches_platform(frontmatter: dict[str, Any]) -> bool:
    platforms = _as_list(frontmatter.get("platforms"))
    if not platforms:
        return True
    current = _current_platform()
    aliases = {
        "win": "windows",
        "windows": "windows",
        "linux": "linux",
        "mac": "macos",
        "macos": "macos",
        "darwin": "macos",
        "all": "all",
        "*": "all",
    }
    for platform in platforms:
        normalized = aliases.get(platform.lower().strip(), platform.lower().strip())
        if normalized in {"all", current}:
            return True
    return False


def build_repo_skill_manifest(skills_dir: Path) -> dict[str, list[int]]:
    manifest: dict[str, list[int]] = {}
    for path in _iter_manifest_files(skills_dir):
        try:
            stat = path.stat()
            manifest[_posix_rel(path, skills_dir)] = [stat.st_mtime_ns, stat.st_size]
        except OSError:
            continue
    return manifest


def default_cache_path(workspace_root: Path) -> Path:
    return workspace_root / ".cache" / "alpharavis" / "repo_skill_manifest.json"


def scan_repo_skills(
    skills_dir: Path,
    *,
    workspace_root: Path,
    use_cache: bool = True,
    cache_path: Path | None = None,
    force: bool = False,
    supporting_file_limit: int = 40,
    include_drafts: bool = False,
) -> dict[str, Any]:
    skills_dir = skills_dir.resolve()
    workspace_root = workspace_root.resolve()
    _ensure_under(skills_dir, workspace_root)

    cache_path = (cache_path or default_cache_path(workspace_root)).resolve()
    _ensure_under(cache_path, workspace_root)
    supporting_file_limit = max(0, min(int(supporting_file_limit), 200))

    manifest = build_repo_skill_manifest(skills_dir) if skills_dir.exists() else {}
    scan_config = {
        "supporting_file_limit": supporting_file_limit,
        "include_drafts": bool(include_drafts),
        "platform": _current_platform(),
    }
    if use_cache and not force:
        cached = _load_snapshot(cache_path, manifest, scan_config)
        if cached is not None:
            cached = dict(cached)
            cached["cache_status"] = "cache_hit"
            cached["cache_path"] = str(cache_path)
            return cached

    skills = []
    if skills_dir.exists():
        for skill_md in _iter_skill_index_files(skills_dir, "SKILL.md"):
            entry = _build_skill_entry(
                skill_md,
                skills_dir=skills_dir,
                workspace_root=workspace_root,
                supporting_file_limit=supporting_file_limit,
            )
            if entry is None:
                continue
            status = str((entry.get("conditions") or {}).get("status") or "active").lower()
            if status == "draft" and not include_drafts:
                continue
            if not skill_matches_platform(entry.get("frontmatter") or {}):
                continue
            skills.append(entry)

    snapshot = {
        "version": SNAPSHOT_VERSION,
        "generated_at": int(time.time()),
        "skills_dir": str(skills_dir),
        "manifest": manifest,
        "config": scan_config,
        "skills": skills,
        "cache_status": "cache_miss" if use_cache else "cache_disabled",
        "cache_path": str(cache_path),
    }
    if use_cache:
        try:
            _write_snapshot(cache_path, snapshot)
            snapshot["cache_status"] = "cache_written"
        except Exception as exc:
            snapshot["cache_status"] = "cache_write_failed"
            snapshot["cache_error"] = str(exc)
    return snapshot


def reload_repo_skill_manifest(
    skills_dir: Path,
    *,
    workspace_root: Path,
    cache_path: Path | None = None,
    supporting_file_limit: int = 40,
    include_drafts: bool = False,
) -> dict[str, Any]:
    workspace_root = workspace_root.resolve()
    cache_path = (cache_path or default_cache_path(workspace_root)).resolve()
    _ensure_under(cache_path, workspace_root)
    before_snapshot = _load_snapshot_unchecked(cache_path)
    before = _skills_by_slug(before_snapshot.get("skills") if before_snapshot else [])
    after_snapshot = scan_repo_skills(
        skills_dir,
        workspace_root=workspace_root,
        use_cache=True,
        cache_path=cache_path,
        force=True,
        supporting_file_limit=supporting_file_limit,
        include_drafts=include_drafts,
    )
    after = _skills_by_slug(after_snapshot.get("skills") or [])

    before_slugs = set(before)
    after_slugs = set(after)
    added = [after[slug] for slug in sorted(after_slugs - before_slugs)]
    removed = [before[slug] for slug in sorted(before_slugs - after_slugs)]
    changed = [
        after[slug]
        for slug in sorted(before_slugs & after_slugs)
        if _skill_change_fingerprint(before[slug]) != _skill_change_fingerprint(after[slug])
    ]
    unchanged = sorted((before_slugs & after_slugs) - {skill["slug"] for skill in changed})

    return {
        "cache_path": str(cache_path),
        "cache_status": after_snapshot.get("cache_status"),
        "added": _trim_skill_list(added),
        "removed": _trim_skill_list(removed),
        "changed": _trim_skill_list(changed),
        "unchanged": unchanged,
        "total": len(after),
        "skills": _trim_skill_list(list(after.values())),
    }


def format_skill_manifest(skills: list[dict[str, Any]], *, max_chars: int = 4000, cache_status: str = "") -> str:
    if not skills:
        return "No valid repo AI skill cards found."
    lines = []
    if cache_status:
        lines.append(f"Repo skill manifest: {cache_status}")
    for skill in skills:
        support_count = len(skill.get("supporting_files") or [])
        support = f" ({support_count} supporting files)" if support_count else ""
        lines.append(
            f"- {skill.get('name') or skill.get('slug')}: {skill.get('description', '')}{support}\n"
            f"  Path: {skill.get('path', '')}"
        )
    text = "\n".join(lines)
    max_chars = max(1000, min(int(max_chars), 12000))
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n[Repo skill manifest truncated.]"
    return text


def repo_skill_hint_context(query: str, skills: list[dict[str, Any]], limit: int) -> str:
    if not skills:
        return ""

    query_terms = {term for term in re.split(r"[^a-zA-Z0-9]+", query.lower()) if len(term) >= 4}
    scored: list[tuple[int, dict[str, Any]]] = []
    for skill in skills:
        haystack = " ".join(
            [
                str(skill.get("name") or ""),
                str(skill.get("description") or ""),
                str(skill.get("slug") or ""),
                " ".join(str(item) for item in (skill.get("supporting_files") or [])),
            ]
        ).lower()
        score = sum(1 for term in query_terms if term in haystack)
        if score:
            scored.append((score, skill))

    scored.sort(key=lambda item: (-item[0], str(item[1].get("name") or item[1].get("slug") or "")))
    selected = [skill for _, skill in scored[: max(1, int(limit))]]
    if not selected:
        return ""

    lines = [
        "Reviewed repo AI skill cards may match this task. They are metadata hints only; "
        "read the full card with read_repo_ai_skill only if needed."
    ]
    for skill in selected:
        support = skill.get("supporting_files") or []
        suffix = " Supporting files exist." if support else ""
        lines.append(f"- {skill.get('name')}: {skill.get('description')}{suffix}")
    return "\n".join(lines)


def resolve_skill_file_path(
    skills_dir: Path,
    skill_name: str,
    requested_file: str = "",
) -> tuple[str, Path]:
    slug = slugify_skill_name(skill_name)
    skill_dir = skills_dir / slug
    requested_file = (requested_file or "").strip().replace("\\", "/")
    if not requested_file:
        return slug, skill_dir / "SKILL.md"

    rel = Path(requested_file)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("supporting file path must be relative to the skill directory")

    if rel.as_posix().lower() == "skill.md":
        return slug, skill_dir / "SKILL.md"

    if len(rel.parts) == 1:
        name = rel.name
        if "." not in name:
            name = f"{name}.md"
        rel = Path("references") / name
    elif rel.parts[0] not in SUPPORTING_DIRS:
        raise ValueError(
            "supporting file path must be SKILL.md or start with references/, templates/, scripts/, or assets/"
        )

    return slug, skill_dir / rel


def render_skill_draft_from_candidate(
    candidate: dict[str, Any],
    *,
    candidate_key: str,
    approval_note: str = "",
) -> str:
    name = str(candidate.get("name") or candidate_key or "skill").strip()
    slug = slugify_skill_name(name, fallback=slugify_skill_name(candidate_key, "skill"))
    trigger = str(candidate.get("trigger") or "").strip()
    description = trigger or f"Reviewed draft workflow generated from AlphaRavis candidate {candidate_key}."
    description = " ".join(description.split())[:600]
    created_at = int(time.time())

    sections = [
        "---",
        f"name: {slug}",
        f"description: {json.dumps(description, ensure_ascii=False)}",
        "status: draft",
        "source: alpharavis-skill-candidate",
        f"source_candidate_key: {candidate_key}",
        "review_required: true",
        "---",
        "",
        f"# {name}",
        "",
        "This draft was exported from an inactive AlphaRavis skill candidate. Review, edit, and move it into a reviewed skill path before relying on it.",
        "",
        "## Trigger",
        "",
        trigger or "- TODO: describe when this skill should be used.",
        "",
        "## Workflow",
        "",
        _as_markdown_block(candidate.get("steps"), fallback="- TODO: write the reviewed workflow steps."),
        "",
        "## Success Signals",
        "",
        _as_markdown_block(candidate.get("success_signals"), fallback="- TODO: define success signals."),
        "",
        "## Safety Notes",
        "",
        _as_markdown_block(candidate.get("safety_notes"), fallback="- Keep this skill non-destructive unless separate tool approvals apply."),
        "",
        "## Evidence",
        "",
        _as_markdown_block(candidate.get("evidence"), fallback="- TODO: add review evidence or examples."),
        "",
        "## Source Task",
        "",
        _as_markdown_block(candidate.get("source_task"), fallback="- TODO: add the originating task context if useful."),
        "",
        "## Review Checklist",
        "",
        "- Confirm the trigger is narrow enough.",
        "- Confirm tool permissions and HITL gates are still respected.",
        "- Confirm no secrets or local-only values are embedded.",
        "- Move out of the draft folder only after human review.",
        "",
        "## Export Metadata",
        "",
        f"- candidate_key: `{candidate_key}`",
        f"- exported_at_unix: `{created_at}`",
        f"- confidence: `{candidate.get('confidence', '')}`",
    ]
    if approval_note.strip():
        sections.extend(["", "## Review Note", "", approval_note.strip()[:1200]])
    return "\n".join(sections).rstrip() + "\n"


def _parse_simple_frontmatter_value(value: str) -> Any:
    value = value.strip().strip("'\"")
    if not value:
        return ""
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [part.strip().strip("'\"") for part in inner.split(",") if part.strip()]
    return value


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = _parse_simple_frontmatter_value(value)
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _as_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _current_platform() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def _ensure_under(path: Path, root: Path) -> None:
    resolved = path.resolve()
    root = root.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path is outside workspace root: {resolved}")


def _posix_rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _iter_skill_index_files(skills_dir: Path, filename: str):
    matches: list[Path] = []
    if not skills_dir.exists():
        return
    for root, dirs, files in os.walk(skills_dir):
        dirs[:] = [directory for directory in dirs if directory not in EXCLUDED_SKILL_DIRS]
        if filename in files:
            matches.append(Path(root) / filename)
    for path in sorted(matches, key=lambda item: item.relative_to(skills_dir).as_posix()):
        yield path


def _iter_manifest_files(skills_dir: Path):
    if not skills_dir.exists():
        return
    for root, dirs, files in os.walk(skills_dir):
        dirs[:] = [directory for directory in dirs if directory not in EXCLUDED_SKILL_DIRS]
        root_path = Path(root)
        for filename in files:
            path = root_path / filename
            rel_parts = path.relative_to(skills_dir).parts
            if filename in {"SKILL.md", "DESCRIPTION.md"} or (
                len(rel_parts) >= 3 and rel_parts[1] in SUPPORTING_DIRS
            ):
                yield path


def _load_snapshot(
    cache_path: Path,
    manifest: dict[str, list[int]],
    scan_config: dict[str, Any],
) -> dict[str, Any] | None:
    snapshot = _load_snapshot_unchecked(cache_path)
    if not snapshot:
        return None
    if snapshot.get("version") != SNAPSHOT_VERSION:
        return None
    if snapshot.get("manifest") != manifest:
        return None
    if snapshot.get("config") != scan_config:
        return None
    return snapshot


def _load_snapshot_unchecked(cache_path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _write_snapshot(cache_path: Path, snapshot: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(cache_path)


def _build_skill_entry(
    skill_md: Path,
    *,
    skills_dir: Path,
    workspace_root: Path,
    supporting_file_limit: int,
) -> dict[str, Any] | None:
    try:
        content = skill_md.read_text(encoding="utf-8")
    except Exception:
        return None

    frontmatter, body = parse_frontmatter(content)
    name = str(frontmatter.get("name") or skill_md.parent.name).strip() or skill_md.parent.name
    slug = slugify_skill_name(name, fallback=skill_md.parent.name)
    description = extract_skill_description(frontmatter, body)
    conditions = extract_skill_conditions(frontmatter)
    stat = skill_md.stat()
    supporting_files = _collect_supporting_files(
        skill_md.parent,
        workspace_root=workspace_root,
        limit=supporting_file_limit,
    )
    return {
        "slug": slug,
        "name": name,
        "description": description,
        "path": _posix_rel(skill_md, workspace_root),
        "skill_dir": _posix_rel(skill_md.parent, workspace_root),
        "category": _category_for(skill_md, skills_dir),
        "platforms": conditions.get("platforms", []),
        "conditions": conditions,
        "supporting_files": supporting_files,
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
        "frontmatter": _json_safe(frontmatter),
    }


def _category_for(skill_md: Path, skills_dir: Path) -> str:
    rel = skill_md.relative_to(skills_dir)
    if len(rel.parts) > 2:
        return "/".join(rel.parts[:-2])
    return "general"


def _collect_supporting_files(skill_dir: Path, *, workspace_root: Path, limit: int) -> list[str]:
    if limit <= 0:
        return []
    files: list[str] = []
    for dirname in SUPPORTING_DIRS:
        root = skill_dir / dirname
        if not root.exists():
            continue
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
            if len(files) >= limit:
                return files
            if path.is_file() and not path.is_symlink():
                try:
                    _ensure_under(path, workspace_root)
                    files.append(path.relative_to(skill_dir).as_posix())
                except Exception:
                    continue
    return files


def _skills_by_slug(skills: Any) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    if not isinstance(skills, list):
        return output
    for skill in skills:
        if not isinstance(skill, dict):
            continue
        slug = str(skill.get("slug") or slugify_skill_name(str(skill.get("name") or ""))).strip()
        if slug:
            output[slug] = skill
    return output


def _skill_change_fingerprint(skill: dict[str, Any]) -> tuple[Any, ...]:
    return (
        skill.get("description"),
        tuple(skill.get("supporting_files") or []),
        skill.get("mtime_ns"),
        skill.get("size"),
        json.dumps(skill.get("conditions") or {}, sort_keys=True, default=str),
    )


def _trim_skill_list(skills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trimmed = []
    for skill in skills:
        trimmed.append(
            {
                "slug": skill.get("slug"),
                "name": skill.get("name"),
                "description": skill.get("description"),
                "path": skill.get("path"),
                "supporting_files": skill.get("supporting_files") or [],
                "conditions": skill.get("conditions") or {},
            }
        )
    return trimmed


def _as_markdown_block(value: Any, *, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    lines = text.splitlines()
    if len(lines) == 1:
        return lines[0]
    return "\n".join(lines)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
