from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Toolset:
    name: str
    description: str
    tools: tuple[str, ...] = ()
    includes: tuple[str, ...] = ()
    mcp_categories: tuple[str, ...] = ()


@dataclass
class ToolsetResolution:
    requested: list[str]
    resolved_toolsets: list[str]
    tools: list[str]
    missing_toolsets: list[str] = field(default_factory=list)
    cycles: list[str] = field(default_factory=list)


@dataclass
class MaterializedToolset:
    requested: list[str]
    resolved_toolsets: list[str]
    tools: list[Any]
    tool_names: list[str]
    missing_tools: list[str]
    missing_toolsets: list[str]
    cycles: list[str]
    mcp_schema_cache: dict[str, list[dict[str, str]]]


TOOLSETS: dict[str, Toolset] = {
    "coding/read": Toolset(
        "coding/read",
        "Read repository, artifacts, reviewed skills, recent turns, and architecture context before planning changes.",
        tools=(
            "read_alpha_ravis_artifact",
            "list_alpha_ravis_artifacts",
            "read_alpha_ravis_architecture",
            "list_repo_ai_skills",
            "read_repo_ai_skill",
            "reload_repo_ai_skills",
            "search_session_history",
            "semantic_memory_search",
        ),
    ),
    "coding/write": Toolset(
        "coding/write",
        "Write bounded artifacts, draft reviewed skills, and delegate coding work through Hermes when enabled.",
        tools=(
            "write_alpha_ravis_artifact",
            "export_skill_candidate_to_repo_draft",
            "record_skill_candidate",
            "check_hermes_agent",
            "call_hermes_agent",
            "build_specialist_report",
        ),
        includes=("coding/read",),
    ),
    "coding/execute": Toolset(
        "coding/execute",
        "Run bounded diagnostics through approved local/SSH/Hermes paths; destructive actions stay gated.",
        tools=(
            "execute_local_command",
            "execute_ssh_command",
            "check_external_service",
            "check_hermes_agent",
            "call_hermes_agent",
            "build_specialist_report",
        ),
        includes=("coding/read",),
    ),
    "media/image": Toolset(
        "media/image",
        "Generate and track images through Pixelle and the media gallery; raw media stays out of context by default.",
        tools=(
            "start_pixelle_remote",
            "start_pixelle_async",
            "check_pixelle_job",
            "register_media_asset",
            "semantic_media_search",
            "plan_media_analysis",
            "check_external_service",
        ),
        mcp_categories=("pixelle", "media", "image"),
    ),
    "media/video": Toolset(
        "media/video",
        "Register videos, preserve URLs/file ids, search indexed media, and plan explicit analysis pipelines.",
        tools=("register_media_asset", "semantic_media_search", "plan_media_analysis", "check_external_service"),
        mcp_categories=("pixelle", "media", "video"),
    ),
    "media/audio": Toolset(
        "media/audio",
        "Track audio metadata and plan transcription; audio analysis is explicit and not dumped into context.",
        tools=("register_media_asset", "plan_media_analysis", "check_external_service"),
        mcp_categories=("media", "audio"),
    ),
    "rag/documents": Toolset(
        "rag/documents",
        "Search existing document RAG and normalize external-document hits without duplicating docs into AlphaRavis.",
        tools=("ask_documents", "semantic_memory_search", "check_external_service"),
    ),
    "rag/memory": Toolset(
        "rag/memory",
        "Search and update thread/global memories, archives, artifacts, sessions, and pgvector chunks.",
        tools=(
            "semantic_memory_search",
            "search_archived_context",
            "read_archive_record",
            "read_archive_collection",
            "search_session_history",
            "search_debugging_lessons",
            "search_agent_memory",
            "record_agent_memory",
            "search_curated_memory",
            "record_curated_memory",
            "search_skill_library",
            "list_skill_candidates",
            "activate_skill_candidate",
            "deactivate_skill",
        ),
    ),
    "system/docker": Toolset(
        "system/docker",
        "Inspect Docker/service status through safe diagnostics before assuming an external dependency works.",
        tools=("check_external_service", "execute_local_command", "owner_get_pixelle_logs"),
    ),
    "system/ssh": Toolset(
        "system/ssh",
        "Owner-gated SSH and remote log inspection paths for configured machines.",
        tools=(
            "execute_ssh_command",
            "owner_check_llama_server",
            "owner_get_llama_server_logs",
            "owner_check_comfyui_server",
            "owner_get_pixelle_logs",
        ),
    ),
    "system/power": Toolset(
        "system/power",
        "Owner-gated model and power lifecycle tools; shutdown/reboot paths require HITL approval.",
        tools=(
            "inspect_model_management_status",
            "plan_embedding_maintenance",
            "run_embedding_memory_jobs",
            "queue_vector_memory_backfill",
            "prepare_comfy_for_pixelle",
            "request_power_management_action",
            "wake_on_lan",
            "owner_check_llama_server",
            "owner_start_llama_server",
            "owner_restart_llama_server",
            "owner_check_comfyui_server",
            "owner_start_comfyui_server",
            "owner_start_all_model_services",
            "owner_shutdown_llama_server",
            "owner_shutdown_comfyui_server",
        ),
    ),
    "web/research": Toolset(
        "web/research",
        "Use web search/research and citation normalization for current external facts.",
        tools=("fast_web_search", "deep_web_research", "normalize_research_sources", "check_external_service"),
    ),
    "skills/repo": Toolset(
        "skills/repo",
        "List, reload, and read reviewed repo skill cards and supporting files on demand.",
        tools=("list_repo_ai_skills", "read_repo_ai_skill", "reload_repo_ai_skills"),
    ),
    "skills/evolution": Toolset(
        "skills/evolution",
        "Record reusable workflow candidates and optionally export disabled repo drafts for human review.",
        tools=(
            "search_skill_library",
            "list_skill_candidates",
            "record_skill_candidate",
            "export_skill_candidate_to_repo_draft",
            "activate_skill_candidate",
            "deactivate_skill",
        ),
        includes=("skills/repo",),
    ),
    "artifacts": Toolset(
        "artifacts",
        "Write/read bounded thread artifacts and handoff reports instead of dumping long material into chat.",
        tools=("write_alpha_ravis_artifact", "read_alpha_ravis_artifact", "list_alpha_ravis_artifacts", "build_specialist_report"),
    ),
    "agent/research": Toolset(
        "agent/research",
        "Research agent bounded tool bundle.",
        includes=("web/research", "rag/documents", "rag/memory", "coding/read", "skills/repo", "artifacts"),
    ),
    "agent/general": Toolset(
        "agent/general",
        "Generalist bounded tool bundle for chat, media, memory, skills, and safe orchestration.",
        includes=(
            "media/image",
            "media/video",
            "media/audio",
            "rag/memory",
            "skills/evolution",
            "artifacts",
            "system/power",
            "web/research",
        ),
        tools=("create_manage_memory_tool", "create_search_memory_tool"),
    ),
    "agent/hermes": Toolset(
        "agent/hermes",
        "Hermes bridge bundle for coding, repo, file, terminal-oriented tasks, and handoff reporting.",
        includes=("coding/read", "coding/write", "coding/execute", "rag/memory", "skills/repo", "artifacts"),
    ),
    "agent/context": Toolset(
        "agent/context",
        "Context retrieval bundle for archives, collections, RAG, memory, media search, and repo skill context.",
        includes=("rag/memory", "rag/documents", "media/video", "skills/repo", "artifacts"),
    ),
    "agent/power": Toolset(
        "agent/power",
        "Power/model management bundle for local owner-specific lifecycle management.",
        includes=("system/power", "system/ssh", "system/docker", "rag/memory", "artifacts"),
    ),
    "agent/crisis": Toolset(
        "agent/crisis",
        "Token-light crisis bundle with safe owner recovery actions only.",
        tools=(
            "owner_check_llama_server",
            "owner_start_llama_server",
            "owner_restart_llama_server",
            "owner_get_llama_server_logs",
            "owner_check_comfyui_server",
            "owner_start_comfyui_server",
            "owner_start_all_model_services",
            "owner_get_pixelle_logs",
            "build_specialist_report",
        ),
    ),
}


_TOOLSET_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("coding/execute", ("terminal", "shell", "docker", "logs", "debug", "ssh", "command", "befehl")),
    ("coding/write", ("implement", "patch", "refactor", "code", "repo", "datei", "file", "fix")),
    ("media/image", ("bild", "image", "pixelle", "comfy", "comfyui")),
    ("media/video", ("video", "frame", "timecode")),
    ("media/audio", ("audio", "transcribe", "transkript")),
    ("rag/documents", ("pdf", "document", "dokument", "rag", "quelle")),
    ("rag/memory", ("memory", "archiv", "archive", "frueher", "damals", "erinner", "pgvector")),
    ("system/power", ("wake", "wol", "shutdown", "ollama", "llama", "power", "strom", "server")),
    ("web/research", ("search", "suche", "recherche", "research", "internet", "web")),
    ("skills/evolution", ("skill", "workflow", "lerne", "candidate")),
]


def get_toolset(name: str) -> Toolset | None:
    return TOOLSETS.get(normalize_toolset_name(name))


def get_all_toolsets() -> dict[str, Toolset]:
    return dict(TOOLSETS)


def get_toolset_names() -> list[str]:
    return sorted(TOOLSETS)


def validate_toolset(name: str) -> bool:
    normalized = normalize_toolset_name(name)
    return normalized in TOOLSETS or normalized in {"all", "*"}


def normalize_toolset_name(name: str) -> str:
    return str(name or "").strip().lower().replace("\\", "/")


def resolve_toolset(name: str, visited: set[str] | None = None, stack: tuple[str, ...] = ()) -> ToolsetResolution:
    normalized = normalize_toolset_name(name)
    if visited is None:
        visited = set()

    if normalized in {"all", "*"}:
        aggregate = resolve_multiple_toolsets(get_toolset_names())
        aggregate.requested = [normalized]
        return aggregate

    if normalized in visited:
        return ToolsetResolution([normalized], [], [], cycles=[" -> ".join([*stack, normalized])])
    toolset = TOOLSETS.get(normalized)
    if toolset is None:
        return ToolsetResolution([normalized], [], [], missing_toolsets=[normalized])

    visited.add(normalized)
    resolved = [normalized]
    tools = set(toolset.tools)
    missing: list[str] = []
    cycles: list[str] = []
    for include in toolset.includes:
        child = resolve_toolset(include, visited=visited, stack=(*stack, normalized))
        resolved.extend(child.resolved_toolsets)
        tools.update(child.tools)
        missing.extend(child.missing_toolsets)
        cycles.extend(child.cycles)
    return ToolsetResolution(
        [normalized],
        sorted(dict.fromkeys(resolved)),
        sorted(tools),
        missing_toolsets=sorted(dict.fromkeys(missing)),
        cycles=sorted(dict.fromkeys(cycles)),
    )


def resolve_multiple_toolsets(names: list[str] | tuple[str, ...] | set[str]) -> ToolsetResolution:
    requested = [normalize_toolset_name(name) for name in names if str(name or "").strip()]
    resolved: list[str] = []
    tools: set[str] = set()
    missing: list[str] = []
    cycles: list[str] = []
    for name in requested:
        item = resolve_toolset(name)
        resolved.extend(item.resolved_toolsets)
        tools.update(item.tools)
        missing.extend(item.missing_toolsets)
        cycles.extend(item.cycles)
    return ToolsetResolution(
        requested,
        sorted(dict.fromkeys(resolved)),
        sorted(tools),
        missing_toolsets=sorted(dict.fromkeys(missing)),
        cycles=sorted(dict.fromkeys(cycles)),
    )


def infer_toolsets_from_text(text: str, *, default: tuple[str, ...] = ("agent/general",)) -> list[str]:
    lowered = str(text or "").lower()
    selected: list[str] = []
    for toolset, keywords in _TOOLSET_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            selected.append(toolset)
    if not selected:
        selected.extend(default)
    if "coding/write" in selected and "coding/read" not in selected:
        selected.append("coding/read")
    if "coding/execute" in selected and "system/docker" not in selected and "docker" in lowered:
        selected.append("system/docker")
    return sorted(dict.fromkeys(selected))


def render_toolset_registry(category: str = "", *, include_tools: bool = True, max_tools: int = 16) -> str:
    selected = normalize_toolset_name(category)
    names = [selected] if selected else get_toolset_names()
    lines = ["AlphaRavis lazy toolsets:"]
    for name in names:
        toolset = TOOLSETS.get(name)
        if toolset is None:
            known = ", ".join(get_toolset_names())
            return f"Unknown toolset `{category}`. Known toolsets: {known}"
        resolved = resolve_toolset(name)
        lines.append(f"- {name}: {toolset.description}")
        if toolset.includes:
            lines.append(f"  includes: {', '.join(toolset.includes)}")
        if include_tools:
            shown = ", ".join(resolved.tools[:max_tools]) if resolved.tools else "(no direct local tools)"
            if len(resolved.tools) > max_tools:
                shown += f", and {len(resolved.tools) - max_tools} more"
            lines.append(f"  resolved tools: {shown}")
        if toolset.mcp_categories:
            lines.append(f"  mcp categories: {', '.join(toolset.mcp_categories)}")
        if resolved.cycles:
            lines.append(f"  warnings: cyclic include skipped: {', '.join(resolved.cycles)}")
    return "\n".join(lines)


def tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", getattr(tool, "__name__", "")) or "")


def build_mcp_schema_cache(server_infos: list[dict[str, Any]] | None) -> dict[str, list[dict[str, str]]]:
    cache: dict[str, list[dict[str, str]]] = {}
    for info in server_infos or []:
        server_name = str(info.get("name") or "").lower()
        for tool_info in info.get("tools", []) or []:
            if not isinstance(tool_info, dict):
                continue
            name = str(tool_info.get("name") or "")
            description = str(tool_info.get("description") or "")
            categories = _mcp_categories_for(server_name, name, description)
            entry = {
                "server": str(info.get("name") or ""),
                "name": name,
                "description": description[:500],
            }
            for category in categories:
                cache.setdefault(category, []).append(entry)
    for category, entries in list(cache.items()):
        deduped = {f"{item['server']}:{item['name']}": item for item in entries}
        cache[category] = sorted(deduped.values(), key=lambda item: (item["server"], item["name"]))
    return cache


def schema_cache_fingerprint(cache: dict[str, list[dict[str, str]]]) -> str:
    payload = json.dumps(cache, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def materialize_toolsets(
    toolset_names: list[str] | tuple[str, ...] | set[str],
    available_tools: dict[str, Any],
    *,
    mcp_tools: list[Any] | None = None,
    mcp_schema_cache: dict[str, list[dict[str, str]]] | None = None,
    include_mcp: bool = True,
) -> MaterializedToolset:
    resolution = resolve_multiple_toolsets(toolset_names)
    selected: list[Any] = []
    missing: list[str] = []
    seen_names: set[str] = set()

    for name in resolution.tools:
        tool = available_tools.get(name)
        if tool is None:
            missing.append(name)
            continue
        actual_name = tool_name(tool) or name
        if actual_name in seen_names:
            continue
        seen_names.add(actual_name)
        selected.append(tool)

    if include_mcp and mcp_tools:
        categories = _mcp_categories_for_toolsets(resolution.resolved_toolsets)
        mcp_by_name = {tool_name(tool): tool for tool in mcp_tools if tool_name(tool)}
        cache = mcp_schema_cache or {}
        for category in categories:
            for schema in cache.get(category, []):
                name = schema.get("name", "")
                tool = mcp_by_name.get(name)
                if tool is None or name in seen_names:
                    continue
                seen_names.add(name)
                selected.append(tool)

    return MaterializedToolset(
        requested=resolution.requested,
        resolved_toolsets=resolution.resolved_toolsets,
        tools=selected,
        tool_names=[tool_name(tool) for tool in selected],
        missing_tools=sorted(dict.fromkeys(missing)),
        missing_toolsets=resolution.missing_toolsets,
        cycles=resolution.cycles,
        mcp_schema_cache=mcp_schema_cache or {},
    )


def toolset_profile(materialized: MaterializedToolset) -> dict[str, Any]:
    return {
        "requested": materialized.requested,
        "resolved": materialized.resolved_toolsets,
        "tool_count": len(materialized.tool_names),
        "tool_names": materialized.tool_names,
        "missing_tools": materialized.missing_tools,
        "missing_toolsets": materialized.missing_toolsets,
        "cycles": materialized.cycles,
        "mcp_schema_categories": sorted(materialized.mcp_schema_cache),
        "mcp_schema_fingerprint": schema_cache_fingerprint(materialized.mcp_schema_cache)
        if materialized.mcp_schema_cache
        else "",
    }


def _mcp_categories_for_toolsets(toolsets: list[str]) -> set[str]:
    categories: set[str] = set()
    for name in toolsets:
        toolset = TOOLSETS.get(name)
        if toolset:
            categories.update(toolset.mcp_categories)
    return categories


def _mcp_categories_for(server_name: str, tool_name_value: str, description: str) -> set[str]:
    text = f"{server_name} {tool_name_value} {description}".lower()
    categories: set[str] = set()
    if any(word in text for word in ("pixelle", "comfy", "image", "workflow", "prompt")):
        categories.update({"pixelle", "media", "image"})
    if "video" in text:
        categories.update({"media", "video"})
    if "audio" in text:
        categories.update({"media", "audio"})
    if any(word in text for word in ("rag", "document", "pdf")):
        categories.update({"rag", "document"})
    if any(word in text for word in ("ssh", "power", "wake", "shutdown")):
        categories.update({"system", "power", "ssh"})
    if not categories:
        categories.add(re.sub(r"[^a-z0-9_-]+", "-", server_name).strip("-") or "mcp")
    return categories
