from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import json
import os
import re
import shlex
import time
from pathlib import Path
from typing import Any

import httpx
from deepagents import create_deep_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.globals import set_llm_cache
from langchain_core.messages import AIMessage, BaseMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_litellm import ChatLiteLLM
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_redis import RedisCache
from langgraph.func import task
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import interrupt
from langgraph_swarm import create_handoff_tool, create_swarm
from langmem import create_manage_memory_tool, create_search_memory_tool
from mcp import ClientSession
from mcp.client.sse import sse_client
from typing_extensions import NotRequired

try:
    from langgraph.config import get_config, get_store
except Exception:  # pragma: no cover - older local LangGraph imports
    get_config = None
    get_store = None

try:
    from langgraph_cua import create_cua
except Exception as exc:  # pragma: no cover - optional local dependency
    create_cua = None
    CUA_IMPORT_ERROR: Exception | None = exc
else:
    CUA_IMPORT_ERROR = None

try:
    from langgraph_sdk.runtime import ServerRuntime
except Exception:  # pragma: no cover - older local CLI imports
    ServerRuntime = Any  # type: ignore[misc,assignment]


class AlphaRavisState(MessagesState):
    active_agent: NotRequired[str]
    active_skill_context: NotRequired[str]
    fast_path_route: NotRequired[str]
    fast_path_locked: NotRequired[bool]
    fast_path_lock_reason: NotRequired[str]
    run_profile: NotRequired[dict[str, Any]]
    thread_id: NotRequired[str]
    thread_key: NotRequired[str]
    context_summary: NotRequired[str]
    archive_summary: NotRequired[str]
    archived_context_keys: NotRequired[list[str]]
    archive_collection_keys: NotRequired[list[str]]
    compressed_archive_keys: NotRequired[list[str]]
    memory_notice: NotRequired[str]
    memory_notice_key: NotRequired[str]
    memory_notice_seen_key: NotRequired[str]
    skill_candidate_keys: NotRequired[list[str]]


class DebuggerState(MessagesState):
    internal_logs: NotRequired[list[str]]


pcs_env = os.getenv("REMOTE_PCS", "{}")
try:
    REMOTE_PCS = json.loads(pcs_env)
except Exception as exc:
    print(f"Error loading REMOTE_PCS: {exc}")
    REMOTE_PCS = {}

SSH_USER = os.getenv("SSH_USER", "root")
SSH_PASS_DEFAULT = os.getenv("SSH_PASS_DEFAULT", "")
PIXELLE_URL = os.getenv("PIXELLE_URL", "http://pixelle:9004")
COMFY_IP = REMOTE_PCS.get("comfy_server", {}).get("ip")
ARCHIVE_INDEX_NS = ("alpharavis", "archive_index")
ARCHIVE_COLLECTION_INDEX_NS = ("alpharavis", "archive_collection_index")
DEBUGGING_LESSON_NS = ("alpharavis", "debugging_lessons")
SKILL_LIBRARY_NS = ("alpharavis", "skill_library")
SKILL_CONTEXT_MESSAGE_ID = "alpharavis_skill_library_context"
MANUAL_COMPRESSION_PATTERNS = [
    "archive diesen abschnitt",
    "archiviere diesen abschnitt",
    "archiviere jetzt",
    "compress now",
    "komprimiere jetzt",
    "komprimiere den chat",
    "komprimier jetzt",
    "manual compression",
]
COMPRESSION_PAUSE_PATTERNS = [
    "keine kompression",
    "ohne kompression",
    "nicht komprimieren",
    "kompression aussetzen",
    "compression off",
    "disable compression",
    "skip compression",
    "no compression",
]
FAST_PATH_DENY_PATTERNS = [
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
    "fehl",
    "git",
    "image",
    "install",
    "kompression",
    "log",
    "memory",
    "mcp",
    "pc",
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
FAST_PATH_FORCE_PATTERNS = [
    "fast path",
    "ohne tools",
    "nur chat",
    "simple chat",
]
OPTIONAL_TOOL_MANIFEST = [
    {
        "name": "Pixelle MCP",
        "status": "lazy",
        "env_flag": "ALPHARAVIS_LOAD_MCP_TOOLS",
        "description": (
            "Optional Pixelle MCP registry for extra Pixelle/workflow/config tools. "
            "Native Pixelle image jobs still work through start_pixelle_remote without loading it."
        ),
    }
]

if not COMFY_IP:
    print("WARNING: 'comfy_server' IP not found in REMOTE_PCS env variable.")


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes"}


def _model(model_name: str | None = None, timeout_seconds: float | None = None) -> ChatLiteLLM:
    return ChatLiteLLM(
        model=model_name or os.getenv("ALPHARAVIS_MODEL", "openai/big-boss"),
        api_base=os.getenv("OPENAI_API_BASE", "http://litellm:4000/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "sk-local-dev"),
        request_timeout=timeout_seconds or float(os.getenv("ALPHARAVIS_LLM_TIMEOUT_SECONDS", "120")),
        max_retries=int(os.getenv("ALPHARAVIS_LLM_MAX_RETRIES", "0")),
        streaming=_env_bool("ALPHARAVIS_LLM_STREAMING", "true"),
    )


def _workspace_root() -> str:
    configured = os.getenv("ALPHARAVIS_WORKSPACE_DIR")
    if configured:
        return configured
    if Path("/workspace").exists():
        return "/workspace"
    return str(Path(__file__).resolve().parents[1])


def _configure_llm_cache() -> None:
    if os.getenv("ALPHARAVIS_ENABLE_REDIS_CACHE", "false").lower() not in {"1", "true", "yes"}:
        return

    redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
    try:
        set_llm_cache(RedisCache(redis_url=redis_url))
    except Exception as exc:
        print(f"WARNING: Redis LLM cache unavailable: {exc}")


def _warn_about_mongo_checkpointer() -> None:
    uri = os.getenv("LS_MONGODB_URI")
    if not uri:
        print(
            "WARNING: langgraph.json selects the Mongo checkpointer. "
            "Set LS_MONGODB_URI to a MongoDB replica-set URI with a database name."
        )
        return

    if not uri.startswith("mongodb+srv://") and "replicaSet=" not in uri:
        print(
            "WARNING: LS_MONGODB_URI should point to a MongoDB replica set "
            "for LangGraph Mongo checkpointing."
        )


@task
async def monitor_pixelle_job(job_id: str, original_thread_id: str) -> dict[str, str]:
    """Poll Pixelle until the job finishes, fails, or times out."""

    interval_seconds = float(os.getenv("PIXELLE_MONITOR_INTERVAL_SECONDS", "10"))
    max_polls = int(os.getenv("PIXELLE_MONITOR_MAX_POLLS", "180"))

    async with httpx.AsyncClient(timeout=30.0) as client:
        for attempt in range(1, max_polls + 1):
            if attempt > 1:
                await asyncio.sleep(interval_seconds)

            try:
                response = await client.get(f"{PIXELLE_URL}/api/status/{job_id}")
                response.raise_for_status()
                data = response.json()
            except Exception as exc:
                return {
                    "status": "monitor_error",
                    "job_id": job_id,
                    "thread_id": original_thread_id,
                    "message": f"Pixelle monitoring failed: {exc}",
                }

            status = data.get("status", "running")
            logs = data.get("logs", "No logs returned.")

            if status == "completed":
                return {
                    "status": "completed",
                    "job_id": job_id,
                    "thread_id": original_thread_id,
                    "message": data.get("result", ""),
                }

            if status == "failed":
                return {
                    "status": "failed",
                    "job_id": job_id,
                    "thread_id": original_thread_id,
                    "message": _format_pixelle_failure(job_id, logs),
                }

    return {
        "status": "timeout",
        "job_id": job_id,
        "thread_id": original_thread_id,
        "message": (
            f"Pixelle job `{job_id}` did not finish after "
            f"{max_polls} polls at {interval_seconds:g}s intervals."
        ),
    }


def _format_pixelle_failure(job_id: str, logs: str) -> str:
    return (
        f"CRITICAL ERROR: Pixelle job `{job_id}` failed.\n"
        "INSTRUCTION FOR DEBUGGING:\n"
        "1. Transfer to `debugger_agent` immediately.\n"
        "2. Pixelle runs as a local Docker container. Check "
        "`docker logs pixelle --tail 50`.\n"
        "3. Also check the LangGraph app logs: "
        "`docker logs langgraph-api --tail 50`.\n"
        "4. If a code error is found, present the proposed fix and wait for "
        "user approval before applying it.\n\n"
        f"Pixelle logs:\n{logs}"
    )


@tool
async def start_pixelle_remote(prompt: str, config: RunnableConfig):
    """Starts a Pixelle image job and monitors it through a durable LangGraph task."""

    current_thread_id = config["configurable"].get("thread_id", "default_thread")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(f"{PIXELLE_URL}/api/run", json={"prompt": prompt})
            response.raise_for_status()
            job_id = response.json().get("job_id")
        except Exception as exc:
            return f"Error: Could not reach Pixelle. ({exc})"

    if not job_id:
        return "Error: Pixelle did not return a job_id."

    result = await monitor_pixelle_job(job_id, current_thread_id)
    if result["status"] == "completed":
        return f"Image ready. Job `{job_id}` completed.\n\n{result['message']}"

    return result["message"]


@tool
async def start_pixelle_async(prompt: str):
    """Start a Pixelle image job and return immediately with a job id."""

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(f"{PIXELLE_URL}/api/run", json={"prompt": prompt})
            response.raise_for_status()
            job_id = response.json().get("job_id")
        except Exception as exc:
            return f"Error: Could not reach Pixelle. ({exc})"

    if not job_id:
        return "Error: Pixelle did not return a job_id."

    return (
        f"Pixelle job started. job_id: {job_id}\n"
        "Use check_pixelle_job with this exact job_id to get the current status. "
        "Do not poll automatically unless the user asks."
    )


@tool
async def check_pixelle_job(job_id: str):
    """Check the current status of a Pixelle image job."""

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{PIXELLE_URL}/api/status/{job_id.strip()}")
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            return f"Pixelle status check failed for `{job_id}`: {exc}"

    status = data.get("status", "running")
    if status == "completed":
        return f"Pixelle job `{job_id}` completed.\n\n{data.get('result', '')}"
    if status == "failed":
        return _format_pixelle_failure(job_id, data.get("logs", "No logs returned."))
    return f"Pixelle job `{job_id}` status: {status}\n\n{data.get('logs', '')}"


@tool
def wake_on_lan(pc_name: str):
    """Sends a magic packet to wake up a remote PC by its configured name."""

    from wakeonlan import send_magic_packet

    pc_info = REMOTE_PCS.get(pc_name)
    if not pc_info or "mac" not in pc_info:
        return f"Error: PC '{pc_name}' not found. Available: {list(REMOTE_PCS.keys())}"

    send_magic_packet(pc_info["mac"])
    return f"System: Magic Packet sent to {pc_name}."


@tool
def execute_ssh_command(pc_name: str, command: str):
    """Executes a shell command on a remote PC via SSH for diagnostics."""

    import subprocess

    pc_info = REMOTE_PCS.get(pc_name)
    if not pc_info or "ip" not in pc_info:
        return f"Error: PC '{pc_name}' not found. Available: {list(REMOTE_PCS.keys())}"

    approval = _require_command_approval("ssh", command, target=pc_name)
    if not approval["approved"]:
        return approval["message"]
    command = approval["command"]

    ssh_target = f"{SSH_USER}@{pc_info['ip']}"
    ssh_opts = ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"]
    ssh_pass = pc_info.get("ssh_pass", SSH_PASS_DEFAULT)

    if ssh_pass:
        cmd = ["sshpass", "-p", ssh_pass, "ssh"] + ssh_opts + [ssh_target, command]
    else:
        cmd = ["ssh"] + ssh_opts + [ssh_target, command]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        return f"Exit Code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    except subprocess.TimeoutExpired:
        return f"Error: SSH command timed out after 45s on '{pc_name}'."
    except Exception as exc:
        return f"SSH connection failed: {exc}"


@tool
def execute_local_command(command: str):
    """Executes a local diagnostic shell command for Docker, logs, or repo inspection."""

    import subprocess

    approval = _require_command_approval("local", command, target="langgraph-api")
    if not approval["approved"]:
        return approval["message"]

    try:
        result = subprocess.run(
            approval["command"],
            shell=True,
            capture_output=True,
            text=True,
            timeout=int(os.getenv("ALPHARAVIS_LOCAL_COMMAND_TIMEOUT_SECONDS", "45")),
        )
        return f"Exit Code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: local command timed out."
    except Exception as exc:
        return f"Local command failed: {exc}"


def _first_shell_word(command: str) -> str:
    try:
        parts = shlex.split(command, posix=True)
    except ValueError:
        return ""
    return parts[0] if parts else ""


def _command_segments(command: str) -> list[str]:
    return [segment.strip() for segment in re.split(r"\s*(?:&&|\|\||;|\|)\s*", command) if segment.strip()]


def _is_read_only_command(command: str) -> bool:
    command = command.strip()
    if not command:
        return False

    dangerous_patterns = [
        r"\b(rm|rmdir|mv|cp|chmod|chown|dd|mkfs|fdisk|parted|mount|umount|truncate|tee)\b",
        r"\b(kill|pkill|killall|reboot|shutdown|poweroff)\b",
        r"\b(apt|apt-get|apk|yum|dnf|pip|pip3|npm|pnpm|yarn)\s+(install|remove|uninstall|upgrade|update|add)\b",
        r"\b(git)\s+(push|commit|merge|rebase|reset|clean|checkout|switch|restore)\b",
        r"\b(docker)\s+(restart|stop|start|kill|rm|rmi|compose\s+(up|down|restart|stop|start|pull|build)|system\s+prune)\b",
        r"\b(systemctl|service)\s+(restart|stop|start|enable|disable|reload)\b",
        r"\b(pm2)\s+(restart|stop|start|delete|reload|save)\b",
        r"\bsed\s+-i\b",
        r"(^|[^<])>(?!>)|>>",
    ]
    lowered = command.lower()
    if any(re.search(pattern, lowered) for pattern in dangerous_patterns):
        return False

    safe_roots = {
        "awk",
        "cat",
        "curl",
        "date",
        "df",
        "docker",
        "du",
        "echo",
        "file",
        "find",
        "free",
        "git",
        "grep",
        "head",
        "hostname",
        "id",
        "journalctl",
        "less",
        "ls",
        "netstat",
        "pm2",
        "ps",
        "pwd",
        "rg",
        "service",
        "ss",
        "stat",
        "systemctl",
        "tail",
        "top",
        "uname",
        "uptime",
        "which",
        "whoami",
    }
    allowed_subcommands = {
        "docker": {"ps", "logs", "inspect", "version", "info", "stats", "compose"},
        "git": {"status", "diff", "log", "show", "branch", "remote", "rev-parse"},
        "pm2": {"list", "status", "logs", "show", "describe", "monit"},
        "service": {"status"},
        "systemctl": {"status", "is-active", "is-enabled", "list-units", "list-timers"},
    }

    for segment in _command_segments(command):
        root = _first_shell_word(segment)
        if root not in safe_roots:
            return False
        if root in allowed_subcommands:
            parts = shlex.split(segment, posix=True)
            subcommand = parts[1] if len(parts) > 1 else ""
            if root == "docker" and subcommand == "compose":
                compose_cmd = parts[2] if len(parts) > 2 else ""
                if compose_cmd not in {"ps", "logs", "config"}:
                    return False
            elif subcommand not in allowed_subcommands[root]:
                return False

    return True


def _require_command_approval(scope: str, command: str, *, target: str) -> dict[str, Any]:
    if os.getenv("ALPHARAVIS_REQUIRE_COMMAND_APPROVAL", "true").lower() not in {"1", "true", "yes"}:
        return {"approved": True, "command": command, "message": ""}

    if _is_read_only_command(command):
        return {"approved": True, "command": command, "message": ""}

    response = interrupt(
        {
            "type": "command_approval",
            "scope": scope,
            "target": target,
            "command": command,
            "risk": "This command can modify state, stop services, delete data, install packages, or is not clearly read-only.",
            "allowed_replies": [
                "approve",
                "reject",
                "replace: <safer command>",
            ],
        }
    )

    if isinstance(response, str):
        response = {"action": response}
    if not isinstance(response, dict):
        return {"approved": False, "command": command, "message": "Command rejected: invalid approval response."}

    action = str(response.get("action", "")).lower().strip()
    if action in {"approve", "approved", "yes", "ja", "genehmigt"}:
        return {"approved": True, "command": command, "message": ""}
    if action in {"replace", "change", "ersetzen", "ändern", "aendern"} and response.get("command"):
        replacement = str(response["command"]).strip()
        if not replacement:
            return {"approved": False, "command": command, "message": "Command rejected: empty replacement."}
        return {"approved": True, "command": replacement, "message": ""}

    return {"approved": False, "command": command, "message": "Command rejected by user approval gate."}


@tool
def fast_web_search(query: str):
    """ONLY for quick facts, weather, or simple questions using 1-2 sources."""

    search = DuckDuckGoSearchRun()
    return search.invoke(query)


@tool
async def deep_web_research(query: str):
    """Use for complex research, comparisons, or deep multi-source web searches."""

    search = TavilySearchResults(max_results=10)
    return await search.ainvoke({"query": query})


@tool
async def ask_documents(query: str):
    """Search local uploaded documents through the RAG API."""

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get("http://rag_api:8000/query", params={"text": query})
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            return f"Document search failed: {exc}"


@tool
def read_alpha_ravis_architecture(query: str = "", max_chars: int = 6000):
    """Read the editable AlphaRavis architecture/capabilities document on demand."""

    configured_path = os.getenv("ALPHARAVIS_ARCHITECTURE_DOC_PATH")
    if configured_path:
        doc_path = Path(configured_path)
    else:
        doc_path = Path(_workspace_root()) / "docs" / "ALPHARAVIS_ARCHITECTURE.md"

    try:
        resolved = doc_path.resolve()
        workspace = Path(_workspace_root()).resolve()
        if workspace not in [resolved, *resolved.parents]:
            return f"Architecture document path is outside the workspace: {resolved}"
        content = resolved.read_text(encoding="utf-8")
    except Exception as exc:
        return f"Could not read AlphaRavis architecture document: {exc}"

    max_allowed = int(os.getenv("ALPHARAVIS_ARCHITECTURE_DOC_MAX_CHARS", "6000"))
    max_chars = max(1000, min(int(max_chars), max_allowed))

    if query:
        lowered_terms = [term for term in re.split(r"\W+", query.lower()) if len(term) >= 4]
        sections = re.split(r"(?m)^## ", content)
        matches = []
        for section in sections:
            haystack = section.lower()
            if any(term in haystack for term in lowered_terms):
                prefix = "" if section.startswith("#") else "## "
                matches.append(prefix + section.strip())
        if matches:
            content = "\n\n".join(matches)

    if len(content) > max_chars:
        return (
            content[:max_chars].rstrip()
            + "\n\n[Truncated. Ask for a narrower AlphaRavis architecture topic if more detail is needed.]"
        )
    return content


@tool
def list_repo_ai_skills(max_chars: int = 4000):
    """List reviewed repo skill cards available under ai-skills/."""

    skills = _list_repo_ai_skill_metadata()
    if isinstance(skills, str):
        return skills
    if not skills:
        return "No valid repo AI skill cards found."

    output = "\n".join(
        f"- {skill['name']}: {skill['description']}\n  Path: {skill['path']}" for skill in skills
    )
    max_chars = max(1000, min(int(max_chars), 8000))
    return output[:max_chars].rstrip()


def _list_repo_ai_skill_metadata() -> list[dict[str, str]] | str:
    skills_dir = Path(_workspace_root()) / "ai-skills"
    try:
        workspace = Path(_workspace_root()).resolve()
        resolved = skills_dir.resolve()
        if workspace not in [resolved, *resolved.parents]:
            return f"AI skills path is outside the workspace: {resolved}"
        if not resolved.exists():
            return []
    except Exception as exc:
        return f"Could not inspect repo AI skills: {exc}"

    skills = []
    for skill_md in sorted(resolved.glob("*/SKILL.md")):
        try:
            text = skill_md.read_text(encoding="utf-8")
        except Exception:
            continue
        name_match = re.search(r"(?m)^name:\s*(.+?)\s*$", text)
        desc_match = re.search(r"(?ms)^description:\s*(.+?)\n---", text)
        name = (name_match.group(1).strip().strip('"') if name_match else skill_md.parent.name)
        description = " ".join((desc_match.group(1).strip().strip('"') if desc_match else "").split())
        skills.append(
            {
                "name": name,
                "description": description,
                "path": f"ai-skills/{skill_md.parent.name}/SKILL.md",
            }
        )
    return skills


def _repo_skill_hint_context(query: str, limit: int) -> str:
    skills = _list_repo_ai_skill_metadata()
    if isinstance(skills, str) or not skills:
        return ""

    query_terms = {term for term in re.split(r"[^a-zA-Z0-9]+", query.lower()) if len(term) >= 4}
    scored: list[tuple[int, dict[str, str]]] = []
    for skill in skills:
        haystack = f"{skill['name']} {skill['description']}".lower()
        score = sum(1 for term in query_terms if term in haystack)
        if score:
            scored.append((score, skill))

    scored.sort(key=lambda item: (-item[0], item[1]["name"]))
    selected = [skill for _, skill in scored[: max(1, limit)]]
    if not selected:
        return ""

    lines = [
        "Reviewed repo AI skill cards may match this task. They are metadata hints only; "
        "read the full card with read_repo_ai_skill only if needed."
    ]
    lines.extend(f"- {skill['name']}: {skill['description']}" for skill in selected)
    return "\n".join(lines)


@tool
def read_repo_ai_skill(skill_name: str, reference_name: str = "", max_chars: int = 8000):
    """Read one reviewed repo AI skill card or one of its markdown references."""

    normalized = re.sub(r"[^a-z0-9-]+", "-", skill_name.lower()).strip("-")
    if not normalized:
        return "Provide a skill_name such as `deepagents-agent-builder`."

    base_dir = Path(_workspace_root()) / "ai-skills" / normalized
    if reference_name:
        safe_reference = Path(reference_name).name
        if not safe_reference.endswith(".md"):
            safe_reference = f"{safe_reference}.md"
        target = base_dir / "references" / safe_reference
    else:
        target = base_dir / "SKILL.md"

    try:
        workspace = Path(_workspace_root()).resolve()
        resolved = target.resolve()
        allowed_root = base_dir.resolve()
        if workspace not in [resolved, *resolved.parents]:
            return f"Skill path is outside the workspace: {resolved}"
        if allowed_root not in [resolved, *resolved.parents]:
            return f"Skill path is outside the requested skill directory: {resolved}"
        content = resolved.read_text(encoding="utf-8")
    except Exception as exc:
        return f"Could not read repo AI skill `{normalized}`: {exc}"

    max_chars = max(1000, min(int(max_chars), 16000))
    if len(content) > max_chars:
        return content[:max_chars].rstrip() + "\n\n[Truncated. Ask for a narrower skill reference if needed.]"
    return content


@tool
def normalize_research_sources(source_notes: str, max_sources: int = 20):
    """Extract unique URLs from research notes and return stable citation numbers."""

    urls = []
    seen = set()
    for match in re.finditer(r"https?://[^\s\]\)>,]+", source_notes):
        url = match.group(0).rstrip(".,;:")
        if url not in seen:
            seen.add(url)
            urls.append(url)

    max_sources = max(1, min(int(max_sources), 50))
    if not urls:
        return "No URLs found. Do not invent citations; mark unsupported claims as uncited or search again."

    lines = ["Use these stable citation numbers for this answer:"]
    for index, url in enumerate(urls[:max_sources], start=1):
        lines.append(f"[{index}] {url}")
    if len(urls) > max_sources:
        lines.append(f"[Truncated {len(urls) - max_sources} additional URLs.]")
    return "\n".join(lines)


@tool
def build_specialist_report(
    agent_id: str,
    summary: str,
    evidence: str = "",
    sources: str = "",
    commands_run: str = "",
    risks: str = "",
    next_actions: str = "",
):
    """Format a specialist handoff report with stable fields."""

    report = {
        "agent_id": agent_id,
        "summary": summary,
        "evidence": evidence,
        "sources": sources,
        "commands_run": commands_run,
        "risks": risks,
        "next_actions": next_actions,
    }
    return json.dumps(report, ensure_ascii=False, indent=2)


@tool
async def search_archived_context(query: str, limit: int = 5, include_other_threads: bool = False):
    """Search archived memory. Defaults to the current chat thread only."""

    if get_store is None:
        return "LangGraph store access is unavailable in this runtime."

    try:
        store = get_store()
    except Exception as exc:
        return f"No LangGraph store is attached to this run: {exc}"

    limit = max(1, min(int(limit), int(os.getenv("ALPHARAVIS_ARCHIVE_SEARCH_LIMIT", "5"))))
    thread_id = _state_thread_id()
    if include_other_threads:
        limit = min(limit, int(os.getenv("ALPHARAVIS_CROSS_THREAD_ARCHIVE_SEARCH_LIMIT", "3")))
        namespaces = [
            (ARCHIVE_INDEX_NS, "Cross-thread archive"),
            (ARCHIVE_COLLECTION_INDEX_NS, "Cross-thread archive collection"),
        ]
    else:
        namespaces = [
            (_thread_archive_ns(thread_id), "Thread archive"),
            (_thread_archive_collection_ns(thread_id), "Thread archive collection"),
        ]

    records: list[tuple[str, Any]] = []
    for namespace, label in namespaces:
        try:
            results = await _maybe_search(store, namespace, query=query, limit=limit)
        except Exception as exc:
            return f"{label} search failed: {exc}"

        for item in results or []:
            records.append((label, item))

    if not records:
        if include_other_threads:
            return "No archived context matched that query across threads."
        return "No archived context matched that query in the current chat thread."

    lines = []
    for label, item in records[:limit]:
        key = _store_item_key(item)
        value = _store_item_value(item)
        if isinstance(value, dict):
            summary = value.get("summary") or value.get("content") or str(value)
            token_estimate = value.get("token_estimate", "unknown")
            source_thread = value.get("thread_key") or value.get("thread_id") or "unknown"
            lines.append(
                f"{label} `{key}` from `{source_thread}` ({token_estimate} tokens est.):\n{summary}"
            )
        else:
            lines.append(f"{label} `{key}`:\n{value}")

    return "\n\n".join(lines)


@tool
async def search_debugging_lessons(query: str, limit: int = 5):
    """Search lessons learned from past debugging failures and successful fixes."""

    if get_store is None:
        return "LangGraph store access is unavailable in this runtime."

    try:
        store = get_store()
    except Exception as exc:
        return f"No LangGraph store is attached to this run: {exc}"

    try:
        results = await _maybe_search(store, DEBUGGING_LESSON_NS, query=query, limit=limit)
    except Exception as exc:
        return f"Lesson search failed: {exc}"

    if not results:
        return "No previous debugging lessons matched that query."

    lines = []
    for item in results:
        value = getattr(item, "value", item)
        if isinstance(value, dict):
            lines.append(
                "\n".join(
                    [
                        f"Problem: {value.get('problem', 'unknown')}",
                        f"Root cause: {value.get('root_cause', 'unknown')}",
                        f"Fix: {value.get('fix', 'unknown')}",
                        f"Signals: {value.get('signals', 'unknown')}",
                    ]
                )
            )
        else:
            lines.append(str(value))
    return "\n\n".join(lines)


@tool
async def record_debugging_lesson(
    problem: str,
    root_cause: str,
    fix: str,
    signals: str = "",
    commands: str = "",
    outcome: str = "",
):
    """Store a durable lesson after a debugging issue is understood or fixed."""

    if get_store is None:
        return "LangGraph store access is unavailable in this runtime."

    try:
        store = get_store()
    except Exception as exc:
        return f"No LangGraph store is attached to this run: {exc}"

    lesson = {
        "problem": problem,
        "root_cause": root_cause,
        "fix": fix,
        "signals": signals,
        "commands": commands,
        "outcome": outcome,
        "created_at": int(time.time()),
    }
    key = hashlib.sha256(json.dumps(lesson, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    await _maybe_put(store, DEBUGGING_LESSON_NS, key, lesson)
    return f"Stored debugging lesson `{key}`."


@tool
def describe_optional_tool_registry():
    """Describe optional lazy-loaded tool registries without loading them."""

    lines = [
        "Optional lazy tool registries known to AlphaRavis:",
    ]
    for entry in OPTIONAL_TOOL_MANIFEST:
        enabled = _env_bool(entry["env_flag"], "false")
        lines.append(
            "\n".join(
                [
                    f"- {entry['name']} ({'enabled' if enabled else 'disabled/lazy'})",
                    f"  Env: {entry['env_flag']}={'true' if enabled else 'false'}",
                    f"  Use: {entry['description']}",
                ]
            )
        )
    return "\n\n".join(lines)


@tool
async def search_agent_memory(agent_id: str, query: str, limit: int = 5, include_global: bool = True):
    """Search durable memories for one agent, optionally including global memories."""

    if get_store is None:
        return "LangGraph store access is unavailable in this runtime."

    try:
        store = get_store()
    except Exception as exc:
        return f"No LangGraph store is attached to this run: {exc}"

    agent_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", agent_id.strip().lower())[:80] or "global"
    limit = max(1, min(int(limit), 10))
    namespaces = [
        (("alpharavis", "agent_memories", agent_id), f"Agent memory `{agent_id}`"),
    ]
    if include_global and agent_id != "global":
        namespaces.append((("alpharavis", "agent_memories", "global"), "Global agent memory"))

    lines = []
    for namespace, label in namespaces:
        try:
            results = await _maybe_search(store, namespace, query=query, limit=limit)
        except Exception as exc:
            return f"{label} search failed: {exc}"

        for item in results or []:
            key = _store_item_key(item)
            value = _store_item_value(item)
            if isinstance(value, dict):
                lines.append(
                    "\n".join(
                        [
                            f"{label} `{key}`:",
                            f"Type: {value.get('memory_type', 'note')}",
                            f"Memory: {value.get('memory', '')}",
                            f"Evidence: {value.get('evidence', '')}",
                        ]
                    )
                )
            elif value:
                lines.append(f"{label} `{key}`:\n{value}")

    if not lines:
        return f"No agent memories matched `{query}` for `{agent_id}`."
    return "\n\n".join(lines[:limit])


@tool
async def record_agent_memory(
    agent_id: str,
    memory: str,
    memory_type: str = "lesson",
    evidence: str = "",
    scope: str = "agent",
):
    """Store a durable agent-specific or global memory after a useful lesson is confirmed."""

    if get_store is None:
        return "LangGraph store access is unavailable in this runtime."

    try:
        store = get_store()
    except Exception as exc:
        return f"No LangGraph store is attached to this run: {exc}"

    target_agent = "global" if scope.lower().strip() == "global" else agent_id
    target_agent = re.sub(r"[^a-zA-Z0-9_-]+", "_", target_agent.strip().lower())[:80] or "global"
    record = {
        "agent_id": target_agent,
        "memory": memory.strip()[:2500],
        "memory_type": memory_type.strip()[:80] or "lesson",
        "evidence": evidence.strip()[:1500],
        "scope": "global" if target_agent == "global" else "agent",
        "created_at": int(time.time()),
    }
    key = hashlib.sha256(json.dumps(record, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    await _maybe_put(store, ("alpharavis", "agent_memories", target_agent), key, record)
    return f"Stored {record['scope']} memory `{key}` for `{target_agent}`."


@tool
async def search_skill_library(query: str, limit: int = 5, include_candidates: bool = False):
    """Search approved workflow skills, optionally including inactive candidates."""

    if get_store is None:
        return "LangGraph store access is unavailable in this runtime."

    try:
        store = get_store()
    except Exception as exc:
        return f"No LangGraph store is attached to this run: {exc}"

    try:
        results = await _maybe_search(store, SKILL_LIBRARY_NS, query=query, limit=limit)
    except Exception as exc:
        return f"Skill-library search failed: {exc}"

    lines = []
    for item in results or []:
        key = _store_item_key(item)
        value = _store_item_value(item)
        if not isinstance(value, dict):
            continue
        status = value.get("status", "candidate")
        if not include_candidates and status != "active":
            continue
        lines.append(_format_skill_record(key, value))

    if lines:
        return "\n\n".join(lines)

    if include_candidates:
        return "No skill-library records matched that query."
    return "No approved active skills matched that query."


@tool
async def record_skill_candidate(
    name: str,
    trigger: str,
    steps: str,
    success_signals: str = "",
    safety_notes: str = "",
    evidence: str = "",
    source_task: str = "",
    confidence: float = 0.5,
):
    """Store a reusable workflow as an inactive skill candidate for later human review."""

    if get_store is None:
        return "LangGraph store access is unavailable in this runtime."

    try:
        store = get_store()
    except Exception as exc:
        return f"No LangGraph store is attached to this run: {exc}"

    confidence = max(0.0, min(1.0, float(confidence)))
    skill = {
        "name": name.strip()[:160],
        "trigger": trigger.strip()[:1200],
        "steps": steps.strip()[:4000],
        "success_signals": success_signals.strip()[:1200],
        "safety_notes": safety_notes.strip()[:1200],
        "evidence": evidence.strip()[:2000],
        "source_task": source_task.strip()[:1200],
        "confidence": confidence,
        "status": "candidate",
        "active": False,
        "human_approval_required": True,
        "created_at": int(time.time()),
    }
    key = hashlib.sha256(
        json.dumps(
            {
                "name": skill["name"],
                "trigger": skill["trigger"],
                "steps": skill["steps"],
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:24]
    await _maybe_put(store, SKILL_LIBRARY_NS, key, skill)
    return (
        f"Stored inactive skill candidate `{key}`. It will not affect routing "
        "until a human promotes it to active."
    )


@tool
async def list_skill_candidates(query: str = "", limit: int = 20, include_active: bool = True):
    """List workflow skill candidates for human review."""

    if get_store is None:
        return "LangGraph store access is unavailable in this runtime."

    try:
        store = get_store()
    except Exception as exc:
        return f"No LangGraph store is attached to this run: {exc}"

    limit = max(1, min(int(limit), 50))
    try:
        results = await _maybe_search(store, SKILL_LIBRARY_NS, query=query, limit=limit)
    except Exception as exc:
        return f"Skill-library listing failed: {exc}"

    lines = []
    for item in results or []:
        key = _store_item_key(item)
        value = _store_item_value(item)
        if not isinstance(value, dict):
            continue
        if not include_active and value.get("status") == "active":
            continue
        lines.append(_format_skill_record(key, value))

    if not lines:
        return "No skill candidates matched that review query."
    return "\n\n".join(lines)


@tool
async def activate_skill_candidate(skill_id: str, approval_note: str = ""):
    """Promote a reviewed skill candidate to active when promotion is explicitly enabled."""

    if not _env_bool("ALPHARAVIS_ALLOW_SKILL_PROMOTION", "false"):
        return (
            "Skill promotion is disabled for safety. Set "
            "ALPHARAVIS_ALLOW_SKILL_PROMOTION=true only while intentionally "
            "promoting reviewed candidates."
        )

    if get_store is None:
        return "LangGraph store access is unavailable in this runtime."

    try:
        store = get_store()
    except Exception as exc:
        return f"No LangGraph store is attached to this run: {exc}"

    item = await _maybe_get(store, SKILL_LIBRARY_NS, skill_id)
    value = _store_item_value(item)
    if not isinstance(value, dict):
        return f"Skill candidate `{skill_id}` was not found."

    value = dict(value)
    value["status"] = "active"
    value["active"] = True
    value["approved_at"] = int(time.time())
    value["approval_note"] = approval_note.strip()[:1200]
    await _maybe_put(store, SKILL_LIBRARY_NS, skill_id, value)
    return f"Activated skill `{skill_id}`."


@tool
async def deactivate_skill(skill_id: str, reason: str = ""):
    """Deactivate an active workflow skill during explicit review mode."""

    if not _env_bool("ALPHARAVIS_ALLOW_SKILL_PROMOTION", "false"):
        return (
            "Skill activation/deactivation is disabled for safety. Set "
            "ALPHARAVIS_ALLOW_SKILL_PROMOTION=true only while intentionally "
            "reviewing skills."
        )

    if get_store is None:
        return "LangGraph store access is unavailable in this runtime."

    try:
        store = get_store()
    except Exception as exc:
        return f"No LangGraph store is attached to this run: {exc}"

    item = await _maybe_get(store, SKILL_LIBRARY_NS, skill_id)
    value = _store_item_value(item)
    if not isinstance(value, dict):
        return f"Skill `{skill_id}` was not found."

    value = dict(value)
    value["status"] = "candidate"
    value["active"] = False
    value["deactivated_at"] = int(time.time())
    value["deactivation_reason"] = reason.strip()[:1200]
    await _maybe_put(store, SKILL_LIBRARY_NS, skill_id, value)
    return f"Deactivated skill `{skill_id}`."


async def _load_pixelle_mcp_tools(stack: contextlib.AsyncExitStack) -> list[Any]:
    try:
        session_manager = sse_client(f"{PIXELLE_URL}/pixelle/mcp/sse")
        streams = await stack.enter_async_context(session_manager)
        session = ClientSession(*streams)
        await session.initialize()
        tools = await load_mcp_tools(session)
        print(f"Loaded {len(tools)} Pixelle MCP tools.")
        return list(tools)
    except Exception as exc:
        print(f"WARNING: Pixelle MCP tools unavailable: {exc}")
        return []


def _message_text(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content", "")
        role = message.get("role") or message.get("type") or "message"
    else:
        content = getattr(message, "content", "")
        role = getattr(message, "type", getattr(message, "role", "message"))

    if isinstance(content, list):
        content = " ".join(str(block) for block in content)

    return f"{role}: {content}"


def _message_to_json(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return dict(message)

    return {
        "type": getattr(message, "type", None),
        "name": getattr(message, "name", None),
        "id": getattr(message, "id", None),
        "content": getattr(message, "content", ""),
        "additional_kwargs": getattr(message, "additional_kwargs", {}),
        "response_metadata": getattr(message, "response_metadata", {}),
    }


def _estimate_tokens(messages: list[Any]) -> int:
    text = "\n".join(_message_text(message) for message in messages)
    return max(1, len(text) // 4)


async def _maybe_put(store: Any, namespace: tuple[str, ...], key: str, value: dict[str, Any]) -> None:
    if hasattr(store, "aput"):
        result = store.aput(namespace, key, value)
    else:
        result = store.put(namespace, key, value)
    if inspect.isawaitable(result):
        await result


async def _maybe_get(store: Any, namespace: tuple[str, ...], key: str) -> Any:
    if hasattr(store, "aget"):
        result = store.aget(namespace, key)
    elif hasattr(store, "get"):
        result = store.get(namespace, key)
    else:
        return None
    if inspect.isawaitable(result):
        result = await result
    return result


async def _maybe_search(store: Any, namespace: tuple[str, ...], *, query: str, limit: int) -> Any:
    if hasattr(store, "asearch"):
        result = store.asearch(namespace, query=query, limit=limit)
    else:
        result = store.search(namespace, query=query, limit=limit)
    if inspect.isawaitable(result):
        result = await result
    return result


def _store_item_value(item: Any) -> Any:
    if item is None:
        return None
    return getattr(item, "value", item)


def _store_item_key(item: Any) -> str:
    key = getattr(item, "key", None)
    if key is not None:
        return str(key)
    if isinstance(item, dict):
        return str(item.get("key") or item.get("id") or "unknown")
    return "unknown"


def _format_skill_record(key: str, value: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Skill `{key}` ({value.get('status', 'candidate')}): {value.get('name', 'unnamed')}",
            f"Trigger: {value.get('trigger', '')}",
            f"Steps: {value.get('steps', '')}",
            f"Success signals: {value.get('success_signals', '')}",
            f"Safety: {value.get('safety_notes', '')}",
        ]
    ).strip()


def _thread_id_from_config() -> str | None:
    if get_config is None:
        return None

    try:
        config = get_config()
    except Exception:
        return None

    if not isinstance(config, dict):
        return None

    configurable = config.get("configurable")
    metadata = config.get("metadata")
    for source in [configurable, metadata, config]:
        if isinstance(source, dict):
            for key in ["thread_id", "thread_key", "conversation_id", "conversationId"]:
                value = source.get(key)
                if value:
                    return str(value)
    return None


def _state_thread_id(state: dict[str, Any] | None = None) -> str:
    if state:
        for key in ["thread_id", "thread_key"]:
            value = state.get(key)
            if value:
                return str(value)
    return _thread_id_from_config() or "global"


def _state_thread_key(state: dict[str, Any] | None = None) -> str:
    if state and state.get("thread_key"):
        return str(state["thread_key"])
    return _state_thread_id(state)


def _thread_archive_ns(thread_id: str) -> tuple[str, ...]:
    return ("alpharavis", "threads", thread_id, "archives")


def _thread_archive_collection_ns(thread_id: str) -> tuple[str, ...]:
    return ("alpharavis", "threads", thread_id, "archive_collections")


def _latest_user_query(messages: list[Any]) -> str:
    for message in reversed(messages):
        role = None
        if isinstance(message, dict):
            role = message.get("role") or message.get("type")
        else:
            role = getattr(message, "type", getattr(message, "role", None))

        if role in {"human", "user"}:
            return _message_text(message)

    return "\n".join(_message_text(message) for message in messages[-4:])


def _compression_paused_by_user(messages: list[Any]) -> bool:
    if not _env_bool("ALPHARAVIS_ALLOW_USER_COMPRESSION_PAUSE", "true"):
        return False

    latest = _latest_user_query(messages).lower()
    return any(pattern in latest for pattern in COMPRESSION_PAUSE_PATTERNS)


def _compression_forced_by_user(messages: list[Any]) -> bool:
    latest = _latest_user_query(messages).lower()
    configured = os.getenv("ALPHARAVIS_MANUAL_COMPRESSION_PATTERNS", "")
    patterns = [
        pattern.strip().lower()
        for pattern in configured.split("|")
        if pattern.strip()
    ] or MANUAL_COMPRESSION_PATTERNS
    return any(pattern in latest for pattern in patterns)


def _profile_update(state: AlphaRavisState, **updates: Any) -> dict[str, Any]:
    profile = dict(state.get("run_profile") or {})
    profile.update(updates)
    return profile


def _fast_path_decision(state: AlphaRavisState) -> tuple[bool, str]:
    if not _env_bool("ALPHARAVIS_ENABLE_FAST_PATH", "true"):
        return False, "fast path disabled"

    if _env_bool("ALPHARAVIS_FAST_PATH_LOCK_AFTER_SWARM", "true") and state.get("fast_path_locked"):
        return False, f"thread already used agent path: {state.get('fast_path_lock_reason', 'locked')}"

    messages = list(state.get("messages", []))
    query = _latest_user_query(messages).strip()
    if not query:
        return False, "no user query"

    lowered = query.lower()
    if "kein fast path" in lowered or "no fast path" in lowered:
        return False, "user disabled fast path"

    deny_hits = [pattern for pattern in FAST_PATH_DENY_PATTERNS if pattern in lowered]
    if deny_hits:
        return False, f"agent/tool keyword: {deny_hits[0]}"

    max_chars = int(os.getenv("ALPHARAVIS_FAST_PATH_MAX_CHARS", "360"))
    if len(query) > max_chars:
        return False, f"query too long: {len(query)} chars"

    if any(pattern in lowered for pattern in FAST_PATH_FORCE_PATTERNS):
        return True, "explicit simple chat request"

    return True, "short non-tool chat"


async def run_profile_start_node(state: AlphaRavisState) -> dict[str, Any]:
    messages = list(state.get("messages", []))
    return {
        "run_profile": {
            "started_at": time.time(),
            "latest_user_chars": len(_latest_user_query(messages)),
            "message_count": len(messages),
            "token_estimate": _estimate_tokens(messages),
        }
    }


async def route_decision_node(state: AlphaRavisState) -> dict[str, Any]:
    use_fast_path, reason = _fast_path_decision(state)
    route = "fast_path" if use_fast_path else "swarm"
    lock_thread = (
        route == "swarm"
        and _env_bool("ALPHARAVIS_FAST_PATH_LOCK_AFTER_SWARM", "true")
        and reason != "fast path disabled"
    )
    updates: dict[str, Any] = {
        "fast_path_route": route,
        "run_profile": _profile_update(
            state,
            route=route,
            route_reason=reason,
            fast_path_locked=bool(state.get("fast_path_locked") or lock_thread),
            route_decided_at=time.time(),
        ),
    }
    if lock_thread:
        updates["fast_path_locked"] = True
        updates["fast_path_lock_reason"] = reason
    elif state.get("fast_path_locked"):
        updates["fast_path_locked"] = True
        updates["fast_path_lock_reason"] = state.get("fast_path_lock_reason", reason)
    return updates


def route_after_decision(state: AlphaRavisState) -> str:
    return "fast_path" if state.get("fast_path_route") == "fast_path" else "swarm"


def _fast_path_bind_kwargs(*, allow_chat_template_kwargs: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_tokens": int(os.getenv("ALPHARAVIS_FAST_PATH_MAX_TOKENS", "256")),
        "temperature": float(os.getenv("ALPHARAVIS_FAST_PATH_TEMPERATURE", "0")),
    }
    if allow_chat_template_kwargs and _env_bool("ALPHARAVIS_FAST_PATH_DISABLE_THINKING", "true"):
        kwargs["chat_template_kwargs"] = {"enable_thinking": False}
    return kwargs


async def fast_chat_node(state: AlphaRavisState) -> dict[str, Any]:
    messages = list(state.get("messages", []))
    prompt = SystemMessage(
        content=(
            "You are AlphaRavis in direct fast-chat mode. Answer the user "
            "normally and concisely. Do not claim to browse, inspect files, "
            "use tools, control PCs, or access archives in this mode. If the "
            "request requires tools, say it needs the agent path."
        )
    )
    started = time.time()
    primary_model = os.getenv("ALPHARAVIS_MODEL", "openai/big-boss")
    used_model = primary_model
    fallback_used = False
    fallback_error = ""

    try:
        primary_llm = _model(
            model_name=primary_model,
            timeout_seconds=float(os.getenv("ALPHARAVIS_FAST_PATH_PRIMARY_TIMEOUT_SECONDS", "20")),
        ).bind(**_fast_path_bind_kwargs(allow_chat_template_kwargs=True))
        response = await primary_llm.ainvoke([prompt, *messages])
    except Exception as exc:
        fallback_error = str(exc)
        fallback_model = os.getenv("ALPHARAVIS_FAST_PATH_FALLBACK_MODEL", "openai/edge-gemma")
        if not _env_bool("ALPHARAVIS_FAST_PATH_ENABLE_FALLBACK", "true") or not fallback_model:
            raise
        fallback_used = True
        used_model = fallback_model
        fallback_llm = _model(
            model_name=fallback_model,
            timeout_seconds=float(os.getenv("ALPHARAVIS_FAST_PATH_FALLBACK_TIMEOUT_SECONDS", "45")),
        ).bind(**_fast_path_bind_kwargs(allow_chat_template_kwargs=False))
        response = await fallback_llm.ainvoke([prompt, *messages])

    response_content = getattr(response, "content", "")
    if isinstance(response_content, list):
        response_content = " ".join(str(block) for block in response_content)
    if not str(response_content).strip():
        response = AIMessage(
            content=(
                "Der schnelle Chat-Pfad hat keine finale Modellantwort erhalten. "
                "Bitte wiederhole die Anfrage mit `kein fast path`, falls der "
                "Agentenpfad genutzt werden soll."
            )
        )
        response_content = response.content

    if _env_bool("ALPHARAVIS_SHOW_FAST_PATH_NOTICE", "true"):
        model_label = f"{used_model} fallback" if fallback_used else f"{used_model} direct"
        lock_text = (
            "Dieser Thread bleibt im Fast Path, bis er einmal den Agentenpfad benutzt."
            if _env_bool("ALPHARAVIS_FAST_PATH_LOCK_AFTER_SWARM", "true")
            else "Thread-Lock nach Agentenpfad ist deaktiviert."
        )
        response = AIMessage(
            content=(
                f"Fast-Path aktiv ({model_label}). {lock_text}\n\n"
                f"{str(response_content).strip()}"
            )
        )

    profile_updates: dict[str, Any] = {
        "route": "fast_path",
        "fast_path_model": used_model,
        "fast_path_fallback_used": fallback_used,
        "fast_path_notice_shown": _env_bool("ALPHARAVIS_SHOW_FAST_PATH_NOTICE", "true"),
        "fast_path_seconds": round(time.time() - started, 3),
    }
    if fallback_error:
        profile_updates["fast_path_primary_error"] = fallback_error[:300]

    return {
        "messages": [response],
        "run_profile": _profile_update(state, **profile_updates),
    }


async def _summarize_messages(
    llm: ChatLiteLLM,
    messages: list[Any],
    existing_summary: str | None,
) -> str:
    history = "\n".join(_message_text(message) for message in messages)
    previous = existing_summary or "No previous summary."
    prompt = (
        "Summarize this conversation history for future retrieval. Preserve "
        "user preferences, unresolved tasks, exact technical facts, file paths, "
        "commands, error messages, decisions, and pending approvals. Keep it "
        "compact but specific.\n\n"
        f"Previous summary:\n{previous}\n\n"
        f"History to archive:\n{history}"
    )
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    return str(response.content)


async def _summarize_archive_records(
    llm: ChatLiteLLM,
    records: list[tuple[str, dict[str, Any]]],
    existing_summary: str | None,
) -> str:
    previous = existing_summary or "No previous archive summary."
    archive_text = "\n\n".join(
        [
            "\n".join(
                [
                    f"Archive key: {key}",
                    f"Token estimate: {value.get('token_estimate', 'unknown')}",
                    f"Summary: {value.get('summary', '')}",
                ]
            )
            for key, value in records
        ]
    )
    prompt = (
        "Create a higher-level memory summary from these conversation archives. "
        "Preserve durable facts, user preferences, recurring tasks, unresolved "
        "decisions, reusable workflow patterns, file paths, commands, errors, "
        "and references to archive keys when exact raw history may be needed. "
        "Keep this compact enough to retrieve later.\n\n"
        f"Previous archive summary:\n{previous}\n\n"
        f"Archives to compress:\n{archive_text}"
    )
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    return str(response.content)


async def _maybe_compact_archives(
    store: Any,
    thread_id: str,
    thread_key: str,
    archived_keys: list[str],
    compressed_keys: list[str],
    collection_keys: list[str],
    existing_summary: str | None,
) -> dict[str, Any]:
    if os.getenv("ALPHARAVIS_ENABLE_HIERARCHICAL_COMPRESSION", "true").lower() not in {
        "1",
        "true",
        "yes",
    }:
        return {}

    pending_keys = [key for key in archived_keys if key not in set(compressed_keys)]
    if not pending_keys:
        return {}

    records: list[tuple[str, dict[str, Any]]] = []
    archive_ns = _thread_archive_ns(thread_id)
    for key in pending_keys:
        item = await _maybe_get(store, archive_ns, key)
        value = _store_item_value(item)
        if isinstance(value, dict):
            records.append((key, value))

    archive_limit = int(os.getenv("ALPHARAVIS_ARCHIVE_TOKEN_LIMIT", "50000"))
    pending_tokens = sum(int(value.get("token_estimate") or 0) for _, value in records)
    if pending_tokens <= archive_limit:
        return {}

    keep_recent = int(os.getenv("ALPHARAVIS_ARCHIVE_KEEP_RECENT_RECORDS", "8"))
    if len(records) <= keep_recent:
        return {}

    records_to_compact = records[:-keep_recent]
    if not records_to_compact:
        return {}

    summary = await _summarize_archive_records(_model(), records_to_compact, existing_summary)
    compacted_keys = [key for key, _ in records_to_compact]
    token_estimate = sum(int(value.get("token_estimate") or 0) for _, value in records_to_compact)
    collection_key = hashlib.sha256(
        f"{time.time()}:{summary}:{','.join(compacted_keys)}".encode("utf-8")
    ).hexdigest()[:24]
    collection_record = {
        "summary": summary,
        "child_archive_keys": compacted_keys,
        "token_estimate": token_estimate,
        "record_count": len(records_to_compact),
        "compressed_at": int(time.time()),
        "thread_id": thread_id,
        "thread_key": thread_key,
    }
    await _maybe_put(store, _thread_archive_collection_ns(thread_id), collection_key, collection_record)
    await _maybe_put(store, ARCHIVE_COLLECTION_INDEX_NS, collection_key, collection_record)

    return {
        "archive_summary": summary,
        "archive_collection_keys": [*collection_keys, collection_key],
        "compressed_archive_keys": [*compressed_keys, *compacted_keys],
        "archive_compression_notice": (
            f"Zusätzlich wurden {len(records_to_compact)} ältere Archivblöcke "
            f"zu einer Hierarchie-Zusammenfassung `{collection_key}` verdichtet."
        ),
    }


async def skill_library_node(state: AlphaRavisState, runtime: Any | None = None) -> dict[str, Any]:
    if os.getenv("ALPHARAVIS_ENABLE_SKILL_LIBRARY", "true").lower() not in {"1", "true", "yes"}:
        return {}

    messages = list(state.get("messages", []))
    query = _latest_user_query(messages)
    repo_hint_limit = int(os.getenv("ALPHARAVIS_REPO_SKILL_HINT_LIMIT", "3"))
    repo_skill_context = _repo_skill_hint_context(query, repo_hint_limit)
    store = getattr(runtime, "store", None) if runtime else None
    if store is None:
        content = repo_skill_context or "Skill library unavailable for this run; continue without saved workflow hints."
        return {
            "messages": [
                SystemMessage(
                    content=content,
                    id=SKILL_CONTEXT_MESSAGE_ID,
                )
            ],
            "active_skill_context": content,
        }

    limit = int(os.getenv("ALPHARAVIS_SKILL_LIBRARY_SEARCH_LIMIT", "3"))
    try:
        results = await _maybe_search(store, SKILL_LIBRARY_NS, query=query, limit=limit)
    except Exception as exc:
        return {
            "messages": [
                SystemMessage(
                    content=f"Skill library search failed: {exc}. Continue without saved workflow hints.",
                    id=SKILL_CONTEXT_MESSAGE_ID,
                )
            ],
            "active_skill_context": "",
        }

    active_skills = []
    for item in results or []:
        key = _store_item_key(item)
        value = _store_item_value(item)
        if isinstance(value, dict) and value.get("status") == "active":
            active_skills.append((key, value))

    sections = []
    if repo_skill_context:
        sections.append(repo_skill_context)

    if not active_skills:
        sections.append(
            "Skill library: no approved active workflow skill matched this task. "
            "Do not invent a saved workflow."
        )
    else:
        max_chars = int(os.getenv("ALPHARAVIS_SKILL_CONTEXT_MAX_CHARS", "2500"))
        body = "\n\n".join(_format_skill_record(key, value) for key, value in active_skills)
        sections.append(
            "Approved AlphaRavis workflow skills matched this task. Treat them as "
            "non-binding hints; keep normal reasoning, tool safety, and human "
            "approval gates in force.\n\n"
            f"{body[:max_chars]}"
        )
    content = "\n\n".join(sections)

    return {
        "messages": [SystemMessage(content=content, id=SKILL_CONTEXT_MESSAGE_ID)],
        "active_skill_context": content,
    }


async def context_guard_node(state: AlphaRavisState, runtime: Any | None = None) -> dict[str, Any]:
    messages = list(state.get("messages", []))
    token_limit = int(os.getenv("ALPHARAVIS_ACTIVE_TOKEN_LIMIT", "10000"))
    token_estimate = _estimate_tokens(messages)
    force_compression = _compression_forced_by_user(messages)

    if _compression_paused_by_user(messages):
        notice_key = hashlib.sha256(
            f"compression-paused:{_latest_user_query(messages)}:{token_estimate}".encode("utf-8")
        ).hexdigest()[:16]
        return {
            "memory_notice": (
                "Kompression wurde fuer diesen Lauf ausgesetzt, weil du es im Chat "
                "so angefordert hast. Wenn der Verlauf zu gross wird, kann die "
                "naechste Modellantwort langsamer oder instabiler werden."
            ),
            "memory_notice_key": notice_key,
        }

    if token_estimate <= token_limit and not force_compression:
        return {}

    keep_last = int(os.getenv("ALPHARAVIS_CONTEXT_KEEP_LAST_MESSAGES", "12"))
    if len(messages) <= keep_last:
        if force_compression:
            notice_key = hashlib.sha256(
                f"compression-skipped-small:{_latest_user_query(messages)}:{len(messages)}".encode("utf-8")
            ).hexdigest()[:16]
            return {
                "memory_notice": (
                    "Manuelle Kompression wurde angefragt, aber der aktive Verlauf "
                    "ist noch zu kurz, um sinnvoll alte Nachrichten zu archivieren."
                ),
                "memory_notice_key": notice_key,
            }
        return {}

    old_messages = messages[:-keep_last]
    recent_messages = messages[-keep_last:]
    summary = await _summarize_messages(
        _model(),
        old_messages,
        state.get("context_summary"),
    )

    archive_key = hashlib.sha256(
        f"{time.time()}:{summary}:{len(old_messages)}".encode("utf-8")
    ).hexdigest()[:24]

    store = getattr(runtime, "store", None) if runtime else None
    archived_keys = list(state.get("archived_context_keys", []))
    archive_collection_keys = list(state.get("archive_collection_keys", []))
    compressed_archive_keys = list(state.get("compressed_archive_keys", []))
    archive_summary = state.get("archive_summary")
    thread_id = _state_thread_id(state)
    thread_key = _state_thread_key(state)
    hierarchy_notice = ""
    if store is not None:
        archive_record = {
            "summary": summary,
            "token_estimate": _estimate_tokens(old_messages),
            "archived_at": int(time.time()),
            "messages": [_message_to_json(message) for message in old_messages],
            "thread_id": thread_id,
            "thread_key": thread_key,
        }
        await _maybe_put(store, _thread_archive_ns(thread_id), archive_key, archive_record)
        await _maybe_put(
            store,
            ARCHIVE_INDEX_NS,
            archive_key,
            {key: value for key, value in archive_record.items() if key != "messages"},
        )
        archived_keys.append(archive_key)
        compact_update = await _maybe_compact_archives(
            store,
            thread_id,
            thread_key,
            archived_keys,
            compressed_archive_keys,
            archive_collection_keys,
            archive_summary,
        )
        archive_summary = compact_update.get("archive_summary", archive_summary)
        archive_collection_keys = compact_update.get("archive_collection_keys", archive_collection_keys)
        compressed_archive_keys = compact_update.get("compressed_archive_keys", compressed_archive_keys)
        hierarchy_notice = compact_update.get("archive_compression_notice", "")

    prefix = "Manuelle Kompression: " if force_compression else ""
    memory_notice = (
        f"{prefix}Ich habe den aktiven Chat-Kontext komprimiert: ca. {_estimate_tokens(old_messages)} "
        f"alte Tokens wurden als Archiv `{archive_key}` gespeichert, die letzten "
        f"{len(recent_messages)} Nachrichten bleiben aktiv."
    )
    if store is None:
        memory_notice += " Es war kein LangGraph Store verfuegbar, daher existiert nur die Summary im Thread."
    if hierarchy_notice:
        memory_notice += f" {hierarchy_notice}"

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            SystemMessage(
                content=(
                    "Earlier conversation was archived to long-term memory. "
                    "Use context_retrieval_agent if exact details are needed.\n\n"
                    f"Summary:\n{summary}"
                )
            ),
            *recent_messages,
        ],
        "context_summary": summary,
        "archive_summary": archive_summary,
        "archived_context_keys": archived_keys,
        "archive_collection_keys": archive_collection_keys,
        "compressed_archive_keys": compressed_archive_keys,
        "memory_notice": memory_notice,
        "memory_notice_key": archive_key,
    }


async def memory_notice_node(state: AlphaRavisState) -> dict[str, Any]:
    if not _env_bool("ALPHARAVIS_SHOW_MEMORY_NOTICES", "true"):
        return {}

    notice = state.get("memory_notice")
    notice_key = state.get("memory_notice_key")
    if not notice or not notice_key:
        return {}
    if notice_key == state.get("memory_notice_seen_key"):
        return {}

    message_id = f"alpharavis_memory_notice_{notice_key}"
    return {
        "messages": [AIMessage(content=f"\n\nMemory-Notice: {notice}", id=message_id)],
        "memory_notice_seen_key": notice_key,
    }


async def run_profile_finish_node(state: AlphaRavisState) -> dict[str, Any]:
    profile = dict(state.get("run_profile") or {})
    started_at = profile.get("started_at")
    if isinstance(started_at, (int, float)):
        profile["total_seconds"] = round(time.time() - started_at, 3)
    profile["finished_at"] = time.time()

    if not _env_bool("ALPHARAVIS_SHOW_RUN_PROFILE", "false"):
        return {"run_profile": profile}

    summary = (
        "\n\nRun-Profile: "
        f"route={profile.get('route', 'unknown')}; "
        f"total={profile.get('total_seconds', '?')}s; "
        f"reason={profile.get('route_reason', 'n/a')}; "
        f"messages={profile.get('message_count', '?')}; "
        f"tokens~={profile.get('token_estimate', '?')}"
    )
    if profile.get("fast_path_seconds") is not None:
        summary += f"; fast_path_llm={profile['fast_path_seconds']}s"
    if profile.get("fast_path_fallback_used"):
        summary += f"; fallback={profile.get('fast_path_model')}"

    return {
        "run_profile": profile,
        "messages": [AIMessage(content=summary, id=f"alpharavis_run_profile_{int(time.time())}")],
    }


def _create_ui_assistant(llm: ChatLiteLLM, handoff_tools: list[Any]):
    if create_cua is not None:
        try:
            computer_worker = create_cua(
                prompt=(
                    "You are the UI Expert. You have access to a virtual Linux "
                    "desktop via DISPLAY :0. Use visual feedback to confirm actions."
                ),
                environment="ubuntu",
            )
            computer_worker.name = "ui_assistant"
            return computer_worker
        except Exception as exc:
            print(f"WARNING: langgraph-cua could not initialize: {exc}")

    reason = f" ({CUA_IMPORT_ERROR})" if CUA_IMPORT_ERROR else ""
    return create_deep_agent(
        model=llm,
        tools=handoff_tools,
        name="ui_assistant",
        system_prompt=(
            "You are the UI Assistant, but direct GUI control is unavailable "
            f"in this runtime{reason}. Explain what UI steps would be needed "
            "and transfer to another agent when the task is not UI-specific."
        ),
    )


def _create_debugger_subgraph(llm: ChatLiteLLM, handoff_tools: list[Any]):
    debugger_worker = create_deep_agent(
        model=llm,
        tools=[
            execute_ssh_command,
            execute_local_command,
            fast_web_search,
            describe_optional_tool_registry,
            search_agent_memory,
            record_agent_memory,
            list_repo_ai_skills,
            read_repo_ai_skill,
            build_specialist_report,
            search_skill_library,
            list_skill_candidates,
            search_debugging_lessons,
            record_debugging_lesson,
            record_skill_candidate,
            *handoff_tools,
        ],
        name="debugger_agent_worker",
        system_prompt=(
            "You are the Debugger Agent. Your only job is to investigate "
            "infrastructure problems.\n\n"
            f"Available PCs: {list(REMOTE_PCS.keys())}\n"
            "ComfyUI is managed via PM2 on `comfy_server`; look for "
            "`comfyui_production` and ignore `comfyui_test`.\n"
            "Pixelle and LangGraph run as local Docker containers.\n\n"
            "Strict rules:\n"
            "1. Search debugger_agent memory and debugging lessons first when an error resembles a past failure.\n"
            "2. Diagnose first; always read logs before proposing a fix.\n"
            "3. Destructive or state-changing commands are guarded by a real approval interrupt.\n"
            "4. If code changes are needed, show the file path, problematic "
            "lines, and proposed fix.\n"
            "5. After a useful diagnosis or confirmed fix, record a debugging lesson "
            "with problem, root cause, fix, signals, and commands.\n"
            "6. When a reusable multi-agent workflow emerges, store it only as "
            "an inactive skill candidate; never assume it is approved. "
            "Optional MCP registries are lazy-loaded; call "
            "describe_optional_tool_registry when you need to know what exists. "
            "Use read_repo_ai_skill when the user asks to build or refactor "
            "AlphaRavis agents from reviewed repo skill cards. "
            "Use build_specialist_report for final handoff reports when "
            "evidence, commands, risks, and next actions matter. "
            "Use agent_id=`debugger_agent` for your own durable memories; use "
            "scope=`global` only for lessons useful to all agents."
        ),
    )

    async def run_debugger(state: DebuggerState) -> dict[str, Any]:
        result = await debugger_worker.ainvoke({"messages": state["messages"]})
        output_messages = list(result.get("messages", []))
        if not output_messages:
            return {
                "messages": [AIMessage(content="Debugger did not return a result.")],
                "internal_logs": ["Debugger returned an empty response."],
            }

        final_message = output_messages[-1]
        internal_logs = [_message_text(message) for message in output_messages[:-1]]
        return {"messages": [final_message], "internal_logs": internal_logs}

    builder = StateGraph(DebuggerState)
    builder.add_node("debugger_investigation", run_debugger)
    builder.add_edge(START, "debugger_investigation")
    builder.add_edge("debugger_investigation", END)
    graph = builder.compile()
    graph.name = "debugger_agent"
    return graph


def _build_graph(mcp_tools: list[Any] | None = None, store: Any | None = None):
    _warn_about_mongo_checkpointer()
    _configure_llm_cache()

    llm = _model()
    mcp_tools = mcp_tools or []

    transfer_to_research = create_handoff_tool(
        agent_name="research_expert",
        description="Transfer to the research expert for deep web or document research.",
    )
    transfer_to_generalist = create_handoff_tool(
        agent_name="general_assistant",
        description="Transfer to the general assistant for normal chat, coding, tools, Pixelle, or PC control.",
    )
    transfer_to_ui = create_handoff_tool(
        agent_name="ui_assistant",
        description="Transfer to the UI assistant for browser, VNC, or desktop automation.",
    )
    transfer_to_debugger = create_handoff_tool(
        agent_name="debugger_agent",
        description="Transfer to the debugger for failed jobs, logs, SSH, Docker, or infrastructure errors.",
    )
    transfer_to_context = create_handoff_tool(
        agent_name="context_retrieval_agent",
        description="Transfer to the context retrieval agent to search archived long-term conversation memory.",
    )

    research_worker = create_deep_agent(
        model=llm,
        tools=[
            deep_web_research,
            ask_documents,
            describe_optional_tool_registry,
            search_agent_memory,
            record_agent_memory,
            read_alpha_ravis_architecture,
            list_repo_ai_skills,
            read_repo_ai_skill,
            normalize_research_sources,
            build_specialist_report,
            transfer_to_generalist,
            transfer_to_debugger,
            transfer_to_context,
        ],
        name="research_expert",
        system_prompt=(
            "You are the Research Expert. Use deep_web_research for deep web "
            "research and ask_documents for local data. Search thoroughly, "
            "Use read_alpha_ravis_architecture only when the user asks about "
            "AlphaRavis itself, its architecture, or its capabilities. "
            "Use agent_id=`research_expert` for research-specific memories. "
            "Optional MCP registries are lazy-loaded; call "
            "describe_optional_tool_registry only when tool availability matters. "
            "Use list_repo_ai_skills/read_repo_ai_skill on demand for reviewed "
            "research workflows such as deep-research-report, market-research, "
            "and competitor-analysis. "
            "For substantial research, follow the DeepAgents research pattern: "
            "plan, choose focused passes, search broadly then narrowly, "
            "normalize citations with normalize_research_sources, synthesize "
            "with caveats, and verify the answer covers the request. Use "
            "build_specialist_report when returning evidence-heavy results to "
            "another AlphaRavis agent. "
            "Use global memories only for stable cross-agent preferences. "
            "return concise conclusions, and transfer to the correct peer when "
            "the task is outside research."
        ),
    )

    general_worker = create_deep_agent(
        model=llm,
        tools=[
            start_pixelle_remote,
            start_pixelle_async,
            check_pixelle_job,
            wake_on_lan,
            fast_web_search,
            describe_optional_tool_registry,
            read_alpha_ravis_architecture,
            list_repo_ai_skills,
            read_repo_ai_skill,
            build_specialist_report,
            search_agent_memory,
            record_agent_memory,
            create_manage_memory_tool(namespace=("memories",)),
            create_search_memory_tool(namespace=("memories",)),
            search_skill_library,
            list_skill_candidates,
            record_skill_candidate,
            activate_skill_candidate,
            deactivate_skill,
            transfer_to_research,
            transfer_to_ui,
            transfer_to_debugger,
            transfer_to_context,
        ]
        + mcp_tools,
        name="general_assistant",
        system_prompt=(
            "You are AlphaRavis's Generalist. Handle quick facts, Pixelle control, "
            "approved tool orchestration, and memory management. Do not use a "
            "raw shell execute path; transfer to debugger_agent for local or "
            "SSH command diagnostics so the approval gate stays in force. "
            "For long Pixelle jobs, prefer start_pixelle_async and return the "
            "job_id unless the user explicitly wants to wait. "
            "Use read_alpha_ravis_architecture only when the user asks what "
            "AlphaRavis is, what it can do, or how the stack works. "
            "Optional MCP registries are lazy-loaded; call "
            "describe_optional_tool_registry when a task may need optional tools. "
            "Use list_repo_ai_skills/read_repo_ai_skill when the user asks to "
            "build, inspect, or improve agents from reviewed repo skill cards. "
            "Use agent_id=`general_assistant` for your own memories. Search "
            "your agent memory before recording a new repeated lesson. "
            "Use approved skill-library entries only as hints. Store new "
            "workflows as inactive skill candidates for human review. "
            "Transfer directly to specialized peers instead of routing through "
            "a supervisor."
        ),
    )

    computer_worker = _create_ui_assistant(
        llm,
        [transfer_to_generalist, transfer_to_research, transfer_to_debugger, transfer_to_context],
    )

    debugger_worker = _create_debugger_subgraph(
        llm,
        [transfer_to_research, transfer_to_generalist, transfer_to_context],
    )

    context_worker = create_deep_agent(
        model=llm,
        tools=[
            search_archived_context,
            search_debugging_lessons,
            describe_optional_tool_registry,
            search_agent_memory,
            record_agent_memory,
            search_skill_library,
            list_skill_candidates,
            list_repo_ai_skills,
            read_repo_ai_skill,
            read_alpha_ravis_architecture,
            build_specialist_report,
            transfer_to_generalist,
            transfer_to_research,
            transfer_to_debugger,
        ],
        name="context_retrieval_agent",
        system_prompt=(
            "You are the Context Retrieval Agent. Search long-term archived "
            "conversation memory and return the precise facts needed by the "
            "active peer. By default, search only the current chat thread. "
            "Set include_other_threads=true only when the user explicitly asks "
            "to search other chats or all archives. Use read_alpha_ravis_architecture "
            "only for questions about AlphaRavis itself. Use agent_id=`context_retrieval_agent` "
            "for retrieval-specific memories. Optional MCP registry details are "
            "available through describe_optional_tool_registry. Repo AI skills can "
            "be listed or read on demand when the user asks for reviewed skill cards. "
            "Use build_specialist_report when returning retrieved facts, source "
            "keys, caveats, and next actions to another agent. Do not answer "
            "unrelated tasks yourself; transfer back."
        ),
    )

    swarm = create_swarm(
        [research_worker, general_worker, computer_worker, debugger_worker, context_worker],
        default_active_agent="general_assistant",
    ).compile(store=store)

    builder = StateGraph(AlphaRavisState)
    builder.add_node("run_profile_start", run_profile_start_node)
    builder.add_node("context_guard_before", context_guard_node)
    builder.add_node("route_decision", route_decision_node)
    builder.add_node("fast_chat", fast_chat_node)
    builder.add_node("skill_library", skill_library_node)
    builder.add_node("alpha_ravis_swarm", swarm)
    builder.add_node("context_guard_after", context_guard_node)
    builder.add_node("memory_notice", memory_notice_node)
    builder.add_node("run_profile_finish", run_profile_finish_node)
    builder.add_edge(START, "run_profile_start")
    builder.add_edge("run_profile_start", "context_guard_before")
    builder.add_edge("context_guard_before", "route_decision")
    builder.add_conditional_edges(
        "route_decision",
        route_after_decision,
        {"fast_path": "fast_chat", "swarm": "skill_library"},
    )
    builder.add_edge("fast_chat", "context_guard_after")
    builder.add_edge("skill_library", "alpha_ravis_swarm")
    builder.add_edge("alpha_ravis_swarm", "context_guard_after")
    builder.add_edge("context_guard_after", "memory_notice")
    builder.add_edge("memory_notice", "run_profile_finish")
    builder.add_edge("run_profile_finish", END)
    return builder.compile(store=store)


def _should_load_mcp(runtime: Any) -> bool:
    if not _env_bool("ALPHARAVIS_LOAD_MCP_TOOLS", "false"):
        return False

    if runtime is None:
        return True

    execution_runtime = getattr(runtime, "execution_runtime", None)
    if execution_runtime is None and hasattr(runtime, "access_context"):
        return False

    return True


def _open_mongodb_store(stack: contextlib.AsyncExitStack):
    if os.getenv("ALPHARAVIS_ENABLE_MONGODB_STORE", "true").lower() not in {"1", "true", "yes"}:
        return None

    uri = os.getenv("LS_MONGODB_URI") or os.getenv("MONGODB_URI")
    if not uri:
        return None

    try:
        from langgraph.store.mongodb import MongoDBStore

        return stack.enter_context(
            MongoDBStore.from_conn_string(
                conn_string=uri,
                db_name=os.getenv("ALPHARAVIS_STORE_DB", "langgraph_memory"),
                collection_name=os.getenv("ALPHARAVIS_STORE_COLLECTION", "long_term_store"),
            )
        )
    except Exception as exc:
        print(f"WARNING: MongoDBStore unavailable, continuing without long-term store: {exc}")
        return None


@contextlib.asynccontextmanager
async def make_graph(runtime: ServerRuntime | None = None):
    """LangGraph CLI entrypoint for the AlphaRavis brain."""

    async with contextlib.AsyncExitStack() as stack:
        mcp_tools = []
        if _should_load_mcp(runtime):
            mcp_tools = await _load_pixelle_mcp_tools(stack)

        store = getattr(runtime, "store", None) if runtime else None
        if store is None:
            store = _open_mongodb_store(stack)

        yield _build_graph(mcp_tools=mcp_tools, store=store)


__all__ = ["make_graph", "monitor_pixelle_job", "start_pixelle_remote"]
