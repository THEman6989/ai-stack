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
from deepagents.backends.local_shell import LocalShellBackend
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

if not COMFY_IP:
    print("WARNING: 'comfy_server' IP not found in REMOTE_PCS env variable.")


def _model() -> ChatLiteLLM:
    return ChatLiteLLM(
        model=os.getenv("ALPHARAVIS_MODEL", "openai/big-boss"),
        api_base=os.getenv("OPENAI_API_BASE", "http://litellm:4000/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "sk-local-dev"),
        request_timeout=float(os.getenv("ALPHARAVIS_LLM_TIMEOUT_SECONDS", "120")),
        max_retries=int(os.getenv("ALPHARAVIS_LLM_MAX_RETRIES", "0")),
        streaming=os.getenv("ALPHARAVIS_LLM_STREAMING", "true").lower() in {"1", "true", "yes"},
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
async def activate_skill_candidate(skill_id: str, approval_note: str = ""):
    """Promote a reviewed skill candidate to active when promotion is explicitly enabled."""

    if os.getenv("ALPHARAVIS_ALLOW_SKILL_PROMOTION", "false").lower() not in {"1", "true", "yes"}:
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
    if os.getenv("ALPHARAVIS_ALLOW_USER_COMPRESSION_PAUSE", "true").lower() not in {
        "1",
        "true",
        "yes",
    }:
        return False

    latest = _latest_user_query(messages).lower()
    return any(pattern in latest for pattern in COMPRESSION_PAUSE_PATTERNS)


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
    store = getattr(runtime, "store", None) if runtime else None
    if store is None:
        return {
            "messages": [
                SystemMessage(
                    content="Skill library unavailable for this run; continue without saved workflow hints.",
                    id=SKILL_CONTEXT_MESSAGE_ID,
                )
            ],
            "active_skill_context": "",
        }

    query = _latest_user_query(messages)
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

    if not active_skills:
        content = (
            "Skill library: no approved active workflow skill matched this task. "
            "Do not invent a saved workflow."
        )
    else:
        max_chars = int(os.getenv("ALPHARAVIS_SKILL_CONTEXT_MAX_CHARS", "2500"))
        body = "\n\n".join(_format_skill_record(key, value) for key, value in active_skills)
        content = (
            "Approved AlphaRavis workflow skills matched this task. Treat them as "
            "non-binding hints; keep normal reasoning, tool safety, and human "
            "approval gates in force.\n\n"
            f"{body[:max_chars]}"
        )

    return {
        "messages": [SystemMessage(content=content, id=SKILL_CONTEXT_MESSAGE_ID)],
        "active_skill_context": content,
    }


async def context_guard_node(state: AlphaRavisState, runtime: Any | None = None) -> dict[str, Any]:
    messages = list(state.get("messages", []))
    token_limit = int(os.getenv("ALPHARAVIS_ACTIVE_TOKEN_LIMIT", "10000"))
    token_estimate = _estimate_tokens(messages)

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

    if token_estimate <= token_limit:
        return {}

    keep_last = int(os.getenv("ALPHARAVIS_CONTEXT_KEEP_LAST_MESSAGES", "12"))
    if len(messages) <= keep_last:
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

    memory_notice = (
        f"Ich habe den aktiven Chat-Kontext komprimiert: ca. {_estimate_tokens(old_messages)} "
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
    if os.getenv("ALPHARAVIS_SHOW_MEMORY_NOTICES", "true").lower() not in {"1", "true", "yes"}:
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
            search_skill_library,
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
            "1. Search debugging lessons first when an error resembles a past failure.\n"
            "2. Diagnose first; always read logs before proposing a fix.\n"
            "3. Destructive or state-changing commands are guarded by a real approval interrupt.\n"
            "4. If code changes are needed, show the file path, problematic "
            "lines, and proposed fix.\n"
            "5. After a useful diagnosis or confirmed fix, record a debugging lesson "
            "with problem, root cause, fix, signals, and commands.\n"
            "6. When a reusable multi-agent workflow emerges, store it only as "
            "an inactive skill candidate; never assume it is approved."
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
    sandbox = LocalShellBackend(root_dir=_workspace_root())
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
            transfer_to_generalist,
            transfer_to_debugger,
            transfer_to_context,
        ],
        name="research_expert",
        system_prompt=(
            "You are the Research Expert. Use deep_web_research for deep web "
            "research and ask_documents for local data. Search thoroughly, "
            "return concise conclusions, and transfer to the correct peer when "
            "the task is outside research."
        ),
    )

    general_worker = create_deep_agent(
        model=llm,
        tools=[
            start_pixelle_remote,
            wake_on_lan,
            fast_web_search,
            create_manage_memory_tool(namespace=("memories",)),
            create_search_memory_tool(namespace=("memories",)),
            search_skill_library,
            record_skill_candidate,
            activate_skill_candidate,
            transfer_to_research,
            transfer_to_ui,
            transfer_to_debugger,
            transfer_to_context,
        ]
        + mcp_tools,
        backend=sandbox,
        name="general_assistant",
        system_prompt=(
            "You are the Generalist. Handle quick facts, Pixelle control, "
            "safe code execution in the sandbox, and memory management. "
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
            search_skill_library,
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
            "to search other chats or all archives. Do not answer unrelated "
            "tasks yourself; transfer back."
        ),
    )

    swarm = create_swarm(
        [research_worker, general_worker, computer_worker, debugger_worker, context_worker],
        default_active_agent="general_assistant",
    ).compile(store=store)

    builder = StateGraph(AlphaRavisState)
    builder.add_node("context_guard_before", context_guard_node)
    builder.add_node("skill_library", skill_library_node)
    builder.add_node("alpha_ravis_swarm", swarm)
    builder.add_node("context_guard_after", context_guard_node)
    builder.add_node("memory_notice", memory_notice_node)
    builder.add_edge(START, "context_guard_before")
    builder.add_edge("context_guard_before", "skill_library")
    builder.add_edge("skill_library", "alpha_ravis_swarm")
    builder.add_edge("alpha_ravis_swarm", "context_guard_after")
    builder.add_edge("context_guard_after", "memory_notice")
    builder.add_edge("memory_notice", END)
    return builder.compile(store=store)


def _should_load_mcp(runtime: Any) -> bool:
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
