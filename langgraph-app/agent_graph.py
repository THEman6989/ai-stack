from __future__ import annotations

import asyncio
import contextlib
import json
import os
from pathlib import Path
from typing import Any

import httpx
from deepagents import create_deep_agent
from deepagents.backends.local_shell import LocalShellBackend
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.globals import set_llm_cache
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_litellm import ChatLiteLLM
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_redis import RedisCache
from langgraph.func import task
from langgraph_supervisor import create_supervisor
from langmem import create_manage_memory_tool, create_search_memory_tool
from mcp import ClientSession
from mcp.client.sse import sse_client

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

if not COMFY_IP:
    print("WARNING: 'comfy_server' IP not found in REMOTE_PCS env variable.")


def _model() -> ChatLiteLLM:
    return ChatLiteLLM(
        model=os.getenv("ALPHARAVIS_MODEL", "edge-gemma"),
        base_url=os.getenv("OPENAI_API_BASE", "http://litellm:4000"),
    )


def _workspace_root() -> str:
    configured = os.getenv("ALPHARAVIS_WORKSPACE_DIR")
    if configured:
        return configured
    if Path("/workspace").exists():
        return "/workspace"
    return str(Path(__file__).resolve().parents[1])


def _configure_llm_cache() -> None:
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
        "1. Delegate to `debugger_agent` immediately.\n"
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


def _create_ui_assistant(llm: ChatLiteLLM):
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
        tools=[],
        name="ui_assistant",
        system_prompt=(
            "You are the UI Assistant, but direct GUI control is unavailable "
            f"in this runtime{reason}. Explain what UI steps would be needed "
            "and ask for a runtime with langgraph-cua, DISPLAY, and VNC when "
            "visual automation is required."
        ),
    )


def _build_graph(mcp_tools: list[Any] | None = None):
    _warn_about_mongo_checkpointer()
    _configure_llm_cache()

    llm = _model()
    sandbox = LocalShellBackend(root_dir=_workspace_root())
    mcp_tools = mcp_tools or []

    research_worker = create_deep_agent(
        model=llm,
        tools=[deep_web_research, ask_documents],
        name="research_expert",
        system_prompt=(
            "You are the Research Expert. Use deep_web_research for deep web "
            "research and ask_documents for local data. Search thoroughly and "
            "return concise, well-sourced conclusions."
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
        ]
        + mcp_tools,
        backend=sandbox,
        name="general_assistant",
        system_prompt=(
            "You are the Generalist. Handle quick facts, Pixelle control, "
            "safe code execution in the sandbox, and memory management."
        ),
    )

    computer_worker = _create_ui_assistant(llm)

    debugger_worker = create_deep_agent(
        model=llm,
        tools=[execute_ssh_command, fast_web_search],
        name="debugger_agent",
        system_prompt=(
            "You are the Debugger Agent. Your only job is to investigate "
            "infrastructure problems.\n\n"
            f"Available PCs: {list(REMOTE_PCS.keys())}\n"
            "ComfyUI is managed via PM2 on `comfy_server`; look for "
            "`comfyui_production` and ignore `comfyui_test`.\n"
            "Pixelle and LangGraph run as local Docker containers.\n\n"
            "Strict rules:\n"
            "1. Diagnose first; always read logs before proposing a fix.\n"
            "2. Never run destructive commands without explicit user approval.\n"
            "3. If code changes are needed, show the file path, problematic "
            "lines, and proposed fix.\n"
            "4. After presenting findings, wait for approval before applying "
            "fixes, restarts, or docker commands."
        ),
    )

    supervisor = create_supervisor(
        [research_worker, general_worker, computer_worker, debugger_worker],
        model=llm,
        prompt=(
            "You are AlphaRavis, the Chief Supervisor of this system. "
            "Delegate tasks to the right specialized worker:\n"
            "- Simple questions, facts, coding, PC/Pixelle control: "
            "general_assistant.\n"
            "- Deep research, complex comparisons, or 5+ sources: "
            "research_expert.\n"
            "- GUI automation, browser control, desktop tasks: ui_assistant.\n"
            "- Errors, failed jobs, infrastructure problems, SSH investigation, "
            "or container issues: debugger_agent.\n"
            "Coordinate workers for a final answer, and never bypass the "
            "debugger_agent approval rule for fixes."
        ),
    )
    return supervisor.compile()


def _should_load_mcp(runtime: Any) -> bool:
    if runtime is None:
        return True

    execution_runtime = getattr(runtime, "execution_runtime", None)
    if execution_runtime is None and hasattr(runtime, "access_context"):
        return False

    return True


@contextlib.asynccontextmanager
async def make_graph(runtime: ServerRuntime | None = None):
    """LangGraph CLI entrypoint for the AlphaRavis brain."""

    async with contextlib.AsyncExitStack() as stack:
        mcp_tools = []
        if _should_load_mcp(runtime):
            mcp_tools = await _load_pixelle_mcp_tools(stack)

        yield _build_graph(mcp_tools=mcp_tools)


__all__ = ["make_graph", "monitor_pixelle_job", "start_pixelle_remote"]
