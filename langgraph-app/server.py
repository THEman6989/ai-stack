import os
import asyncio
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pymongo import MongoClient
from redis import Redis

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_litellm import ChatLiteLLM
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph_supervisor import create_supervisor
from langgraph_cua import create_computer_use_agent
from langchain_redis import RedisCache
# Von: from langchain.globals import set_llm_cache
# Zu:
from langchain_core.globals import set_llm_cache

# --- DeepAgents & Bridge Imports ---
from deepagents.backends.local_shell import LocalShellBackend
from langchain_fastapi_chat_completion.core.base_agent_factory import BaseAgentFactory
from langchain_fastapi_chat_completion.core.create_agent_dto import CreateAgentDto
from langchain_fastapi_chat_completion.fastapi.langchain_openai_api_bridge_fastapi import LangchainOpenaiApiBridgeFastAPI

from langmem import create_manage_memory_tool, create_search_memory_tool
from deepagents import create_deep_agent
from deepagents.backends.local_shell import LocalShellBackend
from langgraphics import LangGraphicsMiddleware

# --- NEU: MCP Imports ---
from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.runnables import RunnableConfig

# =====================================================================
# --- DYNAMIC INFRASTRUCTURE CONFIG ---
# =====================================================================
import json

pcs_env = os.getenv("REMOTE_PCS", "{}")
try:
    REMOTE_PCS = json.loads(pcs_env)
except Exception as e:
    print(f"Error loading REMOTE_PCS: {e}")
    REMOTE_PCS = {}

SSH_USER = os.getenv("SSH_USER", "root")
SSH_PASS_DEFAULT = os.getenv("SSH_PASS_DEFAULT", "")

# Pixelle runs locally in Docker — no remote IP needed
PIXELLE_URL = "http://pixelle:9004"

#COMFY_IP = REMOTE_PCS.get("comfy_server", {}).get("ip")
COMFY_IP = REMOTE_PCS.get("comfy_server", {}).get("ip")
if not COMFY_IP:
    print("⚠️ WARNING: 'comfy_server' IP not found in REMOTE_PCS env variable!")
# =====================================================================

# --- INFRASTRUKTUR ---
mongo_client = MongoClient("mongodb://mongodb:27017")
checkpointer = MongoDBSaver(mongo_client, db_name="langgraph_memory")
store = MongoDBSaver(mongo_client, db_name="langgraph_memory", collection_name="long_term_store")

redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
set_llm_cache(RedisCache(redis_url=redis_url)) # URL String übergeben!
redis_client = Redis.from_url(redis_url)

# --- GEHIRN ---
llm = ChatLiteLLM(model="edge-gemma", base_url="http://litellm:4000")
agent_executor = None
mcp_session_manager = None # Für den sauberen Shutdown des MCP Clients

# --- DER SEPARATE ERROR-VERWALTER (Background Task) ---
async def monitor_pixelle_job(job_id: str, original_thread_id: str):
    """
    Läuft im Hintergrund auf PC A.
    Überwacht PC B und postet das Ergebnis zurück in den Haupt-Chat.
    """
    print(f"👁️ Error-Verwalter gestartet für Remote-Job {job_id}")

    # Separater Kontext für die Überwachung (Monitor-Session)
    monitor_config = {"configurable": {"thread_id": f"monitor_{job_id}"}}

    async with httpx.AsyncClient(timeout=30.0) as client:
        is_running = True
        while is_running:
            await asyncio.sleep(10) # Check alle 10 Sekunden

            try:
                # 1. Status von PC B abrufen
                response = await client.get(f"{PIXELLE_URL}/api/status/{job_id}")
                data = response.json()
                status = data.get("status", "running")
                logs = data.get("logs", "Keine Logs.")

                # 3. VERARBEITUNG DES ERGEBNISSES
                if status == "completed":
                    # NEU: Wir nehmen direkt die echte HTTP-URL (oder den Text),
                    # den Pixelle standardmäßig ausspuckt!
                    original_pixelle_result = data.get("result", "")

                    final_msg = (
                        f"🔔 **Image ready!** Job `{job_id}` completed.\n\n"
                        f"{original_pixelle_result}"
                    )

                    print(f"✅ Job {job_id} fertig. Injiziere Nachricht in Thread {original_thread_id}")
                    await agent_executor.ainvoke(
                        {"messages": [HumanMessage(content=f"SYSTEM NOTIFICATION: {final_msg}")]},
                        config={"configurable": {"thread_id": original_thread_id}}
                    )
                    is_running = False

                elif status == "failed":
                    # System instruction for the Supervisor to trigger the Debugger Agent
                    instruction = (
                        f"CRITICAL ERROR: Pixelle job `{job_id}` failed.\n"
                        f"INSTRUCTION FOR SUPERVISOR:\n"
                        f"1. Delegate to 'debugger_agent' immediately.\n"
                        f"2. Pixelle runs as a local Docker container. Use 'execute_ssh_command' on 'main_pc' "
                        f"   (localhost equivalent) or check Docker logs directly: 'docker logs pixelle --tail 50'.\n"
                        f"3. Also check the LangGraph app logs: 'docker logs langgraph-app --tail 50'.\n"
                        f"4. If a code error is found in server.py or any project file, read the file, "
                        f"   propose the fix in the chat, and WAIT for user approval before applying. Do NOT apply any changes automatically, even if the fix is obvious.\n"
                        f"5. HUMAN-IN-THE-LOOP: Do NOT apply any fix, restart, or docker command automatically. "
                        f"   Present findings and proposed solution first."
                    )

                    await agent_executor.ainvoke(
                        {"messages": [HumanMessage(content=instruction)]},
                        config={"configurable": {"thread_id": original_thread_id}}
                    )
                    is_running = False

            except Exception as e:
                print(f"⚠️ Monitoring-Fehler (PC B evtl. offline?): {e}")
                is_running = False

# --- WERKZEUGE (TOOLS) ---

@tool
async def start_pixelle_remote(prompt: str, config: RunnableConfig):
    """Starts a workflow on the remote ComfyUI PC. The job ID is monitored automatically."""
    current_thread_id = config["configurable"].get("thread_id", "default_thread")
    
    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(f"{PIXELLE_URL}/api/run", json={"prompt": prompt})
            job_id = res.json().get("job_id")
            asyncio.create_task(monitor_pixelle_job(job_id, current_thread_id))
            return f"Confirmed: Job `{job_id}` started on Comfy Server. I will notify you here once it's done."
        except Exception as e:
            return f"Error: Could not reach Comfy Server. ({e})"

@tool
def wake_on_lan(pc_name: str):
    """Sends a magic packet to wake up a remote PC by its name (e.g., 'main_pc')."""
    from wakeonlan import send_magic_packet
    pc_info = REMOTE_PCS.get(pc_name)
    if not pc_info or "mac" not in pc_info:
        return f"Error: PC '{pc_name}' not found. Available: {list(REMOTE_PCS.keys())}"
    
    send_magic_packet(pc_info["mac"])
    return f"System: Magic Packet sent to {pc_name}."


@tool
def execute_ssh_command(pc_name: str, command: str):
    """Executes a shell command on a remote PC via SSH.
    Use this for debugging: reading logs, checking PM2 status, inspecting files.
    Available PCs: see REMOTE_PCS env variable."""
    import subprocess

    pc_info = REMOTE_PCS.get(pc_name)
    if not pc_info or "ip" not in pc_info:
        return f"Error: PC '{pc_name}' not found. Available: {list(REMOTE_PCS.keys())}"

    ip = pc_info["ip"]
    # Per-PC password override, or fall back to global default
    ssh_pass = pc_info.get("ssh_pass", SSH_PASS_DEFAULT)

    ssh_target = f"{SSH_USER}@{ip}"
    ssh_opts = ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"]

    if ssh_pass:
        # Use sshpass for password auth (requires sshpass installed in Dockerfile)
        cmd = ["sshpass", "-p", ssh_pass, "ssh"] + ssh_opts + [ssh_target, command]
    else:
        # Fall back to key-based auth
        cmd = ["ssh"] + ssh_opts + [ssh_target, command]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        return f"Exit Code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    except subprocess.TimeoutExpired:
        return f"Error: SSH command timed out after 45s on '{pc_name}'."
    except Exception as e:
        return f"SSH Connection failed: {str(e)}"

@tool
def fast_web_search(query: str):
    """ONLY for quick facts, weather, or simple questions (max 1-2 sources)."""
    search = DuckDuckGoSearchRun()
    return search.invoke(query)

@tool
async def deep_web_research(query: str):
    """USE THIS TOOL for complex research, comparisons, or when more than 5 websites need to be checked. 
    It uses secure proxies (Tavily) to hide your IP and perform deep searches."""
    search = TavilySearchResults(max_results=10)
    return await search.ainvoke({"query": query})

@tool
async def ask_documents(query: str):
    """Nutze dieses Tool, um Wissen aus deinen lokal hochgeladenen Dokumenten (RAG) abzurufen."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"http://rag_api:8000/query", params={"text": query})
            return response.json()
        except Exception as e:
            return f"Fehler bei der Dokumentensuche: {e}"

#@tool
#def deep_research_agent(topic: str):
    #"""NUR für extrem komplexe Themen nutzen, die eine tiefe Analyse erfordern."""
    #return f"Deep Researcher wurde für '{topic}' gestartet. (Sub-Graph Integration folgt)."

# --- LEBENSZYKLUS DES SERVERS ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_executor, mcp_session_manager
    print("🚀 AlphaRavis wird initialisiert...")

    # FIX: root_dir statt working_directory
    sandbox = LocalShellBackend(root_dir="/workspace")

    # 2. Remote Pixelle MCP Tools via SSE laden
    mcp_tools = []
    try:
        # Verbindung zum internen Pixelle-Container aufbauen
        mcp_session_manager = sse_client(f"{PIXELLE_URL}/pixelle/mcp/sse")
        mcp_streams = await mcp_session_manager.__aenter__()
        session = ClientSession(*mcp_streams)
        await session.initialize()

        mcp_tools = await load_mcp_tools(session)
        print(f"✅ {len(mcp_tools)} Interne Docker MCP Tools von Pixelle geladen.")
    except Exception as e:
        print(f"⚠️ Interner MCP (Pixelle) konnte nicht erreicht werden: {e}")

# 3. Create Specialized Workers (Phase 2 & 3: Supervisor + Deep Research)
    
    # Worker 1: The Research Expert (Uses Tavily for IP protection and local documents)
    research_worker = create_deep_agent(
        model=llm,
        tools=[deep_web_research, ask_documents,],
        name="research_expert",
        system_prompt="You are the Research Expert. Use 'deep_web_research' (Tavily) for deep web research and ask_documents for local data. You search thoroughly and comprehensively."
    )

    # Worker 2: The Generalist (Fast searches, PC control, Pixelle, Code)
    general_worker = create_deep_agent(
        model=llm,
        tools=[
            start_pixelle_remote,
            wake_on_lan,
            fast_web_search,
            create_manage_memory_tool(namespace=("memories",)),
            create_search_memory_tool(namespace=("memories",))
        ] + mcp_tools,
        backend=sandbox,
        name="general_assistant",
        system_prompt="You are the Generalist. Responsible for quick facts (fast_web_search), Pixelle control, code execution in the sandbox, and memory management."
    )
    
    # Worker 3: The Computer Use Agent (Direct GUI control via VNC) 
    computer_worker = create_computer_use_agent(
        model=llm,
        name="ui_assistant",
        system_prompt="""You are the UI Expert. 
        You have access to a virtual Linux desktop via DISPLAY :0.
        You can use a browser, terminal, and other GUI apps.
        Use visual feedback to confirm your actions."""
    )
    # --- NEW: Worker 4: The Debugger Agent ---
    debugger_worker = create_deep_agent(
        model=llm,
        tools=[
            execute_ssh_command,
            fast_web_search,   # To look up error messages online
        ],
        name="debugger_agent",
        system_prompt=f"""You are the Debugger Agent. Your ONLY job is to investigate and fix infrastructure problems.
 
        CAPABILITIES:
        - SSH into remote PCs to read logs and inspect files
        - Available PCs: {list(REMOTE_PCS.keys())}
        - ComfyUI is managed via PM2 on 'comfy_server'. Use these commands:
        - 'pm2 list' → see all processes and their status/port
        - 'pm2 logs comfyui_production --lines 50' → read recent logs
        - 'pm2 restart comfyui_production' → restart (ONLY with user approval!)
        - 'cat /path/to/server.py' → inspect code files
        - Pixelle and LangGraph run as local Docker containers. Use 'docker logs <name> --tail 50'.
        - Process names in PM2: look for 'comfyui_production' for production. Ignore 'comfyui_test'.

        STRICT RULES:
        1. DIAGNOSE first — always read logs before proposing anything.
        2. NEVER run destructive commands (restart, rm, stop, docker-compose down) without explicit user approval.
        3. If you find a code error in a file (e.g. server.py), show the user: the file path, the problematic lines, and your proposed fix.
        4. After presenting findings, WAIT. Only proceed when the user says "yes", "fix it", or "proceed".
        5. After a confirmed fix, you MAY run 'docker compose up -d --build' or 'pm2 restart' to apply it."""


    )

    # 4. THE SUPERVISOR (AlphaRavis Chief Logic)
    agent_executor = create_supervisor(
        [research_worker, general_worker, computer_worker, debugger_worker],
        model=llm,
        prompt="""You are AlphaRavis, the Chief Supervisor of this system.
        Delegate tasks to the right specialized worker:
        - Simple questions, facts, coding, PC/Pixelle control: Use 'general_assistant'.
        - Deep research, complex comparisons, 5+ sources needed: Use 'research_expert'.
        - GUI automation, browser control, desktop tasks: Use 'ui_assistant'.
        - ANY error, failed job, infrastructure problem, SSH investigation, container issues: Use 'debugger_agent'.
        - ALWAYS coordinate the workers to give the user a perfect final answer. Do not do the work yourself, delegate it!
        CRITICAL RULE: The 'debugger_agent' must ALWAYS ask the user for approval before applying any fix. Never bypass this rule.""",
        checkpointer=checkpointer,
        store=store
    )

    print("✅ AlphaRavis Supervisor Ready.")
    yield

    # Cleanup beim Beenden des Servers
    if mcp_session_manager:
        await mcp_session_manager.__aexit__(None, None, None)

# FastAPI initialisieren
app = FastAPI(lifespan=lifespan)

# --- API BRIDGE (Anbindung an LibreChat) ---
class MyEnterpriseAgentFactory(BaseAgentFactory):
    # Das ": CreateAgentDto" ist PFLICHT!
    def create_agent(self, create_agent_dto: CreateAgentDto):
        return agent_executor

# To this:
bridge = LangchainOpenaiApiBridgeFastAPI(
    app=app,
    agent_factory_provider=MyEnterpriseAgentFactory()
)

# And ensure you call the bind method afterwards:
bridge.bind_openai_chat_completion(path="/v1")

