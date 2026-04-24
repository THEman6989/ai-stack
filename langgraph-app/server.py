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
# --- KONFIGURATION FÜR REMOTE PIXELLE (PC B) ---
# =====================================================================
# TRAGE HIER DIE IP-ADRESSE DEINES COMFYUI/PIXELLE-RECHNERS EIN:
PC_B_IP = "192.168.178.50"
PIXELLE_URL = f"http://{PIXELLE_HOST}:9004"
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

                # 2. Monitor-Agent prüft Logs im Hintergrund-Kontext
                await agent_executor.ainvoke(
                    {"messages": [HumanMessage(content=f"Prüfe Logs von PC B: {logs[-500:]}")]},
                    config=monitor_config
                )

                # 3. VERARBEITUNG DES ERGEBNISSES
                if status == "completed":
                    # NEU: Wir nehmen direkt die echte HTTP-URL (oder den Text),
                    # den Pixelle standardmäßig ausspuckt!
                    original_pixelle_result = data.get("result", "")

                    final_msg = (
                        f"🔔 **Update: Dein Bild ist fertig!**\n\n"
                        f"{original_pixelle_result}" # Hier ist Pixelles kompletter Link schon drin!
                    )

                    print(f"✅ Job {job_id} fertig. Injiziere Nachricht in Thread {original_thread_id}")
                    await agent_executor.ainvoke(
                        {"messages": [HumanMessage(content=f"SYSTEM-MELDUNG: {final_msg}")]},
                        config={"configurable": {"thread_id": original_thread_id}}
                    )
                    is_running = False

                elif status == "failed":
                    error_msg = f"❌ **Job {job_id} abgebrochen.** In den Logs wurde ein Fehler erkannt."

                    # Fehler-Info in den Haupt-Chat posten
                    await agent_executor.ainvoke(
                        {"messages": [HumanMessage(content=error_msg)]},
                        config={"configurable": {"thread_id": original_thread_id}}
                    )
                    is_running = False

            except Exception as e:
                print(f"⚠️ Monitoring-Fehler (PC B evtl. offline?): {e}")
                is_running = False

# --- WERKZEUGE (TOOLS) ---

@tool
async def start_pixelle_remote(prompt: str, config: RunnableConfig):
    """Startet einen Workflow auf PC B. Die ID wird automatisch für das Monitoring genutzt."""
    # Hole die aktuelle Thread-ID aus der Config (von LibreChat/LangGraph)
    current_thread_id = config["configurable"].get("thread_id", "default_thread")

    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(f"{PIXELLE_URL}/api/run", json={"prompt": prompt})
            job_id = res.json().get("job_id")

            # Starte Monitor und übergib die aktuelle Thread-ID für den Callback!
            asyncio.create_task(monitor_pixelle_job(job_id, current_thread_id))

            return f"Bestätigt: Job `{job_id}` wurde auf PC B gestartet. Ich melde mich hier in diesem Chat, sobald das Bild fertig ist!"
        except Exception as e:
            return f"Fehler: Konnte PC B nicht erreichen. ({e})"

@tool
def wake_up_big_pc(mac_address: str = "AA:BB:CC:DD:EE:FF"):
    """Weckt den großen Hauptrechner via Wake-on-LAN auf."""
    from wakeonlan import send_magic_packet
    send_magic_packet(mac_address)
    return "System: Magic Packet gesendet."

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
        tools=[deep_web_research, ask_documents, deep_research_agent],
        name="research_expert",
        system_prompt="You are the Research Expert. Use 'deep_web_research' (Tavily) for deep web research and ask_documents for local data. You search thoroughly and comprehensively."
    )

    # Worker 2: The Generalist (Fast searches, PC control, Pixelle, Code)
    general_worker = create_deep_agent(
        model=llm,
        tools=[
            start_pixelle_remote,
            wake_up_big_pc,
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

    # 4. THE SUPERVISOR (AlphaRavis Chief Logic)
    agent_executor = create_supervisor(
        [research_worker, general_worker, computer_worker],
        model=llm,
        prompt="""You are AlphaRavis, the Chief Supervisor of this system. 
        Your job is to delegate tasks to the right specialized worker:
        - For simple questions (facts, weather, single info), coding, or PC/Pixelle control: Use 'general_assistant'.
        - For deep research, complex comparisons, or if more than 5 sources are needed: Use 'research_expert'.
        - ALWAYS coordinate the workers to give the user a perfect final answer. Do not do the work yourself, delegate it!""",
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

