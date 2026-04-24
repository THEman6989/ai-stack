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
from langchain_redis import RedisCache
# Von: from langchain.globals import set_llm_cache
# Zu:
from langchain_core.globals import set_llm_cache

# Change "Fastapi" to "FastAPI"
from langchain_fastapi_chat_completion.fastapi.langchain_openai_api_bridge_fastapi import LangchainOpenaiApiBridgeFastAPI
from langchain_fastapi_chat_completion.core.base_agent_factory import BaseAgentFactory
from langchain_fastapi_chat_completion.core.create_agent_dto import CreateAgentDto

from langmem import create_manage_memory_tool, create_search_memory_tool
from deepagents import create_deep_agent
from deepagents.backends.local_shell import LocalShellBackend

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
PIXELLE_URL = f"http://{PC_B_IP}:8080" # Port ggf. anpassen
# =====================================================================

# --- INFRASTRUKTUR ---
mongo_client = MongoClient("mongodb://mongodb:27017")
checkpointer = MongoDBSaver(mongo_client, db_name="langgraph_memory")
store = MongoDBSaver(mongo_client, db_name="langgraph_memory", collection_name="long_term_store")

# FIX: Pass the connection string to RedisCache instead of the client object
redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
set_llm_cache(RedisCache(redis_url=redis_url))

# If you need the client for other parts of the app:
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
    """Nutze dieses Tool für SCHNELLE Suchen (Wetter, Fakten, News)."""
    search = DuckDuckGoSearchRun()
    return search.invoke(query)

@tool
async def ask_documents(query: str):
    """Nutze dieses Tool, um Wissen aus deinen lokal hochgeladenen Dokumenten (RAG) abzurufen."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"http://rag_api:8000/query", params={"text": query})
            return response.json()
        except Exception as e:
            return f"Fehler bei der Dokumentensuche: {e}"

@tool
def deep_research_agent(topic: str):
    """NUR für extrem komplexe Themen nutzen, die eine tiefe Analyse erfordern."""
    return f"Deep Researcher wurde für '{topic}' gestartet. (Sub-Graph Integration folgt)."

# --- LEBENSZYKLUS DES SERVERS ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_executor, mcp_session_manager

    print("🚀 AlphaRavis wird initialisiert...")

    # 1. Native Sandbox für Code
    sandbox = LocalShellBackend(working_directory="/workspace")

    # 2. Remote Pixelle MCP Tools via SSE laden
    mcp_tools = []
    try:
        # Verbindung zu PC B aufbauen
        mcp_session_manager = sse_client(f"http://{PC_B_IP}:8080/pixelle/mcp/sse")
        mcp_streams = await mcp_session_manager.__aenter__()
        session = ClientSession(*mcp_streams)
        await session.initialize()

        mcp_tools = await load_mcp_tools(session)
        print(f"✅ {len(mcp_tools)} Remote MCP Tools von PC B geladen.")
    except Exception as e:
        print(f"⚠️ Remote MCP (PC B) konnte nicht erreicht werden: {e}")

    # 3. Agenten erstellen
    agent_executor = create_deep_agent(
        model=llm,
        tools=[
            start_pixelle_remote, # <-- Neues asynchrones Remote-Tool
            wake_up_big_pc,
            fast_web_search,
            ask_documents,
            deep_research_agent,
            create_manage_memory_tool(namespace=("memories",)),
            create_search_memory_tool(namespace=("memories",))
        ] + mcp_tools, # <-- Fügt die nativen MCP Tools von PC B hinzu, falls sie direkt genutzt werden sollen
        sandbox=sandbox,
        checkpointer=checkpointer,
        store=store,
        system_prompt="""You are AlphaRavis, the chief agent of this system.
Decide wisely:
- Use 'fast_web_search' for quick internet facts.
- Use 'ask_documents' to find info in your own files and archives.
- Use 'deep_research_agent' for long, complex analyses.
- Use 'start_pixelle_remote' to start image generation or ComfyUI workflows on the remote PC. DO NOT wait for the image, just trigger it.
- You have native computer access via a sandbox to write, read, and run code autonomously.
- Proactively use your memory tools to remember user preferences across chats!"""
    )

    print("✅ AlphaRavis Ready.")
    yield

    # Cleanup beim Beenden des Servers
    if mcp_session_manager:
        await mcp_session_manager.__aexit__(None, None, None)

# FastAPI initialisieren
app = FastAPI(lifespan=lifespan)

# --- API BRIDGE (Anbindung an LibreChat) ---
class MyEnterpriseAgentFactory(BaseAgentFactory):
    # Add ": CreateAgentDto" to the parameter
    def create_agent(self, create_agent_dto: CreateAgentDto):
        return agent_executor

# To this:
bridge = LangchainOpenaiApiBridgeFastAPI(
    app=app,
    agent_factory_provider=MyEnterpriseAgentFactory()
)

# And ensure you call the bind method afterwards:
bridge.bind_openai_chat_completion(path="/v1")

