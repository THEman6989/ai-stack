import os
import asyncio
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pymongo import MongoClient
from redis import Redis

from langchain_core.tools import tool
from langchain_litellm import ChatLiteLLM
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_redis import RedisCache
from langchain.globals import set_llm_cache

from langchain_fastapi_chat_completion.fastapi.langchain_openai_api_bridge_fastapi import LangchainOpenaiApiBridgeFastapi
from langchain_fastapi_chat_completion.core.base_agent_factory import BaseAgentFactory

from langmem import create_manage_memory_tool, create_search_memory_tool
from deepagents import create_deep_agent
from deepagents.backends.local_shell import LocalShellSandbox

# --- INFRASTRUKTUR (Bleibt erhalten) ---
mongo_client = MongoClient("mongodb://mongodb:27017")
checkpointer = MongoDBSaver(mongo_client, db_name="langgraph_memory")
store = MongoDBSaver(mongo_client, db_name="langgraph_memory", collection_name="long_term_store")

redis_client = Redis(host="redis", port=6379)
set_llm_cache(RedisCache(redis_client))

# --- WERKZEUGE (TOOLS) ---

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
        # Verbindung zur rag_api in deinem Docker-Netzwerk
        try:
            response = await client.get(f"http://rag_api:8000/query", params={"text": query})
            return response.json()
        except Exception as e:
            return f"Fehler bei der Dokumentensuche: {e}"

@tool
def deep_research_agent(topic: str):
    """NUR für extrem komplexe Themen nutzen, die eine tiefe Analyse erfordern."""
    return f"Deep Researcher wurde für '{topic}' gestartet. (Sub-Graph Integration folgt)."

# --- GEHIRN ---
llm = ChatLiteLLM(model="edge-gemma", base_url="http://litellm:4000")
agent_executor = None

# --- LEBENSZYKLUS DES SERVERS ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_executor

    print("🚀 AlphaRavis wird initialisiert...")

    # 1. Native Sandbox für Code (Ersetzt OpenCode komplett!)
    sandbox = LocalShellSandbox(cwd="/workspace")

    # 2. ComfyUI MCP Tools laden (Optionaler Block)
    comfy_tools = []
    # try:
    #     from langchain_mcp_adapters.tools import load_mcp_tools
    #     # Hier die URL deines ComfyUI-MCP-Servers einfügen
    #     comfy_tools = await load_mcp_tools("http://comfyui-mcp:8080")
    #     print(f"✅ {len(comfy_tools)} ComfyUI Tools geladen.")
    # except Exception as e:
    #     print(f"⚠️ ComfyUI MCP konnte nicht geladen werden: {e}")

    # 3. Agenten erstellen (Der ultimative Allrounder)
    agent_executor = create_deep_agent(
        model=llm,
        tools=[
            wake_up_big_pc,
            fast_web_search,
            ask_documents,       # Neu: RAG Power
            deep_research_agent,
            create_manage_memory_tool(namespace=("memories",)),
            create_search_memory_tool(namespace=("memories",))
        ] + comfy_tools,
        sandbox=sandbox,        # Native Coding Power
        checkpointer=checkpointer,
        store=store,
        system_prompt="""You are AlphaRavis, the chief agent of this system.
Decide wisely:
- Use 'fast_web_search' for quick internet facts.
- Use 'ask_documents' to find info in your own files and archives.
- Use 'deep_research_agent' for long, complex analyses.
- You have native computer access via a sandbox to write, read, and run code autonomously.
- Proactively use your memory tools to remember user preferences across chats!"""
    )

    print("✅ AlphaRavis Ready.")
    yield

# FastAPI initialisieren
app = FastAPI(lifespan=lifespan)

# --- API BRIDGE (Anbindung an LibreChat) ---
class MyEnterpriseAgentFactory(BaseAgentFactory):
    def create_agent(self, create_agent_dto):
        return agent_executor

bridge = LangchainOpenaiApiBridgeFastapi(
    agent_factory=MyEnterpriseAgentFactory()
)
app.include_router(bridge.router)
