import os
import asyncio
from contextlib import asynccontextmanager
import time
from fastapi import FastAPI
from pymongo import MongoClient
from redis import Redis

from langchain_core.tools import tool
from langchain_litellm import ChatLiteLLM
from deepagents import create_deep_agent
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_redis import RedisCache
from langchain.globals import set_llm_cache

from langchain_fastapi_chat_completion.fastapi.langchain_openai_api_bridge_fastapi import LangchainOpenaiApiBridgeFastapi
from langchain_fastapi_chat_completion.core.base_agent_factory import BaseAgentFactory

from langgraph.store.mongodb import MongoDBSaver
from langgraph.store.base import BaseStore

# --- NEUE MCP IMPORTE FÜR OPENCODE ---
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

from langmem import create_manage_memory_tool, create_search_memory_tool
# Optional: Hier würden wir später den Deep Researcher importieren
# import sys
# sys.path.append("/app/local_deep_researcher")
# from graph import graph as deep_researcher_graph


# --- INFRASTRUKTUR ---
mongo_client = MongoClient("mongodb://mongodb:27017")
checkpointer = MongoDBSaver(mongo_client, db_name="langgraph_memory")

# Dein permanenter Speicher (Langzeitgedächtnis)
store = MongoDBSaver(mongo_client, db_name="langgraph_memory", collection_name="long_term_store")

redis_client = Redis(host="redis", port=6379)
set_llm_cache(RedisCache(redis_client))


# --- WERKZEUGE (TOOLS) ---

@tool
def wake_up_big_pc(mac_address: str = "AA:BB:CC:DD:EE:FF"):
    """Weckt den großen Hauptrechner auf."""
    from wakeonlan import send_magic_packet
    send_magic_packet(mac_address)
    return "System: Magic Packet gesendet."

@tool
def fast_web_search(query: str):
    """Nutze dieses Tool für SCHNELLE Suchen (z.B. Wetter, einfache Fakten, aktuelle News)."""
    search = DuckDuckGoSearchRun()
    return search.invoke(query)

@tool
def deep_research_agent(topic: str):
    """Nutze dieses Tool NUR für KOMPLEXE Themen, die einen extrem langen und detaillierten Bericht erfordern.
    Dieses Tool startet einen Sub-Agenten, der stundenlang das Web analysiert."""
    return f"Ich habe den Deep Researcher beauftragt, einen umfassenden Bericht über '{topic}' zu erstellen. (Hinweis für Entwickler: Submodule muss noch konfiguriert werden)."


# --- GEHIRN (DER CHEF-AGENT) ---
llm = ChatLiteLLM(model="edge-gemma", base_url="http://litellm:4000")

# Wir bereiten die Variable für den Agenten vor, füllen sie aber erst beim Server-Start
agent_executor = None


# --- LEBENSZYKLUS DES SERVERS (Hier wird die MCP-Verbindung hergestellt) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_executor

    print("Starte MCP Verbindung zu OpenCode...")
    # Definiert, wie der npm-Prozess gestartet wird
    server_parameters = StdioServerParameters(
        command="npx",
        args=["-y", "opencode-mcp"],
        env={
            **os.environ,
            # Verweist auf den neuen OpenCode-Docker-Container (muss in der docker-compose.yml sein!)
            "OPENCODE_BASE_URL": "http://opencode-server:4096"
        }
    )

    # Baut die Verbindung zum MCP-Server auf
    async with stdio_client(server_parameters) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Alle 79 Tools von OpenCode dynamisch abrufen
            opencode_tools = await load_mcp_tools(session)
            print(f"✅ Erfolgreich {len(opencode_tools)} OpenCode MCP Tools geladen!")

            # JETZT erstellen wir den Agenten, übergeben das Gedächtnis (store) und bündeln ALLE Tools an einem Ort
            agent_executor = create_deep_agent(
                model=llm,
                tools=[
                    wake_up_big_pc,
                    fast_web_search,
                    deep_research_agent,
                    # --- LANGMEM TOOLS HINZUFÜGEN ---
                    create_manage_memory_tool(namespace=("memories",)),
                    create_search_memory_tool(namespace=("memories",))
                ] + opencode_tools,
                checkpointer=checkpointer,
                store=store, # <--- Dein Langzeitgedächtnis ist hiermit aktiv!
                system_prompt="""You are the chief agent of this system.
                Decide wisely: For quick questions, use 'fast_web_search'.
                If the user requests an in-depth analysis, delegate it to the 'deep_research_agent'.
                You also have access to extremely powerful OpenCode tools (such as opencode_run, write, bash).
                Use your memory tools (manage_memory, search_memory) to actively remember the user's preferences and important facts across sessions!"""
            )

            # Ab hier läuft der Server ganz normal, die Verbindung bleibt aber im Hintergrund offen
            yield


# FastAPI mit dem Lifespan (Start-Routine) initialisieren
app = FastAPI(lifespan=lifespan)

# --- API BRIDGE ---
class MyEnterpriseAgentFactory(BaseAgentFactory):
    def create_agent(self, create_agent_dto):
        return agent_executor

bridge = LangchainOpenaiApiBridgeFastapi(
    agent_factory=MyEnterpriseAgentFactory()
)
app.include_router(bridge.router)
