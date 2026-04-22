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

# Optional: Hier würden wir später den Deep Researcher importieren
# import sys
# sys.path.append("/app/local_deep_researcher")
# from graph import graph as deep_researcher_graph

app = FastAPI()

# --- INFRASTRUKTUR ---
mongo_client = MongoClient("mongodb://mongodb:27017")
checkpointer = MongoDBSaver(mongo_client, db_name="langgraph_memory")

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

    # ⚠️ PLATZHALTER: Hier verknüpfen wir gleich den Code aus dem Submodule!
    # Wir machen das in zwei Schritten, damit dein Container jetzt nicht abstürzt.
    return f"Ich habe den Deep Researcher beauftragt, einen umfassenden Bericht über '{topic}' zu erstellen. (Hinweis für Entwickler: Submodule muss noch konfiguriert werden)."


# --- GEHIRN (DER CHEF-AGENT) ---
llm = ChatLiteLLM(model="edge-gemma", base_url="http://litellm:4000")

agent_executor = create_deep_agent(
    model=llm,
    # NEU: Der Chef hat jetzt 3 Werkzeuge!
    tools=[wake_up_big_pc, fast_web_search, deep_research_agent],
    checkpointer=checkpointer,
    system_prompt="""Du bist der Chef-Agent dieses Systems.
    Entscheide weise: Für kurze Fragen nutzt du 'fast_web_search'.
    Wenn der User aber eine tiefgehende Analyse, ein Essay oder einen wissenschaftlichen Bericht will, delegierst du das an den 'deep_research_agent'."""
)

# --- API BRIDGE ---
class MyEnterpriseAgentFactory(BaseAgentFactory):
    def create_agent(self, create_agent_dto):
        return agent_executor

bridge = LangchainOpenaiApiBridgeFastapi(
    agent_factory=MyEnterpriseAgentFactory()
)
app.include_router(bridge.router)




# Dein neuer permanenter Speicher
# Wir nutzen dieselbe MongoDB wie für die Checkpoints, aber eine andere Collection
store = MongoDBSaver(mongo_client, db_name="langgraph_memory", collection_name="long_term_store")

# Beim Erstellen des Agenten übergeben wir jetzt den Store
agent_executor = create_deep_agent(
    model=llm,
    tools=[wake_up_big_pc, fast_web_search, deep_research_agent],
    checkpointer=checkpointer,
    store=store, # <--- HIER wird das Langzeitgedächtnis aktiviert!
    system_prompt="Du hast Zugriff auf ein Langzeitgedächtnis. Merke dir wichtige Details über den Nutzer."
)
