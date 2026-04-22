import time
from fastapi import FastAPI
from pymongo import MongoClient
from redis import Redis

# Deine Enterprise-Tools (Wie bisher)
from langchain_core.tools import tool
from langchain_litellm import ChatLiteLLM
from deepagents import create_deep_agent
from langgraph.checkpoint.mongodb import MongoDBSaver

# ----------------------------------------------------
# NEU: Importiere die Bridge aus dem lokalen Ordner!
# ----------------------------------------------------
from langchain_fastapi_chat_completion.fastapi.langchain_openai_api_bridge_fastapi import LangchainOpenaiApiBridgeFastapi
from langchain_fastapi_chat_completion.core.base_agent_factory import BaseAgentFactory
from langchain_fastapi_chat_completion.chat_model_adapter.base_openai_compatible_chat_model_adapter import BaseOpenaiCompatibleChatModelAdapter

app = FastAPI()

# --- INFRASTRUKTUR & GEHIRN (Bleibt gleich!) ---
mongo_client = MongoClient("mongodb://mongodb:27017")
checkpointer = MongoDBSaver(mongo_client, db_name="langgraph_memory")

@tool
def wake_up_big_pc(mac_address: str = "AA:BB:CC:DD:EE:FF"):
    """Weckt den großen PC."""
    return "PC geweckt."

llm = ChatLiteLLM(model="edge-gemma", base_url="http://litellm:4000")

agent_executor = create_deep_agent(
    model=llm,
    tools=[wake_up_big_pc],
    checkpointer=checkpointer,
    system_prompt="Du bist ein Deep-Research-Agent."
)

# ----------------------------------------------------
# NEU: Die Integration der importierten API-Bridge
# ----------------------------------------------------
# Die Bridge zwingt uns, eine "Factory" zu schreiben. 
# Sie ruft diese Factory auf, wenn LibreChat eine Anfrage schickt.
class MyEnterpriseAgentFactory(BaseAgentFactory):
    def create_agent(self, create_agent_dto):
        # Hier übergeben wir unseren fertigen LangGraph-Agenten an die Bridge
        return agent_executor

class MyModelAdapter(BaseOpenaiCompatibleChatModelAdapter):
    # Die Bridge braucht auch einen Adapter, um das Modell-Format zu verstehen
    def adapt(self, agent_response):
        return agent_response

# Wir initialisieren die Bridge mit unserer Factory
bridge = LangchainOpenaiApiBridgeFastapi(
    agent_factory=MyEnterpriseAgentFactory(),
    chat_model_adapter=MyModelAdapter()
)

# Wir hängen die Routen (/v1/chat/completions) automatisch an FastAPI an!
app.include_router(bridge.router)