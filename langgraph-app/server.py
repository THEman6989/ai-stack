import time
from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient

# LangChain & LangGraph Enterprise-Komponenten
from langchain_core.tools import tool
from langchain_litellm import ChatLiteLLM
from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_community.tools import DuckDuckGoSearchRun

# Caching & Memory Pakete
from langchain_redis import RedisCache
from langchain.globals import set_llm_cache
from redis import Redis

app = FastAPI()

# ==========================================
# 1. INFRASTRUKTUR: MONGODB & REDIS
# ==========================================
# MongoDB Setup (Geteilt mit LibreChat)
mongo_client = MongoClient("mongodb://mongodb:27017")
checkpointer = MongoDBSaver(mongo_client, db_name="langgraph_memory")

# Redis Semantic Cache Setup
# Dies spart Rechenpower bei identischen Fragen
redis_client = Redis(host="redis", port=6379)
set_llm_cache(RedisCache(redis_client))

# ==========================================
# 2. WERKZEUGE (INKL. RESEARCH)
# ==========================================
@tool
def wake_up_big_pc(mac_address: str = "AA:BB:CC:DD:EE:FF"):
    """Weckt den großen Hauptrechner per Magic Packet."""
    from wakeonlan import send_magic_packet
    send_magic_packet(mac_address)
    return "System: Magic Packet gesendet."

@tool
def deep_web_research(query: str):
    """Sucht im Web nach aktuellen Informationen."""
    search = DuckDuckGoSearchRun()
    return search.invoke(query)

# ==========================================
# 3. DAS GEHIRN (LITELLM PROXY)
# ==========================================
# Wir nutzen dein edge-gemma via LiteLLM Container
llm = ChatLiteLLM(model="edge-gemma", base_url="http://litellm:4000")

agent_executor = create_deep_agent(
    model=llm,
    tools=[wake_up_big_pc, deep_web_research],
    checkpointer=checkpointer,
    system_prompt="Du bist ein Deep-Research-Agent mit lokalem Gedächtnis."
)

# ==========================================
# 4. API FÜR LIBRECHAT & AGENT UI
# ==========================================
class ChatRequest(BaseModel):
    messages: list[dict]
    model: str

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    user_msg = request.messages[-1]["content"]
    
    # Der Checkpointer nutzt diese ID, um den Chat in MongoDB wiederzufinden
    config = {"configurable": {"thread_id": "global_user_thread"}}
    
    result = agent_executor.invoke(
        {"messages": [HumanMessage(content=user_msg)]}, 
        config=config
    )
    
    return {
        "id": "chatcmpl-enterprise",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": result["messages"][-1].content}}]
    }
