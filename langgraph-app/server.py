import time
from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient
from redis import Redis

# LangChain & LangGraph Enterprise-Komponenten
from langchain_core.tools import tool
from langchain_litellm import ChatLiteLLM
from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_redis import RedisCache
from langchain.globals import set_llm_cache

app = FastAPI()

# --- INFRASTRUKTUR ---
# Geteilte MongoDB (Checkpointing)
mongo_client = MongoClient("mongodb://mongodb:27017")
checkpointer = MongoDBSaver(mongo_client, db_name="langgraph_memory")

# Redis Semantic Cache (Turbo)
redis_client = Redis(host="redis", port=6379)
set_llm_cache(RedisCache(redis_client))

# --- TOOLS ---
@tool
def wake_up_big_pc(mac_address: str = "AA:BB:CC:DD:EE:FF"):
    """Weckt den großen Hauptrechner auf."""
    from wakeonlan import send_magic_packet
    send_magic_packet(mac_address)
    return "System: Magic Packet gesendet."

@tool
def deep_web_research(query: str):
    """Sucht im Web nach aktuellen Informationen für Deep Research."""
    search = DuckDuckGoSearchRun()
    return search.invoke(query)

# --- GEHIRN ---
llm = ChatLiteLLM(model="edge-gemma", base_url="http://litellm:4000")

agent_executor = create_deep_agent(
    model=llm,
    tools=[wake_up_big_pc, deep_web_research],
    checkpointer=checkpointer,
    system_prompt="Du bist ein Deep-Research-Agent. Nutze deine Tools für PC-Steuerung und Websuche."
)

# --- API ---
class ChatRequest(BaseModel):
    messages: list[dict]
    model: str

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    user_msg = request.messages[-1]["content"]
    config = {"configurable": {"thread_id": "shared-enterprise-thread"}}
    
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
