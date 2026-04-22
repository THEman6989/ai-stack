from fastapi import FastAPI
from pydantic import BaseModel
import time

from langchain_core.tools import tool
from langchain_community.chat_models import ChatLiteLLM
from deepagents import create_deep_agent  # <--- HIER IST DIE NEUE MAGIE
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI()

# ==========================================
# 1. DEIN EIGENES TOOL
# ==========================================
@tool
def wake_up_big_pc(mac_address: str = "AA:BB:CC:DD:EE:FF"):
    """
    Weckt den großen Hauptrechner auf.
    Nutze dieses Tool, wenn der User dich darum bittet oder du mehr Rechenpower brauchst.
    """
    from wakeonlan import send_magic_packet
    print(">>> AGENT NUTZT TOOL: Sende Magic Packet...")
    send_magic_packet(mac_address)
    time.sleep(2)
    return "System-Antwort: Der große PC wurde erfolgreich aufgeweckt."

# ==========================================
# 2. DEIN "DEEP AGENT"
# ==========================================
llm = ChatLiteLLM(model="edge-gemma", base_url="http://litellm:4000")

# Deep Agents baut den LangGraph automatisch mit allen Spezial-Funktionen zusammen
agent_executor = create_deep_agent(
    model=llm,
    tools=[wake_up_big_pc],
    system_prompt="Du bist ein genialer System-Agent. Du kannst planen, das Dateisystem nutzen und PCs aufwecken."
)

# ==========================================
# 3. DIE SCHNITTSTELLE ZU LIBRECHAT
# ==========================================
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]
    model: str

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    lc_messages = []

    for msg in request.messages:
        if msg.role == "user":
            lc_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            lc_messages.append(AIMessage(content=msg.content))

    print(">>> DEEP AGENT DENKT NACH...")
    # Da create_deep_agent einen LangGraph zurückgibt, rufen wir ihn genauso auf!
    result = agent_executor.invoke({"messages": lc_messages})

    final_reply = result["messages"][-1].content
    print(">>> AGENT ANTWORTET:", final_reply)

    return {
        "id": "chatcmpl-deepagents",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": final_reply},
            "finish_reason": "stop"
        }]
    }
