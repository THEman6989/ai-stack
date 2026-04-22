from fastapi import FastAPI
from pydantic import BaseModel
import time
from wakeonlan import send_magic_packet

app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]
    model: str

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    # Hole die letzte Nachricht des Users aus LibreChat
    user_message = request.messages[-1].content

    # ----------------------------------------------------
    # HIER kommt später der richtige LangGraph/Gemma Code hin.
    # Für jetzt machen wir einen simplen Wake-On-LAN Test!
    # ----------------------------------------------------

    if "weck" in user_message.lower() or "pc" in user_message.lower():
        # Trage hier später die MAC-Adresse deines großen PCs ein
        send_magic_packet('AA:BB:CC:DD:EE:FF')
        time.sleep(2) # Kurze Pause zur Simulation
        reply = "Ich habe das Magic Packet gesendet! Der große PC fährt hoch."
    else:
        reply = f"Dein eigener Server läuft perfekt! Du hast gesagt: '{user_message}'. Sag 'weck den pc', um mein Tool zu testen."

    # Antwort im OpenAI-Format an LibreChat zurücksenden
    return {
        "id": "chatcmpl-custom",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": reply},
            "finish_reason": "stop"
        }]
    }
