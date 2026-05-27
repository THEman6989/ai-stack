# ComfyUI Skill / Swarm-Agent / UI-Tab Integration

## Ziel

ComfyUI soll in AlphaRavis als eigener, feature-gegateter Swarm-Spezialist verfügbar sein und über einen dedizierten ComfyUI-Tab bedienbar werden. Der Tab und der Agent steuern eine bestehende ComfyUI-Instanz im LAN, typischerweise den ComfyPC (`REMOTE_PCS.comfy_server` oder `ALPHARAVIS_COMFYUI_API_BASE`). Pixelle bleibt weiterhin die einfache Bildgenerierungs-Schicht; ComfyUI wird die direkte Workflow-/Modell-/Queue-Schicht.

## Umsetzungsschritte

1. Backend-Client kapseln
   - Neuer `langgraph-app/comfyui_client.py` als kleiner REST-Adapter für ComfyUI.
   - Base-URL aus `ALPHARAVIS_COMFYUI_API_BASE`, fallback `ALPHARAVIS_COMFY_API_BASE`, fallback `ALPHARAVIS_COMFY_HEALTH_URL`, fallback `REMOTE_PCS.comfy_server.ip:8188`, fallback `127.0.0.1:8188`.
   - Keine Secrets und keine IP-Hardcodierung.

2. Swarm-Agent anbinden
   - Neuer Feature-Flag: `ALPHARAVIS_ENABLE_COMFYUI_AGENT=false`.
   - Neue Tools im LangGraph: Status, Queue, Modelle, Workflow submit, Prompt-History.
   - Direkter Workflow-Submit separat gegatet: `ALPHARAVIS_ENABLE_COMFYUI_WORKFLOW_SUBMIT=false`, weil fremde Workflows Custom-Node-Code ausführen können.
   - Neuer `comfyui_agent` im Swarm, Peer-Handoffs analog zum Office-Agent.

3. Toolset/Lazy-Loading
   - Neue Toolsets `comfyui/workflows` und `agent/comfyui` in `alpharavis_toolsets.py`.
   - Intent-Erkennung für ComfyUI, Workflow, checkpoint/model, queue, prompt_id.

4. Media-Gallery API für UI-Tab
   - Safe Status-/Queue-/Model-Proxy unter `/comfy/*`.
   - Optionaler submit endpoint nur wenn `ALPHARAVIS_ENABLE_COMFYUI_WORKFLOW_SUBMIT=true`.

5. DeepAgents UI
   - Neuer `ComfyPanel.tsx` mit Health/Queue/Model-Übersicht und Auftrag-Formular.
   - Auftrag wird direkt an `comfyui_agent` gesendet, wenn `NEXT_PUBLIC_COMFYUI_AGENT_ENABLED=true`.

6. Dokumentation/Tests
   - `.env(exaple)`, Docker env wiring, Open Tasks, Changes, Architecture/Usage aktualisieren.
   - Fokussierte Tests für Toolsets, Swarm-Gating, Media-Proxy-Helfer und UI-Tab-Quellen.
