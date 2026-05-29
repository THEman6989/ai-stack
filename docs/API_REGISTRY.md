# AlphaRavis API Registry

Zentrales Verzeichnis aller API-Endpoints, Services und Ports im
AlphaRavis AI Stack. Eine Quelle der Wahrheit — kein Suchen in `.env`,
Docker-Compose-Files oder Code.

Zuletzt aktualisiert: 2026-05-29

---

## Docker Services (intern)

Diese Services laufen in Docker-Containern auf dem AI-Stack-Host.
Interne Hostnames funktionieren nur innerhalb des Docker-Netzwerks.
Von außen (localhost) sind sie über die gemappten Ports erreichbar.

| Service | Docker Hostname | Port (intern) | Port (extern) | Beschreibung |
|---------|----------------|---------------|---------------|-------------|
| LangGraph API | `langgraph-api` | 2024 | 2024 | Agent Graph Runtime, DeepAgents Swarm |
| API Bridge | `api-bridge` | 8123 | 8123 | OpenAI-kompatibler Bridge (`/v1/chat/completions`, `/v1/responses`) |
| LiteLLM | `litellm` | 4000 | 4000 | Model-Gateway, routed zu llama.cpp, Ollama |
| Hermes Agent | `hermes-agent` | 8642 | 8642 | Coding/System-Spezialist, SSE-Streaming |
| Hermes WebUI | `hermes-webui` | 8643 | 8643 | Browser-UI für Hermes |
| Hermes Orchestrator | `hermes-orch` | 8650 | 8650 | Hermes Task-Orchestrierung, Media-Routing |
| Pixelle MCP | `pixelle` | 9004 | 9004 | MCP-Server für Bildgenerierung (ComfyUI) |
| RAG API | `rag_api` | 8000 | 8000 | Dokumenten-Retrieval, Embedding-Suche |
| Media Gallery | `media-gallery` | 8130 | 8130 | Medien-Assets, Office-Output, Analyse-Cache |
| LibreChat | `librechat` | 3080 | 3080 | Chat-UI (verbindet sich zu Bridge oder Hermes) |
| OpenWebUI | `openwebui` | 8080 | 3090 | Alternative Chat-UI mit Pipeline-Mode |
| DeepAgents UI | — | 3000 | 3000 | AlphaRavis Custom Agent UI (Next.js, geforkt) |
| Agent Custom UI | — | 3000 | 3001 | Alternative Agent-UI |
| Service Dashboard | `service-dashboard` | 8090 | 8090 | Landing Page, Status, Settings |
| Bridge Test UI | `bridge-test-ui` | 8140 | 8140 | Bridge Smoke-Test Web-UI |
| Redis | `redis` | 6379 | 6379 | Session Cache, Queue |
| MongoDB | `mongodb` | 27017 | 27017 | LibreChat Persistenz |
| VectorDB | `vectordb` | 5432 | 5432 | PostgreSQL + pgvector (Embeddings, Memory) |
| OnlyOffice | `onlyoffice` | 80 | 8088 | DocumentServer für ODF→DOCX-Konvertierung |

### Profile / Feature Gates

Nicht alle Services sind immer aktiv. Feature-Gates via Compose-Profile:

```bash
# Standard (Chat + Bridge + Agenten)
docker compose up -d

# Mit OpenWebUI
docker compose --profile openwebui up -d openwebui

# Mit Hermes Dashboard
docker compose --profile hermes-dashboard up -d hermes-dashboard

# Mit OnlyOffice DocumentServer
docker compose --profile odf up -d onlyoffice
```

---

## Externe Dienste — Llama-PC (192.168.178.153)

Der Llama-PC hostet die LLM-Inferenz (llama.cpp) und den Ubuntu Llama Manager.
Er wird über den ESP Power Controller (192.168.178.113) ein-/ausgeschaltet.

| Service | IP:Port | Env-Var | Beschreibung |
|---------|---------|---------|-------------|
| **Ubuntu Llama Manager** | `192.168.178.153:8099` | `ALPHARAVIS_UBUNTU_LLAMA_MANAGER_URL` | Service-Manager für llama.cpp-Instanzen, Power-Management, GPU-Monitoring |
| **BigBoss** (llama.cpp) | `192.168.178.153:8033` | `BIG_BOSS_API_BASE` | Primäre LLM-Inferenz: Qwen3.6-35B-A3B-MTP (MXFP4_MOE), ctx=262k, np=1 |
| **Qwen2B** (llama.cpp) | `192.168.178.153:8001` | — | Sekundäre LLM-Inferenz: Qwen3.5-2B (Q4_1), ctx=60k, np=4 |
| **Ollama** | `192.168.178.153:11434` | `ALPHARAVIS_OLLAMA_BASE_URL` | Ollama-Server für Fallback-Modelle und Embeddings |
| **ESP Power Controller** | `192.168.178.113:80` | `ALPHARAVIS_UBUNTU_LLAMA_ESP_IP` | Physischer Power-Button-Drücker (ESP32/ESP8266) |

### ESP Power Controller API

```
POST /action  {"action":"power-on|power-off|power-cycle|reset",
               "reason":"...", "requested_by":"alpharavis"}
               Auth: Bearer <ALPHARAVIS_UBUNTU_LLAMA_ESP_API_KEY>

POST /cancel  {}  (bricht ausstehende Aktion ab)

GET  /health  (Status)
```

Power-On-Ablauf: ESP wartet 30s → drückt Button 1s → wartet 20s → PC bootet.
Total: ~51s bis Power-Knopf-Release, dann + Boot-Zeit (~60-90s).

### Ubuntu Llama Manager API (Auswahl)

```
GET  /health                          Health-Check
GET  /status                          Service-Status (GPU, llama, power)
GET  /llama/instances                 Alle llama.cpp-Instanzen
POST /llama/start                     llama.cpp starten
POST /llama/stop                      llama.cpp stoppen
POST /esp/action                      ESP-Power-Action (via Manager)
POST /llama/instances/primary/config  Konfiguration patchen (Modell, ctx, np)
POST /ai-stack/llama-no-response      Recovery bei hängendem llama-Server
```

---

## Externe Dienste — Andere PCs

| PC | IP | Env-Var | Dienste |
|----|----|---------|---------|
| **Comfy Server** | `192.168.178.50` | `REMOTE_PCS` (`comfy_server`) | ComfyUI für Bildgenerierung |
| **Unraid Server** | `100.126.202.107` (Tailscale) | — | Ollama-Instanz |

### ComfyUI

- Lokal (AI-Stack-Host): `unix:///workspace/runtime/comfyui.sock` (`ALPHARAVIS_COMFYUI_API_BASE`)
- Public URL: `http://localhost:8188`
- Via Pixelle MCP: Automatische Workflow-Submission
- Comfy Relay: `make comfyui-relay` (Docker→Host-Bridge)

### REMOTE_PCS (.env)

```json
{
  "main_pc":      {"ip": "192.168.178.140", "mac": "AA:BB:CC:DD:EE:FF"},
  "comfy_server": {"ip": "192.168.178.50",  "mac": "11:22:33:44:55:66"}
}
```

> **Achtung:** `main_pc` IP in REMOTE_PCS ist `.140`, aber der Llama-PC ist
> tatsächlich auf `.153`. Die MAC ist ein Platzhalter. Für WOL die echte MAC
> eintragen.

---

## Modell-Routing (LiteLLM)

Das Model-Gateway (LiteLLM, Port 4000) routet Model-Namen zu Backends:

| LiteLLM Model-Name | Backend | API-Base | Hardware |
|-------------------|---------|----------|----------|
| `big-boss` | llama.cpp | `192.168.178.153:8033/v1` | Qwen3.6-35B-A3B (MXFP4) |
| `edge-gemma` | Ollama | `192.168.178.140:11434/v1` ⚠️ | Gemma4:e2b (Fallback) |

> ⚠️ `EDGE_GEMMA_API_BASE` zeigt noch auf `.140` — muss auf `.153` korrigiert
> werden, falls Ollama auf dem Llama-PC läuft.

**AlphaRavis Model-Konfiguration:**
- `ALPHARAVIS_MODEL=openai/big-boss` — primäres Agent-Modell
- `BRIDGE_LLM_HEALTH_MODEL=big-boss` — Bridge Health-Check
- `BRIDGE_LLM_HEALTH_FALLBACK_MODEL=edge-gemma` — Fallback

---

## DeepAgents UI

Custom AlphaRavis Agent UI, basierend auf Next.js 16 (geforkt von LangChain
DeepAgents UI). Läuft auf `localhost:3000`.

### UI-Layout

- **Chat-Tab**: Haupt-Chat-Interface (niemals antasten)
- **Coding-Tab**: Hermes SSE-Streaming + Orchestrator-Status
- **≡ Overlay**: ComfyUI-Workflows, Pixelle-Jobs
- **BelowFold**: Office-Dokumente (nicht im Haupt-Viewport)

### Technische Details

- `react-resizable-panels` Panel: `inline overflow:hidden`
  - Fix: `overflow-y-auto min-h-0` auf Child, `flex-1` (nicht `h-full`)
- Viewport: `min-h-[calc(100vh-4rem)]`, Header innerhalb
- Hero-Sektion: `flex-1`
- Dockerfile: `NEXT_PUBLIC_*` benötigt `ARG`+`ENV` im Builder (Turbopack inline)
  - Ohne → `undefined` im Browser → fällt auf docker-interne URL zurück → broken

---

## ARM Gateway (DeepAgent ARM Gateway)

Leichtgewichtiges Always-On-Gateway für ARM64 SBCs (Radxa ROCK 2A).
Bietet dieselbe URL (`http://<board-ip>:3000`), egal ob der AI-Stack online ist.

### Installation

Repository: `deepagent-arm-gateway/` (gitignored in ai-stack)
Hardware: Radxa ROCK 2A, Armbian, Python FastAPI, SQLite

### Konfiguration (.env)

```
DEEPAGENT_TARGET_URL=http://192.168.1.100:3000     # AI-Stack-PC IP
AI_STACK_QUEUE_INGEST_URL=http://192.168.1.100:8123/api/queue/ingest
GATEWAY_MEDIA_BASE_URL=http://192.168.1.50:3000     # ROCK 2A eigene IP
WAKE_MODE=wol
WAKE_WOL_MAC=00:11:22:33:44:55
REDIRECTOR_URL=http://192.168.1.100:8090            # Service Dashboard
```

### API-Endpoints

| Endpoint | Beschreibung |
|----------|-------------|
| `GET /health` | Gateway Health (immer) |
| `GET /api/gateway/status` | Server-Status (online/offline) |
| `POST /api/gateway/message` | Message senden (+ optional File) |
| `GET /api/gateway/history?session_id=X` | Chat-Verlauf |
| `GET /api/gateway/media` | Media Gallery (Liste) |
| `GET /api/gateway/media/{id}` | Einzelne Mediendatei |
| `POST /api/gateway/wake` | WoL/HTTP-Wake |
| `POST /api/gateway/flush-queue` | Queue flushen (Admin) |

### AI-Stack Integration

- Queue-Ingest: `POST /api/queue/ingest` im `api-bridge` (implementiert in `queue_ingest.py`)
- Media-Flush: Bridge lädt Medien via HTTP → base64 → LangGraph
- Auto-Registrierung: Gateway meldet sich alle 5 Min am Service Dashboard (`POST /api/register-gateway`)

### Speicher-Limits

- Media: 2 GB (`GATEWAY_MEDIA_LIMIT_MB`), oldest-first cleanup
- Text-Queue: 3 GB (`GATEWAY_TEXT_LIMIT_MB`)
- SQLite WAL-Mode für SD-Karten

### Systemd-Services

```
deepagent-gateway.service       # Haupt-Gateway
deepagent-gateway-update.service # Auto-Update (git pull alle 30 Min)
deepagent-gateway-update.timer
deepagent-gateway-register.service # Dashboard-Registrierung
deepagent-gateway-register.timer
```

---

## Netzwerk-Übersicht

```
┌─────────────────────────────────────────────────────────────────┐
│ AI-Stack Host (Docker)                                          │
│                                                                 │
│  localhost:8123  ── Bridge ──► LangGraph (:2024)               │
│  localhost:4000  ── LiteLLM ──► externe Modelle                │
│  localhost:8642  ── Hermes Agent                                │
│  localhost:8643  ── Hermes WebUI                                │
│  localhost:3000  ── DeepAgents UI                               │
│  localhost:8090  ── Service Dashboard                           │
│  localhost:8130  ── Media Gallery                               │
│  localhost:8000  ── RAG API                                     │
│  localhost:9004  ── Pixelle MCP                                 │
│  localhost:3080  ── LibreChat                                   │
│  localhost:3090  ── OpenWebUI                                   │
│  localhost:8188  ── ComfyUI (via Relay)                        │
└─────────────────────────────────────────────────────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
┌──────────────────┐  ┌───────────────┐  ┌──────────────────┐
│ Llama-PC         │  │ Comfy Server  │  │ ARM Gateway      │
│ 192.168.178.153  │  │ 192.168.178.50│  │ (ROCK 2A)        │
│                  │  │               │  │                  │
│ :8033 BigBoss    │  │ ComfyUI       │  │ :3000 Gateway    │
│ :8001 Qwen2B     │  │               │  │ SQLite Queue     │
│ :8099 Llama Mgr  │  │               │  │ Media Cache      │
│ :11434 Ollama    │  │               │  │ WoL              │
└──────────────────┘  └───────────────┘  └──────────────────┘
          ▲
          │ ESP (192.168.178.113:80)
          │ Power Button Control
```

### Feste IPs (präferiert)

- Kein mDNS / `.local` — alle Services haben feste IPs
- ESP: `192.168.178.113`
- Llama-PC: `192.168.178.153`
- Comfy-PC: `192.168.178.50`
- ARM Gateway: `192.168.1.50` (Beispiel, tatsächliche IP konfigurieren)
- Tailscale: `100.126.202.107` (Unraid Server, Tailscale MagicDNS)

---

## Verwandte Dokumente

- `docs/ALPHARAVIS_ARCHITECTURE.md` — System-Design, Container-Rollen
- `docs/ALPHARAVIS_MODEL_MANAGEMENT.md` — Custom Model/Power-Management
- `docs/ALPHARAVIS_USAGE_NOTES.md` — Runtime-Verhalten, Flags
- `docs/HERMES_INTEGRATION.md` — Hermes Service, Delegation
- `docs/deepagent_gateway_plan.md` — ARM Gateway Planung
- `docs/deepagent_gateway_ai_stack_integration.md` — Gateway↔AI-Stack Integration
- `docs/MAKEFILE_README.md` — Makefile-Referenz
