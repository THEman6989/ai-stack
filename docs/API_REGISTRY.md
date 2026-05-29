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

## REST API Routes — LangGraph Container

Alle folgenden APIs leben im `langgraph-app/` Codebase und bauen aus
demselben Docker-Image (`langgraph-app/Dockerfile`), laufen aber als
separate Container.

Route-Suche im Code:
```bash
grep -n '@app\.\(get\|post\|delete\|put\)' langgraph-app/bridge_server.py
grep -n '@app\.\(get\|post\|delete\|put\)' langgraph-app/media_server.py
grep -n '@app\.\(get\|post\|delete\|put\)' langgraph-app/hermes_orch_server.py
grep -n '@router\.\(get\|post\)' langgraph-app/queue_ingest.py
```

---

### API Bridge (`api-bridge:8123`)

Quelle: `langgraph-app/bridge_server.py` (4312 Zeilen)

OpenAI-kompatibler Bridge. Verbindet LibreChat/OpenWebUI → LangGraph Agent.

#### Health & Observability

| Methode | Pfad | Zeile | Beschreibung |
|---------|------|-------|-------------|
| `GET` | `/health` | L3911 | Health Check: Bridge + LangGraph Erreichbarkeit |
| `GET` | `/health/llm-generation` | L3924 | LLM-Generation Health (echter Modell-Call) |
| `GET` | `/_alpharavis/bridge-observer` | L3955 | Stream Observer für Debug/Diagnose |
| `DELETE` | `/_alpharavis/bridge-observer` | L3967 | Observer zurücksetzen |

#### LangGraph Tool (Hermes→AlphaRavis)

| Methode | Pfad | Zeile | Beschreibung |
|---------|------|-------|-------------|
| `POST` | `/tools/langgraph/run` | L3973 | LangGraph Run via Hermes. Erfordert `explicit_user_request=true` |

#### OpenAI Chat Completions API

| Methode | Pfad | Zeile | Beschreibung |
|---------|------|-------|-------------|
| `GET` | `/v1/models` | L4037 | Verfügbare Modelle (`my-agent`, `server-model-manager`) |
| `POST` | `/v1/chat/completions` | L4238 | Chat Completions (stream/non-stream). Parameter: `model`, `messages`, `stream`, `user`, `x-user-id` |
| `POST` | `/v1/responses` | L4053 | Responses API (stateful, `previous_response_id`). Stream via `stream=true` |
| `POST` | `/v1/responses/compact` | L4139 | Response komprimieren (Token-Spar-Modus) |
| `POST` | `/v1/responses/input_tokens` | L4150 | Input-Token-Zählung für Response |
| `GET` | `/v1/responses/{response_id}` | L4171 | Gespeicherte Response abrufen |
| `GET` | `/v1/responses/{response_id}/input_items` | L4186 | Input-Items einer Response |
| `POST` | `/v1/responses/{response_id}/cancel` | L4211 | Laufende Response abbrechen |
| `DELETE` | `/v1/responses/{response_id}` | L4228 | Response löschen |

#### Queue Ingest (ARM Gateway)

Quelle: `langgraph-app/queue_ingest.py` (230 Zeilen)

Im Bridge eingebunden via `app.include_router(queue_ingest_router)`.

| Methode | Pfad | Zeile | Beschreibung |
|---------|------|-------|-------------|
| `POST` | `/api/queue/ingest` | L61 | Offline-Queue vom ARM Gateway flushen. Idempotent (message_id). Leitet an LangGraph weiter. Auth: `AI_STACK_QUEUE_INGEST_TOKEN` |

---

### Media Gallery (`media-gallery:8130`)

Quelle: `langgraph-app/media_server.py` (1918 Zeilen)

Medien-Assets, ComfyUI-Proxy, Office-Dokumente, Asset-Registry.

#### ComfyUI Proxy

| Methode | Pfad | Zeile | Beschreibung |
|---------|------|-------|-------------|
| `GET` | `/comfyui/status` | L220 | ComfyUI System-Status |
| `GET` | `/comfyui/queue` | L225 | ComfyUI Job-Queue |
| `GET` | `/comfyui/models/{folder}` | L230 | Modelle in Ordner (checkpoints, loras, vae, ...) |
| `GET` | `/comfyui/history/{prompt_id}` | L234 | Prompt-History (Output-URLs, Metadata) |
| `POST` | `/comfyui/preflight` | L239 | Workflow-Validierung (Preflight) |
| `POST` | `/comfyui/prompt` | L244 | Workflow ausführen (gated: `ALPHARAVIS_ENABLE_COMFYUI_WORKFLOW_SUBMIT`) |
| `GET` | `/comfyui/view` | L249 | Output-Bild/Video/Asset über internen Transport |
| `POST` | `/comfyui/queue/clear` | L266 | Queue leeren |
| `POST` | `/comfyui/interrupt` | L270 | Aktuelle Ausführung abbrechen |
| `POST` | `/comfyui/free` | L274 | Speicher freigeben |
| `POST` | `/comfyui/outputs/register` | L1538 | Generierte Outputs in Media Gallery registrieren |
| `POST` | `/comfyui/history/{prompt_id}/register` | L1549 | History-Prompt-Outputs registrieren |

#### Office (via OfficeCLI)

| Methode | Pfad | Zeile | Beschreibung |
|---------|------|-------|-------------|
| `GET` | `/office/files` | L1281 | Office-Output-Dateien auflisten |
| `GET` | `/office/templates` | L1286 | Office-Vorlagen auflisten |
| `POST` | `/office/upload` | L1292 | Dokument hochladen |
| `POST` | `/office/template-merge` | L1305 | Template-Merge-Plan |
| `POST` | `/office/validate` | L1310 | Dokument validieren |
| `POST` | `/office/batch` | L1315 | Batch-Job erstellen |
| `POST` | `/office/roundtrip` | L1320 | Roundtrip (ODF↔DOCX) |
| `POST` | `/office/preview` | L1325 | Dokument-Vorschau |
| `POST` | `/office/repair` | L1330 | Dokument reparieren |
| `POST` | `/office/watch/start` | L1335 | File-Watcher starten |
| `POST` | `/office/watch/stop` | L1340 | File-Watcher stoppen |
| `GET` | `/office/watch/status` | L1345 | Watcher-Status |
| `GET` | `/office/blueprints` | L1350 | Office-Blueprints auflisten |
| `GET` | `/office/blueprints/suggest` | L1355 | Blueprint-Vorschläge |
| `POST` | `/office/blueprints/create` | L1360 | Blueprint erstellen |
| `GET` | `/office/validation-results` | L1365 | Validierungsergebnisse abrufen |
| `POST` | `/office/validation-results` | L1375 | Ergebnis speichern |
| `GET` | `/office/batch/jobs` | L1380 | Batch-Jobs auflisten |
| `POST` | `/office/batch/jobs` | L1391 | Batch-Job erstellen |
| `GET` | `/office/batch/jobs/{job_id}` | L1396 | Einzelnen Job abrufen |
| `POST` | `/office/batch/jobs/{job_id}/progress` | L1401 | Job-Fortschritt updaten |
| `POST` | `/office/batch/jobs/update` | L1412 | Job aktualisieren |
| `GET` | `/office/templates/placeholders` | L1423 | Template-Platzhalter |
| `POST` | `/office/templates/merge-form` | L1428 | Merge-Formular |

#### Asset Registry

| Methode | Pfad | Zeile | Beschreibung |
|---------|------|-------|-------------|
| `POST` | `/assets/register` | L1585 | Asset registrieren (ID, URL, Metadata) |
| `POST` | `/assets/upload` | L1672 | Asset hochladen (Multipart) |
| `POST` | `/api/assets/upload` | L1687 | JSON-API-Upload (Agenten, Frontends). Returns `{asset_id, url, ...}` |
| `GET` | `/assets` | L1727 | Alle Assets auflisten |
| `GET` | `/assets/resolve` | L1752 | Asset via query auflösen |

#### Sonstiges

| Methode | Pfad | Zeile | Beschreibung |
|---------|------|-------|-------------|
| `GET` | `/` | L215 | Redirect zu `/gallery` |
| `GET` | `/gallery` | L1774 | Media Gallery HTML-Ansicht |
| `GET` | `/health` | L1567 | Health Check |
| `GET` | `/favicon.svg` | L279 | Favicon |

#### Static Mounts

| Mount | Pfad | Verzeichnis |
|-------|------|-------------|
| `/media` | Statische Dateien | `MEDIA_ROOT` (default: `/media-data`) |
| `/office-output` | Office-Dateien | `OFFICE_OUTPUT_ROOT` (default: `/workspace/office-output`) |

---

### Hermes Orchestrator (`hermes-orch:8650`)

Quelle: `langgraph-app/hermes_orch_server.py` (91 Zeilen)

Streaming-Relay: Pre-loaded AlphaRavis Context → Hermes Agent SSE.

| Methode | Pfad | Zeile | Beschreibung |
|---------|------|-------|-------------|
| `GET` | `/health` | L44 | Health Check |
| `POST` | `/hermes/stream` | L50 | SSE-Stream. Body: `{message, system_prompt?, max_output_chars?}`. Pre-loaded Memory/RAG/Skills/Sessions → Hermes :8642 → SSE-Stream zurück. Speichert Output als AlphaRavis-Artefakt. |

---

### Service Dashboard (`service-dashboard:8090`)

Quelle: `service_redirector_server.py` (stdlib http.server, keine FastAPI-Routen)

| Methode | Pfad | Beschreibung |
|---------|------|-------------|
| `GET` | `/` | Landing Page: Service-Karten, Links, Status |
| `GET` | `/settings` | Settings-UI (mobile-first, PWA). Runtime/Permanent-Override. |
| `POST` | `/api/register-gateway` | ARM Gateway Auto-Registrierung (alle 5 Min) |

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

Quelle: `helper-repos/ubuntu-llama-manager/docs/api.md`

Öffentliche Endpunkte (kein Auth):
```
GET  /health                          Health-Check
GET  /status                          Service-Status (GPU, llama, power)
GET  /models                          Modell-Scan
GET  /models/{id}                     Einzelnes Modell
GET  /llama/status                    Llama-Service-Status
GET  /llama/config                    Llama-Konfiguration
GET  /llama-secondary/status          Secondary-Status
GET  /llama-secondary/config          Secondary-Konfiguration
GET  /llama/instances                 Alle llama.cpp-Instanzen
GET  /llama/instances/{id}            Einzelne Instanz
GET  /reboot/status                   Auto-Reboot-Status
GET  /esp/status                      ESP-Status
GET  /esp/control                     ESP-Web-Control (HTML)
GET  /diagnostics/gpu                 GPU-Diagnose
POST /esp/heartbeat                   ESP-Heartbeat
```

Geschützte Endpunkte (`Authorization: Bearer <API_TOKEN>`):

```
POST /llama/start                     llama.cpp starten
POST /llama/stop                      llama.cpp stoppen
POST /llama/restart                   llama.cpp neustarten
POST /llama/config                    Config patchen
POST /llama/force-kill                Hartes kill -9
POST /llama/switch-model              Modell wechseln (preserved flags)
POST /llama-secondary/start           Secondary starten
POST /llama-secondary/stop            Secondary stoppen
POST /llama-secondary/restart         Secondary neustarten
POST /llama-secondary/config          Secondary Config
POST /llama/instances/{id}/config     Instanz-Konfiguration patchen
POST /reboot/enable                   Auto-Reboot aktivieren
POST /reboot/disable                  Auto-Reboot deaktivieren
POST /reboot/now                      Jetzt rebooten
POST /power/shutdown                  systemctl poweroff (kein ESP-Cycle)
POST /diagnostics/handle-gpu-fault    GPU-Fehler behandeln
POST /ai-stack/diagnose-llama         Nur Diagnose (kein Kill/Restart)
POST /ai-stack/llama-no-response      Recovery bei hängendem Llama
POST /recovery/llama-no-response      Alias für Rückwärtskompatibilität
POST /esp/action                      ESP-Aktion ausführen
POST /esp/cancel                      ESP-Aktion abbrechen
POST /esp/request-power-cycle         Power-Cycle via ESP
POST /esp/request-power-on            Power-On via ESP
POST /esp/request-power-off           Power-Off via ESP
```

---

## Hermes Agent API (`hermes-agent:8642`)

Quelle: `hermes-agent/gateway/platforms/api_server.py` (3524 Zeilen)

OpenAI-kompatibler API-Server. Beliebiges Frontend kann verbinden.

| Methode | Pfad | Beschreibung |
|---------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat Completions (stateless). Header: `X-Hermes-Session-Id`, `X-Hermes-Session-Key` |
| `POST` | `/v1/responses` | Responses API (stateful via `previous_response_id`) |
| `GET` | `/v1/responses/{response_id}` | Response abrufen |
| `DELETE` | `/v1/responses/{response_id}` | Response löschen |
| `GET` | `/v1/models` | Verfügbare Modelle (`hermes-agent`) |
| `GET` | `/v1/capabilities` | Machine-readable Capabilities für UIs |
| `POST` | `/v1/runs` | Run starten → `run_id` (202) |
| `GET` | `/v1/runs/{run_id}` | Run-Status |
| `GET` | `/v1/runs/{run_id}/events` | SSE-Stream: Lifecycle-Events |
| `POST` | `/v1/runs/{run_id}/approval` | Pending Approval auflösen |
| `POST` | `/v1/runs/{run_id}/stop` | Laufenden Run abbrechen |
| `GET` | `/health` | Health Check |
| `GET` | `/health/detailed` | Rich Status (Cross-Container Dashboard) |

Hermes API Base: `http://host.docker.internal:8642/v1` (aus Docker-Containern)

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
