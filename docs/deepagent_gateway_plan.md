# DeepAgent ARM Gateway — Architektur-Plan

> **Ziel:** Ein leichtgewichtiges Gateway (`deepagent-arm-gateway`) auf einem
> Radxa ROCK 2A ARM64 Always-On-Board, das dieselbe Browser-URL/IP:Port
> bereitstellt — unabhängig davon, ob der echte DeepAgent/LangGraph/AI-Server
> online oder offline ist.
>
> **Standalone-Repo:** `deepagent-arm-gateway/` (gitignored in ai-stack, eigenes Git-Repo)

## Entscheidung: Hybrid Reverse-Proxy + Fallback-UI (Ansatz D)

Die drei im Prompt genannten Ansätze wurden evaluiert:

| Ansatz | Bewertung |
|--------|-----------|
| A) Original DeepAgentUI-Frontend wiederverwenden | Die DeepAgents UI ist ein Next.js 16 App mit ~80 MB node_modules, yarn, LangGraph SDK. Läuft auf ROCK 2A theoretisch, aber: braucht lebenden LangGraph-Server, RAM-Verbrauch ist grenzwertig, kein Offline-Modus. |
| B) Eigener leichter Wrapper | Viel Aufwand. Müsste DeepAgents-Look komplett nachbauen und zwei Codebasen pflegen. Overkill für ein Gateway. |
| C) Nur Redirector + simple Fake-Seite | Zu einfach. Der Prompt verlangt Chat-Eingaben, Message-Queue, History, Media-Uploads — mehr als ein Redirector. |
| **D) Hybrid: Reverse Proxy + leichtgewichtige Fallback-UI** | **Gewählt.** Transparenter Pass-Through zum echten DeepAgentUI wenn online; eine minimalistische HTML/JS Single-Page-App als Fallback wenn offline. |

### Warum D optimal ist

1. **Normalbetrieb**: Der Nutzer bekommt die volle DeepAgents UI, Transparenz durch Reverse-Proxy
2. **Offline-Modus**: Leichte statische HTML/JS-Seite (kein Node.js, kein Build-Step)
3. **Selbe URL**: Der Port 3000 bleibt immer derselbe, egal ob online oder offline
4. **Queue**: Messages aus dem Offline-Modus werden in SQLite gespeichert und beim Online-Gehen an den AI-Stack geflusht
5. **ARM64-freundlich**: Python FastAPI (kein Node.js), SQLite, minimale Dependencies
6. **SD-Karten-schonend**: SQLite mit WAL-Mode, keine ständigen Writes, Logs limitiert

## Architektur

```
Browser/Handy (immer: http://<ROCK2A-IP>:3000)
               │
               ▼
    ┌──────────────────────────┐
    │   DeepAgent Gateway      │
    │   (Python FastAPI :3000) │
    │                          │
    │  ┌────────────────────┐  │
    │  │ Health Check Loop   │  │
    │  │ (pollt Target alle  │  │
    │  │  10s via /health)   │  │
    │  └────────┬───────────┘  │
    │           │               │
    │     online │ offline      │
    │  ┌────────┴────────┐     │
    │  │ Reverse Proxy   │     │
    │  │ (httpx pass-    │     │
    │  │  through)       │     │
    │  └────────┬────────┘     │
    │           │               │
    └───────────┼───────────────┘
                │
    ┌───────────┴───────────────┐
    │  Echter DeepAgentUI       │
    │  (Next.js :3000)          │
    │  + LangGraph API :2024    │
    │  + AI-Stack (anderer PC)  │
    └───────────────────────────┘
```

### Fallback-Modus (offline)

```
Browser/Handy → Gateway :3000
  GET  /           → static/fallback.html (Chat-UI)
  POST /api/gateway/message  → SQLite queue
  GET  /api/gateway/history  → SQLite messages
  GET  /api/gateway/status   → {"status": "offline", ...}
  POST /api/gateway/wake     → WoL/HTTP-Wake triggern
```

### Online-Modus

```
Browser/Handy → Gateway :3000
  ALLES           → Reverse Proxy → Echter DeepAgentUI :3000
  (inkl. WebSockets für Streaming)
```

## Gateway-Komponenten

### 1. Python FastAPI App (`deepagent_gateway/`)

- `main.py`: FastAPI-App, Routing, Health-Check-Loop
- `proxy.py`: Reverse-Proxy mit httpx (pass-through inkl. WebSockets)
- `db.py`: SQLite via aiosqlite, Schema-Migrationen
- `queue.py`: Message-Queue-Logik (enqueue, flush, idempotency)
- `wake.py`: Wake-on-LAN / HTTP-Wake Handler
- `config.py`: ENV-Konfiguration

### 2. SQLite Schema

```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT DEFAULT (datetime('now')),
    title TEXT
);

CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES sessions(id),
    role TEXT CHECK(role IN ('user', 'assistant', 'system')),
    content TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    status TEXT DEFAULT 'queued'  -- queued|forwarded|accepted|failed
);

CREATE TABLE media (
    id TEXT PRIMARY KEY,
    message_id TEXT REFERENCES messages(id),
    filename TEXT,
    mime_type TEXT,
    file_path TEXT,  -- relativer Pfad im media_dir
    size_bytes INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE gateway_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT,  -- server_up|server_down|wake_sent|flush_start|flush_end
    detail TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
```

### 3. Fallback UI (`static/fallback.html`)

- Einzelne HTML-Datei mit eingebettetem CSS und Vanilla JS
- Kein Build-Step, kein Framework, kein Node.js
- Funktionalität:
  - Chat-Interface mit Nachrichteneingabe
  - Verlaufsansicht (lädt via `/api/gateway/history`)
  - Status-Anzeige "Server offline / wird gestartet"
  - Media-Upload (Dateien → `/api/gateway/message` als multipart)
  - Automatischer Reload wenn Server online geht

### 4. Reverse Proxy Config

Caddyfile-Beispiel (empfohlen wegen ARM64-Kompatibilität und einfacher Syntax):

```caddyfile
# Alternative: direktes Gateway ohne Caddy, Gateway bindet auf Port 3000
# Caddy nur wenn HTTPS/Terminierung gewünscht
```

Da das Gateway direkt auf Port 3000 bindet, ist Caddy optional (nur für HTTPS).

### 5. AI-Stack Integration

Neuer Endpoint im ai-stack oder Dokumentation:

```
POST /api/queue/ingest
Body: {
  "messages": [
    {"id": "uuid", "role": "user", "content": "...", "session_id": "uuid",
     "media": [{"filename": "...", "url": "http://gateway:3000/api/gateway/media/uuid"}]}
  ]
}
Response: {
  "accepted": ["uuid1", "uuid2"],
  "duplicates": ["uuid3"],  -- bereits verarbeitet
  "failed": []
}
```

Der ai-stack muss:
1. Duplikaterkennung anhand message_id
2. Messages als normale User-Prompts an DeepAgent/LangGraph weiterleiten
3. Status zurückmelden (accepted/duplicate/failed)

### 6. Wake-Funktion

Drei Modi (per ENV `WAKE_MODE`):
- `wol`: Wake-on-LAN Magic Packet an `WAKE_WOL_MAC`
- `http`: HTTP-POST an `WAKE_HTTP_URL`
- `none`: Kein Wake (manuell)

### 7. Deployment

- systemd-Service auf Armbian
- Python 3.10+ (in Armbian enthalten)
- Dependencies: fastapi, uvicorn, httpx, aiosqlite, python-multipart
- Alternativ: docker-compose mit arm64-Image (optional)

## ARM64 / ROCK 2A Besonderheiten

- Keine amd64 Docker-Images
- Falls Docker: `FROM python:3.12-slim` (multi-arch)
- SD-Karte: WAL-Journal, begrenzte Log-Rotation
- RAM: FastAPI mit uvicorn single-worker (~50 MB idle)
- Kein Node.js auf dem Board nötig

## Was NICHT implementiert wird (v1-Scope)

- Keine Echtzeit-Synchronisation (Polling reicht für v1)
- Keine Multi-User Unterstützung (Single-User Gateway)
- Kein Streaming im Offline-Modus (Messages werden nur gequeued)
- Kein WebSocket-Proxy im Offline-Modus (nur online passthrough)

## Umsetzungsplan

1. `deepagent-arm-gateway/deepagent_gateway/main.py` — FastAPI App mit Health-Check-Loop
2. `deepagent_gateway/config.py` — ENV-Konfiguration
3. `deepagent_gateway/db.py` — SQLite Schema + aiosqlite
4. `deepagent_gateway/queue.py` — Message Queue Logik
5. `deepagent_gateway/proxy.py` — Reverse Proxy
6. `deepagent_gateway/wake.py` — Wake-on-LAN / HTTP-Wake
7. `deepagent_gateway/static/fallback.html` — Offline Chat UI
8. `deepagent_gateway/systemd/deepagent-gateway.service` — systemd Unit
9. `deepagent_gateway/tests/test_gateway.py` — Unit Tests
10. `deepagent_gateway/requirements.txt` — Python Dependencies
11. `docs/deepagent_gateway_plan.md` — Dieser Plan (Dokumentation)
