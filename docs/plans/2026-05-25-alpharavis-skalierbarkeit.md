# AlphaRavis Skalierbarkeit — Implementierungsplan

> **Für Hermes:** Diesen Plan Stage für Stage umsetzen. Jede Stage ist in sich
> abgeschlossen und testbar. Keine Stage überspringen — jede baut auf der
> vorherigen auf.

**Ziel:** AlphaRavis von Single-Host Dev-Stack zu einem horizontal skalierbaren
Multi-User-System umbauen, das 5 → 50 → 500 → 1000 gleichzeitige Nutzer bedient.

**Architektur-Ansatz:** Die bestehende Architektur (MongoDB-Checkpointer,
Redis-Leases, asyncio, percentage-based Context Budget) ist bereits
multi-worker-fähig designed. Der Plan baut darauf auf, indem er
Infrastruktur-Schichten davor setzt und den einen kritischen Process-Local-State
(MCP Session Manager) isoliert, statt den Agent-Code umzuschreiben.

**Tech Stack:** Docker Compose → Kubernetes, nginx/traefik, Redis, MongoDB,
Python 3.11, uvicorn, LangGraph Platform, LiteLLM

---

## Architektur-Übersicht: Was bleibt, was ändert sich

```
HEUTE (Single Instance):
  LibreChat ──→ api-bridge (1× uvicorn) ──→ langgraph-api (langgraph dev, 1 Prozess)
                                              │
                                              ├── MongoDB (shared)
                                              ├── Redis (optional)
                                              └── MCP Sessions (process-lokal)

ZIEL (Multi-Instance, Stage 3):
  LibreChat ──→ nginx (load balancer)
                  │
                  ├──→ api-bridge-1 (uvicorn worker)
                  ├──→ api-bridge-2 (uvicorn worker)
                  └──→ api-bridge-N (uvicorn worker)
                         │
                         └──→ langgraph-api-1 ──┐
                             langgraph-api-2 ──┤
                             langgraph-api-N ──┘
                                   │
                                   ├── MongoDB (shared, replica set)
                                   ├── Redis (shared, Leases + Cache)
                                   └── mcp-proxy (eigener Service, zentrale MCP-Sessions)
```

### Process-Local-State-Inventar (was geändert werden muss)

| State | Datei:Zeile | Multi-Worker-Problem | Lösung |
|---|---|---|---|
| `mcp_session_manager` (SSE) | server.py:235 | Jeder Worker baut eigene SSE-Verbindung → bei 10 Workern 10× die gleichen Tools geladen, Pixelle wird 10× verbunden | **MCP Proxy Service** (Stage 2+) |
| `MCP_SCHEMA_CACHE` | agent_graph.py:761 | Read-Only nach Startup → harmlos, jeder Worker lädt selbst | Keine Änderung nötig |
| `GRAPH_*_CONTEXT_RESERVES` | agent_graph.py:1430 | Berechnet beim Graph-Build, identisch pro Worker | Keine Änderung nötig |
| `_ENDPOINT_MODEL_METADATA_CACHE` | model_metadata.py:37 | 5min TTL → jeder Worker cached selbst | Keine Änderung nötig |
| `LAST_GRAPH_ACTIVITY_AT` | agent_graph.py:9793 | Nur Activity-Tracking, kein Shared-State nötig | Keine Änderung nötig |
| `agent_executor` | server.py:235 | Wird pro Prozess neu gebaut | Keine Änderung nötig |
| Runtime settings file | service-dashboard-data | File-basiert, shared Volume | Shared Volume in K8s |

**Fazit:** Nur EIN kritischer State muss angepackt werden: die MCP-Sessions.
Alles andere ist bereits multi-worker-kompatibel oder Read-Only nach Startup.

---

## Stage 1: "Team-Scale" (5–50 User, ~2 Tage)

Ohne Code-Änderungen. Nur Docker Compose + nginx.

### Warum diese Stage zuerst

- Liefert sofortigen Mehrwert (kleines Team kann parallel arbeiten)
- Validiert, dass MongoDB-Checkpointer mit 2+ Workern funktioniert
- Findet versteckte Race-Conditions früh, bevor sie in K8s schwer zu debuggen sind
- Keine Code-Änderung → kein Risiko für bestehende Features

### Task 1.1: nginx Load Balancer aufsetzen

**Dateien:**
- Erstellen: `nginx/default.conf`
- Erstellen: `nginx/Dockerfile`
- Ändern: `docker-compose.yml` (nginx service hinzufügen)

**nginx/default.conf:**
```nginx
upstream api_bridge {
    least_conn;
    server api-bridge:8123;
}

upstream langgraph_api {
    least_conn;
    server langgraph-api:2024;
}

server {
    listen 80;

    # LibreChat
    location / {
        proxy_pass http://librechat:3080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
    }

    # API Bridge (OpenAI-compatible)
    location /v1/ {
        proxy_pass http://api_bridge;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 300s;
        proxy_buffering off;
    }

    # LangGraph API (für UIs direkt)
    location /langgraph/ {
        rewrite ^/langgraph/(.*) /$1 break;
        proxy_pass http://langgraph_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 300s;
    }
}
```

**nginx/Dockerfile:**
```dockerfile
FROM nginx:alpine
COPY default.conf /etc/nginx/conf.d/default.conf
```

**docker-compose.yml** (nginx service hinzufügen, bestehende Service-Ports auf interne IPs ändern):
```yaml
  nginx:
    build: ./nginx
    container_name: nginx
    ports:
      - "80:80"
    depends_on:
      - librechat
      - api-bridge
      - langgraph-api
    restart: always

  api-bridge:
    # ports ändern: nicht mehr nach außen exposen
    ports:
      - "127.0.0.1:8123:8123"  # nur localhost, nginx routed

  langgraph-api:
    ports:
      - "127.0.0.1:2024:2024"  # nur localhost
```

**Verifikation:**
```bash
docker compose up -d --build
curl http://localhost/v1/models                    # → Modell-Liste vom Bridge
curl http://localhost/                             # → LibreChat UI
```

### Task 1.2: api-bridge auf Multi-Worker umstellen

**Dateien:**
- Ändern: `docker-compose.yml` (api-bridge command mit --workers)

```yaml
  api-bridge:
    # command ändern von "uvicorn bridge_server:app" zu:
    command: ["uvicorn", "bridge_server:app", "--host", "0.0.0.0", "--port", "8123", "--workers", "4"]
```

Hinweis: Der Bridge-Server ist stateless (leitet nur Requests an LangGraph weiter,
kein eigener Session-State außer approve-always Cache, der per LibreChat-Thread
lokal ist). `--workers 4` funktioniert sofort.

**Verifikation:**
```bash
docker compose restart api-bridge
docker logs api-bridge | grep -i worker            # → zeigt 4 worker processes
make bridge-smoke                                  # → smoke test besteht
```

### Task 1.3: langgraph-api auf Production Mode umstellen

**Dateien:**
- Ändern: `docker-compose.yml` (langgraph-api command)
- Ändern: `langgraph-app/langgraph.json` (Production-kompatibel prüfen)

Der aktuelle Befehl `langgraph dev` ist der Development-Server. Für Stage 1
reicht es, auf `langgraph serve` (Production-Server mit mehreren Workern)
umzustellen. Alternativ: `langgraph up` wenn LangGraph Platform verfügbar.

```yaml
  langgraph-api:
    # command ändern:
    command: ["sh", "-c", "\
      python /workspace/langgraph-app/patches/patch_langchain_openai_disable_streaming.py && \
      python /workspace/langgraph-app/patches/patch_langchain_openai_responses_tool_streaming.py && \
      rm -f /tmp/.X0-lock && \
      Xvfb :0 -screen 0 1280x720x24 & fluxbox & x11vnc -display :0 -forever -nopw -listen 0.0.0.0 & \
      cd /workspace/langgraph-app && \
      langgraph serve --config langgraph.json --host 0.0.0.0 --port 2024 \
        --n-workers 4 \
        --redis-url redis://redis:6379"]
```

**Verifikation:**
```bash
docker compose restart langgraph-api
docker logs langgraph-api | grep -i "n.workers\|worker"   # → 4 workers
make bridge-smoke                                          # → smoke test besteht
# Test: Zwei simultane Requests
curl -s http://localhost/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"my-agent","messages":[{"role":"user","content":"sag hallo"}]}' &
curl -s http://localhost/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"my-agent","messages":[{"role":"user","content":"sag hallo"}]}' &
wait
# Beide sollten parallel antworten
```

### Task 1.4: Redis Context Leases aktivieren

**Dateien:**
- Ändern: `.env`

```bash
# Von process-local auf Redis umstellen (essentiell für Multi-Worker)
ALPHARAVIS_CONTEXT_LEASE_BACKEND=redis
REDIS_URL=redis://redis:6379
```

**Verifikation:**
```bash
docker compose restart langgraph-api
docker logs langgraph-api | grep -i "context.*lease\|redis"
# → sollte "ContextScheduler using Redis backend" o.ä. zeigen
```

### Task 1.5: Smoke-Test mit parallelen Requests

```bash
# 10 parallele Requests
for i in $(seq 1 10); do
  curl -s http://localhost/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"my-agent\",\"messages\":[{\"role\":\"user\",\"content\":\"sag $i\"}]}" &
done
wait
# Alle 10 sollten innerhalb ~30s antworten (nicht sequentiell 10×30s)
```

---

## Stage 2: "Department-Scale" (50–500 User, ~2 Wochen)

### Task 2.1: MCP Proxy Service (DER kritische Architektur-Change)

**Problem:** Aktuell baut jeder LangGraph-Worker beim Startup eigene SSE MCP-Verbindungen
auf (server.py:235-253, mcp_client.py:601-666). Bei 10 Workern = 10× Pixelle-Verbindungen.
Das skaliert nicht.

**Lösung:** Ein dedizierter `mcp-proxy` Service, der EINMAL alle MCP-Verbindungen
aufbaut und als HTTP-Proxy für Tool-Calls bereitstellt. LangGraph-Worker rufen
Tools über den Proxy auf, statt eigene MCP-Sessions zu halten.

**Dateien:**
- Erstellen: `langgraph-app/mcp_proxy_server.py` (~400 Zeilen)
- Erstellen: `langgraph-app/mcp_proxy_client.py` (~200 Zeilen)
- Ändern: `langgraph-app/mcp_client.py` (Proxy-Mode ergänzen, ~50 Zeilen)
- Ändern: `docker-compose.yml` (mcp-proxy service)
- Ändern: `.env(exaple)` (neue ENV vars dokumentieren)

**Architektur:**
```
mcp-proxy (1 Instanz, hält ALLE MCP-Sessions)
  ├── Pixelle SSE ────→ http://pixelle:9004/pixelle/mcp/sse
  ├── [weitere MCP Server aus mcp.json]
  │
  └── HTTP API:
        POST /tools/{server_name}/{tool_name}   → Tool ausführen
        GET  /tools/{server_name}                → Tool-Liste eines Servers
        GET  /health                             → Health Check

langgraph-api worker-1 ──→ POST mcp-proxy/tools/pixelle/generate_image
langgraph-api worker-2 ──→ POST mcp-proxy/tools/pixelle/generate_image
langgraph-api worker-N ──→ POST mcp-proxy/tools/pixelle/generate_image
```

**mcp_proxy_server.py** (Kernstruktur):
```python
"""MCP Proxy Service — zentrale MCP-Session-Verwaltung für Multi-Worker."""
from __future__ import annotations
import asyncio, json, os, time
from contextlib import AsyncExitStack
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mcp_client import load_mcp_config, RobustMCPServerManager

app = FastAPI()

# Global: eine Session pro Server
_mcp_managers: dict[str, RobustMCPServerManager] = {}
_stack: AsyncExitStack | None = None

class ToolCallRequest(BaseModel):
    arguments: dict = {}
    timeout: int = 120

@app.on_event("startup")
async def startup():
    global _stack
    _stack = AsyncExitStack()
    config, _, warnings = load_mcp_config()
    servers = config.get("mcpServers", {})
    for name, cfg in sorted(servers.items()):
        mgr = RobustMCPServerManager(name, _mcp_connection_from_config(cfg), cfg)
        try:
            await mgr.connect(_stack)
            _mcp_managers[name] = mgr
        except Exception as e:
            print(f"MCP server '{name}' failed: {e}")

@app.on_event("shutdown")
async def shutdown():
    if _stack:
        await _stack.aclose()

@app.post("/tools/{server_name}/{tool_name}")
async def call_tool(server_name: str, tool_name: str, req: ToolCallRequest):
    mgr = _mcp_managers.get(server_name)
    if not mgr:
        raise HTTPException(404, f"Server '{server_name}' not found")
    try:
        result = await mgr.call_tool(tool_name, req.arguments, timeout=req.timeout)
        return {"result": result}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/tools/{server_name}")
async def list_tools(server_name: str):
    mgr = _mcp_managers.get(server_name)
    if not mgr:
        raise HTTPException(404)
    return {"tools": [{"name": t.name, "description": t.description} for t in mgr.tools]}

@app.get("/health")
async def health():
    return {"servers": list(_mcp_managers.keys()), "healthy": True}
```

**docker-compose.yml** Ergänzung:
```yaml
  mcp-proxy:
    build:
      context: .
      dockerfile: ./langgraph-app/Dockerfile
    container_name: mcp-proxy
    command: ["uvicorn", "mcp_proxy_server:app", "--host", "0.0.0.0", "--port", "8135"]
    ports:
      - "127.0.0.1:8135:8135"
    volumes:
      - ./langgraph-app:/app
      - ./:/workspace:ro
    env_file:
      - .env
    environment:
      - PIXELLE_URL=${PIXELLE_URL:-http://pixelle:9004}
    restart: always
    healthcheck:
      test: ["CMD-SHELL", "curl -sS http://localhost:8135/health || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 10
```

**mcp_client.py** Änderung (Proxy-Mode):
```python
# In load_robust_mcp_tools(), vor dem Laden:
if _env_bool("ALPHARAVIS_MCP_PROXY_MODE", "false"):
    return await _load_mcp_tools_via_proxy()
# ... bestehender Code für direkte MCP-Verbindungen bleibt
```

**mcp_proxy_client.py** (neu):
```python
"""MCP-Tool-Proxy-Client — wrapped Tools die HTTP-Calls zum Proxy machen."""
import httpx
from langchain_core.tools import BaseTool

MCP_PROXY_URL = os.getenv("ALPHARAVIS_MCP_PROXY_URL", "http://mcp-proxy:8135")

async def _load_mcp_tools_via_proxy():
    """Hole Tool-Schemas vom Proxy und wrappe sie als HTTP-Call-Tools."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{MCP_PROXY_URL}/health")
        servers = resp.json()["servers"]

    tools = []
    for server in servers:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{MCP_PROXY_URL}/tools/{server}")
            for tool_info in resp.json()["tools"]:
                tool = _make_proxy_tool(server, tool_info)
                tools.append(tool)
    return tools

def _make_proxy_tool(server_name: str, tool_info: dict) -> BaseTool:
    """Erzeugt ein LangChain-Tool das HTTP-Calls zum Proxy macht."""
    # ... Implementation mit httpx
```

**Verifikation:**
```bash
# MCP Proxy Health Check
curl http://localhost:8135/health
# → {"servers": ["pixelle"], "healthy": true}

# Tool-Liste über Proxy
curl http://localhost:8135/tools/pixelle
# → [{"name": "generate_image", "description": "..."}, ...]

# LangGraph mit Proxy-Mode starten
ALPHARAVIS_MCP_PROXY_MODE=true docker compose restart langgraph-api
make bridge-smoke
```

### Task 2.2: LiteLLM für Skalierung vorbereiten

**Ziel:** LiteLLM als zentralen Gateway behalten, aber für Last vorbereiten.

**Dateien:**
- Ändern: `docker-compose.yml` (litellm mit Redis-Backing)
- Ändern: `litellm-config/config.yaml`

```yaml
  litellm:
    environment:
      # Redis für Rate-Limiting + Caching
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      # Request queue
      - LITELLM_MAX_PARALLEL_REQUESTS=100
      - LITELLM_REQUEST_TIMEOUT=600
```

### Task 2.3: MongoDB Connection-Pool Tuning

**Dateien:**
- Ändern: `.env(exaple)` (Doku)
- Ändern: `docs/ALPHARAVIS_ARCHITECTURE.md` (Doku)

```bash
# .env
MONGO_MAX_POOL_SIZE=100        # Default ist 100, für 500 User erhöhen
MONGO_MIN_POOL_SIZE=10
MONGO_SERVER_SELECTION_TIMEOUT_MS=5000
```

Hinweis: MongoDB Connection-Pool wird über den Connection-String gesteuert.
LangGraph-SDK `AsyncMongoCheckpointer` akzeptiert `max_pool_size` im Constructor.

### Task 2.4: Monitoring (Prometheus + Grafana)

**Dateien:**
- Erstellen: `monitoring/prometheus.yml`
- Erstellen: `monitoring/docker-compose.yml`
- Ändern: `docker-compose.yml` (Metric-Exports aktivieren)

```yaml
# monitoring/docker-compose.yml
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

**Verifikation:**
```bash
docker compose -f monitoring/docker-compose.yml up -d
curl http://localhost:9090/api/v1/status/config  # Prometheus läuft
curl http://localhost:3000                        # Grafana läuft
```

### Task 2.5: Dokumentation aktualisieren

**Dateien:**
- Ändern: `docs/ALPHARAVIS_ARCHITECTURE.md`
- Ändern: `docs/ALPHARAVIS_CHANGES.md`
- Ändern: `.env(exaple)`

Neue ENV vars dokumentieren:
```bash
# Skalierung
ALPHARAVIS_CONTEXT_LEASE_BACKEND=redis
ALPHARAVIS_MCP_PROXY_MODE=true
ALPHARAVIS_MCP_PROXY_URL=http://mcp-proxy:8135
MONGO_MAX_POOL_SIZE=100
```

---

## Stage 3: "Enterprise-Scale" (500–1000 User, ~4 Wochen)

### Task 3.1: Kubernetes Migration

**Dateien:**
- Erstellen: `k8s/namespace.yaml`
- Erstellen: `k8s/configmap.yaml`
- Erstellen: `k8s/secrets.yaml`
- Erstellen: `k8s/deployment-api-bridge.yaml`
- Erstellen: `k8s/deployment-langgraph.yaml`
- Erstellen: `k8s/deployment-mcp-proxy.yaml`
- Erstellen: `k8s/deployment-litellm.yaml`
- Erstellen: `k8s/service-api-bridge.yaml`
- Erstellen: `k8s/service-langgraph.yaml`
- Erstellen: `k8s/ingress.yaml`
- Erstellen: `k8s/hpa-api-bridge.yaml`
- Erstellen: `k8s/hpa-langgraph.yaml`

**k8s/deployment-langgraph.yaml** (Auszug):
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langgraph-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: langgraph-api
  template:
    spec:
      containers:
      - name: langgraph-api
        image: alpharavis/langgraph-app:latest
        command: ["langgraph", "serve", "--config", "langgraph.json",
                  "--host", "0.0.0.0", "--port", "2024",
                  "--n-workers", "4",
                  "--redis-url", "$(REDIS_URL)"]
        envFrom:
        - configMapRef:
            name: alpharavis-config
        - secretRef:
            name: alpharavis-secrets
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"
        readinessProbe:
          httpGet:
            path: /health
            port: 2024
          initialDelaySeconds: 30
          periodSeconds: 10
```

**k8s/hpa-langgraph.yaml:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: langgraph-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: langgraph-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Task 3.2: Rate Limiting Layer

**Dateien:**
- Erstellen: `langgraph-app/rate_limiter.py` (~150 Zeilen)
- Ändern: `langgraph-app/bridge_server.py` (Rate-Limit-Middleware)
- Ändern: `.env(exaple)` (Doku)

**rate_limiter.py:**
```python
"""Redis-backed Rate Limiter für Bridge + LangGraph."""
import time, os
import redis.asyncio as redis

class RateLimiter:
    def __init__(self, redis_url: str, max_requests: int = 10, window_seconds: int = 60):
        self.redis = redis.from_url(redis_url)
        self.max_requests = max_requests
        self.window = window_seconds

    async def check(self, key: str) -> tuple[bool, int]:
        """Returns (allowed, remaining_requests)."""
        now = time.time()
        window_start = now - self.window
        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, self.window + 10)
        _, count, _, _ = await pipe.execute()
        remaining = max(0, self.max_requests - count)
        return count <= self.max_requests, remaining
```

**bridge_server.py** Middleware:
```python
from rate_limiter import RateLimiter

_rate_limiter: RateLimiter | None = None

@app.on_event("startup")
async def setup_rate_limiter():
    global _rate_limiter
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
    max_req = int(os.getenv("BRIDGE_RATE_LIMIT_MAX", "60"))
    _rate_limiter = RateLimiter(redis_url, max_requests=max_req)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if _rate_limiter:
        client_ip = request.client.host if request.client else "unknown"
        allowed, remaining = await _rate_limiter.check(f"rate:{client_ip}")
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded", "retry_after": 60}
            )
    return await call_next(request)
```

### Task 3.3: LLM-Provider auf externe API umstellen

**Begründung:** Bei 1000 parallelen Usern ist ein lokaler llama.cpp-Server das Bottleneck.
Externe APIs (OpenRouter, Anthropic) haben praktisch unendliche Parallelkapazität.

**Dateien:**
- Ändern: `.env` (Produktion)
- Ändern: `litellm-config/config.yaml` (neue Routen)

```bash
# .env für Enterprise Deployment
OPENAI_API_BASE=https://openrouter.ai/api/v1
OPENAI_API_KEY=sk-or-v1-xxx
ALPHARAVIS_RESPONSES_MODEL=anthropic/claude-sonnet-4
ALPHARAVIS_DEEPAGENTS_RESPONSES_MODEL=anthropic/claude-sonnet-4

# Lokale llama.cpp Instanzen deaktivieren
ALPHARAVIS_ENABLE_MODEL_MANAGEMENT=false
```

Hinweis: Die AlphaRavis-Architektur ist provider-agnostisch. Über LiteLLM oder
OpenRouter kann jedes Modell genutzt werden. Kein Code-Change nötig — nur `.env`.

### Task 3.4: Lasttest-Suite

**Dateien:**
- Erstellen: `tests/load/locustfile.py`
- Erstellen: `tests/load/README.md`

```python
# tests/load/locustfile.py
from locust import HttpUser, task, between

class AlphaRavisUser(HttpUser):
    wait_time = between(1, 5)

    @task(3)
    def simple_chat(self):
        self.client.post("/v1/chat/completions", json={
            "model": "my-agent",
            "messages": [{"role": "user", "content": "Was ist 2+2?"}],
            "stream": False
        })

    @task(1)
    def complex_chat(self):
        self.client.post("/v1/chat/completions", json={
            "model": "my-agent",
            "messages": [{"role": "user", "content": "Analysiere die Architektur von Docker"}],
            "stream": False
        })
```

**Verifikation:**
```bash
pip install locust
locust -f tests/load/locustfile.py --host=http://localhost
# Öffne http://localhost:8089, starte mit 50 users, 5 spawn rate
# Erhöhe schrittweise auf 100, 200, 500, 1000
# Beobachte: Response Time, Failure Rate, RPS
```

### Task 3.5: Failover & Resilience

**Dateien:**
- Erstellen: `k8s/pod-disruption-budget.yaml`
- Ändern: `k8s/deployment-*.yaml` (podAntiAffinity)

```yaml
# k8s/pod-disruption-budget.yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: langgraph-api-pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: langgraph-api
```

```yaml
# In deployment-langgraph.yaml, podAntiAffinity:
affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchLabels:
            app: langgraph-api
        topologyKey: kubernetes.io/hostname
```

---

## Zusammenfassung: Aufwand pro Stage

| Stage | User | Zeit | Code-Änderungen | Risiko |
|---|---|---|---|---|
| 1: Team | 5–50 | 2 Tage | 0 Zeilen Python, nur nginx + docker-compose | Minimal |
| 2: Department | 50–500 | 2 Wochen | ~650 Zeilen (MCP Proxy + Client), Monitoring YAML | Mittel |
| 3: Enterprise | 500–1000 | 4 Wochen | ~150 Zeilen (Rate Limiter), K8s Manifests | Moderat |

**Kern-Erkenntnis:** Die AlphaRavis-Architektur (MongoDB-Checkpoints, Redis-Leases,
asyncio, percentage-based Context Budget) ist bereits für horizontale Skalierung
ausgelegt. Der einzige echte Code-Change ist der MCP Proxy Service in Stage 2.
Der Rest ist Infrastruktur-Konfiguration.

**Quick Wins (sofort umsetzbar, Stage 0):**
```bash
# Redis Context Leases aktivieren (schon implementiert, nur config)
echo "ALPHARAVIS_CONTEXT_LEASE_BACKEND=redis" >> .env

# Bridge Multi-Worker testen
docker compose exec api-bridge uvicorn bridge_server:app --workers 2 --port 8124

# Parallele Smoke-Tests
for i in 1 2 3 4 5; do make bridge-smoke & done; wait
```
