# 🚀 AlphaRavis

**AlphaRavis** is a state-of-the-art, fully local all-in-one AI stack. It orchestrates local and cloud-based LLMs, equips them with long-term memory, provides them with powerful code execution and web research tools, and bundles everything into a sleek, professional user interface.

This repository combines RAG, agentic workflows (LangGraph), model routing (LiteLLM), code execution (MCP/OpenCode), and a modern chat interface (LibreChat / LangFlow).

## Current AlphaRavis Notes

- Human usage guide: [`docs/ALPHARAVIS_USAGE_NOTES.md`](docs/ALPHARAVIS_USAGE_NOTES.md)
- Architecture/capability guide: [`docs/ALPHARAVIS_ARCHITECTURE.md`](docs/ALPHARAVIS_ARCHITECTURE.md)
- Hermes integration guide: [`docs/HERMES_INTEGRATION.md`](docs/HERMES_INTEGRATION.md)

Important current behavior:

- Simple non-tool chat can use the fast path and skip the swarm.
- Complex tool, debug, memory, research, Pixelle, code, SSH, Docker, and PC-control tasks use the agent path.
- `edge-gemma` is only a small starter/crisis fallback for simple fast-path chat, not a second boss for complex workflows.
- Fast-path replies are visibly marked, and a thread is locked out of fast path after it uses the normal agent path.
- Optional MCP tool loading is off by default to avoid slowing every chat.
- MCP uses DeepAgents-style `mcp.json` config, with stdio servers disabled by default.
- Hermes can be used directly from LibreChat or as an optional bounded coding/system sub-agent inside AlphaRavis.
- A Hermes-inspired MemoryKernel keeps curated always-memory small, indexes turns for search, and writes large notes to artifacts.
- Agent handoffs now use structured packets plus a handoff-context guard so long workflows keep the task brief, open work, and verification state active.
- Optional pgvector semantic memory stores a source catalog plus full retrieval chunks for memories, turns, archives, artifacts, skills, lessons, and federated RAG hits while MongoDB remains the source of truth.
- Agents can inspect a lazy optional-tool manifest and use scoped agent/global memories.
- Reviewed repo skill cards live in `ai-skills/`, starting with DeepAgents agent-building and research workflows, and can be read on demand by AlphaRavis.
- Risky local or SSH commands require approval before execution.

## Makefile Workflow

Use the Makefile for the common stack operations:

```bash
make help                        # show install/runtime targets and variables
make install                     # guided .env sync, streaming/profile selection, submodules, optional build/start
make update                      # git pull, profile selection, submodules, Docker build, stack restart
make install-fullstreaming        # set full Responses tool streaming, init submodules, build, start
make install-chat-fullstreaming   # set Chat Completions streaming, init submodules, build, start
make profiles                     # show runtime profiles and the .env values they write
make streaming STREAMING=full     # only update .env streaming settings
make media-vision VISION_ENABLED=true VISION_URL=http://host:port/v1 VISION_MODEL=model-name
make up-fullstreaming             # set full streaming and recreate langgraph-api/api-bridge/test UI
make update                       # git pull, optional submodule update, optional env edit
make status                       # show URLs, streaming mode, profiles, and docker compose ps
make up                           # docker compose up -d --build, including bridge-test-ui
make down                         # docker compose down
make bridge-smoke                 # one small OpenAI-compatible request against api-bridge
make hermes-smoke                 # one small OpenAI-compatible request against Hermes
```

Tailscale HTTPS helper:

```bash
make tailscale-plan TAILSCALE_HOST=<device>.<tailnet>.ts.net
make tailscale-overrides TAILSCALE_HOST=<device>.<tailnet>.ts.net
```

The helper reads the Service Dashboard catalog and prepares Tailscale Serve
HTTPS routes for the local HTTP services inside your Tailnet. It does not run
Tailscale Funnel and does not publish services to the public internet.

Runtime profiles accepted by `make install STREAMING=...`, `make update`, and
`make streaming STREAMING=...`:

- `responses-hybrid` (`hybrid`): stable default. Responses API, no-tool calls
  may stream, tool-bound calls stay non-streaming.
- `responses-full` (`full`): Responses API full streaming with the AlphaRavis
  experimental tool-streaming patch enabled.
- `responses-nonstreaming` (`nonstreaming`): Responses API with internal
  model streaming disabled.
- `chat-full` (`chat`): Chat Completions API through ChatLiteLLM with
  `ALPHARAVIS_LLM_STREAMING=true`.
- `chat-nonstreaming`: Chat Completions API through ChatLiteLLM with streaming
  disabled.

Useful install examples:

```bash
make install STREAMING=full PROFILES=openwebui
make install STREAMING=chat-full PROFILES=none
make install STREAMING=hybrid START=no BUILD=no SUBMODULES=yes PROFILES=none
make install VISION_ENABLED=true VISION_URL=http://192.168.178.50:8080/v1 VISION_MODEL=vision-embed
make update VISION_URL=http://192.168.178.50:8080/v1 VISION_MODEL=vision-embed
make up VISION_URL=http://192.168.178.50:8080/v1 VISION_MODEL=vision-embed
```

`VISION_URL` writes `ALPHARAVIS_VISION_EMBEDDING_MODEL_URL` for a dedicated
OpenAI-compatible vision embedding server, for example a small llama.cpp server
on another machine. When it is set on `make up`, the Makefile updates `.env`
before starting Docker Compose.

Important endpoints:

- Service Dashboard: `http://localhost:8090`
- LibreChat: `http://localhost:3080`
- LangGraph API: `http://localhost:2024`
- OpenAI-compatible AlphaRavis bridge: `http://localhost:8123/v1`
- Bridge Test UI: `http://localhost:8140`
- Hermes OpenAI-compatible API: `http://localhost:8642/v1`

LibreChat has named custom endpoints for `LangGraph Agent` and `Hermes Agent`.
If a separate `OpenAI` provider appears, it comes from LibreChat's generic
OpenAI integration. By default this stack hides that extra bucket with
`LIBRECHAT_OPENAI_API_KEY=` and `LIBRECHAT_OPENAI_REVERSE_PROXY=`.

## ✨ Key Features

- 💬 **LibreChat Frontend:** A ChatGPT-like UI acting as the primary interface for all your agents.
- 🧠 **LangGraph Orchestrator:** An intelligent agentic "brain" that delegates tasks, such as switching between fast web searches and hours-long deep research cycles.
- 💾 **Persistent Memory:** Integrated short-term and long-term memory using MongoDB (Checkpointer & BaseStore) and LangMem to maintain context across days or weeks.
- 🛠️ **MCP (Model Context Protocol):** Full access to your local workspace via OpenCode. The agent can autonomously read, write, and execute code.
- 📚 **Local RAG:** Document analysis and vector search based on PostgreSQL (pgvector) for private data processing.
- 🔀 **LiteLLM Routing:** A central gateway for all models, allowing you to use Ollama, Llama.cpp, or OpenAI via a single API.

## 🏗️ System Architecture

AlphaRavis is composed of several microservices orchestrated via Docker Compose:

1. **Frontend:** `librechat` (main interface) and `agent-custom-ui` (specialized agent view).
2. **Logic Layer:** `langgraph-api` (The Python-based agentic server managing workflows).
3. **Model Gateway:** `litellm` (Distributes prompts to local servers or cloud providers).
4. **Data & Cache:** `mongodb` (State & Long-Term Memory), `vectordb`/pgvector (optional semantic memory/search index), and `redis` (Semantic Caching).
5. **Tools:** `opencode-server` (Headless code worker) and `rag_api` (Document embedding service).

## 🚀 Quickstart

### Prerequisites
- Docker & Docker Compose
- Node.js (for MCP tool execution)
- (Optional) Local LLMs running via Ollama or Llama.cpp.

### Installation

1. **Clone the repository with submodules:**
   ```bash
   git clone --recurse-submodules https://github.com/THEman6989/ai-stack
   cd ai-stack
2. **Update submodules:**
   ```bash
   git submodule update --init --recursive --remote
