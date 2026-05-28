# AlphaRavis — Fully Local AI Stack

**AlphaRavis** is a local-first, multi-agent AI system that orchestrates LLMs through a LangGraph "brain", equips them with persistent memory and powerful tools, and exposes everything through multiple polished UIs.

One `make install` — and you get a full AI operating system.

## Quickstart

```bash
git clone --recurse-submodules https://github.com/THEman6989/ai-stack
cd ai-stack
make config          # browser UI for .env settings
make install         # init submodules, build, start
make status          # verify all services
```

Open the **Service Dashboard** at `http://localhost:8090` for clickable links to every running service.

For detailed operator reference: `make help` or [`docs/MAKEFILE_README.md`](docs/MAKEFILE_README.md).

## What You Get

| Service | Port | What It Does |
|---|---|---|
| **Service Dashboard** | 8090 | Landing page + runtime settings UI for all services |
| **LangGraph API** | 2024 | The brain — runs `alpha_ravis` agent graph with tool routing |
| **API Bridge** | 8123 | OpenAI-compatible `/v1/chat/completions` + `/v1/responses` |
| **LiteLLM** | 4000 | Central model gateway to Ollama, llama.cpp, cloud APIs |
| **Hermes Agent** | 8642 | Coding/system specialist with terminal, file, web tools |
| **MongoDB** | 27017 | Persistent state, long-term memory, session storage |
| **pgvector** | 5432 | Optional semantic memory + document RAG index |
| **RAG API** | 8000 | Document ingestion and vector search backend |
| **Redis** | 6379 | Semantic caching layer |
| **Media Gallery** | 8130 | Hosts uploaded media, galleries, Office output, analysis assets |
| **Bridge Test UI** | 8140 | Diagnostic UI for streaming/protocol debugging |

### User Interfaces

| UI | Port | Description |
|---|---|---|
| **Deep Agents UI** | 3000 | Primary LangGraph-native inspection and chat UI (forked) |
| **LibreChat** | 3080 | ChatGPT-style interface with LangGraph + Hermes endpoints |
| **Agent Custom UI** | 3001 | Alternative AlphaRavis frontend |
| **OpenWebUI** | 3090 | Optional second frontend (profile: `openwebui`) |
| **Hermes Dashboard** | 9119 | Optional Hermes dashboard (profile: `hermes-dashboard`) |
| **LangGraph Studio** | — | Connect via `https://smith.langchain.com/studio/?baseUrl=http://localhost:2024` |

### Optional / Feature-Gated

| Feature | Gate | Docs |
|---|---|---|
| **OfficeCLI** — create Word, Excel, PowerPoint via CLI | `ALPHARAVIS_ENABLE_OFFICECLI=true` | [`OFFICECLI_AGENT_REFERENCE.md`](docs/OFFICECLI_AGENT_REFERENCE.md) |
| **MCP Tools** — model context protocol servers | `ALPHARAVIS_LOAD_MCP_TOOLS=true` | `langgraph-app/mcp.json` |
| **Model Management** — inspect and control remote llama.cpp/Ollama | `ALPHARAVIS_ENABLE_MODEL_MANAGEMENT=true` | [`ALPHARAVIS_MODEL_MANAGEMENT.md`](docs/ALPHARAVIS_MODEL_MANAGEMENT.md) |
| **Vision Embedding** — multimodal semantic memory | `ALPHARAVIS_ENABLE_VISION_VECTOR_MEMORY=true` | [`ALPHARAVIS_ARCHITECTURE.md`](docs/ALPHARAVIS_ARCHITECTURE.md) |
| **Video Analysis** — frame extraction and analysis | `make video-analysis ENABLED=true` | [`ALPHARAVIS_USAGE_NOTES.md`](docs/ALPHARAVIS_USAGE_NOTES.md) |
| **Tailscale HTTPS** — secure remote access | Default enabled; `make tailscale-disable` | [`MAKEFILE_README.md`](docs/MAKEFILE_README.md) |
| **AionUi ACP** — custom agent UI via stdio adapter | `ALPHARAVIS_ENABLE_ACP=true` | [`AIONUI_LANGGRAPH_ACP_INTEGRATION.md`](docs/AIONUI_LANGGRAPH_ACP_INTEGRATION.md) |

## Core Capabilities

**Agentic Brain (LangGraph)**
- Multi-agent swarm with Planner → Router → Executor → Reviewer flow
- Fast path for simple non-tool chats; agent path for complex tasks
- DeepAgents-style subagents for long-running work
- Context compression, token budgeting, and semantic memory retrieval

**Persistent Memory**
- MongoDB-backed checkpointer for session state
- Curated always-memory via Hermes-inspired MemoryKernel
- Optional pgvector semantic memory with source catalog
- Session history search for cross-session recall

**Tool Ecosystem**
- Shell, file I/O, and SSH with approval gates
- Web search (Tavily), browser automation
- Code execution via Hermes agent delegation
- Document RAG with pgvector backend
- Pixelle MCP for image generation/analysis
- OfficeCLI for document creation (Word, Excel, PowerPoint)

**Streaming Profiles**
```bash
make streaming STREAMING=hybrid      # stable default: stream text, non-stream tools
make streaming STREAMING=full        # full streaming with tool calls
make streaming STREAMING=nonstreaming
make streaming STREAMING=chat-full   # Chat Completions full streaming
```

## Architecture

```
┌─────────────┐  ┌──────────────┐  ┌────────────────┐
│ DeepAgents  │  │  LibreChat   │  │  Agent Custom  │
│    UI :3000 │  │       :3080  │  │     UI :3001   │
└──────┬──────┘  └──────┬───────┘  └───────┬────────┘
       │                │                  │
       │          ┌─────▼──────┐           │
       │          │ API Bridge │           │
       │          │   :8123    │           │
       │          │  OpenAI    │           │
       │          │ Compatible │           │
       │          └─────┬──────┘           │
       │                │                  │
       └────────┐       │       ┌──────────┘
                │       │       │
              ┌─▼───────▼───────▼─┐
              │  LangGraph API    │
              │  alpha_ravis :2024│
              │  (Brain/Agents)   │
              └──┬──────┬──────┬──┘
                 │      │      │
    ┌────────────▼─┐ ┌──▼──┐ ┌▼──────────┐
    │   LiteLLM    │ │ RAG │ │  Hermes    │
    │    :4000     │ │:8000│ │   :8642    │
    │ (Model Gate) │ │     │ │ (Coding)   │
    └──────┬───────┘ └──┬──┘ └────────────┘
           │            │
    ┌──────▼──────┐ ┌───▼──────┐
    │ Ollama      │ │ pgvector │
    │ llama.cpp   │ │ MongoDB  │
    │ Cloud APIs  │ │ Redis    │
    └─────────────┘ └──────────┘
```

For full details: [`docs/ALPHARAVIS_ARCHITECTURE.md`](docs/ALPHARAVIS_ARCHITECTURE.md).

## Daily Operations

```bash
make help                          # full target reference
make config                        # browser settings editor
make install                       # first-time setup
make update                        # git pull + rebuild + restart
make up                            # start stack
make down                          # stop stack
make status                        # service health
make logs                          # tail all logs

# Streaming switches
make streaming STREAMING=hybrid    # switch streaming profile
make up-fullstreaming              # enable full streaming

# Network modes
make tailscale-apply               # HTTPS remote access
make tailscale-disable             # local LAN HTTP access

# Diagnostics
make bridge-smoke                  # test bridge connectivity
make hermes-smoke                  # test Hermes connectivity
make media-smoke                   # test media gallery
make comfyui-smoke                 # test ComfyUI host/proxy/socket integration
```

## Documentation

| Document | Covers |
|---|---|
| [`docs/ALPHARAVIS_ARCHITECTURE.md`](docs/ALPHARAVIS_ARCHITECTURE.md) | Full system design, container roles, data flow, runtime profiles |
| [`docs/ALPHARAVIS_USAGE_NOTES.md`](docs/ALPHARAVIS_USAGE_NOTES.md) | Human-facing behavior: what runs when, what's automatic, what needs asking |
| [`docs/ALPHARAVIS_CHANGES.md`](docs/ALPHARAVIS_CHANGES.md) | Intentional local runtime changes, patches, compatibility decisions |
| [`docs/MAKEFILE_README.md`](docs/MAKEFILE_README.md) | Complete Makefile target and variable reference |
| [`docs/HERMES_INTEGRATION.md`](docs/HERMES_INTEGRATION.md) | Hermes agent setup, LibreChat direct vs AlphaRavis delegation modes |
| [`docs/ALPHARAVIS_RESPONSES_COMPATIBILITY.md`](docs/ALPHARAVIS_RESPONSES_COMPATIBILITY.md) | OpenAI Responses API surface: implemented features, streaming events |
| [`docs/ALPHARAVIS_MODEL_MANAGEMENT.md`](docs/ALPHARAVIS_MODEL_MANAGEMENT.md) | Remote model/power management, embedder scheduling, crisis fallback |
| [`docs/ALPHARAVIS_RAG_HANDOFF.md`](docs/ALPHARAVIS_RAG_HANDOFF.md) | RAG/compression, large-paste ingest, retrieval routing |
| [`docs/ALPHARAVIS_OPEN_TASKS.md`](docs/ALPHARAVIS_OPEN_TASKS.md) | Active backlog, current implementation state, open work |
| [`docs/AIONUI_LANGGRAPH_ACP_INTEGRATION.md`](docs/AIONUI_LANGGRAPH_ACP_INTEGRATION.md) | AionUi ACP adapter setup and usage |
| [`docs/AIONUI_OFFICE_INTEGRATION.md`](docs/AIONUI_OFFICE_INTEGRATION.md) | OfficeCLI analysis and AlphaRavis integration design |
| [`docs/OFFICECLI_AGENT_REFERENCE.md`](docs/OFFICECLI_AGENT_REFERENCE.md) | Compact OfficeCLI command reference for agents |
| [`docs/ALPHARAVIS_DEEPAGENTS_IMPROVEMENTS.md`](docs/ALPHARAVIS_DEEPAGENTS_IMPROVEMENTS.md) | DeepAgents integration: subagents, skills, structured reports |
| [`docs/ALPHARAVIS_UI_INTEGRATION_TEMPLATE.md`](docs/ALPHARAVIS_UI_INTEGRATION_TEMPLATE.md) | Contract for browser UIs connecting to the AlphaRavis brain |
| [`docs/ALPHARAVIS_RESPONSES_FULL_STREAMING_PLAN.md`](docs/ALPHARAVIS_RESPONSES_FULL_STREAMING_PLAN.md) | Investigation notes for Responses streaming with tool calls |

## Prerequisites

- Docker & Docker Compose v2
- (Optional) Local LLMs running via Ollama or llama.cpp
- (Optional) Node.js for MCP tool servers

## Install Variants

```bash
# Default: Tailscale HTTPS, hybrid streaming
make install

# LAN HTTP mode (no Tailscale)
make install TAILSCALE_AUTO=off

# Specific streaming profile
make install STREAMING=full
make install STREAMING=chat-full PROFILES=openwebui

# With vision/media
make install VISION_ENABLED=true VISION_URL=http://host:port/v1 VISION_MODEL=model-name
```

## Repo Structure

```
ai-stack/
├── langgraph-app/          AlphaRavis brain: agent graph, toolsets, bridge, MCP
├── hermes-agent/           Hermes coding agent (Git submodule)
├── rag-api-repo/           Document RAG backend (Git submodule)
├── langchain-bridge-repo/  Bridge helper library (Git submodule)
├── local-deep-researcher-repo/  Deep research workflows (Git submodule)
├── pixelle-mcp-custom/     Pixelle MCP for image tools
├── submodules/
│   ├── deep-agents-ui/     Primary LangGraph-native UI (forked)
│   └── OfficeCLI/          Office document CLI tool
├── agent-custom-ui/        Alternative agent frontend
├── scripts/                Setup, config server, patches
├── docker/                 Custom Dockerfiles
├── docs/                   All documentation
├── tests/                  Integration and protocol tests
├── .env(exaple)            Documented settings template
├── docker-compose.yml      Full stack definition
├── Makefile                Operator interface
└── librechat.yaml          LibreChat endpoint configuration
```
