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
- Optional pgvector semantic memory can index previews/source keys for memories, turns, archives, artifacts, skills, and lessons while MongoDB remains the source of truth.
- Agents can inspect a lazy optional-tool manifest and use scoped agent/global memories.
- Reviewed repo skill cards live in `ai-skills/`, starting with DeepAgents agent-building and research workflows, and can be read on demand by AlphaRavis.
- Risky local or SSH commands require approval before execution.

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
