# 🚀 AlphaRavis

**AlphaRavis** is a state-of-the-art, fully local all-in-one AI stack. It orchestrates local and cloud-based LLMs, equips them with long-term memory, provides them with powerful code execution and web research tools, and bundles everything into a sleek, professional user interface.

This repository combines RAG, agentic workflows (LangGraph), model routing (LiteLLM), code execution (MCP/OpenCode), and a modern chat interface (LibreChat / LangFlow).

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
4. **Data & Cache:** `mongodb` (State & Long-Term Memory) and `redis` (Semantic Caching).
5. **Tools:** `opencode-server` (Headless code worker) and `rag_api` (Document embedding service).

## 🚀 Quickstart

### Prerequisites
- Docker & Docker Compose
- Node.js (for MCP tool execution)
- (Optional) Local LLMs running via Ollama or Llama.cpp.

### Installation

1. **Clone the repository with submodules:**
   ```bash
   git clone --recurse-submodules <your-repo-url>
   cd ai-stack
