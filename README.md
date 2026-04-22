# 🚀 AlphaRavis

**AlphaRavis** ist eine lokale, hochmoderne All-in-One AI-Stack-Lösung. Es orchestriert lokale und cloud-basierte LLMs, verbindet sie mit einem Langzeitgedächtnis, verleiht ihnen Werkzeuge zur Code-Ausführung und Web-Recherche und bündelt alles in einer eleganten Benutzeroberfläche.

Dieses Repository vereint RAG, Agentic Workflows (LangGraph), Model-Routing (LiteLLM), Code-Ausführung (MCP/OpenCode) und ein modernes Chat-Interface (LibreChat / LangFlow).

## ✨ Kernfunktionen

- 💬 **LibreChat Frontend:** Eine ChatGPT-ähnliche UI als Hauptschnittstelle für alle Agenten.
- 🧠 **LangGraph Chef-Agent:** Ein intelligenter Orchestrator-Agent, der Aufgaben delegiert (z.B. schnelle Suchen vs. stundenlange Deep-Research).
- 💾 **Persistentes Gedächtnis:** Kurzzeit- und Langzeitgedächtnis via MongoDB (Checkpointer & BaseStore) für kontextbezogene Unterhaltungen über Tage hinweg.
- 🛠️ **MCP (Model Context Protocol):** Voller Zugriff auf lokale Workspaces via OpenCode. Der Agent kann Code lesen, schreiben und ausführen.
- 📚 **Lokales RAG:** Dokumenten-Analyse und Vector-Suche auf Basis von PostgreSQL (pgvector).
- 🔀 **LiteLLM Routing:** Ein zentraler Gateway für alle Modelle (Ollama, Llama.cpp, OpenAI, etc.).
- 🌊 **LangFlow Ready:** Architektur vorbereitet für die visuelle Erstellung von Agenten-Flows.

## 🏗️ Systemarchitektur

AlphaRavis besteht aus mehreren Microservices, die per Docker Compose orchestriert werden:

1. **Frontend:** `librechat` (und optional `agent-custom-ui`)
2. **Brain / Logic:** `langgraph-api` (Dein maßgeschneiderter Python-Server mit Agenten-Logik)
3. **Model Gateway:** `litellm` (Verteilt Prompts an den großen PC oder kleine lokale Modelle)
4. **Memory & Cache:** `mongodb` (State & Long-Term Memory) + `redis` (LLM Caching)
5. **Tools:** `opencode-server` (Code-Arbeiter) + `rag_api` (Dokumenten-Vektorisierung)

## 🚀 Quickstart

### Voraussetzungen
- Docker & Docker Compose
- (Optional) Lokale LLMs via Ollama oder Llama.cpp im Netzwerk verfügbar.
- Wake-on-LAN fähiger Hauptrechner (falls große Modelle ausgelagert werden).

### Installation & Start

1. **Repository klonen:**
   ```bash
   git clone --recurse-submodules <deine-repo-url>
   cd ai-stack
