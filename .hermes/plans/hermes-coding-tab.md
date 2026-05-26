# Hermes Coding Tab — Implementation Plan

## Goal

Add a "Coding" tab to Deep Agents UI that connects directly to Hermes Agent,
with two switchable modes and AlphaRavis tool forwarding.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                Deep Agents UI (:3000)               │
│  ┌──────────┬──────────────┬──────────────────────┐ │
│  │ Chat Tab │  Office Tab  │  Coding Tab (NEU)    │ │
│  │ →Swarm   │  →OfficeCLI  │  →Hermes Agent       │ │
│  └──────────┴──────────────┴──────────────────────┘ │
│                                                     │
│  Provider Switch: [AlphaRavis Swarm] [Hermes]       │
│  Mode Switch:    [Direct] [Orchestrated]            │
└─────────────────────────────────────────────────────┘
```

## Phase 1: Coding Tab — Hermes Direct (SSE Live)

### 1a. `HermesPanel.tsx` — new component

- SSE connection to `http://localhost:8642/v1/chat/completions`
- Stream parsing: `text_delta`, `tool_call`, `tool_result`, `pending_approval`
- Chat UI: messages, tool-call cards, approve/deny buttons
- Workspace file browser (read from mounted `/workspace`)
- Session list from Hermes `state.db` (SQLite read-only)

### 1b. Page integration — `page.tsx`

- Add "Coding" button next to Chat/Office toggle
- `activeView` extended: `"chat" | "office" | "coding"`
- Config: `NEXT_PUBLIC_HERMES_API_URL=http://localhost:8642/v1`

### 1c. Docker env vars

```env
NEXT_PUBLIC_HERMES_API_URL=http://localhost:8642/v1
NEXT_PUBLIC_HERMES_API_KEY=sk-hermes-local
```

## Phase 2: Orchestrated Mode (AlphaRavis Middleware)

### 2a. Hermes Bridge in LangGraph

New tool: `call_hermes_streaming(task, context, stream=true)`

- Streams Hermes SSE through AlphaRavis
- Captures full output → `write_alpha_ravis_artifact("hermes-run-{id}")`
- Records memory: `record_agent_memory(what_was_done)`
- Relays SSE events to browser in real-time

### 2b. Pre-Load Context

Before calling Hermes:
1. `semantic_memory_search(task)` → inject key memories
2. `search_session_history(task)` → inject session summaries
3. `agentic_rag_retrieve(task)` → inject RAG chunks
4. `list_repo_ai_skills()` → inject skill names
Pack everything as compact context block.

### 2c. Tool Forwarding

Hermes can request AlphaRavis tools via structured output:
```json
{"need_alpha_ravis_tool": "search_memory", "query": "auth.py refactoring"}
```
→ AlphaRavis executes tool, appends result, retries Hermes call (max 1 retry).

## Phase 3: Mode Switch in UI

### 3a. Provider dropdown (bottom bar)

```
[AlphaRavis Swarm ▼]  → Chat Tab
[Hermes Direct    ▼]  → Coding Tab (Phase 1)
[Hermes + Alpha   ▼]  → Coding Tab (Phase 2)
```

### 3b. Mode toggle per coding session

- "Direct": bare Hermes, no AlphaRavis overhead
- "Orchestrated": AlphaRavis pre-loads context, captures artifacts, forwards tools

## Files Changed

| File | Change |
|---|---|
| `submodules/deep-agents-ui/src/app/page.tsx` | Add Coding tab + provider switch |
| `submodules/deep-agents-ui/src/app/components/HermesPanel.tsx` | NEW — Hermes SSE chat |
| `submodules/deep-agents-ui/src/app/hooks/useHermesChat.ts` | NEW — SSE stream hook |
| `langgraph-app/agent_graph.py` | `call_hermes_streaming()` tool |
| `docker-compose.yml` | Hermes env vars for deep-agents-ui |
| `.env(exaple)` | Document new vars |
