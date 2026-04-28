# AlphaRavis Architecture And Capabilities

This file is the editable on-demand self-description for AlphaRavis.
It is intentionally not injected into every chat. Agents should read it only
when the user asks what AlphaRavis is, how it works, what it can do, or how the
stack is wired.

## Identity

AlphaRavis is a multi-agent AI system built around a native LangGraph "brain"
and an OpenAI-compatible FastAPI bridge for LibreChat.

The system is designed to be:

- Inspectable through LangGraph Studio and DeepAgents UI.
- Usable from LibreChat through OpenAI-compatible chat/completions endpoints.
- Modular, so the agent brain and the LibreChat bridge can evolve separately.
- Safe by default for debugging commands through human approval gates.
- Memory-aware, with active context compression and long-term retrieval.

## High-Level Containers

The current Docker architecture is split into these main roles:

- `langgraph-api`: the brain. Runs the LangGraph graph `alpha_ravis`.
- `api-bridge`: the mouth. Exposes OpenAI-compatible `/v1/models`,
  `/v1/chat/completions`, and a compatibility `/v1/responses` wrapper for
  LibreChat or other OpenAI-style clients.
- `litellm`: model gateway. Routes AlphaRavis model calls to configured backends such as llama.cpp or Ollama.
- `mongodb`: LangGraph checkpointing and long-term store backing.
- `vectordb`: Postgres with pgvector. It can act as an optional semantic
  search sidecar for AlphaRavis memory; it does not replace MongoDB.
- `redis`: optional LLM cache, not the primary checkpointer in this phase.
- `deep-agents-ui`: visual UI for DeepAgents/LangGraph-style inspection.
- `librechat`: the normal chat UI for the user.
- `rag_api`: local document search backend when available.
- Pixelle/MCP services: image generation and Pixelle tool integration when available.
- `hermes-agent`: optional external coding/system agent reached through its
  OpenAI-compatible API on the host.

## Hermes Integration

Hermes is integrated as an optional external coding/system agent, not as a
replacement for the AlphaRavis LangGraph brain.

Supported paths:

```text
LibreChat -> Hermes Agent
LibreChat -> AlphaRavis LangGraph bridge
AlphaRavis LangGraph -> Hermes coding sub-agent
Hermes -> AlphaRavis LangGraph tool endpoint, only when explicitly enabled
```

LibreChat has a separate custom endpoint for Hermes in `librechat.yaml`.
The default Docker-side base URL is:

```text
HERMES_API_BASE=http://host.docker.internal:8642/v1
HERMES_MODEL=hermes-agent
```

For containers to reach a host-running Hermes gateway on Linux, Hermes should
bind to `API_SERVER_HOST=0.0.0.0` rather than only `127.0.0.1`.

LangGraph can call Hermes through the `hermes_coding_agent` swarm worker when:

```text
ALPHARAVIS_ENABLE_HERMES_AGENT=true
```

The Hermes worker is meant for coding, file analysis, terminal-oriented
diagnosis, project-structure inspection, and implementation guidance. It calls
Hermes with a system guard that forbids calling LangGraph back from that run.

The reverse path is optional and disabled by default:

```text
BRIDGE_ENABLE_LANGGRAPH_TOOL=false
POST /tools/langgraph/run
```

That endpoint requires `explicit_user_request=true` in the request body. This
prevents Hermes from silently invoking LangGraph unless the user explicitly asks
for AlphaRavis/LangGraph/custom-agent flow.

## MCP Integration

AlphaRavis uses a DeepAgents-style MCP configuration pattern.

Default config:

```text
langgraph-app/mcp.json
```

The config uses the familiar shape:

```json
{
  "mcpServers": {
    "pixelle": {
      "type": "sse",
      "url": "${PIXELLE_URL}/pixelle/mcp/sse"
    }
  }
}
```

MCP tools remain lazy by default:

```text
ALPHARAVIS_LOAD_MCP_TOOLS=false
```

When enabled, AlphaRavis loads configured MCP servers through
`MultiServerMCPClient`, prefixes tool names by server when supported, records
server/tool metadata for `describe_optional_tool_registry`, and keeps stdio MCP
servers disabled unless explicitly trusted:

```text
ALPHARAVIS_MCP_ALLOW_STDIO=false
```

This keeps the useful DeepAgents MCP pattern without letting arbitrary project
MCP configs start local processes by accident.

## Core Request Flow

1. The user chats in LibreChat.
2. LibreChat calls `api-bridge` using an OpenAI-compatible request.
3. The bridge maps the LibreChat conversation id to a deterministic LangGraph thread id.
4. The bridge calls the LangGraph API graph `alpha_ravis`.
5. LangGraph runs the agent swarm, tools, memory guards, and approval gates.
6. The bridge streams or returns the final response back to LibreChat.

The bridge separates visible chat UI from internal LangGraph state. LibreChat can
keep its full visible history, while LangGraph keeps checkpointed thread state
and compressed memory.

## Bridge Behavior

The bridge is implemented in `langgraph-app/bridge_server.py`.

Important behavior:

- It exposes `/v1/models`.
- It exposes `/v1/chat/completions`.
- It exposes `/v1/responses` as a compatibility wrapper around the same
  LangGraph run path.
- It supports non-streaming and OpenAI-compatible SSE streaming.
- It can stream LangGraph message events.
- It can optionally forward reasoning/thinking deltas as a separate SSE delta
  field when `BRIDGE_STREAM_REASONING_EVENTS=true`. Normal visible content
  still strips reasoning blocks.
- It can optionally emit short visible status/activity messages.
- It handles LangGraph human approval interrupts and lets the user reply with:
  - `approve`
  - `reject`
  - `replace: <safer command>`

The bridge uses `BRIDGE_MESSAGE_SYNC_MODE=delta` by default. This means that
after an existing LangGraph thread has state, the bridge sends only new user
messages into LangGraph instead of re-sending the whole LibreChat history every
turn. This keeps LangGraph compression useful and avoids old messages being
reintroduced forever.

## Agent Graph

The main graph lives in `langgraph-app/agent_graph.py`.

The graph id is:

```text
alpha_ravis
```

The graph is built as:

```text
START
  -> run_profile_start
  -> route_decision
  -> fast_chat OR planner
  -> memory_kernel_before when the agent path is selected
  -> skill_library when the agent path is selected
  -> handoff_context_guard when the agent path is selected
  -> alpha_ravis_swarm when the agent path is selected
  -> memory_kernel_after when the agent path is selected
  -> context_guard_after
  -> memory_notice
  -> run_profile_finish
  -> END
```

## Agents

AlphaRavis currently uses a swarm-style multi-agent setup.

### General Assistant

Default agent for normal tasks.

Capabilities:

- General chat.
- Pixelle image job start and async status checks.
- Wake-on-LAN for configured remote PCs.
- Fast web search.
- LangMem manage/search memory tools.
- Skill candidate creation.
- Safe handoff to specialists through structured handoff packets.

Safety:

- The General Assistant does not get a raw DeepAgents shell backend.
- Local and SSH command diagnostics are routed to the Debugger Agent, where
  AlphaRavis command approval gates are enforced.

### Research Expert

Handles deeper research.

Capabilities:

- Tavily-based deeper web search.
- Local document search through the RAG API.
- Handoff to general, debugger, or context retrieval.

### Debugger Agent

Handles infrastructure problems and failed jobs.

Capabilities:

- SSH diagnostics against configured remote PCs.
- Local Docker/log/repo diagnostics from the LangGraph container.
- Past debugging lesson search.
- Debugging lesson recording.
- Skill candidate recording when a reusable workflow emerges.

Safety:

- Destructive or state-changing commands trigger a LangGraph human approval interrupt.
- Read-only diagnostics such as logs and status checks can run without approval.

### Hermes Coding Agent

Optional specialist that delegates bounded coding/system tasks to an external
Hermes Agent API.

Capabilities:

- Check Hermes API reachability.
- Ask Hermes for coding, file-analysis, terminal-oriented diagnosis, project
  inspection, patch planning, or implementation guidance.
- Return structured handoff reports to the swarm.

Safety:

- Disabled until `ALPHARAVIS_ENABLE_HERMES_AGENT=true`.
- Calls Hermes with an anti-recursion system prompt.
- Does not expose AlphaRavis command approval bypasses.
- If Hermes needs LangGraph, the request is transferred back inside AlphaRavis
  instead of recursively calling Hermes again.

### UI Assistant

Handles browser, VNC, and desktop-style tasks when the optional UI stack is available.

### Context Retrieval Agent

Handles archived memory retrieval.

Capabilities:

- Search archived context from the current chat thread.
- Search debugging lessons.
- Search active workflow skills.
- Search other chat archives only when explicitly requested through `include_other_threads=true`.

### Power Management Agent

Handles local hardware and model lifecycle planning when advanced custom model
management is enabled:

```text
ALPHARAVIS_ENABLE_MODEL_MANAGEMENT=true
ALPHARAVIS_ENABLE_ADVANCED_MODEL_MANAGEMENT=true
```

It is disabled by default so normal single-model stacks continue to use only
the standard `big-boss` route.

Capabilities:

- Inspect big llama.cpp reachability.
- Inspect Ollama running models on the small management node.
- Inspect ComfyUI readiness before Pixelle work.
- Plan safe embedding windows for `memory-embed`.
- Send Wake-on-LAN through the existing configured `REMOTE_PCS` tool.
- Request shutdown/service/model-switch actions through a curated external
  action endpoint.

Safety:

- Shutdowns, service starts/stops, Ollama model switches, and embedding-job
  execution are dry-run by default.
- Real actions require:

```text
ALPHARAVIS_MODEL_MGMT_ACTION_URL=<your curated tool endpoint>
ALPHARAVIS_MODEL_MGMT_ALLOW_ACTIONS=true
```

- The action endpoint receives a small JSON object:

```json
{"action": "wake_pc", "payload": {"target": "comfy_server", "reason": "..."}}
```

This is intentionally separate from prompt-generated SSH commands.

## Memory Layers

AlphaRavis has multiple memory layers.

### MemoryKernel

The MemoryKernel is the Hermes-inspired learning layer around the LangGraph
swarm. It is not a replacement for checkpointing.

On the normal agent path it does four small jobs:

1. Prefetches tiny curated memories that match the current turn.
2. Adds an invisible memory nudge every `ALPHARAVIS_MEMORY_NUDGE_INTERVAL`
   user turns.
3. Indexes completed turns into a thread-scoped session-history namespace.
4. Gives compression a small list of memory-worthy details to preserve.

Fast Path skips the MemoryKernel so simple chat stays cheap.

When `ALPHARAVIS_VECTOR_BACKEND=pgvector`, the MemoryKernel also writes a source
catalog and full retrieval chunks to pgvector for newly saved memories, session
turns, archives, archive collections, artifacts, debugging lessons, and skill
candidates. MongoDB/store/artifact files remain the source of truth, while
pgvector is the searchable Inhaltsverzeichnis and chunk index.

Relevant settings:

```text
ALPHARAVIS_ENABLE_MEMORY_KERNEL=true
ALPHARAVIS_MEMORY_NUDGE_INTERVAL=10
ALPHARAVIS_MEMORY_KERNEL_PRECOMPRESS_NOTES=true
ALPHARAVIS_VECTOR_BACKEND=pgvector
```

### LangGraph Checkpoints

LangGraph checkpointing stores thread state. This includes message state,
active agent state, compression summaries, and other graph state fields.

The checkpointer is configured through `langgraph.json` and MongoDB.

### LangMem Memories

LangMem tools are available for normal durable memories. They are separate from
the raw chat history and can store user preferences or useful persistent facts.

### Agent-Specific Memories

AlphaRavis also has explicit agent-scoped memories:

```text
alpharavis / agent_memories / general_assistant
alpharavis / agent_memories / research_expert
alpharavis / agent_memories / debugger_agent
alpharavis / agent_memories / context_retrieval_agent
alpharavis / agent_memories / global
```

Agents are instructed to search their own memory first and global memory second.
Global memory is for stable cross-agent preferences or lessons. Agent-specific
memory is for habits, recurring issues, or lessons that belong to one role.

### Curated Always Memory

Curated memory is the small Hermes-style memory layer. It is separate from raw
chat archives and separate from long LangMem memories.

Curated memory should contain only stable facts:

- user preferences,
- environment facts,
- recurring tool quirks,
- lessons that reduce future correction.

It should not contain long logs, one-off task progress, or full procedures.
Those belong in thread archives, artifacts, or skills.

Agents can use:

```text
search_curated_memory
record_curated_memory
```

The MemoryKernel may inject a tiny matching curated-memory block into the
agent path. It is fenced as background context, not user input.

Default limits:

```text
ALPHARAVIS_ALWAYS_MEMORY_MAX_ITEMS=6
ALPHARAVIS_ALWAYS_MEMORY_MAX_CHARS=2200
ALPHARAVIS_CURATED_MEMORY_ENTRY_MAX_CHARS=1200
```

### Session-History Search

Hermes uses SQLite + FTS5 for past-session recall. AlphaRavis now mirrors that
pattern through the LangGraph Store:

```text
alpharavis / threads / <thread_id> / session_turns
alpharavis / session_turn_index
```

The normal search mode is current-thread only. Cross-thread search is available
only when a tool call explicitly sets `include_other_threads=true`.

The implementation uses LangGraph Store search, so it can benefit from the
active store backend's text or vector search behavior without dumping whole
threads into the prompt.

Agents can use:

```text
search_session_history
```

This is useful when the user says things like "what did we do earlier in this
chat?" without loading the whole raw archive.

### Debugging Lessons

The debugger can store lessons learned from infrastructure failures:

- problem
- root cause
- fix
- signals
- commands
- outcome

These are used to avoid repeating old debugging mistakes.

### Skill Library

The skill library stores reusable workflow patterns.

Important safety rule:

- New workflows are stored as inactive `candidate` skills.
- Candidates do not affect routing.
- Promotion to active skill is disabled by default.
- Active skills are still non-binding hints, not automatic execution.
- Candidate listing is available for review. Activation and deactivation require
  `ALPHARAVIS_ALLOW_SKILL_PROMOTION=true`.

### Repo AI Skills

Version-controlled skill cards can live under `ai-skills/`.

The first reviewed skill card is:

```text
ai-skills/deepagents-agent-builder/SKILL.md
```

Additional reviewed research skill cards include:

```text
ai-skills/deep-research-report/SKILL.md
ai-skills/market-research/SKILL.md
ai-skills/competitor-analysis/SKILL.md
ai-skills/hermes-agent-integration/SKILL.md
```

These cards are not injected into every chat by default; agents should read them
only when the user asks for matching agent-building or research workflows.

Agents can use:

```text
list_repo_ai_skills
read_repo_ai_skill
```

These tools are restricted to the repo `ai-skills/` directory.

Before the agent path, AlphaRavis may inject a tiny metadata hint for matching
repo skill cards. This hint contains only names and descriptions. Full skill
instructions are loaded only when an agent calls `read_repo_ai_skill`.

## Context Compression

AlphaRavis uses two compression tiers.

### Chat Compression

When the active LangGraph message window exceeds `ALPHARAVIS_ACTIVE_TOKEN_LIMIT`,
older messages are summarized after the current run has produced its answer.
Compression no longer runs before the task starts.

Default:

```text
ALPHARAVIS_ACTIVE_TOKEN_LIMIT=10000
ALPHARAVIS_CONTEXT_KEEP_LAST_MESSAGES=12
ALPHARAVIS_ENABLE_POST_RUN_COMPRESSION=true
```

What happens:

1. Older messages are summarized.
2. The current task brief and latest handoff packet are preserved verbatim when
   available.
3. Recent messages are kept verbatim.
4. A thread-specific archive record is stored.
5. The active LangGraph message list is replaced by:
   - the current task brief, if present
   - the latest handoff packet, if present
   - one summary system message
   - the recent messages
6. A visible Memory-Notice can be returned to LibreChat.

### Handoff Context Guard

Before the swarm starts, AlphaRavis can run a lighter handoff-context guard. It
does not wait until the final post-run compression threshold. If the active
message window is already too large, the guard compresses the beginning of the
run into a handoff summary, archives the exact original messages, and keeps the
important coordination material active:

- current task brief
- planner execution plan
- MemoryKernel and skill hints
- latest handoff packet
- recent messages

Default:

```text
ALPHARAVIS_ENABLE_HANDOFF_CONTEXT_GUARD=true
ALPHARAVIS_HANDOFF_CONTEXT_TOKEN_LIMIT=8500
ALPHARAVIS_HANDOFF_CONTEXT_KEEP_LAST_MESSAGES=16
ALPHARAVIS_HANDOFF_PACKET_MAX_CHARS=4000
ALPHARAVIS_HANDOFF_SUMMARY_MAX_CHARS=2600
```

Agents are instructed to call `build_specialist_report` before `transfer_to_*`.
That report is the handoff packet and should state completed work, evidence,
commands/files/tools, verification status, risks, open tasks, and the exact
next-agent instruction.

The user can pause compression for one run by saying things such as:

- `keine Kompression`
- `nicht komprimieren`
- `skip compression`
- `no compression`

The user can force compression for one run by saying things such as:

- `komprimiere jetzt`
- `archiviere jetzt`
- `compress now`

Custom force phrases can be set with `ALPHARAVIS_MANUAL_COMPRESSION_PATTERNS`.

### Archive Collection Compression

When many archive records accumulate inside one chat thread, older archive
records can be summarized into an archive collection.

Default:

```text
ALPHARAVIS_ARCHIVE_TOKEN_LIMIT=50000
ALPHARAVIS_ARCHIVE_KEEP_RECENT_RECORDS=8
```

Raw archive records are not deleted. Archive collections are summaries with
references to child archive keys.

### Artifacts

Long reports, logs, plans, and intermediate notes should go to artifacts
instead of chat. Artifacts are disk-backed and indexed in the LangGraph Store.

Default root:

```text
/workspace/artifacts/alpharavis
```

Agents can use:

```text
write_alpha_ravis_artifact
read_alpha_ravis_artifact
list_alpha_ravis_artifacts
```

Artifacts are thread-scoped by default, with optional cross-thread listing only
when explicitly requested. The artifact index stores metadata and a small
preview; the full content stays on disk.

### Semantic Vector Memory And Source Catalog

AlphaRavis uses the existing `vectordb` Postgres/pgvector service as a semantic
retrieval index. MongoDB remains the ground truth for checkpoints, Store data,
archives, and thread state; pgvector stores a searchable index built from the
complete original source data.

Default:

```text
ALPHARAVIS_VECTOR_BACKEND=pgvector
ALPHARAVIS_ENABLE_PGVECTOR_MEMORY=true
```

For each new source, AlphaRavis writes:

- one catalog/Inhaltsverzeichnis row generated from the full original data
- full overlapping retrieval chunks
- `source_type`, such as `session_turn`, `archive`, `artifact`, `skill`,
  `curated_memory`, `agent_memory`, `debugging_lesson`, or `external_document`
- `source_key`, which points back to MongoDB/Store/artifact/RAG source
- `thread_id`, `thread_key`, chunk position, metadata, embedding model, and
  embedding vector

The catalog row lists source metadata, headings, file paths, URLs, code symbols,
database/RAG topics, and a chunk map. It is not model-invented memory; it is
extracted from the original data so retrieval can later answer "what was in
this conversation/source?" without loading every raw record first.

Session turns use a sliding window by default:

```text
ALPHARAVIS_PGVECTOR_SESSION_WINDOW_TURNS=2
```

That means the indexed text for a new turn includes the previous completed turn
plus the current one, so references such as "that bug" keep their context.

Artifacts and archives are chunked with overlap:

```text
ALPHARAVIS_PGVECTOR_CHUNK_MAX_CHARS=6000
ALPHARAVIS_PGVECTOR_CHUNK_OVERLAP_CHARS=800
```

This means long code, logs, reports, and compressed archives are fully
retrievable without becoming one oversized embedding input.

The tool exposed to agents is:

```text
semantic_memory_search
```

By default it searches the current thread plus global memories and federates
with `rag_api` for external document hits. It searches other AlphaRavis threads
only when a tool call explicitly sets `include_other_threads=true`. Enabling
this backend indexes new records from that point onward. Existing MongoDB/store
history is intentionally not bulk-backfilled automatically.

## Thread Isolation

Archive memory is scoped by LangGraph thread id:

```text
alpharavis / threads / <thread_id> / archives
alpharavis / threads / <thread_id> / archive_collections
```

This prevents normal retrieval from mixing different LibreChat conversations.

Cross-thread retrieval exists only as an explicit search mode and is limited by
`ALPHARAVIS_CROSS_THREAD_ARCHIVE_SEARCH_LIMIT`.

## Tool Safety

SSH and local shell diagnostic tools use a command classifier.

Read-only examples:

- `docker ps`
- `docker logs ...`
- `git status`
- `git diff`
- `ls`
- `cat`

Risky examples that require approval:

- deletion
- file moves
- service restarts
- Docker stop/restart/up/down operations
- package installs
- git commit/push/reset
- shell redirection

If approval is needed, LangGraph interrupts the run and asks the user.

## Pixelle

Pixelle image jobs can be started through either:

- `start_pixelle_remote`, which starts and waits through a durable LangGraph
  monitoring task.
- `start_pixelle_async`, which returns a job id immediately.
- `check_pixelle_job`, which checks the job id later.

Monitoring is implemented as a LangGraph `@task`, so it is visible and resumable
inside LangGraph execution rather than being a loose FastAPI background task.

Before a Pixelle job starts, AlphaRavis can run a ComfyUI preflight through the
model-management layer:

```text
ALPHARAVIS_ENABLE_MODEL_MANAGEMENT=true
ALPHARAVIS_ENABLE_ADVANCED_MODEL_MANAGEMENT=true
ALPHARAVIS_PIXELLE_PREPARE_COMFY=true
ALPHARAVIS_COMFY_HEALTH_URL=http://<comfy-ip>:8188/system_stats
```

If ComfyUI is reachable, Pixelle starts normally. If ComfyUI is offline,
AlphaRavis can request a wake action, but real power actions stay dry-run until
the curated action endpoint is configured. By default Pixelle warns and still
tries the job; set this to block instead:

```text
ALPHARAVIS_PIXELLE_BLOCK_IF_COMFY_OFFLINE=true
```

If Pixelle MCP tools are unavailable, AlphaRavis should fail cleanly and route
debuggable context instead of crashing.

## Model Backend

AlphaRavis talks to LiteLLM through an OpenAI-compatible client.

LiteLLM routes `big-boss` to the configured llama.cpp server and can also route
other configured models such as Ollama models.

The current controlled fallback path is:

```text
fast path only: big-boss -> edge-gemma
```

`big-boss` uses the llama.cpp OpenAI-compatible `/v1` endpoint.
`edge-gemma` uses the Ollama OpenAI-compatible `/v1` endpoint.

Global LiteLLM fallback is intentionally not enabled for every request.
`edge-gemma` is treated as a small starter/crisis model, not as a second boss.
Complex swarm/tool workflows stay on `big-boss` and should fail visibly if the
large backend is unavailable. Only the direct fast-chat path can fall back to
`edge-gemma`, controlled by the `ALPHARAVIS_FAST_PATH_*` variables in `.env`.

The bridge also exposes:

```text
GET /health/llm-generation
```

This endpoint performs a real tiny generation against the primary model and the
fallback model. It is meant to detect the "server is online but generation is
stuck" failure mode.

If the bridge returns a timeout but `/v1/models` and health endpoints work, the
most likely cause is that the model generation backend is busy, stuck, or too
slow for the current timeout. That does not automatically mean the bridge is
broken.

Automatic power actions such as SSH shutdown or Wake-on-LAN are intentionally
not run by a hidden background watchdog. The Power Management Agent can inspect
and plan them, and Wake-on-LAN can be called explicitly, but destructive actions
must go through a curated endpoint or the debugger approval gate.

The custom model-management layer is off by default:

```text
ALPHARAVIS_ENABLE_MODEL_MANAGEMENT=false
ALPHARAVIS_ENABLE_ADVANCED_MODEL_MANAGEMENT=false
ALPHARAVIS_ENABLE_CRISIS_MANAGER=false
```

When enabled for the custom local setup, the embedding model lives on the Ollama
management node. Because that node may not be able to keep both the chat/crisis
model and the embedding model loaded, AlphaRavis plans embedding windows instead
of blindly loading the model:

```text
ALPHARAVIS_EMBEDDING_LOAD_POLICY=idle_or_big_llm_active
ALPHARAVIS_MODEL_IDLE_SECONDS=600
ALPHARAVIS_OLLAMA_CHAT_MODEL=gemma4:e2b
ALPHARAVIS_OLLAMA_EMBED_MODEL=Q78KG/gte-Qwen2-1.5B-instruct
```

The intended flow is:

1. Keep MongoDB/store as source of truth.
2. Queue pgvector indexing work safely.
3. When the system is idle or `big-boss` is reachable, switch Ollama into the
   embedding model window.
4. Run queued embedding jobs.
5. Restore the small chat/crisis model if needed.

## Fast Path And Run Profile

Short non-tool chat requests can use a direct fast path:

```text
START
  -> run_profile_start
  -> route_decision
  -> fast_chat
  -> context_guard_after
  -> memory_notice
  -> run_profile_finish
  -> END
```

Fast path skips skill-library retrieval and the swarm. It is meant for simple
chat, wording, translation, or short explanations. It is not used for debugging,
tools, files, Pixelle, memory/archive retrieval, research, Docker, SSH, PC
control, or architecture questions.

Fast-path replies are visibly marked when:

```text
ALPHARAVIS_SHOW_FAST_PATH_NOTICE=true
```

When `ALPHARAVIS_FAST_PATH_LOCK_AFTER_SWARM=true`, a thread that once routes to
the normal agent/swarm path is locked out of fast path for future turns. This
keeps mixed complex threads from later falling back to the simple route.

For llama.cpp/Qwen-style models, fast path passes:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

This prevents simple replies from spending seconds generating hidden reasoning
tokens before returning a tiny answer.

Set `ALPHARAVIS_FAST_PATH_DISABLE_THINKING=false` to allow hidden thinking in
fast path again.

Optional MCP tools are also not loaded by default during graph construction:

```text
ALPHARAVIS_LOAD_MCP_TOOLS=false
```

This avoids paying MCP startup cost on every simple chat. Native tools such as
`start_pixelle_remote`, `start_pixelle_async`, and `check_pixelle_job` remain
available without loading the Pixelle MCP tool registry. Set the flag to `true`
only when those extra MCP-provided tools are needed.

Agents can call `describe_optional_tool_registry` to see configured MCP servers,
load status, warning messages, and loaded tool names without loading the
registry during normal graph startup.

The normal agent path remains:

```text
route_decision
  -> skill_library
  -> alpha_ravis_swarm
```

Every run stores a `run_profile` object in LangGraph state with route, reason,
message count, estimated tokens, timing, and fast-path fallback information.
Set `ALPHARAVIS_SHOW_RUN_PROFILE=true` only when you want this profile appended
visibly in chat; otherwise inspect it in LangGraph Studio or DeepAgents UI.

## Observability

Available observation points:

- LibreChat: user-facing chat and visible Memory-Notice or approval prompts.
- Bridge SSE stream: OpenAI-compatible chunks, optionally with Status messages.
- LangGraph Studio: graph nodes, state, checkpoints, time travel.
- DeepAgents UI: agent-oriented visual inspection.
- Docker logs: service-level debugging.

## How Agents Should Use This File

Do not include this file in every response.

Read this file only when the user asks about:

- what AlphaRavis is
- what AlphaRavis can do
- how the architecture works
- bridge behavior
- memory/compression behavior
- skill library behavior
- safety/approval behavior
- debugging capabilities

When answering, summarize the relevant part. Do not dump the whole file unless
the user explicitly asks for the full document.
