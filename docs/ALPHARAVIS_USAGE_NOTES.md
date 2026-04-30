# AlphaRavis User Notes

This file explains what AlphaRavis uses automatically and what is used only
when you ask for it. It is meant for humans first, and agents may also read it
when asked how the system behaves.

## Daily Interface

Use LibreChat for normal chatting. It talks to `api-bridge`, which forwards the
request into the LangGraph `alpha_ravis` brain.

Use LangGraph Studio or DeepAgents UI when you want to inspect internal graph
steps, state, checkpoints, run profiles, memory compression, or agent routing.

## Fast Path

Short non-tool questions can take the fast path. This path:

- runs through LangGraph state,
- skips skill-library retrieval,
- skips the swarm,
- calls the model directly,
- can fall back to `edge-gemma` only for simple chat.

Fast-path replies are visibly marked by default:

```text
ALPHARAVIS_SHOW_FAST_PATH_NOTICE=true
```

Once one turn in a chat thread uses the normal agent/swarm path, that thread is
locked out of fast path by default:

```text
ALPHARAVIS_FAST_PATH_LOCK_AFTER_SWARM=true
```

This prevents a complex conversation from bouncing back into the simple route
later.

The fast path is not used when the message looks like it needs tools, research,
debugging, Docker, SSH, Pixelle, code/files, memory/archive search, PC control,
or AlphaRavis architecture details.

Optional MCP tool loading is off by default:

```text
ALPHARAVIS_LOAD_MCP_TOOLS=false
```

That prevents slow MCP startup from affecting every chat. AlphaRavis now uses a
DeepAgents-style MCP config loader: `mcp.json` / `.mcp.json` files describe
servers, while the agent can inspect the server manifest before tools are
loaded. The native Pixelle HTTP tool can still start Pixelle jobs without
loading the extra MCP registry.

Agents can still see a short manifest of optional registries through the
`describe_optional_tool_registry` tool, so they know Pixelle MCP exists and how
it can be enabled without paying the startup cost by default.

Default MCP config:

```text
ALPHARAVIS_MCP_CONFIG_PATH=/workspace/langgraph-app/mcp.json
ALPHARAVIS_MCP_TOOL_PREFIX=true
ALPHARAVIS_MCP_ALLOW_STDIO=false
ALPHARAVIS_MCP_STRICT=false
```

`ALPHARAVIS_MCP_ALLOW_STDIO=false` is intentional: stdio MCP can start local
processes, so only remote HTTP/SSE MCP servers are trusted by default.

For llama.cpp/Qwen-style models, fast path also disables hidden thinking with:

```text
ALPHARAVIS_FAST_PATH_DISABLE_THINKING=true
```

This keeps tiny replies from spending seconds on invisible reasoning tokens.
Set it to `false` if you explicitly want hidden thinking even in fast path.

To force the normal agent path for one message, write:

```text
kein fast path
```

## Big Model And Small Fallback

`big-boss` is the main model on the llama.cpp server.

`edge-gemma` is a small starter/crisis model on the Ollama management machine.
It is not intended for complex agent workflows or risky tool decisions.

Current rule:

```text
simple fast path may use edge-gemma as fallback
normal swarm/tool path stays on big-boss
```

If the big server is down, complex requests should fail visibly instead of
silently running on the weaker model.

Direct no-tool LangGraph model calls can use the Responses API:

```text
ALPHARAVIS_LLM_API_MODE=responses
ALPHARAVIS_RESPONSES_API_BASE=http://litellm:4000/v1
ALPHARAVIS_RESPONSES_MODEL=big-boss
```

This applies to direct calls such as planner, fast path, and summarizers. The
DeepAgents tool workers can also use LangChain's `ChatOpenAI` Responses mode:

```text
ALPHARAVIS_DEEPAGENTS_API_MODE=responses
ALPHARAVIS_DEEPAGENTS_RESPONSES_API_BASE=http://litellm:4000/v1
ALPHARAVIS_DEEPAGENTS_RESPONSES_MODEL=big-boss
ALPHARAVIS_DEEPAGENTS_RESPONSES_OUTPUT_VERSION=responses/v1
```

Set `ALPHARAVIS_DEEPAGENTS_API_MODE=chat_completions` to return only DeepAgents
tool workers to the older ChatLiteLLM path. Set
`ALPHARAVIS_DEEPAGENTS_REQUIRE_RESPONSES=true` only when you want startup to
fail instead of falling back. Set `ALPHARAVIS_RESPONSES_REQUIRE_NATIVE=true`
only when you want direct no-tool calls to fail instead of falling back to Chat
Completions.

## Model And Power Management

AlphaRavis has a custom `model_management.py` layer for your split hardware
setup.

Important idea:

- The Ollama management node is mainly for startup/crisis work.
- The embedding model should be loaded only during safe windows.
- A safe window means either the system has been idle long enough or the big
  llama.cpp server is up, so normal chat does not depend on the small Ollama
  node.

Default controls keep this custom layer completely off:

```text
ALPHARAVIS_ENABLE_MODEL_MANAGEMENT=false
ALPHARAVIS_ENABLE_ADVANCED_MODEL_MANAGEMENT=false
ALPHARAVIS_ENABLE_OWNER_POWER_TOOLS=false
ALPHARAVIS_ENABLE_CRISIS_MANAGER=false
ALPHARAVIS_ENABLE_POWER_MANAGEMENT=false
ALPHARAVIS_MODEL_MGMT_ALLOW_ACTIONS=false
ALPHARAVIS_POWER_MANAGER_MODEL=openai/edge-gemma
ALPHARAVIS_CRISIS_MANAGER_MODEL=openai/edge-gemma
ALPHARAVIS_MODEL_IDLE_SECONDS=600
ALPHARAVIS_EMBEDDING_LOAD_POLICY=idle_or_big_llm_active
```

With these defaults, AlphaRavis uses the normal `big-boss` route and does not
create the Power Management Agent. Enable the layer only on the custom hardware
setup that needs it.

The advanced hooks become visible only after:

```text
ALPHARAVIS_ENABLE_MODEL_MANAGEMENT=true
ALPHARAVIS_ENABLE_ADVANCED_MODEL_MANAGEMENT=true
ALPHARAVIS_ENABLE_OWNER_POWER_TOOLS=true
```

Even then, real shutdowns, service changes, Ollama model switching, and
embedding-job runs stay disabled until you provide:

```text
ALPHARAVIS_MODEL_MGMT_ACTION_URL=
ALPHARAVIS_MODEL_MGMT_API_KEY=
ALPHARAVIS_MODEL_MGMT_ALLOW_ACTIONS=true
```

`make model-management` can write these ENV switches interactively. The result
is still plain `.env`, so you can copy that `.env` to another machine and skip
the Make step later.

When enabled, the Power Management Agent handles questions like:

```text
check model management status
plane ein Embedding-Fenster
run embedding memory jobs
pruefe ob ComfyUI fuer Pixelle bereit ist
```

Owner-specific host/MAC/start-command defaults live in
`langgraph-app/owner_power_tools.py`. Keep real passwords in your private `.env`
through `ALPHARAVIS_OWNER_SSH_PASS`, not in git.

## Hermes Mode

Hermes is available as a separate optional coding/system agent.

Direct mode:

```text
LibreChat -> Hermes Agent
```

Use this for coding, terminal-oriented work, file operations, and direct agent
tasks when Hermes gateway is running.

AlphaRavis mode:

```text
LibreChat -> AlphaRavis -> hermes_coding_agent -> Hermes
```

Use this when AlphaRavis should stay the main supervisor but delegate a bounded
coding/system subtask to Hermes.

Required Hermes gateway settings:

```text
API_SERVER_ENABLED=true
API_SERVER_HOST=0.0.0.0
API_SERVER_PORT=8642
API_SERVER_KEY=<same as HERMES_API_KEY>
```

AlphaRavis settings:

```text
HERMES_API_BASE=http://host.docker.internal:8642/v1
HERMES_API_KEY=sk-hermes-local
HERMES_MODEL=hermes-agent
ALPHARAVIS_ENABLE_HERMES_AGENT=false
```

Keep `ALPHARAVIS_ENABLE_HERMES_AGENT=false` until Hermes is actually reachable.

Reverse mode is disabled by default:

```text
BRIDGE_ENABLE_LANGGRAPH_TOOL=false
```

When enabled, Hermes can call `POST /tools/langgraph/run`, but only with
`explicit_user_request=true`. This is the loop guard: Hermes may use LangGraph
only when you explicitly ask it to.

## Tools

Tools are used only when an agent chooses them for the task.

Examples that trigger tool-capable paths:

- "debugge den Fehler"
- "schau Docker logs"
- "starte meinen PC"
- "generiere ein Bild"
- "suche in meinen Dokumenten"
- "lies die Architektur von AlphaRavis"
- "suche in alten Archiven"

Risky local or SSH commands require a human approval interrupt before execution.
Reply with:

```text
approve
reject
replace: <safer command>
```

## Pixelle Jobs

For image generation, `start_pixelle_remote` starts a job and waits through a
durable LangGraph `@task`. This is best when you want AlphaRavis to stay with
the job until it finishes.

`start_pixelle_async` starts the job and returns a `job_id` immediately. This is
better for long jobs. Later you can ask:

```text
check_pixelle_job <job_id>
```

Before a Pixelle job starts, AlphaRavis can check ComfyUI:

```text
ALPHARAVIS_ENABLE_MODEL_MANAGEMENT=true
ALPHARAVIS_ENABLE_ADVANCED_MODEL_MANAGEMENT=true
ALPHARAVIS_ENABLE_OWNER_POWER_TOOLS=true
ALPHARAVIS_PIXELLE_PREPARE_COMFY=true
ALPHARAVIS_COMFY_HEALTH_URL=http://<comfy-ip>:8188/system_stats
```

If ComfyUI is offline, AlphaRavis warns. It only blocks the Pixelle job when:

```text
ALPHARAVIS_PIXELLE_BLOCK_IF_COMFY_OFFLINE=true
```

For the owner hardware setup, Pixelle preflight can also use the direct
owner-tool Wake-on-LAN path:

```text
ALPHARAVIS_ENABLE_OWNER_POWER_TOOLS=true
ALPHARAVIS_PIXELLE_OWNER_WAKE_COMFY=true
ALPHARAVIS_PIXELLE_OWNER_WAKE_WAIT_SECONDS=30
```

If Pixelle fails, the returned message includes debugger-ready context and asks
for Pixelle/LangGraph logs instead of crashing silently.

## Bridge Streaming

LibreChat normally uses:

```text
POST /v1/chat/completions
```

The bridge also offers:

```text
POST /v1/responses
```

`/v1/responses` is a compatibility wrapper around the same LangGraph run path.
Chat Completions remains available for LibreChat compatibility. Clients that
support richer Responses output items should prefer:

```text
BRIDGE_PREFERRED_API_MODE=responses
```

LibreChat custom endpoints may still call `/v1/chat/completions` depending on
LibreChat provider support. AlphaRavis exposes `/v1/responses` and uses OpenAPI
`3.1.0`; clients that support the newer Responses endpoint can call it directly.

Reasoning/thinking is stripped from normal visible answer text. If a client can
handle a separate reasoning delta field, enable:

```text
BRIDGE_STREAM_REASONING_EVENTS=true
BRIDGE_REASONING_DELTA_FIELD=reasoning_content
```

If LibreChat shows that reasoning as normal text, turn it back off.

## Bridge Context Hygiene

The bridge strips internal AlphaRavis context blocks from visible output by
default:

```text
BRIDGE_SCRUB_INTERNAL_CONTEXT=true
```

This protects LibreChat streaming output even when an internal tag such as
`<memory-context>` is split across multiple SSE deltas. The internal context can
still exist inside LangGraph state and Studio debugging; it is just not emitted
as normal assistant text.

Explicit user references can be expanded before the message reaches LangGraph:

```text
BRIDGE_ENABLE_CONTEXT_REFERENCES=true
BRIDGE_CONTEXT_REFERENCES_FETCH_URLS=true
```

Supported forms:

```text
@file:langgraph-app/agent_graph.py:10-40
@folder:docs
@diff
@staged
@git:3
@url:https://example.com/page
```

Files are resolved under `BRIDGE_CONTEXT_REFERENCE_WORKSPACE_ROOT` or the
AI-stack repo root by default. Sensitive credential/config paths such as `.env`,
`.ssh`, `.aws`, `.kube`, and `.docker` are refused. Large references are bounded
by:

```text
BRIDGE_CONTEXT_REFERENCE_SOFT_RATIO=0.25
BRIDGE_CONTEXT_REFERENCE_HARD_RATIO=0.50
BRIDGE_CONTEXT_REFERENCE_CONTEXT_LENGTH=128000
```

Reference warnings and injected-token estimates are recorded in the LangGraph
`run_profile`.

Hard request cutoffs:

```text
ALPHARAVIS_HARD_CONTEXT_TOKEN_LIMIT=128000
BRIDGE_HARD_INPUT_TOKEN_LIMIT=128000
```

The bridge checks before sending a request to LangGraph. The graph checks again
before invoking any model.

## Memory And Compression

Active chat compression happens automatically after the current LangGraph run
finishes when the thread grows above `ALPHARAVIS_ACTIVE_TOKEN_LIMIT`.

When compression happens, AlphaRavis can show a visible `Memory-Notice`.
The current task brief and latest handoff packet are preserved verbatim when
available, so the next run still knows the plan, completed work, open tasks,
and verification state.

Before the swarm starts, AlphaRavis also has a handoff-context guard. If the
context is already too large, it compresses the beginning of the current run
into a handoff summary and archives the exact original messages, while keeping
the task brief, memory/skill hints, latest handoff packet, and recent messages
active.

Useful handoff settings:

```text
ALPHARAVIS_ENABLE_HANDOFF_CONTEXT_GUARD=true
ALPHARAVIS_HANDOFF_CONTEXT_TOKEN_LIMIT=8500
ALPHARAVIS_HANDOFF_CONTEXT_KEEP_LAST_MESSAGES=16
ALPHARAVIS_HANDOFF_PACKET_MAX_CHARS=4000
ALPHARAVIS_HANDOFF_SUMMARY_MAX_CHARS=2600
```

Agents are instructed to call `build_specialist_report` before `transfer_to_*`.
That JSON report is the handoff packet.

To pause compression for one run, say one of:

```text
keine Kompression
nicht komprimieren
skip compression
no compression
```

To force compression for one run, say one of:

```text
komprimiere jetzt
archiviere jetzt
compress now
```

The force phrases can be replaced with a pipe-separated ENV value:

```text
ALPHARAVIS_MANUAL_COMPRESSION_PATTERNS=komprimiere jetzt|archive now
```

Archive search is thread-scoped by default. Other chat archives are searched
only when you explicitly ask for cross-thread archive search.

## MemoryKernel

The MemoryKernel is the small learning layer inspired by Hermes.

It runs only on the normal agent path. It does not run on Fast Path.

What it does:

- loads tiny curated memories when they match the current turn,
- reminds agents every few turns to save useful durable facts,
- indexes finished turns for later search,
- helps compression preserve memory-worthy details.

Useful settings:

```text
ALPHARAVIS_ENABLE_MEMORY_KERNEL=true
ALPHARAVIS_MEMORY_NUDGE_INTERVAL=10
ALPHARAVIS_ALWAYS_MEMORY_MAX_ITEMS=6
ALPHARAVIS_ALWAYS_MEMORY_MAX_CHARS=2200
```

Curated memory is for compact stable facts, not full chat history. Good
examples:

```text
User prefers concise German explanations.
The big llama.cpp server is the preferred backend for complex work.
Pixelle failures should first check job status before SSH debugging.
```

Long logs, reports, or implementation notes should go to artifacts instead.

## Semantic Vector Memory

pgvector memory is the semantic Inhaltsverzeichnis for AlphaRavis. MongoDB and
LangGraph still own checkpoints, store data, archives, and thread state, but
pgvector stores a catalog plus full retrieval chunks generated from the
complete original source data.

Default:

```text
ALPHARAVIS_VECTOR_BACKEND=pgvector
ALPHARAVIS_ENABLE_PGVECTOR_MEMORY=true
ALPHARAVIS_PGVECTOR_CATALOG_ENABLED=true
ALPHARAVIS_PGVECTOR_STORE_FULL_CHUNKS=true
```

Requirements:

```text
ALPHARAVIS_PGVECTOR_DATABASE_URL=postgresql://postgres:<password>@vectordb:5432/rag_api
ALPHARAVIS_PGVECTOR_EMBEDDING_BASE_URL=http://litellm:4000/v1
ALPHARAVIS_PGVECTOR_EMBEDDING_MODEL=memory-embed
ALPHARAVIS_PGVECTOR_FALLBACK_EMBEDDING_MODEL=memory-embed-fallback
```

`memory-embed` is a LiteLLM route. The default example routes to an
OpenAI-compatible Ollama embedding model such as
`Q78KG/gte-Qwen2-1.5B-instruct`. The fallback route points to `bge-m3`.

Agents can call:

```text
semantic_memory_search
```

It searches the current thread plus global memories by default and also queries
the existing RAG API for external documents. It searches other AlphaRavis
threads only when `include_other_threads=true` is explicitly requested.
After enabling pgvector memory, new records are indexed automatically. Old
MongoDB/store history is not bulk-backfilled by default, to avoid a surprise
embedding job over many chats.

New records go into a durable embedding queue by default:

```text
ALPHARAVIS_PGVECTOR_INDEX_MODE=queue
ALPHARAVIS_PGVECTOR_QUEUE_TABLE=alpharavis_embedding_jobs
ALPHARAVIS_EMBEDDING_JOB_BATCH_SIZE=10
```

The Power Management Agent can drain that queue with `run_embedding_memory_jobs`.
It is allowed when the big llama.cpp server is active or the system has been
idle long enough, depending on `ALPHARAVIS_EMBEDDING_LOAD_POLICY`.

To let LangGraph drain the queue automatically:

```text
ALPHARAVIS_ENABLE_EMBEDDING_SCHEDULER=true
ALPHARAVIS_EMBEDDING_SCHEDULER_INTERVAL_SECONDS=120
```

The lifecycle runner pauses if the small Ollama chat/crisis model is already
loaded and `ALPHARAVIS_EMBEDDING_UNLOAD_CHAT_MODEL=false`. This avoids stealing
the management node from crisis work. If you want the runner to unload the small
chat model for embedding windows, set:

```text
ALPHARAVIS_EMBEDDING_UNLOAD_CHAT_MODEL=true
```

Manual bounded backfill is available through:

```text
queue vector memory backfill
```

The optional daemon is default off and requires a query:

```text
ALPHARAVIS_ENABLE_VECTOR_BACKFILL_DAEMON=true
ALPHARAVIS_VECTOR_BACKFILL_QUERY=project-name-or-topic
ALPHARAVIS_VECTOR_BACKFILL_LIMIT_PER_SOURCE=10
```

That daemon searches existing Store indexes and queues matching records. It is
not an automatic full-history backfill.

## Session Search And Artifacts

AlphaRavis now keeps an indexed per-turn history, similar in spirit to Hermes
session search.

Agents can search it with:

```text
search_session_history
```

Current-thread search is default. Other threads are searched only when the user
explicitly asks for it.

Artifacts are disk-backed files for large outputs:

```text
write_alpha_ravis_artifact
read_alpha_ravis_artifact
list_alpha_ravis_artifacts
```

Default artifact root:

```text
/workspace/artifacts/alpharavis
```

Use artifacts when a result is too large for chat but should still be
recoverable later.

## Agent-Specific Memories

Agents also have scoped durable memories:

```text
alpharavis / agent_memories / general_assistant
alpharavis / agent_memories / research_expert
alpharavis / agent_memories / debugger_agent
alpharavis / agent_memories / context_retrieval_agent
alpharavis / agent_memories / global
```

The intended rule is:

- search the active agent's own memory first,
- include global memories for stable cross-agent preferences,
- record new memories only after a useful lesson or repeated preference is clear,
- keep thread archives separate from reusable agent memories.

The agent knows which memory to use from its role prompt. For example, the
debugger uses `agent_id=debugger_agent`, while cross-agent lessons use
`scope=global`.

## Skill Library

The skill library stores reusable workflow patterns.

Safety rules:

- New workflows become inactive candidates.
- Candidates do not affect routing.
- Active skills are hints, not automatic execution.
- Promotion is disabled unless `ALPHARAVIS_ALLOW_SKILL_PROMOTION=true`.

Useful review commands:

```text
zeige Skill-Kandidaten
aktiviere Skill <key>
deaktiviere Skill <key>
```

Activation/deactivation is blocked unless review mode is enabled:

```text
ALPHARAVIS_ALLOW_SKILL_PROMOTION=true
```

Reviewed repo skill cards under `ai-skills/` are different from Mongo skill
candidates. AlphaRavis may inject only a tiny metadata hint when a card seems
relevant. It reads the full card only through `read_repo_ai_skill` when needed.

```text
ALPHARAVIS_REPO_SKILL_HINT_LIMIT=3
```

## Run Profile

Every run stores timing and routing data in LangGraph state as `run_profile`.

Typical fields:

- route: `fast_path` or `swarm`
- route_reason
- message_count
- token_estimate
- total_seconds
- fast_path_seconds
- fast_path_fallback_used

Set `ALPHARAVIS_SHOW_RUN_PROFILE=true` if you want this appended visibly in
LibreChat. Otherwise inspect it in LangGraph Studio or DeepAgents UI.

## Current Optimization Notes

Already available:

- OpenAI-compatible LibreChat bridge
- LangGraph native brain
- fast path for simple chat
- run profile state
- skill-library candidate listing and review-mode activation/deactivation
- reviewed repo skill-card hints and on-demand skill-card reading
- DeepAgents-style MCP config loading, disabled by default for faster simple chat
- optional Hermes direct endpoint for LibreChat and Hermes coding sub-agent for AlphaRavis
- fast-path hidden-thinking disable for llama.cpp/Qwen-style models
- visible fast-path notices and thread lockout after agent path
- graph-level and bridge-level hard context cutoffs
- agent-specific and global memory tools
- MemoryKernel with curated always-memory, turn indexing, and compression hints
- session-history search over indexed turns
- disk-backed AlphaRavis artifacts for large reports/logs/plans
- thread-scoped memory archives
- manual one-run chat compression
- structured specialist reports for research/debug/context handoffs
- async Pixelle start/status tools for long image jobs
- visible memory notices
- command approval gate
- LLM generation health endpoint
- owner-only power tools for llama.cpp and ComfyUI, default off
- protected owner shutdown tools behind human approval
- token-light crisis preflight/recovery agent, default off
- OpenAPI 3.1 bridge schema and richer Responses streaming event names
- Responses-native direct LangGraph calls for planner/fast-path/summarizers
- Responses-native DeepAgents model binding through LangChain `ChatOpenAI`,
  feature-flagged with ChatLiteLLM fallback
- `make model-management` / `make owner-model-management` for custom hardware setup
- durable pgvector embedding queue, scheduler, manual queue runner, and bounded
  backfill queueing
- Pixelle owner wake guard for ComfyUI, default off through model management
- safe media handling in the Bridge: URL/file-id/type metadata is passed instead
  of raw images/videos unless explicitly enabled
- media-gallery service for Pixelle outputs and uploaded/linked media metadata
- separate optional media/vision pgvector table to avoid text/vision dimension
  conflicts
- lazy tool category registry for coding, media, RAG, and system tool families
- OpenWebUI optional frontend profile using the AlphaRavis Bridge
- Hermes healthcheck/fallback before bounded coding-agent calls

Still open / planned next:

- mid-run backend watchdog and crisis recovery for timeouts/502s after a graph
  run already started
- post-crisis readiness gate before continuing to the normal planner
- live smoke test against your llama.cpp Responses endpoint for full DeepAgents
  tool calls
- agent time/tool/handoff budget guard
- richer activity stream in LibreChat
- test whether LibreChat shows `reasoning_content` in a separate reasoning
  panel before enabling reasoning streaming by default
- optional parallel agent execution with dependency groups
- full video analysis pipeline: keyframes, timecodes, captions, transcription,
  and frame-level vision embeddings
- true internal dynamic tool binding/unbinding per run; current implementation
  exposes category manifests and keeps concrete LangGraph tools available

## Media And Uploads

By default the Bridge does not forward raw media blocks into LangGraph:

```text
BRIDGE_ALLOW_RAW_MEDIA_CONTEXT=false
BRIDGE_MEDIA_CONTEXT_MODE=metadata
```

That means uploads/links arrive as metadata markers containing fields such as
type, file id, URL, mime type, or title. This prevents a video or image blob from
filling the LLM context. Use `register_media_asset` to save a URL/file id in the
media gallery. Use `semantic_media_search` only after
`ALPHARAVIS_ENABLE_VISION_VECTOR_MEMORY=true` and a compatible vision embedding
route exists.

Pixelle output URLs are registered automatically when the job result contains a
media URL. The gallery runs at:

```text
http://localhost:8130/gallery
```

Video analysis remains planned, not automatic. The agent should say this
clearly and use `plan_media_analysis` when the user asks what would happen.

## OpenWebUI

OpenWebUI is optional:

```text
docker compose --profile openwebui up -d openwebui
```

It uses the AlphaRavis Bridge as the OpenAI-compatible provider. In OpenWebUI,
set Function Calling to `Native` for capable models. Keep web search disabled
until SearXNG or another search backend is configured. Passthrough is useful for
Responses/custom endpoints, but it forwards upstream requests with the configured
OpenWebUI provider key, so keep it owner-only or disable it on shared instances.
