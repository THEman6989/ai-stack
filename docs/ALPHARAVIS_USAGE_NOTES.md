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

## Memory And Compression

Active chat compression happens automatically when the LangGraph thread grows
above `ALPHARAVIS_ACTIVE_TOKEN_LIMIT`.

When compression happens, AlphaRavis can show a visible `Memory-Notice`.

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

Optional pgvector memory is a search index, not a second source of truth.
MongoDB/LangGraph still own checkpoints, store data, archives, and thread state.
pgvector stores only compact search cards with a preview, metadata, source key,
thread id, and embedding.

Default:

```text
ALPHARAVIS_VECTOR_BACKEND=off
ALPHARAVIS_ENABLE_PGVECTOR_MEMORY=false
```

Enable:

```text
ALPHARAVIS_VECTOR_BACKEND=pgvector
```

Requirements:

```text
ALPHARAVIS_PGVECTOR_DATABASE_URL=postgresql://postgres:<password>@vectordb:5432/rag_api
ALPHARAVIS_PGVECTOR_EMBEDDING_BASE_URL=http://litellm:4000/v1
ALPHARAVIS_PGVECTOR_EMBEDDING_MODEL=memory-embed
```

`memory-embed` is a LiteLLM route. The default example routes to an
OpenAI-compatible Ollama embedding model such as `nomic-embed-text`. Pull or
configure that model before enabling vector memory.

Agents can call:

```text
semantic_memory_search
```

It searches the current thread plus global memories by default. It searches
other threads only when `include_other_threads=true` is explicitly requested.
After enabling pgvector memory, new records are indexed automatically. Old
MongoDB/store history is not bulk-backfilled by default, to avoid a surprise
embedding job over many chats.

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

Planned next:

- backend watchdog as a safe debugger tool
- agent time/tool/handoff budget guard
- richer activity stream in LibreChat
