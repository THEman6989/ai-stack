# AlphaRavis Architecture And Capabilities

This file is the editable on-demand self-description for AlphaRavis.
It is intentionally not injected into every chat. Agents should read it only
when the user asks what AlphaRavis is, how it works, what it can do, or how the
stack is wired.

## Identity

AlphaRavis is a multi-agent AI system built around a native LangGraph "brain",
an OpenAI-compatible FastAPI bridge for LibreChat, and a separate ACP adapter
for AionUi/custom-agent clients.

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
- `alpharavis_acp_adapter.py`: optional AionUi bridge. Runs as a JSON-RPC-over-stdio
  custom ACP agent and maps LangGraph streams to AionUi text chunks, toolcards,
  plan updates, and permission requests.
- `litellm`: model gateway. Routes AlphaRavis model calls to configured
  backends such as llama.cpp or Ollama. Its proxy metadata lives in the
  `litellm` Postgres database.
- External `ubuntu-llama-manager`: optional owner-run API for a Ubuntu
  llama.cpp host. LangGraph can inspect its `/health`, `/status`, `/models`,
  and `/llama/instances` endpoints and, when model-management actions are
  enabled, request no-response recovery, control managed llama services, run
  ESP/server power actions, or patch a `primary`/`secondary` llama.cpp instance
  model/context through `/llama/instances/{id}/config`. Optional direct ESP
  mode can call the ESP `/action` or `/cancel` endpoints when the Ubuntu host
  itself is off. This service is the control plane only; AlphaRavis does not use
  it as a tokenizer proxy.
- Direct llama.cpp runtime plane: for each Manager-discovered instance,
  AlphaRavis can call the selected llama-server directly for `/apply-template`,
  `/tokenize`, `/slots`, `/v1/models`, `/completion`, and
  `/v1/chat/completions`. The context scheduler uses that runtime plane for
  hard token counts and context leases (process-local by default, optional
  Redis-backed for multi-worker setups) before async LLM calls.
  `--kv-unified` enables dynamic shared-pool planning; without it, planning is
  conservative per slot.
- Background lane: small independent read-only jobs can overlap with the main
  graph work. Small LLM side jobs, such as Router, Judge, Summarizer, RAG
  compression, and chunk ranking, still go through direct llama-server token
  counting and ContextLease admission, with a lower background utilization cap.
  Dangerous write, restart, recovery, and power actions remain outside this
  lane.
- Percentage-based context budget router (`ai_stack/context_budget/router.py`):
  `DynamicServerState` probes live llama-server `/slots`, `/props`, `/metrics`.
  `PercentageBudgetPolicy` computes safety reserves and output budgets as
  percentages of the detected context pool (no fixed token constants).
  `PriorityAwareRouter` classifies requests into 7 priority levels and routes
  based on dynamic free context, task priority, and configurable thresholds.
  Primary agents (critical_main_agent, coding_agent) get full usable context;
  secondary agents use percentage caps. Budget notice injection tells the model
  about output limits in natural language.
- Parallel task executor (`ai_stack/parallel_executor/`): Parses planner output
  into a structured `TaskDAG` with classification, file conflict detection,
  chokepoint detection, and parallelization analysis. When the feature flag is
  enabled, `planner_node` asks BigBoss to append a machine-readable
  `<parallel-execution-plan>{...}</parallel-execution-plan>` JSON block with
  `parallel_possible`, per-task `parallel` hints, groups, dependencies, files,
  model class, risk, and rationale. The parser prefers that block when present
  and falls back to legacy bullet parsing otherwise. BigBoss hints are advisory:
  `parallel=false` forces serial execution, safe planner groups are preserved,
  and deterministic safety checks still override unsafe parallel hints.
  Independent tasks are grouped for real concurrent `asyncio.gather()` execution;
  overlapping write globs, chokepoint files, dependencies, merge/review, and
  constrained big-model/context resources remain serial. Git worktree isolation
  (`worktree_manager.py`) is adapted from Hermes CLI patterns. Abstract worker
  spawner interface (`worker_spawner.py`) with `DryRunWorker` mock and
  `DirectLLMWorker` for real LLM calls. `executor.py` runs parallel groups
  concurrently via `asyncio.gather()`, then serial chain, then merge/review.
  `file_lock.py` provides process-local file/glob locking for concurrent write
  safety, including concrete-path vs wildcard-glob conflicts.
  Feature-flagged via `ALPHARAVIS_PARALLEL_TASK_EXECUTION=false` (default OFF).
  When disabled, the `parallel_executor` graph node returns `{}` (no-op), the
  planner prompt is unchanged, and the existing sequential swarm path is
  unchanged. When enabled, workers run concurrently before the swarm, results are
  collected and merged.

  **HermesWorker** (`worker_spawner.py`): Wraps the existing `call_hermes_agent`
  path as a controlled WorkerSpawner. Hermes gets a bounded task with context,
  allowed tools, and a context budget — it cannot call AlphaRavis/LangGraph back
  or spawn further workers autonomously. Feature-flagged via
  `ALPHARAVIS_PARALLEL_HERMES_WORKER=false`.

  **ParallelContextPlanner** (`context_planner.py`): Conservative admission
  control for KV-unified context pools (e.g. BigBoss np=4, 320k). `SlotBudget`
  tracks pool usage with an 8% global safety reserve, supporting asymmetric
  distribution (Worker A 120k, Worker B 30k). `ContextPreEstimator` tokenizes
  worker material (RAG, files, tools) via llama.cpp `/tokenize` API — material
  is NOT loaded into the estimator's own context. Workers are refused admission
  when the pool is full. Feature-flagged via
  `ALPHARAVIS_PARALLEL_CONTEXT_PLANNER=false`.
- `Server Model Manager`: a dedicated LangGraph/Bridge access mode for
  `power_management_agent`. LibreChat sees it as the `server-model-manager`
  model/preset on the existing AlphaRavis Bridge, while native LangGraph
  callers can pass `active_agent="power_management_agent"` and
  `selected_toolsets=["agent/power"]`. Its default LangGraph model is
  `openai/server-model-manager`, a LiteLLM route intended to prefer BigBoss and
  fall back to Edge Gemma. The agent carries a comprehensive set of power and
  model management tools controlling two separate hardware paths:

  **Llama Server (ESP + Ubuntu Llama Manager API):**
  `request_ubuntu_server_power_action` (power-on/off/cycle, reset, shutdown,
  reboot via Manager or direct ESP, gated by `confirmed=true`),
  `control_ubuntu_llama_service`, `configure_ubuntu_llama_instance`.

  **ComfyUI Server (SSH + Wake-on-LAN):**
  `owner_check_comfyui_server`, `owner_start_comfyui_server` (WoL),
  `owner_shutdown_comfyui_server` (SSH, HITL-gated),
  `wake_on_lan`, `prepare_comfy_for_pixelle`.

  **Monitoring, Recovery, Embedding:**
  `inspect_ubuntu_llama_manager`, `diagnose_ubuntu_llama_no_response`,
  `recover_ubuntu_llama_no_response`, `check_ollama_models`,
  `load_embedding_model`, `run_embedding_jobs`, `queue_*_vector_backfill`.

  **Owner SSH Tools** (gated by `ALPHARAVIS_ENABLE_OWNER_POWER_TOOLS=true`):
  `owner_start_llama_server`, `owner_restart_llama_server`,
  `owner_shutdown_llama_server` (prefer Ubuntu Manager API for llama).
- `mongodb`: LangGraph checkpointing and long-term store backing.
- `vectordb`: Postgres with pgvector. It can act as an optional semantic
  search sidecar for AlphaRavis memory; it does not replace MongoDB.
- `redis`: optional LLM cache and shared context-lease store for multi-worker
  deployments. When `ALPHARAVIS_CONTEXT_LEASE_BACKEND=redis`, the
  `ContextScheduler` uses Redis for atomic cross-worker context admission
  (Lua-script-backed `HSET` + capacity check). Default `local` uses a
  process-local dict — sufficient for single-worker setups.
- `deep-agents-ui`: AlphaRavis custom agent UI (forked from langchain-ai/deep-agents-ui,
  in `submodules/`). Features: chat, threads, tasks, tool approval, subagent
  indicators, multimodal file upload (picker, drag/drop, paste), chat openers,
  hardened thread rename/delete, artifact system, file preview panel, lightweight
  diff viewer, on-demand Monaco editor for explicit code preview, Office tab for
  OfficeCLI document creation/editing, ComfyUI tab for remote ComfyPC status/
  model/queue handling, and skills indicator. Connects directly to LangGraph API
  on port 2024 and to `media-gallery` for lightweight Office/ComfyUI browser
  endpoints. Future UI ports should follow `docs/ALPHARAVIS_UI_INTEGRATION_TEMPLATE.md`.
- `service-dashboard`: lightweight local redirector UI on port `8090` that
  lists host and Docker URLs for the stack services, separating Web Interfaces,
  APIs, and Infrastructure. API cards expose copyable Tailscale HTTPS, local
  HTTP, and Tailnet HTTP addresses when the Tailscale override payload is
  present. Cards are directly clickable. LiteLLM and Pixelle are represented as
  separate Web UI and API/MCP endpoint cards where appropriate. Experimental
  LangGraph specialist visual ports are infrastructure/TCP entries, not normal
  click-to-open web cards. It also serves `/settings`, a mobile/PWA-friendly
  settings UI generated from `.env(exaple)` that can write temporary runtime
  overrides or persist validated keys to `.env`.
- `librechat`: the normal chat UI for the user.
- `rag_api`: local document search backend when available. Its LangChain
  PGVector tables live in the separate `rag_api` Postgres database.
- Pixelle/MCP services: image generation and Pixelle tool integration when available.
- `hermes-agent`: optional external coding/system agent reached through its
  OpenAI-compatible API on the host.

`tailscale_https_routes.py` is a prepared operator helper, not a running
container. It imports the service-dashboard catalog, derives local HTTP targets
such as `http://127.0.0.1:3080`, and can configure Tailscale Serve HTTPS routes
on matching Tailnet-reachable ports. It does not use Tailscale Funnel and does
not expose services to the public internet. The redirector can prefer the
generated `service-dashboard-data/tailscale_service_urls.json` URLs when
present. The dashboard route itself is included by default on port `8090` for
the Makefile Tailscale targets; operators can opt out with
`TAILSCALE_DASHBOARD=false` or `--exclude-dashboard`. Normal install/update/up
Makefile flows call `tailscale-auto`, which defaults to `tailscale-apply`.
Before Docker starts, those flows set `ALPHARAVIS_DOCKER_HOST_BIND=127.0.0.1`
so Docker-published application ports do not conflict with Tailscale Serve on
the Tailnet IP. Set `TAILSCALE_AUTO=off` to switch the stack to LAN HTTP mode:
the Makefile disables managed Tailscale Serve routes, removes dashboard HTTPS
overrides, writes `ALPHARAVIS_DOCKER_HOST_BIND=0.0.0.0`, and then the normal
Docker start/recreate step publishes application ports on all host interfaces.
Use `TAILSCALE_AUTO=keep` when a run should leave the current network exposure
mode untouched. The helper's sudo mode defaults to `auto`, so it requests sudo
only after the non-sudo Tailscale CLI attempt fails with a permissions error.
For visible URLs that include a path, `tailscale_https_routes.py` keeps that
path in the public dashboard link but uses the service root as the default
Tailscale Serve upstream target. This avoids mounting a whole UI behind an
upstream `/gallery` or `/v1` path and breaking sub-links.

## Install And Runtime Profiles

The Makefile is the supported operator entrypoint for local setup. It delegates
stateful install/configuration work to `scripts/alpharavis_setup.py` so the
interactive wizard, one-shot targets, and direct script calls all update `.env`
through the same code path.

Common flows:

```bash
make install
make update
make config
make install-fullstreaming
make install-chat-fullstreaming
make profiles
make streaming STREAMING=full
make up-fullstreaming
make up-chat-fullstreaming
make status
```

`make config` is the central human-editable configuration surface. It starts a
local dependency-free web UI, opens it in the browser, groups settings using the
sections in `.env(exaple)`, pre-fills current `.env` values, exposes boolean
values as True/False controls, and saves through the same root `.env` file that
Docker Compose and setup commands read. Per-field reset restores the documented
default for that key; reset-all asks for confirmation before restoring every
shown setting to `.env(exaple)`.

For day-to-day runtime control after the stack is up, the Service Dashboard
exposes `/settings`. The Settings UI parses every key from `.env(exaple)`,
annotates it with current `.env` values and any temporary runtime override, and
offers search, category chips, importance filters, changed/runtime filters, and
mobile-first controls, including a sort mode that preserves `.env(exaple)`
insertion order. The UI uses compact setting rows, local browser
favorites, generated fallback descriptions for undocumented keys, and inferred
dropdowns for common mode/profile settings. `Temporary anwenden` writes
`service-dashboard-data/runtime_settings.json`; LangGraph reads that file before
each new run and applies the values into `os.environ`, so new chat turns can
pick up runtime changes without rewriting `.env`. `Permanent speichern` writes
validated keys back to `.env` after browser confirmation. The dashboard card and
Tailscale helper both treat `/settings` as a normal dashboard Web Interface.

`make install` now does more than copy `.env`: it syncs missing defaults from
`.env(exaple)`, lets the operator choose a runtime API/streaming profile,
optionally configures the existing model-management/media/OpenWebUI sections,
stores Docker Compose profiles such as `openwebui`, initializes submodules, and
can build/start the stack. `make update` uses the same menu, then updates
submodules and builds/starts the stack by default.

The streaming profiles write these architectural modes into `.env`:

| Profile | Core effect |
| --- | --- |
| `responses-hybrid` | Responses API, `streaming=true`, `disable_streaming=tool_calling`, experimental tool-stream patch disabled. This is the stable default. |
| `responses-full` | Responses API, `streaming=true`, `disable_streaming=false`, `ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING=true`. This enables experimental tool-bound full streaming. |
| `responses-nonstreaming` | Responses API with internal DeepAgents/ChatLiteLLM streaming disabled. |
| `chat-full` | Direct calls and DeepAgents workers use Chat Completions through ChatLiteLLM with `ALPHARAVIS_LLM_STREAMING=true`. |
| `chat-nonstreaming` | Direct calls and DeepAgents workers use Chat Completions through ChatLiteLLM with `ALPHARAVIS_LLM_STREAMING=false`. |

Docker Compose profiles are stored in `.env` as `COMPOSE_PROFILES`. For example,
`COMPOSE_PROFILES=openwebui` makes normal `docker compose up` / `make up` include
the optional OpenWebUI service.

## AionUi ACP Adapter

AionUi can use AlphaRavis through a separate custom ACP agent:

```text
python /workspace/langgraph-app/alpharavis_acp_adapter.py
```

This adapter is not the OpenAI bridge. It does not expose `/v1/chat/completions`
or `/v1/responses`; it speaks ACP-style JSON-RPC over stdio and calls the native
LangGraph API on `LANGGRAPH_API_URL`.

Supported ACP flow:

```text
AionUi -> alpharavis_acp_adapter.py -> langgraph-api -> alpha_ravis
```

The adapter maps LangGraph events to AionUi-native UI updates:

- message deltas -> `agent_message_chunk`
- node/status summaries -> `agent_thought_chunk`
- planner state -> `plan`
- tool calls/results -> `tool_call` / `tool_call_update`
- command approval interrupts -> `session/request_permission`

It strips internal AlphaRavis context blocks, redacts common secrets, truncates
tool outputs, and keeps tool logs out of thought/reasoning updates. Detailed
setup is in `docs/AIONUI_LANGGRAPH_ACP_INTEGRATION.md`.

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

Docker builds do not modify the upstream `hermes-agent` submodule directly.
Instead, the Hermes containers apply parent-repo patches from
`patches/hermes-agent/` to `/opt/hermes` at startup via
`scripts/hermes_patched_entrypoint.sh`, which calls
`scripts/apply_hermes_agent_patches.sh` before delegating to the original Hermes
entrypoint. If Hermes startup or kanban migration behavior differs from
upstream, check `docs/ALPHARAVIS_CHANGES.md`, the Compose Hermes entrypoint, and
those patch files first.

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

## ComfyUI / ComfyPC Integration

ComfyUI is integrated in two layers so LAN control stays centralized and the
browser does not need to talk directly to the remote ComfyPC:

```text
DeepAgents UI ComfyUI tab -> media-gallery /comfyui/* -> ComfyUI REST API
AlphaRavis LangGraph -> comfyui_agent / comfyui tools -> ComfyUI REST API
Pixelle flows -> prepare_comfy_for_pixelle / ComfyPC power tools -> Pixelle
```

`langgraph-app/comfyui_client.py` resolves the ComfyUI base URL in this order:

```text
ALPHARAVIS_COMFYUI_API_BASE
ALPHARAVIS_COMFY_HEALTH_URL
REMOTE_PCS[ALPHARAVIS_COMFY_PC] + ALPHARAVIS_COMFYUI_PORT
http://127.0.0.1:8188
```

The narrow `comfyui/workflows` toolset exposes status, queue, model listing,
history lookup, and gated workflow submission. The dedicated `comfyui_agent`
peer is only registered when `ALPHARAVIS_ENABLE_COMFYUI_AGENT=true`; other swarm
workers then get `transfer_to_comfyui` handoff tools. Workflow submission remains
separately gated by `ALPHARAVIS_ENABLE_COMFYUI_WORKFLOW_SUBMIT=false` by default
because arbitrary ComfyUI workflows/custom nodes have Python-code trust level.

`media-gallery` exposes `/comfyui/status`, `/comfyui/queue`, and
`/comfyui/models/{folder}` as lightweight browser-safe proxy endpoints. The
ComfyUI tab uses those endpoints for health/queue/model visibility and uses
agent-launcher prompts for substantial generate/inspect/fix workflow tasks.

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
    },
    "officecli": {
      "type": "stdio",
      "command": "officecli",
      "args": ["mcp", "start"],
      "enabled_env": "ALPHARAVIS_ENABLE_OFFICECLI_MCP"
    }
  }
}
```

MCP tools remain lazy by default:

```text
ALPHARAVIS_LOAD_MCP_TOOLS=false
```

When enabled, AlphaRavis loads configured MCP servers through the robust
``mcp_client`` module (``langgraph-app/mcp_client.py``), which wraps
``langchain_mcp_adapters`` with Hermes-style resilience:

- **Reconnect** with exponential backoff on connection loss.
- **Circuit breaker** (3-state) that short-circuits after 3 consecutive
  failures for 60s to prevent iteration-burn retry loops.
- **Per-server timeouts** configurable via ``timeout`` and ``connect_timeout``
  in ``mcp.json``.
- **Error classification** into auth, transient, and permanent errors with
  appropriate response messages.

The loader prefixes tool names by server when supported, records server/tool
metadata for ``describe_optional_tool_registry``, supports per-server
`enabled_env` gates for config entries that should exist but remain disabled by
default, and keeps stdio MCP servers disabled unless explicitly trusted:

```text
ALPHARAVIS_MCP_ALLOW_STDIO=false
```

This keeps the useful DeepAgents MCP pattern without letting arbitrary project
MCP configs start local processes by accident.

AlphaRavis also has a Hermes-style toolset layer in
`langgraph-app/alpharavis_toolsets.py`. Toolsets are composable categories such
such as `coding/read`, `media/image`, `rag/memory`, `office/documents`,
`system/docker`, and `system/power`. The run start node infers likely toolsets
request and stores them in `run_profile.selected_toolsets`; the planner injects
only a short category context on the agent path. MCP schemas are cached by
category, and concrete MCP tools are bound only to matching specialist bundles
instead of attaching every loaded MCP tool to the generalist. Specialist
workers now also bind their local tools from those materialized bundles at graph
build time, with handoff tools added explicitly. The loaded per-agent bundle
profiles are recorded in `run_profile.loaded_toolsets`.

Office is now represented twice by design: `office/documents` remains the narrow
OfficeCLI/MCP tool category, while `agent/office` is the dedicated swarm-agent
bundle for multi-step document workflows. When `ALPHARAVIS_ENABLE_OFFICE_AGENT`
is enabled, `_build_graph()` adds `office_agent` as a peer worker in the existing
swarm and exposes `transfer_to_office` from the other specialists. The Deep
Agents UI Office tab can submit `active_agent=office_agent`, so opening the Office
workflow path starts directly on the Office specialist instead of relying on the
generalist to infer and hand off. Lightweight list/upload/status/placeholder
calls still go straight to `media-gallery`; create/edit/template/batch/repair and
preview workflows go through `office_agent`.

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
- It exposes `/v1/responses` over the same LangGraph run path, with
  Responses-style output items, semantic SSE lifecycle events, and bridge-local
  `GET`/`DELETE /v1/responses/{response_id}` retrieval support.
- It also exposes bridge-local Responses compatibility routes for
  `/v1/responses/{response_id}/input_items`, `/v1/responses/{response_id}/cancel`,
  and `/v1/responses/input_tokens`. `previous_response_id` works when the
  referenced response is still in the local bridge cache.
- It returns explicit unsupported errors for OpenAI-hosted features that are not
  genuinely implemented by AlphaRavis, including background Responses,
  Conversations, hosted client-supplied tools, structured output formats,
  non-text output modalities, prompt-template references, and encrypted
  `/v1/responses/compact`.
- It publishes an OpenAPI `3.1.0` schema.
- It supports non-streaming and OpenAI-compatible SSE streaming. Chat
  Completions streams data-only `chat.completion.chunk` events; Responses
  streams typed semantic events such as `response.output_text.delta`.
- It can stream LangGraph message events.
- It can optionally forward reasoning/thinking deltas as a separate SSE delta
  field when `BRIDGE_STREAM_REASONING_EVENTS=true`. Normal visible content
  still strips reasoning blocks.
- Responses streaming has a separate default-on reasoning switch,
  `BRIDGE_RESPONSES_STREAM_REASONING_EVENTS=true`, so LibreChat's Responses
  reasoning pane can receive explicit provider reasoning and visible local-model
  thinking without enabling the legacy Chat Completions reasoning field.
- LangGraph `planner` updates are translated to reasoning deltas with
  `alpha_reasoning_kind=internal_plan`; the Bridge Test UI renders them in a
  dedicated Planer pane, while LibreChat receives them in its single reasoning
  channel.
- It can optionally emit short visible status/activity messages.
- It handles LangGraph human approval interrupts and lets the user reply with:
  - `approve`
  - `reject`
  - `replace: <safer command>`
  - `approve always` / `immer erlauben`
- `approve always` stores a bridge-local allow entry for the exact
  scope/target/command in the current LibreChat thread only. It does not create
  a global command bypass and is lost on `api-bridge` restart.
- LibreChat's custom endpoint path does not provide an AlphaRavis-native
  approval button callback. AionUI/ACP maps the same interrupt to
  `session/request_permission`; LibreChat uses chat-text approval unless a
  LibreChat-specific permission event is added later.

The detailed Responses compatibility matrix is documented in
`docs/ALPHARAVIS_RESPONSES_COMPATIBILITY.md`.

The bridge uses `BRIDGE_MESSAGE_SYNC_MODE=delta` by default. This means that
after an existing LangGraph thread has state, the bridge sends only new user
messages into LangGraph instead of re-sending the whole LibreChat history every
turn. This keeps LangGraph compression useful and avoids old messages being
reintroduced forever.

The bridge also owns first-pass context hygiene:

- `BRIDGE_SCRUB_INTERNAL_CONTEXT=true` removes internal blocks such as
  `<memory-context>...</memory-context>` from visible OpenAI/LibreChat output,
  including streamed chunks where the tag can be split across deltas.
- `BRIDGE_ENABLE_CONTEXT_REFERENCES=true` lets explicit user references such as
  `@file:...`, `@folder:...`, `@diff`, `@staged`, `@git:3`, and `@url:...` be
  expanded into bounded context blocks before LangGraph planning.
- Context references resolve under the AI-stack repo root by default and refuse
  sensitive credential/config paths. Warnings and injected-token estimates are
  copied into `run_profile` as `bridge_context_references`.
- Large file, git, and URL context references use Hermes-style head/tail
  truncation. The middle is replaced with a marker that points the agent back to
  exact file/archive tools when the full source is needed.

Stable prompt material is separated from ephemeral run context. A tiny
`<stable-runtime-context>` block carries platform hints, archive policy, and
toolset policy; the current task brief, planner context, MemoryKernel hints,
skill hints, and handoff packet remain separate protected messages.

## File Safety

AlphaRavis keeps file access policy in one local helper:

```text
langgraph-app/file_safety.py
```

This module is inspired by Hermes file safety but is not a Hermes runtime
dependency. It is used by:

- bridge context references (`@file`, `@folder`)
- architecture and reviewed repo-skill readers
- disk-backed AlphaRavis artifact reads/writes
- media gallery downloads

The guard blocks sensitive credential/config paths, internal caches, shell
profiles, and OS/system paths before direct reads, lists, writes, or future
delete helpers run. Examples include `.env`, `.ssh`, `.aws`, `.kube`, `.docker`,
`.git`, `.cache`, and common shell profile files.

Optional owner-wide write confinement:

```text
ALPHARAVIS_WRITE_SAFE_ROOT=
```

When this is set, AlphaRavis write/delete helpers must stay under that root in
addition to their tool-specific roots such as the artifact root or media root.

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
  -> pre_run_context_guard
  -> route_decision
  -> hard_context_stop OR fast_chat OR crisis_preflight
  -> crisis_manager when owner crisis recovery is enabled and the big LLM preflight fails
  -> planner
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

### Supporting Modules

agent_graph.py delegates pure logic to focused helper modules to keep
orchestration and implementation separate:

| Module | Responsibility |
|---|---|
| `source_content.py` | Content-type detection, keyword/entity/symbol extraction, line-range parsing, classifier JSON, text windows |
| `command_safety.py` | SSH command classification (read-only vs dangerous), command word/segment helpers |
| `prompt_assembly.py` | Stable prompt context, environment hints, policy prompts, FAST_PATH routing patterns |
| `context_compressor.py` | Message compression, token estimation, ratio token limits, summary preparation |
| `model_metadata.py` | Context length discovery, model metadata, provider context length overrides |
| `retrieval_router.py` | RAG backend selection, hit scoring, reranking, source ingest routing |
| `alpharavis_toolsets.py` | Toolset dataclasses, resolution, materialization, MCP schema cache |
| `error_classifier.py` | Error reason classification (auth, rate_limit, context_overflow, etc.) |
| `file_safety.py` | File read/write safety decisions, sensitive path detection |
| `internal_context.py` | Internal context block scrubbing (streaming + batch) |
| `compression_redact.py` | Secret redaction from tool output and messages |
| `media_analysis.py` | Media (image/video) preparation for model input |
| `operational_logging.py` | Structured operational logging |
| `owner_power_tools.py` | Owner-gated server power commands (llama, ComfyUI) |
| `repo_skills.py` | AI skill card scanning, manifest, draft export |
| `responses_client.py` | Direct `/v1/responses` API client |
| `provider_hardening.py` | Provider hardening (timeouts, retries, compatibility) |
| `document_ingest.py` | Document file loading for RAG ingest |
| `run_state_manager.py` | Run checkpoint save/load/resume |
| `rag_api_client.py` | External RAG API client |
| `rag_pins_manager.py` | Active RAG source pin management |
| `runtime_settings.py` | Runtime overrides from config files |
| `context_references.py` | @file:/@git:/@diff: reference parsing and expansion |
| `curated_memory_review.py` | Curated memory candidate review workflow |
| `maintenance_helpers.py` | Maintenance scheduling decisions |

See `AGENTS.md` Module Boundary Rules for when to extract vs keep in agent_graph.py.

## Agents

AlphaRavis currently uses a swarm-style multi-agent setup.

Direct no-tool model calls inside the graph can use
`ALPHARAVIS_LLM_API_MODE=responses`. That path calls the OpenAI-compatible
`/v1/responses` endpoint on LiteLLM or llama.cpp for planner, fast path, and
summary calls. `langgraph-app/provider_hardening.py` hardens that path with
Hermes-style compatibility retries: unsupported parameters can be removed or
mapped once, Kimi/Moonshot-style models can omit server-managed temperature, and
LiteLLM remains the default gateway abstraction.

The provider hardening layer also exposes small provider profiles. These
profiles only adjust OpenAI-compatible request shape and fallback policy; they
do not introduce native Anthropic/Gemini/Kimi transports. The default `auto`
profile keeps local LiteLLM conservative and records profile metadata in
operational logs/run profiles for diagnosis.

Tool-heavy DeepAgents workers can use Responses-native tool binding through
LangChain `ChatOpenAI`:

```text
ALPHARAVIS_DEEPAGENTS_API_MODE=responses
ALPHARAVIS_DEEPAGENTS_RESPONSES_API_BASE=http://litellm:4000/v1
ALPHARAVIS_DEEPAGENTS_RESPONSES_OUTPUT_VERSION=responses/v1
ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING=true
ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=tool_calling
ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING=false
```

This keeps DeepAgents on its native `create_agent(...)` path while swapping the
model object underneath it. AlphaRavis applies a local startup patch equivalent
to the important part of langchain-ai/langchain PR #35457. That patch fixes the
`AsyncStream` crash when LangChain routes
`disable_streaming="tool_calling"` calls through non-streaming OpenAI code
paths.

The `langgraph-api` container also runs
`langgraph-app/patches/patch_langchain_openai_responses_tool_streaming.py` at
startup. That second patch is inert unless
`ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING=true`. When enabled, it adjusts
LangChain Responses stream conversion so reasoning items, function-call indexes,
partial argument chunks, and final tool-call emission stay coherent for the
local LiteLLM/llama.cpp stack.

Streaming remains configurable for future provider/library upgrades:

```text
# force fully non-streaming internal model calls
ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING=false
ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=true

# experimental full streaming with tool-bound calls
ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING=true
ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=false
ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING=true

# default patched LangChain hybrid: stream unless tools are passed
ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING=true
ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=tool_calling
ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING=false
```

If a local provider has a Responses/tool-call bug, set
`ALPHARAVIS_DEEPAGENTS_API_MODE=chat_completions` or leave
`ALPHARAVIS_DEEPAGENTS_REQUIRE_RESPONSES=false` to fall back to ChatLiteLLM.
For a deliberate Chat Completions full-streaming runtime, use:

```text
ALPHARAVIS_LLM_API_MODE=chat_completions
ALPHARAVIS_DEEPAGENTS_API_MODE=chat_completions
ALPHARAVIS_LLM_STREAMING=true
BRIDGE_PREFERRED_API_MODE=chat_completions
```

The Makefile shortcut is:

```bash
make streaming STREAMING=chat-full
make up-chat-fullstreaming
```

Every specialist prompt includes a local specialist-planning rule. The global
planner creates the compact task contract once before the swarm; each specialist
then adapts that contract into its own role-specific plan before doing work.

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
- Local document search through AlphaRavis pgvector by default, with `rag_api`
  still available as an adapter/reference backend.
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
- Load exact raw archive records by `archive_key`.
- Load archive collection tables of contents by `collection_key`.
- Search debugging lessons.
- Search active workflow skills.
- Search other chat archives only when explicitly requested through `include_other_threads=true`.

### Power Management Agent

Handles local hardware and model lifecycle planning when advanced custom model
management is enabled:

```text
ALPHARAVIS_ENABLE_MODEL_MANAGEMENT=true
ALPHARAVIS_ENABLE_ADVANCED_MODEL_MANAGEMENT=true
ALPHARAVIS_ENABLE_OWNER_POWER_TOOLS=true
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
Document and large-paste RAG uses this AlphaRavis-owned pgvector backend by
default through `ALPHARAVIS_DOCUMENT_RAG_BACKEND=alpharavis_pgvector`; `rag_api`
is a selectable adapter for compatibility and comparison, not the primary
source of truth. Source digest dedup is active by default for identical scoped
source keys, so repeated identical source content can reuse an existing
pgvector catalog/chunk set instead of embedding the same source again.

### Memory Tier Policy

AlphaRavis treats memory as explicit tiers. LangGraph owns the policy decision
for which tier is active in a run; the Bridge and Observer only display the
resulting metadata.

| Tier | Source of truth | Loaded every turn | Retrieved on demand | Compaction rule |
| --- | --- | --- | --- | --- |
| Latest task tail | LangGraph checkpoint `messages` | Yes | No | Protected by head/middle/tail selection; can only be reduced by hard rescue when the request would not fit. |
| Active compaction summary | LangGraph checkpoint summary message plus `context_summary` / `handoff_context_summary` | Yes, after compression | No | Can be summarized again, but must keep archive references and current-task constraints. |
| Raw compression archive | LangGraph Store / Mongo thread archive record | No | `read_archive_record`, `query_archive`, bounded archive tools | Never inject the full raw archive automatically; use windows/search slices for exact old context. |
| Archive collection | LangGraph Store archive-collection record and pgvector catalog | No | Archive collection tools and vector recall | It is a table of contents over child archives; answer exact details only after reading relevant child raw archives. |
| Document / large-paste source | Raw source store plus AlphaRavis pgvector chunks, optional `rag_api` mirror | Only as a compact source marker | `read_source_chunks`, `read_raw_source`, RAG tools | Source marker stays in active context; exact neighboring code/log/prose comes from bounded raw-source reads. |
| Vector recall chunks | AlphaRavis pgvector | No | Semantic/vector search | Retrieval snippets are pointers and evidence, not the source of truth when exact wording or neighboring lines matter. |
| MemoryKernel facts | LangGraph Store curated memories and small matching hints | Only tiny matched hints | `search_curated_memory` / MemoryKernel lookup | Store stable preferences, environment facts, and recurring lessons; do not store raw logs, pasted docs, or one-off task state. |
| Temporary workflow state | LangGraph checkpoint state and run profile | Current run only | No | May be overwritten by graph nodes; durable only if explicitly archived, indexed, or recorded as curated memory. |
| Observer / run telemetry | Bridge observations and run-profile metadata | No | Observer/debug APIs | Debug/status surface only; not model memory and not an answer source. |

Curated-memory review is a separate promotion path. Candidate extraction writes
review records with `status=pending`; those candidates are not part of
always-memory. `accept_curated_memory_candidate` is the promotion boundary and
persists the reviewed memory through the same curated-memory store/index path as
`record_curated_memory`. `reject_curated_memory_candidate` leaves an auditable
rejection record without writing memory.

The active model should prefer the smallest tier that can answer safely: current
task tail first, active summary for old high-level context, vector recall for
finding candidates, then bounded raw archive/source reads for exact evidence.

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

### Run-State Resume Manager

LangGraph checkpoints persist graph state at checkpoint boundaries. AlphaRavis
also keeps a smaller operator-facing run-state checkpoint in MongoDB through
`langgraph-app/run_state_manager.py`. This is the recovery record for disrupted
agent runs where the operator needs the latest plan/task state after a
llama.cpp/LiteLLM disconnect or LangGraph process restart.

The run-state manager stores one latest `current` record per thread in
`ALPHARAVIS_RUN_STATE_DB` / `ALPHARAVIS_RUN_STATE_COLLECTION` and replaces it
atomically as a full Mongo document. The record includes phase, status,
`current_task_brief`, `planner_context`, `planner_last_key`, selected toolsets,
active agent, compact run profile, and provider error classification. Completed
runs are marked `completed`; interrupted provider/swarm runs stay
`awaiting_resume`.

The same manager also exposes generic workflow-record helpers for feature-local
status/history without creating separate state-manager implementations. Records
are keyed by `namespace` and `workflow_id`, support status/file filters, and are
used by the Office managed-workflow layer for validation history, validation
badges, managed batch progress/error counters, and template-merge form state.

On the next message in the same thread, an open checkpoint restores the planner
context and task brief. By default AlphaRavis asks the user whether to continue
and keeps the job saved if no answer arrives; operators can enable automatic
continuation with `ALPHARAVIS_RUN_STATE_AUTO_RESUME=true`.

The Bridge Observer exposes the same records through `GET /api/resume-runs` and
an `Awaiting Resume` panel. That panel is intentionally operator-facing: it
lists the saved thread, phase, task brief, selected toolsets, and resume phrase
without changing LangGraph's same-thread resume semantics.

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
reload_repo_ai_skills
```

These tools are restricted to the repo `ai-skills/` directory.

Before the agent path, AlphaRavis may inject a tiny metadata hint for matching
repo skill cards. This hint contains only names and descriptions. Full skill
instructions are loaded only when an agent calls `read_repo_ai_skill`.

The repo skill index borrows the Hermes manifest pattern: `SKILL.md`,
`DESCRIPTION.md`, and supporting-file mtimes/sizes are cached under
`.cache/alpharavis/repo_skill_manifest.json`. `reload_repo_ai_skills` forces a
rescan and returns added/removed/changed/unchanged status without changing skill
promotion state.

Reviewed disk skills may carry supporting files in `references/`, `templates/`,
`scripts/`, and `assets/`. `read_repo_ai_skill` can load those by relative path,
while the central file-safety guard keeps access inside the requested skill
directory.

Mongo/LangGraph Store skill candidates remain separate from reviewed disk
skills. `record_skill_candidate` writes inactive candidates. `activate_skill_candidate`
still requires `ALPHARAVIS_ALLOW_SKILL_PROMOTION=true`. The optional
`export_skill_candidate_to_repo_draft` tool requires
`ALPHARAVIS_ALLOW_SKILL_DRAFT_EXPORT=true` and writes review-only drafts under
`ai-skills/_drafts/<slug>/SKILL.md`; exporting a draft does not activate the
candidate or make it a normal routing hint.

## Context Compression

AlphaRavis uses a Hermes-style active compression engine plus a separate archive
collection tier.

The active engine lives in `langgraph-app/context_compressor.py` and is shared by
both trigger points:

- `pre_run_context_guard`: pre-route trigger that compacts old active thread
  state before the hard context cutoff and before fast-path/agent-path model
  calls.
- `handoff_context_guard`: pre-swarm trigger when planner/memory/skill setup
  made an agent-path run too large.
- `context_guard_after`: post-run safety net after the current answer is done.

There are not two competing active compression algorithms anymore. Both paths
protect the same kind of material, compress only the middle, and write the raw
removed messages into archives.

### Chat Compression

When the active LangGraph message window exceeds `ALPHARAVIS_ACTIVE_TOKEN_LIMIT`,
the shared compressor runs after the current graph run has produced its answer.
It does not compress in the middle of a task.

Default:

```text
ALPHARAVIS_COMPRESSION_ENGINE=hermes_style
ALPHARAVIS_ENABLE_PERCENT_CONTEXT_LIMITS=true
ALPHARAVIS_COMPRESSION_TRIGGER_RATIO=0.50
ALPHARAVIS_ACTIVE_CONTEXT_TRIGGER_RATIO=0.50
ALPHARAVIS_HANDOFF_CONTEXT_TRIGGER_RATIO=0.50
ALPHARAVIS_HARD_CONTEXT_RATIO=0.95
ALPHARAVIS_ENABLE_PRE_RUN_COMPRESSION=true
ALPHARAVIS_ENABLE_HARD_CONTEXT_TRIM=true
ALPHARAVIS_HARD_CONTEXT_TRIM_RATIO=0.80
ALPHARAVIS_ENABLE_STATIC_CONTEXT_RESERVE=true
ALPHARAVIS_STATIC_CONTEXT_RESERVE_TOKENS=0
ALPHARAVIS_USE_AGENT_SPECIFIC_CONTEXT_RESERVE=true
ALPHARAVIS_ENABLE_FINAL_LLM_BUDGET_GUARD=true
ALPHARAVIS_ENABLE_FINAL_BUDGET_RESCUE=true
ALPHARAVIS_FINAL_BUDGET_RESCUE_MAX_PASSES=3
ALPHARAVIS_ENABLE_PROVIDER_OVERFLOW_RETRY=true
ALPHARAVIS_ENABLE_PROVIDER_CONTEXT_LIMIT_RETRY=true
ALPHARAVIS_DYNAMIC_COMPRESSION_UNTIL_BUDGET=true
ALPHARAVIS_DYNAMIC_COMPRESSION_MAX_PASSES=6
ALPHARAVIS_DYNAMIC_COMPRESSION_HARD_MAX_PASSES=12
ALPHARAVIS_ENABLE_POST_RUN_COMPRESSION=true
ALPHARAVIS_COMPRESSION_PROTECT_FIRST_MESSAGES=3
ALPHARAVIS_COMPRESSION_PROTECT_LAST_MESSAGES=3
ALPHARAVIS_COMPRESSION_TAIL_TOKEN_RATIO=0.20
ALPHARAVIS_COMPRESSION_TAIL_SOFT_CEILING_RATIO=1.5
ALPHARAVIS_PRE_RUN_COMPRESSION_MAX_PASSES=3
```

With percentage limits enabled, AlphaRavis resolves the model context length
first and then computes the actual guard thresholds:

```text
compression_trigger = context_length * ALPHARAVIS_*_TRIGGER_RATIO
hard_cutoff         = context_length * ALPHARAVIS_HARD_CONTEXT_RATIO
```

For a 128k llama.cpp context and the default 50 percent trigger, handoff and
pre-run, handoff, and post-run compression start around 64k estimated tokens,
while the hard stop starts around 121k. The pre-run guard runs before the hard
stop; if normal compression cannot rescue a thread already above the hard limit,
hard trim removes old active messages while preserving the latest user turn. If
the endpoint cannot report context length, the fallback values
`ALPHARAVIS_MODEL_CONTEXT_LENGTH` and `ALPHARAVIS_DEFAULT_CONTEXT_LENGTH` are
used. Set `ALPHARAVIS_ENABLE_PERCENT_CONTEXT_LIMITS=false` to return to the
fixed legacy limits `ALPHARAVIS_ACTIVE_TOKEN_LIMIT` and
`ALPHARAVIS_HANDOFF_CONTEXT_TOKEN_LIMIT`.

Context-length discovery follows the Hermes idea but stays local to
AlphaRavis:

```text
ALPHARAVIS_AUTO_DISCOVER_CONTEXT_LENGTH=true
ALPHARAVIS_CONTEXT_DISCOVERY_API_BASE=
ALPHARAVIS_CONTEXT_DISCOVERY_API_KEY=
ALPHARAVIS_CONTEXT_DISCOVERY_MODEL=
ALPHARAVIS_CONTEXT_DISCOVERY_TIMEOUT_SECONDS=2
```

If `ALPHARAVIS_CONTEXT_DISCOVERY_API_BASE` is empty, AlphaRavis prefers the
direct `BIG_BOSS_API_BASE`, then Responses/OpenAI base URLs. It queries
OpenAI-compatible `/models` metadata and, for llama.cpp, `/v1/props` or `/props`
to read the actually allocated `n_ctx`.

What happens:

1. The engine protects the head: policy/system context, current task brief,
   planner context, MemoryKernel context, skill hint, and latest handoff packet.
2. It protects a recent tail Hermes-style: a small hard minimum by message
   count plus a token budget. The default is the latest 3 messages, then older
   messages until roughly 20 percent of the compression limit is active. A
   soft ceiling prevents a single older oversized message from dragging in a
   huge tail after the minimum is already satisfied. Like Hermes, the latest
   user/human message is always anchored into the tail so the active request is
   never compressed into a reference-only summary.
   If the resulting protected tail still exceeds
   `ALPHARAVIS_COMPRESSION_OVERSIZED_TAIL_RATIO` of the compression budget,
   default 60 percent, AlphaRavis rebalances it: older tail messages move back
   into the compressible middle while the latest user message remains anchored
   by default. This keeps "last three messages" protection from making a huge
   uncompressible tail.
   If the tail is critically oversized above
   `ALPHARAVIS_COMPRESSION_OVERSIZED_TAIL_FORCE_MIDDLE_RATIO`, default 80
   percent, the latest user-message anchor is released too and the huge active
   request can be archived/compressed. This is a last-resort pressure rule; for
   huge pasted documents, the preferred path is still Large-Paste RAG before
   compression.
3. The middle is summarized. Tool outputs are pruned into informative previews
   before the summary call, repeated old tool outputs are deduplicated by hash,
   tool-call JSON arguments are shortened without breaking JSON, and secrets are
   redacted.
4. Previous summaries are updated iteratively instead of starting from zero.
5. A thread-specific raw archive record is stored with the removed messages.
   The archive keeps the original structure and useful content, but credential
   values are redacted so secrets are not leaked into summaries or logs.
6. The raw archive record is queued/indexed in pgvector when vector memory is
   enabled.
7. The active LangGraph message list is replaced by:
   - protected head
   - one reference-only compaction summary
   - one tiny archived-context policy note
   - protected tail
8. A visible Memory-Notice can be returned to LibreChat.

The compaction summary is reference-only. It explicitly tells the next agent
that previous turns are already handled and that it should answer only the latest
user request.

Pre-run compression follows Hermes' safety shape: estimate the active request,
compact when it is over the threshold, re-estimate, and continue until the full
request is under budget. The default dynamic cap is
`ALPHARAVIS_DYNAMIC_COMPRESSION_MAX_PASSES=6`, with
`ALPHARAVIS_DYNAMIC_COMPRESSION_HARD_MAX_PASSES` as the absolute safety cap.
Set `ALPHARAVIS_DYNAMIC_COMPRESSION_UNTIL_BUDGET=false` to return to the legacy
fixed `ALPHARAVIS_PRE_RUN_COMPRESSION_MAX_PASSES` behavior. Like Hermes, the
preflight budget is not just visible chat messages. At
graph build time AlphaRavis estimates the largest DeepAgents static prompt/tool
schema overhead and reserves that budget from the active-message threshold.
`ALPHARAVIS_STATIC_CONTEXT_RESERVE_TOKENS=0` means auto-reserve; set a positive
value only to force an operator override. When
`ALPHARAVIS_USE_AGENT_SPECIFIC_CONTEXT_RESERVE=true`, AlphaRavis uses the
selected or active agent's own reserve once routing/toolset context is known and
falls back to the largest reserve only before that point. This is why the normal
path should stay below the discovered model context without requiring the bridge
to reject the request first.

The same reserve is applied to handoff and post-run compression thresholds, so
tool-heavy agent runs compact back under the effective budget after they add
tool traces or handoff packets. The active state is therefore kept ready for the
next full model request, not merely under the raw chat-history threshold.

`final_budget_rescue` runs immediately before the Swarm model node. It inspects
the full request budget snapshot and, when the active request would exceed the
effective threshold, forces Hermes-style compression until the request is under
the effective budget or the dynamic pass cap is reached before the model
invocation. With dynamic mode disabled, it uses the legacy
`ALPHARAVIS_FINAL_BUDGET_RESCUE_MAX_PASSES` cap.
If the request is still over the hard budget and hard trim is enabled, it trims
old active messages while keeping the latest user turn.

If the provider still raises a classified context overflow or payload-too-large
error from the Swarm model path, `ALPHARAVIS_ENABLE_PROVIDER_OVERFLOW_RETRY`
lets AlphaRavis run final budget rescue once and retry the Swarm invocation
with the compressed state. `ALPHARAVIS_ENABLE_PROVIDER_CONTEXT_LIMIT_RETRY`
also extracts provider-reported context limits from errors such as llama.cpp
`n_ctx_slot` messages and recomputes the retry budget from that smaller real
window.

The final model-call budget guard mirrors Hermes' request estimate more closely
than the active-message compressor can: immediately before direct LLM calls and
inside DeepAgents model invocations, AlphaRavis estimates messages plus bound
tool schemas, model kwargs, and any system prompt that the agent runtime has
materialized into the message list. It logs `llm.request_budget.estimated` with
message/tool/system split counts and warns near or above the LangGraph hard
limit. This guard is observational by default; compression still happens in the
pre-run LangGraph state where old messages can be archived instead of silently
dropped.

Agents can call `inspect_context_budget` to see the detected context length,
discovery base URL, active and hard thresholds, effective thresholds after
static reserves, the derived `compression_summary_budget`, all agent-specific
reserves, archive counts, and whether the current estimate needs compression or
hard rescue. This is the canonical runtime-facing place to read these numbers:
agents and tools should use that budget snapshot or `model_metadata.get_model_context_length(...)`
instead of inventing small static context assumptions.

The Bridge test UI Observer shows the latest recorded context-budget snapshot in
a dedicated `Context Budget` section for each request, including message tokens,
reserve, request estimate, active/effective limits, and hard/effective limits.
It also has a `Source Ingest` section that renders LangGraph-owned source
conversion metadata from `receive.source_ingests`: source key, title, content
type, source status, character counts, indexed/queued backend, RAG-active state,
and whether the active prompt was replaced by a marker. This is display-only;
the Bridge does not own the large-message decision.

The Observer has a `Shrinking` section that turns compression metadata into
operator-readable cards. Each card represents one compression scope
(`pre_run_compression`, `final_budget_rescue`, `post_run_compression`, or
`handoff_context`) and shows:

- active tokens before and after the pass
- shrink percentage and a progress bar
- request-budget before/after when available
- dynamic pass count and whether the pass reached budget
- head/middle/tail message counts and middle-token estimate
- summary-prompt token budget, whether that prompt was pruned, and archive key
- summary prompt overhead/payload budget so wrapper text and protected notes do
  not silently eat the model window
- chunking status, chunk count, chunk payload budget, omitted chunk chars, chunk
  output token cap, and whether the final chunk-synthesis prompt was pruned

The Observer detail pane still has the raw `Kompression` tab for exact JSON.
That tab is useful when the visual Shrinking card suggests a bug, for example a
large before/after gap that does not match `request_tokens_after`, a missing
archive key, `summary_failed=true`, or `summary_chunk_omitted_chars > 0`.

The current compressor borrows mature single-agent ideas from Hermes but does
not import Hermes at runtime. AlphaRavis keeps its own LangGraph-state design,
raw archives, archive collections, MemoryKernel, skill context, and pgvector
retrieval. The ported mechanisms are local helpers:

- image-aware and tool-argument-aware token estimation, with real API usage
  values preferred when model metadata contains them
- percentage-based trigger thresholds calibrated from discovered context length
- JSON-safe tool-call-argument truncation
- tool-output deduplication for the summary prompt only
- conservative summary-prompt budget pruning so the summary model call itself
  stays below the discovered context window while raw middle messages remain in
  the archive
- tool-specific summaries for terminal, file, search, browser/web, and generic
  tools
- anti-thrashing based on `compression_stats.last_compression_savings_pct` and
  `compression_stats.ineffective_compression_count`
- summary failure cooldown with a visible fail-safe reference summary instead
  of silent context loss
- iterative summary updates that keep still-valid facts, remove obsolete points,
  and move completed work into `Progress Done`

Relevant additional settings:

```text
ALPHARAVIS_COMPRESSION_TOOL_ARGS_MAX_CHARS=1500
ALPHARAVIS_COMPRESSION_TOOL_ARGS_HEAD_CHARS=1000
ALPHARAVIS_COMPRESSION_TOOL_ARGS_TAIL_CHARS=300
ALPHARAVIS_COMPRESSION_DEDUP_MIN_CHARS=200
ALPHARAVIS_WORKFLOW_EVENT_OUTPUT_MAX_CHARS=900
ALPHARAVIS_WORKFLOW_EVENT_OUTPUT_HEAD_CHARS=620
ALPHARAVIS_WORKFLOW_EVENT_OUTPUT_TAIL_CHARS=180
ALPHARAVIS_COMPRESSION_IMAGE_TOKEN_ESTIMATE=1600
ALPHARAVIS_COMPRESSION_ANTI_THRASHING_ENABLED=true
ALPHARAVIS_COMPRESSION_MIN_SAVINGS_RATIO=0.10
ALPHARAVIS_COMPRESSION_FAILURE_COOLDOWN_SECONDS=600
ALPHARAVIS_COMPRESSION_SUMMARY_RATIO=0.20
ALPHARAVIS_COMPRESSION_SUMMARY_MIN_TOKENS=1200
ALPHARAVIS_COMPRESSION_SUMMARY_MAX_TOKENS=0
ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_RATIO=0.75
ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_MIN_TOKENS=8192
ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_MAX_TOKENS=0
ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_CHARS_PER_TOKEN=2.0
ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_OVERHEAD_RESERVE_TOKENS=512
ALPHARAVIS_ENABLE_COMPACT_INSTRUCTIONS=true
ALPHARAVIS_COMPACT_INSTRUCTIONS_MAX_CHARS=1200
ALPHARAVIS_COMPRESSION_ENABLE_CHUNKED_SUMMARY=false
ALPHARAVIS_COMPRESSION_SUMMARY_CHUNK_RATIO=0.03
ALPHARAVIS_COMPRESSION_SUMMARY_CHUNK_MIN_TOKENS=300
ALPHARAVIS_COMPRESSION_SUMMARY_CHUNK_MAX_TOKENS=0
ALPHARAVIS_COMPRESSION_SUMMARY_CHUNK_OVERLAP_CHARS=1000
ALPHARAVIS_COMPRESSION_SUMMARY_MAX_CHUNKS=12
ALPHARAVIS_DEFAULT_CONTEXT_LENGTH=128000
ALPHARAVIS_MODEL_CONTEXT_LENGTH=128000
```

Summary budgets are ratio-first and use the model context window, not the
smaller active-compression target. This keeps two concepts separate:

- `compression_token_limit`: how small the active LangGraph message state
  should become after compaction, usually the effective active limit.
- `summary_context_token_limit`: how much context the summary model can use for
  the internal summary call, derived from discovered model context length,
  including llama.cpp `/props` / `n_ctx` when available.

`*_MAX_TOKENS=0` means no fixed absolute cap; a positive max is only an
operator override for a smaller hard cap. With a 128k model window and a
75 percent prompt ratio, the summary-prompt budget can therefore be 96k even
when the active compression target is 64k. The same derived values are exposed
in `inspect_context_budget` under `compression_summary_budget` so downstream
agents can see `summary_prompt_tokens`, `summary_output_tokens`, and
`summary_chunk_output_tokens` without duplicating the math.

Focused compaction is a bounded summary-selection hint, not a new runtime task.
When enabled by `ALPHARAVIS_ENABLE_COMPACT_INSTRUCTIONS=true`, the latest user
message can include `<focus_topic>...</focus_topic>`,
`<compact_instructions>...</compact_instructions>`, `/compact ...`,
`@compact ...`, or `@focus ...`. AlphaRavis extracts up to
`ALPHARAVIS_COMPACT_INSTRUCTIONS_MAX_CHARS` and passes that text into one-shot
and chunked compression prompts as "User compaction instructions." The prompt
explicitly says these instructions only decide what the summary should preserve.
They are also recorded in compression archive metadata/run profiles and rendered
as `Compact Focus` in Observer `Shrinking` cards. This keeps compression
transparent and local; AlphaRavis does not use opaque provider-native compact
items as the primary format.

Chunked summary compression is experimental and off by default for ordinary
over-budget runs. It exists for the case where normal active compression
selected a large middle section, but the summary-model prompt budget would
otherwise force a blunt head/tail prune. The compressor may still force chunked
summary when oversized-tail rescue moves the latest user message into the
compressible middle, because that is the "otherwise cannot fit" path for a
huge pasted file/request. The default path remains the simpler bounded one-shot
summary when no oversized-tail rescue is involved.

When `ALPHARAVIS_COMPRESSION_ENABLE_CHUNKED_SUMMARY=true`, or oversized-tail
rescue forces the latest user message into compression, and the summary input
exceeds `ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_*`, AlphaRavis changes only the
summary-generation step:

1. The selected middle messages are still archived in full redacted form before
   they leave active context.
2. The prepared middle summary input is split into bounded chunks sized from
   the ratio-derived summary-prompt budget and
   `ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_CHARS_PER_TOKEN`. A positive
   `ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_MAX_TOKENS` can still cap that value,
   but the default `0` lets it scale with the discovered context. Before sizing
   a chunk, AlphaRavis estimates the prompt wrapper/protected-note overhead and
   subtracts that plus
   `ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_OVERHEAD_RESERVE_TOKENS`; this avoids
   the chunk call itself exceeding the summary model context.
3. Adjacent chunks overlap by
   `ALPHARAVIS_COMPRESSION_SUMMARY_CHUNK_OVERLAP_CHARS` so details near chunk
   boundaries are less likely to disappear.
4. Each chunk is summarized with the same reference-only section contract.
5. A final synthesis pass merges the intermediate chunk summaries into the one
   active compaction summary that remains in LangGraph state.
6. If chunking itself fails, the normal fail-safe fallback summary is used and
   the archive reference remains available.

The important debugging fields are:

```text
summary_chunking_used
summary_chunk_count
summary_chunk_chars
summary_chunk_prompt_token_limit
summary_chunk_payload_token_limit
summary_chunk_prompt_overhead_tokens
summary_chunk_overlap_chars
summary_chunk_max_chunks
summary_chunk_omitted_chars
summary_chunk_output_tokens
summary_chunk_summary_tokens_estimate
summary_chunk_synthesis_pruned
summary_chunk_synthesis_tokens_estimate
summary_chunk_synthesis_payload_token_limit
summary_chunk_synthesis_prompt_overhead_tokens
```

`summary_chunk_omitted_chars` should normally be `0`. A positive value means
the prepared middle exceeded `ALPHARAVIS_COMPRESSION_SUMMARY_MAX_CHUNKS` and
some prepared summary input was not sent to the summary model. Exact raw
messages are still in the archive, and the final synthesis prompt receives an
explicit note to mention archive lookup for omitted middle details, but the
active summary may be less complete. Increase the max chunks or reduce chunk
size/overlap only after checking model latency. The Observer Shrinking section
exposes these fields directly so a live llama.cpp run can be inspected without
reading raw JSON first.

Compression also has a structured progress event surface. `compress_messages`
can receive a callback and records the same event list in archive metadata and
run-profile compression debug fields. The event names are:

```text
compression.started
compression.precompact
compression.workflow_events.compacted
compression.chunk.started
compression.chunk.completed
compression.synthesis.started
compression.synthesis.completed
compression.synthesis.failed
compression.skipped
compression.completed
compression.postcompact
```

The Bridge maps the latest compression event from LangGraph updates to compact
`context_compaction` reasoning/status activity. The full timeline remains in
Observer raw compression metadata. This is intentionally a status surface; it
does not change the active summary contract or make the Bridge own compression.
Large-source ingest uses the separate `large_ingest.*` / `document_ingest.*`
event namespace.

Tool and workflow telemetry is compacted before the summary prompt receives the
middle context. Assistant tool-call requests, tool results, duplicate tool
outputs, and long action logs are collapsed into a separate
`Workflow / Tool Event Compact Log`; normal chat messages stay in the normal
middle-message section. The compact log is stored in archive metadata and in
the raw archive record beside the redacted original messages. This keeps action
history inspectable without making tool logs behave like ordinary user/assistant
conversation. Observer `Shrinking` cards expose the compact event counts and a
bounded `Workflow / Tool Events` preview for the selected compression scope.

`compression.precompact` is emitted after the graph has selected
head/middle/tail and before the summary model is called. It records the reason,
scope, token pressure, H/M/T counts and indexes, summary prompt pressure, and
whether chunked summary will run. `compression.postcompact` is appended after
an archive key is allocated and records the archive key, before/after token
estimates, H/M/T counts, summary failure state, and chunking result.
`compression.workflow_events.compacted` records the compacted workflow event
counts and compact-log size.

`compression_stats` is stored in LangGraph state and currently contains:

```text
last_compression_savings_pct
ineffective_compression_count
summary_failure_cooldown_until
last_summary_error
last_summary_failed_at
last_summary_fallback_used
```

This is the small AlphaRavis equivalent of Hermes' ContextEngine status. A full
pluggable ContextEngine abstraction was not copied because LangGraph already
uses explicit nodes (`handoff_context_guard` and `context_guard_after`), and a
larger plugin layer would add complexity without improving the current graph.

The active context does not receive all archive collections. It receives only a
small policy note:

```text
Archived context is available via semantic_memory_search; retrieve before
relying on old details.
```

### Handoff Context Guard

Before the swarm starts, AlphaRavis can run the same active compressor with the
smaller `ALPHARAVIS_HANDOFF_CONTEXT_TOKEN_LIMIT`. If the active message window is
already too large, the guard compresses the middle into a handoff summary,
archives the removed messages as redacted raw archive records, and keeps the
important coordination material active:

- current task brief
- planner execution plan
- MemoryKernel and skill hints
- latest handoff packet
- recent tail messages

Default:

```text
ALPHARAVIS_ENABLE_HANDOFF_CONTEXT_GUARD=true
ALPHARAVIS_HANDOFF_CONTEXT_TOKEN_LIMIT=12000
ALPHARAVIS_HANDOFF_PACKET_MAX_CHARS=4000
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

Raw archive records are not deleted. Archive collections are not normal active
chat context. They are a thread-scoped Inhaltsverzeichnis / router over older
raw archive records.

Archive collections contain:

- collection key
- child archive keys
- covered range
- main topics
- important files
- commands/tools used
- errors/signals
- decisions
- open tasks
- retrieval keywords

Both raw archives and archive collections are indexed in pgvector. Retrieval
works like this:

1. `semantic_memory_search` searches current-thread vector memory by default.
2. If a hit is `source_type=archive_collection`, the LLM reads
   `child_archive_keys`.
3. The LLM calls `read_archive_record` for only the relevant raw archive keys.
4. The answer is based on the loaded raw archive content.

Cross-thread archive retrieval remains off by default and requires an explicit
tool call with `include_other_threads=true`.

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
preview; the full content stays on disk. Artifact writes route through
`retrieval_router.ingest_source(source_type="artifact", preferred_backend="auto")`
after the file/store record is created, so artifact metadata uses the same
normalized RAG fields as documents and large pasted sources.

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
Archive and archive-collection sources choose that profile from their content:
code fences/common source syntax use the code profile, log/traceback lines use
the log profile, and normal conversation archives stay on the chat profile.
When `ALPHARAVIS_PGVECTOR_SECTION_LEVEL_ARCHIVE_SPLITTING=true`, mixed archive
and archive-collection sources are split by ordered prose/log/code/config
sections before chunking. Each section uses the matching chunk profile while
preserving original order. This keeps one compressed archive searchable across
conversation, log, config, and code parts without flattening everything into one
profile.

The tool exposed to agents is:

```text
semantic_memory_search
query_source
query_sources
query_archive
agentic_rag_retrieve
read_archive_record
read_archive_collection
```

By default it searches the current thread plus global memories and federates
with `rag_api` for external document hits. `semantic_memory_search` returns
structured hits containing `source_type`, `source_key`, `title`, `score`,
`preview_text`, `metadata`, and `child_archive_keys` when present.
When an agent already knows the relevant `source_key`, archive key, or RAG
`file_id`, `query_source`, `query_sources`, and `query_archive` run a scoped
semantic search against only those sources. This keeps known-source questions
from pulling unrelated chunks into context and mirrors the `rag_api`
`/query`/`/query_multiple` pattern for external documents.
If `ALPHARAVIS_ENABLE_RAG_RERANKING=true`, these scoped hits are reranked before
grading/context-packet construction. Deterministic reranking is local; model
reranking calls the configured llama.cpp Qwen3-Reranker endpoint and falls back
to deterministic reranking on endpoint errors when configured.
When the question needs the Agentic-RAG control loop, `agentic_rag_retrieve`
runs source-scoped retrieval, deterministic grading by default, one optional
query rewrite, and returns a bounded `context_packet` plus `graph_trace`. When
`ALPHARAVIS_AGENTIC_RAG_LLM_GRADING=true`, the router can call an optional
structured-output LLM grader and falls back to deterministic grading on errors.
It is an explicit tool path, not automatic archive injection.
Thread-level RAG activation metadata is carried separately:

```text
rag_active
active_rag_file_ids
active_source_keys
rag_activation_reason
archive_rag_mode
```

Manual/operator pins are stored in a shared Mongo collection through
`langgraph-app/rag_pins_manager.py` when available. The LangGraph pin/unpin
tools and the Bridge Observer `RAG Pins` panel both read and write that store,
so UI pin changes affect the same active-RAG prefetch path as agent tool calls.

Vector backfill is available as exact commands in addition to query search:

- `queue_current_thread_vector_backfill` indexes stored session turns,
  artifacts, archives, and archive collections for the active thread.
- `queue_recent_artifact_vector_backfill` indexes the last N artifact records.
- `queue_selected_source_vector_backfill` resolves explicit raw-source,
  large-paste, artifact, archive, archive-collection, or session-turn keys.

Model-management actions are split between direct primitives and lifecycle
flows. `check_ollama_models`, `load_embedding_model`,
`unload_ollama_model`, and `run_embedding_jobs` perform the concrete Ollama or
pgvector operation. `run_embedding_memory_jobs` wraps those primitives in the
safe embedding-maintenance policy.

Crisis recovery has two entry points: preflight before the planner and mid-run
recovery after provider failures. Mid-run recovery classifies timeout,
connection, 502/server, overload, and rate-limit failures, runs a capped Crisis
Manager attempt, applies a readiness gate, and retries the swarm only when the
primary backend is ready. Caps are stored in run profile metadata and include
attempt count, wall-clock time, and a recursive-loop guard.

`ingest_source(...)` sets these fields for explicit document and large-paste
sources so later graph nodes can auto-retrieve bounded chunks from those
sources. Compression archives set `rag_active=false` and
`archive_rag_mode=tool_only`. On the Agent path, archive auto-intent is enabled
by default by `ALPHARAVIS_ARCHIVE_AUTO_ON_INTENT_AGENT_DEFAULT=true`, so
current-thread archives can still be checked by the small classifier when no
active document/source prefetch is taking precedence. Fast Path bypasses this
node. Set `archive_rag_mode=manual` to keep archives strictly tool-only for a
thread.
Important architecture rule: large-paste, RAG, and compression policy is owned
by LangGraph. The Bridge, Deep Agents UI, ACP clients, and direct callers should
forward input and surface metadata; they must not become the place that decides
whether a large paste is RAG/source/summary. The central LangGraph flow is:

1. Paired manual markers (`/rag ... /rag`, `/rake ... /rake`,
   `/index ... /index`, `/ingest ... /ingest`, `/big-context ... /big-context`)
   and fenced `<big-context name="...">...</big-context>` blocks force
   immediate source ingest.
2. Plain large human pastes are not automatically indexed just because they are
   long. With the default
   `ALPHARAVIS_LARGE_PASTE_RAG_AUTO_STAGE=post_compression`, old context first
   goes through normal pre-run compression.
3. After that compression pass, `large_paste_post_compression_node`
   re-estimates the active request. Only if the request is still above
   `ALPHARAVIS_LARGE_PASTE_RAG_POST_COMPRESSION_TRIGGER_RATIO` of the active
   compression budget, default `0.80`, does LangGraph call
   `ingest_source(source_type="large_paste")` for the document/code/log body.
4. Successful or queued ingest replaces the full active paste with a compact
   source marker. The marker contains the source key, optional RAG file id,
   title, retrieval instruction, and a short `Source manifest` line with content
   type, character counts, chunk stats, source digest, and indexed/queued
   backends. The original raw text is preserved in raw-source storage and/or the
   selected ingest backend, not kept as full active prompt text. If no explicit
   question/task is detected, the marker tells the model to ask what to extract
   or analyze instead of doing broad unsupported analysis.
5. If the marker replacement still leaves too much non-document chatter active,
   the same node can run one follow-up compression pass at
   `ALPHARAVIS_LARGE_PASTE_POST_RAG_COMPRESSION_TRIGGER_RATIO`, default `0.80`.
6. Independently, the compressor protects a recent tail but rebalances it when
   that tail exceeds `ALPHARAVIS_COMPRESSION_OVERSIZED_TAIL_RATIO`, default
   `0.60`, of the compression budget: older tail messages move back into the
   compressible middle while the latest user message stays anchored by default.
7. If the protected tail is still critically oversized above
   `ALPHARAVIS_COMPRESSION_OVERSIZED_TAIL_FORCE_MIDDLE_RATIO`, default `0.80`,
   even the latest user message can be released into the compressible
   middle/archive. For a huge latest paste, the Large-Paste RAG post-compression
   path is deferred to first; if compression still has to summarize a too-large
   middle block, chunked summary is forced so the summary model prompt itself
   does not overflow.

Before ingest, AlphaRavis classifies the large paste as `document`,
`instruction`, `mixed`, or `unknown` using deterministic markers/heuristics and,
for long mixed/noisy pastes, the safe Qwen3.5 2B classifier as a bounded
structure helper. Instruction-like pastes are indexed as `large_instruction` in
AlphaRavis pgvector for exact lookup but do not activate automatic document RAG;
the replacement message contains a condensed instruction brief. Mixed pastes
keep active document RAG, include the condensed instruction brief beside the
source handle, and strip obvious instruction text from the indexed document body
when a document/data section can be separated. This prevents prompt
instructions from becoming ordinary search material while still keeping the
important source content retrievable by source key.
`active_rag_prefetch_node` later consumes the active source/file ids and injects
only bounded retrieved chunks in an `<active-rag-context>` system message. If
there are only current-thread archive keys, Agent-path runs also ask the safe
Qwen3.5 2B classifier whether the latest request is archive recall and which
bounded `search_query` to use. Confirmed recall requests can prefetch bounded
chunks from the most recent `ALPHARAVIS_ARCHIVE_AUTO_ON_INTENT_MAX_ARCHIVES`
archives. The classifier is only a structure/query helper; invalid JSON,
endpoint errors, timeouts, or low confidence fall back to the local
archive-recall condenser/heuristic. The prompt tells the classifier to reject
new current-source/media tasks, including uploads, files, images, videos, URLs,
Pixelle outputs, and active sources, unless the user explicitly asks for
older/archive context. This prevents old archive hits from polluting a concrete
"use this video/file/source now" task.
The Bridge Test UI includes a classifier probe for this small-model path. It
can run local fallback checks for short direct, long noisy, instruction-only,
document-only, mixed, and simulated down/invalid/timeout cases, or call the
configured Qwen endpoint for the semantic cases.
LibreChat document uploads use the same activation path: the Bridge registers
downloadable `file` / `input_file` document parts with media-gallery, maps the
stored media path into the LangGraph workspace, sends it as
`pending_document_ingests`, and `run_profile_start_node` loads it through
LangChain document loaders before calling `ingest_source(...)`.
The AlphaRavis pgvector path follows the same retriever shape as `rag_api`:
`query + source key(s) + k`, where a single source maps to `$eq` semantics and
multiple sources map to `$in` semantics. It can also apply an optional
pgvector-distance cutoff with `ALPHARAVIS_PGVECTOR_DISTANCE_THRESHOLD`, matching
the intent of `rag_api`'s `RAG_DISTANCE_THRESHOLD` while keeping the setting
separate for AlphaRavis's own table.

Compression archives can optionally be mirrored into `rag_api` as a secondary
retrieval index with:

```text
ALPHARAVIS_ENABLE_RAG_ARCHIVE_MIRROR=true
```

The raw archive still lives in MongoDB/LangGraph Store and remains the source of
truth. The mirror uses `file_id=archive:<archive_key>` and lets
`query_archive(...)` retrieve bounded chunks through `rag_api` before falling
back to AlphaRavis pgvector. This path is default-off so normal archive
compression behavior does not change unless the operator enables it.
The Bridge Test UI Observer exposes an `Archive RAG Smoke` panel that exercises
the same mirror-and-query path and reports acceptance checks without loading the
whole archive into active model context.
It also exposes `Memory Embed Tester`, a diagnostic-only probe for validating
the configured text or vision embedding endpoint before that endpoint is used by
AlphaRavis pgvector or the optional `rag_api` mirror.

It searches other AlphaRavis threads only when a tool call explicitly sets
`include_other_threads=true`. Enabling this backend indexes new records from
that point onward. Existing MongoDB/store history is intentionally not
bulk-backfilled automatically.

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
ALPHARAVIS_ENABLE_OWNER_POWER_TOOLS=true
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
ALPHARAVIS_ENABLE_OWNER_POWER_TOOLS=false
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

The optional scheduler performs step 4 repeatedly when enabled:

```text
ALPHARAVIS_ENABLE_EMBEDDING_SCHEDULER=true
```

The optional vector backfill daemon is deliberately bounded. It only searches
existing Store indexes for `ALPHARAVIS_VECTOR_BACKFILL_QUERY` and queues matching
records. It is not a startup-time full-history import.

## Hard Context Cutoff

AlphaRavis has two hard cutoff layers:

```text
BRIDGE_HARD_INPUT_TOKEN_LIMIT=0
ALPHARAVIS_HARD_CONTEXT_TOKEN_LIMIT=128000
ALPHARAVIS_HARD_CONTEXT_RATIO=0.95
```

The bridge-level cutoff is disabled by default. LangGraph owns the normal hard
context decision so it can first use model context discovery, pre-run
compression, and hard trim on the active checkpointed context before routing to
fast path, planner, swarm, or crisis manager. When percentage limits are
enabled, the graph hard cutoff is computed from discovered context length and
`ALPHARAVIS_HARD_CONTEXT_RATIO`; `ALPHARAVIS_HARD_CONTEXT_TOKEN_LIMIT=0` still
disables the graph hard stop explicitly. Set `BRIDGE_HARD_INPUT_TOKEN_LIMIT` to
a positive value only when an operator explicitly wants the bridge to reject
oversized raw requests before LangGraph sees them.

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

The same tool also exposes AlphaRavis's first-level lazy tool categories:

```text
coding/read
coding/write
coding/execute
media/image
media/video
media/audio
rag/documents
rag/memory
system/docker
system/ssh
system/power
```

The model starts with category awareness and only calls concrete tools when the
active task needs them. Specialist workers bind bounded local/MCP bundles from
`alpharavis_toolsets.py` at graph build time, and `run_profile.loaded_toolsets`
records the loaded per-agent profiles.

## Media, Vision, And pgvector Dimensions

Text memory and vision/media memory use separate pgvector tables:

```text
ALPHARAVIS_PGVECTOR_TABLE=alpharavis_memory_vectors
ALPHARAVIS_VISION_PGVECTOR_TABLE=alpharavis_media_vectors
```

This avoids mixing embeddings with different dimensions in one `vector(...)`
column. If a future multimodal embedding model returns one shared dimension for
text, image, and video-frame queries, the same model can still be used behind
the vision route. Until then, media records are linked by `source_key`,
`file_id`, `thread_id`, and metadata instead of being forced into the text
vector table.

Media is safe-by-default:

- LibreChat/OpenWebUI media blocks are reduced to URL/file-id/type metadata by
  the bridge unless `BRIDGE_ALLOW_RAW_MEDIA_CONTEXT=true`.
- Incoming image/video/document blocks are mirrored through `media-gallery`
  first when the corresponding Bridge flags are enabled. Image/video blocks are
  rewritten only in the AlphaRavis-facing media marker to the stable gallery URL;
  document blocks keep their original URL marker but also receive a
  `pending_document_ingests` entry for LangGraph RAG ingest. LibreChat's visible
  attachment record and original upload storage stay unchanged.
- Pixelle output URLs are registered with `media-gallery`.
- The gallery downloads/stores returned assets under `media-data` and records
  metadata in MongoDB. For OfficeCLI output it also mounts
  `/workspace/office-output` and exposes static download URLs under
  `/office-output/<relative-path>`, lightweight JSON listings at `/office/files`
  and `/office/templates`, a constrained `/office/upload` endpoint for
  `.docx`, `.pptx`, and `.xlsx` files, and Phase-5 plan endpoints
  (`/office/template-merge`, `/office/validate`, `/office/batch`,
  `/office/roundtrip`) that return safe quoted OfficeCLI command plans without
  starting a managed job. Phase-6 managed workflow endpoints add non-destructive
  Office follow-ups: `/office/preview` plans sibling HTML/PNG previews,
  `/office/repair` plans a `<name>-repaired.<ext>` copy instead of overwriting
  the original, `/office/watch/start|stop|status` tracks the watch lifecycle
  while keeping the standalone OfficeCLI watch URL compatible with an embedded
  UI iframe, and `/office/blueprints`, `/office/blueprints/create`, plus
  `/office/blueprints/suggest` expose blueprint listing/creation/hints for
  polished existing documents. Office file records include direct links to
  existing sibling preview artifacts named `<name>-preview.png` and
  `<name>-preview.html`, while those preview artifacts are hidden from normal
  document records. The shared output root and upload results are chowned to
  configurable host ownership (`ALPHARAVIS_OFFICE_OUTPUT_HOST_UID/GID`, default
  `1000:1000`) so Docker-created files remain editable on the bind-mounted host
  checkout. Office outputs are not copied into MongoDB by this path.
- `media-gallery` accepts normal HTTP(S) image/video URLs and inline `data:`
  image/video blocks. Inline payloads are written to disk but not copied back
  into MongoDB asset metadata.
- The media-gallery service stores optional original/processed derivation
  fields and exposes `/gallery?view=all|original|processed` for operator
  inspection with copyable stable media URLs. Gallery/API listing can
  filter by `thread_id`, `thread_key`, or `group_id`, sort by date/name/type,
  and group by date, no section, thread, group, or media type. The gallery UI
  defaults to collapsible date sections and hides technical source/thread/group
  metadata from cards so the mobile view stays focused on previews, date/time,
  copy, open, and upload actions. The Gallery also accepts direct browser
  uploads through `POST /assets/upload`, stores them in `media-data`, records
  them as original `gallery_upload` assets in MongoDB, and redirects back to the
  chronological view. It serves its own `/favicon.svg` so browser tabs and
  mobile shortcuts show a Media Gallery identity instead of the default browser
  icon.
- Media-gallery Mongo state is split conceptually:
  - `assets`: one row per file/media asset
  - `references`: where that asset appeared in chat/tool context
  - `alpharavis_embedding_jobs`: durable index queue, including
    `media_analysis` jobs. Stale `running` rows are claimable again after
    `ALPHARAVIS_EMBEDDING_JOB_STALE_AFTER_SECONDS` so interrupted dev reloads
    or container restarts do not strand queued document/media ingest forever.
  - `alpharavis_media_vectors`: searchable frame/media embedding rows
- Registration is metadata-only by default. Immediate registration-time vision
  indexing requires an explicit `index=true` tool argument or
  `ALPHARAVIS_MEDIA_REGISTER_INDEX_ON_REGISTER=true`.
- Gallery presence does not mean indexed. Agents must check
  `inspect_media_index_status` or `inspect_embedding_queue_status` before
  claiming that a video's visual contents are searchable.
- Automatic video indexing is controlled by:

```text
ALPHARAVIS_MEDIA_AUTO_INDEX_ENABLED=true
ALPHARAVIS_MEDIA_AUTO_INDEX_USER_UPLOADS=true
ALPHARAVIS_MEDIA_AUTO_INDEX_PIXELLE_MCP_OUTPUTS=false
ALPHARAVIS_MEDIA_AUTO_INDEX_LINK_REFERENCES=false
ALPHARAVIS_MEDIA_INDEX_VERSION=2026-05-12-v1
ALPHARAVIS_MEDIA_VISION_EMBEDDING_MODEL_CARD=vision-embed
```

- Dedupe is based on the media source key plus model-card id, media index
  version, and chunking-config hash. Multiple chat references should create
  multiple reference records, not repeated full video embeddings.
- Optional vision embeddings are experimental and written only when
  `ALPHARAVIS_ENABLE_VISION_VECTOR_MEMORY=true`. The normal memory/RAG bring-up
  path uses text embeddings only through LiteLLM `memory-embed`, defaulting to
  an Ollama-native `ollama/qwen3-embedding:0.6b` route.
- The vision embedding client prefers a direct external model URL when
  configured:

```text
ALPHARAVIS_VISION_EMBEDDING_MODEL_URL=http://<vision-embedding-host>:<port>/v1
ALPHARAVIS_VISION_EMBEDDING_MODEL=<model-name>
```

  If that is empty, it falls back to `ALPHARAVIS_VISION_EMBEDDING_BASE_URL`,
  then `VISION_EMBEDDING_API_BASE`, then the text pgvector/OpenAI-compatible
  base URL. This supports either a dedicated llama.cpp vision embedding server
  or the existing LiteLLM `vision-embed` route. The Makefile can write the
  direct route during install/update/start via `VISION_URL` and `VISION_MODEL`.
- `prepare_media_for_model` is the explicit video preparation tool. Its default
  fallback is `register_only`; it downloads and extracts frames only for
  `analyze` or `index` decisions and only when
  `ALPHARAVIS_VIDEO_ANALYSIS_ENABLED=true`.
- Prepared videos produce timestamped frame files and a manifest under
  `ALPHARAVIS_VIDEO_ANALYSIS_CACHE_ROOT`, with public URLs derived from the
  media-gallery `/media/...` path.
- `inspect_media_index_status` lets the context retrieval agent check which
  media/frame records are already in `alpharavis_media_vectors` and whether
  matching media-analysis jobs are still queued/running/failed in
  `alpharavis_embedding_jobs`.
- `inspect_embedding_queue_status` exposes the shared queue status for text,
  archive, artifact, memory, session-turn, and media-analysis indexing work.
- The Bridge Observer polls the same queue status endpoint for operator
  visibility, showing pending/running/failed/done counts, recent active queued
  source jobs, and per-source chunk progress while large-source embeddings
  drain asynchronously. Queue rows store a small progress JSON document that is
  updated by `run_embedding_jobs` after each indexed chunk.
- `prepare_media_for_model(mode="index")` creates durable `media_analysis`
  jobs in the same queue that text/context indexing already uses, so
  `run_embedding_memory_jobs` can drain both text and video work.
- Optional audio transcription, frame captioning, and richer scene grouping
  remain future provider/pipeline work.

## OpenWebUI

OpenWebUI is an optional second frontend, not a second brain. It should point to
the AlphaRavis Bridge:

```text
OPENAI_API_BASE_URL=http://api-bridge:8123/v1
```

OpenWebUI passthrough is enabled in the example env so clients can use the
bridge's OpenAI-compatible surface. Native tool calling must still be enabled
per model in the OpenWebUI UI when the chosen model supports it. AlphaRavis
keeps LangGraph routing, memory, RAG, Hermes delegation, Pixelle, and approval
rules.

## Tool Calling Mode

AlphaRavis's LangGraph/DeepAgents workers use LangChain tools, not the old
prompt-only "pretend to call a tool" style. When configured with:

```text
ALPHARAVIS_DEEPAGENTS_API_MODE=responses
```

the DeepAgents model binding uses LangChain `ChatOpenAI` with the Responses API
path and `output_version=responses/v1`, falling back to Chat Completions only
when the runtime/provider cannot support it. OpenWebUI's "Native" setting is
separate: it controls how OpenWebUI calls tools inside OpenWebUI chats, while
AlphaRavis still performs its own LangGraph-native tool execution behind the
Bridge.

DeepAgents Responses token streaming is intentionally disabled by default for
the current local stack. Tool loops still work normally: the model returns a
complete tool call, LangGraph executes the tool, the tool result is added back
to state, and the next model step continues. What is disabled is live token
deltas from the internal DeepAgents model call. The bridge can still expose
Responses-style SSE events to clients.

The `alpha_ravis_swarm` node is itself a compiled nested LangGraph. The Bridge
therefore requests LangGraph streams with `BRIDGE_STREAM_SUBGRAPHS=true` so
worker `messages/partial` events from that nested Swarm graph are forwarded to
LibreChat and the Bridge Test UI. If this is disabled, the Bridge only sees the
completed top-level Swarm result and cannot stream worker tokens in real time.

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
- Local rotating AlphaRavis logs:

```text
logs/operational/alpharavis.log
logs/operational/alpharavis.jsonl
logs/debug/alpharavis-debug.log
logs/debug/alpharavis-debug.jsonl
```

`langgraph-app/operational_logging.py` records timestamped operational events
for bridge requests, run start/finish, route decisions, LLM call duration and
failures, Pixelle/ComfyUI preflight, semantic memory search, and dependency
health. Operational logs are always meant for owner debugging, not model
context. The separate debug-all logger is disabled by default and can be turned
on only while diagnosing noisy issues.

```text
ALPHARAVIS_OPERATIONAL_LOGGING=true
ALPHARAVIS_DEBUG_ALL_LOGGING=false
ALPHARAVIS_LOG_RETENTION_DAYS=4
```

The file and JSONL formatters redact obvious secrets before writing. In Docker,
`langgraph-api` and `api-bridge` mount the shared host folder `./logs` to
`/logs`.

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
