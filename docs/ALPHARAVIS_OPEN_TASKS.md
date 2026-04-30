# AlphaRavis Open Tasks

This is the running backlog for features that are intentionally prepared but
not fully wired yet.

## Custom Model / Power Management

Status: prepared, default off. Owner tool file exists and safe owner tools are wired.

Implemented:

- `langgraph-app/model_management.py` exists as the custom hardware layer.
- `.env(exaple)` contains all switches and defaults them off.
- `make model-management` can write the relevant `.env` switches.
- `power_management_agent` is only registered when:

```text
ALPHARAVIS_ENABLE_MODEL_MANAGEMENT=true
ALPHARAVIS_ENABLE_ADVANCED_MODEL_MANAGEMENT=true
```
- Owner-only tools from `langgraph-app/owner_power_tools.py` are available when:

```text
ALPHARAVIS_ENABLE_OWNER_POWER_TOOLS=true
```
- Safe owner actions are wired:
  - check llama server
  - start/restart llama server
  - read llama logs
  - check/wake ComfyUI
  - start all model services
  - read Pixelle logs when Docker is reachable
- Protected owner actions are wired through human approval:
  - shutdown llama server
  - shutdown ComfyUI server
- `power_management_agent` uses `ALPHARAVIS_POWER_MANAGER_MODEL` when advanced
  model management is enabled.

Still needed:

- Provide the curated external action endpoint:

```text
ALPHARAVIS_MODEL_MGMT_ACTION_URL=
ALPHARAVIS_MODEL_MGMT_API_KEY=
ALPHARAVIS_MODEL_MGMT_ALLOW_ACTIONS=true
```

- Populate remaining safe real actions:
  - `check_ollama_models`
  - `load_embedding_model`
  - `unload_ollama_model`
  - `run_embedding_jobs`
- Populate remaining HITL/destructive actions if you really want them:
  - `reboot_server`
  - `kill_process`
  - `delete_files`
- Restrict `delete_files` to explicitly allowed temp/work folders.
- Decide whether `wake_pc` should stay as direct Wake-on-LAN or also route
  through the curated action endpoint.

## Crisis Manager

Status: minimal preflight/recovery agent implemented, default off.

Implemented:

- Enabled only by:

```text
ALPHARAVIS_ENABLE_MODEL_MANAGEMENT=true
ALPHARAVIS_ENABLE_ADVANCED_MODEL_MANAGEMENT=true
ALPHARAVIS_ENABLE_OWNER_POWER_TOOLS=true
ALPHARAVIS_ENABLE_CRISIS_MANAGER=true
ALPHARAVIS_CRISIS_MANAGER_MODEL=openai/edge-gemma
ALPHARAVIS_POWER_MANAGER_MODEL=openai/edge-gemma
```
- Use the small Ollama model only as a crisis moderator, not for normal complex
  work.
- Preflight check runs before the normal planner path.
- Automatically runs non-destructive checks and safe starts through owner tools:
  - status checks
  - logs/read-only probes
  - `start_llama_server`
  - `restart_llama_server`
- Sends the user a short `Crisis-Notice` while recovery is happening.
- After the recovery attempt, routes back to the normal planner path so the
  original user request can continue.
- Destructive shutdown tools are not given to the crisis agent.

Still needed:

- Trigger crisis recovery on mid-run main-model failures such as timeout, 502,
  connection errors, or LiteLLM backend generation health failure.
- Add a post-recovery readiness gate before continuing to the planner.
- Add full hard caps from the ENV placeholders:
  - max recovery attempts
  - max wall-clock time
  - no recursive crisis loops
- Add read-only Ollama/LiteLLM checks:
  - `check_ollama_models`
  - LiteLLM generation smoke status

ENV placeholders already exist:

```text
ALPHARAVIS_CRISIS_AUTO_ACTIONS=check_llama_server|check_ollama_models|check_comfyui|start_llama_server|restart_llama_server|wake_pc
ALPHARAVIS_CRISIS_HITL_ACTIONS=shutdown_server|reboot_server|kill_process|delete_files
ALPHARAVIS_CRISIS_MAX_ATTEMPTS=1
ALPHARAVIS_CRISIS_TIMEOUT_SECONDS=120
```

## Embedding Queue And pgvector

Status: pgvector retrieval chunks, catalog rows, durable queueing, a manual
model-lifecycle queue runner, optional scheduler, and bounded Store-index
backfill queueing are implemented.

Implemented:

- `ALPHARAVIS_PGVECTOR_INDEX_MODE=queue` stores new indexing work in Postgres.
- `alpharavis_embedding_jobs` keeps pending/failed/running/done queue state.
- `inspect_model_management_status` shows queue status.
- `run_embedding_memory_jobs` loads the configured Ollama embedding model when
  allowed and drains queued pgvector jobs.
- The runner may work while big-boss is active, so the small Ollama node can be
  used for embeddings without taking over complex chat.
- `ALPHARAVIS_ENABLE_EMBEDDING_SCHEDULER=true` drains the queue periodically.
- `queue_vector_memory_backfill` queues bounded old Store records by query.
- `ALPHARAVIS_ENABLE_VECTOR_BACKFILL_DAEMON=true` can repeat that bounded
  backfill search, but only when `ALPHARAVIS_VECTOR_BACKFILL_QUERY` is set.

Still needed:

- More precise convenience backfill commands:
  - index this exact thread without a search query
  - index last N artifacts by timestamp
  - index selected document/source keys from the external RAG backend
- Active-job awareness for Pixelle/MCP jobs beyond the current big-LLM/Ollama
  model probes.

## Media / Vision Memory

Status: safe media metadata handling, media-gallery service, and a separate
vision pgvector table are implemented. Full image/video understanding is still
provider/pipeline work.

Implemented:

- Bridge strips raw media blocks from chat context by default and preserves
  metadata markers.
- `media-gallery` can register/download image, video, audio, or document URLs
  and exposes `/gallery`.
- Pixelle job results are scanned for media URLs and registered when present.
- `register_media_asset`, `semantic_media_search`, and `plan_media_analysis`
  tools exist.
- Vision/media embeddings use `alpharavis_media_vectors`, separate from the text
  table, so vector dimensions do not collide.

Still needed:

- Connect a real vision embedding endpoint and enable:

```text
ALPHARAVIS_ENABLE_VISION_VECTOR_MEMORY=true
```

- Build video analysis:
  - stable URL/file mapping from LibreChat/Mongo attachments
  - keyframe extraction
  - scene/timecode grouping
  - optional audio transcription
  - frame captions
  - frame-level vision embeddings
- Build image analysis:
  - captioning
  - OCR
  - explicit user-triggered vision analysis
- Add exact mapping from LibreChat upload ids to gallery assets if LibreChat
  stores the file only inside its Mongo/filesystem layer.

## OpenWebUI

Status: optional Compose profile exists and points to the AlphaRavis Bridge.

Still needed:

- Start and verify:

```text
docker compose --profile openwebui up -d openwebui
make openwebui-smoke
```

- In OpenWebUI UI, set capable AlphaRavis models to Native function calling.
- Configure SearXNG or another web-search backend before enabling web search.
- Decide whether passthrough should stay enabled in your deployment:

```text
OPENWEBUI_ENABLE_OPENAI_API_PASSTHROUGH=true
```

## Lazy Tool Loading

Status: category registry exists and agents can inspect it with
`describe_optional_tool_registry(category=...)`.

Still needed:

- True per-run dynamic internal tool binding/unbinding.
- Cache concrete MCP tool schemas by category and only expose loaded subsets.
- Store loaded tool-set metadata in `run_profile`.

Clarification:

- `ALPHARAVIS_PGVECTOR_INDEX_MODE=background` still exists for best-effort
  async indexing, but the default example now uses `queue`.
- The model lifecycle runner can load the embedding model and drain jobs. It
  does not unload the small chat/crisis model by default. If that model is
  already loaded, the runner skips unless `ALPHARAVIS_EMBEDDING_UNLOAD_CHAT_MODEL=true`.

## Pixelle / ComfyUI Power Flow

Status: preflight hook exists, default off.

Implemented:

- Pixelle can run with durable `@task` monitoring or async job id polling.
- ComfyUI preflight can warn or block before Pixelle starts.
- The generic model-management preflight can request `wake_pc` through the
  curated action endpoint when that endpoint is configured.
- Owner power tools include a direct ComfyUI wake helper for manual/power-agent
  use.

Still needed:

- Set a real ComfyUI health URL:

```text
ALPHARAVIS_COMFY_HEALTH_URL=http://<comfy-ip>:8188/system_stats
```

- Decide whether Pixelle should warn-and-continue or block when ComfyUI is
  offline:

```text
ALPHARAVIS_PIXELLE_BLOCK_IF_COMFY_OFFLINE=false
```

- Decide whether Pixelle preflight should call the owner ComfyUI wake helper
  directly or stay routed through the curated action endpoint.

## Bridge

Status: Chat Completions remains compatible; Responses API wrapper,
Responses-style streaming events, direct Responses calls, and DeepAgents
Responses model binding exist.

Implemented:

- `/v1/chat/completions`
- `/v1/responses`
- OpenAPI schema version `3.1.0`
- `response.output_item.*`, `response.output_text.*`, and optional
  `response.reasoning_text.*` stream events
- bridge-level hard request cutoff before LangGraph is called
- direct no-tool LangGraph calls can use `/v1/responses` with:

```text
ALPHARAVIS_LLM_API_MODE=responses
```
- DeepAgents workers can bind tools through Responses with:

```text
ALPHARAVIS_DEEPAGENTS_API_MODE=responses
```

Still needed:

- Test whether LibreChat preserves `reasoning_content` as a separate reasoning
  panel or shows it as normal text.
- Keep `BRIDGE_STREAM_REASONING_EVENTS=false` until verified.
- Live smoke-test DeepAgents tool calls against the actual llama.cpp Responses
  backend and keep ChatLiteLLM fallback enabled until that passes repeatedly.

## Parallel Agent Work

Status: planned, not active.

Still needed:

- Extend planner output with dependency groups:
  - independent tasks may run in parallel
  - dependent tasks stay sequential
- Add a bounded parallel execution node or worker pattern.
- Require each parallel branch to produce a `build_specialist_report`.
- Merge reports into one final handoff packet.
- Keep tool conflict rules so two agents do not edit or control the same target
  at the same time.

## DeepAgents / Hermes Skills

Status: skill cards exist.

Still needed:

- Use the DeepAgents and Hermes skill cards as templates when adding new agents.
- Extract more stable reusable skills from completed workflows.
- Keep promotion manual through the existing skill-library review flow.
