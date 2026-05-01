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
- `/v1/responses/{response_id}`
- `/v1/responses/{response_id}/input_items`
- `/v1/responses/{response_id}/cancel`
- `/v1/responses/input_tokens`
- explicit unsupported response for `/v1/responses/compact`
- OpenAPI schema version `3.1.0`
- `response.output_item.*`, `response.output_text.*`, and optional
  `response.reasoning_text.*` stream events
- local `previous_response_id` continuation through `BRIDGE_RESPONSES_STORE`
- explicit errors for unsupported hosted Responses features instead of silent
  fake support
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

## Hermes Deep-Code Followups

Status: reference analysis done against the local Hermes Agent checkout at
`C:\experi\ai\hermes-agent`. These are adoption candidates only; Hermes should
remain a reference and optional external agent, not a runtime dependency for
AlphaRavis.

Already adopted or partly adopted:

- `agent/context_compressor.py`, `agent/model_metadata.py`, and `agent/redact.py`
  inspired AlphaRavis's active compression hardening:
  - JSON-safe tool-call argument truncation
  - tool-output pruning and duplicate-output backreferences for summary prompts
  - anti-thrashing via `compression_stats`
  - failure cooldown
  - image/tool-argument-aware token estimation
  - percentage-based context-length triggers with local model context discovery
- `agent/context_engine.py` inspired the lightweight AlphaRavis
  `compression_stats` state. A full plugin-style context engine is not needed
  yet because AlphaRavis compression also writes archives and pgvector records.
- Hermes skill ideas are represented by reviewed repo skill cards under
  `ai-skills/`, plus the Store-backed skill-library candidate flow.

Implementation chunks:

Do not implement the whole Hermes followup list in one pass. Work in these
chunks so every step stays testable and can be disabled independently.

### Chunk 1: Context Hygiene First

Status: implemented.

Implemented files:

```text
langgraph-app/internal_context.py
langgraph-app/context_references.py
langgraph-app/bridge_server.py
langgraph-app/agent_graph.py
tests/test_context_hygiene.py
```

AlphaRavis-specific integration:

- The scrubber is attached at the bridge output layer, so LibreChat receives
  clean visible text while LangGraph Studio can still inspect internal state.
- Context references are resolved relative to the AI-stack repo root by default,
  not relative to an arbitrary process directory.
- Reference metadata is passed into LangGraph state as
  `bridge_context_references` and copied into `run_profile`.
- Sensitive paths such as `.env`, `.ssh`, `.aws`, `.kube`, and `.docker` are
  refused before file content is attached.
- URL reference fetching is controlled independently by
  `BRIDGE_CONTEXT_REFERENCES_FETCH_URLS`.

Goal:

- Prevent accidental context leaks into LibreChat.
- Add explicit context-reference handling without dumping uncontrolled files.

Scope:

- Implement the streaming internal-context scrubber from Hermes
  `agent/memory_manager.py`.
- Add tests where `<memory-context>` or archive/internal tags are split across
  multiple SSE deltas.
- Implement a minimal AlphaRavis context-reference preprocessor inspired by
  Hermes `agent/context_references.py`:
  - `@file`
  - `@folder`
  - `@diff`
  - `@staged`
  - `@git`
  - `@url`
- Add context-budget protection and `allowed_root` path checks.
- Record warnings/refusals in `run_profile`.

Acceptance:

- LibreChat never receives hidden memory/internal blocks as normal assistant
  text, even when streaming chunks split the tag boundaries.
- Explicit references attach bounded context blocks.
- Oversized references warn or refuse cleanly instead of silently filling the
  prompt.

### Chunk 2: Error Router And Recovery Decisions

Status: implemented as compact AlphaRavis-local classifier.

Implemented files:

```text
langgraph-app/error_classifier.py
langgraph-app/responses_client.py
langgraph-app/bridge_server.py
langgraph-app/agent_graph.py
tests/test_error_classifier.py
```

AlphaRavis-specific integration:

- Responses direct calls now raise `AlphaRavisAPIError` with a structured
  classification instead of plain `RuntimeError` for HTTP/transport failures.
- The bridge formats visible errors by class, for example `context_overflow`,
  `timeout`, `server_error`, `overloaded`, `rate_limit`, and `format_error`.
- When activity events are enabled, the bridge can emit a short classified
  error status event before the visible error message.
- Planner, fast-path fallback, crisis preflight, and crisis-manager failures
  record classification metadata in `run_profile`.
- The classifier is intentionally compact; it does not import Hermes or bring in
  cloud-provider billing/credential rotation behavior.

Goal:

- Stop treating every backend issue as the same failure.

Scope:

- Port a compact AlphaRavis-local classifier from Hermes
  `agent/error_classifier.py`.
- Wire it into:
  - `responses_client.py`
  - bridge non-streaming/streaming errors
  - graph crisis/preflight metadata
- Map decisions:
  - `context_overflow` -> compression/hard-cutoff message
  - timeout/502/overloaded/connection -> crisis-manager candidate
  - rate limit/server busy -> retry/backoff or visible status
  - format/unsupported parameter -> Responses/Chat fallback or parameter strip

Acceptance:

- `run_profile` shows the classified reason.
- Advanced model-management recovery can use the classification later.
- Normal users get a useful message instead of a generic backend crash.

### Chunk 3: Central File Safety

Status: implemented as shared AlphaRavis-local file safety guard.

Implemented files:

```text
langgraph-app/file_safety.py
langgraph-app/context_references.py
langgraph-app/agent_graph.py
langgraph-app/media_server.py
tests/test_file_safety.py
```

AlphaRavis-specific integration:

- `file_safety.py` centralizes read/list/write/delete checks for sensitive
  credential/config paths, internal caches, shell profiles, and OS/system paths.
- `BRIDGE_ENABLE_CONTEXT_REFERENCES` file/folder reads now call the central
  read/list guard instead of carrying separate safety rules.
- `read_alpha_ravis_architecture`, `read_repo_ai_skill`,
  `write_alpha_ravis_artifact`, and `read_alpha_ravis_artifact` now pass through
  the same guard.
- Media gallery downloads verify the target path before writing under
  `ALPHARAVIS_MEDIA_ROOT`.
- `ALPHARAVIS_WRITE_SAFE_ROOT` can optionally force AlphaRavis write/delete
  helpers under a single owner-approved root.

Goal:

- Future coding/file/power tools share one safety policy.

Scope:

- Add `langgraph-app/file_safety.py`, inspired by Hermes
  `agent/file_safety.py`.
- Protect sensitive paths:
  - `.ssh`
  - `.aws`
  - `.kube`
  - `.docker`
  - `.env`
  - shell profiles
  - credential files
  - OS/system paths
- Add optional:

```text
ALPHARAVIS_WRITE_SAFE_ROOT=
```

- Make owner/coding/Hermes delegation tools call this module before destructive
  file operations.

Acceptance:

- Sensitive writes are blocked before tool execution.
- Reads that could expose internal caches or secrets return a safe refusal.
- Destructive actions still require HITL where already configured.

### Chunk 4: Skill Evolution And Self-Crystallizing Workflows

Status: implemented as safe repo skill manifest/cache plus review-only draft export.

Implemented files:

```text
langgraph-app/repo_skills.py
langgraph-app/agent_graph.py
tests/test_repo_skills.py
.env(exaple)
docs/ALPHARAVIS_ARCHITECTURE.md
docs/ALPHARAVIS_USAGE_NOTES.md
```

AlphaRavis-specific integration:

- `repo_skills.py` adds a Hermes-style mtime/size manifest cache for reviewed
  `ai-skills/` cards and their supporting folders.
- `reload_repo_ai_skills` reports added/removed/changed/unchanged disk skill
  status without changing Mongo skill candidate promotion state.
- `read_repo_ai_skill` can now read safe supporting files under `references/`,
  `templates/`, `scripts/`, and `assets/` in addition to `SKILL.md`.
- `export_skill_candidate_to_repo_draft` can write review-only drafts under
  `ai-skills/_drafts/<slug>/SKILL.md` when
  `ALPHARAVIS_ALLOW_SKILL_DRAFT_EXPORT=true`; candidates stay inactive.
- Normal repo skill hints still contain only compact metadata and never inject
  full skills into every run.

Goal:

- Keep AlphaRavis's safe candidate-review model, while borrowing Hermes's better
  disk-skill ergonomics.

Current AlphaRavis behavior:

- `record_skill_candidate` stores reusable workflows in Mongo/LangGraph Store as
  inactive candidates.
- `activate_skill_candidate` and `deactivate_skill` only work when:

```text
ALPHARAVIS_ALLOW_SKILL_PROMOTION=true
```

- Reviewed repo skill cards live under `ai-skills/`.
- The graph injects only small repo-skill metadata hints; full `SKILL.md` content
  is loaded only through `read_repo_ai_skill`.

Hermes behavior to learn from:

- Disk skills are first-class `SKILL.md` files.
- `prompt_builder.py` caches a skill manifest based on `SKILL.md` and
  `DESCRIPTION.md` mtime/size.
- `skill_commands.py` can reload skills and return added/removed/unchanged
  status.
- Loaded skills include supporting folders such as `references`, `templates`,
  `scripts`, and `assets`.
- Hermes encourages saving difficult repeated workflows as skills, but the
  AlphaRavis version must still keep promotion/manual review.

Scope:

- Add a repo-skill manifest cache for `ai-skills/`.
- Add a `reload_repo_ai_skills` or status tool that reports changes without
  changing promotion state.
- Add an optional exporter from reviewed Store skill candidate to a draft
  `ai-skills/<slug>/SKILL.md`, default off and review-only.
- Keep auto-created skills inactive until human review.
- Add better skill metadata conditions later:
  - required tool categories
  - fallback-only skills
  - platform compatibility

Acceptance:

- AlphaRavis can crystallize workflows into candidates automatically.
- It does not silently make a candidate active.
- Reviewed disk skills become faster and more ergonomic to use.

### Chunk 4.5: Operational Logging And Dependency Trace Files

Status: implemented as local rotating operational/debug log files.

Implemented files:

```text
langgraph-app/operational_logging.py
langgraph-app/agent_graph.py
langgraph-app/bridge_server.py
tests/test_operational_logging.py
.env(exaple)
docker-compose.yml
.gitignore
docs/ALPHARAVIS_ARCHITECTURE.md
docs/ALPHARAVIS_USAGE_NOTES.md
```

AlphaRavis-specific integration:

- Operational logs default to `logs/operational/alpharavis.log` and
  `logs/operational/alpharavis.jsonl`.
- The optional all-debug logger writes to `logs/debug/` only when
  `ALPHARAVIS_DEBUG_ALL_LOGGING=true`.
- Both loggers use daily rotation and keep `ALPHARAVIS_LOG_RETENTION_DAYS`
  backups, default 4 days.
- Logs include timestamps, severity, component, event, dependency, thread/run
  hints, duration, status, and redacted error data.
- `agent_graph.py` logs run start/finish, route decisions, LLM call
  duration/failure, Pixelle/ComfyUI preflight/job status, and semantic memory
  search results.
- `bridge_server.py` logs OpenAI-compatible bridge requests, Responses/Chat
  start/completion, LangGraph stream/wait failures, and LLM health probes.
- Docker mounts `./logs` to `/logs` for both `langgraph-api` and `api-bridge`.

Goal:

- Have local, time-correlated operational evidence even without LangSmith.
- Keep normal logs compact and enable a separate all-debug mode only while
  diagnosing issues.

### Chunk 5: True Lazy Toolsets

Status: implemented for static graph compile-time bundles and MCP category
filtering. Full per-node runtime rebinding remains future work if LangGraph
tool binding becomes hot-swappable.

Goal:

- Move from "the model sees a manifest" to actual bounded tool binding.

Scope:

- Use Hermes `toolsets.py` as the design reference.
- Define composable AlphaRavis toolsets:
  - `coding/read`
  - `coding/write`
  - `coding/execute`
  - `media/image`
  - `media/video`
  - `rag/documents`
  - `rag/memory`
  - `system/docker`
  - `system/ssh`
  - `system/power`
- Keep high-level categories visible.
- Bind concrete tools only after planner/agent chooses the set.
- Cache MCP schemas by category.

Acceptance:

- Done: `run_profile` records selected toolsets and loaded per-agent toolset
  profiles.
- Done: toolset includes detect cycles and cannot recurse forever.
- Done: MCP schemas are cached by category and only matching MCP tools are
  attached to the specialist bundles.
- Done: fast/simple chats still do not pay MCP/tool context cost.

### Chunk 6: Optional Usage, Pricing, And Rate-Limit Telemetry

Goal:

- Capture useful usage/rate-limit metadata without forcing cloud-style pricing
  into a local setup.

Default:

```text
ALPHARAVIS_ENABLE_USAGE_TELEMETRY=false
ALPHARAVIS_ENABLE_COST_ESTIMATION=false
ALPHARAVIS_SHOW_RATE_LIMITS=false
```

Reason:

- Your normal setup is local llama.cpp/Ollama, so cost estimation is not needed
  for daily use.
- Token/usage telemetry can still be useful for compression triggers and
  debugging when enabled.

Scope:

- Borrow only the useful parts from Hermes:
  - `usage_pricing.py` for normalized usage shape
  - `rate_limit_tracker.py` for `x-ratelimit-*` headers
- Mark local models as `local/included`, not paid.
- Add a future Make helper:

```text
make telemetry
```

or include it under `make configure`:

```text
Enable usage telemetry? [y/N]
Enable cost estimation for hosted APIs? [y/N]
Show rate-limit headers? [y/N]
```

Acceptance:

- All telemetry is off by default.
- Compression can use real API usage when present.
- Pricing output never appears unless explicitly enabled.

### Chunk 7: Prompt Assembly And Provider Hardening

Status: implemented for stable prompt context, head/tail context-reference
truncation, and direct Responses compatibility retries. Chat fallback for
DeepAgents remains controlled by the existing `ALPHARAVIS_DEEPAGENTS_API_MODE`
and `ALPHARAVIS_DEEPAGENTS_REQUIRE_RESPONSES` switches.

Goal:

- Make prompt assembly and provider fallback more robust without a huge provider
  rewrite.

Scope:

- Add WSL/Windows environment hints from Hermes `prompt_builder.py`.
- Separate stable prompt material from ephemeral task/memory/skill context.
- Improve head/tail truncation of loaded context files.
- Borrow selected provider-hardening ideas from Hermes `auxiliary_client.py`:
  - unsupported parameter retry
  - model-specific token/temperature quirks
  - safe Chat fallback when Responses tool-calling is broken

Acceptance:

- Done: no provider adapter became a hard dependency.
- Done: LiteLLM remains the default abstraction.
- Done: Responses remains preferred where it is stable.
- Done: direct Responses calls retry once after unsupported parameter errors.

### Chunk 8: Maintenance And Metadata Helpers

Goal:

- Improve long-term quality after the main runtime path is stable.

Scope:

- Offline archive/trajectory compression evaluator from Hermes
  `trajectory_compressor.py`.
- Optional shell hooks/approval allowlists from `shell_hooks.py`.
- Thread/archive title helper from `title_generator.py`.
- Candidate insight extraction from `insights.py`, review-only.

Acceptance:

- These are maintenance/admin helpers, not mandatory runtime features.
- Nothing here should affect normal LibreChat use unless enabled.

High priority:

1. Context reference preprocessor.

   Status: implemented in Chunk 1. Future refinement can add richer URL
   extraction or browser/VPN-backed fetching, but the safe bridge-side
   preprocessor is wired.

   Reference:

   ```text
   C:\experi\ai\hermes-agent\agent\context_references.py
   parse_context_references
   preprocess_context_references
   _expand_file_reference
   _expand_folder_reference
   _expand_git_reference
   _fetch_url_content
   _resolve_path
   ```

   AlphaRavis target:

   ```text
   langgraph-app/bridge_server.py
   langgraph-app/agent_graph.py
   ```

   Needed behavior:

   - Support explicit `@file`, `@folder`, `@diff`, `@staged`, `@git`, and `@url`
     references before planning.
   - Resolve paths relative to the repo/workspace and keep an `allowed_root`
     guard so references cannot silently escape the intended workspace.
   - Use context budget thresholds similar to Hermes:
     - soft warning around 25 percent of context
     - hard refusal around 50 percent of context
   - Attach files/folders/diffs as explicit context blocks rather than letting
     LibreChat full-history sync or prompt text dump arbitrary data.
   - Record reference warnings in `run_profile`.

2. Streaming internal-context scrubber.

   Status: implemented in Chunk 1 for bridge visible output and Responses
   wrapper streams.

   Reference:

   ```text
   C:\experi\ai\hermes-agent\agent\memory_manager.py
   sanitize_context
   StreamingContextScrubber
   build_memory_context_block
   ```

   AlphaRavis target:

   ```text
   langgraph-app/bridge_server.py
   langgraph-app/agent_graph.py
   ```

   Needed behavior:

   - Keep `<memory-context>...</memory-context>` and similar internal context
     blocks from leaking into LibreChat visible output.
   - Handle SSE chunk boundaries. A simple one-shot regex is not enough because
     opening and closing tags may arrive in different deltas.
   - Keep memory/context visible in Deep Agent/LangGraph debugging where useful,
     but scrub it from normal assistant text unless explicitly requested.

3. API error classification router.

   Status: implemented in Chunk 2 as `langgraph-app/error_classifier.py`.

   Reference:

   ```text
   C:\experi\ai\hermes-agent\agent\error_classifier.py
   FailoverReason
   ClassifiedError
   classify_api_error
   _classify_by_status
   _classify_by_error_code
   _classify_by_message
   ```

   AlphaRavis target:

   ```text
   langgraph-app/responses_client.py
   langgraph-app/bridge_server.py
   langgraph-app/agent_graph.py
   ```

   Needed behavior:

   - Classify `context_overflow` as compression/hard-cutoff work, not a generic
     backend crash.
   - Classify timeout, 502, overloaded, and connection failures as crisis-manager
     candidates when advanced model management is enabled.
   - Classify rate limits and temporary server errors as retry/backoff.
   - Classify format errors as Responses/Chat fallback or unsupported-parameter
     stripping.
   - Store the classified reason in `run_profile` and bridge status events.

4. Central file read/write safety.

   Reference:

   ```text
   C:\experi\ai\hermes-agent\agent\file_safety.py
   is_write_denied
   get_read_block_error
   get_safe_write_root
   ```

   AlphaRavis target:

   ```text
   langgraph-app/file_safety.py
   langgraph-app/owner_power_tools.py
   future file/coding tools
   ```

   Needed behavior:

   - Block writes to sensitive paths such as `.ssh`, `.aws`, `.kube`, `.docker`,
     `.env`, shell profiles, credential files, and system directories.
   - Add optional `ALPHARAVIS_WRITE_SAFE_ROOT`.
   - Block reads of internal cache/vector/secret files when those could become
     prompt-injection or credential leaks.
   - Make future Hermes/deep coding delegation obey the same safety policy.

Medium priority:

5. Disk skill index and manifest cache.

   Reference:

   ```text
   C:\experi\ai\hermes-agent\agent\prompt_builder.py
   _build_skills_manifest
   _load_skills_snapshot
   _write_skills_snapshot
   _parse_skill_file
   _skill_should_show
   build_skills_system_prompt

   C:\experi\ai\hermes-agent\agent\skill_commands.py
   _build_skill_message
   scan_skill_commands
   reload_skills
   build_skill_invocation_message
   ```

   AlphaRavis target:

   ```text
   ai-skills/
   langgraph-app/agent_graph.py
   docs/ALPHARAVIS_USAGE_NOTES.md
   ```

   Needed behavior:

   - Add a manifest cache for repo skills so full `SKILL.md` scans do not run
     every time.
   - Respect skill metadata such as required tools/toolsets, platform guards, and
     fallback-only behavior.
   - Include supporting folders (`references`, `templates`, `scripts`, `assets`)
     in the loaded skill message, with paths resolved relative to the skill
     directory.
   - Add a reload/status command or tool that reports added/removed/unchanged
     skills without auto-promoting Store skill candidates.

6. True lazy toolset resolver.

   Reference:

   ```text
   C:\experi\ai\hermes-agent\toolsets.py
   TOOLSETS
   get_toolset
   resolve_toolset
   resolve_multiple_toolsets
   get_all_toolsets
   validate_toolset
   ```

   AlphaRavis target:

   ```text
   langgraph-app/agent_graph.py
   OPTIONAL_TOOL_REGISTRY
   describe_optional_tool_registry
   ```

   Needed behavior:

   - Replace the current manifest-only approximation with composable toolsets
     such as `coding/read`, `coding/write`, `coding/execute`, `media/video`,
     `rag/memory`, `system/power`.
   - Keep category descriptions visible to the model, but bind concrete tools
     only after the planner or agent selects the category.
   - Cache MCP tool schemas per category.
   - Record selected and loaded toolsets in `run_profile`.
   - Prevent recursive/cyclic toolset includes.

7. Usage, cost, and rate-limit telemetry.

   Reference:

   ```text
   C:\experi\ai\hermes-agent\agent\usage_pricing.py
   CanonicalUsage
   normalize_usage
   estimate_usage_cost

   C:\experi\ai\hermes-agent\agent\rate_limit_tracker.py
   parse_rate_limit_headers
   format_rate_limit_display
   format_rate_limit_compact
   ```

   AlphaRavis target:

   ```text
   langgraph-app/responses_client.py
   langgraph-app/bridge_server.py
   langgraph-app/agent_graph.py
   run_profile
   ```

   Needed behavior:

   - Normalize usage across LiteLLM, llama.cpp, and future hosted providers.
   - Track input, output, reasoning, cache-read, and cache-write tokens.
   - Mark local llama.cpp/Ollama costs as local/included instead of fake money.
   - Parse `x-ratelimit-*` headers when present and show compact status in
     bridge/debug output.
   - Use real usage values for compression decisions whenever available.

8. Prompt assembly and context-file cache hygiene.

   Reference:

   ```text
   C:\experi\ai\hermes-agent\agent\prompt_builder.py
   build_environment_hints
   build_context_files_prompt
   _truncate_content

   C:\experi\ai\hermes-agent\agent\prompt_caching.py
   ```

   AlphaRavis target:

   ```text
   langgraph-app/agent_graph.py
   docs/ALPHARAVIS_ARCHITECTURE.md
   ```

   Needed behavior:

   - Separate stable system prompt material from ephemeral task, memory, skill,
     and handoff context.
   - Add WSL/Windows path hints when the workspace path indicates a mixed
     Windows/Linux environment.
   - Truncate context files by preserving useful head/tail regions and scan
     hints, not just naive first-N characters.
   - Keep stable prompt-cache candidates stable so future provider-side prompt
     caching can work better.

Lower priority / future:

9. Offline trajectory/archive compression evaluator.

   Reference:

   ```text
   C:\experi\ai\hermes-agent\trajectory_compressor.py
   ```

   AlphaRavis target:

   ```text
   archive collections
   vector backfill tools
   maintenance scripts
   ```

   Needed behavior:

   - Batch-evaluate old thread/archive compression quality.
   - Track success/failure metrics for collection summaries.
   - Use it for maintenance/backfill, not the live chat path.

10. Shell hooks and approval allowlists.

    Reference:

    ```text
    C:\experi\ai\hermes-agent\agent\shell_hooks.py
    ```

    AlphaRavis target:

    ```text
    langgraph-app/owner_power_tools.py
    future terminal/file tools
    ```

    Needed behavior:

    - Optional pre/post hooks around shell/system actions.
    - Strict allowlist and audit trail.
    - No automatic destructive hook execution without HITL.

11. Provider adapter hardening.

    Reference:

    ```text
    C:\experi\ai\hermes-agent\agent\auxiliary_client.py
    C:\experi\ai\hermes-agent\agent\codex_responses_adapter.py
    C:\experi\ai\hermes-agent\agent\anthropic_adapter.py
    C:\experi\ai\hermes-agent\agent\gemini_native_adapter.py
    ```

    AlphaRavis target:

    ```text
    langgraph-app/responses_client.py
    langgraph-app/bridge_server.py
    ```

    Needed behavior:

    - Strip unsupported parameters and retry where safe.
    - Map model-specific max-output-token and temperature behavior.
    - Keep Chat Completions fallback for providers with broken Responses tools.
    - Add direct non-OpenAI providers only if LiteLLM is not enough.

12. Thread title and insight helpers.

    Reference:

    ```text
    C:\experi\ai\hermes-agent\agent\title_generator.py
    C:\experi\ai\hermes-agent\agent\insights.py
    ```

    AlphaRavis target:

    ```text
    archive titles
    archive collections
    LibreChat/bridge metadata
    curated memory review
    ```

    Needed behavior:

    - Generate short stable titles for archive records and archive collections.
    - Extract candidate user/system insights for review without auto-promoting
      them into always-memory.
    - Keep this separate from raw archives and pgvector source-of-truth rules.
