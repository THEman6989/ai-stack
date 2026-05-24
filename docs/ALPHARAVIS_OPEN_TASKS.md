# AlphaRavis Open Tasks

This is the running backlog for features that are intentionally prepared but
not fully wired yet.

## Deep Agents UI

Status: Implemented and second hardening pass applied. Build/lint verification
passes; live browser smoke test still needed.

Implemented:

- `submodules/deep-agents-ui` is the canonical forked UI submodule.
- Multimodal upload is wired in `ChatInterface`: file picker, drag/drop via
  `dropRef`, paste via `handlePaste`, content preview, processing state,
  success/error toasts, remove-all control, timestamped pasted-image filenames,
  and content blocks in `sendMessage`.
- Chat openers, hardened thread rename/delete (pending state, Escape cancel,
  duplicate rename guard, metadata title display, active-thread delete recovery),
  artifact rendering, file preview panel, lightweight diff rendering, on-demand
  Monaco editor, and skills indicator are present.
- Dependency hygiene: `monaco-editor` peer dependency explicit, `yarn.lock`
  regenerated, unused `diff` / `@types/diff` removed.
- Docker hygiene: `.dockerignore` keeps local `node_modules`/`.next` out of the
  build context, and the Dockerfile now uses `yarn install --frozen-lockfile`.
- Next.js build config hygiene: `tsconfig.json` is aligned with Next 16 so the
  production build does not need to mutate it in-container.
- UI integration template: `docs/ALPHARAVIS_UI_INTEGRATION_TEMPLATE.md` plus
  `.hermes/templates/alpha-ravis-ui-integration-template.md` pointer.

Still needed:

- Live browser smoke test on port 3000: file picker upload, drag/drop upload,
  paste upload, attachment remove/remove-all, preview panel, lightweight diff
  rendering, code preview before/after `Open Monaco editor`, and thread
  rename/delete including active-thread deletion recovery.
- Decide later whether to tackle AionUi Tier 3 items: i18n, inline tool-result
  streaming, and conversation tabs.

## Parallel Task Execution (Stage 2)

Status: Stage 2 implemented (executor, worker spawn, merge/review). Live
testing and Codex/Hermes adapters still needed.

Implemented:

- `ai_stack/parallel_executor/executor.py`: `ParallelExecutor` runs parallel
  groups concurrently via `asyncio.gather()`, then serial chain sequentially,
  then merge/review. `build_execution_plan()` converts `TaskDAG` to ordered
  `ExecutionPlan`.
- `ai_stack/parallel_executor/worker_spawner.py`: Added `DirectLLMWorker`
  that integrates with `_ainvoke_direct_text` via callable injection (avoids
  circular imports with agent_graph).
- `agent_graph.py`: Added `parallel_executor` graph node wired between
  `final_budget_rescue` and `swarm_trace_start`. When disabled, returns `{}`
  (complete no-op).
- `_parallel_executor_node` builds executor with DirectLLM worker, runs DAG,
  logs results, and appends result messages to state.
- 41 tests covering Stage 1 + Stage 2 (execution plan, dry-run executor,
  failing worker, merge/review, report serialization). All 91 tests pass.

Still needed:

- Codex CLI adapter and Hermes agent adapter for `WorkerSpawner` interface.
- Live test with real BigBoss calls through the DirectLLM worker.
- Profile parallel execution overhead and resource usage.
- DeepAgents Responses compatibility testing with parallel executor messages.

## Percentage-Based Context Budget Router

Status: Implemented. Live integration verification still needed.

- Module: `ai_stack/context_budget/router.py` with `DynamicServerState`,
  `PercentageBudgetPolicy`, `PriorityAwareRouter`.
- Scheduler integration: `ContextScheduler` now uses the router for dynamic
  max_tokens calculation.
- `LlamaCppRuntimeClient` now supports `/props` and `/metrics`.
- Secondary model context updated to 60k (`ALPHARAVIS_SMALL_MODEL_CONTEXT=60000`).

Still needed:

- Live-test the router with real `/slots` data from the primary llama-server.
- Verify that dynamic context detection works correctly after a server restart
  with different `--ctx-size`.
- Profile routing overhead in the hot path.

## Ubuntu Llama Manager Integration

Status: LangGraph tool surface implemented; live manager validation still
needed.

Implemented:

- AlphaRavis can be configured with `ALPHARAVIS_UBUNTU_LLAMA_MANAGER_URL` and
  `ALPHARAVIS_UBUNTU_LLAMA_MANAGER_API_KEY` to call the external
  `ubuntu-llama-manager` API. Operators can now enter
  `ALPHARAVIS_UBUNTU_LLAMA_MANAGER_IP` / `..._PORT` and
  `ALPHARAVIS_UBUNTU_LLAMA_ESP_IP` / `..._PORT`; full URL settings remain
  optional overrides.
- LangGraph exposes tools to inspect manager health/status/models/instances,
  diagnose a no-response llama.cpp server, start/stop/restart managed llama
  services, run gated ESP/server power actions, run gated recovery, and patch
  `primary` or `secondary` instance model/context/parallel/command settings.
- Real recovery/config writes are dry-runs unless
  `ALPHARAVIS_MODEL_MGMT_ALLOW_ACTIONS=true`.
- The power/model-management agent gets the full tool set. The crisis manager
  gets inspect, diagnose, service control, ESP power-cycle, and gated recovery,
  so it can restore the big backend without changing model context.
- LibreChat exposes a dedicated `Server Model Manager` preset using Bridge model
  id `server-model-manager`; the Bridge routes it to native LangGraph
  `power_management_agent` with `agent/power` selected. Native LangGraph callers
  can pass the same `active_agent` / `selected_toolsets` fields directly.
- The dedicated manager uses `ALPHARAVIS_SERVER_MODEL_MANAGER_MODEL` first
  (`openai/server-model-manager` by default), a LiteLLM route intended to
  prefer BigBoss and fall back to Edge Gemma.
- Destructive server/ESP tools require an extra `confirmed=true` after the
  agent states the exact target and tool; start/restart/power-on remain direct
  once actions are enabled.
- Crisis-manager enablement no longer depends on owner SSH tools when advanced
  model management is enabled; Ubuntu Llama Manager can provide the recovery
  surface.
- Model Manager now has real Ollama/queue actions:
  `check_ollama_models`, `load_embedding_model`, `unload_ollama_model`, and
  `run_embedding_jobs`.
- Mid-run crisis recovery catches recoverable provider failures, runs a bounded
  Crisis Manager attempt, waits for a readiness gate, and retries the swarm
  only after the primary backend is reported ready.
- Crisis recovery is capped by max attempts, max wall-clock, per-action timeout,
  and a recursive-loop guard.
- Focused unit coverage verifies dry-run gating, context bounds, and protected
  diagnose payloads.
- `configure_ubuntu_llama_instance` can set bounded llama.cpp parallel slots
  (`parallel_slots=2` maps to `--parallel 2`) for the small 2B or BigBoss
  instance when VRAM headroom is safe. The configured max is
  `ALPHARAVIS_UBUNTU_LLAMA_PARALLEL_MAX`.
- The AI-stack now has a separate dynamic context scheduler layer:
  `UbuntuLlamaManagerClient` is control-plane only, `LlamaCppRuntimeClient`
  talks directly to each selected llama-server for `/apply-template`,
  `/tokenize`, `/slots`, `/v1/models`, `/completion`, and
  `/v1/chat/completions`, and `ContextScheduler` reserves process-local
  context leases before async LLM calls when
  `ALPHARAVIS_CONTEXT_SCHEDULER_ENABLED` is active.
- A background task lane is available for small independent work. Read-only
  tools can run concurrently; small LLM side jobs still need direct
  llama-server token counting plus a ContextLease and are capped by
  `ALPHARAVIS_BACKGROUND_CONTEXT_MAX_UTILIZATION` so they do not starve the
  main agent. The memory-kernel curated-memory and semantic-memory prefetches
  now run through this read-only parallel path.
- Runtime config is discovered per instance from Manager response data and the
  saved command: `host`, `port`, `base_url`, `ctx_total`, `parallel`, and
  `kv_unified`. When Manager-reported localhost would be unreachable from
  AlphaRavis, the runtime URL is derived from the Manager host or
  `ALPHARAVIS_LLAMA_RUNTIME_HOST_OVERRIDE`.

Still needed:

- Configure the real manager URL/token in private `.env` and live-test:
  `inspect_ubuntu_llama_manager`,
  `diagnose_ubuntu_llama_no_response`, and a controlled dry-run/real
  `request_ubuntu_server_power_action`, `control_ubuntu_llama_service`, and
  `configure_ubuntu_llama_instance` context change.
- Decide later model-manager policy for automatic context escalation and
  rollback, including when the secondary 2B model should move between 8K and
  16K and when the primary can be raised toward 200K.
- Improve automatic parallel-slot policy: temporarily use `parallel_slots=2`
  for safe short-context windows, but restore `parallel_slots=1` before
  concurrent high-context BigBoss work can exhaust VRAM.
- Wire more optional side jobs, such as Planner, Router/Judge, RAG compression,
  summarization, and chunk ranking, into the background lane where their inputs
  are independent and cancellation is safe.
- Extend managed-run lifecycle tracking for the big llama host. The prompt and
  ENV policy now distinguish "already running" from "powered on for this
  request"; ComfyUI/Pixelle has concrete delayed shutdown scheduling, while
  BigBoss automatic post-run shutdown should be live-tested and made durable
  before enabling it by default.
- Add Manager API aliases if desired:
  `POST /llama/instances/{id}/restart` and
  `POST /llama/instances/{id}/stop`. The current AI-stack client intentionally
  uses the documented primary/secondary endpoints from helper `docs/api.md`.
- Promote context leases from process-local memory to Redis if
  multiple `langgraph-api` workers need a shared global budget.
  Implemented 2026-05-24: `RedisLeaseStore` with atomic Lua-script admission,
  lazy `redis.asyncio` import, TTL-based stale-lease expiry, and graceful
  fallback to `LocalLeaseStore`. Controlled by `ALPHARAVIS_CONTEXT_LEASE_BACKEND`
  (default `local`). See `ai_stack/context_budget/leases.py`.
- Wire the scheduler's "not enough context" decisions into automatic
  compression/RAG-chunk reduction/max-output lowering/retry routing. The
  scheduler currently returns a structured decision and blocks admission rather
  than blindly starting the call.
- Live-test direct runtime token counting against the real primary and
  secondary llama-server ports, including `/apply-template`, `/tokenize`,
  `/slots`, and `/v1/models`.
- Live-test the new Ollama model actions against the real host once the private
  `.env` points at the target Ollama instance.

## Operator Config UI

Status: implemented for root `.env` editing and dashboard runtime settings.

Implemented:

- `make config` starts a local browser UI backed by
  `scripts/alpharavis_config_server.py`.
- The UI uses `.env(exaple)` as the canonical default/template source, groups
  settings by its documented sections, and saves current values into `.env`.
- Boolean settings use True/False controls, secret-looking keys use password
  inputs, and URL-like values are directly editable.
- Each setting can reset to its documented default. Reset all is available in
  the bottom-right action bar and asks for confirmation before changing the
  in-browser values; Save persists the result to `.env`.
- `make install` and `make update` keep the existing terminal prompts as a
  fallback path, while `make config` is the intended central place for broader
  configuration changes.
- The Service Dashboard now exposes `/settings` as a modern mobile/PWA-friendly
  Settings WebUI and primary dashboard card.
- `/settings` loads all keys from `.env(exaple)`, shows current `.env` values
  plus active runtime overrides, and infers toggles, dropdowns, number fields,
  URL fields, and password fields from defaults/comments/key names.
- The Settings WebUI uses compact setting rows, quieter accents, equal-size
  temporary/permanent actions, local favorites, and generated fallback
  descriptions for undocumented `.env(exaple)` keys.
- Filters support search, category chips,
  important/normal/low/changed/runtime/favorite views, and sorting by
  importance, alphabet, section, or changed state.
- `Temporary anwenden` stores overrides in
  `service-dashboard-data/runtime_settings.json`; LangGraph reloads this file
  before each new run so new chat turns pick up temporary values without
  rewriting `.env`.
- `Permanent speichern` writes validated template keys to `.env` after browser
  confirmation, with a local "nicht mehr fragen" option.

Still needed:

- Done: Live browser polish pass on the owner machine for very narrow mobile-sized windows.
- Done: Tailscale HTTPS route was live-tested at
  `https://cachyos-comfyui.tail852b38.ts.net:8090/settings` with `200 OK`.
- Still recommended: open `/settings` from an iOS Safari home-screen shortcut
  and check the actual standalone PWA chrome/safe-area behavior on-device.

## Agent Run State And Resume

Status: first durable run-state layer implemented.

Implemented:

- `langgraph-app/run_state_manager.py` stores one latest Mongo-backed
  `current` checkpoint per thread with phase/status, task brief, planner
  context, selected toolsets, active agent, compact run profile, and provider
  error classification.
- The graph saves checkpoints at run start, planner success/failure, swarm
  start/failure/completion, and run finish. Completed runs are marked
  `completed`; interrupted provider/llama.cpp runs remain `awaiting_resume`.
- On a later message in the same thread, open checkpoints restore the plan/task
  brief and trigger a same-thread prompt asking whether to continue. Manual
  confirmation is the default; `ALPHARAVIS_RUN_STATE_AUTO_RESUME=true` enables
  automatic continuation.
- The Bridge Observer exposes an `Awaiting Resume` panel and
  `GET /api/resume-runs` endpoint for listing saved `awaiting_resume` runs with
  thread id/key, phase, task brief, active agent, selected toolsets, and the
  same-thread resume phrase.
- Focused verification: `pytest -q tests/test_run_state_manager.py` passed.

Still needed:

- Live-test a forced llama.cpp/LiteLLM disconnect during a long swarm run and
  confirm the user-facing resume prompt and `ja, weiter` path in LibreChat.

## Service Dashboard And Tailscale HTTPS

Status: dashboard implemented; Tailscale HTTPS helper wired for operator use;
the dashboard route is included by default.

Implemented:

- `service-dashboard` runs on `http://localhost:8090` as part of the base
  Docker Compose stack.
- `service_redirector_server.py` serves the dark service-card redirector plus
  `/services.json` and `/health`.
- `make up`, `make install`, and `make update` include the dashboard through
  the base Compose stack; `make service-dashboard` starts only the dashboard.
- `bridge-test-ui` remains included in base stack startup and has
  `make test-ui` for targeted startup.
- `tailscale_https_routes.py` can plan/apply Tailscale Serve HTTPS routes for
  local HTTP services inside the Tailnet and can write
  `tailscale_service_urls.json` redirector override data. It does not use
  Tailscale Funnel or public-internet exposure.
- Makefile targets exist:
  - `make tailscale-plan`
  - `make tailscale-overrides`
  - `make tailscale-apply`
  - `make tailscale-disable`
  - `make tailscale-status`
- Normal operator flows call Tailscale automatically:
  - `make install` and install profile targets
  - `make update`
  - `make update-no-start`
  - `make up`
  - `make up-fullstreaming`
  - `make up-chat-fullstreaming`
  Use `TAILSCALE_AUTO=off` to skip the automatic apply step for one run.
- Tailscale sudo mode defaults to `auto`: retry with sudo only after a
  permissions-style Tailscale CLI failure. Use `TAILSCALE_SUDO=true` to force
  sudo or `TAILSCALE_SUDO=never` to disable retry.
- `service_redirector_server.py` automatically prefers generated Tailscale
  HTTPS URLs from `service-dashboard-data/tailscale_service_urls.json` when
  `ALPHARAVIS_SERVICE_DASHBOARD_URL_MODE=auto`.
- `tailscale_https_routes.py` now includes the `service-dashboard` Tailnet HTTPS
  route by default for `plan`, `write-overrides`, `apply`, and `disable`.
  Operators can opt out with `--exclude-dashboard`,
  `ALPHARAVIS_TAILSCALE_INCLUDE_DASHBOARD=false`, or
  `make ... TAILSCALE_DASHBOARD=false`.
- Focused local verification completed on 2026-05-13:
  - `python tailscale_https_routes.py plan --tailscale-host test-device.tailnet.ts.net`
    includes `https://test-device.tailnet.ts.net:8090` for the dashboard.
  - `python tailscale_https_routes.py plan --tailscale-host test-device.tailnet.ts.net --exclude-dashboard`
    omits the dashboard.
  - `pytest -q tests/test_tailscale_https_routes.py` passes.

Implemented follow-up:

- Dashboard UX refinement completed:
  - `service_redirector_server.py` separates primary `Web Interfaces` from
    collapsible `APIs` and `Infrastructure` sections.
  - API/backend cards show copyable HTTPS, local HTTP, and Tailnet HTTP
    addresses when the Tailscale override payload contains a `tailscale_host`.
  - Cards are directly clickable again; the explicit action label remains
    visible as `Öffnen`.
  - LiteLLM is listed both as a Web UI (`http://localhost:4000`) and as an API
    endpoint (`http://localhost:4000/v1`).
  - Pixelle is listed as a Web UI on `http://localhost:9004`; the streamable
    HTTP MCP endpoint is listed separately as
    `http://localhost:9004/pixelle/mcp`.
  - LangGraph specialist visual ports (`8760`, `8762`, `8764`, `8766`, `8768`)
    are no longer primary Web Interface cards because they are experimental
    internal agent ports and may not serve a standalone browser UI. They are
    marked as TCP/internal infrastructure entries, so the dashboard does not
    treat them as click-to-open Web UIs or Tailscale HTTPS web routes.
  - Service cards now include consistent compact letter logos; Media Gallery is
    no longer an unbranded/yellow placeholder card.
  - The dashboard serves `/favicon.svg`.
- Media Gallery visual refresh completed:
  - `/gallery` now has a modern dark responsive layout, sticky desktop header,
    mobile-friendly filters, a visible `MG` brand mark, and `/favicon.svg`.
  - The 2026-05-22 follow-up made date sections the default, kept collapsible
    sections optional, removed technical file/source metadata from card bodies,
    and tuned the grid/cards for smaller iPhone-sized screens.
  - The filter Apply action no longer wraps below the first select on medium
    widths; it stays at the row end on desktop and becomes a full-width action
    below filters on mobile.
  - A native Upload control now posts to `/assets/upload`, stores selected
    browser/mobile files as original `gallery_upload` assets in `media-data`,
    records them in MongoDB, and returns to the date-sorted Gallery.
  - Asset cards keep image/video previews and copy/open actions; operator
    filters and alternate grouping modes remain available when needed.
- Tailscale path handling fixed:
  - `tailscale_https_routes.py` keeps the public click path, for example
    `/gallery`, but proxies Tailscale Serve to the service root
    `http://127.0.0.1:<port>` unless a service explicitly defines
    `tailscale_target_path`. This avoids sub-link "Not Found" issues caused by
    serving a whole UI under a path-bound upstream target.

Still needed:

- Live-test `tailscale serve --bg --https=<port>` from another allowed Tailnet
  device after Tailscale HTTPS certificates are enabled for the tailnet. This
  is now an operator/live-network validation, not an implementation blocker.
- Optional: replace the compact letter logos with custom SVG/PNG brand assets
  later if a full visual identity set is wanted.

## Responses Streaming Follow-up

Status: local PR #35457-style patch applied; hybrid streaming mode passes.

Implemented:

- LangGraph container packages were updated to:

```text
langchain-openai==1.2.1
langchain==1.2.18
langchain-core==1.3.3
langgraph==1.1.10
deepagents==0.5.9
openai==2.36.0
litellm==1.83.0
```

- Patched DeepAgents Responses hybrid mode works with:

```text
ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING=true
ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=tool_calling
```

- Live test after the update:
  - `/v1/responses` Fast Path returned 200.
  - `/v1/responses` Agent Path returned 200 in about 48 seconds.
  - External `/v1/responses` SSE streaming returned chunks.
- Live test after the local patch:
  - Direct `ChatOpenAI(use_responses_api=True, streaming=True,
    disable_streaming="tool_calling")` with a bound tool returned
    `DIRECT_TOOL_STREAM_TEST_OK`.
  - Bridge `/v1/responses` Agent Path streaming returned
    `PATCHED_AGENT_STREAM_OK`.
- Focused full-streaming probe after the env-gated tool-stream patch:
  - raw `/v1/responses` SSE included function-call events
  - LangChain no-tool Responses streaming passed
  - LangChain `create_react_agent` streaming executed `marker_tool` exactly once
  - `invalid_tool_calls=0`

Still needed:

- Track the upstream LangChain issue and remove the local patch when
  `langchain-openai` includes the fix.
- Retest the LiteLLM proxy after its Docker image reports a newer package than
  `litellm==1.82.6`.
- Keep full streaming
  `ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=false` experimental as a
  stack default until Bridge-level repeated smoke tests pass and the local
  provider consistently handles Responses tool-choice behavior.
- Consider a two-phase final-answer stream:
  - tool-capable worker calls stay hybrid/non-streaming for reliable tool JSON
  - after tool execution, run a final answer call without tools and stream that
    text to the client
  - this requires orchestration changes because current DeepAgents workers keep
    tools bound on final answer turns too
- Profile and reduce normal agent-path latency. Live Docker measurements on
  2026-05-12 showed:
  - direct LiteLLM chat call: about `1.8 s`
  - Bridge fast path: about `2.3 s`
  - Bridge `/v1/responses` agent path with `kein fast path`: about `54 s`
  - the same run spent about `29.8 s` in the planner call, then continued into
    the worker/model stage
  - queued runs can add large apparent latency because the current local
    LangGraph runtime reported only one active background worker
  Implemented diagnostic aid: `bridge-test-ui` now includes a request
  waterfall trace, and the Bridge propagates a `trace_id` into LangGraph input
  for non-streaming Responses/Chat Completions runs. Follow-up tracing showed
  the 50s agent-path case was dominated by planner output (~28s and 62k chars)
  plus slow semantic memory prefetch (~20s). Planner calls are now capped and
  memory prefetch has a default 4s per-step timeout; a Docker smoke then reduced
  the same forced agent-path probe to about `11.2 s`.
  The Bridge Test UI now uses a streaming proxy and browser-side SSE parsing by
  default, so it no longer buffers streamed Bridge responses into one JSON
  result. It is also part of the normal `make up`, `make install`, and
  `make update` stack flow; explicit `make build` and streaming recreate targets
  include `bridge-test-ui` too. Follow-up: measure true visible text first-token
  latency separately from lifecycle/activity SSE latency, add equivalent
  Bridge/LangGraph trace metadata for streaming SSE, then decide whether to
  shorten/bypass planner work for trivial prompts, increase local worker
  concurrency where safe, or route simple UI greetings through the fast path
  earlier.
- Bridge Observer implemented:
  - `api-bridge` keeps an in-memory ring buffer of recent Bridge observations
    for real LibreChat and Test UI traffic.
  - `bridge-test-ui` exposes `/observer` as a full-page table view with
    `Senden` and `Empfang` tabs plus `Nur Kontext` / `Vollansicht` modes.
  - The send-side context view shows the raw incoming messages, derived
    `thread_key` / `thread_id`, and the exact `model_context_messages` payload
    prepared for LangGraph.
  - The receive-side view shows output/reasoning/status data captured by the
    Bridge.
- Critical context/threading follow-up:
  - Investigated: LibreChat's observed chat-completions payload did not include
    `conversationId` / `conversation_id`; it sent `user` as
    `69ee2b264b635fe48c9913b5`. The Bridge was using `body.user` as the
    LangGraph `thread_key`, so separate visible LibreChat chats could share the
    same persistent LangGraph thread. The failing request's prepared model
    context was only one `hi` message, but the reused LangGraph thread already
    contained dozens of old messages plus stored reasoning/thinking blocks. The
    Bridge now defaults to ephemeral threads unless an explicit conversation
    id/header is present, and Observer records the existing LangGraph state
    profile beside the prepared model context.
  - Implemented: `pre_run_context_guard` now runs before `route_decision`, so
    an old explicit thread can compact old context before the hard cutoff. If a
    thread is already above the hard limit and normal compression fails or
    remains too large, hard trim removes old active messages while preserving
    the latest user turn and records the result in `run_profile`.
  - Implemented: active-context token estimates now ignore UI
    reasoning/thinking blocks and provider usage metadata in the graph,
    compressor, and model metadata estimator, because those are not model input
    context. Verified with a synthetic LangGraph runtime probe that forces
    compression failure under a tiny hard limit: old messages are removed and
    the latest user message remains active. Remaining live verification:
    exercise a real long explicit thread with LibreChat once enough old state is
    available and confirm the visible request is rescued rather than refused.

## RAG / Retrieval Router

Status: hybrid retrieval router foundation implemented; explicit Agentic-RAG
tool exposed; thread activation metadata and large-paste context replacement
implemented; native AlphaRavis pgvector is now the default document/large-paste
backend; LibreChat document upload auto-ingest has a first guarded bridge path;
live large-paste latency still needs runtime tuning.

Implemented:

- `retrieval_router.py` centralizes source ingest selection and scoped retrieval
  over AlphaRavis pgvector plus optional `rag_api`.
- Compression archives remain owned by AlphaRavis and are indexed in pgvector by
  default; `rag_api` archive mirroring stays default-off unless explicitly
  enabled.
- External document / large-ingest style sources route toward AlphaRavis
  pgvector by default through `ingest_source(...)`.
- `query_source`, `query_sources`, and `query_archive` search known source keys
  without loading full archives.
- `agentic_rag_retrieve` is exposed from `agent_graph.py` as an explicit tool.
  It runs retrieve, deterministic grade, optional query rewrite, retry, and
  returns a bounded `context_packet` plus `graph_trace`.
- `ingest_source(...)` returns thread-aware RAG activation metadata. External
  documents and large-paste style sources set `rag_active=true` with
  `active_source_keys`, optional `active_rag_file_ids`, and
  `rag_activation_reason=document_ingest|large_paste`. Compression archives set
  `rag_active=false` and `archive_rag_mode=tool_only`; Agent-path
  archive-intent checking is now default-on separately through
  `ALPHARAVIS_ARCHIVE_AUTO_ON_INTENT_AGENT_DEFAULT=true`, while
  `archive_rag_mode=manual` remains the strict tool-only override.
- `AlphaRavisState` can carry `rag_active`, `active_rag_file_ids`,
  `active_source_keys`, `rag_activation_reason`, and `archive_rag_mode`.
  Run-profile snapshots expose these fields for observer/debugging.
- Large human paste messages are detected at `run_profile_start_node`, but
  automatic paste-to-RAG now runs after pre-run compression by default. Plain
  large pastes are deferred while old context is compressed first; after that,
  `large_paste_post_compression_node` re-estimates the active request and only
  indexes/replaces the paste when it still consumes at least
  `ALPHARAVIS_LARGE_PASTE_RAG_POST_COMPRESSION_TRIGGER_RATIO` of the active
  compression budget, default `0.80`. Manual `/rag ... /rag` blocks still force
  source indexing immediately. After a RAG/source replacement, the same node can
  run a follow-up compression pass for remaining non-document chatter when the
  active context still exceeds
  `ALPHARAVIS_LARGE_PASTE_POST_RAG_COMPRESSION_TRIGGER_RATIO`, default `0.80`.
- Large-paste intent classification is implemented without an extra model call
  in the hot path. Document/unknown pastes keep the existing document-RAG
  behavior. Instruction-like pastes are indexed as `large_instruction` for exact
  lookup but do not automatically activate document RAG; the active context gets
  a condensed instruction brief. Mixed instruction+document pastes keep active
  RAG for the document/data parts, preserve a condensed instruction brief in
  the replacement marker, and strip obvious instruction text from the indexed
  document body when a document section can be separated.
- Large-paste ingest decisions now record an Observer-visible run-profile event
  timeline: `large_ingest.started`, `large_ingest.completed`,
  `large_ingest.failed`, or `large_ingest.skipped`, including elapsed time,
  status, backend, and skip/failure metadata where available.
- Direct pgvector document/large-paste indexing can now emit
  `large_ingest.chunk_indexed` / `document_ingest.chunk_indexed` progress
  events into the run profile. The Bridge streaming path turns those run-profile
  ingest events into status/reasoning activity lines when activity streaming is
  enabled.
- The durable embedding queue can reclaim stale `running` rows after
  `ALPHARAVIS_EMBEDDING_JOB_STALE_AFTER_SECONDS` (default `900`). This prevents
  document/upload ingest from staying stuck if a dev reload or container restart
  happens after a job is claimed but before it is marked `done` / `failed`.
- LibreChat document upload auto-ingest first slice is implemented. The Bridge
  registers `file` / `input_file` document parts with media-gallery, maps the
  returned media-gallery path into the LangGraph workspace path, sends
  `pending_document_ingests` into the graph state, and `run_profile_start_node`
  loads the file through LangChain document loaders before routing it through
  `ingest_source(...)`.
- Source digest dedup is now active by default with
  `ALPHARAVIS_PGVECTOR_DEDUP_SOURCES=true`: if an identical source digest is
  already indexed for the same scoped source key, pgvector embed/upsert is
  skipped and the existing source is reused.
- `active_rag_prefetch_node` consumes active document/large-paste RAG metadata
  after memory prefetch and injects only a bounded `<active-rag-context>` system
  message. Archive-only Agent-path state can also run the safe Qwen3.5 2B
  archive-intent classifier; confirmed recall uses bounded source-scoped
  archive retrieval, while Fast Path bypasses this node.
- Live runtime check on 2026-05-19: LiteLLM now drops unsupported optional
  embedding params only for Ollama-backed routes, so LangChain/OpenAIEmbeddings
  through `rag_api` no longer fails on `encoding_format=base64` when
  `memory-embed` uses Ollama. OpenAI-compatible/llama.cpp routes keep parameter
  dropping disabled. `Archive RAG Smoke` passed with `acceptance_ok=true` and
  returned a bounded `rag_api` hit.
- LiteLLM's proxy metadata is isolated into a separate `litellm` Postgres
  database. `rag_api` keeps the `rag_api` database for LangChain PGVector
  tables, so LiteLLM Prisma startup migrations no longer touch
  `langchain_pg_collection` / `langchain_pg_embedding`.
- `ALPHARAVIS_DOCUMENT_RAG_BACKEND=alpharavis_pgvector` is the default for
  document/large-paste ingest. `rag_api` remains selectable with
  `ALPHARAVIS_DOCUMENT_RAG_BACKEND=rag_api` or `both`, but is no longer the
  product default.
- Bridge Test UI now includes `Native Document RAG Smoke`, which indexes a
  document/large-paste source through AlphaRavis pgvector and retrieves bounded
  chunks without calling `rag_api`.
- Live runtime check on 2026-05-19: `Native Document RAG Smoke` with
  `source_type=large_paste` passed in about `3.0 s` with `acceptance_ok=true`,
  `rag_api_not_used=true`, `active_source_key_recorded=true`, and two bounded
  pgvector hits.

Still needed:

- Implemented first slice: document/PDF/DOCX upload paths route through
  `ingest_source(...)` when LibreChat sends a `file` / `input_file` document
  part with a downloadable URL. `ingest_document_file` remains available for
  explicit server-local files. Live Bridge-compatible `input_file` smoke on
  2026-05-19 verified media-gallery registration and
  `pending_document_ingests` handoff; the queued pgvector job was then drained
  through the embedding queue and produced pgvector Catalog+Chunk rows for the
  upload source key. Remaining follow-up: browser-test real LibreChat PDF/DOCX
  uploads and extend mapping only if LibreChat emits a new attachment shape not
  covered by the current parser.
- Implemented: LangChain document-loader normalization for file-like
  sources now exists in `langgraph-app/document_ingest.py`. PDF, DOCX, HTML,
  Markdown, plain text, CSV/JSON/YAML/log files can be loaded into normalized
  document text plus source metadata before reaching `ingest_source(...)`.
- Add an optional LangChain PGVector / retriever evaluation path without
  replacing AlphaRavis pgvector as the default source of truth. The evaluation
  must compare metadata fidelity, thread scoping, archive ownership, catalog
  rows, queue behavior, and retrieval quality before any migration.
- Implemented first slice: add a LangChain-style retriever adapter around
  AlphaRavis source-scoped retrieval. `AlphaRavisSourceRetriever` exposes
  async `aget_relevant_documents` / `ainvoke` and returns LangChain `Document`
  objects when `langchain_core.documents` is available, otherwise a compatible
  local document shape. Follow-up: wire this adapter into future graph nodes
  where a Retriever interface is cleaner than router payloads.
- Implemented: optional reranking behind the router, default-off with
  `ALPHARAVIS_ENABLE_RAG_RERANKING=false`. Deterministic mode blends lexical
  query/chunk overlap with backend vector score and annotates hits with
  `rerank_score`. Model mode uses the configured llama.cpp Qwen3-Reranker
  endpoint (`ALPHARAVIS_RAG_RERANKER_URL`, default
  `http://192.168.178.140:8000`) and falls back to deterministic reranking when
  `ALPHARAVIS_RAG_RERANKER_FALLBACK_DETERMINISTIC=true`.
- Implemented first slice: optional LLM structured-output grading for
  Agentic-RAG is available behind `ALPHARAVIS_AGENTIC_RAG_LLM_GRADING=false`.
  The router accepts an LLM grader callback, falls back cleanly to deterministic
  grading on errors, and records the grading strategy in the trace. The example
  grader is `openai/big-boss`; the grader call disables hidden thinking with
  `enable_thinking=false` / `preserve_thinking=false` to keep the JSON judgment
  cheap. Current product decision: do not make this a required RAG feature while
  the Qwen3 reranker is active. Keep LLM grading default-off as an optional
  debug/comparison path only; use the reranker plus deterministic grading as the
  normal policy.
- Implemented first slice: streaming ingest progress for direct large
  documents/pastes. Run-profile start/completion/failure/skip events now include
  chunk-level progress for direct pgvector writes and Bridge activity extraction
  can surface those events.
- Implemented follow-up: asynchronously drained queue jobs now persist per-job
  progress JSON while `run_embedding_jobs` indexes chunks. The Bridge Observer
  `Embedding Queue` panel shows per-source planned chunks, completed chunks,
  percentage, latest event, status, and thread id.
- Implemented: add queue-only large document ingest for very large
  sources, with progress and a source handle returned before all chunks are
  embedded. The retrieval router now distinguishes queued pgvector work with
  `index_status=queued` and `queued_backends=["alpharavis_pgvector"]`, and
  Large Paste replacement markers can return a source handle while embeddings
  are still queued.
- Implemented first slice: chunk/source digest metadata is now used for scoped
  source dedup. Repeated identical sources with the same scoped source key skip
  embed/upsert and return a `deduped:...` backend result. Large-paste source
  keys are now content-based within the thread so repeated identical paste
  content lands on the same source key. Remaining follow-up: cross-source chunk
  reuse if we later need dedup across different source keys without losing
  exact reconstruction semantics.
- Implemented: guarded source reads for known source keys when bounded semantic
  retrieval is not enough. `read_source_chunks` returns ordered pgvector chunks
  for a known `source_key`, scoped to the current thread by default, and capped
  by `ALPHARAVIS_SOURCE_READ_MAX_CHUNKS` /
  `ALPHARAVIS_SOURCE_READ_MAX_CHARS`. `read_raw_source` reads a bounded raw
  Store slice for newly ingested documents and large pastes; it supports
  `search`, `start`, and `max_chars` so the model can page through exact source
  text without injecting the whole source.
- Implemented: user/operator RAG pin/unpin controls for thread-level active
  source sets. `pin_active_rag_sources`, `unpin_active_rag_sources`, and
  `inspect_active_rag_sources` persist per-thread active source/file ids in a
  Mongo-backed shared pin store when available, and
  `active_rag_prefetch_node` merges those pins with state-derived active RAG
  metadata. The Bridge Observer exposes a `RAG Pins` panel and
  `GET/POST /api/rag-pins` so operators can pin/unpin/clear thread sources
  directly.
- Implemented: code-aware LangChain splitter options for source-code profiles.
  Known code languages use LangChain language splitters when available and
  code-specific separator fallback otherwise.
- Rebuild/recreate the LangGraph container and live-test the new
  `langchain-text-splitters` dependency in the running stack.
- Tune large-paste runtime performance for very large real chat runs. A previous
  live two-turn large-paste test reached
  embedding, but a 27-chunk embedding batch and the later chat-model call hit
  the current 180s runtime timeouts before the document backend default moved to
  AlphaRavis pgvector. Candidate fixes: shorter large-paste chunks, queue-only
  ingest with progress, smaller backend batches, or a larger Bridge/LangGraph
  timeout only after backend throughput is understood. The default
  `memory-embed` route has been switched from the slower 4b model to
  `qwen3-embedding:0.6b`, with a new
  `RAG_COLLECTION_NAME=alpharavis_qwen06` collection to avoid vector-dimension
  collisions with old 4b rows. Run-profile ingest events are implemented;
  direct-ingest chunk progress can now be surfaced through Bridge activity
  events. Queue-drained background progress remains a follow-up.
    Follow-up probe on 2026-05-20 after enabling GPU acceleration on the
    Ollama host: `qwen3-embedding:4b` loaded and returned 2560-dim vectors,
    with warm probes around 12.8s for 2048 chars and 22.3s for 8192 chars;
    `ollama ps` reported about 2.38GB VRAM for the loaded 4B model. The 0.6B
    baseline remained faster, around 7.2s and 10.9s for the same warm probes.
    If 4B becomes the active route, use a separate 2560-dim collection rather
    than `alpharavis_qwen06`.
- Live-test the new large-paste intent split with real prompt-instruction,
  document, and mixed LibreChat examples. The deterministic classifier is the
  default because it costs no VRAM; optional small-model tie-breaking can be
  added later only if live examples show ambiguous classifications.
- Add optional archive `auto_on_intent` behavior after live quality/latency is
  measured. Keep archive-only threads passive unless this mode is explicitly
  enabled.
- Implemented: model reranking is live after restarting the llama.cpp Qwen3
  reranker with a larger physical batch than `-ub 64`. Direct GPU probes on
  2026-05-20 completed 3 docs / 277 tokens in about 0.51s and 10 docs / 1028
  tokens in about 2.03s. An in-container AlphaRavis router probe reported
  `strategy=llamacpp_qwen3_reranker`, `fallback_used=false`, and 0.426s for
  three candidates. The CPU reranker experiment was about 4x slower, so the
  current practical default is GPU reranking while keeping deterministic
  fallback enabled.
- Implemented: Bridge Test UI now includes `RAG Load Probe`
  (`/api/rag-load-probe`) to run concurrent embedding and reranker probes across
  configurable rough-token steps, with optional real Bridge `/v1/responses`
  queries. First live GPU probe with `qwen3-embedding:4b` plus GPU Qwen3
  reranker passed 400, 1000, 4000, 10000, 20000, and 40000 rough-token steps.
  The 4B embedding server reported `prompt_eval_count=4095` from 10k upward,
  so larger inputs should be treated as accepted but capped/truncated by the
  current embedding context unless the server context is raised. A smaller
  Bridge query probe passed 400 and 1000 rough-token steps through
  `/v1/responses`.
- LLM structured-output grading is not needed for the normal RAG path while the
  Qwen3 reranker is active. Keep it default-off as an optional comparison/debug
  mode only; do not track it as a blocker for RAG feature completion.
- Implemented first slice: active RAG prefetch now prepares a bounded
  retrieval query before embedding. Short questions stay direct, long/noisy
  turns are locally condensed and capped, and ambiguous long prompts can call
  the always-on small classifier model on the Big-Boss host port `8001`
  (`unsloth/Qwen3.5-2B-GGUF:Q4_1`, 8k context, reasoning off) for JSON labels,
  a short retrieval query, and line ranges. If
  `ALPHARAVIS_RAG_CLASSIFIER_API_BASE` is empty, AlphaRavis derives it from
  `BIG_BOSS_API_BASE` by replacing the port with `8001`. Fallback remains the
  local query condensation path when the classifier fails or returns invalid
  JSON.
- Implemented follow-up: large-paste ingest can use the same small classifier
  for long mixed prompts. Instruction/question line ranges are stripped from
  the indexed document body, the classifier retrieval query and line ranges are
  stored in ingest metadata/run_profile, and mixed replacement markers preserve
  the current query/task lines so the active user request is not lost.
- Implemented follow-up: source ingest metadata now includes deterministic
  `content_type=code|log|config|prose|table|mixed`, `source_title`,
  `source_keywords`, `source_entities`, and `source_symbols` for
  document-upload, explicit document-file, large-paste, and compression-archive
  sources. The pgvector chunk profile honors `content_type=log|code|config|prose`
  so logs/config/code can use better chunking hints without another model call.
- Implemented follow-up: long-prompt route classification can use the existing
  small Qwen3.5 classifier only after the normal short FastPath decision rejects
  a prompt for length. High-confidence `direct_query` / `noisy_query` labels
  with no document/instruction ranges may use the direct answer path; document,
  mixed, instruction, low-confidence, tool-keyword, and very large prompts stay
  on the agent path. Short FastPath behavior is unchanged.
- Implemented follow-up: archive-recall query condensation now builds a compact
  stronger archive/RAG query from vague prompts such as "wie war das nochmal mit
  X" plus recent thread context. The planner stores the suggested query in
  `run_profile`, and `context_retrieval_agent` has a
  `condense_archive_recall_query` tool for vague recall searches.
- Not planned as a separate model call: RAG/source sufficiency. The agent graph
  carries a standing prompt hint instead: for code/log sources, treat RAG hits
  as pointers and use `read_raw_source` for exact surrounding original text when
  snippets are insufficient.
- Planned: add tests and a Test UI/Observer probe for the small-model
  classifier path. It should verify short direct queries, long noisy queries,
  instruction-only large prompts, document-only large prompts, and mixed
  instruction+document prompts. Fallback must remain the existing local
  heuristic classifier when the model endpoint is down, times out, or returns
  invalid JSON.

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

- Decide whether `wake_pc` should stay as direct Wake-on-LAN or also route
  through the curated action endpoint.

Note: All previously listed "still needed" actions are now implemented:
- `check_ollama_models`, `load_embedding_model`, `unload_ollama_model`,
  `run_embedding_jobs` — implemented 2026-05-22 (see CHANGES).
- `reboot_server` — exists as `request_ubuntu_server_power_action(action="reboot-now")`
  via Ubuntu Llama Manager, and `owner_shutdown_llama_server()` + WoL restart
  via SSH for the owner power tools path.
- `kill_process` — exists as `owner_restart_llama_server()` (`pkill -f llama-server`)
  plus Ubuntu Manager service-control and ESP power-cycle.
- `delete_files` — intentionally not implemented (too dangerous for an agent).

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

Note: The previously listed "still needed" Crisis Manager items were all
implemented 2026-05-22 (see CHANGES):
- Mid-run crisis recovery for timeout, 502, connection errors, overload,
  rate-limit → handled by `error_classifier` + bounded crisis attempt.
- Post-recovery readiness gate → checks primary backend reachable before retry.
- Hard caps: `ALPHARAVIS_CRISIS_MAX_ATTEMPTS`, `ALPHARAVIS_CRISIS_MAX_WALL_CLOCK_SECONDS`,
  `ALPHARAVIS_CRISIS_ACTION_TIMEOUT_SECONDS`. Recursive-loop guard via
  `crisis_recovery_attempted` flag.
- Read-only Ollama/LiteLLM checks: `check_ollama_models`, `inspect_model_management_status`.

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
- `ALPHARAVIS_ENABLE_EMBEDDING_SCHEDULER=true` drains the queue periodically
  after real graph inactivity. `run_profile_start_node` updates an in-process
  activity timestamp, and the scheduler waits for
  `ALPHARAVIS_EMBEDDING_SCHEDULER_IDLE_AFTER_SECONDS` before running queued
  embedding jobs.
- `queue_vector_memory_backfill` queues bounded old Store records by query.
- `ALPHARAVIS_ENABLE_VECTOR_BACKFILL_DAEMON=true` can repeat that bounded
  backfill search, but only when `ALPHARAVIS_VECTOR_BACKFILL_QUERY` is set.

Still needed:

- More precise convenience backfill commands:
  Implemented 2026-05-22 (see CHANGES):
  - `queue_current_thread_vector_backfill` — index this exact thread
  - `queue_recent_artifact_vector_backfill` — index last N artifacts
  - `queue_selected_source_vector_backfill` — index selected source keys
- Active-job awareness for Pixelle/MCP jobs beyond the current big-LLM/Ollama
  model probes.

## Media / Vision Memory

Status: safe media metadata handling, media-gallery service, a separate vision
pgvector table, explicit video-analysis preparation, and media-index status
inspection are implemented. Full caption/OCR/transcription remains
provider/pipeline work.

Implemented:

- Bridge strips raw media blocks from chat context by default, preserves
  metadata markers, and automatically mirrors incoming image/video blocks into
  `media-gallery` when `BRIDGE_MEDIA_GALLERY_AUTO_REGISTER_IMAGES=true` and
  `BRIDGE_MEDIA_GALLERY_AUTO_REGISTER_VIDEOS=true`.
- LibreChat's normal `AlphaRavis Responses` model spec accepts video uploads on
  the `LangGraph Agent` endpoint. The LibreChat container applies
  `scripts/patch_librechat_video_uploads.js` at startup so this local endpoint
  emits `video_url` attachments instead of forcing the generic custom
  endpoint/text-upload path. The patch covers both the backend encoder and the
  client upload menu/drag-drop bundle, and preserves `videos`/`audios` in
  LibreChat's prompt formatter. It also patches LibreChat's OpenAI Responses
  converter so `video_url` reaches the Bridge as `input_video` while still
  setting `useResponsesApi: true` for the assistant response.
- The AlphaRavis-facing marker is rewritten to the stable gallery URL after a
  successful mirror. LibreChat's original visible attachment/file record stays
  untouched in this phase.
- `media-gallery` can register/download image, video, audio, or document URLs
  and exposes `/gallery`. Bridge mirroring accepts HTTP(S) URLs and inline
  `data:` image/video payloads while omitting inline payloads from Mongo
  metadata. Gallery `/assets` and `/gallery` support thread/group filters plus
  date/name/type/kind/thread/group sorting; `/gallery` can group by
  day+group, thread, group, date, or media type.
- Pixelle job results are scanned for media URLs and registered when present.
- `register_media_asset`, `semantic_media_search`, and `plan_media_analysis`
  tools exist.
- Vision/media embeddings use `alpharavis_media_vectors`, separate from the text
  table, so vector dimensions do not collide.
- `prepare_media_for_model` decides `register_only`, `pass_through`, `analyze`,
  or `index`; it only downloads video for explicit analyze/index modes and when
  `ALPHARAVIS_VIDEO_ANALYSIS_ENABLED=true`.
- `inspect_media_index_status` lets agents check which media/frame records are
  already present in `alpharavis_media_vectors`, and also reports matching
  pending/running/failed/done media-analysis queue records.
- `inspect_embedding_queue_status` lets agents answer general queue questions
  for text, archive, and media-analysis jobs in `alpharavis_embedding_jobs`.
- `prepare_media_for_model(mode="index")` queues video analysis/indexing as a
  durable `media_analysis` job in the same embedding queue used by text,
  archives, artifacts, memories, and session turns.
- Media-gallery registration now separates media assets from chat/tool
  appearances through the Mongo `references` collection; repeated mentions of
  one video should create references, not duplicate full embeddings.
- Media indexing dedupes by media source key, media vision model-card id,
  `ALPHARAVIS_MEDIA_INDEX_VERSION`, and the video chunking-config hash.
- Auto-index policy is ENV-controlled for user uploads, Pixelle MCP / ComfyUI
  outputs, and link references:

```text
ALPHARAVIS_MEDIA_AUTO_INDEX_ENABLED=true
ALPHARAVIS_MEDIA_AUTO_INDEX_USER_UPLOADS=true
ALPHARAVIS_MEDIA_AUTO_INDEX_PIXELLE_MCP_OUTPUTS=false
ALPHARAVIS_MEDIA_AUTO_INDEX_LINK_REFERENCES=false
```
- Video frame extraction uses `ffprobe`/`ffmpeg`, bounded FPS, bounded
  `ALPHARAVIS_VIDEO_ANALYSIS_MAX_FRAMES`, timestamped frame manifests, and the
  model-card defaults in `langgraph-app/model_cards.json`.
- `make video-analysis ENABLED=true FPS=1 MAX_FRAMES=100` can write the core
  analysis switches into `.env`.
- `make media-vision`, `make install`, `make update`, `make up`,
  `make up-fullstreaming`, and `make up-chat-fullstreaming` accept
  `VISION_ENABLED`, `VISION_URL`, `VISION_BASE_URL`, `VISION_MODEL`, and
  `VISION_FALLBACK` so a dedicated external vision embedding server can be
  written into `.env` before the stack starts.

Still needed:

- Connect a real vision embedding endpoint and enable:

```text
ALPHARAVIS_ENABLE_VISION_VECTOR_MEMORY=true
ALPHARAVIS_VISION_EMBEDDING_MODEL_URL=http://<vision-embedding-host>:<port>/v1
ALPHARAVIS_VISION_EMBEDDING_MODEL=<model-name-served-by-that-endpoint>
```

  Prepared: `.env(exaple)`, Docker env wiring, setup prompts, Makefile install
  and up/update arguments, and the `vector_memory` client now prefer
  `ALPHARAVIS_VISION_EMBEDDING_MODEL_URL` for a dedicated external
  llama.cpp/OpenAI-compatible vision embedding server.
  Captioning/OCR/transcription remain future work.

- Build the Meet/media-gallery integration as the operator-facing video rack.
  The current `media-gallery` already has its own port and Mongo-backed asset
  registration, but the UI and analysis pipeline are still basic.

  Current implementation checkpoint from 2026-05-12:

  - `BRIDGE_ALLOW_RAW_MEDIA_CONTEXT=false` and
    `BRIDGE_MEDIA_CONTEXT_MODE=metadata` are already the default path in
    `bridge_server.py`, `.env(exaple)`, and Docker Compose.
  - The Bridge converts OpenAI/LibreChat media content parts into metadata
    markers instead of forwarding raw media blocks to LangGraph.
  - `context_retrieval_agent` already has `semantic_media_search` for indexed
    media references and `inspect_media_index_status` for processed/indexed
    media plus pending queue status.
  - The shared `alpharavis_embedding_jobs` queue now carries both text/archive
    embedding jobs and video `media_analysis` jobs. `run_embedding_memory_jobs`
    drains both kinds through the existing model-management embedding window.
  - `/assets/resolve` can map a copied gallery/source URL back to the Mongo
    asset and its recorded references.
  - Incoming LibreChat/Responses video blocks are now copied into
    `media-data` through the Bridge and media-gallery before LangGraph context
    is built; the LLM marker points at the media-gallery URL.
  - `plan_media_analysis` remains explanatory. The real bounded preparation
    path is `prepare_media_for_model`.

  Goal:

  - Use the Meet/media-gallery service as the place where all videos from chat,
    uploads, Pixelle MCP outputs, and future Meet-server flows become visible.
  - Keep MongoDB/media-gallery metadata as the source of truth for original
    uploads and processed outputs.
  - Make every asset usable by link in later chats, either as a pass-through
    URL for Pixelle or as an analysis target that AlphaRavis downloads and
    preprocesses.
  - Preserve the relation between a user-supplied source video, the chat turn
    or Pixelle request that used it, and the processed video returned by Pixelle.

  Research note from 2026-05-12:

  - The active local target mentioned by the operator is assumed to be
    `Qwen/Qwen3.6-35B-A3B` unless the runtime model id says otherwise.
  - The official Qwen3.6 model card says it is a causal language model with a
    vision encoder and a native context length of 262,144 tokens, extendable up
    to 1,010,000 tokens with YaRN.
  - Its Hugging Face `preprocessor_config.json` has image
    `longest_edge=16777216`, `shortest_edge=65536`, `patch_size=16`,
    `temporal_patch_size=2`, and `merge_size=2`.
  - Its `video_preprocessor_config.json` has video
    `longest_edge=25165824`, `shortest_edge=4096`, `patch_size=16`,
    `temporal_patch_size=2`, and `merge_size=2`.
  - The model card's vLLM video example says default video sampling is
    `fps=2`, configurable through `mm_processor_kwargs`. For AlphaRavis, keep
    the operator default stricter at `1 fps max` because the requested local
    behavior prioritizes predictable load over maximum frame recall.
  - The same model card recommends increasing the video preprocessor
    `longest_edge` to `469762048` for hour-scale long-video workloads; keep
    this as an optional advanced model-card value, not the default.

  Data model plan:

  - Extend media asset records with derivation fields:
    - `asset_kind`: `original`, `processed`, `reference`, or `unknown`
    - `origin`: `librechat_upload`, `chat_url`, `pixelle_output`,
      `meet_server`, or `manual_register`
    - `parent_asset_id`
    - `root_asset_id`
    - `derivation_group_id`
    - `source_message_id`
    - `result_message_id`
    - `tool_call_id` or Pixelle `job_id`
    - `processing_provider`, for example `pixelle`
    - `processing_prompt` or compact prompt hash
    - `public_url`
    - `download_url`
    - `local_path`
    - `thumbnail_path` or `preview_path`
    - `duration_seconds`, `width`, `height`, `fps`, and `bytes`
  - Keep original videos distinct from processed videos, but group them under
    one derivation tree so "All" can show source and result together.
  - Add a stable lookup from LibreChat/Mongo upload ids to media-gallery assets
    if the file exists only in LibreChat's Mongo/filesystem layer.
  - Add idempotent registration keyed by source URL, upload id, local path, or
    source hash so repeated registration does not create duplicate gallery
    cards.
  - Preserve existing media-server filter stages. First audit the current
    filters, then insert derivation/grouping logic after safe metadata
    extraction and before gallery rendering/download.

  UI plan:

  - Replace the current simple `/gallery` HTML with a real work UI, still
    served by the media-gallery/Meet service port unless a separate frontend is
    justified later.
  - Add tabs or segmented controls:
    - `All`
    - `Original`
    - `Processed`
  - In `All`, group original input videos and Pixelle/processed result videos
    together by `derivation_group_id` or `root_asset_id`.
  - In `Original`, show only uploaded/input/reference videos.
  - In `Processed`, show only Pixelle/generated/processed outputs.
  - Use dense video cards with:
    - thumbnail or lightweight preview
    - title/source label
    - original/processed badge
    - thread/chat marker
    - Pixelle job/result marker when present
    - duration/resolution/filesize metadata
    - link/copy action in a small bottom-right menu
    - open/download actions
  - Do not autoplay every full video by default. Use `preload="metadata"` plus
    posters/thumbnails first. Add optional hover preview or low-rate muted
    preview clips only after performance is measured.
  - Generate thumbnails and tiny preview clips as background media jobs. The UI
    must remain useful when thumbnails are pending.
  - Add filters for media type, thread, source, date, and processing provider
    once the basic Original/Processed/All flow works.
  - Verify the UI on desktop and mobile with real videos before marking done.

  Link and ingestion plan:

  - Every gallery card needs a stable public/media URL that can be copied and
    pasted into another chat.
  - Default behavior must remain metadata-only: pasted or uploaded videos are
    not pulled into model context unless the user explicitly asks to analyze,
    inspect, describe, summarize, transcribe, compare, or otherwise understand
    the media content.
  - The copied link must be acceptable as:
    - a normal media reference for AlphaRavis
    - a pass-through input URL for Pixelle when the user asks to create a new
      video from it
    - a downloadable analysis target when the user asks to inspect/analyze it
  - Add explicit tool behavior:
    - For "send this to Pixelle", pass the URL through and avoid downloading
      unless Pixelle requires a local file.
    - For "analyze this video", download or resolve the asset into the media
      analysis cache, then preprocess frames for the target model.
    - For "copy link", return only the stable media URL, not internal local
      paths or Mongo ids.
  - Prefer a dedicated `analyze_media_asset` / `prepare_media_for_model` tool
    over a new agent. The model can decide when to call that tool from user
    intent; the tool should enforce the hard rules, caps, MIME checks, and
    frame sampling. The context retrieval agent should retrieve references and
    timecoded indexed hits, not perform heavy video preprocessing itself.
  - Add an intent/decision helper inside that tool:
    - `pass_through`: keep URL only for Pixelle or another downstream service.
    - `register_only`: save metadata/gallery entry, no download.
    - `analyze`: download/resolve locally, sample frames, and build bounded
      model-ready content.
    - `index`: enqueue media analysis/indexing for future retrieval without
      answering from raw media immediately.
    - The LLM may choose the mode, but the default fallback must be
      `register_only`, not `analyze`.
  - Add safety checks around download URLs:
    - allowed schemes
    - size limit
    - media MIME validation
    - path confinement under `ALPHARAVIS_MEDIA_ROOT`
    - optional signed/internal token later if the media service is exposed
      outside localhost.

  Video analysis rack plan:

  - Add a separate analysis pipeline instead of pushing raw videos into LLM
    context.
  - Resolve the model card for the active model id before preprocessing:
    - static built-in entry for `Qwen/Qwen3.6-35B-A3B`
    - optional JSON/YAML override for local aliases such as `big-boss`
    - fallback defaults when the runtime model has no vision card
  - Model-card fields should include:
    - `supports_images`
    - `supports_video`
    - `native_context_tokens`
    - `image_longest_edge`
    - `image_shortest_edge`
    - `video_longest_edge`
    - `video_shortest_edge`
    - `patch_size`
    - `temporal_patch_size`
    - `merge_size`
    - `preferred_video_fps`
    - `max_video_fps`
    - `max_frames`
    - provider-specific payload knobs such as
      `mm_processor_kwargs.fps` / `do_sample_frames`
  - Add ENV defaults:

```text
ALPHARAVIS_VIDEO_ANALYSIS_ENABLED=true
ALPHARAVIS_VIDEO_ANALYSIS_FPS=1
ALPHARAVIS_VIDEO_ANALYSIS_MAX_FPS=1
ALPHARAVIS_VIDEO_ANALYSIS_MAX_FRAMES=100
ALPHARAVIS_VIDEO_ANALYSIS_MAX_DOWNLOAD_BYTES=2147483648
ALPHARAVIS_VIDEO_ANALYSIS_MODEL_CARD_PATH=/workspace/langgraph-app/model_cards.json
ALPHARAVIS_VIDEO_ANALYSIS_PUBLIC_MEDIA_ROOT=/workspace/media-data
ALPHARAVIS_VIDEO_ANALYSIS_CACHE_ROOT=/workspace/media-data/analysis-cache
ALPHARAVIS_VIDEO_ANALYSIS_INCLUDE_AUDIO=false
ALPHARAVIS_VIDEO_ANALYSIS_TRANSCRIBE_AUDIO=false
```

  - Add Makefile/setup support:
    - `make media-vision` should be able to write the video-analysis switches.
    - Add a direct target such as `make video-analysis FPS=1 MAX_FRAMES=100`
      if that is simpler for repeat use.
    - `make status` should show whether video analysis is enabled, FPS, frame
      cap, cache root, and model-card path.
  - Sampling rule:
    - Never sample more than `ALPHARAVIS_VIDEO_ANALYSIS_MAX_FPS`.
    - Default to at most one frame per second.
    - For videos whose duration in seconds is less than or equal to the frame
      cap, sample one frame per second.
    - For longer videos, select at most `MAX_FRAMES` frames evenly across the
      full duration, so a one-hour video with `MAX_FRAMES=100` stays near 100
      frames instead of trying to keep one frame per second.
    - Preserve timestamps for every extracted frame.
  - Preprocessing rule:
    - Scale frames according to the active model card's video/image limits.
    - Keep aspect ratio.
    - Store extracted frames and metadata in the analysis cache.
    - Mark the model payload as video, not as unrelated still images, whenever
      the provider/server supports video content parts.
    - If the llama.cpp/OpenAI-compatible route cannot accept native video
      blocks, send a bounded sequence of timestamped image frames with a clear
      system/user message that they are sampled frames from one video.
  - Retrieval/RAG behavior:
    - Store analysis metadata, frame timestamps, captions, optional transcript,
      and embeddings under `alpharavis_media_vectors`.
    - Keep MongoDB/media-gallery as the asset source of truth and pgvector as
      the searchable index.
    - Allow later prompts such as "analyze this video" or "use this link as
      input for a new video" to resolve the asset by URL or asset id.
    - Use `inspect_embedding_queue_status` when the user asks how much indexing
      work is still pending.
    - Use `inspect_media_index_status` to distinguish "not indexed yet",
      "queued", "running", "failed", and "indexed".

  Implementation phases:

  1. Audit current media-gallery and Meet-server routes, existing media filters,
     LibreChat upload metadata, Pixelle result registration, and Mongo asset
     records.
  2. Extend the Mongo asset schema and registration API for original/processed
     grouping without breaking current `/assets/register` callers. Implemented
     for optional derivation fields, `asset_kind`, origin, parent/root asset,
     and derivation group fields.
  3. Add URL copy/download/open affordances and stable public links.
     Implemented for copy/open links in `/gallery`; signed links remain future
     work if the gallery is exposed outside localhost.
  4. Build the improved gallery UI with Original/Processed/All grouping.
     Partially implemented in the media server's `/gallery` route with
     `view=all|original|processed` tabs, derivation/thread/group sections, and
     query controls for media type, thread/group filters, date/name/type
     sorting, and grouping mode.
  5. Add thumbnail/preview generation and avoid heavy autoplay.
  6. Add model-card config and Qwen3.6 defaults. Implemented.
  7. Add video download, ffprobe/ffmpeg keyframe extraction, adaptive frame
     sampling, scaling, and analysis-cache storage. Implemented for explicit
     video analysis, without captions/transcription.
  8. Add the dedicated media-analysis preparation tool and wire explicit
     decisions for pass-through-to-Pixelle vs download-for-analysis.
     Implemented.
  9. Add optional frame captions, audio transcription, and media-vector indexing.
     Frame-level vision indexing is implemented through the shared durable
     embedding queue when vision pgvector is enabled; captioning and
     transcription remain future work.
  10. Add Makefile/setup/status controls and smoke tests. Partially
     implemented: `make video-analysis`, setup/status output, helper tests, and
     bridge media tests exist; live Docker/UI smoke remains needed.
  11. Mirror LibreChat-origin video input into `media-data` automatically
      before LangGraph sees it. Implemented for Bridge-facing HTTP(S) and
      inline `data:` video blocks; rewriting the visible LibreChat message card
      itself remains intentionally out of scope for this phase.
  11a. Mirror LibreChat-origin image input into `media-data` automatically
      before LangGraph sees it. Implemented through the same Bridge/media-gallery
      path as videos; image and video blocks from one chat share the same
      thread/group metadata so the gallery can display them together.
  12. Run a real Docker/LibreChat upload smoke with `AlphaRavis Responses`:
      - send a chat video through LibreChat
      - verify the Bridge registers it in `media-gallery`
      - verify the stable gallery URL appears in AlphaRavis-facing context
      - verify the gallery card is created and the stored bytes resolve through
        the media service URL
      - partial 2026-05-16: LibreChat stored `vitpose_00001.mp4` as a local
        `video/mp4` attachment and the restarted prompt formatter preserves
        `video_url` content parts. Follow-up Bridge Observer evidence showed
        LibreChat's Responses converter still reduced the request to
        `input_text`; the local patch now converts `video_url` to `input_video`.
        Repeat the browser send after reload to verify the Bridge/media-gallery
        side end-to-end.
  13. Investigate an explicit LibreChat-visible rewrite phase:
      - either rewrite the persisted LibreChat message/file metadata to the
        Media Gallery URL
      - or keep LibreChat's native attachment UI untouched and add a clearly
        linked gallery reference beside it
      - document the MongoDB/file-system mutation boundary before enabling this
        path, because this changes what the operator sees in historical chats
  14. Promote lightweight motion previews from "possible" to a measured UI
      follow-up:
      - poster first
      - muted hover/focus preview or tiny generated preview clip
      - no full autoplay grid until desktop/mobile performance is measured

  Acceptance:

  - A LibreChat-uploaded original video appears in the gallery as `Original`.
  - A Pixelle-generated or processed result appears as `Processed`.
  - `All` shows source and processed result grouped together when one was
    derived from the other.
  - Copying a card link and pasting it into a new chat gives AlphaRavis enough
    metadata to pass it to Pixelle or download it for analysis.
  - A prompt that only says to use a video as Pixelle input does not download or
    sample the video in AlphaRavis.
  - A prompt that explicitly asks to analyze the video calls the media-analysis
    preparation tool and stays within FPS/frame/download caps.
  - The UI can show many videos without starting all full video streams at
    once.
  - A single card can expose a deliberate moving preview without turning the
    whole gallery into an autoplay wall.
  - `ALPHARAVIS_VIDEO_ANALYSIS_MAX_FRAMES=100` keeps a one-hour video bounded
    near 100 sampled frames, while a ten-second video samples up to ten frames
    at the default one frame per second.
  - The model payload preserves video semantics where the active provider
    supports it; otherwise AlphaRavis states that it is sending sampled
    timestamped frames from a video.
  - The current media-server filter stages remain intact and covered by smoke
    tests.

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

Status: implemented for the current DeepAgents static graph-binding model.
Agents can inspect categories with
`describe_optional_tool_registry(category=...)`, MCP schemas are cached by
category, and specialist workers bind bounded materialized toolsets instead of
the old broad local tool lists.

Implemented:

- `alpharavis_toolsets.py` defines composable toolsets for coding, media, RAG,
  system/power, research, Hermes, debugger, context, and UI roles.
- `agent_graph.py` materializes those toolsets at graph build and binds each
  specialist to its own resolved local/MCP bundle; handoff tools are added
  explicitly outside the category bundle.
- MCP schemas are cached by category and only matching loaded MCP tools are
  attached to the specialist bundle that selected that category.
- `run_profile.selected_toolsets` records the likely categories inferred from
  the latest user message, and `run_profile.loaded_toolsets` records the
  materialized per-agent profiles, including tool names, missing tools,
  missing toolsets, cycle warnings, MCP categories, and schema fingerprint.
- The Hermes bridge agent no longer receives raw local/SSH execute tools from
  the coding execute category; terminal-oriented work stays delegated through
  Hermes or the debugger agent.
- Focused verification completed on 2026-05-13:
  - `pytest -q tests/test_alpharavis_toolsets.py` passes.

Open items:

- None for the current supported DeepAgents graph-binding model. Runtime
  hot-swapping can be revisited only if LangGraph/DeepAgents exposes a safe
  per-node rebinding API; it is not tracked as an active AlphaRavis open task.

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
  reasoning stream events; LibreChat compatibility still needs the
  `response.reasoning.*` normalization listed below
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

Current live-test status:
- Done: direct no-tool LangGraph calls use Responses successfully for
  fast-path/planner style calls.
- Done: DeepAgents can use Responses successfully with the patched LangChain
  hybrid streaming mode:

```text
ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING=true
ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=tool_calling
```

- Not stable yet: full internal Responses streaming for tool-bound DeepAgents
  calls as the default stack mode. The focused probe passes with
  `ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING=true`, but earlier full
  streaming failed with `item['content'] is empty` and Bridge-level soak testing
  is still needed.

Still needed:

- LibreChat Responses/reasoning/tool UI pass:
  - Status: implemented and live-smoked through the recreated `api-bridge`
    and `librechat` containers on 2026-05-11.
  - Added two explicit LibreChat model specs in `librechat.yaml` and updated
    the config to `version: 1.3.9` with `interface.presets: false` so
    `modelSpecs` is not competing with default presets:
    - `AlphaRavis Chat` using the existing custom endpoint through
      `/v1/chat/completions`.
    - `AlphaRavis Responses` using the same custom endpoint with
      `useResponsesApi: true`, `reasoning_summary: "detailed"` or `"auto"`,
      `reasoning_effort`, and `verbosity`.
  - Keep the Chat Completions model spec as the legacy/stable path.
  - Make the Responses model spec the path for LibreChat's reasoning bubble,
    tool execution timeline, and agent progress visibility.
  - Updated `langgraph-app/bridge_server.py` Responses streaming from the
    old `response.reasoning_text.delta/done` shape to LibreChat/Open
    Responses compatible events:
    - `response.output_item.added` for a `type: "reasoning"` item
    - `response.content_part.added` with `part.type: "reasoning_text"`
    - `response.reasoning.delta`
    - `response.reasoning.done`
    - `response.content_part.done`
    - `response.output_item.done`
  - Added `logprobs: []` to every `response.output_text.delta` and
    `response.output_text.done` event, matching LibreChat v0.8.5 validation.
  - Included the final reasoning item in the completed Response object's
    `output` array when reasoning text or summaries were emitted.
  - Preserved the old Chat Completions reasoning path by continuing to emit
    `delta.reasoning_content` when `BRIDGE_STREAM_REASONING_EVENTS=true`.
  - Do not promise raw OpenAI chain-of-thought. OpenAI-hosted reasoning models
    expose reasoning summaries, not raw reasoning tokens; full visible thinking
    is only possible when the selected local/OpenAI-compatible provider emits
    visible `reasoning_content`, `reasoning`, or `<think>` text.
  - Mapped LangGraph tool activity to Responses tool items when enabled:
    - tool-call start -> `function_call` output item
    - tool-call args -> `response.function_call_arguments.delta/done`
    - tool result -> `function_call_output` item
    - tool completion/failure -> matching `response.output_item.done`
  - Reuse the proven tool extraction patterns from
    `langgraph-app/alpharavis_acp_adapter.py` so tool names, call IDs, args,
    status, file locations, and output snippets stay consistent across AionUI
    and LibreChat.
  - Emit agent/node progress separately from final assistant text. Candidate
    sources:
    - LangGraph `updates` node names such as `general_assistant`,
      `debugger_agent`, `hermes_coding_agent`, `context_retrieval_agent`, and
      `power_management_agent`
    - LangChain `on_tool_start`, `on_tool_end`, and `on_tool_error` events
    - DeepAgents tool call messages and tool result messages
  - Bridge Test UI follow-up implemented:
    - LangGraph `updates` from the `planner` node are emitted as
      `response.reasoning.delta` with `alpha_reasoning_kind=internal_plan`.
    - `BRIDGE_RESPONSES_STREAM_REASONING_EVENTS=true` is the Responses default,
      so explicit provider reasoning and visible `<think>` blocks can reach
      LibreChat's reasoning path even when the legacy Chat Completions reasoning
      flag is disabled.
    - The Test UI renders live Status, Reasoning, and Planer panes above the
      chat transcript, while LibreChat still receives the combined reasoning
      stream.
    - LangGraph `messages/metadata` now maps `messages/partial` IDs to their
      source node, so streamed planner tokens are removed from visible answer
      text and routed only to internal-plan reasoning.
    - The Test UI live Status, Reasoning, and Planer panes can be expanded for
      longer diagnostics.
    - Text-only model conversion and bridge visible-output cleanup suppress
      `[thinking content block omitted]` placeholders.
    - `BRIDGE_RESPONSES_OUTPUT_DELTA_MAX_CHARS=1` and
      `BRIDGE_RESPONSES_REASONING_DELTA_MAX_CHARS=1` split visible answer text
      and model/plan reasoning into character-level SSE deltas for smoother
      rendering. Status events remain whole status lines.
    - `BRIDGE_STREAM_SUBGRAPHS=true` enables LangGraph subgraph streaming so
      the nested `alpha_ravis_swarm` workers can forward `messages/partial`
      token deltas instead of only returning the completed Swarm result.
  - Remaining streaming gap:
    - Responses hybrid mode still disables true model streaming for
      tool-bound LangChain calls. Use the Chat Completions `chat-full` profile
      or the experimental Responses full-streaming profile when investigating
      provider-level tool-call token streaming.
  - Added focused tests in `tests/test_bridge_responses.py`:
    - no `response.reasoning_text.*` events in Responses streams
    - `response.reasoning.delta/done` events contain `sequence_number`,
      `item_id`, `output_index`, `content_index`, and text/delta fields
    - output text delta/done events include `logprobs: []`
    - completed Response output can contain both a `reasoning` item and an
      assistant `message` item
    - tool start/result events become valid `function_call` and
      `function_call_output` items
    - `/v1/chat/completions` streaming still emits normal text and optional
      `reasoning_content`
  - Live smoke checks completed:
    - `POST /v1/responses stream=true` emitted `response.reasoning.delta/done`
      plus `response.output_text.delta/done` with `logprobs: []`.
    - Agent-path `POST /v1/responses stream=true` emitted LangGraph node
      activity as reasoning deltas.
    - Tool-path `POST /v1/responses stream=true` emitted `function_call`,
      `response.function_call_arguments.delta/done`, and
      `function_call_output` items.
    - `POST /v1/chat/completions stream=true` still works as the fallback path.
  - Approval UX status:
    - OpenAI Responses supports MCP approval request/response items for remote
      MCP tools, but LibreChat's custom endpoint path does not expose an
      AlphaRavis-native click-to-approve permission callback.
    - `api-bridge` therefore keeps the chat-text approval path for LibreChat:
      `approve`, `reject`, `replace: <safer command>`, `approve always`, and
      `immer erlauben`.
    - `approve always` / `immer erlauben` stores an exact scope/target/command
      allow entry for the current LibreChat thread only, in bridge process
      memory. It is cleared by `api-bridge` restart and is not global.
  - Still verify visually in the LibreChat browser UI that `AlphaRavis
    Responses` renders reasoning/tool activity in the intended panes.
- llama.cpp/local-model visible thinking follow-up:
  - User backend is llama.cpp/local models behind the OpenAI-compatible stack,
    not OpenAI-hosted reasoning models. Do not assume OpenAI raw chain-of-thought
    restrictions apply to the local backend; instead preserve whatever visible
    thinking the local provider actually emits.
  - Recommended runtime shape is valid and should remain supported:
    `LibreChat -> api-bridge /v1/responses -> LangGraph -> llama.cpp
    /v1/chat/completions`. The outer LibreChat-facing Bridge can speak
    Responses even when the internal LangGraph-to-llama.cpp model call uses
    Chat Completions. The outer Responses translation is enough for LibreChat's
    reasoning bubble, tool timeline, and LangGraph activity UI.
  - Current code status:
    - `langgraph-app/bridge_server.py::_message_reasoning_content` already
      extracts visible reasoning from `reasoning_content`, `reasoning`,
      `additional_kwargs.reasoning_content`, and list content blocks with
      `type: "thinking"` or `type: "reasoning"`.
    - `langgraph-app/bridge_server.py::_message_content` already skips list
      content blocks with `type: "thinking"` or `type: "reasoning"`.
    - Added `_VisibleThinkingSplitter` in `langgraph-app/bridge_server.py` for
      normal string content containing visible local-model thinking markers.
      It supports `<think>...</think>` and `<thinking>...</thinking>`, handles
      split marker boundaries across chunks, routes inside-thinking text to
      reasoning output, routes outside text to assistant output, and suppresses
      the marker text itself.
    - The splitter is wired into both external streaming paths:
      `_stream_responses` emits extracted thinking through
      `response.reasoning.delta`, and `_stream_chat_events` emits extracted
      thinking through the configured Chat Completions reasoning delta field.
    - The fallback state-read path used when no token was streamed also splits
      final `_last_ai_content(...)` text so stored `<think>` blocks do not leak
      into the visible final answer.
    - Explicit provider reasoning fields still win. If a part already exposes
      `reasoning_content` or `reasoning`, string `<think>` blocks in that part
      are stripped from visible output but not duplicated into reasoning.
    - `StreamingInternalContextScrubber` remains after the split: answer text
      goes through the content scrubber and reasoning text goes through the
      reasoning scrubber.
  - Added focused tests in `tests/test_bridge_responses.py`:
    - Responses stream: `<think>plan</think>Answer` emits
      `response.reasoning.delta` containing `plan` and `response.output_text`
      containing only `Answer`.
    - Responses stream with split markers across chunks routes thinking
      correctly and does not leak `<think>` or `</think>` into
      `response.output_text.done` or the completed Response object's assistant
      message.
    - Chat Completions stream emits `delta.reasoning_content` for thinking and
      normal `delta.content` for the answer.
    - Explicit provider reasoning fields still work and are not double-counted
      when string `<think>` blocks are also present.
  - Live smoke after patch:
    - Use `AlphaRavis Responses` in LibreChat with the llama.cpp model and a
      prompt that reliably produces a visible `<think>` block.
    - Confirm the `<think>` body appears in LibreChat's reasoning area and the
      final assistant message does not include the raw markers.
    - Confirm tool calls still appear as `function_call`/`function_call_output`
      items during the same run.
- Re-test DeepAgents internal Responses token streaming after LiteLLM,
  `langchain-openai`, or llama.cpp upgrades. Keep the stable non-streaming
  DeepAgents default until repeated smoke tests pass.
- Updated `docs/ALPHARAVIS_RESPONSES_COMPATIBILITY.md` so the documented
  streaming event list matches the actual LibreChat-compatible event surface.
- Updated `langchain-bridge-repo` separately:
  - added Chat Completions `reasoning_content` passthrough for streamed chunks
    where LangChain exposes visible reasoning
  - added tests for reasoning passthrough
  - committed and pushed to `THEman6989/langchain-fastapi-chat-completion` as
    `3e647bf Preserve streamed reasoning content`
  - main repo submodule pointer now needs to be included with the main
    ai-stack commit when the surrounding Bridge/LibreChat changes are committed

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

Status: skill cards exist and the provider-error-hardening workflow has been
promoted into a reviewed repo skill card.

Implemented:

- Added `ai-skills/provider-error-hardening/SKILL.md` for Hermes-style provider
  failure work:
  - classify first
  - retry only safe unsupported-parameter failures
  - preserve original and retry errors
  - document smoke/runtime evidence
  - keep LiteLLM/LangChain as the main AlphaRavis routing layer

Still needed:

- Use the DeepAgents and Hermes skill cards as templates when adding new agents.
- Continue extracting stable reusable skills from completed workflows when they
  repeat across sessions.
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
  - Hermes-style recent-tail protection with a 3-message minimum plus token
    budget/soft ceiling, rather than preserving 16 latest messages verbatim
  - latest user-message anchoring so the active request stays outside the
    reference-only summary
  - oversized-tail rebalancing: if the protected recent tail is still above
    `ALPHARAVIS_COMPRESSION_OVERSIZED_TAIL_RATIO` of the compression budget
    (default 60%), older tail messages move back into the compressible middle
    while the latest user message remains anchored
  - critical oversized-tail release: if the protected tail is above
    `ALPHARAVIS_COMPRESSION_OVERSIZED_TAIL_FORCE_MIDDLE_RATIO` (default 80%),
    even the latest user-message anchor can be released into compression/archive
  - multi-pass pre-run compression that re-estimates after each pass before
    falling back to hard trim
  - pre-run static prompt/tool reserve, so active-message thresholds leave room
    for the actual DeepAgents system prompt and tool schemas
  - the same static reserve is applied to handoff and post-run compression
    thresholds
  - final model-call budget estimation that counts active messages plus model
    kwargs and bound DeepAgents tool schemas before invocation
  - direct `archive_key` / `read_archive_record` references in active
    compaction summaries
- `agent/context_engine.py` inspired the lightweight AlphaRavis
  `compression_stats` state. A full plugin-style context engine is not needed
  yet because AlphaRavis compression also writes archives and pgvector records.
- Hermes skill ideas are represented by reviewed repo skill cards under
  `ai-skills/`, plus the Store-backed skill-library candidate flow.

Current Hermes-style context hardening plan:

- Implemented: local context discovery queries llama.cpp/OpenAI-compatible
  endpoints directly and uses `/props` or `/v1/props` for actual `n_ctx`.
- Implemented: run/profile budget accounting includes message tokens, static
  prompt/tool reserve, request estimate, active/hard thresholds, and archive
  counts.
- Implemented: make static context reserve agent-specific instead of always
  reserving the largest DeepAgents tool/schema block.
- Implemented: add an `inspect_context_budget` tool for operators and agents to
  see context length source, active/effective limits, reserve details, archive
  counts, and whether compression or hard rescue is needed.
- Implemented: add a final LangGraph budget-rescue node before the Swarm model
  call. If the assembled request would exceed the effective budget, it should
  run aggressive Hermes-style compression and archive the middle before the
  model sees the request.
- Implemented: provider-side context-overflow prevention now happens via
  `final_budget_rescue` before the Swarm model call. If the Swarm provider call
  still raises `context_overflow` or `payload_too_large`, AlphaRavis runs rescue
  once and retries the Swarm invocation with compressed state.
- Implemented: provider overflow retry parses provider-reported context limits
  from errors such as llama.cpp `n_ctx_slot` / "maximum context length" messages
  and recomputes the rescue budget from that smaller real limit.
- Still future: retry an already-failed provider call inside deeply nested
  third-party subgraphs when the exception happens after the AlphaRavis Swarm
  wrapper no longer has clean state ownership.
- Implemented: add archive-recall nudges when the latest user message clearly
  refers to old compressed context, so agents prefer `context_retrieval_agent`
  and `read_archive_record(...)` over guessing from summaries.
- Implemented: make multi-pass preflight and final budget rescue configurable as
  "until under full request budget" with a bounded maximum pass count. Node
  tests cover provider-reported budget snapshots and dynamic multi-pass final
  rescue.
- Implemented: live UI/Observer validation after a real over-budget llama.cpp
  run. On 2026-05-15 `/props` reported `n_ctx=128000`; a 61-message request
  with about 84.9k raw tokens compacted before the model call and returned
  `LIVE_OVER_BUDGET_OK`. Observer record `obs_23d3389c9d5f` showed
  `context_length=128000`, `request_tokens=23383`,
  `effective_active_limit=56300`, `pre_run_compression_passes=1`, and
  `pre_run_compression_budget_met=true`.
- Implemented: optional chunked summary compression is available behind
  `ALPHARAVIS_COMPRESSION_ENABLE_CHUNKED_SUMMARY=false` for ordinary
  over-budget runs. It also activates automatically when oversized-tail rescue
  has to move the latest user message into the compressible middle and the
  summary-model prompt would otherwise be pruned. It summarizes bounded chunks,
  then synthesizes one active reference summary while preserving exact raw
  messages in the archive. Observer/debug metadata includes chunk counts,
  omitted chars, prompt overhead/payload budgets, output budget,
  oversized-tail-forced chunking, and synthesis-pruning status. If max chunks
  omit prepared summary input, the final synthesis prompt now explicitly tells
  the model to mention archive lookup for omitted middle details.
- Implemented: summary max settings are ratio-first. `*_MAX_TOKENS=0` disables
  a fixed absolute cap, so summary output, summary prompt payload, and chunk
  output budgets scale from the effective compression limit derived from the
  discovered model context length. `inspect_context_budget` exposes the same
  derived values under `compression_summary_budget`.
- Implemented: Bridge Test UI Observer has a visual `Shrinking` section in
  addition to the raw `Kompression` JSON tab. It shows before/after tokens,
  shrink percentage, pass counts, budget status, head/middle/tail counts,
  prompt pruning, chunking, omitted chunk chars, chunk output budget, synthesis
  pruning, and archive key per compression scope.
- Implemented: Bridge Test UI Observer has a `Chunking Lab` for local chunked
  summary diagnostics. It starts runs through `POST /api/chunking/runs`, exposes
  results through `GET /api/chunking/runs/{run_id}`, uses the real AlphaRavis
  context compressor with chunked summary enabled, and renders action logs,
  summary calls, tool-pruning stats, prompt overhead/payload budgets, chunk
  omissions, synthesis pruning, and acceptance checks. The diagnostic uses a
  deterministic synthetic web-like corpus plus generated tool traces and
  variable prompt load, so it is available without external network fetches.
  The lab distinguishes `summary_mode=stub` from `summary_mode=real_llm`;
  `stub` validates plumbing only, while `real_llm` calls the configured summary
  model and is required for latency/quality evidence.
- Still future: run a live llama.cpp over-budget check for both explicit
  `ALPHARAVIS_COMPRESSION_ENABLE_CHUNKED_SUMMARY=true` and the automatic
  oversized-tail rescue path. The live Observer acceptance criteria are
  `summary_failed=false`, `summary_chunking_used=true`,
  `summary_chunk_omitted_chars=0`, and budget success for the relevant
  pre-run/final-rescue compression scope. Until that evidence exists, keep the
  global opt-in flag false for ordinary compression profiles.

External context-management learnings to evaluate:

- Planned: add a compact feature matrix for external systems and keep the
  AlphaRavis decision explicit. Current references:
  - OpenAI Responses `/responses/compact`: stateless compaction with opaque
    encrypted compaction items and verbatim prior user messages.
  - Claude Code `/compact [instructions]`: manual/auto compaction, focus
    instructions, `PreCompact` hooks, and CLAUDE.md summary instructions.
  - Cursor summarization: product-level chat summarization and separate smart
    condensation for large files/folders.
  - Letta/MemGPT: explicit in-context memory plus out-of-context archival and
    recall memory tiers.
  - LangChain/LangGraph: reusable trim/summarization middleware and state/store
    primitives, but no single opinionated AlphaRavis-style archive contract.
  - Google ADK: sliding-window event compaction plus Session/State/Memory split.
- Planned: run an external chunking/compaction comparison spike and record the
  result in architecture docs before changing defaults. Compare three distinct
  mechanisms rather than treating all "chunking" as one thing:
  - Document/RAG chunking: split large sources into chunks, embed, store, and
    retrieve only relevant chunks. Letta documents this explicitly for archival
    memory; AlphaRavis should map this to `rag_api` plus federated
    `semantic_memory_search`.
  - Conversation compaction: shrink old turns while preserving active task
    continuity. OpenAI `/responses/compact`, Claude `/compact [instructions]`,
    Hermes, and AlphaRavis all fit here, but OpenAI is opaque while AlphaRavis
    should stay inspectable.
  - Workflow/event compaction: summarize agent workflow events/tool telemetry
    separately from chat messages. Google ADK is the main reference here; this
    may become a separate AlphaRavis action-log compaction path.
- Planned: source links for the comparison spike:
  - https://platform.openai.com/docs/guides/conversation-state?api-mode=responses
  - https://platform.openai.com/docs/api-reference/responses/compact?api-mode=responses
  - https://docs.anthropic.com/en/docs/claude-code/slash-commands
  - https://google.github.io/adk-docs/context/compaction/
  - https://docs.letta.com/guides/ade/archival-memory
  - https://docs.letta.com/concepts/memory-management
- Planned: derive concrete AlphaRavis improvement proposals from that spike.
  Initial candidates:
  - Add Claude-style focus instructions for both manual compaction and chunked
    summary runs.
  - Keep OpenAI-style provider-native opaque compact items optional only; never
    replace AlphaRavis readable summaries and raw/retrievable archives.
  - Use Letta-style archival/RAG chunking as the default answer for large pasted
    documents instead of lossy conversation compression.
  - Implemented first ADK-style event-log compaction for tool/action telemetry
    so UI/archive history can stay inspectable without bloating model context.
  - Add progress events for chunked compression and RAG ingest so users see
    `chunk 1/N`, `chunk 2/N`, synthesis, and completion states in the chat/UI.
- Planned: compare `rag_api` document chunking with AlphaRavis' own pgvector
  retrieval backend and document what should be reused, federated, or kept
  separate. Current code-level anchors:
  - `rag_api` uses LangChain `RecursiveCharacterTextSplitter` with `CHUNK_SIZE`
    and `CHUNK_OVERLAP`, stores chunks under `file_id`, adds `user_id` and
    `digest` metadata, supports batch embedding, and queries by `file_id` or
    `file_ids`.
  - AlphaRavis `vector_memory.py` already implements a RAG-like pgvector backend
    with its own embedding model, semantic section splitting, overlap, full
    chunk storage, source catalog rows/table-of-contents, HNSW pgvector indexes,
    source metadata, and `semantic_memory_search`.
  - Decision to make explicit: `rag_api` is the generic external document RAG
    service, while AlphaRavis pgvector is the agent-memory/archive/artifact
    retrieval layer. They are conceptually similar, but ownership and source of
    truth differ.
- Partially implemented: evaluate improvements to port from `rag_api` into AlphaRavis
  pgvector where useful:
  - Implemented: configurable splitter profile per source type now includes
    `ALPHARAVIS_PGVECTOR_SPLITTER=auto|langchain|alpharavis`. In `auto`,
    explicit document and large-paste sources use LangChain
    `RecursiveCharacterTextSplitter` when available, while chat/archive/code/log
    profiles keep the AlphaRavis splitter. Archive and archive-collection
    sources now scan content before choosing their profile: code fences/common
    code syntax use the code profile, log/traceback lines use the log profile,
    and normal conversations stay chat. Chunk size and overlap still use the
    existing AlphaRavis ENV profile knobs.
  - Still open: section-level mixed archive splitting, where one archive can
    split prose, logs, and code blocks with different per-section strategies.
  - `file_id`/`source_key`-scoped targeted search for big-message ingest;
  - chunk digest/dedup metadata for repeated pasted text and repeated archive
    chunks;
  - batch embedding and bounded queue behavior for very large ingests;
  - distance/similarity threshold tuning so weak vector hits do not consume
    downstream LLM context;
  - optional full-document context reconstruction endpoint/tool when the user
    explicitly asks to load a known source, with hard token safeguards.
- Planned: refactor the RAG write/ingest and retrieval path toward a
  LangChain-native internal implementation while keeping AlphaRavis ownership
  of archives, thread metadata, access checks, and context-budget decisions.
  This should make the implementation less "attached from the side" than the
  first `rag_api` integration, without throwing away the useful `rag_api`
  patterns already copied.
  - Keep `rag_api` as the reference implementation for the current pragmatic
    document flow: `file_id`, `file_ids`, LangChain splitter, batch embedding,
    digest metadata, pgvector filtering, and optional distance threshold.
  - Use LangChain/LangGraph directly inside AlphaRavis for reusable primitives:
    loaders, splitters, embedding clients, vector-store adapters, retrievers,
    contextual compression/reranking, and future graph nodes.
  - Do not hand archive ownership to LangChain. MongoDB/LangGraph Store remains
    the source of truth for raw archives, compression lineage, redaction state,
    `archive_key`, `thread_id`, and `read_archive_record(...)`.
  - Build a stable AlphaRavis-facing interface first:
    `ingest_source(...)`, `query_source(...)`, `query_sources(...)`,
    `query_archive(...)`, and later `rerank_chunks(...)`. Internally the router
    may call `rag_api`, AlphaRavis pgvector, or LangChain retrievers.
    First ingest slice implemented: `retrieval_router.ingest_source(...)`
    exists as the stable write-side router entrypoint. It validates source
    content, chooses AlphaRavis pgvector, `rag_api`, or both from source type,
    metadata, environment flags, and `preferred_backend`, returns normalized
    `rag_file_id`, `rag_index_status`, `indexed_backends`, `index_status`,
    backend results, warnings, and errors. Current routing rules keep external
    documents/large pastes on AlphaRavis pgvector by default
    (`ALPHARAVIS_DOCUMENT_RAG_BACKEND=alpharavis_pgvector`), keep archives in
    AlphaRavis pgvector unless `ALPHARAVIS_ENABLE_RAG_ARCHIVE_MIRROR=true` or
    the caller explicitly requests `rag_api`, and allow
    `ALPHARAVIS_DOCUMENT_RAG_BACKEND=rag_api|both` / `preferred_backend=both`
    for adapter comparison. Follow-up: wire future document upload flows through
    this function instead of direct backend calls.
    Implemented first product call-site: compression archive creation now calls
    `ingest_source(source_type="archive", preferred_backend="auto")` instead of
    directly mirroring to `rag_api` and separately indexing pgvector. The archive
    record stores the router's normalized `ingest_status`, `rag_file_id`,
    `rag_index_status`, `rag_indexed_at`, `indexed_backends`, and
    `ingest_errors`. Large-paste ingest and `write_alpha_ravis_artifact(...)`
    now also use the router. Remaining call-site class to watch: future
    non-chat manual ingest commands. Explicit document/PDF/DOCX upload, chat
    `/ingest`/large-paste flows, and AlphaRavis artifacts route through
    `ingest_source(...)`.
  - Move backend selection out of `agent_graph.py` and into
    `retrieval_router.py`, so LangGraph nodes/tools call one AlphaRavis API
    instead of knowing about `rag_api`, pgvector, mirrors, and fallback rules.
    First slice implemented: source-key query backend orchestration now lives in
    `retrieval_router.query_sources_with_backends(...)`. `agent_graph.py` keeps
    the LangGraph tool wrappers, archive-key lookup, store access, and logging,
    but delegates source query execution to the router. The router combines
    AlphaRavis pgvector hits and explicit `rag_api` file-id hits, normalizes
    backend provenance, and keeps archive-only `rag_api` retrieval passive unless
    a mirror/file id is explicitly supplied.
  - Keep result normalization in AlphaRavis: every backend should return bounded
    chunks with source metadata, score fields, backend provenance, thread/source
    scope, and instructions that raw archive reads are a separate explicit step.
  - Add a small evaluation matrix before replacing working code: compare
    `rag_api` HTTP path, direct LangChain pgvector retriever, AlphaRavis
    pgvector source search, and optional reranked results on the same archive
    and document questions.
  - Local reference checkout: `helper-repos/awesome-rag` contains the
    noworneverev/Awesome-RAG catalogue. Use it as a discovery map for projects
    and patterns, especially LangChain, LlamaIndex, Dify, Flowise, Haystack,
    RAGFlow, Cognita, fastRAG, AutoRAG, FlashRAG, GraphRAG, vector stores,
    memory systems, evaluation frameworks, document parsers, and model serving.
    Do not import runtime code from this catalogue; inspect linked projects and
    official docs before copying implementation ideas. Local read-through
    finding: the checked-out Awesome-RAG content is mostly a catalogue; its
    `techniques.md` and `papers.md` pages are still placeholders, so it is
    useful for discovery but not a direct LangGraph implementation template.
  - Local LangGraph agentic-RAG Schablone:
    `helper-repos/langgraph-agentic-rag-template` contains a downloaded copy of
    the current LangChain docs page plus the archived
    `langgraph_agentic_rag.ipynb` example. Use the docs page as authoritative;
    the notebook is useful as concrete code but marks itself archival. Pattern
    to adapt:
    - `agent` / `generate_query_or_respond`: decide direct answer vs retrieval;
    - `retrieve`: call AlphaRavis `retrieval_router`, not a demo vector store;
    - `grade_documents`: relevance gate before using retrieved chunks;
    - `rewrite_question`: sharpen weak archive/document queries;
    - `generate_answer`: answer from bounded chunks, with raw archive reads only
      as an explicit fallback.
    Follow-up: implement an AlphaRavis-specific agentic-RAG graph slice using
    this loop around `query_sources_with_backends(...)` and thread-aware
    `rag_active` metadata.
    First router-level slice implemented:
    `retrieval_router.agentic_rag_retrieve(...)` now wraps
    `query_sources_with_backends(...)` with the Schablone steps
    `retrieve -> grade_documents -> rewrite_question -> retrieve retry ->
    generate_answer context packet`. It uses a deterministic relevance gate and
    query rewrite first, not an LLM grader yet, so it is safe to test locally.
    It returns `graph_trace`, `grade`, `context_packet`, `next_action`,
    `final_query`, and `rewritten_query`.
    Implemented follow-up: `agent_graph.py` exposes this as the
    `agentic_rag_retrieve` tool and the RAG/memory toolset includes it. It is
    still explicit tool use, not automatic archive injection. Current policy:
    use deterministic grading plus the active Qwen3 reranker; LLM grading stays
    optional/default-off and is not needed for the normal RAG path.
    Handoff for new context windows:
    `docs/ALPHARAVIS_RAG_HANDOFF.md` summarizes the user intent, implemented
    files, current routing behavior, LangGraph Schablone mapping, verification
    commands, and the next best steps.
- Implemented: added a more RAG-API-like "ask this source" retrieval surface for
  AlphaRavis pgvector. Agents can still call broad `semantic_memory_search(...)`
  and then `read_archive_record(...)`, but `rag_api` also has a simpler and more
  model-friendly pattern: `query + file_id(s) -> relevant chunks`. The
  AlphaRavis tool/API shaped like:
  - `query_source(query, source_type, source_key, limit=...)`
  - `query_archive(query, archive_key, limit=...)`
  - `query_sources(query, source_keys=[...], source_type=..., limit=...)`
  returns chunk text, score/similarity, chunk index/count, source metadata, and
  retrieval instructions. The local pgvector path mirrors `rag_api`'s `$eq` /
  `$in` filter shape and optional distance-threshold semantics through
  `ALPHARAVIS_PGVECTOR_DISTANCE_THRESHOLD`. This gives the model exactly the
  interface it usually wants: "I know which archive/document/source matters;
  fetch only the relevant parts for this question."
- Planned: copy the useful `rag_api` retrieval pattern, not blindly copy the
  whole backend. Relevant patterns to adapt:
  - filter vector search by `file_id`/`source_key`;
  - support multi-source filtered search like `/query_multiple`;
  - include authorization/thread scope checks before returning chunks;
  - expose distance/similarity threshold behavior;
  - keep raw archive reads separate from chunk search so exact-history loads stay
    deliberate.
- Planned: decide archive indexing ownership explicitly instead of drifting into
  two competing RAG stores. Current recommendation is hybrid, not a full
  migration:
  - Keep MongoDB/LangGraph Store as the source of truth for AlphaRavis archives,
    because archive records include thread lineage, compression metadata,
    covered message ranges, raw redacted messages, archive collections, and
    `read_archive_record(...)` semantics.
  - Keep AlphaRavis pgvector as the primary index for agent memory, archives,
    artifacts, skills, and thread-scoped retrieval because it already stores
    `source_type`, `source_key`, `thread_id`, `thread_key`, catalog rows, chunk
    maps, and archive-specific metadata.
  - Use AlphaRavis pgvector as the default backend for external documents and
    large pasted document ingests, so the product continues toward the native
    implementation. Keep `rag_api` selectable as an adapter/reference backend
    for `file_id`-scoped document search, LangChain loaders, and generic
    document chunking comparisons.
  - Consider optionally mirroring archive text into `rag_api` only as a secondary
    federated index, with `file_id=archive:<archive_key>` and clear ownership
    metadata. Do not make `rag_api` the only archive source unless it can preserve
    thread isolation, raw archive retrieval, collection hierarchy, and redaction
    guarantees.
  - Add an evaluation task before any migration: compare archive recall quality,
    latency, metadata fidelity, dedup behavior, and operational complexity for
    AlphaRavis pgvector-only vs `rag_api` mirror vs full `rag_api` archive index.
- Partially implemented path: build the hybrid archive/document retrieval
  wrapper instead of replacing `vector_memory.py`.
  - Keep archive source of truth in MongoDB/LangGraph Store. The raw compressed
    middle messages, redaction state, thread lineage, archive collections, and
    exact `read_archive_record(...)` behavior stay owned by AlphaRavis.
  - Add an optional `rag_api` mirror for large text sources, especially
    compression archives, artifacts, and future big pasted documents. Use stable
    file ids such as `archive:<archive_key>` and `artifact:<artifact_id>`.
  - Store mirror metadata on the AlphaRavis source record:
    - `rag_file_id`
    - `rag_index_status`
    - `rag_indexed_at`
    - `rag_chunk_count` when available
    - `indexed_backends`, for example `["alpharavis_pgvector", "rag_api"]`
  - Add a small `rag_api_client.py` rather than importing the FastAPI app
    directly into LangGraph. Use the HTTP API first so `rag_api` keeps its own
    config, vector-store setup, batch embedding behavior, distance threshold,
    and LangChain/ExtendedPgVector internals isolated.
    Implemented: `langgraph-app/rag_api_client.py` wraps `/embed`, `/query`,
    and `/query_multiple`.
  - On archive creation, after the Store write succeeds, optionally mirror the
    archive `content` into `rag_api` when enabled. If mirroring fails, keep the
    AlphaRavis archive valid and record the failure in metadata/run profile;
    retrieval falls back to AlphaRavis pgvector.
    Partially implemented: `ALPHARAVIS_ENABLE_RAG_ARCHIVE_MIRROR=false` is the
    default-off flag. When enabled, newly created compression archives are sent
    to `rag_api` as `file_id=archive:<archive_key>` and the archive record gets
    `rag_file_id`, `rag_index_status`, `rag_indexed_at`, and
    `indexed_backends`. Failures are recorded without invalidating the archive.
  - Make `query_archive(...)`, `query_source(...)`, and `query_sources(...)`
    route through a retrieval wrapper:
    - prefer AlphaRavis pgvector for external/big documents by default, and use
      `rag_api /query` or `/query_multiple` only when a `rag_file_id` mirror
      exists or the document backend is explicitly set to `rag_api` / `both`;
    - fall back to AlphaRavis `vector_memory.py` source-key search when no
      mirror exists, the source is agent-memory/catalog/media, or `rag_api` is
      unavailable;
    - never load full raw archives automatically just because an archive key is
      present.
    Partially implemented: `query_archive(...)` checks existing archive mirror
    metadata and queries `rag_api` first when a ready `rag_file_id` exists, then
    still returns AlphaRavis pgvector fallback hits. `query_source(...)` and
    `query_sources(...)` use the same `rag_api` client for source/file-id
    queries while keeping AlphaRavis pgvector source-key search.
  - Planned: add optional reranking support in the AlphaRavis retrieval router.
    Current `rag_api` does vector top-k retrieval plus optional
    `RAG_DISTANCE_THRESHOLD`; it does not implement a real cross-encoder or
    reranker pass. The AlphaRavis wrapper should support:
    - retrieve a wider candidate set from `rag_api`/pgvector, for example top
      20-50;
    - rerank `query + chunk` pairs with a configurable backend such as
      local `bge-reranker`, Jina/Cohere-compatible rerank APIs, or a future
      LiteLLM/OpenAI-compatible rerank route;
    - return only the final top 3-8 chunks to the LLM with original vector score,
      rerank score, source metadata, and backend provenance;
    - keep reranking default-off until latency and quality are measured in the
      Bridge Test UI, because reranking improves precision but costs another
      model/API call per candidate chunk;
    - add a smoke/probe surface that compares vector-only vs reranked results on
      archive/document questions.
  - Planned: add a thread-aware RAG activation router so retrieval is used at
    the right time instead of always injecting vector hits. The routing rule
    should distinguish user-ingested documents from passive archives:
    - If the user adds a PDF, pasted document, artifact, or other explicit
      non-archive source to a thread and it is indexed in the configured
      document RAG backend, mark the thread as RAG-active. Future LLM calls in
      that thread should automatically run source-scoped retrieval against the
      indexed document(s) and inject only bounded top chunks, because the user
      intentionally gave the thread a document context.
    - If the thread only has compression archives, keep RAG passive by default.
      Do not automatically search every archive on every turn. Instead expose
      `query_archive(...)` / `search_archive(...)` as the model-facing tool and
      use lightweight prompt/user-intent heuristics such as "wie war das
      nochmal", "hatten wir", "Archiv", "Dokument", "in meinem Repo", or a
      referenced archive/source id to trigger retrieval.
    - If both explicit documents and archives exist, automatically retrieve from
      the active document set first, and let the agent call archive retrieval
      only when the question asks for older compressed context or the document
      hits are insufficient.
    - Store the decision inputs as thread/source metadata, for example
      `rag_active=true`, `active_rag_file_ids`, `active_source_keys`,
      `rag_activation_reason=document_ingest|large_paste|manual_pin`, and
      `archive_rag_mode=tool_only|auto_on_intent|manual`.
    - Add an override so the operator/user can pin or unpin RAG for a thread
      without changing archive storage. This keeps normal short chats cheap,
      makes explicit document conversations behave like document-RAG, and avoids
      pulling 100k-token archives into context just because they exist.
  - Add an operator smoke surface in Bridge Test UI / Observer so the archive
    RAG foundation can be validated without manual curl commands.
    Implemented: `Archive RAG Smoke` posts to `/api/archive-rag-smoke`, mirrors
    a small archive as `file_id=archive:<archive_key>`, queries that source, and
    returns acceptance checks plus runtime errors as JSON. Live smoke currently
    reaches `rag_api` and LiteLLM, but fails because LiteLLM's `memory-embed`
    backend at `192.168.178.140:11434` refuses the connection. After the
    embedding backend is online, rerun this smoke and expect `acceptance_ok=true`
    with at least one bounded hit containing the archive retrieval rule.
  - Planned: build a minimal `memory-embed` backend bring-up path before adding
    more RAG features. This is separate from deciding what memory content should
    be indexed. The immediate work is: pick/reach a text embedding backend,
    validate the OpenAI-compatible `/v1/embeddings` or Ollama `/api/embed`
    surface, measure input-size and latency limits, then wire that stable route
    into AlphaRavis pgvector and the optional `rag_api` archive mirror.
    Decision: use a normal text embedding model first via Ollama behind
    LiteLLM. Default config now targets
    `EMBEDDING_LITELLM_MODEL=ollama/qwen3-embedding:0.6b` and
    `EMBEDDING_API_BASE=http://<ollama-host>:11434`. The operator must pull
    `qwen3-embedding:0.6b` on the Ollama host, then recreate LiteLLM and rerun
    the Memory Embed Tester. Treat 32k tokens as the expected context target for
    `qwen3-embedding:0.6b`, but validate the actual accepted input size and
    latency with the tester because Ollama/server limits may differ. Keep the
    OpenAI-compatible path documented for future llama.cpp/LM Studio embedding
    backends by setting
    `EMBEDDING_LITELLM_MODEL=openai/<served-model>` and a `/v1` base URL.
    Live result: LiteLLM `memory-embed` is functional and returns 2560-dim
    vectors. Performance on the current Ollama host is slow for large chunks
    (2048 chars ~20s, 4096 chars ~42s, 8192 chars ~86s). Defaults now use
    profile-specific token chunking: default 900 tokens / 125 overlap, chat
    700 / 100, logs 1200 / 75, code 600 / 80, with
    `ALPHARAVIS_PGVECTOR_CHARS_PER_TOKEN=4.0` and 45s embedding timeout.
    Follow-up: decide whether to keep this throughput, use smaller queue
    batches, or move embeddings to a faster OpenAI-compatible backend.
    Additional probe: `qwen3-embedding:0.6b` works through Ollama `/api/embed`,
    returns 1024-dim vectors, reports a 32768-token context, and completed a
    131072-char / ~32768-rough-token probe in ~40.5s. `aroxima/gte-qwen2-1.5b-
    instruct` reports 131072 context and 1536 embedding length in metadata, but
    Ollama rejects `/api/embed` with HTTP 501, so it is not usable as an Ollama
    embedding backend in this setup. Decision: switch `memory-embed` from
    `qwen3-embedding:4b` to `qwen3-embedding:0.6b` because speed is more
    important than vector dimensionality for the current local RAG flow.
  - Planned: add real vision-embedding backend support after the text
    `memory-embed` route is proven. Test whether the target llama.cpp/OpenAI-
    compatible server accepts the intended vision payload and returns vectors;
    do not assume Ollama can serve the selected vision embedding model. Keep the
    first implementation as a capability probe plus config documentation before
    indexing production media through it.
    Decision: vision embedding remains default-off and experimental. It should
    only be enabled with an explicit flag/argument after text embedding is
    stable, because it may require a separate RTX 3090/remote backend and more
    model-management logic.
    Implemented probe: Bridge Test UI Observer now has `Memory Embed Tester`,
    which can target a custom IP/base URL, model, OpenAI-compatible or Ollama
    endpoint, text or experimental vision payload, and step through increasing
    input sizes until rejection or a slow-response threshold.
    Live result: the panel is served by Bridge Test UI, but the default
    `memory-embed` probe currently fails at LiteLLM with HTTP 500 because the
    downstream embedding backend is unreachable. Bring up or repoint the
    embedding backend, then rerun the text probe before testing vision payloads.
  - Keep `read_archive_record(...)` as an explicit exact-history tool for the
    rare cases where chunk retrieval is not enough. The model should first use
    `query_archive(...)` to find relevant chunks and only call
    `read_archive_record(...)` when exact raw turns are required.
  - Avoid unbounded storage duplication. The raw archive exists once in
    MongoDB/Store. Retrieval indexes may duplicate chunk text and embeddings;
    this is expected RAG index overhead. Do not mirror every small memory or
    metadata-only catalog row into `rag_api`; mirror only large text sources
    where the stronger document retriever is useful.
  - Preserve thread isolation by carrying `thread_id`, `thread_key`, and source
    ownership metadata in `rag_api` document metadata. `query_archive(...)`
    must check the AlphaRavis source record before querying the mirror so a
    caller cannot use arbitrary `file_id` values to bypass thread boundaries.
  - Later, split current responsibilities into clearer modules:
    - `vector_memory.py`: AlphaRavis pgvector backend, catalog rows, agent
      memory/archive/artifact/media indexes.
    - `rag_api_client.py`: `/embed`/upload/query/query_multiple/document lookup
      client.
    - `retrieval_router.py`: backend selection and result normalization for
      `query_source`, `query_sources`, and `query_archive`.
    Implemented first slice: `rag_api_client.py` and a small
    `retrieval_router.py` exist. Backend selection still mostly lives in
    `agent_graph.py` and should be moved behind the router in a follow-up.
  - Acceptance:
    - A compression archive records an `archive_key` and, when enabled, a
      `rag_file_id=archive:<archive_key>` mirror.
    - `query_archive(query, archive_key)` returns relevant chunks without
      loading the whole raw archive.
    - If `rag_api` is down or the mirror is missing, `query_archive` falls back
      to AlphaRavis pgvector source-key search.
    - `read_archive_record(...)` remains available but is not part of the
      normal first retrieval step.
    - A 100k-token archive can be searched by question with bounded returned
      chunks and without injecting the full archive into active model context.
- Planned: document the conceptual answer for operators: RAG is not "only
  searching", but a pipeline: ingest/load -> split/chunk -> embed -> store in a
  vector/index backend -> retrieve/rerank relevant chunks -> ground the LLM
  answer in those chunks. Semantic search is the retrieval part of RAG; an
  embedding model plus pgvector backend is enough to build a local RAG retrieval
  system when paired with ingestion and chunk management.
- Implemented first slice: operator/user-provided `focus_topic` and
  `compact_instructions` can now guide AlphaRavis compression. The latest user
  message may include `<focus_topic>...</focus_topic>`,
  `<compact_instructions>...</compact_instructions>`, `/compact ...`,
  `@compact ...`, or `@focus ...`; the bounded extraction feeds both one-shot
  and chunked summary prompts, is recorded in archive metadata/run profiles, and
  is shown in Observer `Shrinking` as `Compact Focus`. Implemented follow-up:
  the Bridge Test UI `Chunking Lab` now has a `Compact Instructions` field and
  includes that value plus progress actions in its diagnostic JSON.
- Implemented: `PreCompact` / `PostCompact` runtime events for AlphaRavis.
  `compression.precompact` records reason, scope, token pressure, selected
  head/middle/tail counts and indexes, summary prompt pressure, and whether
  chunking will run before the summary model call. `compression.postcompact`
  records the archive key, before/after token estimates, H/M/T counts, summary
  failure state, and chunking result after the archive key is allocated. This is
  Observer/debug-only for now; do not run arbitrary hooks in the main path until
  there is a clear sandbox/timeout story.
- Planned: keep AlphaRavis transparent rather than adopting OpenAI-style opaque
  compaction items as the primary local format. If an opaque/provider-native
  compaction item is ever used, wrap it beside the existing readable summary and
  raw archive record, not instead of them.
- Implemented first policy table: AlphaRavis now has a documented memory-tier
  policy in `docs/ALPHARAVIS_ARCHITECTURE.md` and a shorter operator version in
  `docs/ALPHARAVIS_USAGE_NOTES.md`. It separates latest task tail, active
  compaction summary, raw archive records, archive collections, document/source
  records, vector recall chunks, durable MemoryKernel facts, temporary workflow
  state, and Observer/run telemetry.
- Implemented first Google-ADK-style workflow event compaction separately from
  chat-message compression. Tool-call requests, tool outputs, duplicates, and
  long action logs are collapsed into a `Workflow / Tool Event Compact Log`
  before the summary prompt. The compact log is stored in archive metadata and
  the raw archive record, while redacted original messages remain available for
  exact archive reads. Implemented polish: the Observer `Shrinking` cards now
  show workflow event counts, tool-call/result counts, compact-log character
  size, and a bounded `Workflow / Tool Events` preview instead of leaving the
  log only in raw compression metadata.
- Planned: add a quality rubric for promoting chunked summary from opt-in to a
  default profile. Minimum evidence should include live over-budget runs,
  static prompt plus variable prompt load, real LLM latency, before/after manual
  inspection, archive-recall success for omitted details, and no silent loss
  when `summary_chunk_omitted_chars > 0`.
- Partly implemented: stream/surface fine-grained compression and ingest
  progress without making normal chat noisy.
  The Bridge already has a status path through Responses reasoning deltas and
  Chat Completions `reasoning_content` activity chunks. The compressor now has
  a progress callback/event channel and records the timeline in archive metadata
  and run-profile debug fields. Implemented event names:
  - `compression.started`
  - `compression.precompact` with reason, scope, token pressure, H/M/T counts,
    prompt pressure, and chunking decision
  - `compression.workflow_events.compacted` with compacted tool/action event
    counts and compact-log character size
  - `compression.chunk.started` / `compression.chunk.completed` with
    `index`, `total`, source char/token estimates, elapsed time, and output
    summary chars
  - `compression.synthesis.started` / `compression.synthesis.completed`
  - `compression.postcompact` with archive key, before/after token estimates,
    H/M/T counts, summary failure state, and chunking result
  - `large_ingest.started`, `large_ingest.chunk_indexed`, and
    `large_ingest.completed` for RAG-backed big-message ingest
  - failure/fallback events when summary generation fails or the system switches
    to archive/RAG-only behavior.
- Implemented first status surface: Bridge streaming summarizes the latest
  compression event from LangGraph updates as `context_compaction`
  reasoning/status activity, and large-ingest events already surface as
  `large_ingest`. The exact event list remains in Observer raw compression or
  source-ingest metadata. Remaining feature polish: add profile-controlled
  fallback plaintext progress lines only for clients that cannot display
  reasoning/status events.
- Planned: make status emission configurable and not too noisy:
  `ALPHARAVIS_STREAM_COMPRESSION_PROGRESS=true`, max update frequency, and a
  plaintext fallback flag. Observer/Test UI should show the same events in the
  live status panel and persist them with the compression/ingest run metadata.
- Planned: do not copy Cursor/OpenAI automatic truncation behavior that silently
  drops old context. AlphaRavis should prefer explicit compression, explicit
  hard-trim notices, archive lookup instructions, and Observer-visible stats.
- Core implemented: handle the case where the first/latest user message itself
  contains a huge pasted document such as 100k/130k+ characters. LangGraph owns
  the large-message decision for every client path: plain large messages first
  run through pre-run compression, then `large_paste_post_compression_node`
  indexes/replaces the document/code/log body only if the active request is
  still above the configured post-compression trigger ratio. The ingest path
  splits/indexes the body through the router/pgvector chunking path, stores raw
  source text, records chunk stats/source digest in `source_manifest`, and
  replaces the active paste with a source handle plus any detected current
  question/task. A focused 130k first-message test covers chunk progress,
  manifest stats, marker replacement, and the no-question fallback.
  - Preferred backend: use AlphaRavis pgvector for large pasted documents by
    default, through the same `ingest_source(...)` router used by LangGraph.
    Keep `rag_api` selectable as an adapter/reference path for comparison. The
    LangGraph ingest layer should create a stable source key, index the text
    through the router, then keep only the manifest and user question in active
    context.
  - Archive relationship: do not duplicate the full document into both RAG and
    AlphaRavis active archives by default. Store a small AlphaRavis manifest
    archive with the RAG `file_id`, source title, chunk/index stats, and raw-text
    preservation policy. If exact raw preservation outside RAG is required, make
    that an explicit `preserve_raw=true` ingest mode.
  - Implemented follow-up: explicit syntax supports paired `/ingest`,
    `/big-context`, and fenced `<big-context name="...">...</big-context>`
    blocks so operators can opt into ingestion from normal LibreChat without a
    custom UI. "Protect" means preserve exact raw text in raw-source/archive
    storage, not keep the whole body in active model context.
  - Implemented follow-up: auto-ingest status is now compactly visible. Large
    paste replacement records a `source_manifest` in `large_paste_ingests`, the
    active marker includes a short `Source manifest` line with chunk/index
    stats, and the Bridge Observer renders a `Big Message / Source Ingest`
    section from LangGraph metadata. Do not add an ask-before-ingest step for
    the normal auto path: auto-ingest only happens after compression/budget
    checks show the full paste does not fit. Still open: optional profile
    control for how loudly the UI surfaces the conversion. Never silently
    truncate the latest user text.
  - Implemented follow-up: if the huge pasted text includes an explicit
    question, keep that question active and retrieve relevant chunks before
    answering. If there is no task/question, create only a lightweight
    manifest/table-of-contents and ask what to extract or analyze.
  - Summary policy: do not make a lossy whole-document summary the source of
    truth for unknown future questions. Raw archived chunks remain authoritative;
    any overview summary must point back to chunk/archive keys.
  - UI follow-up: the Observer now has a `Big Message / Source Ingest` panel
    with source handles, queued/indexed backend state, character counts, and
    chunk stats. Remaining display/control polish: richer before/after active
    prompt view and retrieval previews. LangGraph remains the source of truth
    for the decision and metadata.
  - Tools/API: expose source-key lookup for agents, e.g. reuse
    `semantic_memory_search` plus `read_archive_record(...)` where possible, or
    add a narrower `read_large_context_chunk(source_key, chunk_id)` tool if raw
    archive records become too coarse.
  - Acceptance: a first-message 130k-character document path is covered by a
    focused unit smoke for chunk progress, manifest stats, source marker
    replacement, and no-question fallback. Remaining live acceptance: run the
    same flow through LibreChat/Browser against the real stack and verify
    provider context safety plus grounded retrieval quality.

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

Status: implemented for static graph compile-time specialist bundles, MCP
category filtering, and run-profile toolset metadata.

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
- Done: DeepAgents Responses streaming is ENV-controlled and documented with a
  stable default plus experimental full/hybrid streaming opt-ins.

### Chunk 8: Maintenance And Metadata Helpers

Status: partially implemented for deterministic title/insight helpers.

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
- Done: `langgraph-app/maintenance_helpers.py` can suggest short deterministic
  thread/archive titles and extract review-only insight candidates without
  auto-promoting them into memory.
- Done: `suggest_thread_title` and `extract_review_insights` tools expose those
  helpers to agents.
- Still future: offline archive/trajectory compression evaluator and optional
  shell hooks/approval allowlists.

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

6. Usage, cost, and rate-limit telemetry.

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

7. Prompt assembly and context-file cache hygiene.

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

8. Offline trajectory/archive compression evaluator.

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

9. Shell hooks and approval allowlists.

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

10. Provider adapter hardening.

    Status: staged AlphaRavis-local hardening. Phase A/B is implemented for
    direct Responses compatibility retries, retry failure diagnostics,
    ChatLiteLLM fallback kwarg cleanup, and run-profile/log visibility for
    compatibility retries. Do not copy the full Hermes provider stack into
    AlphaRavis; keep LiteLLM and LangChain as the main routing layer, then add
    small compatibility guards where local OpenAI-compatible endpoints are
    known to reject harmless parameters.

    Reference:

    ```text
    C:\experi\ai\hermes-agent\agent\auxiliary_client.py
    C:\experi\ai\hermes-agent\run_agent.py
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

    Implementation plan:

    - Phase A: centralize small request-shape hardening in
      `langgraph-app/provider_hardening.py` and use it from direct Responses
      calls and ChatLiteLLM construction/binds. Adopt only the Hermes patterns
      that are low-risk for AlphaRavis:
      - omit `temperature` for providers/models that manage sampling
        server-side, such as Kimi/Moonshot-style endpoints
      - drop `None` values before sending request payloads
      - retry direct `/v1/responses` once when a local endpoint rejects
        harmless parameters such as `parallel_tool_calls`, `truncation`,
        `store`, `metadata`, `temperature`, or token-limit spellings
      - map token-limit spellings between `max_output_tokens`, `max_tokens`,
        and `max_completion_tokens` instead of failing planner/summary calls
      - apply the same pre-send cleanup to ChatLiteLLM `model_kwargs` so
        fallback Chat Completions does not reintroduce parameters that the
        selected local endpoint is known to reject
      - implemented: failed compatibility retries now preserve the original
        provider error and include the retry failure in the classified error
    - Phase B: add structured telemetry from the compatibility layer into
      `run_profile` / operational logs:
      - request mode
      - provider/model/base-url family
      - stripped parameters
      - retry reason
      - fallback mode when direct Responses falls back to ChatLiteLLM
      - implemented for compatibility retries: direct Responses completion logs
        include retry metadata, returned AI messages carry
        `responses_compatibility_retry`, and planner/fast-path run profiles set
        `provider_hardening_last_retry`
    - Phase C: expand provider profiles only when needed by real runtime
      evidence:
      - GPT-5/OpenAI-compatible endpoints that require `max_completion_tokens`
      - providers that require Responses instead of Chat Completions
      - endpoints that reject sampling knobs on reasoning models
      - local llama.cpp/LiteLLM quirks discovered by smoke tests
      - implemented: `provider_hardening.py` now has a small provider-profile
        layer. `auto` keeps the local LiteLLM/llama.cpp path conservative,
        detects Kimi/Moonshot sampling behavior, maps direct OpenAI/GitHub
        Chat token limits to `max_completion_tokens`, and exposes explicit
        profile overrides for evidence-backed cases such as
        `responses_required`.
    - Phase D: keep direct non-OpenAI adapters out of AlphaRavis unless
      LiteLLM cannot represent a required feature. If added, make them optional
      and documented, not a replacement for the current gateway.
      - implemented: no direct provider adapter was added. The compatibility
        layer records a disabled direct-adapter policy and keeps LiteLLM /
        LangChain as the route; future direct adapters must have focused
        evidence, docs, and tests.

    Acceptance:

    - Direct Responses planner/summary calls retry once after safe unsupported
      parameter errors and preserve the original error if the safer retry also
      fails.
    - ChatLiteLLM fallback calls receive the same conservative cleanup for
      known server-managed sampling parameters.
    - Provider profile metadata is attached to direct Responses AI messages and
      operational logs, and planner/fast-path run profiles can record the
      active provider hardening profile.
    - `responses_required` / `ALPHARAVIS_CHAT_FALLBACK_MODE=responses_required`
      blocks silent ChatLiteLLM fallback when runtime evidence says the
      provider requires Responses.
    - Direct non-OpenAI adapters remain out of AlphaRavis; the documented path
      is still the OpenAI-compatible LiteLLM/LangChain gateway.
    - The behavior is ENV-controlled and default-safe.
    - Focused unit tests cover both pre-send hardening and retry payload
      rewriting.
    - `docs/ALPHARAVIS_CHANGES.md` and `docs/ALPHARAVIS_USAGE_NOTES.md`
      describe the runtime knobs and remaining limits.

11. Thread title and insight helpers.

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
    - Done: candidate user/system insights can be extracted into a pending
      review queue, listed, accepted, or rejected. Accept is the only path that
      writes curated memory.
    - Keep this separate from raw archives and pgvector source-of-truth rules.

## Aktuelle Zusammenfassung der offenen Aufgaben (Stand: 12. Mai 2026)

*Diese Zusammenfassung dient dem schnellen Überblick für den Anwender. Die KI sollte bei der Bearbeitung stets die detaillierten Sektionen oben sowie die verlinkten Architektur-Dokumente prüfen, um keine technischen Details oder Abhängigkeiten zu übersehen.*

### Prioritäre Baustellen:
1.  **Responses & Streaming Stabilität:** Finalisierung des Full-Streaming-Modus (derzeit experimentell) und Entfernung lokaler Patches, sobald Upstream-Fixes für LangChain verfügbar sind. Optimierung der Latenz im Agenten-Pfad.
2.  **Model & Power Management:** Live-Validierung der realen Ollama/Embedding-Actions, Context-Policy und des Ubuntu-Manager-Action-Endpoints.
3.  **Crisis Manager:** Live-Validierung der Mid-run-Recovery bei echten Timeout-/502-/Connection-Fehlern.
4.  **Media & Vision:** Ausbau der Media-Gallery zur zentralen Video-Verwaltung (Meet-Integration), inklusive Vision-Embeddings und Frame-Analyse.
5.  **OpenWebUI:** Verifizierung der Integration und Konfiguration der Web-Suche (SearXNG).

### Kürzlich erledigt:
- **Lazy Tool Loading:** Spezialisten binden jetzt materialisierte, bounded
  Toolset-Bundles inklusive MCP-Kategoriecache; die Profile stehen in
  `run_profile`.
- **Service Dashboard / Tailscale:** Das Dashboard wird von den
  Tailscale-Helper-Zielen standardmäßig auf Port `8090` eingeplant; Opt-out ist
  weiterhin möglich.
- **Model/Crisis/Backfill/Streaming:** Reale Ollama/Embedding-Actions,
  automatische primary/secondary Context-Policy mit Rollback, Mid-run-Crisis-Recovery
  mit Hard Caps, exakte Embedding-Backfill-Commands, guarded Full Tool Streaming
  und optionales Two-phase-Final-Streaming sind implementiert.

*Hinweis: Diese Zusammenfassung sollte regelmäßig aktualisiert und verfeinert werden, um den Projektfortschritt präzise abzubilden.*

## RAG Feature-Konzentrat (Stand: 21. Mai 2026)

Diese Liste ist der kurze Arbeitszettel fuer die naechsten RAG-Feature-Updates.
Grosse LibreChat-/Browser-/Live-Lasttests kommen danach separat; beim
Implementieren trotzdem weiter fokussierte Unit-/Smoke-Tests laufen lassen.

1. **Section-level Mixed Archive Splitting: implementiert, Live-Qualitaet offen**

   `vector_memory.chunk_text(...)` segmentiert `archive` und
   `archive_collection` Quellen jetzt in geordnete Prosa-/Log-/Code-/Config-
   Abschnitte und verwendet pro Abschnitt das passende Chunk-Profil. Der Schalter
   ist `ALPHARAVIS_PGVECTOR_SECTION_LEVEL_ARCHIVE_SPLITTING=true`; pro Quelle
   kann `section_level_splitting=false` gesetzt werden. Noch offen: echte
   Archiv-Retrieval-Qualitaet mit gemischten Chat/Log/Code-Archiven vergleichen.

2. **Archive `auto_on_intent`: implementiert, Agent-Pfad default-on**

   `active_rag_prefetch_node` fragt auf dem Agent-Pfad bei vorhandenen
   aktuellen Thread-Archiven standardmaessig den sicheren Qwen3.5-2B-Classifier,
   ob die aktuelle Anfrage alte/compressed Archive meint. Der Classifier liefert
   strict JSON mit `archive_recall`, `search_query`, `confidence` und `reason`;
   bei Ausfall/Timeout/ungueltigem JSON greift der lokale Archive-Recall-
   Condenser als Fallback. Nur bei bestaetigtem Intent werden die letzten
   `ALPHARAVIS_ARCHIVE_AUTO_ON_INTENT_MAX_ARCHIVES` Archive source-scoped
   durchsucht. Fast Path laeuft daran vorbei. Aktuelle Upload-/Datei-/Bild-/
   Video-/URL-/Pixelle-/Source-Aufgaben muessen als `archive_recall=false`
   klassifiziert werden, sofern der User nicht explizit alten Archivkontext
   verlangt. `archive_rag_mode=manual` bleibt der Opt-out. Noch offen: grosse
   LibreChat-/Browser-/Live-Beispiele auf False-Positive-Verhalten messen.
