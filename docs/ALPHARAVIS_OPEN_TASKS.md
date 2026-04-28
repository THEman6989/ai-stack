# AlphaRavis Open Tasks

This is the running backlog for features that are intentionally prepared but
not fully wired yet.

## Custom Model / Power Management

Status: prepared, default off.

Implemented:

- `langgraph-app/model_management.py` exists as the custom hardware layer.
- `.env(exaple)` contains all switches and defaults them off.
- `make model-management` can write the relevant `.env` switches.
- `power_management_agent` is only registered when:

```text
ALPHARAVIS_ENABLE_MODEL_MANAGEMENT=true
ALPHARAVIS_ENABLE_ADVANCED_MODEL_MANAGEMENT=true
```

Still needed:

- Provide the curated external action endpoint:

```text
ALPHARAVIS_MODEL_MGMT_ACTION_URL=
ALPHARAVIS_MODEL_MGMT_API_KEY=
ALPHARAVIS_MODEL_MGMT_ALLOW_ACTIONS=true
```

- Populate safe real actions:
  - `check_llama_server`
  - `check_ollama_models`
  - `check_comfyui`
  - `start_llama_server`
  - `restart_llama_server`
  - `wake_pc`
  - `load_embedding_model`
  - `unload_ollama_model`
  - `run_embedding_jobs`
- Populate HITL/destructive actions:
  - `shutdown_server`
  - `reboot_server`
  - `kill_process`
  - `delete_files`
- Restrict `delete_files` to explicitly allowed temp/work folders.
- Decide whether `wake_pc` should stay as direct Wake-on-LAN or also route
  through the curated action endpoint.

## Crisis Manager

Status: documented, not implemented as an active retry node yet.

Desired behavior:

- Enabled only by:

```text
ALPHARAVIS_ENABLE_MODEL_MANAGEMENT=true
ALPHARAVIS_ENABLE_ADVANCED_MODEL_MANAGEMENT=true
ALPHARAVIS_ENABLE_CRISIS_MANAGER=true
ALPHARAVIS_CRISIS_MANAGER_MODEL=openai/edge-gemma
```

- Trigger on main-model failures such as timeout, 502, connection errors, or
  LiteLLM backend generation health failure.
- Use the small Ollama model only as a crisis moderator, not for normal complex
  work.
- Automatically run non-destructive checks and safe starts:
  - status checks
  - logs/read-only probes
  - `wake_pc`
  - `start_llama_server`
  - `restart_llama_server`
- Interrupt for destructive actions:
  - shutdown
  - reboot
  - kill process
  - delete files
- Send the user a short status message while recovery is happening.
- After health checks pass, retry the original user request against `big-boss`.
- Add hard caps:
  - max recovery attempts
  - max wall-clock time
  - no recursive crisis loops

ENV placeholders already exist:

```text
ALPHARAVIS_CRISIS_AUTO_ACTIONS=check_llama_server|check_ollama_models|check_comfyui|start_llama_server|restart_llama_server|wake_pc
ALPHARAVIS_CRISIS_HITL_ACTIONS=shutdown_server|reboot_server|kill_process|delete_files
ALPHARAVIS_CRISIS_MAX_ATTEMPTS=1
ALPHARAVIS_CRISIS_TIMEOUT_SECONDS=120
```

## Embedding Queue And pgvector

Status: pgvector retrieval chunks are implemented; model lifecycle runner is
not fully populated.

Still needed:

- A real embedding job queue runner behind `run_embedding_jobs`.
- Manual backfill tools:
  - index this thread
  - index last N artifacts
  - index selected document/source keys
- Queue visibility in status output.
- Optional idle scheduler that starts only after no active LangGraph/Pixelle/MCP
  work is running.

## Pixelle / ComfyUI Power Flow

Status: preflight hook exists, default off.

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

- Wire wake/start actions so ComfyUI can be woken before submitting image jobs.

## Bridge

Status: Chat Completions is primary; Responses API wrapper exists.

Still needed:

- Test whether LibreChat preserves `reasoning_content` as a separate reasoning
  panel or shows it as normal text.
- Keep `BRIDGE_STREAM_REASONING_EVENTS=false` until verified.
- Decide if `/v1/responses` should become a first-class external endpoint or
  stay a compatibility wrapper.

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
