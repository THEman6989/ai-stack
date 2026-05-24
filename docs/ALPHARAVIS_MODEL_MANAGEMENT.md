# AlphaRavis Custom Model And Power Management

This document describes the custom local-hardware layer implemented in
`langgraph-app/model_management.py`.

## Why It Exists

The Ollama management machine is not the main thinking machine. It is useful
for startup, crisis handling, Wake-on-LAN style operations, and limited fallback
work. It should not receive complex agent workflows when the large llama.cpp
server is expected to handle them.

The embedding model also lives on this Ollama machine. Because the machine may
not keep both the small chat/crisis model and the embedding model loaded at the
same time, AlphaRavis treats embedding work as a scheduled maintenance window.

## Default Safe Behavior

By default, this whole custom layer is off:

```text
ALPHARAVIS_ENABLE_MODEL_MANAGEMENT=false
ALPHARAVIS_ENABLE_SERVER_MODEL_MANAGER=true
ALPHARAVIS_ENABLE_ADVANCED_MODEL_MANAGEMENT=false
ALPHARAVIS_ENABLE_CRISIS_MANAGER=false
ALPHARAVIS_ENABLE_OWNER_POWER_TOOLS=false
```

The dedicated Server Model Manager agent stays available by default through
`ALPHARAVIS_ENABLE_SERVER_MODEL_MANAGER=true`. The older broad model-management
planning layer remains off until `ALPHARAVIS_ENABLE_MODEL_MANAGEMENT=true`.

The dashboard Settings UI at `http://localhost:8090/settings` can filter by the
`model` category or search for `MODEL_MGMT`, `UBUNTU_LLAMA`, `CRISIS`, and
related keys. Use `Temporary anwenden` for runtime experiments on new chat
turns, or `Permanent speichern` when the value should be written to `.env`.

When `ALPHARAVIS_ENABLE_MODEL_MANAGEMENT=true`, AlphaRavis can inspect and plan:

```text
inspect_model_management_status
plan_embedding_maintenance
inspect_ubuntu_llama_manager
diagnose_ubuntu_llama_no_response
recover_ubuntu_llama_no_response
control_ubuntu_llama_service
request_ubuntu_server_power_action
configure_ubuntu_llama_instance
prepare_comfy_for_pixelle
request_power_management_action
```

Advanced hooks are separate:

```text
ALPHARAVIS_ENABLE_ADVANCED_MODEL_MANAGEMENT=true
```

That enables the `power_management_agent`, Pixelle ComfyUI preflight hooks, and
the crisis-manager routing surface. The crisis manager no longer requires
owner SSH tools when the Ubuntu Llama Manager API is configured.

Real power/model actions are still disabled by default:

```text
ALPHARAVIS_ENABLE_POWER_MANAGEMENT=false
ALPHARAVIS_MODEL_MGMT_ALLOW_ACTIONS=false
ALPHARAVIS_MODEL_MGMT_ACTION_URL=
```

That means shutdowns, service restarts, Ollama unload/load actions, and
embedding-job runners return dry-run plans until a curated endpoint is provided.
Ubuntu Llama Manager write/recovery actions are gated by the same
`ALPHARAVIS_MODEL_MGMT_ALLOW_ACTIONS=true` switch.

## Ubuntu Llama Manager And Server API

AlphaRavis can call the separate private `ubuntu-llama-manager` service when
configured:

The AI-stack now treats that Manager strictly as the control plane. It reads
instance status and saved commands, and it asks the Manager to change
model/context/parallel/command config or to stop/restart/recover services. It
does not proxy tokenization through the Manager.

Runtime information comes from the selected llama-server directly. The
`LlamaCppRuntimeClient` calls `/apply-template`, `/tokenize`, `/slots`,
`/v1/models`, `/completion`, and `/v1/chat/completions` on the instance
`base_url` discovered from `/llama/instances`. Chat token counting renders the
prompt with `/apply-template` and counts it with `/tokenize` on the same
server. This path is required for hard context-budget decisions.

`ContextScheduler` parses `--ctx-size`/`-c`, `--parallel`/`-np`, and
`--kv-unified` from each saved command and creates a lease before async LLM
calls. With `--kv-unified`, capacity is the shared context pool times
`ALPHARAVIS_CONTEXT_SAFETY_FACTOR`; without it, capacity is conservatively
`ctx_total // parallel` times that factor. The current lease store is
process-local, so a multi-worker deployment still needs a Redis/Postgres lease
store before it can enforce one global budget across workers.

Small independent side tasks use a separate background lane. Non-LLM read-only
work can overlap with the main agent. Small LLM side tasks, such as Router,
Judge, Summarizer, RAG compression, and chunk ranking, still reserve a normal
ContextLease after direct llama-server token counting, but their admission is
capped by `ALPHARAVIS_BACKGROUND_CONTEXT_MAX_UTILIZATION`. Low-priority or
speculative tasks can be skipped or cancelled under pressure. Write, restart,
recovery, and power actions are not admitted through this lane.

```text
ALPHARAVIS_UBUNTU_LLAMA_MANAGER_IP=<llama-host-ip>
ALPHARAVIS_UBUNTU_LLAMA_MANAGER_PORT=8099
ALPHARAVIS_UBUNTU_LLAMA_MANAGER_URL=http://<llama-host>:8099
ALPHARAVIS_UBUNTU_LLAMA_MANAGER_API_KEY=<API_TOKEN>
ALPHARAVIS_UBUNTU_LLAMA_ESP_IP=<esp-ip>
ALPHARAVIS_UBUNTU_LLAMA_ESP_PORT=80
ALPHARAVIS_UBUNTU_LLAMA_ESP_URL=http://<esp-ip>
ALPHARAVIS_UBUNTU_LLAMA_ESP_API_KEY=<ESP_AUTH_TOKEN>
ALPHARAVIS_UBUNTU_LLAMA_CONTEXT_MIN=512
ALPHARAVIS_UBUNTU_LLAMA_CONTEXT_MAX=262144
ALPHARAVIS_BACKGROUND_TASKS_ENABLED=true
ALPHARAVIS_BACKGROUND_CONTEXT_MAX_UTILIZATION=0.70
```

For normal setup, enter the IP fields. `ALPHARAVIS_UBUNTU_LLAMA_MANAGER_URL`
and `ALPHARAVIS_UBUNTU_LLAMA_ESP_URL` are optional full-URL overrides; when
empty, AlphaRavis builds `http://<ip>:<port>` for the manager and
`http://<esp-ip>` for the ESP default port 80.

This is server management as well as model management. The LangGraph tools map
to the manager API:

- `inspect_ubuntu_llama_manager`: reads `/health`, `/status`, `/models`, and
  `/llama/instances`, so the returned `/status` covers llama services, reboot,
  GPU power/health, API state, and ESP status.
- `diagnose_ubuntu_llama_no_response`: posts to `/ai-stack/diagnose-llama`
  without executing recovery.
- `recover_ubuntu_llama_no_response`: posts to `/ai-stack/llama-no-response`
  only when model-management actions are enabled.
- `control_ubuntu_llama_service`: posts to `/llama/start|stop|restart`,
  `/llama-secondary/start|stop|restart`, or `/llama/force-kill`, gated by
  action settings.
- `request_ubuntu_server_power_action`: posts gated server/ESP actions such as
  `/esp/action`, `/esp/cancel`, `/reboot/now`, `/reboot/enable`,
  `/reboot/disable`, `/power/shutdown`, or `/diagnostics/handle-gpu-fault`.
  When `direct_esp=true`, power/reset/cancel actions go directly to the ESP
  endpoint because the Ubuntu Manager API is unavailable while the host is off.
  Requests like "turn BigBoss on" mean power on the Ubuntu llama.cpp host:
  use `action="power-on"` through the Manager when it is reachable, or
  `direct_esp=true` when the host is off and the Manager cannot respond.
- `configure_ubuntu_llama_instance`: patches `primary` or `secondary` via
  `/llama/instances/{id}/config`, supporting `model`, `model_flag`,
  `context_size`, `parallel_slots`, `command`, and `restart`.

This is the preferred route for changing llama.cpp context windows, for
example moving the secondary 2B instance from 8K to 16K or temporarily raising
the primary instance toward a larger context budget. The crisis manager receives
status, diagnosis, service restart, ESP power-cycle, and gated recovery tools;
context/model reconfiguration is reserved for the power/model-management
surface.

`parallel_slots=2` maps to llama.cpp `--parallel 2` through the Ubuntu Llama
Manager and is capped by `ALPHARAVIS_UBUNTU_LLAMA_PARALLEL_MAX` (default `2`).
Use it only for a safe VRAM window: the small 2B instance and short-context
BigBoss work can benefit, but high-context concurrent BigBoss work should roll
affected instances back to `parallel_slots=1` before the context-heavy jobs
start.

Destructive tools have a second confirmation guard. `power-on`, service
`start`, service `restart`, reboot timer enable/disable, and ESP cancel can run
when actions are enabled. `power-off`, `power-cycle`, `reset`, `shutdown`,
`reboot-now`, `gpu-fault`, service `stop`, and `force-kill` return
`needs_confirmation=true` unless the caller passes `confirmed=true` after the
operator has confirmed the exact target and tool.

## Managed Lifecycle

AlphaRavis distinguishes machines that were already online from machines it
woke only for the current request.

- For Pixelle, `prepare_comfy_for_pixelle` returns `woke_for_request` when it
  had to wake ComfyUI. If
  `ALPHARAVIS_PIXELLE_AUTO_SHUTDOWN_COMFY_AFTER_JOB=true`, the Pixelle tools
  schedule a delayed ComfyUI shutdown after job completion or failure. The
  shutdown is skipped when ComfyUI was already reachable before the job.
- For BigBoss/the Ubuntu llama.cpp host, the Power/Model Manager prompt treats
  `ALPHARAVIS_BIG_LLM_AUTO_SHUTDOWN_AFTER_MANAGED_RUN` as a policy marker:
  only shut down through Ubuntu Llama Manager after a run if the host was
  powered on specifically for that request and the configured idle delay passed.
  A host that was already running must be left alone.

Relevant settings:

```text
ALPHARAVIS_PIXELLE_AUTO_SHUTDOWN_COMFY_AFTER_JOB=false
ALPHARAVIS_PIXELLE_AUTO_SHUTDOWN_DELAY_SECONDS=600
ALPHARAVIS_BIG_LLM_AUTO_SHUTDOWN_AFTER_MANAGED_RUN=false
ALPHARAVIS_BIG_LLM_AUTO_SHUTDOWN_DELAY_SECONDS=600
```

## Dedicated Server Model Manager

LibreChat exposes a separate `Server Model Manager` preset using Bridge model
id `server-model-manager`. Bridge requests for that model are routed into the
same native LangGraph graph with:

```json
{
  "active_agent": "power_management_agent",
  "selected_toolsets": ["agent/power"],
  "server_model_manager_mode": true
}
```

Native LangGraph/DeepAgents callers can pass the same fields directly in graph
input; the manager is not exclusive to the Bridge.

The manager model policy is:

```text
ALPHARAVIS_SERVER_MODEL_MANAGER_MODEL=openai/server-model-manager
ALPHARAVIS_SERVER_MODEL_MANAGER_FALLBACK_MODEL=openai/edge-gemma
```

`openai/server-model-manager` is a LiteLLM route intended to prefer BigBoss and
fall back to Edge Gemma. Prompts and tool arguments for this agent are kept
intentionally small because the fallback model is limited and prone to wrong
tool choices.

## Embedding Window Logic

The default policy is:

```text
ALPHARAVIS_EMBEDDING_LOAD_POLICY=idle_or_big_llm_active
ALPHARAVIS_MODEL_IDLE_SECONDS=600
```

AlphaRavis recommends loading the embedding model only when:

- the system has been inactive for the configured idle window, or
- the big llama.cpp server is reachable, so chat work can stay on `big-boss`.

The planned sequence is:

1. Finish the current user-facing run.
2. Keep MongoDB/store as ground truth.
3. Queue pgvector indexing work.
4. During a safe window, unload the Ollama chat model if needed.
5. Load `ALPHARAVIS_OLLAMA_EMBED_MODEL`.
6. Run queued embedding jobs.
7. Restore the small Ollama chat/crisis model if needed.

## Pixelle And ComfyUI

Pixelle is the image job API. ComfyUI is the backend that may live on a machine
that is not always awake.

Before Pixelle starts, AlphaRavis can check ComfyUI when advanced model
management and Pixelle preflight are both enabled:

```text
ALPHARAVIS_ENABLE_MODEL_MANAGEMENT=true
ALPHARAVIS_ENABLE_ADVANCED_MODEL_MANAGEMENT=true
ALPHARAVIS_PIXELLE_PREPARE_COMFY=true
ALPHARAVIS_COMFY_HEALTH_URL=http://<comfy-ip>:8188/system_stats
```

If ComfyUI is unreachable:

- AlphaRavis warns by default.
- It can request a wake action through the curated action endpoint.
- It blocks the Pixelle job only when
  `ALPHARAVIS_PIXELLE_BLOCK_IF_COMFY_OFFLINE=true`.

## Interfaces Still To Populate

These are intentionally left as interfaces until the safe tools are curated:

- `ALPHARAVIS_MODEL_MGMT_ACTION_URL`: one HTTP endpoint that receives
  `{"action": "...", "payload": {...}}`.
- Optional: higher-level policies for when `configure_ubuntu_llama_instance`
  should raise or lower context automatically after a crisis or budget warning.
- `wake_pc`: wake a configured PC.
- `shutdown_pc`: safely shut down a configured PC.
- `start_service` / `stop_service`: service lifecycle controls.
- `load_embedding_model` / `unload_ollama_model`: Ollama model lifecycle.
- `run_embedding_jobs`: process queued pgvector embedding work.
- ComfyUI health URL for the real image backend.
- Crisis-manager routing node and retry-original-request logic.

## Owner Power Tools

Owner-specific tools live in:

```text
langgraph-app/owner_power_tools.py
```

They are enabled only when all of these are true:

```text
ALPHARAVIS_ENABLE_MODEL_MANAGEMENT=true
ALPHARAVIS_ENABLE_ADVANCED_MODEL_MANAGEMENT=true
ALPHARAVIS_ENABLE_OWNER_POWER_TOOLS=true
```

The file contains editable host/IP/MAC/start-command defaults derived from the
exported OpenWebUI tools. Real passwords are intentionally not committed; use:

```text
ALPHARAVIS_OWNER_SSH_PASS=<private password>
```

The helper uses `sshpass -e` so the password is supplied through the process
environment instead of a visible command-line argument. On Linux, install
`sshpass` on the host/container that runs these owner tools, or replace the
implementation with SSH keys in `owner_power_tools.py`.

Safe owner actions include:

- check llama server
- start/restart llama server
- read llama logs
- check/wake ComfyUI
- start all model services
- read Pixelle logs when Docker is reachable

Protected owner actions use the LangGraph human approval interrupt:

- shutdown llama server
- shutdown ComfyUI server

Future crisis-manager guard rails are already represented as ENV placeholders:

```text
ALPHARAVIS_CRISIS_MANAGER_MODEL=openai/edge-gemma
ALPHARAVIS_CRISIS_MAX_ATTEMPTS=1
ALPHARAVIS_CRISIS_TIMEOUT_SECONDS=120
ALPHARAVIS_CRISIS_AUTO_ACTIONS=check_llama_server|check_ollama_models|check_comfyui|start_llama_server|restart_llama_server|wake_pc
ALPHARAVIS_CRISIS_HITL_ACTIONS=shutdown_server|reboot_server|kill_process|delete_files
```

The power-management agent also uses a small model by default:

```text
ALPHARAVIS_POWER_MANAGER_MODEL=openai/edge-gemma
ALPHARAVIS_POWER_MANAGER_TIMEOUT_SECONDS=90
```

The agent should not invent SSH commands for these actions. It should either
use the curated endpoint, Wake-on-LAN, or transfer to the debugger where the
approval gate is active.
