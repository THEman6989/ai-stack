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
ALPHARAVIS_ENABLE_ADVANCED_MODEL_MANAGEMENT=false
ALPHARAVIS_ENABLE_CRISIS_MANAGER=false
ALPHARAVIS_ENABLE_OWNER_POWER_TOOLS=false
```

The standard stack uses `big-boss` through LiteLLM and does not create the
Power Management Agent. This keeps the repo usable for normal single-model
setups.

When `ALPHARAVIS_ENABLE_MODEL_MANAGEMENT=true`, AlphaRavis can inspect and plan:

```text
inspect_model_management_status
plan_embedding_maintenance
prepare_comfy_for_pixelle
request_power_management_action
```

Advanced hooks are separate:

```text
ALPHARAVIS_ENABLE_ADVANCED_MODEL_MANAGEMENT=true
```

That enables the `power_management_agent`, Pixelle ComfyUI preflight hooks, and
the future crisis-manager routing surface.

Real power/model actions are still disabled by default:

```text
ALPHARAVIS_ENABLE_POWER_MANAGEMENT=false
ALPHARAVIS_MODEL_MGMT_ALLOW_ACTIONS=false
ALPHARAVIS_MODEL_MGMT_ACTION_URL=
```

That means shutdowns, service restarts, Ollama unload/load actions, and
embedding-job runners return dry-run plans until a curated endpoint is provided.

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
