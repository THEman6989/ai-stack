# AlphaRavis Changes

This file records important local changes that affect runtime behavior,
compatibility, or operations. Keep detailed rationale here so future upgrades
can tell which patches are intentional and which ones can be removed.

## 2026-05-11 - LibreChat Command Approval Memory

### Summary

LibreChat command approvals still use the chat-text fallback because the
external custom endpoint path does not expose an AlphaRavis-native clickable
permission callback.

Accepted replies while a command approval interrupt is pending:

```text
approve
reject
replace: <safer command>
approve always
immer erlauben
```

`approve always` / `immer erlauben` stores a bridge-local allow entry for the
exact scope/target/command in the current LibreChat thread only. It is not a
global allowlist and is cleared when `api-bridge` restarts.

### Why This Was Needed

OpenAI Responses has MCP approval request/response items for remote MCP tools,
and AionUI/ACP has a native `session/request_permission` flow. LibreChat's
custom OpenAI-compatible endpoint path does not provide that same AlphaRavis
permission callback, so the robust path is to keep text approvals and make the
"remember this exact command in this chat" case explicit.

### Files Changed

- `langgraph-app/bridge_server.py`
  - parses `approve always` / `immer erlauben`
  - stores exact command fingerprints in process memory per thread
  - auto-resumes only when the same pending interrupt reappears in that thread
- `tests/test_bridge_responses.py`
  - covers command-memory parsing and exact-command matching
- Responses, usage, architecture, and open-task docs
  - document the LibreChat limitation and the supported fallback commands

### Verification

```text
pytest -q tests/test_bridge_responses.py tests/test_alpharavis_acp_adapter.py
36 passed

pytest -q tests
110 passed
```

## 2026-05-11 - Responses / DeepAgents Streaming Fix

### Summary

AlphaRavis now runs LangGraph/DeepAgents through the Responses API by default
and enables LangChain's hybrid streaming mode for DeepAgents:

```text
ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING=true
ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=tool_calling
```

The hybrid mode means:

- model calls without bound tools may stream tokens
- model calls with tools are routed through non-streaming model calls
- the Bridge can still expose `/v1/responses` SSE events to clients

This is different from full streaming. Full streaming with
`ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=false` still sends streaming
tool-capable model calls to the provider and remains experimental for the local
LiteLLM/llama.cpp stack.

### Why This Was Needed

LangChain documents `disable_streaming="tool_calling"` as a way to bypass
streaming when tools are passed. In the installed `langchain-openai==1.2.1`,
that routing reached non-streaming `_generate` / `_agenerate` code paths, but
the payload still included `stream=true`. The OpenAI client then returned a
Stream/AsyncStream object where LangChain expected a completed response object.

Observed failure:

```text
AttributeError: 'AsyncStream' object has no attribute 'error'
```

The issue is tracked upstream:

```text
https://github.com/langchain-ai/langchain/issues/35436
```

### Fix Choice

Two upstream PR/fork fixes were compared locally:

- `https://github.com/langchain-ai/langchain/pull/35440`
  - small three-line fix in `_get_invocation_params`
- `https://github.com/langchain-ai/langchain/pull/35457`
  - forces `payload["stream"] = False` in `_generate` and `_agenerate`
  - includes regression tests
  - documents that `tool_calling` disables streaming for all calls while tools
    are bound, not only calls that eventually produce tool calls

AlphaRavis applies the PR #35457 approach because it patches the concrete
non-streaming code paths that crashed in the local repro.

### Files Changed

- `langgraph-app/patches/patch_langchain_openai_disable_streaming.py`
  - startup patch for `langchain_openai.chat_models.base`
  - idempotent; exits if already applied
- `docker-compose.yml`
  - runs the patch before `langgraph dev`
  - default DeepAgents Responses streaming set to hybrid mode
- `langgraph-app/Dockerfile`
  - mirrors the same startup patch for image defaults
- `.env(exaple)`
  - documents the new streaming flags and defaults
- `langgraph-app/requirements.txt`
  - updated/pinned LangChain, LangGraph, DeepAgents, OpenAI package versions
- Responses/usage/architecture docs
  - document the patch, current limitations, and verification results

### Verification

Package state in `langgraph-api` after update:

```text
langchain-openai==1.2.1
langchain==1.2.18
langchain-core==1.3.3
langgraph==1.1.10
deepagents==0.5.9
openai==2.36.0
litellm==1.83.0
```

Runtime checks:

```text
patch_marker True
STREAMING=true
DISABLE=tool_calling
```

Direct repro after patch:

```text
DIRECT_TOOL_STREAM_TEST_OK events=36
```

Bridge `/v1/responses` Agent Path streaming after patch:

```text
PATCHED_AGENT_STREAM_OK
```

Local tests:

```text
83 passed
```

### Remaining Limitations

- Full internal Responses streaming with tools
  `ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=false` is still
  experimental.
- Hybrid mode cannot know in advance whether the model will produce a tool call.
  It only knows whether tools are bound to the call. Therefore, when tools are
  bound, LangChain bypasses internal token streaming for that whole model call.
- The external Bridge SSE stream can still emit final output text as Responses
  events. This is not the same as token-by-token provider streaming for every
  internal tool-capable model call.
- Remove the startup patch only after `langchain-openai` ships the upstream fix
  and the direct tool-calling repro passes without local modification.

### Future Design: Text Streaming With Safe Tool Execution

The safest design for "show text while preserving reliable tools" is not full
streaming of tool-call chunks. Tool-call arguments are structured JSON and may
arrive split across many chunks; executing them before the final chunk is unsafe.

A safer two-phase design would be:

1. Run tool-capable planner/worker turns with
   `disable_streaming="tool_calling"` so tool calls are complete before
   execution.
2. Execute tools only after the complete model response is available.
3. After tools finish, run a final answer model call without bound tools and
   stream that text token-by-token to the UI.

This would allow visible final-answer token streaming while keeping tool calls
reliable. It requires changes in the agent orchestration layer, because current
DeepAgents/React-style workers keep tools bound on every model call, including
the final answer turn.

Detailed follow-up plan:

```text
docs/ALPHARAVIS_RESPONSES_FULL_STREAMING_PLAN.md
```

## 2026-05-11 - Responses Full-Streaming Probe Instrumentation

### Summary

The follow-up plan now has concrete instrumentation in:

```text
scripts/probe_responses_tool_streaming.py
```

The script records both sides of the suspected failure boundary before any
runtime patch is attempted:

- raw `/v1/responses` SSE events from LiteLLM/llama.cpp
- direct LangChain `ChatOpenAI(... use_responses_api=True)` no-tool streaming
- LangGraph `create_react_agent(...).astream_events(..., version="v2")`
  chunks with `content`, `tool_call_chunks`, `tool_calls`,
  `invalid_tool_calls`, metadata, exceptions, and tracebacks

Artifacts are written under:

```text
artifacts/alpharavis/responses_streaming_probe/<run-id>/
```

That directory is ignored by git because the probe may capture prompts, model
outputs, provider headers, and tracebacks.

### Usage

Inside `langgraph-api`:

```bash
python /workspace/scripts/probe_responses_tool_streaming.py
```

From the host against the exposed LiteLLM port:

```bash
python scripts/probe_responses_tool_streaming.py --base-url http://127.0.0.1:4000/v1
```

The script exits with `0` only if all enabled probes pass. It still writes
`summary.json` and JSONL artifacts when a probe fails, so failed runs are the
expected input for deciding the next patch point.

Current probe result:

```text
run_id: codex_probe_20260511_repo_artifacts
classification: provider_litellm_or_openai_sdk
low_level_responses_sse: HTTP 408 from LiteLLM after 30 seconds
langchain_no_tool_astream: same HTTP 408
langchain_react_agent_astream_events: same HTTP 408
```

The actionable conclusion is to keep the existing hybrid runtime mode and not
apply a LangChain tool-stream buffering patch yet. The provider stream must be
fixed or bypassed first.

Follow-up after restarting the local Lamma/LAMMPS backend:

```text
run_id: codex_probe_after_lamma_restart_classified_20260511
classification: langchain_openai_conversion
raw /v1/responses SSE: ok, function-call events observed
LangChain no-tool Responses stream: ok
LangChain create_react_agent Responses stream: failed with item['content'] is empty
```

That narrowed the failure from provider availability to LangChain's Responses
stream chunk conversion/aggregation.

### Experimental Full-Streaming Patch

An env-gated patch now exists at:

```text
langgraph-app/patches/patch_langchain_openai_responses_tool_streaming.py
```

It is disabled by default and only applies when:

```text
ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING=true
```

The patch keeps the production default hybrid mode unchanged, but allows
explicit experiments with:

```text
ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=false
```

Patch behavior:

- keeps provider-reused `function_call` output indexes separate from the prior
  reasoning item
- emits final reasoning content from `response.output_item.done`
- suppresses partial `response.function_call_arguments.delta` chunks so
  incomplete JSON is not parsed as `invalid_tool_calls`
- emits one complete LangChain tool call when `response.output_item.done` for
  the function call arrives
- upgrades the earlier partial experimental patch in-place if a running
  container already has it

Verification after applying the experimental patch inside `langgraph-api`:

```text
run_id: codex_probe_experimental_patch_v5_no_force_20260511
classification: not_reproduced
raw /v1/responses SSE: ok, function-call events observed
LangChain no-tool Responses stream: ok
LangChain create_react_agent Responses stream: ok
invalid_tool_calls: 0
marker_tool_ends: 1
```

The low-level probe with `--force-tool-choice` produced no raw tool-call events
on this local provider, while the LangChain agent path still passed. Use the
default non-forced probe as the local validation path unless LiteLLM/llama.cpp
starts enforcing Responses `tool_choice` consistently.

The hybrid runtime defaults remain unchanged until Bridge-level full-streaming
checks also pass and the experimental patch has more soak time.

## 2026-05-11 - Makefile Install And Streaming Profile Refresh

### Summary

The Makefile install flow has been updated so a fresh local setup can choose
the current AlphaRavis runtime mode instead of inheriting old defaults.

New Makefile flows:

```bash
make install
make update
make install-fullstreaming
make install-chat-fullstreaming
make profiles
make streaming STREAMING=full
make streaming STREAMING=chat-full
make up-fullstreaming
make up-chat-fullstreaming
make status
```

`make install` now delegates to `scripts/alpharavis_setup.py` with explicit
install options for:

- runtime API/streaming profile
- submodule initialization
- Docker Compose profiles such as `openwebui`
- optional image build
- optional stack start

The script writes `.env` directly through the existing safe key-update helper,
so interactive and non-interactive Makefile targets use the same behavior.

### Streaming Profiles

The setup helper now supports these profiles:

```text
responses-hybrid       -> stable default Responses mode
responses-full         -> experimental full Responses tool streaming
responses-nonstreaming -> Responses mode without internal streaming
chat-full              -> Chat Completions with ChatLiteLLM streaming enabled
chat-nonstreaming      -> Chat Completions with ChatLiteLLM streaming disabled
```

Aliases remain available for short Makefile use:

```text
hybrid       -> responses-hybrid
full         -> responses-full
nonstreaming -> responses-nonstreaming
chat         -> chat-full
```

The `responses-full` profile sets:

```text
ALPHARAVIS_LLM_API_MODE=responses
ALPHARAVIS_LLM_STREAMING=true
ALPHARAVIS_DEEPAGENTS_API_MODE=responses
ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING=true
ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=false
ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING=true
BRIDGE_ENABLE_RESPONSES_API=true
BRIDGE_PREFERRED_API_MODE=responses
```

The `responses-hybrid` profile keeps the stable runtime:

```text
ALPHARAVIS_LLM_API_MODE=responses
ALPHARAVIS_LLM_STREAMING=true
ALPHARAVIS_DEEPAGENTS_API_MODE=responses
ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING=true
ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=tool_calling
ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING=false
```

The `chat-full` profile sets:

```text
ALPHARAVIS_LLM_API_MODE=chat_completions
ALPHARAVIS_LLM_STREAMING=true
ALPHARAVIS_DEEPAGENTS_API_MODE=chat_completions
BRIDGE_PREFERRED_API_MODE=chat_completions
```

`make update` now uses the same profile menu as `make install`, updates
submodules by default, and runs `docker compose up -d --build` by default after
the update. `make update-no-start` keeps the update/build flow but does not
start the stack.

### Files Changed

- `Makefile`
  - added `help`, `streaming`, `fullstreaming`, `hybrid-streaming`,
    `nonstreaming`, `chat-completions`, `chat-fullstreaming`,
    `chat-nonstreaming`, `install-fullstreaming`, `install-hybrid`,
    `install-nonstreaming`, `install-chat`, `install-chat-fullstreaming`,
    `install-chat-nonstreaming`, `profiles`, `update-no-start`,
    `up-fullstreaming`, and `up-chat-fullstreaming`
  - `make install` now accepts `STREAMING`, `SUBMODULES`, `BUILD`, `START`, and
    `PROFILES`
  - `make update` now accepts `UPDATE_STREAMING`, `UPDATE_SUBMODULES`,
    `UPDATE_BUILD`, `UPDATE_START`, and `UPDATE_PROFILES`
- `scripts/alpharavis_setup.py`
  - added streaming-profile application and status reporting
  - added numbered terminal profile selection with an info view showing exact
    env values
  - added Compose profile persistence through `COMPOSE_PROFILES`
  - added install/update-time build/start orchestration
- `.env(exaple)`
  - added documented `COMPOSE_PROFILES`
  - moved `ALPHARAVIS_LLM_STREAMING` into the main model route section because
    runtime profiles update it directly
- `README.md`, `docs/ALPHARAVIS_ARCHITECTURE.md`,
  `docs/ALPHARAVIS_USAGE_NOTES.md`, and
  `docs/ALPHARAVIS_RESPONSES_COMPATIBILITY.md`
  - document the current install and streaming architecture
- `tests/test_alpharavis_setup.py`
  - covers full-streaming env values, env update behavior, mode detection, and
    Compose profile normalization

### Verification

Local verification after the Makefile/setup changes:

```text
pytest -q tests -> 101 passed
docker compose config --quiet -> ok
git diff --check -> ok
py_compile setup/probe/test files -> ok
```

The local `.env` was then set through the same helper used by Makefile:

```bash
python scripts/alpharavis_setup.py streaming --streaming-mode full
docker compose up -d --force-recreate langgraph-api api-bridge
```

Container ENV after recreate:

```text
ALPHARAVIS_LLM_API_MODE=responses
ALPHARAVIS_DEEPAGENTS_API_MODE=responses
ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING=true
ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=false
ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING=true
BRIDGE_ENABLE_RESPONSES_API=true
BRIDGE_PREFERRED_API_MODE=responses
```

Full-streaming probe after the Makefile/ENV activation:

```text
run_id: codex_probe_makefile_fullstreaming_v2_20260511
classification: not_reproduced
raw /v1/responses SSE: ok, function-call events observed
LangChain no-tool Responses stream: ok
LangChain create_react_agent Responses stream: ok
invalid_tool_calls: 0
marker_tool_ends: 1
```

## 2026-05-11 - Hermes-Agent Local Patch Handling

### Summary

AlphaRavis keeps upstream `hermes-agent` as a submodule, but local fixes that
are needed for this stack live in the parent repo under:

```text
patches/hermes-agent/
```

The Docker containers apply those patches automatically at startup through:

```text
scripts/apply_hermes_agent_patches.sh
```

`docker-compose.yml` builds Hermes from the upstream submodule, mounts the
parent repo read-only at `/workspace`, and uses
`scripts/hermes_patched_entrypoint.sh` as a wrapper around the original Hermes
entrypoint. The wrapper runs the patch script against `/opt/hermes`, then hands
control back to `/opt/hermes/docker/entrypoint.sh`.

```text
alpharavis/hermes-agent:local
```

### Why This Exists

Submodule commits must exist in their own upstream repository. If the parent
repo points at a local-only `hermes-agent` commit, other machines and GitHub
cannot reproduce the checkout. Storing AlphaRavis-specific changes as parent
repo patches keeps the submodule clean and makes local changes reproducible.

### Current Hermes Patch

```text
patches/hermes-agent/kanban-db-duplicate-column-guard.patch
```

This patch makes Hermes kanban optional-column migrations tolerate SQLite
`duplicate column name` races/errors for these columns:

- `consecutive_failures`
- `worker_pid`
- `last_failure_error`

If Hermes kanban/task startup fails around duplicate SQLite columns, check this
patch and the Hermes startup patch command first.

### Manual Development Helper

For local debugging outside Docker, apply the same patch directly to the
submodule with:

```bash
scripts/apply_hermes_agent_patches.sh
```

The normal Docker path does not require this manual step.
