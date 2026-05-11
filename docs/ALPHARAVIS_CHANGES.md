# AlphaRavis Changes

This file records important local changes that affect runtime behavior,
compatibility, or operations. Keep detailed rationale here so future upgrades
can tell which patches are intentional and which ones can be removed.

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
