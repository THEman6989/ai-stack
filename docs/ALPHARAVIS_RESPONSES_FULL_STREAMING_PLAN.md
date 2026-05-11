# AlphaRavis Responses Full Streaming Plan

This plan is written for a fresh context. It contains the current state, what
was already fixed, what still needs investigation, and the exact next steps to
make Responses streaming with tools as good as possible without breaking tool
execution.

## Current Date And Workspace

- Date when written: 2026-05-11
- Repo: `/mnt/cc13def8-75e4-4260-ac61-e0008db37b92/@home/amin/experi/ai-satck-dev/ai-stack`
- Main runtime: Docker Compose
- Relevant services:
  - `langgraph-api`
  - `api-bridge`
  - `litellm`
  - external/local llama.cpp / Lamma2 Plus backend behind LiteLLM

## Current Working State

LangGraph/DeepAgents use Responses mode by default:

```text
ALPHARAVIS_LLM_API_MODE=responses
ALPHARAVIS_DEEPAGENTS_API_MODE=responses
ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING=true
ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=tool_calling
ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING=false
```

Meaning:

- no-tool model calls may stream
- model calls with tools are routed non-streaming by LangChain
- Bridge `/v1/responses` can still emit SSE events to clients
- full internal tool-call streaming remains experimental
- the experimental LangChain tool-streaming patch is available but disabled by
  default

## Important Files Already Changed

- `langgraph-app/patches/patch_langchain_openai_disable_streaming.py`
  - applies a startup patch to installed `langchain_openai.chat_models.base`
  - implements the important behavior from langchain-ai/langchain PR #35457
  - idempotent via marker:

```text
# AlphaRavis patch for langchain-ai/langchain#35436
```

- `docker-compose.yml`
  - runs the startup patch before `langgraph dev`
  - defaults DeepAgents Responses streaming to hybrid mode

- `langgraph-app/Dockerfile`
  - mirrors the same startup patch in image default command

- `.env(exaple)`
  - documents the hybrid streaming env defaults

- `langgraph-app/patches/patch_langchain_openai_responses_tool_streaming.py`
  - env-gated experimental patch for LangChain Responses tool streaming
  - only applies when `ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING=true`
  - upgrades an already partially patched running container in-place

- `langgraph-app/requirements.txt`
  - currently pins:

```text
langchain-openai==1.2.1
langchain==1.2.18
langchain-core==1.3.3
langgraph==1.1.10
deepagents==0.5.9
openai==2.36.0
```

- Documentation updated:
  - `docs/ALPHARAVIS_CHANGES.md`
  - `docs/ALPHARAVIS_RESPONSES_COMPATIBILITY.md`
  - `docs/ALPHARAVIS_USAGE_NOTES.md`
  - `docs/ALPHARAVIS_ARCHITECTURE.md`
  - `docs/ALPHARAVIS_OPEN_TASKS.md`

## What Was Fixed

Original failure:

```text
AttributeError: 'AsyncStream' object has no attribute 'error'
```

Cause:

- `streaming=True`
- `disable_streaming="tool_calling"`
- tools were bound
- LangChain routed to non-streaming `_generate` / `_agenerate`
- but `langchain-openai` still sent `stream=true` in the payload
- OpenAI SDK returned `AsyncStream`
- LangChain expected a completed response object

Fix applied:

- force `payload["stream"] = False` in non-streaming OpenAI code paths
- based on upstream PR:

```text
https://github.com/langchain-ai/langchain/pull/35457
```

Alternative smaller PR was reviewed but not chosen:

```text
https://github.com/langchain-ai/langchain/pull/35440
```

Root issue:

```text
https://github.com/langchain-ai/langchain/issues/35436
```

## Verified Results

After patch:

```text
patch_marker True
STREAMING=true
DISABLE=tool_calling
```

Direct repro passed:

```text
DIRECT_TOOL_STREAM_TEST_OK events=36
```

Bridge Responses Agent Path streaming passed:

```text
PATCHED_AGENT_STREAM_OK
```

Local tests:

```text
pytest -q tests
83 passed
```

## Important Conceptual Clarification

Hybrid mode is not full tool streaming.

`disable_streaming=tool_calling` means:

- if tools are bound to a model call, LangChain bypasses internal streaming for
  the whole call
- it cannot wait until a tool call appears, because it does not know in advance
  whether the model will produce text only or a tool call
- it only knows that tools are available on that request

The user's desired behavior is:

1. stream normal text/reasoning immediately
2. buffer tool-call chunks
3. execute the tool only after the complete tool-call JSON is available
4. continue the agent loop

This is conceptually correct, but it is not what the current DeepAgents /
LangChain hybrid mode does.

## Why Full Tool Streaming Needs Investigation

LangChain Python already has relevant machinery:

- `langchain_openai.chat_models.base._stream_responses`
- `langchain_openai.chat_models.base._astream_responses`
- `langchain_openai.chat_models.base._convert_responses_chunk_to_generation_chunk`
- `AIMessageChunk.tool_call_chunks`
- `AIMessageChunk.__add__`
- final parsing of `tool_call_chunks` into `tool_calls`

The critical Responses event path is:

```text
response.output_text.delta
response.output_item.added where item.type == function_call
response.function_call_arguments.delta
response.output_item.done
response.completed
```

Known local risk from prior live tests:

```text
item['content'] is empty
```

This may come from LiteLLM, llama.cpp, the OpenAI SDK object shape, or
LangChain conversion/aggregation. Do not assume the bug is only in LangChain.

## LangChain.js Status

There does not appear to be the exact same Python `AsyncStream` issue in
LangChain.js, because that was Python OpenAI SDK behavior plus Python
`langchain-openai` payload construction.

However, LangChain.js has related streaming/tool-call/Responses issues:

- `https://github.com/langchain-ai/langchainjs/issues/8049`
  - OpenAI Responses API streaming tool-call arguments parsed incorrectly
  - closed by `#8107`
- `https://github.com/langchain-ai/langchainjs/issues/8577`
  - Responses API streaming does not work in LangGraph because callbacks were
    not receiving token events
- `https://github.com/langchain-ai/langchainjs/issues/8518`
  - streaming toolcall response missing for no-parameter functions
- `https://github.com/langchain-ai/langgraphjs/issues/1289`
  - parallel tool calls in streaming mode malformed/merged
- `https://github.com/langchain-ai/langgraphjs/issues/1667`
  - tool call fails in LangGraph Studio because messages stream mode cannot be
    disabled

Conclusion: JS has the same class of problem, not necessarily the same exact
bug. Streaming tool calls are tricky across both Python and JS.

## Local LangChain Source Already Cloned

During investigation, upstream Python LangChain was cloned to:

```text
/tmp/langchain-upstream
```

Branches fetched:

```text
pr-35440
pr-35457
dotuananh-fix
anandesh-fix
```

If `/tmp` was cleared, reclone/fetch:

```bash
git clone --filter=blob:none https://github.com/langchain-ai/langchain.git /tmp/langchain-upstream
cd /tmp/langchain-upstream
git fetch origin pull/35440/head:pr-35440
git fetch origin pull/35457/head:pr-35457
git fetch https://github.com/dotuananh0712/langchain.git fix/disable-streaming-tool-calling:dotuananh-fix
git fetch https://github.com/Anandesh-Sharma/langchain.git fix/disable-streaming-tool-calling-35436:anandesh-fix
```

## Next Goal

Determine whether full internal Responses tool streaming can be made safe for
AlphaRavis:

```text
ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING=true
ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=false
```

The goal is not to execute partial tool calls. The goal is:

- text chunks visible as early as possible
- tool-call chunks buffered
- complete tool calls executed only after final assembled JSON
- no malformed `invalid_tool_calls`
- no `item['content'] is empty`
- no duplicate final text in Bridge SSE

## Exact Investigation Plan

### 1. Keep Current Hybrid Mode Stable

Do not remove the existing PR #35457-style startup patch.

Before experiments, record baseline:

```bash
docker exec langgraph-api sh -lc 'printf "STREAMING=$ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING\nDISABLE=$ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING\n"'
docker exec langgraph-api sh -lc 'python - << "PY"
from pathlib import Path
import langchain_openai.chat_models.base as base
text = Path(base.__file__).read_text()
print("patch_marker", "AlphaRavis patch for langchain-ai/langchain#35436" in text)
print(base.__file__)
PY'
pytest -q tests
```

Expected:

```text
STREAMING=true
DISABLE=tool_calling
patch_marker True
83 passed
```

### 2. Add A Focused Full-Streaming Probe Script

Create a script, preferably:

```text
scripts/probe_responses_tool_streaming.py
```

Status: implemented. The script writes `low_level_responses_sse.jsonl`,
`langchain_no_tool_astream.jsonl`,
`langchain_react_agent_astream_events.jsonl`, and `summary.json` into a
timestamped run directory below the artifact root.

It should run inside `langgraph-api` or against the same container environment.

The script should test direct `ChatOpenAI`, not the full AlphaRavis graph first:

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def marker_tool(value: str) -> str:
    """Return a marker value."""
    return "TOOL_RETURN_" + value
```

Model config:

```python
ChatOpenAI(
    model="big-boss",
    base_url="http://litellm:4000/v1",
    api_key="sk-local-dev",
    streaming=True,
    disable_streaming=False,
    use_responses_api=True,
    max_retries=0,
    timeout=120,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
```

The script must collect:

- raw LangChain events from `agent.astream_events(..., version="v2")`
- every `AIMessageChunk`
- `content`
- `tool_call_chunks`
- `tool_calls`
- `invalid_tool_calls`
- response metadata
- exceptions and tracebacks

Persist JSONL artifacts under:

```text
artifacts/alpharavis/responses_streaming_probe/
```

### 3. Add A Lower-Level OpenAI SDK Probe

Also probe LiteLLM/llama.cpp directly through the OpenAI Python SDK or `httpx`
against:

```text
http://litellm:4000/v1/responses
```

Use `stream=true` and one forced/simple tool if possible. Capture raw SSE/event
objects before LangChain conversion.

Purpose:

- decide if malformed data starts before LangChain
- compare raw provider events to LangChain chunks

Status: implemented through `httpx` SSE capture in
`scripts/probe_responses_tool_streaming.py`.

Observed result from `codex_probe_20260511_repo_artifacts`:

```text
bucket: provider_litellm_or_openai_sdk
direct /v1/responses stream: HTTP 408 from LiteLLM after 30 seconds
LangChain no-tool Responses stream: same HTTP 408
LangChain create_react_agent Responses stream: same HTTP 408
```

Artifacts:

```text
artifacts/alpharavis/responses_streaming_probe/codex_probe_20260511_repo_artifacts/
```

This means the current failure happens before LangChain can assemble tool-call
chunks. Do not enable `ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=false`
as a default from this result.

Updated result after restarting the local Lamma/LAMMPS backend:

```text
run_id: codex_probe_after_lamma_restart_classified_20260511
bucket: langchain_openai_conversion
direct /v1/responses stream: ok, function-call events observed
LangChain no-tool Responses stream: ok
LangChain create_react_agent Responses stream: item['content'] is empty
```

That result made the LangChain Responses conversion layer the correct
experimental patch point.

### 4. Classify Failure

Classify any failure into one of these buckets:

1. Provider/LiteLLM/llama.cpp sends malformed Responses events.
2. OpenAI Python SDK object shape differs from what LangChain expects.
3. `langchain-openai` conversion creates malformed `AIMessageChunk`.
4. `AIMessageChunk` aggregation parses partial JSON too early into
   `invalid_tool_calls`.
5. LangGraph/DeepAgents agent loop sees partial chunks and reacts incorrectly.
6. Bridge SSE extraction emits duplicate or incomplete text/tool data.

Do not patch before the bucket is clear.

### 5. Candidate Patch Points

Patch only with an env gate, for example:

```text
ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING=true
```

Possible patch locations:

1. `langchain_openai.chat_models.base._convert_responses_chunk_to_generation_chunk`
   - suppress `invalid_tool_calls` during partial tool argument chunks
   - ensure final `response.output_item.done` emits one complete tool call
   - status: implemented experimentally in
     `langgraph-app/patches/patch_langchain_openai_responses_tool_streaming.py`
2. Wrapper around `ChatOpenAI._astream_responses`
   - buffer tool chunks internally
   - yield text chunks immediately
   - only yield tool chunks after complete item done
3. AlphaRavis model wrapper in `agent_graph.py`
   - safer than globally patching LangChain
   - harder because DeepAgents expects normal LangChain model interfaces
4. Two-phase AlphaRavis orchestration
   - keep tool turns hybrid/non-streaming
   - after tools finish, run a final no-tool answer call and stream that
   - safest user-visible improvement, less invasive than full tool streaming

### 6. Preferred Implementation Order

1. Probe and log raw events.
2. If raw events are malformed, do not patch LangChain. Keep hybrid mode and
   pursue two-phase final-answer streaming.
3. If raw events are clean but LangChain chunks are malformed, patch
   `langchain-openai` conversion with an AlphaRavis startup patch.
4. If LangChain chunks are clean but DeepAgents reacts incorrectly, avoid global
   LangChain patching and implement two-phase final-answer streaming.
5. Only make full streaming default after:
   - direct full streaming probe passes
   - agent full streaming probe passes
   - Bridge `/v1/responses` stream passes
   - no `invalid_tool_calls`
   - no `item['content'] is empty`
   - `pytest -q tests` passes

Current experimental patch probe result:

```text
run_id: codex_probe_experimental_patch_v5_no_force_20260511
bucket: not_reproduced
direct /v1/responses stream: ok, function-call events observed
LangChain no-tool Responses stream: ok
LangChain create_react_agent Responses stream: ok
invalid_tool_calls: 0
marker_tool_ends: 1
```

After the Makefile/setup flow was updated and the local `.env` was switched to
`STREAMING=full`, the recreated runtime also passed:

```text
run_id: codex_probe_makefile_fullstreaming_v2_20260511
bucket: not_reproduced
direct /v1/responses stream: ok, function-call events observed
LangChain no-tool Responses stream: ok
LangChain create_react_agent Responses stream: ok
invalid_tool_calls: 0
marker_tool_ends: 1
```

The low-level probe with `--force-tool-choice` did not observe raw tool-call
events on this local provider, even though the LangChain agent path passed. Do
not treat forced `tool_choice` as the validation source until the LiteLLM /
llama.cpp Responses layer enforces it consistently.

Operational shortcuts added after the probe:

```bash
make streaming STREAMING=full
make install-fullstreaming
make up-fullstreaming
make streaming STREAMING=chat-full
make up-chat-fullstreaming
```

These targets write the required `.env` combination, including
`ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING=true` for Responses full
streaming or `ALPHARAVIS_LLM_STREAMING=true` for Chat Completions streaming,
instead of requiring manual edits.

### 7. Acceptance Tests

Add or run these checks:

```text
Direct ChatOpenAI Responses no-tool streaming returns text chunks.
Direct ChatOpenAI Responses tool streaming assembles one valid tool call.
Direct create_react_agent full streaming executes marker_tool exactly once.
Bridge /v1/responses stream=true Agent Path returns output_text.delta chunks.
Bridge /v1/responses stream=true has no duplicate final text.
Bridge /v1/chat/completions stream=true still works.
pytest -q tests == 83 passed or better.
```

### 8. Do Not Break These Existing Behaviors

- `BRIDGE_RESPONSES_ALLOW_CLIENT_TOOLS=false` remains safe default.
- Client-supplied Responses tools are not executed by the Bridge.
- Internal LangGraph tools still work.
- Fast Path still works.
- `edge-gemma` fallback is still known broken and should not be confused with
  this streaming work.
- Do not revert unrelated dirty worktree entries:
  - `hermes-agent`
  - `.cache/`
  - `old cobvsresation-chaty.txt`
  - `scripts/aionui_alpharavis_docker.sh`

## Practical Recommendation

Start with probes, not patches. The next implementation step should be
instrumentation that records raw Responses events and LangChain chunks for the
same tool-call request. Once the exact failure point is known, the patch can be
small and defensible.

If the provider stream is not clean, use the two-phase final-answer streaming
design instead of trying to repair partial tool-call streams.
