# AlphaRavis Responses Compatibility

AlphaRavis exposes an OpenAI-compatible Responses surface through
`langgraph-app/bridge_server.py`. The goal is honest compatibility: supported
features behave like Responses objects/events; unsupported OpenAI-hosted
features return explicit structured errors instead of being silently ignored.

## Implemented

- `POST /v1/responses`
  - foreground, non-streaming Response objects
  - `stream=true` semantic SSE events
  - `instructions`
  - string input
  - message-list input
  - text content parts
  - media/file content parts as safe metadata markers unless raw media context
    is explicitly enabled
  - tool-output input items as text context
  - `previous_response_id` when the previous response is still in the local
    bridge cache
  - `store`
  - `metadata`
  - usage estimates with `input_tokens_details`, `output_tokens_details`, and
    `total_tokens`
- `GET /v1/responses/{response_id}`
  - returns an explicit `retrieve_stream_not_supported` error for
    `?stream=true`; replay streaming of stored responses is not faked
- `DELETE /v1/responses/{response_id}`
- `GET /v1/responses/{response_id}/input_items`
- `POST /v1/responses/{response_id}/cancel`
  - returns a Response object only for in-progress/background responses
  - returns `response_not_cancellable` for completed foreground responses
- `POST /v1/responses/input_tokens`
  - returns bridge-side approximate input token counts

## Streaming Events

The bridge streams typed SSE events:

- `response.created`
- `response.in_progress`
- `response.output_item.added`
- `response.content_part.added`
- `response.output_text.delta`
- `response.output_text.done`
- `response.reasoning.delta`
- `response.reasoning.done`
- `response.function_call_arguments.delta`
- `response.function_call_arguments.done`
- `response.content_part.done`
- `response.output_item.done`
- `response.completed`

`response.output_text.delta` and `response.output_text.done` include
`logprobs: []` for LibreChat v0.8.5/Open Responses validation compatibility.
Visible reasoning is emitted as a real `type: "reasoning"` output item with
`reasoning_text` content. Internal LangGraph tool activity can be represented
as `function_call` and `function_call_output` output items; client-supplied
Responses tools are still not executed by the bridge.

For broad client compatibility the bridge can append `data: [DONE]` after the
semantic lifecycle events via `BRIDGE_RESPONSES_DONE_SENTINEL=true`.

## Explicitly Unsupported

These are not faked:

- `background=true`
- OpenAI Conversations via `conversation`
- `prompt` template references
- OpenAI-hosted client tools such as web search, file search, code interpreter,
  computer use, shell tools, or arbitrary client-supplied Responses tools
- non-text output modalities
- `text.format` values other than plain text
- encrypted `POST /v1/responses/compact`
- streaming retrieval via `GET /v1/responses/{response_id}?stream=true`

AlphaRavis has its own LangGraph tools, memory, RAG, compression, and archive
retrieval. Those features remain inside the graph rather than being exposed as
OpenAI-hosted Responses tools.

## Important Env Flags

```env
BRIDGE_ENABLE_RESPONSES_API=true
BRIDGE_PREFERRED_API_MODE=responses
BRIDGE_RESPONSES_STORE=true
BRIDGE_RESPONSES_STORE_MAX=200
BRIDGE_RESPONSES_DONE_SENTINEL=true
BRIDGE_RESPONSES_ALLOW_CLIENT_TOOLS=false
BRIDGE_STREAM_REASONING_EVENTS=true
BRIDGE_RESPONSES_STREAM_TOOL_EVENTS=true
BRIDGE_RESPONSES_STREAM_ACTIVITY_EVENTS=true
BRIDGE_RESPONSES_TOOL_OUTPUT_MAX_CHARS=8000
```

Keep `BRIDGE_RESPONSES_ALLOW_CLIENT_TOOLS=false` unless you intentionally want
to accept tool metadata without executing those tools. The safe default is to
reject unsupported client tool requests.

LangGraph-internal model calls have separate Responses flags:

```env
ALPHARAVIS_LLM_API_MODE=responses
ALPHARAVIS_RESPONSES_API_BASE=http://litellm:4000/v1
ALPHARAVIS_RESPONSES_MODEL=big-boss
ALPHARAVIS_DEEPAGENTS_API_MODE=responses
ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING=true
ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=tool_calling
```

AlphaRavis applies a startup patch equivalent to the important part of
langchain-ai/langchain PR #35457, so LangChain's documented
`disable_streaming="tool_calling"` hybrid no longer crashes with `AsyncStream`
when tool-bound Responses calls are routed through non-streaming code paths.
The bridge's external `/v1/responses` SSE stream is separate from the internal
DeepAgents model streaming setting.

Verified package state after the May 2026 update:

```text
langgraph-api:
  langchain-openai==1.2.1
  langchain==1.2.18
  langchain-core==1.3.3
  langgraph==1.1.10
  deepagents==0.5.9
  openai==2.36.0
  litellm==1.83.0

litellm proxy image:
  litellm==1.82.6
```

The DeepAgents upgrade did not by itself fix internal Responses streaming with
tool-capable LangChain calls. Before the local patch, a direct repro using
`ChatOpenAI(use_responses_api=True, streaming=True,
disable_streaming="tool_calling")` with a bound tool failed in
`langchain_openai` with:

```text
AttributeError: 'AsyncStream' object has no attribute 'error'
```

After applying the local PR #35457-style patch, the same direct repro passes
and the Bridge `/v1/responses` Agent Path returns streamed SSE output. Keep the
patch until `langchain-openai` ships the upstream fix and the repro still passes
without local modification.

## Internal Streaming Modes

There are three separate modes for DeepAgents Responses model calls:

| Mode | Env | What happens |
| --- | --- | --- |
| Fully non-streaming | `STREAMING=false`, `DISABLE_STREAMING=true`, `EXPERIMENTAL_BUFFER_TOOL_STREAMING=false` | Every internal model call waits for a complete response before LangChain continues. Stable, but no internal token stream. |
| Hybrid default | `STREAMING=true`, `DISABLE_STREAMING=tool_calling`, `EXPERIMENTAL_BUFFER_TOOL_STREAMING=false` | LangChain may stream calls without tools. Calls with tools are sent non-streaming so tool-call JSON is complete before execution. |
| Full streaming | `STREAMING=true`, `DISABLE_STREAMING=false`, `EXPERIMENTAL_BUFFER_TOOL_STREAMING=true` | Tool-bound model calls are also streamed. This passed the focused AlphaRavis LangChain/React-agent probe, but remains experimental as the default stack mode. |

Use the Makefile to set the matching `.env` values:

```bash
make streaming STREAMING=hybrid
make streaming STREAMING=full
make streaming STREAMING=nonstreaming
```

The Makefile also exposes Chat Completions runtime profiles. These switch both
direct calls and DeepAgents workers to ChatLiteLLM instead of LangChain
Responses:

| Profile | Env | What happens |
| --- | --- | --- |
| `chat-full` | `ALPHARAVIS_LLM_API_MODE=chat_completions`, `ALPHARAVIS_DEEPAGENTS_API_MODE=chat_completions`, `ALPHARAVIS_LLM_STREAMING=true` | Chat Completions path with LangGraph/LangChain message streaming enabled where the provider supports it. |
| `chat-nonstreaming` | same API mode values, `ALPHARAVIS_LLM_STREAMING=false` | Chat Completions path with internal streaming disabled. |

Use:

```bash
make streaming STREAMING=chat-full
make up-chat-fullstreaming
```

The hybrid mode is not "stream only until a tool call appears." LangChain cannot
know whether the next model response will contain a tool call before it asks the
model. It only knows whether tools are bound to the request. Therefore, if tools
are bound, `tool_calling` bypasses internal streaming for that whole model call.

This still improves the previous state:

- before the patch, hybrid mode crashed in `langchain_openai`
- after the patch, hybrid mode works and avoids malformed streamed tool calls
- external `/v1/responses` SSE events still work through the Bridge
- full internal tool-call streaming remains opt-in for future provider fixes

## LibreChat Notes

LibreChat has two intended AlphaRavis model specs:

- `AlphaRavis Responses`: `useResponsesApi: true`, reasoning summaries enabled,
  and the Open Responses event stream for reasoning/tool/activity UI.
- `AlphaRavis Chat`: the legacy `/v1/chat/completions` fallback path.

`librechat.yaml` uses config `version: 1.3.9`, keeps model selection and
parameters visible, and disables the deprecated presets UI so `modelSpecs`
remain the selected AlphaRavis entry points.

Live smoke on 2026-05-11 confirmed that the bridge emits LibreChat-compatible
Responses reasoning events, LangGraph node activity, and function-call/tool
output items through `POST /v1/responses` with `stream=true`.

Command approval interrupts are not exposed to LibreChat as native clickable
approval buttons on the external custom endpoint path. The bridge instead uses
chat-text approval for both Chat Completions and Responses requests:

```text
approve
reject
replace: <safer command>
approve always
immer erlauben
```

`approve always` / `immer erlauben` remembers only the exact
scope/target/command in the current LibreChat thread, in bridge process memory.
It is not a global allowlist and it is cleared by an `api-bridge` restart.
AionUI/ACP has a separate native `session/request_permission` permission UI;
LibreChat would need a custom frontend/backend permission event to get the same
one-click approval UX for AlphaRavis.

OpenAI-hosted reasoning models do not expose raw chain-of-thought through the
API; they expose reasoning summaries. Full visible thinking appears only when
the selected local/OpenAI-compatible provider emits visible reasoning fields
such as `reasoning_content`, `reasoning`, or `<think>` text.

For local llama.cpp-style models that stream visible thinking inside normal
string content, the bridge splits `<think>...</think>` and
`<thinking>...</thinking>` before emitting client events. The marker body is
sent as Responses reasoning or Chat Completions `reasoning_content`; the final
assistant text does not include the raw markers. Explicit provider reasoning
fields still take precedence so the same thinking is not emitted twice.
