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
- `response.content_part.done`
- `response.output_item.done`
- `response.completed`

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
| Fully non-streaming | `STREAMING=false`, `DISABLE_STREAMING=true` | Every internal model call waits for a complete response before LangChain continues. Stable, but no internal token stream. |
| Hybrid default | `STREAMING=true`, `DISABLE_STREAMING=tool_calling` | LangChain may stream calls without tools. Calls with tools are sent non-streaming so tool-call JSON is complete before execution. |
| Full streaming | `STREAMING=true`, `DISABLE_STREAMING=false` | Tool-bound model calls are also streamed. This is still experimental with the local LiteLLM/llama.cpp stack. |

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

LibreChat may still call `/v1/chat/completions` depending on its provider
adapter. That path remains available. If the active LibreChat provider supports
Responses directly, point it at `/v1/responses`; otherwise AlphaRavis still uses
the same LangGraph brain behind the Chat Completions compatibility endpoint.
