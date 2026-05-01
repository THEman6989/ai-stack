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

## LibreChat Notes

LibreChat may still call `/v1/chat/completions` depending on its provider
adapter. That path remains available. If the active LibreChat provider supports
Responses directly, point it at `/v1/responses`; otherwise AlphaRavis still uses
the same LangGraph brain behind the Chat Completions compatibility endpoint.
