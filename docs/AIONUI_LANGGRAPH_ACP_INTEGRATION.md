# AionUi + AlphaRavis ACP Integration

AlphaRavis has a separate ACP adapter for AionUi custom agents:

```bash
python /workspace/langgraph-app/alpharavis_acp_adapter.py
```

This adapter is intentionally separate from the OpenAI-compatible FastAPI
bridge. LibreChat keeps using `api-bridge` on port `8123` and
`/v1/chat/completions`; AionUi talks to the stdio ACP adapter, and the adapter
talks to `langgraph-api` on port `2024`.

## Setup In AionUi

1. Open AionUi custom agents / ACP agent configuration.
2. Add a new custom agent named `AlphaRavis LangGraph`.
3. Use one of these launch styles.

Docker/container path:

```text
Command: python
Args: /workspace/langgraph-app/alpharavis_acp_adapter.py
```

Windows checkout path:

```text
Command: python
Args: C:\experi\ai\ai-stack\langgraph-app\alpharavis_acp_adapter.py
```

Windows helper:

```text
Command: C:\experi\ai\ai-stack\scripts\run_alpharavis_acp_adapter.cmd
Args:
```

Linux helper:

```text
Command: /workspace/scripts/run_alpharavis_acp_adapter.sh
Args:
```

## URLs

When AionUi runs on the host and LangGraph is published locally:

```env
LANGGRAPH_API_URL=http://localhost:2024
```

When the adapter runs inside the Docker network:

```env
LANGGRAPH_API_URL=http://langgraph-api:2024
```

The graph id stays:

```env
LANGGRAPH_ASSISTANT_ID=alpha_ravis
```

## Environment

```env
ALPHARAVIS_ACP_WORKSPACE=/workspace
ALPHARAVIS_ACP_TOOL_OUTPUT_MAX_CHARS=8000
ALPHARAVIS_ACP_TRACE_DETAIL=summary
ALPHARAVIS_ACP_SCRUB_INTERNAL_CONTEXT=true
ALPHARAVIS_ACP_DEBUG_IO=false
ALPHARAVIS_ACP_STREAM_HEARTBEAT_SECONDS=5
ALPHARAVIS_ACP_DEBUG_EVENT_PAYLOAD_CHARS=12000
ALPHARAVIS_ACP_DEBUG_STATUS_TO_AION=true
ALPHARAVIS_ACP_ALLOW_FILE_WRITES=false
ALPHARAVIS_ACP_SEND_AVAILABLE_COMMANDS=true
ALPHARAVIS_ACP_RUN_TIMEOUT_SECONDS=300
```

`ALPHARAVIS_ACP_TRACE_DETAIL` values:

- `summary`: short LangGraph node/status summaries
- `debug`: includes additional non-message event names and reasoning summaries
- `off`: only assistant text/toolcards

Debug-related behavior:

- `ALPHARAVIS_ACP_DEBUG_IO=true` logs raw JSON-RPC in/out packets to `stderr`.
- `ALPHARAVIS_ACP_TRACE_DETAIL=debug` logs structured timing traces to `stderr`.
- `ALPHARAVIS_ACP_STREAM_HEARTBEAT_SECONDS` controls how often the adapter logs
  that it is still waiting for the LangGraph/LLM stream.
- `ALPHARAVIS_ACP_DEBUG_STATUS_TO_AION=true` mirrors short wait/status updates
  into AionUi as `agent_thought_chunk` when trace detail is `debug`.
- `ALPHARAVIS_ACP_DEBUG_EVENT_PAYLOAD_CHARS` caps large raw event fields in
  debug logs.

## Debugging The Handshake

Set:

```env
ALPHARAVIS_ACP_DEBUG_IO=true
```

The adapter logs incoming and outgoing JSON-RPC packets to `stderr` only. It
never writes debug text to `stdout`, because stdout must remain pure JSON-RPC for
AionUi. Debug packets are secret-scrubbed before printing.

For a full timing trace while debugging slow first-token behavior, set:

```env
ALPHARAVIS_ACP_DEBUG_IO=true
ALPHARAVIS_ACP_TRACE_DETAIL=debug
ALPHARAVIS_ACP_STREAM_HEARTBEAT_SECONDS=5
```

The `stderr` trace then shows:

- JSON-RPC receive/dispatch/write timings
- request start/end timings
- prompt start/end timings
- LangGraph client creation and thread ensure timings
- each LangGraph stream event with event gaps
- `langgraph.stream.waiting` heartbeats while no event has arrived
- `langgraph.stream.first_text_delta` when the first visible token arrives
- tool start/end, permission request/response, and cancel timings

If AionUi appears to wait for 40 seconds before the first answer, look for
`langgraph.thread.ensure.end`, repeated `langgraph.stream.waiting`, and
`langgraph.stream.first_text_delta`. Those three timestamps show whether the
delay is connection setup, LangGraph scheduling, model first-token latency, or
tool/approval waiting.

You should see AionUi send:

```text
initialize
session/new
session/prompt
```

and AlphaRavis answer with normal JSON-RPC responses plus `session/update`
notifications.

## Supported Incoming ACP Methods

- `initialize`
- `session/new`
- `session/load`
- `session/prompt`
- `session/send_message`
- `session/cancel`
- `session/close`
- `session/set_config_option`
- `session/response`
- `session/permission_response`
- `fs/read_text_file`
- `fs/write_text_file`

Filesystem calls are restricted to `ALPHARAVIS_ACP_WORKSPACE`. Path traversal is
blocked. Sensitive paths such as `.env`, private keys, tokens, and secrets are
blocked. Writes are disabled by default and require:

```env
ALPHARAVIS_ACP_ALLOW_FILE_WRITES=true
```

## Streamed Events

The adapter currently emits:

- `agent_message_chunk` for visible assistant text
- `agent_thought_chunk` for short status/reasoning summaries, never private
  chain-of-thought
- `tool_call` for toolcard start
- `tool_call_update` for toolcard completion/failure
- `plan` for planner entries when LangGraph exposes them
- `available_commands_update` after session start
- `session/request_permission` for command approval interrupts

Toolcards include:

- `kind`: `read`, `edit`, or `execute`
- `locations`: file paths found in tool args or diff output
- Markdown diff output when a unified diff is returned

AionUi's `ToolCallContentItem` type supports `type: "diff"` on `tool_call`
payloads. Its `tool_call_update` type is narrower in the local repo, so
completed diff output is sent as Markdown diff content for compatibility.

## Permission Flow

AlphaRavis supports both AionUi response styles:

- JSON-RPC response with the same `id` as `session/request_permission`
- method calls to `session/response` or `session/permission_response`

Mapping:

- `allow_once` / `allow_always` -> LangGraph resume `approve`
- `reject_once` / `reject_always` -> LangGraph resume `reject`
- replace/edit-style responses are parsed when present, but not required by the
  UI.

The adapter never auto-approves destructive commands.

## Cancel Behavior

`session/cancel` always marks the ACP session as locally cancelled. If the
LangGraph SDK stream exposes a run id and the SDK provides a cancel method, the
adapter attempts a real LangGraph cancel. If that is unavailable, cancellation
is local-only and documented in the JSON-RPC result.

If AionUi sends another `session/prompt` while a previous prompt for the same
session is still running, the adapter interrupts the previous in-process prompt,
attempts a best-effort LangGraph cancel, sends the old JSON-RPC request a
`stopReason: cancelled` response, and starts the new prompt with a fresh local
session state. This avoids a common ACP hang where the second message waits
behind an abandoned first stream.

## Safety

The adapter:

- strips AlphaRavis internal context blocks before visible output,
- redacts common secrets, bearer tokens, and private keys,
- truncates tool outputs,
- keeps tool output out of thought/reasoning events,
- blocks file access outside the configured workspace,
- keeps writes disabled unless explicitly enabled.

## References

- [OpenCode ACP Support](https://open-code.ai/en/docs/acp)
- [AionUi ACP Setup](https://github.com/iOfficeAI/AionUi/wiki/ACP-Setup)
