# AionUi + AlphaRavis ACP Integration

AlphaRavis exposes a separate ACP adapter for AionUi custom agents:

```bash
python /workspace/langgraph-app/alpharavis_acp_adapter.py
```

This is intentionally separate from the OpenAI-compatible FastAPI bridge. LibreChat and `/v1/chat/completions` keep using `api-bridge` on port `8123`; AionUi talks to the new stdio adapter, and the adapter talks to `langgraph-api` on port `2024`.

## Why ACP

OpenCode documents ACP as a JSON-RPC-over-stdio subprocess protocol for editor/client-to-agent communication. AionUi's ACP setup says custom external agents can be launched as CLI tools if they speak ACP. AlphaRavis follows that shape here: AionUi starts the adapter process, then sends `initialize`, `session/new`, and `session/prompt`.

References:

- [OpenCode ACP Support](https://open-code.ai/en/docs/acp)
- [AionUi ACP Setup](https://github.com/iOfficeAI/AionUi/wiki/ACP-Setup)

## Configuration

In AionUi, add a custom ACP agent with:

```text
Command: python
Args: /workspace/langgraph-app/alpharavis_acp_adapter.py
```

For a Windows checkout, use the local path instead:

```text
Command: python
Args: C:\experi\ai\ai-stack\langgraph-app\alpharavis_acp_adapter.py
```

Or use the helper:

```text
Command: C:\experi\ai\ai-stack\scripts\run_alpharavis_acp_adapter.cmd
Args:
```

Set environment variables as needed:

```env
LANGGRAPH_API_URL=http://localhost:2024
LANGGRAPH_ASSISTANT_ID=alpha_ravis
ALPHARAVIS_ACP_WORKSPACE=/workspace
ALPHARAVIS_ACP_TOOL_OUTPUT_MAX_CHARS=8000
ALPHARAVIS_ACP_TRACE_DETAIL=summary
ALPHARAVIS_ACP_SCRUB_INTERNAL_CONTEXT=true
```

Docker-internal default:

```env
LANGGRAPH_API_URL=http://langgraph-api:2024
```

Host/AionUi default:

```env
LANGGRAPH_API_URL=http://localhost:2024
```

## Supported ACP Methods

Incoming from AionUi:

- `initialize`
- `session/new`
- `session/load`
- `session/prompt`
- `session/send_message`
- `session/cancel`
- `session/close`
- `session/set_config_option`

Outgoing to AionUi:

- `session/update`
- `session/request_permission`

## Event Mapping

LangGraph message deltas become:

```text
session/update -> agent_message_chunk
```

Short status or node updates become:

```text
session/update -> agent_thought_chunk
```

These are only summaries, not private chain-of-thought.

Tool calls become:

```text
session/update -> tool_call
session/update -> tool_call_update
```

Plans become:

```text
session/update -> plan
```

Command approval interrupts become:

```text
session/request_permission
```

The adapter maps AionUi's permission answer back to LangGraph resume commands:

- `allow_once` -> `{"action": "approve"}`
- `reject_once` -> `{"action": "reject"}`

## Safety

The adapter:

- strips AlphaRavis internal context blocks before visible output,
- redacts common secrets, bearer tokens, and private keys,
- truncates tool outputs,
- keeps tool output out of thought/reasoning events,
- never auto-approves destructive commands.

## Limitations

The first version maps common LangGraph SDK stream shapes: `messages`, `updates`, tool-start/tool-end callback events, AIMessage tool calls, ToolMessages, and command approval interrupts. If a future LangGraph SDK emits a new event shape, the adapter falls back to short status summaries instead of dumping raw event payloads.

The adapter does not replace the OpenAI bridge. Use it when you want AionUi's ACP toolcards and permission UI rather than LibreChat's OpenAI-compatible chat surface.
