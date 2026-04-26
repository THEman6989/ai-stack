# Hermes Integration

Hermes is optional. AlphaRavis stays the LangGraph supervisor; Hermes is added
as a strong coding/system specialist.

## Modes

### Mode A: LibreChat To Hermes

LibreChat can talk directly to Hermes through the `Hermes Agent` custom endpoint
in `librechat.yaml`.

Start Hermes outside the stack:

```bash
API_SERVER_ENABLED=true API_SERVER_HOST=0.0.0.0 API_SERVER_PORT=8642 API_SERVER_KEY=sk-hermes-local hermes gateway
```

Docker-side config:

```text
HERMES_API_BASE=http://host.docker.internal:8642/v1
HERMES_API_KEY=sk-hermes-local
HERMES_MODEL=hermes-agent
```

Use this when you want Hermes directly for coding, terminal-oriented work, file
tasks, or debugging.

### Mode B: LibreChat To AlphaRavis

This is the existing route:

```text
LibreChat -> api-bridge:8123 -> langgraph-api:2024 -> alpha_ravis
```

Use this for multi-agent workflows, memory, Pixelle, research, custom tools, and
normal AlphaRavis orchestration.

### Mode C: AlphaRavis To Hermes

Enable only after Hermes gateway is reachable:

```text
ALPHARAVIS_ENABLE_HERMES_AGENT=true
```

AlphaRavis then has a `hermes_coding_agent` swarm worker. It calls Hermes via
OpenAI-compatible `/v1/chat/completions` and asks for a bounded structured
coding/system result.

Loop guard:

- AlphaRavis may call Hermes for one bounded subtask.
- The Hermes call is instructed not to call AlphaRavis/LangGraph back.
- Hermes output is limited by `HERMES_MAX_OUTPUT_CHARS`.
- Hermes calls time out after `HERMES_TIMEOUT_SECONDS`.

### Mode D: Hermes To AlphaRavis

Disabled by default:

```text
BRIDGE_ENABLE_LANGGRAPH_TOOL=false
```

If enabled, Hermes can call:

```text
POST http://api-bridge:8123/tools/langgraph/run
```

Body:

```json
{
  "explicit_user_request": true,
  "thread_key": "hermes-session-id",
  "message": "Run this bounded AlphaRavis/LangGraph task.",
  "timeout_seconds": 60
}
```

If `BRIDGE_LANGGRAPH_TOOL_API_KEY` is set, send:

```text
Authorization: Bearer <BRIDGE_LANGGRAPH_TOOL_API_KEY>
```

This endpoint rejects calls unless `explicit_user_request=true`, so Hermes can
use LangGraph only when the user explicitly asks for it.

## Windows Networking Note

The Docker services use `host.docker.internal` to reach Hermes on the host. On
Linux this is mapped through `extra_hosts: host-gateway`.

If Hermes binds only to `127.0.0.1`, Linux containers usually cannot reach it
through the host-gateway address. Start Hermes with:

```text
API_SERVER_HOST=0.0.0.0
```

On Windows, if Hermes is running but unreachable from containers, check Windows
Firewall rules for the Hermes Python process and port `8642`.
