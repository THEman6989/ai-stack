# Hermes Integration

Hermes is optional. AlphaRavis stays the LangGraph supervisor; Hermes is added
as a strong coding/system specialist.

## Docker Service

Hermes is now part of the main Docker Compose stack as `hermes-agent`.
The service is built from the `hermes-agent` Git submodule because no stable
public Docker image was available under the expected GHCR name.

Default endpoints:

```text
inside Docker: http://hermes-agent:8642/v1
from host:     http://localhost:8642/v1
```

Important `.env` values:

```text
HERMES_API_BASE=http://hermes-agent:8642/v1
HERMES_EXTERNAL_API_BASE=http://localhost:8642/v1
HERMES_API_KEY=sk-hermes-local
HERMES_MODEL=hermes-agent
HERMES_INFERENCE_PROVIDER=custom
HERMES_INFERENCE_MODEL=big-boss
HERMES_OPENAI_BASE_URL=http://litellm:4000/v1
HERMES_OPENAI_API_KEY=sk-local-dev
```

`HERMES_MODEL` is the model id advertised to LibreChat and AlphaRavis.
`HERMES_INFERENCE_MODEL` is the real model Hermes uses behind the scenes.

## Modes

### Mode A: LibreChat To Hermes

LibreChat can talk directly to Hermes through the `Hermes Agent` custom endpoint
in `librechat.yaml`.

Docker-side config:

```text
HERMES_API_BASE=http://hermes-agent:8642/v1
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

## LibreChat OpenAI Bucket

If LibreChat shows a separate `OpenAI` provider in addition to `LangGraph Agent`
and `Hermes Agent`, it is usually because LibreChat inherited `OPENAI_API_KEY`
from `.env` or has `OPENAI_REVERSE_PROXY` set.

The current Compose file keeps that generic bucket hidden by default:

```text
LIBRECHAT_OPENAI_API_KEY=
LIBRECHAT_OPENAI_REVERSE_PROXY=
```

The OpenAI-compatible bridge is still externally available at:

```text
http://localhost:8123/v1
```

If you intentionally want the generic LibreChat `OpenAI` entry to point at the
AlphaRavis bridge, set:

```text
LIBRECHAT_OPENAI_API_KEY=sk-1234
LIBRECHAT_OPENAI_REVERSE_PROXY=http://api-bridge:8123/v1
```

## Windows Networking Note

The default Docker setup no longer needs `host.docker.internal` for Hermes
because Hermes runs inside the same Compose network.

If you run Hermes outside Docker instead, set:

```text
HERMES_API_BASE=http://host.docker.internal:8642/v1
```

and start Hermes with `API_SERVER_HOST=0.0.0.0`. On Windows, if a host-run
Hermes is still unreachable from containers, check Windows Firewall rules for
the Hermes Python process and port `8642`.
