# AlphaRavis User Notes

This file explains what AlphaRavis uses automatically and what is used only
when you ask for it. It is meant for humans first, and agents may also read it
when asked how the system behaves.

## Daily Interface

Open the local service dashboard when you want a clickable overview of the
running stack:

```text
http://localhost:8090
```

It lists LibreChat, LangGraph, the AlphaRavis Bridge, Bridge Test UI, Hermes,
LiteLLM, media/RAG services, optional UIs, and database endpoints. `make up`,
`make install`, and `make update` include it in the base Docker Compose stack;
`make service-dashboard` starts only that redirector.

For the complete Makefile target and argument reference, including install,
update, streaming profiles, Tailscale/LAN network modes, media/vision settings,
and smoke checks, see [`MAKEFILE_README.md`](MAKEFILE_README.md).

A Tailscale HTTPS helper can make the local HTTP services reachable through
Tailscale Serve inside your Tailnet and make the dashboard show those HTTPS
URLs:

```bash
make tailscale-plan TAILSCALE_HOST=<device>.<tailnet>.ts.net
make tailscale-overrides TAILSCALE_HOST=<device>.<tailnet>.ts.net
make tailscale-apply TAILSCALE_HOST=<device>.<tailnet>.ts.net
```

The helper reads the redirector's service list, keeps only local HTTP services,
and plans `tailscale serve --bg --https=<port>` routes such as
`http://localhost:3080` -> `https://<device>.<tailnet>.ts.net:3080`.
The dashboard itself is included by default as
`https://<device>.<tailnet>.ts.net:8090`, so after `make tailscale-apply` you
can open the same service cards from another allowed Tailnet device. To opt out
of the dashboard route, pass `TAILSCALE_DASHBOARD=false` to the Make target or
use `--exclude-dashboard` with `tailscale_https_routes.py`.
It does not run Tailscale Funnel and does not publish services to the public
internet; access stays limited to devices allowed in your Tailnet.
Sudo handling defaults to `TAILSCALE_SUDO=auto`: the helper tries the Tailscale
command normally first, and only prompts for a sudo password if the CLI reports
a permissions error. Use `TAILSCALE_SUDO=true` to force sudo from the start or
`TAILSCALE_SUDO=never` to disable sudo retry. The password is not stored.
`make tailscale-overrides` only writes
`service-dashboard-data/tailscale_service_urls.json`; it does not change
Tailscale Serve state. In dashboard `auto` mode, that JSON file makes the cards
prefer Tailscale HTTPS URLs while still showing the original local URL on each
card.

`make install`, `make update`, `make update-no-start`, `make up`,
`make up-fullstreaming`, and `make up-chat-fullstreaming` run
`tailscale-apply` automatically around their normal stack work. Before Docker
starts, the Makefile writes `ALPHARAVIS_DOCKER_HOST_BIND=127.0.0.1` so Docker
does not try to bind the Tailnet IP ports already owned by Tailscale Serve.
After Docker starts, it applies the Tailnet HTTPS routes and writes the
dashboard override JSON in one step.

Set `TAILSCALE_AUTO=off` when you explicitly want LAN HTTP mode instead:

```bash
make install TAILSCALE_AUTO=off
make update TAILSCALE_AUTO=off
make up TAILSCALE_AUTO=off
```

That disables the managed Tailscale Serve routes, removes the dashboard HTTPS
override JSON, writes `ALPHARAVIS_DOCKER_HOST_BIND=0.0.0.0`, and then Docker
recreates affected containers during the normal start step so the app ports are
reachable through the host's LAN IP again. Use `TAILSCALE_AUTO=keep` only when
you want a run to leave the current Tailscale/Docker bind mode untouched.

Use LibreChat for normal chatting. It talks to `api-bridge`, which forwards the
request into the LangGraph `alpha_ravis` brain.

Use LangGraph Studio or DeepAgents UI when you want to inspect internal graph
steps, state, checkpoints, run profiles, memory compression, or agent routing.

For debugging LibreChat-specific issues, use the minimal Bridge test UI. It is
part of the normal base stack, so `make up`, `make install`, and `make update`
build/start it through Docker Compose. To rebuild or start only that UI:

```bash
make test-ui
```

Then open:

```text
http://localhost:8140
```

This UI keeps only an in-browser RAM chat history, sends through
`bridge-test-ui -> api-bridge`, and can switch between Responses and Chat
Completions. It intentionally does not use LibreChat storage, presets, or UI
state. The Trace panel shows a small waterfall for the current request so you
can see whether time is spent in the browser/test UI, Bridge setup, LangGraph
wait, fast-chat primary model call, fallback model call, or backend failure.
Normal sends use streaming: the UI forwards `stream=true`, reads SSE events as
they arrive, and renders Responses `response.output_text.delta` or Chat
Completions `delta.content` live. When reasoning is present, each assistant
message gets a collapsed `Reasoning` panel fed by Responses
`response.reasoning.delta` or Chat Completions reasoning delta fields. If the
agent path only emits a complete final AI message, the UI still streams
lifecycle/activity events first, but visible answer text can arrive as one
final delta.
Inside the `Reasoning` panel, LangGraph lifecycle statuses are shown in a
separate `Status` block. Model-provided reasoning text, when the Bridge emits
it, is shown below that as `Modell-Reasoning`. In the default local stack,
`BRIDGE_STREAM_REASONING_EVENTS=false`, so the panel may show statuses only
unless explicit reasoning forwarding is enabled for debugging.
Planner output from the Agent Path is internal. The Bridge routes streamed text
from the LangGraph `planner` node into the reasoning channel for both Responses
and Chat Completions, marked as `internal_plan`, so it can be inspected without
being appended to the visible assistant answer. The Test UI shows it as an
`Interner Plan` block.
Use `Verlauf leeren` before an isolated test. It clears the visible messages and
starts a new backend session, so the next prompt does not resume an older
LangGraph thread from a previous browser reload.
Responses-mode history is sent as structured message items with roles, not as a
synthetic `Chat history:` prompt. Internal task/planner blocks are scrubbed from
Bridge-visible output.
Assistant messages show a route badge: `Fast Path` for the simple direct route,
`Agent Path` for planner/swarm/tool routing, and `Hard Stop` when context limits
block the run. The status line mirrors the same route while the stream is
active.
The Trace table compacts consecutive answer-text deltas by default, so token or
character streaming does not flood the waterfall. Enable `Delta-Details` in the
Trace header when you need the raw per-delta timing rows for a stuck or jittery
stream.

## Fast Path

Short non-tool questions can take the fast path. This path:

- runs through LangGraph state,
- skips skill-library retrieval,
- skips the swarm,
- calls the model directly,
- can fall back to `edge-gemma` only for simple chat.

Fast-path replies are visibly marked by default:

```text
ALPHARAVIS_SHOW_FAST_PATH_NOTICE=true
ALPHARAVIS_FAST_PATH_NOTICE_TEXT=Fastpath
```

When enabled, the marker is appended after the model answer as a short label;
the response does not include execution-mode explanatory text.

Once one turn in a chat thread uses the normal agent/swarm path, that thread is
locked out of fast path by default:

```text
ALPHARAVIS_FAST_PATH_LOCK_AFTER_SWARM=true
```

This prevents a complex conversation from bouncing back into the simple route
later.

The fast path is not used when the message looks like it needs tools, research,
debugging, Docker, SSH, Pixelle, code/files, memory/archive search, PC control,
or AlphaRavis architecture details.

Optional MCP tool loading is off by default:

```text
ALPHARAVIS_LOAD_MCP_TOOLS=false
```

That prevents slow MCP startup from affecting every chat. AlphaRavis now uses a
DeepAgents-style MCP config loader: `mcp.json` / `.mcp.json` files describe
servers, while the agent can inspect the server manifest before tools are
loaded. The native Pixelle HTTP tool can still start Pixelle jobs without
loading the extra MCP registry.

Agents can still see a short manifest of optional registries through the
`describe_optional_tool_registry` tool, so they know Pixelle MCP exists and how
it can be enabled without paying the startup cost by default.

The manifest now comes from composable AlphaRavis toolsets. Typical categories
are `coding/read`, `coding/write`, `media/image`, `rag/memory`, `system/docker`,
`system/ssh`, and `system/power`. The planner records likely categories in
`run_profile.selected_toolsets`; each specialist gets only its bounded bundle
and matching MCP category tools. Those bundles are materialized at graph build
time from `alpharavis_toolsets.py`; handoff tools are added explicitly so agents
can still transfer work without receiving unrelated concrete tools. The loaded
per-agent bundles are visible in `run_profile.loaded_toolsets`.

Default MCP config:

```text
ALPHARAVIS_MCP_CONFIG_PATH=/workspace/langgraph-app/mcp.json
ALPHARAVIS_MCP_TOOL_PREFIX=true
ALPHARAVIS_MCP_ALLOW_STDIO=false
ALPHARAVIS_MCP_STRICT=false
```

`ALPHARAVIS_MCP_ALLOW_STDIO=false` is intentional: stdio MCP can start local
processes, so only remote HTTP/SSE MCP servers are trusted by default.

For llama.cpp/Qwen-style models, fast path also disables hidden thinking with:

```text
ALPHARAVIS_FAST_PATH_DISABLE_THINKING=true
```

This keeps tiny replies from spending seconds on invisible reasoning tokens.
Set it to `false` if you explicitly want hidden thinking even in fast path.

To force the normal agent path for one message, write:

```text
kein fast path
```

## Big Model And Small Fallback

`big-boss` is the main model on the llama.cpp server.

`edge-gemma` is a small starter/crisis model on the Ollama management machine.
It is not intended for complex agent workflows or risky tool decisions.

Current rule:

```text
simple fast path may use edge-gemma as fallback
normal swarm/tool path stays on big-boss
```

If the big server is down, complex requests should fail visibly instead of
silently running on the weaker model.

Direct no-tool LangGraph model calls can use the Responses API:

```text
ALPHARAVIS_LLM_API_MODE=responses
ALPHARAVIS_RESPONSES_API_BASE=http://litellm:4000/v1
ALPHARAVIS_RESPONSES_MODEL=big-boss
```

This applies to direct calls such as planner, fast path, and summarizers. The
DeepAgents tool workers can also use LangChain's `ChatOpenAI` Responses mode:

```text
ALPHARAVIS_DEEPAGENTS_API_MODE=responses
ALPHARAVIS_DEEPAGENTS_RESPONSES_API_BASE=http://litellm:4000/v1
ALPHARAVIS_DEEPAGENTS_RESPONSES_MODEL=big-boss
ALPHARAVIS_DEEPAGENTS_RESPONSES_OUTPUT_VERSION=responses/v1
ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING=true
ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=tool_calling
ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING=false
```

Set `ALPHARAVIS_DEEPAGENTS_API_MODE=chat_completions` to return only DeepAgents
tool workers to the older ChatLiteLLM path. Chat Completions streaming is
controlled by:

```text
ALPHARAVIS_LLM_STREAMING=true
```

Set
`ALPHARAVIS_DEEPAGENTS_REQUIRE_RESPONSES=true` only when you want startup to
fail instead of falling back. Set `ALPHARAVIS_RESPONSES_REQUIRE_NATIVE=true`
only when you want direct no-tool calls to fail instead of falling back to Chat
Completions.

The DeepAgents Responses streaming flags are separate from the external bridge
stream. The stable local default is:

```text
ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING=true
ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=tool_calling
```

AlphaRavis applies a local startup patch equivalent to the important part of
langchain-ai/langchain PR #35457. Without that patch, LangChain's documented
hybrid mode can crash because the non-streaming `_generate` / `_agenerate`
paths still send `stream=true` to the provider.

Live testing after the patch showed:

- direct `ChatOpenAI(use_responses_api=True, streaming=True,
  disable_streaming="tool_calling")` with a bound tool passes
- Bridge `/v1/responses` Agent Path streaming returns SSE output text chunks
- full streaming with `ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=false`
  now has an env-gated AlphaRavis patch and passed the focused
  LangChain/React-agent probe, but remains experimental as a runtime default

Those modes remain available as opt-ins for provider/library upgrades:

```text
# force fully non-streaming internal model calls
ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING=false
ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=true

# experimental full streaming
ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING=true
ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=false
ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING=true

# default patched LangChain hybrid
ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING=true
ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=tool_calling
ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING=false
```

The Makefile can write those combinations for you:

```bash
make config
make install STREAMING=full
make install STREAMING=chat-full
make streaming STREAMING=hybrid
make streaming STREAMING=chat-nonstreaming
make up-fullstreaming
```

Use `make config` when you want to edit the stack settings visually instead of
answering repeated terminal prompts during install/update. It opens a local
browser page backed by `.env`: current values are already filled in from `.env`,
defaults come from `.env(exaple)`, booleans use True/False buttons, and Save
writes all shown values back to `.env`. Each row has Reset for that one key, and
Reset all asks for confirmation before restoring all documented defaults.

If Tailscale Serve is already listening on the Tailnet IP for the same service
ports, Docker's published-port host bind must be localhost before recreating
containers:

```text
ALPHARAVIS_DOCKER_HOST_BIND=127.0.0.1
```

The default template is `0.0.0.0` for normal LAN exposure, while the default
Makefile runtime mode changes it to `127.0.0.1` when Tailscale HTTPS is active.
This applies to the user-facing UI/API ports, including Pixelle on `9004`; the
database debug ports are still published directly for local development.
To switch a running stack back to LAN HTTP mode:

```bash
make tailscale-disable
```

To switch it back to Tailscale HTTPS mode:

```bash
make tailscale-apply
```

Direct Responses calls have a small compatibility retry layer:

```text
ALPHARAVIS_RESPONSES_UNSUPPORTED_PARAM_RETRY=true
ALPHARAVIS_RESPONSES_OMIT_TEMPERATURE_MODE=auto
ALPHARAVIS_PROVIDER_PROFILE=auto
ALPHARAVIS_PROVIDER_REQUIRE_RESPONSES_MODE=auto
ALPHARAVIS_CHAT_FALLBACK_MODE=auto
ALPHARAVIS_RESPONSES_TOKEN_LIMIT_PARAM_MODE=auto
ALPHARAVIS_CHAT_OMIT_TEMPERATURE_MODE=auto
ALPHARAVIS_CHAT_TOKEN_LIMIT_PARAM_MODE=auto
```

If a local endpoint rejects a harmless parameter such as `parallel_tool_calls`,
`truncation`, `temperature`, or a token-limit spelling, AlphaRavis retries once
with the safer payload instead of failing the whole planner/summary call.
When that safer retry also fails, the raised provider error includes both the
original failure and the compatibility-retry failure so operators can diagnose
the real backend behavior.

The same conservative cleanup is applied to ChatLiteLLM fallback kwargs. Local
LiteLLM/llama.cpp calls keep `max_tokens` by default; direct OpenAI/GitHub
GPT-4o/o-series/GPT-5-style endpoints can be auto-mapped to
`max_completion_tokens`.

When a direct Responses compatibility retry succeeds, agents can see retry
metadata in `run_profile.provider_hardening_last_retry` for planner/fast-path
runs. This is diagnostic metadata only; it does not change routing by itself.

Provider profiles are deliberately small request-shape profiles, not new
provider adapters. `auto` keeps the local LiteLLM/llama.cpp route conservative,
detects Kimi/Moonshot-style server-managed sampling, and maps direct
OpenAI/GitHub reasoning-style Chat calls to `max_completion_tokens` when that
family is detected. Use `ALPHARAVIS_PROVIDER_PROFILE=responses_required` or
`ALPHARAVIS_CHAT_FALLBACK_MODE=responses_required` only after runtime evidence
shows that Chat Completions fallback is broken for a provider.

Direct non-OpenAI adapters stay out of AlphaRavis by policy. Prefer routing
through LiteLLM/LangChain; add a direct adapter only when the gateway cannot
represent a required feature and the change has focused docs and tests.

Hermes-inspired maintenance helpers are available as review tools:

- `suggest_thread_title` creates a short deterministic title for a thread,
  archive, or collection.
- `extract_review_insights` returns candidate user/system insights with
  `review_required=true`; it never promotes them into always-memory.

## Model And Power Management

AlphaRavis has a custom `model_management.py` layer for your split hardware
setup.

Important idea:

- The Ollama management node is mainly for startup/crisis work.
- The embedding model should be loaded only during safe windows.
- A safe window means either the system has been idle long enough or the big
  llama.cpp server is up, so normal chat does not depend on the small Ollama
  node.

Default controls keep this custom layer completely off:

```text
ALPHARAVIS_ENABLE_MODEL_MANAGEMENT=false
ALPHARAVIS_ENABLE_ADVANCED_MODEL_MANAGEMENT=false
ALPHARAVIS_ENABLE_OWNER_POWER_TOOLS=false
ALPHARAVIS_ENABLE_CRISIS_MANAGER=false
ALPHARAVIS_ENABLE_POWER_MANAGEMENT=false
ALPHARAVIS_MODEL_MGMT_ALLOW_ACTIONS=false
ALPHARAVIS_POWER_MANAGER_MODEL=openai/edge-gemma
ALPHARAVIS_CRISIS_MANAGER_MODEL=openai/edge-gemma
ALPHARAVIS_MODEL_IDLE_SECONDS=600
ALPHARAVIS_EMBEDDING_LOAD_POLICY=idle_or_big_llm_active
```

With these defaults, AlphaRavis uses the normal `big-boss` route and does not
create the Power Management Agent. Enable the layer only on the custom hardware
setup that needs it.

The advanced hooks become visible only after:

```text
ALPHARAVIS_ENABLE_MODEL_MANAGEMENT=true
ALPHARAVIS_ENABLE_ADVANCED_MODEL_MANAGEMENT=true
ALPHARAVIS_ENABLE_OWNER_POWER_TOOLS=true
```

Even then, real shutdowns, service changes, Ollama model switching, and
embedding-job runs stay disabled until you provide:

```text
ALPHARAVIS_MODEL_MGMT_ACTION_URL=
ALPHARAVIS_MODEL_MGMT_API_KEY=
ALPHARAVIS_MODEL_MGMT_ALLOW_ACTIONS=true
```

`make model-management` can write these ENV switches interactively. The result
is still plain `.env`, so you can copy that `.env` to another machine and skip
the Make step later.

When enabled, the Power Management Agent handles questions like:

```text
check model management status
plane ein Embedding-Fenster
run embedding memory jobs
pruefe ob ComfyUI fuer Pixelle bereit ist
```

Owner-specific host/MAC/start-command defaults live in
`langgraph-app/owner_power_tools.py`. Keep real passwords in your private `.env`
through `ALPHARAVIS_OWNER_SSH_PASS`, not in git.

## Hermes Mode

Hermes is available as a separate optional coding/system agent.

Direct mode:

```text
LibreChat -> Hermes Agent
```

Use this for coding, terminal-oriented work, file operations, and direct agent
tasks when Hermes gateway is running.

AlphaRavis mode:

```text
LibreChat -> AlphaRavis -> hermes_coding_agent -> Hermes
```

Use this when AlphaRavis should stay the main supervisor but delegate a bounded
coding/system subtask to Hermes.

Required Hermes gateway settings:

```text
API_SERVER_ENABLED=true
API_SERVER_HOST=0.0.0.0
API_SERVER_PORT=8642
API_SERVER_KEY=<same as HERMES_API_KEY>
```

AlphaRavis settings:

```text
HERMES_API_BASE=http://host.docker.internal:8642/v1
HERMES_API_KEY=sk-hermes-local
HERMES_MODEL=hermes-agent
ALPHARAVIS_ENABLE_HERMES_AGENT=false
```

Keep `ALPHARAVIS_ENABLE_HERMES_AGENT=false` until Hermes is actually reachable.

The Hermes Docker services automatically apply local parent-repo patches from:

```text
patches/hermes-agent/
```

The upstream `hermes-agent` submodule should stay clean. The container startup
entrypoint wrapper runs `scripts/apply_hermes_agent_patches.sh` against
`/opt/hermes` before starting Hermes. For local non-Docker debugging, run the
same script without `HERMES_PATCH_TARGET_DIR` to apply the patch set to the
submodule.

Reverse mode is disabled by default:

```text
BRIDGE_ENABLE_LANGGRAPH_TOOL=false
```

When enabled, Hermes can call `POST /tools/langgraph/run`, but only with
`explicit_user_request=true`. This is the loop guard: Hermes may use LangGraph
only when you explicitly ask it to.

## Tools

Tools are used only when an agent chooses them for the task.

Examples that trigger tool-capable paths:

- "debugge den Fehler"
- "schau Docker logs"
- "starte meinen PC"
- "generiere ein Bild"
- "suche in meinen Dokumenten"
- "lies die Architektur von AlphaRavis"
- "suche in alten Archiven"

Risky local or SSH commands require a human approval interrupt before execution.
Reply with:

```text
approve
reject
replace: <safer command>
approve always
immer erlauben
```

`approve`/`ja` approves only the pending command once. `approve always` or
`immer erlauben` approves the pending command and remembers that exact
scope/target/command combination for the current LibreChat thread. The memory
is bridge-local and process-local, so it is intentionally lost when the
`api-bridge` container restarts.

LibreChat does not currently expose AlphaRavis command approvals as native
clickable approval buttons on the external custom endpoint path. AionUI/ACP has
native permission requests; LibreChat uses the chat-text fallback above unless
LibreChat itself is extended with a custom permission event.

## Pixelle Jobs

For image generation, `start_pixelle_remote` starts a job and waits through a
durable LangGraph `@task`. This is best when you want AlphaRavis to stay with
the job until it finishes.

`start_pixelle_async` starts the job and returns a `job_id` immediately. This is
better for long jobs. Later you can ask:

```text
check_pixelle_job <job_id>
```

Before a Pixelle job starts, AlphaRavis can check ComfyUI:

```text
ALPHARAVIS_ENABLE_MODEL_MANAGEMENT=true
ALPHARAVIS_ENABLE_ADVANCED_MODEL_MANAGEMENT=true
ALPHARAVIS_ENABLE_OWNER_POWER_TOOLS=true
ALPHARAVIS_PIXELLE_PREPARE_COMFY=true
ALPHARAVIS_COMFY_HEALTH_URL=http://<comfy-ip>:8188/system_stats
```

If ComfyUI is offline, AlphaRavis warns. It only blocks the Pixelle job when:

```text
ALPHARAVIS_PIXELLE_BLOCK_IF_COMFY_OFFLINE=true
```

For the owner hardware setup, Pixelle preflight can also use the direct
owner-tool Wake-on-LAN path:

```text
ALPHARAVIS_ENABLE_OWNER_POWER_TOOLS=true
ALPHARAVIS_PIXELLE_OWNER_WAKE_COMFY=true
ALPHARAVIS_PIXELLE_OWNER_WAKE_WAIT_SECONDS=30
```

If Pixelle fails, the returned message includes debugger-ready context and asks
for Pixelle/LangGraph logs instead of crashing silently.

## Bridge Streaming

LibreChat normally uses:

```text
POST /v1/chat/completions
```

The bridge also offers:

```text
POST /v1/responses
```

`/v1/responses` uses the same LangGraph run path, but exposes a Responses-style
contract externally. Non-stream responses return `object=response` with output
items, usage estimates with `input_tokens_details`, metadata,
`previous_response_id`, `tools`, `reasoning`, `completed_at`, and other
Responses fields. `previous_response_id` works against the bridge-local response
store and injects the previous stored output into the next LangGraph run.
Streamed responses use semantic SSE events such as:

```text
response.created
response.in_progress
response.output_item.added
response.content_part.added
response.output_text.delta
response.output_text.done
response.content_part.done
response.output_item.done
response.completed
```

The bridge also supports the bridge-local management endpoints that clients
expect from the Responses surface:

```text
GET /v1/responses/{response_id}
GET /v1/responses/{response_id}/input_items
DELETE /v1/responses/{response_id}
POST /v1/responses/{response_id}/cancel
POST /v1/responses/input_tokens
POST /v1/responses/compact
```

`cancel` returns the normal Response object only for cancellable background
responses; AlphaRavis currently runs foreground jobs, so completed responses
return an explicit `response_not_cancellable` error. `compact` returns a clear
`501 compact_not_supported` because OpenAI's compact endpoint returns encrypted
opaque items; AlphaRavis uses its own active compression plus archive retrieval
instead. `GET /v1/responses/{response_id}?stream=true` returns
`retrieve_stream_not_supported` rather than replaying a fake event history.

The response cache is controlled by:

```text
BRIDGE_RESPONSES_STORE=true
BRIDGE_RESPONSES_STORE_MAX=200
BRIDGE_RESPONSES_DONE_SENTINEL=true
BRIDGE_RESPONSES_ALLOW_CLIENT_TOOLS=false
```

`BRIDGE_RESPONSES_ALLOW_CLIENT_TOOLS=false` is intentional. OpenAI-hosted
Responses tools such as web search, file search, code interpreter, computer use,
or shell execution are not executed by this bridge. AlphaRavis has its own
LangGraph tools. If a client sends `tools`, the bridge returns a structured
error instead of pretending to run unsupported hosted tools. Structured output
formats and non-text output modalities also return explicit unsupported errors
until they are wired to a real AlphaRavis capability.

Chat Completions remains available for LibreChat compatibility. Clients that
support Responses output items should prefer:

```text
BRIDGE_PREFERRED_API_MODE=responses
```

LibreChat custom endpoints may still call `/v1/chat/completions` depending on
LibreChat provider support. If LibreChat is configured with a provider/client
mode that can call Responses, point it to `/v1/responses`; otherwise the bridge
keeps Chat Completions as the compatibility surface while the internal
AlphaRavis path stays the same. AlphaRavis exposes OpenAPI `3.1.0`.
See `docs/ALPHARAVIS_RESPONSES_COMPATIBILITY.md` for the full implemented vs.
explicitly unsupported Responses matrix.

Reasoning/thinking is stripped from normal visible answer text. If a client can
handle a separate reasoning delta field, enable:

```text
BRIDGE_STREAM_REASONING_EVENTS=true
BRIDGE_REASONING_DELTA_FIELD=reasoning_content
```

If LibreChat shows that reasoning as normal text, turn it back off.

For `/v1/responses`, the bridge now has a separate default-on switch:

```text
BRIDGE_STREAM_SUBGRAPHS=true
BRIDGE_RESPONSES_STREAM_REASONING_EVENTS=true
BRIDGE_RESPONSES_OUTPUT_DELTA_MAX_CHARS=1
BRIDGE_RESPONSES_REASONING_DELTA_MAX_CHARS=1
```

This keeps LibreChat's Responses path receiving planner output, explicit model
reasoning, and visible local-model thinking in the Responses reasoning stream.
The Bridge Test UI additionally splits those same events into live Status,
Reasoning, and Planer panes for debugging; LibreChat receives them together in
its reasoning channel.
The Test UI also has a live `Kontext` terminal. It shows compaction events in
yellow and hard-cutoff / hard-trim events in red when the Bridge receives
semantic context activity from LangGraph.
`BRIDGE_STREAM_SUBGRAPHS=true` is required for the nested AlphaRavis Swarm
workers to stream token-level partials through the top-level LangGraph run.
Visible assistant text and model/plan reasoning are split into character-level
Responses deltas for smoother rendering. Status lines remain whole status
events. That split is only at the Bridge boundary; if the upstream
Swarm/DeepAgents node does not emit partial message events, the first visible
answer delta still waits for that node to finish.

For request/context debugging, open the Bridge Test UI observer:

```text
http://127.0.0.1:8140/observer
```

It is a full-page table of recent Bridge requests. Select a row, then switch
between `Senden` and `Empfang`; `Nur Kontext` shows the exact context prepared
for LangGraph plus the existing LangGraph state profile, while `Vollansicht`
shows the full captured Bridge payload.

LibreChat's OpenAI-compatible chat-completions calls commonly send `user` as
the account id rather than a conversation id. The Bridge therefore does not use
`body.user` as the LangGraph thread key by default; requests without
`conversationId`, `conversation_id`, `x-conversation-id`, or `x-thread-id` use
isolated ephemeral LangGraph threads. Set `BRIDGE_ALLOW_USER_THREAD_KEY=true`
only for clients where `user` is known to be a stable per-chat identifier.

## Error Classification

AlphaRavis classifies backend/API failures before showing them to LibreChat or
recording them in `run_profile`:

```text
ALPHARAVIS_ENABLE_ERROR_CLASSIFIER=true
BRIDGE_SHOW_ERROR_CLASSIFICATION=true
```

The classifier labels failures as:

```text
context_overflow
payload_too_large
timeout
server_error
overloaded
rate_limit
format_error
auth
model_not_found
unknown
```

It does not execute dangerous recovery actions by itself. It only tells the
bridge and graph which recovery path is appropriate: compress active context,
retry/back off, strip unsupported parameters or fall back, surface an auth/model
configuration error, or hand the situation to the Crisis Manager when advanced
model management is enabled.

Planner errors, fast-path primary-model fallback, crisis preflight failures, and
crisis-manager failures store this classification metadata in `run_profile`.
When bridge activity events are enabled, LibreChat can also receive a short
status line such as `Fehler klassifiziert: timeout; Aktion: crisis_recovery.`

## Bridge Context Hygiene

The bridge strips internal AlphaRavis context blocks from visible output by
default:

```text
BRIDGE_SCRUB_INTERNAL_CONTEXT=true
```

This protects LibreChat streaming output even when an internal tag such as
`<memory-context>` is split across multiple SSE deltas. The internal context can
still exist inside LangGraph state and Studio debugging; it is just not emitted
as normal assistant text.

Explicit user references can be expanded before the message reaches LangGraph:

```text
BRIDGE_ENABLE_CONTEXT_REFERENCES=true
BRIDGE_CONTEXT_REFERENCES_FETCH_URLS=true
```

Supported forms:

```text
@file:langgraph-app/agent_graph.py:10-40
@folder:docs
@diff
@staged
@git:3
@url:https://example.com/page
```

Files are resolved under `BRIDGE_CONTEXT_REFERENCE_WORKSPACE_ROOT` or the
AI-stack repo root by default. Sensitive credential/config paths such as `.env`,
`.ssh`, `.aws`, `.kube`, and `.docker` are refused. Large references are bounded
by:

```text
BRIDGE_CONTEXT_REFERENCE_SOFT_RATIO=0.25
BRIDGE_CONTEXT_REFERENCE_HARD_RATIO=0.50
BRIDGE_CONTEXT_REFERENCE_CONTEXT_LENGTH=128000
BRIDGE_CONTEXT_REFERENCE_MAX_FILE_CHARS=20000
BRIDGE_CONTEXT_REFERENCE_MAX_GIT_CHARS=20000
BRIDGE_CONTEXT_REFERENCE_MAX_URL_CHARS=12000
```

Large file/git/URL references are truncated head+tail, not just first-N chars.
That keeps openings, imports, command headers, final errors, and tail context
while telling the agent to use exact file/archive tools when it needs the full
source.

Reference warnings and injected-token estimates are recorded in the LangGraph
`run_profile`.

Hard request cutoffs:

```text
ALPHARAVIS_HARD_CONTEXT_TOKEN_LIMIT=128000
BRIDGE_HARD_INPUT_TOKEN_LIMIT=0
ALPHARAVIS_ENABLE_STATIC_CONTEXT_RESERVE=true
ALPHARAVIS_STATIC_CONTEXT_RESERVE_TOKENS=0
ALPHARAVIS_USE_AGENT_SPECIFIC_CONTEXT_RESERVE=true
ALPHARAVIS_ENABLE_FINAL_LLM_BUDGET_GUARD=true
ALPHARAVIS_ENABLE_FINAL_BUDGET_RESCUE=true
ALPHARAVIS_FINAL_BUDGET_RESCUE_MAX_PASSES=3
ALPHARAVIS_ENABLE_PROVIDER_OVERFLOW_RETRY=true
ALPHARAVIS_ENABLE_PROVIDER_CONTEXT_LIMIT_RETRY=true
ALPHARAVIS_DYNAMIC_COMPRESSION_UNTIL_BUDGET=true
ALPHARAVIS_DYNAMIC_COMPRESSION_MAX_PASSES=6
ALPHARAVIS_DYNAMIC_COMPRESSION_HARD_MAX_PASSES=12
ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_RATIO=0.75
ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_MIN_TOKENS=8192
ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_MAX_TOKENS=0
ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_CHARS_PER_TOKEN=2.0
ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_OVERHEAD_RESERVE_TOKENS=512
ALPHARAVIS_COMPRESSION_ENABLE_CHUNKED_SUMMARY=false
ALPHARAVIS_COMPRESSION_SUMMARY_CHUNK_RATIO=0.03
ALPHARAVIS_COMPRESSION_SUMMARY_CHUNK_MIN_TOKENS=300
ALPHARAVIS_COMPRESSION_SUMMARY_CHUNK_MAX_TOKENS=0
ALPHARAVIS_COMPRESSION_SUMMARY_CHUNK_OVERLAP_CHARS=1000
ALPHARAVIS_COMPRESSION_SUMMARY_MAX_CHUNKS=12
```

The bridge cutoff is disabled by default. LangGraph owns the normal hard
context decision, asks the model endpoint for its context length when discovery
is enabled, and then runs pre-run compression/trim on the active thread state
before invoking any model. Set `BRIDGE_HARD_INPUT_TOKEN_LIMIT` to a positive
value only for deployments that intentionally want the bridge to reject raw
oversized requests before LangGraph can compact them.

`ALPHARAVIS_ENABLE_STATIC_CONTEXT_RESERVE` makes pre-run compression reserve
budget for the largest configured DeepAgents system prompt and tool schema.
For summary sizing, `*_RATIO` values are computed from the active compression
model context limit, not the smaller target that active messages must shrink
under. A `*_MAX_TOKENS` value of `0` means "do not apply a fixed absolute cap;
use the ratio-derived value." Set a positive max only when you intentionally
want to pin a smaller operator safety cap for slow or expensive summary models.
In the Observer Shrinking cards, `Compress Limit` is the active-state shrink
target and `Summary Context` is the model context used to size the internal
summary call.
`ALPHARAVIS_COMPRESSION_ENABLE_CHUNKED_SUMMARY` is an experimental opt-in for
very large compression windows: if the summary prompt would otherwise be
pruned, LangGraph summarizes the middle in chunks and then synthesizes the chunk
summaries. The default remains `false`; the existing bounded one-shot summary
path stays active unless the flag is enabled.
When testing chunking, watch the Observer `Shrinking` section:

- `One-shot` means the normal bounded summary path was used.
- `Chunking` means the summary input exceeded its own prompt budget and was
  summarized chunk-by-chunk before synthesis.
- `Prompt Pruned` tells you the one-shot summary input would have been clipped.
- `Prompt Payload`, `Prompt Overhead`, `Chunk Payload`, and `Chunk Overhead`
  show how much of the summary-model window was left for actual middle content
  after wrapper/protected-note text was reserved.
- `Chunk Count`, `Chunk Omitted`, `Chunk Output`, and `Synth Pruned` show
  whether chunking covered the prepared middle cleanly.

If `Chunk Omitted` is positive, exact raw context is still archived, but the
active summary did not receive every prepared chunk. The synthesis prompt will
tell the model to mention archive lookup for omitted middle details. Treat that
as a tuning signal for `ALPHARAVIS_COMPRESSION_SUMMARY_MAX_CHUNKS` or the chunk
sizing settings before making chunking a default.

The Bridge Test UI Observer also has a `Chunking Lab` panel. It starts local
diagnostic runs with `POST /api/chunking/runs` and exposes the same run through
`GET /api/chunking/runs/{run_id}` so operators or agents can inspect the result
without scraping the page. The lab uses AlphaRavis's real context compressor
against a deterministic synthetic web-like corpus, generated tool traces, and
optional variable prompt load. It does not mean the Bridge itself chunks normal
requests. Treat a lab run as healthy when `summary_failed=false`,
`summary_chunking_used=true`, `summary_chunk_omitted_chars=0`, and the rendered
acceptance status is OK.

Use the lab's `Summary Mode` selector carefully:

- `Stub schnell` is a deterministic harness. It checks chunk splitting,
  prompt-overhead accounting, tool pruning/deduplication, API status, and the
  UI, but it does not call the summary LLM.
- `Real LLM` calls an OpenAI-compatible `/chat/completions` summary model. This
  is the mode to use for latency, quality, and promotion evidence.

For `Real LLM`, the Test UI reads `TEST_UI_CHUNKING_SUMMARY_API_BASE`,
`TEST_UI_CHUNKING_SUMMARY_API_KEY`, `TEST_UI_CHUNKING_SUMMARY_MODEL`, and
`TEST_UI_CHUNKING_SUMMARY_TIMEOUT_SECONDS` when set. Otherwise it falls back to
`OPENAI_API_BASE`, `OPENAI_API_KEY`, `ALPHARAVIS_MODEL`, and a 240 second
per-summary-call timeout.

After a lab run completes, open the collapsible Before/After panels to inspect
the prepared compression input and the final synthesized summary side by side.
The same content is available from the run API under `result.comparison`.
`TEST_UI_CHUNKING_TEXT_CAPTURE_CHARS` controls how many characters are retained
for browser/API display; the diagnostic can truncate the display capture even
when the compressor processed the full chunking window.

This mirrors Hermes' preflight estimate, where tool schemas are counted before
the model call. Leave `ALPHARAVIS_STATIC_CONTEXT_RESERVE_TOKENS=0` for automatic
reserve calculation; set a positive value only when you want to pin the reserve
manually. The same reserve is used for handoff and post-run compression, so the
thread is compacted back to a budget that leaves room for the next real agent
request. With `ALPHARAVIS_USE_AGENT_SPECIFIC_CONTEXT_RESERVE=true`, AlphaRavis
uses the selected/active agent's own reserve once that is known and falls back
to the largest agent reserve earlier in the run.

`ALPHARAVIS_ENABLE_FINAL_LLM_BUDGET_GUARD` keeps a final Hermes-style estimate
at the actual model invocation point. It includes active messages plus tool
schemas and model kwargs, and it logs a warning when the assembled request is
near or above the LangGraph hard context budget.

`ALPHARAVIS_ENABLE_FINAL_BUDGET_RESCUE` adds a final compression checkpoint
immediately before the Swarm model call. If the full request estimate is still
over the effective budget, AlphaRavis archives and compresses the middle again
until the request is under budget, bounded by
`ALPHARAVIS_DYNAMIC_COMPRESSION_MAX_PASSES` and
`ALPHARAVIS_DYNAMIC_COMPRESSION_HARD_MAX_PASSES`, before calling the model.
Use the `inspect_context_budget` tool to inspect the same budget fields from
inside an agent run. The tool also returns `compression_summary_budget`, which
contains the dynamic summary output, summary prompt, and chunk output token
budgets derived from the currently detected context length and effective active
limit. Use those fields instead of hardcoding small summary limits in agents or
tools.

`ALPHARAVIS_ENABLE_PROVIDER_OVERFLOW_RETRY` adds one rescue-and-retry path
around the Swarm model invocation if the provider still reports
`context_overflow` or `payload_too_large`. With
`ALPHARAVIS_ENABLE_PROVIDER_CONTEXT_LIMIT_RETRY=true`, the retry path parses
provider error text for the real context window and recomputes the rescue budget
from that smaller limit.

The Bridge Test UI Observer (`/observer`) includes a `Context Budget` strip for
each observed request when budget data is available. It shows the assembled
request budget, effective active/hard limits, remaining budget, detected and
provider-reported context length, rescue pass counts, retry status, and whether
pre-run/final rescue reached budget. Directly below that, `Shrinking` shows
compression cards for pre-run, final rescue, post-run, and handoff scopes when
those scopes exist. Use it to check:

- how much the active state shrank (`Before`, `After`, `Shrink`)
- whether multi-pass compression stopped under budget (`Passes`, `Budget OK`)
- which part was summarized (`Head/Middle/Tail`, `Middle Tokens`)
- whether the summary prompt was pruned or chunked
- whether the archive key exists for exact recall

The `Kompression` detail tab still shows the same data as JSON, including
archive keys, before/after tokens, summary status/errors, prompt-pruning
metrics, chunking metrics, and tool-pruning counters. Non-streaming Bridge
requests now record the same final budget and compression snapshots as
streaming update paths.

## File Safety

AlphaRavis uses one shared local guard for direct file reads/lists/writes:

```text
langgraph-app/file_safety.py
```

It blocks sensitive credential/config paths and internal caches before tool
execution, including `.env`, `.ssh`, `.aws`, `.kube`, `.docker`, shell profiles,
`.git`, `.cache`, and OS/system paths for writes/deletes.

Optional write-root enforcement:

```text
ALPHARAVIS_WRITE_SAFE_ROOT=
```

Leave it empty for the normal Docker layout. Set it to `/workspace` or another
owner-approved directory if you want all AlphaRavis write/delete helpers to be
confined under one root. The artifact and media tools still also enforce their
own artifact/media roots.

## Memory And Compression

Active chat compression happens automatically before route selection and again
after the current LangGraph run finishes when the thread grows above
`ALPHARAVIS_ACTIVE_TOKEN_LIMIT`.

When compression happens, AlphaRavis can show a visible `Memory-Notice`.
The current task brief and latest handoff packet are preserved verbatim when
available, so the next run still knows the plan, completed work, open tasks,
and verification state.

Before route selection, the pre-run guard compresses old active thread state.
This is the guard that prevents the hard cutoff from blocking a small latest
message on an old thread. If normal compression cannot reduce a context that is
already beyond the hard limit, hard trim removes old active messages while
preserving the latest user turn and records the trim in `run_profile`.

Before the swarm starts, AlphaRavis also has a handoff-context guard. If the
agent-path context is already too large after planner/memory/skill setup, it
compresses the beginning of the current run into a handoff summary and archives
the exact original messages, while keeping the task brief, memory/skill hints,
latest handoff packet, and recent messages active.

Useful handoff settings:

```text
ALPHARAVIS_ENABLE_PRE_RUN_COMPRESSION=true
ALPHARAVIS_ENABLE_HARD_CONTEXT_TRIM=true
ALPHARAVIS_HARD_CONTEXT_TRIM_RATIO=0.80
ALPHARAVIS_ENABLE_HANDOFF_CONTEXT_GUARD=true
ALPHARAVIS_HANDOFF_CONTEXT_TOKEN_LIMIT=8500
ALPHARAVIS_HANDOFF_CONTEXT_KEEP_LAST_MESSAGES=16
ALPHARAVIS_HANDOFF_PACKET_MAX_CHARS=4000
ALPHARAVIS_HANDOFF_SUMMARY_MAX_CHARS=2600
```

Agents are instructed to call `build_specialist_report` before `transfer_to_*`.
That JSON report is the handoff packet.

To pause compression for one run, say one of:

```text
keine Kompression
nicht komprimieren
skip compression
no compression
```

To force compression for one run, say one of:

```text
komprimiere jetzt
archiviere jetzt
compress now
```

The force phrases can be replaced with a pipe-separated ENV value:

```text
ALPHARAVIS_MANUAL_COMPRESSION_PATTERNS=komprimiere jetzt|archive now
```

Archive search is thread-scoped by default. Other chat archives are searched
only when you explicitly ask for cross-thread archive search.

## MemoryKernel

The MemoryKernel is the small learning layer inspired by Hermes.

It runs only on the normal agent path. It does not run on Fast Path.

What it does:

- loads tiny curated memories when they match the current turn,
- reminds agents every few turns to save useful durable facts,
- indexes finished turns for later search,
- helps compression preserve memory-worthy details.

Useful settings:

```text
ALPHARAVIS_ENABLE_MEMORY_KERNEL=true
ALPHARAVIS_MEMORY_NUDGE_INTERVAL=10
ALPHARAVIS_ALWAYS_MEMORY_MAX_ITEMS=6
ALPHARAVIS_ALWAYS_MEMORY_MAX_CHARS=2200
```

Curated memory is for compact stable facts, not full chat history. Good
examples:

```text
User prefers concise German explanations.
The big llama.cpp server is the preferred backend for complex work.
Pixelle failures should first check job status before SSH debugging.
```

Long logs, reports, or implementation notes should go to artifacts instead.

## Semantic Vector Memory

pgvector memory is the semantic Inhaltsverzeichnis for AlphaRavis. MongoDB and
LangGraph still own checkpoints, store data, archives, and thread state, but
pgvector stores a catalog plus full retrieval chunks generated from the
complete original source data.

Default:

```text
ALPHARAVIS_VECTOR_BACKEND=pgvector
ALPHARAVIS_ENABLE_PGVECTOR_MEMORY=true
ALPHARAVIS_PGVECTOR_CATALOG_ENABLED=true
ALPHARAVIS_PGVECTOR_STORE_FULL_CHUNKS=true
```

Requirements:

```text
ALPHARAVIS_PGVECTOR_DATABASE_URL=postgresql://postgres:<password>@vectordb:5432/rag_api
ALPHARAVIS_PGVECTOR_EMBEDDING_BASE_URL=http://litellm:4000/v1
ALPHARAVIS_PGVECTOR_EMBEDDING_MODEL=memory-embed
ALPHARAVIS_PGVECTOR_FALLBACK_EMBEDDING_MODEL=memory-embed-fallback
```

`memory-embed` is a LiteLLM route. The default example now routes to Ollama's
native embedding API with `ollama/qwen3-embedding:0.6b`; pull it on the Ollama
host first:

```bash
ollama pull qwen3-embedding:0.6b
```

The fallback route points to `ollama/bge-m3`. For a future OpenAI-compatible
embedding backend such as llama.cpp or LM Studio, set
`EMBEDDING_LITELLM_MODEL=openai/<served-model>` and
`EMBEDDING_API_BASE=http://<embedding-host>:<port>/v1`, then keep AlphaRavis
itself pointed at LiteLLM's `memory-embed` route.
`scripts/render_litellm_config.py` renders the LiteLLM config at container
startup. It enables `drop_params=true` only on routes whose resolved model id
starts with `ollama/`, so LangChain/OpenAIEmbeddings can use the local Ollama
embedding route even when it sends optional OpenAI parameters such as
`encoding_format`. If the route is changed to `openai/<served-model>` for
llama.cpp, LM Studio, or another OpenAI-compatible embedding server, the
renderer leaves parameter dropping disabled for that route.
`qwen3-embedding:0.6b` is expected to support roughly a 32k-token embedding
context. Use the Bridge Test UI `Memory Embed Tester` to confirm the real
accepted size and latency on the running server; its default max probe size is
set near 32k rough tokens.
On the current Ollama host, the previous 4b route returned 2560-dimensional
vectors but became slow well before 32k. The 0.6b default returns
1024-dimensional vectors and is the practical throughput choice; AlphaRavis
pgvector chunking remains smaller and profile-specific:

When changing embedding dimensions, use a new `rag_api` collection. The default
is now `RAG_COLLECTION_NAME=alpharavis_qwen06` so old 4b vectors do not mix with
new 0.6b vectors in LangChain PGVector searches.

LiteLLM and `rag_api` intentionally use separate Postgres databases on the same
`vectordb` server:

```text
litellm -> litellm
rag_api -> rag_api
```

Keep them separate. LiteLLM runs Prisma migrations at startup; sharing the
`rag_api` database can remove or invalidate LangChain PGVector tables.

```text
ALPHARAVIS_PGVECTOR_CHARS_PER_TOKEN=4.0
ALPHARAVIS_PGVECTOR_CHUNK_TOKENS=900
ALPHARAVIS_PGVECTOR_CHUNK_OVERLAP_TOKENS=125
ALPHARAVIS_PGVECTOR_CHAT_CHUNK_TOKENS=700
ALPHARAVIS_PGVECTOR_CHAT_CHUNK_OVERLAP_TOKENS=100
ALPHARAVIS_PGVECTOR_LOG_CHUNK_TOKENS=1200
ALPHARAVIS_PGVECTOR_LOG_CHUNK_OVERLAP_TOKENS=75
ALPHARAVIS_PGVECTOR_CODE_CHUNK_TOKENS=600
ALPHARAVIS_PGVECTOR_CODE_CHUNK_OVERLAP_TOKENS=80
ALPHARAVIS_PGVECTOR_EMBEDDING_TIMEOUT_SECONDS=45
```

AlphaRavis chooses the chunk profile from `source_type`, filename/path metadata,
Markdown code fences, and common code/log syntax. Code detection is intentionally
heuristic for now; a later Tree-sitter/AST splitter can cut by function/class
boundaries more precisely.

Use the 32k context window for capability testing and exceptional large-query
cases, not as the normal archive/memory chunk size.

Agents can call:

```text
semantic_memory_search
query_source
query_sources
query_archive
agentic_rag_retrieve
```

It searches the current thread plus global memories by default and also queries
the existing RAG API for external documents. It searches other AlphaRavis
threads only when `include_other_threads=true` is explicitly requested.
Use `query_source` / `query_sources` when the agent already has a source key,
archive key, artifact key, or external RAG `file_id` and should retrieve only
relevant chunks from that known source. Use `query_archive` for a known archive
key before deciding whether the raw `read_archive_record` payload is needed.
Use `agentic_rag_retrieve` when the known-source question needs a
retrieve/grade/rewrite pass and a bounded `context_packet` for a grounded
answer. It stays tool-only; it does not automatically load complete archives
into active context.
For newly ingested explicit documents or large pastes, AlphaRavis records
thread-level RAG metadata (`rag_active`, active source keys, optional external
file ids, and activation reason) so a later auto-retrieval node can use bounded
chunks. Compression archives remain passive by default with
`archive_rag_mode=tool_only`.
Large pasted messages above `ALPHARAVIS_LARGE_PASTE_RAG_MIN_CHARS` are indexed
first; after successful indexing, the model sees a compact retrieval marker
instead of the entire paste.
Document and large-paste ingest now defaults to the AlphaRavis-owned pgvector
backend, not `rag_api`:

```text
ALPHARAVIS_DOCUMENT_RAG_BACKEND=alpharavis_pgvector
```

Set it to `rag_api` to use the current external adapter for comparison, or
`both` for dual indexing during evaluation. When AlphaRavis pgvector is the only
document backend, active threads keep `active_source_keys` but do not add
`active_rag_file_ids`, so automatic prefetch does not call `rag_api` unless the
source was actually mirrored there.
When `ALPHARAVIS_ENABLE_ACTIVE_RAG_PREFETCH=true`, active document/large-paste
threads automatically prefetch bounded chunks into `<active-rag-context>`.
Archive-only threads remain tool-only unless a future `auto_on_intent` archive
mode is enabled.
If weak pgvector hits are too noisy, set
`ALPHARAVIS_PGVECTOR_DISTANCE_THRESHOLD` for AlphaRavis's own pgvector table.
This is separate from `rag_api`'s `RAG_DISTANCE_THRESHOLD`, but uses the same
distance-cutoff idea: lower pgvector distance means a stronger match.
After enabling pgvector memory, new records are indexed automatically. Old
MongoDB/store history is not bulk-backfilled by default, to avoid a surprise
embedding job over many chats.

New records go into a durable embedding queue by default:

```text
ALPHARAVIS_PGVECTOR_INDEX_MODE=queue
ALPHARAVIS_PGVECTOR_QUEUE_TABLE=alpharavis_embedding_jobs
ALPHARAVIS_EMBEDDING_JOB_BATCH_SIZE=10
```

The Power Management Agent can drain that queue with `run_embedding_memory_jobs`.

Archive mirroring into `rag_api` is prepared but off by default:

```text
ALPHARAVIS_ENABLE_RAG_ARCHIVE_MIRROR=false
ALPHARAVIS_RETRIEVAL_PREFER_RAG_MIRRORS=true
ALPHARAVIS_RAG_ENTITY_ID=alpharavis
```

When enabled, newly created compression archives are still stored normally in
MongoDB/LangGraph Store, but AlphaRavis also uploads the archive text to
`rag_api` as `file_id=archive:<archive_key>`. `query_archive(...)` can then use
the `rag_api` mirror for bounded chunk retrieval and fall back to AlphaRavis
pgvector if the mirror is missing or unavailable. `read_archive_record(...)`
remains the explicit raw-history tool and is not the normal first retrieval
step.

The Bridge Test UI Observer has an `Archive RAG Smoke` panel for this path. It
creates a small archive-shaped payload, sends it to `rag_api`, queries it back,
and reports acceptance checks plus runtime errors. A failed smoke with
`memory-embed` connection errors means the wrapper reached LiteLLM but the
configured embedding backend is not available.
The same Observer page has `Memory Embed Tester` for bringing that backend up:
enter the target base URL/IP, model name, OpenAI-compatible or Ollama mode,
choose text or experimental vision input, then run the probe. It reports
embedding dimensions, latency per input size, max accepted chars/tokens, and
whether the backend rejected or became too slow.
It is allowed when the big llama.cpp server is active or the system has been
idle long enough, depending on `ALPHARAVIS_EMBEDDING_LOAD_POLICY`.

To let LangGraph drain the queue automatically:

```text
ALPHARAVIS_ENABLE_EMBEDDING_SCHEDULER=true
ALPHARAVIS_EMBEDDING_SCHEDULER_INTERVAL_SECONDS=120
```

The lifecycle runner pauses if the small Ollama chat/crisis model is already
loaded and `ALPHARAVIS_EMBEDDING_UNLOAD_CHAT_MODEL=false`. This avoids stealing
the management node from crisis work. If you want the runner to unload the small
chat model for embedding windows, set:

```text
ALPHARAVIS_EMBEDDING_UNLOAD_CHAT_MODEL=true
```

Manual bounded backfill is available through:

```text
queue vector memory backfill
```

The optional daemon is default off and requires a query:

```text
ALPHARAVIS_ENABLE_VECTOR_BACKFILL_DAEMON=true
ALPHARAVIS_VECTOR_BACKFILL_QUERY=project-name-or-topic
ALPHARAVIS_VECTOR_BACKFILL_LIMIT_PER_SOURCE=10
```

That daemon searches existing Store indexes and queues matching records. It is
not an automatic full-history backfill.

## Session Search And Artifacts

AlphaRavis now keeps an indexed per-turn history, similar in spirit to Hermes
session search.

Agents can search it with:

```text
search_session_history
```

Current-thread search is default. Other threads are searched only when the user
explicitly asks for it.

Artifacts are disk-backed files for large outputs:

```text
write_alpha_ravis_artifact
read_alpha_ravis_artifact
list_alpha_ravis_artifacts
```

Default artifact root:

```text
/workspace/artifacts/alpharavis
```

Use artifacts when a result is too large for chat but should still be
recoverable later.

Artifact reads and writes go through the central file-safety guard, then through
the artifact-root check. This means an artifact cannot escape its root, and a
misconfigured artifact path cannot be used to read or write secrets.

## Agent-Specific Memories

Agents also have scoped durable memories:

```text
alpharavis / agent_memories / general_assistant
alpharavis / agent_memories / research_expert
alpharavis / agent_memories / debugger_agent
alpharavis / agent_memories / context_retrieval_agent
alpharavis / agent_memories / global
```

The intended rule is:

- search the active agent's own memory first,
- include global memories for stable cross-agent preferences,
- record new memories only after a useful lesson or repeated preference is clear,
- keep thread archives separate from reusable agent memories.

The agent knows which memory to use from its role prompt. For example, the
debugger uses `agent_id=debugger_agent`, while cross-agent lessons use
`scope=global`.

## Skill Library

The skill library stores reusable workflow patterns.

Safety rules:

- New workflows become inactive candidates.
- Candidates do not affect routing.
- Active skills are hints, not automatic execution.
- Promotion is disabled unless `ALPHARAVIS_ALLOW_SKILL_PROMOTION=true`.

Useful review commands:

```text
zeige Skill-Kandidaten
aktiviere Skill <key>
deaktiviere Skill <key>
```

Activation/deactivation is blocked unless review mode is enabled:

```text
ALPHARAVIS_ALLOW_SKILL_PROMOTION=true
```

Reviewed repo skill cards under `ai-skills/` are different from Mongo skill
candidates. AlphaRavis may inject only a tiny metadata hint when a card seems
relevant. It reads the full card only through `read_repo_ai_skill` when needed.
The repo skill index is cached by an mtime/size manifest so normal runs do not
rescan every skill file.

```text
ALPHARAVIS_REPO_SKILL_HINT_LIMIT=3
ALPHARAVIS_REPO_SKILL_CACHE=true
ALPHARAVIS_REPO_SKILL_CACHE_PATH=.cache/alpharavis/repo_skill_manifest.json
ALPHARAVIS_REPO_SKILL_SUPPORTING_FILE_LIMIT=40
ALPHARAVIS_REPO_SKILL_INCLUDE_DRAFTS=false
```

Use `reload_repo_ai_skills` when you add, remove, or edit disk skills and want
AlphaRavis to report what changed. Reloading only refreshes the manifest; it
does not promote Mongo skill candidates or change routing.

Disk skills may have supporting files in:

```text
references/
templates/
scripts/
assets/
```

Use `read_repo_ai_skill("skill-name", "references/file.md")` or the matching
supporting path to load one of those files. The same central file-safety guard
keeps reads inside the requested skill directory.

Candidate export is separate from activation. `record_skill_candidate` still
creates an inactive Mongo candidate. When review mode is intentionally enabled,
`export_skill_candidate_to_repo_draft` can write a draft under
`ai-skills/_drafts/<slug>/SKILL.md`; the Store candidate remains inactive.

```text
ALPHARAVIS_ALLOW_SKILL_DRAFT_EXPORT=false
ALPHARAVIS_REPO_SKILL_DRAFT_DIR=ai-skills/_drafts
```

## Run Profile

Every run stores timing and routing data in LangGraph state as `run_profile`.

Typical fields:

- route: `fast_path` or `swarm`
- route_reason
- message_count
- token_estimate
- total_seconds
- fast_path_seconds
- fast_path_fallback_used

Set `ALPHARAVIS_SHOW_RUN_PROFILE=true` if you want this appended visibly in
LibreChat. Otherwise inspect it in LangGraph Studio or DeepAgents UI.

## Operational Logging

AlphaRavis has a local rotating log layer in addition to LangGraph Studio and
optional LangSmith tracing.

Default files:

```text
logs/operational/alpharavis.log
logs/operational/alpharavis.jsonl
```

Enable the noisier all-debug logger only while diagnosing a problem:

```text
logs/debug/alpharavis-debug.log
logs/debug/alpharavis-debug.jsonl
```

The operational logger records route decisions, run start/finish, bridge
requests, LLM call duration/failures, Pixelle/ComfyUI preflight, pgvector/RAG
search status, and dependency health. It does not intentionally write full chat
content into the normal log. The debug-all logger can include more routing/tool
detail, but values are still secret-redacted and truncated.

```text
ALPHARAVIS_OPERATIONAL_LOGGING=true
ALPHARAVIS_DEBUG_ALL_LOGGING=false
ALPHARAVIS_LOG_RETENTION_DAYS=4
ALPHARAVIS_LOG_DIR=logs
```

In Docker, `langgraph-api` and `api-bridge` mount `./logs` into `/logs`, so both
services write to the same host-side log folder.

## Current Optimization Notes

Already available:

- OpenAI-compatible LibreChat bridge
- LangGraph native brain
- fast path for simple chat
- run profile state
- skill-library candidate listing and review-mode activation/deactivation
- reviewed repo skill-card hints and on-demand skill-card reading
- DeepAgents-style MCP config loading, disabled by default for faster simple chat
- optional Hermes direct endpoint for LibreChat and Hermes coding sub-agent for AlphaRavis
- fast-path hidden-thinking disable for llama.cpp/Qwen-style models
- visible fast-path notices and thread lockout after agent path
- graph-level and bridge-level hard context cutoffs
- agent-specific and global memory tools
- MemoryKernel with curated always-memory, turn indexing, and compression hints
- session-history search over indexed turns
- disk-backed AlphaRavis artifacts for large reports/logs/plans
- thread-scoped memory archives
- manual one-run chat compression
- structured specialist reports for research/debug/context handoffs
- async Pixelle start/status tools for long image jobs
- visible memory notices
- command approval gate
- LLM generation health endpoint
- owner-only power tools for llama.cpp and ComfyUI, default off
- protected owner shutdown tools behind human approval
- token-light crisis preflight/recovery agent, default off
- OpenAPI 3.1 bridge schema and richer Responses streaming event names
- Responses-native direct LangGraph calls for planner/fast-path/summarizers
- Responses-native DeepAgents model binding through LangChain `ChatOpenAI`,
  feature-flagged with ChatLiteLLM fallback; internal DeepAgents token
  streaming is disabled by default until the LiteLLM/llama.cpp Responses
  streaming path is stable
- `make model-management` / `make owner-model-management` for custom hardware setup
- durable pgvector embedding queue, scheduler, manual queue runner, and bounded
  backfill queueing
- Pixelle owner wake guard for ComfyUI, default off through model management
- safe media handling in the Bridge: URL/file-id/type metadata is passed instead
  of raw images/videos unless explicitly enabled
- media-gallery service for Pixelle MCP outputs and uploaded/linked media metadata
- separate optional media/vision pgvector table to avoid text/vision dimension
  conflicts
- lazy toolset binding for specialist coding, media, RAG, system, Hermes,
  debugger, and context bundles
- OpenWebUI optional frontend profile using the AlphaRavis Bridge
- Hermes healthcheck/fallback before bounded coding-agent calls

Still open / planned next:

- mid-run backend watchdog and crisis recovery for timeouts/502s after a graph
  run already started
- post-crisis readiness gate before continuing to the normal planner
- remove the local LangChain #35436 patch after `langchain-openai` ships an
  upstream fix and the direct tool-calling repro still passes
- agent time/tool/handoff budget guard
- richer activity stream in LibreChat
- test whether LibreChat shows `reasoning_content` in a separate reasoning
  panel before enabling reasoning streaming by default
- optional parallel agent execution with dependency groups
- full video analysis pipeline: keyframes, timecodes, captions, transcription,
  and frame-level vision embeddings
- true internal dynamic tool binding/unbinding per run; current implementation
  exposes category manifests and keeps concrete LangGraph tools available

## Media And Uploads

By default the Bridge does not forward raw media blocks into LangGraph:

```text
BRIDGE_ALLOW_RAW_MEDIA_CONTEXT=false
BRIDGE_MEDIA_CONTEXT_MODE=metadata
BRIDGE_MEDIA_GALLERY_AUTO_REGISTER_VIDEOS=true
```

That means uploads/links arrive as metadata markers containing fields such as
type, file id, URL, mime type, or title. This prevents a video or image blob from
filling the LLM context. Incoming video blocks are mirrored into `media-data`
through `media-gallery` first, then the AlphaRavis-facing marker uses the stable
gallery URL. LibreChat's visible attachment card and original upload file are
not rewritten by this Bridge step. Use `register_media_asset` to save other
URL/file-id media manually. Registration is metadata-only by default; set the
tool's `index=true` only when you explicitly want immediate vision indexing.

For LibreChat video uploads, use the normal `AlphaRavis Responses` model spec.
The LibreChat container applies `scripts/patch_librechat_video_uploads.js` at
startup so the `LangGraph Agent` endpoint sends videos as `video_url`
attachments to the AlphaRavis Bridge while still using `useResponsesApi: true`.
The patch covers both the backend video encoder and the browser upload
menu/drag-drop bundle, plus LibreChat's prompt formatter that otherwise keeps
`image_urls` but drops `videos` before the provider HTTP request. In Responses
mode it also patches LibreChat's OpenAI Responses converter so `video_url` parts
become `input_video` instead of being filtered out after formatting. If the
browser still shows only `Upload as Text` after a restart, reload the LibreChat
page once so the local service worker unregisters and clears Workbox caches,
then reload once more so the patched client bundle is loaded.
Do not choose LibreChat's `Upload as Text` option for videos; LibreChat can parse
the video bytes as huge text, hit its own agent-context pruning limit, and fail
before the Bridge receives the request.

When the user explicitly asks to analyze, inspect, describe, summarize,
transcribe, compare, or index a video, AlphaRavis can use
`prepare_media_for_model`. That tool decides between `register_only`,
`pass_through`, `analyze`, and `index`; it only downloads video for
`analyze`/`index`, and only when:

```text
ALPHARAVIS_VIDEO_ANALYSIS_ENABLED=true
```

The video preparation path uses `ffprobe`/`ffmpeg`, samples at no more than
`ALPHARAVIS_VIDEO_ANALYSIS_MAX_FPS`, caps frames with
`ALPHARAVIS_VIDEO_ANALYSIS_MAX_FRAMES`, stores timestamped frames/manifests
under the media-data analysis cache, and indexes sampled frames into
`alpharavis_media_vectors` when `ALPHARAVIS_ENABLE_VISION_VECTOR_MEMORY=true`
and a compatible vision embedding route exists.

Use `semantic_media_search` to search indexed media semantically. Use
`inspect_media_index_status` to check whether media/frame records have already
been processed by the vision embedding path. Use
`inspect_embedding_queue_status` to see how much text/archive/media indexing
work is still pending, running, failed, or done in `alpharavis_embedding_jobs`.

When `prepare_media_for_model` is called with `mode=index`, it queues a durable
`media_analysis` job in the same embedding queue used for text, archive,
artifact, memory, and session-turn indexing. `run_embedding_memory_jobs` drains
that shared queue during the existing embedding/model-management window.

Pixelle output URLs are registered automatically when the job result contains a
media URL. The gallery runs at:

```text
http://localhost:8130/gallery
```

Use the gallery tabs to inspect grouped assets:

```text
http://localhost:8130/gallery?view=all
http://localhost:8130/gallery?view=original
http://localhost:8130/gallery?view=processed
```

Video analysis remains explicit, not automatic. For Pixelle input, AlphaRavis
should pass the copied URL through without downloading unless the downstream
service requires a local file.

Media gallery presence does not mean the visual content is indexed. A gallery
asset means AlphaRavis knows the file URL/path and metadata. Indexing status is
separate:

```text
inspect_media_index_status
inspect_embedding_queue_status
```

The media server also records chat/tool appearances as media references, so one
file can appear in multiple chat turns without being embedded repeatedly.
Automatic indexing can be tuned:

```text
ALPHARAVIS_MEDIA_AUTO_INDEX_ENABLED=true
ALPHARAVIS_MEDIA_AUTO_INDEX_USER_UPLOADS=true
ALPHARAVIS_MEDIA_AUTO_INDEX_PIXELLE_MCP_OUTPUTS=false
ALPHARAVIS_MEDIA_AUTO_INDEX_LINK_REFERENCES=false
ALPHARAVIS_MEDIA_INDEX_VERSION=2026-05-12-v1
ALPHARAVIS_MEDIA_VISION_EMBEDDING_MODEL_CARD=vision-embed
```

Vision embeddings are experimental and remain off by default. Do not enable
them for the normal memory/RAG bring-up path; first get the text
`memory-embed` route green. Later, vision embeddings can use either the normal
LiteLLM route or a dedicated external OpenAI-compatible server, but only after
the Memory Embed Tester proves the selected endpoint accepts the vision payload
and returns vectors. For a separate llama.cpp vision embedding server, set the
direct model URL:

```bash
make media-vision VISION_ENABLED=true \
  VISION_URL=http://<vision-embedding-host>:<port>/v1 \
  VISION_MODEL=<model-name-served-by-that-endpoint>

make up VISION_URL=http://<vision-embedding-host>:<port>/v1 \
  VISION_MODEL=<model-name-served-by-that-endpoint>
```

The same `VISION_*` variables are accepted by `make install`, `make update`,
`make up-fullstreaming`, and `make up-chat-fullstreaming`. They write:

```text
ALPHARAVIS_ENABLE_VISION_VECTOR_MEMORY=true
ALPHARAVIS_VISION_EMBEDDING_MODEL_URL=http://<vision-embedding-host>:<port>/v1
ALPHARAVIS_VISION_EMBEDDING_MODEL=<model-name-served-by-that-endpoint>
ALPHARAVIS_VISION_EMBEDDING_API_KEY=sk-local-dev
```

If `ALPHARAVIS_VISION_EMBEDDING_MODEL_URL` is empty, AlphaRavis uses
`ALPHARAVIS_VISION_EMBEDDING_BASE_URL`, then `VISION_EMBEDDING_API_BASE`, then
the text pgvector/OpenAI base fallback. Captioning/OCR/transcription are still
separate future work; this only configures the vector embedding route.

There is no second queue table for vision. Vision/video indexing jobs use the
shared durable `alpharavis_embedding_jobs` queue with `job_type=media_analysis`;
`run_embedding_memory_jobs` drains those jobs during the normal embedding
maintenance window.

The dedupe key is based on media source key, model-card id, index version, and
chunking config. If the same video is referenced five times, AlphaRavis should
store five references but one index for that model/version/config.

## OpenWebUI

OpenWebUI is optional:

```text
docker compose --profile openwebui up -d openwebui
```

It uses the AlphaRavis Bridge as the OpenAI-compatible provider. In OpenWebUI,
set Function Calling to `Native` for capable models. Keep web search disabled
until SearXNG or another search backend is configured. Passthrough is useful for
Responses/custom endpoints, but it forwards upstream requests with the configured
OpenWebUI provider key, so keep it owner-only or disable it on shared instances.
