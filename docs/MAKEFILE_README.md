# AlphaRavis Makefile README

This is the operator reference for the AlphaRavis `Makefile`. Use it when you
want to know which target to run, which variables it accepts, and how Tailscale
HTTPS mode differs from plain LAN HTTP mode.

## Quick Commands

```bash
make help
make config
make install
make update
make status
make up
make down
```

Common first-run flow:

```bash
make config
make install
make status
```

Common update flow:

```bash
make update
make status
```

## Network Modes

The default Makefile mode is Tailscale Serve HTTPS mode.

In that mode, Docker application ports bind to localhost only:

```text
ALPHARAVIS_DOCKER_HOST_BIND=127.0.0.1
```

Tailscale Serve owns the Tailnet IP ports and proxies HTTPS traffic back to
`http://127.0.0.1:<port>`. Local HTTP still works on the machine itself, for
example `http://localhost:3080`, while Tailnet devices use HTTPS.

Enable or refresh Tailscale HTTPS mode:

```bash
make tailscale-apply
```

Switch to LAN HTTP mode:

```bash
make tailscale-disable
```

LAN HTTP mode disables the managed Tailscale Serve routes, removes the service
dashboard HTTPS override file, writes:

```text
ALPHARAVIS_DOCKER_HOST_BIND=0.0.0.0
```

and recreates Docker services so application ports are reachable from the host's
LAN IP.

The same switch can be used during normal install/update/up:

```bash
make install TAILSCALE_AUTO=off
make update TAILSCALE_AUTO=off
make up TAILSCALE_AUTO=off
```

Use `TAILSCALE_AUTO=keep` when you want a run to leave the current network mode
untouched:

```bash
make up TAILSCALE_AUTO=keep
```

## Main Targets

| Target | Purpose |
| --- | --- |
| `make help` | Print the built-in Makefile help. |
| `make config` | Open the local browser UI for editing `.env` from `.env(exaple)` defaults. |
| `make install` | Sync missing `.env` defaults, choose runtime settings, optionally update submodules, build, and start. Defaults to Tailscale HTTPS mode. |
| `make update` | Run `git pull --ff-only`, update runtime settings, optionally update submodules, build, and restart. Defaults to Tailscale HTTPS mode. |
| `make update-no-start` | Same update path, but builds without starting/recreating the full stack. |
| `make status` | Print service URLs, runtime profile, network exposure mode, and `docker compose ps`. |
| `make up` | Run `docker compose up -d --build` after applying the selected network mode. |
| `make down` | Run `docker compose down`. |
| `make logs` | Follow Compose logs with `--tail=120`. |
| `make build` | Build the core AlphaRavis images. |
| `make submodules` | Update submodules recursively from configured remote branches. |

## Configuration Targets

| Target | Purpose |
| --- | --- |
| `make configure` | Terminal prompt for important `.env` values. |
| `make profiles` | Print runtime profiles and the `.env` values they write. |
| `make streaming STREAMING=<mode>` | Update runtime/streaming `.env` values only. |
| `make model-management` | Configure custom model management and owner power-tool settings. |
| `make owner-model-management` | Alias for `make model-management`. |
| `make media-vision ...` | Configure media gallery and vision embedding settings. |
| `make vision-embedding ...` | Alias for `make media-vision`. |
| `make video-analysis ...` | Configure video analysis settings. |
| `make openwebui` | Configure optional OpenWebUI settings. |

## Runtime Profile Targets

These are shortcuts around `make install` or `make streaming`.

| Target | Effect |
| --- | --- |
| `make install-fullstreaming` | Install/start with Responses full streaming. |
| `make install-hybrid` | Install/start with Responses hybrid streaming. |
| `make install-nonstreaming` | Install/start with Responses non-streaming. |
| `make install-chat` | Install/start with Chat Completions streaming. |
| `make install-chat-fullstreaming` | Install/start with Chat Completions streaming. |
| `make install-chat-nonstreaming` | Install/start with Chat Completions non-streaming. |
| `make fullstreaming` | Set Responses full streaming in `.env`. |
| `make full-streaming` | Alias for `make fullstreaming`. |
| `make hybrid-streaming` | Set Responses hybrid streaming in `.env`. |
| `make nonstreaming` | Set Responses non-streaming in `.env`. |
| `make chat-completions` | Set Chat Completions streaming in `.env`. |
| `make chat-fullstreaming` | Set Chat Completions streaming in `.env`. |
| `make chat-nonstreaming` | Set Chat Completions non-streaming in `.env`. |
| `make up-fullstreaming` | Set Responses full streaming, then recreate `langgraph-api`, `api-bridge`, and `bridge-test-ui`. |
| `make up-chat-fullstreaming` | Set Chat Completions streaming, then recreate `langgraph-api`, `api-bridge`, and `bridge-test-ui`. |

## UI And Service Targets

| Target | Purpose |
| --- | --- |
| `make service-dashboard` | Start only the AlphaRavis Service Dashboard. |
| `make dashboard` | Alias for `make service-dashboard`. |
| `make test-ui` | Start or rebuild only the Bridge Test UI. |

## Tailscale Targets

| Target | Purpose |
| --- | --- |
| `make tailscale-plan` | Print planned Tailscale Serve HTTPS routes. Does not change Tailscale state. |
| `make tailscale-overrides` | Write dashboard HTTPS override JSON only. Does not change Tailscale Serve state. |
| `make tailscale-apply` | Set Docker app ports to localhost, recreate services, apply Tailscale Serve HTTPS routes, and refresh dashboard overrides. |
| `make tailscale-disable` | Disable managed Tailscale Serve routes, remove dashboard overrides, set Docker app ports to `0.0.0.0`, and recreate services. |
| `make disable-tailscale` | Alias for `make tailscale-disable`. |
| `make tailscale-status` | Print `tailscale serve status`. |

Internal helper targets also exist:

| Target | Purpose |
| --- | --- |
| `make tailscale-prep` | Apply the pre-Docker network-mode step used by install/update/up. |
| `make tailscale-auto` | Apply the post-Docker automatic Tailscale step used by install/update/up. |
| `make tailscale-routes-apply` | Apply only the Tailscale Serve routes and dashboard overrides. |
| `make tailscale-routes-disable` | Disable only the managed Tailscale Serve routes and remove dashboard overrides. |

Prefer the public `tailscale-apply` and `tailscale-disable` targets unless you
are debugging the Makefile flow itself.

## Smoke Targets

| Target | Purpose |
| --- | --- |
| `make bridge-smoke` | Send a small OpenAI-compatible request to the AlphaRavis bridge. |
| `make hermes-smoke` | Send a small OpenAI-compatible request to Hermes. |
| `make media-smoke` | Check the media gallery health endpoint. |
| `make openwebui-smoke` | Check the OpenWebUI HTTP endpoint. |

## Variables

Make variables are passed as `NAME=value`:

```bash
make install STREAMING=chat-full START=yes PROFILES=openwebui
```

### Core

| Variable | Default | Used by | Meaning |
| --- | --- | --- | --- |
| `PYTHON` | `python` | Most targets | Python executable. |
| `COMPOSE` | `docker compose` | Docker targets | Docker Compose command. |

### Install

| Variable | Default | Values | Meaning |
| --- | --- | --- | --- |
| `STREAMING` | `prompt` | `prompt`, `keep`, `hybrid`, `full`, `nonstreaming`, `chat`, `chat-full`, `chat-nonstreaming`, full names below | Runtime profile for `make install`. |
| `SUBMODULES` | `prompt` | `prompt`, `yes`, `no` | Whether install updates submodules. |
| `BUILD` | `prompt` | `prompt`, `yes`, `no` | Whether install builds images when it is not starting the stack. Ignored when `START=yes`. |
| `START` | `prompt` | `prompt`, `yes`, `no` | Whether install starts/recreates the stack. |
| `PROFILES` | `prompt` | `prompt`, `keep`, `none`, `openwebui`, `hermes-dashboard`, comma-separated profiles | Compose profiles to write/use. |

Examples:

```bash
make install STREAMING=full PROFILES=openwebui
make install STREAMING=chat-full START=yes SUBMODULES=yes PROFILES=none
make install START=no BUILD=yes
```

### Update

| Variable | Default | Values | Meaning |
| --- | --- | --- | --- |
| `UPDATE_STREAMING` | `prompt` | Same as `STREAMING` | Runtime profile for `make update`. |
| `UPDATE_SUBMODULES` | `yes` | `prompt`, `yes`, `no` | Whether update refreshes submodules from configured remotes. |
| `UPDATE_BUILD` | `yes` | `prompt`, `yes`, `no` | Whether update builds images when it is not starting the stack. |
| `UPDATE_START` | `yes` | `prompt`, `yes`, `no` | Whether update starts/recreates the stack. |
| `UPDATE_PROFILES` | `prompt` | Same as `PROFILES` | Compose profiles to write/use during update. |

Examples:

```bash
make update
make update UPDATE_STREAMING=keep UPDATE_SUBMODULES=no
make update-no-start UPDATE_STREAMING=chat-full
```

### Runtime Profiles

Accepted profile names:

| Profile | Aliases | Effect |
| --- | --- | --- |
| `responses-hybrid` | `hybrid`, `responses`, `stable`, `tool_calling` | Responses API; no-tool calls may stream, tool-bound DeepAgents calls stay non-streaming. Stable default. |
| `responses-full` | `full`, `fullstreaming`, `full-streaming`, `responses-fullstreaming` | Responses API full streaming with the experimental tool-streaming patch. |
| `responses-nonstreaming` | `nonstreaming`, `non-streaming`, `false`, `off`, `none`, `no` | Responses API with internal model streaming disabled. |
| `chat-full` | `chat`, `chat-completions`, `chat_completions`, `legacy` | Chat Completions mode with ChatLiteLLM streaming enabled. |
| `chat-nonstreaming` | `chat-non-streaming`, `chat-completions-nonstreaming` | Chat Completions mode with streaming disabled. |

### Tailscale And Network Exposure

| Variable | Default | Values | Meaning |
| --- | --- | --- | --- |
| `TAILSCALE_AUTO` | `apply` | `apply`, `on`, `yes`, `off`, `false`, `no`, `lan`, `keep`, `skip`, `none` | Controls automatic network-mode handling around install/update/up. Default enables Tailscale HTTPS mode. `off` means LAN HTTP mode. `keep` leaves current mode untouched. |
| `TAILSCALE_HOST` | empty | MagicDNS host, for example `device.tailnet.ts.net` | Optional explicit Tailnet hostname. If omitted, the helper auto-detects it from `tailscale status --json` when needed. |
| `TAILSCALE_SUDO` | `auto` | `auto`, `true`, `false`, `always`, `never` | Sudo behavior for Tailscale CLI commands. `auto` retries with sudo only after a permission error. |
| `TAILSCALE_DASHBOARD` | `true` | `true`, `false` | Whether to include the Service Dashboard itself in Tailscale routes. |
| `TAILSCALE_EXTRA` | empty | Extra CLI args | Advanced passthrough to `tailscale_https_routes.py`. |

Examples:

```bash
make tailscale-plan TAILSCALE_HOST=cachyos-comfyui.tail852b38.ts.net
make tailscale-apply TAILSCALE_SUDO=auto
make tailscale-disable
make up TAILSCALE_AUTO=off
make update TAILSCALE_AUTO=keep
```

### Config UI

| Variable | Default | Meaning |
| --- | --- | --- |
| `CONFIG_HOST` | `127.0.0.1` | Host for the config browser UI server. |
| `CONFIG_PORT` | `8765` | Port for the config browser UI server. |

Example:

```bash
CONFIG_HOST=127.0.0.1 CONFIG_PORT=8765 make config
```

### Media And Vision

| Variable | Default | Meaning |
| --- | --- | --- |
| `VISION_ENABLED` | empty | `true`, `false`, or `keep` for vision vector memory. |
| `VISION_URL` | empty | Direct OpenAI-compatible `/v1` URL for a dedicated vision embedding server. |
| `VISION_BASE_URL` | empty | LiteLLM/OpenAI-compatible base URL for fallback vision embeddings. |
| `VISION_MODEL` | empty | Primary vision embedding model id. |
| `VISION_FALLBACK` | empty | Fallback vision embedding model id. |

Accepted by `make media-vision`, `make install`, `make update`, `make up`,
`make up-fullstreaming`, and `make up-chat-fullstreaming`.

Examples:

```bash
make media-vision VISION_ENABLED=true VISION_URL=http://192.168.178.50:8080/v1 VISION_MODEL=vision-embed
make up VISION_URL=http://192.168.178.50:8080/v1 VISION_MODEL=vision-embed
make update VISION_ENABLED=keep VISION_MODEL=vision-embed
```

### Video Analysis

| Variable | Default | Values | Meaning |
| --- | --- | --- | --- |
| `ENABLED` | `keep` | `keep`, `true`, `false` | Enable or disable video analysis. |
| `FPS` | empty | Number | Sample FPS and max FPS for analysis. |
| `MAX_FRAMES` | empty | Number | Maximum sampled frames. |

Example:

```bash
make video-analysis ENABLED=true FPS=1 MAX_FRAMES=100
```

## Important URLs

Local URLs are still useful in both modes from the host machine itself:

| Service | Local URL |
| --- | --- |
| Service Dashboard | `http://localhost:8090` |
| LibreChat | `http://localhost:3080` |
| LangGraph API | `http://localhost:2024` |
| OpenAI-compatible AlphaRavis bridge | `http://localhost:8123/v1` |
| Bridge Test UI | `http://localhost:8140` |
| Hermes API | `http://localhost:8642/v1` |
| LiteLLM | `http://localhost:4000/v1` |
| RAG API | `http://localhost:8000` |
| Media Gallery | `http://localhost:8130/gallery` |
| DeepAgents UI | `http://localhost:3000` |
| Agent Custom UI | `http://localhost:3001` |
| Pixelle MCP | `http://localhost:9004` |

When Tailscale HTTPS mode is active, the Service Dashboard can show HTTPS URLs
from `service-dashboard-data/tailscale_service_urls.json`, for example:

```text
https://<device>.<tailnet>.ts.net:3080
https://<device>.<tailnet>.ts.net:9004
```

## Troubleshooting

If Docker fails with an error like `address already in use` on a port also used
by Tailscale Serve, switch to Tailscale HTTPS mode:

```bash
make tailscale-apply
```

If Tailnet HTTPS is not desired and you want direct LAN HTTP access, switch to
LAN HTTP mode:

```bash
make tailscale-disable
```

If you only want to inspect what Tailscale routes would be applied:

```bash
make tailscale-plan
```

If you want to avoid changing network mode during a one-off run:

```bash
make up TAILSCALE_AUTO=keep
```
