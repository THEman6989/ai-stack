# Hermes Patterns For AlphaRavis

Source inspected: `C:/experi/ai/hermes-agent`.

## Gateway API

Hermes has an OpenAI-compatible API-server plan and gateway integration.

Expected endpoints:

- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /health`

Typical environment:

```text
API_SERVER_ENABLED=true
API_SERVER_PORT=8642
API_SERVER_KEY=<secret>
```

For Docker callers, use:

```text
http://host.docker.internal:8642/v1
```

The API is intended to work with Open WebUI, LobeChat, LibreChat, AnythingLLM,
and normal OpenAI-compatible clients.

## Memory Pattern

Hermes keeps bounded curated memory in small prompt-injected stores:

- `MEMORY.md`: agent notes, environment facts, conventions, learned patterns.
- `USER.md`: user profile, preferences, expectations.

Config examples from Hermes:

```yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200
  user_char_limit: 1375
  nudge_interval: 10
  flush_min_turns: 6
```

AlphaRavis equivalent:

- Keep large thread history in LangGraph archives.
- Keep durable small facts in agent/global memories.
- Keep reusable procedures in repo or Mongo skills.
- Add memory nudges only when a repeated preference or lesson is clear.

## Skills Pattern

Hermes treats skills as reusable procedures the agent can load and follow.

Important ideas to borrow:

- Skill creation nudge after complex tool work.
- External read-only skill directories.
- Skill chaining when a task needs multiple procedures.
- Install heavy/niche skills separately instead of loading everything.

Config concepts:

```yaml
skills:
  creation_nudge_interval: 15
  external_dirs:
    - ~/.agents/skills
    - /home/shared/team-skills
```

AlphaRavis equivalent:

- Reviewed repo skills live in `ai-skills/`.
- Mongo skill candidates stay inactive until promoted.
- Stable Hermes workflows should be ported into `ai-skills/` so AlphaRavis can
  use them without calling Hermes every time.

## Toolsets Pattern

Hermes groups tools into toolsets:

- `web`
- `terminal`
- `file`
- `browser`
- `skills`
- `todo`
- `memory`
- `session_search`
- composites such as `debugging` and `safe`

AlphaRavis can borrow this by giving each specialist a narrow tool set instead
of giving every tool to every agent.

## MCP Pattern

Hermes supports MCP servers in config:

```yaml
mcp_servers:
  time:
    command: uvx
    args: ["mcp-server-time"]
  notion:
    url: https://mcp.notion.com/mcp
```

AlphaRavis already uses a DeepAgents-style `mcp.json` loader. If Hermes should
call AlphaRavis, expose AlphaRavis through a guarded OpenAPI/MCP bridge and
describe it as explicit-use-only.

## Safe Hermes Delegation Pattern

When AlphaRavis calls Hermes:

1. Give Hermes one bounded task.
2. Include only relevant context.
3. Tell Hermes not to call LangGraph/AlphaRavis back.
4. Set timeout and output limit.
5. Ask for structured result:
   - summary
   - actions/recommendations
   - files/commands
   - risks
   - next step

When Hermes calls AlphaRavis:

1. Require explicit user request.
2. Require `explicit_user_request=true` in the tool payload.
3. Use a separate thread key.
4. Tell AlphaRavis not to call Hermes back.
5. Return a compact structured result.

## Improvements To Consider

- Port Hermes-style memory nudges into AlphaRavis agent memory.
- Add a Hermes-skill importer that converts selected Hermes skills to
  `ai-skills/<name>/SKILL.md`.
- Add per-agent toolset presets in AlphaRavis, modeled after Hermes toolsets.
- Add a session-search specialist modeled after Hermes session search.
- Add skill-use analytics, similar to Hermes insights/top skills.
