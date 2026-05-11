# Repository Instructions For Agents

This is the root guide for agents working in this repository. It applies to the
whole `ai-stack` checkout unless a nested `AGENTS.md` gives more specific
instructions, for example inside the `hermes-agent` submodule.

## Orientation

Before changing behavior, read the relevant docs instead of guessing from one
file:

- `README.md`: operator-facing overview, common Makefile commands, service URLs.
- `docs/ALPHARAVIS_OPEN_TASKS.md`: active backlog and current implementation
  state. Start here for "what is still needed".
- `docs/ALPHARAVIS_ARCHITECTURE.md`: system design, container roles, runtime
  profiles, memory, tool routing, bridge/client architecture.
- `docs/ALPHARAVIS_USAGE_NOTES.md`: human-facing runtime behavior and settings.
- `docs/ALPHARAVIS_CHANGES.md`: intentional local runtime changes, patches,
  compatibility decisions, and verification notes.
- `docs/ALPHARAVIS_RESPONSES_COMPATIBILITY.md`: OpenAI-compatible
  `/v1/responses` and streaming compatibility details.
- `docs/ALPHARAVIS_RESPONSES_FULL_STREAMING_PLAN.md`: investigation notes for
  Responses streaming with tool calls.
- `docs/HERMES_INTEGRATION.md`: Hermes service, direct LibreChat usage, and
  AlphaRavis-to-Hermes delegation.
- `docs/AIONUI_LANGGRAPH_ACP_INTEGRATION.md`: ACP adapter behavior and AionUi
  debugging.
- `docs/ALPHARAVIS_MODEL_MANAGEMENT.md`: custom model/power management.

## Documentation Rules

Document non-trivial work as part of the change. Do not leave future agents to
infer runtime behavior only from code.

- Update `docs/ALPHARAVIS_OPEN_TASKS.md` whenever a task moves from planned to
  implemented, partially implemented, blocked, or verified. Keep remaining work
  explicit.
- Update `docs/ALPHARAVIS_CHANGES.md` for behavior changes, local patches,
  dependency/version changes, runtime defaults, compatibility decisions, or
  operationally important fixes. Include concise rationale and verification.
- Update `docs/ALPHARAVIS_ARCHITECTURE.md` when service boundaries, data flow,
  agents, memory, tool routing, streaming architecture, or deployment profiles
  change.
- Update `docs/ALPHARAVIS_USAGE_NOTES.md` when a human operator needs to know a
  new workflow, flag, default, limitation, or UI behavior.
- Update the focused doc for the touched area, for example Responses
  compatibility, Hermes, AionUi ACP, model management, or deep-agent
  improvements.
- Prefer linking to the canonical doc over duplicating long explanations in
  multiple files.

## Runtime And Operations

Use the Makefile as the supported operator interface unless a task requires a
lower-level command:

```bash
make help
make status
make install
make update
make streaming STREAMING=hybrid
make up
make bridge-smoke
make hermes-smoke
```

Primary services:

- `langgraph-api`: LangGraph brain, graph id `alpha_ravis`.
- `api-bridge`: OpenAI-compatible bridge for LibreChat on port `8123`.
- `librechat`: normal user chat UI on port `3080`.
- `litellm`: model gateway on port `4000`.
- `hermes-agent`: optional coding/system specialist on port `8642`.
- `deep-agents-ui`: inspection UI.

The stable default for DeepAgents Responses mode is hybrid streaming:

```text
ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING=true
ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=tool_calling
```

Do not make experimental full tool streaming the default unless the probe,
Bridge smoke tests, and relevant docs all support that change.

## Editing Guidelines

- Keep changes scoped to the requested behavior and the owning module.
- Do not rewrite unrelated generated data, logs, caches, artifacts, or local
  runtime state.
- Treat `.env` as local runtime state. Add or document defaults in
  `.env(exaple)` instead of assuming `.env` changes should be committed.
- Be careful with submodules such as `hermes-agent`,
  `langchain-bridge-repo`, `rag-api-repo`, and `local-deep-researcher-repo`.
  If a submodule has its own instructions, follow them inside that directory.
- Parent-repo Hermes behavior is often implemented through patches in
  `patches/hermes-agent/` and scripts in `scripts/`; check those before editing
  upstream Hermes files directly.
- Preserve dirty user work. Do not revert files you did not intentionally
  change.

## Verification

Run the narrowest useful test first, then broaden when touching shared paths.
Do not consider a code change complete without running at least one relevant
`pytest` target or an equivalent focused test command for the touched area.
For behavior changes, prefer a narrow test first and widen only when shared
paths are involved.

Common checks:

```bash
pytest -q tests
pytest -q tests/test_bridge_responses.py
pytest -q tests/test_responses_streaming_probe.py
python scripts/alpharavis_setup.py status
python scripts/alpharavis_setup.py bridge-smoke
```

For Docker/runtime changes, inspect the relevant service after restart:

```bash
docker compose ps
docker logs --tail 200 api-bridge
docker logs --tail 200 langgraph-api
```

For LibreChat/Responses work, verify both protocol behavior and user-facing UI
expectations when possible:

- `/v1/responses stream=true` emits valid semantic SSE events.
- `/v1/chat/completions stream=true` remains compatible.
- Reasoning/thinking appears in the intended reasoning channel, not as raw
  marker text in the final answer.
- Tool activity remains represented as tool/function-call events where
  documented.

If a live UI check is still needed, leave it explicitly in
`docs/ALPHARAVIS_OPEN_TASKS.md` with the exact thing to verify.

## Commit Hygiene

Do not commit or push unless the user asks. When the user asks for a commit or
for the work to be finished on GitHub, include related code, tests, and
documentation together so the repository state stays explainable, then push the
result to the remote branch.
