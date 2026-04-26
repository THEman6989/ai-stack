# DeepAgents Patterns For AlphaRavis

This reference is distilled from the local DeepAgents repo at `C:/experi/ai/deepagents`.

## What DeepAgents Adds

DeepAgents is a LangGraph/LangChain agent harness. `create_deep_agent` returns a compiled LangGraph graph built from:

- a base system prompt for autonomous task work,
- `TodoListMiddleware` for planning,
- `FilesystemMiddleware` for `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, and optional `execute`,
- `SubAgentMiddleware` for the `task` tool,
- summarization middleware for context compaction and backend offload,
- optional `SkillsMiddleware` for on-demand skills,
- optional `MemoryMiddleware` for always-loaded `AGENTS.md` memory,
- optional human approval middleware,
- optional filesystem permissions,
- provider profiles and prompt caching.

The central design idea is not "more agents everywhere". It is: give the main agent a small, reliable harness for planning, file-backed context, delegation, compression, and safety.

## `create_deep_agent` Anatomy

Important parameters:

- `model`: model string or chat model instance.
- `tools`: domain-specific tools added to the built-in DeepAgents tools.
- `system_prompt`: domain role instructions prepended before the base DeepAgents prompt.
- `subagents`: sync, compiled, or async subagent specs.
- `skills`: skill source directories. The model sees metadata first and reads `SKILL.md` on demand.
- `memory`: `AGENTS.md` paths. These are always loaded into the system prompt.
- `backend`: file and execution backend.
- `interrupt_on`: LangGraph human-in-the-loop interrupts for named tools.
- `response_format`: structured output schema for specialists.
- `checkpointer`, `store`, `cache`: passed through to LangGraph/LangChain.

DeepAgents automatically adds a `general-purpose` subagent if no subagent with that name is supplied.

## Middleware Order

Main agent base stack:

1. Todo list middleware.
2. Skills middleware, if `skills` is set.
3. Filesystem middleware.
4. Subagent middleware.
5. Summarization middleware.
6. Tool-call patching middleware.
7. Async subagent middleware, if async subagents exist.
8. User middleware.
9. Provider profile middleware and prompt cache.
10. Memory middleware, if `memory` is set.
11. Human-in-the-loop middleware, if `interrupt_on` is set.
12. Permission middleware last.

Subagents receive their own middleware stack and do not automatically leak parent private state such as `skills_metadata` or `memory_contents`.

## Subagent Patterns

### Sync Task Subagents

Use `SubAgent` specs when the parent should block until the specialist returns one final result.

Good for:

- independent codebase exploration,
- research slices,
- data analysis,
- review passes,
- heavy context work that would bloat the parent thread.

Do not use for trivial one-tool actions or when the parent must inspect every intermediate step.

### Compiled Subagents

Use `CompiledSubAgent` when the specialist is a custom LangGraph graph. Its state must include `messages`.

Good for:

- debugger subgraphs with private logs,
- workflows with custom reducers,
- specialists with explicit nodes rather than one generic agent loop.

### Async Subagents

Use async subagents for remote Agent Protocol/LangGraph servers. The main agent receives a task id and can later check, update, cancel, or list tasks.

Good for:

- long research jobs,
- long Pixelle/image pipelines,
- remote heavy compute,
- tasks the user can continue around.

Rule: after launching async work, report the task id and stop. Do not poll in a loop.

## Skills

DeepAgents skills use progressive disclosure:

1. The agent sees name, description, path, and optional allowed tools.
2. The agent reads the full `SKILL.md` only when the task matches.
3. Supporting files are loaded only when needed.

Skill sources are searched in order. Later sources override earlier skills with the same name.

Use skills for reusable procedures. Use memory for stable preferences and identity. Use tools for deterministic actions.

## Memory

DeepAgents `MemoryMiddleware` loads `AGENTS.md` files into the system prompt on every run. It also instructs the agent to update memory immediately when the user teaches a lasting rule or preference.

Use memory for:

- stable user preferences,
- project conventions,
- agent identity,
- durable tool-use guidance.

Do not store:

- passwords,
- API keys,
- one-off task details,
- transient conversation state.

For AlphaRavis, keep large architecture docs on demand. Only small identity and durable behavior rules should become always-loaded memory.

## Backends

`StateBackend` stores files in LangGraph state. It is good for ephemeral thread files.

`FilesystemBackend` reads and writes real files. It is useful for local development and persistent workspace files, but it can expose secrets if pointed too broadly.

`LocalShellBackend` adds unrestricted shell execution. It is powerful and risky. If used, require human approval or isolate it.

`CompositeBackend` routes path prefixes to different backends. This is useful for separating:

- `/workspace/` project files,
- `/memories/` durable memory,
- `/large_tool_results/` offloaded tool output,
- `/conversation_history/` summarized archives.

## Summarization

DeepAgents summarization:

- compacts older messages when a token or context fraction trigger is reached,
- keeps a recent message window,
- writes full offloaded history to `/conversation_history/<thread_id>.md`,
- replaces older active context with a summary message,
- can expose `compact_conversation` for manual compaction.

AlphaRavis already has its own two-tier compression. The useful DeepAgents lesson is the file-backed archive pointer pattern: summary in active context, raw details recoverable by path or retrieval.

## AlphaRavis Improvement Checklist

### P0: Close the raw `execute` bypass

AlphaRavis currently creates the General Assistant with a shell-capable backend. DeepAgents can expose its built-in `execute` tool whenever the backend supports execution. That can bypass the custom AlphaRavis `execute_local_command` approval classifier if not guarded.

Safer options:

- use `StateBackend` for normal agents and keep shell through explicit approved tools only,
- or pass `interrupt_on={"execute": True, "write_file": True, "edit_file": True}` with a checkpointer,
- or create a custom backend/tool wrapper that routes shell execution through the existing approval classifier.

### P1: Register repo skills as static approved skills

The new `ai-skills/` folder should become a static approved skill source. AlphaRavis can then distinguish:

- repo skills: reviewed and version-controlled,
- Mongo skill candidates: inactive until promoted,
- agent memories: scoped lessons, not workflow definitions.

Current DeepAgents-derived repo skills:

- `deepagents-agent-builder`: build new AlphaRavis agents with DeepAgents primitives.
- `deep-research-report`: plan, research, synthesize, cite, and verify deep research.
- `market-research`: define markets, estimate TAM/SAM/SOM, map trends and GTM risks.
- `competitor-analysis`: compare competitors, substitutes, positioning, pricing, and opportunities.

These came from the local DeepAgents examples:

- `examples/deep_research`: research orchestration, subagent delegation, search budgets, citation consolidation.
- `examples/deploy-gtm-agent`: market researcher as a sync subagent, content work as async follow-up, GTM synthesis.
- `examples/deploy-gtm-agent/skills/competitor-analysis`: compact competitive analysis workflow.

### P1: Add an on-demand skill reader

Expose a narrow tool such as `read_repo_skill(skill_name)` that can read only:

```text
ai-skills/<skill-name>/SKILL.md
ai-skills/<skill-name>/references/*.md
```

This gives AlphaRavis access to skill cards without injecting them into every chat.

### P1: Use response schemas for specialist reports

Debugger, research, and context agents should optionally return structured data:

- summary,
- evidence,
- commands_run,
- risks,
- next_actions,
- referenced_files.

This makes bridge/activity UI and DeepAgents UI easier to render.

### P2: Split heavy jobs into async subagents

Use the async subagent pattern for:

- long research,
- Pixelle generation and monitoring,
- repository-wide audits,
- model/backend health investigations.

This keeps LibreChat responsive and lets the user ask for status later.

### P2: Make context compression manually callable

Add a visible tool or command equivalent to `compact_conversation` so the user can say:

```text
komprimiere jetzt
```

and get a visible result with archive pointers.

### P2: Use CompositeBackend path routing

Route different file classes instead of letting every tool see the same filesystem:

- state backend for scratch,
- store backend for durable memories,
- workspace backend for repo-visible files,
- separate artifact area for large tool results.

### P3: Promote skills through review, not automatic learning

Keep the current candidate/promotion model. A workflow may be recorded as a candidate, but active use should require explicit review.
