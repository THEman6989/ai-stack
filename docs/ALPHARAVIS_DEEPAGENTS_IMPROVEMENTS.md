# AlphaRavis DeepAgents Improvement Notes

Source reviewed: `C:/experi/ai/deepagents`.

## What Matters Most

DeepAgents is useful because it standardizes long-running agent work:

- plan with todos,
- isolate heavy work in subagents,
- keep reusable procedures as skills,
- keep durable guidance in memory,
- offload large context to files or archives,
- enforce safety at tools and backends.

AlphaRavis already uses some of this: `create_deep_agent`, swarm agents, a debugger subgraph, compression, skill candidates, and agent-specific memories.

## Best Next Improvements

1. Close the raw DeepAgents `execute` risk.
   The General Assistant uses a shell-capable backend, so DeepAgents can expose its built-in `execute` tool. That should be guarded or removed so shell commands always go through AlphaRavis approval.

2. Make `ai-skills/` a static reviewed skill source.
   Version-controlled skill cards should be treated as approved reference workflows. Mongo skill candidates should stay inactive until promoted.

3. Keep `list_repo_ai_skills` and `read_repo_ai_skill(skill_name)` narrow.
   These tools read only from `ai-skills/<skill-name>/...`, so AlphaRavis can use skill cards on demand without loading every skill into every chat.

4. Add structured reports for specialists.
   Debugger and research agents should return fields like `summary`, `evidence`, `commands_run`, `risks`, and `next_actions`.

5. Use async subagents for long work.
   Pixelle jobs, deep research, and large audits can run as background tasks with task ids instead of blocking LibreChat.

6. Consider a manual `compact_conversation` command.
   AlphaRavis already compresses automatically. A user-visible manual command would make compression feel controlled and inspectable.

## New Skill Card

The reusable DeepAgents builder skill lives at:

```text
ai-skills/deepagents-agent-builder/SKILL.md
```

Detailed reference:

```text
ai-skills/deepagents-agent-builder/references/deepagents-patterns.md
```
