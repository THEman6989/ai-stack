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

## Implemented Improvement Set

1. Close the raw DeepAgents `execute` risk.
   The General Assistant no longer receives a shell-capable DeepAgents backend. Shell and SSH diagnostics now stay on the Debugger Agent tools, where AlphaRavis approval interrupts apply.

2. Make `ai-skills/` a static reviewed skill source.
   Version-controlled skill cards are treated as approved reference workflows. Mongo skill candidates stay inactive until promoted.

3. Keep `list_repo_ai_skills` and `read_repo_ai_skill(skill_name)` narrow.
   These tools read only from `ai-skills/<skill-name>/...`, so AlphaRavis can use skill cards on demand without loading every skill into every chat. The graph also injects only tiny metadata hints for matching repo skill cards.

4. Add structured reports for specialists.
   Debugger, research, and context agents can return fields like `summary`, `evidence`, `commands_run`, `risks`, and `next_actions` through `build_specialist_report`.

5. Use async subagents for long work.
   Pixelle now has an async start/status tool pair, so the agent can return a job id instead of blocking LibreChat.

6. Consider a manual `compact_conversation` command.
   AlphaRavis now supports one-run manual compression phrases such as `komprimiere jetzt` and returns the normal visible Memory-Notice.

7. Adopt the DeepAgents MCP config pattern.
   AlphaRavis now reads `mcp.json` / `.mcp.json` style MCP configs, loads tools
   with `MultiServerMCPClient` only when lazy MCP loading is enabled, prefixes
   tools by server where supported, exposes MCP metadata in
   `describe_optional_tool_registry`, and keeps stdio MCP disabled by default.

## New Skill Card

The reusable DeepAgents-derived skill cards live at:

```text
ai-skills/deepagents-agent-builder/SKILL.md
ai-skills/deep-research-report/SKILL.md
ai-skills/market-research/SKILL.md
ai-skills/competitor-analysis/SKILL.md
```

Detailed reference:

```text
ai-skills/deepagents-agent-builder/references/deepagents-patterns.md
```
