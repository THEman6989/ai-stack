---
name: deepagents-agent-builder
description: Design and implement new AlphaRavis or LangGraph agents using DeepAgents principles such as create_deep_agent, task subagents, skills, memory, filesystem backends, summarization, permissions, and human approval. Use when the user asks to add a new agent, build a specialized agent, convert a workflow into an agent, or improve AlphaRavis with DeepAgents-style architecture.
---

# DeepAgents Agent Builder

Use this skill to build agents that follow the DeepAgents harness pattern instead of ad hoc tool wiring.

## Core Workflow

1. Inspect the current AlphaRavis graph before editing:
   - `langgraph-app/agent_graph.py`
   - `docs/ALPHARAVIS_ARCHITECTURE.md`
   - existing tools, memory tools, handoff tools, and safety gates.
2. Decide the agent shape:
   - Use `create_deep_agent` for a normal specialist with tools, todos, filesystem context, skills, and summarization.
   - Use a wrapped `StateGraph` when the specialist needs private state, such as internal logs.
   - Use an async subagent pattern for long-running remote work that should return a task id.
   - Use a plain tool when the behavior is deterministic and does not need an autonomous loop.
3. Define the role contract:
   - Give the agent one job.
   - List when it should transfer to peers.
   - Give it an `agent_id` for scoped memories.
   - Tell it when to use optional tools or skills.
   - Specify output shape, especially if the parent agent must consume the result.
4. Select the DeepAgents primitives:
   - `tools` for domain actions.
   - `subagents` for isolated context work.
   - `skills` for on-demand workflows.
   - `memory` for always-loaded small guidance.
   - `backend` for files, artifacts, and shell behavior.
   - `interrupt_on` or AlphaRavis approval tools for risky actions.
   - `response_format` when structured reports are safer than free text.
5. Integrate through handoffs:
   - Add a `create_handoff_tool` for the new agent.
   - Add relevant incoming handoff tools to existing peers.
   - Add the worker to `create_swarm`.
   - Keep the default active agent broad and conservative.
6. Validate:
   - Run Python import or compile checks.
   - Start `langgraph dev` only when runtime validation is needed.
   - Test simple routing, handoff routing, and failure behavior.

## DeepAgents Rules To Preserve

- Plan before long work with todos.
- Keep heavyweight work in subagents when it can be isolated.
- Keep the parent thread concise by returning synthesized results.
- Use skills through progressive disclosure: metadata first, full `SKILL.md` only when needed.
- Use memory for stable rules and user preferences, not temporary facts.
- Offload large intermediate context to files or archives instead of dumping it into chat.
- Enforce safety at tool/backend level. Do not rely on prompt text alone.

## AlphaRavis Agent Template

Use this shape for a normal new AlphaRavis worker:

```python
transfer_to_new_agent = create_handoff_tool(
    agent_name="new_agent",
    description="Transfer to the new specialist for <specific domain>.",
)

new_worker = create_deep_agent(
    model=llm,
    tools=[
        domain_tool,
        search_agent_memory,
        record_agent_memory,
        read_alpha_ravis_architecture,
        transfer_to_generalist,
        transfer_to_debugger,
        transfer_to_context,
    ],
    name="new_agent",
    system_prompt=(
        "You are the New Agent. Your only job is ... "
        "Use agent_id=`new_agent` for your durable memories. "
        "Transfer back when the task is outside your domain."
    ),
)
```

Then add `new_worker` to `create_swarm([...])` and expose handoffs from peers that should be able to call it.

## Detailed Reference

Read `references/deepagents-patterns.md` when you need exact DeepAgents architecture details, middleware behavior, backend tradeoffs, or the AlphaRavis improvement checklist.
