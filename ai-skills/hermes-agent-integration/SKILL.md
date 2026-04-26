---
name: hermes-agent-integration
description: Integrate Hermes Agent with AlphaRavis, LangGraph, LibreChat, MCP/OpenAPI tool flows, coding-agent delegation, Hermes skills, Hermes memory, toolsets, gateway API, and anti-recursion safety. Use when adding Hermes as a direct LibreChat endpoint, a LangGraph sub-agent, or a source of reusable coding/system workflow skills.
---

# Hermes Agent Integration

Use this skill when AlphaRavis should work with Hermes Agent or borrow Hermes
patterns for coding agents, memories, skills, or tool routing.

## Core Workflow

1. Inspect the current AlphaRavis integration points:
   - `librechat.yaml`
   - `docker-compose.yml`
   - `langgraph-app/agent_graph.py`
   - `langgraph-app/bridge_server.py`
   - `docs/ALPHARAVIS_ARCHITECTURE.md`
2. Keep the roles separate:
   - LibreChat is the UI.
   - AlphaRavis/LangGraph is the supervisor and custom-agent orchestrator.
   - Hermes is a bounded coding/system specialist.
3. For LibreChat direct mode, configure Hermes as an OpenAI-compatible custom endpoint.
4. For LangGraph mode, call Hermes through a bounded tool or specialist worker.
5. For Hermes-to-LangGraph mode, require explicit user intent and a hard timeout.
6. Prevent loops:
   - LangGraph may call Hermes for one bounded subtask.
   - Hermes may call LangGraph only when the user explicitly asks.
   - Any reverse call must say not to call Hermes back.

## AlphaRavis Rules

- Keep `ALPHARAVIS_ENABLE_HERMES_AGENT=false` until Hermes gateway is running.
- Use `HERMES_API_BASE=http://host.docker.internal:8642/v1` from Docker.
- Keep Hermes output bounded with `HERMES_TIMEOUT_SECONDS` and `HERMES_MAX_OUTPUT_CHARS`.
- Do not let Hermes bypass AlphaRavis command approval gates.
- Treat Hermes skills as reusable workflow patterns. Port stable workflows into `ai-skills/` when they reduce future Hermes calls.
- Put project-independent lessons into global memory; put Hermes-specific lessons into `agent_id=hermes_coding_agent`.

## When To Read The Reference

Read `references/hermes-patterns.md` when you need details about:

- Hermes gateway OpenAI API settings,
- Hermes memory design,
- Hermes skills/routines,
- toolsets and MCP,
- what AlphaRavis can borrow from Hermes next.
