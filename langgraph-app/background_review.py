"""LLM-powered background review / curation for AlphaRavis.

Feature-flagged: ALPHARAVIS_ENABLE_BACKGROUND_REVIEW (default OFF).

After the agent completes a turn, this module runs a lightweight curation pass
using the LLM to extract durable memories and skill candidates from the
conversation — the LLM IS the curator. No regex, no mechanical extraction.

Architecture:
  - review_conversation() — takes messages, returns curated items
  - CURATION_PROMPT — structured system prompt for the curation LLM
  - _parse_curation_response() — extracts JSON from LLM output
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any


# ── Curation prompt ─────────────────────────────────────────────────────────

CURATION_PROMPT = (
    "You are AlphaRavis' background reviewer. Your job: analyze the conversation "
    "above and extract durable knowledge. You are a CURATOR — do not dump raw "
    "conversation text. Process, structure, and distill.\n\n"

    "Return a single JSON object with these keys:\n"
    "  • \"memories\": list of {memory, memory_type, evidence} — compact, declarative "
    "facts. memory_type is one of: fact, preference, convention, tool_quirk, "
    "environment. Evidence is a short quote or reference from the conversation.\n"
    "  • \"skills\": list of {name, trigger, steps, success_signals, safety_notes, "
    "evidence} — reusable workflows discovered in this turn. Name MUST be at the "
    "class level (e.g. 'debugging-docker-containers' not 'fix-port-8080-today').\n"
    "  • \"nothing_to_save\": boolean — set true only if genuinely nothing is worth "
    "keeping.\n\n"

    "WHAT TO CAPTURE as memories:\n"
    "  • User preferences: 'prefers concise responses', 'wants German output', "
    "'hates markdown in terminal'. Especially corrections and frustration signals "
    "('stop doing X', 'don't format like this', 'I hate when you Y').\n"
    "  • Environment facts: 'Project uses pytest with xdist', 'Docker host is "
    "192.168.1.100', 'ComfyUI runs on port 8188'.\n"
    "  • Tool quirks: 'httpx hangs from Docker containers — use aiohttp instead', "
    "'blockbuster monkey-patches os.getcwd()'.\n"
    "  • Conventions: 'Commit messages use module: description format', 'Feature "
    "flags default OFF'.\n\n"

    "WHAT TO CAPTURE as skills:\n"
    "  • Non-trivial techniques, fixes, workarounds, or debugging paths that a "
    "future session would benefit from.\n"
    "  • Reusable workflows: 'how to set up a new MCP server', 'how to debug "
    "Docker networking issues'.\n"
    "  • A loaded skill that turned out to be wrong or missing steps → capture "
    "the correction.\n\n"

    "WHAT NOT TO CAPTURE:\n"
    "  • Environment-dependent failures: missing binaries, 'command not found', "
    "unconfigured credentials. These are not durable rules.\n"
    "  • Negative claims about tools: 'X tool is broken', 'cannot use Y'. These "
    "harden into self-imposed constraints.\n"
    "  • One-off task narratives: 'analyze this PR', 'summarize today's market'.\n"
    "  • Session-specific transient errors that resolved during the conversation.\n"
    "  • Task progress, PR numbers, commit SHAs, completed-work logs.\n\n"

    "FORMAT RULES:\n"
    "  • Memories: write as declarative facts, not instructions. 'User prefers "
    "concise responses' ✓ — 'Always respond concisely' ✗.\n"
    "  • Skills: name at CLASS level. Trigger is WHEN to use it. Steps are WHAT "
    "to do. Success signals are HOW to know it worked. Safety notes are WHAT to "
    "watch out for.\n"
    "  • Evidence: short direct quote or reference. Not the whole conversation.\n\n"

    "Be ACTIVE. Most turns produce at least one memory. A pass that saves nothing "
    "is usually a missed opportunity. But 'nothing_to_save: true' IS valid when "
    "the turn was pure chitchat or a simple lookup.\n\n"

    "Return ONLY the JSON object, no markdown, no explanation."
)


# ── Public API ──────────────────────────────────────────────────────────────

def build_curation_messages(
    messages: list[dict[str, Any]],
    *,
    max_messages: int = 20,
) -> list[dict[str, Any]]:
    """Build the message list for the curation LLM call.

    Takes the last N user + assistant messages (drops system messages and
    tool outputs to save tokens) and prepends the curation prompt.
    """
    # Filter to user + assistant only, keep last N
    filtered = [
        m for m in messages
        if m.get("role") in ("user", "assistant")
        and m.get("content")
    ]
    recent = filtered[-max_messages:]

    # Build compact representation
    lines = []
    for m in recent:
        role = m["role"]
        content = str(m.get("content", ""))
        # Truncate very long messages
        if len(content) > 2000:
            content = content[:2000].rstrip() + "\n[truncated]"
        lines.append(f"[{role}]: {content}")

    conversation = "\n\n".join(lines)

    return [
        {"role": "system", "content": CURATION_PROMPT},
        {"role": "user", "content": f"Review this conversation:\n\n{conversation}"},
    ]


def parse_curation_response(text: str) -> dict[str, Any]:
    """Parse the LLM's JSON response, with fallback for malformed output."""
    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON block from markdown
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Fallback: nothing parsable
    return {"nothing_to_save": True, "memories": [], "skills": []}


async def review_conversation(
    messages: list[dict[str, Any]],
    *,
    llm_call: Any,  # async callable: (messages) -> str
    max_messages: int = 20,
) -> dict[str, Any]:
    """Run the curation LLM pass and return structured results.

    Args:
        messages: Full conversation messages (OpenAI format).
        llm_call: Async function that takes messages list and returns text.
        max_messages: Max user+assistant messages to include.

    Returns:
        Dict with 'memories', 'skills', 'nothing_to_save', and timing info.
    """
    started = time.perf_counter()
    curation_messages = build_curation_messages(messages, max_messages=max_messages)

    try:
        raw = await llm_call(curation_messages)
        result = parse_curation_response(raw)
    except Exception as exc:
        result = {
            "nothing_to_save": True,
            "memories": [],
            "skills": [],
            "error": str(exc)[:200],
        }

    result["_curation_duration_seconds"] = round(time.perf_counter() - started, 2)
    result["_curation_messages"] = len(curation_messages[1]["content"])
    return result
