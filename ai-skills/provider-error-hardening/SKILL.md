---
name: provider-error-hardening
description: Diagnose and harden AlphaRavis or Hermes-style provider failures, including unsupported parameters, token-limit spelling mismatches, Responses-vs-Chat fallback, auth/rate-limit/server classifications, and operational smoke-test evidence.
requires_tool_categories: [coding/read, coding/write]
---

# Provider Error Hardening

Use this skill when a model backend, LiteLLM route, Hermes gateway, or
OpenAI-compatible provider fails with provider errors, unsupported parameters,
bad token-limit names, missing auth, server errors, rate limits, or broken
Responses tool behavior.

## Workflow

1. Capture the exact failing path:
   - external client path, such as LibreChat, Bridge Test UI, or Hermes
   - endpoint path, such as `/v1/chat/completions` or `/v1/responses`
   - provider/model/base URL family
   - status code and response body
2. Classify before changing behavior:
   - auth or missing key
   - rate limit or overload
   - backend 5xx
   - context overflow or payload too large
   - unsupported parameter or schema mismatch
   - Responses tool-streaming conversion failure
3. Prefer a narrow compatibility retry:
   - remove harmless rejected parameters
   - map token-limit spelling only for providers known to need it
   - keep local LiteLLM/llama.cpp defaults unchanged unless evidence says otherwise
   - preserve the original error if the retry also fails
4. Keep fallback explicit:
   - direct Responses may fall back to ChatLiteLLM when not required
   - DeepAgents Responses should stay hybrid by default until repeated smoke tests pass
   - do not silently switch to hosted providers
5. Add operational evidence:
   - focused unit test for the classifier or retry payload
   - smoke result for the affected service when reachable
   - documentation entry in `docs/ALPHARAVIS_CHANGES.md`
   - remaining work in `docs/ALPHARAVIS_OPEN_TASKS.md`

## Hermes Reference Points

- `hermes-agent/agent/auxiliary_client.py`
- `hermes-agent/run_agent.py`
- `hermes-agent/agent/error_classifier.py`
- `hermes-agent/agent/rate_limit_tracker.py`
- `hermes-agent/agent/usage_pricing.py`

Borrow the pattern, not the whole adapter stack. AlphaRavis should keep LiteLLM
and LangChain as the main route unless a provider feature cannot be represented
there.

## Success Signals

- The user-facing error is classified and actionable.
- Retry behavior is visible in logs or `run_profile`.
- A failed retry includes both the original and retry failure.
- Runtime docs say which flags control the behavior.
- Tests cover the exact compatibility behavior added.
