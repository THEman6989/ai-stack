# Delegate Agent — Feature Parity mit Hermes (Gap-Closure Plan)

> **Fuer Hermes/AlphaRavis:** Plan Task fuer Task abarbeiten. Reihenfolge einhalten.

**Goal:** AlphaRavis `delegate_task` auf vollen Hermes-Feature-Level bringen.
Alle 8 Gaps aus dem Side-by-Side-Code-Trace schliessen.

**Architecture:** `delegate_agent.py` wird erweitert um Provider-Override,
Toolset-Intersection, Heartbeat-Keepalive, verbesserte System-Prompts.
`agent_graph.py` bekommt neue ENV-Vars und uebergibt Provider-Config an
`run_sub_agent()`.

**Aktueller Stand:** 862 Zeilen, 22 Tests gruen, lauffaehig aber "Light"-Sub-Agent.

---

## GAP UEBERSICHT

| # | Gap | Groesse | Impact |
|---|-----|---------|--------|
| 1 | Provider/Credential-Override | LARGE | Sub-Agent kann nicht auf anderem Model laufen |
| 2 | Toolset-Intersection mit Parent | MEDIUM | Sub-Agent kriegt alle 22 Tools statt Parent-Menge |
| 3 | Heartbeat/Gateway-Activity-Keepalive | MEDIUM | Gateway killt Parent waehrend Sub-Agent laeuft |
| 4 | Blocked-Tools-Liste | SMALL | Kein Schutz vor gefaehrlichen Tools |
| 5 | Workspace-Hint im System-Prompt | SMALL | Sub-Agent errät /workspace/ statt echten Pfad |
| 6 | Orchestrator-Rollen-Prompt | SMALL | Nur ein Satz, Hermes hat ausfuehrliche Anleitung |
| 7 | Output-Format im System-Prompt | SMALL | Hermes hat spezifischere Vorgaben |
| 8 | Fallback/Retry-Resilience | LARGE | Kein Error-Recovery bei API-Fehlern |

---

### Task 1: ENV-Vars + Config-Struktur

**Objective:** Neue ENV-Vars in `.env(exaple)` + `delegate_agent.py` Konstanten

**Files:**
- Modify: `.env(exaple)`
- Modify: `langgraph-app/delegate_agent.py:255-279`

**Neue ENV-Vars:**
```
# Delegate Provider Override (leer = Parent-Provider)
ALPHARAVIS_DELEGATE_PROVIDER=
ALPHARAVIS_DELEGATE_MODEL=
ALPHARAVIS_DELEGATE_API_BASE=
ALPHARAVIS_DELEGATE_API_KEY=

# Heartbeat (verhindert Gateway-Timeout waehrend Sub-Agent laeuft)
ALPHARAVIS_DELEGATE_HEARTBEAT_ENABLED=true
ALPHARAVIS_DELEGATE_HEARTBEAT_INTERVAL_SECONDS=30

# Toolset Control
ALPHARAVIS_DELEGATE_INTERSECT_PARENT_TOOLS=true
ALPHARAVIS_DELEGATE_BLOCKED_TOOLS=clarify,memory,send_message

# Fallback (retry bei API-Fehlern)
ALPHARAVIS_DELEGATE_MAX_RETRIES=2
ALPHARAVIS_DELEGATE_RETRY_DELAY_SECONDS=5

# Workspace
ALPHARAVIS_DELEGATE_WORKSPACE_HINT=
```

**Step 1:** `.env(exaple)` Eintraege hinzufuegen
**Step 2:** `delegate_agent.py` Konstanten definieren
**Step 3:** `git add .env(exaple) langgraph-app/delegate_agent.py && git commit -m "feat(delegate): ENV vars for provider-override, heartbeat, toolset-control, fallback"`

---

### Task 2: Provider/Credential-Override (GAP 1 — LARGE)

**Objective:** `run_sub_agent()` akzeptiert `_provider`, `_model_name`, `_api_base`,
`_api_key`. Wenn gesetzt, baut Sub-Agent eigenen `ChatOpenAI`-Client statt
Parent-`_model_fn` zu nutzen.

**Files:**
- Modify: `langgraph-app/delegate_agent.py:421-546` (run_sub_agent signature + model construction)
- Modify: `langgraph-app/agent_graph.py:6900-6915` (uebergibt Provider-Params)

**Aenderung in delegate_agent.py:**

```python
async def run_sub_agent(
    *,
    goal: str,
    context: str = "",
    tools: dict[str, Any] | None = None,
    tool_names: list[str] | None = None,
    max_iterations: int = 30,
    timeout_seconds: int = 600,
    max_output_chars: int = 8000,
    depth: int = 0,
    parent_id: str | None = None,
    _model_fn: Any = None,
    _tool_name_fn: Any = None,
    _store: Any = None,
    _thread_id: str = "",
    _thread_key: str = "",
    _router_ingest_source: Any = None,
    # --- NEU: Provider Override ---
    _provider: str = "",
    _model_name: str = "",
    _api_base: str = "",
    _api_key: str = "",
) -> dict[str, Any]:
```

Model-Construction (ersetzt Zeile 528-545):
```python
    # Provider override: build eigene ChatOpenAI-Instanz
    if _provider and _model_name:
        try:
            from langchain_openai import ChatOpenAI
            model_kwargs = {
                "model": _model_name,
                "temperature": float(os.getenv("ALPHARAVIS_DELEGATE_TEMPERATURE", "0.1")),
                "max_tokens": int(os.getenv("ALPHARAVIS_DELEGATE_MAX_TOKENS", "4096")),
            }
            if _api_base:
                model_kwargs["base_url"] = _api_base
            if _api_key:
                model_kwargs["api_key"] = _api_key
            if tool_schemas:
                model_kwargs["tools"] = tool_schemas
            model = ChatOpenAI(**model_kwargs)
        except ImportError:
            model = _model_fn(model_kwargs) if _model_fn else None
    elif _model_fn:
        # Legacy: use parent's model function
        model_kwargs = {...}  # existing code
        model = _model_fn(model_kwargs)
    else:
        model = None
```

**Aenderung in agent_graph.py (delegate_task @tool):**
```python
# Resolve provider override from ENV
_provider = os.getenv("ALPHARAVIS_DELEGATE_PROVIDER", "").strip()
_model_name = os.getenv("ALPHARAVIS_DELEGATE_MODEL", "").strip()
_api_base = os.getenv("ALPHARAVIS_DELEGATE_API_BASE", "").strip()
_api_key = os.getenv("ALPHARAVIS_DELEGATE_API_KEY", "").strip()

return await _run_sub_agent(
    ...
    _provider=_provider,
    _model_name=_model_name,
    _api_base=_api_base,
    _api_key=_api_key,
)
```

**Step 1:** `delegate_agent.py` — `run_sub_agent()` Signatur + Model-Construction um Provider-Logik erweitern
**Step 2:** `agent_graph.py` — ENV-Resolution vor `_run_sub_agent()`-Call
**Step 3:** `git commit -m "feat(delegate): provider/model override for sub-agents"`
**Verify:** Sub-Agent kann auf `ALPHARAVIS_DELEGATE_PROVIDER=openrouter ALPHARAVIS_DELEGATE_MODEL=qwen/qwen-2.5-7b-instruct` laufen

---

### Task 3: Fallback/Retry bei API-Fehlern (GAP 8 — LARGE)

**Objective:** Wenn `model.ainvoke()` fehlschlaeigt, retry mit exponential backoff.
Max `ALPHARAVIS_DELEGATE_MAX_RETRIES` Versuche.

**Files:**
- Modify: `langgraph-app/delegate_agent.py:570-607` (_task_loop)

**Aenderung:**
```python
max_retries = int(os.getenv("ALPHARAVIS_DELEGATE_MAX_RETRIES", "2"))
retry_delay = float(os.getenv("ALPHARAVIS_DELEGATE_RETRY_DELAY_SECONDS", "5"))

for turn in range(max_iterations):
    ...
    for attempt in range(max_retries + 1):
        try:
            response = await model.ainvoke(messages)
            break  # success
        except Exception as exc:
            if attempt < max_retries and _is_retryable_error(exc):
                delay = retry_delay * (2 ** attempt)
                LOGGER.warning("Agent %s: API error (attempt %d/%d), retrying in %.1fs: %s",
                               agent_id, attempt+1, max_retries+1, delay, exc)
                await asyncio.sleep(delay)
            else:
                raise  # out of retries or non-retryable
    ...
```

**Helper `_is_retryable_error()`:**
```python
def _is_retryable_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    retryable = {"rate limit", "timeout", "overloaded", "server error",
                 "503", "502", "429", "connection", "capacity"}
    return any(pattern in msg for pattern in retryable)
```

**Step 1:** `_is_retryable_error()` helper in `delegate_agent.py`
**Step 2:** Retry-Loop in `_task_loop()`
**Step 3:** `git commit -m "feat(delegate): retry with exponential backoff on API errors"`
**Verify:** Simulierter API-Fehler → Sub-Agent retried 2x, dann Abbruch mit Status "failed"

---

### Task 4: Heartbeat/Gateway-Keepalive (GAP 3 — MEDIUM)

**Objective:** Waehrend Sub-Agent laeuft, periodisch Parent-Activity touchten
damit Gateway nicht "inactivity timeout" feuert.

**Files:**
- Modify: `langgraph-app/delegate_agent.py:421-670` (run_sub_agent + _task_loop)

**Aenderung:**
```python
HEARTBEAT_ENABLED = os.getenv("ALPHARAVIS_DELEGATE_HEARTBEAT_ENABLED", "true").strip().lower() in ("1", "true", "yes")
HEARTBEAT_INTERVAL = float(os.getenv("ALPHARAVIS_DELEGATE_HEARTBEAT_INTERVAL_SECONDS", "30"))

async def _heartbeat_loop(agent_id: str, cancel_evt, parent_touch_fn, started: float):
    """Periodisch Parent-Activity touchten solange Sub-Agent laeuft."""
    while not cancel_evt.is_set():
        try:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            if cancel_evt.is_set():
                break
            if parent_touch_fn:
                elapsed = time.perf_counter() - started
                parent_touch_fn(f"delegate_task: sub-agent {agent_id} working ({elapsed:.0f}s)")
        except asyncio.CancelledError:
            break
        except Exception:
            pass

# In run_sub_agent() — nach Registration, vor _task_loop:
heartbeat_task = None
if HEARTBEAT_ENABLED and _parent_touch_fn:
    heartbeat_task = asyncio.create_task(
        _heartbeat_loop(agent_id, cancel_evt, _parent_touch_fn, started)
    )

try:
    result = await asyncio.wait_for(_task_loop(), timeout=timeout)
finally:
    if heartbeat_task:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
```

**Step 1:** `_heartbeat_loop()` async function in `delegate_agent.py`
**Step 2:** Heartbeat-Lifecycle in `run_sub_agent()` integrieren
**Step 3:** `agent_graph.py` — `_parent_touch_fn` an `run_sub_agent()` uebergeben (optional)
**Step 4:** `git commit -m "feat(delegate): heartbeat keepalive verhindert Gateway-Timeout"`
**Verify:** Sub-Agent laeuft 60s → kein Gateway-Timeout

---

### Task 5: Toolset-Intersection + Blocked-Tools (GAP 2 + 4 — MEDIUM+SMALL)

**Objective:** Sub-Agent kriegt nur Tools die der Parent auch hat (Intersection).
Zusaetzlich Blocklist fuer gefaehrliche Tools.

**Files:**
- Modify: `langgraph-app/delegate_agent.py:470-486` (tool selection)
- Modify: `langgraph-app/agent_graph.py:6835-6860` (_delegate_tool_list)

**Aenderung in delegate_agent.py:**
```python
# Blocked tools — never available to sub-agents
DELEGATE_BLOCKED_TOOLS = frozenset(
    os.getenv("ALPHARAVIS_DELEGATE_BLOCKED_TOOLS", "clarify,memory,send_message")
    .replace(" ", "").split(",")
) if os.getenv("ALPHARAVIS_DELEGATE_BLOCKED_TOOLS", "") else frozenset()

INTERSECT_PARENT_TOOLS = os.getenv("ALPHARAVIS_DELEGATE_INTERSECT_PARENT_TOOLS", "true").strip().lower() in ("1", "true", "yes")

# In run_sub_agent() — tool selection:
selected_tools: dict[str, Any] = {}
if tools:
    for name, tool_obj in tools.items():
        # Blocklist check
        if name in DELEGATE_BLOCKED_TOOLS:
            continue
        # Toolset filter (from parent's tool_names)
        if clean_tool_names and name not in clean_tool_names:
            continue
        selected_tools[name] = tool_obj
```

**Aenderung in agent_graph.py:**
```python
# Keine Aenderung noetig — _delegate_tool_list ist schon sauber.
# delegate_task selbst ist nicht in der Liste → Sub-Agent kann nicht rekursiv spawnen.
# Blocks passieren in delegate_agent.py.
```

**Step 1:** `DELEGATE_BLOCKED_TOOLS` + `INTERSECT_PARENT_TOOLS` ENV-Resolution
**Step 2:** Blocklist-Check in Tool-Selection
**Step 3:** `git commit -m "feat(delegate): toolset intersection + blocked-tools blocklist"`
**Verify:** Sub-Agent mit `tool_names=["execute_local_command", "clarify"]` → `clarify` wird geblockt

---

### Task 6: Workspace-Hint im System-Prompt (GAP 5 — SMALL)

**Objective:** Sub-Agent kriegt echten Workspace-Pfad im System-Prompt,
nicht den "/workspace/"-Guess.

**Files:**
- Modify: `langgraph-app/delegate_agent.py:488-518` (system_prompt)
- Modify: `langgraph-app/agent_graph.py:6900-6915` (uebergibt workspace)

**Aenderung in delegate_agent.py:**
```python
# Resolve workspace hint
workspace_hint = os.getenv("ALPHARAVIS_DELEGATE_WORKSPACE_HINT", "").strip()
if not workspace_hint:
    # Fallback: discover from parent
    workspace_hint = os.getenv("TERMINAL_CWD", os.getcwd())

workspace_note = ""
if workspace_hint and os.path.isdir(workspace_hint):
    workspace_note = (
        f"\nWorkspace: {workspace_hint}\n"
        "IMPORTANT: Use this exact path for file operations. "
        "Never assume /workspace/... or any container-style path.\n"
    )

system_prompt = (
    f"You are AlphaRavis sub-agent '{agent_id}' (depth {depth}/{max_depth}). "
    ...
    f"{workspace_note}"
    ...
)
```

**Step 1:** Workspace-Resolution + NOTE im System-Prompt
**Step 2:** `agent_graph.py` uebergibt CWD an `run_sub_agent()`
**Step 3:** `git commit -m "feat(delegate): workspace hint prevents /workspace/ assumptions"`

---

### Task 7: Orchestrator-Rolle + Output-Format (GAP 6 + 7 — SMALL)

**Objective:** Verbesserte System-Prompts: Orchestrator-Rolle mit Delegations-Anleitung,
Output-Format mit spezifischen Sektionen wie Hermes.

**Files:**
- Modify: `langgraph-app/delegate_agent.py:488-518`

**Aenderung:**
```python
# Orchestrator guidance (wenn depth < max_depth)
orchestrator_guidance = ""
if depth < max_depth:
    orchestrator_guidance = (
        "\n## Subagent Spawning (Orchestrator)\n"
        "You have access to spawn sub-agents via delegate_task.\n\n"
        "WHEN to delegate:\n"
        "- The goal splits into 2+ independent subtasks (parallel).\n"
        "- A subtask is reasoning-heavy and would flood your context.\n\n"
        "WHEN NOT to delegate:\n"
        "- Single-step work — do it directly.\n"
        "- Trivial tasks you can finish in 1-2 tool calls.\n"
        "- Pass-through: don't re-delegate your entire goal to one worker.\n\n"
        "Coordinate results and synthesize before reporting to parent.\n"
    )

# Output format (wie Hermes)
output_format = (
    "## Output Format\n"
    "When done, return a final answer with:\n"
    "  ## Summary — what you accomplished\n"
    "  ## Key Findings — discoveries, answers\n"
    "  ## Actions — commands/files used\n"
    "  ## Issues — problems encountered (if any)\n"
    "  ## Recommendation — next step for the parent agent"
)
```

**Step 1:** Orchestrator-Guidance + Output-Format in System-Prompt
**Step 2:** `git commit -m "feat(delegate): orchestrator role guidance + structured output format"`

---

### Task 8: Tests + Verifikation

**Objective:** Neue Tests fuer alle Gaps. Bestehende 22 Tests muessen gruen bleiben.

**Files:**
- Modify: `tests/test_delegate_agent.py` (neue Tests)
- Create: `tests/test_delegate_gaps.py` (Gap-spezifische Tests)

**Neue Tests:**
```python
# Gap 1: Provider Override
def test_run_sub_agent_accepts_provider_params()
def test_provider_override_builds_chatopenai()

# Gap 3: Heartbeat
def test_heartbeat_touches_parent_activity()
def test_heartbeat_stops_on_cancel()

# Gap 2+4: Toolset
def test_blocked_tools_are_filtered()
def test_toolset_intersection_removes_extra_tools()

# Gap 8: Retry
def test_retry_on_rate_limit_error()
def test_non_retryable_error_fails_immediately()
def test_exponential_backoff_delay()

# Gap 5: Workspace
def test_workspace_hint_in_system_prompt()

# Gap 6+7: Prompt
def test_orchestrator_guidance_present_when_depth_lt_max()
def test_output_format_sections_in_system_prompt()
```

**Step 1:** Neue Tests schreiben (TDD — erst RED)
**Step 2:** `pytest tests/ -q -k 'delegate'` → alle gruen
**Step 3:** `git commit -m "test(delegate): gap-closure verification tests"`

---

### Task 9: Docs + Final Commit

**Objective:** CHANGES.md, ARCHITECTURE.md, Skill-Referenz aktualisieren.

**Files:**
- Modify: `docs/ALPHARAVIS_CHANGES.md`
- Modify: `docs/ALPHARAVIS_ARCHITECTURE.md`
- Modify: Skill `alpharavis-development` → `references/delegate-agent-architecture.md`

**Step 1:** `docs/ALPHARAVIS_CHANGES.md` — neuer Eintrag "2026-06-01 — Delegate Agent Gap-Closure"
**Step 2:** Skill-Referenz aktualisieren
**Step 3:** `git commit -m "docs: delegate gap-closure documentation"`
**Step 4:** Finaler Test-Run: `pytest tests/ -q` → alle gruen

---

## AUSFUEHRUNGSREIHENFOLGE

1. Task 1 — ENV-Vars (Foundation fuer alle anderen)
2. Task 2 — Provider Override (groesster Gap)
3. Task 3 — Retry/Fallback (Resilience)
4. Task 4 — Heartbeat (Gateway-Stabilitaet)
5. Task 5 — Toolset-Control (Safety)
6. Task 6 — Workspace-Hint (UX)
7. Task 7 — Prompt-Verbesserung (UX)
8. Task 8 — Tests (alles auf einmal am Ende, TDD-pro-Task)
9. Task 9 — Docs

**Geschaetzte Zeit: 45-60 Minuten fuer alle 9 Tasks.**
