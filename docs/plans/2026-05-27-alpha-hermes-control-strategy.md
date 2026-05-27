# Alpha-Hermes-Kontrollstrategie + Node-Verdrahtung

> **Status: Stages 1–4 implementiert.** Nur Live-Smoke-Test mit BigBoss np=4
> und automatische np-Anpassung sind noch offen.

**Ziel:** `_parallel_executor_node` so verdrahten, dass:
1. HermesWorker für Write-Tasks genutzt wird (bestehende `call_hermes_agent`)
2. ParallelContextPlanner vor dem Spawn Budgets schätzt und Admission kontrolliert
3. BigBoss-Registry die neuen Worker-Typen kennt
4. Alles backward-compatible und feature-flagged bleibt

**Alpha-Hermes-Kontrollstrategie (Kern-Idee):**
Hermes ist ein Worker, kein Orchestrator. Der Big Boss plant, Hermes führt aus.
Zwischen Plan und Ausführung sitzt der ContextPlanner als Admission-Gate.
Hermes bekommt pro Task: bounded prompt, context budget, erlaubte Tools.
Wenn Hermes mehr Kontext braucht → NEED_MORE_CONTEXT → Supervisor entscheidet.
Wenn Hermes fertig ist → Ergebnis geht zurück in den Swarm, nicht direkt zum User.

---

## Stage 1: Guarded Import erweitern (≈15min)

### Task 1.1: Import-Block in agent_graph.py

```python
_PARALLEL_EXECUTOR_AVAILABLE = False
try:
    from ai_stack.parallel_executor import (
        DirectLLMWorker,
        HermesWorker,
        ParallelExecutor,
        ParallelContextPlanner,
        TaskDAG,
        analyze_parallelization,
        build_execution_plan,
        log_parallelization_decision,
        parallel_context_planner_enabled,
        parallel_execution_enabled,
        parallel_hermes_worker_enabled,
        parallel_planner_instruction_block,
        parse_planner_text_into_tasks,
    )
    _PARALLEL_EXECUTOR_AVAILABLE = True
except ImportError:
    # ... stubs ...
```

### Task 1.2: Fallback-Stubs

```python
except ImportError:
    def parallel_execution_enabled() -> bool: return False
    def parallel_context_planner_enabled() -> bool: return False
    def parallel_hermes_worker_enabled() -> bool: return False
    def parallel_planner_instruction_block() -> str: return ""
```

---

## Stage 2: _parallel_executor_node neu verdrahten (≈1h)

### Aktueller Flow:
```
state.parallel_dag → tasks → TaskDAG → build_execution_plan → DirectLLMWorker → execute → report
```

### Neuer Flow:
```
state.parallel_dag → tasks → TaskDAG → build_execution_plan
  │
  ├─→ [ContextPlanner enabled?]
  │     ├─ YES: estimate_all() → admit_all() → budgets
  │     └─ NO:  skip (budgets=None)
  │
  ├─→ [HermesWorker enabled?]
  │     ├─ YES: HermesWorker(hermes_fn=call_hermes_agent) für write-Tasks
  │     └─ NO:  DirectLLMWorker für alle
  │
  └─→ execute mit budgets → report → state-Update
```

### Task 2.1: Hermes-Callable bauen

`HermesWorker` braucht `call_hermes_agent` als `hermes_fn`. Die Funktion ist in
agent_graph.py definiert (Zeile 5911). Kein circular import — wir wrappen sie
inline, wie bei `DirectLLMWorker`:

```python
async def _parallel_hermes_fn(task: str, context: str, max_output_chars: int) -> str:
    return await call_hermes_agent(
        task=task,
        context=context,
        max_output_chars=max_output_chars,
    )
```

### Task 2.2: ContextPlanner-Integration

```python
if parallel_context_planner_enabled():
    planner = ParallelContextPlanner(
        pool_total=...,
        parallel_slots=...,
    )
    estimates = await planner.estimate_all(tasks, task_brief=task_brief)
    admission = planner.admit_all(estimates)
    
    if not admission.ok:
        # Fallback: serial mode, or reduce/re-estimate
        ...
    
    # Assign budgets to tasks
    budgets = {
        tid: admission.budget_for(tid)
        for tid in admission.admitted
    }
else:
    budgets = {}
```

### Task 2.3: Worker-Auswahl

```python
for task in tasks:
    if parallel_hermes_worker_enabled() and task.write_enabled:
        worker = HermesWorker()
        worker.set_hermes_fn(_parallel_hermes_fn)
    else:
        worker = DirectLLMWorker()
        worker.set_llm_fn(_parallel_llm_fn)
    
    budget = budgets.get(task.task_id, 0)
    result = await worker.spawn(
        task,
        task_brief=task_brief,
        context_budget=budget,
    )
```

### Task 2.4: GLOBAL_WORKER_REGISTRY befüllen

Am Ende der guarded-import-Sektion, wenn `_PARALLEL_EXECUTOR_AVAILABLE`:

```python
GLOBAL_WORKER_REGISTRY.register("direct_llm", DirectLLMWorker())
GLOBAL_WORKER_REGISTRY.register("hermes", HermesWorker())
```

---

## Stage 3: Alpha-Hermes-Kontrollstrategie (≈30min Design)

### Prinzip: Hermes als Controlled Execution Layer

```
BigBoss (Planner)        ContextPlanner (Gate)       Hermes (Worker)
     │                          │                        │
     │  Task plan + DAG         │                        │
     ├─────────────────────────►│                        │
     │                          │  estimate + admit      │
     │                          ├───────────────────────►│
     │                          │  bounded task + budget │
     │                          │                        │
     │                          │         result         │
     │                          │◄───────────────────────┤
     │       merged results     │                        │
     │◄─────────────────────────┤                        │
     │                          │                        │
     ▼                          ▼                        ▼
  Swarm                    Admission Log             Artifacts
```

### Kontrollpunkte (wo AlphaRavis eingreift):

1. **Plan-Validation:** Planner-DAG wird vor Ausführung geparst und auf unsichere
   Parallel-Gruppen geprüft (file conflicts, chokepoints).

2. **Admission Gate:** ContextPlanner verweigert Worker-Start wenn Pool voll.
   Kein Worker läuft ohne Budget.

3. **Budget Enforcement:** `MAX_CONTEXT_BUDGET` im Worker-Prompt. Worker muss
   `NEED_MORE_CONTEXT` signalisieren wenn Budget zu knapp.

4. **Tool Restriction:** Hermes bekommt `X-AlphaRavis-Disable-LangGraph-Tool: true`
   Header. Keine AlphaRavis/LangGraph-Tools.

5. **Output Truncation:** `max_output_chars` begrenzt Hermes-Output. Truncation
   durch AlphaRavis, nicht Hermes.

6. **Result Merge:** Ergebnisse fließen in `ExecutionReport`, nicht direkt in
   User-Response. Merge/Review-Schritt vor Swarm-Weitergabe.

7. **Error Escalation:** Hermes-Fehler werden als `WorkerResult(status="failed")`
   gemeldet. Kein Hermes-Retry ohne Supervisor-Entscheidung.

---

## Stage 4: BigBoss-Konfiguration dynamisch erkennen (≈30min)

### Task 4.1: Pool-Daten aus ContextScheduler beziehen

Statt hart `pool_total=320000, parallel_slots=4` zu coden, sollte der
`_parallel_executor_node` die Werte aus dem ContextScheduler/BigBoss-Instance
beziehen:

```python
scheduler = get_context_scheduler()
if scheduler:
    await scheduler.refresh_instances_from_manager()
    bigboss = scheduler.instances.get("primary")
    if bigboss:
        pool_total = bigboss.ctx_total
        parallel_slots = bigboss.parallel
        kv_unified = bigboss.kv_unified
```

### Task 4.2: Planner-Prompt für asymmetrische Budget-Hints

Der Planner sollte wissen, dass asymmetrische Verteilung möglich ist:

```
Parallel execution hint: KV-unified pool of {pool_total} tokens with
{parallel_slots} slots. Budgets are asymmetric — a heavy analysis task
may get 120k while a light summary gets only 30k. Mark heavy tasks with
model=big_model and light tasks with model=small_model.
```

---

## Stage 5: Tests (≈30min)

### Task 5.1: Node-Integration-Tests

```python
class TestParallelExecutorNodeIntegration:
    def test_hermes_worker_selected_for_write_tasks(self, ...):
        """When ALPHARAVIS_PARALLEL_HERMES_WORKER=true, write tasks use HermesWorker."""
    
    def test_context_planner_admits_and_assigns_budgets(self, ...):
        """ContextPlanner estimates tokens and assigns budgets to workers."""
    
    def test_node_noop_when_disabled(self, ...):
        """When feature flags OFF, node returns {}."""
    
    def test_fallback_when_pool_full(self, ...):
        """When pool is full, refused tasks are reported, not silently dropped."""
```

---

## Feature Flags (Übersicht)

| Flag | Default | Beschreibung |
|------|---------|-------------|
| `ALPHARAVIS_PARALLEL_TASK_EXECUTION` | false | Master-Switch Parallel Executor |
| `ALPHARAVIS_PARALLEL_HERMES_WORKER` | false | HermesWorker statt DirectLLM für Write-Tasks |
| `ALPHARAVIS_PARALLEL_CONTEXT_PLANNER` | false | Budget-Estimation + Admission Control |
| `ALPHARAVIS_PARALLEL_CONTEXT_SAFETY_RESERVE` | 0.08 | 8% globale Pool-Reserve |
| `ALPHARAVIS_PARALLEL_WORKER_MAX_CONTEXT_RATIO` | 0.85 | Max % des Pools pro Worker |

---

## Was dieser Plan explizit NICHT macht

- **Keine automatische np-Anpassung** — np bleibt statisch, Scheduler managed nur Admission
- **Keine Worker-Preemption** — laufende Worker werden nicht unterbrochen
- **Keine dynamische Pool-Größenänderung** — ctx_total wird vom Server gelesen, nicht geändert
- **Keine Hermes→AlphaRavis Callbacks** — Hermes bleibt one-way Worker
