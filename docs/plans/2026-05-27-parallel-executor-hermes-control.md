# Parallel Executor — Kontrollierte Hermes-Integration + Konservativer Context Scheduler

> **Für Hermes:** Diesen Plan implementieren. Keinen Codex-Adapter. Keinen komplett
> neuen Hermes-Adapter. Bestehende Hermes-Swarm-Integration nutzen.

**Ziel:** Den Parallel Task Executor mit Hermes als kontrolliertem Worker-Layer
ausstatten und einen konservativen Context Scheduler einbauen, der verhindert,
dass parallele Workers das globale Kontextlimit (320k unified bei np=4)
sprengen.

**Architektur-Ansatz:** Hermes wird als Worker-Typ in das bestehende
`WorkerSpawner`-Interface eingehängt. Der Orchestrator (Big Boss / Supervisor)
bleibt in der Kontrolle. Ein neuer `ParallelContextPlanner` schätzt vorab
Token-Budgets via llama.cpp `/tokenize` API und managed Admission.

---

## Architektur-Übersicht

```
Planner (BigBoss, np=4, 320k KV-unified)
  │
  ├─→ <parallel-execution-plan> JSON
  │
  ▼
_parallel_execution_hook() → TaskDAG
  │
  ▼
ParallelContextPlanner (NEU)
  │
  ├─→ Sammelt Worker-Material (RAG, Tools, Files)
  ├─→ Tokenisiert via llama.cpp /tokenize (NICHT in eigenen Kontext)
  ├─→ Berechnet konservatives Budget pro Worker
  ├─→ Prüft globale Admission (active_leases + new_budget ≤ pool * safety_factor)
  │
  ▼
_parallel_executor_node()
  │
  ├─→ HermesWorker (bestehende call_hermes_agent) ──→ Hermes API :8642
  ├─→ DirectLLMWorker (BigBoss direkt)              ──→ BigBoss :8033
  │
  ▼
Merge/Review → Ergebnisse zurück in Swarm
```

---

## Stage 1: HermesWorker (≈2h)

### Task 1.1: HermesWorker implementieren

**Datei:** `ai_stack/parallel_executor/worker_spawner.py`

`HermesWorker` implementiert das `WorkerSpawner`-Interface und wrappt die
bestehende `call_hermes_agent`-Funktion (agent_graph.py:5911):

```python
class HermesWorker(WorkerSpawner):
    """Worker that delegates to the external Hermes Agent API.
    
    Uses the existing call_hermes_agent tool path. Hermes gets:
    - task: the PlannedTask title + description
    - context: task_brief, affected files, dependencies, budget info
    - max_output_chars: derived from context budget
    """
    
    def __init__(self, *, model_override: str = "", api_base: str = ""):
        self.model_override = model_override
        self.api_base = api_base
    
    async def spawn(self, task: PlannedTask, *,
                    worktree: WorktreeInfo | None = None,
                    task_brief: str = "",
                    extra_context: str = "",
                    max_tokens: int = 4096,
                    **kwargs) -> WorkerResult:
        # Baut Prompt aus task.title + task_brief + extra_context
        # Ruft call_hermes_agent(task=..., context=..., max_output_chars=...)
        # Gibt WorkerResult zurück
```

**Hermes bekommt pro Worker:**
- `task`: Task-Titel + spezifische Anweisung
- `context`: Task-Brief, Abhängigkeiten, betroffene Files, Budget-Info, erlaubte Tools
- `max_output_chars`: aus Context-Budget abgeleitet (max 8000)

**Hermes wird NICHT:**
- Eigenständig weitere Tasks spawnen
- Den Orchestrator-Flow kapern
- AlphaRavis/LangGraph-Tools aufrufen (Header `X-AlphaRavis-Disable-LangGraph-Tool: true`)

### Task 1.2: HermesWorker registrieren

In `worker_spawner.py`: `GLOBAL_WORKER_REGISTRY` um `"hermes"` Eintrag erweitern.

### Task 1.3: _parallel_executor_node anpassen

Statt nur `DirectLLMWorker`:
- `hermes_tasks` aus TaskDAG filtern (Tasks mit `worker_type="hermes"`)
- `llm_tasks` (Tasks mit `worker_type="direct_llm"`)
- Beide Worker-Typen parallel spawnen

---

## Stage 2: Konservativer ParallelContextPlanner (≈4h)

### Task 2.1: ContextPreEstimator

**Neue Datei:** `ai_stack/parallel_executor/context_planner.py`

```python
@dataclass
class WorkerContextEstimate:
    task_id: str
    prompt_tokens: int        # Aus /tokenize
    rag_tokens: int           # RAG-Material tokenisiert
    tool_output_tokens: int   # Erwartete Tool-Ausgaben
    file_snippet_tokens: int  # Betroffene Dateien (erste 200 Zeilen)
    total_estimated: int      # Summe
    safety_overhead: int      # 20% Puffer
    recommended_budget: int   # total_estimated + safety_overhead

class ContextPreEstimator:
    """Schätzt Token-Budgets pro Worker VOR dem Start."""
    
    def __init__(self, runtime_client: LlamaCppRuntimeClient):
        self.runtime = runtime_client
    
    async def estimate_worker(self, task: PlannedTask, *,
                              task_brief: str = "",
                              rag_material: list[str] | None = None,
                              affected_files: list[str] | None = None,
                              ) -> WorkerContextEstimate:
        """Tokenisiert alles Material separat via /tokenize."""
        # 1. Prompt tokenisieren
        # 2. RAG-Material tokenisieren  
        # 3. File-Snippets tokenisieren
        # 4. Tool-Output-Puffer schätzen
        # 5. 20% Safety overhead
```

**Wichtig:** Das Material wird NUR tokenisiert, nicht in den eigenen Kontext
geladen. Die `/tokenize` API nimmt den Text und gibt nur Token-Count zurück.

### Task 2.2: Slot-aware Admission Control

`ParallelContextPlanner` managed die Admission:

```python
@dataclass
class SlotBudget:
    """Budget-Tracking für KV-unified bei np=4."""
    pool_total: int           # 320000
    parallel_slots: int       # 4
    kv_unified: bool          # True
    safety_reserve_pct: float # 0.08 (8% Reserve)
    
    active_budgets: dict[str, int]  # task_id → zugewiesenes Budget
    
    @property
    def available(self) -> int:
        used = sum(self.active_budgets.values())
        reserve = int(self.pool_total * self.safety_reserve_pct)
        return max(0, self.pool_total - used - reserve)
    
    def can_admit(self, requested: int) -> bool:
        return self.available >= requested
    
    def admit(self, task_id: str, budget: int) -> bool:
        if not self.can_admit(budget):
            return False
        self.active_budgets[task_id] = budget
        return True
    
    def release(self, task_id: str):
        self.active_budgets.pop(task_id, None)
```

**Regeln:**
1. Kein Worker startet, wenn `available < requested`
2. 8% globale Reserve (`ALPHARAVIS_PARALLEL_CONTEXT_SAFETY_RESERVE`)
3. Asymmetrische Verteilung: Worker A kriegt 120k, Worker B nur 30k
4. Budget wird beim Worker-Start reserviert, beim Completion freigegeben
5. Wenn nicht genug Kontext frei → Worker warten (asyncio.Event) oder Task wird
   kleiner geschnitten

### Task 2.3: Worker-Budget-Enforcement

Jeder Worker bekommt sein Budget als Teil des System-Prompts:

```
MAX_CONTEXT_BUDGET: 120000 tokens
Your total context (prompt + output + tool results) MUST NOT exceed this.
If you approach the limit, summarize intermediate results.
If you need more, signal NEED_MORE_CONTEXT to the supervisor.
```

Der Worker-Prompt enthält:
- Klare Task-Beschreibung
- Erlaubte Tools
- Output-Format
- Maximales Token-/Kontextbudget
- Eskalationspfad (NEED_MORE_CONTEXT)

---

## Stage 3: BigBoss-Konfiguration (≈1h)

### Task 3.1: RuntimeConfig für BigBoss

In `policies.py` / `RuntimeConfig`:

```python
# BigBoss target config
bigboss_config = RuntimeConfig(
    ctx_total=320000,
    parallel=4,          # max np=4
    kv_unified=True,
)

@property
def conservative_ctx_per_slot(self) -> int:
    """Bei KV-unified: nicht einfach ctx/parallel.
    Asymmetrische Verteilung möglich — gibt floor-Wert."""
    if self.kv_unified:
        return max(1, int(self.ctx_total * 0.25))  # 80k floor
    return max(1, self.ctx_total // max(1, self.parallel))
```

### Task 3.2: 2B-Modell bleibt bei 60k

Keine Änderung. `ALPHARAVIS_SMALL_MODEL_CONTEXT=60000` bleibt.

### Task 3.3: Kontext-Test vorbereiten

- `/slots` Polling vor Parallel-Start
- Logging: `parallel_context.total_free`, `parallel_context.per_slot_estimated`
- Warnung wenn `free_context < 20%` vor Parallel-Start

---

## Stage 4: Integration + Tests (≈2h)

### Task 4.1: _parallel_executor_node mit ContextPlanner

```python
async def _parallel_executor_node(state: AlphaRavisState) -> dict[str, Any]:
    # ... bestehende Checks ...
    
    # === NEU: Context Planning ===
    planner = ParallelContextPlanner(
        scheduler=get_context_scheduler(),
        safety_reserve_pct=0.08,
    )
    
    # Pre-estimate all workers
    estimates = await planner.estimate_all(
        tasks, 
        task_brief=task_brief,
        rag_material=state.get("rag_context") or [],
    )
    
    # Admission: assign budgets, refuse if over capacity
    admission = await planner.admit_all(estimates)
    if not admission.ok:
        # Fall back: run serial, or reduce RAG material, or wait
        ...
    
    # Spawn with assigned budgets
    workers = []
    for task in tasks:
        if task.worker_type == "hermes":
            worker = HermesWorker()
        else:
            worker = DirectLLMWorker()
        worker.set_budget(admission.budgets[task.task_id])
        workers.append(worker)
    
    # Execute with budgets
    ...
```

### Task 4.2: Tests

- `TestHermesWorker`: Mock Hermes API, testet Prompt-Bau, Budget-Enforcement
- `TestParallelContextPlanner`: Testet Token-Schätzung, Admission, Reserve
- `TestSlotBudget`: Testet asymmetrische Verteilung, Refusal bei voll
- `TestContextEnforcement`: Worker überschreitet Budget → NEED_MORE_CONTEXT

### Task 4.3: Smoke-Test

```bash
# 1. BigBoss mit np=4, 320k, KV-unified starten
# 2. ALPHARAVIS_PARALLEL_TASK_EXECUTION=true setzen
# 3. Task mit 4 unabhängigen Subtasks stellen
# 4. Prüfen: alle 4 Worker starten, keiner sprengt Kontext
# 5. /slots prüfen: kein Slot bei 100%
```

---

## Was dieser Plan NICHT macht

- **Keinen Codex-Adapter** — explizit ausgeschlossen
- **Keinen komplett neuen Hermes-Adapter** — bestehende `call_hermes_agent` wird gewrappt
- **Keine Alpha-Hermes-Kontrollstrategie** — kommt in separatem Plan
- **Keine Änderung am 2B-Modell** — bleibt bei 60k
- **Kein automatisches np-Umschalten** — np=4 ist statisch, der Scheduler managed nur die Admission

---

## Feature Flags (alle default OFF)

```text
ALPHARAVIS_PARALLEL_TASK_EXECUTION=false          # Master switch
ALPHARAVIS_PARALLEL_HERMES_WORKER=false            # HermesWorker statt DirectLLM
ALPHARAVIS_PARALLEL_CONTEXT_PLANNER=false          # ContextPreEstimator + Admission
ALPHARAVIS_PARALLEL_CONTEXT_SAFETY_RESERVE=0.08    # 8% globale Reserve
ALPHARAVIS_PARALLEL_WORKER_MAX_CONTEXT_RATIO=0.85  # Max % des Pools pro Worker
```

---

## Offene Punkte (für späteren Plan)

- Alpha-Hermes-Kontrollstrategie (Schritt-für-Schritt-Intervention)
- Automatische np-Anpassung je nach Kontext-Last
- Worker-Priorisierung (welcher Worker kriegt wie viel Budget bei Knappheit)
- Preemption (laufenden Worker unterbrechen wenn kritischer Task kommt)
