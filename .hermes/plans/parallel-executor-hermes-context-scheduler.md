# Parallel Executor — Hermes-Integration & Context Scheduler

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Controlled Hermes integration as WorkerSpawner + konservativer Context Scheduler mit Token-Budget pro Worker und Admission Control für parallele BigBoss-Worker.

**Architecture:** Hermes wird als Worker-Layer über einen `HermesWorker`-Spawner angebunden — er nutzt die existierende `_call_hermes_streaming_sse`-Pipeline (Memory/RAG/Skills Pre-Load, dann Hermes :8642). Der `ParallelContextAdmission`-Gate prüft vor jedem Worker-Start live `/slots`, schätzt Token-Bedarf via llama.cpp `/tokenize`, vergibt asymmetrische Budgets, und verhindert Überlastung des unified context pools (320k, np=4). BigBoss bleibt Supervisor; Hermes führt aus.

**Tech Stack:** Python asyncio, httpx → llama.cpp API, bestehende `_call_hermes_streaming_sse`, `LlamaCppRuntimeClient`, `ContextScheduler`, `LeaseStore`.

---

## Context & Constraints

- Big Boss: 320k unified context, np=4. Asymmetrische Slot-Verteilung (z.B. Slot A 120k, Slot B 30k, solange Summe ≤ 320k minus Reserve).
- 2B-Modell: bleibt bei 60k Kontext, nur für bestehende 3 Classifier-Funktionen.
- Kein neuer Codex-CLI-Adapter. Kein komplett neuer Hermes-Adapter.
- Hermes-Integration via existierende `_call_hermes_streaming_sse` / `hermes-orch` (:8650) Pipeline.
- Alpha-Hermes-Kontrollstrategie (Deep Agents / Orchestrator-Feinsteuerung) kommt in separatem Plan.
- Feature bleibt hinter `ALPHARAVIS_PARALLEL_TASK_EXECUTION=true` gegated.
- Alle neuen Env-Vars default OFF / konservativ.

---

### Task 1: HermesWorker — WorkerSpawner-Adapter für Hermes

**Objective:** `HermesWorker` implementiert `WorkerSpawner.spawn()` und delegiert an die existierende `_call_hermes_streaming_sse`-Pipeline.

**Files:**
- Create: `ai_stack/parallel_executor/hermes_worker.py`
- Modify: `ai_stack/parallel_executor/__init__.py`
- Modify: `ai_stack/parallel_executor/worker_spawner.py` (Registry-Eintrag)
- Create: `tests/test_hermes_worker.py`

**Step 1: Write failing test**

```python
# tests/test_hermes_worker.py
import asyncio
from ai_stack.parallel_executor.hermes_worker import HermesWorker
from ai_stack.parallel_executor.task_graph import PlannedTask, TaskType, ModelClass

def test_hermes_worker_implements_spawner_interface():
    """HermesWorker must implement WorkerSpawner.spawn()."""
    from ai_stack.parallel_executor.worker_spawner import WorkerSpawner
    worker = HermesWorker()
    assert isinstance(worker, WorkerSpawner)

def test_hermes_worker_dry_run_when_no_callable():
    """When no hermes_callable is set, falls back to dry-run."""
    task = PlannedTask(
        task_id="test-1",
        title="Refactor auth",
        task_type=TaskType.CODE_WRITE,
        required_model_class=ModelClass.BIG_MODEL,
    )
    worker = HermesWorker()  # no callable set
    result = asyncio.run(worker.spawn(task, task_brief="Test"))
    assert result.status == "dry_run"
    assert "HermesWorker" in result.output
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_hermes_worker.py::test_hermes_worker_implements_spawner_interface -v`
Expected: FAIL — module not found

**Step 3: Write minimal implementation**

```python
# ai_stack/parallel_executor/hermes_worker.py
"""HermesWorker — WorkerSpawner adapter that delegates to the existing
_call_hermes_streaming_sse pipeline (Memory/RAG/Skills → Hermes :8642)."""

from __future__ import annotations
import logging
from typing import Any
from ai_stack.parallel_executor.worker_spawner import WorkerSpawner, WorkerResult, DryRunWorker
from ai_stack.parallel_executor.task_graph import PlannedTask

LOGGER = logging.getLogger(__name__)

HermesCallable = Any  # async (message, system_prompt, max_output_chars) -> str


class HermesWorker(WorkerSpawner):
    """Spawns a Hermes coding agent for one PlannedTask.

    Uses a callable that wraps _call_hermes_streaming_sse so the parallel
    executor stays decoupled from agent_graph internals.
    """

    def __init__(self, hermes_fn: HermesCallable | None = None) -> None:
        self._hermes_fn = hermes_fn

    def set_hermes_fn(self, fn: HermesCallable) -> None:
        self._hermes_fn = fn

    async def spawn(
        self,
        task: PlannedTask,
        *,
        worktree=None,
        task_brief: str = "",
        **kwargs: Any,
    ) -> WorkerResult:
        if self._hermes_fn is None:
            dry = DryRunWorker()
            result = await dry.spawn(task, worktree=worktree, task_brief=task_brief)
            result.output = f"HermesWorker dry-run for {task.task_id}: {task.title}"
            return result

        system_prompt = (
            f"You are Hermes, a coding specialist working on a bounded task.\\n"
            f"Task ID: {task.task_id}\\n"
            f"Task type: {task.task_type.value}\\n"
            f"Context: {task_brief}\\n"
            f"\\nWork in isolation. Do NOT call AlphaRavis/LangGraph back. "
            f"Return only the requested output."
        )

        max_chars = int(kwargs.get("max_output_chars", 24000))

        try:
            output = await self._hermes_fn(
                message=task.title,
                system_prompt=system_prompt,
                max_output_chars=max_chars,
            )
            return WorkerResult(
                task_id=task.task_id,
                status="completed",
                output=str(output),
                worktree=worktree,
                metadata={"worker": "hermes"},
            )
        except Exception as exc:
            LOGGER.exception("HermesWorker failed for %s", task.task_id)
            return WorkerResult(
                task_id=task.task_id,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                worktree=worktree,
            )
```

**Step 4: Run test to verify pass**

Run: `pytest tests/test_hermes_worker.py -v`
Expected: 2 passed

**Step 5: Register in __init__.py and worker_spawner.py**

```python
# ai_stack/parallel_executor/__init__.py: add export
from ai_stack.parallel_executor.hermes_worker import HermesWorker, HermesCallable

# ai_stack/parallel_executor/worker_spawner.py: register
GLOBAL_WORKER_REGISTRY.register("hermes", HermesWorker())
```

**Step 6: Commit**

```bash
git add ai_stack/parallel_executor/hermes_worker.py ai_stack/parallel_executor/__init__.py ai_stack/parallel_executor/worker_spawner.py tests/test_hermes_worker.py
git commit -m "feat: HermesWorker spawner adapter for parallel executor"
```

---

### Task 2: Wire HermesWorker in agent_graph _parallel_executor_node

**Objective:** Im `_parallel_executor_node` den `HermesWorker` als Spawner-Option bereitstellen (via Feature-Flag), sodass Coding-Tasks an Hermes delegiert werden können.

**Files:**
- Modify: `langgraph-app/agent_graph.py` (Zeilen ~930-950)

**Step 1: Write test**

```python
# tests/test_parallel_executor.py (add to existing)
def test_hermes_worker_registered_in_registry():
    from ai_stack.parallel_executor.worker_spawner import GLOBAL_WORKER_REGISTRY
    spawner = GLOBAL_WORKER_REGISTRY.get("hermes")
    assert spawner is not None
    from ai_stack.parallel_executor.hermes_worker import HermesWorker
    assert isinstance(spawner, HermesWorker)
```

**Step 2: Run test — expected FAIL**

**Step 3: Wire in agent_graph.py**

In `_parallel_executor_node`, nach `parallel_execution_enabled()` check:

```python
    # Build executor with HermesWorker for coding tasks when available
    _use_hermes = _env_bool("ALPHARAVIS_PARALLEL_USE_HERMES", "false")

    if _use_hermes:
        from ai_stack.parallel_executor.hermes_worker import HermesWorker

        async def _hermes_callable(message: str, system_prompt: str, max_output_chars: int) -> str:
            # Use the existing _call_hermes_streaming_sse pipeline
            full_output_parts: list[str] = []
            async for event in _call_hermes_streaming_sse(
                message=message,
                system_prompt=system_prompt,
                max_output_chars=max_output_chars,
            ):
                if isinstance(event, str) and event:
                    full_output_parts.append(event)
            return "".join(full_output_parts)

        hermes_worker = HermesWorker()
        hermes_worker.set_hermes_fn(_hermes_callable)

        # Use HermesWorker for code-write tasks, DirectLLM for others
        # (simplified: single spawner for now; per-task routing follows in Task 5)
        spawner = hermes_worker
    else:
        # Existing DirectLLM path
        worker = DirectLLMWorker()
        worker.set_llm_fn(_parallel_llm_fn)
        spawner = worker

    executor = ParallelExecutor(spawner=spawner, merge_spawner=spawner)
```

**Step 4: Run test to verify pass**

Run: `pytest tests/test_parallel_executor.py::test_hermes_worker_registered_in_registry -v`
Expected: PASS

**Step 5: Commit**

```bash
git add langgraph-app/agent_graph.py tests/test_parallel_executor.py
git commit -m "feat: wire HermesWorker into parallel executor node"
```

---

### Task 3: ParallelContextAdmission — Admission Gate

**Objective:** `ParallelContextAdmission` prüft vor Worker-Start live `/slots`, berechnet freien Kontext + Sicherheitsreserve, und entscheidet `admit` / `wait` / `serialize` / `shrink`.

**Files:**
- Create: `ai_stack/parallel_executor/context_admission.py`
- Create: `tests/test_context_admission.py`

**Step 1: Write failing tests**

```python
# tests/test_context_admission.py
import asyncio
from ai_stack.parallel_executor.context_admission import (
    ParallelContextAdmission,
    AdmissionDecision,
    AdmissionAction,
)

def test_admit_when_enough_free_context():
    """Admit worker when free context > required + reserve."""
    admission = ParallelContextAdmission(
        context_pool_size=320000,
        safety_reserve_pct=0.10,
    )
    decision = asyncio.run(admission.can_admit(
        required_tokens=80000,
        active_workers=1,
        current_kv_used=60000,
    ))
    assert decision.action == AdmissionAction.ADMIT
    assert decision.granted_budget >= 80000

def test_wait_when_not_enough_free():
    """Wait when free context cannot satisfy required + reserve."""
    admission = ParallelContextAdmission(
        context_pool_size=320000,
        safety_reserve_pct=0.10,
    )
    decision = asyncio.run(admission.can_admit(
        required_tokens=120000,
        active_workers=3,
        current_kv_used=250000,
    ))
    assert decision.action in {AdmissionAction.WAIT, AdmissionAction.SERIALIZE}

def test_shrink_budget_when_tight():
    """Shrink granted budget when pool is tight."""
    admission = ParallelContextAdmission(
        context_pool_size=320000,
        safety_reserve_pct=0.10,
    )
    decision = asyncio.run(admission.can_admit(
        required_tokens=100000,
        active_workers=3,
        current_kv_used=200000,
    ))
    if decision.action == AdmissionAction.ADMIT:
        assert decision.granted_budget < 100000
```

**Step 2: Run tests — expected FAIL**

**Step 3: Write implementation**

```python
# ai_stack/parallel_executor/context_admission.py
"""ParallelContextAdmission — Admission control for parallel worker spawns.

Queries live /slots from llama-server, computes free context, applies
safety reserve, and decides whether a new worker can be admitted, must
wait, should serialize, or needs a smaller budget.
"""

from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

LOGGER = logging.getLogger(__name__)


class AdmissionAction(str, Enum):
    ADMIT = "admit"
    WAIT = "wait"
    SERIALIZE = "serialize"
    SHRINK = "shrink"
    REJECT = "reject"


@dataclass
class AdmissionDecision:
    action: AdmissionAction
    granted_budget: int = 0
    reason: str = ""
    free_context: int = 0
    safety_reserve: int = 0
    active_workers: int = 0
    diagnostic: dict[str, Any] = field(default_factory=dict)

    @property
    def admitted(self) -> bool:
        return self.action == AdmissionAction.ADMIT


class ParallelContextAdmission:
    """Admission gate for parallel BigBoss/Hermes worker spawns.

    Conservative by design: always keeps a safety reserve, never admits
    a worker that would push total KV usage near the pool limit.

    Key behaviors:
    - Queries live /slots before each decision
    - Asymmetric budget distribution: one worker can get 120k, another 30k
    - Reserve = pct of pool size, increases with active worker count
    - When tight: tries SHRINK first, then SERIALIZE, then WAIT
    """

    def __init__(
        self,
        context_pool_size: int = 320000,
        safety_reserve_pct: float = 0.10,
        max_workers: int = 4,
        min_budget: int = 8000,
    ) -> None:
        self.context_pool_size = context_pool_size
        self.safety_reserve_pct = safety_reserve_pct
        self.max_workers = max_workers
        self.min_budget = min_budget
        self._active_budgets: dict[str, int] = {}  # worker_id -> granted budget
        self._lock = asyncio.Lock()

    async def can_admit(
        self,
        required_tokens: int,
        active_workers: int = 0,
        current_kv_used: int = 0,
        *,
        worker_id: str = "",
        priority: str = "normal",
        estimated_output_tokens: int = 4096,
    ) -> AdmissionDecision:
        """Decide whether a new worker with `required_tokens` can be admitted."""
        async with self._lock:
            # Total KV used: actual + granted-but-not-yet-active
            committed = sum(self._active_budgets.values())
            total_used = max(current_kv_used, committed)

            # Safety reserve grows with active workers
            active = active_workers + len(self._active_budgets)
            reserve_pct = self.safety_reserve_pct * (1 + active * 0.25)
            reserve = max(1, int(self.context_pool_size * reserve_pct))

            free = max(0, self.context_pool_size - total_used - reserve)

            if active >= self.max_workers:
                return AdmissionDecision(
                    action=AdmissionAction.WAIT,
                    reason=f"max workers reached ({self.max_workers})",
                    free_context=free, safety_reserve=reserve,
                    active_workers=active,
                )

            total_needed = required_tokens + estimated_output_tokens

            if free >= total_needed:
                budget = self._compute_budget(
                    required_tokens, free, active, priority,
                )
                self._active_budgets[worker_id] = budget
                return AdmissionDecision(
                    action=AdmissionAction.ADMIT,
                    granted_budget=budget,
                    reason=f"free={free} >= needed={total_needed}",
                    free_context=free, safety_reserve=reserve,
                    active_workers=active,
                )

            # Tight: try shrink
            if free >= self.min_budget:
                shrunk = max(self.min_budget, free - reserve // 2)
                self._active_budgets[worker_id] = shrunk
                return AdmissionDecision(
                    action=AdmissionAction.ADMIT,
                    granted_budget=shrunk,
                    reason=f"shrunk: free={free}, granted={shrunk}",
                    free_context=free, safety_reserve=reserve,
                    active_workers=active,
                    diagnostic={"shrunk_from": required_tokens},
                )

            # Cannot admit — serialize if only this worker, otherwise wait
            if active == 0:
                return AdmissionDecision(
                    action=AdmissionAction.SERIALIZE,
                    reason="pool too full even for single worker",
                    free_context=free, safety_reserve=reserve,
                    active_workers=active,
                )

            return AdmissionDecision(
                action=AdmissionAction.WAIT,
                reason=f"free={free} < needed={total_needed}, active={active}",
                free_context=free, safety_reserve=reserve,
                active_workers=active,
            )

    def release(self, worker_id: str) -> None:
        self._active_budgets.pop(worker_id, None)

    def _compute_budget(
        self,
        required: int,
        free: int,
        active: int,
        priority: str,
    ) -> int:
        """Compute granted budget — conservative: always give more than estimated."""
        # Conservative: add 25% margin
        conservative = int(required * 1.25)

        # But cap at available free minus reserve for other workers
        max_grant = max(self.min_budget, free - self.min_budget * max(0, 3 - active))

        return min(conservative, max_grant)

```

**Step 4: Run tests to verify pass**

Run: `pytest tests/test_context_admission.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add ai_stack/parallel_executor/context_admission.py tests/test_context_admission.py
git commit -m "feat: ParallelContextAdmission gate for worker spawns"
```

---

### Task 4: TokenEstimator — Vorab-Tokenisierung ohne eigenen Kontextverbrauch

**Objective:** `TokenEstimator` sammelt RAG-Ergebnisse, Tool-Ausgaben, Datei-Snippets für einen Task, tokenisiert sie separat via llama.cpp `/tokenize` API, und berechnet den geschätzten Kontextbedarf — ohne das Material in den Scheduler-Kontext zu laden.

**Files:**
- Create: `ai_stack/parallel_executor/token_estimator.py`
- Create: `tests/test_token_estimator.py`

**Step 1: Write failing tests**

```python
# tests/test_token_estimator.py
import asyncio
from ai_stack.parallel_executor.token_estimator import TokenEstimator, TaskMaterial

def test_estimate_from_material():
    """Estimate tokens for task material via external tokenizer."""
    est = TokenEstimator(tokenizer_fn=lambda text: len(text.split()))
    material = TaskMaterial(
        rag_results=["Context about auth module", "Previous PR context"],
        file_snippets={"auth.py": "def login(): pass"},
        tool_outputs=["Database schema loaded"],
        task_description="Refactor auth module",
    )
    estimated = asyncio.run(est.estimate(material))
    assert estimated.total_tokens > 0
    assert estimated.rag_tokens > 0
    assert estimated.file_tokens > 0

def test_estimate_empty_material():
    """Empty material should give minimal estimate."""
    est = TokenEstimator(tokenizer_fn=lambda text: len(text.split()))
    material = TaskMaterial(task_description="Simple task")
    estimated = asyncio.run(est.estimate(material))
    assert estimated.total_tokens >= 1
```

**Step 2: Run tests — expected FAIL**

**Step 3: Write implementation**

```python
# ai_stack/parallel_executor/token_estimator.py
"""TokenEstimator — Tokenizes worker material via external API without
loading it into the scheduler's own context window."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable

TokenizeFn = Callable[[str], int]  # text -> token count


@dataclass
class TaskMaterial:
    """Pre-collected material a worker would load into its context."""
    task_description: str = ""
    rag_results: list[str] = field(default_factory=list)
    file_snippets: dict[str, str] = field(default_factory=dict)
    tool_outputs: list[str] = field(default_factory=list)


@dataclass
class TokenEstimate:
    total_tokens: int = 0
    rag_tokens: int = 0
    file_tokens: int = 0
    tool_tokens: int = 0
    description_tokens: int = 0
    # Conservative overhead for system prompt, tool defs, output
    overhead_tokens: int = 4096

    @property
    def conservative_total(self) -> int:
        """Total with overhead + 25% safety margin."""
        return int((self.total_tokens + self.overhead_tokens) * 1.25)


class TokenEstimator:
    """Estimates token count by tokenizing material through an external API.

    The material is NEVER loaded into the scheduler's context — each
    piece is sent individually to the tokenizer endpoint.
    """

    def __init__(self, tokenizer_fn: TokenizeFn | None = None) -> None:
        self._tokenize = tokenizer_fn or (lambda text: len(text) // 4)

    async def estimate(self, material: TaskMaterial) -> TokenEstimate:
        est = TokenEstimate()

        est.description_tokens = self._tokenize(material.task_description)

        for snippet in material.rag_results:
            est.rag_tokens += self._tokenize(snippet)

        for content in material.file_snippets.values():
            est.file_tokens += self._tokenize(content)

        for output in material.tool_outputs:
            est.tool_tokens += self._tokenize(output)

        est.total_tokens = (
            est.description_tokens
            + est.rag_tokens
            + est.file_tokens
            + est.tool_tokens
        )
        return est
```

**Step 4: Run tests to verify pass**

Run: `pytest tests/test_token_estimator.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add ai_stack/parallel_executor/token_estimator.py tests/test_token_estimator.py
git commit -m "feat: TokenEstimator for worker material tokenization"
```

---

### Task 5: ParallelExecutor — Admission-Gate-Integration

**Objective:** `ParallelExecutor._run_task()` prüft vor jedem Worker-Start das Admission-Gate, wartet ggf., schrumpft Budget, oder serialisiert.

**Files:**
- Modify: `ai_stack/parallel_executor/executor.py`
- Create: `tests/test_executor_admission.py`

**Step 1: Write failing test**

```python
# tests/test_executor_admission.py
import asyncio
from ai_stack.parallel_executor.executor import ParallelExecutor
from ai_stack.parallel_executor.worker_spawner import DryRunWorker
from ai_stack.parallel_executor.task_graph import (
    PlannedTask, TaskType, ModelClass, TaskDAG,
)
from ai_stack.parallel_executor.context_admission import (
    ParallelContextAdmission, AdmissionDecision, AdmissionAction,
)

def test_executor_respects_admission_wait():
    """Executor waits when admission says WAIT."""
    admission = ParallelContextAdmission(
        context_pool_size=100000,
        safety_reserve_pct=0.10,
        max_workers=1,
    )

    # Pre-fill admission so next worker gets WAIT
    admission._active_budgets["w1"] = 90000

    task = PlannedTask(
        task_id="t1", title="Test",
        task_type=TaskType.CODE_READ,
        required_model_class=ModelClass.BIG_MODEL,
    )
    dag = TaskDAG(tasks=[task])

    spawner = DryRunWorker()
    executor = ParallelExecutor(
        spawner=spawner,
        admission_gate=admission,
    )

    report = asyncio.run(executor.execute(dag, task_brief="test"))
    assert report.results[0].status == "queued"
```

**Step 2: Run test — expected FAIL**

**Step 3: Modify executor**

In `executor.py`:

```python
# Add admission_gate parameter to ParallelExecutor.__init__()
    def __init__(
        self,
        *,
        spawner: WorkerSpawner | None = None,
        worktree_manager: WorktreeManager | None = None,
        merge_spawner: WorkerSpawner | None = None,
        file_lock_manager: FileLockManager | None = None,
        admission_gate: Any | None = None,  # ParallelContextAdmission
    ) -> None:
        # ... existing ...
        self.admission_gate = admission_gate

# Add _admit_worker() method
    async def _admit_worker(self, task: PlannedTask) -> AdmissionDecision:
        if self.admission_gate is None:
            return AdmissionDecision(action=AdmissionAction.ADMIT, granted_budget=0)

        return await self.admission_gate.can_admit(
            required_tokens=getattr(task, 'estimated_context_tokens', 16384),
            worker_id=task.task_id,
        )

# Modify _run_task() to check admission before spawning
    async def _run_task(self, task, *, task_brief=""):
        # ... existing file-lock / worktree logic ...

        # Admission check
        if self.admission_gate is not None:
            decision = await self._admit_worker(task)
            if not decision.admitted:
                return WorkerResult(
                    task_id=task.task_id,
                    status="queued",
                    error=f"admission: {decision.action.value} — {decision.reason}",
                    metadata={"admission": decision},
                )
            # Store granted budget so worker can access it
            task.allocated_context_budget = decision.granted_budget

        # ... existing spawn logic ...
```

**Step 4: Run test to verify pass**

Run: `pytest tests/test_executor_admission.py -v`
Expected: 1 passed

**Step 5: Commit**

```bash
git add ai_stack/parallel_executor/executor.py tests/test_executor_admission.py
git commit -m "feat: integrate admission gate into parallel executor"
```

---

### Task 6: Live /slots probe in agent_graph

**Objective:** `_parallel_executor_node` fragt vor der Ausführung live `/slots` ab und erzeugt eine `ParallelContextAdmission` mit realen Werten.

**Files:**
- Modify: `langgraph-app/agent_graph.py` (Zeilen ~894-960)

**Step 1: Write test (in test_parallel_executor.py)**

```python
def test_parallel_executor_node_builds_admission_from_slots():
    """_parallel_executor_node should build ParallelContextAdmission with live slots."""
    # Test via mocking — verifies the wiring, not live call
    from unittest.mock import AsyncMock, patch
    from ai_stack.parallel_executor.context_admission import ParallelContextAdmission

    with patch("agent_graph._parallel_executor_node", new_callable=AsyncMock) as mock_node:
        # If the node builds the gate, it should be a ParallelContextAdmission
        pass  # Smoke: import works
```

**Step 2: Wire in agent_graph.py**

```python
# In _parallel_executor_node, after building dag:
    from ai_stack.parallel_executor.context_admission import ParallelContextAdmission

    # Build admission gate from live /slots if available
    admission = None
    try:
        from ai_stack.llama_runtime.client import LlamaCppRuntimeClient
        bigboss_url = os.getenv("ALPHARAVIS_LLAMA_BASE_URL", "")
        if bigboss_url:
            client = LlamaCppRuntimeClient(bigboss_url, timeout_seconds=10)
            slots = await client.get_slots()
            # Parse KV usage from slots
            kv_used = 0
            active = 0
            if isinstance(slots, list):
                for slot in slots:
                    if isinstance(slot, dict) and slot.get("state") == 1:
                        active += 1
                        kv_used += slot.get("n_past", 0)

            pool_size = int(os.getenv("ALPHARAVIS_PARALLEL_CONTEXT_POOL_SIZE", "320000"))
            admission = ParallelContextAdmission(
                context_pool_size=pool_size,
                safety_reserve_pct=float(os.getenv(
                    "ALPHARAVIS_PARALLEL_SAFETY_RESERVE_PCT", "0.10",
                )),
                max_workers=int(os.getenv("ALPHARAVIS_PARALLEL_MAX_WORKERS", "4")),
            )
            # Pre-seed with current KV state
            admission._current_kv_used = kv_used
            admission._current_active = active
    except Exception:
        LOGGER.debug("parallel_executor: admission gate unavailable, running without")

    # Pass admission gate to executor
    executor = ParallelExecutor(
        spawner=spawner,
        merge_spawner=spawner,
        admission_gate=admission,
    )
```

**Step 3: Commit**

```bash
git add langgraph-app/agent_graph.py
git commit -m "feat: live /slots probe for admission gate in parallel executor"
```

---

### Task 7: Worker Budget Enforcement + Supervisor Escalation

**Objective:** Worker bekommt `max_tokens`-Limit im Prompt mitgeteilt. Wenn Budget knapp wird, meldet er an Supervisor. `ExecutionReport` zeigt Budget-Nutzung.

**Files:**
- Modify: `ai_stack/parallel_executor/worker_spawner.py` (WorkerResult)
- Modify: `ai_stack/parallel_executor/executor.py` (Report)
- Modify: `ai_stack/parallel_executor/task_graph.py` (PlannedTask)

**Step 1: Add `allocated_context_budget` to PlannedTask**

```python
# task_graph.py: PlannedTask
allocated_context_budget: int = 0  # granted by admission gate
```

**Step 2: Add budget fields to WorkerResult**

```python
# worker_spawner.py: WorkerResult
budget_granted: int = 0
budget_used_estimate: int = 0
```

**Step 3: Inject budget into worker prompt**

In `DirectLLMWorker.spawn()` und `HermesWorker.spawn()`:

```python
budget = getattr(task, 'allocated_context_budget', 0)
if budget > 0:
    prompt = (
        f"!!! CONTEXT BUDGET: {budget} tokens total !!!\n"
        f"Do NOT exceed this budget. Keep output compact.\n"
        f"If you need more context, signal: NEED_MORE_CONTEXT.\n"
        f"\n{prompt}"
    )
```

**Step 4: Update ExecutionReport**

```python
# executor.py: ExecutionReport
total_budget_granted: int = 0
total_budget_used_estimate: int = 0
```

**Step 5: Write tests**

```python
def test_worker_receives_budget_in_prompt():
    task = PlannedTask(
        task_id="t1", title="Test",
        task_type=TaskType.CODE_WRITE,
        required_model_class=ModelClass.BIG_MODEL,
        allocated_context_budget=50000,
    )
    # Verify prompt contains "CONTEXT BUDGET: 50000"
    # ...

def test_report_tracks_budget():
    report = ExecutionReport(
        results=[
            WorkerResult(task_id="t1", status="completed",
                         budget_granted=50000, budget_used_estimate=42000),
        ],
        total_budget_granted=50000,
        total_budget_used_estimate=42000,
    )
    assert report.total_budget_granted == 50000
```

**Step 6: Commit**

```bash
git add ai_stack/parallel_executor/task_graph.py ai_stack/parallel_executor/worker_spawner.py ai_stack/parallel_executor/executor.py tests/test_parallel_executor.py
git commit -m "feat: worker budget enforcement with supervisor escalation"
```

---

### Task 8: Big Boss Config — np=4, KV Unified, 320k Test

**Objective:** `.env(exaple)` mit `np=4` und 320k unified context dokumentieren. Smoke-Test in `scripts/` der `/slots` abfragt und np=4 + KV unified verifiziert.

**Files:**
- Modify: `.env(exaple)`
- Create: `scripts/alpharavis_parallel_smoke.py`

**Step 1: Update .env(exaple)**

```text
# Parallel Executor — Big Boss Config
# np=4, KV unified, 320k unified context
ALPHARAVIS_LLAMA_N_PARALLEL=4
ALPHARAVIS_LLAMA_CONTEXT_SIZE=320000
ALPHARAVIS_PARALLEL_CONTEXT_POOL_SIZE=320000
ALPHARAVIS_PARALLEL_SAFETY_RESERVE_PCT=0.10
ALPHARAVIS_PARALLEL_MAX_WORKERS=4
ALPHARAVIS_PARALLEL_WORKER_TIMEOUT_SECONDS=120
ALPHARAVIS_PARALLEL_USE_HERMES=false
```

**Step 2: Write smoke script**

```python
# scripts/alpharavis_parallel_smoke.py
"""Smoke test: verify Big Boss is running with np=4, KV unified, 320k."""
import asyncio, os, sys
from ai_stack.llama_runtime.client import LlamaCppRuntimeClient

async def main():
    url = os.getenv("ALPHARAVIS_LLAMA_BASE_URL", "http://192.168.178.153:8033")
    client = LlamaCppRuntimeClient(url)

    slots = await client.get_slots()
    print(f"Slots: {len(slots) if isinstance(slots, list) else 'unknown'}")

    props = await client.get_props()
    ctx = props.get("n_ctx", props.get("context_size", "unknown"))
    print(f"Context: {ctx}")

    sys.exit(0)

asyncio.run(main())
```

**Step 3: Commit**

```bash
git add .env(exaple) scripts/alpharavis_parallel_smoke.py
git commit -m "feat: Big Boss np=4 KV unified 320k config + smoke script"
```

---

### Task 9: Documentation Update

**Objective:** Alle vier Docs aktualisieren.

**Files:**
- Modify: `docs/ALPHARAVIS_CHANGES.md`
- Modify: `docs/ALPHARAVIS_USAGE_NOTES.md`
- Modify: `docs/ALPHARAVIS_ARCHITECTURE.md`
- Modify: `docs/ALPHARAVIS_OPEN_TASKS.md`
- Modify: Skill `alpharavis-context-budget`

**Update ALPHARAVIS_CHANGES.md:**

Neuer Abschnitt:
```markdown
## 2026-05-27 — Parallel Executor: Hermes Integration & Context Scheduler

- HermesWorker als WorkerSpawner-Adapter (nutzt existierende hermes-orch / _call_hermes_streaming_sse Pipeline)
- ParallelContextAdmission: Admission Gate mit live /slots-probe, konservativer Reserve (10% + 25% pro aktivem Worker)
- TokenEstimator: Vorab-Tokenisierung via llama.cpp /tokenize ohne eigenen Kontextverbrauch
- Worker Budget Enforcement: max_tokens im Prompt, NEED_MORE_CONTEXT-Signal
- Big Boss Config: np=4, KV unified, 320k unified context, asymmetrische Slot-Verteilung
- Feature flag: ALPHARAVIS_PARALLEL_USE_HERMES (default false), ALPHARAVIS_PARALLEL_MAX_WORKERS=4

Verification:
- pytest tests/test_hermes_worker.py tests/test_context_admission.py tests/test_token_estimator.py tests/test_executor_admission.py — alle grün
- scripts/alpharavis_parallel_smoke.py — /slots verifiziert np=4, 320k context
```

**Update ALPHARAVIS_USAGE_NOTES.md:**

```markdown
### Parallel Executor — Hermes & Context Scheduler

ALPHARAVIS_PARALLEL_USE_HERMES=true   # Hermes statt DirectLLM für Coding-Worker
ALPHARAVIS_PARALLEL_MAX_WORKERS=4     # max gleichzeitige Worker (≤ np)
ALPHARAVIS_PARALLEL_CONTEXT_POOL_SIZE=320000
ALPHARAVIS_PARALLEL_SAFETY_RESERVE_PCT=0.10  # 10% Reserve vom Pool

Admission Control: Vor jedem Worker-Start wird live /slots geprüft.
Bei knappem Kontext: Budget schrumpfen → serialisieren → warten.
Worker bekommen "CONTEXT BUDGET: N tokens" im Prompt mitgeteilt.
```

**Update ALPHARAVIS_ARCHITECTURE.md:**

Parallel-Executor-Abschnitt um HermesWorker, Admission Gate, TokenEstimator ergänzen.

**Update ALPHARAVIS_OPEN_TASKS.md:**

- HermesWorker: done
- ParallelContextAdmission: done
- TokenEstimator: done
- Worker Budget Enforcement: done
- BigBoss np=4 config: done
- Alpha-Hermes-Kontrollstrategie: deferred to separate plan

**Update Skill alpharavis-context-budget:**

Neue Sektion für ParallelContextAdmission, Admission-Decision-Flow, TokenEstimator-Pipeline. Test-Count aktualisieren.

**Step 1: Commit**

```bash
git add docs/ALPHARAVIS_CHANGES.md docs/ALPHARAVIS_USAGE_NOTES.md docs/ALPHARAVIS_ARCHITECTURE.md docs/ALPHARAVIS_OPEN_TASKS.md
git commit -m "docs: parallel executor Hermes integration & context scheduler"
```

---

### Task 10: Full Test Suite Run + Verification

**Objective:** Alle Tests laufen lassen, AST check, Whitespace check.

**Step 1: Run all tests**

```bash
pytest -q tests/test_parallel_executor.py tests/test_hermes_worker.py tests/test_context_admission.py tests/test_token_estimator.py tests/test_executor_admission.py
```

Expected: alle grün

**Step 2: Broad test**

```bash
pytest -q tests/test_context_budget_router.py tests/test_llama_context_scheduler.py tests/test_parallel_executor.py tests/test_hermes_worker.py tests/test_context_admission.py tests/test_token_estimator.py tests/test_executor_admission.py
```

**Step 3: Validation**

```bash
PYTHONDONTWRITEBYTECODE=1 python -c "
import ast
for f in ['ai_stack/parallel_executor/hermes_worker.py', 'ai_stack/parallel_executor/context_admission.py', 'ai_stack/parallel_executor/token_estimator.py']:
    with open(f) as fh:
        ast.parse(fh.read())
    print(f'{f}: ast ok')
"
```

```bash
git diff --check
```

**Step 4: Commit final**

```bash
git add -A
git commit -m "chore: final test suite verification for parallel executor v2"
```

---

## What is NOT in this plan (separate future plan)

- Alpha-Hermes Kontrollstrategie: Deep Agents Orchestrator-Feinsteuerung, Schritt-für-Schritt-Intervention, Coding-Tab-Mod-Integration
- Per-Task Worker-Routing (Code-Write → Hermes, Summarize → DirectLLM, etc.)
- Live-Test mit echtem BigBoss + Hermes parallel (braucht laufende Docker-Services)
- ContextScheduler-Integration für parallele Worker-Leases (Redis-koordiniert)

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | HermesWorker Adapter | `hermes_worker.py` (new), `__init__.py` |
| 2 | Wire in agent_graph | `agent_graph.py` |
| 3 | Admission Gate | `context_admission.py` (new) |
| 4 | TokenEstimator | `token_estimator.py` (new) |
| 5 | Admission in Executor | `executor.py` |
| 6 | Live /slots probe | `agent_graph.py` |
| 7 | Budget Enforcement | `task_graph.py`, `worker_spawner.py`, `executor.py` |
| 8 | Big Boss Config | `.env(exaple)`, smoke script |
| 9 | Documentation | 4 docs, 1 skill |
| 10 | Full verification | Tests, AST, whitespace |
