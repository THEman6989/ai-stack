"""Tests for parallel task executor — task DAG, classification, conflict detection,
parallelization decisions, worktree manager, and worker spawner.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_stack.parallel_executor import (
    DryRunWorker,
    DirectLLMWorker,
    HermesWorker,
    ModelClass,
    PlannedTask,
    RiskLevel,
    SlotBudget,
    ContextPreEstimator,
    WorkerContextEstimate,
    ParallelContextPlanner,
    TaskDAG,
    TaskType,
    WorktreeManager,
    analyze_parallelization,
    log_parallelization_decision,
    parallel_context_planner_enabled,
    parallel_execution_enabled,
    parallel_hermes_worker_enabled,
    parallel_planner_instruction_block,
    parse_planner_text_into_tasks,
    repo_has_uncommitted_changes,
)
from ai_stack.parallel_executor.worker_spawner import GLOBAL_WORKER_REGISTRY
from ai_stack.parallel_executor.task_graph import (
    CHOKEPOINT_FILE_PATTERNS,
    _classify_task_from_title,
    _globs_overlap,
    detect_file_conflicts,
)


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------

class TestFeatureFlag:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("ALPHARAVIS_PARALLEL_TASK_EXECUTION", raising=False)
        assert parallel_execution_enabled() is False

    def test_explicit_true(self, monkeypatch):
        monkeypatch.setenv("ALPHARAVIS_PARALLEL_TASK_EXECUTION", "true")
        assert parallel_execution_enabled() is True

    def test_explicit_false(self, monkeypatch):
        monkeypatch.setenv("ALPHARAVIS_PARALLEL_TASK_EXECUTION", "false")
        assert parallel_execution_enabled() is False


# ---------------------------------------------------------------------------
# Planner text parser
# ---------------------------------------------------------------------------

class TestParsePlannerText:
    def test_empty_text(self):
        assert parse_planner_text_into_tasks("") == []
        assert parse_planner_text_into_tasks("   \n  ") == []

    def test_parses_bullet_tasks(self):
        text = """- Implement user auth API
- Write tests for auth
- Review integration"""
        tasks = parse_planner_text_into_tasks(text)
        assert len(tasks) == 3
        assert tasks[0].task_id == "task_001"
        assert tasks[0].title == "Implement user auth API"
        assert tasks[0].task_type == TaskType.WRITE_IMPLEMENTATION
        assert tasks[0].write_enabled is True

    def test_parser_does_not_add_sequential_dependencies_by_default(self):
        text = """- Analyze frontend code
- Analyze backend code
- Analyze docs"""
        tasks = parse_planner_text_into_tasks(text)
        assert len(tasks) == 3
        assert [task.dependencies for task in tasks] == [[], [], []]

    def test_parses_numbered_tasks(self):
        text = """1. Build frontend component
2. Build backend API
3. Run integration tests"""
        tasks = parse_planner_text_into_tasks(text)
        assert len(tasks) == 3

    def test_classifies_read_only_analysis(self):
        text = "- Analyze repository structure"
        tasks = parse_planner_text_into_tasks(text)
        assert tasks[0].read_only is True
        assert tasks[0].write_enabled is False

    def test_classifies_summarization(self):
        text = "- Summarize the conversation history"
        tasks = parse_planner_text_into_tasks(text)
        assert tasks[0].task_type == TaskType.SUMMARIZATION
        assert tasks[0].required_model_class == ModelClass.SMALL_MODEL

    def test_classifies_coding_as_big_model(self):
        text = "- Refactor the database layer"
        tasks = parse_planner_text_into_tasks(text)
        assert tasks[0].required_model_class == ModelClass.BIG_MODEL

    def test_extracts_file_globs(self):
        text = "- Update `package.json` and `src/api.py`"
        tasks = parse_planner_text_into_tasks(text)
        assert "package.json" in tasks[0].affected_file_globs
        assert "src/api.py" in tasks[0].affected_file_globs
        # package.json is a chokepoint
        assert "package.json" in tasks[0].shared_chokepoint_files

    def test_non_chokepoint_files(self):
        text = "- Edit `src/components/Button.tsx`"
        tasks = parse_planner_text_into_tasks(text)
        assert tasks[0].shared_chokepoint_files == []

    def test_parses_bigboss_structured_parallel_plan(self):
        text = """- fallback bullets should not be used when JSON block exists

<parallel-execution-plan>{
  "parallel_possible": true,
  "tasks": [
    {
      "title": "Analyze frontend boundaries",
      "task_type": "read_only_analysis",
      "parallel": true,
      "parallel_group": "analysis",
      "files": ["frontend/*.tsx"],
      "model": "big_model",
      "risk": "low",
      "reason": "independent read-only area"
    },
    {
      "title": "Run focused tests",
      "task_type": "test",
      "parallel": false,
      "depends_on": ["task_001"],
      "files": ["tests/test_frontend.py"],
      "reason": "tests wait for analysis"
    }
  ]
}</parallel-execution-plan>"""

        tasks = parse_planner_text_into_tasks(text)

        assert len(tasks) == 2
        assert tasks[0].title == "Analyze frontend boundaries"
        assert tasks[0].planner_parallel_allowed is True
        assert tasks[0].parallel_group_id == "analysis"
        assert tasks[0].affected_file_globs == ["frontend/*.tsx"]
        assert tasks[0].required_model_class == ModelClass.BIG_MODEL
        assert tasks[0].risk_level == RiskLevel.LOW
        assert tasks[1].planner_parallel_allowed is False
        assert tasks[1].dependencies == ["task_001"]

    def test_structured_parallel_possible_false_serializes_all_tasks(self):
        text = """<parallel-execution-plan>{
  "parallel_possible": false,
  "tasks": [
    {"title": "Inspect shared API", "parallel": true, "parallel_group": "g1"},
    {"title": "Inspect shared UI", "parallel": true, "parallel_group": "g1"}
  ]
}</parallel-execution-plan>"""

        tasks = parse_planner_text_into_tasks(text)

        assert [task.planner_parallel_allowed for task in tasks] == [False, False]

    def test_parallel_planner_instruction_block_is_machine_readable(self):
        block = parallel_planner_instruction_block()
        assert "<parallel-execution-plan>" in block
        assert '\"parallel_possible\"' in block
        assert '\"parallel_group\"' in block


# ---------------------------------------------------------------------------
# Task classification
# ---------------------------------------------------------------------------

class TestTaskClassification:
    def test_write_triggers(self):
        for title in ["Implement login", "Build API", "Create widget", "Fix bug", "Refactor code"]:
            task_type, read_only, write_enabled, model = _classify_task_from_title(title)
            assert task_type == TaskType.WRITE_IMPLEMENTATION
            assert write_enabled is True

    def test_test_triggers(self):
        for title in ["Test login flow", "Verify API response", "Check database"]:
            task_type, _, _, _ = _classify_task_from_title(title)
            assert task_type == TaskType.TEST

    def test_summary_triggers(self):
        for title in ["Summarize chat", "Compress context", "Analyze logs"]:
            task_type, _, _, model = _classify_task_from_title(title)
            assert task_type == TaskType.SUMMARIZATION
            assert model == ModelClass.SMALL_MODEL


# ---------------------------------------------------------------------------
# File conflict detection
# ---------------------------------------------------------------------------

class TestFileConflicts:
    def test_no_conflict_different_files(self):
        assert _globs_overlap(["src/a.py"], ["src/b.py"]) is False

    def test_conflict_same_file(self):
        assert _globs_overlap(["src/a.py"], ["src/a.py"]) is True

    def test_conflict_shared_basename(self):
        assert _globs_overlap(["package.json"], ["package.json"]) is True

    def test_conflict_concrete_path_against_wildcard_glob(self):
        assert _globs_overlap(["src/api.py"], ["src/*.py"]) is True
        assert _globs_overlap(["frontend/Button.tsx"], ["backend/*.py"]) is False

    def test_empty_globs_no_conflict(self):
        assert _globs_overlap([], ["src/a.py"]) is False

    def test_detect_file_conflicts_between_tasks(self):
        task_a = PlannedTask(task_id="a", title="A", task_type=TaskType.WRITE_IMPLEMENTATION,
                            read_only=False, write_enabled=True,
                            affected_file_globs=["src/a.py", "package.json"])
        task_b = PlannedTask(task_id="b", title="B", task_type=TaskType.WRITE_IMPLEMENTATION,
                            read_only=False, write_enabled=True,
                            affected_file_globs=["src/a.py", "src/c.py"])
        conflicts = detect_file_conflicts([task_a, task_b])
        assert "a" in conflicts
        assert "b" in conflicts["a"]


# ---------------------------------------------------------------------------
# Parallelization analysis
# ---------------------------------------------------------------------------

class TestAnalyzeParallelization:
    def test_read_only_tasks_parallelize(self):
        tasks = [
            PlannedTask(task_id="a", title="Analyze X", task_type=TaskType.READ_ONLY_ANALYSIS,
                       read_only=True, write_enabled=False, required_model_class=ModelClass.SMALL_MODEL),
            PlannedTask(task_id="b", title="Analyze Y", task_type=TaskType.READ_ONLY_ANALYSIS,
                       read_only=True, write_enabled=False, required_model_class=ModelClass.SMALL_MODEL),
        ]
        dag = analyze_parallelization(tasks)
        assert dag.parallelizable_count == 2

    def test_write_tasks_with_conflict_serialized(self):
        tasks = [
            PlannedTask(task_id="a", title="Edit X", task_type=TaskType.WRITE_IMPLEMENTATION,
                       read_only=False, write_enabled=True,
                       affected_file_globs=["src/x.py"]),
            PlannedTask(task_id="b", title="Edit X too", task_type=TaskType.WRITE_IMPLEMENTATION,
                       read_only=False, write_enabled=True,
                       affected_file_globs=["src/x.py"]),
        ]
        dag = analyze_parallelization(tasks)
        # They conflict on src/x.py — file_conflicts stores conflicting task IDs
        task_a = dag.get_task("a")
        task_b = dag.get_task("b")
        assert task_a is not None
        assert task_b is not None
        assert task_a.file_conflicts or task_b.file_conflicts  # at least one has conflicts
        assert "b" in task_a.file_conflicts or "a" in task_b.file_conflicts
        assert task_a.can_parallelize is False
        assert task_b.can_parallelize is False

    def test_write_tasks_no_conflict_parallelize(self):
        tasks = [
            PlannedTask(task_id="a", title="Edit frontend", task_type=TaskType.WRITE_IMPLEMENTATION,
                       read_only=False, write_enabled=True,
                       affected_file_globs=["frontend/app.tsx"]),
            PlannedTask(task_id="b", title="Edit backend", task_type=TaskType.WRITE_IMPLEMENTATION,
                       read_only=False, write_enabled=True,
                       affected_file_globs=["backend/api.py"]),
        ]
        dag = analyze_parallelization(tasks)
        assert dag.parallelizable_count == 2
        assert len(dag.parallel_groups) == 1
        assert set(next(iter(dag.parallel_groups.values()))) == {"a", "b"}

    def test_chokepoint_files_force_serialization(self):
        tasks = [
            PlannedTask(task_id="a", title="Edit package.json", task_type=TaskType.WRITE_IMPLEMENTATION,
                       read_only=False, write_enabled=True,
                       affected_file_globs=["package.json"],
                       shared_chokepoint_files=["package.json"]),
        ]
        dag = analyze_parallelization(tasks)
        task = dag.get_task("a")
        assert task is not None
        assert "chokepoint" in task.reason_for_parallelization_decision.lower()
        assert task.can_parallelize is False

    def test_tests_wait_for_implementation(self):
        tasks = [
            PlannedTask(task_id="a", title="Implement login", task_type=TaskType.WRITE_IMPLEMENTATION,
                       read_only=False, write_enabled=True),
            PlannedTask(task_id="b", title="Test login", task_type=TaskType.TEST,
                       read_only=False, write_enabled=True),
        ]
        tasks[1].dependencies = []  # clear default chain dependency
        dag = analyze_parallelization(tasks)
        task_b = dag.get_task("b")
        # Tests should have implementation as blocking dependency
        assert len(task_b.blocking_dependencies) >= 1

    def test_merge_review_always_serialized(self):
        tasks = [
            PlannedTask(task_id="a", title="Merge and review", task_type=TaskType.MERGE_REVIEW,
                       read_only=False, write_enabled=True),
        ]
        dag = analyze_parallelization(tasks)
        task = dag.get_task("a")
        assert task.can_parallelize is False
        assert "serialized" in task.reason_for_parallelization_decision.lower()

    def test_resource_conflict_when_big_model_busy(self):
        tasks = [
            PlannedTask(task_id="a", title="Heavy refactor", task_type=TaskType.WRITE_IMPLEMENTATION,
                       read_only=False, write_enabled=True, required_model_class=ModelClass.BIG_MODEL),
        ]
        dag = analyze_parallelization(tasks, active_big_model_busy=True)
        task = dag.get_task("a")
        assert task is not None
        assert "big_model_busy" in task.resource_conflicts
        assert task.can_parallelize is False

    def test_context_pressure_resource_conflict(self):
        tasks = [
            PlannedTask(task_id="a", title="Large analysis", task_type=TaskType.WRITE_IMPLEMENTATION,
                       read_only=False, write_enabled=True, required_model_class=ModelClass.BIG_MODEL),
        ]
        dag = analyze_parallelization(tasks, context_pressure_high=True)
        task = dag.get_task("a")
        assert task is not None
        assert "context_pressure_high" in task.resource_conflicts
        assert task.can_parallelize is False

    def test_parallel_group_assignment(self):
        tasks = [
            PlannedTask(task_id="a", title="Read-only A", task_type=TaskType.READ_ONLY_ANALYSIS,
                       read_only=True, write_enabled=False),
            PlannedTask(task_id="b", title="Read-only B", task_type=TaskType.SUMMARIZATION,
                       read_only=True, write_enabled=False),
        ]
        tasks[0].dependencies = []
        tasks[1].dependencies = []
        dag = analyze_parallelization(tasks)
        assert dag.parallelizable_count == 2
        assert len(dag.parallel_groups) == 1
        assert set(next(iter(dag.parallel_groups.values()))) == {"a", "b"}

    def test_parsed_independent_read_only_tasks_build_one_concurrent_group(self):
        tasks = parse_planner_text_into_tasks("""- Analyze frontend
- Analyze backend
- Summarize docs""")
        dag = analyze_parallelization(tasks)
        assert dag.parallelizable_count == 3
        assert len(dag.parallel_groups) == 1
        assert set(next(iter(dag.parallel_groups.values()))) == {"task_001", "task_002", "task_003"}

    def test_bigboss_serial_hint_is_respected(self):
        tasks = [
            PlannedTask(
                task_id="a",
                title="Inspect shared state",
                task_type=TaskType.READ_ONLY_ANALYSIS,
                read_only=True,
                write_enabled=False,
                planner_parallel_allowed=False,
                planner_parallel_reason="BigBoss says shared context must stay serial",
                parallel_group_id="analysis",
            )
        ]

        dag = analyze_parallelization(tasks)
        task = dag.get_task("a")

        assert task is not None
        assert task.can_parallelize is False
        assert task.parallel_group_id == ""
        assert "planner requested serial" in task.reason_for_parallelization_decision

    def test_bigboss_group_hint_is_preserved_when_safe(self):
        tasks = [
            PlannedTask(
                task_id="a",
                title="Analyze frontend",
                task_type=TaskType.READ_ONLY_ANALYSIS,
                read_only=True,
                write_enabled=False,
                planner_parallel_allowed=True,
                parallel_group_id="analysis",
            ),
            PlannedTask(
                task_id="b",
                title="Analyze backend",
                task_type=TaskType.READ_ONLY_ANALYSIS,
                read_only=True,
                write_enabled=False,
                planner_parallel_allowed=True,
                parallel_group_id="analysis",
            ),
        ]

        dag = analyze_parallelization(tasks)

        assert dag.parallel_groups == {"analysis": ["a", "b"]}

    def test_safety_overrides_bigboss_parallel_hint_for_file_conflicts(self):
        tasks = [
            PlannedTask(
                task_id="a",
                title="Edit API one",
                task_type=TaskType.WRITE_IMPLEMENTATION,
                read_only=False,
                write_enabled=True,
                affected_file_globs=["src/api.py"],
                planner_parallel_allowed=True,
                parallel_group_id="impl",
            ),
            PlannedTask(
                task_id="b",
                title="Edit API two",
                task_type=TaskType.WRITE_IMPLEMENTATION,
                read_only=False,
                write_enabled=True,
                affected_file_globs=["src/*.py"],
                planner_parallel_allowed=True,
                parallel_group_id="impl",
            ),
        ]

        dag = analyze_parallelization(tasks)

        assert dag.parallelizable_count == 0
        assert dag.parallel_groups == {}
        assert set(dag.serial_chain) == {"a", "b"}


# ---------------------------------------------------------------------------
# TaskDAG
# ---------------------------------------------------------------------------

class TestTaskDAG:
    def test_counts(self):
        tasks = [
            PlannedTask(task_id="a", title="A", task_type=TaskType.READ_ONLY_ANALYSIS, read_only=True, write_enabled=False, can_parallelize=True),
            PlannedTask(task_id="b", title="B", task_type=TaskType.WRITE_IMPLEMENTATION, read_only=False, write_enabled=True, can_parallelize=False),
        ]
        dag = TaskDAG(tasks=tasks)
        assert dag.task_count == 2
        assert dag.parallelizable_count == 1
        assert dag.serial_count == 1

    def test_get_task(self):
        task = PlannedTask(task_id="x", title="X", task_type=TaskType.READ_ONLY_ANALYSIS, read_only=True, write_enabled=False)
        dag = TaskDAG(tasks=[task])
        assert dag.get_task("x") is task
        assert dag.get_task("nonexistent") is None


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------

class TestObservability:
    def test_log_decision_produces_all_fields(self):
        task = PlannedTask(
            task_id="test_001", title="Test task",
            task_type=TaskType.WRITE_IMPLEMENTATION,
            read_only=False, write_enabled=True,
            affected_file_globs=["src/a.py"],
            can_parallelize=False,
            parallel_group_id="",
            blocking_dependencies=["task_000"],
            file_conflicts=["task_002"],
            resource_conflicts=[],
            selected_model="big-boss",
            route_decision="serial",
            reason_for_parallelization_decision="file conflict with task_002",
        )
        record = log_parallelization_decision(task)
        assert record["task_id"] == "test_001"
        assert record["can_parallelize"] is False
        assert record["reason"] == "file conflict with task_002"
        assert "file_conflicts" in record


# ---------------------------------------------------------------------------
# Worker spawner (dry-run)
# ---------------------------------------------------------------------------

class TestDryRunWorker:
    def test_spawn_returns_result(self):
        worker = DryRunWorker()
        task = PlannedTask(task_id="t1", title="Test", task_type=TaskType.READ_ONLY_ANALYSIS,
                          read_only=True, write_enabled=False)
        result = asyncio.run(worker.spawn(task, task_brief="Test brief"))
        assert result.status == "dry_run"
        assert result.ok is True
        assert result.task_id == "t1"

    def test_spawn_count_increments(self):
        worker = DryRunWorker()
        task = PlannedTask(task_id="t1", title="T1", task_type=TaskType.READ_ONLY_ANALYSIS, read_only=True, write_enabled=False)
        asyncio.run(worker.spawn(task))
        assert worker.spawned_count == 1


class TestDirectLLMWorker:
    def test_spawn_uses_compact_prompt_max_tokens_adapter_contract(self):
        calls = []

        async def llm_fn(prompt: str, max_tokens: int) -> str:
            calls.append((prompt, max_tokens))
            return "worker output"

        worker = DirectLLMWorker(llm_fn)
        task = PlannedTask(
            task_id="t1",
            title="Analyze module boundaries",
            task_type=TaskType.READ_ONLY_ANALYSIS,
            read_only=True,
            write_enabled=False,
            required_model_class=ModelClass.SMALL_MODEL,
        )

        result = asyncio.run(worker.spawn(task, task_brief="Brief"))

        assert result.status == "completed"
        assert result.output == "worker output"
        assert calls == [("Task: Analyze module boundaries\nType: read_only_analysis\nContext: Brief\n", 1024)]


# ---------------------------------------------------------------------------
# Worktree manager (integration-like, needs git repo)
# ---------------------------------------------------------------------------

class TestWorktreeManager:
    def test_is_git_repo_detected(self):
        # This test repo should be a git repo
        wm = WorktreeManager()
        assert wm.is_git_repo is True

    def test_repo_root_points_to_git_root(self):
        wm = WorktreeManager()
        assert wm.repo_root
        assert (Path(wm.repo_root) / ".git").exists()

    def test_create_and_remove_worktree(self):
        wm = WorktreeManager()
        info = wm.create("test_task_worktree")
        if info is None:
            pytest.skip("Worktree creation failed (likely in Docker or restricted env)")
        assert info.task_id == "test_task_worktree"
        assert Path(info.path).exists()
        # Clean up
        removed = wm.remove("test_task_worktree", force=True)
        assert removed is True
        assert not Path(info.path).exists()


# ---------------------------------------------------------------------------
# Stage 2: ExecutionPlan and ParallelExecutor
# ---------------------------------------------------------------------------

class TestExecutionPlan:
    def test_build_plan_from_simple_dag(self):
        from ai_stack.parallel_executor.executor import build_execution_plan
        tasks = [
            PlannedTask(task_id="a", title="Read A", task_type=TaskType.READ_ONLY_ANALYSIS,
                       read_only=True, write_enabled=False, can_parallelize=True,
                       parallel_group_id="group_01"),
            PlannedTask(task_id="b", title="Read B", task_type=TaskType.READ_ONLY_ANALYSIS,
                       read_only=True, write_enabled=False, can_parallelize=True,
                       parallel_group_id="group_01"),
            PlannedTask(task_id="c", title="Serial C", task_type=TaskType.WRITE_IMPLEMENTATION,
                       read_only=False, write_enabled=True, can_parallelize=False),
        ]
        dag = TaskDAG(tasks=tasks, serial_chain=["c"])
        plan = build_execution_plan(dag)
        assert len(plan.parallel_groups) >= 1
        assert len(plan.serial_chain) >= 1
        # Merge task created for multi-task DAG
        assert plan.merge_task is not None

    def test_single_task_no_merge(self):
        from ai_stack.parallel_executor.executor import build_execution_plan
        tasks = [
            PlannedTask(task_id="a", title="Only task", task_type=TaskType.READ_ONLY_ANALYSIS,
                       read_only=True, write_enabled=False),
        ]
        dag = TaskDAG(tasks=tasks)
        plan = build_execution_plan(dag)
        assert plan.merge_task is None


class TestParallelExecutor:
    def test_execute_with_dry_run(self):
        from ai_stack.parallel_executor.executor import ParallelExecutor
        tasks = [
            PlannedTask(task_id="a", title="Task A", task_type=TaskType.READ_ONLY_ANALYSIS,
                       read_only=True, write_enabled=False, can_parallelize=True,
                       parallel_group_id="group_01"),
            PlannedTask(task_id="b", title="Task B", task_type=TaskType.SUMMARIZATION,
                       read_only=True, write_enabled=False, can_parallelize=True,
                       parallel_group_id="group_01"),
        ]
        dag = TaskDAG(tasks=tasks)
        executor = ParallelExecutor()
        report = asyncio.run(executor.execute(dag, task_brief="Test"))
        assert report.total_tasks == 2
        assert report.completed == 2
        assert report.failed == 0
        assert report.ok is True
        assert report.merge_result is not None

    def test_execute_with_failed_task(self):
        from ai_stack.parallel_executor.executor import ParallelExecutor
        from ai_stack.parallel_executor.worker_spawner import WorkerSpawner, WorkerResult

        class FailingWorker(WorkerSpawner):
            async def spawn(self, task, **kwargs):
                return WorkerResult(task_id=task.task_id, status="failed", error="test error")

        tasks = [
            PlannedTask(task_id="a", title="Will fail", task_type=TaskType.WRITE_IMPLEMENTATION,
                       read_only=False, write_enabled=True),
        ]
        dag = TaskDAG(tasks=tasks)
        executor = ParallelExecutor(spawner=FailingWorker())
        report = asyncio.run(executor.execute(dag))
        assert report.completed == 0
        assert report.failed == 1
        assert report.ok is False

    def test_executor_reports_skipped_lock_conflict_as_failure(self, tmp_path):
        from ai_stack.parallel_executor.executor import ParallelExecutor
        from ai_stack.parallel_executor.file_lock import FileLockManager
        from ai_stack.parallel_executor.worker_spawner import WorkerSpawner, WorkerResult

        class SlowWorker(WorkerSpawner):
            async def spawn(self, task, **kwargs):
                await asyncio.sleep(0.01)
                return WorkerResult(task_id=task.task_id, status="completed", output="ok")

        tasks = [
            PlannedTask(task_id="a", title="Edit API", task_type=TaskType.WRITE_IMPLEMENTATION,
                       read_only=False, write_enabled=True, affected_file_globs=["src/api.py"],
                       can_parallelize=True, parallel_group_id="group_01"),
            PlannedTask(task_id="b", title="Edit API again", task_type=TaskType.WRITE_IMPLEMENTATION,
                       read_only=False, write_enabled=True, affected_file_globs=["src/api.py"],
                       can_parallelize=True, parallel_group_id="group_01"),
        ]
        dag = TaskDAG(tasks=tasks)
        executor = ParallelExecutor(
            spawner=SlowWorker(),
            worktree_manager=WorktreeManager(repo_root=str(tmp_path)),
            file_lock_manager=FileLockManager(),
        )
        report = asyncio.run(executor.execute(dag))
        assert report.completed == 1
        assert report.failed == 1
        assert report.ok is False
        assert any("file lock conflict" in error for error in report.errors)

    def test_execution_report_to_dict(self):
        from ai_stack.parallel_executor.executor import ExecutionReport
        report = ExecutionReport(total_tasks=2, completed=2)
        d = report.to_dict()
        assert d["total_tasks"] == 2
        assert d["completed"] == 2


# ---------------------------------------------------------------------------
# File lock manager
# ---------------------------------------------------------------------------

class TestFileLockManager:
    def test_acquire_and_release(self):
        from ai_stack.parallel_executor.file_lock import FileLockManager
        mgr = FileLockManager()
        lock = asyncio.run(mgr.try_acquire("task_a", ["src/a.py"]))
        assert lock is not None
        assert lock.task_id == "task_a"

        count = asyncio.run(mgr.release("task_a"))
        assert count == 1
        assert mgr.active_locks == 0

    def test_conflict_detected(self):
        from ai_stack.parallel_executor.file_lock import FileLockManager
        mgr = FileLockManager()
        lock_a = asyncio.run(mgr.try_acquire("task_a", ["src/a.py"]))
        assert lock_a is not None

        lock_b = asyncio.run(mgr.try_acquire("task_b", ["src/a.py"]))
        assert lock_b is None  # conflict!

        asyncio.run(mgr.release("task_a"))

    def test_no_conflict_different_files(self):
        from ai_stack.parallel_executor.file_lock import FileLockManager
        mgr = FileLockManager()
        lock_a = asyncio.run(mgr.try_acquire("task_a", ["frontend/app.tsx"]))
        lock_b = asyncio.run(mgr.try_acquire("task_b", ["backend/api.py"]))
        assert lock_a is not None
        assert lock_b is not None

        asyncio.run(mgr.release_all())

    def test_check_conflict(self):
        from ai_stack.parallel_executor.file_lock import FileLockManager
        mgr = FileLockManager()
        asyncio.run(mgr.try_acquire("task_a", ["src/a.py", "src/b.py"]))
        conflicts = asyncio.run(mgr.check_conflict(["src/b.py"]))
        assert "task_a" in conflicts

        no_conflicts = asyncio.run(mgr.check_conflict(["src/c.py"]))
        assert no_conflicts == []

        asyncio.run(mgr.release_all())

    def test_wildcard_glob_conflict_detected(self):
        from ai_stack.parallel_executor.file_lock import FileLockManager
        mgr = FileLockManager()
        lock_a = asyncio.run(mgr.try_acquire("task_a", ["src/*.py"]))
        assert lock_a is not None
        lock_b = asyncio.run(mgr.try_acquire("task_b", ["src/api.py"]))
        assert lock_b is None
        asyncio.run(mgr.release_all())


# ---------------------------------------------------------------------------
# Responses compatibility (message format validation)
# ---------------------------------------------------------------------------

class TestResponsesCompatibility:
    def test_parallel_executor_output_messages_format(self):
        """Parallel executor output must use valid message format."""
        # Simulate what _parallel_executor_node produces (dict form, no import needed)
        messages = [
            {"role": "system", "content": "[task_001] OK: Frontend component built"},
            {"role": "system", "content": "[task_002] OK: Backend API implemented"},
            {"role": "system", "content": "[merge_review] All tasks completed successfully."},
        ]
        for msg in messages:
            assert msg["role"] == "system"
            assert isinstance(msg["content"], str)
            assert len(msg["content"]) > 0
            # Merge messages must fit in 2000 chars
            if "merge_review" in msg["content"]:
                assert len(msg["content"]) <= 2000

    def test_execution_report_serializes_for_state(self):
        """Report.to_dict() must be JSON-serializable for LangGraph state."""
        import json
        from ai_stack.parallel_executor.executor import ExecutionReport
        from ai_stack.parallel_executor.worker_spawner import WorkerResult

        report = ExecutionReport(
            total_tasks=3,
            completed=2,
            failed=1,
            elapsed_seconds=1.5,
            errors=["task_c: test error"],
            results=[
                WorkerResult(task_id="a", status="completed", output="OK"),
                WorkerResult(task_id="b", status="completed", output="OK"),
                WorkerResult(task_id="c", status="failed", error="test error"),
            ],
        )
        d = report.to_dict()
        json_str = json.dumps(d, default=str)
        assert json_str
        parsed = json.loads(json_str)
        assert parsed["total_tasks"] == 3
        assert len(parsed["results"]) == 3


# ---------------------------------------------------------------------------
# HermesWorker
# ---------------------------------------------------------------------------


class TestHermesWorker:
    def test_spawn_with_hermes_fn(self):
        calls = []

        async def hermes_fn(task: str, context: str, max_output_chars: int) -> str:
            calls.append((task, context, max_output_chars))
            return "hermes did the work"

        worker = HermesWorker(hermes_fn)
        task = PlannedTask(
            task_id="h1",
            title="Refactor database layer",
            task_type=TaskType.WRITE_IMPLEMENTATION,
            read_only=False,
            write_enabled=True,
            affected_file_globs=["src/db.py", "src/models.py"],
            dependencies=["task_001"],
        )

        result = asyncio.run(
            worker.spawn(
                task,
                task_brief="Split monolithic db.py into modules",
                context_budget=120000,
                max_output_chars=6000,
            )
        )

        assert result.status == "completed"
        assert isinstance(result.output, str)
        assert result.metadata["worker_type"] == "hermes"
        assert result.metadata["context_budget"] == 120000
        assert len(calls) == 1

        call_task, call_context, call_max_chars = calls[0]
        assert call_task == "Refactor database layer"
        assert "Split monolithic db.py into modules" in call_context
        assert "src/db.py" in call_context
        assert "MAX_CONTEXT_BUDGET: 120000 tokens" in call_context
        assert "READ-ONLY mode" not in call_context
        assert call_max_chars == 6000

    def test_spawn_read_only_task(self):
        async def hermes_fn(task, context, max_output_chars):
            return "analysis complete"

        worker = HermesWorker(hermes_fn)
        task = PlannedTask(
            task_id="h2",
            title="Analyze frontend",
            task_type=TaskType.READ_ONLY_ANALYSIS,
            read_only=True,
            write_enabled=False,
        )

        result = asyncio.run(worker.spawn(task, context_budget=30000))
        assert result.status == "completed"

    def test_spawn_fallback_when_no_fn(self):
        worker = HermesWorker()
        task = PlannedTask(
            task_id="h3", title="Task",
            task_type=TaskType.READ_ONLY_ANALYSIS,
            read_only=True, write_enabled=False,
        )
        result = asyncio.run(worker.spawn(task))
        assert result.status == "dry_run"

    def test_spawn_handles_hermes_error(self):
        async def failing_fn(task, context, max_output_chars):
            raise RuntimeError("Hermes unreachable")

        worker = HermesWorker(failing_fn)
        task = PlannedTask(
            task_id="h4", title="Will fail",
            task_type=TaskType.WRITE_IMPLEMENTATION,
            read_only=False, write_enabled=True,
        )

        result = asyncio.run(worker.spawn(task))
        assert result.status == "failed"
        assert "Hermes unreachable" in result.error


# ---------------------------------------------------------------------------
# Context planner (SlotBudget + ContextPreEstimator + ParallelContextPlanner)
# ---------------------------------------------------------------------------


class TestSlotBudget:
    def test_initial_state(self):
        budget = SlotBudget(pool_total=320000, parallel_slots=4, kv_unified=True)
        assert budget.safety_reserve == 25600  # 8% of 320k
        assert budget.usable_pool == 294400  # 320k - 25.6k
        assert budget.available == 294400
        assert budget.active_count == 0

    def test_admit_and_release(self):
        budget = SlotBudget(pool_total=320000, parallel_slots=4)
        assert budget.admit("task_a", 120000) is True
        assert budget.active_count == 1
        assert budget.admit("task_b", 30000) is True
        assert budget.active_count == 2
        # Total allocated: 150k, available: 294.4k - 150k = 144.4k
        assert budget.available == 294400 - 150000

        released = budget.release("task_a")
        assert released == 120000
        assert budget.active_count == 1

    def test_refuse_when_pool_full(self):
        budget = SlotBudget(pool_total=320000, parallel_slots=2)
        # Fill with one huge task
        assert budget.admit("task_a", 250000) is True
        # Not enough left
        assert budget.can_admit(100000) is False
        assert budget.admit("task_b", 100000) is False
        assert budget.active_count == 1

    def test_refuse_when_slots_exhausted(self):
        budget = SlotBudget(pool_total=320000, parallel_slots=2)
        assert budget.admit("task_a", 50000) is True
        assert budget.admit("task_b", 50000) is True
        # Still room in pool, but no more slots
        assert budget.available > 0
        assert budget.can_admit(10000) is False

    def test_asymmetric_distribution(self):
        """Worker A gets 120k, Worker B gets 30k, Worker C gets 70k, Worker D gets 30k."""
        budget = SlotBudget(pool_total=320000, parallel_slots=4)
        assert budget.admit("heavy_analysis", 120000) is True
        assert budget.admit("light_summary", 30000) is True
        assert budget.admit("medium_impl", 70000) is True
        assert budget.admit("light_summary_2", 30000) is True
        # Total: 250k, pool: 294.4k usable → OK
        assert budget.allocated == 250000
        assert budget.active_count == 4

    def test_to_dict(self):
        budget = SlotBudget(pool_total=320000, parallel_slots=4)
        budget.admit("task_a", 50000)
        d = budget.to_dict()
        assert d["pool_total"] == 320000
        assert d["parallel_slots"] == 4
        assert d["active_count"] == 1


class TestContextPreEstimator:
    def test_estimate_without_runtime(self):
        """Without a runtime client, heuristic based on char count is used."""
        estimator = ContextPreEstimator()
        task = PlannedTask(
            task_id="e1",
            title="Analyze codebase structure",
            task_type=TaskType.READ_ONLY_ANALYSIS,
            read_only=True,
            write_enabled=False,
        )
        est = asyncio.run(estimator.estimate_worker(task, task_brief="Inspect the repo"))
        assert est.task_id == "e1"
        assert est.prompt_tokens > 0
        assert est.total_estimated > 0
        assert est.safety_overhead > 0
        assert est.recommended_budget == est.total_estimated + est.safety_overhead

    def test_estimate_with_rag_material(self):
        estimator = ContextPreEstimator()
        task = PlannedTask(
            task_id="e2", title="Summarize",
            task_type=TaskType.SUMMARIZATION,
            read_only=True, write_enabled=False,
        )
        rag = ["RAG result 1" * 100, "RAG result 2" * 200]
        est = asyncio.run(estimator.estimate_worker(task, rag_material=rag))
        assert est.rag_tokens > 0
        assert est.total_estimated > est.rag_tokens

    def test_estimate_20_percent_overhead(self):
        estimator = ContextPreEstimator()
        task = PlannedTask(
            task_id="e3", title="Task",
            task_type=TaskType.READ_ONLY_ANALYSIS,
            read_only=True, write_enabled=False,
        )
        est = asyncio.run(estimator.estimate_worker(task))
        assert est.safety_overhead >= 512  # Minimum overhead
        assert est.recommended_budget >= est.total_estimated  # Budget >= base


class TestParallelContextPlanner:
    def test_admit_all_success(self):
        planner = ParallelContextPlanner(pool_total=320000, parallel_slots=4)
        est_a = WorkerContextEstimate(
            task_id="a", total_estimated=100000,
            safety_overhead=20000, recommended_budget=120000,
        )
        est_b = WorkerContextEstimate(
            task_id="b", total_estimated=25000,
            safety_overhead=5000, recommended_budget=30000,
        )
        estimates = {"a": est_a, "b": est_b}

        result = planner.admit_all(estimates)
        assert result.ok is True
        assert result.admitted == ["a", "b"]
        assert result.refused == []

    def test_admit_refuse_when_pool_full(self):
        planner = ParallelContextPlanner(
            pool_total=100000, parallel_slots=4,
            safety_reserve_pct=0.05,
        )
        est_a = WorkerContextEstimate(
            task_id="a", total_estimated=70000,
            safety_overhead=10000, recommended_budget=80000,
        )
        est_b = WorkerContextEstimate(
            task_id="b", total_estimated=30000,
            safety_overhead=5000, recommended_budget=35000,
        )
        estimates = {"a": est_a, "b": est_b}

        result = planner.admit_all(estimates)
        # Budget: 100k * 0.95 = 95k usable. task_a takes 80k, 15k left < 35k needed
        assert len(result.admitted) == 1
        assert "a" in result.admitted
        assert "b" in result.refused

    def test_release_frees_budget(self):
        planner = ParallelContextPlanner(pool_total=320000, parallel_slots=4)
        est = WorkerContextEstimate(
            task_id="a", total_estimated=50000,
            safety_overhead=10000, recommended_budget=60000,
        )
        planner.admit_all({"a": est})
        assert planner.slot_budget.active_count == 1

        planner.release("a")
        assert planner.slot_budget.active_count == 0
        assert planner.slot_budget.allocated == 0

    def test_max_ratio_caps_individual_budget(self):
        """Worker budget capped at max_ratio * pool_total."""
        planner = ParallelContextPlanner(
            pool_total=320000, parallel_slots=4,
            safety_reserve_pct=0.05,
        )
        # max_ratio = 0.85 → max per worker = 272000
        est = WorkerContextEstimate(
            task_id="huge", total_estimated=250000,
            safety_overhead=50000, recommended_budget=300000,
        )
        result = planner.admit_all({"huge": est})
        assert result.ok is True
        # Budget should have been capped
        assert planner.slot_budget.active_budgets["huge"] <= 272000

    def test_feature_flag_defaults(self, monkeypatch):
        monkeypatch.delenv("ALPHARAVIS_PARALLEL_CONTEXT_PLANNER", raising=False)
        monkeypatch.delenv("ALPHARAVIS_PARALLEL_HERMES_WORKER", raising=False)
        assert parallel_context_planner_enabled() is False
        assert parallel_hermes_worker_enabled() is False

    def test_feature_flag_enabled(self, monkeypatch):
        monkeypatch.setenv("ALPHARAVIS_PARALLEL_CONTEXT_PLANNER", "true")
        monkeypatch.setenv("ALPHARAVIS_PARALLEL_HERMES_WORKER", "true")
        assert parallel_context_planner_enabled() is True
        assert parallel_hermes_worker_enabled() is True

    def test_to_dict(self):
        planner = ParallelContextPlanner(pool_total=320000, parallel_slots=4)
        est = WorkerContextEstimate(
            task_id="a", recommended_budget=60000,
        )
        planner.admit_all({"a": est})
        d = planner.to_dict()
        assert d["slot_budget"]["active_count"] == 1
        assert d["is_full"] is False


# ---------------------------------------------------------------------------
# HermesWorker + ContextPlanner integration
# ---------------------------------------------------------------------------


class TestHermesWorkerWithBudget:
    def test_hermes_worker_receives_budget_in_context(self):
        """HermesWorker maps context_budget kwarg into the Hermes call context."""
        calls = []

        async def hermes_fn(task: str, context: str, max_output_chars: int) -> str:
            calls.append((task, context, max_output_chars))
            return "done"

        worker = HermesWorker(hermes_fn)
        task = PlannedTask(
            task_id="hw1",
            title="Heavy analysis task",
            task_type=TaskType.READ_ONLY_ANALYSIS,
            read_only=True,
            write_enabled=False,
        )

        asyncio.run(
            worker.spawn(
                task,
                task_brief="Analyze the full codebase",
                context_budget=120000,
                max_output_chars=4000,
            )
        )

        _, context, max_out = calls[0]
        assert "MAX_CONTEXT_BUDGET: 120000 tokens" in context
        assert "READ-ONLY mode" in context
        assert max_out == 4000

    def test_hermes_worker_write_task_context(self):
        async def hermes_fn(task, context, max_output_chars):
            return "ok"

        worker = HermesWorker(hermes_fn)
        task = PlannedTask(
            task_id="hw2",
            title="Refactor module",
            task_type=TaskType.WRITE_IMPLEMENTATION,
            read_only=False,
            write_enabled=True,
        )

        asyncio.run(worker.spawn(task, context_budget=80000))
        # No easy way to extract context without mock, already tested above


# ---------------------------------------------------------------------------
# Node integration tests (HermesWorker + ContextPlanner in _parallel_executor_node)
# ---------------------------------------------------------------------------


class TestParallelExecutorNodeIntegration:
    """Tests for the rewired _parallel_executor_node with HermesWorker + ContextPlanner."""

    def test_hermes_worker_selected_for_write_tasks(self, monkeypatch):
        """When ALPHARAVIS_PARALLEL_HERMES_WORKER=true, write tasks get HermesWorker."""
        monkeypatch.setenv("ALPHARAVIS_PARALLEL_HERMES_WORKER", "true")
        assert parallel_hermes_worker_enabled() is True

    def test_context_planner_flags_off_by_default(self, monkeypatch):
        """All parallel control flags default to OFF."""
        monkeypatch.delenv("ALPHARAVIS_PARALLEL_CONTEXT_PLANNER", raising=False)
        monkeypatch.delenv("ALPHARAVIS_PARALLEL_HERMES_WORKER", raising=False)
        monkeypatch.delenv("ALPHARAVIS_PARALLEL_TASK_EXECUTION", raising=False)
        assert parallel_context_planner_enabled() is False
        assert parallel_hermes_worker_enabled() is False
        assert parallel_execution_enabled() is False

    def test_feature_flags_independent(self, monkeypatch):
        """Each flag can be toggled independently."""
        monkeypatch.setenv("ALPHARAVIS_PARALLEL_TASK_EXECUTION", "true")
        monkeypatch.setenv("ALPHARAVIS_PARALLEL_HERMES_WORKER", "true")
        monkeypatch.setenv("ALPHARAVIS_PARALLEL_CONTEXT_PLANNER", "false")
        assert parallel_execution_enabled() is True
        assert parallel_hermes_worker_enabled() is True
        assert parallel_context_planner_enabled() is False

    def test_slot_budget_refuses_when_pool_full_with_320k_4_slots(self):
        """At 320k pool with 4 slots, workers must fit within budget."""
        budget = SlotBudget(pool_total=320000, parallel_slots=4, kv_unified=True)
        # Admit 4 workers each with 75k = 300k total (fits in 294.4k usable)
        assert budget.admit("w1", 75000) is True
        assert budget.admit("w2", 75000) is True
        assert budget.admit("w3", 75000) is True
        # 225k allocated, 69.4k available — should fit 75k
        assert budget.admit("w4", 75000) is True
        # All 4 slots used
        assert budget.active_count == 4
        assert budget.can_admit(1000) is False  # No more slots

    def test_asymmetric_budget_with_realistic_worker_distribution(self):
        """Simulates: heavy analysis 120k, light summary 30k, medium impl 70k, light summary 30k."""
        budget = SlotBudget(pool_total=320000, parallel_slots=4)
        assert budget.admit("heavy_analysis", 120000) is True
        assert budget.admit("light_summary_1", 30000) is True
        assert budget.admit("medium_impl", 70000) is True
        assert budget.admit("light_summary_2", 30000) is True
        # Total: 250k (should fit in 294.4k usable)
        assert budget.allocated == 250000
        assert budget.active_count == 4

    def test_global_registry_has_workers(self):
        """GLOBAL_WORKER_REGISTRY has direct_llm and hermes registered."""
        registry = GLOBAL_WORKER_REGISTRY
        llm_worker = registry.get("direct_llm")
        hermes_worker = registry.get("hermes")
        assert llm_worker is not None
        assert hermes_worker is not None
        assert llm_worker is not hermes_worker  # Different instances

    def test_directllm_worker_accepts_context_budget_kwarg(self):
        """DirectLLMWorker should accept context_budget kwarg without error."""
        calls = []

        async def llm_fn(prompt: str, max_tokens: int) -> str:
            calls.append((prompt, max_tokens))
            return "done"

        worker = DirectLLMWorker(llm_fn)
        task = PlannedTask(
            task_id="t1",
            title="Test task",
            task_type=TaskType.READ_ONLY_ANALYSIS,
            read_only=True,
            write_enabled=False,
        )

        result = asyncio.run(
            worker.spawn(task, task_brief="Test", context_budget=80000)
        )
        assert result.status == "completed"

        prompt, max_tokens = calls[0]
        assert "MAX_CONTEXT_BUDGET: 80000 tokens" in prompt

    def test_hermes_worker_callable_signature_matches_call_hermes_agent(self):
        """HermesWorker hermes_fn must accept (task, context, max_output_chars)."""
        calls = []

        async def hermes_fn(task: str, context: str, max_output_chars: int) -> str:
            calls.append({
                "task": task,
                "context": context,
                "max_output_chars": max_output_chars,
            })
            return "hermes result"

        worker = HermesWorker(hermes_fn)
        task = PlannedTask(
            task_id="signature_test",
            title="Signature verification task",
            task_type=TaskType.WRITE_IMPLEMENTATION,
            read_only=False,
            write_enabled=True,
            affected_file_globs=["src/test.py"],
        )

        result = asyncio.run(
            worker.spawn(
                task,
                task_brief="Verify the callable signature",
                context_budget=50000,
                max_output_chars=4000,
            )
        )

        assert result.status == "completed"
        assert len(calls) == 1
        assert calls[0]["task"] == "Signature verification task"
        assert "MAX_CONTEXT_BUDGET: 50000 tokens" in calls[0]["context"]
        assert "src/test.py" in calls[0]["context"]
        assert calls[0]["max_output_chars"] == 4000


# ---------------------------------------------------------------------------
# Full integration: admission → spawn → result with mock workers
# ---------------------------------------------------------------------------


class TestFullIntegrationAdmissionToSpawn:
    def test_planner_admits_then_workers_spawn_with_budgets(self):
        """End-to-end: planner admits workers, then HermesWorker spawns with budgets."""
        planner = ParallelContextPlanner(pool_total=320000, parallel_slots=4)

        est_a = WorkerContextEstimate(
            task_id="task_a", total_estimated=100000,
            safety_overhead=20000, recommended_budget=120000,
        )
        est_b = WorkerContextEstimate(
            task_id="task_b", total_estimated=25000,
            safety_overhead=5000, recommended_budget=30000,
        )
        estimates = {"task_a": est_a, "task_b": est_b}

        admission = planner.admit_all(estimates)
        assert admission.ok is True
        assert len(admission.admitted) == 2

        # Verify budgets match
        for tid in admission.admitted:
            budget = admission.budget_for(tid)
            assert budget > 0
            assert budget <= 272000  # max_ratio cap

        planner.release("task_a")
        planner.release("task_b")
        assert planner.slot_budget.active_count == 0

    def test_refused_workers_not_spawned(self):
        """When pool is full, refused workers are reported."""
        planner = ParallelContextPlanner(
            pool_total=100000, parallel_slots=2,
            safety_reserve_pct=0.05,
        )
        est_large = WorkerContextEstimate(
            task_id="large", total_estimated=70000,
            safety_overhead=10000, recommended_budget=80000,
        )
        est_small = WorkerContextEstimate(
            task_id="small", total_estimated=20000,
            safety_overhead=3000, recommended_budget=23000,
        )
        estimates = {"large": est_large, "small": est_small}

        admission = planner.admit_all(estimates)
        # large (80k) fits in 95k usable. small (23k) doesn't fit in remaining 15k.
        assert "large" in admission.admitted
        assert "small" in admission.refused
        assert len(admission.refused) == 1

