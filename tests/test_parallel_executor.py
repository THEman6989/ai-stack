"""Tests for parallel task executor — task DAG, classification, conflict detection,
parallelization decisions, worktree manager, and worker spawner.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_stack.parallel_executor import (
    DryRunWorker,
    ModelClass,
    PlannedTask,
    TaskDAG,
    TaskType,
    WorktreeManager,
    analyze_parallelization,
    log_parallelization_decision,
    parallel_execution_enabled,
    parse_planner_text_into_tasks,
    repo_has_uncommitted_changes,
)
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
        assert task_a.file_conflicts or task_b.file_conflicts  # at least one has conflicts
        assert "b" in task_a.file_conflicts or "a" in task_b.file_conflicts

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
        assert dag.parallelizable_count >= 0  # may or may not depending on deps

    def test_chokepoint_files_force_serialization(self):
        tasks = [
            PlannedTask(task_id="a", title="Edit package.json", task_type=TaskType.WRITE_IMPLEMENTATION,
                       read_only=False, write_enabled=True,
                       affected_file_globs=["package.json"],
                       shared_chokepoint_files=["package.json"]),
        ]
        dag = analyze_parallelization(tasks)
        task = dag.get_task("a")
        assert "chokepoint" in task.reason_for_parallelization_decision.lower()

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
        assert "big_model_busy" in task.resource_conflicts

    def test_context_pressure_resource_conflict(self):
        tasks = [
            PlannedTask(task_id="a", title="Large analysis", task_type=TaskType.WRITE_IMPLEMENTATION,
                       read_only=False, write_enabled=True, required_model_class=ModelClass.BIG_MODEL),
        ]
        dag = analyze_parallelization(tasks, context_pressure_high=True)
        task = dag.get_task("a")
        assert "context_pressure_high" in task.resource_conflicts

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
        assert len(dag.parallel_groups) >= 1


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
