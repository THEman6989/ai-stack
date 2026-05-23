"""Structured task DAG for parallel execution.

Parses the planner's text output into a structured task graph,
classifies tasks, detects file conflicts, and decides which tasks
can run in parallel.

Key design:
- The planner still produces text bullets (backward compat).
- This module optionally parses those into a structured DAG when
  ALPHARAVIS_PARALLEL_TASK_EXECUTION=true.
- When false, the planner text flows through unchanged.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums and constants
# ---------------------------------------------------------------------------


class TaskType(str, Enum):
    READ_ONLY_ANALYSIS = "read_only_analysis"
    WRITE_IMPLEMENTATION = "write_implementation"
    TEST = "test"
    INTEGRATION_REVIEW = "integration_review"
    MERGE_REVIEW = "merge_review"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"


class ModelClass(str, Enum):
    SMALL_MODEL = "small_model"
    BIG_MODEL = "big_model"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Files that should serialize tasks or require merge review when touched
CHOKEPOINT_FILE_PATTERNS: tuple[str, ...] = (
    "package.json",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "docker-compose.yml",
    "docker-compose.yaml",
    "Dockerfile",
    ".env",
    ".env.example",
    ".env(exaple)",
    "README.md",
    "pyproject.toml",
    "Cargo.toml",
    "go.mod",
    "Gemfile",
    "**/migrations/**",
    "**/schemas.py",
    "**/types.py",
    "**/client.py",
    "**/api_client.py",
)


# ---------------------------------------------------------------------------
# Structured task representation
# ---------------------------------------------------------------------------


@dataclass
class PlannedTask:
    """One task inside a parallel execution plan."""

    task_id: str
    title: str
    task_type: TaskType
    read_only: bool
    write_enabled: bool
    affected_file_globs: list[str] = field(default_factory=list)
    forbidden_file_globs: list[str] = field(default_factory=list)
    shared_chokepoint_files: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    can_parallelize: bool = False
    parallel_group_id: str = ""
    required_model_class: ModelClass = ModelClass.BIG_MODEL
    risk_level: RiskLevel = RiskLevel.MEDIUM
    reason_for_parallelization_decision: str = ""

    # Computed during graph analysis
    blocking_dependencies: list[str] = field(default_factory=list)
    file_conflicts: list[str] = field(default_factory=list)
    resource_conflicts: list[str] = field(default_factory=list)
    selected_model: str = ""
    route_decision: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "task_type": self.task_type.value,
            "read_only": self.read_only,
            "write_enabled": self.write_enabled,
            "affected_file_globs": self.affected_file_globs,
            "forbidden_file_globs": self.forbidden_file_globs,
            "shared_chokepoint_files": self.shared_chokepoint_files,
            "dependencies": self.dependencies,
            "can_parallelize": self.can_parallelize,
            "parallel_group_id": self.parallel_group_id,
            "required_model_class": self.required_model_class.value,
            "risk_level": self.risk_level.value,
            "reason_for_parallelization_decision": self.reason_for_parallelization_decision,
            "blocking_dependencies": self.blocking_dependencies,
            "file_conflicts": self.file_conflicts,
            "resource_conflicts": self.resource_conflicts,
            "selected_model": self.selected_model,
            "route_decision": self.route_decision,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlannedTask":
        return cls(
            task_id=data.get("task_id", ""),
            title=data.get("title", ""),
            task_type=TaskType(data.get("task_type", "read_only_analysis")),
            read_only=data.get("read_only", True),
            write_enabled=data.get("write_enabled", False),
            affected_file_globs=list(data.get("affected_file_globs", [])),
            forbidden_file_globs=list(data.get("forbidden_file_globs", [])),
            shared_chokepoint_files=list(data.get("shared_chokepoint_files", [])),
            dependencies=list(data.get("dependencies", [])),
            can_parallelize=data.get("can_parallelize", False),
            parallel_group_id=data.get("parallel_group_id", ""),
            required_model_class=ModelClass(data.get("required_model_class", "big_model")),
            risk_level=RiskLevel(data.get("risk_level", "medium")),
            reason_for_parallelization_decision=data.get("reason_for_parallelization_decision", ""),
            blocking_dependencies=list(data.get("blocking_dependencies", [])),
            file_conflicts=list(data.get("file_conflicts", [])),
            resource_conflicts=list(data.get("resource_conflicts", [])),
            selected_model=data.get("selected_model", ""),
            route_decision=data.get("route_decision", ""),
        )


@dataclass
class TaskDAG:
    """A directed acyclic graph of tasks for parallel execution."""

    tasks: list[PlannedTask] = field(default_factory=list)
    parallel_groups: dict[str, list[str]] = field(default_factory=dict)  # group_id -> [task_ids]
    serial_chain: list[str] = field(default_factory=list)  # tasks that must run sequentially
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def task_count(self) -> int:
        return len(self.tasks)

    @property
    def parallelizable_count(self) -> int:
        return sum(1 for t in self.tasks if t.can_parallelize)

    @property
    def serial_count(self) -> int:
        return self.task_count - self.parallelizable_count

    def get_task(self, task_id: str) -> PlannedTask | None:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None


# ---------------------------------------------------------------------------
# Planner text parser (extracts structured tasks from planner bullets)
# ---------------------------------------------------------------------------


def parse_planner_text_into_tasks(planner_text: str) -> list[PlannedTask]:
    """Convert planner bullet output into structured PlannedTask list.

    Heuristic: lines starting with '-' or '•' or numbers are tasks.
    Falls back gracefully — if parsing fails, returns empty list.
    The caller should fall back to the normal sequential swarm path.
    """
    if not planner_text or not planner_text.strip():
        return []

    lines = planner_text.strip().splitlines()
    tasks: list[PlannedTask] = []
    task_index = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Detect bullet / numbered task lines
        is_task_line = (
            stripped.startswith(("- ", "• ", "* ", "+ "))
            or bool(re.match(r"^\d+[\.\)]\s", stripped))
        )
        if not is_task_line:
            continue

        # Clean the title
        title = re.sub(r"^[-•*+\d\.\)\s]+", "", stripped).strip()
        if not title:
            continue

        task_index += 1
        task_id = f"task_{task_index:03d}"

        # Classify the task
        task_type, read_only, write_enabled, model_class = _classify_task_from_title(title)
        affected, forbidden, chokepoints = _extract_file_globs(title)

        tasks.append(
            PlannedTask(
                task_id=task_id,
                title=title,
                task_type=task_type,
                read_only=read_only,
                write_enabled=write_enabled,
                affected_file_globs=affected,
                forbidden_file_globs=forbidden,
                shared_chokepoint_files=chokepoints,
                required_model_class=model_class,
                risk_level=_assess_risk(title, task_type, write_enabled),
            )
        )

    # Assign dependencies: each task depends on the previous one in order by default
    # (parser doesn't know real dependencies; the graph analyzer can refine)
    for i in range(1, len(tasks)):
        if not tasks[i].dependencies:
            tasks[i].dependencies = [tasks[i - 1].task_id]

    return tasks


def _classify_task_from_title(title: str) -> tuple[TaskType, bool, bool, ModelClass]:
    """Heuristic classification from the task title text."""
    lowered = title.lower()

    # Write/implementation tasks
    write_triggers = [
        "implement", "build", "create", "write", "code", "patch", "fix",
        "refactor", "add", "change", "modify", "update", "edit", "generate",
        "docker", "deploy",
    ]
    if any(trigger in lowered for trigger in write_triggers):
        return TaskType.WRITE_IMPLEMENTATION, False, True, ModelClass.BIG_MODEL

    # Test tasks
    test_triggers = ["test", "verify", "validate", "check", "assert"]
    if any(trigger in lowered for trigger in test_triggers):
        return TaskType.TEST, False, True, ModelClass.BIG_MODEL

    # Summarization / analysis
    summary_triggers = ["summarize", "summary", "compress", "extract", "analyze", "analysis", "review"]
    if any(trigger in lowered for trigger in summary_triggers):
        return TaskType.SUMMARIZATION, True, False, ModelClass.SMALL_MODEL

    # Classification
    classify_triggers = ["classify", "route", "judge", "categorize", "detect"]
    if any(trigger in lowered for trigger in classify_triggers):
        return TaskType.CLASSIFICATION, True, False, ModelClass.SMALL_MODEL

    # Default: read-only analysis
    return TaskType.READ_ONLY_ANALYSIS, True, False, ModelClass.SMALL_MODEL


def _extract_file_globs(title: str) -> tuple[list[str], list[str], list[str]]:
    """Extract file globs mentioned in the task title.

    Returns (affected, forbidden, chokepoints).
    """
    affected: list[str] = []
    forbidden: list[str] = []
    chokepoints: list[str] = []

    # Find file paths in backticks or standalone
    path_pattern = re.findall(r"`([^`]+\.[a-zA-Z]+)`", title)
    paths = [p.strip() for p in path_pattern if p.strip()]

    for path in paths:
        basename = os.path.basename(path)
        affected.append(path)
        if _is_chokepoint(path):
            chokepoints.append(path)

    return affected, forbidden, chokepoints


def _is_chokepoint(path: str) -> bool:
    """Check if a file path matches any chokepoint pattern.

    Matches the full path first. Falls back to basename matching only
    for patterns without wildcards in the basename portion (avoids
    false positives like '**' matching everything).
    """
    import fnmatch
    for pattern in CHOKEPOINT_FILE_PATTERNS:
        if fnmatch.fnmatch(path, pattern):
            return True
        pattern_basename = os.path.basename(pattern)
        # Only do basename matching for exact filenames (no wildcards)
        if "*" not in pattern_basename and "?" not in pattern_basename and "[" not in pattern_basename:
            if fnmatch.fnmatch(os.path.basename(path), pattern_basename):
                return True
    return False


def _assess_risk(title: str, task_type: TaskType, write_enabled: bool) -> RiskLevel:
    lowered = title.lower()
    if any(word in lowered for word in ("delete", "remove", "drop", "destroy", "migration")):
        return RiskLevel.HIGH
    if write_enabled:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


# ---------------------------------------------------------------------------
# File conflict detection
# ---------------------------------------------------------------------------


def _globs_overlap(globs_a: list[str], globs_b: list[str]) -> bool:
    """Simple overlap check: any glob from A appears in B (exact match or shared basename)."""
    if not globs_a or not globs_b:
        return False
    set_a = set(globs_a)
    set_b = set(globs_b)
    if set_a & set_b:
        return True
    # Also check basename overlap
    basenames_a = {os.path.basename(g) for g in globs_a}
    basenames_b = {os.path.basename(g) for g in globs_b}
    return bool(basenames_a & basenames_b)


def detect_file_conflicts(tasks: list[PlannedTask]) -> dict[str, list[str]]:
    """Return {task_id: [conflicting_task_ids]}."""
    conflicts: dict[str, list[str]] = {}
    for i, task_a in enumerate(tasks):
        for task_b in tasks[i + 1 :]:
            if _globs_overlap(task_a.affected_file_globs, task_b.affected_file_globs):
                conflicts.setdefault(task_a.task_id, []).append(task_b.task_id)
                conflicts.setdefault(task_b.task_id, []).append(task_a.task_id)
    return conflicts


# ---------------------------------------------------------------------------
# Parallelization decision engine
# ---------------------------------------------------------------------------


def analyze_parallelization(
    tasks: list[PlannedTask],
    *,
    active_big_model_busy: bool = False,
    context_pressure_high: bool = False,
) -> TaskDAG:
    """Analyze tasks and decide which can run in parallel.

    Returns a TaskDAG with parallelization decisions, group assignments,
    and detailed reasoning for each task.
    """
    if not tasks:
        return TaskDAG()

    file_conflicts = detect_file_conflicts(tasks)
    dag = TaskDAG(tasks=tasks)
    group_counter = 0

    for task in tasks:
        reasons: list[str] = []
        blocking: list[str] = []
        task_file_conflicts: list[str] = []
        resource_conflicts: list[str] = []
        can_parallel = True

        # --- Rule 1: Dependencies must be serialized ---
        for dep_id in task.dependencies:
            dep_task = next((t for t in tasks if t.task_id == dep_id), None)
            if dep_task:
                blocking.append(dep_id)
                reasons.append(f"depends on {dep_id}")

        # --- Rule 2: Write tasks must not have overlapping file globs ---
        if task.write_enabled:
            conflicting = file_conflicts.get(task.task_id, [])
            if conflicting:
                task_file_conflicts = list(conflicting)
                reasons.append(f"file conflict with {', '.join(conflicting)}")

        # --- Rule 3: Chokepoint files force serialization ---
        if task.shared_chokepoint_files:
            reasons.append(f"chokepoint files: {', '.join(task.shared_chokepoint_files[:3])}")

        # --- Rule 4: Test tasks wait for implementation ---
        if task.task_type == TaskType.TEST:
            impl_tasks = [t.task_id for t in tasks if t.task_type == TaskType.WRITE_IMPLEMENTATION]
            if impl_tasks:
                blocking.extend(impl_tasks)
                reasons.append("tests must wait for implementation tasks")

        # --- Rule 5: Merge/review always serialized ---
        if task.task_type in {TaskType.MERGE_REVIEW, TaskType.INTEGRATION_REVIEW}:
            reasons.append("merge/review must be serialized")
            can_parallel = False

        # --- Rule 6: Resource constraints ---
        if task.required_model_class == ModelClass.BIG_MODEL and active_big_model_busy:
            resource_conflicts.append("big_model_busy")
            reasons.append("big model busy; consider queue/defer or small model")

        if task.required_model_class == ModelClass.BIG_MODEL and context_pressure_high:
            resource_conflicts.append("context_pressure_high")
            reasons.append("high context pressure; consider chunk/summarize first")

        # --- Rule 7: Read-only tasks can parallelize freely ---
        # Only if not already forced serial by an earlier rule.
        if can_parallel and task.read_only and not task.write_enabled and not task.shared_chokepoint_files:
            if not reasons:
                reasons.append("read-only task, no conflicts")
            can_parallel = True

        # --- Rule 8: Write tasks can parallelize if no file conflicts ---
        # Only if not already forced serial by an earlier rule.
        if can_parallel and task.write_enabled and not task_file_conflicts and not task.shared_chokepoint_files:
            can_parallel = True
            reasons.append("write task, no file conflicts")

        # --- Assign parallel group ---
        if can_parallel and not blocking:
            group_counter += 1
            task.parallel_group_id = f"group_{group_counter:02d}"
            reasons.append(f"assigned to {task.parallel_group_id}")
        elif blocking and can_parallel and not task_file_conflicts:
            # Can parallelize after dependencies complete
            task.can_parallelize = False  # must wait for deps first
            reasons.append("can parallelize after dependencies complete")
        else:
            reasons.append("serialized")

        # --- Finalize ---
        task.can_parallelize = can_parallel and not blocking
        task.blocking_dependencies = blocking
        task.file_conflicts = task_file_conflicts
        task.resource_conflicts = resource_conflicts
        task.reason_for_parallelization_decision = "; ".join(reasons) if reasons else "no decision"
        task.selected_model = (
            "big-boss" if task.required_model_class == ModelClass.BIG_MODEL else "small-2b"
        )
        task.route_decision = (
            "parallel" if task.can_parallelize else "serial"
        )

    # --- Build parallel groups ---
    for task in dag.tasks:
        if task.can_parallelize and task.parallel_group_id:
            dag.parallel_groups.setdefault(task.parallel_group_id, []).append(task.task_id)

    # --- Build serial chain ---
    dag.serial_chain = [t.task_id for t in dag.tasks if not t.can_parallelize]

    return dag


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------


def parallel_execution_enabled() -> bool:
    return os.getenv("ALPHARAVIS_PARALLEL_TASK_EXECUTION", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------


def log_parallelization_decision(task: PlannedTask) -> dict[str, Any]:
    """Produce an observability record for one task's parallelization decision."""
    return {
        "task_id": task.task_id,
        "title": task.title,
        "can_parallelize": task.can_parallelize,
        "parallel_group_id": task.parallel_group_id,
        "blocking_dependencies": task.blocking_dependencies,
        "file_conflicts": task.file_conflicts,
        "resource_conflicts": task.resource_conflicts,
        "selected_model": task.selected_model,
        "route_decision": task.route_decision,
        "reason": task.reason_for_parallelization_decision,
    }
