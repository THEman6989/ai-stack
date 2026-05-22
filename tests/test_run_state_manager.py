from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

import run_state_manager  # noqa: E402


class _FakeCollection:
    def __init__(self) -> None:
        self.records: dict[str, dict] = {}
        self.replacements: list[tuple[dict, dict, bool]] = []

    def replace_one(self, query: dict, record: dict, upsert: bool = False) -> None:
        self.replacements.append((query, record, upsert))
        self.records[str(query["_id"])] = dict(record)

    def find_one(self, query: dict):
        record = self.records.get(str(query["_id"]))
        return dict(record) if record else None


def test_save_run_checkpoint_replaces_current_atomically(monkeypatch) -> None:
    collection = _FakeCollection()
    monkeypatch.setattr(run_state_manager, "_collection", lambda: collection)

    first = run_state_manager.save_run_checkpoint(
        thread_id="thread-1",
        thread_key="chat-1",
        phase="planner",
        status="running",
        state={"current_task_brief": "brief", "planner_context": "plan"},
    )
    second = run_state_manager.save_run_checkpoint(
        thread_id="thread-1",
        thread_key="chat-1",
        phase="alpha_ravis_swarm",
        status="failed",
        state={"current_task_brief": "brief", "planner_context": "plan"},
        error="connection reset",
        error_classification={"reason": "timeout_or_network"},
    )

    assert first["saved"] is True
    assert second["saved"] is True
    assert len(collection.records) == 1
    record = collection.records["thread-1:current"]
    assert record["phase"] == "alpha_ravis_swarm"
    assert record["status"] == "failed"
    assert record["current_task_brief"] == "brief"
    assert record["planner_context"] == "plan"
    assert record["error"] == "connection reset"
    assert collection.replacements[-1][2] is True


def test_resume_updates_skip_completed_checkpoint() -> None:
    assert run_state_manager.resume_updates_from_checkpoint({"status": "completed"}) == {}


def test_resume_updates_restore_plan_fields() -> None:
    updates = run_state_manager.resume_updates_from_checkpoint(
        {
            "thread_id": "thread-1",
            "phase": "planner",
            "status": "failed",
            "current_task_brief": "brief",
            "planner_context": "plan",
            "planner_last_key": "abc",
            "selected_toolsets": ["repo"],
            "error": "server disconnected",
        }
    )

    assert updates["current_task_brief"] == "brief"
    assert updates["planner_context"] == "plan"
    assert updates["planner_last_key"] == "abc"
    assert updates["selected_toolsets"] == ["repo"]
    assert updates["run_resume_checkpoint"]["phase"] == "planner"
    assert updates["run_resume_checkpoint"]["status"] == "failed"


def test_atomic_write_json_replaces_complete_file(tmp_path: Path) -> None:
    target = tmp_path / "state.json"

    run_state_manager.atomic_write_json(target, {"status": "running"})
    run_state_manager.atomic_write_json(target, {"status": "completed", "phase": "finish"})

    assert target.read_text(encoding="utf-8").strip().endswith("}")
    assert '"status": "completed"' in target.read_text(encoding="utf-8")
    assert not (tmp_path / ".state.json.tmp").exists()
