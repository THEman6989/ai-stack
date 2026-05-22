from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

import run_review_manager  # noqa: E402


def test_run_review_manager_saves_loads_and_delivers_pending_review(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "reviews.json"
    monkeypatch.setenv("ALPHARAVIS_ASYNC_REVIEW_STORE_PATH", str(path))

    record = run_review_manager.save_run_review(
        "thread-1",
        thread_key="conv-1",
        task_brief="Do the thing",
        review_text="Missing verification.",
    )

    assert record["status"] == "pending"
    pending = run_review_manager.load_pending_run_review("thread-1")
    assert pending is not None
    assert pending["review_text"] == "Missing verification."

    run_review_manager.mark_run_review_delivered("thread-1")

    assert run_review_manager.load_pending_run_review("thread-1") is None
