from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

from maintenance_helpers import extract_review_insight_candidates, generate_thread_title  # noqa: E402


def test_generate_thread_title_is_short_and_deterministic() -> None:
    title = generate_thread_title("User: Please debug Hermes provider routing and smoke tests today.", max_words=6)

    assert title == "Please debug Hermes provider routing smoke"


def test_extract_review_insight_candidates_is_review_only() -> None:
    candidates = extract_review_insight_candidates(
        "I prefer local llama.cpp defaults for daily use. Remember that approvals should stay per thread.",
    )

    assert len(candidates) == 2
    assert candidates[0]["review_required"] is True
    assert candidates[0]["kind"] == "user_preference"
    assert candidates[1]["kind"] == "explicit_memory_request"


if __name__ == "__main__":
    test_generate_thread_title_is_short_and_deterministic()
    test_extract_review_insight_candidates_is_review_only()
