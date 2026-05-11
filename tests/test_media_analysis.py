from __future__ import annotations

import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

import media_analysis  # noqa: E402
import vector_memory  # noqa: E402


def test_decide_media_mode_defaults_to_register_only() -> None:
    assert media_analysis.decide_media_mode("nutze dieses Video spaeter", "auto") == "register_only"


def test_decide_media_mode_passes_pixelle_inputs_through() -> None:
    assert media_analysis.decide_media_mode("mach daraus mit Pixelle ein neues Video", "auto") == "pass_through"


def test_decide_media_mode_detects_explicit_analysis() -> None:
    assert media_analysis.decide_media_mode("analysiere dieses Video", "auto") == "analyze"


def test_sampling_plan_caps_long_videos() -> None:
    card = {"preferred_video_fps": 1, "max_video_fps": 1, "max_frames": 100}
    plan = media_analysis._sampling_plan(3600, card)
    assert plan["max_frames"] == 100
    assert plan["estimated_frames"] <= 100
    assert plan["fps"] < 1


def test_sampling_plan_keeps_short_videos_near_one_fps() -> None:
    card = {"preferred_video_fps": 1, "max_video_fps": 1, "max_frames": 100}
    plan = media_analysis._sampling_plan(10, card)
    assert plan["fps"] == 1
    assert plan["estimated_frames"] == 10


def test_resolve_model_card_uses_big_boss_alias() -> None:
    card = media_analysis.resolve_model_card("big-boss")
    assert card["supports_video"] is True
    assert card["native_context_tokens"] == 262144


def test_prepare_media_register_only_does_not_download() -> None:
    result = asyncio.run(
        media_analysis.prepare_media_for_model(
            media_url="https://example.test/source.mp4",
            user_goal="nutze das spaeter",
            mode="auto",
        )
    )
    assert result["ok"] is True
    assert result["mode"] == "register_only"
    assert result["downloaded"] is False
    assert result["decision"] == "metadata_only"


def test_media_chunking_hash_changes_with_frame_cap(monkeypatch) -> None:
    monkeypatch.setenv("ALPHARAVIS_VIDEO_ANALYSIS_MAX_FRAMES", "100")
    first = vector_memory._media_chunking_config_hash()
    monkeypatch.setenv("ALPHARAVIS_VIDEO_ANALYSIS_MAX_FRAMES", "128")
    second = vector_memory._media_chunking_config_hash()
    assert first != second


def test_media_model_card_prefers_media_specific_env(monkeypatch) -> None:
    monkeypatch.setenv("ALPHARAVIS_MEDIA_VISION_EMBEDDING_MODEL_CARD", "qwen3vl-video-embed-v1")
    assert vector_memory._media_model_card_id("") == "qwen3vl-video-embed-v1"
