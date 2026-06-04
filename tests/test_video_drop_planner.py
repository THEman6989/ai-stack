from __future__ import annotations

import json
import socket
import sys
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

import video_drop_planner  # noqa: E402


def _write_pcm16_wav(path: Path, samples: list[float], sample_rate: int = 22050) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        payload = bytearray()
        for sample in samples:
            value = max(-1.0, min(1.0, sample))
            payload.extend(int(value * 32767).to_bytes(2, byteorder="little", signed=True))
        wf.writeframes(bytes(payload))


def test_detect_audio_drop_candidates_finds_rms_jump(tmp_path: Path) -> None:
    sample_rate = 22050
    samples: list[float] = []
    # Low-energy intro, then a clear high-energy beat/drop region.
    samples.extend([0.015] * sample_rate)
    samples.extend([0.7 if index % 2 == 0 else -0.7 for index in range(sample_rate // 2)])
    samples.extend([0.08] * (sample_rate // 2))
    wav_path = tmp_path / "drop.wav"
    _write_pcm16_wav(wav_path, samples, sample_rate=sample_rate)

    candidates = video_drop_planner.detect_audio_drop_candidates(wav_path, fps=30, max_candidates=3)

    assert candidates
    assert abs(candidates[0].time_seconds - 1.0) < 0.2
    assert candidates[0].frame_index in range(24, 37)
    assert "rms jump" in candidates[0].reason


def test_merge_drop_candidates_keeps_strongest_inside_window() -> None:
    merged = video_drop_planner.merge_drop_candidates(
        [
            video_drop_planner.DropCandidate(1.0, 30, 0.4, "weak"),
            video_drop_planner.DropCandidate(1.2, 36, 0.9, "strong"),
            video_drop_planner.DropCandidate(3.0, 90, 0.6, "later"),
        ],
        merge_seconds=0.5,
    )

    assert [candidate.reason for candidate in merged] == ["strong", "later"]


def test_detect_visual_transition_suggests_black_frame_reset() -> None:
    scores = [
        video_drop_planner.FrameChangeScore(248, 8.267, "a.jpg", 0.01),
        video_drop_planner.FrameChangeScore(249, 8.300, "b.jpg", 0.02),
        video_drop_planner.FrameChangeScore(250, 8.333, "black.jpg", 0.5, is_black=True),
        video_drop_planner.FrameChangeScore(251, 8.367, "new.jpg", 0.72),
        video_drop_planner.FrameChangeScore(252, 8.400, "new2.jpg", 0.03),
    ]

    transition = video_drop_planner.detect_visual_transition(scores, beat_frame=250, fps=30, black_frame_ms=60)

    assert transition["visual_change_frame"] == 251
    assert transition["first_new_outfit_frame"] == 251
    assert transition["black_frame_count"] == 2
    assert transition["black_frame_start"] == 249
    assert transition["insert_black_frame"] is True


def test_detect_visual_transition_skips_black_frame_when_it_has_largest_diff() -> None:
    scores = [
        video_drop_planner.FrameChangeScore(248, 8.267, "old-a.jpg", 0.01),
        video_drop_planner.FrameChangeScore(249, 8.300, "old-b.jpg", 0.02),
        video_drop_planner.FrameChangeScore(250, 8.333, "black.jpg", 0.95, is_black=True),
        video_drop_planner.FrameChangeScore(251, 8.367, "new.jpg", 0.20),
        video_drop_planner.FrameChangeScore(252, 8.400, "new2.jpg", 0.03),
    ]

    transition = video_drop_planner.detect_visual_transition(scores, beat_frame=250, fps=30, black_frame_ms=60)

    assert transition["visual_change_frame"] == 251
    assert transition["first_new_outfit_frame"] == 251
    assert transition["last_old_outfit_frame"] == 249


def test_frame_change_scores_map_candidate_fps_to_source_frame_numbers(tmp_path: Path) -> None:
    frames = []
    for index in range(3):
        frame = tmp_path / f"frame-{index}.jpg"
        frame.write_bytes(bytes([index + 1]) * 128)
        frames.append(frame)

    scores = video_drop_planner.compute_frame_change_scores(
        frames,
        window_start_seconds=1.0,
        extraction_fps=10.0,
        source_fps=30.0,
    )

    assert [score.time_seconds for score in scores] == [1.0, 1.1, 1.2]
    assert [score.frame_index for score in scores] == [30, 33, 36]


def test_resolve_video_input_rejects_local_paths_by_default(tmp_path: Path) -> None:
    local_video = tmp_path / "source.mp4"
    local_video.write_bytes(b"fake")

    try:
        video_drop_planner.resolve_video_input(str(local_video), tmp_path / "run")
    except ValueError as exc:
        assert "Local video paths are disabled" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("local paths must be disabled by default")


def test_resolve_video_input_allows_local_paths_under_configured_root(monkeypatch, tmp_path: Path) -> None:
    local_video = tmp_path / "source.mp4"
    local_video.write_bytes(b"fake")
    monkeypatch.setenv("ALPHARAVIS_VIDEO_DROP_PLANNER_ALLOW_LOCAL_PATHS", "true")
    monkeypatch.setenv("ALPHARAVIS_VIDEO_DROP_PLANNER_MEDIA_ROOT", str(tmp_path))

    assert video_drop_planner.resolve_video_input(str(local_video), tmp_path / "run") == local_video.resolve()


def test_remote_url_validation_blocks_private_hosts_by_default(monkeypatch) -> None:
    monkeypatch.setattr(
        video_drop_planner.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))],
    )

    try:
        video_drop_planner.resolve_video_input("http://example.test/source.mp4", Path("/tmp/run"))
    except ValueError as exc:
        assert "Private, loopback" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("private hosts must be blocked by default")


def test_build_drop_plan_maps_outfits_and_windows() -> None:
    probe = video_drop_planner.VideoProbe(duration_seconds=12.0, fps=30.0, width=720, height=1280, has_audio=True)
    candidate = video_drop_planner.DropCandidate(8.417, 253, 0.91, "strong onset")
    transition = {
        253: {
            "visual_change_frame": 254,
            "last_old_outfit_frame": 251,
            "first_new_outfit_frame": 254,
            "insert_black_frame": True,
            "black_frame_start": 252,
            "black_frame_count": 2,
            "visual_confidence": 0.92,
            "reason": "max diff",
        }
    }

    plan = video_drop_planner.build_drop_plan(
        source_video="/tmp/source.mp4",
        probe=probe,
        outfit_images=[{"id": None, "url": "media://red.png"}],
        workflow_name="outfit_change_beatdrop",
        audio_candidates=[candidate],
        visual_transitions=transition,
    )

    assert plan["plan_type"] == "video_outfit_drop_plan"
    assert plan["drop_count"] == 1
    drop = plan["drops"][0]
    assert drop["selected_outfit_image"] == "outfit_1"
    assert drop["window_start_frame"] == 223
    assert drop["window_end_frame"] == 283
    assert drop["first_new_outfit_frame"] == 254


def test_plan_from_file_can_create_manifest_without_frame_extraction(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake video")
    wav = tmp_path / "audio.wav"
    _write_pcm16_wav(wav, [0.0] * 100)

    monkeypatch.setattr(
        video_drop_planner,
        "probe_video",
        lambda path: video_drop_planner.VideoProbe(duration_seconds=5.0, fps=25.0, width=640, height=360, has_audio=True),
    )
    monkeypatch.setattr(video_drop_planner, "extract_audio_to_wav", lambda video_path, out_dir: wav)
    monkeypatch.setattr(
        video_drop_planner,
        "detect_audio_drop_candidates",
        lambda wav_path, fps, max_candidates: [video_drop_planner.DropCandidate(2.0, 50, 0.8, "test")],
    )

    plan = video_drop_planner.plan_video_outfit_drops_from_file(
        source,
        output_dir=tmp_path / "plan",
        outfit_images=["media://outfit.png"],
        extract_frames=False,
    )

    assert plan["ok"] is True
    assert plan["drops"][0]["beat_frame"] == 50
    assert Path(plan["plan_path"]).exists()
    saved = json.loads(Path(plan["plan_path"]).read_text())
    assert saved["plan_path"] == plan["plan_path"]
