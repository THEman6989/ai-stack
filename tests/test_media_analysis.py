from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
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


def test_vision_embedding_model_url_overrides_litellm_base(monkeypatch) -> None:
    monkeypatch.setenv("ALPHARAVIS_VISION_EMBEDDING_MODEL_URL", "http://vision-box:8080/v1/")
    monkeypatch.setenv("ALPHARAVIS_VISION_EMBEDDING_BASE_URL", "http://litellm:4000/v1")
    monkeypatch.setenv("VISION_EMBEDDING_API_BASE", "http://backend:11434/v1")

    assert vector_memory._vision_embedding_base_url() == "http://vision-box:8080/v1"


def test_vision_embedding_base_url_falls_back_to_litellm_backend_env(monkeypatch) -> None:
    monkeypatch.delenv("ALPHARAVIS_VISION_EMBEDDING_MODEL_URL", raising=False)
    monkeypatch.delenv("ALPHARAVIS_VISION_EMBEDDING_BASE_URL", raising=False)
    monkeypatch.setenv("VISION_EMBEDDING_API_BASE", "http://vision-backend:11434/v1")

    assert vector_memory._vision_embedding_base_url() == "http://vision-backend:11434/v1"


def test_vector_chunk_profile_detects_code_and_chat(monkeypatch) -> None:
    monkeypatch.delenv("ALPHARAVIS_PGVECTOR_CHUNK_MAX_CHARS", raising=False)
    monkeypatch.delenv("ALPHARAVIS_PGVECTOR_CHUNK_OVERLAP_CHARS", raising=False)

    assert vector_memory._chunk_profile("artifact", "app.py", {}, "def run():\n    return 1") == "code"
    assert vector_memory._chunk_profile("large_paste", "notes", {"content_type": "log"}, "plain text") == "log"
    assert vector_memory._chunk_profile("large_paste", "settings", {"content_type": "config"}, "plain text") == "code"
    assert vector_memory._chunk_profile("archive", "chat archive", {}, "user: hello\nassistant: hi") == "chat"
    assert vector_memory._chunk_profile("archive", "code archive", {}, "```python\ndef archived():\n    return 1\n```") == "code"
    assert vector_memory._chunk_profile("archive_collection", "ops archive", {}, "ERROR service crashed\nTraceback") == "log"
    assert vector_memory._chunk_max_chars(source_type="archive") == 2800
    assert vector_memory._chunk_max_chars(source_type="archive", text="def archived():\n    return 1") == 2400
    assert vector_memory._chunk_max_chars(source_type="artifact", title="app.py", text="def run():\n    return 1") == 2400


def test_vector_chunk_text_accepts_profile_metadata(monkeypatch) -> None:
    monkeypatch.delenv("ALPHARAVIS_PGVECTOR_CHUNK_MAX_CHARS", raising=False)
    monkeypatch.delenv("ALPHARAVIS_PGVECTOR_CHUNK_OVERLAP_CHARS", raising=False)
    monkeypatch.setenv("ALPHARAVIS_PGVECTOR_CODE_CHUNK_TOKENS", "300")
    text = "\n".join(f"def fn_{index}():\n    return {index}" for index in range(120))

    chunks = vector_memory.chunk_text(text, source_type="artifact", title="module.py")

    assert len(chunks) > 1
    assert all(len(chunk) <= 1300 for chunk in chunks)


def test_vector_chunk_text_uses_langchain_splitter_for_large_paste(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeRecursiveCharacterTextSplitter:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def split_text(self, text):
            return ["langchain chunk one", "langchain chunk two"]

    monkeypatch.delenv("ALPHARAVIS_PGVECTOR_SPLITTER", raising=False)
    monkeypatch.delenv("ALPHARAVIS_PGVECTOR_CHUNK_MAX_CHARS", raising=False)
    monkeypatch.delenv("ALPHARAVIS_PGVECTOR_CHUNK_OVERLAP_CHARS", raising=False)
    monkeypatch.setenv("ALPHARAVIS_PGVECTOR_CHUNK_TOKENS", "150")
    monkeypatch.setenv("ALPHARAVIS_PGVECTOR_CHUNK_OVERLAP_TOKENS", "25")
    monkeypatch.setattr(vector_memory, "RecursiveCharacterTextSplitter", FakeRecursiveCharacterTextSplitter)

    chunks = vector_memory.chunk_text(
        "Document:\n\n" + ("large paste content " * 100),
        source_type="large_paste",
    )

    assert chunks == ["langchain chunk one", "langchain chunk two"]
    assert calls[0]["chunk_size"] == 600
    assert calls[0]["chunk_overlap"] == 100


def test_archive_chunk_text_splits_mixed_sections_by_profile(monkeypatch) -> None:
    monkeypatch.delenv("ALPHARAVIS_PGVECTOR_CHUNK_MAX_CHARS", raising=False)
    monkeypatch.delenv("ALPHARAVIS_PGVECTOR_CHUNK_OVERLAP_CHARS", raising=False)
    monkeypatch.setenv("ALPHARAVIS_PGVECTOR_SECTION_LEVEL_ARCHIVE_SPLITTING", "true")

    text = "\n".join(
        [
            "Conversation summary",
            "We discussed deployment rollback and archive recall.",
            "",
            "2026-05-21 ERROR api-bridge failed to answer",
            "2026-05-21 INFO api-bridge recovered",
            "",
            "```python",
            "def archive_marker():",
            "    return 'ARCHIVE_MIXED_SPLIT'",
            "```",
            "",
            "Final prose note about the fix.",
        ]
    )

    chunks = vector_memory.chunk_text(text, source_type="archive", title="mixed archive")

    assert len(chunks) >= 3
    assert "Conversation summary" in chunks[0]
    assert any("ERROR api-bridge" in chunk for chunk in chunks)
    assert any("archive_marker" in chunk for chunk in chunks)
    assert chunks.index(next(chunk for chunk in chunks if "ERROR api-bridge" in chunk)) < chunks.index(
        next(chunk for chunk in chunks if "archive_marker" in chunk)
    )


def test_vector_chunk_text_can_force_alpharavis_splitter(monkeypatch) -> None:
    class FailingRecursiveCharacterTextSplitter:
        def __init__(self, **kwargs):
            raise AssertionError("LangChain splitter should not be constructed")

    monkeypatch.setenv("ALPHARAVIS_PGVECTOR_SPLITTER", "alpharavis")
    monkeypatch.setenv("ALPHARAVIS_PGVECTOR_CHUNK_TOKENS", "100")
    monkeypatch.setenv("ALPHARAVIS_PGVECTOR_CHUNK_OVERLAP_TOKENS", "10")
    monkeypatch.setattr(vector_memory, "RecursiveCharacterTextSplitter", FailingRecursiveCharacterTextSplitter)

    chunks = vector_memory.chunk_text(
        "\n\n".join(f"Paragraph {index}. " + ("x" * 40) for index in range(30)),
        source_type="large_paste",
    )

    assert len(chunks) > 1
    assert all("Paragraph" in chunk for chunk in chunks)


def test_upsert_memory_record_adds_source_and_chunk_digests(monkeypatch) -> None:
    inserted: list[dict[str, object]] = []

    @dataclass
    class FakeEmbedding:
        vector: list[float]
        model: str = "fake-embed"

    async def fake_embed_text(text):
        return FakeEmbedding([0.1, 0.2, 0.3])

    async def immediate_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(vector_memory, "is_enabled", lambda: True)
    monkeypatch.setattr(vector_memory, "chunk_text", lambda *args, **kwargs: ["chunk one", "chunk two"])
    monkeypatch.setattr(vector_memory, "embed_text", fake_embed_text)
    monkeypatch.setattr(vector_memory.asyncio, "to_thread", immediate_to_thread)
    monkeypatch.setattr(vector_memory, "_ensure_schema_sync", lambda dimensions: None)
    monkeypatch.setattr(vector_memory, "_delete_source_sync", lambda **kwargs: None)
    monkeypatch.setattr(vector_memory, "_catalog_enabled", lambda: True)
    monkeypatch.setattr(vector_memory, "_insert_chunk_sync", lambda **kwargs: inserted.append(kwargs))

    result = asyncio.run(
        vector_memory.upsert_memory_record(
            source_type="large_paste",
            source_key="source-1",
            title="Source One",
            content="chunk one\n\nchunk two",
            thread_id="thread-1",
            metadata={"origin": "test"},
        )
    )

    assert result == "large_paste:source-1:2"
    catalog = inserted[0]["metadata"]
    first_chunk = inserted[1]["metadata"]
    second_chunk = inserted[2]["metadata"]
    assert catalog["source_digest"] == vector_memory._content_digest("chunk one\n\nchunk two")
    assert catalog["source_digest_algorithm"] == "sha256-normalized-text"
    assert first_chunk["source_digest"] == catalog["source_digest"]
    assert first_chunk["chunk_digest"] == vector_memory._content_digest("chunk one")
    assert second_chunk["chunk_digest"] == vector_memory._content_digest("chunk two")
    assert first_chunk["digest_algorithm"] == "sha256-normalized-text"


def test_upsert_memory_record_skips_existing_identical_source_digest(monkeypatch) -> None:
    calls: dict[str, int] = {"embed": 0, "delete": 0, "insert": 0}

    async def fake_embed_text(text):
        calls["embed"] += 1
        raise AssertionError("duplicate source should not embed")

    async def immediate_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(vector_memory, "is_enabled", lambda: True)
    monkeypatch.setattr(vector_memory, "chunk_text", lambda *args, **kwargs: ["same chunk"])
    monkeypatch.setattr(vector_memory, "embed_text", fake_embed_text)
    monkeypatch.setattr(vector_memory.asyncio, "to_thread", immediate_to_thread)
    monkeypatch.setattr(
        vector_memory,
        "_source_digest_match_sync",
        lambda **kwargs: {"chunk_count": 1, "source_key": kwargs["source_key"]},
    )
    monkeypatch.setattr(vector_memory, "_delete_source_sync", lambda **kwargs: calls.__setitem__("delete", calls["delete"] + 1))
    monkeypatch.setattr(vector_memory, "_insert_chunk_sync", lambda **kwargs: calls.__setitem__("insert", calls["insert"] + 1))

    result = asyncio.run(
        vector_memory.upsert_memory_record(
            source_type="large_paste",
            source_key="source-1",
            title="Source One",
            content="same chunk",
            thread_id="thread-1",
        )
    )

    assert result == "deduped:large_paste:source-1:1"
    assert calls == {"embed": 0, "delete": 0, "insert": 0}


def test_upsert_memory_record_emits_chunk_progress(monkeypatch) -> None:
    progress: list[dict[str, object]] = []

    @dataclass
    class FakeEmbedding:
        vector: list[float]
        model: str = "fake-embed"

    async def fake_embed_text(text):
        return FakeEmbedding([0.1, 0.2, 0.3])

    async def immediate_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(vector_memory, "is_enabled", lambda: True)
    monkeypatch.setattr(vector_memory, "chunk_text", lambda *args, **kwargs: ["chunk one", "chunk two"])
    monkeypatch.setattr(vector_memory, "embed_text", fake_embed_text)
    monkeypatch.setattr(vector_memory.asyncio, "to_thread", immediate_to_thread)
    monkeypatch.setattr(vector_memory, "_source_digest_match_sync", lambda **kwargs: None)
    monkeypatch.setattr(vector_memory, "_ensure_schema_sync", lambda dimensions: None)
    monkeypatch.setattr(vector_memory, "_delete_source_sync", lambda **kwargs: None)
    monkeypatch.setattr(vector_memory, "_catalog_enabled", lambda: False)
    monkeypatch.setattr(vector_memory, "_insert_chunk_sync", lambda **kwargs: None)

    asyncio.run(
        vector_memory.upsert_memory_record(
            source_type="large_paste",
            source_key="source-1",
            title="Source One",
            content="chunk one\n\nchunk two",
            thread_id="thread-1",
            progress_callback=progress.append,
        )
    )

    assert [event["event"] for event in progress] == ["large_ingest.chunk_indexed", "large_ingest.chunk_indexed"]
    assert progress[0]["chunk_number"] == 1
    assert progress[1]["chunk_number"] == 2
    assert progress[1]["chunk_count"] == 2


def test_read_source_chunks_clamps_bounds(monkeypatch) -> None:
    calls: dict[str, object] = {}

    async def immediate_to_thread(func, *args, **kwargs):
        calls.update(kwargs)
        return {"source_key": kwargs["source_key"], "chunks": []}

    monkeypatch.setattr(vector_memory, "is_enabled", lambda: True)
    monkeypatch.setattr(vector_memory.asyncio, "to_thread", immediate_to_thread)
    monkeypatch.setenv("ALPHARAVIS_SOURCE_READ_MAX_CHUNKS", "3")
    monkeypatch.setenv("ALPHARAVIS_SOURCE_READ_MAX_CHARS", "1000")

    result = asyncio.run(
        vector_memory.read_source_chunks(
            source_key="source-1",
            source_type="large_paste",
            thread_id="thread-1",
            max_chunks=99,
            max_chars=99999,
        )
    )

    assert result["source_key"] == "source-1"
    assert calls["max_chunks"] == 3
    assert calls["max_chars"] == 1000
    assert calls["thread_id"] == "thread-1"
