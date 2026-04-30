from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

from context_references import parse_context_references, preprocess_context_references  # noqa: E402
from internal_context import StreamingInternalContextScrubber, sanitize_internal_context  # noqa: E402


def test_sanitize_internal_context_full_buffer() -> None:
    text = "visible <memory-context>secret memory</memory-context> done"
    assert sanitize_internal_context(text) == "visible  done"


def test_streaming_scrubber_handles_split_tags() -> None:
    scrubber = StreamingInternalContextScrubber()
    chunks = [
        "hello <memo",
        "ry-context>secret",
        " still secret</memory-",
        "context> world",
    ]

    visible = "".join(scrubber.feed(chunk) for chunk in chunks) + scrubber.flush()

    assert visible == "hello  world"
    assert "secret" not in visible


def test_streaming_scrubber_flushes_non_tag_tail() -> None:
    scrubber = StreamingInternalContextScrubber()

    visible = scrubber.feed("plain <memo") + scrubber.flush()

    assert visible == "plain <memo"


def test_parse_context_references_file_range() -> None:
    refs = parse_context_references("Bitte lies @file:`src/app.py`:2-4 und @diff")

    assert len(refs) == 2
    assert refs[0].kind == "file"
    assert refs[0].target == "src/app.py"
    assert refs[0].line_start == 2
    assert refs[0].line_end == 4
    assert refs[1].kind == "diff"


def test_file_reference_expands_inside_workspace() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "src"
        src.mkdir()
        (src / "app.py").write_text("line1\nline2\nline3\n", encoding="utf-8")

        result = asyncio.run(
            preprocess_context_references(
                "Nutze @file:src/app.py:2-3 fuer die Antwort.",
                cwd=root,
                allowed_root=root,
                context_length=1000,
            )
        )

    assert result.expanded
    assert not result.blocked
    assert "line2\nline3" in result.message
    assert "line1" not in result.message
    assert "Attached Context" in result.message


def test_sensitive_file_reference_is_warning_only() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / ".env").write_text("SECRET=abc", encoding="utf-8")

        result = asyncio.run(
            preprocess_context_references(
                "Nutze @file:.env bitte.",
                cwd=root,
                allowed_root=root,
                context_length=1000,
            )
        )

    assert result.expanded
    assert result.warnings
    assert "sensitive" in result.warnings[0]
    assert "SECRET=abc" not in result.message


def test_context_reference_budget_can_block_large_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "large.txt").write_text("x" * 2000, encoding="utf-8")

        result = asyncio.run(
            preprocess_context_references(
                "Nutze @file:large.txt.",
                cwd=root,
                allowed_root=root,
                context_length=100,
                hard_ratio=0.25,
            )
        )

    assert result.blocked
    assert "Attached Context" not in result.message
    assert "refused" in result.message


def test_url_reference_uses_injected_fetcher() -> None:
    async def fetcher(url: str) -> str:
        return f"content from {url}"

    result = asyncio.run(
        preprocess_context_references(
            "Lies @url:https://example.test/page",
            cwd=ROOT,
            allowed_root=ROOT,
            context_length=1000,
            url_fetcher=fetcher,
        )
    )

    assert result.expanded
    assert "content from https://example.test/page" in result.message


def _run_all() -> None:
    tests = [
        test_sanitize_internal_context_full_buffer,
        test_streaming_scrubber_handles_split_tags,
        test_streaming_scrubber_flushes_non_tag_tail,
        test_parse_context_references_file_range,
        test_file_reference_expands_inside_workspace,
        test_sensitive_file_reference_is_warning_only,
        test_context_reference_budget_can_block_large_file,
        test_url_reference_uses_injected_fetcher,
    ]
    for test in tests:
        test()


if __name__ == "__main__":
    _run_all()
