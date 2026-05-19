from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

import document_ingest  # noqa: E402


class _FakeLoader:
    def __init__(self, documents):
        self._documents = documents

    def load(self):
        return self._documents


def test_loaded_documents_to_text_preserves_part_and_page_metadata() -> None:
    text, metadata = document_ingest.loaded_documents_to_text(
        [
            SimpleNamespace(page_content="First page text.", metadata={"page": 0, "source": "doc.pdf"}),
            {"page_content": "Second page text.", "metadata": {"page_number": 2}},
            SimpleNamespace(page_content="", metadata={"page": 3}),
        ]
    )

    assert "[document_part=1 page=1]" in text
    assert "First page text." in text
    assert "[document_part=2 page=2]" in text
    assert "Second page text." in text
    assert "document_part=3" not in text
    assert metadata[0]["source"] == "doc.pdf"
    assert metadata[2]["page"] == 3


def test_load_document_file_uses_loader_and_returns_normalized_payload(tmp_path, monkeypatch) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("ignored by fake loader", encoding="utf-8")

    def fake_loader_for_path(path: Path):
        assert path == source.resolve()
        return _FakeLoader(
            [
                SimpleNamespace(
                    page_content="AlphaRavis document loader text.",
                    metadata={"source": str(path), "page": 1},
                )
            ]
        )

    monkeypatch.setattr(document_ingest, "_loader_for_path", fake_loader_for_path)

    result = document_ingest.load_document_file(source)

    assert result["ok"] is True
    assert result["title"] == "notes.txt"
    assert result["text_chars"] == len(result["text"])
    assert "AlphaRavis document loader text." in result["text"]
    assert result["metadata"]["filename"] == "notes.txt"
    assert result["metadata"]["extension"] == ".txt"
    assert result["metadata"]["loader"] == "_FakeLoader"
    assert result["metadata"]["document_part_count"] == 1


def test_load_document_file_reports_missing_file(tmp_path) -> None:
    result = document_ingest.load_document_file(tmp_path / "missing.pdf")

    assert result["ok"] is False
    assert result["text"] == ""
    assert result["error"] == "document file does not exist"


def test_load_document_file_reports_loader_failure(tmp_path, monkeypatch) -> None:
    source = tmp_path / "broken.pdf"
    source.write_text("not a pdf", encoding="utf-8")

    def fake_loader_for_path(path: Path):
        raise RuntimeError(f"cannot load {path.name}")

    monkeypatch.setattr(document_ingest, "_loader_for_path", fake_loader_for_path)

    result = document_ingest.load_document_file(source)

    assert result["ok"] is False
    assert result["metadata"]["filename"] == "broken.pdf"
    assert result["error"] == "RuntimeError: cannot load broken.pdf"
