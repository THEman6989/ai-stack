from __future__ import annotations

from pathlib import Path
from typing import Any


def _document_text(document: Any) -> str:
    if isinstance(document, dict):
        return str(document.get("page_content") or document.get("content") or document.get("text") or "")
    return str(getattr(document, "page_content", "") or getattr(document, "content", "") or "")


def _document_metadata(document: Any) -> dict[str, Any]:
    if isinstance(document, dict):
        metadata = document.get("metadata") or {}
    else:
        metadata = getattr(document, "metadata", {}) or {}
    return metadata if isinstance(metadata, dict) else {"raw_metadata": metadata}


def _loader_for_path(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        return PyPDFLoader(str(path))
    if suffix in {".docx", ".doc"}:
        from langchain_community.document_loaders import Docx2txtLoader

        return Docx2txtLoader(str(path))
    if suffix in {".html", ".htm"}:
        from langchain_community.document_loaders import BSHTMLLoader

        return BSHTMLLoader(str(path))
    if suffix in {".md", ".markdown", ".txt", ".log", ".csv", ".json", ".yaml", ".yml"}:
        from langchain_community.document_loaders import TextLoader

        return TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
    from langchain_community.document_loaders import TextLoader

    return TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)


def loaded_documents_to_text(documents: list[Any]) -> tuple[str, list[dict[str, Any]]]:
    parts: list[str] = []
    metadata_rows: list[dict[str, Any]] = []
    for index, document in enumerate(documents):
        text = _document_text(document).strip()
        metadata = _document_metadata(document)
        metadata_rows.append(metadata)
        if not text:
            continue
        page_label = metadata.get("page") or metadata.get("page_number") or index + 1
        parts.append(f"[document_part={index + 1} page={page_label}]\n{text}")
    return "\n\n".join(parts).strip(), metadata_rows


def load_document_file(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        return {
            "ok": False,
            "path": str(resolved),
            "text": "",
            "metadata": {},
            "error": "document file does not exist",
        }
    try:
        loader = _loader_for_path(resolved)
        documents = loader.load()
    except Exception as exc:
        return {
            "ok": False,
            "path": str(resolved),
            "text": "",
            "metadata": {"filename": resolved.name, "extension": resolved.suffix.lower()},
            "error": f"{type(exc).__name__}: {exc}",
        }

    text, parts_metadata = loaded_documents_to_text(list(documents or []))
    return {
        "ok": bool(text),
        "path": str(resolved),
        "title": resolved.name,
        "text": text,
        "text_chars": len(text),
        "metadata": {
            "filename": resolved.name,
            "extension": resolved.suffix.lower(),
            "loader": type(loader).__name__,
            "document_part_count": len(parts_metadata),
            "document_parts_metadata": parts_metadata[:20],
        },
        "error": "" if text else "loader returned no text",
    }
