from __future__ import annotations

import asyncio
import hashlib
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    try:
        from langchain_text_splitters import Language
    except Exception:
        Language = None  # type: ignore[assignment]
except Exception as exc:  # pragma: no cover - optional dependency during local tests.
    RecursiveCharacterTextSplitter = None  # type: ignore[assignment]
    Language = None  # type: ignore[assignment]
    LANGCHAIN_TEXT_SPLITTERS_IMPORT_ERROR: Exception | None = exc
else:
    LANGCHAIN_TEXT_SPLITTERS_IMPORT_ERROR = None

try:
    import psycopg
    from psycopg import sql
    from psycopg.types.json import Jsonb
except Exception as exc:  # pragma: no cover - optional dependency
    psycopg = None
    sql = None
    Jsonb = None
    PSYCOPG_IMPORT_ERROR: Exception | None = exc
else:
    PSYCOPG_IMPORT_ERROR = None


class VectorMemoryError(RuntimeError):
    """Raised when the optional pgvector sidecar cannot complete a request."""


@dataclass(frozen=True)
class EmbeddingResult:
    vector: list[float]
    model: str


def _env_bool(name: str, default: str = "false") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "y", "on"}


def _media_index_version() -> str:
    return os.getenv("ALPHARAVIS_MEDIA_INDEX_VERSION", "2026-05-12-v1").strip() or "2026-05-12-v1"


def _media_model_card_id(model_id: str = "") -> str:
    return (
        model_id
        or os.getenv("ALPHARAVIS_MEDIA_VISION_EMBEDDING_MODEL_CARD")
        or os.getenv("ALPHARAVIS_VIDEO_ANALYSIS_MODEL_ID")
        or os.getenv("ALPHARAVIS_VISION_EMBEDDING_MODEL")
        or "vision-embed"
    ).strip()


def _media_chunking_config_hash() -> str:
    raw = "|".join(
        [
            os.getenv("ALPHARAVIS_VIDEO_ANALYSIS_FPS", "1"),
            os.getenv("ALPHARAVIS_VIDEO_ANALYSIS_MAX_FPS", "1"),
            os.getenv("ALPHARAVIS_VIDEO_ANALYSIS_MAX_FRAMES", "100"),
            os.getenv("ALPHARAVIS_VIDEO_ANALYSIS_MAX_FRAME_SIDE", ""),
            os.getenv("ALPHARAVIS_VIDEO_ANALYSIS_INCLUDE_AUDIO", "false"),
            os.getenv("ALPHARAVIS_VIDEO_ANALYSIS_TRANSCRIBE_AUDIO", "false"),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def is_enabled() -> bool:
    backend = os.getenv("ALPHARAVIS_VECTOR_BACKEND", "off").strip().lower()
    return backend == "pgvector" or _env_bool("ALPHARAVIS_ENABLE_PGVECTOR_MEMORY", "false")


def _require_psycopg() -> None:
    if psycopg is None or sql is None or Jsonb is None:
        raise VectorMemoryError(f"psycopg is unavailable: {PSYCOPG_IMPORT_ERROR}")


def _database_url() -> str:
    configured = os.getenv("ALPHARAVIS_PGVECTOR_DATABASE_URL", "").strip()
    if configured:
        return configured
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    return f"postgresql://postgres:{password}@vectordb:5432/rag_api"


def _table_name() -> str:
    table_name = os.getenv("ALPHARAVIS_PGVECTOR_TABLE", "alpharavis_memory_vectors")
    table_name = re.sub(r"[^a-zA-Z0-9_]+", "_", table_name).strip("_")
    return table_name or "alpharavis_memory_vectors"


def _queue_table_name() -> str:
    table_name = os.getenv("ALPHARAVIS_PGVECTOR_QUEUE_TABLE", "alpharavis_embedding_jobs")
    table_name = re.sub(r"[^a-zA-Z0-9_]+", "_", table_name).strip("_")
    return table_name or "alpharavis_embedding_jobs"


def _vision_table_name() -> str:
    table_name = os.getenv("ALPHARAVIS_VISION_PGVECTOR_TABLE", "alpharavis_media_vectors")
    table_name = re.sub(r"[^a-zA-Z0-9_]+", "_", table_name).strip("_")
    return table_name or "alpharavis_media_vectors"


def _table_identifier():
    if sql is None:
        raise VectorMemoryError("psycopg.sql is unavailable.")
    return sql.Identifier(_table_name())


def _queue_table_identifier():
    if sql is None:
        raise VectorMemoryError("psycopg.sql is unavailable.")
    return sql.Identifier(_queue_table_name())


def _vision_table_identifier():
    if sql is None:
        raise VectorMemoryError("psycopg.sql is unavailable.")
    return sql.Identifier(_vision_table_name())


def vision_is_enabled() -> bool:
    return is_enabled() and _env_bool("ALPHARAVIS_ENABLE_VISION_VECTOR_MEMORY", "false")


def _chunk_chars_from_tokens(env_name: str, default_tokens: int) -> int:
    token_count = max(100, int(os.getenv(env_name, str(default_tokens))))
    chars_per_token = max(1.0, float(os.getenv("ALPHARAVIS_PGVECTOR_CHARS_PER_TOKEN", "4.0")))
    return max(400, int(token_count * chars_per_token))


def _chunk_overlap_chars_from_tokens(env_name: str, default_tokens: int, max_chars: int) -> int:
    token_count = max(0, int(os.getenv(env_name, str(default_tokens))))
    chars_per_token = max(1.0, float(os.getenv("ALPHARAVIS_PGVECTOR_CHARS_PER_TOKEN", "4.0")))
    return min(max(0, int(token_count * chars_per_token)), max_chars // 2)


def _legacy_chunk_max_chars() -> int | None:
    raw = os.getenv("ALPHARAVIS_PGVECTOR_CHUNK_MAX_CHARS", "").strip()
    return max(400, int(raw)) if raw else None


def _legacy_chunk_overlap_chars(max_chars: int) -> int | None:
    raw = os.getenv("ALPHARAVIS_PGVECTOR_CHUNK_OVERLAP_CHARS", "").strip()
    return min(max(0, int(raw)), max_chars // 2) if raw else None


def _chunk_profile(source_type: str = "", title: str = "", metadata: dict[str, Any] | None = None, text: str = "") -> str:
    metadata = metadata or {}
    explicit = str(metadata.get("chunk_profile") or os.getenv("ALPHARAVIS_PGVECTOR_CHUNK_PROFILE", "")).strip().lower()
    if explicit in {"default", "chat", "archive", "log", "code"}:
        return explicit
    content_type = str(metadata.get("content_type") or metadata.get("source_content_type") or "").strip().lower()
    if content_type == "log":
        return "log"
    if content_type in {"code", "config"}:
        return "code"
    if content_type == "prose":
        return "default"

    source = str(source_type or "").lower()
    sample = text[:6000]
    looks_like_code = bool(
        "```" in sample
        or re.search(
            r"^\s*(?:async\s+def|def|class|function|import|from|const|let|var|SELECT|CREATE TABLE)\b",
            sample,
            re.MULTILINE,
        )
    )
    looks_like_log = bool(
        re.search(r"^\s*(?:\d{4}-\d{2}-\d{2}|INFO|WARN|WARNING|ERROR|DEBUG|Traceback|Exception:)", sample, re.MULTILINE)
    )
    pathish = " ".join(
        str(value or "")
        for value in [
            title,
            metadata.get("path"),
            metadata.get("file_path"),
            metadata.get("filename"),
            metadata.get("source_path"),
            metadata.get("source_key"),
        ]
    ).lower()
    code_ext = (
        ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".kt",
        ".c", ".h", ".cpp", ".hpp", ".cs", ".php", ".rb", ".swift", ".scala",
        ".sh", ".bash", ".zsh", ".ps1", ".sql", ".html", ".css", ".scss",
        ".json", ".yaml", ".yml", ".toml", ".xml",
    )
    log_ext = (".log", ".out", ".err", ".trace")

    if source in {"code", "source_code", "repo_file"} or any(pathish.endswith(ext) or ext in pathish for ext in code_ext):
        return "code"
    if source in {"log", "logs", "terminal", "command_output"} or any(pathish.endswith(ext) for ext in log_ext):
        return "log"
    if source in {"archive", "archive_collection"}:
        if looks_like_code:
            return "code"
        if looks_like_log:
            return "log"
        return "chat"
    if source in {"archive", "archive_collection", "session_turn", "chat", "conversation"}:
        return "chat"

    if looks_like_code:
        return "code"
    if looks_like_log:
        return "log"
    return "default"


def _chunk_max_chars(source_type: str = "", title: str = "", metadata: dict[str, Any] | None = None, text: str = "") -> int:
    profile = _chunk_profile(source_type, title, metadata, text)
    legacy = _legacy_chunk_max_chars()
    if legacy is not None and profile == "default":
        return legacy
    defaults = {
        "default": ("ALPHARAVIS_PGVECTOR_CHUNK_TOKENS", 900),
        "chat": ("ALPHARAVIS_PGVECTOR_CHAT_CHUNK_TOKENS", 700),
        "archive": ("ALPHARAVIS_PGVECTOR_CHAT_CHUNK_TOKENS", 700),
        "log": ("ALPHARAVIS_PGVECTOR_LOG_CHUNK_TOKENS", 1200),
        "code": ("ALPHARAVIS_PGVECTOR_CODE_CHUNK_TOKENS", 600),
    }
    env_name, default_tokens = defaults.get(profile, defaults["default"])
    return _chunk_chars_from_tokens(env_name, default_tokens)


def _chunk_overlap_chars(
    max_chars: int,
    source_type: str = "",
    title: str = "",
    metadata: dict[str, Any] | None = None,
    text: str = "",
) -> int:
    profile = _chunk_profile(source_type, title, metadata, text)
    legacy = _legacy_chunk_overlap_chars(max_chars)
    if legacy is not None and profile == "default":
        return legacy
    defaults = {
        "default": ("ALPHARAVIS_PGVECTOR_CHUNK_OVERLAP_TOKENS", 125),
        "chat": ("ALPHARAVIS_PGVECTOR_CHAT_CHUNK_OVERLAP_TOKENS", 100),
        "archive": ("ALPHARAVIS_PGVECTOR_CHAT_CHUNK_OVERLAP_TOKENS", 100),
        "log": ("ALPHARAVIS_PGVECTOR_LOG_CHUNK_OVERLAP_TOKENS", 75),
        "code": ("ALPHARAVIS_PGVECTOR_CODE_CHUNK_OVERLAP_TOKENS", 80),
    }
    env_name, default_tokens = defaults.get(profile, defaults["default"])
    return _chunk_overlap_chars_from_tokens(env_name, default_tokens, max_chars)


def _preview_chars() -> int:
    return max(200, int(os.getenv("ALPHARAVIS_PGVECTOR_PREVIEW_CHARS", "900")))


def _splitter_mode(metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata or {}
    raw = str(metadata.get("splitter") or metadata.get("chunk_splitter") or os.getenv("ALPHARAVIS_PGVECTOR_SPLITTER", "auto")).strip().lower()
    aliases = {
        "": "auto",
        "default": "auto",
        "native": "alpharavis",
        "internal": "alpharavis",
        "legacy": "alpharavis",
        "own": "alpharavis",
        "recursive": "langchain",
        "recursive_character": "langchain",
        "recursive_character_text_splitter": "langchain",
    }
    return aliases.get(raw, raw if raw in {"auto", "langchain", "alpharavis", "code", "tree_sitter"} else "auto")


def _langchain_splitter_source_default(source_type: str, profile: str) -> bool:
    normalized = str(source_type or "").strip().lower().replace("-", "_")
    return profile == "default" and normalized in {
        "external_document",
        "document",
        "pdf",
        "uploaded_document",
        "artifact_document",
        "large_paste",
        "large_ingest",
    }


def _should_use_langchain_splitter(
    *,
    source_type: str = "",
    title: str = "",
    metadata: dict[str, Any] | None = None,
    text: str = "",
) -> bool:
    mode = _splitter_mode(metadata)
    if mode == "alpharavis":
        return False
    if RecursiveCharacterTextSplitter is None:
        return False
    if mode == "langchain":
        return True
    profile = _chunk_profile(source_type, title, metadata, text)
    return _langchain_splitter_source_default(source_type, profile)


def _source_code_language(source_type: str = "", title: str = "", metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata or {}
    explicit = str(metadata.get("language") or metadata.get("code_language") or "").strip().lower()
    if explicit:
        return explicit
    pathish = " ".join(
        str(value or "")
        for value in [
            title,
            metadata.get("path"),
            metadata.get("file_path"),
            metadata.get("filename"),
            metadata.get("source_path"),
            metadata.get("source_key"),
            source_type,
        ]
    ).lower()
    mapping = {
        ".py": "python",
        ".pyi": "python",
        ".js": "js",
        ".jsx": "js",
        ".ts": "ts",
        ".tsx": "ts",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".kt": "kotlin",
        ".c": "c",
        ".h": "cpp",
        ".cpp": "cpp",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".php": "php",
        ".rb": "ruby",
        ".swift": "swift",
        ".scala": "scala",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
        ".ps1": "powershell",
        ".sql": "sql",
        ".html": "html",
        ".css": "css",
        ".scss": "css",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".xml": "xml",
    }
    for suffix, language in mapping.items():
        if pathish.endswith(suffix) or suffix in pathish:
            return language
    return ""


def _langchain_language_enum(language: str) -> Any | None:
    if Language is None:
        return None
    aliases = {
        "javascript": "JS",
        "js": "JS",
        "typescript": "TS",
        "ts": "TS",
        "python": "PYTHON",
        "py": "PYTHON",
        "go": "GO",
        "golang": "GO",
        "java": "JAVA",
        "kotlin": "KOTLIN",
        "rust": "RUST",
        "rs": "RUST",
        "cpp": "CPP",
        "c++": "CPP",
        "c": "C",
        "csharp": "CSHARP",
        "c#": "CSHARP",
        "php": "PHP",
        "ruby": "RUBY",
        "rb": "RUBY",
        "swift": "SWIFT",
        "scala": "SCALA",
        "markdown": "MARKDOWN",
        "md": "MARKDOWN",
        "html": "HTML",
        "latex": "LATEX",
        "solidity": "SOL",
    }
    enum_name = aliases.get(str(language or "").strip().lower())
    return getattr(Language, enum_name, None) if enum_name else None


def _code_separators(language: str = "") -> list[str]:
    language = str(language or "").lower()
    common = ["\nclass ", "\ndef ", "\nasync def ", "\nfunction ", "\nexport ", "\nimport ", "\n\n", "\n", " ", ""]
    if language in {"python", "py"}:
        return ["\nclass ", "\ndef ", "\nasync def ", "\n@", "\n\n", "\n", " ", ""]
    if language in {"js", "javascript", "ts", "typescript"}:
        return ["\nexport class ", "\nclass ", "\nexport function ", "\nfunction ", "\nconst ", "\nlet ", "\n\n", "\n", " ", ""]
    if language in {"go", "golang"}:
        return ["\nfunc ", "\ntype ", "\npackage ", "\nimport ", "\n\n", "\n", " ", ""]
    if language in {"rust", "rs"}:
        return ["\nfn ", "\nimpl ", "\nstruct ", "\nenum ", "\nmod ", "\nuse ", "\n\n", "\n", " ", ""]
    if language in {"java", "kotlin", "cpp", "c", "csharp", "c#"}:
        return ["\nclass ", "\ninterface ", "\nstruct ", "\npublic ", "\nprivate ", "\nprotected ", "\n\n", "\n", " ", ""]
    if language in {"sql"}:
        return ["\nCREATE ", "\nALTER ", "\nSELECT ", "\nINSERT ", "\nUPDATE ", "\nDELETE ", "\n\n", "\n", " ", ""]
    return common


def _code_aware_chunk_text(text: str, *, max_chars: int, overlap: int, language: str = "") -> list[str]:
    if RecursiveCharacterTextSplitter is not None:
        language_enum = _langchain_language_enum(language)
        try:
            if language_enum is not None:
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=language_enum,
                    chunk_size=max_chars,
                    chunk_overlap=overlap,
                    keep_separator=True,
                )
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=max_chars,
                    chunk_overlap=overlap,
                    keep_separator=True,
                    separators=_code_separators(language),
                )
            chunks = [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]
            if chunks:
                return chunks
        except Exception as exc:
            print(f"WARNING: code-aware LangChain splitter failed; falling back to AlphaRavis splitter: {exc}")
    return []


def _langchain_chunk_text(text: str, *, max_chars: int, overlap: int) -> list[str]:
    if RecursiveCharacterTextSplitter is None:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap,
        keep_separator=True,
        separators=["\n\n", "\n", " ", ""],
    )
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]


def _section_level_archive_splitting_enabled(metadata: dict[str, Any] | None = None) -> bool:
    metadata = metadata or {}
    explicit = str(metadata.get("section_level_splitting") or metadata.get("mixed_archive_splitting") or "").strip().lower()
    if explicit in {"0", "false", "no", "off", "disabled"}:
        return False
    if explicit in {"1", "true", "yes", "on", "enabled"}:
        return True
    return _env_bool("ALPHARAVIS_PGVECTOR_SECTION_LEVEL_ARCHIVE_SPLITTING", "true")


def _is_archive_source_type(source_type: str) -> bool:
    return str(source_type or "").strip().lower().replace("-", "_") in {"archive", "archive_collection"}


def _archive_line_profile(line: str, *, in_fence: bool) -> str:
    stripped = line.strip()
    if in_fence or stripped.startswith("```"):
        return "code"
    if re.search(r"^\s*(?:\d{4}-\d{2}-\d{2}[T\s]|\[[^\]]+\]\s*)?(?:INFO|WARN|WARNING|ERROR|DEBUG|TRACE)\b", line):
        return "log"
    if re.search(r"^\s*(?:Traceback|Exception|Caused by:|at\s+[\w.$]+\(.*\))", line):
        return "log"
    if re.search(r"^\s*(?:async\s+def|def|class|function|import|from|const|let|var|SELECT|CREATE TABLE)\b", line):
        return "code"
    if re.search(r"^\s*[\w.-]+\s*[:=]\s*[^=].*$", line) and len(stripped) < 240:
        return "config"
    return "prose"


def _archive_section_chunk_profile(section_profile: str) -> str:
    if section_profile == "log":
        return "log"
    if section_profile in {"code", "config"}:
        return "code"
    return "chat"


def _mixed_archive_sections(text: str) -> list[tuple[str, str]]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines(keepends=True)
    if not lines:
        return []
    sections: list[tuple[str, str]] = []
    buffer: list[str] = []
    current_profile = ""
    in_fence = False

    for line in lines:
        stripped = line.strip()
        profile = _archive_line_profile(line, in_fence=in_fence)
        if not stripped and current_profile:
            profile = current_profile
        if buffer and profile != current_profile:
            sections.append((current_profile, "".join(buffer).strip()))
            buffer = []
        current_profile = profile
        buffer.append(line)
        if stripped.startswith("```"):
            in_fence = not in_fence

    if buffer:
        sections.append((current_profile or "prose", "".join(buffer).strip()))
    sections = [(profile, section) for profile, section in sections if section]
    meaningful_profiles = {
        profile
        for profile, section in sections
        if section.strip() and profile in {"code", "log", "config", "prose"}
    }
    if len(meaningful_profiles) < 2:
        return []
    return sections


def _chunk_mixed_archive_sections(
    text: str,
    *,
    source_type: str,
    title: str,
    metadata: dict[str, Any] | None = None,
) -> list[str]:
    chunks: list[str] = []
    for section_profile, section in _mixed_archive_sections(text):
        chunk_profile = _archive_section_chunk_profile(section_profile)
        section_metadata = {**(metadata or {}), "chunk_profile": chunk_profile, "section_profile": section_profile}
        max_chars = _chunk_max_chars(source_type=source_type, title=title, metadata=section_metadata, text=section)
        overlap = _chunk_overlap_chars(max_chars, source_type=source_type, title=title, metadata=section_metadata, text=section)
        for semantic_section in _semantic_sections(section):
            chunks.extend(_split_large_section(semantic_section, max_chars, overlap))
    return [chunk for chunk in chunks if chunk]


def _embedding_models() -> list[str]:
    models = [os.getenv("ALPHARAVIS_PGVECTOR_EMBEDDING_MODEL", "memory-embed").strip()]
    fallback = os.getenv("ALPHARAVIS_PGVECTOR_FALLBACK_EMBEDDING_MODEL", "memory-embed-fallback").strip()
    if fallback and fallback not in models:
        models.append(fallback)
    return [model for model in models if model]


def _vision_embedding_models() -> list[str]:
    models = [os.getenv("ALPHARAVIS_VISION_EMBEDDING_MODEL", "vision-embed").strip()]
    fallback = os.getenv("ALPHARAVIS_VISION_EMBEDDING_FALLBACK_MODEL", "").strip()
    if fallback and fallback not in models:
        models.append(fallback)
    return [model for model in models if model]


def _vision_embedding_base_url() -> str:
    return (
        os.getenv("ALPHARAVIS_VISION_EMBEDDING_MODEL_URL")
        or os.getenv("ALPHARAVIS_VISION_EMBEDDING_BASE_URL")
        or os.getenv("VISION_EMBEDDING_API_BASE")
        or os.getenv("ALPHARAVIS_PGVECTOR_EMBEDDING_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or "http://litellm:4000/v1"
    ).rstrip("/")


def _vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(f"{float(value):.9g}" for value in embedding) + "]"


def _distance_threshold() -> float | None:
    raw = os.getenv("ALPHARAVIS_PGVECTOR_DISTANCE_THRESHOLD", "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise VectorMemoryError(f"Invalid ALPHARAVIS_PGVECTOR_DISTANCE_THRESHOLD={raw!r}") from exc


def _record_id(source_type: str, source_key: str, thread_id: str, scope: str, chunk_index: int) -> str:
    raw = f"{source_type}:{source_key}:{thread_id}:{scope}:{chunk_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _content_digest(text: str) -> str:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _catalog_enabled() -> bool:
    return _env_bool("ALPHARAVIS_PGVECTOR_CATALOG_ENABLED", "true")


def _extract_matches(pattern: str, text: str, limit: int = 30) -> list[str]:
    matches = []
    seen = set()
    for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
        value = (match.group(1) if match.groups() else match.group(0)).strip()
        value = value.strip("`'\".,;:()[]{}")
        if value and value not in seen:
            seen.add(value)
            matches.append(value)
        if len(matches) >= limit:
            break
    return matches


def _compact_metadata(metadata: dict[str, Any], limit: int = 40) -> list[str]:
    lines = []
    for key, value in sorted((metadata or {}).items(), key=lambda item: str(item[0]))[:limit]:
        if isinstance(value, (dict, list, tuple)):
            value_text = str(value)[:500]
        else:
            value_text = str(value)[:500]
        lines.append(f"- {key}: {value_text}")
    return lines


def build_catalog_text(
    *,
    source_type: str,
    source_key: str,
    title: str,
    content: str,
    chunks: list[str],
    thread_id: str,
    thread_key: str,
    scope: str,
    metadata: dict[str, Any],
) -> str:
    headings = _extract_matches(r"^\s{0,3}#{1,6}\s+(.+)$", content)
    file_paths = _extract_matches(
        r"((?:[A-Za-z]:\\|/|\.{1,2}/)?[\w .\-\\/]+?\.(?:py|ts|tsx|js|jsx|md|json|yaml|yml|toml|env|txt|pdf|docx|sql|sh|ps1|go|rs|java|cpp|c|h))",
        content,
    )
    urls = _extract_matches(r"(https?://[^\s\]\)>,]+)", content)
    code_langs = _extract_matches(r"^```([a-zA-Z0-9_+.-]+)?\s*$", content)
    functions = _extract_matches(r"^\s*(?:async\s+def|def|class|function)\s+([A-Za-z_][\w]*)", content)
    db_terms = [
        term
        for term in [
            "mongodb",
            "postgres",
            "pgvector",
            "redis",
            "sqlite",
            "mysql",
            "mariadb",
            "qdrant",
            "weaviate",
            "milvus",
            "chroma",
            "rag",
            "embedding",
        ]
        if re.search(rf"\b{re.escape(term)}\b", content, re.IGNORECASE)
    ]

    chunk_lines = []
    for index, chunk in enumerate(chunks[:80]):
        first_line = next((line.strip() for line in chunk.splitlines() if line.strip()), "")
        chunk_lines.append(f"- chunk {index + 1}/{len(chunks)}: {first_line[:220]}")

    sections = [
        "AlphaRavis source catalog generated from complete original source data.",
        f"source_type: {source_type}",
        f"source_key: {source_key}",
        f"title: {title}",
        f"thread_id: {thread_id or 'global'}",
        f"thread_key: {thread_key or thread_id or 'global'}",
        f"scope: {scope}",
        f"source_chars: {len(content)}",
        f"chunk_count: {len(chunks)}",
    ]

    metadata_lines = _compact_metadata(metadata)
    if metadata_lines:
        sections.append("metadata:\n" + "\n".join(metadata_lines))
    if headings:
        sections.append("headings:\n" + "\n".join(f"- {item}" for item in headings))
    if file_paths:
        sections.append("file_paths:\n" + "\n".join(f"- {item}" for item in file_paths))
    if urls:
        sections.append("urls:\n" + "\n".join(f"- {item}" for item in urls))
    if code_langs:
        sections.append("code_languages:\n" + "\n".join(f"- {item or 'plain'}" for item in code_langs))
    if functions:
        sections.append("code_symbols:\n" + "\n".join(f"- {item}" for item in functions))
    if db_terms:
        sections.append("database_or_rag_topics:\n" + "\n".join(f"- {item}" for item in db_terms))
    if chunk_lines:
        sections.append("chunk_table_of_contents:\n" + "\n".join(chunk_lines))

    return "\n\n".join(sections)


def _looks_like_code_boundary(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    return bool(
        stripped.startswith(("# ", "## ", "### ", "#### ", "```"))
        or re.match(r"^(class|def|async def|function|const|let|var|export|import)\b", stripped)
        or re.match(r"^[-*]\s+`?[\w./\\-]+\.(py|ts|tsx|js|jsx|md|json|yaml|yml|toml|go|rs|java|cpp|c|h)`?", stripped)
    )


def _semantic_sections(text: str) -> list[str]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines(keepends=True)
    sections: list[str] = []
    buffer: list[str] = []
    in_fence = False

    for line in lines:
        stripped = line.strip()
        starts_new = bool(buffer and not in_fence and _looks_like_code_boundary(line))
        if starts_new:
            sections.append("".join(buffer).strip())
            buffer = []

        buffer.append(line)
        if stripped.startswith("```"):
            in_fence = not in_fence

    if buffer:
        sections.append("".join(buffer).strip())

    return [section for section in sections if section]


def _split_large_section(section: str, max_chars: int, overlap: int) -> list[str]:
    if len(section) <= max_chars:
        return [section]

    chunks = []
    start = 0
    while start < len(section):
        end = min(start + max_chars, len(section))
        if end < len(section):
            boundary = max(section.rfind("\n\n", start, end), section.rfind("\n", start, end))
            if boundary > start + max_chars // 2:
                end = boundary
        chunks.append(section[start:end].strip())
        if end >= len(section):
            break
        start = max(end - overlap, start + 1)
    return [chunk for chunk in chunks if chunk]


def chunk_text(
    text: str,
    *,
    source_type: str = "",
    title: str = "",
    metadata: dict[str, Any] | None = None,
) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    if _is_archive_source_type(source_type) and _section_level_archive_splitting_enabled(metadata):
        mixed_chunks = _chunk_mixed_archive_sections(text, source_type=source_type, title=title, metadata=metadata)
        if mixed_chunks:
            return mixed_chunks

    max_chars = _chunk_max_chars(source_type=source_type, title=title, metadata=metadata, text=text)
    overlap = _chunk_overlap_chars(max_chars, source_type=source_type, title=title, metadata=metadata, text=text)
    profile = _chunk_profile(source_type, title, metadata, text)
    mode = _splitter_mode(metadata)
    if profile == "code" or mode in {"code", "tree_sitter"}:
        chunks = _code_aware_chunk_text(
            text,
            max_chars=max_chars,
            overlap=overlap,
            language=_source_code_language(source_type, title, metadata),
        )
        if chunks:
            return chunks
    if _should_use_langchain_splitter(source_type=source_type, title=title, metadata=metadata, text=text):
        try:
            chunks = _langchain_chunk_text(text, max_chars=max_chars, overlap=overlap)
        except Exception as exc:
            print(f"WARNING: LangChain text splitter failed; falling back to AlphaRavis splitter: {exc}")
        else:
            if chunks:
                return chunks

    sections: list[str] = []
    for section in _semantic_sections(text):
        sections.extend(_split_large_section(section, max_chars, overlap))

    chunks: list[str] = []
    current = ""
    for section in sections:
        if not current:
            current = section
            continue
        candidate = f"{current}\n\n{section}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        chunks.append(current)
        prefix = current[-overlap:].strip() if overlap else ""
        current = f"{prefix}\n\n{section}".strip() if prefix else section

    if current:
        chunks.append(current)
    return chunks


async def _embed_text_with_model(text: str, model: str) -> EmbeddingResult:
    base_url = os.getenv(
        "ALPHARAVIS_PGVECTOR_EMBEDDING_BASE_URL",
        os.getenv("OPENAI_API_BASE", "http://litellm:4000/v1"),
    ).rstrip("/")
    api_key = os.getenv(
        "ALPHARAVIS_PGVECTOR_EMBEDDING_API_KEY",
        os.getenv("OPENAI_API_KEY", os.getenv("LITELLM_MASTER_KEY", "sk-local-dev")),
    )
    timeout = float(os.getenv("ALPHARAVIS_PGVECTOR_EMBEDDING_TIMEOUT_SECONDS", "20"))
    payload = {"model": model, "input": text[:_chunk_max_chars(text=text)]}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{base_url}/embeddings", headers=headers, json=payload)
    if response.status_code >= 400:
        raise VectorMemoryError(f"{model} returned HTTP {response.status_code}: {response.text[:500]}")

    data = response.json()
    try:
        embedding = data["data"][0]["embedding"]
    except Exception as exc:
        raise VectorMemoryError(f"{model} response did not contain data[0].embedding: {data!r}") from exc

    if not isinstance(embedding, list) or not embedding:
        raise VectorMemoryError(f"{model} returned an empty or invalid vector.")
    return EmbeddingResult(vector=[float(value) for value in embedding], model=model)


def _media_input_payload(*, media_url: str, caption: str, media_type: str) -> Any:
    media_url = (media_url or "").strip()
    caption = (caption or "").strip()
    media_type = (media_type or "image").strip().lower()
    if media_type == "text" or not media_url:
        return caption
    if media_type == "image":
        parts: list[dict[str, Any]] = [
            {"type": "input_image", "image_url": media_url},
        ]
    elif media_type == "video":
        parts = [
            {"type": "input_video", "video_url": media_url},
        ]
    else:
        parts = [
            {"type": "input_file", "file_url": media_url},
        ]
    if caption:
        parts.insert(0, {"type": "input_text", "text": caption})
    return parts


async def _embed_media_with_model(
    *,
    media_url: str,
    caption: str,
    media_type: str,
    model: str,
) -> EmbeddingResult:
    base_url = _vision_embedding_base_url()
    api_key = os.getenv(
        "ALPHARAVIS_VISION_EMBEDDING_API_KEY",
        os.getenv("ALPHARAVIS_PGVECTOR_EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", os.getenv("LITELLM_MASTER_KEY", "sk-local-dev"))),
    )
    timeout = float(os.getenv("ALPHARAVIS_VISION_EMBEDDING_TIMEOUT_SECONDS", os.getenv("ALPHARAVIS_PGVECTOR_EMBEDDING_TIMEOUT_SECONDS", "30")))
    payload = {
        "model": model,
        "input": _media_input_payload(media_url=media_url, caption=caption, media_type=media_type),
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{base_url}/embeddings", headers=headers, json=payload)
    if response.status_code >= 400:
        raise VectorMemoryError(f"{model} returned HTTP {response.status_code}: {response.text[:500]}")

    data = response.json()
    try:
        embedding = data["data"][0]["embedding"]
    except Exception as exc:
        raise VectorMemoryError(f"{model} response did not contain data[0].embedding: {data!r}") from exc

    if not isinstance(embedding, list) or not embedding:
        raise VectorMemoryError(f"{model} returned an empty or invalid vector.")
    return EmbeddingResult(vector=[float(value) for value in embedding], model=model)


async def embed_media(*, media_url: str = "", caption: str = "", media_type: str = "image") -> EmbeddingResult:
    caption = (caption or "").strip()
    media_url = (media_url or "").strip()
    if not caption and not media_url:
        raise VectorMemoryError("Cannot embed media without media_url or caption.")

    errors = []
    for model in _vision_embedding_models():
        try:
            return await _embed_media_with_model(
                media_url=media_url,
                caption=caption,
                media_type=media_type,
                model=model,
            )
        except Exception as exc:
            errors.append(f"{model}: {exc}")

    if _env_bool("ALPHARAVIS_VISION_EMBEDDING_FALLBACK_TEXT", "true") and caption:
        try:
            return await embed_text(caption)
        except Exception as exc:
            errors.append(f"text fallback: {exc}")

    raise VectorMemoryError("All vision embedding models failed: " + " | ".join(errors))


async def embed_text(text: str) -> EmbeddingResult:
    text = (text or "").strip()
    if not text:
        raise VectorMemoryError("Cannot embed empty text.")

    errors = []
    for model in _embedding_models():
        try:
            return await _embed_text_with_model(text, model)
        except Exception as exc:
            errors.append(f"{model}: {exc}")
    raise VectorMemoryError("All embedding models failed: " + " | ".join(errors))


def _ensure_schema_sync(dimensions: int) -> None:
    _require_psycopg()
    table_name = _table_name()
    table = sql.Identifier(table_name)
    hnsw_enabled = _env_bool("ALPHARAVIS_PGVECTOR_CREATE_HNSW_INDEX", "true")

    with psycopg.connect(_database_url(), autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {table} (
                        id TEXT PRIMARY KEY,
                        namespace TEXT NOT NULL,
                        scope TEXT NOT NULL,
                        thread_id TEXT,
                        thread_key TEXT,
                        source_type TEXT NOT NULL,
                        source_key TEXT NOT NULL,
                        title TEXT,
                        content TEXT NOT NULL,
                        chunk_text TEXT NOT NULL DEFAULT '',
                        catalog_text TEXT NOT NULL DEFAULT '',
                        preview_text TEXT NOT NULL DEFAULT '',
                        chunk_index INTEGER NOT NULL DEFAULT 0,
                        chunk_count INTEGER NOT NULL DEFAULT 1,
                        is_catalog BOOLEAN NOT NULL DEFAULT false,
                        embedding_model TEXT NOT NULL DEFAULT '',
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding vector({dimensions}) NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                ).format(table=table, dimensions=sql.Literal(dimensions))
            )
            for column, ddl in [
                ("chunk_text", "TEXT NOT NULL DEFAULT ''"),
                ("catalog_text", "TEXT NOT NULL DEFAULT ''"),
                ("preview_text", "TEXT NOT NULL DEFAULT ''"),
                ("chunk_index", "INTEGER NOT NULL DEFAULT 0"),
                ("chunk_count", "INTEGER NOT NULL DEFAULT 1"),
                ("is_catalog", "BOOLEAN NOT NULL DEFAULT false"),
                ("embedding_model", "TEXT NOT NULL DEFAULT ''"),
            ]:
                cur.execute(sql.SQL("ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} " + ddl).format(
                    table=table,
                    column=sql.Identifier(column),
                ))
            cur.execute(
                sql.SQL(
                    "UPDATE {table} SET chunk_text = content WHERE chunk_text = ''"
                ).format(table=table)
            )
            cur.execute(
                sql.SQL(
                    "UPDATE {table} SET preview_text = LEFT(content, %s) WHERE preview_text = ''"
                ).format(table=table),
                (_preview_chars(),),
            )
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {index} "
                    "ON {table} (namespace, scope, thread_id, source_type, source_key)"
                ).format(
                    index=sql.Identifier(f"{table_name}_scope_idx"),
                    table=table,
                )
            )
            if hnsw_enabled:
                try:
                    cur.execute(
                        sql.SQL(
                            "CREATE INDEX IF NOT EXISTS {index} "
                            "ON {table} USING hnsw (embedding vector_cosine_ops)"
                        ).format(
                            index=sql.Identifier(f"{table_name}_embedding_hnsw_idx"),
                            table=table,
                        )
                    )
                except Exception as exc:
                    print(f"WARNING: pgvector HNSW index unavailable; semantic search will still work: {exc}")


def _ensure_vision_schema_sync(dimensions: int) -> None:
    _require_psycopg()
    table_name = _vision_table_name()
    table = sql.Identifier(table_name)
    hnsw_enabled = _env_bool("ALPHARAVIS_VISION_PGVECTOR_CREATE_HNSW_INDEX", "true")

    with psycopg.connect(_database_url(), autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {table} (
                        id TEXT PRIMARY KEY,
                        namespace TEXT NOT NULL,
                        scope TEXT NOT NULL DEFAULT 'thread',
                        thread_id TEXT,
                        thread_key TEXT,
                        source_type TEXT NOT NULL,
                        source_key TEXT NOT NULL,
                        file_id TEXT NOT NULL DEFAULT '',
                        media_type TEXT NOT NULL DEFAULT 'unknown',
                        media_url TEXT NOT NULL DEFAULT '',
                        title TEXT NOT NULL DEFAULT '',
                        caption TEXT NOT NULL DEFAULT '',
                        frame_index INTEGER NOT NULL DEFAULT 0,
                        frame_timecode TEXT NOT NULL DEFAULT '',
                        embedding_model TEXT NOT NULL DEFAULT '',
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding vector({dimensions}) NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                ).format(table=table, dimensions=sql.Literal(dimensions))
            )
            for column, ddl in [
                ("file_id", "TEXT NOT NULL DEFAULT ''"),
                ("media_type", "TEXT NOT NULL DEFAULT 'unknown'"),
                ("media_url", "TEXT NOT NULL DEFAULT ''"),
                ("title", "TEXT NOT NULL DEFAULT ''"),
                ("caption", "TEXT NOT NULL DEFAULT ''"),
                ("frame_index", "INTEGER NOT NULL DEFAULT 0"),
                ("frame_timecode", "TEXT NOT NULL DEFAULT ''"),
                ("embedding_model", "TEXT NOT NULL DEFAULT ''"),
            ]:
                cur.execute(
                    sql.SQL("ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} " + ddl).format(
                        table=table,
                        column=sql.Identifier(column),
                    )
                )
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {index} "
                    "ON {table} (namespace, scope, thread_id, media_type, source_type, source_key, file_id)"
                ).format(
                    index=sql.Identifier(f"{table_name}_scope_idx"),
                    table=table,
                )
            )
            if hnsw_enabled:
                try:
                    cur.execute(
                        sql.SQL(
                            "CREATE INDEX IF NOT EXISTS {index} "
                            "ON {table} USING hnsw (embedding vector_cosine_ops)"
                        ).format(
                            index=sql.Identifier(f"{table_name}_embedding_hnsw_idx"),
                            table=table,
                        )
                    )
                except Exception as exc:
                    print(f"WARNING: vision pgvector HNSW index unavailable; semantic media search will still work: {exc}")


def _delete_source_sync(
    *,
    namespace: str,
    scope: str,
    thread_id: str,
    source_type: str,
    source_key: str,
) -> None:
    table = _table_identifier()
    with psycopg.connect(_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    DELETE FROM {table}
                    WHERE namespace = %s
                      AND scope = %s
                      AND COALESCE(thread_id, '') = %s
                      AND source_type = %s
                      AND source_key = %s
                    """
                ).format(table=table),
                (namespace, scope, thread_id or "", source_type, source_key),
            )
        conn.commit()


def _source_digest_match_sync(
    *,
    namespace: str,
    scope: str,
    thread_id: str,
    source_type: str,
    source_key: str,
    source_digest: str,
) -> dict[str, Any] | None:
    try:
        _require_psycopg()
        table = _table_identifier()
        with psycopg.connect(_database_url()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        SELECT id, source_key, title, chunk_count, metadata, updated_at
                        FROM {table}
                        WHERE namespace = %s
                          AND scope = %s
                          AND COALESCE(thread_id, '') = %s
                          AND source_type = %s
                          AND source_key = %s
                          AND is_catalog = true
                          AND metadata->>'source_digest' = %s
                        ORDER BY updated_at DESC
                        LIMIT 1
                        """
                    ).format(table=table),
                    (namespace, scope, thread_id or "", source_type, source_key, source_digest),
                )
                row = cur.fetchone()
                if not row:
                    return None
                record = {
                    "id": row[0],
                    "source_key": row[1],
                    "title": row[2],
                    "chunk_count": row[3],
                    "metadata": row[4] if isinstance(row[4], dict) else {},
                    "updated_at": row[5].isoformat() if hasattr(row[5], "isoformat") else str(row[5] or ""),
                }
                return record
    except Exception:
        return None


def _ensure_queue_schema_sync() -> None:
    _require_psycopg()
    table_name = _queue_table_name()
    table = sql.Identifier(table_name)
    with psycopg.connect(_database_url(), autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {table} (
                        id TEXT PRIMARY KEY,
                        namespace TEXT NOT NULL,
                        scope TEXT NOT NULL,
                        thread_id TEXT,
                        source_type TEXT NOT NULL,
                        source_key TEXT NOT NULL,
                        title TEXT,
                        payload JSONB NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending',
                        attempts INTEGER NOT NULL DEFAULT 0,
                        last_error TEXT NOT NULL DEFAULT '',
                        progress JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        available_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                ).format(table=table)
            )
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {index} ON {table} (status, available_at, created_at)"
                ).format(index=sql.Identifier(f"{table_name}_status_idx"), table=table)
            )
            cur.execute(
                sql.SQL("ALTER TABLE {table} ADD COLUMN IF NOT EXISTS progress JSONB NOT NULL DEFAULT '{{}}'::jsonb").format(table=table)
            )


def _enqueue_memory_record_sync(payload: dict[str, Any]) -> str:
    _require_psycopg()
    _ensure_queue_schema_sync()
    table = _queue_table_identifier()
    job_id = _record_id(
        str(payload.get("source_type") or "memory"),
        str(payload.get("source_key") or ""),
        str(payload.get("thread_id") or ""),
        str(payload.get("scope") or "thread"),
        0,
    )
    with psycopg.connect(_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {table} (
                        id, namespace, scope, thread_id, source_type, source_key,
                        title, payload, status, attempts, last_error, available_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'pending', 0, '', now())
                    ON CONFLICT (id) DO UPDATE SET
                        namespace = EXCLUDED.namespace,
                        scope = EXCLUDED.scope,
                        thread_id = EXCLUDED.thread_id,
                        source_type = EXCLUDED.source_type,
                        source_key = EXCLUDED.source_key,
                        title = EXCLUDED.title,
                        payload = EXCLUDED.payload,
                        status = CASE
                            WHEN {table}.status = 'running' THEN {table}.status
                            WHEN {table}.status = 'done' AND EXCLUDED.payload->>'dedupe_done' = 'true' THEN {table}.status
                            ELSE 'pending'
                        END,
                        last_error = CASE
                            WHEN {table}.status = 'done' AND EXCLUDED.payload->>'dedupe_done' = 'true' THEN {table}.last_error
                            ELSE ''
                        END,
                        updated_at = now()
                    """
                ).format(table=table),
                (
                    job_id,
                    str(payload.get("namespace") or "alpharavis"),
                    str(payload.get("scope") or "thread"),
                    str(payload.get("thread_id") or ""),
                    str(payload.get("source_type") or "memory"),
                    str(payload.get("source_key") or ""),
                    str(payload.get("title") or "")[:500],
                    Jsonb(payload),
                ),
            )
        conn.commit()
    return job_id


async def enqueue_memory_record(
    *,
    source_type: str,
    source_key: str,
    title: str,
    content: str,
    thread_id: str = "",
    thread_key: str = "",
    scope: str = "thread",
    namespace: str = "alpharavis",
    metadata: dict[str, Any] | None = None,
) -> str:
    if not is_enabled():
        return ""
    source_type = re.sub(r"[^a-zA-Z0-9_-]+", "_", source_type.strip().lower())[:80] or "memory"
    source_key = str(source_key or "").strip()
    source_digest = _content_digest(content)
    if (
        _env_bool("ALPHARAVIS_PGVECTOR_DEDUP_SOURCES", "true")
        and source_key
        and source_digest
    ):
        match = await asyncio.to_thread(
            _source_digest_match_sync,
            namespace=namespace,
            scope=scope or "thread",
            thread_id=thread_id or "",
            source_type=source_type,
            source_key=source_key,
            source_digest=source_digest,
        )
        if match:
            return f"deduped:{source_type}:{source_key}:{int(match.get('chunk_count') or 0)}"
    payload = {
        "source_type": source_type,
        "source_key": source_key,
        "title": title,
        "content": content,
        "thread_id": thread_id,
        "thread_key": thread_key,
        "scope": scope,
        "namespace": namespace,
        "metadata": metadata or {},
    }
    return await asyncio.to_thread(_enqueue_memory_record_sync, payload)


async def enqueue_media_analysis_record(
    *,
    media_url: str,
    user_goal: str = "",
    mode: str = "index",
    media_type: str = "unknown",
    source_key: str = "",
    title: str = "",
    model_id: str = "",
    thread_id: str = "",
    thread_key: str = "",
    scope: str = "thread",
    namespace: str = "alpharavis",
    metadata: dict[str, Any] | None = None,
) -> str:
    if not is_enabled():
        return ""
    media_source_key = str(source_key or media_url or "").strip()
    if not media_source_key:
        raise VectorMemoryError("source_key or media_url is required for queued media analysis.")
    model_card_id = _media_model_card_id(model_id)
    index_version = _media_index_version()
    chunking_hash = _media_chunking_config_hash()
    index_key = hashlib.sha256(
        f"{media_source_key}|{model_card_id}|{index_version}|{chunking_hash}".encode("utf-8")
    ).hexdigest()[:32]
    payload = {
        "job_kind": "media_analysis",
        "source_type": "media_analysis",
        "source_key": index_key,
        "media_source_key": media_source_key,
        "title": title or media_source_key,
        "content": user_goal or title or media_source_key,
        "media_url": media_url,
        "user_goal": user_goal,
        "mode": mode,
        "media_type": media_type,
        "model_id": model_id or model_card_id,
        "model_card_id": model_card_id,
        "index_version": index_version,
        "chunking_config_hash": chunking_hash,
        "dedupe_done": True,
        "thread_id": thread_id,
        "thread_key": thread_key,
        "scope": scope,
        "namespace": namespace,
        "metadata": metadata or {},
    }
    return await asyncio.to_thread(_enqueue_memory_record_sync, payload)


def _insert_chunk_sync(
    *,
    record_id: str,
    namespace: str,
    scope: str,
    thread_id: str,
    thread_key: str,
    source_type: str,
    source_key: str,
    title: str,
    chunk: str,
    catalog_text: str,
    preview: str,
    chunk_index: int,
    chunk_count: int,
    is_catalog: bool,
    embedding_model: str,
    metadata: dict[str, Any],
    embedding: list[float],
) -> None:
    table = _table_identifier()
    vector = _vector_literal(embedding)
    with psycopg.connect(_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {table} (
                        id, namespace, scope, thread_id, thread_key, source_type,
                        source_key, title, content, chunk_text, catalog_text, preview_text,
                        chunk_index, chunk_count, is_catalog, embedding_model, metadata, embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (id) DO UPDATE SET
                        namespace = EXCLUDED.namespace,
                        scope = EXCLUDED.scope,
                        thread_id = EXCLUDED.thread_id,
                        thread_key = EXCLUDED.thread_key,
                        source_type = EXCLUDED.source_type,
                        source_key = EXCLUDED.source_key,
                        title = EXCLUDED.title,
                        content = EXCLUDED.content,
                        chunk_text = EXCLUDED.chunk_text,
                        catalog_text = EXCLUDED.catalog_text,
                        preview_text = EXCLUDED.preview_text,
                        chunk_index = EXCLUDED.chunk_index,
                        chunk_count = EXCLUDED.chunk_count,
                        is_catalog = EXCLUDED.is_catalog,
                        embedding_model = EXCLUDED.embedding_model,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding,
                        updated_at = now()
                    """
                ).format(table=table),
                (
                    record_id,
                    namespace,
                    scope,
                    thread_id or "",
                    thread_key or "",
                    source_type,
                    source_key,
                    title[:500],
                    chunk,
                    chunk,
                    catalog_text,
                    preview,
                    chunk_index,
                    chunk_count,
                    is_catalog,
                    embedding_model,
                    Jsonb(metadata or {}),
                    vector,
                ),
            )
        conn.commit()


async def upsert_memory_record(
    *,
    source_type: str,
    source_key: str,
    title: str,
    content: str,
    thread_id: str = "",
    thread_key: str = "",
    scope: str = "thread",
    namespace: str = "alpharavis",
    metadata: dict[str, Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], Any] | None = None,
) -> str:
    if not is_enabled():
        return ""

    source_type = re.sub(r"[^a-zA-Z0-9_-]+", "_", source_type.strip().lower())[:80] or "memory"
    source_key = str(source_key or "").strip()
    if not source_key:
        raise VectorMemoryError("source_key is required for vector memory indexing.")

    chunks = chunk_text(content, source_type=source_type, title=title, metadata=metadata) if _env_bool("ALPHARAVIS_PGVECTOR_STORE_FULL_CHUNKS", "true") else [
        content[:_preview_chars()].strip()
    ]
    if not chunks:
        raise VectorMemoryError("content is required for vector memory indexing.")

    metadata = metadata or {}
    source_digest = _content_digest(content)
    if _env_bool("ALPHARAVIS_PGVECTOR_DEDUP_SOURCES", "true"):
        match = await asyncio.to_thread(
            _source_digest_match_sync,
            namespace=namespace,
            scope=scope or "thread",
            thread_id=thread_id or "",
            source_type=source_type,
            source_key=source_key,
            source_digest=source_digest,
        )
        if match:
            if progress_callback is not None:
                result = progress_callback(
                    {
                        "event": "large_ingest.deduped",
                        "source_type": source_type,
                        "source_key": source_key,
                        "chunk_count": int(match.get("chunk_count") or len(chunks)),
                        "source_digest": source_digest,
                    }
                )
                if asyncio.iscoroutine(result):
                    await result
            return f"deduped:{source_type}:{source_key}:{int(match.get('chunk_count') or len(chunks))}"

    first_embedding = await embed_text(f"{title.strip()}\n\n{chunks[0]}".strip())
    dimensions = len(first_embedding.vector)
    await asyncio.to_thread(_ensure_schema_sync, dimensions)
    await asyncio.to_thread(
        _delete_source_sync,
        namespace=namespace,
        scope=scope or "thread",
        thread_id=thread_id or "",
        source_type=source_type,
        source_key=source_key,
    )

    chunk_count = len(chunks)
    catalog_text = build_catalog_text(
        source_type=source_type,
        source_key=source_key,
        title=title or source_key,
        content=content,
        chunks=chunks,
        thread_id=thread_id or "",
        thread_key=thread_key or "",
        scope=scope or "thread",
        metadata=metadata,
    )

    if _catalog_enabled():
        catalog_embedding = await embed_text(catalog_text)
        if len(catalog_embedding.vector) != dimensions:
            raise VectorMemoryError(
                f"Catalog embedding dimension differs from chunk embedding: {dimensions} -> {len(catalog_embedding.vector)}"
            )
        await asyncio.to_thread(
            _insert_chunk_sync,
            record_id=_record_id(source_type, source_key, thread_id or "", scope or "thread", -1),
            namespace=namespace,
            scope=scope or "thread",
            thread_id=thread_id or "",
            thread_key=thread_key or "",
            source_type=source_type,
            source_key=source_key,
            title=f"Catalog: {title or source_key}",
            chunk=catalog_text,
            catalog_text=catalog_text,
            preview=catalog_text[:_preview_chars()],
            chunk_index=-1,
            chunk_count=chunk_count,
            is_catalog=True,
            embedding_model=catalog_embedding.model,
            metadata={
                **metadata,
                "is_catalog": True,
                "chunk_count": chunk_count,
                "source_text_chars": len(content),
                "source_digest": source_digest,
                "source_digest_algorithm": "sha256-normalized-text",
            },
            embedding=catalog_embedding.vector,
        )

    for index, chunk in enumerate(chunks):
        embedding = first_embedding if index == 0 else await embed_text(f"{title.strip()}\n\n{chunk}".strip())
        if len(embedding.vector) != dimensions:
            raise VectorMemoryError(
                f"Embedding dimension changed within one record: {dimensions} -> {len(embedding.vector)}"
            )
        record_id = _record_id(source_type, source_key, thread_id or "", scope or "thread", index)
        chunk_metadata = {
            **metadata,
            "chunk_index": index,
            "chunk_count": chunk_count,
            "source_text_chars": len(content),
            "source_digest": source_digest,
            "chunk_digest": _content_digest(chunk),
            "digest_algorithm": "sha256-normalized-text",
        }
        await asyncio.to_thread(
            _insert_chunk_sync,
            record_id=record_id,
            namespace=namespace,
            scope=scope or "thread",
            thread_id=thread_id or "",
            thread_key=thread_key or "",
            source_type=source_type,
            source_key=source_key,
            title=title or source_key,
            chunk=chunk,
            catalog_text="",
            preview=chunk[:_preview_chars()],
            chunk_index=index,
            chunk_count=chunk_count,
            is_catalog=False,
            embedding_model=embedding.model,
            metadata=chunk_metadata,
            embedding=embedding.vector,
        )
        if progress_callback is not None:
            result = progress_callback(
                {
                    "event": "large_ingest.chunk_indexed",
                    "source_type": source_type,
                    "source_key": source_key,
                    "chunk_index": index,
                    "chunk_number": index + 1,
                    "chunk_count": chunk_count,
                    "chunk_chars": len(chunk),
                    "chunk_digest": chunk_metadata["chunk_digest"],
                    "source_digest": source_digest,
                }
            )
            if asyncio.iscoroutine(result):
                await result
    return f"{source_type}:{source_key}:{chunk_count}"


def _insert_vision_sync(
    *,
    record_id: str,
    namespace: str,
    scope: str,
    thread_id: str,
    thread_key: str,
    source_type: str,
    source_key: str,
    file_id: str,
    media_type: str,
    media_url: str,
    title: str,
    caption: str,
    frame_index: int,
    frame_timecode: str,
    embedding_model: str,
    metadata: dict[str, Any],
    embedding: list[float],
) -> None:
    table = _vision_table_identifier()
    vector = _vector_literal(embedding)
    with psycopg.connect(_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {table} (
                        id, namespace, scope, thread_id, thread_key, source_type,
                        source_key, file_id, media_type, media_url, title, caption,
                        frame_index, frame_timecode, embedding_model, metadata, embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (id) DO UPDATE SET
                        namespace = EXCLUDED.namespace,
                        scope = EXCLUDED.scope,
                        thread_id = EXCLUDED.thread_id,
                        thread_key = EXCLUDED.thread_key,
                        source_type = EXCLUDED.source_type,
                        source_key = EXCLUDED.source_key,
                        file_id = EXCLUDED.file_id,
                        media_type = EXCLUDED.media_type,
                        media_url = EXCLUDED.media_url,
                        title = EXCLUDED.title,
                        caption = EXCLUDED.caption,
                        frame_index = EXCLUDED.frame_index,
                        frame_timecode = EXCLUDED.frame_timecode,
                        embedding_model = EXCLUDED.embedding_model,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding,
                        updated_at = now()
                    """
                ).format(table=table),
                (
                    record_id,
                    namespace,
                    scope or "thread",
                    thread_id or "",
                    thread_key or "",
                    source_type,
                    source_key,
                    file_id or "",
                    media_type or "unknown",
                    media_url or "",
                    title[:500],
                    caption,
                    int(frame_index),
                    frame_timecode or "",
                    embedding_model,
                    Jsonb(metadata or {}),
                    vector,
                ),
            )
        conn.commit()


async def upsert_media_record(
    *,
    source_type: str,
    source_key: str,
    media_type: str,
    media_url: str = "",
    file_id: str = "",
    title: str = "",
    caption: str = "",
    thread_id: str = "",
    thread_key: str = "",
    scope: str = "thread",
    namespace: str = "alpharavis",
    frame_index: int = 0,
    frame_timecode: str = "",
    metadata: dict[str, Any] | None = None,
) -> str:
    if not vision_is_enabled():
        return ""

    source_type = re.sub(r"[^a-zA-Z0-9_-]+", "_", source_type.strip().lower())[:80] or "media"
    media_type = re.sub(r"[^a-zA-Z0-9_-]+", "_", media_type.strip().lower())[:40] or "unknown"
    source_key = str(source_key or file_id or media_url or "").strip()
    if not source_key:
        raise VectorMemoryError("source_key, file_id, or media_url is required for media vector indexing.")

    title = title or source_key
    caption = (caption or title or source_key).strip()
    embedding = await embed_media(media_url=media_url, caption=caption, media_type=media_type)
    await asyncio.to_thread(_ensure_vision_schema_sync, len(embedding.vector))
    raw_id = f"{source_type}:{source_key}:{file_id}:{thread_id}:{scope}:{media_type}:{frame_index}:{frame_timecode}"
    record_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:32]
    await asyncio.to_thread(
        _insert_vision_sync,
        record_id=record_id,
        namespace=namespace,
        scope=scope or "thread",
        thread_id=thread_id or "",
        thread_key=thread_key or "",
        source_type=source_type,
        source_key=source_key,
        file_id=file_id or "",
        media_type=media_type,
        media_url=media_url or "",
        title=title,
        caption=caption,
        frame_index=int(frame_index),
        frame_timecode=frame_timecode or "",
        embedding_model=embedding.model,
        metadata=metadata or {},
        embedding=embedding.vector,
    )
    return f"{source_type}:{source_key}:{media_type}:{frame_index}"


def _media_index_status_sync(
    *,
    namespace: str,
    thread_id: str,
    source_key: str,
    media_type: str,
    include_other_threads: bool,
    limit: int,
) -> list[dict[str, Any]]:
    _require_psycopg()
    table = _vision_table_identifier()
    where = ["namespace = %s"]
    params: list[Any] = [namespace]

    if not include_other_threads:
        where.append("(thread_id = %s OR thread_id = '' OR thread_id IS NULL)")
        params.append(thread_id or "")
    if source_key:
        where.append("(source_key = %s OR payload->>'media_source_key' = %s OR payload->>'media_url' = %s)")
        params.extend([source_key, source_key, source_key])
    if media_type and media_type != "all":
        if media_type == "video":
            where.append("(media_type = %s OR source_type = 'video_frame')")
            params.append(media_type)
        else:
            where.append("media_type = %s")
            params.append(media_type)
    params.append(limit)

    query = sql.SQL(
        """
        SELECT
            id, scope, thread_id, thread_key, source_type, source_key, file_id,
            media_type, media_url, title, caption, frame_index, frame_timecode,
            embedding_model, metadata, created_at, updated_at
        FROM {table}
        WHERE {where_clause}
        ORDER BY updated_at DESC
        LIMIT %s
        """
    ).format(table=table, where_clause=sql.SQL(" AND ").join(sql.SQL(item) for item in where))

    with psycopg.connect(_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

    records = []
    for row in rows:
        record = dict(zip(columns, row))
        if hasattr(record.get("created_at"), "isoformat"):
            record["created_at"] = record["created_at"].isoformat()
        if hasattr(record.get("updated_at"), "isoformat"):
            record["updated_at"] = record["updated_at"].isoformat()
        records.append(record)
    return records


async def media_index_status(
    *,
    thread_id: str = "",
    source_key: str = "",
    media_type: str = "all",
    include_other_threads: bool = False,
    limit: int = 50,
    namespace: str = "alpharavis",
) -> list[dict[str, Any]]:
    if not vision_is_enabled():
        return []
    media_type = re.sub(r"[^a-zA-Z0-9_-]+", "_", media_type.strip().lower())[:40] or "all"
    return await asyncio.to_thread(
        _media_index_status_sync,
        namespace=namespace,
        thread_id=thread_id,
        source_key=source_key.strip(),
        media_type=media_type,
        include_other_threads=include_other_threads,
        limit=max(1, min(int(limit), int(os.getenv("ALPHARAVIS_VISION_INDEX_STATUS_LIMIT", "50")))),
    )


def _media_queue_status_sync(
    *,
    thread_id: str,
    source_key: str,
    include_other_threads: bool,
    limit: int,
) -> list[dict[str, Any]]:
    _require_psycopg()
    _ensure_queue_schema_sync()
    table = _queue_table_identifier()
    where = ["source_type = 'media_analysis'"]
    params: list[Any] = []
    if not include_other_threads:
        where.append("(thread_id = %s OR thread_id = '' OR thread_id IS NULL)")
        params.append(thread_id or "")
    if source_key:
        where.append("source_key = %s")
        params.append(source_key)
    params.append(limit)
    query = sql.SQL(
        """
        SELECT id, namespace, scope, thread_id, source_type, source_key, title,
               payload, status, attempts, last_error, created_at, updated_at
        FROM {table}
        WHERE {where_clause}
        ORDER BY updated_at DESC
        LIMIT %s
        """
    ).format(table=table, where_clause=sql.SQL(" AND ").join(sql.SQL(item) for item in where))
    with psycopg.connect(_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
    records = []
    for row in rows:
        record = dict(zip(columns, row))
        if hasattr(record.get("created_at"), "isoformat"):
            record["created_at"] = record["created_at"].isoformat()
        if hasattr(record.get("updated_at"), "isoformat"):
            record["updated_at"] = record["updated_at"].isoformat()
        records.append(record)
    return records


async def media_queue_status(
    *,
    thread_id: str = "",
    source_key: str = "",
    include_other_threads: bool = False,
    limit: int = 50,
) -> list[dict[str, Any]]:
    if not is_enabled():
        return []
    return await asyncio.to_thread(
        _media_queue_status_sync,
        thread_id=thread_id,
        source_key=source_key.strip(),
        include_other_threads=include_other_threads,
        limit=max(1, min(int(limit), int(os.getenv("ALPHARAVIS_VISION_INDEX_STATUS_LIMIT", "50")))),
    )


def _claim_embedding_jobs_sync(limit: int, max_attempts: int) -> list[dict[str, Any]]:
    _require_psycopg()
    _ensure_queue_schema_sync()
    table = _queue_table_identifier()
    stale_after_seconds = max(
        60,
        int(os.getenv("ALPHARAVIS_EMBEDDING_JOB_STALE_AFTER_SECONDS", "900")),
    )
    with psycopg.connect(_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    WITH claimed AS (
                        SELECT id
                        FROM {table}
                        WHERE (
                            status IN ('pending', 'failed')
                            OR (
                                status = 'running'
                                AND updated_at <= now() - (%s * interval '1 second')
                            )
                        )
                          AND attempts < %s
                          AND available_at <= now()
                        ORDER BY created_at
                        LIMIT %s
                        FOR UPDATE SKIP LOCKED
                    )
                    UPDATE {table}
                    SET status = 'running',
                        attempts = attempts + 1,
                        last_error = CASE
                            WHEN status = 'running'
                            THEN %s
                            ELSE last_error
                        END,
                        updated_at = now()
                    WHERE id IN (SELECT id FROM claimed)
                    RETURNING id, payload, attempts
                    """
                ).format(table=table),
                (
                    stale_after_seconds,
                    max_attempts,
                    limit,
                    f"Reclaimed stale running job after {stale_after_seconds}s.",
                ),
            )
            rows = cur.fetchall()
        conn.commit()
    return [{"id": row[0], "payload": row[1], "attempts": row[2]} for row in rows]


def _update_embedding_job_progress_sync(job_id: str, progress: dict[str, Any]) -> None:
    _require_psycopg()
    _ensure_queue_schema_sync()
    table = _queue_table_identifier()
    with psycopg.connect(_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    UPDATE {table}
                    SET progress = progress || %s::jsonb,
                        updated_at = now()
                    WHERE id = %s
                    """
                ).format(table=table),
                (Jsonb(progress or {}), job_id),
            )
        conn.commit()


def _finish_embedding_job_sync(job_id: str, *, ok: bool, error: str = "", result: Any = None) -> None:
    _require_psycopg()
    table = _queue_table_identifier()
    status = "done" if ok else "failed"
    final_progress = {
        "ok": bool(ok),
        "finished_at": int(time.time()),
        **({"result": str(result)[:1000]} if result is not None else {}),
    }
    with psycopg.connect(_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    UPDATE {table}
                    SET status = %s,
                        last_error = %s,
                        progress = progress || %s::jsonb,
                        available_at = CASE WHEN %s THEN now() ELSE now() + interval '5 minutes' END,
                        updated_at = now()
                    WHERE id = %s
                    """
                ).format(table=table),
                (status, error[:2000], Jsonb(final_progress), ok, job_id),
            )
        conn.commit()


def _queue_job_planned_chunks(payload: dict[str, Any]) -> int:
    if not isinstance(payload, dict):
        return 0
    if payload.get("job_kind") == "media_analysis":
        return 0
    try:
        chunks = chunk_text(
            str(payload.get("content") or ""),
            source_type=str(payload.get("source_type") or ""),
            title=str(payload.get("title") or ""),
            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
        )
        return len(chunks)
    except Exception:
        return 0


def _source_indexed_chunk_count_sync(*, namespace: str, scope: str, thread_id: str, source_type: str, source_key: str) -> int:
    try:
        _require_psycopg()
        table = _table_identifier()
        with psycopg.connect(_database_url()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        SELECT COALESCE(MAX(chunk_count), COUNT(*))
                        FROM {table}
                        WHERE namespace = %s
                          AND scope = %s
                          AND COALESCE(thread_id, '') = %s
                          AND source_type = %s
                          AND source_key = %s
                          AND is_catalog = false
                        """
                    ).format(table=table),
                    (namespace, scope, thread_id or "", source_type, source_key),
                )
                row = cur.fetchone()
                return int(row[0] or 0) if row else 0
    except Exception:
        return 0


def _queue_stats_sync() -> dict[str, Any]:
    _require_psycopg()
    _ensure_queue_schema_sync()
    table = _queue_table_identifier()
    with psycopg.connect(_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT status, COUNT(*) FROM {table} GROUP BY status ORDER BY status").format(table=table)
            )
            counts = {str(status): int(count) for status, count in cur.fetchall()}
            cur.execute(
                sql.SQL(
                    """
                    SELECT id, namespace, scope, thread_id, source_type, source_key, title,
                           status, attempts, last_error, payload, progress, updated_at
                    FROM {table}
                    WHERE status IN ('pending', 'failed', 'running')
                    ORDER BY updated_at DESC
                    LIMIT 12
                    """
                ).format(table=table)
            )
            columns = [desc[0] for desc in cur.description]
            rows = [dict(zip(columns, row)) for row in cur.fetchall()]
    for row in rows:
        if hasattr(row.get("updated_at"), "isoformat"):
            row["updated_at"] = row["updated_at"].isoformat()
    source_progress: list[dict[str, Any]] = []
    for row in rows:
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        progress = row.get("progress") if isinstance(row.get("progress"), dict) else {}
        planned = int(progress.get("chunk_count") or _queue_job_planned_chunks(payload))
        completed = int(progress.get("chunk_number") or 0)
        if str(row.get("status") or "") == "done" and planned:
            completed = planned
        if str(row.get("status") or "") == "running":
            completed = max(0, min(completed, planned or completed))
        source_progress.append(
            {
                "id": row.get("id"),
                "status": row.get("status"),
                "source_type": row.get("source_type"),
                "source_key": row.get("source_key"),
                "title": row.get("title"),
                "thread_id": row.get("thread_id"),
                "planned_chunks": planned,
                "completed_chunks": completed,
                "progress": round(completed / planned, 4) if planned else None,
                "last_event": progress.get("event", ""),
                "last_error": row.get("last_error", ""),
                "updated_at": row.get("updated_at"),
            }
        )
    return {"table": _queue_table_name(), "counts": counts, "recent_active": rows, "source_progress": source_progress}


async def queue_stats() -> dict[str, Any]:
    if not is_enabled():
        return {"enabled": False}
    return await asyncio.to_thread(_queue_stats_sync)


async def run_embedding_jobs(limit: int = 10) -> dict[str, Any]:
    if not is_enabled():
        return {"ok": False, "message": "pgvector memory is disabled"}

    limit = max(1, min(int(limit), int(os.getenv("ALPHARAVIS_EMBEDDING_JOB_MAX_BATCH", "25"))))
    max_attempts = max(1, int(os.getenv("ALPHARAVIS_EMBEDDING_JOB_MAX_ATTEMPTS", "3")))
    jobs = await asyncio.to_thread(_claim_embedding_jobs_sync, limit, max_attempts)
    results = []
    for job in jobs:
        payload = dict(job.get("payload") or {})
        job_id = str(job["id"])
        try:
            if payload.get("job_kind") == "media_analysis":
                result = await _run_media_analysis_job(payload)
            else:
                async def progress_callback(event: dict[str, Any], *, _job_id: str = job_id):
                    await asyncio.to_thread(_update_embedding_job_progress_sync, _job_id, dict(event))

                result = await upsert_memory_record(**payload, progress_callback=progress_callback)
            await asyncio.to_thread(_finish_embedding_job_sync, job_id, ok=True, result=result)
            results.append({"id": job_id, "ok": True, "result": result})
        except Exception as exc:
            await asyncio.to_thread(_finish_embedding_job_sync, job_id, ok=False, error=str(exc))
            results.append({"id": job_id, "ok": False, "error": str(exc)[:500]})

    return {
        "ok": all(item["ok"] for item in results) if results else True,
        "processed": len(results),
        "results": results,
        "stats": await queue_stats(),
    }


async def _run_media_analysis_job(payload: dict[str, Any]) -> dict[str, Any]:
    if not vision_is_enabled():
        raise VectorMemoryError("Vision/media vector memory is disabled. Set ALPHARAVIS_ENABLE_VISION_VECTOR_MEMORY=true.")
    try:
        from media_analysis import prepare_media_for_model as _prepare_video
    except Exception as exc:  # pragma: no cover - optional runtime helper
        raise VectorMemoryError(f"media_analysis helper unavailable: {exc}") from exc

    prepared = await _prepare_video(
        media_url=str(payload.get("media_url") or ""),
        user_goal=str(payload.get("user_goal") or payload.get("content") or ""),
        mode=str(payload.get("mode") or "index"),
        media_type=str(payload.get("media_type") or "unknown"),
        source_key=str(payload.get("source_key") or ""),
        title=str(payload.get("title") or ""),
        model_id=str(payload.get("model_id") or ""),
        thread_id=str(payload.get("thread_id") or ""),
    )
    if not prepared.get("ok"):
        raise VectorMemoryError(str(prepared.get("error") or prepared))

    indexed = []
    errors = []
    source_key = str(prepared.get("source_key") or payload.get("media_source_key") or payload.get("source_key") or "")
    media_url = str(prepared.get("media_url") or payload.get("media_url") or "")
    title = str(prepared.get("title") or payload.get("title") or source_key)
    user_goal = str(payload.get("user_goal") or "")
    for frame in prepared.get("frames", [])[: int(os.getenv("ALPHARAVIS_VIDEO_ANALYSIS_MAX_FRAMES", "100"))]:
        frame_url = str(frame.get("public_url") or "")
        if not frame_url:
            continue
        frame_index = int(frame.get("frame_index") or 0)
        timecode = str(frame.get("timecode") or "")
        caption = (
            f"Sampled frame from video `{title}` at {timecode}. "
            f"Original video URL: {media_url}. User goal: {user_goal[:500]}"
        )
        try:
            vector_key = await upsert_media_record(
                source_type="video_frame",
                source_key=source_key,
                file_id=str(prepared.get("manifest_path") or ""),
                media_type="image",
                media_url=frame_url,
                title=title,
                caption=caption,
                thread_id=str(payload.get("thread_id") or ""),
                thread_key=str(payload.get("thread_key") or payload.get("thread_id") or ""),
                scope="global",
                namespace=str(payload.get("namespace") or "alpharavis"),
                frame_index=frame_index,
                frame_timecode=timecode,
                metadata={
                    **(payload.get("metadata") or {}),
                    "parent_media_url": media_url,
                    "frame": frame,
                    "manifest_path": prepared.get("manifest_path", ""),
                    "manifest_url": prepared.get("manifest_url", ""),
                    "analysis_mode": prepared.get("mode", "index"),
                    "user_goal": user_goal[:1000],
                    "queued_job": True,
                    "origin_thread_id": str(payload.get("thread_id") or ""),
                    "origin_thread_key": str(payload.get("thread_key") or ""),
                    "model_card_id": str(payload.get("model_card_id") or ""),
                    "index_version": str(payload.get("index_version") or ""),
                    "chunking_config_hash": str(payload.get("chunking_config_hash") or ""),
                },
            )
            indexed.append({"frame_index": frame_index, "timecode": timecode, "vector_key": vector_key})
        except Exception as exc:
            errors.append(f"frame {frame_index}: {exc}")

    if errors and not indexed:
        raise VectorMemoryError("Media analysis prepared frames but indexing failed: " + " | ".join(errors[:5]))
    return {
        "prepared": {
            "source_key": source_key,
            "media_url": media_url,
            "frame_count": prepared.get("frame_count", 0),
            "manifest_path": prepared.get("manifest_path", ""),
            "manifest_url": prepared.get("manifest_url", ""),
        },
        "indexed_frame_count": len(indexed),
        "indexed_frames": indexed[:20],
        "errors": errors[:20],
    }


def _search_sync(
    *,
    query_embedding: list[float],
    namespace: str,
    thread_id: str,
    source_type: str,
    source_keys: list[str],
    include_other_threads: bool,
    limit: int,
) -> list[dict[str, Any]]:
    _require_psycopg()
    table = _table_identifier()
    vector = _vector_literal(query_embedding)
    distance_threshold = _distance_threshold()
    where = ["namespace = %s"]
    params: list[Any] = [namespace]

    if not include_other_threads:
        where.append("(thread_id = %s OR thread_id = '' OR thread_id IS NULL)")
        params.append(thread_id or "")

    if source_type and source_type != "all":
        where.append("source_type = %s")
        params.append(source_type)

    if source_keys:
        where.append("source_key IN (" + ", ".join(["%s"] * len(source_keys)) + ")")
        params.extend(source_keys)

    if distance_threshold is not None:
        where.append("(embedding <=> %s::vector) <= %s")
        params.extend([vector, distance_threshold])

    params = [vector, vector, *params, vector, limit]
    query = sql.SQL(
        """
        SELECT
            id, scope, thread_id, thread_key, source_type, source_key,
            title, content, chunk_text, catalog_text, preview_text, chunk_index,
            chunk_count, is_catalog, embedding_model, metadata, created_at, updated_at,
            1 - (embedding <=> %s::vector) AS similarity,
            embedding <=> %s::vector AS distance
        FROM {table}
        WHERE {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """
    ).format(table=table, where_clause=sql.SQL(" AND ").join(sql.SQL(item) for item in where))

    with psycopg.connect(_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

    records = []
    for row in rows:
        record = dict(zip(columns, row))
        if hasattr(record.get("created_at"), "isoformat"):
            record["created_at"] = record["created_at"].isoformat()
        if hasattr(record.get("updated_at"), "isoformat"):
            record["updated_at"] = record["updated_at"].isoformat()
        records.append(record)
    return records


def _search_vision_sync(
    *,
    query_embedding: list[float],
    namespace: str,
    thread_id: str,
    media_type: str,
    include_other_threads: bool,
    limit: int,
) -> list[dict[str, Any]]:
    _require_psycopg()
    table = _vision_table_identifier()
    vector = _vector_literal(query_embedding)
    where = ["namespace = %s"]
    params: list[Any] = [namespace]

    if not include_other_threads:
        where.append("(thread_id = %s OR thread_id = '' OR thread_id IS NULL)")
        params.append(thread_id or "")

    if media_type and media_type != "all":
        where.append("media_type = %s")
        params.append(media_type)

    params = [vector, *params, vector, limit]
    query = sql.SQL(
        """
        SELECT
            id, scope, thread_id, thread_key, source_type, source_key, file_id,
            media_type, media_url, title, caption, frame_index, frame_timecode,
            embedding_model, metadata, created_at, updated_at,
            1 - (embedding <=> %s::vector) AS similarity
        FROM {table}
        WHERE {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """
    ).format(table=table, where_clause=sql.SQL(" AND ").join(sql.SQL(item) for item in where))

    with psycopg.connect(_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

    records = []
    for row in rows:
        record = dict(zip(columns, row))
        if hasattr(record.get("created_at"), "isoformat"):
            record["created_at"] = record["created_at"].isoformat()
        if hasattr(record.get("updated_at"), "isoformat"):
            record["updated_at"] = record["updated_at"].isoformat()
        record["source"] = "alpharavis_media_pgvector"
        records.append(record)
    return records


async def semantic_search(
    *,
    query: str,
    thread_id: str = "",
    source_type: str = "all",
    source_key: str = "",
    source_keys: list[str] | None = None,
    include_other_threads: bool = False,
    limit: int = 5,
    namespace: str = "alpharavis",
) -> list[dict[str, Any]]:
    if not is_enabled():
        return []

    query = (query or "").strip()
    if not query:
        raise VectorMemoryError("query is required for semantic vector search.")

    source_type = re.sub(r"[^a-zA-Z0-9_-]+", "_", source_type.strip().lower())[:80] or "all"
    normalized_source_keys = [
        str(item).strip()
        for item in ([source_key] if source_key else []) + list(source_keys or [])
        if str(item).strip()
    ]
    normalized_source_keys = list(dict.fromkeys(normalized_source_keys))[:50]
    query_embedding = await embed_text(query)
    return await asyncio.to_thread(
        _search_sync,
        query_embedding=query_embedding.vector,
        namespace=namespace,
        thread_id=thread_id,
        source_type=source_type,
        source_keys=normalized_source_keys,
        include_other_threads=include_other_threads,
        limit=max(1, min(int(limit), int(os.getenv("ALPHARAVIS_PGVECTOR_SEARCH_LIMIT", "5")))),
    )


def _read_source_chunks_sync(
    *,
    namespace: str,
    thread_id: str,
    source_type: str,
    source_key: str,
    include_other_threads: bool,
    max_chunks: int,
    max_chars: int,
) -> dict[str, Any]:
    _require_psycopg()
    table = _table_identifier()
    where = ["namespace = %s", "source_key = %s", "is_catalog = false"]
    params: list[Any] = [namespace, source_key]
    if source_type and source_type != "all":
        where.append("source_type = %s")
        params.append(source_type)
    if not include_other_threads:
        where.append("(thread_id = %s OR thread_id = '' OR thread_id IS NULL)")
        params.append(thread_id or "")
    params.append(max_chunks)
    query = sql.SQL(
        """
        SELECT id, scope, thread_id, thread_key, source_type, source_key, title,
               chunk_text, preview_text, chunk_index, chunk_count, metadata,
               created_at, updated_at
        FROM {table}
        WHERE {where_clause}
        ORDER BY chunk_index ASC
        LIMIT %s
        """
    ).format(table=table, where_clause=sql.SQL(" AND ").join(sql.SQL(item) for item in where))
    with psycopg.connect(_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

    records = []
    used_chars = 0
    truncated = False
    for row in rows:
        record = dict(zip(columns, row))
        text = str(record.get("chunk_text") or record.get("preview_text") or "")
        remaining = max_chars - used_chars
        if remaining <= 0:
            truncated = True
            break
        if len(text) > remaining:
            text = text[:remaining].rstrip()
            truncated = True
        used_chars += len(text)
        if hasattr(record.get("created_at"), "isoformat"):
            record["created_at"] = record["created_at"].isoformat()
        if hasattr(record.get("updated_at"), "isoformat"):
            record["updated_at"] = record["updated_at"].isoformat()
        record["chunk_text"] = text
        records.append(record)
        if truncated:
            break
    return {
        "source_key": source_key,
        "source_type": source_type,
        "thread_id": thread_id,
        "chunk_count_returned": len(records),
        "max_chunks": max_chunks,
        "max_chars": max_chars,
        "returned_chars": used_chars,
        "truncated": truncated or len(rows) > len(records),
        "chunks": records,
    }


async def read_source_chunks(
    *,
    source_key: str,
    thread_id: str = "",
    source_type: str = "all",
    include_other_threads: bool = False,
    max_chunks: int = 8,
    max_chars: int = 12000,
    namespace: str = "alpharavis",
) -> dict[str, Any]:
    if not is_enabled():
        return {"source_key": source_key, "chunks": [], "warning": "pgvector memory is disabled"}
    source_key = str(source_key or "").strip()
    if not source_key:
        raise VectorMemoryError("source_key is required for source chunk reads.")
    source_type = re.sub(r"[^a-zA-Z0-9_-]+", "_", source_type.strip().lower())[:80] or "all"
    return await asyncio.to_thread(
        _read_source_chunks_sync,
        namespace=namespace,
        thread_id=thread_id,
        source_type=source_type,
        source_key=source_key,
        include_other_threads=include_other_threads,
        max_chunks=max(1, min(int(max_chunks), int(os.getenv("ALPHARAVIS_SOURCE_READ_MAX_CHUNKS", "20")))),
        max_chars=max(200, min(int(max_chars), int(os.getenv("ALPHARAVIS_SOURCE_READ_MAX_CHARS", "30000")))),
    )


async def semantic_media_search(
    *,
    query: str,
    thread_id: str = "",
    media_type: str = "all",
    include_other_threads: bool = False,
    limit: int = 5,
    namespace: str = "alpharavis",
) -> list[dict[str, Any]]:
    if not vision_is_enabled():
        return []

    query = (query or "").strip()
    if not query:
        raise VectorMemoryError("query is required for semantic media vector search.")

    media_type = re.sub(r"[^a-zA-Z0-9_-]+", "_", media_type.strip().lower())[:40] or "all"
    if _env_bool("ALPHARAVIS_VISION_QUERY_USES_VISION_MODEL", "true"):
        query_embedding = await embed_media(media_url="", caption=query, media_type="text")
    else:
        query_embedding = await embed_text(query)
    return await asyncio.to_thread(
        _search_vision_sync,
        query_embedding=query_embedding.vector,
        namespace=namespace,
        thread_id=thread_id,
        media_type=media_type,
        include_other_threads=include_other_threads,
        limit=max(1, min(int(limit), int(os.getenv("ALPHARAVIS_VISION_PGVECTOR_SEARCH_LIMIT", "5")))),
    )
