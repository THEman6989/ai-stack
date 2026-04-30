from __future__ import annotations

import asyncio
import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx

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
    return os.getenv(name, default).lower() in {"1", "true", "yes"}


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


def _chunk_max_chars() -> int:
    return max(1000, int(os.getenv("ALPHARAVIS_PGVECTOR_CHUNK_MAX_CHARS", "6000")))


def _chunk_overlap_chars() -> int:
    max_chars = _chunk_max_chars()
    overlap = max(0, int(os.getenv("ALPHARAVIS_PGVECTOR_CHUNK_OVERLAP_CHARS", "800")))
    return min(overlap, max_chars // 2)


def _preview_chars() -> int:
    return max(200, int(os.getenv("ALPHARAVIS_PGVECTOR_PREVIEW_CHARS", "900")))


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


def _vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(f"{float(value):.9g}" for value in embedding) + "]"


def _record_id(source_type: str, source_key: str, thread_id: str, scope: str, chunk_index: int) -> str:
    raw = f"{source_type}:{source_key}:{thread_id}:{scope}:{chunk_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


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


def chunk_text(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    max_chars = _chunk_max_chars()
    overlap = _chunk_overlap_chars()
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
    payload = {"model": model, "input": text[:_chunk_max_chars()]}
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
    base_url = os.getenv(
        "ALPHARAVIS_VISION_EMBEDDING_BASE_URL",
        os.getenv("ALPHARAVIS_PGVECTOR_EMBEDDING_BASE_URL", os.getenv("OPENAI_API_BASE", "http://litellm:4000/v1")),
    ).rstrip("/")
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
                        status = CASE WHEN {table}.status = 'running' THEN {table}.status ELSE 'pending' END,
                        last_error = '',
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
) -> str:
    if not is_enabled():
        return ""

    source_type = re.sub(r"[^a-zA-Z0-9_-]+", "_", source_type.strip().lower())[:80] or "memory"
    source_key = str(source_key or "").strip()
    if not source_key:
        raise VectorMemoryError("source_key is required for vector memory indexing.")

    chunks = chunk_text(content) if _env_bool("ALPHARAVIS_PGVECTOR_STORE_FULL_CHUNKS", "true") else [
        content[:_preview_chars()].strip()
    ]
    if not chunks:
        raise VectorMemoryError("content is required for vector memory indexing.")

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
    metadata = metadata or {}
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
            metadata={**metadata, "is_catalog": True, "chunk_count": chunk_count, "source_text_chars": len(content)},
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


def _claim_embedding_jobs_sync(limit: int, max_attempts: int) -> list[dict[str, Any]]:
    _require_psycopg()
    _ensure_queue_schema_sync()
    table = _queue_table_identifier()
    with psycopg.connect(_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    WITH claimed AS (
                        SELECT id
                        FROM {table}
                        WHERE status IN ('pending', 'failed')
                          AND attempts < %s
                          AND available_at <= now()
                        ORDER BY created_at
                        LIMIT %s
                        FOR UPDATE SKIP LOCKED
                    )
                    UPDATE {table}
                    SET status = 'running',
                        attempts = attempts + 1,
                        updated_at = now()
                    WHERE id IN (SELECT id FROM claimed)
                    RETURNING id, payload, attempts
                    """
                ).format(table=table),
                (max_attempts, limit),
            )
            rows = cur.fetchall()
        conn.commit()
    return [{"id": row[0], "payload": row[1], "attempts": row[2]} for row in rows]


def _finish_embedding_job_sync(job_id: str, *, ok: bool, error: str = "") -> None:
    _require_psycopg()
    table = _queue_table_identifier()
    status = "done" if ok else "failed"
    with psycopg.connect(_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    UPDATE {table}
                    SET status = %s,
                        last_error = %s,
                        available_at = CASE WHEN %s THEN now() ELSE now() + interval '5 minutes' END,
                        updated_at = now()
                    WHERE id = %s
                    """
                ).format(table=table),
                (status, error[:2000], ok, job_id),
            )
        conn.commit()


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
                    SELECT id, source_type, source_key, title, status, attempts, last_error, updated_at
                    FROM {table}
                    WHERE status IN ('pending', 'failed', 'running')
                    ORDER BY updated_at DESC
                    LIMIT 5
                    """
                ).format(table=table)
            )
            columns = [desc[0] for desc in cur.description]
            rows = [dict(zip(columns, row)) for row in cur.fetchall()]
    for row in rows:
        if hasattr(row.get("updated_at"), "isoformat"):
            row["updated_at"] = row["updated_at"].isoformat()
    return {"table": _queue_table_name(), "counts": counts, "recent_active": rows}


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
            result = await upsert_memory_record(**payload)
            await asyncio.to_thread(_finish_embedding_job_sync, job_id, ok=True)
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


def _search_sync(
    *,
    query_embedding: list[float],
    namespace: str,
    thread_id: str,
    source_type: str,
    include_other_threads: bool,
    limit: int,
) -> list[dict[str, Any]]:
    _require_psycopg()
    table = _table_identifier()
    vector = _vector_literal(query_embedding)
    where = ["namespace = %s"]
    params: list[Any] = [namespace]

    if not include_other_threads:
        where.append("(thread_id = %s OR thread_id = '' OR thread_id IS NULL)")
        params.append(thread_id or "")

    if source_type and source_type != "all":
        where.append("source_type = %s")
        params.append(source_type)

    params = [vector, *params, vector, limit]
    query = sql.SQL(
        """
        SELECT
            id, scope, thread_id, thread_key, source_type, source_key,
            title, content, chunk_text, catalog_text, preview_text, chunk_index,
            chunk_count, is_catalog, embedding_model, metadata, created_at, updated_at,
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
    query_embedding = await embed_text(query)
    return await asyncio.to_thread(
        _search_sync,
        query_embedding=query_embedding.vector,
        namespace=namespace,
        thread_id=thread_id,
        source_type=source_type,
        include_other_threads=include_other_threads,
        limit=max(1, min(int(limit), int(os.getenv("ALPHARAVIS_PGVECTOR_SEARCH_LIMIT", "5")))),
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
