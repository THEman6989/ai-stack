from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
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


def _table_identifier():
    if sql is None:
        raise VectorMemoryError("psycopg.sql is unavailable.")
    return sql.Identifier(_table_name())


def _max_content_chars() -> int:
    return max(500, int(os.getenv("ALPHARAVIS_PGVECTOR_MAX_CONTENT_CHARS", "8000")))


def _vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(f"{float(value):.9g}" for value in embedding) + "]"


def _record_id(source_type: str, source_key: str, thread_id: str, scope: str) -> str:
    raw = f"{source_type}:{source_key}:{thread_id}:{scope}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


async def embed_text(text: str) -> list[float]:
    text = (text or "").strip()
    if not text:
        raise VectorMemoryError("Cannot embed empty text.")

    base_url = os.getenv(
        "ALPHARAVIS_PGVECTOR_EMBEDDING_BASE_URL",
        os.getenv("OPENAI_API_BASE", "http://litellm:4000/v1"),
    ).rstrip("/")
    model = os.getenv("ALPHARAVIS_PGVECTOR_EMBEDDING_MODEL", "memory-embed").strip()
    api_key = os.getenv(
        "ALPHARAVIS_PGVECTOR_EMBEDDING_API_KEY",
        os.getenv("OPENAI_API_KEY", os.getenv("LITELLM_MASTER_KEY", "sk-local-dev")),
    )
    timeout = float(os.getenv("ALPHARAVIS_PGVECTOR_EMBEDDING_TIMEOUT_SECONDS", "20"))
    payload = {"model": model, "input": text[:_max_content_chars()]}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{base_url}/embeddings", headers=headers, json=payload)
    if response.status_code >= 400:
        raise VectorMemoryError(f"Embedding endpoint returned HTTP {response.status_code}: {response.text[:500]}")

    data = response.json()
    try:
        embedding = data["data"][0]["embedding"]
    except Exception as exc:
        raise VectorMemoryError(f"Embedding response did not contain data[0].embedding: {data!r}") from exc

    if not isinstance(embedding, list) or not embedding:
        raise VectorMemoryError("Embedding response contained an empty or invalid vector.")
    return [float(value) for value in embedding]


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
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding vector({dimensions}) NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                ).format(table=table, dimensions=sql.Literal(dimensions))
            )
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {index} "
                    "ON {table} (namespace, scope, thread_id, source_type)"
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


def _upsert_sync(
    *,
    record_id: str,
    namespace: str,
    scope: str,
    thread_id: str,
    thread_key: str,
    source_type: str,
    source_key: str,
    title: str,
    content: str,
    metadata: dict[str, Any],
    embedding: list[float],
) -> None:
    _ensure_schema_sync(len(embedding))
    table = _table_identifier()
    vector = _vector_literal(embedding)

    with psycopg.connect(_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {table} (
                        id, namespace, scope, thread_id, thread_key, source_type,
                        source_key, title, content, metadata, embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (id) DO UPDATE SET
                        namespace = EXCLUDED.namespace,
                        scope = EXCLUDED.scope,
                        thread_id = EXCLUDED.thread_id,
                        thread_key = EXCLUDED.thread_key,
                        source_type = EXCLUDED.source_type,
                        source_key = EXCLUDED.source_key,
                        title = EXCLUDED.title,
                        content = EXCLUDED.content,
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
                    content[:_max_content_chars()],
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

    content = (content or "").strip()
    if not content:
        raise VectorMemoryError("content is required for vector memory indexing.")

    embedding_input = f"{title.strip()}\n\n{content}".strip()
    embedding = await embed_text(embedding_input)
    record_id = _record_id(source_type, source_key, thread_id or "", scope or "thread")
    await asyncio.to_thread(
        _upsert_sync,
        record_id=record_id,
        namespace=namespace,
        scope=scope or "thread",
        thread_id=thread_id or "",
        thread_key=thread_key or "",
        source_type=source_type,
        source_key=source_key,
        title=title or source_key,
        content=content,
        metadata=metadata or {},
        embedding=embedding,
    )
    return record_id


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
            title, content, metadata, created_at, updated_at,
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
        query_embedding=query_embedding,
        namespace=namespace,
        thread_id=thread_id,
        source_type=source_type,
        include_other_threads=include_other_threads,
        limit=max(1, min(int(limit), int(os.getenv("ALPHARAVIS_PGVECTOR_SEARCH_LIMIT", "5")))),
    )
