from __future__ import annotations

import os
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

from rag_api_client import RagApiClientError
from rag_api_client import mirror_text as _rag_mirror_text
from rag_api_client import query_sources as _rag_query_sources


def env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def archive_rag_file_id(archive_key: str) -> str:
    return f"archive:{archive_key}"


def rag_archive_mirror_enabled() -> bool:
    return env_bool("ALPHARAVIS_ENABLE_RAG_ARCHIVE_MIRROR", "false")


def prefer_rag_mirrors() -> bool:
    return env_bool("ALPHARAVIS_RETRIEVAL_PREFER_RAG_MIRRORS", "true")


def _source_type_key(source_type: str) -> str:
    return str(source_type or "").strip().lower().replace("-", "_")


_DOCUMENT_RAG_SOURCE_TYPES = {
    "external_document",
    "document",
    "pdf",
    "uploaded_document",
    "artifact_document",
}
_LARGE_PASTE_RAG_SOURCE_TYPES = {
    "large_paste",
    "large_ingest",
}
_DOCUMENT_BACKEND_ALIASES = {
    "alpha": "alpharavis_pgvector",
    "alpharavis": "alpharavis_pgvector",
    "alpharavis_pgvector": "alpharavis_pgvector",
    "internal": "alpharavis_pgvector",
    "pgvector": "alpharavis_pgvector",
    "vector": "alpharavis_pgvector",
    "rag": "rag_api",
    "rag_api": "rag_api",
    "rag-api": "rag_api",
    "external": "rag_api",
    "both": "both",
    "dual": "both",
}


def document_rag_backend() -> str:
    raw = os.getenv("ALPHARAVIS_DOCUMENT_RAG_BACKEND", "alpharavis_pgvector").strip().lower()
    return _DOCUMENT_BACKEND_ALIASES.get(raw, "alpharavis_pgvector")


def _is_document_rag_source(source_type: str) -> bool:
    normalized_type = _source_type_key(source_type)
    return normalized_type in _DOCUMENT_RAG_SOURCE_TYPES or normalized_type in _LARGE_PASTE_RAG_SOURCE_TYPES


def normalize_source_keys(source_keys: Any, *, source_key: str = "") -> list[str]:
    raw_items: list[Any]
    if isinstance(source_keys, str):
        raw_items = [part.strip() for part in source_keys.split(",")]
    elif isinstance(source_keys, (list, tuple, set)):
        raw_items = list(source_keys)
    elif source_keys:
        raw_items = [source_keys]
    else:
        raw_items = []
    if source_key:
        raw_items.insert(0, source_key)
    normalized = [str(item).strip() for item in raw_items if str(item).strip()]
    return list(dict.fromkeys(normalized))[:50]


def vector_result_to_tool_hit(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": metadata}
    preview = str(record.get("preview_text") or record.get("chunk_text") or record.get("content") or "")
    preview_chars = int(os.getenv("ALPHARAVIS_PGVECTOR_RESULT_PREVIEW_CHARS", "900"))
    if len(preview) > preview_chars:
        preview = preview[:preview_chars].rstrip() + "\n[Vector result preview truncated.]"
    similarity = record.get("similarity")
    distance = record.get("distance")
    child_archive_keys = metadata.get("child_archive_keys") or record.get("child_archive_keys") or []
    return {
        "source_type": record.get("source_type", "memory"),
        "source_key": record.get("source_key", "unknown"),
        "title": record.get("title") or record.get("source_key") or "untitled",
        "score": similarity,
        "similarity": similarity,
        "distance": distance,
        "preview_text": preview,
        "chunk_text": str(record.get("chunk_text") or record.get("content") or ""),
        "thread_id": record.get("thread_id") or "",
        "thread_key": record.get("thread_key") or record.get("thread_id") or "",
        "chunk_index": record.get("chunk_index"),
        "chunk_count": record.get("chunk_count"),
        "is_catalog": bool(record.get("is_catalog")),
        "embedding_model": record.get("embedding_model") or "",
        "metadata": metadata,
        "child_archive_keys": child_archive_keys,
        "retrieval_backend": record.get("retrieval_backend") or "alpharavis_pgvector",
    }


def rag_allowed_for_source_type(source_type: str, rag_source_keys: list[str] | None = None) -> bool:
    return source_type in {"all", "external_document", "document"} or rag_source_keys is not None


_QUERY_STOPWORDS = {
    "a",
    "about",
    "again",
    "also",
    "an",
    "and",
    "auf",
    "aus",
    "bei",
    "bitte",
    "das",
    "dass",
    "dem",
    "den",
    "der",
    "die",
    "dies",
    "diesem",
    "dieser",
    "do",
    "doch",
    "ein",
    "eine",
    "einen",
    "einer",
    "es",
    "for",
    "gab",
    "haben",
    "hatten",
    "how",
    "ich",
    "in",
    "is",
    "ist",
    "mal",
    "me",
    "mit",
    "noch",
    "nochmal",
    "of",
    "on",
    "sag",
    "the",
    "to",
    "und",
    "war",
    "was",
    "we",
    "wie",
    "wir",
    "zu",
    "zum",
    "zur",
}


def _query_terms(text: str) -> set[str]:
    terms = set()
    for token in re.findall(r"[A-Za-zÄÖÜäöüß0-9_+#.-]{3,}", str(text or "").lower()):
        normalized = token.strip("._-")
        if normalized and normalized not in _QUERY_STOPWORDS:
            terms.add(normalized)
    return terms


def _hit_text(hit: dict[str, Any]) -> str:
    metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    return "\n".join(
        str(value or "")
        for value in [
            hit.get("title"),
            hit.get("source_key"),
            hit.get("chunk_text"),
            hit.get("preview_text"),
            metadata.get("filename"),
            metadata.get("file_id"),
        ]
    )


def score_retrieval_hit(query: str, hit: dict[str, Any]) -> float:
    query_terms = _query_terms(query)
    if not query_terms:
        return 0.0
    hit_terms = _query_terms(_hit_text(hit))
    lexical_score = len(query_terms & hit_terms) / max(1, len(query_terms))

    similarity = hit.get("similarity")
    distance = hit.get("distance")
    score = hit.get("score")
    numeric_score = 0.0
    if isinstance(similarity, (int, float)):
        numeric_score = max(numeric_score, min(1.0, max(0.0, float(similarity))))
    if isinstance(distance, (int, float)):
        numeric_score = max(numeric_score, min(1.0, max(0.0, 1.0 - float(distance))))
    if isinstance(score, (int, float)) and similarity is None and distance is None:
        # Some backends return distance as score, some similarity. Treat small
        # values as distance-like and large values as similarity-like.
        value = float(score)
        numeric_score = max(numeric_score, 1.0 - value if 0.0 <= value <= 1.0 else 0.0)

    return round(max(lexical_score, numeric_score * 0.8), 4)


def grade_retrieval_hits(
    *,
    query: str,
    hits: list[dict[str, Any]],
    min_relevance: float | None = None,
    max_hits: int | None = None,
) -> dict[str, Any]:
    threshold = float(
        min_relevance
        if min_relevance is not None
        else os.getenv("ALPHARAVIS_AGENTIC_RAG_MIN_RELEVANCE", "0.18")
    )
    scored = []
    for hit in hits:
        relevance_score = score_retrieval_hit(query, hit)
        enriched = {**hit, "relevance_score": relevance_score}
        scored.append(enriched)
    scored.sort(key=lambda item: item.get("relevance_score", 0.0), reverse=True)
    if max_hits is not None:
        scored = scored[: max(1, int(max_hits))]
    relevant = [hit for hit in scored if float(hit.get("relevance_score") or 0.0) >= threshold]
    rejected = [hit for hit in scored if hit not in relevant]
    decision = "generate_answer" if relevant else "rewrite_question" if hits else "no_results"
    return {
        "decision": decision,
        "min_relevance": threshold,
        "relevant_hits": relevant,
        "rejected_hits": rejected,
        "relevant_count": len(relevant),
        "rejected_count": len(rejected),
    }


def rewrite_retrieval_query(query: str, *, source_keys: list[str] | None = None) -> str:
    text = str(query or "").strip()
    replacements = [
        r"\bwie war das nochmal (mit|bei|zu|über|ueber)?\b",
        r"\bwas hatten wir nochmal (zu|über|ueber|mit)?\b",
        r"\bhatten wir (nicht )?(mal )?\b",
        r"\bkannst du (nochmal )?(nach)?schauen\b",
        r"\berinner(st)? (du )?(dich )?\b",
        r"\bim archiv\b",
        r"\bin den archiven\b",
        r"\bwas war\b",
    ]
    rewritten = text.lower()
    for pattern in replacements:
        rewritten = re.sub(pattern, " ", rewritten, flags=re.IGNORECASE)
    rewritten = re.sub(r"[?!.:,;]+", " ", rewritten)
    terms = [term for term in re.findall(r"[A-Za-zÄÖÜäöüß0-9_+#.-]{3,}", rewritten) if term not in _QUERY_STOPWORDS]
    deduped = list(dict.fromkeys(terms))
    if source_keys:
        deduped.extend([key for key in normalize_source_keys(source_keys) if key not in deduped])
    candidate = " ".join(deduped).strip()
    return candidate or text


def build_grounded_context_packet(
    *,
    query: str,
    hits: list[dict[str, Any]],
    max_chars: int | None = None,
) -> dict[str, Any]:
    max_chars = int(max_chars or os.getenv("ALPHARAVIS_AGENTIC_RAG_CONTEXT_MAX_CHARS", "6000"))
    chunks: list[dict[str, Any]] = []
    used_chars = 0
    for index, hit in enumerate(hits, start=1):
        text = str(hit.get("chunk_text") or hit.get("preview_text") or "").strip()
        if not text:
            continue
        remaining = max_chars - used_chars
        if remaining <= 0:
            break
        clipped = text[:remaining].rstrip()
        used_chars += len(clipped)
        chunks.append(
            {
                "rank": index,
                "source_type": hit.get("source_type", ""),
                "source_key": hit.get("source_key", ""),
                "title": hit.get("title", ""),
                "retrieval_backend": hit.get("retrieval_backend", ""),
                "relevance_score": hit.get("relevance_score"),
                "chunk_text": clipped,
                "truncated": len(text) > len(clipped),
                "metadata": hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {},
            }
        )
    return {
        "query": query,
        "chunks": chunks,
        "chunk_count": len(chunks),
        "used_chars": used_chars,
        "max_chars": max_chars,
        "instructions": (
            "Use these bounded chunks as retrieval context for the answer. "
            "Do not infer from missing archive content. If exact old turns are required, "
            "call read_archive_record(...) for the specific relevant archive key only."
        ),
    }


def rag_file_id_for_source(source_type: str, source_key: str, metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata if isinstance(metadata, dict) else {}
    explicit = str(metadata.get("rag_file_id") or metadata.get("file_id") or "").strip()
    if explicit:
        return explicit
    normalized_type = _source_type_key(source_type)
    source_key = str(source_key or "").strip()
    if normalized_type == "archive":
        return archive_rag_file_id(source_key)
    if normalized_type in {"artifact", "artifact_document"}:
        return f"artifact:{source_key}"
    return source_key


def rag_activation_metadata(
    *,
    source_type: str,
    source_key: str,
    rag_file_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = metadata if isinstance(metadata, dict) else {}
    normalized_type = _source_type_key(source_type)
    source_key = str(source_key or "").strip()
    rag_file_id = str(rag_file_id or "").strip()
    explicit_reason = str(metadata.get("rag_activation_reason") or "").strip()
    archive_mode = str(metadata.get("archive_rag_mode") or "").strip()

    if metadata.get("rag_active") is True:
        return {
            "rag_active": True,
            "active_rag_file_ids": normalize_source_keys(metadata.get("active_rag_file_ids"), source_key=rag_file_id),
            "active_source_keys": normalize_source_keys(metadata.get("active_source_keys"), source_key=source_key),
            "rag_activation_reason": explicit_reason or "manual_pin",
            "archive_rag_mode": archive_mode or "manual",
        }

    if normalized_type in _DOCUMENT_RAG_SOURCE_TYPES:
        return {
            "rag_active": True,
            "active_rag_file_ids": normalize_source_keys(metadata.get("active_rag_file_ids"), source_key=rag_file_id),
            "active_source_keys": normalize_source_keys(metadata.get("active_source_keys"), source_key=source_key),
            "rag_activation_reason": explicit_reason or "document_ingest",
            "archive_rag_mode": archive_mode or "tool_only",
        }

    if normalized_type in _LARGE_PASTE_RAG_SOURCE_TYPES:
        return {
            "rag_active": True,
            "active_rag_file_ids": normalize_source_keys(metadata.get("active_rag_file_ids"), source_key=rag_file_id),
            "active_source_keys": normalize_source_keys(metadata.get("active_source_keys"), source_key=source_key),
            "rag_activation_reason": explicit_reason or "large_paste",
            "archive_rag_mode": archive_mode or "tool_only",
        }

    return {
        "rag_active": False,
        "active_rag_file_ids": normalize_source_keys(metadata.get("active_rag_file_ids")),
        "active_source_keys": normalize_source_keys(metadata.get("active_source_keys")),
        "rag_activation_reason": explicit_reason,
        "archive_rag_mode": (archive_mode or "tool_only") if normalized_type == "archive" else archive_mode,
    }


def should_mirror_to_rag(
    *,
    source_type: str,
    content: str,
    preferred_backend: str = "auto",
    metadata: dict[str, Any] | None = None,
) -> bool:
    metadata = metadata if isinstance(metadata, dict) else {}
    preferred = str(preferred_backend or "auto").strip().lower()
    if preferred in {"rag", "rag_api", "rag_api_only", "both"}:
        return True
    if preferred in {"none", "pgvector", "alpharavis_pgvector", "vector"}:
        return False
    if metadata.get("rag_active") is True or metadata.get("force_rag") is True:
        return True

    normalized_type = _source_type_key(source_type)
    if normalized_type == "archive":
        return rag_archive_mirror_enabled()
    if _is_document_rag_source(normalized_type):
        return document_rag_backend() in {"rag_api", "both"}

    min_chars = int(os.getenv("ALPHARAVIS_RAG_AUTO_MIRROR_MIN_CHARS", "20000"))
    return normalized_type not in {"memory", "agent_memory", "catalog"} and len(content or "") >= max(1, min_chars)


def should_index_pgvector(
    *,
    source_type: str,
    preferred_backend: str = "auto",
    metadata: dict[str, Any] | None = None,
) -> bool:
    metadata = metadata if isinstance(metadata, dict) else {}
    preferred = str(preferred_backend or "auto").strip().lower()
    if preferred in {"none", "rag_api_only"}:
        return False
    if preferred in {"pgvector", "alpharavis_pgvector", "vector", "both"}:
        return True
    if metadata.get("skip_pgvector") is True:
        return False

    normalized_type = _source_type_key(source_type)
    if _is_document_rag_source(normalized_type):
        backend = document_rag_backend()
        if backend in {"alpharavis_pgvector", "both"}:
            return True
        return env_bool("ALPHARAVIS_INGEST_INDEX_DOCUMENTS_IN_PGVECTOR", "false")
    return True


def _backend_result_is_success(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    if not text:
        return False
    return not any(marker in text for marker in ("failed", "unavailable", "disabled"))


async def ingest_source(
    *,
    source_type: str,
    source_key: str,
    title: str,
    content: str,
    thread_id: str = "",
    thread_key: str = "",
    scope: str = "thread",
    metadata: dict[str, Any] | None = None,
    preferred_backend: str = "auto",
    pgvector_index: Callable[..., Awaitable[str | None]] | None = None,
    rag_mirror_func: Callable[..., Awaitable[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Index one AlphaRavis source through the configured retrieval backends.

    This is the stable router entrypoint for future LangChain-native ingestion.
    The first slice deliberately delegates to existing pgvector and rag_api
    paths so callers can move behind the router without changing storage
    ownership.
    """

    started = time.perf_counter()
    source_type = _source_type_key(source_type) or "memory"
    source_key = str(source_key or "").strip()
    title = str(title or source_key or "untitled").strip()
    content = str(content or "")
    metadata = metadata if isinstance(metadata, dict) else {}
    warnings: list[str] = []
    errors: list[dict[str, str]] = []
    indexed_backends: list[str] = []
    backend_results: dict[str, Any] = {}
    rag_file_id = rag_file_id_for_source(source_type, source_key, metadata)

    if not source_key:
        return {
            "source_type": source_type,
            "source_key": source_key,
            "index_status": "failed",
            "indexed_backends": [],
            "warnings": ["source_key is required."],
            "errors": [{"stage": "validate", "error": "source_key is required."}],
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
    if not content.strip():
        return {
            "source_type": source_type,
            "source_key": source_key,
            "index_status": "failed",
            "indexed_backends": [],
            "warnings": ["content is required."],
            "errors": [{"stage": "validate", "error": "content is required."}],
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }

    if should_index_pgvector(source_type=source_type, preferred_backend=preferred_backend, metadata=metadata):
        if pgvector_index is None:
            warnings.append("AlphaRavis pgvector ingest callback was not provided.")
        else:
            try:
                pgvector_result = await pgvector_index(
                    source_type=source_type,
                    source_key=source_key,
                    title=title,
                    content=content,
                    thread_id=thread_id,
                    thread_key=thread_key,
                    scope=scope,
                    metadata={**metadata, "rag_file_id": rag_file_id} if rag_file_id else metadata,
                )
                backend_results["alpharavis_pgvector"] = pgvector_result
                if _backend_result_is_success(pgvector_result):
                    indexed_backends.append("alpharavis_pgvector")
                else:
                    warnings.append(str(pgvector_result or "AlphaRavis pgvector indexing returned no result."))
            except Exception as exc:
                errors.append({"stage": "alpharavis_pgvector", "error": str(exc)[:500]})

    if should_mirror_to_rag(
        source_type=source_type,
        content=content,
        preferred_backend=preferred_backend,
        metadata=metadata,
    ):
        mirror = rag_mirror_func or _rag_mirror_text
        filename = str(metadata.get("filename") or metadata.get("file_name") or title or source_key).strip()
        if "." not in filename.rsplit("/", 1)[-1]:
            filename = f"{filename}.txt"
        try:
            rag_payload = await mirror(
                file_id=rag_file_id,
                text=content,
                filename=filename,
            )
            backend_results["rag_api"] = rag_payload
            indexed_backends.append("rag_api")
            rag_indexed_at = int(time.time())
        except Exception as exc:
            rag_indexed_at = None
            errors.append({"stage": "rag_api", "error": str(exc)[:500]})
    else:
        rag_indexed_at = None

    indexed_backends = list(dict.fromkeys(indexed_backends))
    index_status = "indexed" if indexed_backends and not errors else "partial" if indexed_backends else "failed"
    activation = rag_activation_metadata(
        source_type=source_type,
        source_key=source_key,
        rag_file_id=rag_file_id,
        metadata=metadata,
    )
    if "rag_api" not in indexed_backends and not normalize_source_keys(metadata.get("active_rag_file_ids")):
        activation = {**activation, "active_rag_file_ids": []}
    result_metadata = {
        **metadata,
        "source_type": source_type,
        "source_key": source_key,
        "rag_file_id": rag_file_id,
        "rag_index_status": "indexed" if "rag_api" in indexed_backends else "failed" if any(
            item.get("stage") == "rag_api" for item in errors
        ) else metadata.get("rag_index_status", ""),
        "rag_indexed_at": rag_indexed_at or metadata.get("rag_indexed_at"),
        "indexed_backends": indexed_backends,
        **activation,
    }
    return {
        "source_type": source_type,
        "source_key": source_key,
        "title": title,
        "thread_id": thread_id,
        "thread_key": thread_key,
        "scope": scope,
        "rag_file_id": rag_file_id,
        "rag_index_status": result_metadata.get("rag_index_status"),
        "rag_indexed_at": result_metadata.get("rag_indexed_at"),
        "indexed_backends": indexed_backends,
        "index_status": index_status,
        **activation,
        "backend_results": backend_results,
        "metadata": result_metadata,
        "warnings": warnings,
        "errors": errors,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


async def query_sources_with_backends(
    *,
    query: str,
    source_keys: list[str],
    source_type: str = "all",
    limit: int = 5,
    include_other_threads: bool = False,
    thread_id: str = "",
    pgvector_search: Callable[..., Awaitable[list[dict[str, Any]]]] | None = None,
    pgvector_available: bool = False,
    pgvector_import_error: Exception | None = None,
    rag_query_func: Callable[..., Awaitable[tuple[list[dict[str, Any]], str]]] | None = None,
    rag_source_keys: list[str] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    query = str(query or "").strip()
    source_keys = normalize_source_keys(source_keys)
    if not query:
        return {"results": [], "warnings": ["query is required."]}
    if not source_keys:
        return {"query": query, "results": [], "warnings": ["source_key is required."]}

    limit = max(1, min(int(limit), int(os.getenv("ALPHARAVIS_PGVECTOR_SEARCH_LIMIT", "5"))))
    if include_other_threads:
        limit = min(limit, int(os.getenv("ALPHARAVIS_CROSS_THREAD_VECTOR_SEARCH_LIMIT", "3")))

    vector_results: list[dict[str, Any]] = []
    vector_warning = ""
    if pgvector_available and pgvector_search is not None:
        try:
            vector_results = await pgvector_search(
                query=query,
                thread_id=thread_id,
                source_type=source_type,
                source_keys=source_keys,
                include_other_threads=include_other_threads,
                limit=limit,
            )
        except Exception as exc:
            vector_warning = f"AlphaRavis pgvector source query failed cleanly: {exc}"
    elif pgvector_import_error:
        vector_warning = f"AlphaRavis pgvector memory is unavailable: {pgvector_import_error}"
    else:
        vector_warning = "AlphaRavis pgvector memory is disabled."

    rag_results: list[dict[str, Any]] = []
    rag_warning = ""
    rag_lookup_keys = normalize_source_keys(rag_source_keys if rag_source_keys is not None else source_keys)
    if (
        env_bool("ALPHARAVIS_ENABLE_RAG_FEDERATED_SEARCH", "true")
        and rag_allowed_for_source_type(source_type, rag_source_keys=rag_source_keys)
        and rag_lookup_keys
        and rag_query_func is not None
    ):
        rag_results, rag_warning = await rag_query_func(query, rag_lookup_keys, limit)

    memory_hits = [vector_result_to_tool_hit(record) for record in vector_results[:limit]]
    document_hits = rag_results[:limit]
    warnings = [warning for warning in [vector_warning, rag_warning] if warning]
    return {
        "query": query,
        "source_keys": source_keys,
        "rag_source_keys": rag_lookup_keys,
        "include_other_threads": include_other_threads,
        "source_type_filter": source_type,
        "retrieval_policy": (
            "These hits are filtered to the requested source_key/file_id values. "
            "Use chunk_text for grounded answers. For archive hits, call read_archive_record "
            "only when exact raw archived turns are needed; do not load unrelated archives."
        ),
        "results": [*memory_hits, *document_hits],
        "warnings": warnings,
        "backend_counts": {
            "alpharavis_pgvector": len(memory_hits),
            "rag_api": len(document_hits),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


async def agentic_rag_retrieve(
    *,
    query: str,
    source_keys: list[str],
    source_type: str = "all",
    limit: int = 5,
    include_other_threads: bool = False,
    thread_id: str = "",
    pgvector_search: Callable[..., Awaitable[list[dict[str, Any]]]] | None = None,
    pgvector_available: bool = False,
    pgvector_import_error: Exception | None = None,
    rag_query_func: Callable[..., Awaitable[tuple[list[dict[str, Any]], str]]] | None = None,
    rag_source_keys: list[str] | None = None,
    allow_rewrite: bool = True,
    max_context_chars: int | None = None,
) -> dict[str, Any]:
    """Run the first AlphaRavis agentic-RAG loop around source retrieval.

    This mirrors the official LangGraph Agentic RAG template at the router
    level: retrieve, grade retrieved documents, rewrite weak queries once, then
    prepare a bounded context packet for a future generate-answer node.
    """

    trace: list[dict[str, Any]] = []
    first = await query_sources_with_backends(
        query=query,
        source_keys=source_keys,
        source_type=source_type,
        limit=limit,
        include_other_threads=include_other_threads,
        thread_id=thread_id,
        pgvector_search=pgvector_search,
        pgvector_available=pgvector_available,
        pgvector_import_error=pgvector_import_error,
        rag_query_func=rag_query_func,
        rag_source_keys=rag_source_keys,
    )
    trace.append({"node": "retrieve", "query": query, "result_count": len(first.get("results", []))})
    grade = grade_retrieval_hits(query=query, hits=list(first.get("results", [])), max_hits=limit)
    trace.append({"node": "grade_documents", "decision": grade["decision"], "relevant_count": grade["relevant_count"]})

    final_query = query
    final_retrieval = first
    final_grade = grade
    rewritten_query = ""
    if allow_rewrite and grade["decision"] == "rewrite_question":
        rewritten_query = rewrite_retrieval_query(query, source_keys=source_keys)
        trace.append({"node": "rewrite_question", "query": rewritten_query})
        if rewritten_query and rewritten_query != query:
            retry = await query_sources_with_backends(
                query=rewritten_query,
                source_keys=source_keys,
                source_type=source_type,
                limit=limit,
                include_other_threads=include_other_threads,
                thread_id=thread_id,
                pgvector_search=pgvector_search,
                pgvector_available=pgvector_available,
                pgvector_import_error=pgvector_import_error,
                rag_query_func=rag_query_func,
                rag_source_keys=rag_source_keys,
            )
            retry_grade = grade_retrieval_hits(query=rewritten_query, hits=list(retry.get("results", [])), max_hits=limit)
            trace.append(
                {
                    "node": "retrieve",
                    "query": rewritten_query,
                    "result_count": len(retry.get("results", [])),
                    "retry": True,
                }
            )
            trace.append(
                {
                    "node": "grade_documents",
                    "decision": retry_grade["decision"],
                    "relevant_count": retry_grade["relevant_count"],
                    "retry": True,
                }
            )
            if retry_grade["relevant_count"] >= grade["relevant_count"]:
                final_query = rewritten_query
                final_retrieval = retry
                final_grade = retry_grade

    context_packet = build_grounded_context_packet(
        query=final_query,
        hits=list(final_grade.get("relevant_hits", [])),
        max_chars=max_context_chars,
    )
    next_action = "generate_answer" if context_packet["chunk_count"] else "no_grounded_context"
    trace.append({"node": "generate_answer" if context_packet["chunk_count"] else "stop", "next_action": next_action})
    return {
        "query": query,
        "final_query": final_query,
        "rewritten_query": rewritten_query,
        "source_keys": final_retrieval.get("source_keys", normalize_source_keys(source_keys)),
        "rag_source_keys": final_retrieval.get("rag_source_keys", normalize_source_keys(rag_source_keys or [])),
        "source_type_filter": source_type,
        "retrieval": final_retrieval,
        "grade": final_grade,
        "context_packet": context_packet,
        "next_action": next_action,
        "graph_trace": trace,
        "warnings": final_retrieval.get("warnings", []),
    }


async def mirror_archive_text(
    *,
    archive_key: str,
    content: str,
    title: str = "",
) -> dict[str, Any]:
    file_id = archive_rag_file_id(archive_key)
    filename = f"{file_id.replace(':', '_')}.txt"
    payload = await _rag_mirror_text(file_id=file_id, text=content, filename=filename)
    return {
        "rag_file_id": file_id,
        "rag_index_status": "indexed" if payload.get("status", True) else "unknown",
        "rag_indexed_at": int(time.time()),
        "rag_response": payload,
        "title": title,
    }


async def query_rag_sources(
    *,
    query: str,
    file_ids: list[str],
    limit: int,
) -> tuple[list[dict[str, Any]], str]:
    try:
        return await _rag_query_sources(query, file_ids, limit=limit), ""
    except RagApiClientError as exc:
        return [], str(exc)
    except Exception as exc:
        return [], f"rag_api unavailable: {exc}"
