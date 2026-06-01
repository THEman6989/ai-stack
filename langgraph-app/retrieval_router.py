from __future__ import annotations

import os
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from env_utils import env_bool
from rag_api_client import RagApiClientError
from rag_api_client import mirror_text as _rag_mirror_text
from rag_api_client import query_sources as _rag_query_sources

try:
    from langchain_core.documents import Document as LangChainDocument
except Exception:  # pragma: no cover - optional in lean local test envs.
    LangChainDocument = None  # type: ignore[assignment]


@dataclass
class AlphaRavisDocument:
    page_content: str
    metadata: dict[str, Any]




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
    source_key = record.get("source_key", "unknown")
    source_id = record.get("source_id") or source_key
    raw_ref = record.get("raw_ref")
    if not isinstance(raw_ref, dict):
        raw_ref = metadata.get("raw_ref") if isinstance(metadata.get("raw_ref"), dict) else {}
    return {
        "source_type": record.get("source_type", "memory"),
        "source_id": source_id,
        "source_key": source_key,
        "version": record.get("version") or metadata.get("version") or metadata.get("source_digest") or "v1",
        "title": record.get("title") or source_key or "untitled",
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
        "raw_ref": raw_ref,
        "created_at": record.get("created_at") or "",
        "updated_at": record.get("updated_at") or "",
        "child_archive_keys": child_archive_keys,
        "retrieval_backend": record.get("retrieval_backend") or "alpharavis_pgvector",
    }


def rag_allowed_for_source_type(source_type: str, rag_source_keys: list[str] | None = None) -> bool:
    return source_type in {"all", "external_document", "document"} or rag_source_keys is not None


_QUERY_STOPWORDS = {
    # English
    "a", "about", "again", "also", "an", "and", "are", "as", "at", "be",
    "but", "by", "can", "could", "do", "does", "for", "from", "had", "has",
    "have", "how", "if", "in", "is", "it", "its", "just", "like", "me",
    "more", "my", "no", "not", "now", "of", "on", "or", "our", "out",
    "shall", "should", "so", "some", "than", "that", "the", "their", "then",
    "there", "these", "they", "this", "to", "up", "very", "was", "we",
    "were", "what", "when", "which", "who", "why", "will", "with", "would",
    "you", "your",
    # German
    "aber", "als", "am", "an", "auch", "auf", "aus", "bei", "bin", "bis",
    "bitte", "da", "dadurch", "dafür", "dagegen", "daher", "damit", "dann",
    "daran", "darauf", "daraus", "darf", "darin", "darum", "das", "dass",
    "davon", "davor", "dazu", "dein", "deine", "dem", "den", "denn", "der",
    "des", "deshalb", "dich", "die", "dies", "diese", "diesem", "diesen",
    "dieser", "dir", "doch", "dort", "du", "durch", "ein", "eine", "einem",
    "einen", "einer", "es", "etwa", "etwas", "euch", "für", "gab", "ganz",
    "gegen", "geht", "gibt", "habe", "haben", "hast", "hat", "hatte",
    "hatten", "heute", "hier", "ich", "ihm", "ihn", "ihnen", "ihr", "ihre",
    "im", "immer", "in", "ist", "ja", "jetzt", "kann", "kannst", "kein",
    "keine", "kommen", "können", "machen", "mal", "man", "mehr", "mein",
    "meine", "mich", "mir", "mit", "muss", "nach", "nein", "nicht",
    "nichts", "nie", "noch", "nochmal", "nun", "nur", "ob", "oben",
    "oder", "ohne", "schon", "sehr", "sein", "seine", "sich", "sie",
    "sind", "so", "soll", "sondern", "sonst", "über", "und", "uns", "unser",
    "unter", "vom", "von", "vor", "war", "warum", "was", "wegen", "weil",
    "weitere", "welche", "welcher", "wenn", "wer", "werden", "wie", "wieder",
    "wir", "wird", "wo", "wollen", "wurde", "zu", "zum", "zur",
    # NLTK/ISO additions
    "ander", "anderer", "anderes", "deinen", "deines", "derer",
    "euer", "eure", "eurem", "euren", "eurer", "eures",
    "folgende", "folgenden", "hab", "jedes", "keines",
    "unserem", "unseren", "unseres", "warst",
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


async def maybe_llm_grade_retrieval_hits(
    *,
    query: str,
    hits: list[dict[str, Any]],
    deterministic_grade: dict[str, Any],
    llm_grade_func: Callable[..., Awaitable[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if not env_bool("ALPHARAVIS_AGENTIC_RAG_LLM_GRADING", "false"):
        return {**deterministic_grade, "grading_strategy": "deterministic"}
    if llm_grade_func is None:
        return {
            **deterministic_grade,
            "grading_strategy": "deterministic",
            "llm_grading": {"enabled": True, "used": False, "warning": "llm_grade_func was not provided"},
        }
    try:
        llm_grade = await llm_grade_func(
            query=query,
            hits=hits,
            deterministic_grade=deterministic_grade,
        )
    except Exception as exc:
        return {
            **deterministic_grade,
            "grading_strategy": "deterministic",
            "llm_grading": {"enabled": True, "used": False, "error": str(exc)[:500]},
        }
    if not isinstance(llm_grade, dict) or not isinstance(llm_grade.get("relevant_hits"), list):
        return {
            **deterministic_grade,
            "grading_strategy": "deterministic",
            "llm_grading": {"enabled": True, "used": False, "warning": "llm grader returned invalid payload"},
        }
    relevant = [hit for hit in llm_grade.get("relevant_hits", []) if isinstance(hit, dict)]
    rejected = [hit for hit in llm_grade.get("rejected_hits", []) if isinstance(hit, dict)]
    return {
        **deterministic_grade,
        **llm_grade,
        "decision": "generate_answer" if relevant else "rewrite_question" if hits else "no_results",
        "relevant_hits": relevant,
        "rejected_hits": rejected,
        "relevant_count": len(relevant),
        "rejected_count": len(rejected),
        "grading_strategy": str(llm_grade.get("grading_strategy") or "llm_structured_output"),
        "llm_grading": {"enabled": True, "used": True},
    }


def rerank_retrieval_hits(
    *,
    query: str,
    hits: list[dict[str, Any]],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    reranked = []
    for index, hit in enumerate(hits):
        rerank_score = score_retrieval_hit(query, hit)
        backend_score = hit.get("similarity")
        if not isinstance(backend_score, (int, float)):
            backend_score = hit.get("score")
        combined_score = rerank_score
        if isinstance(backend_score, (int, float)):
            combined_score = max(combined_score, min(1.0, max(0.0, float(backend_score))) * 0.8)
        reranked.append(
            {
                **hit,
                "rerank_score": round(float(combined_score), 4),
                "rerank_original_rank": index + 1,
                "rerank_strategy": "deterministic_lexical_vector_blend",
            }
        )
    reranked.sort(key=lambda item: (float(item.get("rerank_score") or 0.0), -int(item.get("rerank_original_rank") or 0)), reverse=True)
    return reranked[: max(1, int(limit))] if limit is not None else reranked


def _reranker_mode() -> str:
    raw = os.getenv("ALPHARAVIS_RAG_RERANKER_MODE", "deterministic").strip().lower()
    if raw in {"model", "llamacpp", "llama_cpp", "qwen3", "qwen3_reranker"}:
        return "model"
    if raw in {"auto"}:
        return "model" if os.getenv("ALPHARAVIS_RAG_RERANKER_URL", "").strip() else "deterministic"
    return "deterministic"


def _reranker_url() -> str:
    base = os.getenv("ALPHARAVIS_RAG_RERANKER_URL", "http://192.168.178.140:8000").strip().rstrip("/")
    endpoint = os.getenv("ALPHARAVIS_RAG_RERANKER_ENDPOINT", "/reranking").strip() or "/reranking"
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        return endpoint
    return f"{base}/{endpoint.lstrip('/')}"


def _reranker_document_text(hit: dict[str, Any], *, max_chars: int) -> str:
    text = str(hit.get("chunk_text") or hit.get("preview_text") or hit.get("content") or "").strip()
    if max_chars > 0:
        text = text[:max_chars].rstrip()
    return text or str(hit.get("title") or hit.get("source_key") or "empty")


async def _call_model_reranker(query: str, documents: list[str]) -> Any:
    try:
        import httpx
    except Exception as exc:  # pragma: no cover - dependency should exist in runtime.
        raise RuntimeError(f"httpx unavailable for model reranker: {exc}") from exc

    timeout = float(os.getenv("ALPHARAVIS_RAG_RERANKER_TIMEOUT_SECONDS", "8"))
    payload: dict[str, Any] = {
        "query": query,
        "documents": documents,
    }
    model = os.getenv("ALPHARAVIS_RAG_RERANKER_MODEL", "").strip()
    if model:
        payload["model"] = model

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(_reranker_url(), json=payload)
    if response.status_code >= 400:
        raise RuntimeError(f"reranker HTTP {response.status_code}: {response.text[:500]}")
    return response.json()


def _parse_model_rerank_scores(payload: Any, *, expected_count: int) -> dict[int, float]:
    if isinstance(payload, dict):
        items = payload.get("results")
        if items is None:
            items = payload.get("data")
        if items is None:
            items = payload.get("rerank_results")
    else:
        items = payload
    if not isinstance(items, list):
        raise RuntimeError("reranker returned no results array")

    scores: dict[int, float] = {}
    for rank, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        raw_index = (
            item.get("index")
            if item.get("index") is not None
            else item.get("document_index")
            if item.get("document_index") is not None
            else item.get("id")
        )
        try:
            index = int(raw_index)
        except Exception:
            index = rank
        raw_score = (
            item.get("relevance_score")
            if item.get("relevance_score") is not None
            else item.get("score")
            if item.get("score") is not None
            else item.get("rerank_score")
        )
        try:
            score = float(raw_score)
        except Exception:
            continue
        if 0 <= index < expected_count:
            scores[index] = score
    if not scores:
        raise RuntimeError("reranker returned no parseable scores")
    return scores


async def rerank_retrieval_hits_with_fallback(
    *,
    query: str,
    hits: list[dict[str, Any]],
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    mode = _reranker_mode()
    if mode == "model" and hits:
        max_hits = max(1, min(len(hits), int(os.getenv("ALPHARAVIS_RAG_RERANKER_MAX_HITS", "20"))))
        max_chars = max(16, int(os.getenv("ALPHARAVIS_RAG_RERANKER_MAX_CHARS", "700")))
        candidates = hits[:max_hits]
        documents = [_reranker_document_text(hit, max_chars=max_chars) for hit in candidates]
        try:
            payload = await _call_model_reranker(query, documents)
            scores = _parse_model_rerank_scores(payload, expected_count=len(candidates))
            reranked: list[dict[str, Any]] = []
            for index, hit in enumerate(candidates):
                score = scores.get(index, float("-inf"))
                reranked.append(
                    {
                        **hit,
                        "rerank_score": round(float(score), 6) if score != float("-inf") else 0.0,
                        "rerank_original_rank": index + 1,
                        "rerank_strategy": "llamacpp_qwen3_reranker",
                        "rerank_backend": _reranker_url(),
                    }
                )
            if len(hits) > len(candidates):
                for index, hit in enumerate(hits[len(candidates):], start=len(candidates)):
                    reranked.append(
                        {
                            **hit,
                            "rerank_score": 0.0,
                            "rerank_original_rank": index + 1,
                            "rerank_strategy": "llamacpp_qwen3_reranker_unscored_tail",
                            "rerank_backend": _reranker_url(),
                        }
                    )
            reranked.sort(
                key=lambda item: (float(item.get("rerank_score") or 0.0), -int(item.get("rerank_original_rank") or 0)),
                reverse=True,
            )
            metadata = {
                "enabled": True,
                "strategy": "llamacpp_qwen3_reranker",
                "backend": _reranker_url(),
                "model": os.getenv("ALPHARAVIS_RAG_RERANKER_MODEL", "qwen3-reranker-0.6b"),
                "fallback_used": False,
                "scored_count": len(scores),
                "candidate_count": len(candidates),
            }
            return (reranked[: max(1, int(limit))] if limit is not None else reranked), metadata, ""
        except Exception as exc:
            if not env_bool("ALPHARAVIS_RAG_RERANKER_FALLBACK_DETERMINISTIC", "true"):
                raise
            warning = f"Model reranker failed; used deterministic fallback: {exc}"
            fallback = rerank_retrieval_hits(query=query, hits=hits, limit=limit)
            metadata = {
                "enabled": True,
                "strategy": "deterministic_lexical_vector_blend",
                "requested_strategy": "llamacpp_qwen3_reranker",
                "backend": _reranker_url(),
                "fallback_used": True,
            }
            return fallback, metadata, warning

    fallback = rerank_retrieval_hits(query=query, hits=hits, limit=limit)
    return fallback, {
        "enabled": True,
        "strategy": "deterministic_lexical_vector_blend",
        "fallback_used": False,
    }, ""


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


def retrieval_hits_to_documents(hits: list[dict[str, Any]]) -> list[Any]:
    documents = []
    document_cls = LangChainDocument or AlphaRavisDocument
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        text = str(hit.get("chunk_text") or hit.get("preview_text") or "").strip()
        if not text:
            continue
        metadata = {
            "source_type": hit.get("source_type", ""),
            "source_key": hit.get("source_key", ""),
            "title": hit.get("title", ""),
            "retrieval_backend": hit.get("retrieval_backend", ""),
            "score": hit.get("score"),
            "similarity": hit.get("similarity"),
            "distance": hit.get("distance"),
            "relevance_score": hit.get("relevance_score"),
            "rerank_score": hit.get("rerank_score"),
            "chunk_index": hit.get("chunk_index"),
            "chunk_count": hit.get("chunk_count"),
            **(hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}),
        }
        documents.append(document_cls(page_content=text, metadata=metadata))
    return documents


class AlphaRavisSourceRetriever:
    """Small LangChain-compatible async retriever adapter over AlphaRavis router callbacks."""

    def __init__(
        self,
        *,
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
    ) -> None:
        self.source_keys = normalize_source_keys(source_keys)
        self.source_type = source_type
        self.limit = limit
        self.include_other_threads = include_other_threads
        self.thread_id = thread_id
        self.pgvector_search = pgvector_search
        self.pgvector_available = pgvector_available
        self.pgvector_import_error = pgvector_import_error
        self.rag_query_func = rag_query_func
        self.rag_source_keys = rag_source_keys

    async def aget_relevant_documents(self, query: str) -> list[Any]:
        payload = await query_sources_with_backends(
            query=query,
            source_keys=self.source_keys,
            source_type=self.source_type,
            limit=self.limit,
            include_other_threads=self.include_other_threads,
            thread_id=self.thread_id,
            pgvector_search=self.pgvector_search,
            pgvector_available=self.pgvector_available,
            pgvector_import_error=self.pgvector_import_error,
            rag_query_func=self.rag_query_func,
            rag_source_keys=self.rag_source_keys,
        )
        grade = grade_retrieval_hits(query=query, hits=list(payload.get("results", [])), max_hits=self.limit)
        return retrieval_hits_to_documents(list(grade.get("relevant_hits", [])))

    async def ainvoke(self, query: str, *args: Any, **kwargs: Any) -> list[Any]:
        return await self.aget_relevant_documents(query)


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


def _backend_result_is_queued(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text.startswith("queued:") or text in {"scheduled", "queue disabled"}


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
    queued_backends: list[str] = []
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
                    if _backend_result_is_queued(pgvector_result):
                        queued_backends.append("alpharavis_pgvector")
                    else:
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
    queued_backends = list(dict.fromkeys(queued_backends))
    if indexed_backends and not errors and not queued_backends:
        index_status = "indexed"
    elif queued_backends and not errors and not indexed_backends:
        index_status = "queued"
    elif indexed_backends or queued_backends:
        index_status = "partial"
    else:
        index_status = "failed"
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
        "queued_backends": queued_backends,
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
        "queued_backends": queued_backends,
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
    results = [*memory_hits, *document_hits]
    reranking_metadata: dict[str, Any] = {"enabled": True, "strategy": ""}
    rerank_warning = ""
    if results:
        results, reranking_metadata, rerank_warning = await rerank_retrieval_hits_with_fallback(
            query=query,
            hits=results,
            limit=limit,
        )
    warnings = [warning for warning in [vector_warning, rag_warning, rerank_warning] if warning]
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
        "results": results,
        "reranking": reranking_metadata,
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
    llm_grade_func: Callable[..., Awaitable[dict[str, Any]]] | None = None,
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
    grade = await maybe_llm_grade_retrieval_hits(
        query=query,
        hits=list(first.get("results", [])),
        deterministic_grade=grade,
        llm_grade_func=llm_grade_func,
    )
    trace.append(
        {
            "node": "grade_documents",
            "decision": grade["decision"],
            "relevant_count": grade["relevant_count"],
            "strategy": grade.get("grading_strategy", "deterministic"),
        }
    )

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
            retry_grade = await maybe_llm_grade_retrieval_hits(
                query=rewritten_query,
                hits=list(retry.get("results", [])),
                deterministic_grade=retry_grade,
                llm_grade_func=llm_grade_func,
            )
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
                    "strategy": retry_grade.get("grading_strategy", "deterministic"),
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
