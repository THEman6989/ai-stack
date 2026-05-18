from __future__ import annotations

import os
import time
import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any

import httpx


class RagApiClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class RagApiConfig:
    base_url: str
    timeout_seconds: float
    bearer_token: str = ""
    entity_id: str = "alpharavis"


def rag_api_config() -> RagApiConfig:
    return RagApiConfig(
        base_url=os.getenv("ALPHARAVIS_RAG_API_URL", "http://rag_api:8000").rstrip("/"),
        timeout_seconds=float(os.getenv("ALPHARAVIS_RAG_FEDERATED_TIMEOUT_SECONDS", "20")),
        bearer_token=os.getenv("ALPHARAVIS_RAG_API_BEARER_TOKEN", "").strip(),
        entity_id=os.getenv("ALPHARAVIS_RAG_ENTITY_ID", "alpharavis").strip() or "alpharavis",
    )


def _local_jwt(config: RagApiConfig) -> str:
    secret = os.getenv("ALPHARAVIS_RAG_JWT_SECRET") or os.getenv("JWT_SECRET") or ""
    if not secret:
        return ""
    now = int(time.time())
    payload = {
        "id": config.entity_id,
        "sub": config.entity_id,
        "iat": now,
        "exp": now + int(os.getenv("ALPHARAVIS_RAG_JWT_TTL_SECONDS", "3600")),
    }
    try:
        import jwt

        return jwt.encode(payload, secret, algorithm="HS256")
    except Exception:
        header = {"alg": "HS256", "typ": "JWT"}

        def encode_part(value: dict[str, Any]) -> bytes:
            raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
            return base64.urlsafe_b64encode(raw).rstrip(b"=")

        signing_input = b".".join([encode_part(header), encode_part(payload)])
        signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
        return b".".join([signing_input, base64.urlsafe_b64encode(signature).rstrip(b"=")]).decode("ascii")


def _headers(config: RagApiConfig) -> dict[str, str]:
    token = config.bearer_token or _local_jwt(config)
    return {"Authorization": f"Bearer {token}"} if token else {}


def auth_mode(config: RagApiConfig | None = None) -> str:
    config = config or rag_api_config()
    if config.bearer_token:
        return "bearer_token"
    if _local_jwt(config):
        return "local_jwt"
    return "none"


def _normalize_document_hit(item: Any) -> dict[str, Any] | None:
    document = item
    score = None
    if isinstance(item, (list, tuple)) and item:
        document = item[0]
        if len(item) > 1:
            score = item[1]

    if isinstance(document, dict):
        page_content = str(document.get("page_content") or document.get("content") or document.get("text") or "")
        metadata = document.get("metadata") or {}
    else:
        page_content = str(getattr(document, "page_content", "") or getattr(document, "content", ""))
        metadata = getattr(document, "metadata", {}) or {}

    if not page_content.strip():
        return None

    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": metadata}

    preview_chars = int(os.getenv("ALPHARAVIS_RAG_RESULT_PREVIEW_CHARS", "1400"))
    chunk = page_content[:preview_chars].rstrip()
    if len(page_content) > preview_chars:
        chunk += "\n[RAG chunk preview truncated.]"

    file_id = metadata.get("file_id") or metadata.get("source") or metadata.get("path") or "unknown"
    filename = metadata.get("filename") or metadata.get("file_name") or metadata.get("source") or file_id
    return {
        "source_type": "external_document",
        "source_key": str(file_id),
        "title": str(filename),
        "score": score,
        "distance": score,
        "preview_text": chunk,
        "chunk_text": chunk,
        "metadata": metadata,
        "retrieval_backend": "rag_api",
    }


async def query_sources(
    query: str,
    file_ids: list[str],
    *,
    limit: int = 5,
    config: RagApiConfig | None = None,
) -> list[dict[str, Any]]:
    config = config or rag_api_config()
    file_ids = [str(item).strip() for item in file_ids if str(item).strip()]
    if not file_ids:
        return []

    async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
        if len(file_ids) == 1:
            response = await client.post(
                f"{config.base_url}/query",
                json={"query": query, "file_id": file_ids[0], "k": limit, "entity_id": config.entity_id},
                headers=_headers(config),
            )
        else:
            response = await client.post(
                f"{config.base_url}/query_multiple",
                json={"query": query, "file_ids": file_ids, "k": limit},
                headers=_headers(config),
            )

    if response.status_code == 404:
        return []
    if response.status_code >= 400:
        raise RagApiClientError(f"rag_api query returned HTTP {response.status_code}: {response.text[:500]}")

    payload = response.json()
    if not isinstance(payload, list):
        raise RagApiClientError("rag_api query returned an unexpected non-list payload.")

    hits: list[dict[str, Any]] = []
    for item in payload:
        hit = _normalize_document_hit(item)
        if hit:
            hits.append(hit)
    return hits


async def mirror_text(
    *,
    file_id: str,
    text: str,
    filename: str = "alpharavis-archive.txt",
    content_type: str = "text/plain",
    config: RagApiConfig | None = None,
) -> dict[str, Any]:
    config = config or rag_api_config()
    file_id = str(file_id).strip()
    if not file_id:
        raise RagApiClientError("file_id is required for rag_api mirror.")
    if not str(text).strip():
        raise RagApiClientError("text is required for rag_api mirror.")

    files = {"file": (filename, text.encode("utf-8", "ignore"), content_type)}
    data = {"file_id": file_id, "entity_id": config.entity_id}
    async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
        response = await client.post(
            f"{config.base_url}/embed",
            data=data,
            files=files,
            headers=_headers(config),
        )

    if response.status_code >= 400:
        raise RagApiClientError(f"rag_api embed returned HTTP {response.status_code}: {response.text[:500]}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise RagApiClientError("rag_api embed returned an unexpected non-object payload.")
    return payload
