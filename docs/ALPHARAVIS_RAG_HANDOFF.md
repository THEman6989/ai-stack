# AlphaRavis RAG / Retrieval Handoff

Date: 2026-05-19

This handoff captures the intent behind the current RAG work so a new context
window can continue without re-deriving the design.

## User Intent

The goal is not to blindly replace AlphaRavis memory with `rag_api`, and not to
copy a demo project wholesale. The goal is to build a strong AlphaRavis RAG
layer that:

- keeps AlphaRavis archives, compression metadata, thread lineage, redaction
  state, and exact raw archive reads owned by AlphaRavis;
- uses proven RAG patterns from `rag_api` and LangGraph/LangChain internally;
- avoids ever loading a full 100k-token archive into the active LLM context just
  because the archive exists;
- lets the system retrieve only bounded relevant chunks when a document/archive
  question needs old context;
- later supports reranking, thread-aware RAG activation, and optional vision
  embeddings, but keeps vision default-off for now.

Important mental model from the user:

```text
Embedding model = makes query/document vectors
pgvector = searches vectors
RAG API / router = coordinates retrieval
Chat LLM = answers from retrieved chunks
```

## Current Architecture Direction

AlphaRavis owns:

- MongoDB/LangGraph Store archive records
- `archive_key`, `thread_id`, `thread_key`
- compression archives and raw redacted messages
- `read_archive_record(...)`
- access/thread scoping
- context-budget decisions

Reusable RAG pieces come from:

- `rag_api` patterns: `file_id`, `file_ids`, LangChain splitter, batch
  embedding, digest metadata, pgvector filters, distance threshold
- LangGraph Agentic-RAG pattern: retrieve, grade documents, rewrite weak query,
  generate answer from bounded context
- future LangChain components: loaders, splitters, retrievers, contextual
  compression, rerankers

Do not hand archive source-of-truth to LangChain or `rag_api`.

## Local References

Read these first:

- `docs/ALPHARAVIS_OPEN_TASKS.md`
- `docs/ALPHARAVIS_CHANGES.md`
- `helper-repos/langgraph-agentic-rag-template/README.md`
- `helper-repos/langgraph-agentic-rag-template/langgraph_agentic_rag.ipynb`
- `helper-repos/langgraph-agentic-rag-template/langchain_agentic_rag_doc.html`
- `helper-repos/awesome-rag/README.md`

The LangGraph notebook is archival, but useful as a concrete code sample. The
current docs page is authoritative. The reusable shape is:

```text
agent / generate_query_or_respond
  -> retrieve
  -> grade_documents
  -> rewrite_question if weak
  -> generate_answer with bounded retrieved chunks
```

## Implemented So Far

New / important files:

- `langgraph-app/rag_api_client.py`
- `langgraph-app/retrieval_router.py`
- `tests/test_rag_api_client.py`
- `tests/test_retrieval_router.py`
- `tests/test_source_scoped_retrieval.py`
- `docs/ALPHARAVIS_RAG_HANDOFF.md`
- `helper-repos/langgraph-agentic-rag-template/`
- `helper-repos/awesome-rag/`

Implemented retrieval APIs/tools:

- `query_source(...)`
- `query_sources(...)`
- `query_archive(...)`
- `agentic_rag_retrieve(...)` as an AlphaRavis tool backed by the router-level
  retrieve/grade/rewrite/context-packet loop

Implemented router functions:

- `ingest_source(...)`
- `query_sources_with_backends(...)`
- `agentic_rag_retrieve(...)`
- `grade_retrieval_hits(...)`
- `rewrite_retrieval_query(...)`
- `build_grounded_context_packet(...)`

Compression archives now use `retrieval_router.ingest_source(...)` for
write-side backend selection. Archive records now store normalized router
metadata:

```text
ingest_status
rag_file_id
rag_index_status
rag_indexed_at
indexed_backends
ingest_errors
rag_active
active_rag_file_ids
active_source_keys
rag_activation_reason
archive_rag_mode
```

## Current Behavior

Archive default:

- store raw archive in AlphaRavis Store/MongoDB
- index/search via AlphaRavis pgvector
- mirror to `rag_api` only when `ALPHARAVIS_ENABLE_RAG_ARCHIVE_MIRROR=true` or
  explicit backend preference requests it

External document / PDF / large paste default:

- route toward `rag_api` by default through `ingest_source(...)`
- do not also index in AlphaRavis pgvector unless
  `ALPHARAVIS_INGEST_INDEX_DOCUMENTS_IN_PGVECTOR=true` or
  `preferred_backend=both`
- return thread-activation metadata with `rag_active=true`,
  `active_source_keys`, `active_rag_file_ids`, and
  `rag_activation_reason=document_ingest|large_paste`
- large human messages are now detected in `run_profile_start_node`; after
  successful `ingest_source(source_type="large_paste")`, the active chat context
  gets a compact retrieval marker instead of the full pasted text

Archive / compression default:

- `rag_active=false`
- `archive_rag_mode=tool_only`
- archive chunks stay available through explicit tools such as
  `query_archive(...)` and `agentic_rag_retrieve(...)`

Agentic-RAG router slice:

- retrieves via `query_sources_with_backends(...)`
- grades retrieved chunks deterministically
- rewrites vague archive questions such as "wie war das nochmal ..." once
- retries retrieval
- returns a bounded `context_packet` for a future answer node/tool
- is exposed through `agent_graph.py` as the `agentic_rag_retrieve` tool for
  explicit source-scoped RAG calls

Active document / large-paste RAG:

- `active_rag_prefetch_node` runs after memory prefetch and before skill/handoff
  preparation when `rag_active=true`
- it retrieves from `active_source_keys` / `active_rag_file_ids` with bounded
  `agentic_rag_retrieve(...)`
- it injects only a compact `<active-rag-context>` system message
- archive-only state with `archive_rag_mode=tool_only` stays passive

## Embedding / Chunking Decisions

Vision embeddings are default-off and experimental.

Text embedding current config targets Ollama/LiteLLM:

```text
EMBEDDING_LITELLM_MODEL=ollama/qwen3-embedding:0.6b
EMBEDDING_API_BASE=http://192.168.178.140:11434
ALPHARAVIS_ENABLE_VISION_VECTOR_MEMORY=false
```

Live findings:

- `qwen3-embedding:4b`: works, 2560-dim, but slow for large chunks
- `qwen3-embedding:0.6b`: current default; works, 1024-dim, much faster when
  speed matters more than vector dimension
- `aroxima/gte-qwen2-1.5b-instruct`: reports embedding metadata but Ollama
  rejects `/api/embed` with HTTP 501 in this setup
- LiteLLM must drop unsupported params when `rag_api` uses
  LangChain/OpenAIEmbeddings against an Ollama-backed `memory-embed` route,
  because that client sends `encoding_format=base64` and Ollama does not accept
  it. Keep this route-scoped: `scripts/render_litellm_config.py` adds
  `drop_params=true` only when the resolved LiteLLM model id starts with
  `ollama/`, so llama.cpp/OpenAI-compatible embedding routes keep their normal
  request parameters.
- Archive RAG smoke passed after the LiteLLM param-drop fix and LangChain
  PGVector table initialization. Large-paste live testing still hits runtime
  timeouts with the current 4b embedding route when a paste expands to many
  chunks.
- The `rag_api` collection default is `RAG_COLLECTION_NAME=alpharavis_qwen06`
  after switching the default embedding model to 0.6b. Do not mix old 2560-dim
  qwen3-embedding:4b rows and new 1024-dim qwen3-embedding:0.6b rows in one
  LangChain PGVector collection.
- LiteLLM and `rag_api` must not share the same Postgres database. LiteLLM uses
  the `litellm` database for Prisma proxy metadata; `rag_api` uses the `rag_api`
  database for LangChain PGVector. Sharing them caused LiteLLM startup sanity
  migrations to remove or invalidate `langchain_pg_collection`.

Current chunking direction:

- standard: 900 tokens / 125 overlap
- chat/archive: 700 / 100
- logs: 1200 / 75
- code: 600 / 80

Code detection is heuristic for now. AST/Tree-sitter splitting is still a
follow-up.

## Next Best Steps

1. Route future document/PDF upload paths through `ingest_source(...)`.

2. Add optional archive auto-on-intent behavior. Keep compression archives
   passive by default; only enable archive auto-retrieval when
   `archive_rag_mode=auto_on_intent` and intent heuristics are proven.

3. Add optional reranking behind the router.
   Desired flow:

```text
pgvector/rag_api top 20-50
  -> reranker
  -> final top 3-8 chunks
  -> LLM context
```

Reranking should be default-off until measured in the Test UI.

4. Later add optional LLM structured-output grading for `grade_documents`.
   Current deterministic grader is intentional because it is fast and testable.

## Verification Commands

Use focused tests first:

```bash
pytest -q tests/test_retrieval_router.py tests/test_source_scoped_retrieval.py tests/test_rag_api_client.py tests/test_agent_context_budget.py
```

Broader current RAG-related smoke:

```bash
pytest -q tests/test_retrieval_router.py tests/test_source_scoped_retrieval.py tests/test_rag_api_client.py tests/test_agent_context_budget.py tests/test_bridge_test_ui.py tests/test_media_analysis.py
```

Syntax check:

```bash
PYTHONPYCACHEPREFIX=/tmp/alpharavis-pycache python -m py_compile langgraph-app/retrieval_router.py langgraph-app/agent_graph.py langgraph-app/rag_api_client.py
```

Notebook reference check:

```bash
python -c "import json; json.load(open('helper-repos/langgraph-agentic-rag-template/langgraph_agentic_rag.ipynb')); print('notebook ok')"
```

## Cautions

- Preserve dirty user work. Do not revert unrelated changes.
- Do not make `rag_api` the only archive source of truth.
- Do not silently inject full archives into active context.
- Do not make vision embeddings default-on.
- Do not copy the LangGraph demo vector store or hard-coded models directly.
- Keep changes documented in `ALPHARAVIS_OPEN_TASKS.md` and
  `ALPHARAVIS_CHANGES.md`.
