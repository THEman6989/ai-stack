# AlphaRavis RAG / Retrieval Handoff

Date: 2026-05-19

This handoff captures the intent behind the current RAG work so a new context
window can continue without re-deriving the design.

## Current Snapshot

The current direction is AlphaRavis-native RAG first. `rag_api` is no longer the
default document/large-paste backend; it is kept as an adapter/reference path
for comparison and compatibility.

Latest local follow-up in this working tree:

- large pasted user content is not automatically indexed on every long paste
  anymore. Auto-ingest waits until the active context is within
  `ALPHARAVIS_LARGE_PASTE_RAG_COMPRESSION_MARGIN_TOKENS` tokens of compression
  pressure, default `5000`.
- paired `/rag ... /rag`, `/rake ... /rake`, `/index ... /index`, or
  `/ingest ... /ingest` blocks still force immediate source indexing.
- large-paste intent is classified locally as `document`, `instruction`,
  `mixed`, or `unknown` before ingest. Instruction-like pastes become
  `large_instruction` sources and the active message keeps a condensed
  instruction brief instead of treating the whole prompt as a document.
- large-paste ingest now records a run-profile event timeline for Observer and
  later UI progress plumbing: `large_ingest.started`,
  `large_ingest.completed`, `large_ingest.failed`, or
  `large_ingest.skipped`.
- queued pgvector ingest is now represented explicitly as
  `index_status=queued` with `queued_backends=["alpharavis_pgvector"]`.
  Large Paste can still replace the active text with a source handle while the
  embedding queue is pending, but retrieval will only return chunks after the
  queue is drained.
- explicit server-local document ingest now has an Agent tool:
  `ingest_document_file`. It reads only under
  `ALPHARAVIS_DOCUMENT_INGEST_ROOT`, loads through LangChain document loaders,
  routes through `ingest_source(...)`, and can pin the resulting active RAG
  source for the current thread.
- when a huge newest paste cannot stay protected in the recent tail, compression
  may move it into the compressible middle. In that oversized-tail rescue path,
  chunked summary compression is forced if the summary prompt would otherwise
  be pruned, even while the global chunking flag stays default-off.
- the exact raw compressed middle still goes into the AlphaRavis compression
  archive, so the active summary is bounded but exact text remains retrievable
  through archive/RAG tooling.

Most recent commits:

```text
4fae3a2 Harden large context RAG and media handling
a57017c Add native document RAG smoke
d7d5c85 Default document RAG to AlphaRavis pgvector
86c15f6 Isolate LiteLLM proxy database
77dc70c Scope LiteLLM embedding params and default to qwen 0.6b
```

Current defaults:

```text
ALPHARAVIS_DOCUMENT_RAG_BACKEND=alpharavis_pgvector
EMBEDDING_LITELLM_MODEL=ollama/qwen3-embedding:0.6b
RAG_COLLECTION_NAME=alpharavis_qwen06
ALPHARAVIS_ENABLE_LARGE_PASTE_RAG_INGEST=true
ALPHARAVIS_ENABLE_LARGE_PASTE_INTENT_CLASSIFIER=true
ALPHARAVIS_LARGE_PASTE_RAG_MIN_CHARS=20000
ALPHARAVIS_LARGE_PASTE_RAG_COMPRESSION_MARGIN_TOKENS=5000
ALPHARAVIS_COMPRESSION_REBALANCE_OVERSIZED_TAIL=true
ALPHARAVIS_COMPRESSION_OVERSIZED_TAIL_RATIO=0.60
ALPHARAVIS_COMPRESSION_OVERSIZED_TAIL_FORCE_MIDDLE_RATIO=0.80
ALPHARAVIS_COMPRESSION_ENABLE_CHUNKED_SUMMARY=false
ALPHARAVIS_PGVECTOR_SPLITTER=auto
ALPHARAVIS_DOCUMENT_INGEST_ROOT=
ALPHARAVIS_ENABLE_RAG_RERANKING=false
```

Live verification after the latest slice:

- Bridge Test UI serves `Native Document RAG Smoke`.
- `POST /api/native-document-rag-smoke` passed live with
  `acceptance_ok=true`, `hit_count=2`, `pgvector_backend_selected=true`, and
  `rag_api_not_used=true`.
- A follow-up `source_type=large_paste` native smoke passed live in about
  `3.0 s` with `acceptance_ok=true`, `rag_api_not_used=true`,
  `active_source_key_recorded=true`, and two bounded pgvector hits.
- Focused tests passed: `42 passed` across Bridge Test UI, retrieval router,
  context budget, and `rag_api_client` tests.
- Current focused local RAG/Memory/Loader tests passed: `65 passed` across
  document ingest, context budget, retrieval router, media analysis, and
  source-scoped retrieval tests.
- `bridge-test-ui` is running on `127.0.0.1:8140`.

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
- `langgraph-app/document_ingest.py`
- `langgraph-app/test_ui_server.py` native/document RAG smoke surface
- `tests/test_rag_api_client.py`
- `tests/test_retrieval_router.py`
- `tests/test_source_scoped_retrieval.py`
- `tests/test_bridge_test_ui.py`
- `docs/ALPHARAVIS_RAG_HANDOFF.md`
- `helper-repos/langgraph-agentic-rag-template/`
- `helper-repos/awesome-rag/`

Implemented retrieval APIs/tools:

- `query_source(...)`
- `query_sources(...)`
- `query_archive(...)`
- `ingest_document_file(...)`
- `agentic_rag_retrieve(...)` as an AlphaRavis tool backed by the router-level
  retrieve/grade/rewrite/context-packet loop
- `pin_active_rag_sources(...)`, `unpin_active_rag_sources(...)`, and
  `inspect_active_rag_sources(...)`
- `read_source_chunks(...)`

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

- route toward AlphaRavis-owned pgvector by default through `ingest_source(...)`
  with `ALPHARAVIS_DOCUMENT_RAG_BACKEND=alpharavis_pgvector`
- explicit server-local files can be loaded with `ingest_document_file(...)`;
  supported loader profiles include PDF, DOCX, HTML, Markdown, plain text,
  CSV/JSON/YAML, and logs. The tool is guarded by
  `ALPHARAVIS_DOCUMENT_INGEST_ROOT` and does not read arbitrary server paths.
- use `ALPHARAVIS_DOCUMENT_RAG_BACKEND=rag_api` for the current external
  adapter, or `both` for evaluation/dual indexing
- return thread-activation metadata with `rag_active=true`,
  `active_source_keys`, optional `active_rag_file_ids`, and
  `rag_activation_reason=document_ingest|large_paste`
- when only AlphaRavis pgvector indexed the source, keep `active_rag_file_ids`
  empty so active prefetch does not call `rag_api`
- large human messages are now detected in `run_profile_start_node`, but
  automatic paste-to-RAG waits until the active context is within
  `ALPHARAVIS_LARGE_PASTE_RAG_COMPRESSION_MARGIN_TOKENS` tokens of compression
  pressure, default `5000`. Paired `/rag ... /rag` blocks force source indexing
  regardless of current context margin. After successful ingest, the active chat
  context gets a compact retrieval marker instead of the full pasted text/block.
- large-paste intent is classified before ingest as `document`, `instruction`,
  `mixed`, or `unknown` without an extra model call. Instruction-like pastes are
  indexed as `large_instruction` for exact lookup but do not auto-activate
  document RAG; the replacement marker carries a condensed instruction brief.
  Mixed pastes keep active RAG, carry the instruction brief, and strip obvious
  instruction text from the indexed document body when separable.

Archive / compression default:

- `rag_active=false`
- `archive_rag_mode=tool_only`
- archive chunks stay available through explicit tools such as
  `query_archive(...)` and `agentic_rag_retrieve(...)`
- pre-run compression runs after large-paste auto-ingest, so a huge paste that
  crossed the 5000-token margin should already have a source/RAG marker before
  the compressor shrinks the active messages.
- oversized-tail rebalancing keeps ordinary recent messages protected, but if
  the protected tail itself exceeds the force threshold, the latest user message
  may be compressed and archived rather than blocking the model run.
- chunked summary remains globally default-off for ordinary compression, but is
  forced for the oversized latest-tail rescue path when the selected middle is
  too large for a one-shot summary prompt.

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
- it retrieves from `active_source_keys` and, only when a source was mirrored to
  an external adapter, `active_rag_file_ids`, with bounded
  `agentic_rag_retrieve(...)`
- it injects only a compact `<active-rag-context>` system message
- archive-only state with `archive_rag_mode=tool_only` stays passive
- Bridge Test UI exposes `Native Document RAG Smoke` to validate the
  AlphaRavis-owned pgvector path without `rag_api`.

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
  PGVector table initialization.
- Native Document RAG smoke passed through AlphaRavis pgvector without `rag_api`
  after switching document/large-paste default routing to
  `ALPHARAVIS_DOCUMENT_RAG_BACKEND=alpharavis_pgvector`.
- A previous large-paste live test hit 180s timeouts while routing through
  `rag_api` embedding batches. Treat that as historical evidence for why the
  next live test should use the native AlphaRavis pgvector route and queue/
  progress controls.
- The `rag_api` collection default is `RAG_COLLECTION_NAME=alpharavis_qwen06`
  after switching the default embedding model to 0.6b. Do not mix old 2560-dim
  qwen3-embedding:4b rows and new 1024-dim qwen3-embedding:0.6b rows in one
  LangChain PGVector collection.
- LiteLLM and `rag_api` must not share the same Postgres database. LiteLLM uses
  the `litellm` database for Prisma proxy metadata; `rag_api` uses the `rag_api`
  database for LangChain PGVector. Sharing them caused LiteLLM startup sanity
  migrations to remove or invalidate `langchain_pg_collection`.

Current chunking direction:

- splitter mode: `ALPHARAVIS_PGVECTOR_SPLITTER=auto`
  - explicit documents and large-paste sources use LangChain
    `RecursiveCharacterTextSplitter` when `langchain-text-splitters` is
    available
  - chat/archive/code/log profiles keep the AlphaRavis splitter unless the
    operator sets `ALPHARAVIS_PGVECTOR_SPLITTER=langchain`
  - set `ALPHARAVIS_PGVECTOR_SPLITTER=alpharavis` to force the local fallback
    everywhere
- standard: 900 tokens / 125 overlap
- chat/archive: 700 / 100
- logs: 1200 / 75
- code: 600 / 80

Code detection is heuristic for now. AST/Tree-sitter splitting is still a
follow-up.

## Next Best Steps

1. Run a true Bridge/LibreChat large-paste E2E against native AlphaRavis
   pgvector.

   Acceptance: first turn with a large pasted source gets replaced by the
   compact retrieval marker, the next user question triggers
   `<active-rag-context>` from `active_source_keys`, the returned chunks come
   from `alpharavis_pgvector`, and `rag_api` is not called unless explicitly
   configured.

2. Connect the actual LibreChat document/PDF upload handoff to
   `ingest_document_file(...)` / `ingest_source(...)`.

   The explicit server-local file tool is implemented. Remaining work is the
   bridge/upload contract: pass a trusted server-side path, not only `file_id`
   metadata. Keep the AlphaRavis pgvector backend as the default and treat
   `rag_api` as an adapter/reference path unless explicitly requested.

3. Stream ingest progress events for large document/paste work.

   Run-profile events for start/completion/failure/skip are now recorded for
   Large Paste and visible through Observer metadata. Remaining work is to emit
   live status/progress events during long-running ingest, including
   `large_ingest.chunk_indexed`, before considering larger Bridge timeouts.

4. Add optional archive auto-on-intent behavior. Keep compression archives
   passive by default; only enable archive auto-retrieval when
   `archive_rag_mode=auto_on_intent` and intent heuristics are proven.

5. Add optional reranking behind the router.
   Desired flow:

```text
pgvector/rag_api top 20-50
  -> reranker
  -> final top 3-8 chunks
  -> LLM context
```

Reranking should be default-off until measured in the Test UI.

6. Later add optional LLM structured-output grading for `grade_documents`.
   Current deterministic grader is intentional because it is fast and testable.

## Verification Commands

Use focused tests first:

```bash
pytest -q tests/test_document_ingest.py tests/test_agent_context_budget.py tests/test_retrieval_router.py tests/test_media_analysis.py tests/test_source_scoped_retrieval.py
```

Broader current RAG-related smoke:

```bash
pytest -q tests/test_retrieval_router.py tests/test_source_scoped_retrieval.py tests/test_rag_api_client.py tests/test_agent_context_budget.py tests/test_bridge_test_ui.py tests/test_media_analysis.py
```

Native Document RAG live smoke:

```bash
python -c "import httpx,json; payload={'source_key':'native_doc_live_smoke','source_type':'large_paste','document_text':'Runtime marker: NATIVE_PGVECTOR_RAG_SMOKE. Decision: explicit documents and large pasted sources should use AlphaRavis-owned pgvector by default. rag_api remains only an adapter or comparison backend.','query':'Welche native AlphaRavis-RAG-Regel steht im Dokument?','limit':3}; r=httpx.post('http://127.0.0.1:8140/api/native-document-rag-smoke',json=payload,timeout=180); print(r.status_code); data=r.json(); print(json.dumps({'status':data.get('status'),'acceptance_ok':data.get('acceptance_ok'),'errors':data.get('errors'),'hit_count':data.get('hit_count'),'acceptance':data.get('acceptance')},ensure_ascii=False,indent=2))"
```

Syntax check:

```bash
PYTHONPYCACHEPREFIX=/tmp/alpharavis-pycache python -m py_compile langgraph-app/document_ingest.py langgraph-app/vector_memory.py langgraph-app/retrieval_router.py langgraph-app/agent_graph.py
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
