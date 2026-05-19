# AlphaRavis Changes

This file records important local changes that affect runtime behavior,
compatibility, or operations. Keep detailed rationale here so future upgrades
can tell which patches are intentional and which ones can be removed.

## 2026-05-19 - Current Local Follow-Up Summary

This working-tree slice updates four related areas:

- Large-paste RAG is now budget-aware. Automatic ingest waits until the active
  context is within
  `ALPHARAVIS_LARGE_PASTE_RAG_COMPRESSION_MARGIN_TOKENS=5000` tokens of
  compression pressure. Paired `/rag ... /rag`, `/rake ... /rake`,
  `/index ... /index`, and `/ingest ... /ingest` still force indexing.
- Large-paste intent is classified locally before ingest. Document-like pastes
  activate normal document RAG; instruction-like pastes are indexed as
  `large_instruction` for exact lookup and keep a condensed active instruction
  brief; mixed pastes keep active RAG while stripping obvious instruction text
  from the indexed document body when separable.
- Large-paste ingest records Observer-visible run-profile events:
  `large_ingest.started`, `large_ingest.completed`, `large_ingest.failed`, or
  `large_ingest.skipped`, with elapsed time and backend/status metadata.
- Compression can rebalance oversized protected tails. Above the configured
  60% target it moves older tail messages back into the compressible middle;
  above the 80% force threshold it may move the latest user message too. If
  that oversized latest-tail rescue would prune the summary prompt, chunked
  summary compression is forced for that rescue even though ordinary chunked
  summary remains globally default-off.
- Media handling now mirrors image inputs through `media-gallery` as well as
  videos. Image and video parts from the same chat share thread/group metadata,
  and `/assets` plus `/gallery` gained thread/group filters, grouping, and
  date/name/type sorting.

Focused verification for the current local slice:

```text
pytest -q tests/test_context_compressor.py
pytest -q tests/test_bridge_responses.py tests/test_media_server.py
pytest -q tests/test_context_compressor.py tests/test_agent_context_budget.py tests/test_retrieval_router.py tests/test_bridge_responses.py tests/test_media_server.py
PYTHONPYCACHEPREFIX=/tmp/alpharavis-pycache python -m py_compile \
  langgraph-app/context_compressor.py \
  langgraph-app/agent_graph.py \
  langgraph-app/bridge_server.py \
  langgraph-app/media_server.py
```

Additional live RAG smoke on 2026-05-19:

```text
POST http://127.0.0.1:8140/api/native-document-rag-smoke
source_type=large_paste
status=passed
acceptance_ok=true
rag_api_not_used=true
hit_count=2
elapsed_seconds=2.971
```

## 2026-05-19 - LiteLLM Embedding Param Compatibility For rag_api

Document and large-paste ingest now defaults to the AlphaRavis-owned pgvector
backend:

```text
ALPHARAVIS_DOCUMENT_RAG_BACKEND=alpharavis_pgvector
```

`rag_api` remains available as an adapter with
`ALPHARAVIS_DOCUMENT_RAG_BACKEND=rag_api`, or as a dual-indexing comparison path
with `both`, but it is no longer the default document/large-paste destination.
This keeps the current work moving toward the AlphaRavis-native RAG
implementation rather than making `rag_api` the product. For pgvector-only
document ingests, active thread state keeps `active_source_keys` and leaves
`active_rag_file_ids` empty, so automatic active-RAG prefetch does not call
`rag_api` unless the source was actually mirrored there.
Bridge Test UI now includes `Native Document RAG Smoke`
(`/api/native-document-rag-smoke`) to validate this path directly: it indexes a
document/large-paste source through AlphaRavis pgvector, retrieves bounded
chunks through the router, and checks that `rag_api` was not used.

LiteLLM now uses its own Postgres database:

```text
DATABASE_URL=postgresql://postgres:<password>@vectordb:5432/litellm
```

`rag_api` remains on the `rag_api` database for LangChain PGVector tables. Fresh
Postgres volumes create the extra LiteLLM database via
`docker-initdb/010-create-litellm-db.sql`. This prevents LiteLLM's Prisma
migrations and post-migration sanity diff from modifying or dropping
`rag_api`'s `langchain_pg_collection` / `langchain_pg_embedding` tables.
Verified by force-recreating `litellm`, confirming its live `DATABASE_URL`
points at `/litellm`, confirming the `rag_api` LangChain tables still exist,
and rerunning `Archive RAG Smoke` successfully after the restart.

LiteLLM config is now rendered at container startup by
`scripts/render_litellm_config.py`. The base `litellm-config/config.yaml` keeps
`drop_params` unset globally; the renderer adds `drop_params: true` only to
routes whose resolved model id starts with `ollama/`.

Rationale: `rag_api` uses LangChain/OpenAIEmbeddings, and the current client
sends `encoding_format=base64` on embedding requests. Ollama-backed LiteLLM
embedding routes reject that optional OpenAI parameter unless LiteLLM is allowed
to drop unsupported provider params for that Ollama route. AlphaRavis' own
`vector_memory.py` did not send this parameter, so the failure only appeared
through the `rag_api` mirror and query path.

The compatibility setting is deliberately route-scoped. If
`EMBEDDING_LITELLM_MODEL=openai/<served-model>` points to a llama.cpp, LM Studio,
or other OpenAI-compatible embedding server that accepts OpenAI embedding
parameters, the renderer does not enable `drop_params` for that route.

Live verification on the local stack:

```text
POST /v1/embeddings model=memory-embed encoding_format=base64 -> 200
POST /api/archive-rag-smoke -> status=passed, acceptance_ok=true
```

The first Archive RAG smoke after fixing LiteLLM exposed a one-time
initialization issue where `langchain_pg_collection` did not exist yet. Importing
`rag_api`'s vector store initialized the LangChain PGVector tables in Postgres;
the next Archive RAG smoke mirrored and queried a bounded archive chunk
successfully.

Large-paste live follow-up still needs performance work. A two-turn test with a
large pasted source reached `rag_api` embedding for 27 chunks, but the embedding
batch hit 180s LiteLLM timeouts and the later chat-model path also timed out at
the Bridge/LangGraph 180s boundary. This validates the compatibility fix, but
not a fast end-to-end large-paste user workflow on the current embedding/model
host.

## 2026-05-18 - RAG Reference Checkout And LangChain-Native Router Plan

Added a local helper checkout for RAG architecture research:

```text
helper-repos/awesome-rag
helper-repos/langgraph-agentic-rag-template
```

`awesome-rag` is the `noworneverev/Awesome-RAG` catalogue. It is not runtime
code and should not be imported by AlphaRavis. It is a reference map for
comparing RAG-related projects and patterns such as LangChain, LlamaIndex, Dify,
Flowise, Haystack, RAGFlow, Cognita, fastRAG, AutoRAG, FlashRAG, GraphRAG,
vector stores, memory systems, evaluation frameworks, document parsers, and
model-serving tools.

`langgraph-agentic-rag-template` contains local copies of:

```text
https://docs.langchain.com/oss/python/langgraph/agentic-rag
https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_agentic_rag.ipynb
```

The current docs page is authoritative. The downloaded notebook is useful as a
concrete code sample, but it marks itself archival. The reusable pattern for
AlphaRavis is the graph loop: decide whether to retrieve, call the retriever
tool, grade retrieved chunks, rewrite weak queries, and generate an answer from
bounded context. The demo vector store, web loader, and hard-coded model choices
should not be copied into runtime.

The implementation direction recorded in the open tasks is now:

- keep AlphaRavis as owner of archives, thread metadata, access checks, context
  budgeting, and exact raw archive reads;
- use LangChain/LangGraph primitives internally for loaders, splitters,
  embeddings, vector stores, retrievers, contextual compression, and future
  reranking where they fit;
- preserve the useful `rag_api` design points already validated locally:
  `file_id`/`file_ids`, LangChain splitter, batch embedding, digest metadata,
  pgvector filtering, and optional distance threshold;
- move backend selection into `retrieval_router.py` behind AlphaRavis APIs such
  as `ingest_source(...)`, `query_source(...)`, `query_sources(...)`, and
  `query_archive(...)`.

This is a planning/reference change. No runtime behavior changes from the
helper checkout alone.

First implementation slice after the plan: `retrieval_router.py` now owns the
source-key retrieval backend orchestration through
`query_sources_with_backends(...)`. LangGraph tools in `agent_graph.py` still
own user-facing tool signatures, archive-key mirror lookup, store access, and
logging, but delegate source query execution to the router.

The router currently:

- normalizes source keys;
- queries AlphaRavis pgvector through injected callbacks;
- queries explicit `rag_api` file ids through the existing client wrapper;
- returns normalized backend provenance and backend hit counts;
- keeps archive-only `rag_api` retrieval passive unless an explicit mirror/file
  id is supplied.

Second implementation slice: `retrieval_router.py` now also exposes
`ingest_source(...)` as the write-side router entrypoint. It validates source
input, decides whether to call AlphaRavis pgvector, `rag_api`, or both, and
returns normalized write metadata:

```text
rag_file_id
rag_index_status
indexed_backends
index_status
backend_results
warnings
errors
```

The current routing policy is conservative:

- external documents, PDFs, uploaded documents, and large-paste style sources
  index into AlphaRavis pgvector by default;
- archive sources stay in AlphaRavis pgvector by default, and only mirror to
  `rag_api` when `ALPHARAVIS_ENABLE_RAG_ARCHIVE_MIRROR=true` or the caller
  explicitly asks for `rag_api`/`both`;
- small memory/catalog style sources stay AlphaRavis pgvector-owned;
- `ALPHARAVIS_DOCUMENT_RAG_BACKEND=rag_api|both` and `preferred_backend=both`
  exist for adapter comparison and smoke tests.

This is still a router foundation. Existing product flows are not all migrated
to `ingest_source(...)` yet; compression archive creation and large-paste ingest
are behind the router, while future document upload paths still need to move
behind it next. After those call sites are behind the router, a direct LangChain
retriever backend can be added
without changing LangGraph tool code.

First product call-site migration: compression archive creation now writes
through `retrieval_router.ingest_source(...)`. The archive Store record keeps
AlphaRavis archive ownership, raw content, redacted messages, thread lineage,
and `read_archive_record(...)` semantics, while ingest/index metadata now comes
from the router:

```text
ingest_status
rag_file_id
rag_index_status
rag_indexed_at
indexed_backends
ingest_errors
```

This preserves the existing default behavior: archive pgvector indexing remains
the normal path, and `rag_api` archive mirroring only happens when the existing
mirror flag or explicit router preference enables it. The important structural
change is that archive write-side backend selection is now centralized in the
same router that query-side retrieval uses.

Third implementation slice: `retrieval_router.agentic_rag_retrieve(...)` now
implements the first router-level version of the LangGraph Agentic-RAG
Schablone:

```text
retrieve
grade_documents
rewrite_question
retrieve retry
generate_answer context packet
```

This is not yet a full LangGraph subgraph and it does not call an LLM grader.
It deliberately starts with deterministic local relevance scoring and query
rewrite so the behavior is testable without model/network dependencies. The
function returns:

```text
graph_trace
grade
context_packet
next_action
final_query
rewritten_query
```

The generated `context_packet` is bounded by
`ALPHARAVIS_AGENTIC_RAG_CONTEXT_MAX_CHARS` and includes instructions not to load
raw archives unless exact old turns are required. It is now exposed from
`agent_graph.py` as an explicit tool; optional structured-output LLM grading or
reranking remains future work once latency/quality are measured.

Follow-up implementation: `agent_graph.py` now exposes the router loop as the
`agentic_rag_retrieve` tool. The tool returns the router payload, including
`graph_trace`, `grade`, `context_packet`, `next_action`, `final_query`, and
`rewritten_query`; it does not automatically inject archive content into the
active model context. For `source_type="archive"`, the tool keeps AlphaRavis
pgvector search scoped to the archive key and additionally queries an existing
`rag_api` mirror only when archive metadata advertises a ready
`archive:<archive_key>` file id. The tool is registered in the RAG/memory
toolset and the context retrieval agent prompt now prefers it when a
source-scoped question needs retrieve/grade/rewrite behavior.

Next metadata slice: `retrieval_router.ingest_source(...)` now emits
thread-aware RAG activation fields:

```text
rag_active
active_rag_file_ids
active_source_keys
rag_activation_reason
archive_rag_mode
```

External documents, PDFs, uploaded documents, artifact documents, and large
paste/large-ingest sources mark RAG active for the thread and carry the relevant
source/file ids. Compression archives remain passive with
`archive_rag_mode=tool_only`, so old archived context is still reached through
explicit retrieval tools instead of automatic full-archive injection.
`AlphaRavisState` now has matching fields and run-profile snapshots expose the
current values for debugging.

Large-paste ingest is now wired at `run_profile_start_node`. If a human message
exceeds `ALPHARAVIS_LARGE_PASTE_RAG_MIN_CHARS` and
`ALPHARAVIS_ENABLE_LARGE_PASTE_RAG_INGEST=true`, AlphaRavis first checks the
current context budget. Automatic paste-to-RAG only runs when the estimated
active context has at most
`ALPHARAVIS_LARGE_PASTE_RAG_COMPRESSION_MARGIN_TOKENS` tokens left before the
compression trigger, default `5000`. If there is still more room, the paste
stays in active context and `large_paste_ingests` records a skipped decision.
Manual `/rag ... /rag` blocks force indexing of the marked block regardless of
current context margin. Only after successful indexing does the active message
or marked block get replaced with a compact retrieval marker containing the
`source_key`/`rag_file_id` and a short preview.

Large-paste handling now classifies the paste intent before ingest without
calling another model. Document-like and unknown pastes keep the existing
document-RAG path. Instruction-like pastes are indexed as `large_instruction`
through AlphaRavis pgvector for exact lookup, but they do not activate automatic
document RAG; the active message is replaced with a condensed instruction brief
that the model should follow. Mixed instruction+document pastes keep active RAG
for source-scoped retrieval and include the condensed instruction brief in the
replacement marker. For mixed pastes with a recognizable document section,
obvious instruction text is stripped from the indexed document body while the
instruction brief stays active. Run-profile `large_paste_ingests` records
`paste_intent`, confidence, instruction/document scores, context-margin data,
manual-block status, whether instruction text was stripped from the index, and
an `events` timeline with `large_ingest.started`, `large_ingest.completed`,
`large_ingest.failed`, or `large_ingest.skipped` entries for Observer/debugging.

Active-RAG prefetch is now wired into the LangGraph path after memory prefetch
and before skill/handoff preparation. When `rag_active=true` and active
source/file ids exist, `active_rag_prefetch_node` runs bounded
`agentic_rag_retrieve(...)` and injects a compact `<active-rag-context>` system
message. It skips trivial queries, missing sources, router failures, and
archive-only state with `archive_rag_mode=tool_only`.

Operator knobs:

```text
ALPHARAVIS_ENABLE_ACTIVE_RAG_PREFETCH=true
ALPHARAVIS_ACTIVE_RAG_PREFETCH_LIMIT=4
ALPHARAVIS_ACTIVE_RAG_PREFETCH_CONTEXT_CHARS=5000
ALPHARAVIS_ACTIVE_RAG_PREFETCH_MIN_QUERY_CHARS=8
```

Added `docs/ALPHARAVIS_RAG_HANDOFF.md` as the explicit context-window handoff
for this RAG effort. It records the user intent, local reference files,
implemented router functions, current defaults, next steps, verification
commands, and cautions.

Verification:

```text
pytest -q tests/test_retrieval_router.py tests/test_source_scoped_retrieval.py tests/test_rag_api_client.py tests/test_agent_context_budget.py tests/test_bridge_test_ui.py tests/test_media_analysis.py
PYTHONPYCACHEPREFIX=/tmp/alpharavis-pycache python -m py_compile langgraph-app/retrieval_router.py langgraph-app/agent_graph.py langgraph-app/rag_api_client.py
python -c "import json; json.load(open('helper-repos/langgraph-agentic-rag-template/langgraph_agentic_rag.ipynb')); print('notebook ok')"
```

## 2026-05-18 - Default-Off RAG API Archive Mirror Foundation

Added the first implementation slice for hybrid archive/document retrieval.
The existing AlphaRavis archive path remains the default. When
`ALPHARAVIS_ENABLE_RAG_ARCHIVE_MIRROR=true`, newly created compression archives
are also uploaded to `rag_api` through `/embed` with
`file_id=archive:<archive_key>`. The raw archive remains in the
MongoDB/LangGraph Store as the source of truth; the `rag_api` copy is only a
retrieval index.

New local modules:

```text
langgraph-app/rag_api_client.py
langgraph-app/retrieval_router.py
```

`rag_api_client.py` wraps the existing `rag_api` HTTP surface instead of
importing the FastAPI app into LangGraph. It supports `/embed`, `/query`, and
`/query_multiple`, preserving the `rag_api` chunking, batching, embedding,
distance-threshold, and LangChain/ExtendedPgVector behavior inside that service.

Archive records can now carry mirror metadata:

```text
rag_file_id
rag_index_status
rag_indexed_at
indexed_backends
```

`query_archive(...)` checks existing archive metadata and can query the
`rag_api` mirror first when a ready `rag_file_id` exists. It still falls back to
AlphaRavis pgvector source-key search, and `read_archive_record(...)` remains
the explicit exact-history path.

Verification:

```text
pytest -q tests/test_rag_api_client.py tests/test_bridge_test_ui.py tests/test_source_scoped_retrieval.py tests/test_alpharavis_toolsets.py
PYTHONPYCACHEPREFIX=/tmp/alpharavis-pycache python -m py_compile langgraph-app/test_ui_server.py langgraph-app/rag_api_client.py langgraph-app/retrieval_router.py langgraph-app/agent_graph.py langgraph-app/vector_memory.py langgraph-app/alpharavis_toolsets.py
```

Live smoke through the Bridge Test UI was added under Observer ->
`Archive RAG Smoke`. It posts to `/api/archive-rag-smoke`, mirrors a small
archive payload as `file_id=archive:<archive_key>`, then queries the same source.
The endpoint returns structured acceptance fields and runtime errors instead of
failing the page with HTTP 500.

Current local live result on 2026-05-18:

```text
GET /observer: 200, Archive RAG Smoke panel present
POST /api/archive-rag-smoke: 200, status=failed
rag_file_id: archive:live_smoke_archive
failure: LiteLLM memory-embed route cannot connect to 192.168.178.140:11434
```

This validates the wrapper/UI/control path and identifies the remaining runtime
blocker as embedding backend availability, not the AlphaRavis retrieval wrapper.

The Observer also now includes `Memory Embed Tester`. It calls a chosen
embedding base URL directly, supports OpenAI-compatible `/v1/embeddings`,
Ollama `/api/embed`, and Ollama `/api/embeddings`, and can probe text or an
experimental vision payload. The probe doubles input size across bounded steps
and reports latency, embedding dimensions, status code, max accepted chars, and
whether it stopped because of rejection, error, or a slow-response threshold.
This is a diagnostic surface only; it does not change the active memory backend.
Live probe against `http://litellm:4000/v1`, model `memory-embed`, currently
returns HTTP 500 because the downstream embedding backend is unreachable; the
tester reports this as `status=failed`, `stop_reason=rejected`.

Updated the default text embedding route to use Ollama-native LiteLLM model ids:

```text
EMBEDDING_LITELLM_MODEL=ollama/qwen3-embedding:0.6b
EMBEDDING_FALLBACK_LITELLM_MODEL=ollama/bge-m3
EMBEDDING_API_BASE=http://192.168.178.140:11434
```

Operators should run `ollama pull qwen3-embedding:0.6b` on the Ollama host before
recreating LiteLLM. The OpenAI-compatible alternative remains supported by
setting `EMBEDDING_LITELLM_MODEL=openai/<served-model>` and
`EMBEDDING_API_BASE=http://<embedding-host>:<port>/v1`. Vision embeddings remain
experimental and default-off through `ALPHARAVIS_ENABLE_VISION_VECTOR_MEMORY`.
The Memory Embed Tester default max probe size was raised to about 32k rough
tokens so it can validate the expected `qwen3-embedding:0.6b` context window.
Earlier live measurement through LiteLLM showed the 4b route is functional with
2560-dimensional vectors, but latency rises quickly on the current Ollama host:
1024 chars took 10.3s, 2048 chars took 20.1s, 4096 chars took 41.7s, and
8192 chars took 86.2s. Because of that, AlphaRavis pgvector defaults were tuned
to smaller profile-specific chunks:

```text
ALPHARAVIS_PGVECTOR_SPLITTER=auto
ALPHARAVIS_PGVECTOR_CHARS_PER_TOKEN=4.0
ALPHARAVIS_PGVECTOR_CHUNK_TOKENS=900
ALPHARAVIS_PGVECTOR_CHUNK_OVERLAP_TOKENS=125
ALPHARAVIS_PGVECTOR_CHAT_CHUNK_TOKENS=700
ALPHARAVIS_PGVECTOR_CHAT_CHUNK_OVERLAP_TOKENS=100
ALPHARAVIS_PGVECTOR_LOG_CHUNK_TOKENS=1200
ALPHARAVIS_PGVECTOR_LOG_CHUNK_OVERLAP_TOKENS=75
ALPHARAVIS_PGVECTOR_CODE_CHUNK_TOKENS=600
ALPHARAVIS_PGVECTOR_CODE_CHUNK_OVERLAP_TOKENS=80
ALPHARAVIS_PGVECTOR_EMBEDDING_TIMEOUT_SECONDS=45
```

Code/log/chat detection uses source type, filename/path metadata, Markdown
fences, and common syntax/log-line signals. Code splitting is still heuristic;
AST/Tree-sitter function/class chunking remains a follow-up.

Follow-up: `ALPHARAVIS_PGVECTOR_SPLITTER=auto` now uses LangChain's
`RecursiveCharacterTextSplitter` for explicit document and large-paste sources
when `langchain-text-splitters` is installed. Chunk size and overlap still come
from the AlphaRavis ENV profile knobs above. Set
`ALPHARAVIS_PGVECTOR_SPLITTER=langchain` to force LangChain splitting, or
`alpharavis` to force the local fallback.

Follow-up: AlphaRavis pgvector chunk metadata now records stable normalized-text
digests. Catalog rows include `source_digest` and
`source_digest_algorithm=sha256-normalized-text`; chunk rows include
`source_digest`, `chunk_digest`, and `digest_algorithm`. This is the first step
toward repeated paste/archive chunk deduplication. The retrieval/storage policy
still keeps all chunks for now; digest-based skip/reuse behavior remains a
follow-up.

Follow-up: the retrieval router now distinguishes queued pgvector ingest from
completed indexing. A pgvector result such as `queued:<job_id>` returns
`index_status=queued`, leaves `indexed_backends` empty, and records
`queued_backends=["alpharavis_pgvector"]`. Large Paste replacement still returns
a source handle in this state, but the marker says the source is queued rather
than already indexed. Retrieval may return no chunks until the embedding queue
is drained.

Follow-up: RAG active-source pins are now tool-accessible and store-backed per
thread. `pin_active_rag_sources`, `unpin_active_rag_sources`, and
`inspect_active_rag_sources` let an agent/operator persist active source keys
and optional `rag_api` file ids in the LangGraph Store. `active_rag_prefetch`
merges those pins with state-derived document/large-paste activation metadata,
so pinned sources can trigger bounded prefetch even when the current graph state
does not already carry `rag_active=true`.

Follow-up: `read_source_chunks` provides a guarded exact-source read path for
known AlphaRavis pgvector `source_key` values. It reads ordered non-catalog
chunks scoped to the current thread by default and caps output with
`ALPHARAVIS_SOURCE_READ_MAX_CHUNKS` and `ALPHARAVIS_SOURCE_READ_MAX_CHARS`.
This gives agents a controlled fallback when semantic retrieval finds the right
source but a few exact adjacent chunks are needed.

Follow-up: the retrieval router now has a LangChain-style document/retriever
adapter. `retrieval_hits_to_documents(...)` converts AlphaRavis hits into
LangChain `Document` objects when available, and `AlphaRavisSourceRetriever`
exposes async `aget_relevant_documents` / `ainvoke` over
`query_sources_with_backends(...)`. This keeps AlphaRavis ownership, thread
scope, and metadata while letting future graph nodes consume a Retriever-like
interface.

Follow-up: optional deterministic reranking is available behind
`ALPHARAVIS_ENABLE_RAG_RERANKING=false`. When enabled, router results are
annotated with `rerank_score`, `rerank_original_rank`, and
`rerank_strategy=deterministic_lexical_vector_blend`, then sorted before
grading/context-packet construction. External reranker model calls remain
future work.

Follow-up: `langgraph-app/document_ingest.py` adds the first LangChain-native
document loader layer for file-like RAG sources. It selects `PyPDFLoader`,
`Docx2txtLoader`, `BSHTMLLoader`, or `TextLoader` by extension, returns
normalized text with document-part markers, and preserves per-part metadata for
later `ingest_source(...)` calls. The runtime requirements now include `pypdf`
and `docx2txt` for PDF/DOCX loader support. This helper is internal foundation;
the live upload path still needs an explicit file-location handoff before it
can be wired safely.

Follow-up: `ingest_document_file` is now available as a guarded Agent RAG tool.
It reads only under `ALPHARAVIS_DOCUMENT_INGEST_ROOT` (defaulting to the
AlphaRavis workspace), loads the file through `document_ingest.py`, indexes it
through `ingest_source(...)` with AlphaRavis pgvector callbacks, and optionally
persists the resulting active source pin for the current thread. This covers
explicit server-local PDF/DOCX/HTML/Markdown/text ingest while keeping the
LibreChat upload auto-routing follow-up separate.

Treat 32k as model context capacity, not as a practical per-chunk ingest size on
this backend.

Additional Ollama embedding probes:

```text
aroxima/gte-qwen2-1.5b-instruct:
  qwen2.context_length=131072
  qwen2.embedding_length=1536
  /api/embed result: HTTP 501, "this model does not support embeddings"

qwen3-embedding:0.6b:
  qwen3.context_length=32768
  qwen3.embedding_length=1024
  8192 chars  (~2048 rough tokens): 19.1s
  16384 chars (~4096 rough tokens): 44.8s
  32768 chars (~8192 rough tokens): 43.5s
  65536 chars (~16384 rough tokens): 46.0s
  131072 chars (~32768 rough tokens): 40.5s
```

The `0.6b` model is now the default because throughput matters more than vector
dimension for the current local RAG flow. It returns smaller 1024-dimensional
vectors but is much faster than the 4b route on this host.

The `rag_api` collection default was moved to `RAG_COLLECTION_NAME=alpharavis_qwen06`
at the same time. This avoids mixing old 2560-dimensional qwen3-embedding:4b
rows with new 1024-dimensional qwen3-embedding:0.6b rows in the same LangChain
PGVector collection.

## 2026-05-18 - Source-Scoped Retrieval Tools

AlphaRavis now exposes `query_source`, `query_sources`, and `query_archive`
beside the broad `semantic_memory_search` tool. These tools let agents search
only known `source_key` values, archive keys, artifact keys, or external RAG
`file_id` values when the relevant source is already known.

The pgvector search path now accepts bounded `source_key` filters. For
`external_document`/document-style searches, the same tools call the existing
`rag_api` `/query` and `/query_multiple` endpoints directly instead of first
listing every document id. Results keep the normal structured hit shape with
chunk text, similarity/score, source metadata, and retrieval policy guidance.
The local AlphaRavis pgvector query mirrors the `rag_api` retriever contract:
single-source lookup behaves like `file_id $eq`, multi-source lookup behaves
like `file_id $in`, and `ALPHARAVIS_PGVECTOR_DISTANCE_THRESHOLD` can apply a
pgvector distance cutoff analogous to `rag_api`'s `RAG_DISTANCE_THRESHOLD`.

Verification:

```text
pytest -q tests/test_source_scoped_retrieval.py
```

## 2026-05-18 - Bridge Test UI Chunking Lab

The Bridge Test UI Observer now includes a local `Chunking Lab` panel for
compression diagnostics. This is a test/observer feature only; the Bridge still
does not perform active-context chunking in normal request handling. The panel
starts a local diagnostic run through `POST /api/chunking/runs`, polls
`GET /api/chunking/runs/{run_id}`, and renders running action logs, tool-pruning
stats, chunk counts, prompt overhead/payload budgets, synthesis pruning, summary
failure state, and acceptance checks.

The diagnostic uses AlphaRavis's real `context_compressor.compress_messages`
helper with chunked summary enabled, a large synthetic web-like corpus by
default, generated tool-call/tool-output traces, and optional variable prompt
load (`current_task_brief`, handoff packet, MemoryKernel context, and skill
context). It is deterministic and does not fetch external internet text, so it
can run in local Docker/offline developer environments while still exercising
the static-plus-variable prompt overhead path.

The lab now distinguishes summary backends explicitly. `summary_mode=stub` is a
fast deterministic harness for UI/API/chunk-budget plumbing and does not call an
LLM. `summary_mode=real_llm` calls an OpenAI-compatible `/chat/completions`
summary model using `TEST_UI_CHUNKING_SUMMARY_*` settings, falling back to the
normal `OPENAI_API_BASE`, `OPENAI_API_KEY`, and `ALPHARAVIS_MODEL` environment.
Only `real_llm` runs should be treated as latency or quality evidence for
promoting chunked summary compression.

Each completed run now also exposes a collapsible Before/After comparison in
the Observer: the prepared compression input before chunking and the final
active summary after synthesis. The same data is returned in the run API under
`result.comparison`, with browser/API capture capped by
`TEST_UI_CHUNKING_TEXT_CAPTURE_CHARS` so very large diagnostic bodies do not
make the page unusable.

Verification:

```text
pytest -q tests/test_bridge_test_ui.py
```

## 2026-05-16 - LibreChat Video Upload Responses Patch

`librechat.yaml` now lets the normal `AlphaRavis Responses` model spec accept
video uploads on the existing `LangGraph Agent` endpoint. LibreChat v0.8.5 only
serializes videos as `video_url` attachments for a small provider set by
default, so the LibreChat container runs
`scripts/patch_librechat_video_uploads.js` before startup. The patch makes the
configured AlphaRavis endpoint video-capable without exposing a misleading
OpenRouter/provider shim in the UI.

The same startup patch also updates LibreChat's client upload menu/drag-drop
bundle. Without that UI patch, LibreChat can still show only `Upload as Text`
for videos even though the backend encoder is already able to emit `video_url`
parts.

LibreChat can store uploaded videos as normal local message attachments while
still dropping them before the provider HTTP request. The startup patch now also
updates LibreChat's prompt formatter so user-message `videos` and `audios`
arrays are preserved alongside `image_urls` in the final message content sent
to the AlphaRavis Bridge.

For `useResponsesApi: true`, LibreChat also passes those formatted messages
through `@librechat/agents` / LangChain's OpenAI Responses converter. LibreChat
v0.8.5's converter only forwarded text, images, and files, so `video_url` parts
were still dropped and the Bridge Observer showed only `input_text`. The startup
patch now converts `video_url` to `input_video` and allows existing
`input_video` parts through for the local Responses path.

LibreChat's bundled PWA service worker can keep serving the old upload menu
from Workbox cache. The patch therefore also replaces `registerSW.js` and
`sw.js` with cache-clearing/unregistering scripts for the local stack, and
rewrites compressed `.gz`/`.br` variants for patched browser assets.

The model spec still sets `useResponsesApi: true`; text, video attachment, and
assistant output remain on LibreChat's Responses path.

```text
endpoint: LangGraph Agent -> http://api-bridge:8123/v1
model spec: AlphaRavis Responses
useResponsesApi: true
allowed MIME: video/*, image/*, application/octet-stream
```

Use this spec when sending videos through LibreChat. Do not use LibreChat's
`Upload as Text` option for videos; that can still extract/base64-like text and
consume the entire LibreChat agent context before AlphaRavis sees the request.
The Bridge already mirrors incoming `video_url`/`input_video` parts into
`media-gallery` and converts raw media to metadata markers before LangGraph
context construction.

Verification on 2026-05-16:

- LibreChat stored `vitpose_00001.mp4` as a local `video/mp4` message
  attachment, not as a text upload.
- Before the prompt-formatter patch, the Bridge request still had only text
  messages because LibreChat dropped `message.videos`.
- After restart, `formatMessage()` preserves a test `video_url` content part,
  and `@librechat/agents` converts it to Responses `input_video`.
- `pytest -q tests/test_bridge_responses.py` passes.

## 2026-05-15 - Hermes-Style Tail And LangGraph-Owned Hard Context

### Summary

AlphaRavis compression now protects the recent tail closer to Hermes: the
default hard minimum is 3 latest messages, while the real tail size is governed
by `ALPHARAVIS_COMPRESSION_TAIL_TOKEN_RATIO` and
`ALPHARAVIS_COMPRESSION_TAIL_SOFT_CEILING_RATIO`. This removes the previous
default that kept at least 16 latest messages verbatim, which could leave a
large uncompressible tail. The latest user/human message is now anchored into
the protected tail, matching Hermes' active-task preservation rule. Pre-run
compression now also mirrors Hermes' multi-pass preflight loop: estimate,
compress, re-estimate, and retry up to
`ALPHARAVIS_PRE_RUN_COMPRESSION_MAX_PASSES` before hard trim is used as a final
fallback.

Follow-up: oversized-tail rebalancing is now enabled. If the protected recent
tail still consumes more than `ALPHARAVIS_COMPRESSION_OVERSIZED_TAIL_RATIO` of
the compression budget, default 60 percent, older tail messages are moved back
into the compressible middle. The latest user message stays anchored by default
through `ALPHARAVIS_COMPRESSION_KEEP_LATEST_USER_WHEN_REBALANCING_TAIL=true`.
If the protected tail is critically oversized above
`ALPHARAVIS_COMPRESSION_OVERSIZED_TAIL_FORCE_MIDDLE_RATIO`, default 80 percent,
the latest user anchor is released too so the huge request can be archived and
compressed instead of blocking the active window.
Compression metadata records `oversized_tail_rebalanced`,
`oversized_tail_tokens_before`, `oversized_tail_token_target`,
`oversized_tail_force_latest_user_to_middle`, and moved indexes for
Observer/debugging.

The bridge hard input cutoff is disabled by default
(`BRIDGE_HARD_INPUT_TOKEN_LIMIT=0`). LangGraph now owns the normal hard-context
decision so it can use discovered llama.cpp/model context length, pre-run
compression, and hard trim before deciding whether a thread is still too large.

Active compaction summaries also include a direct archive reference block with
`source_type: archive`, the `archive_key`, and a `read_archive_record(...)`
hint so agents can inspect exact removed context instead of guessing from the
summary.

Pre-run compression now reserves static model-call overhead before deciding
whether the active thread is below budget. The reserve is computed from the
largest configured DeepAgents system prompt and tool schema budget, matching
Hermes' habit of counting tool schemas during preflight. The route hard stop
also checks active messages plus this reserve, so the graph is not comparing
only chat history against a limit that the actual model request will exceed.
Handoff and post-run compression also use the reserve, keeping the active state
ready for the next model request after tool traces or handoff packets are added.
The reserve is now tracked per configured agent and used per selected/active
agent when possible, with the maximum reserve kept as a safe fallback.

AlphaRavis now has a final budget rescue node immediately before the Swarm
model invocation. It checks the same full request budget snapshot, forces
Hermes-style compression when needed, and only falls back to hard trim if the
request is still over the hard budget. Agents also get an `inspect_context_budget`
tool for visibility into detected context length, effective limits, reserves,
and archive counts.
If the provider still reports context overflow or payload-too-large from the
Swarm path, AlphaRavis now runs that rescue path once and retries the Swarm
invocation with compressed state.

Pre-run and final rescue compression now run in "until under budget" mode by
default. Instead of stopping after the old fixed 3-pass cap, the nodes
re-estimate the full request budget after every compression pass and continue
until the message window plus static reserve is below the effective active
limit, bounded by `ALPHARAVIS_DYNAMIC_COMPRESSION_MAX_PASSES` and
`ALPHARAVIS_DYNAMIC_COMPRESSION_HARD_MAX_PASSES`.

Provider overflow retry also parses provider-reported context limits from
messages such as llama.cpp `n_ctx_slot` or "maximum context length is ..."
errors. When a smaller real limit is reported, the retry stores it in
`provider_reported_context_limit`, recomputes active/hard thresholds from that
limit, and exposes the classification in `provider_context_error` /
`run_profile`.

The compression summarizer now has its own conservative prompt budget. Large
middle sections are pruned head/tail before the summary-model call while exact
raw messages remain in the archive. This fixes the observed live failure where
the active request was rescued, but the summary model itself received a
138k-token prompt against a 128k llama.cpp context and fell back to the
reference-only fail-safe summary.

The Bridge Test UI Observer now has a `Context Budget` strip so operators can
see message/reserve/request/limit numbers without digging through raw JSON. It
also shows remaining budget, detected/provider context length, rescue pass
counts, retry state, and whether pre-run/final rescue landed under budget. The
Observer now also has a dedicated `Shrinking` section. It renders compression
metadata as one card per scope, with before/after tokens, shrink percentage,
request budget after rescue, pass counts, budget status, head/middle/tail
counts, prompt-pruning status, chunking status, chunk omissions, chunk output
budget, synthesis-pruning status, and archive key. The raw `Kompression` tab is
still available for exact JSON when the visual cards point to a suspicious run.
The Bridge records those budget/compression fields for non-streaming
`runs.wait` calls too, not only event-stream update paths.

Final LLM invocations now also emit a Hermes-style request-budget estimate.
The estimate includes active messages plus model kwargs and bound DeepAgents
tool schemas, so warnings are based on the prompt that is actually about to be
sent rather than only the user-visible chat history.

Planner prompts now nudge archive retrieval when a user appears to reference
older compressed context and archive keys are available, steering agents toward
`context_retrieval_agent` / `read_archive_record(...)` instead of guessing.

### Verification

```text
pytest -q tests/test_context_compressor.py tests/test_agent_context_budget.py tests/test_bridge_responses.py
pytest -q tests/test_bridge_test_ui.py
PYTHONPYCACHEPREFIX=/tmp/alpharavis-pyc python -m py_compile langgraph-app/agent_graph.py langgraph-app/model_metadata.py langgraph-app/context_compressor.py
python scripts/alpharavis_setup.py bridge-smoke
Live llama.cpp over-budget check on 2026-05-15: `/props` reported n_ctx=128000.
A 61-message request with about 84.9k raw tokens produced
LIVE_OVER_BUDGET_OK after pre-run compression. Observer record
obs_23d3389c9d5f showed context_length=128000, request_tokens=23383,
effective_active_limit=56300, pre_run_compression_passes=1, and
pre_run_compression_budget_met=true.
Follow-up after summary prompt budgeting: the same shape of request produced
LIVE_SUMMARY_OK with summary_failed=false. The Compression tab showed
summary_prompt_pruned=true, summary_prompt_original_tokens_estimate=69413,
summary_prompt_tokens_estimate=21113, and archive_key=1b952224efdad5ef57d1fbb5.
```

Added experimental opt-in chunked summary compression behind
`ALPHARAVIS_COMPRESSION_ENABLE_CHUNKED_SUMMARY=false`. When enabled, oversized
summary windows are split into bounded overlapping chunks, each chunk is
summarized, and a final synthesis pass builds the active reference-only summary.
The raw middle is still archived, and Observer compression metadata now exposes
`summary_chunking_used`, chunk counts, chunk omissions, chunk output budget, and
synthesis-pruning stats. Chunk sizing now subtracts estimated summary prompt
wrapper/protected-note overhead plus
`ALPHARAVIS_COMPRESSION_SUMMARY_PROMPT_OVERHEAD_RESERVE_TOKENS` before deciding
how much middle payload fits in each chunk, so the summary model is less likely
to hit its own context limit. Summary output, summary prompt, and per-chunk
summary output max settings now treat `0` as "no fixed absolute cap"; the
ratio-derived budget then scales from the effective compression token limit,
which in turn comes from discovered model context length such as llama.cpp
`n_ctx`. `inspect_context_budget` now exposes those derived values under
`compression_summary_budget`, so agents and operators can retrieve the runtime
summary/prompt/chunk budgets from the same central snapshot instead of copying
small static constants. `summary_chunk_omitted_chars` is intentionally surfaced
because it is the first thing to check for quality bugs: it should stay at `0`
in normal successful chunked runs; a positive value means the synthesis prompt
now explicitly instructs the summary model to mention archive lookup for details
beyond the configured max chunk count.

Follow-up fix: the active-state compression target and the summary model
context are now passed separately. A 128k llama.cpp server can use a 96k
summary-prompt budget even when AlphaRavis is shrinking active messages under a
64k effective active limit. Observer compression records expose both values as
`compression_token_limit` and `summary_context_token_limit`.

## 2026-05-14 - Makefile Operator Reference

### Summary

Added `docs/MAKEFILE_README.md` as the canonical operator README for Makefile
targets and variables. It documents install/update/up flows, runtime
streaming profiles, Tailscale HTTPS versus LAN HTTP network modes, config UI
settings, media/vision and video-analysis variables, service targets, smoke
checks, important URLs, and troubleshooting commands.

The root `README.md` now keeps a short Makefile quickstart and links to the
Makefile README instead of duplicating the full target list. Usage notes also
link to the same Makefile README from the Daily Interface section.

### Verification

```text
make -n help
python -m py_compile scripts/alpharavis_setup.py tailscale_https_routes.py
pytest -q tests/test_alpharavis_setup.py tests/test_tailscale_https_routes.py
```

## 2026-05-14 - Tailscale/LAN Network Mode Switching

### Summary

The Makefile now treats Tailscale HTTPS and plain LAN HTTP as explicit network
exposure modes instead of only applying Tailscale routes after Docker starts.
Default `make install`, `make update`, and `make up` runs prepare Tailscale mode
first by writing `ALPHARAVIS_DOCKER_HOST_BIND=127.0.0.1`, then start/recreate
Docker services, then apply Tailscale Serve HTTPS routes. This avoids Docker
trying to bind Tailnet ports already owned by Tailscale Serve.

`TAILSCALE_AUTO=off` is now the LAN HTTP mode. It disables the managed
Tailscale Serve routes, removes the dashboard HTTPS override JSON, writes
`ALPHARAVIS_DOCKER_HOST_BIND=0.0.0.0`, and lets the normal Docker start step
publish application ports on all host interfaces. `TAILSCALE_AUTO=keep` is the
new no-op mode for runs that should leave the current exposure mode untouched.

Direct operator commands:

```text
make tailscale-apply
make tailscale-disable
make install TAILSCALE_AUTO=off
make update TAILSCALE_AUTO=off
```

### Verification

```text
python -m py_compile scripts/alpharavis_setup.py tailscale_https_routes.py
make -n install START=no SUBMODULES=no TAILSCALE_AUTO=off
make -n up TAILSCALE_AUTO=apply
pytest -q tests/test_alpharavis_setup.py tests/test_tailscale_https_routes.py
```

## 2026-05-13 - Make Config Browser UI

### Summary

`make config` now starts a local dependency-free browser UI for editing the
root `.env` file. The UI reads `.env(exaple)` as the canonical default/template
source, groups settings by its documented sections, pre-fills current `.env`
values, and saves changes back to `.env`.

Operator behavior:

```text
make config
CONFIG_HOST=127.0.0.1 CONFIG_PORT=8765 make config
```

The config server opens the browser automatically when possible and prints the
local URL for headless shells. Boolean values are shown as True/False controls,
URL-like values get normal text inputs, secret-looking keys use password
inputs, and every row has a per-key Reset button. The bottom-right Reset all
button asks for confirmation before restoring all shown values to
`.env(exaple)` defaults; Save writes the resulting values to `.env`.

`make install` and `make update` keep their existing terminal prompts for
fallback/non-browser use, but the intended central place to edit large config
sets is now `make config`.

### Verification

```text
pytest -q tests/test_alpharavis_config_server.py tests/test_alpharavis_setup.py
python -m py_compile scripts/alpharavis_config_server.py scripts/alpharavis_setup.py
```

## 2026-05-13 - Lazy Toolset Binding And Dashboard Tailscale Default

### Summary

AlphaRavis now uses the lazy toolset resolver for the specialist workers'
actual local/MCP tool binding, not only for registry text and profiles. At graph
build time, each specialist gets its materialized bounded bundle:

- `research_expert` -> research/RAG/media/context tools
- `general_assistant` -> general media/memory/skills/power categories
- `debugger_agent` -> approved local/SSH diagnostics, Docker/service checks,
  debugging lessons, and reports
- `hermes_coding_agent` -> Hermes delegation, coding context, memory, skills,
  and artifacts, but not raw local/SSH execute tools
- `context_retrieval_agent` -> archives, memory, media index status, skills,
  and artifacts
- optional power/crisis agents -> their owner/model-management bundles

Handoff tools are still added explicitly so routing remains available even when
a specialist's own toolset is narrow. `run_profile.loaded_toolsets` records the
per-agent materialized profiles, and `run_profile.selected_toolsets` continues
to record categories inferred from the latest user message.

The Tailscale helper now includes the service dashboard itself by default.
`make tailscale-plan`, `make tailscale-overrides`, `make tailscale-apply`, and
`make tailscale-disable` therefore include port `8090` unless the operator opts
out with:

```text
TAILSCALE_DASHBOARD=false
```

or direct script flags:

```text
--exclude-dashboard
ALPHARAVIS_TAILSCALE_INCLUDE_DASHBOARD=false
```

This keeps the operator flow simple: after Tailscale Serve is applied, the same
dashboard card surface is reachable inside the Tailnet over HTTPS.

### Verification

```text
PYTHONPYCACHEPREFIX=/tmp/alpharavis-pycache python -m py_compile \
  langgraph-app/alpharavis_toolsets.py \
  langgraph-app/agent_graph.py \
  tailscale_https_routes.py \
  tests/test_tailscale_https_routes.py \
  tests/test_alpharavis_toolsets.py

pytest -q tests/test_alpharavis_toolsets.py tests/test_tailscale_https_routes.py

python tailscale_https_routes.py plan --tailscale-host test-device.tailnet.ts.net
python tailscale_https_routes.py plan --tailscale-host test-device.tailnet.ts.net --exclude-dashboard
```

## 2026-05-13 - Vision Embedding Endpoint Configuration

### Summary

Media/vector indexing can now target a dedicated external vision embedding
server without changing the normal LiteLLM route. The vision embedding client
prefers:

```text
ALPHARAVIS_VISION_EMBEDDING_MODEL_URL
ALPHARAVIS_VISION_EMBEDDING_BASE_URL
VISION_EMBEDDING_API_BASE
ALPHARAVIS_PGVECTOR_EMBEDDING_BASE_URL
OPENAI_API_BASE
```

This lets the operator run a small OpenAI-compatible llama.cpp server on
another machine and point AlphaRavis directly at it:

```text
ALPHARAVIS_ENABLE_VISION_VECTOR_MEMORY=true
ALPHARAVIS_VISION_EMBEDDING_MODEL_URL=http://<vision-embedding-host>:<port>/v1
ALPHARAVIS_VISION_EMBEDDING_MODEL=<model-name>
```

Docker, setup prompts, and the Makefile now expose the direct URL. `make
media-vision`, `make install`, `make update`, `make up`, `make
up-fullstreaming`, and `make up-chat-fullstreaming` accept:

```text
VISION_ENABLED=true
VISION_URL=http://<vision-embedding-host>:<port>/v1
VISION_MODEL=<model-name>
VISION_FALLBACK=<fallback-model>
```

When `VISION_URL` is passed to `make up`, the Makefile writes the `.env` values
before Docker Compose starts. Captioning, OCR, and transcription remain
separate future work.

Vision/video indexing uses the existing durable `alpharavis_embedding_jobs`
queue with `job_type=media_analysis`; there is no separate vision queue table.

### Verification

```text
pytest -q tests/test_media_analysis.py
pytest -q tests/test_alpharavis_setup.py

PYTHONPYCACHEPREFIX=/tmp/alpharavis-pycache python -m py_compile \
  langgraph-app/vector_memory.py \
  scripts/alpharavis_setup.py \
  tests/test_media_analysis.py \
  tests/test_alpharavis_setup.py
```

## 2026-05-13 - Service Dashboard Redirector

AlphaRavis now has a lightweight `service-dashboard` Compose service on
`http://localhost:8090`. It serves `service_redirector_server.py`, a standalone
Python stdlib dashboard that lists the stack's main UI/API/database endpoints
as dark clickable service cards plus `/services.json` and `/health`.

`make up`, `make install`, and `make update` start it with the base Compose
stack. `make service-dashboard` or `make dashboard` starts only this redirector.
The existing Bridge Test UI remains part of the normal base stack and keeps its
dedicated `make test-ui` target.

`tailscale_https_routes.py` can generate and apply Tailscale Serve HTTPS routes
for the redirector's local HTTP services inside the Tailnet. It deliberately
uses `tailscale serve`, not Tailscale Funnel, so it does not publish services to
the public internet. The Makefile exposes safe planning and override-generation
targets:

```text
make tailscale-plan TAILSCALE_HOST=<device>.<tailnet>.ts.net
make tailscale-overrides TAILSCALE_HOST=<device>.<tailnet>.ts.net
```

`make tailscale-apply` runs `tailscale serve --bg --https=<port>` for each
local HTTP service, writes
`service-dashboard-data/tailscale_service_urls.json`, and restarts the
dashboard. The dashboard reads that file automatically in `auto` mode and shows
the generated HTTPS URLs instead of localhost URLs while preserving local URLs
on each card.

Follow-up: the normal operator flows now run that apply step automatically.
`make install`, all install profile targets, `make update`,
`make update-no-start`, `make up`, `make up-fullstreaming`, and
`make up-chat-fullstreaming` call `tailscale-auto`, which defaults to
`tailscale-apply`. This older note originally treated `TAILSCALE_AUTO=off` as a
skip flag; as of 2026-05-14 it is the explicit LAN HTTP mode, documented above
in "Tailscale/LAN Network Mode Switching".

Follow-up: Tailscale sudo handling now defaults to `auto`. The helper first
tries the Tailscale CLI without sudo. If it gets a permissions-style failure,
it asks for the sudo password and retries the command through `sudo -S`.
Operators can force old behavior with `TAILSCALE_SUDO=true` or disable sudo
retry with `TAILSCALE_SUDO=never`.

## 2026-05-13 - Hermes-Inspired Provider Error Handling Phase A-D

### Summary

The provider-hardening follow-up now has a staged plan in
`docs/ALPHARAVIS_OPEN_TASKS.md`, focused on Hermes-style error handling without
copying Hermes's full provider adapter stack into AlphaRavis.

Phase A/B adds a central compatibility layer in
`langgraph-app/provider_hardening.py` and applies it to direct internal
Responses calls plus ChatLiteLLM fallback kwargs:

- Direct `/v1/responses` calls still retry once after safe unsupported-parameter
  errors.
- If the compatibility retry also fails, the raised/classified error now keeps
  both the original provider error and the retry error.
- Successful direct Responses calls attach `responses_compatibility_retry` to
  the returned AI message when a retry happened; planner and fast-path nodes
  copy that into `run_profile.provider_hardening_last_retry`.
- ChatLiteLLM fallback calls omit server-managed `temperature` for
  Kimi/Moonshot-style endpoints and can map `max_tokens` to
  `max_completion_tokens` for direct OpenAI/GitHub GPT-4o/o-series/GPT-5-style
  endpoints.
- Local LiteLLM/llama.cpp defaults stay conservative: `max_tokens` is not
  remapped unless the endpoint/model profile calls for it.
- A reviewed repo skill, `provider-error-hardening`, now captures the reusable
  Hermes-style workflow for provider failures.
- Chunk 8 maintenance helpers now include deterministic thread/archive title
  suggestions and review-only insight extraction; they do not auto-promote
  memory.

Phase C/D completes the staged provider-hardening plan without importing the
Hermes provider stack:

- Provider profiles are narrow request-shape profiles, not transports.
  `ALPHARAVIS_PROVIDER_PROFILE=auto` preserves local LiteLLM/llama.cpp behavior,
  detects Kimi/Moonshot server-managed sampling, and maps direct OpenAI/GitHub
  reasoning-style Chat token limits to `max_completion_tokens`.
- `ALPHARAVIS_PROVIDER_PROFILE=responses_required` or
  `ALPHARAVIS_CHAT_FALLBACK_MODE=responses_required` blocks silent fallback from
  direct Responses calls to ChatLiteLLM when runtime evidence shows that Chat
  Completions is broken for a provider.
- Direct Responses and ChatLiteLLM operational logs include provider-profile
  metadata. Planner and fast-path messages can copy the active profile into
  `run_profile.provider_hardening_profile`.
- Direct non-OpenAI adapters remain disabled by policy. AlphaRavis keeps
  LiteLLM/LangChain as the provider route unless a future, documented feature
  gap proves a direct adapter is required.

### Hermes Operational Fix

The Hermes Compose entrypoint now synchronizes persisted
`$HERMES_HOME/config.yaml` model fields from AlphaRavis env values before
starting Hermes. This fixes a stale-volume failure where Hermes still routed to
OpenRouter/Claude and returned `HTTP 502` because OpenRouter auth was missing,
even though `.env` configured `custom`, `big-boss`, and local LiteLLM.

After recreating `hermes-agent`, the log showed:

```text
Synced Hermes model config from AlphaRavis env: default=big-boss, provider=custom, base_url=http://litellm:4000/v1
```

`make hermes-smoke` then returned assistant content `OK`.

### Verification

```text
pytest -q tests/test_provider_hardening.py \
  tests/test_responses_client_error_handling.py \
  tests/test_error_classifier.py \
  tests/test_maintenance_helpers.py \
  tests/test_alpharavis_setup.py \
  tests/test_repo_skills.py

PYTHONPYCACHEPREFIX=/tmp/alpharavis-pycache python -m py_compile \
  langgraph-app/provider_hardening.py \
  langgraph-app/responses_client.py \
  langgraph-app/agent_graph.py \
  langgraph-app/maintenance_helpers.py \
  scripts/alpharavis_setup.py \
  tests/test_provider_hardening.py \
  tests/test_responses_client_error_handling.py \
  tests/test_maintenance_helpers.py

bash -n scripts/hermes_patched_entrypoint.sh scripts/apply_hermes_agent_patches.sh
docker compose up -d --force-recreate hermes-agent
make hermes-smoke
```

## 2026-05-13 - Bridge Test UI Streaming Proxy

### Summary

`bridge-test-ui` now sends through a streaming proxy by default. The browser
posts to `/api/send_stream`; the test UI server forwards to either
`/v1/responses` or `/v1/chat/completions` with `stream=true`, then proxies the
Bridge SSE stream back to the browser. The browser reads the response body as a
stream, renders `response.output_text.delta` or Chat Completions
`delta.content` as it arrives, and records the raw SSE events plus a browser-side
stream trace.

Assistant messages now also get a collapsed `Reasoning` panel when reasoning is
present. Responses `response.reasoning.delta` events and Chat Completions
`delta.reasoning_content` / `delta.reasoning` fields are appended live while
the normal answer text continues to render separately.

Follow-up fix: the browser SSE parser escapes newline regexes inside the Python
HTML string (`\\r?\\n`) so the delivered JavaScript remains valid and the Send
button handler can attach normally.

Follow-up session fix: `Verlauf leeren` now creates a new backend session id in
addition to clearing browser messages. Previously the UI kept a persistent
`session_id` in `localStorage`, so a visually empty test UI could still resume
an old LangGraph thread and make prompts look like stale hidden context was
being injected.

Follow-up context leak fix: Bridge output scrubbing now treats
`<current-task-brief>` and `<execution-plan>` as internal-only blocks, and the
state fallback no longer returns the last non-AI message as assistant text. The
Bridge Test UI also sends Responses history as structured message items instead
of flattening it into a synthetic `Chat history: ...` user prompt. This prevents
internal task briefs or synthetic prompt wrappers from appearing as assistant
answers when an agent turn emits no visible final AI text.

If a streamed run produces no visible final AI message but LangGraph state has a
failed trace step, the Bridge now emits a concise failure message instead of an
empty assistant response. This surfaced the actual local failure during testing:
the agent swarm model call failed through LiteLLM with `InternalServerError` /
provider connection error for model group `big-boss`.

The Bridge Test UI now shows a per-message route badge. It marks streamed
responses as `Fast Path`, `Agent Path`, or `Hard Stop` based on LangGraph
stream activity such as `fast_chat`, `planner`, `memory_kernel`, `skill_library`,
and `swarm`, and mirrors that route in the top status line.

Trace readability follow-up: the Test UI now compacts consecutive answer-text
delta trace rows by default. The raw per-delta rows are still available with
the `Delta-Details` checkbox in the Trace header. This keeps normal traces
readable while preserving precise timing diagnostics for streaming stalls.

Reasoning panel follow-up: the Test UI separates streamed LangGraph lifecycle
statuses from model-provided reasoning. `Status: ...` deltas are displayed in a
dedicated `Status` block, while non-status reasoning deltas stream into a
`Modell-Reasoning` block below it. This is a UI-only split; the Bridge event
streaming protocol is unchanged.

Planner-stream follow-up: streamed text emitted by the LangGraph `planner` node
is now treated as internal reasoning instead of visible assistant output in both
Responses and Chat Completions streaming. Responses events include
`alpha_reasoning_kind=internal_plan`; Chat Completions deltas include the same
AlphaRavis marker next to `reasoning_content`. The Test UI renders those deltas
in an `Interner Plan` block, while the final swarm answer remains normal
`output_text` / `delta.content`.

Bridge Test UI streaming follow-up: LangGraph `updates` payloads from the
`planner` node are now converted into `response.reasoning.delta` events with
`alpha_reasoning_kind=internal_plan`, so the plan is visible as soon as the
planner update arrives instead of waiting for the final swarm message. Responses
reasoning extraction is enabled by default through
`BRIDGE_RESPONSES_STREAM_REASONING_EVENTS=true`; the older
`BRIDGE_STREAM_REASONING_EVENTS` flag still controls the Chat Completions
reasoning field. The Test UI also has fixed live panes for Status, Reasoning,
and Planer while keeping the per-message collapsed reasoning details.

Planner visibility fix: LangGraph can stream planner tokens as
`messages/partial` while sending their node metadata separately in a preceding
`messages/metadata` event. The Bridge now tracks message IDs to their
LangGraph node and keeps per-message delta buffers, so planner partials route
only to internal reasoning and do not appear in the visible assistant message.
The Test UI live panes can be expanded with a `Gross` / `Klein` toggle for
longer plans.

Reasoning hygiene follow-up: text-only conversion and visible bridge output now
suppress `[thinking content block omitted]` / `[reasoning content block
omitted]` placeholders. Real provider reasoning still goes to the reasoning
stream when it is exposed as `reasoning_content`, `reasoning`, or visible
`<think>` text.

Responses delta-smoothing follow-up: visible assistant text and model/plan
reasoning are split into character-level SSE deltas through
`BRIDGE_RESPONSES_OUTPUT_DELTA_MAX_CHARS=1` and
`BRIDGE_RESPONSES_REASONING_DELTA_MAX_CHARS=1`. Status events remain whole
status lines. This keeps Bridge Test UI and LibreChat Responses rendering from
receiving visibly jumpy multi-token chunks when the provider or LangGraph emits
larger partials.

Swarm streaming fix: Bridge LangGraph run streaming now sets
`stream_subgraphs=true` through `BRIDGE_STREAM_SUBGRAPHS=true`. The
AlphaRavis Swarm is a nested compiled graph under the top-level
`alpha_ravis_swarm` node; without subgraph streaming, the Bridge only saw the
completed Swarm result and could not forward worker `messages/partial` token
deltas. Direct LangGraph probing with subgraphs enabled produced hundreds of
worker partials with the first text-like event around 2-3 seconds instead of
waiting for the full Swarm run to finish.

Bridge Observer follow-up: `api-bridge` now stores a bounded in-memory request
observer buffer, exposed at `/_alpharavis/bridge-observer`. `bridge-test-ui`
adds a full-page `/observer` view with a wide request table, `Senden` /
`Empfang` tabs, and `Nur Kontext` / `Vollansicht` modes. The context view shows
the raw incoming messages, derived thread key/id, and the exact
`model_context_messages` payload prepared for LangGraph, which is intended to
debug LibreChat thread-id and hard-context-cutoff issues.

LibreChat thread isolation follow-up: the Bridge no longer treats `body.user`
as a conversation/thread key by default. LibreChat sends that field as the user
identity on the observed chat-completions path, so using it as a thread key made
separate visible chats share the same persistent LangGraph state. The Bridge now
requires an explicit conversation id/header for stable threads and otherwise
uses an ephemeral LangGraph thread for that request. The Observer also records a
LangGraph state profile so old-state reuse is visible alongside the exact model
context. Active-context token estimates now strip UI reasoning/thinking blocks
and provider usage metadata before enforcing the hard context limit.

Fast-path reply marker follow-up: visible fast-path replies now append only the
short `ALPHARAVIS_FAST_PATH_NOTICE_TEXT` marker, defaulting to `Fastpath`, after
the model answer. The previous explanatory prefix about the fast-path route and
thread lock is no longer included in the assistant text.

Pre-run context compression fix: the LangGraph flow now runs
`pre_run_context_guard` between `run_profile_start` and `route_decision`. This
means old checkpointed thread state can be compacted before the hard context
cutoff and before fast-path or swarm model calls. If a thread is already above
the hard limit and normal Hermes-style compression fails or cannot reduce it
enough, `ALPHARAVIS_ENABLE_HARD_CONTEXT_TRIM=true` removes old active messages
while preserving the latest user turn and records the trim in `run_profile`.
The shared context estimator now ignores UI-only `thinking` / `reasoning`
blocks in both compression and model metadata token estimates.

Bridge Test UI context terminal follow-up: Bridge streaming now emits semantic
context activity reasoning events for `context_compaction` and `context_hard`
when LangGraph reports pre-run/post-run/handoff compression, hard trim, or hard
cutoff state. The Test UI shows those events in a dedicated `Kontext` live
terminal: compaction is yellow and hard trim/cutoff is red.

Docker/Tailscale operational follow-up: published ports for Hermes,
LangGraph, the bridge, media gallery, Bridge Test UI, LiteLLM, RAG API,
DeepAgents UI, custom agent UI, LibreChat, OpenWebUI, service dashboard,
Hermes dashboard, and Pixelle can now be bound to a specific host address
through `ALPHARAVIS_DOCKER_HOST_BIND`. The default remains `0.0.0.0`; this
local run used `127.0.0.1` because Tailscale Serve was already listening on the
tailnet IP ports and proxying to localhost.

Operational follow-up: `bridge-test-ui` is treated as a normal base-stack
service for operator workflows. `make up`, `make install`, and `make update`
already run Docker Compose without a service filter, so they build/start it with
the rest of the stack. The explicit `make build`, `make up-fullstreaming`, and
`make up-chat-fullstreaming` targets now include `bridge-test-ui`, and
`make status` lists `http://localhost:8140`.

The old `/api/send` JSON route remains available as a non-streaming fallback and
diagnostic path.

### Current Limitation

This fixes the Test UI buffering problem. It does not force LangGraph or
DeepAgents to produce token-level final-answer chunks when an internal agent
turn only emits a complete AI message. In that case the UI still shows early
SSE lifecycle, activity, and reasoning/status events, but visible answer text
arrives when the Bridge receives it from LangGraph.

### Verification

```text
PYTHONPYCACHEPREFIX=/tmp/alpharavis-pycache python -m py_compile \
  langgraph-app/test_ui_server.py tests/test_bridge_test_ui.py

PYTHONPYCACHEPREFIX=/tmp/alpharavis-pycache python -m py_compile \
  scripts/alpharavis_setup.py

make -n build
make -n up-fullstreaming
make -n up-chat-fullstreaming

python scripts/alpharavis_setup.py status

pytest -q tests/test_bridge_test_ui.py tests/test_bridge_responses.py \
  tests/test_context_hygiene.py
```

## 2026-05-12 - Bridge Test UI Waterfall Trace

### Summary

`bridge-test-ui` now shows a per-request waterfall trace. Each request gets a
`trace_id` in metadata and the Bridge carries it through to LangGraph input.
The UI displays browser/server elapsed time plus Bridge and LangGraph timing
steps such as thread setup, run payload preparation, LangGraph wait duration,
fast-chat start, primary/fallback LLM call duration, and completion/failure.
The browser code avoids hard dependencies on `crypto.randomUUID` and disables
HTML caching so a reload picks up the current test UI script.

This is intentionally local and lightweight. It complements LangSmith instead
of replacing it: LangSmith can inspect LangGraph internals, while this trace is
focused on the operator path from browser to Bridge to LangGraph/model backend.

The trace was expanded to cover the normal agent path, including route
decision, planner duration, memory prefetch, skill lookup, handoff guard, and
swarm start/finish markers.

Agent-path latency was reduced by bounding two previously unbounded or
over-large steps:

- Planner calls now default to `ALPHARAVIS_PLANNER_MAX_TOKENS=768`,
  `ALPHARAVIS_PLANNER_TEMPERATURE=0`, and
  `ALPHARAVIS_PLANNER_DISABLE_THINKING=true`.
- Memory-kernel prefetch substeps now default to
  `ALPHARAVIS_MEMORY_PREFETCH_STEP_TIMEOUT_SECONDS=4`, so slow
  semantic/pgvector retrieval cannot block every agent-path reply for ~20s.
- DeepAgents model calls are forced through text-only content normalization by
  default (`ALPHARAVIS_FORCE_TEXT_ONLY_AGENT_MODEL_CONTENT=true`). This keeps
  middleware-generated `content[].type` blocks from reaching local
  OpenAI-compatible llama.cpp/LiteLLM backends that only accept string message
  content.

### Verification

```text
PYTHONPYCACHEPREFIX=/tmp/alpharavis-pycache python -m py_compile \
  langgraph-app/test_ui_server.py \
  langgraph-app/bridge_server.py \
  langgraph-app/agent_graph.py

pytest -q \
  tests/test_bridge_responses.py::test_run_wait_content_reads_nested_langgraph_values_state \
  tests/test_bridge_responses.py::test_response_object_has_stable_ids_and_usage
```

Docker smoke after rebuilding `bridge-test-ui`, `api-bridge`, and
`langgraph-api`:

```text
Responses: trace_codexsmoke returned TRACE_TEST_OK with Bridge and LangGraph steps.
Chat Completions: trace_codexchat returned TRACE_CHAT_OK with Bridge and LangGraph steps.
Browser compatibility fix smoke: page returned `Cache-Control: no-store`, no
direct `crypto.randomUUID().replaceAll` dependency remained, and `/api/send`
returned UI_FIX_OK through the Trace path.

Latency follow-up smoke:

Fast path:
  total about 2.1s
  route_decision about 0.14s
  primary LLM about 1.17s

Agent path before planner/memory caps:
  total about 50.2s
  planner about 28.1s
  memory/skill/handoff pre-swarm gap about 19.8s

Agent path after caps:
  total about 11.2s
  planner about 4.0s, plan_chars=765
  curated memory about 0.0s
  semantic memory timed out at 4.0s
  swarm about 1.9s

Tool-list regression:
  prompt "welche tools hast du" no longer fails with
  "unsupported content[].type"; it returned a tool overview through the Swarm.
  Total about 19.9s, with planner about 3.0s, semantic memory capped at 4.0s,
  and swarm about 11.8s.
```

## 2026-05-12 - Minimal Bridge Test UI

### Summary

Added `bridge-test-ui`, a small FastAPI/HTML test surface for isolating
LibreChat from Bridge/LangGraph failures. It serves on:

```text
http://localhost:8140
```

The UI stores chat history only in browser memory and posts through its own
proxy endpoint to `api-bridge`, with a protocol switch for Responses vs Chat
Completions. It also shows the last raw Bridge response so UI/persistence
problems can be separated from backend model errors.

### Verification

```text
PYTHONPYCACHEPREFIX=/tmp/alpharavis-pycache python -m py_compile langgraph-app/test_ui_server.py
docker compose config --quiet
docker compose up -d --build bridge-test-ui
```

Container health returned:

```text
{"ok":true,"bridge_base_url":"http://api-bridge:8123/v1","model":"my-agent"}
```

Live proxy calls reached `api-bridge`, but the current model backend returned
`InternalServerError` from LiteLLM for both `big-boss` and the `edge-gemma`
fallback. That confirms the new UI bypasses LibreChat and exposes the active
backend failure directly.

## 2026-05-12 - Bridge Non-Streaming Agent Output Extraction

### Summary

The Bridge now unwraps LangGraph `{"values": ...}` state objects before reading
the final AI message. This fixes successful non-streaming `/v1/responses`
agent-path runs that previously completed with an empty `output_text` because
the Bridge looked for `messages` on the wrapper object instead of inside
`values`.

### Verification

```text
pytest -q tests/test_bridge_responses.py
```

Live Docker retest after recreating `langgraph-api` and `api-bridge`:

```text
POST /v1/responses
  input="kein fast path. Antworte exakt: RESP_AGENT_AFTER_FIX_OK"
  -> 200, output_text="RESP_AGENT_AFTER_FIX_OK"

POST /v1/chat/completions
  user="Antworte exakt mit CHAT_AFTER_FIX_OK"
  -> 200, content contains "CHAT_AFTER_FIX_OK"
```

## 2026-05-19 - Bridge Mirrors Chat Images And Groups Media Gallery Assets

### Summary

Incoming LibreChat/OpenAI Responses image blocks now use the same safe
media-gallery mirror path as videos:

```text
BRIDGE_MEDIA_GALLERY_AUTO_REGISTER_IMAGES=true
BRIDGE_MEDIA_GALLERY_AUTO_REGISTER_VIDEOS=true
```

The Bridge rewrites the AlphaRavis-facing media marker to the stable
media-gallery URL after registration. LibreChat's original visible attachment
and upload record remain untouched. Images and videos from the same chat share
`thread_id`, `thread_key`, and `group_id`, so gallery views can group them
together without duplicating video semantics.

The media server now supports thread/group filters and date/name/type/kind
sorting on `/assets` and `/gallery`. `/gallery` also exposes grouping modes for
day+group, thread, group, date, and media type.

### Verification

```text
pytest -q tests/test_bridge_responses.py tests/test_media_server.py
PYTHONPYCACHEPREFIX=/tmp/alpharavis-pycache python -m py_compile \
  langgraph-app/bridge_server.py \
  langgraph-app/media_server.py
```

## 2026-05-19 - Oversized Latest Paste Forces Chunked Compression Rescue

### Summary

Ordinary over-budget compression still keeps chunked summary off by default:

```text
ALPHARAVIS_COMPRESSION_ENABLE_CHUNKED_SUMMARY=false
```

However, when oversized-tail rescue has to move the latest user message into
the compressible middle, the compressor now forces chunked summary if the
summary prompt would otherwise be pruned. This covers the case where a pasted
file/request is too large to fit as protected latest context. Exact raw middle
content is still archived, and any large-paste RAG/source reference created
before compression remains available for later lookup.

### Verification

```text
pytest -q tests/test_context_compressor.py
PYTHONPYCACHEPREFIX=/tmp/alpharavis-pycache python -m py_compile \
  langgraph-app/context_compressor.py
```

## 2026-05-12 - Bridge Mirrors Incoming Chat Videos Into Media Gallery

### Summary

Incoming LibreChat/Open Responses video blocks are now mirrored into
`media-gallery` before AlphaRavis constructs metadata-only model context:

```text
BRIDGE_MEDIA_GALLERY_AUTO_REGISTER_VIDEOS=true
```

After a successful mirror, the Bridge-held video part points at the stable
media-gallery URL, so the resulting LangGraph context marker references
`/media/...` instead of carrying the original inline video URL/blob. LibreChat's
visible attachment card and original local upload record are not rewritten in
this phase.

`media-gallery` now accepts both HTTP(S) video sources and inline `data:` video
blocks. Inline bytes are saved under `media-data`, while Mongo asset metadata
stores an omitted placeholder rather than the full base64 payload.

### Verification

```text
PYTHONPYCACHEPREFIX=/tmp/alpharavis-pycache python -m py_compile \
  langgraph-app/bridge_server.py \
  langgraph-app/media_server.py \
  tests/test_bridge_responses.py \
  tests/test_media_server.py

pytest -q \
  tests/test_bridge_responses.py \
  tests/test_media_server.py \
  tests/test_media_analysis.py \
  tests/test_alpharavis_toolsets.py

docker compose config --quiet
```

## 2026-05-12 - Explicit Video Analysis Preparation

### Summary

AlphaRavis now has an explicit media preparation path for videos:

```text
prepare_media_for_model
inspect_media_index_status
```

Media remains safe-by-default. The Bridge still converts raw media content
parts into metadata markers, and media registration no longer implies
vision-index processing unless explicitly requested.

Media gallery presence is not treated as indexed. Assets and chat references
are separate from vector index records. Automatic indexing is ENV-controlled
for user uploads, Pixelle MCP / ComfyUI outputs, and link references, and
media-analysis jobs dedupe by media source key plus model-card id, index
version, and chunking config hash.

The preparation tool decides between:

- `register_only`: store metadata, no download
- `pass_through`: keep the URL for Pixelle/downstream tools, no download
- `analyze`: download, probe, sample bounded frames, and index sampled frames
- `index`: queue a durable `media_analysis` job in `alpharavis_embedding_jobs`
  for retrieval-oriented indexing

Video download/frame extraction only runs when:

```text
ALPHARAVIS_VIDEO_ANALYSIS_ENABLED=true
```

Frame embeddings are written only when:

```text
ALPHARAVIS_ENABLE_VISION_VECTOR_MEMORY=true
```

### Why This Was Needed

Videos should not silently enter the model context or vision embedding path.
The model can decide from the user's wording whether a URL should be passed to
Pixelle, merely registered, or actually analyzed, while the tool enforces
download limits, FPS, frame caps, cache paths, and model-card defaults.

### Files Changed

- `langgraph-app/media_analysis.py`
  - model-card resolution, mode decision, bounded download, `ffprobe` probing,
    `ffmpeg` frame extraction, timestamped manifests
- `langgraph-app/model_cards.json`
  - Qwen3.6-35B-A3B defaults and `big-boss` aliases
- `langgraph-app/agent_graph.py`
  - `prepare_media_for_model`, `inspect_media_index_status`, metadata-only
    registration default, `inspect_embedding_queue_status`, context-agent wiring
- `langgraph-app/media_server.py`
  - optional derivation/original/processed fields on `/assets/register`
  - separate Mongo `references` collection for chat/tool appearances of a
    media asset
  - `/assets/resolve` to map copied gallery/source URLs back to Mongo asset
    metadata and references
  - `/assets` filtering by `asset_kind`
  - `/gallery` tabs for All/Original/Processed and per-card copy/open actions
- `langgraph-app/vector_memory.py`
  - media analysis enqueueing through `alpharavis_embedding_jobs`, media queue
    status query, media index status query, and video searches that include
    indexed `video_frame` records
- `langgraph-app/Dockerfile`
  - installs `ffmpeg` / `ffprobe`
- `.env(exaple)`, `Makefile`, `scripts/alpharavis_setup.py`
  - video-analysis switches, `make video-analysis`, status output
  - auto-indexing switches for user uploads, Pixelle MCP / ComfyUI outputs,
    link references, media index version, and media vision model-card id
- docs and tests
  - usage/architecture/open-task notes plus focused media-analysis and Bridge
    tests

### Verification

```text
PYTHONPYCACHEPREFIX=/tmp/alpharavis-pycache python -m py_compile \
  langgraph-app/media_analysis.py \
  langgraph-app/vector_memory.py \
  langgraph-app/agent_graph.py \
  scripts/alpharavis_setup.py

pytest -q \
  tests/test_media_analysis.py \
  tests/test_bridge_responses.py::test_responses_input_supports_instructions_and_content_parts \
  tests/test_alpharavis_toolsets.py

12 passed
```

## 2026-05-11 - LibreChat Command Approval Memory

### Summary

LibreChat command approvals still use the chat-text fallback because the
external custom endpoint path does not expose an AlphaRavis-native clickable
permission callback.

Accepted replies while a command approval interrupt is pending:

```text
approve
reject
replace: <safer command>
approve always
immer erlauben
```

`approve always` / `immer erlauben` stores a bridge-local allow entry for the
exact scope/target/command in the current LibreChat thread only. It is not a
global allowlist and is cleared when `api-bridge` restarts.

### Why This Was Needed

OpenAI Responses has MCP approval request/response items for remote MCP tools,
and AionUI/ACP has a native `session/request_permission` flow. LibreChat's
custom OpenAI-compatible endpoint path does not provide that same AlphaRavis
permission callback, so the robust path is to keep text approvals and make the
"remember this exact command in this chat" case explicit.

### Files Changed

- `langgraph-app/bridge_server.py`
  - parses `approve always` / `immer erlauben`
  - stores exact command fingerprints in process memory per thread
  - auto-resumes only when the same pending interrupt reappears in that thread
- `tests/test_bridge_responses.py`
  - covers command-memory parsing and exact-command matching
- Responses, usage, architecture, and open-task docs
  - document the LibreChat limitation and the supported fallback commands

### Verification

```text
pytest -q tests/test_bridge_responses.py tests/test_alpharavis_acp_adapter.py
36 passed

pytest -q tests
110 passed
```

## 2026-05-11 - Responses / DeepAgents Streaming Fix

### Summary

AlphaRavis now runs LangGraph/DeepAgents through the Responses API by default
and enables LangChain's hybrid streaming mode for DeepAgents:

```text
ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING=true
ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=tool_calling
```

The hybrid mode means:

- model calls without bound tools may stream tokens
- model calls with tools are routed through non-streaming model calls
- the Bridge can still expose `/v1/responses` SSE events to clients

This is different from full streaming. Full streaming with
`ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=false` still sends streaming
tool-capable model calls to the provider and remains experimental for the local
LiteLLM/llama.cpp stack.

### Why This Was Needed

LangChain documents `disable_streaming="tool_calling"` as a way to bypass
streaming when tools are passed. In the installed `langchain-openai==1.2.1`,
that routing reached non-streaming `_generate` / `_agenerate` code paths, but
the payload still included `stream=true`. The OpenAI client then returned a
Stream/AsyncStream object where LangChain expected a completed response object.

Observed failure:

```text
AttributeError: 'AsyncStream' object has no attribute 'error'
```

The issue is tracked upstream:

```text
https://github.com/langchain-ai/langchain/issues/35436
```

### Fix Choice

Two upstream PR/fork fixes were compared locally:

- `https://github.com/langchain-ai/langchain/pull/35440`
  - small three-line fix in `_get_invocation_params`
- `https://github.com/langchain-ai/langchain/pull/35457`
  - forces `payload["stream"] = False` in `_generate` and `_agenerate`
  - includes regression tests
  - documents that `tool_calling` disables streaming for all calls while tools
    are bound, not only calls that eventually produce tool calls

AlphaRavis applies the PR #35457 approach because it patches the concrete
non-streaming code paths that crashed in the local repro.

### Files Changed

- `langgraph-app/patches/patch_langchain_openai_disable_streaming.py`
  - startup patch for `langchain_openai.chat_models.base`
  - idempotent; exits if already applied
- `docker-compose.yml`
  - runs the patch before `langgraph dev`
  - default DeepAgents Responses streaming set to hybrid mode
- `langgraph-app/Dockerfile`
  - mirrors the same startup patch for image defaults
- `.env(exaple)`
  - documents the new streaming flags and defaults
- `langgraph-app/requirements.txt`
  - updated/pinned LangChain, LangGraph, DeepAgents, OpenAI package versions
- Responses/usage/architecture docs
  - document the patch, current limitations, and verification results

### Verification

Package state in `langgraph-api` after update:

```text
langchain-openai==1.2.1
langchain==1.2.18
langchain-core==1.3.3
langgraph==1.1.10
deepagents==0.5.9
openai==2.36.0
litellm==1.83.0
```

Runtime checks:

```text
patch_marker True
STREAMING=true
DISABLE=tool_calling
```

Direct repro after patch:

```text
DIRECT_TOOL_STREAM_TEST_OK events=36
```

Bridge `/v1/responses` Agent Path streaming after patch:

```text
PATCHED_AGENT_STREAM_OK
```

Local tests:

```text
83 passed
```

### Remaining Limitations

- Full internal Responses streaming with tools
  `ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=false` is still
  experimental.
- Hybrid mode cannot know in advance whether the model will produce a tool call.
  It only knows whether tools are bound to the call. Therefore, when tools are
  bound, LangChain bypasses internal token streaming for that whole model call.
- The external Bridge SSE stream can still emit final output text as Responses
  events. This is not the same as token-by-token provider streaming for every
  internal tool-capable model call.
- Remove the startup patch only after `langchain-openai` ships the upstream fix
  and the direct tool-calling repro passes without local modification.

### Future Design: Text Streaming With Safe Tool Execution

The safest design for "show text while preserving reliable tools" is not full
streaming of tool-call chunks. Tool-call arguments are structured JSON and may
arrive split across many chunks; executing them before the final chunk is unsafe.

A safer two-phase design would be:

1. Run tool-capable planner/worker turns with
   `disable_streaming="tool_calling"` so tool calls are complete before
   execution.
2. Execute tools only after the complete model response is available.
3. After tools finish, run a final answer model call without bound tools and
   stream that text token-by-token to the UI.

This would allow visible final-answer token streaming while keeping tool calls
reliable. It requires changes in the agent orchestration layer, because current
DeepAgents/React-style workers keep tools bound on every model call, including
the final answer turn.

Detailed follow-up plan:

```text
docs/ALPHARAVIS_RESPONSES_FULL_STREAMING_PLAN.md
```

## 2026-05-11 - Responses Full-Streaming Probe Instrumentation

### Summary

The follow-up plan now has concrete instrumentation in:

```text
scripts/probe_responses_tool_streaming.py
```

The script records both sides of the suspected failure boundary before any
runtime patch is attempted:

- raw `/v1/responses` SSE events from LiteLLM/llama.cpp
- direct LangChain `ChatOpenAI(... use_responses_api=True)` no-tool streaming
- LangGraph `create_react_agent(...).astream_events(..., version="v2")`
  chunks with `content`, `tool_call_chunks`, `tool_calls`,
  `invalid_tool_calls`, metadata, exceptions, and tracebacks

Artifacts are written under:

```text
artifacts/alpharavis/responses_streaming_probe/<run-id>/
```

That directory is ignored by git because the probe may capture prompts, model
outputs, provider headers, and tracebacks.

### Usage

Inside `langgraph-api`:

```bash
python /workspace/scripts/probe_responses_tool_streaming.py
```

From the host against the exposed LiteLLM port:

```bash
python scripts/probe_responses_tool_streaming.py --base-url http://127.0.0.1:4000/v1
```

The script exits with `0` only if all enabled probes pass. It still writes
`summary.json` and JSONL artifacts when a probe fails, so failed runs are the
expected input for deciding the next patch point.

Current probe result:

```text
run_id: codex_probe_20260511_repo_artifacts
classification: provider_litellm_or_openai_sdk
low_level_responses_sse: HTTP 408 from LiteLLM after 30 seconds
langchain_no_tool_astream: same HTTP 408
langchain_react_agent_astream_events: same HTTP 408
```

The actionable conclusion is to keep the existing hybrid runtime mode and not
apply a LangChain tool-stream buffering patch yet. The provider stream must be
fixed or bypassed first.

Follow-up after restarting the local Lamma/LAMMPS backend:

```text
run_id: codex_probe_after_lamma_restart_classified_20260511
classification: langchain_openai_conversion
raw /v1/responses SSE: ok, function-call events observed
LangChain no-tool Responses stream: ok
LangChain create_react_agent Responses stream: failed with item['content'] is empty
```

That narrowed the failure from provider availability to LangChain's Responses
stream chunk conversion/aggregation.

### Experimental Full-Streaming Patch

An env-gated patch now exists at:

```text
langgraph-app/patches/patch_langchain_openai_responses_tool_streaming.py
```

It is disabled by default and only applies when:

```text
ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING=true
```

The patch keeps the production default hybrid mode unchanged, but allows
explicit experiments with:

```text
ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=false
```

Patch behavior:

- keeps provider-reused `function_call` output indexes separate from the prior
  reasoning item
- emits final reasoning content from `response.output_item.done`
- suppresses partial `response.function_call_arguments.delta` chunks so
  incomplete JSON is not parsed as `invalid_tool_calls`
- emits one complete LangChain tool call when `response.output_item.done` for
  the function call arrives
- upgrades the earlier partial experimental patch in-place if a running
  container already has it

Verification after applying the experimental patch inside `langgraph-api`:

```text
run_id: codex_probe_experimental_patch_v5_no_force_20260511
classification: not_reproduced
raw /v1/responses SSE: ok, function-call events observed
LangChain no-tool Responses stream: ok
LangChain create_react_agent Responses stream: ok
invalid_tool_calls: 0
marker_tool_ends: 1
```

The low-level probe with `--force-tool-choice` produced no raw tool-call events
on this local provider, while the LangChain agent path still passed. Use the
default non-forced probe as the local validation path unless LiteLLM/llama.cpp
starts enforcing Responses `tool_choice` consistently.

The hybrid runtime defaults remain unchanged until Bridge-level full-streaming
checks also pass and the experimental patch has more soak time.

## 2026-05-11 - Makefile Install And Streaming Profile Refresh

### Summary

The Makefile install flow has been updated so a fresh local setup can choose
the current AlphaRavis runtime mode instead of inheriting old defaults.

New Makefile flows:

```bash
make install
make update
make install-fullstreaming
make install-chat-fullstreaming
make profiles
make streaming STREAMING=full
make streaming STREAMING=chat-full
make up-fullstreaming
make up-chat-fullstreaming
make status
```

`make install` now delegates to `scripts/alpharavis_setup.py` with explicit
install options for:

- runtime API/streaming profile
- submodule initialization
- Docker Compose profiles such as `openwebui`
- optional image build
- optional stack start

The script writes `.env` directly through the existing safe key-update helper,
so interactive and non-interactive Makefile targets use the same behavior.

### Streaming Profiles

The setup helper now supports these profiles:

```text
responses-hybrid       -> stable default Responses mode
responses-full         -> experimental full Responses tool streaming
responses-nonstreaming -> Responses mode without internal streaming
chat-full              -> Chat Completions with ChatLiteLLM streaming enabled
chat-nonstreaming      -> Chat Completions with ChatLiteLLM streaming disabled
```

Aliases remain available for short Makefile use:

```text
hybrid       -> responses-hybrid
full         -> responses-full
nonstreaming -> responses-nonstreaming
chat         -> chat-full
```

The `responses-full` profile sets:

```text
ALPHARAVIS_LLM_API_MODE=responses
ALPHARAVIS_LLM_STREAMING=true
ALPHARAVIS_DEEPAGENTS_API_MODE=responses
ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING=true
ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=false
ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING=true
BRIDGE_ENABLE_RESPONSES_API=true
BRIDGE_PREFERRED_API_MODE=responses
```

The `responses-hybrid` profile keeps the stable runtime:

```text
ALPHARAVIS_LLM_API_MODE=responses
ALPHARAVIS_LLM_STREAMING=true
ALPHARAVIS_DEEPAGENTS_API_MODE=responses
ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING=true
ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=tool_calling
ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING=false
```

The `chat-full` profile sets:

```text
ALPHARAVIS_LLM_API_MODE=chat_completions
ALPHARAVIS_LLM_STREAMING=true
ALPHARAVIS_DEEPAGENTS_API_MODE=chat_completions
BRIDGE_PREFERRED_API_MODE=chat_completions
```

`make update` now uses the same profile menu as `make install`, updates
submodules by default, and runs `docker compose up -d --build` by default after
the update. `make update-no-start` keeps the update/build flow but does not
start the stack.

### Files Changed

- `Makefile`
  - added `help`, `streaming`, `fullstreaming`, `hybrid-streaming`,
    `nonstreaming`, `chat-completions`, `chat-fullstreaming`,
    `chat-nonstreaming`, `install-fullstreaming`, `install-hybrid`,
    `install-nonstreaming`, `install-chat`, `install-chat-fullstreaming`,
    `install-chat-nonstreaming`, `profiles`, `update-no-start`,
    `up-fullstreaming`, and `up-chat-fullstreaming`
  - `make install` now accepts `STREAMING`, `SUBMODULES`, `BUILD`, `START`, and
    `PROFILES`
  - `make update` now accepts `UPDATE_STREAMING`, `UPDATE_SUBMODULES`,
    `UPDATE_BUILD`, `UPDATE_START`, and `UPDATE_PROFILES`
- `scripts/alpharavis_setup.py`
  - added streaming-profile application and status reporting
  - added numbered terminal profile selection with an info view showing exact
    env values
  - added Compose profile persistence through `COMPOSE_PROFILES`
  - added install/update-time build/start orchestration
- `.env(exaple)`
  - added documented `COMPOSE_PROFILES`
  - moved `ALPHARAVIS_LLM_STREAMING` into the main model route section because
    runtime profiles update it directly
- `README.md`, `docs/ALPHARAVIS_ARCHITECTURE.md`,
  `docs/ALPHARAVIS_USAGE_NOTES.md`, and
  `docs/ALPHARAVIS_RESPONSES_COMPATIBILITY.md`
  - document the current install and streaming architecture
- `tests/test_alpharavis_setup.py`
  - covers full-streaming env values, env update behavior, mode detection, and
    Compose profile normalization

### Verification

Local verification after the Makefile/setup changes:

```text
pytest -q tests -> 101 passed
docker compose config --quiet -> ok
git diff --check -> ok
py_compile setup/probe/test files -> ok
```

The local `.env` was then set through the same helper used by Makefile:

```bash
python scripts/alpharavis_setup.py streaming --streaming-mode full
docker compose up -d --force-recreate langgraph-api api-bridge
```

Container ENV after recreate:

```text
ALPHARAVIS_LLM_API_MODE=responses
ALPHARAVIS_DEEPAGENTS_API_MODE=responses
ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING=true
ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=false
ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING=true
BRIDGE_ENABLE_RESPONSES_API=true
BRIDGE_PREFERRED_API_MODE=responses
```

Full-streaming probe after the Makefile/ENV activation:

```text
run_id: codex_probe_makefile_fullstreaming_v2_20260511
classification: not_reproduced
raw /v1/responses SSE: ok, function-call events observed
LangChain no-tool Responses stream: ok
LangChain create_react_agent Responses stream: ok
invalid_tool_calls: 0
marker_tool_ends: 1
```

## 2026-05-11 - Hermes-Agent Local Patch Handling

### Summary

AlphaRavis keeps upstream `hermes-agent` as a submodule, but local fixes that
are needed for this stack live in the parent repo under:

```text
patches/hermes-agent/
```

The Docker containers apply those patches automatically at startup through:

```text
scripts/apply_hermes_agent_patches.sh
```

`docker-compose.yml` builds Hermes from the upstream submodule, mounts the
parent repo read-only at `/workspace`, and uses
`scripts/hermes_patched_entrypoint.sh` as a wrapper around the original Hermes
entrypoint. The wrapper runs the patch script against `/opt/hermes`, then hands
control back to `/opt/hermes/docker/entrypoint.sh`.

```text
alpharavis/hermes-agent:local
```

### Why This Exists

Submodule commits must exist in their own upstream repository. If the parent
repo points at a local-only `hermes-agent` commit, other machines and GitHub
cannot reproduce the checkout. Storing AlphaRavis-specific changes as parent
repo patches keeps the submodule clean and makes local changes reproducible.

### Current Hermes Patch

```text
patches/hermes-agent/kanban-db-duplicate-column-guard.patch
```

This patch makes Hermes kanban optional-column migrations tolerate SQLite
`duplicate column name` races/errors for these columns:

- `consecutive_failures`
- `worker_pid`
- `last_failure_error`

If Hermes kanban/task startup fails around duplicate SQLite columns, check this
patch and the Hermes startup patch command first.

### Manual Development Helper

For local debugging outside Docker, apply the same patch directly to the
submodule with:

```bash
scripts/apply_hermes_agent_patches.sh
```

The normal Docker path does not require this manual step.
