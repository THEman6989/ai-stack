# AlphaRavis RAG / Retrieval Handoff

Date: 2026-05-21

This handoff captures the intent behind the current RAG work so a new context
window can continue without re-deriving the design.

## Continue Here

This is the current operator handoff for the completed RAG/compression,
classifier, Observer, Service Dashboard, Tailscale, and Media Gallery feature
slice. The important point is that Large Paste, compression, RAG, raw-source
reads, archive recall, and the small Qwen3.5 classifier are now one coordinated
LangGraph-owned path, not separate experiments.

Current completed slice:

- Large-paste ingest is LangGraph-owned. Clients forward input and display
  metadata; they do not decide whether a paste becomes source/RAG/summary.
- Plain huge pastes are handled after compression by default. Manual
  `/rag`, `/rake`, `/index`, `/ingest`, `/big-context`, and fenced
  `<big-context>` blocks force source ingest.
- Large-paste source markers now include compact source manifests with content
  type, char counts, chunk/index stats, digest, backends, source key, and
  current-task/question handling. If no clear question exists, the marker asks
  what to analyze instead of inventing broad work.
- Raw-source and archive reads stay bounded. `read_source_chunks(...)`,
  `read_raw_source(...)`, and `read_archive_record(...)` are the exact-context
  tools after RAG points to the right source.
- Section-level mixed archive splitting is implemented for archives and archive
  collections, so prose/log/code/config sections keep order but use matching
  chunk profiles.
- Current-thread archive auto-intent is now default-on for Agent-path runs when
  archive keys exist. Fast Path never runs this node. The Qwen3.5 2B classifier
  must reject current upload/file/image/video/URL/Pixelle/active-source tasks
  unless the user explicitly asks for older/archive context.
- Explicit document/PDF-style uploads, artifacts, and large-paste sources now
  route through `retrieval_router.ingest_source(...)` where implemented.
- PGVector is now treated as the RAG/memory/skill search head, not just an
  embedding-to-ID lookup. `alpharavis_memory_vectors` schema compatibility adds
  canonical `source_id`, top-level `version`, and optional `raw_ref` beside the
  existing `source_key`, `metadata`, timestamps, and mandatory `chunk_text` /
  `content`. Semantic retrieval should answer from returned chunks by default;
  Mongo/raw-source reads are for full originals, neighboring context, complete
  chats, or original media/tool payloads.
- The Test UI/Observer includes the Small-Qwen classifier probe and queued
  embedding progress polling.
- Workflow/tool-event compaction metadata is shown in Observer shrinking cards.
- Service Dashboard separates Web Interfaces, APIs, and Infrastructure; cards
  are clickable again, use `Öffnen`, and expose local/Tailnet/HTTPS addresses
  for APIs. Pixelle and LiteLLM are represented as both UI/API where relevant.
- Media Gallery has a refreshed responsive dark layout and its own favicon.
- Tailscale path handling keeps the public dashboard path but proxies Serve to
  service roots unless a service explicitly overrides the upstream path.

Runtime endpoints and model roles:

- Big-Boss LLM base is expected from
  `BIG_BOSS_API_BASE=http://100.71.57.22:8033/v1`.
- The always-on small classifier is Qwen3.5 2B Q4_1 on the same Big-Boss host
  at port `8001`. If `ALPHARAVIS_RAG_CLASSIFIER_API_BASE` is empty,
  AlphaRavis derives `http://100.71.57.22:8001/v1` from `BIG_BOSS_API_BASE`.
- The Qwen3.5 2B classifier is only for structure/query work: long/noisy
  retrieval-query condensation, mixed prompt line ranges, and large-paste ingest
  refinement, long-prompt route classification, and archive auto-on-intent
  recall classification. It is not the answer model, not the reranker, not the
  embedding model, and not a FastPass replacement.
- The current classifier server context was recently set to 8k. Recommendation:
  test serving Qwen3.5 2B at 16k context, because archive auto-on-intent and
  mixed large-paste line-range classification benefit from more recent thread
  context. Keep classifier output small (`384`-`512` tokens), preserve bounded
  classifier windows, and compare 8k vs 16k latency/JSON validity before
  making 16k the assumed runtime.
- The active reranker endpoint is still configured separately as llama.cpp
  Qwen3-Reranker-0.6B at `http://192.168.178.140:8000/reranking`.
- The user tested CPU reranking and reported it about 4x slower. Current working
  assumption is GPU reranking plus deterministic fallback unless VRAM pressure
  proves otherwise.
- `qwen3-embedding:4b` works after GPU acceleration, but is still too slow for
  the normal path and appears capped around the embedding-server context on very
  large inputs. Keep `qwen3-embedding:0.6b` as the practical default unless a
  separate 4B collection is explicitly being measured.

Current Large Paste / RAG policy:

- Do not immediately send every large human paste into RAG.
- Manual paired markers (`/rag ... /rag`, `/rake ... /rake`,
  `/index ... /index`, `/ingest ... /ingest`, `/big-context ... /big-context`)
  and fenced `<big-context name="...">...</big-context>` blocks still force
  immediate indexing.
- Plain long pastes first go through the normal pre-run compression path so old
  context is reduced before deciding what to do with the newest paste.
- After that, `large_paste_post_compression_node` checks whether the active
  request is still above
  `ALPHARAVIS_LARGE_PASTE_RAG_POST_COMPRESSION_TRIGGER_RATIO`, default `0.80`.
  Only then is the document/code/log part indexed into RAG/raw-source and
  replaced with a source marker.
- If the source marker replacement still leaves too much non-document chatter in
  context, a follow-up compression pass can run at
  `ALPHARAVIS_LARGE_PASTE_POST_RAG_COMPRESSION_TRIGGER_RATIO`, default `0.80`.
- Code and logs are valid document-like sources, but Agent answers should not
  rely only on short RAG snippets when exact surrounding code/log context
  matters. The Agent prompt tells the model to use bounded original-source tools
  for that.

Raw source and archive access:

- New document/large-paste ingest also writes a raw source-of-truth record into
  the LangGraph Store/Mongo-backed state.
- `read_source_chunks(...)` reads ordered indexed chunks.
- `read_raw_source(source_key, ...)` reads bounded raw slices by source key,
  optional search phrase, and offset. This is the right tool after RAG points to
  a relevant code/log/document source but exact neighboring text is needed.
- `read_archive_record(...)` is now bounded too; it should return a window, not
  silently dump a full archive.
- Compression archives are indexed by AlphaRavis pgvector by default. Optional
  `rag_api` mirroring remains secondary/default-off.

Classifier behavior:

- Short direct retrieval queries stay direct.
- Long/noisy retrieval prompts are locally condensed and capped before
  embedding.
- Ambiguous long prompts can call Qwen3.5 2B for strict JSON. The code falls
  back to local condensation if the endpoint is down, times out, or returns bad
  JSON.
- For mixed large-paste prompts, classifier line ranges can remove instruction
  and question lines from the indexed document body while preserving the active
  current task in the marker.
- Archive `auto_on_intent` uses the same Qwen3.5 2B classifier for strict JSON:
  `archive_recall`, `search_query`, `confidence`, and `reason`. If the
  classifier is down, times out, low-confidence, or returns invalid JSON,
  AlphaRavis falls back to the local archive-recall condenser/heuristic.
- Agent-path archive auto-intent is default-on through
  `ALPHARAVIS_ARCHIVE_AUTO_ON_INTENT_AGENT_DEFAULT=true`. Fast Path bypasses it;
  `archive_rag_mode=manual` is the strict per-thread opt-out.
- The JSON parser has been hardened for truncated outputs, and the Bridge Test
  UI/Observer now has a Small-Qwen classifier probe with short direct, long
  noisy, instruction-only, document-only, mixed, and simulated
  down/invalid/timeout fallback cases.

Idle embedding scheduler:

- Queued embedding jobs should drain only after real graph inactivity.
- `run_profile_start_node` refreshes an in-process activity timestamp.
- `_embedding_scheduler_loop` waits for
  `ALPHARAVIS_EMBEDDING_SCHEDULER_IDLE_AFTER_SECONDS=600` before draining the
  durable embedding queue.
- The old local `.env` key
  `ALPHARAVIS_EMBEDDING_SCHEDULER_LAST_ACTIVITY_AGE_SECONDS=999999` was replaced
  with `ALPHARAVIS_EMBEDDING_SCHEDULER_IDLE_AFTER_SECONDS=600`.

Docs and env hygiene:

- The user explicitly asked that any new default added to `.env(exaple)` should
  also be added to local `.env`.
- Keep updating `docs/ALPHARAVIS_OPEN_TASKS.md` when planned items move state,
  and `docs/ALPHARAVIS_CHANGES.md` for runtime behavior changes.

Focused verification already run for this slice:

```text
PYTHONPYCACHEPREFIX=/tmp/alpharavis-pycache python -m py_compile \
  langgraph-app/agent_graph.py langgraph-app/context_compressor.py
pytest -q tests/test_agent_context_budget.py \
  tests/test_context_compressor.py tests/test_retrieval_router.py
git diff --check
docker compose up -d --no-deps --force-recreate langgraph-api
docker compose ps langgraph-api

PYTHONPYCACHEPREFIX=/tmp/alpharavis-pycache python -m py_compile \
  langgraph-app/agent_graph.py
pytest -q tests/test_agent_context_budget.py
pytest -q tests/test_bridge_test_ui.py tests/test_bridge_responses.py \
  tests/test_media_server.py tests/test_tailscale_https_routes.py \
  tests/test_service_redirector_server.py
```

Result: compile passed, the focused pytest set passed with `76 passed`,
`git diff --check` passed, and `langgraph-api` restarted healthy with the new
runtime env values visible inside the container.
Latest focused check for this commit: `tests/test_agent_context_budget.py`,
`tests/test_bridge_test_ui.py`, `tests/test_bridge_responses.py`,
`tests/test_media_server.py`, `tests/test_tailscale_https_routes.py`, and
`tests/test_service_redirector_server.py` passed together with `133 passed`;
`git diff --check` is clean.

Next work agreed with the user:

1. Run an 8k-vs-16k Qwen3.5 2B classifier context comparison. Measure latency,
   JSON validity, archive-recall query quality, and mixed large-paste
   line-range quality before assuming 16k as the practical default.
2. Browser/live-test the new deterministic source metadata and long-prompt route
   classifier behavior with real LibreChat examples before changing any user UI
   defaults.
3. Live-test archive auto-intent false positives with real tasks, especially:
   current video/image/file/Pixelle tasks, "wie war das nochmal" recall, and
   noisy long questions that mention old work but include an active source.
4. Live-test Tailscale HTTPS routes from another Tailnet device after Tailnet
   certificates are enabled.
5. Do not add a separate RAG sufficiency model call for now. The user decided
   the big Agent model can decide whether chunks are enough, as long as the
   raw-source/archive tools and prompt hint are always available in Agent mode.

## Current Snapshot

The current direction is AlphaRavis-native RAG first. `rag_api` is no longer the
default document/large-paste backend; it is kept as an adapter/reference path
for comparison and compatibility.

Latest local follow-up in this working tree:

- large pasted user content is not automatically indexed on every long paste
  anymore. Auto-ingest is deferred until after pre-run compression by default;
  then `large_paste_post_compression_node` indexes/replaces the paste only if
  the active request still exceeds
  `ALPHARAVIS_LARGE_PASTE_RAG_POST_COMPRESSION_TRIGGER_RATIO` of the active
  compression budget, default `0.80`. If the source marker replacement still
  leaves too much non-document chatter active, the same node can run a follow-up
  compression pass at
  `ALPHARAVIS_LARGE_PASTE_POST_RAG_COMPRESSION_TRIGGER_RATIO`, default `0.80`.
- paired `/rag ... /rag`, `/rake ... /rake`, `/index ... /index`, or
  `/ingest ... /ingest` blocks still force immediate source indexing.
- active RAG prefetch now prepares a bounded retrieval query before embedding:
  short questions stay direct, long/noisy turns are locally condensed and capped,
  and ambiguous long turns can call the always-on small classifier on the
  Big-Boss host port `8001`.
- large-paste intent is classified locally as `document`, `instruction`,
  `mixed`, or `unknown` before ingest. Instruction-like pastes become
  `large_instruction` sources and the active message keeps a condensed
  instruction brief instead of treating the whole prompt as a document.
- for long large-paste/mixed prompts, the same small classifier can refine the
  local intent decision. Its instruction/question line ranges are removed from
  the indexed document body, while retrieval query and line ranges are recorded
  in metadata/run_profile and the mixed marker preserves the current task lines.
- document-file, LibreChat document-upload, large-paste, and compression-archive
  ingest now add deterministic source metadata: `content_type`,
  `source_title`, `source_keywords`, `source_entities`, and `source_symbols`.
  pgvector chunking honors `content_type=log|code|config|prose`; config uses
  the code-style chunk profile.
- long-prompt direct-vs-agent routing can consult the existing Qwen3.5 2B
  classifier only after the normal FastPath decision rejects a prompt for
  length. High-confidence `direct_query` / `noisy_query` with no
  document/instruction line ranges may use direct answer mode. Short FastPath,
  tool-keyword denials, document/mixed/instruction prompts, low confidence, and
  very large prompts keep the existing agent path.
- archive-recall query condensation is implemented for vague follow-ups such as
  "wie war das nochmal mit X". Planner hints now include a stronger suggested
  archive/RAG search query from recent thread context, and the context retrieval
  agent has `condense_archive_recall_query`.
- Agent-path archive auto-intent now asks the safe Qwen3.5 2B classifier by
  default when current-thread archive keys exist. It decides whether the latest
  request is archive recall and which bounded `search_query` to use. The local
  archive-recall condenser remains the fallback for endpoint, timeout,
  low-confidence, or JSON failures. Fast Path bypasses this node, and
  `archive_rag_mode=manual` is the strict opt-out.
- section-level mixed archive splitting is implemented without an LLM call:
  archive/archive-collection text is segmented into ordered prose/log/code/config
  sections and chunked with the matching profile while preserving order.
- large-paste ingest now records a run-profile event timeline for Observer and
  later UI progress plumbing: `large_ingest.started`,
  `large_ingest.completed`, `large_ingest.failed`, or
  `large_ingest.skipped`. Direct pgvector writes can also append
  `large_ingest.chunk_indexed` / `document_ingest.chunk_indexed`.
- large-paste replacement now also records a compact `source_manifest` in
  `large_paste_ingests`, includes a short `Source manifest` line in the active
  marker, and the Bridge Observer renders a `Big Message / Source Ingest`
  section from LangGraph metadata. The manifest includes chunk/index stats and
  source digest when available. Explicit ingest syntax also supports paired
  `/big-context ... /big-context` and fenced
  `<big-context name="...">...</big-context>` blocks. If no explicit question is
  detected, the marker asks what to extract/analyze instead of doing broad
  unsupported analysis. The Bridge only surfaces the graph decision; LangGraph
  still owns the large-message/RAG/compression policy.
- focused compression instructions are implemented as a bounded chat-tag path:
  `<focus_topic>...</focus_topic>`, `<compact_instructions>...</compact_instructions>`,
  `/compact ...`, `@compact ...`, or `@focus ...` feed the one-shot/chunked
  summary prompt and archive metadata as compaction-selection hints. They do
  not become new agent tasks. The Observer `Shrinking` cards show `Compact
  Focus`; `/compact clear` or `compression focus off` clears stored focus. The
  Chunking Lab now has a `Compact Instructions` field for the same path.
- compression now records structured progress events in archive metadata and
  run-profile debug fields: `compression.started`, `compression.precompact`,
  `compression.workflow_events.compacted`, `compression.chunk.started/completed`,
  `compression.synthesis.started/completed`, `compression.synthesis.failed`,
  `compression.skipped`, `compression.completed`, and
  `compression.postcompact`. PreCompact carries reason/scope/token pressure,
  selected H/M/T, and chunking decision; workflow compaction carries compacted
  tool/action event counts; PostCompact carries the archive key and final
  before/after result metadata. Bridge streaming summarizes the latest event as
  compact `context_compaction` status/reasoning activity; the full event list
  stays available in Observer JSON.
- tool/workflow telemetry is compacted separately from normal chat messages.
  Tool-call requests, tool outputs, duplicate outputs, and long action logs go
  into `Workflow / Tool Event Compact Log` in archive metadata/content. Redacted
  original messages remain in the raw archive for exact reads. Observer
  `Shrinking` cards render the compact event counts and a bounded
  `Workflow / Tool Events` preview for the selected compression scope.
- Memory tiers are now documented as an AlphaRavis policy: latest task tail,
  active compaction summary, raw archive, archive collection, source/RAG record,
  vector recall, MemoryKernel facts, temporary workflow state, and Observer
  telemetry stay separate. Exact old text should come from bounded raw archive
  or raw-source reads after vector/RAG has found the relevant source.
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
- LibreChat document uploads have a first automatic path: Bridge document parts
  are registered with media-gallery, mapped into `pending_document_ingests`, and
  loaded by `run_profile_start_node` before normal RAG prefetch.
- source digest dedup is active by default for identical scoped source keys, and
  large-paste source keys are content-based within the thread.
- optional LLM structured-output grading exists behind
  `ALPHARAVIS_AGENTIC_RAG_LLM_GRADING=false`; deterministic grading remains the
  default/fallback. The configured example grader is `openai/big-boss`, and the
  grader call disables hidden thinking for speed.
- current product decision: do not require Big-Boss LLM grading while the Qwen3
  reranker is active. Keep LLM grading default-off as a debug/comparison path;
  the normal RAG policy is Qwen3 reranker plus deterministic grading/fallback.
- router reranking now supports a real llama.cpp/Qwen3-Reranker model backend
  through `ALPHARAVIS_RAG_RERANKER_MODE=model`. It falls back to deterministic
  lexical/vector reranking when the endpoint is down, times out, or returns a
  bad payload.
- `write_alpha_ravis_artifact(...)` now indexes artifacts through
  `retrieval_router.ingest_source(...)`, same as compression archives,
  document uploads, and large-paste ingest. Future manual ingest commands
  should use the same router entrypoint instead of calling pgvector or
  `rag_api` directly.
- The Test UI/Observer has a Small Qwen classifier probe. Local/fallback mode
  covers short direct, long noisy, instruction-only, document-only, mixed, and
  simulated endpoint-down/invalid-JSON/timeout fallback cases; real-Qwen mode
  calls the configured small classifier endpoint for the semantic cases.
- The Observer now polls `/api/embedding-queue/status` and renders pending,
  running, failed, done, progress, and recent active jobs for the shared
  pgvector embedding queue.
- archive and archive-collection chunk profiling now scans content before
  choosing a profile: code fences/common source syntax use the code profile,
  log/traceback-heavy archives use the log profile, and normal conversation
  archives stay chat.
- when a huge newest paste cannot stay protected in the recent tail, compression
  may move it into the compressible middle. In that oversized-tail rescue path,
  chunked summary compression is forced if the summary prompt would otherwise
  be pruned, even while the global chunking flag stays default-off.
- the exact raw compressed middle still goes into the AlphaRavis compression
  archive, so the active summary is bounded but exact text remains retrievable
  through archive/RAG tooling.

Most recent commits:

```text
d0ffeab Add LangChain-backed RAG document ingest
4fae3a2 Harden large context RAG and media handling
a57017c Add native document RAG smoke
d7d5c85 Default document RAG to AlphaRavis pgvector
86c15f6 Isolate LiteLLM proxy database
77dc70c Scope LiteLLM embedding params and default to qwen 0.6b
```

Current important settings:

```text
ALPHARAVIS_DOCUMENT_RAG_BACKEND=alpharavis_pgvector
EMBEDDING_LITELLM_MODEL=ollama/qwen3-embedding:0.6b
RAG_COLLECTION_NAME=alpharavis_qwen06
ALPHARAVIS_ENABLE_LARGE_PASTE_RAG_INGEST=true
ALPHARAVIS_ENABLE_LARGE_PASTE_INTENT_CLASSIFIER=true
ALPHARAVIS_LARGE_PASTE_RAG_MIN_CHARS=20000
ALPHARAVIS_DEFER_LARGE_PASTE_RAG_UNTIL_AFTER_COMPRESSION=true
ALPHARAVIS_LARGE_PASTE_RAG_AUTO_STAGE=post_compression
ALPHARAVIS_LARGE_PASTE_RAG_POST_COMPRESSION_TRIGGER_RATIO=0.80
ALPHARAVIS_LARGE_PASTE_POST_RAG_COMPRESSION_ENABLED=true
ALPHARAVIS_LARGE_PASTE_POST_RAG_COMPRESSION_TRIGGER_RATIO=0.80
ALPHARAVIS_COMPRESSION_REBALANCE_OVERSIZED_TAIL=true
ALPHARAVIS_COMPRESSION_OVERSIZED_TAIL_RATIO=0.60
ALPHARAVIS_COMPRESSION_OVERSIZED_TAIL_FORCE_MIDDLE_RATIO=0.80
ALPHARAVIS_COMPRESSION_ENABLE_CHUNKED_SUMMARY=false
ALPHARAVIS_PGVECTOR_SPLITTER=auto
ALPHARAVIS_PGVECTOR_SECTION_LEVEL_ARCHIVE_SPLITTING=true
ALPHARAVIS_DOCUMENT_INGEST_ROOT=
ALPHARAVIS_ARCHIVE_AUTO_ON_INTENT_MAX_ARCHIVES=5
ALPHARAVIS_ENABLE_ARCHIVE_AUTO_INTENT_CLASSIFIER=true
ALPHARAVIS_ARCHIVE_AUTO_INTENT_MIN_CONFIDENCE=0.6
ALPHARAVIS_ARCHIVE_AUTO_INTENT_CLASSIFIER_MAX_TOKENS=384
ALPHARAVIS_ENABLE_RAG_RERANKING=true
ALPHARAVIS_RAG_RERANKER_MODE=model
ALPHARAVIS_RAG_RERANKER_URL=http://192.168.178.140:8000
ALPHARAVIS_RAG_RERANKER_ENDPOINT=/reranking
ALPHARAVIS_RAG_RERANKER_MODEL=qwen3-reranker-0.6b
ALPHARAVIS_RAG_RERANKER_FALLBACK_DETERMINISTIC=true
ALPHARAVIS_RETRIEVAL_QUERY_MAX_CHARS=1500
ALPHARAVIS_RETRIEVAL_DIRECT_QUERY_MAX_CHARS=1500
ALPHARAVIS_ENABLE_RETRIEVAL_QUERY_CLASSIFIER=true
ALPHARAVIS_RETRIEVAL_QUERY_CLASSIFIER_MIN_CHARS=6000
ALPHARAVIS_RAG_CLASSIFIER_API_BASE=
ALPHARAVIS_RAG_CLASSIFIER_MODEL=unsloth/Qwen3.5-2B-GGUF:Q4_1
ALPHARAVIS_RAG_CLASSIFIER_MAX_TOKENS=512
ALPHARAVIS_ENABLE_LARGE_PASTE_SMALL_CLASSIFIER=true
ALPHARAVIS_LARGE_PASTE_SMALL_CLASSIFIER_MIN_CHARS=6000
ALPHARAVIS_AGENTIC_RAG_LLM_GRADING=true
ALPHARAVIS_AGENTIC_RAG_GRADER_MODEL=openai/big-boss
ALPHARAVIS_AGENTIC_RAG_GRADER_TIMEOUT_SECONDS=25
ALPHARAVIS_PGVECTOR_DEDUP_SOURCES=true
BRIDGE_DOCUMENT_RAG_AUTO_INGEST=true
ALPHARAVIS_EMBEDDING_JOB_STALE_AFTER_SECONDS=900
```

Live verification after the latest slice:

- Bridge Test UI serves `Native Document RAG Smoke`.
- `POST /api/native-document-rag-smoke` passed live with
  `acceptance_ok=true`, `hit_count=2`, `pgvector_backend_selected=true`, and
  `rag_api_not_used=true`.
- Rebuilt `langgraph-api`, `api-bridge`, and `media-gallery`; `/v1/models` and
  media-gallery `/health` returned 200.
- A Bridge-compatible `input_file` upload smoke registered the uploaded
  Markdown document in media-gallery and sent `pending_document_ingests` to
  LangGraph. With the local default `ALPHARAVIS_PGVECTOR_INDEX_MODE=queue`, the
  uploaded document appeared in `alpharavis_embedding_jobs`; after queue drain it
  produced two pgvector rows (Catalog+Chunk) with `source_type=uploaded_document`.
- A follow-up `source_type=large_paste` native smoke passed live in about
  `3.0 s` with `acceptance_ok=true`, `rag_api_not_used=true`,
  `active_source_key_recorded=true`, and two bounded pgvector hits.
- Focused tests passed: `42 passed` across Bridge Test UI, retrieval router,
  context budget, and `rag_api_client` tests.
- Current focused local RAG/Memory/Loader/Bridge tests passed: `118 passed`
  across document ingest, context budget, retrieval router, media analysis,
  source-scoped retrieval, and Bridge Responses tests.
- `langgraph-api` was restarted and verified healthy with the active runtime
  settings for model reranking and Big-Boss LLM grading.
- The llama.cpp reranker was restarted with a larger physical batch after the
  earlier `-ub 64` failure. `/reranking` now works on the GPU server at
  `http://192.168.178.140:8000`: a 3-document / 277-token direct probe took
  about `0.51 s`, a 10-document / 1028-token direct probe took about `2.03 s`,
  and an in-container AlphaRavis router probe reported
  `strategy=llamacpp_qwen3_reranker`, `fallback_used=false`, and `0.426 s` for
  three candidates.
- Follow-up embedding probe after enabling GPU acceleration on the Ollama host:
  `qwen3-embedding:4b` loads and embeds through Ollama `/api/embed`, returns
  2560-dimensional vectors, and is resident with about `2.38 GB` reported
  `size_vram`. Warm probes completed 2048 chars in about `12.8 s` and 8192
  chars in about `22.3 s`. `qwen3-embedding:0.6b` remained faster in the same
  window, about `7.2 s` and `10.9 s`, and returns 1024-dimensional vectors.
- Bridge Test UI now includes `RAG Load Probe` at `/api/rag-load-probe`. It
  runs embedding and reranker calls concurrently for configured rough-token
  steps, optionally sends real Bridge `/v1/responses` queries, and surfaces
  per-step embedding/reranker/LLM timings in the Observer UI.
- First combined GPU load probe with `qwen3-embedding:4b` plus GPU Qwen3
  reranker passed all configured rough-token steps `400,1000,4000,10000,20000,
  40000`. The 4B embedding route returned 2560-dimensional vectors throughout;
  elapsed embedding time rose from about `5.2 s` at 400 rough tokens to about
  `45.7 s` for the 40k rough-token request. The server reported
  `prompt_eval_count=4095` from the 10k step upward, so treat larger requests as
  accepted but effectively capped/truncated around the model context unless the
  embedding server context is raised.
- A smaller combined probe with real Bridge `/v1/responses` calls passed at
  400 and 1000 rough-token steps. The Bridge LLM responses took about `29.6 s`
  and `25.6 s`, while concurrent embedding and reranker calls also succeeded.
- The Qwen3.5 2B classifier server on the Big-Boss host port `8001` is reachable
  and returns JSON through `/v1/chat/completions`. A simple live prompt produced
  a bounded retrieval query in about `1.34 s`. AlphaRavis derives the classifier
  base URL from `BIG_BOSS_API_BASE` by replacing the port with `8001` unless
  `ALPHARAVIS_RAG_CLASSIFIER_API_BASE` is explicitly set.
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
- `read_raw_source(...)` for bounded raw Store reads of newly ingested
  document/large-paste source records. Use this after chunk/RAG lookup when the
  model needs exact surrounding source text without loading the whole document.

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
- archive-only Agent-path state runs the bounded Qwen3.5 2B archive-intent
  check by default; current upload/file/image/video/URL/Pixelle/source tasks
  must not receive archive context unless old/archive context is explicit
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

- `qwen3-embedding:4b`: works, 2560-dim, and is usable after GPU acceleration,
  but remains roughly 2x slower than 0.6B on the current 8192-char warm probe.
  Use a separate 2560-dim collection if routing pgvector/RAG to it.
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
  - archive and archive-collection sources scan their content before choosing a
    profile: code fences/common code syntax use `code`, log/traceback lines use
    `log`, otherwise they stay `chat`
  - chat/archive/code/log profiles keep the AlphaRavis splitter unless the
    operator sets `ALPHARAVIS_PGVECTOR_SPLITTER=langchain`
  - set `ALPHARAVIS_PGVECTOR_SPLITTER=alpharavis` to force the local fallback
    everywhere
- standard: 900 tokens / 125 overlap
- chat/archive: 700 / 100
- logs: 1200 / 75
- code: 600 / 80

Code/log detection is heuristic for now. AST/Tree-sitter splitting and
section-level mixed archive splitting are still follow-ups. The mixed archive
splitter should segment prose, logs, and fenced/source-code blocks first, then
apply the matching per-section chunk profile without losing original order or
archive/source metadata.

Current reranking direction:

- `ALPHARAVIS_ENABLE_RAG_RERANKING=true` enables router reranking.
- `ALPHARAVIS_RAG_RERANKER_MODE=model` calls the local llama.cpp reranker at
  `http://192.168.178.140:8000/reranking`.
- The expected model/server is Qwen3-Reranker-0.6B started with llama.cpp
  `--embedding --pooling rank --reranking`.
- If the model endpoint is down, times out, or returns a bad payload,
  `ALPHARAVIS_RAG_RERANKER_FALLBACK_DETERMINISTIC=true` falls back to
  deterministic lexical/vector reranking and records the fallback in warnings.

## Next Best Steps

1. Compare GPU vs CPU serving for the live llama.cpp Qwen3 reranker.

   GPU serving is now functional after raising the physical batch from `-ub 64`
   to a larger value. Next, test a CPU run with `-ngl 0 -c 2048 -b 512 -ub 512`
   while the 4B embedding model stays on GPU, then compare latency and VRAM
   pressure. Early operator feedback says CPU is about 4x slower, so the likely
   default is GPU reranking plus fallback enabled unless VRAM contention appears.

2. Run a true Bridge/LibreChat large-paste E2E against native AlphaRavis
   pgvector.

   Acceptance: first turn with a large pasted source gets replaced by the
   compact retrieval marker, the next user question triggers
   `<active-rag-context>` from `active_source_keys`, the returned chunks come
   from `alpharavis_pgvector`, and `rag_api` is not called unless explicitly
   configured.

3. Browser-test real LibreChat document/PDF/DOCX uploads.

   The Bridge-compatible `input_file` handoff is implemented and live-smoked,
   but a real LibreChat browser upload still needs verification. Confirm that
   LibreChat emits one of the covered `file` / `input_file` attachment shapes,
   media-gallery stores the file, `pending_document_ingests` reaches LangGraph,
   and the queued pgvector job becomes searchable after the queue drains.

4. Live-check queue progress polling for asynchronous queue-drained jobs.

   Direct pgvector writes already emit chunk progress into run-profile activity.
   First UI slice implemented: the Observer polls the shared embedding queue and
   shows pending/running/failed/done counts plus recent active jobs. Remaining
   work is a live large-upload/paste check and optional richer per-source
   progress once queue jobs expose total chunk counts after claiming.

5. Live-check section-level mixed archive splitting.

   Implemented locally: archive/archive-collection chunking can segment ordered
   prose/log/code/config sections and apply the matching profile per section.
   Remaining work is live quality comparison on real mixed archives and tuning
   the heuristics if section boundaries are too aggressive.

6. Keep the Qwen3 reranker as the normal ranking policy.

   The reranker is the intended active model-side quality step. Big-Boss
   LLM-grading is not needed in the normal path while reranking is active; keep
   it default-off and use it only for explicit comparison/debug probes.

7. Live-check Agent-path archive auto-on-intent behavior.

   Implemented locally as Agent-path default behavior via
   `ALPHARAVIS_ARCHIVE_AUTO_ON_INTENT_AGENT_DEFAULT=true`. Compression archives
   still do not activate document RAG, but archive-only Agent runs ask the safe
   Qwen3.5 2B classifier for strict JSON (`archive_recall`, `search_query`,
   `confidence`, `reason`) and fall back to local archive-recall condensation on
   endpoint/timeout/JSON failures. Remaining work is measuring latency, answer
   quality, and false positives on real LibreChat recall examples.

8. Compare Qwen3.5 2B classifier context at 8k vs 16k.

   Current local note: the operator set the classifier context to 8k. 16k is
   likely useful for archive recall and mixed-paste line ranges if latency and
   JSON validity remain acceptable. Do not just raise AlphaRavis limits; the
   llama.cpp/Qwen server itself must be served with the larger context. Keep
   classifier response budgets small and compare the same prompts under both
   contexts.

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
