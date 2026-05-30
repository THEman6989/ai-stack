# PGVector Search Head + Mongo Raw Store Implementation Plan

> **For Hermes:** Use `subagent-driven-development` only if this plan is executed task-by-task. This file is a plan only; do not implement code from it until explicitly requested.

**Goal:** Make AlphaRavis consistently use MongoDB/LangGraph Store as raw long-term storage and Postgres+PGVector as the primary search/index layer for RAG, memories, skills, sessions, artifacts, tool runs, and media metadata — with PGVector always storing retrievable `chunk_text`, not just embeddings and IDs.

**Architecture:** MongoDB remains the source of truth for raw/full records. PGVector becomes the fast search head: every indexed row stores `embedding + chunk_text + source_id/source_key + metadata + created_at + version`, so normal semantic recall returns usable snippets directly. MongoDB/raw-source reads are only used when the agent needs the full original document, neighboring context, complete chat history, original media/tool payloads, or update/delete authority.

**Tech Stack:** `langgraph-app/vector_memory.py`, `langgraph-app/retrieval_router.py`, `langgraph-app/agent_graph.py`, `langgraph-app/repo_skills.py`, LangGraph Store/MongoDB, Postgres pgvector, existing embedding route `memory-embed`, existing reranker route, pytest.

---

## Current Evidence From Repo Inspection

Already partially true:

- `langgraph-app/vector_memory.py` creates `alpharavis_memory_vectors` with `content`, `chunk_text`, `catalog_text`, `preview_text`, `source_key`, `metadata`, `created_at`, and `updated_at`.
- `_insert_chunk_sync()` inserts the full chunk into both `content` and `chunk_text`.
- `semantic_search()` / `_search_sync()` return `chunk_text`, `metadata`, `created_at`, `updated_at`, similarity, and distance directly from PGVector.
- `retrieval_router.vector_result_to_tool_hit()` prefers `preview_text`, then `chunk_text`, then `content`, so normal RAG results can answer from PGVector without a Mongo lookup.
- `record_curated_memory()` creates raw records in the LangGraph Store and indexes created memories into PGVector with `source_type="curated_memory"`.
- `record_skill_candidate()` stores raw skill candidates in `SKILL_LIBRARY_NS` and indexes them into PGVector with `source_type="skill"`.
- `delete_memory_record()` exists in `vector_memory.py` and is called by `record_curated_memory(action="delete")`.

Gaps to close:

- PGVector table has no top-level `version` column and no explicit `source_id` alias. It currently relies on `source_key` and metadata.
- Curated memory `action="update"` updates Mongo/LangGraph Store but does not re-index the PGVector row in the inspected code path.
- Skill `activate_skill_candidate()` / `deactivate_skill()` update Mongo/LangGraph Store status but do not refresh/delete the existing PGVector skill index row, so semantic skill search can show stale status metadata.
- Repo AI skills under `ai-skills/` are scanned/cached by `repo_skills.py`; they are not clearly backfilled into PGVector as first-class `source_type="repo_skill"` index rows.
- There is no single documented storage contract saying: "Mongo raw source, PGVector search head, PGVector rows always contain chunk text."
- Tool runs/media metadata/raw events are conceptually Mongo-side today, but the exact PGVector index contract should be explicit and test-covered before widening indexing.

## Non-Negotiable Rules

1. Default behavior must remain safe and backward-compatible.
2. New expensive or broad indexing paths must be feature-flagged default OFF unless they are pure schema compatibility changes.
3. Existing PGVector rows must continue to work; migrations use `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` only.
4. Never store only `embedding + mongo_id`; every PGVector row must contain a useful `chunk_text` or typed textual surrogate.
5. MongoDB/LangGraph Store remains the write authority for raw/full objects.
6. PGVector is allowed to duplicate text chunks because it is the search head.
7. Updates/deletes must keep Mongo and PGVector in sync where possible; failures should return warnings, not silently lie.
8. No `.env` behavior change without docs and `.env(exaple)` entry if a new flag/default is introduced.

## Target Storage Contract

### MongoDB / LangGraph Store: Raw Source Of Truth

Used for:

- Full original documents and large pastes.
- Full chat/session turns and archive records.
- Curated memory raw records.
- Skill library records and candidates.
- Artifacts and generated outputs.
- Tool-run/event payloads.
- Media metadata, manifests, frame lists, original URLs, file records.

Each raw record should expose:

```text
source_id        stable canonical ID; usually same as current source_key
source_type      document | archive | curated_memory | skill | repo_skill | session_turn | artifact | tool_run | media_metadata | ...
title            short human-readable name
text             full text or textual surrogate when raw payload is non-text
metadata         original metadata and routing hints
created_at       raw record creation time
updated_at       raw record update time
version          content/index version, e.g. source_digest or vN
```

### PGVector: Search Head / Chunk Index

Every row should contain:

```text
id               stable row ID, including source_id + chunk_index + version/scope where needed
namespace        usually alpharavis
scope            thread | global | user | skill_library | artifact | media | ...
thread_id        optional thread binding
thread_key       optional readable thread key
source_type      typed surface
source_id        canonical stable source ID (new alias, same value as source_key initially)
source_key       existing backwards-compatible ID
version          top-level source/index version
chunk_text       full retrievable chunk text or textual surrogate
content          backwards-compatible copy of chunk_text
catalog_text     optional source-level table-of-contents row
preview_text     bounded preview
chunk_index      -1 for catalog, 0..N for chunks
chunk_count      total chunks for source
is_catalog       bool
embedding_model  model that produced vector
metadata         JSONB; includes raw_ref, source_digest, chunk_digest, content_type, status, etc.
embedding        vector(...)
created_at       row creation time
updated_at       row update time
```

`metadata.raw_ref` should point back to the raw store when available:

```json
{
  "raw_ref": {
    "store": "langgraph_store",
    "namespace": ["alpharavis", "curated_memory"],
    "key": "abc123"
  }
}
```

For external/raw Mongo collections, use:

```json
{
  "raw_ref": {
    "store": "mongodb",
    "db": "alpharavis_state",
    "collection": "tool_runs",
    "id": "..."
  }
}
```

## Implementation Tasks

### Task 1: Add a focused storage contract doc section

**Objective:** Document the architecture before changing behavior.

**Files:**
- Modify: `docs/ALPHARAVIS_ARCHITECTURE.md`
- Modify: `docs/ALPHARAVIS_RAG_HANDOFF.md`
- Modify: `docs/ALPHARAVIS_CHANGES.md`
- Modify: `docs/ALPHARAVIS_OPEN_TASKS.md`

**Steps:**
1. Add a section named `Mongo Raw Store + PGVector Search Head` to architecture docs.
2. State that PGVector rows must store `chunk_text` and metadata, not just IDs.
3. State when Mongo/raw tools are needed: full document, surrounding chunks, complete chat, original tool/media payloads.
4. Add an open-task entry tracking implementation state until all tests pass.
5. Add a changes entry only after code changes are done.

**Verification:**
- `python -m pytest tests/test_retrieval_router.py -q` still passes after docs-only change if running a quick sanity check is desired.

### Task 2: Add PGVector schema compatibility columns

**Objective:** Make the target contract explicit in the table schema without breaking existing rows.

**Files:**
- Modify: `langgraph-app/vector_memory.py`
- Test: `tests/test_source_scoped_retrieval.py` or new `tests/test_vector_memory_contract.py`

**Implementation detail:**
In `_ensure_schema_sync(dimensions)`, add columns with `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`:

```python
("source_id", "TEXT NOT NULL DEFAULT ''"),
("version", "TEXT NOT NULL DEFAULT 'v1'"),
("raw_ref", "JSONB NOT NULL DEFAULT '{}'::jsonb"),
```

Then backfill:

```sql
UPDATE {table} SET source_id = source_key WHERE source_id = '';
UPDATE {table} SET version = COALESCE(metadata->>'version', metadata->>'source_digest', 'v1') WHERE version = 'v1';
```

**Verification:**
- Unit-test that `_ensure_schema_sync()` emits/adds the compatibility columns via a fake psycopg cursor, or test the generated migration path if existing tests already stub psycopg.
- Existing search rows without `source_id` still return normally.

### Task 3: Carry `source_id`, `version`, and `raw_ref` through inserts

**Objective:** Ensure every new row writes the canonical fields.

**Files:**
- Modify: `langgraph-app/vector_memory.py`
- Test: `tests/test_vector_memory_contract.py`

**Steps:**
1. Extend `_insert_chunk_sync()` args with `source_id`, `version`, `raw_ref`.
2. Insert those columns in the SQL insert/update statement.
3. In `upsert_memory_record()`, derive:
   - `source_id = metadata.get("source_id") or source_key`
   - `version = metadata.get("version") or metadata.get("source_digest") or source_digest[:16]`
   - `raw_ref = metadata.get("raw_ref") or {}`
4. Ensure catalog row and normal chunk rows both get the same `source_id`, `version`, and `raw_ref`.

**Verification:**
- Test that inserted params include `chunk_text == chunk`, `content == chunk`, `source_id == source_key`, and a non-empty `version`.

### Task 4: Return the new contract fields from semantic search

**Objective:** PGVector search results expose everything needed for 80–90% of RAG answers.

**Files:**
- Modify: `langgraph-app/vector_memory.py`
- Modify: `langgraph-app/retrieval_router.py`
- Test: `tests/test_retrieval_router.py`

**Steps:**
1. Add `source_id`, `version`, and `raw_ref` to `_search_sync()` SELECT.
2. Add those fields to returned records.
3. Update `vector_result_to_tool_hit()` so tool hits include:
   - `source_id`
   - `source_key`
   - `version`
   - `raw_ref`
   - `chunk_text`
   - `preview_text`
   - `metadata`
4. Keep `source_key` as the backward-compatible ID everywhere.

**Verification:**
- Add a retrieval-router test where a PGVector record with `source_id`, `version`, `raw_ref`, and `chunk_text` is converted to a tool hit without any Mongo fetch.

### Task 5: Fix curated-memory update re-indexing

**Objective:** Memory updates must update both Mongo/LangGraph Store and PGVector.

**Files:**
- Modify: `langgraph-app/agent_graph.py`
- Test: add/extend an agent_graph memory test, likely in existing memory-related test module if present; otherwise create focused source test.

**Current gap:** `record_curated_memory(action="update")` updates the raw store and returns before `_maybe_index_vector_memory()` is called.

**Steps:**
1. After `_maybe_put()` calls in update path, call `_maybe_index_vector_memory()` with:
   - `source_type="curated_memory"`
   - `source_key=memory_id`
   - title based on memory type
   - content containing memory + evidence
   - scope `memory_scope`
   - metadata including `created_at`, `updated_at`, `origin_thread_id`, and `raw_ref`
2. If indexing fails, return a warning just like create path.
3. Preserve existing successful update message shape as much as possible.

**Verification:**
- Test update calls `_maybe_index_vector_memory()` once with source_key equal to memory_id.
- Test update still succeeds when vector indexing returns a warning.

### Task 6: Keep curated-memory delete in sync and test it

**Objective:** Ensure delete removes PGVector rows and future regressions are caught.

**Files:**
- Modify tests only unless behavior is broken.
- Test: memory delete focused test.

**Steps:**
1. Add test that `record_curated_memory(action="delete")` calls `_pgvector_delete_memory_record(source_key=memory_id)`.
2. Assert warning includes `pgvector cleanup skipped/failed` only when deletion returns false.
3. Confirm `vector_memory.delete_memory_record()` deletes by `namespace + source_type + source_key`.

**Verification:**
- `pytest -q <new-memory-test>`.

### Task 7: Re-index skill activation/deactivation state

**Objective:** Skill semantic search must not show stale candidate/active metadata.

**Files:**
- Modify: `langgraph-app/agent_graph.py`
- Test: skill-library focused test.

**Current gap:** `activate_skill_candidate()` and `deactivate_skill()` update `SKILL_LIBRARY_NS` but do not refresh PGVector.

**Steps:**
1. Extract a helper near skill tools:
   ```python
   async def _index_skill_library_record(skill_id: str, value: dict[str, Any]) -> str:
       ...
   ```
2. Use it from `record_skill_candidate()`, `activate_skill_candidate()`, and `deactivate_skill()`.
3. Include `status`, `active`, `approved_at`, `deactivated_at`, `version`, and `raw_ref` in metadata.
4. Set `source_type="skill"` or split into `skill_candidate` / `skill_active` only if retrieval filters need that. Prefer keeping `source_type="skill"` and using `metadata.status` to avoid migration churn.

**Verification:**
- Test `activate_skill_candidate()` updates store and calls `_maybe_index_vector_memory()` with `metadata.status == "active"`.
- Test `deactivate_skill()` re-indexes with `metadata.status == "candidate"` or deletes if final design chooses delete.

### Task 8: Index reviewed repo skills as first-class PGVector sources

**Objective:** Skills under `ai-skills/` should be searchable semantically, not only listed by manifest keywords.

**Files:**
- Modify: `langgraph-app/repo_skills.py`
- Modify: `langgraph-app/agent_graph.py`
- Test: `tests/test_repo_skills.py` or new `tests/test_repo_skill_vector_index.py`

**Steps:**
1. Add a pure helper in `repo_skills.py` that builds index payloads from scanned skill entries:
   ```python
   def skill_entry_to_index_document(entry: dict[str, Any]) -> dict[str, Any]
   ```
2. Include skill name, description, trigger/frontmatter, path, supporting-file names, and maybe the first bounded part of `SKILL.md` body.
3. In `reload_repo_ai_skills()`, after scan, optionally enqueue/upsert PGVector records for reviewed skills.
4. Feature flag if broad indexing is expensive:
   - `ALPHARAVIS_ENABLE_REPO_SKILL_VECTOR_INDEX=false` initially if this adds new background work.
   - Or index only on explicit `reload_repo_ai_skills()` call without new default background work.
5. Use `source_type="repo_skill"`, `source_key=<slug>`, `scope="skill_library"`, `raw_ref` with file path.

**Verification:**
- Test payload contains `chunk_text` source text, `source_key == slug`, and `metadata.path`.
- Test disabled flag avoids vector calls.
- Test explicit reload indexes when enabled.

### Task 9: Normalize RAG/document ingest metadata

**Objective:** Uploaded documents, large pastes, archives, and artifacts all produce comparable PGVector rows.

**Files:**
- Modify: `langgraph-app/retrieval_router.py`
- Modify: `langgraph-app/agent_graph.py` only where ingestion calls are made
- Test: `tests/test_retrieval_router.py`, `tests/test_agent_context_budget.py`

**Steps:**
1. Ensure `retrieval_router.ingest_source(...)` or the call site always passes metadata fields:
   - `source_id`
   - `version` / `source_digest`
   - `content_type`
   - `raw_ref`
   - `created_at` if available
2. Keep `source_key` unchanged for compatibility.
3. For `read_source_chunks()` and source markers, mention that PGVector chunks already include text and raw reads are for more context.

**Verification:**
- Existing large-paste/document tests still pass.
- New test asserts metadata passed to `_maybe_index_vector_memory()` includes `source_digest` and `raw_ref` where available.

### Task 10: Add optional tool-run/event indexing with textual surrogates

**Objective:** Tool-runs and raw events become semantically searchable without dumping huge JSON into the model.

**Files:**
- Likely create or extend a module near run state/tool event storage.
- Candidate files after inspection: `langgraph-app/run_state_manager.py`, `langgraph-app/agent_graph.py`, possibly a new small helper `langgraph-app/event_indexing.py`.
- Tests: new focused test.

**Design:**
PGVector `chunk_text` for tool runs should be a compact textual surrogate, not arbitrary full JSON:

```text
Tool run: execute_local_command
Status: success
Command: pytest -q tests/test_retrieval_router.py
Summary: 12 passed
Important output: ...bounded excerpt...
```

Raw full payload stays in Mongo.

**Feature flag:**
- `ALPHARAVIS_ENABLE_TOOL_EVENT_VECTOR_INDEX=false` default OFF.

**Verification:**
- Test that output is bounded and secrets are redacted before indexing.
- Test flag off keeps existing behavior exactly unchanged.

### Task 11: Add optional media metadata indexing

**Objective:** Media search can find videos/images/files by generated captions, manifest metadata, URLs, and user goals.

**Files:**
- Modify: `langgraph-app/vector_memory.py` only if existing media table is enough.
- Modify media ingestion/registration site if needed.
- Tests: `tests/test_media_analysis.py` or new focused test.

**Current state:** `alpharavis_media_vectors` already stores caption-like text and media metadata for frames/media. Keep this separate from text memory table unless unification is explicitly beneficial.

**Steps:**
1. Confirm every media vector row has a meaningful `caption` or textual surrogate.
2. Add `version` and `raw_ref` compatibility columns to media vector schema too if missing.
3. Ensure media search returns caption/metadata directly.

**Verification:**
- Existing media vector tests pass.
- New schema test covers `version`/`raw_ref` if added.

### Task 12: Make retrieval policy explicit in tool outputs

**Objective:** Agents learn the intended lookup behavior from tool results.

**Files:**
- Modify: `langgraph-app/agent_graph.py`
- Modify: `langgraph-app/retrieval_router.py`

**Steps:**
1. In `semantic_memory_search()` retrieval policy, state:
   - PGVector hits contain usable chunks.
   - Do not fetch Mongo/raw source unless full/neighbor/original context is needed.
2. In `query_source()` / `query_sources()`, state the same source-key narrowing rule.
3. Keep German/English acceptable; tool policy can be English.

**Verification:**
- Source tests assert policy text contains `chunk_text` or `PGVector hits contain usable chunks`.

### Task 13: Add a storage contract smoke script or diagnostic endpoint

**Objective:** Operator can verify the live stack follows the contract.

**Files:**
- Optional create: `scripts/pgvector_contract_smoke.py`
- Optional Makefile target: `make pgvector-contract-smoke`
- Docs if target is added: `README.md`, `docs/MAKEFILE_README.md`, `docs/ALPHARAVIS_USAGE_NOTES.md`, `docs/ALPHARAVIS_CHANGES.md`

**Checks:**
1. Connect to PGVector.
2. Inspect `alpharavis_memory_vectors` columns.
3. Insert or locate a small test source.
4. Semantic search returns `chunk_text`, `source_id/source_key`, metadata, version.
5. Optional raw ref can resolve to Mongo/LangGraph Store if test fixture uses one.

**Feature caution:**
Only add a Makefile target if Amin wants operator-facing smoke. Otherwise keep validation as pytest-level for now.

### Task 14: Backfill existing rows safely

**Objective:** Existing PGVector rows get `source_id`, `version`, and raw-ref metadata where possible.

**Files:**
- Optional create: `scripts/backfill_pgvector_contract.py`
- Docs: `docs/ALPHARAVIS_USAGE_NOTES.md` if operator-run.

**Steps:**
1. Default dry-run.
2. Report counts by source_type.
3. Fill `source_id=source_key` where empty.
4. Fill `version` from `metadata.source_digest`, `metadata.version`, or `updated_at` hash.
5. Do not rewrite embeddings.
6. Do not delete old rows.

**Feature flag / safety:**
No automatic destructive cleanup. Any old duplicate cleanup should be a separate explicit approval step.

**Verification:**
- Dry run prints planned row counts.
- Apply mode updates only missing compatibility fields.

### Task 15: Tests and verification pass

**Narrow tests first:**

```bash
pytest -q tests/test_retrieval_router.py
pytest -q tests/test_source_scoped_retrieval.py
pytest -q tests/test_repo_skills.py
```

**New focused tests to add/run depending on implementation:**

```bash
pytest -q tests/test_vector_memory_contract.py
pytest -q tests/test_curated_memory_vector_sync.py
pytest -q tests/test_repo_skill_vector_index.py
```

**Broader regression after shared paths:**

```bash
pytest -q tests
```

**Runtime smoke if stack is running:**

```bash
python scripts/alpharavis_setup.py status
python scripts/alpharavis_setup.py bridge-smoke
```

## Acceptance Criteria

The implementation is done when all are true:

1. PGVector memory rows have retrievable `chunk_text` for all text-like indexed content.
2. PGVector rows expose `source_key` and `source_id` (or documented alias) plus `metadata`, `created_at`, `updated_at`, and `version`.
3. Curated memory create/update/delete keep Mongo/LangGraph Store and PGVector synchronized or return explicit warnings.
4. Skill candidates and active/deactivated state are reflected in PGVector metadata.
5. Reviewed repo skills are either explicitly indexed as `repo_skill` rows or the plan records why only candidate skills are indexed for now.
6. RAG/document/archive/artifact ingests pass source/version/raw-ref metadata into PGVector.
7. Retrieval tools return useful chunks directly from PGVector and only direct agents toward raw Mongo/source reads for full/neighbor/original context.
8. Tests cover schema compatibility, insert contract, retrieval output contract, memory update/delete sync, and skill indexing state.
9. Docs state the storage rule clearly: PGVector is the search head, MongoDB is the raw store.

## Risks / Tradeoffs

- Text duplication is intentional: PGVector duplicates chunks from Mongo/raw store to avoid 80–90% of extra Mongo lookups.
- Wider indexing increases storage. Mitigate with bounded chunks, metadata limits, optional flags for tool/media event indexing, and later cleanup tools.
- Stale dual-store data is the main risk. Mitigate with update/delete sync tests and explicit warning returns.
- Skill indexing can accidentally expose draft/candidate content in search. Mitigate with metadata status filters and human-review flags.
- Tool-run indexing can leak secrets if raw output is indexed. Mitigate with existing redaction helpers and bounded textual surrogates only.

## Suggested Execution Order

1. Schema compatibility columns and search return fields.
2. Curated memory update/delete sync tests.
3. Skill candidate activation/deactivation reindexing.
4. Repo skill vector indexing.
5. RAG/document metadata normalization.
6. Optional tool-run/media metadata indexing.
7. Docs + smoke/backfill scripts.

This keeps the core promise first: PGVector search returns useful text chunks directly, Mongo is only needed when full raw context is required.
