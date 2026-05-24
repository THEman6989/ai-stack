# AlphaRavis agent_graph.py — Refactor Analysis

Date: 2026-05-24
File under analysis: `langgraph-app/agent_graph.py` (13,960 lines, 574 KB)

## Executive Summary

agent_graph.py is the central orchestration file for the AlphaRavis LangGraph pipeline.
It contains ~200 functions spanning graph wiring, tool definitions, business logic,
content analysis, compression, context budgeting, and prompt construction.

This is not a rewrite plan. The goal is to extract code that does not need to live
directly inside agent_graph.py — keeping it as the orchestrator, moving pure helpers
and standalone concerns to existing or new modules.

---

## 0. Alignment With Existing Docs

This analysis was cross-checked against the project documentation:

### `docs/ALPHARAVIS_ARCHITECTURE.md`
- Line 18: "Modular, so the agent brain and the LibreChat bridge can evolve separately."
  → This is about service-level modularity (LangGraph brain vs bridge server), not internal file splitting of agent_graph.py.
  The graph orchestration file can be large as long as service boundaries stay clean.

### `docs/ALPHARAVIS_CHANGES.md`
- Line 58: "No refactor, no rewrite" (about the parallel executor hook in planner_node)
  → Good precedent: the parallel executor was added as a minimal hook without restructuring agent_graph.py.
  This analysis follows the same pattern — extract only what clearly belongs elsewhere.

### `docs/ALPHARAVIS_OPEN_TASKS.md`
- Line 1907-1918: "Move backend selection out of `agent_graph.py` and into `retrieval_router.py`"
  → This confirms an existing plan to extract RAG backend selection. Partially implemented:
  `query_sources_with_backends` already lives in `retrieval_router.py`. Still remaining in agent_graph.py:
  `_rag_federated_search()`, `_rag_query_sources()`, `_query_sources_impl()`, `_rag_file_id_for_archive()`,
  `_rag_file_ids_for_archives()`, `_load_thread_rag_pins()`, `_write_thread_rag_pins()`.
  → These are flagged in the doc as "should be moved behind the router in a follow-up."
  → **My analysis lists these in Section C (debatable) — aligned with the existing doc plan.**

- Line 2172-2181: "Later, split current responsibilities into clearer modules: `vector_memory.py`,
  `rag_api_client.py`, `retrieval_router.py`. Backend selection still mostly lives in `agent_graph.py`
  and should be moved behind the router in a follow-up."
  → These modules already exist. The remaining work is moving the backend selection logic
  from agent_graph.py into them. My Phase 2 does not touch these — they're Phase 3-4 material.

### Conclusion
My Phase 2 plan (pure helpers: `source_content.py`, `command_safety.py`, extending existing modules)
**does not conflict with any doc-planned work.** The doc-recognized extraction targets (RAG backend
selection, retrieval routing) are in my Phase 3-4 category, not Phase 2.

---

## 1. What Already Exists (Existing Module Landscape)

These modules are ALREADY separated and imported by agent_graph.py:

| Module | Lines | Responsibility |
|---|---|---|
| `context_compressor.py` | 1,822 | Message selection, compression, token estimation, tool result summarization |
| `retrieval_router.py` | 1,195 | RAG backend routing, hit scoring, reranking, graded retrieval |
| `model_management.py` | 1,169 | UbuntuLlamaManager, Ollama, embedding lifecycle, power actions |
| `alpharavis_toolsets.py` | 552 | Toolset dataclasses, resolution, materialization, schema cache |
| `context_references.py` | 601 | @file:/@git:/@diff: reference parsing and expansion |
| `model_metadata.py` | 462 | Context length discovery, token estimation, model metadata |
| `operational_logging.py` | 429 | Structured operational logging |
| `media_analysis.py` | 413 | Media (image/video) preparation for model |
| `error_classifier.py` | 339 | Error classification: reason enum, retry/compress/fail decisions |
| `file_safety.py` | 299 | File read/write safety decisions and dangerous path detection |
| `owner_power_tools.py` | 303 | Owner-gated shutdown/start/restart for llama/ComfyUI |
| `compression_redact.py` | 118 | Secret redaction from tool output and messages |
| `internal_context.py` | 150 | Internal context block scrubbing (streaming + batch) |
| `prompt_assembly.py` | 137 | Stable prompt context, environment hints, truncation |
| `repo_skills.py` | 656 | AI skill card scanning, manifest, draft export |
| `responses_client.py` | 275 | Direct `/v1/responses` API client |
| `provider_hardening.py` | 327 | Provider hardening (timeouts, retries) |
| `document_ingest.py` | ? | Document file loading |
| `run_state_manager.py` | 195 | Run checkpoint save/load/resume |
| `rag_api_client.py` | 191 | External RAG API client |
| `rag_pins_manager.py` | 137 | Active RAG source pin management |
| `runtime_settings.py` | 38 | Runtime overrides from config files |
| `curated_memory_review.py` | 134 | Curated memory candidate review |
| `maintenance_helpers.py` | 97 | Maintenance scheduling decisions |

Also available from `ai_stack/`:
- `ai_stack/context_budget/`: scheduler, policies, leases, background task runner
- `ai_stack/parallel_executor/`: task graph DAG, worktree manager, worker spawner

---

## 2. What's Still Inside agent_graph.py — By Responsibility

### A. Graph Orchestration (KEEP — this IS the orchestrator)

| Function | Lines ~ | Risk |
|---|---|---|
| `_build_graph()` | 1160 | HIGH — core graph assembly, agent creation, handoff wiring |
| `run_profile_start_node()` | 120 | HIGH — sets up run profile, trace, runtime settings |
| `resume_prompt_node()` | 10 | HIGH — graph edge |
| `route_after_run_profile_start()` | 4 | HIGH — graph routing |
| `large_paste_post_compression_node()` | 96 | HIGH — post-compression large paste inject + ingest |
| `route_decision_node()` | 106 | HIGH — primary routing decision |
| `route_after_decision()` | 7 | HIGH — graph routing |
| `hard_context_stop_node()` | 12 | HIGH — graph terminal node |
| `crisis_preflight_node()` | 43 | HIGH — llama backend readiness check |
| `route_after_crisis_preflight()` | 4 | HIGH — graph routing |
| `planner_node()` | 193 | HIGH — calls LLM planner, creates execution plan |
| `fast_chat_node()` | 149 | HIGH — direct LLM answer path |
| `pre_run_context_guard_node()` | 163 | HIGH — pre-swarm compression gate |
| `handoff_context_guard_node()` | 103 | HIGH — pre-handoff compression gate |
| `final_budget_rescue_node()` | 112 | HIGH — context overflow rescue compression |
| `memory_kernel_prefetch_node()` | 199 | HIGH — memory + archive recall before swarm |
| `memory_kernel_sync_node()` | 84 | HIGH — memory sync after swarm |
| `active_rag_prefetch_node()` | 122 | HIGH — RAG pin context injection |
| `skill_library_node()` | 68 | HIGH — skill context injection |
| `context_guard_node()` | 120 | HIGH — post-swarm context compression |
| `memory_notice_node()` | 17 | HIGH — memory availability notice |
| `run_profile_finish_node()` | 72 | HIGH — finalize run, async reviewer |
| `swarm_trace_start_node()` | 12 | HIGH — trace markers |
| `swarm_trace_finish_node()` | 13 | HIGH — trace markers |
| `run_swarm_with_context_retry()` | 145 | HIGH — swarm invoke with crisis retry |
| `run_crisis_manager()` | 51 | HIGH — crisis agent invocation |
| `track_marker_node()` | 8 | HIGH — graph tracer |
| `_create_ui_assistant()` | 30 | HIGH — sub-agent creation (used in graph) |
| `_create_debugger_subgraph()` | 60 | HIGH — sub-agent creation (used in graph) |
| `make_graph()` | 24 | HIGH — top-level entry point |
| `_should_load_mcp()` | 12 | HIGH — graph initialization |
| `_open_mongodb_store()` | 21 | HIGH — graph initialization |
| `_embedding_scheduler_loop()` | 38 | HIGH — background daemon |
| `_vector_backfill_daemon_loop()` | 38 | HIGH — background daemon |
| `_cancel_background_tasks()` | 8 | HIGH — cleanup |

**Verdict: All of these STAY in agent_graph.py.** They are the orchestration.

### B. Pure Helper Functions — LOW RISK to extract

#### B1. Config/env helpers (already duplicated in some modules)

| Function | Lines | Depends on | Depended by | Target |
|---|---|---|---|---|
| `_env_bool()` | 5 | os.getenv | ~50 places | **MOVE to `prompt_assembly.py`** (already has `env_bool`) |
| `_env_float()` | 10 | os.getenv | ~15 places | **MOVE to `prompt_assembly.py`** |
| `_env_disable_streaming()` | 7 | _env_bool | few | **MOVE to `prompt_assembly.py`** |

Risk: LOW. These already exist in `prompt_assembly.py` (`env_bool`), `context_compressor.py` (`_env_bool`), `model_metadata.py` (`_env_bool`). The agent_graph.py versions have a different signature for `_env_bool` (adds parameter default). Unify by making `prompt_assembly.env_bool` accept an optional default, then replace all agent_graph.py calls.

#### B2. Message content utilities

| Function | Lines | Depends on | Depended by | Target |
|---|---|---|---|---|
| `_plain_text_content()` | 32 | None (pure) | `_message_with_plain_text_content` | **MOVE to `context_compressor.py`** — already has `_content_to_text` |
| `_message_with_plain_text_content()` | 17 | `_plain_text_content` | `_model_input_messages`, `_message_for_context_estimate` | **MOVE to `context_compressor.py`** |
| `_plain_text_messages()` | 6 | `_message_with_plain_text_content` | few | **MOVE to `context_compressor.py`** |
| `_model_input_messages()` | 18 | `_message_with_plain_text_content` | few | **MOVE to `context_compressor.py`** |
| `_message_text()` | 14 | pure | few | **ALREADY in `context_compressor.py`** — remove duplicate |
| `_message_to_json()` | 14 | pure | `_message_for_context_estimate` | **ALREADY in `context_compressor.py`** as `_message_mapping` — remove duplicate |
| `_message_id()` | 5 | pure | few | **ALREADY in `context_compressor.py`** — remove duplicate |
| `_message_content_text()` | 10 | pure | few | **ALREADY in `context_compressor.py`** as `message_text` — remove duplicate |
| `_message_role_name()` | 6 | pure | few | **ALREADY in `context_compressor.py`** as `message_role` — remove duplicate |
| `_is_remove_message()` | 4 | pure | few | **ALREADY in `context_compressor.py`** — possible |

Risk: VERY LOW. These are already duplicated between agent_graph.py and context_compressor.py. Consolidation removes 60+ lines of duplicate code.

#### B3. Token estimation glue

| Function | Lines | Depends on | Depended by | Target |
|---|---|---|---|---|
| `_estimate_text_tokens()` | 4 | `_compressor_estimate_tokens` | few | **KEEP** — 4-line delegator, fine |
| `_tool_schema_for_budget()` | 11 | pure | few | **MOVE to `context_compressor.py`** |
| `_estimate_tool_schema_tokens()` | 10 | `_compressor_estimate_tokens` | few | **MOVE to `context_compressor.py`** |
| `_estimate_request_tokens()` | 20 | `_compressor_estimate_tokens` | few | **MOVE to `context_compressor.py`** |
| `_estimate_tokens()` | 2 | `_compressor_estimate_tokens` | ~15 places | **KEEP** — 2-line delegator, fine |
| `_message_for_context_estimate()` | 13 | `_message_to_json` | `_estimate_tokens` | **MOVE to `context_compressor.py`** |

Risk: LOW. Pure computation delegating to context_compressor.

#### B4. Source content analysis (pure functions, no LLM calls, no state)

| Function | Lines | Depends on | Depended by | Target |
|---|---|---|---|---|
| `_detect_source_content_type()` | 68 | pure regex | `_source_metadata_summary` | **MOVE to new `source_content.py`** |
| `_extract_source_keywords()` | 14 | `_SOURCE_STOPWORDS` | few | **MOVE to `source_content.py`** |
| `_extract_source_entities()` | 19 | `_SOURCE_STOPWORDS` | few | **MOVE to `source_content.py`** |
| `_extract_source_symbols()` | 22 | pure regex | few | **MOVE to `source_content.py`** |
| `_source_title_from_text()` | 11 | pure | `_source_metadata_summary` | **MOVE to `source_content.py`** |
| `_source_metadata_summary()` | 22 | `_detect_source_content_type`, | `_ingest_large_paste_messages` | **MOVE to `source_content.py`** |
| `_SOURCE_STOPWORDS` | 30 | none | `_extract_source_keywords`, | **MOVE to `source_content.py`** |

Risk: LOW. All are pure functions with no LLM calls, no state, no external deps except stdlib modules. Only used by agent_graph.py internals and `_ingest_large_paste_messages`.

#### B5. Line range / text parsing utilities

| Function | Lines | Depends on | Depended by | Target |
|---|---|---|---|---|
| `_line_ranges_from_text()` | 6 | pure regex | `_classify_prompt_for_retrieval` | **MOVE to `source_content.py`** |
| `_classifier_window_text()` | 29 | pure | `_classify_prompt_for_retrieval` | **MOVE to `source_content.py`** |
| `_local_retrieval_query()` | 25 | pure regex | `_prepare_retrieval_query` | **MOVE to `source_content.py`** |
| `_parse_classifier_json()` | 32 | pure | `_classify_prompt_for_retrieval` | **MOVE to `source_content.py`** |
| `_normalize_line_ranges()` | 20 | pure | `_line_range_indexes` | **MOVE to `source_content.py`** |
| `_line_range_indexes()` | 11 | `_normalize_line_ranges` | `_text_from_line_ranges` | **MOVE to `source_content.py`** |
| `_text_from_line_ranges()` | 12 | `_line_range_indexes` | few | **MOVE to `source_content.py`** |
| `_strip_line_ranges_from_text()` | 41 | `_normalize_line_ranges` | few | **MOVE to `source_content.py`** |
| `_tail_question_line_ranges()` | 14 | pure | few | **MOVE to `source_content.py`** |
| `_bounded_text_window()` | 29 | pure | few | **MOVE to `source_content.py`** |

Risk: LOW. All are pure text/parsing functions. No LLM calls. No external state.

#### B6. Numeric / ratio utilities

| Function | Lines | Depends on | Depended by | Target |
|---|---|---|---|---|
| `_ratio_token_limit()` | 17 | `_env_float` | `_ratio_token_limit_for_context` | **MOVE to `context_compressor.py`** |
| `_ratio_token_limit_for_context()` | 18 | `_ratio_token_limit` | `_context_budget_snapshot` | **MOVE to `context_compressor.py`** |
| `_effective_context_limit()` | 7 | pure | `_context_budget_snapshot` | **MOVE to `context_compressor.py`** |

Risk: LOW. Simple numeric calculations already used by compression logic.

#### B7. Namespace / store key helpers

| Function | Lines | Depends on | Depended by | Target |
|---|---|---|---|---|
| `_thread_archive_ns()` | 4 | none | widely used | **KEEP** — 4-line tuple builder, too coupled to graph constants |
| `_thread_archive_collection_ns()` | 4 | none | widely used | **KEEP** — same reason |
| `_thread_source_record_ns()` | 4 | none | widely used | **KEEP** |
| `_thread_session_turn_ns()` | 4 | none | widely used | **KEEP** |
| `_thread_artifact_ns()` | 4 | none | widely used | **KEEP** |
| `_thread_rag_config_ns()` | 4 | none | widely used | **KEEP** |
| `_curated_memory_ns()` | 4 | none | widely used | **KEEP** |
| `_split_csv_env()` | 5 | none | few | **MOVE to `prompt_assembly.py`** |
| `_sanitize_store_scope()` | 4 | none | few | **KEEP** — too small |
| `_curated_memory_scope()` | 10 | none | few | **KEEP** — business logic for scope decision |

Risk: LOW for the namespace helpers to stay. They're 4-line tuple constructors. Moving them would require importing namespace constants across modules, creating coupling.

#### B8. Store helpers

| Function | Lines | Depends on | Depended by | Target |
|---|---|---|---|---|
| `_maybe_put()` | 9 | get_store | widely used | **KEEP** — generic store helper, fine |
| `_maybe_get()` | 12 | get_store | widely used | **KEEP** — same |
| `_maybe_search()` | 10 | get_store | widely used | **KEEP** — same |
| `_store_item_value()` | 6 | pure | widely used | **KEEP** — minimal |
| `_store_item_key()` | 5 | pure | widely used | **KEEP** — minimal |

Risk: LOW. These are thin wrappers around the LangGraph Store API. They could go to a `store_helpers.py` but the benefit is marginal.

### C. Medium-Complexity Logic — MEDIUM RISK to extract

#### C1. Context budget snapshot & discovery

| Function | Lines | Risk | Depends on | Target |
|---|---|---|---|---|
| `_context_discovery_model()` | 7 | LOW | os.getenv | **MOVE to `model_metadata.py`** |
| `_context_discovery_base_url()` | 8 | LOW | os.getenv | **MOVE to `model_metadata.py`** |
| `_context_discovery_api_key()` | 7 | LOW | os.getenv | **MOVE to `model_metadata.py`** |
| `_detected_context_length()` | 15 | MEDIUM | `_get_model_context_length` (from model_metadata) | **MOVE to `model_metadata.py`** |
| `_provider_context_length_override()` | 14 | LOW | pure | **MOVE to `model_metadata.py`** |
| `_context_budget_snapshot()` | 42 | MEDIUM | Many _ratio*, _estimate*, _detect* functions | **MOVE to new `context_budget_state.py`** |
| `_static_context_reserve_tokens()` | 17 | MEDIUM | global graph state | **KEEP** — uses GRAPH_STATIC_CONTEXT_RESERVE_TOKENS |
| `_static_context_reserve_detail()` | 12 | MEDIUM | global graph state | **KEEP** |
| `_register_static_context_reserve()` | 25 | MEDIUM | global graph state | **KEEP** |
| `_agent_name_from_toolsets()` | 19 | MEDIUM | global graph state | **KEEP** |
| `_active_context_token_limit()` | 9 | MEDIUM | `_ratio_token_limit_for_context` | **MOVE to `context_budget_state.py`** |
| `_handoff_context_token_limit()` | 9 | MEDIUM | `_ratio_token_limit_for_context` | **MOVE to `context_budget_state.py`** |
| `_hard_context_token_limit()` | 4 | MEDIUM | `_hard_context_token_limit_for_context` | **MOVE to `context_budget_state.py`** |
| `_hard_context_token_limit_for_context()` | 11 | MEDIUM | `_ratio_token_limit` | **MOVE to `context_budget_state.py`** |
| `inspect_context_budget()` | 10 | LOW | `_context_budget_snapshot` | **KEEP** — @tool function, graph wiring |

Risk: MEDIUM. `_context_budget_snapshot()` is called from multiple graph nodes and the error handler. Its dependencies are spread across agent_graph.py. Best to extract the pure calculation part to a new module and keep the state-dependent wrappers in agent_graph.py.

#### C2. Source storage and ingest

| Function | Lines | Risk | Target |
|---|---|---|---|
| `_store_raw_source_record()` | 54 | MEDIUM | **MOVE to new `source_storage.py`** |
| `_load_raw_source_record()` | 30 | LOW | **MOVE to `source_storage.py`** |
| `_ingest_large_paste_messages()` | 329 | HIGH — complex, many deps | **MOVE to new `large_paste_ingest.py`** |
| `_ingest_pending_document_uploads()` | 155 | MEDIUM | **MOVE to `large_paste_ingest.py`** |

Risk: HIGH for `_ingest_large_paste_messages` (330 lines, deeply intertwined). Should be wrapped, not moved directly. Keep old function in place, have new module expose the same signature, and route through feature flag.

#### C3. Large paste classification

| Function | Lines | Risk | Target |
|---|---|---|---|
| `_large_paste_intent_classifier_enabled()` | 4 | LOW | **MOVE to `large_paste_ingest.py`** |
| `_large_paste_small_classifier_enabled()` | 4 | LOW | **MOVE to `large_paste_ingest.py`** |
| `_large_paste_small_classifier_min_chars()` | 4 | LOW | **MOVE to `large_paste_ingest.py`** |
| `_classify_large_paste_intent()` | 70 | MEDIUM | **MOVE to `large_paste_ingest.py`** |
| `_large_paste_intent_from_small_classifier()` | 35 | MEDIUM | **MOVE to `large_paste_ingest.py`** |
| `_classify_large_paste_for_ingest()` | 39 | MEDIUM | **MOVE to `large_paste_ingest.py`** |
| `_large_paste_instruction_brief()` | 35 | LOW | **MOVE to `large_paste_ingest.py`** |
| `_large_paste_question_brief()` | 8 | LOW | **MOVE to `large_paste_ingest.py`** |
| `_large_paste_document_body_for_index()` | 38 | LOW | **MOVE to `large_paste_ingest.py`** |
| `_manual_large_paste_blocks()` | 39 | LOW | **MOVE to `large_paste_ingest.py`** |
| `_large_paste_marker()` | 78 | LOW | **MOVE to `large_paste_ingest.py`** |
| `_replace_message_content()` | 16 | LOW | **MOVE to `context_compressor.py`** |
| `_large_paste_ingest_enabled()` + config | ~40 | LOW | **MOVE to `large_paste_ingest.py`** |
| `_large_paste_auto_should_ingest()` | 68 | MEDIUM | **MOVE to `large_paste_ingest.py`** |

Risk: MEDIUM. The functions form a self-contained subsystem that only interacts with agent_graph.py through `_ingest_large_paste_messages` and `_large_paste_post_compression_node`.

### D. SSH / Command Safety (MEDIUM RISK)

| Function | Lines | Target |
|---|---|---|
| `_first_shell_word()` | 7 | **MOVE to `file_safety.py`** (or new `command_safety.py`) |
| `_command_segments()` | 4 | **MOVE to `file_safety.py`** |
| `_is_read_only_command()` | 79 | **MOVE to `file_safety.py`** |
| `_require_command_approval()` | 38 | **MOVE to `file_safety.py`** |

Risk: MEDIUM. These functions are already about safety decisions (like `file_safety.py`). However, `_require_command_approval` calls `interrupt()` from `langgraph.types` — this is a graph-specific function. Best to separate: the pure check (`_is_read_only_command`) moves to `command_safety.py`, the graph-`interrupt()` wrapper stays.

### E. MCP Configuration (MEDIUM RISK)

| Function | Lines | Target |
|---|---|---|
| `_mcp_transport()` | 4 | **MOVE to new `mcp_config.py`** (or extend `alpharavis_toolsets.py`) |
| `_resolve_mcp_path()` | 8 | **MOVE to `mcp_config.py`** |
| `_mcp_config_candidate_paths()` | 27 | **MOVE to `mcp_config.py`** |
| `_expand_mcp_config_value()` | 10 | **MOVE to `mcp_config.py`** |
| `_load_mcp_config_from_paths()` | 47 | **MOVE to `mcp_config.py`** |
| `_mcp_connection_from_config()` | 20 | **MOVE to `mcp_config.py`** |
| `_load_configured_mcp_tools()` | 78 | **MOVE to `mcp_config.py`** |

Risk: MEDIUM. These functions are only used during `make_graph()` initialization. They have clear internal coupling. Moving them to a separate `mcp_config.py` keeps the MCP concerns isolated.

### F. Pixelle Media Pipeline (MEDIUM-HIGH RISK)

| Function | Lines | Target |
|---|---|---|
| `monitor_pixelle_job()` | 53 | **KEEP** — @tool function, part of graph/tool wiring |
| `_format_pixelle_failure()` | 15 | **MOVE to `media_analysis.py`** |
| `_pixelle_preflight()` | 77 | **MOVE to `media_analysis.py`** |
| `_pixelle_preflight_notice()` | 12 | **MOVE to `media_analysis.py`** |
| `_pixelle_preflight_woke_comfy()` | 12 | **MOVE to `media_analysis.py`** |
| `_remember_pixelle_lifecycle()` | 12 | **MOVE to `media_analysis.py`** |
| `_shutdown_comfy_after_pixelle_delay()` | 30 | **MOVE to `media_analysis.py`** |
| `_schedule_comfy_shutdown_if_woke()` | 22 | **MOVE to `media_analysis.py`** |
| `_extract_media_urls()` | 9 | **MOVE to `media_analysis.py`** |
| `_media_type_from_value()` | 14 | **MOVE to `media_analysis.py`** |
| `_media_auto_index_enabled()` | 21 | **MOVE to `media_analysis.py`** |
| `_register_media_asset()` | 101 | **MOVE to `media_analysis.py`** |
| `_media_registration_summary()` | 17 | **MOVE to `media_analysis.py`** |
| `_register_pixelle_media_from_result()` | 32 | **MOVE to `media_analysis.py`** |

Risk: MEDIUM-HIGH. The `@tool`-decorated public functions (`start_pixelle_remote`, `check_pixelle_job`, etc.) MUST stay because they're registered as tools in `_build_graph()`. The private helpers can move to `media_analysis.py` which already has `prepare_media_for_model` and `decide_media_mode`. But the module already exists there, so this is extending, not creating.

### G. Model Construction Helpers (HIGH RISK)

| Function | Lines | Target |
|---|---|---|
| `_model()` | 18 | **KEEP** — called from many graph nodes |
| `_agent_thinking_bind_kwargs()` | 11 | **MOVE to `model_metadata.py`** |
| `_planner_bind_kwargs()` | 12 | **MOVE to `model_metadata.py`** |
| `_deepagents_responses_enabled()` | 5 | **KEEP** — feature flag for graph decisions |
| `_deepagents_responses_extra_body()` | 22 | **MOVE to `responses_client.py`** |
| `_deepagents_responses_streaming_policy()` | 30 | **MOVE to `responses_client.py`** |
| `_deepagents_responses_model()` | 35 | **MOVE** — but high coupling to `ChatLiteLLM` instantiation |
| `_agent_model()` | 5 | **KEEP** — thin delegator |
| `_deep_agent_model()` | 17 | **KEEP** — model construction with env vars |
| `_server_model_manager_model()` | 18 | **KEEP** — similar, specific to power/crisis agents |
| `_budget_guarded_agent_model()` | 116 | **MOVE** — major function, wraps model with budget guards |
| `_create_budgeted_deep_agent()` | 12 | **KEEP** — thin delegator |
| `_text_only_agent_model()` | 54 | **MOVE** — removes vision/tool capabilities |
| `_responses_direct_calls_enabled()` | 4 | **KEEP** — feature flag |
| `_state_trace_id()` | 5 | **KEEP** — graph state access |
| `_state_trace_started()` | 5 | **KEEP** — graph state access |
| `_trace_step()` | 13 | **KEEP** — graph tracing |
| `_trace_updates()` | 4 | **KEEP** — graph tracing |
| `_ainvoke_direct_model()` | 163 | HIGH RISK — complex, response-mode-aware model call |
| `_ainvoke_direct_text()` | 23 | MEDIUM RISK — simpler text model call |
| `_direct_model_compatibility_retry()` | 8 | **MOVE to `responses_client.py`** |
| `_direct_model_provider_profile()` | 8 | **MOVE to `responses_client.py`** |
| `_model_management_enabled()` etc | ~100 | **KEEP** — feature flags used throughout graph |

Risk: HIGH. These are deeply coupled to the graph's model creation and call strategy. Don't move them in early phases. Only after Phase 3 wrapping.

### H. Context Budget / Llama Scheduler Integration (HIGH RISK)

| Function | Lines | Target |
|---|---|---|
| `_log_model_request_budget()` | 40 | **MOVE to `operational_logging.py`** (if operational, keeps graph clean) |
| `_context_scheduler_setting()` | 4 | **KEEP** — feature flag |
| `_context_scheduler_enabled()` | 14 | **KEEP** — feature flag |
| `_max_output_tokens_from_kwargs()` | 13 | **MOVE to `model_metadata.py`** |
| `_context_priority_for_purpose()` | 11 | **MOVE to `ai_stack/context_budget/` or new `context_budget_state.py`** |
| `_background_context_for_purpose()` | 24 | **MOVE to `context_budget_state.py`** |
| `_preferred_llama_instance_for_model()` | 10 | **MOVE to `context_budget_state.py`** |
| `_reserve_llama_context_lease()` | 48 | **MOVE to `context_budget_state.py`** |
| `_release_llama_context_lease()` | 9 | **MOVE to `context_budget_state.py`** |
| `_handle_llama_context_response()` | 9 | **MOVE to `context_budget_state.py`** |
| `_bound_tools_from_args()` | 11 | **MOVE to `context_compressor.py`** |
| `_register_static_context_reserve()` | 25 | **KEEP** — modifies global graph state |

Risk: HIGH. These functions are the bridge between the graph and the llama.cpp context scheduler. Moving them requires careful interface design.

### I. Large Constants / Pattern Lists

| Constant | Type | Target |
|---|---|---|
| `FAST_PATH_DENY_PATTERNS` | list | **MOVE to `prompt_assembly.py`** |
| `FAST_PATH_FORCE_PATTERNS` | list | **MOVE to `prompt_assembly.py`** |
| `MANUAL_COMPRESSION_PATTERNS` | list | **MOVE to `context_compressor.py`** |
| `COMPRESSION_PAUSE_PATTERNS` | list | **MOVE to `context_compressor.py`** |
| `AGENT_POLICY_PROMPT` (composite) | str | **KEEP** — built from policy prompts below |
| `HANDOFF_POLICY_PROMPT` | str | **MOVE to `prompt_assembly.py`** |
| `ARCHIVE_RETRIEVAL_POLICY_PROMPT` | str | **MOVE to `prompt_assembly.py`** |
| `CODE_WINDOW_POLICY_PROMPT` | str | **MOVE to `prompt_assembly.py`** |
| `SPECIALIST_LOCAL_PLAN_PROMPT` | str | **MOVE to `prompt_assembly.py`** |
| `OPTIONAL_TOOL_MANIFEST` | list | **MOVE to `alpharavis_toolsets.py`** |
| `MCP_SERVER_INFOS` (global mutable) | list | **KEEP** — graph initialization state |
| `MCP_SCHEMA_CACHE` (global mutable) | dict | **KEEP** — graph initialization state |
| `GRAPH_TOOLSET_PROFILE` (global mutable) | dict | **KEEP** — graph initialization state |
| `REMOTE_PCS` | dict | **MOVE to `model_management.py`** — already accessed from there |
| `SOURCE_RECORD_INDEX_NS` etc | tuple | **KEEP** — namespace constants too coupled |
| `*_MESSAGE_ID` constants | str | **KEEP** — graph-global identifiers |

Risk: LOW for consts. Can simply move the strings and change the import.

### J. Tool Functions (@tool-decorated) — MUST STAY

ALL `@tool`-decorated functions and their async equivalents MUST stay in agent_graph.py because:
1. They are registered in `_build_graph()` via direct function references
2. Moving them would require `_build_graph()` to import from the new module, creating circular import risk since `_build_graph()` already imports heavily
3. The cost of moving them is higher than keeping them

This includes (~1,200 lines of tool functions):
- `start_pixelle_remote`, `check_pixelle_job`, `register_media_asset`, `semantic_media_search`
- `check_external_service`, `inspect_model_management_status`, `check_ollama_models`
- `inspect_ubuntu_llama_manager`, `control_ubuntu_llama_service`
- `wake_on_lan`, `execute_ssh_command`, `execute_local_command`
- `fast_web_search`, `deep_web_research`, `ask_documents`
- `read_alpha_ravis_architecture`, `locate_repo_surface`, `list_repo_ai_skills`
- `create_curated_memory_review_candidates`, `search_curated_memory`
- `query_source`, `query_sources`, `ingest_document_file`, `agentic_rag_retrieve`
- `write_alpha_ravis_artifact`, `read_alpha_ravis_artifact`, `list_alpha_ravis_artifacts`
- `check_hermes_agent`, `call_hermes_agent`
- `search_archived_context`, `search_session_history`, `record_curated_memory`
- `search_agent_memory`, `record_agent_memory`, `search_skill_library`
- `activate_skill_candidate`, `deactivate_skill`
- All `owner_*` functions
- `apply_model_context_policy`, `prepare_comfy_for_pixelle`, `request_power_management_action`
- `describe_optional_tool_registry` (complex)

**Verdict: These 50+ tool functions STAY.**

---

## 3. Proposed Module Boundaries

### New modules to create:

1. **`langgraph-app/source_content.py`** (~250 lines)
   - Content type detection, keyword/entity/symbol extraction, title extraction
   - Line range parsing, classifier window text, JSON parsing
   - Pure functions, no LLM calls
   - Dependencies: only stdlib (re)

2. **`langgraph-app/large_paste_ingest.py`** (~600 lines)
   - Large paste classification, intent detection
   - Ingest workflow (`_ingest_large_paste_messages`)
   - Status: recommended for Phase 3 wrapping (MEDIUM-HIGH risk)

3. **`langgraph-app/command_safety.py`** (~150 lines)
   - SSH command safety checks, pipe splitting
   - Read-only command classifier
   - Dependencies: stdlib (re, shlex)

4. **`langgraph-app/mcp_config.py`** (~200 lines)
   - MCP config file discovery, loading, transport resolution
   - Tool loading from MCP server configurations
   - Only used during graph init

5. **`langgraph-app/context_budget_state.py`** (~300 lines)
   - Context budget snapshot, token limit calculations
   - Llama context lease management
   - Priority routing for budget purposes

### Existing modules to extend:

6. **`prompt_assembly.py`** (+50 lines)
   - Add policy prompt constants
   - Add FAST_PATH patterns
   - Add `_env_bool`, `_split_csv_env`

7. **`context_compressor.py`** (+80 lines)
   - Add message-to-json helpers, token estimation wrappers
   - Add build_tool_call_map, bounded_text_window
   - Add ratio_token_limit utilities

8. **`model_metadata.py`** (+50 lines)
   - Add context discovery model/base_url/api_key helpers
   - Add provider context length override
   - Add max_output_tokens_from_kwargs

9. **`media_analysis.py`** (+250 lines)
   - Add pixelle preflight, shutdown scheduling
   - Add media URL extraction, auto-index decisions
   - Add media registration helpers

10. **`operational_logging.py`** (+40 lines)
    - Add `_log_model_request_budget` (or keep thin wrapper in agent_graph.py)

---

## 4. Phased Refactor Plan

### Phase 1 — Define shared types/interfaces only (no behavior change)
- No changes. Documentation phase.

### Phase 2 — Extract low-risk pure functions
Files created:
- `source_content.py` (~250 lines extracted)
- `command_safety.py` (~150 lines extracted)

Files extended:
- `prompt_assembly.py` (+50 lines: constants, env helpers)
- `context_compressor.py` (+80 lines: message utils, ratio utils)
- `model_metadata.py` (+50 lines: context discovery helpers)

agent_graph.py reduction: ~530 lines removed

**Tests to run:**
```bash
pytest -q tests/
python scripts/alpharavis_setup.py bridge-smoke
```

**Rollback:** Remove new imports, they were just re-exports. No behavior change.

### Phase 3 — Wrap risky existing logic behind stable interfaces
Files created:
- `large_paste_ingest.py` (wrapper module that calls old functions in agent_graph.py)
- `context_budget_state.py` (similar wrapper)

Files extended:
- `media_analysis.py` (+100 lines: pixelle helpers)
- `operational_logging.py` (+40 lines)

agent_graph.py: functions stay but are wrapped by new module interfaces.
New code calls wrappers; old code continues unchanged.

**Feature flag:** `ALPHARAVIS_USE_MODULAR_LARGE_PASTE=false` (default)
When enabled, new modular path is used. When disabled, old code runs.

**Tests:** smoke test + manual ingestion test.

### Phase 4 — Move larger responsibilities after tests pass
Once Phase 3 wrappers are proven stable with the feature flag enabled:
- Move `_ingest_large_paste_messages` body to `large_paste_ingest.py`
- Remove wrapper/facade
- Change feature flag default to `true`

agent_graph.py reduction at this phase: ~600 more lines

### Phase 5 — Parallel executor clean module boundaries
Already extracted to `ai_stack/parallel_executor/`. The hook in agent_graph.py
(`_parallel_execution_hook`) is a thin delegator. No changes needed.

---

## 5. What STAYS in agent_graph.py (and why)

| Section | Lines ~ | Why it stays |
|---|---|---|
| Import block | 460 | All imports already clean |
| State classes (AlphaRavisState, DebuggerState) | 55 | Graph state definition — belongs with graph |
| Service URL constants + namespace tuples | 60 | Infrastructure configuration |
| Message ID constants | 15 | Graph-global identifiers |
| Graph globals (MCP_SERVER_INFOS, schema cache, toolset profile) | 15 | Initialized during graph build |
| AGENT_POLICY_PROMPT (composite) | 10 | Assembled from sub-prompts that will be imported |
| All `@tool`-decorated functions | 1,200 | Registered directly in `_build_graph()` |
| All graph node functions (~24 nodes) | 1,800 | Core orchestration — each is a graph step |
| `_build_graph()` + nested helpers | 1,160 | Graph assembly — the heart of orchestration |
| `make_graph()` + daemon loops | 130 | Top-level entry points |
| Store helpers (`_maybe_put`, `_maybe_get`, `_maybe_search`) | 45 | Thin wrappers used everywhere |
| Namespace helpers | 50 | Too coupled to graph constants |
| Model construction helpers | 250 | Deeply coupled to `ChatLiteLLM` instantiation in graph context |
| `_budget_guarded_agent_model()` | 116 | Wraps model with budget — graph-specific |
| Feature flag functions (`_model_management_enabled()`, `_crisis_manager_enabled()`, etc.) | 100 | Used throughout graph nodes |
| `_parallel_execution_hook()` | 37 | Graph integration point |
| `_classified_error_profile()` | 55 | Called from crisis retry, swarm, preflight |
| `_scan_persistent_context()` | 12 | Graph state access |
| `_available_agent_names()` | 16 | Graph-global fact |
| `_configure_llm_cache()` | 11 | Graph init |
| `_warn_about_mongo_checkpointer()` | 17 | Graph init |
| `_workspace_root()`, `_file_safety_unavailable()` | 20 | Graph init |
| `_check_read_path/list_path/write_path()` | 30 | Tool-adjacent safety wrappers |
| Compression helper utilities (~30 functions) | 500 | Called from multiple graph nodes |
| Graph node helper functions (`_drop_previous_compaction_messages`, `_message_stable_key`, etc.) | 200 | Used by graph nodes only |
| `_summarize_archive_records()`, `_collect_curated_memory_context()` | 200 | Called from memory kernel nodes |
| Vector memory + backfill queue utilities | 300 | Tool-adjacent store operations |
| Retrieval query utilities (`_latest_user_query`, `_recent_turn_window_text`, etc.) | 150 | Graph node helpers |
| Run state helpers (`_profile_update`, `_save_run_state_checkpoint`, etc.) | 150 | Graph state management |
| `_tool_name_for_profile()`, `_dedupe_tools()`, `_tools_by_name()` | 30 | Graph tool management |
| `_materialized_profile()`, `_infer_selected_toolsets()` | 60 | Toolset materialization |
| `_toolset_context_for_request()`, `_stable_prompt_context()` | 30 | Prompt construction |
| `_fast_path_decision()`, `_long_prompt_direct_route_decision()` | 70 | Route decision logic |
| `_planner_needed()`, `_looks_like_coding_task()` | 30 | Planner pre-checks |
| Async reviewer (`_async_reviewer_enabled()`, `_run_async_review_snapshot()`, etc.) | 110 | Post-swarm review pipeline |
| Vector memory `_maybe_index_vector_memory` + `_format_vector_result` | 120 | Tool-adjacent |
| RAG helpers (`_normalize_rag_document_hit`, `_load_thread_rag_pins`, etc.) | 100 | Tool-adjacent |

**Total that stays: ~9,500 lines (68% of file)**

---

## 6. What Gets Extracted (and what doesn't)

### Phase 2 — Safe extractions (~530 lines):

| From agent_graph.py | To module | Lines |
|---|---|---|
| `_detect_source_content_type()` | `source_content.py` | 68 |
| `_extract_source_keywords()` | `source_content.py` | 14 |
| `_extract_source_entities()` | `source_content.py` | 19 |
| `_extract_source_symbols()` | `source_content.py` | 22 |
| `_source_title_from_text()` | `source_content.py` | 11 |
| `_source_metadata_summary()` | `source_content.py` | 22 |
| `_SOURCE_STOPWORDS` | `source_content.py` | 30 |
| `_line_ranges_from_text()` | `source_content.py` | 6 |
| `_classifier_window_text()` | `source_content.py` | 29 |
| `_local_retrieval_query()` | `source_content.py` | 25 |
| `_parse_classifier_json()` | `source_content.py` | 32 |
| `*_line_range*()` (4 functions) | `source_content.py` | 54 |
| `_tail_question_line_ranges()` | `source_content.py` | 14 |
| `_bounded_text_window()` | `source_content.py` | 29 |
| `_first_shell_word()`, `_command_segments()`, `_is_read_only_command()` | `command_safety.py` | 90 |
| `HANDOFF_POLICY_PROMPT` + 3 others | `prompt_assembly.py` | 20 |
| `FAST_PATH_DENY/FORCE_PATTERNS` | `prompt_assembly.py` | 20 |
| `MANUAL_COMPRESSION_PATTERNS` | `context_compressor.py` | 6 |
| `COMPRESSION_PAUSE_PATTERNS` | `context_compressor.py` | 6 |
| `_ratio_token_limit()` | `context_compressor.py` | 17 |
| `_ratio_token_limit_for_context()` | `context_compressor.py` | 18 |
| `_context_discovery_model/url/key()` | `model_metadata.py` | 22 |
| `_provider_context_length_override()` | `model_metadata.py` | 14 |
| `_message_for_context_estimate()` | `context_compressor.py` | 13 |

### Too risky to extract now:

- `_budget_guarded_agent_model()` (116 lines) — deeply embedded in the model creation chain; moving it would require extracting the entire model creation subsystem which touches `ChatLiteLLM`, `DeepAgents`, streaming policies, and tool binding. HIGH RISK.
- `_ainvoke_direct_model()` (163 lines) — complex provider-aware model invocation with streaming/response-mode dispatch. HIGH RISK.
- `_build_graph()` (1,160 lines) — the graph IS the orchestration. STAYS.
- `_run_hermes_style_compression()` (120 lines) — calls into `context_compressor.compress_messages` but wraps with graph state, archive storage, and node update semantics. STAYS as orchestration glue.
- All graph node functions — orchestration. STAYS.
- All `@tool` functions — directly registered in `_build_graph()`. STAYS.

---

## 7. Import Architecture (after Phase 2)

```
agent_graph.py
  ├── source_content.py (pure text analysis)
  │     └── (no deps beyond stdlib)
  ├── command_safety.py (SSH cmd classification)
  │     └── (no deps beyond stdlib)
  ├── prompt_assembly.py (extended with constants)
  │     └── (no new deps)
  ├── context_compressor.py (extended with ratio utils, message helpers)
  │     ├── compression_redact.py
  │     └── model_metadata.py
  ├── model_metadata.py (extended with discovery helpers)
  └── (existing imports unchanged)
```

No circular imports introduced. `source_content.py` and `command_safety.py` have zero internal deps.

---

## 8. What Behavior Stays Unchanged

- Graph topology (nodes, edges, conditional routing)
- All tool signatures and behavior
- All compression decisions and thresholds
- All crisis recovery logic
- All planner/fast-path/routing logic
- All context budget calculations (numbers identical)
- All feature flags — existing defaults preserved
- All streaming/response mode behavior
- All agent system prompts

Extracted functions are pure — same inputs produce same outputs.

---

## 9. Verification Plan

After each phase:

```bash
# Import check — graph must load
cd langgraph-app
python -c "from agent_graph import make_graph; print('OK')"

# Full test suite
pytest -q tests/ -x

# Bridge smoke test (needs running services)
python scripts/alpharavis_setup.py bridge-smoke

# Check no circular imports
python -c "
import agent_graph
import source_content
import command_safety
print('All imports clean')
"
```

---

## 10. What's Still Too Risky / Unfinished

1. **`_build_graph()` body extraction**: The 1,160-line graph builder could be split into helper functions for each agent creation (`_build_research_agent`, `_build_general_agent`, etc.). This is a mechanical split, low risk, but big change. Recommended for a separate PR after Phase 2 is stable.

2. **`_ingest_large_paste_messages`**: 329 lines, deeply intertwined with store operations, classification, and vector backfill. Phase 3 wrapping first, then Phase 4 move.

3. **Model construction subsystem**: `_model()`, `_budget_guarded_agent_model()`, `_deepagents_responses_model()`, `_text_only_agent_model()` form a tightly coupled chain. Creating a `model_factory.py` would need careful interface design. Leave for a separate focused PR.

4. **Tool function extraction**: 50+ `@tool` functions (1,200+ lines) could technically move to separate modules (`tools/pixelle.py`, `tools/rag.py`, `tools/memory.py`, etc.) but the registration in `_build_graph()` uses direct function references. Moving them would require either: (a) import from new modules in `_build_graph()`, risking circular imports, or (b) passing tool functions through `make_graph()` parameters. Either approach changes the initialization signature. **Recommendation: leave tools in place for now.**

5. **Compression glue functions**: The 500 lines of compression helpers (`_drop_previous_compaction_messages`, `_hard_context_trim_update`, `_store_compression_archive`, etc.) are called from multiple graph nodes and modify graph state. They're graph orchestration, not pure logic. Extract only if needed.

---

## 11. Summary Statistics

| Metric | Before | After Phase 2 |
|---|---|---|
| agent_graph.py lines | 13,960 | ~13,430 (-530) |
| New modules created | 0 | 2 (`source_content.py`, `command_safety.py`) |
| Existing modules extended | 0 | 4 (`prompt_assembly.py`, `context_compressor.py`, `model_metadata.py`, `media_analysis.py`) |
| Functions extracted | 0 | ~25 |
| Pure functions extracted | 0 | ~20 (@250 lines of testable logic) |
| Graph nodes touched | 0 | 0 (orchestration unchanged) |
| Circular imports introduced | 0 | 0 |
| Feature flags changed | 0 | 0 |
| Tests added | 0 | ~10 (for extracted pure functions) |
