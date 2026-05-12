# AlphaRavis Open Tasks

This is the running backlog for features that are intentionally prepared but
not fully wired yet.

## Responses Streaming Follow-up

Status: local PR #35457-style patch applied; hybrid streaming mode passes.

Implemented:

- LangGraph container packages were updated to:

```text
langchain-openai==1.2.1
langchain==1.2.18
langchain-core==1.3.3
langgraph==1.1.10
deepagents==0.5.9
openai==2.36.0
litellm==1.83.0
```

- Patched DeepAgents Responses hybrid mode works with:

```text
ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING=true
ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=tool_calling
```

- Live test after the update:
  - `/v1/responses` Fast Path returned 200.
  - `/v1/responses` Agent Path returned 200 in about 48 seconds.
  - External `/v1/responses` SSE streaming returned chunks.
- Live test after the local patch:
  - Direct `ChatOpenAI(use_responses_api=True, streaming=True,
    disable_streaming="tool_calling")` with a bound tool returned
    `DIRECT_TOOL_STREAM_TEST_OK`.
  - Bridge `/v1/responses` Agent Path streaming returned
    `PATCHED_AGENT_STREAM_OK`.
- Focused full-streaming probe after the env-gated tool-stream patch:
  - raw `/v1/responses` SSE included function-call events
  - LangChain no-tool Responses streaming passed
  - LangChain `create_react_agent` streaming executed `marker_tool` exactly once
  - `invalid_tool_calls=0`

Still needed:

- Track the upstream LangChain issue and remove the local patch when
  `langchain-openai` includes the fix.
- Retest the LiteLLM proxy after its Docker image reports a newer package than
  `litellm==1.82.6`.
- Keep full streaming
  `ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=false` experimental as a
  stack default until Bridge-level repeated smoke tests pass and the local
  provider consistently handles Responses tool-choice behavior.
- Consider a two-phase final-answer stream:
  - tool-capable worker calls stay hybrid/non-streaming for reliable tool JSON
  - after tool execution, run a final answer call without tools and stream that
    text to the client
  - this requires orchestration changes because current DeepAgents workers keep
    tools bound on final answer turns too
- Profile and reduce normal agent-path latency. Live Docker measurements on
  2026-05-12 showed:
  - direct LiteLLM chat call: about `1.8 s`
  - Bridge fast path: about `2.3 s`
  - Bridge `/v1/responses` agent path with `kein fast path`: about `54 s`
  - the same run spent about `29.8 s` in the planner call, then continued into
    the worker/model stage
  - queued runs can add large apparent latency because the current local
    LangGraph runtime reported only one active background worker
  Follow-up: measure streaming first-token latency separately, then decide
  whether to shorten/bypass planner work for trivial prompts, increase local
  worker concurrency where safe, or route simple UI greetings through the fast
  path earlier.

## Custom Model / Power Management

Status: prepared, default off. Owner tool file exists and safe owner tools are wired.

Implemented:

- `langgraph-app/model_management.py` exists as the custom hardware layer.
- `.env(exaple)` contains all switches and defaults them off.
- `make model-management` can write the relevant `.env` switches.
- `power_management_agent` is only registered when:

```text
ALPHARAVIS_ENABLE_MODEL_MANAGEMENT=true
ALPHARAVIS_ENABLE_ADVANCED_MODEL_MANAGEMENT=true
```
- Owner-only tools from `langgraph-app/owner_power_tools.py` are available when:

```text
ALPHARAVIS_ENABLE_OWNER_POWER_TOOLS=true
```
- Safe owner actions are wired:
  - check llama server
  - start/restart llama server
  - read llama logs
  - check/wake ComfyUI
  - start all model services
  - read Pixelle logs when Docker is reachable
- Protected owner actions are wired through human approval:
  - shutdown llama server
  - shutdown ComfyUI server
- `power_management_agent` uses `ALPHARAVIS_POWER_MANAGER_MODEL` when advanced
  model management is enabled.

Still needed:

- Provide the curated external action endpoint:

```text
ALPHARAVIS_MODEL_MGMT_ACTION_URL=
ALPHARAVIS_MODEL_MGMT_API_KEY=
ALPHARAVIS_MODEL_MGMT_ALLOW_ACTIONS=true
```

- Populate remaining safe real actions:
  - `check_ollama_models`
  - `load_embedding_model`
  - `unload_ollama_model`
  - `run_embedding_jobs`
- Populate remaining HITL/destructive actions if you really want them:
  - `reboot_server`
  - `kill_process`
  - `delete_files`
- Restrict `delete_files` to explicitly allowed temp/work folders.
- Decide whether `wake_pc` should stay as direct Wake-on-LAN or also route
  through the curated action endpoint.

## Crisis Manager

Status: minimal preflight/recovery agent implemented, default off.

Implemented:

- Enabled only by:

```text
ALPHARAVIS_ENABLE_MODEL_MANAGEMENT=true
ALPHARAVIS_ENABLE_ADVANCED_MODEL_MANAGEMENT=true
ALPHARAVIS_ENABLE_OWNER_POWER_TOOLS=true
ALPHARAVIS_ENABLE_CRISIS_MANAGER=true
ALPHARAVIS_CRISIS_MANAGER_MODEL=openai/edge-gemma
ALPHARAVIS_POWER_MANAGER_MODEL=openai/edge-gemma
```
- Use the small Ollama model only as a crisis moderator, not for normal complex
  work.
- Preflight check runs before the normal planner path.
- Automatically runs non-destructive checks and safe starts through owner tools:
  - status checks
  - logs/read-only probes
  - `start_llama_server`
  - `restart_llama_server`
- Sends the user a short `Crisis-Notice` while recovery is happening.
- After the recovery attempt, routes back to the normal planner path so the
  original user request can continue.
- Destructive shutdown tools are not given to the crisis agent.

Still needed:

- Trigger crisis recovery on mid-run main-model failures such as timeout, 502,
  connection errors, or LiteLLM backend generation health failure.
- Add a post-recovery readiness gate before continuing to the planner.
- Add full hard caps from the ENV placeholders:
  - max recovery attempts
  - max wall-clock time
  - no recursive crisis loops
- Add read-only Ollama/LiteLLM checks:
  - `check_ollama_models`
  - LiteLLM generation smoke status

ENV placeholders already exist:

```text
ALPHARAVIS_CRISIS_AUTO_ACTIONS=check_llama_server|check_ollama_models|check_comfyui|start_llama_server|restart_llama_server|wake_pc
ALPHARAVIS_CRISIS_HITL_ACTIONS=shutdown_server|reboot_server|kill_process|delete_files
ALPHARAVIS_CRISIS_MAX_ATTEMPTS=1
ALPHARAVIS_CRISIS_TIMEOUT_SECONDS=120
```

## Embedding Queue And pgvector

Status: pgvector retrieval chunks, catalog rows, durable queueing, a manual
model-lifecycle queue runner, optional scheduler, and bounded Store-index
backfill queueing are implemented.

Implemented:

- `ALPHARAVIS_PGVECTOR_INDEX_MODE=queue` stores new indexing work in Postgres.
- `alpharavis_embedding_jobs` keeps pending/failed/running/done queue state.
- `inspect_model_management_status` shows queue status.
- `run_embedding_memory_jobs` loads the configured Ollama embedding model when
  allowed and drains queued pgvector jobs.
- The runner may work while big-boss is active, so the small Ollama node can be
  used for embeddings without taking over complex chat.
- `ALPHARAVIS_ENABLE_EMBEDDING_SCHEDULER=true` drains the queue periodically.
- `queue_vector_memory_backfill` queues bounded old Store records by query.
- `ALPHARAVIS_ENABLE_VECTOR_BACKFILL_DAEMON=true` can repeat that bounded
  backfill search, but only when `ALPHARAVIS_VECTOR_BACKFILL_QUERY` is set.

Still needed:

- More precise convenience backfill commands:
  - index this exact thread without a search query
  - index last N artifacts by timestamp
  - index selected document/source keys from the external RAG backend
- Active-job awareness for Pixelle/MCP jobs beyond the current big-LLM/Ollama
  model probes.

## Media / Vision Memory

Status: safe media metadata handling, media-gallery service, a separate vision
pgvector table, explicit video-analysis preparation, and media-index status
inspection are implemented. Full caption/OCR/transcription remains
provider/pipeline work.

Implemented:

- Bridge strips raw media blocks from chat context by default, preserves
  metadata markers, and automatically mirrors incoming video blocks into
  `media-gallery` when `BRIDGE_MEDIA_GALLERY_AUTO_REGISTER_VIDEOS=true`.
- The AlphaRavis-facing marker is rewritten to the stable gallery URL after a
  successful mirror. LibreChat's original visible attachment/file record stays
  untouched in this phase.
- `media-gallery` can register/download image, video, audio, or document URLs
  and exposes `/gallery`. Video mirroring accepts HTTP(S) URLs and inline
  `data:` video payloads while omitting the inline payload from Mongo metadata.
- Pixelle job results are scanned for media URLs and registered when present.
- `register_media_asset`, `semantic_media_search`, and `plan_media_analysis`
  tools exist.
- Vision/media embeddings use `alpharavis_media_vectors`, separate from the text
  table, so vector dimensions do not collide.
- `prepare_media_for_model` decides `register_only`, `pass_through`, `analyze`,
  or `index`; it only downloads video for explicit analyze/index modes and when
  `ALPHARAVIS_VIDEO_ANALYSIS_ENABLED=true`.
- `inspect_media_index_status` lets agents check which media/frame records are
  already present in `alpharavis_media_vectors`, and also reports matching
  pending/running/failed/done media-analysis queue records.
- `inspect_embedding_queue_status` lets agents answer general queue questions
  for text, archive, and media-analysis jobs in `alpharavis_embedding_jobs`.
- `prepare_media_for_model(mode="index")` queues video analysis/indexing as a
  durable `media_analysis` job in the same embedding queue used by text,
  archives, artifacts, memories, and session turns.
- Media-gallery registration now separates media assets from chat/tool
  appearances through the Mongo `references` collection; repeated mentions of
  one video should create references, not duplicate full embeddings.
- Media indexing dedupes by media source key, media vision model-card id,
  `ALPHARAVIS_MEDIA_INDEX_VERSION`, and the video chunking-config hash.
- Auto-index policy is ENV-controlled for user uploads, Pixelle MCP / ComfyUI
  outputs, and link references:

```text
ALPHARAVIS_MEDIA_AUTO_INDEX_ENABLED=true
ALPHARAVIS_MEDIA_AUTO_INDEX_USER_UPLOADS=true
ALPHARAVIS_MEDIA_AUTO_INDEX_PIXELLE_MCP_OUTPUTS=false
ALPHARAVIS_MEDIA_AUTO_INDEX_LINK_REFERENCES=false
```
- Video frame extraction uses `ffprobe`/`ffmpeg`, bounded FPS, bounded
  `ALPHARAVIS_VIDEO_ANALYSIS_MAX_FRAMES`, timestamped frame manifests, and the
  model-card defaults in `langgraph-app/model_cards.json`.
- `make video-analysis ENABLED=true FPS=1 MAX_FRAMES=100` can write the core
  analysis switches into `.env`.

Still needed:

- Connect a real vision embedding endpoint and enable:

```text
ALPHARAVIS_ENABLE_VISION_VECTOR_MEMORY=true
```

- Build the Meet/media-gallery integration as the operator-facing video rack.
  The current `media-gallery` already has its own port and Mongo-backed asset
  registration, but the UI and analysis pipeline are still basic.

  Current implementation checkpoint from 2026-05-12:

  - `BRIDGE_ALLOW_RAW_MEDIA_CONTEXT=false` and
    `BRIDGE_MEDIA_CONTEXT_MODE=metadata` are already the default path in
    `bridge_server.py`, `.env(exaple)`, and Docker Compose.
  - The Bridge converts OpenAI/LibreChat media content parts into metadata
    markers instead of forwarding raw media blocks to LangGraph.
  - `context_retrieval_agent` already has `semantic_media_search` for indexed
    media references and `inspect_media_index_status` for processed/indexed
    media plus pending queue status.
  - The shared `alpharavis_embedding_jobs` queue now carries both text/archive
    embedding jobs and video `media_analysis` jobs. `run_embedding_memory_jobs`
    drains both kinds through the existing model-management embedding window.
  - `/assets/resolve` can map a copied gallery/source URL back to the Mongo
    asset and its recorded references.
  - Incoming LibreChat/Responses video blocks are now copied into
    `media-data` through the Bridge and media-gallery before LangGraph context
    is built; the LLM marker points at the media-gallery URL.
  - `plan_media_analysis` remains explanatory. The real bounded preparation
    path is `prepare_media_for_model`.

  Goal:

  - Use the Meet/media-gallery service as the place where all videos from chat,
    uploads, Pixelle MCP outputs, and future Meet-server flows become visible.
  - Keep MongoDB/media-gallery metadata as the source of truth for original
    uploads and processed outputs.
  - Make every asset usable by link in later chats, either as a pass-through
    URL for Pixelle or as an analysis target that AlphaRavis downloads and
    preprocesses.
  - Preserve the relation between a user-supplied source video, the chat turn
    or Pixelle request that used it, and the processed video returned by Pixelle.

  Research note from 2026-05-12:

  - The active local target mentioned by the operator is assumed to be
    `Qwen/Qwen3.6-35B-A3B` unless the runtime model id says otherwise.
  - The official Qwen3.6 model card says it is a causal language model with a
    vision encoder and a native context length of 262,144 tokens, extendable up
    to 1,010,000 tokens with YaRN.
  - Its Hugging Face `preprocessor_config.json` has image
    `longest_edge=16777216`, `shortest_edge=65536`, `patch_size=16`,
    `temporal_patch_size=2`, and `merge_size=2`.
  - Its `video_preprocessor_config.json` has video
    `longest_edge=25165824`, `shortest_edge=4096`, `patch_size=16`,
    `temporal_patch_size=2`, and `merge_size=2`.
  - The model card's vLLM video example says default video sampling is
    `fps=2`, configurable through `mm_processor_kwargs`. For AlphaRavis, keep
    the operator default stricter at `1 fps max` because the requested local
    behavior prioritizes predictable load over maximum frame recall.
  - The same model card recommends increasing the video preprocessor
    `longest_edge` to `469762048` for hour-scale long-video workloads; keep
    this as an optional advanced model-card value, not the default.

  Data model plan:

  - Extend media asset records with derivation fields:
    - `asset_kind`: `original`, `processed`, `reference`, or `unknown`
    - `origin`: `librechat_upload`, `chat_url`, `pixelle_output`,
      `meet_server`, or `manual_register`
    - `parent_asset_id`
    - `root_asset_id`
    - `derivation_group_id`
    - `source_message_id`
    - `result_message_id`
    - `tool_call_id` or Pixelle `job_id`
    - `processing_provider`, for example `pixelle`
    - `processing_prompt` or compact prompt hash
    - `public_url`
    - `download_url`
    - `local_path`
    - `thumbnail_path` or `preview_path`
    - `duration_seconds`, `width`, `height`, `fps`, and `bytes`
  - Keep original videos distinct from processed videos, but group them under
    one derivation tree so "All" can show source and result together.
  - Add a stable lookup from LibreChat/Mongo upload ids to media-gallery assets
    if the file exists only in LibreChat's Mongo/filesystem layer.
  - Add idempotent registration keyed by source URL, upload id, local path, or
    source hash so repeated registration does not create duplicate gallery
    cards.
  - Preserve existing media-server filter stages. First audit the current
    filters, then insert derivation/grouping logic after safe metadata
    extraction and before gallery rendering/download.

  UI plan:

  - Replace the current simple `/gallery` HTML with a real work UI, still
    served by the media-gallery/Meet service port unless a separate frontend is
    justified later.
  - Add tabs or segmented controls:
    - `All`
    - `Original`
    - `Processed`
  - In `All`, group original input videos and Pixelle/processed result videos
    together by `derivation_group_id` or `root_asset_id`.
  - In `Original`, show only uploaded/input/reference videos.
  - In `Processed`, show only Pixelle/generated/processed outputs.
  - Use dense video cards with:
    - thumbnail or lightweight preview
    - title/source label
    - original/processed badge
    - thread/chat marker
    - Pixelle job/result marker when present
    - duration/resolution/filesize metadata
    - link/copy action in a small bottom-right menu
    - open/download actions
  - Do not autoplay every full video by default. Use `preload="metadata"` plus
    posters/thumbnails first. Add optional hover preview or low-rate muted
    preview clips only after performance is measured.
  - Generate thumbnails and tiny preview clips as background media jobs. The UI
    must remain useful when thumbnails are pending.
  - Add filters for media type, thread, source, date, and processing provider
    once the basic Original/Processed/All flow works.
  - Verify the UI on desktop and mobile with real videos before marking done.

  Link and ingestion plan:

  - Every gallery card needs a stable public/media URL that can be copied and
    pasted into another chat.
  - Default behavior must remain metadata-only: pasted or uploaded videos are
    not pulled into model context unless the user explicitly asks to analyze,
    inspect, describe, summarize, transcribe, compare, or otherwise understand
    the media content.
  - The copied link must be acceptable as:
    - a normal media reference for AlphaRavis
    - a pass-through input URL for Pixelle when the user asks to create a new
      video from it
    - a downloadable analysis target when the user asks to inspect/analyze it
  - Add explicit tool behavior:
    - For "send this to Pixelle", pass the URL through and avoid downloading
      unless Pixelle requires a local file.
    - For "analyze this video", download or resolve the asset into the media
      analysis cache, then preprocess frames for the target model.
    - For "copy link", return only the stable media URL, not internal local
      paths or Mongo ids.
  - Prefer a dedicated `analyze_media_asset` / `prepare_media_for_model` tool
    over a new agent. The model can decide when to call that tool from user
    intent; the tool should enforce the hard rules, caps, MIME checks, and
    frame sampling. The context retrieval agent should retrieve references and
    timecoded indexed hits, not perform heavy video preprocessing itself.
  - Add an intent/decision helper inside that tool:
    - `pass_through`: keep URL only for Pixelle or another downstream service.
    - `register_only`: save metadata/gallery entry, no download.
    - `analyze`: download/resolve locally, sample frames, and build bounded
      model-ready content.
    - `index`: enqueue media analysis/indexing for future retrieval without
      answering from raw media immediately.
    - The LLM may choose the mode, but the default fallback must be
      `register_only`, not `analyze`.
  - Add safety checks around download URLs:
    - allowed schemes
    - size limit
    - media MIME validation
    - path confinement under `ALPHARAVIS_MEDIA_ROOT`
    - optional signed/internal token later if the media service is exposed
      outside localhost.

  Video analysis rack plan:

  - Add a separate analysis pipeline instead of pushing raw videos into LLM
    context.
  - Resolve the model card for the active model id before preprocessing:
    - static built-in entry for `Qwen/Qwen3.6-35B-A3B`
    - optional JSON/YAML override for local aliases such as `big-boss`
    - fallback defaults when the runtime model has no vision card
  - Model-card fields should include:
    - `supports_images`
    - `supports_video`
    - `native_context_tokens`
    - `image_longest_edge`
    - `image_shortest_edge`
    - `video_longest_edge`
    - `video_shortest_edge`
    - `patch_size`
    - `temporal_patch_size`
    - `merge_size`
    - `preferred_video_fps`
    - `max_video_fps`
    - `max_frames`
    - provider-specific payload knobs such as
      `mm_processor_kwargs.fps` / `do_sample_frames`
  - Add ENV defaults:

```text
ALPHARAVIS_VIDEO_ANALYSIS_ENABLED=true
ALPHARAVIS_VIDEO_ANALYSIS_FPS=1
ALPHARAVIS_VIDEO_ANALYSIS_MAX_FPS=1
ALPHARAVIS_VIDEO_ANALYSIS_MAX_FRAMES=100
ALPHARAVIS_VIDEO_ANALYSIS_MAX_DOWNLOAD_BYTES=2147483648
ALPHARAVIS_VIDEO_ANALYSIS_MODEL_CARD_PATH=/workspace/langgraph-app/model_cards.json
ALPHARAVIS_VIDEO_ANALYSIS_PUBLIC_MEDIA_ROOT=/workspace/media-data
ALPHARAVIS_VIDEO_ANALYSIS_CACHE_ROOT=/workspace/media-data/analysis-cache
ALPHARAVIS_VIDEO_ANALYSIS_INCLUDE_AUDIO=false
ALPHARAVIS_VIDEO_ANALYSIS_TRANSCRIBE_AUDIO=false
```

  - Add Makefile/setup support:
    - `make media-vision` should be able to write the video-analysis switches.
    - Add a direct target such as `make video-analysis FPS=1 MAX_FRAMES=100`
      if that is simpler for repeat use.
    - `make status` should show whether video analysis is enabled, FPS, frame
      cap, cache root, and model-card path.
  - Sampling rule:
    - Never sample more than `ALPHARAVIS_VIDEO_ANALYSIS_MAX_FPS`.
    - Default to at most one frame per second.
    - For videos whose duration in seconds is less than or equal to the frame
      cap, sample one frame per second.
    - For longer videos, select at most `MAX_FRAMES` frames evenly across the
      full duration, so a one-hour video with `MAX_FRAMES=100` stays near 100
      frames instead of trying to keep one frame per second.
    - Preserve timestamps for every extracted frame.
  - Preprocessing rule:
    - Scale frames according to the active model card's video/image limits.
    - Keep aspect ratio.
    - Store extracted frames and metadata in the analysis cache.
    - Mark the model payload as video, not as unrelated still images, whenever
      the provider/server supports video content parts.
    - If the llama.cpp/OpenAI-compatible route cannot accept native video
      blocks, send a bounded sequence of timestamped image frames with a clear
      system/user message that they are sampled frames from one video.
  - Retrieval/RAG behavior:
    - Store analysis metadata, frame timestamps, captions, optional transcript,
      and embeddings under `alpharavis_media_vectors`.
    - Keep MongoDB/media-gallery as the asset source of truth and pgvector as
      the searchable index.
    - Allow later prompts such as "analyze this video" or "use this link as
      input for a new video" to resolve the asset by URL or asset id.
    - Use `inspect_embedding_queue_status` when the user asks how much indexing
      work is still pending.
    - Use `inspect_media_index_status` to distinguish "not indexed yet",
      "queued", "running", "failed", and "indexed".

  Implementation phases:

  1. Audit current media-gallery and Meet-server routes, existing media filters,
     LibreChat upload metadata, Pixelle result registration, and Mongo asset
     records.
  2. Extend the Mongo asset schema and registration API for original/processed
     grouping without breaking current `/assets/register` callers. Implemented
     for optional derivation fields, `asset_kind`, origin, parent/root asset,
     and derivation group fields.
  3. Add URL copy/download/open affordances and stable public links.
     Implemented for copy/open links in `/gallery`; signed links remain future
     work if the gallery is exposed outside localhost.
  4. Build the improved gallery UI with Original/Processed/All grouping.
     Partially implemented in the media server's `/gallery` route with
     `view=all|original|processed` tabs and derivation-group sections.
  5. Add thumbnail/preview generation and avoid heavy autoplay.
  6. Add model-card config and Qwen3.6 defaults. Implemented.
  7. Add video download, ffprobe/ffmpeg keyframe extraction, adaptive frame
     sampling, scaling, and analysis-cache storage. Implemented for explicit
     video analysis, without captions/transcription.
  8. Add the dedicated media-analysis preparation tool and wire explicit
     decisions for pass-through-to-Pixelle vs download-for-analysis.
     Implemented.
  9. Add optional frame captions, audio transcription, and media-vector indexing.
     Frame-level vision indexing is implemented through the shared durable
     embedding queue when vision pgvector is enabled; captioning and
     transcription remain future work.
  10. Add Makefile/setup/status controls and smoke tests. Partially
     implemented: `make video-analysis`, setup/status output, helper tests, and
     bridge media tests exist; live Docker/UI smoke remains needed.
  11. Mirror LibreChat-origin video input into `media-data` automatically
      before LangGraph sees it. Implemented for Bridge-facing HTTP(S) and
      inline `data:` video blocks; rewriting the visible LibreChat message card
      itself remains intentionally out of scope for this phase.
  12. Run a real Docker/LibreChat upload smoke:
      - send a chat video through LibreChat
      - verify the Bridge registers it in `media-gallery`
      - verify the stable gallery URL appears in AlphaRavis-facing context
      - verify the gallery card is created and the stored bytes resolve through
        the media service URL
  13. Investigate an explicit LibreChat-visible rewrite phase:
      - either rewrite the persisted LibreChat message/file metadata to the
        Media Gallery URL
      - or keep LibreChat's native attachment UI untouched and add a clearly
        linked gallery reference beside it
      - document the MongoDB/file-system mutation boundary before enabling this
        path, because this changes what the operator sees in historical chats
  14. Promote lightweight motion previews from "possible" to a measured UI
      follow-up:
      - poster first
      - muted hover/focus preview or tiny generated preview clip
      - no full autoplay grid until desktop/mobile performance is measured

  Acceptance:

  - A LibreChat-uploaded original video appears in the gallery as `Original`.
  - A Pixelle-generated or processed result appears as `Processed`.
  - `All` shows source and processed result grouped together when one was
    derived from the other.
  - Copying a card link and pasting it into a new chat gives AlphaRavis enough
    metadata to pass it to Pixelle or download it for analysis.
  - A prompt that only says to use a video as Pixelle input does not download or
    sample the video in AlphaRavis.
  - A prompt that explicitly asks to analyze the video calls the media-analysis
    preparation tool and stays within FPS/frame/download caps.
  - The UI can show many videos without starting all full video streams at
    once.
  - A single card can expose a deliberate moving preview without turning the
    whole gallery into an autoplay wall.
  - `ALPHARAVIS_VIDEO_ANALYSIS_MAX_FRAMES=100` keeps a one-hour video bounded
    near 100 sampled frames, while a ten-second video samples up to ten frames
    at the default one frame per second.
  - The model payload preserves video semantics where the active provider
    supports it; otherwise AlphaRavis states that it is sending sampled
    timestamped frames from a video.
  - The current media-server filter stages remain intact and covered by smoke
    tests.

- Build image analysis:
  - captioning
  - OCR
  - explicit user-triggered vision analysis
- Add exact mapping from LibreChat upload ids to gallery assets if LibreChat
  stores the file only inside its Mongo/filesystem layer.

## OpenWebUI

Status: optional Compose profile exists and points to the AlphaRavis Bridge.

Still needed:

- Start and verify:

```text
docker compose --profile openwebui up -d openwebui
make openwebui-smoke
```

- In OpenWebUI UI, set capable AlphaRavis models to Native function calling.
- Configure SearXNG or another web-search backend before enabling web search.
- Decide whether passthrough should stay enabled in your deployment:

```text
OPENWEBUI_ENABLE_OPENAI_API_PASSTHROUGH=true
```

## Lazy Tool Loading

Status: category registry exists and agents can inspect it with
`describe_optional_tool_registry(category=...)`.

Still needed:

- True per-run dynamic internal tool binding/unbinding.
- Cache concrete MCP tool schemas by category and only expose loaded subsets.
- Store loaded tool-set metadata in `run_profile`.

Clarification:

- `ALPHARAVIS_PGVECTOR_INDEX_MODE=background` still exists for best-effort
  async indexing, but the default example now uses `queue`.
- The model lifecycle runner can load the embedding model and drain jobs. It
  does not unload the small chat/crisis model by default. If that model is
  already loaded, the runner skips unless `ALPHARAVIS_EMBEDDING_UNLOAD_CHAT_MODEL=true`.

## Pixelle / ComfyUI Power Flow

Status: preflight hook exists, default off.

Implemented:

- Pixelle can run with durable `@task` monitoring or async job id polling.
- ComfyUI preflight can warn or block before Pixelle starts.
- The generic model-management preflight can request `wake_pc` through the
  curated action endpoint when that endpoint is configured.
- Owner power tools include a direct ComfyUI wake helper for manual/power-agent
  use.

Still needed:

- Set a real ComfyUI health URL:

```text
ALPHARAVIS_COMFY_HEALTH_URL=http://<comfy-ip>:8188/system_stats
```

- Decide whether Pixelle should warn-and-continue or block when ComfyUI is
  offline:

```text
ALPHARAVIS_PIXELLE_BLOCK_IF_COMFY_OFFLINE=false
```

- Decide whether Pixelle preflight should call the owner ComfyUI wake helper
  directly or stay routed through the curated action endpoint.

## Bridge

Status: Chat Completions remains compatible; Responses API wrapper,
Responses-style streaming events, direct Responses calls, and DeepAgents
Responses model binding exist.

Implemented:

- `/v1/chat/completions`
- `/v1/responses`
- `/v1/responses/{response_id}`
- `/v1/responses/{response_id}/input_items`
- `/v1/responses/{response_id}/cancel`
- `/v1/responses/input_tokens`
- explicit unsupported response for `/v1/responses/compact`
- OpenAPI schema version `3.1.0`
- `response.output_item.*`, `response.output_text.*`, and optional
  reasoning stream events; LibreChat compatibility still needs the
  `response.reasoning.*` normalization listed below
- local `previous_response_id` continuation through `BRIDGE_RESPONSES_STORE`
- explicit errors for unsupported hosted Responses features instead of silent
  fake support
- bridge-level hard request cutoff before LangGraph is called
- direct no-tool LangGraph calls can use `/v1/responses` with:

```text
ALPHARAVIS_LLM_API_MODE=responses
```
- DeepAgents workers can bind tools through Responses with:

```text
ALPHARAVIS_DEEPAGENTS_API_MODE=responses
```

Current live-test status:
- Done: direct no-tool LangGraph calls use Responses successfully for
  fast-path/planner style calls.
- Done: DeepAgents can use Responses successfully with the patched LangChain
  hybrid streaming mode:

```text
ALPHARAVIS_DEEPAGENTS_RESPONSES_STREAMING=true
ALPHARAVIS_DEEPAGENTS_RESPONSES_DISABLE_STREAMING=tool_calling
```

- Not stable yet: full internal Responses streaming for tool-bound DeepAgents
  calls as the default stack mode. The focused probe passes with
  `ALPHARAVIS_EXPERIMENTAL_BUFFER_TOOL_STREAMING=true`, but earlier full
  streaming failed with `item['content'] is empty` and Bridge-level soak testing
  is still needed.

Still needed:

- LibreChat Responses/reasoning/tool UI pass:
  - Status: implemented and live-smoked through the recreated `api-bridge`
    and `librechat` containers on 2026-05-11.
  - Added two explicit LibreChat model specs in `librechat.yaml` and updated
    the config to `version: 1.3.9` with `interface.presets: false` so
    `modelSpecs` is not competing with default presets:
    - `AlphaRavis Chat` using the existing custom endpoint through
      `/v1/chat/completions`.
    - `AlphaRavis Responses` using the same custom endpoint with
      `useResponsesApi: true`, `reasoning_summary: "detailed"` or `"auto"`,
      `reasoning_effort`, and `verbosity`.
  - Keep the Chat Completions model spec as the legacy/stable path.
  - Make the Responses model spec the path for LibreChat's reasoning bubble,
    tool execution timeline, and agent progress visibility.
  - Updated `langgraph-app/bridge_server.py` Responses streaming from the
    old `response.reasoning_text.delta/done` shape to LibreChat/Open
    Responses compatible events:
    - `response.output_item.added` for a `type: "reasoning"` item
    - `response.content_part.added` with `part.type: "reasoning_text"`
    - `response.reasoning.delta`
    - `response.reasoning.done`
    - `response.content_part.done`
    - `response.output_item.done`
  - Added `logprobs: []` to every `response.output_text.delta` and
    `response.output_text.done` event, matching LibreChat v0.8.5 validation.
  - Included the final reasoning item in the completed Response object's
    `output` array when reasoning text or summaries were emitted.
  - Preserved the old Chat Completions reasoning path by continuing to emit
    `delta.reasoning_content` when `BRIDGE_STREAM_REASONING_EVENTS=true`.
  - Do not promise raw OpenAI chain-of-thought. OpenAI-hosted reasoning models
    expose reasoning summaries, not raw reasoning tokens; full visible thinking
    is only possible when the selected local/OpenAI-compatible provider emits
    visible `reasoning_content`, `reasoning`, or `<think>` text.
  - Mapped LangGraph tool activity to Responses tool items when enabled:
    - tool-call start -> `function_call` output item
    - tool-call args -> `response.function_call_arguments.delta/done`
    - tool result -> `function_call_output` item
    - tool completion/failure -> matching `response.output_item.done`
  - Reuse the proven tool extraction patterns from
    `langgraph-app/alpharavis_acp_adapter.py` so tool names, call IDs, args,
    status, file locations, and output snippets stay consistent across AionUI
    and LibreChat.
  - Emit agent/node progress separately from final assistant text. Candidate
    sources:
    - LangGraph `updates` node names such as `general_assistant`,
      `debugger_agent`, `hermes_coding_agent`, `context_retrieval_agent`, and
      `power_management_agent`
    - LangChain `on_tool_start`, `on_tool_end`, and `on_tool_error` events
    - DeepAgents tool call messages and tool result messages
  - Added focused tests in `tests/test_bridge_responses.py`:
    - no `response.reasoning_text.*` events in Responses streams
    - `response.reasoning.delta/done` events contain `sequence_number`,
      `item_id`, `output_index`, `content_index`, and text/delta fields
    - output text delta/done events include `logprobs: []`
    - completed Response output can contain both a `reasoning` item and an
      assistant `message` item
    - tool start/result events become valid `function_call` and
      `function_call_output` items
    - `/v1/chat/completions` streaming still emits normal text and optional
      `reasoning_content`
  - Live smoke checks completed:
    - `POST /v1/responses stream=true` emitted `response.reasoning.delta/done`
      plus `response.output_text.delta/done` with `logprobs: []`.
    - Agent-path `POST /v1/responses stream=true` emitted LangGraph node
      activity as reasoning deltas.
    - Tool-path `POST /v1/responses stream=true` emitted `function_call`,
      `response.function_call_arguments.delta/done`, and
      `function_call_output` items.
    - `POST /v1/chat/completions stream=true` still works as the fallback path.
  - Approval UX status:
    - OpenAI Responses supports MCP approval request/response items for remote
      MCP tools, but LibreChat's custom endpoint path does not expose an
      AlphaRavis-native click-to-approve permission callback.
    - `api-bridge` therefore keeps the chat-text approval path for LibreChat:
      `approve`, `reject`, `replace: <safer command>`, `approve always`, and
      `immer erlauben`.
    - `approve always` / `immer erlauben` stores an exact scope/target/command
      allow entry for the current LibreChat thread only, in bridge process
      memory. It is cleared by `api-bridge` restart and is not global.
  - Still verify visually in the LibreChat browser UI that `AlphaRavis
    Responses` renders reasoning/tool activity in the intended panes.
- llama.cpp/local-model visible thinking follow-up:
  - User backend is llama.cpp/local models behind the OpenAI-compatible stack,
    not OpenAI-hosted reasoning models. Do not assume OpenAI raw chain-of-thought
    restrictions apply to the local backend; instead preserve whatever visible
    thinking the local provider actually emits.
  - Recommended runtime shape is valid and should remain supported:
    `LibreChat -> api-bridge /v1/responses -> LangGraph -> llama.cpp
    /v1/chat/completions`. The outer LibreChat-facing Bridge can speak
    Responses even when the internal LangGraph-to-llama.cpp model call uses
    Chat Completions. The outer Responses translation is enough for LibreChat's
    reasoning bubble, tool timeline, and LangGraph activity UI.
  - Current code status:
    - `langgraph-app/bridge_server.py::_message_reasoning_content` already
      extracts visible reasoning from `reasoning_content`, `reasoning`,
      `additional_kwargs.reasoning_content`, and list content blocks with
      `type: "thinking"` or `type: "reasoning"`.
    - `langgraph-app/bridge_server.py::_message_content` already skips list
      content blocks with `type: "thinking"` or `type: "reasoning"`.
    - Added `_VisibleThinkingSplitter` in `langgraph-app/bridge_server.py` for
      normal string content containing visible local-model thinking markers.
      It supports `<think>...</think>` and `<thinking>...</thinking>`, handles
      split marker boundaries across chunks, routes inside-thinking text to
      reasoning output, routes outside text to assistant output, and suppresses
      the marker text itself.
    - The splitter is wired into both external streaming paths:
      `_stream_responses` emits extracted thinking through
      `response.reasoning.delta`, and `_stream_chat_events` emits extracted
      thinking through the configured Chat Completions reasoning delta field.
    - The fallback state-read path used when no token was streamed also splits
      final `_last_ai_content(...)` text so stored `<think>` blocks do not leak
      into the visible final answer.
    - Explicit provider reasoning fields still win. If a part already exposes
      `reasoning_content` or `reasoning`, string `<think>` blocks in that part
      are stripped from visible output but not duplicated into reasoning.
    - `StreamingInternalContextScrubber` remains after the split: answer text
      goes through the content scrubber and reasoning text goes through the
      reasoning scrubber.
  - Added focused tests in `tests/test_bridge_responses.py`:
    - Responses stream: `<think>plan</think>Answer` emits
      `response.reasoning.delta` containing `plan` and `response.output_text`
      containing only `Answer`.
    - Responses stream with split markers across chunks routes thinking
      correctly and does not leak `<think>` or `</think>` into
      `response.output_text.done` or the completed Response object's assistant
      message.
    - Chat Completions stream emits `delta.reasoning_content` for thinking and
      normal `delta.content` for the answer.
    - Explicit provider reasoning fields still work and are not double-counted
      when string `<think>` blocks are also present.
  - Live smoke after patch:
    - Use `AlphaRavis Responses` in LibreChat with the llama.cpp model and a
      prompt that reliably produces a visible `<think>` block.
    - Confirm the `<think>` body appears in LibreChat's reasoning area and the
      final assistant message does not include the raw markers.
    - Confirm tool calls still appear as `function_call`/`function_call_output`
      items during the same run.
- Re-test DeepAgents internal Responses token streaming after LiteLLM,
  `langchain-openai`, or llama.cpp upgrades. Keep the stable non-streaming
  DeepAgents default until repeated smoke tests pass.
- Updated `docs/ALPHARAVIS_RESPONSES_COMPATIBILITY.md` so the documented
  streaming event list matches the actual LibreChat-compatible event surface.
- Updated `langchain-bridge-repo` separately:
  - added Chat Completions `reasoning_content` passthrough for streamed chunks
    where LangChain exposes visible reasoning
  - added tests for reasoning passthrough
  - committed and pushed to `THEman6989/langchain-fastapi-chat-completion` as
    `3e647bf Preserve streamed reasoning content`
  - main repo submodule pointer now needs to be included with the main
    ai-stack commit when the surrounding Bridge/LibreChat changes are committed

## Parallel Agent Work

Status: planned, not active.

Still needed:

- Extend planner output with dependency groups:
  - independent tasks may run in parallel
  - dependent tasks stay sequential
- Add a bounded parallel execution node or worker pattern.
- Require each parallel branch to produce a `build_specialist_report`.
- Merge reports into one final handoff packet.
- Keep tool conflict rules so two agents do not edit or control the same target
  at the same time.

## DeepAgents / Hermes Skills

Status: skill cards exist.

Still needed:

- Use the DeepAgents and Hermes skill cards as templates when adding new agents.
- Extract more stable reusable skills from completed workflows.
- Keep promotion manual through the existing skill-library review flow.

## Hermes Deep-Code Followups

Status: reference analysis done against the local Hermes Agent checkout at
`C:\experi\ai\hermes-agent`. These are adoption candidates only; Hermes should
remain a reference and optional external agent, not a runtime dependency for
AlphaRavis.

Already adopted or partly adopted:

- `agent/context_compressor.py`, `agent/model_metadata.py`, and `agent/redact.py`
  inspired AlphaRavis's active compression hardening:
  - JSON-safe tool-call argument truncation
  - tool-output pruning and duplicate-output backreferences for summary prompts
  - anti-thrashing via `compression_stats`
  - failure cooldown
  - image/tool-argument-aware token estimation
  - percentage-based context-length triggers with local model context discovery
- `agent/context_engine.py` inspired the lightweight AlphaRavis
  `compression_stats` state. A full plugin-style context engine is not needed
  yet because AlphaRavis compression also writes archives and pgvector records.
- Hermes skill ideas are represented by reviewed repo skill cards under
  `ai-skills/`, plus the Store-backed skill-library candidate flow.

Implementation chunks:

Do not implement the whole Hermes followup list in one pass. Work in these
chunks so every step stays testable and can be disabled independently.

### Chunk 1: Context Hygiene First

Status: implemented.

Implemented files:

```text
langgraph-app/internal_context.py
langgraph-app/context_references.py
langgraph-app/bridge_server.py
langgraph-app/agent_graph.py
tests/test_context_hygiene.py
```

AlphaRavis-specific integration:

- The scrubber is attached at the bridge output layer, so LibreChat receives
  clean visible text while LangGraph Studio can still inspect internal state.
- Context references are resolved relative to the AI-stack repo root by default,
  not relative to an arbitrary process directory.
- Reference metadata is passed into LangGraph state as
  `bridge_context_references` and copied into `run_profile`.
- Sensitive paths such as `.env`, `.ssh`, `.aws`, `.kube`, and `.docker` are
  refused before file content is attached.
- URL reference fetching is controlled independently by
  `BRIDGE_CONTEXT_REFERENCES_FETCH_URLS`.

Goal:

- Prevent accidental context leaks into LibreChat.
- Add explicit context-reference handling without dumping uncontrolled files.

Scope:

- Implement the streaming internal-context scrubber from Hermes
  `agent/memory_manager.py`.
- Add tests where `<memory-context>` or archive/internal tags are split across
  multiple SSE deltas.
- Implement a minimal AlphaRavis context-reference preprocessor inspired by
  Hermes `agent/context_references.py`:
  - `@file`
  - `@folder`
  - `@diff`
  - `@staged`
  - `@git`
  - `@url`
- Add context-budget protection and `allowed_root` path checks.
- Record warnings/refusals in `run_profile`.

Acceptance:

- LibreChat never receives hidden memory/internal blocks as normal assistant
  text, even when streaming chunks split the tag boundaries.
- Explicit references attach bounded context blocks.
- Oversized references warn or refuse cleanly instead of silently filling the
  prompt.

### Chunk 2: Error Router And Recovery Decisions

Status: implemented as compact AlphaRavis-local classifier.

Implemented files:

```text
langgraph-app/error_classifier.py
langgraph-app/responses_client.py
langgraph-app/bridge_server.py
langgraph-app/agent_graph.py
tests/test_error_classifier.py
```

AlphaRavis-specific integration:

- Responses direct calls now raise `AlphaRavisAPIError` with a structured
  classification instead of plain `RuntimeError` for HTTP/transport failures.
- The bridge formats visible errors by class, for example `context_overflow`,
  `timeout`, `server_error`, `overloaded`, `rate_limit`, and `format_error`.
- When activity events are enabled, the bridge can emit a short classified
  error status event before the visible error message.
- Planner, fast-path fallback, crisis preflight, and crisis-manager failures
  record classification metadata in `run_profile`.
- The classifier is intentionally compact; it does not import Hermes or bring in
  cloud-provider billing/credential rotation behavior.

Goal:

- Stop treating every backend issue as the same failure.

Scope:

- Port a compact AlphaRavis-local classifier from Hermes
  `agent/error_classifier.py`.
- Wire it into:
  - `responses_client.py`
  - bridge non-streaming/streaming errors
  - graph crisis/preflight metadata
- Map decisions:
  - `context_overflow` -> compression/hard-cutoff message
  - timeout/502/overloaded/connection -> crisis-manager candidate
  - rate limit/server busy -> retry/backoff or visible status
  - format/unsupported parameter -> Responses/Chat fallback or parameter strip

Acceptance:

- `run_profile` shows the classified reason.
- Advanced model-management recovery can use the classification later.
- Normal users get a useful message instead of a generic backend crash.

### Chunk 3: Central File Safety

Status: implemented as shared AlphaRavis-local file safety guard.

Implemented files:

```text
langgraph-app/file_safety.py
langgraph-app/context_references.py
langgraph-app/agent_graph.py
langgraph-app/media_server.py
tests/test_file_safety.py
```

AlphaRavis-specific integration:

- `file_safety.py` centralizes read/list/write/delete checks for sensitive
  credential/config paths, internal caches, shell profiles, and OS/system paths.
- `BRIDGE_ENABLE_CONTEXT_REFERENCES` file/folder reads now call the central
  read/list guard instead of carrying separate safety rules.
- `read_alpha_ravis_architecture`, `read_repo_ai_skill`,
  `write_alpha_ravis_artifact`, and `read_alpha_ravis_artifact` now pass through
  the same guard.
- Media gallery downloads verify the target path before writing under
  `ALPHARAVIS_MEDIA_ROOT`.
- `ALPHARAVIS_WRITE_SAFE_ROOT` can optionally force AlphaRavis write/delete
  helpers under a single owner-approved root.

Goal:

- Future coding/file/power tools share one safety policy.

Scope:

- Add `langgraph-app/file_safety.py`, inspired by Hermes
  `agent/file_safety.py`.
- Protect sensitive paths:
  - `.ssh`
  - `.aws`
  - `.kube`
  - `.docker`
  - `.env`
  - shell profiles
  - credential files
  - OS/system paths
- Add optional:

```text
ALPHARAVIS_WRITE_SAFE_ROOT=
```

- Make owner/coding/Hermes delegation tools call this module before destructive
  file operations.

Acceptance:

- Sensitive writes are blocked before tool execution.
- Reads that could expose internal caches or secrets return a safe refusal.
- Destructive actions still require HITL where already configured.

### Chunk 4: Skill Evolution And Self-Crystallizing Workflows

Status: implemented as safe repo skill manifest/cache plus review-only draft export.

Implemented files:

```text
langgraph-app/repo_skills.py
langgraph-app/agent_graph.py
tests/test_repo_skills.py
.env(exaple)
docs/ALPHARAVIS_ARCHITECTURE.md
docs/ALPHARAVIS_USAGE_NOTES.md
```

AlphaRavis-specific integration:

- `repo_skills.py` adds a Hermes-style mtime/size manifest cache for reviewed
  `ai-skills/` cards and their supporting folders.
- `reload_repo_ai_skills` reports added/removed/changed/unchanged disk skill
  status without changing Mongo skill candidate promotion state.
- `read_repo_ai_skill` can now read safe supporting files under `references/`,
  `templates/`, `scripts/`, and `assets/` in addition to `SKILL.md`.
- `export_skill_candidate_to_repo_draft` can write review-only drafts under
  `ai-skills/_drafts/<slug>/SKILL.md` when
  `ALPHARAVIS_ALLOW_SKILL_DRAFT_EXPORT=true`; candidates stay inactive.
- Normal repo skill hints still contain only compact metadata and never inject
  full skills into every run.

Goal:

- Keep AlphaRavis's safe candidate-review model, while borrowing Hermes's better
  disk-skill ergonomics.

Current AlphaRavis behavior:

- `record_skill_candidate` stores reusable workflows in Mongo/LangGraph Store as
  inactive candidates.
- `activate_skill_candidate` and `deactivate_skill` only work when:

```text
ALPHARAVIS_ALLOW_SKILL_PROMOTION=true
```

- Reviewed repo skill cards live under `ai-skills/`.
- The graph injects only small repo-skill metadata hints; full `SKILL.md` content
  is loaded only through `read_repo_ai_skill`.

Hermes behavior to learn from:

- Disk skills are first-class `SKILL.md` files.
- `prompt_builder.py` caches a skill manifest based on `SKILL.md` and
  `DESCRIPTION.md` mtime/size.
- `skill_commands.py` can reload skills and return added/removed/unchanged
  status.
- Loaded skills include supporting folders such as `references`, `templates`,
  `scripts`, and `assets`.
- Hermes encourages saving difficult repeated workflows as skills, but the
  AlphaRavis version must still keep promotion/manual review.

Scope:

- Add a repo-skill manifest cache for `ai-skills/`.
- Add a `reload_repo_ai_skills` or status tool that reports changes without
  changing promotion state.
- Add an optional exporter from reviewed Store skill candidate to a draft
  `ai-skills/<slug>/SKILL.md`, default off and review-only.
- Keep auto-created skills inactive until human review.
- Add better skill metadata conditions later:
  - required tool categories
  - fallback-only skills
  - platform compatibility

Acceptance:

- AlphaRavis can crystallize workflows into candidates automatically.
- It does not silently make a candidate active.
- Reviewed disk skills become faster and more ergonomic to use.

### Chunk 4.5: Operational Logging And Dependency Trace Files

Status: implemented as local rotating operational/debug log files.

Implemented files:

```text
langgraph-app/operational_logging.py
langgraph-app/agent_graph.py
langgraph-app/bridge_server.py
tests/test_operational_logging.py
.env(exaple)
docker-compose.yml
.gitignore
docs/ALPHARAVIS_ARCHITECTURE.md
docs/ALPHARAVIS_USAGE_NOTES.md
```

AlphaRavis-specific integration:

- Operational logs default to `logs/operational/alpharavis.log` and
  `logs/operational/alpharavis.jsonl`.
- The optional all-debug logger writes to `logs/debug/` only when
  `ALPHARAVIS_DEBUG_ALL_LOGGING=true`.
- Both loggers use daily rotation and keep `ALPHARAVIS_LOG_RETENTION_DAYS`
  backups, default 4 days.
- Logs include timestamps, severity, component, event, dependency, thread/run
  hints, duration, status, and redacted error data.
- `agent_graph.py` logs run start/finish, route decisions, LLM call
  duration/failure, Pixelle/ComfyUI preflight/job status, and semantic memory
  search results.
- `bridge_server.py` logs OpenAI-compatible bridge requests, Responses/Chat
  start/completion, LangGraph stream/wait failures, and LLM health probes.
- Docker mounts `./logs` to `/logs` for both `langgraph-api` and `api-bridge`.

Goal:

- Have local, time-correlated operational evidence even without LangSmith.
- Keep normal logs compact and enable a separate all-debug mode only while
  diagnosing issues.

### Chunk 5: True Lazy Toolsets

Status: implemented for static graph compile-time bundles and MCP category
filtering. Full per-node runtime rebinding remains future work if LangGraph
tool binding becomes hot-swappable.

Goal:

- Move from "the model sees a manifest" to actual bounded tool binding.

Scope:

- Use Hermes `toolsets.py` as the design reference.
- Define composable AlphaRavis toolsets:
  - `coding/read`
  - `coding/write`
  - `coding/execute`
  - `media/image`
  - `media/video`
  - `rag/documents`
  - `rag/memory`
  - `system/docker`
  - `system/ssh`
  - `system/power`
- Keep high-level categories visible.
- Bind concrete tools only after planner/agent chooses the set.
- Cache MCP schemas by category.

Acceptance:

- Done: `run_profile` records selected toolsets and loaded per-agent toolset
  profiles.
- Done: toolset includes detect cycles and cannot recurse forever.
- Done: MCP schemas are cached by category and only matching MCP tools are
  attached to the specialist bundles.
- Done: fast/simple chats still do not pay MCP/tool context cost.

### Chunk 6: Optional Usage, Pricing, And Rate-Limit Telemetry

Goal:

- Capture useful usage/rate-limit metadata without forcing cloud-style pricing
  into a local setup.

Default:

```text
ALPHARAVIS_ENABLE_USAGE_TELEMETRY=false
ALPHARAVIS_ENABLE_COST_ESTIMATION=false
ALPHARAVIS_SHOW_RATE_LIMITS=false
```

Reason:

- Your normal setup is local llama.cpp/Ollama, so cost estimation is not needed
  for daily use.
- Token/usage telemetry can still be useful for compression triggers and
  debugging when enabled.

Scope:

- Borrow only the useful parts from Hermes:
  - `usage_pricing.py` for normalized usage shape
  - `rate_limit_tracker.py` for `x-ratelimit-*` headers
- Mark local models as `local/included`, not paid.
- Add a future Make helper:

```text
make telemetry
```

or include it under `make configure`:

```text
Enable usage telemetry? [y/N]
Enable cost estimation for hosted APIs? [y/N]
Show rate-limit headers? [y/N]
```

Acceptance:

- All telemetry is off by default.
- Compression can use real API usage when present.
- Pricing output never appears unless explicitly enabled.

### Chunk 7: Prompt Assembly And Provider Hardening

Status: implemented for stable prompt context, head/tail context-reference
truncation, and direct Responses compatibility retries. Chat fallback for
DeepAgents remains controlled by the existing `ALPHARAVIS_DEEPAGENTS_API_MODE`
and `ALPHARAVIS_DEEPAGENTS_REQUIRE_RESPONSES` switches.

Goal:

- Make prompt assembly and provider fallback more robust without a huge provider
  rewrite.

Scope:

- Add WSL/Windows environment hints from Hermes `prompt_builder.py`.
- Separate stable prompt material from ephemeral task/memory/skill context.
- Improve head/tail truncation of loaded context files.
- Borrow selected provider-hardening ideas from Hermes `auxiliary_client.py`:
  - unsupported parameter retry
  - model-specific token/temperature quirks
  - safe Chat fallback when Responses tool-calling is broken

Acceptance:

- Done: no provider adapter became a hard dependency.
- Done: LiteLLM remains the default abstraction.
- Done: Responses remains preferred where it is stable.
- Done: direct Responses calls retry once after unsupported parameter errors.
- Done: DeepAgents Responses streaming is ENV-controlled and documented with a
  stable default plus experimental full/hybrid streaming opt-ins.

### Chunk 8: Maintenance And Metadata Helpers

Goal:

- Improve long-term quality after the main runtime path is stable.

Scope:

- Offline archive/trajectory compression evaluator from Hermes
  `trajectory_compressor.py`.
- Optional shell hooks/approval allowlists from `shell_hooks.py`.
- Thread/archive title helper from `title_generator.py`.
- Candidate insight extraction from `insights.py`, review-only.

Acceptance:

- These are maintenance/admin helpers, not mandatory runtime features.
- Nothing here should affect normal LibreChat use unless enabled.

High priority:

1. Context reference preprocessor.

   Status: implemented in Chunk 1. Future refinement can add richer URL
   extraction or browser/VPN-backed fetching, but the safe bridge-side
   preprocessor is wired.

   Reference:

   ```text
   C:\experi\ai\hermes-agent\agent\context_references.py
   parse_context_references
   preprocess_context_references
   _expand_file_reference
   _expand_folder_reference
   _expand_git_reference
   _fetch_url_content
   _resolve_path
   ```

   AlphaRavis target:

   ```text
   langgraph-app/bridge_server.py
   langgraph-app/agent_graph.py
   ```

   Needed behavior:

   - Support explicit `@file`, `@folder`, `@diff`, `@staged`, `@git`, and `@url`
     references before planning.
   - Resolve paths relative to the repo/workspace and keep an `allowed_root`
     guard so references cannot silently escape the intended workspace.
   - Use context budget thresholds similar to Hermes:
     - soft warning around 25 percent of context
     - hard refusal around 50 percent of context
   - Attach files/folders/diffs as explicit context blocks rather than letting
     LibreChat full-history sync or prompt text dump arbitrary data.
   - Record reference warnings in `run_profile`.

2. Streaming internal-context scrubber.

   Status: implemented in Chunk 1 for bridge visible output and Responses
   wrapper streams.

   Reference:

   ```text
   C:\experi\ai\hermes-agent\agent\memory_manager.py
   sanitize_context
   StreamingContextScrubber
   build_memory_context_block
   ```

   AlphaRavis target:

   ```text
   langgraph-app/bridge_server.py
   langgraph-app/agent_graph.py
   ```

   Needed behavior:

   - Keep `<memory-context>...</memory-context>` and similar internal context
     blocks from leaking into LibreChat visible output.
   - Handle SSE chunk boundaries. A simple one-shot regex is not enough because
     opening and closing tags may arrive in different deltas.
   - Keep memory/context visible in Deep Agent/LangGraph debugging where useful,
     but scrub it from normal assistant text unless explicitly requested.

3. API error classification router.

   Status: implemented in Chunk 2 as `langgraph-app/error_classifier.py`.

   Reference:

   ```text
   C:\experi\ai\hermes-agent\agent\error_classifier.py
   FailoverReason
   ClassifiedError
   classify_api_error
   _classify_by_status
   _classify_by_error_code
   _classify_by_message
   ```

   AlphaRavis target:

   ```text
   langgraph-app/responses_client.py
   langgraph-app/bridge_server.py
   langgraph-app/agent_graph.py
   ```

   Needed behavior:

   - Classify `context_overflow` as compression/hard-cutoff work, not a generic
     backend crash.
   - Classify timeout, 502, overloaded, and connection failures as crisis-manager
     candidates when advanced model management is enabled.
   - Classify rate limits and temporary server errors as retry/backoff.
   - Classify format errors as Responses/Chat fallback or unsupported-parameter
     stripping.
   - Store the classified reason in `run_profile` and bridge status events.

4. Central file read/write safety.

   Reference:

   ```text
   C:\experi\ai\hermes-agent\agent\file_safety.py
   is_write_denied
   get_read_block_error
   get_safe_write_root
   ```

   AlphaRavis target:

   ```text
   langgraph-app/file_safety.py
   langgraph-app/owner_power_tools.py
   future file/coding tools
   ```

   Needed behavior:

   - Block writes to sensitive paths such as `.ssh`, `.aws`, `.kube`, `.docker`,
     `.env`, shell profiles, credential files, and system directories.
   - Add optional `ALPHARAVIS_WRITE_SAFE_ROOT`.
   - Block reads of internal cache/vector/secret files when those could become
     prompt-injection or credential leaks.
   - Make future Hermes/deep coding delegation obey the same safety policy.

Medium priority:

5. Disk skill index and manifest cache.

   Reference:

   ```text
   C:\experi\ai\hermes-agent\agent\prompt_builder.py
   _build_skills_manifest
   _load_skills_snapshot
   _write_skills_snapshot
   _parse_skill_file
   _skill_should_show
   build_skills_system_prompt

   C:\experi\ai\hermes-agent\agent\skill_commands.py
   _build_skill_message
   scan_skill_commands
   reload_skills
   build_skill_invocation_message
   ```

   AlphaRavis target:

   ```text
   ai-skills/
   langgraph-app/agent_graph.py
   docs/ALPHARAVIS_USAGE_NOTES.md
   ```

   Needed behavior:

   - Add a manifest cache for repo skills so full `SKILL.md` scans do not run
     every time.
   - Respect skill metadata such as required tools/toolsets, platform guards, and
     fallback-only behavior.
   - Include supporting folders (`references`, `templates`, `scripts`, `assets`)
     in the loaded skill message, with paths resolved relative to the skill
     directory.
   - Add a reload/status command or tool that reports added/removed/unchanged
     skills without auto-promoting Store skill candidates.

6. True lazy toolset resolver.

   Reference:

   ```text
   C:\experi\ai\hermes-agent\toolsets.py
   TOOLSETS
   get_toolset
   resolve_toolset
   resolve_multiple_toolsets
   get_all_toolsets
   validate_toolset
   ```

   AlphaRavis target:

   ```text
   langgraph-app/agent_graph.py
   OPTIONAL_TOOL_REGISTRY
   describe_optional_tool_registry
   ```

   Needed behavior:

   - Replace the current manifest-only approximation with composable toolsets
     such as `coding/read`, `coding/write`, `coding/execute`, `media/video`,
     `rag/memory`, `system/power`.
   - Keep category descriptions visible to the model, but bind concrete tools
     only after the planner or agent selects the category.
   - Cache MCP tool schemas per category.
   - Record selected and loaded toolsets in `run_profile`.
   - Prevent recursive/cyclic toolset includes.

7. Usage, cost, and rate-limit telemetry.

   Reference:

   ```text
   C:\experi\ai\hermes-agent\agent\usage_pricing.py
   CanonicalUsage
   normalize_usage
   estimate_usage_cost

   C:\experi\ai\hermes-agent\agent\rate_limit_tracker.py
   parse_rate_limit_headers
   format_rate_limit_display
   format_rate_limit_compact
   ```

   AlphaRavis target:

   ```text
   langgraph-app/responses_client.py
   langgraph-app/bridge_server.py
   langgraph-app/agent_graph.py
   run_profile
   ```

   Needed behavior:

   - Normalize usage across LiteLLM, llama.cpp, and future hosted providers.
   - Track input, output, reasoning, cache-read, and cache-write tokens.
   - Mark local llama.cpp/Ollama costs as local/included instead of fake money.
   - Parse `x-ratelimit-*` headers when present and show compact status in
     bridge/debug output.
   - Use real usage values for compression decisions whenever available.

8. Prompt assembly and context-file cache hygiene.

   Reference:

   ```text
   C:\experi\ai\hermes-agent\agent\prompt_builder.py
   build_environment_hints
   build_context_files_prompt
   _truncate_content

   C:\experi\ai\hermes-agent\agent\prompt_caching.py
   ```

   AlphaRavis target:

   ```text
   langgraph-app/agent_graph.py
   docs/ALPHARAVIS_ARCHITECTURE.md
   ```

   Needed behavior:

   - Separate stable system prompt material from ephemeral task, memory, skill,
     and handoff context.
   - Add WSL/Windows path hints when the workspace path indicates a mixed
     Windows/Linux environment.
   - Truncate context files by preserving useful head/tail regions and scan
     hints, not just naive first-N characters.
   - Keep stable prompt-cache candidates stable so future provider-side prompt
     caching can work better.

Lower priority / future:

9. Offline trajectory/archive compression evaluator.

   Reference:

   ```text
   C:\experi\ai\hermes-agent\trajectory_compressor.py
   ```

   AlphaRavis target:

   ```text
   archive collections
   vector backfill tools
   maintenance scripts
   ```

   Needed behavior:

   - Batch-evaluate old thread/archive compression quality.
   - Track success/failure metrics for collection summaries.
   - Use it for maintenance/backfill, not the live chat path.

10. Shell hooks and approval allowlists.

    Reference:

    ```text
    C:\experi\ai\hermes-agent\agent\shell_hooks.py
    ```

    AlphaRavis target:

    ```text
    langgraph-app/owner_power_tools.py
    future terminal/file tools
    ```

    Needed behavior:

    - Optional pre/post hooks around shell/system actions.
    - Strict allowlist and audit trail.
    - No automatic destructive hook execution without HITL.

11. Provider adapter hardening.

    Reference:

    ```text
    C:\experi\ai\hermes-agent\agent\auxiliary_client.py
    C:\experi\ai\hermes-agent\agent\codex_responses_adapter.py
    C:\experi\ai\hermes-agent\agent\anthropic_adapter.py
    C:\experi\ai\hermes-agent\agent\gemini_native_adapter.py
    ```

    AlphaRavis target:

    ```text
    langgraph-app/responses_client.py
    langgraph-app/bridge_server.py
    ```

    Needed behavior:

    - Strip unsupported parameters and retry where safe.
    - Map model-specific max-output-token and temperature behavior.
    - Keep Chat Completions fallback for providers with broken Responses tools.
    - Add direct non-OpenAI providers only if LiteLLM is not enough.

12. Thread title and insight helpers.

    Reference:

    ```text
    C:\experi\ai\hermes-agent\agent\title_generator.py
    C:\experi\ai\hermes-agent\agent\insights.py
    ```

    AlphaRavis target:

    ```text
    archive titles
    archive collections
    LibreChat/bridge metadata
    curated memory review
    ```

    Needed behavior:

    - Generate short stable titles for archive records and archive collections.
    - Extract candidate user/system insights for review without auto-promoting
      them into always-memory.
    - Keep this separate from raw archives and pgvector source-of-truth rules.
