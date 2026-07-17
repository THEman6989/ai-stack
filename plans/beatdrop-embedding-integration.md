# AlphaRavis × ComfyUI BeatDrop Embedding — Integration Plan

> **Für Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** AlphaRavis steuert den `BeatDropSelectorEmbeddingNode` via `ai_stack_config_json`, empfängt die selektierten Outfit-Bilder per HTTP von ComfyUI, zeigt sie im Chat an, und implementiert einen User-Feedback-Loop für iterative Outfit-Auswahl.

**Architecture:** AlphaRavis-LangGraph-Agent sendet Workflow mit `ai_stack_config_json` an ComfyUI (über bestehenden `comfyui_client.py`), pollt auf Completion, lädt `ai_stack_context`-Output (enthält Bild-Pfade), lädt Bilder per HTTP `/view` API, sendet sie in den Chat, und wartet auf User-Feedback für nächsten Durchlauf.

**Tech Stack:** Python 3.12, httpx, LangGraph, ComfyUI REST API, Pillow (für Bild-Download-Verifikation)

---

## Task 1: `ai_stack_config_json` Generator bauen

**Objective:** Helper-Funktion die aus User-Chat-Kontext + Memory das JSON für `ai_stack_config_json` baut.

**Files:**
- Create: `langgraph-app/beatdrop_embedding_config.py`

**Step 1: Datei anlegen mit Basis-Struktur**

```python
"""Build ai_stack_config_json payload for BeatDropSelectorEmbeddingNode."""

from __future__ import annotations

import json
from typing import Any


def build_embedding_config(
    *,
    extra_instructions: str = "",
    phases: dict[int, str] | None = None,
    text_query_scene_fit: str = "",
    text_query_change_target: str = "",
    reranker_query: str = "",
    conversation_id: str = "",
    weights: dict[str, float] | None = None,
    thresholds: dict[str, float] | None = None,
    history: dict[str, float | int] | None = None,
    max_frames_per_window: int | None = None,
    max_candidate_images: int | None = None,
    use_vlm_fallback: bool | None = None,
) -> str:
    """Build ai_stack_config_json for BeatDropSelectorEmbeddingNode.

    Only non-None values are included in the output. The node
    merges them over its built-in defaults.
    """
    cfg: dict[str, Any] = {}

    # Text overrides
    if extra_instructions:
        cfg["extra_instructions"] = extra_instructions
    if text_query_scene_fit:
        cfg["text_query_scene_fit"] = text_query_scene_fit
    if text_query_change_target:
        cfg["text_query_change_target"] = text_query_change_target
    if reranker_query:
        cfg["reranker_query"] = reranker_query
    if conversation_id:
        cfg["conversation_id"] = conversation_id

    # Per-phase text queries
    if phases:
        cfg["text_query_per_phase"] = {
            str(k): v for k, v in phases.items()
        }

    # Weights
    if weights:
        cfg["weights"] = {k: v for k, v in weights.items() if v is not None}

    # Thresholds
    if thresholds:
        cfg["thresholds"] = {k: v for k, v in thresholds.items() if v is not None}

    # History
    if history:
        cfg["history"] = {k: v for k, v in history.items() if v is not None}

    # Limits
    if max_frames_per_window is not None:
        cfg["max_frames_per_window"] = max_frames_per_window
    if max_candidate_images is not None:
        cfg["max_candidate_images"] = max_candidate_images
    if use_vlm_fallback is not None:
        cfg["use_vlm_fallback"] = use_vlm_fallback

    return json.dumps(cfg) if cfg else "{}"
```

**Step 2: Test schreiben**

```python
# tests/test_beatdrop_embedding_config.py

def test_build_minimal():
    result = build_embedding_config(extra_instructions="test")
    assert "extra_instructions" in result
    assert '"extra_instructions": "test"' in result

def test_build_full():
    result = build_embedding_config(
        extra_instructions="Streetwear",
        phases={0: "jacket", 1: "casual"},
        weights={"change_strength": 0.60},
        history={"penalty": 8.0},
    )
    d = json.loads(result)
    assert d["extra_instructions"] == "Streetwear"
    assert d["text_query_per_phase"]["0"] == "jacket"
    assert d["weights"]["change_strength"] == 0.60

def test_build_empty():
    assert build_embedding_config() == "{}"
```

Run: `pytest tests/test_beatdrop_embedding_config.py -v`

**Step 3: Commit**

```bash
git add langgraph-app/beatdrop_embedding_config.py tests/test_beatdrop_embedding_config.py
git commit -m "feat: add ai_stack_config_json generator for BeatDrop embedding node"
```

---

## Task 2: `ai_stack_context` Parser bauen

**Objective:** Parser der den `ai_stack_context` JSON-Output des Nodes liest und in Python-Dataclasses normalisiert.

**Files:**
- Create: `langgraph-app/beatdrop_embedding_context.py`

**Step 1: Dataclass + Parser**

```python
"""Parse ai_stack_context output from BeatDropSelectorEmbeddingNode."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PhaseDecision:
    phase: int
    beat_time: float | None
    is_drop: bool
    num_outfits: int
    selected_frames: list[int]
    rejected_frames: list[int]
    vlm_overrides: dict | None = None

@dataclass
class SavedFile:
    type: str  # "contact_sheet" | "selected_frame"
    path: str
    filename: str
    frame_index: int | None = None

@dataclass
class BeatdropSelectionContext:
    thread_id: str
    stage_used: str
    phase_decisions: list[PhaseDecision]
    total_selected: int
    saved_files: list[SavedFile]
    selections_dir: str
    embedding_top10: list[dict] = field(default_factory=list)
    folder_assignments: list[dict] = field(default_factory=list)
    instructions: str = ""
    raw_json: str = ""


def parse_ai_stack_context(raw: str) -> BeatdropSelectionContext:
    """Parse the ai_stack_context JSON string into structured data."""
    data = json.loads(raw)

    phase_decisions = []
    for pd in data.get("phase_decisions", []):
        phase_decisions.append(PhaseDecision(
            phase=pd.get("phase", 0),
            beat_time=pd.get("beat_time"),
            is_drop=pd.get("is_drop", False),
            num_outfits=pd.get("num_outfits", 2),
            selected_frames=pd.get("selected_frames", []),
            rejected_frames=pd.get("rejected_frames", []),
            vlm_overrides=pd.get("vlm_overrides"),
        ))

    saved_files = []
    for sf in data.get("saved_files", []):
        saved_files.append(SavedFile(
            type=sf.get("type", ""),
            path=sf.get("path", ""),
            filename=sf.get("filename", ""),
            frame_index=sf.get("frame_index"),
        ))

    folder_info = data.get("folder_source", {})
    return BeatdropSelectionContext(
        thread_id=data.get("thread_id", ""),
        stage_used=data.get("stage_used", ""),
        phase_decisions=phase_decisions,
        total_selected=data.get("total_selected", 0),
        saved_files=saved_files,
        selections_dir=data.get("selections_dir", ""),
        embedding_top10=data.get("embedding_top10", []),
        folder_assignments=folder_info.get("assignments", []),
        instructions=data.get("instructions", ""),
        raw_json=raw,
    )
```

**Step 2: Test**

```python
# tests/test_beatdrop_embedding_context.py

def test_parse_minimal():
    raw = json.dumps({
        "thread_id": "abc",
        "stage_used": "embedding",
        "phase_decisions": [{
            "phase": 0, "selected_frames": [3], "rejected_frames": [5],
            "num_outfits": 2, "is_drop": True
        }],
        "total_selected": 1,
        "saved_files": [],
        "selections_dir": "/tmp/test",
    })
    ctx = parse_ai_stack_context(raw)
    assert ctx.thread_id == "abc"
    assert ctx.stage_used == "embedding"
    assert len(ctx.phase_decisions) == 1
    assert ctx.phase_decisions[0].selected_frames == [3]
```

Run: `pytest tests/test_beatdrop_embedding_context.py -v`

**Step 3: Commit**

```bash
git add langgraph-app/beatdrop_embedding_context.py tests/
git commit -m "feat: add ai_stack_context parser for BeatDrop embedding node"
```

---

## Task 3: HTTP Image Download von ComfyUI

**Objective:** Funktion die Bilder vom ComfyUI-Server per `/view` API herunterlädt.

**Files:**
- Modify: `langgraph-app/comfyui_client.py` (neue Methode)

**Step 1: `download_selection_image` zu ComfyUIClient hinzufügen**

In `comfyui_client.py`, zur bestehenden Klasse hinzufügen:

```python
async def download_selection_image(
    self,
    filename: str,
    subfolder: str = "_beatdrop_selections",
    output_dir: str | None = None,
) -> bytes:
    """Download a selection image from the ComfyUI output directory.

    Uses the /view API endpoint. Returns raw image bytes.
    """
    params = {
        "filename": filename,
        "type": "output",
        "subfolder": subfolder,
    }
    url = f"{self.base_url}/view?{urlencode(params)}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content

async def download_all_selections(
    self,
    saved_files: list[dict],
    subfolder: str = "_beatdrop_selections",
) -> dict[str, bytes]:
    """Download all selection images. Returns {filename: bytes}."""
    results = {}
    for sf in saved_files:
        fname = sf.get("filename", "")
        if not fname:
            continue
        try:
            results[fname] = await self.download_selection_image(
                fname, subfolder=subfolder,
            )
        except Exception as e:
            print(f"Failed to download {fname}: {e}")
    return results
```

**Step 2: Rauchtest**

```python
# scripts/smoke_beatdrop_download.py
import asyncio
from langgraph_app.comfyui_client import ComfyUIClient

async def main():
    client = ComfyUIClient(base_url="http://192.168.x.x:8188")
    img = await client.download_selection_image(
        "frame_0003.png",
        subfolder="_beatdrop_selections/thread_abc",
    )
    print(f"Downloaded {len(img)} bytes")

asyncio.run(main())
```

**Step 3: Commit**

```bash
git add langgraph-app/comfyui_client.py
git commit -m "feat: add HTTP image download for ComfyUI beatdrop selections"
```

---

## Task 4: ComfyUI Workflow mit `ai_stack_config` injecten

**Objective:** Bestehenden Workflow-Submit um `ai_stack_config_json`-Injection erweitern.

**Files:**
- Modify: `langgraph-app/comfyui_client.py` (oder Workflow-Library)

**Step 1: Workflow-Patch-Funktion**

```python
def inject_beatdrop_config(
    workflow: dict[str, Any],
    ai_stack_config_json: str,
    node_title: str = "BeatDropSelectorEmbeddingNode",
) -> dict[str, Any]:
    """Inject ai_stack_config_json into a BeatDropSelectorEmbeddingNode in the workflow."""
    import copy
    wf = copy.deepcopy(workflow)

    # Find the node by class_type
    for node_id, node in wf.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type", "")
        title = node.get("_meta", {}).get("title", "")

        if class_type == node_title or title == node_title:
            inputs = node.setdefault("inputs", {})
            inputs["ai_stack_config_json"] = ai_stack_config_json
            print(f"Injected ai_stack_config_json into node '{node_id}' ({class_type})")
            return wf

    raise ValueError(f"Node '{node_title}' not found in workflow")
```

**Step 2: Test**

```python
def test_inject_beatdrop_config():
    wf = {
        "1": {"class_type": "LoadImage", "inputs": {}},
        "2": {"class_type": "BeatDropSelectorEmbeddingNode", "inputs": {
            "candidate_folders": "/test",
        }},
    }
    result = inject_beatdrop_config(wf, '{"extra_instructions":"test"}')
    assert result["2"]["inputs"]["ai_stack_config_json"] == '{"extra_instructions":"test"}'
    # Verify original inputs still there
    assert result["2"]["inputs"]["candidate_folders"] == "/test"
```

**Step 3: Commit**

```bash
git add langgraph-app/comfyui_client.py tests/
git commit -m "feat: inject ai_stack_config_json into BeatDrop workflow"
```

---

## Task 5: Agent-Tool: Drop-Plan laden + Bilder downloaden

**Objective:** AlphaRavis lädt EIN Artefakt (`drop_plan.json`) per HTTP von ComfyUI, parsed es, und lädt dann die referenzierten Bilder. Kein Workflow-Output-Parsing nötig.

**Files:**
- Create: `langgraph-app/beatdrop_embedding_agent.py`

**Warum PlanWriter-Pattern:**
Der Node speichert JETZT `drop_plan.json` als Datei neben den Bildern:
```
ComfyUI/output/_beatdrop_selections/{thread_id}/
  ├── drop_plan.json        ← EIN Artefakt
  ├── contact_sheet.png
  ├── frame_0003.png
  └── frame_0007.png
```

AlphaRavis lädt nur `drop_plan.json` — der Rest steht da drin.

**Step 1: Agent-Tool implementieren**

```python
"""Agent tool: load BeatDrop drop_plan from ComfyUI."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from beatdrop_embedding_config import build_embedding_config
from beatdrop_embedding_context import parse_ai_stack_context
from comfyui_client import ComfyUIClient, inject_beatdrop_config


async def select_beatdrop_outfits(
    *,
    comfyui_client: ComfyUIClient,
    workflow: dict[str, Any],
    extra_instructions: str,
    phases: dict[int, str] | None = None,
    conversation_id: str = "",
    max_wait_seconds: int = 120,
    **config_overrides: Any,
) -> dict[str, Any]:
    """Run BeatDropSelectorEmbeddingNode and return structured results.

    Flow:
    1. Build ai_stack_config_json
    2. Inject into workflow, submit, poll for completion
    3. Download drop_plan.json via /view API  ← EIN Artefakt
    4. Parse + download referenced images
    5. Return chat-ready result
    """
    # 1. Build config
    config_json = build_embedding_config(
        extra_instructions=extra_instructions,
        phases=phases,
        conversation_id=conversation_id,
        **config_overrides,
    )

    # 2. Inject into workflow + submit
    wf = inject_beatdrop_config(workflow, config_json)
    prompt_id = await comfyui_client.submit_workflow(wf)
    await comfyui_client.wait_for_completion(prompt_id, timeout=max_wait_seconds)

    # 3. Download drop_plan.json — the ONE artifact
    subfolder = f"_beatdrop_selections/{conversation_id}" if conversation_id else "_beatdrop_selections"
    try:
        plan_bytes = await comfyui_client.download_selection_image(
            "drop_plan.json", subfolder=subfolder,
        )
        ai_context_raw = plan_bytes.decode("utf-8")
    except Exception as e:
        return {"error": f"Failed to download drop_plan.json: {e}"}

    # 4. Parse context
    ctx = parse_ai_stack_context(ai_context_raw)

    # 5. Download images referenced in the plan
    images = {}
    for sf in ctx.saved_files:
        fname = sf.filename
        if fname == "drop_plan.json":
            continue  # already loaded
        try:
            images[fname] = await comfyui_client.download_selection_image(
                fname, subfolder=subfolder,
            )
        except Exception as e:
            print(f"Failed to download {fname}: {e}")

    # 6. Build chat message
    chat_lines = ["Hier die selektierten Outfits:"]
    for pd in ctx.phase_decisions:
        folder_name = "?"
        for fa in ctx.folder_assignments:
            if fa.get("phase") == pd.phase:
                folder_name = fa.get("folder", "?")
                break
        chat_lines.append(
            f"Phase {pd.phase} ({folder_name}): "
            f"Frames {', '.join(str(f) for f in pd.selected_frames)} "
            f"({len(pd.selected_frames)} ausgewählt)"
        )
    chat_lines.append(f"\nStage: {ctx.stage_used}")
    chat_lines.append(f"Instructions: {ctx.instructions}")
    chat_lines.append("\nPassen die Outfits? Oder anderer Durchlauf?")

    return {
        "context": ctx,
        "images": images,
        "chat_message": "\n".join(chat_lines),
    }
```

**Step 2: In Agent-Toolset registrieren**

In `langgraph-app/agent_graph.py` (oder wo Tools registriert werden):

```python
from beatdrop_embedding_agent import select_beatdrop_outfits

# Als Tool registrieren (je nach Tool-Registry):
tools.append(select_beatdrop_outfits)
```

**Step 3: Smoketest**

```bash
python scripts/smoke_beatdrop_select.py
```

**Step 4: Commit**

```bash
git add langgraph-app/beatdrop_embedding_agent.py langgraph-app/agent_graph.py
git commit -m "feat: add select_beatdrop_outfits agent tool"
```

---

## Task 6: Chat-Integration + Feedback-Loop

**Objective:** Bilder im Chat anzeigen und User-Feedback ("passt nicht") → neuen Durchlauf starten.

**Files:**
- Modify: `langgraph-app/beatdrop_embedding_agent.py`
- Modify: `langgraph-app/agent_graph.py` (State erweitern)

**Step 1: State um Beatdrop-Feedback erweitern**

```python
# Im Agent-State (TypedDict oder Pydantic):
class AgentState(TypedDict):
    # ... bestehende Felder ...
    beatdrop_iteration: int          # Zähler für Durchläufe
    beatdrop_last_context: str       # JSON vom letzten ai_stack_context
    beatdrop_rejected_frames: list   # Frames die User abgelehnt hat
```

**Step 2: Feedback-Loop in select_beatdrop_outfits**

```python
async def select_beatdrop_outfits_with_feedback(
    state: AgentState,
    ...,
) -> AgentState:
    """Run selection + present to user + handle feedback loop."""

    # Track iteration
    iteration = state.get("beatdrop_iteration", 0)

    # If user rejected previous selection, increase history penalty
    history_override = {}
    if iteration > 0 and state.get("beatdrop_rejected_frames"):
        history_override = {
            "penalty": 10.0 + iteration * 3.0,  # stronger each round
            "decay_rate": 0.15,                  # slower decay
        }

    config_json = build_embedding_config(
        extra_instructions=state.get("user_instructions", ""),
        history=history_override if history_override else None,
        conversation_id=state.get("conversation_id", ""),
    )

    # ... (wie Task 5: submit, poll, parse, download) ...

    # Update state
    state["beatdrop_iteration"] = iteration + 1
    state["beatdrop_last_context"] = context_json

    # Send to user: images + "Passen die?"
    # (Message-Sending hängt vom Chat-Adapter ab)

    return state
```

**Step 3: User-Feedback-Handler**

```python
def handle_beatdrop_feedback(state: AgentState, user_response: str) -> AgentState:
    """Process user feedback on outfit selection."""
    response = user_response.lower().strip()

    if any(w in response for w in ("ja", "passt", "ok", "gut", "perfekt")):
        state["beatdrop_approved"] = True
        return state

    # User wants different selection
    rejected = state.get("beatdrop_rejected_frames", [])
    last_ctx = json.loads(state.get("beatdrop_last_context", "{}"))
    for pd in last_ctx.get("phase_decisions", []):
        rejected.extend(pd.get("selected_frames", []))
    state["beatdrop_rejected_frames"] = list(set(rejected))
    state["beatdrop_approved"] = False

    # Extract new instructions from user response
    state["user_instructions"] = user_response

    return state
```

**Step 4: Commit**

```bash
git add langgraph-app/beatdrop_embedding_agent.py langgraph-app/agent_graph.py
git commit -m "feat: add beatdrop chat integration + feedback loop"
```

---

## Task 7: Memory-Integration

**Objective:** `ai_stack_context` in `record_curated_memory` speichern für spätere Iterationen.

**Files:**
- Modify: `langgraph-app/beatdrop_embedding_agent.py`

**Step 1: Memory-Speicherung nach erfolgreicher Selektion**

```python
async def persist_beatdrop_selection(state: AgentState) -> None:
    """Store beatdrop selection in curated memory."""
    ctx_raw = state.get("beatdrop_last_context", "")
    if not ctx_raw:
        return

    ctx = json.loads(ctx_raw)

    # Build memory entry
    memory_entry = {
        "type": "beatdrop_selection",
        "thread_id": ctx.get("thread_id", ""),
        "stage_used": ctx.get("stage_used", ""),
        "timestamp": ctx.get("timestamp", 0),
        "summary": {
            "total_selected": ctx.get("total_selected", 0),
            "phases": [
                {
                    "phase": pd.get("phase"),
                    "selected_frames": pd.get("selected_frames"),
                    "folder": next(
                        (fa.get("folder") for fa in ctx.get("folder_source", {}).get("assignments", [])
                         if fa.get("phase") == pd.get("phase")),
                        "unknown",
                    ),
                }
                for pd in ctx.get("phase_decisions", [])
            ],
        },
        "instructions": ctx.get("instructions", ""),
    }

    # Store via LangGraph Store or record_curated_memory
    await state.get("store").put(
        ("beatdrop_selections", ctx.get("thread_id", "unknown")),
        "latest",
        memory_entry,
    )
```

**Step 2: Memory beim nächsten Run als Kontext nutzen**

```python
def load_previous_selection(state: AgentState) -> dict | None:
    """Load previous beatdrop selection from memory for context."""
    store = state.get("store")
    if not store:
        return None

    try:
        item = store.get(
            ("beatdrop_selections", state.get("conversation_id", "")),
            "latest",
        )
        return item.value if item else None
    except Exception:
        return None
```

**Step 3: Commit**

```bash
git add langgraph-app/beatdrop_embedding_agent.py
git commit -m "feat: persist beatdrop selections to memory"
```

---

## Task 8: End-to-End Smoketest

**Objective:** Kompletten Flow durchtesten: Config → Workflow → Poll → Parse → Download → Chat.

**Files:**
- Create: `scripts/smoke_beatdrop_e2e.py`

**Step 1: E2E-Test-Script**

```python
#!/usr/bin/env python3
"""End-to-end smoketest: BeatDrop Embedding Node via AlphaRavis."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "langgraph-app"))

from beatdrop_embedding_config import build_embedding_config
from comfyui_client import ComfyUIClient


async def main():
    client = ComfyUIClient(base_url="http://192.168.1.100:8188")

    # 1. Build config
    cfg = build_embedding_config(
        extra_instructions="Streetwear, Phase 0 mit Jacke, Phase 1 ohne",
        phases={0: "jacket streetwear urban", 1: "casual no jacket summer"},
        weights={"change_strength": 0.60},
        conversation_id="smoke_test_001",
    )
    print(f"Config: {cfg[:200]}...")

    # 2. Load workflow
    wf_path = Path("workflows/beatdrop_embedding_test.json")
    if wf_path.exists():
        wf = json.loads(wf_path.read_text())
    else:
        print("ERROR: Workflow file not found. Create a test workflow first.")
        print("  → ComfyUI: add BeatDropSelectorEmbeddingNode with candidate_folders set")
        print("  → Export (API format) → save as workflows/beatdrop_embedding_test.json")
        return

    # 3. Inject config
    from comfyui_client import inject_beatdrop_config
    wf = inject_beatdrop_config(wf, cfg)

    # 4. Submit + poll
    prompt_id = await client.submit_workflow(wf)
    print(f"Submitted: {prompt_id}")

    result = await client.wait_for_completion(prompt_id, timeout=300)
    print(f"Result: {json.dumps(result, indent=2)[:500]}...")

    print("\n✓ E2E smoketest complete")


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Ausführen**

```bash
python scripts/smoke_beatdrop_e2e.py
```

Erwartet: `✓ E2E smoketest complete`

**Step 3: Commit**

```bash
git add scripts/smoke_beatdrop_e2e.py
git commit -m "test: add beatdrop embedding e2e smoketest"
```

---

## Abhängigkeiten & Voraussetzungen

- [x] `BeatDropSelectorEmbeddingNode` in ComfyUI registriert
- [x] `ai_stack_config_json` Input unterstützt alle Overrides
- [x] `ai_stack_context` Output liefert strukturiertes JSON
- [x] Bilder werden in `ComfyUI/output/_beatdrop_selections/` gespeichert
- [ ] AlphaRavis kann ComfyUI-Server per HTTP erreichen (fixe IP)

## Verifikation nach allen Tasks

```bash
# Unit tests
pytest tests/test_beatdrop_embedding_config.py tests/test_beatdrop_embedding_context.py -v

# Integration test
# 1. ComfyUI Workflow mit BeatDropSelectorEmbeddingNode erstellen
# 2. candidate_folders auf /path/to/outfits/ setzen
# 3. Als API-Format exportieren → workflows/beatdrop_embedding_test.json
# 4. E2E-Test ausführen:
python scripts/smoke_beatdrop_e2e.py
```
