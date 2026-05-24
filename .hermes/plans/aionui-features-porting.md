# AionUi Features — Porting Candidates for deep-agents-ui

> **Status Update: 2026-05-24** — Tier 1 & 2 complete. Tier 3 pending.
> deep-agents-ui fork: `THEman6989/deep-agents-ui` (submodule `submodules/deep-agents-ui/`)

---

## Tier 1 — Quick Wins ✅ DONE

### Diff Viewer ✅
**Datei:** `src/app/components/DiffViewer.tsx`
**Integration:** `ToolCallBox.tsx` — Auto-detect `isDiffContent()`, render DiffViewer statt `<pre>`.
**Dependency:** `diff` (npm)
**Build:** ✅

### Skills Indicator ✅
**Datei:** `src/app/components/SkillsIndicator.tsx`
**Integration:** `ChatInterface.tsx` — CloudLightning icon in input row, tooltip mit Skill-Namen.
**State:** `skills` field added to `StateType` in `useChat.ts`.
**Build:** ✅

---

## Tier 2 — File Preview ✅ DONE

### File Preview Panel ✅
**Datei:** `src/app/components/FilePreviewPanel.tsx`
**Features:**
- Tab-basiertes Multi-File-Preview
- Code → Monaco Editor (lazy-loaded)
- Markdown → MarkdownContent rendering
- Diff → DiffViewer component
- Auto-detection von File-Extension → Language
**Integration:** `ChatInterface.tsx` — "Preview"-Button (Eye icon) im Files-Row, Overlay-Panel.
**Dependencies:** `@monaco-editor/react`, `diff`
**Build:** ✅

### Monaco Code Editor ✅
Im FilePreviewPanel integriert. Read-only, Dark-Theme, Minimap disabled, Word-Wrap.
Wird lazy geladen (dynamic import), SSR disabled.

---

## Tier 3 — Nice to Have ⏳ PENDING

### i18n / Mehrsprachigkeit
**Plan:** `react-i18next` oder `next-i18next`. Key-basierte Strings. Erst en+de. ~3-4h.
**Status:** Nicht gestartet.

### Inline-Tool-Execution-Streaming
**Plan:** LangGraph `on_tool_end` Events abfangen, Zwischenergebnisse in ToolCallBox live anzeigen. ~3h.
**Status:** Nicht gestartet.

### Conversation Tabs
**Plan:** Tab-Leiste über Chat, mehrere Threads parallel offen. `useStream` pro Tab. ~4-5h.
**Status:** Nicht gestartet.

---

## Tier 4 — Zukunftsmusik

### Agent Model Selector (~2h)
### Version History via Checkpoints (~5h)
### Workspace File Tree (Streaming) (~3-4h)

---

## Completed Summary

| Feature | Datei | Build |
|---|---|---|
| Diff Viewer | `src/app/components/DiffViewer.tsx` | ✅ |
| Skills Indicator | `src/app/components/SkillsIndicator.tsx` | ✅ |
| File Preview Panel | `src/app/components/FilePreviewPanel.tsx` | ✅ |
| Monaco Editor | In FilePreviewPanel (lazy) | ✅ |
| Markdown Preview | In FilePreviewPanel | ✅ |
