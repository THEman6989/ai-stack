# Deep Agents UI — AlphaRavis Migration Plan

> **Status: IMPLEMENTED + HARDENED** (2026-05-24) — Planned features are ported and the review shortcomings have been fixed. Remaining check: live browser smoke test including Monaco-on-button behavior, upload flows, and thread rename/delete recovery.
> **Branch:** dev2

**Goal:** Port missing features from agent-custom-ui (agent-chat-ui fork) into submodules/deep-agents-ui (THEman6989 fork), making it the primary AlphaRavis chat UI.

**Architecture:** deep-agents-ui is a Next.js 16 + React 19 app. Features are added as new components/hooks in `src/app/components/` and `src/app/hooks/`. The ChatProvider context is extended to carry new state (files, openers, skills). All additions follow existing patterns (shadcn/ui, Tailwind, framer-motion).

**Tech Stack:** Next.js 16, React 19, TypeScript, Tailwind CSS, @langchain/langgraph-sdk, @langchain/core, shadcn/ui (Radix), framer-motion, SWR, Monaco Editor

**Source of truth for ported code:** `agent-custom-ui/src/` (fork of langchain-ai/agent-chat-ui)

---

## Final Status: 16/16 Tasks Done + 3 Bonus Features + Hardening Fixes

### Phase 1: File Upload ✅ COMPLETE

| Task | Status | File |
|---|---|---|
| 1.1 Port multimodal-utils.ts | ✅ | `src/lib/multimodal-utils.ts` — Base64, Type-Guards |
| 1.2 Port file-validation.ts | ✅ | `src/lib/file-validation.ts` — Validierung, Duplikate |
| 1.3 Port useFileUpload hook | ✅ | `src/app/hooks/useFileUpload.ts` — Drag&Drop, Paste, Input |
| 1.4 ContentBlocksPreview | ✅ | `src/app/components/ContentBlocksPreview.tsx` |
| 1.5 MultimodalPreview | ✅ | `src/app/components/MultimodalPreview.tsx` — Bilder/PDFs |
| 1.6 Wire into ChatInterface | ✅ | Paperclip-Button, hidden input, preview, content blocks in sendMessage, `dropRef`, paste handler, drag-over overlay, processing state, remove-all |

### Phase 2: Thread Rename/Delete ✅ COMPLETE (rewritten)

| Task | Status | File |
|---|---|---|
| 2.1 Thread operations | ✅ | Direkt in ThreadList.tsx — `useClient()` statt separatem Hook; empty-title guard, pending states, metadata title display, active-thread delete recovery |
| 2.2 Thread helpers | ✅ | Inline — `client.threads.update()`, `client.threads.delete()` |
| 2.3 Edit/delete UI | ✅ | Hover-icons (Pencil/Trash2), inline rename input, window.confirm delete |

### Phase 3: Chat Openers ✅ COMPLETE

| Task | Status | File |
|---|---|---|
| 3.1 Port ChatOpeners | ✅ | `src/app/components/ChatOpeners.tsx` — Carousel, framer-motion |
| 3.2 Wire into ChatInterface | ✅ | 8 default AlphaRavis prompts, shown on empty chat |

### Phase 4: Artifact System ✅ COMPLETE

| Task | Status | File |
|---|---|---|
| 4.1 Port artifact system | ✅ | `src/app/components/artifact.tsx` — Portal-basiert, Provider |
| 4.2 Wire into layout | ✅ | `src/app/layout.tsx` — ArtifactProvider + Portals |

### Phase 5: Polish ✅ COMPLETE

| Task | Status |
|---|---|
| 5.1 Dark mode | ✅ Already supported via Radix colors |
| 5.2 Docker build | ✅ `docker compose build --no-cache deep-agents-ui` passes |
| 5.3 Docs updated | ✅ CHANGES.md, ARCHITECTURE.md, USAGE_NOTES.md, OPEN_TASKS.md, AGENTS.md |
| 5.4 Lockfile hygiene | ✅ `yarn.lock` regenerated; Monaco peer dependency explicit; unused `diff` package removed |
| 5.5 Docker context hygiene | ✅ `.dockerignore` added to avoid sending `node_modules`/`.next` in build context |
| 5.6 Next build config hygiene | ✅ `tsconfig.json` aligned with Next.js 16 build-time requirements |

---

## Bonus Features (from AionUi Analysis)

| Feature | Status | File |
|---|---|---|
| **Diff Viewer** | ✅ | `src/app/components/DiffViewer.tsx` — Color-coded, auto-detect |
| **Skills Indicator** | ✅ | `src/app/components/SkillsIndicator.tsx` — CloudLightning icon, tooltip |
| **File Preview Panel** | ✅ | `src/app/components/FilePreviewPanel.tsx` — lightweight code preview by default, Monaco only after button, Markdown, Diff |

---

## Files Changed (submodules/deep-agents-ui)

### Modified
```
package.json                          (+framer-motion, @monaco-editor/react, monaco-editor)
src/app/components/ChatInterface.tsx  (+FileUpload, +ChatOpeners, +Skills, +Preview)
src/app/components/ThreadList.tsx     (+Rename/Delete inline)
src/app/hooks/useChat.ts             (+contentBlocks, +skills in StateType)
src/app/layout.tsx                    (+ArtifactProvider)
src/app/components/ToolCallBox.tsx    (+DiffViewer detection)
yarn.lock                             (regenerated dependency lockfile)
.dockerignore                         (small reproducible Docker build context)
```

### New
```
src/lib/multimodal-utils.ts
src/lib/file-validation.ts
src/app/hooks/useFileUpload.ts
src/app/components/ContentBlocksPreview.tsx
src/app/components/MultimodalPreview.tsx
src/app/components/ChatOpeners.tsx
src/app/components/artifact.tsx
src/app/components/DiffViewer.tsx
src/app/components/SkillsIndicator.tsx
src/app/components/FilePreviewPanel.tsx
src/lib/diff-utils.ts
```

## Verification (2026-05-24 hardening pass)

- `docker run --rm -v "$PWD/submodules/deep-agents-ui:/app" -w /app node:20-alpine yarn install --frozen-lockfile` passes.
- `docker run --rm -v "$PWD/submodules/deep-agents-ui:/app" -w /app node:20-alpine yarn lint` passes with 0 errors / 7 existing Fast Refresh warnings.
- `git diff --check` and `git -C submodules/deep-agents-ui diff --check` pass.
- Static added-line scan found no secrets/dangerous eval/shell patterns.
- Independent review passed after fixing the ThreadList blur/Escape rename edge case.
- `docker compose build --no-cache deep-agents-ui` passes with `yarn install --frozen-lockfile`.
- `yarn.lock` now contains `framer-motion`, `@monaco-editor/react`, and `monaco-editor`; unused `diff` / `@types/diff` entries were removed.

## Remaining live check

- Browser smoke test on `deep-agents-ui`: file picker, drag & drop, paste upload, attachment remove/remove-all, preview panel, lightweight diff rendering, code preview before/after `Open Monaco editor`, thread rename/delete including active-thread deletion recovery.

---

## Git History (submodule)

```
4472526 feat: wire FilePreviewPanel into ChatInterface
d85aff6 feat: File Preview Panel (Monaco + Markdown + Diff)
1eac7f2 feat: Diff Viewer + Skills Indicator
b0003af feat: AlphaRavis customizations — file upload, chat openers, thread rename/delete, artifact system
f6a4f34 (upstream) Merge pull request #111 from langchain-ai/fix/cve-2026-44578-next
```

## Fork Info

- **Fork URL:** https://github.com/THEman6989/deep-agents-ui
- **Upstream:** https://github.com/langchain-ai/deep-agents-ui
- **Submodule path:** `submodules/deep-agents-ui/`
