# Deep Agents UI — AlphaRavis Migration Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.
> **Branch:** dev2

**Goal:** Port missing features from agent-custom-ui (agent-chat-ui fork) into submodules/deep-agents-ui (THEman6989 fork), making it the primary AlphaRavis chat UI.

**Architecture:** deep-agents-ui is a Next.js 16 + React 19 app. Features are added as new components/hooks in `src/app/components/` and `src/app/hooks/`. The ChatProvider context is extended to carry new state (files, openers). All additions follow existing patterns (shadcn/ui, Tailwind, framer-motion).

**Tech Stack:** Next.js 16, React 19, TypeScript, Tailwind CSS, @langchain/langgraph-sdk, @langchain/core, shadcn/ui (Radix), framer-motion, SWR

**Source of truth for ported code:** `agent-custom-ui/src/` (fork of langchain-ai/agent-chat-ui)

---

## Pre-Flight

- [ ] `cd submodules/deep-agents-ui && yarn install && yarn build` — verify clean build
- [ ] `cd submodules/deep-agents-ui && yarn dev` — verify dev server starts
- [ ] Check `docker compose up deep-agents-ui` still works after submodule move

---

## Phase 1: File Upload (Critical — Amin's #1 Priority)

### Task 1.1: Port multimodal utility types and helpers

**Objective:** Copy `lib/multimodal-utils.ts` from agent-custom-ui, adapting imports.

**Files:**
- Create: `submodules/deep-agents-ui/src/lib/multimodal-utils.ts`
- Source: `agent-custom-ui/src/lib/multimodal-utils.ts`

**Step 1:** Read source file, understand exports.
**Step 2:** Copy to `src/lib/multimodal-utils.ts`, fix any import paths.
**Step 3:** Verify no TypeScript errors: `npx tsc --noEmit`

---

### Task 1.2: Port file validation logic

**Objective:** Copy `lib/file-validation.ts` which handles file type checks, size limits, duplicate detection.

**Files:**
- Create: `submodules/deep-agents-ui/src/lib/file-validation.ts`
- Source: `agent-custom-ui/src/lib/file-validation.ts`

**Step 1:** Copy file, fix imports to point to `./multimodal-utils`.
**Step 2:** Add `@langchain/core` types if needed.
**Step 3:** Verify: `npx tsc --noEmit`

---

### Task 1.3: Port useFileUpload hook

**Objective:** Copy `hooks/use-file-upload.tsx` — the core drag-and-drop + file input hook.

**Files:**
- Create: `submodules/deep-agents-ui/src/app/hooks/useFileUpload.ts`
- Source: `agent-custom-ui/src/hooks/use-file-upload.tsx`

**Step 1:** Copy, adapt imports to point to `@/lib/file-validation` and `@/lib/multimodal-utils`.
**Step 2:** Ensure `ContentBlock` types are imported correctly from `@langchain/core/messages`.
**Step 3:** Verify: `npx tsc --noEmit`

---

### Task 1.4: Create ContentBlocksPreview component

**Objective:** Copy `ContentBlocksPreview.tsx` — shows thumbnails/previews of uploaded files.

**Files:**
- Create: `submodules/deep-agents-ui/src/app/components/ContentBlocksPreview.tsx`
- Source: `agent-custom-ui/src/components/thread/ContentBlocksPreview.tsx`

**Step 1:** Copy, adapt imports.
**Step 2:** Verify: `npx tsc --noEmit`

---

### Task 1.5: Create MultimodalPreview component

**Objective:** Copy `MultimodalPreview.tsx` — renders images/PDFs inline.

**Files:**
- Create: `submodules/deep-agents-ui/src/app/components/MultimodalPreview.tsx`
- Source: `agent-custom-ui/src/components/thread/MultimodalPreview.tsx`

**Step 1:** Copy, adapt imports.
**Step 2:** Verify: `npx tsc --noEmit`

---

### Task 1.6: Wire file upload into ChatInterface

**Objective:** Add file upload button (paperclip icon), drag-and-drop zone, and content blocks preview to `ChatInterface.tsx`.

**Files:**
- Modify: `submodules/deep-agents-ui/src/app/components/ChatInterface.tsx`

**Step 1:** Add import for `useFileUpload` hook and `Paperclip` icon from lucide-react.
**Step 2:** Add `useFileUpload()` call in component body.
**Step 3:** Add hidden `<input type="file">` and a paperclip button that triggers it.
**Step 4:** Add `onDragOver`/`onDrop` handlers to the chat container div.
**Step 5:** Show `ContentBlocksPreview` above the input when content blocks exist.
**Step 6:** Wire `contentBlocks` into the `sendMessage` call (pass as multimodal content).
**Step 7:** Verify: `yarn build` succeeds, manually test drag-and-drop of an image.

---

## Phase 2: Thread Rename & Delete

### Task 2.1: Port thread operation hooks

**Objective:** Copy thread edit/delete logic from agent-custom-ui.

**Files:**
- Create: `submodules/deep-agents-ui/src/app/hooks/useThreadOperations.ts`
- Source: `agent-custom-ui/src/components/thread/history/hooks/useThreadOperations.ts`
- Create: `submodules/deep-agents-ui/src/app/hooks/useThreadItemEdit.ts`
- Source: `agent-custom-ui/src/components/thread/history/hooks/useThreadItemEdit.ts`

**Step 1:** Copy both hooks, adapt imports.
**Step 2:** Verify: `npx tsc --noEmit`

---

### Task 2.2: Port thread utility functions

**Objective:** Copy `threadHelpers.ts`.

**Files:**
- Create: `submodules/deep-agents-ui/src/app/utils/threadHelpers.ts`
- Source: `agent-custom-ui/src/components/thread/history/utils/threadHelpers.ts`

**Step 1:** Copy, adapt imports.
**Step 2:** Verify: `npx tsc --noEmit`

---

### Task 2.3: Add edit/delete UI to ThreadList

**Objective:** Add right-click or long-press menu to ThreadList items with "Rename" and "Delete" options.

**Files:**
- Modify: `submodules/deep-agents-ui/src/app/components/ThreadList.tsx`

**Step 1:** Import `useThreadOperations` and `useThreadItemEdit` hooks.
**Step 2:** Add a dropdown menu (using existing shadcn/ui patterns) on each thread item.
**Step 3:** Wire "Rename" → inline edit mode (simple text input replacing the title).
**Step 4:** Wire "Delete" → confirmation dialog → call deleteThread operation.
**Step 5:** Call `onMutateReady` callback after mutations to refresh the list.
**Step 6:** Verify: `yarn build`, manual test — rename a thread, delete a thread.

---

## Phase 3: Chat Openers

### Task 3.1: Port ChatOpeners component

**Objective:** Copy `ChatOpeners.tsx` — carousel of suggested prompts shown when chat is empty.

**Files:**
- Create: `submodules/deep-agents-ui/src/app/components/ChatOpeners.tsx`
- Source: `agent-custom-ui/src/components/thread/ChatOpeners.tsx`

**Step 1:** Copy, adapt imports (use existing `cn` utility, add `framer-motion` if not present).
**Step 2:** Check if `framer-motion` is in package.json — if not, `yarn add framer-motion`.
**Step 3:** Verify: `npx tsc --noEmit`

---

### Task 3.2: Wire ChatOpeners into ChatInterface

**Objective:** Show ChatOpeners when messages array is empty. Define default openers.

**Files:**
- Modify: `submodules/deep-agents-ui/src/app/components/ChatInterface.tsx`

**Step 1:** Define default openers array (e.g., "Was kann AlphaRavis?", "Analysiere den Code in...", "Starte eine Recherche zu...").
**Step 2:** Import and render `<ChatOpeners>` when `processedMessages.length === 0`.
**Step 3:** On opener click, call `sendMessage(openerText)`.
**Step 4:** Verify: `yarn build`, open fresh chat → see openers, click one → message sends.

---

## Phase 4: Artifact System

### Task 4.1: Port artifact system

**Objective:** Copy the artifact portal system (`artifact.tsx`, `FullDescriptionModal.tsx`) that renders agent-generated HTML/React content in a dedicated panel.

**Files:**
- Create: `submodules/deep-agents-ui/src/app/components/artifact/artifact.tsx`
- Source: `agent-custom-ui/src/components/thread/artifact.tsx`
- Create: `submodules/deep-agents-ui/src/app/components/artifact/FullDescriptionModal.tsx`
- Source: `agent-custom-ui/src/components/thread/FullDescriptionModal.tsx`

**Step 1:** Copy artifact.tsx — this is a React portal system with context provider.
**Step 2:** Copy FullDescriptionModal.tsx.
**Step 3:** Adapt imports to new paths.
**Step 4:** Verify: `npx tsc --noEmit`

---

### Task 4.2: Wire artifact panel into layout

**Objective:** Add artifact panel (resizable sidebar) to the main layout when artifacts are present.

**Files:**
- Modify: `submodules/deep-agents-ui/src/app/page.tsx` (or `layout.tsx`)

**Step 1:** Import `ArtifactSlot` provider and wrap the chat interface.
**Step 2:** Add `ArtifactTitle` and `ArtifactContent` placeholder areas (can be a slide-out panel or resizable sidebar).
**Step 3:** Verify: `yarn build`, manually test with an artifact-producing message.

---

## Phase 5: Polish & Integration

### Task 5.1: Add dark mode support

**Objective:** Add next-themes dark mode toggle if not present.

**Files:**
- Modify: `submodules/deep-agents-ui/src/app/layout.tsx`

**Step 1:** Check if dark mode already works (deep-agents-ui uses Radix colors, may already support it).
**Step 2:** If not, add `next-themes` ThemeProvider.
**Step 3:** Add a theme toggle button to the header.

---

### Task 5.2: Verify Docker build

**Objective:** Ensure `docker compose build deep-agents-ui` succeeds after all changes.

**Step 1:** Run `docker compose build deep-agents-ui`.
**Step 2:** If failing, debug Dockerfile path or dependency issues.
**Step 3:** Run `docker compose up deep-agents-ui` and verify on `localhost:3000`.

---

### Task 5.3: Update docs

**Objective:** Update documentation to reflect new UI status.

**Files:**
- Modify: `docs/ALPHARAVIS_ARCHITECTURE.md` — update deep-agents-ui description
- Modify: `docs/ALPHARAVIS_CHANGES.md` — add entry for fork + migration
- Modify: `docs/ALPHARAVIS_OPEN_TASKS.md` — mark UI migration as complete

---

## Task Summary

| Phase | Tasks | Priority | Est. Time |
|---|---|---|---|
| 1. File Upload | 6 | CRITICAL | ~2-3h |
| 2. Thread Rename/Delete | 3 | HIGH | ~1h |
| 3. Chat Openers | 2 | MEDIUM | ~30min |
| 4. Artifact System | 2 | MEDIUM | ~1h |
| 5. Polish & Integration | 3 | LOW | ~30min |
| **Total** | **16** | | **~5-6h** |
