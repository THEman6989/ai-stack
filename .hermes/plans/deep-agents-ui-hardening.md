# Deep Agents UI Hardening Plan

> **For Hermes:** Implement directly in the AlphaRavis `submodules/deep-agents-ui` fork; keep changes minimal and verify with Docker build/lint.

**Goal:** Fix review shortcomings and apply the 12-point UI hardening pass without rewriting the fork.

**Architecture:** Keep the existing deep-agents-ui architecture. The lightweight DiffViewer remains the default for patch/agent-change review. Monaco is available only behind an explicit button so the heavy editor is loaded on demand. Thread and upload flows get small robustness improvements. The UI integration contract is documented in a canonical template.

**Tech Stack:** Next.js 16, React 19, TypeScript, Tailwind, LangGraph SDK, Yarn 1.

---

## Tasks

1. Inspect agent-template/deep-agents architecture and current fork state.
   - Files: `submodules/deep-agents-ui/README.md`, `src/app/hooks/useChat.ts`, `src/app/components/ChatInterface.tsx`.
   - Result: no old `agent-template` implementation was found; created `docs/ALPHARAVIS_UI_INTEGRATION_TEMPLATE.md` and `.hermes/templates/alpha-ravis-ui-integration-template.md` as the canonical future template.

2. Fix file upload wiring.
   - Modified: `src/app/components/ChatInterface.tsx`.
   - Wired `dropRef`, `handlePaste`, and `dragOver` directly into the composer.

3. Clean dependencies and lockfile.
   - Modified: `submodules/deep-agents-ui/package.json`, `yarn.lock`.
   - Added `monaco-editor`, kept `@monaco-editor/react`, removed unused `diff` / `@types/diff`.

4. Keep DiffViewer lightweight.
   - Modified: `src/app/components/DiffViewer.tsx`, `src/lib/diff-utils.ts`.
   - DiffViewer remains for patch preview, agent-change review, small comparisons, and fast UI.

5. Lazy-load Monaco only on user action.
   - Modified: `src/app/components/FilePreviewPanel.tsx`.
   - Code files now open as a cheap `<pre>` preview first.
   - Monaco loads only after pressing `Open Monaco editor`.

6. Harden Docker reproducibility.
   - Modified: `docker/deep-agents-ui/Dockerfile`.
   - Switched back to `yarn install --frozen-lockfile` now that `yarn.lock` is current.
   - `.dockerignore` keeps build context small.

7. Improve upload UX.
   - Modified: `src/app/hooks/useFileUpload.ts`, `src/lib/file-validation.ts`, `ChatInterface.tsx`.
   - Added processing state, success/error toasts, remove-all control, upload button disabling during processing, and timestamp names for generic pasted images.

8. Harden thread rename/delete UX.
   - Modified: `src/app/components/ThreadList.tsx`.
   - Empty rename rejected; unchanged rename skips API call; loading states added; Escape-cancel suppresses blur-save; duplicate pending rename saves are guarded; metadata title display is honored in the thread list; delete keeps confirmation and recovers if the active thread was deleted.

9. Document UI integration contract.
   - Added: `docs/ALPHARAVIS_UI_INTEGRATION_TEMPLATE.md`.
   - Added: `.hermes/templates/alpha-ravis-ui-integration-template.md` pointer.

10. Update operator docs and open tasks.
    - Update `ALPHARAVIS_CHANGES.md`, `ALPHARAVIS_ARCHITECTURE.md`, `ALPHARAVIS_USAGE_NOTES.md`, and `ALPHARAVIS_OPEN_TASKS.md`.

11. Verify build/lint.
    - ✅ `docker run --rm -v "$PWD/submodules/deep-agents-ui:/app" -w /app node:20-alpine yarn install --frozen-lockfile` passes.
    - ✅ `docker run --rm -v "$PWD/submodules/deep-agents-ui:/app" -w /app node:20-alpine yarn lint` passes with 0 errors / 0 warnings (all 7 Fast Refresh warnings fixed via extraction of non-component exports).
    - ✅ `git diff --check` and `git -C submodules/deep-agents-ui diff --check` pass.
    - ✅ Static added-line scan found no secrets/dangerous eval/shell patterns.
    - ✅ Independent review passed after fixing the ThreadList blur/Escape rename edge case.
    - ✅ `docker compose build --no-cache deep-agents-ui` passes with `yarn install --frozen-lockfile` and a small Docker build context.

12. Manual browser smoke.
    - Still requires running stack/browser on port `3000`.
    - Checklist lives in `docs/ALPHARAVIS_UI_INTEGRATION_TEMPLATE.md`.

13. Fix Fast Refresh warnings (0 warnings).
    - Extracted `buttonVariants` → `src/components/ui/button-variants.ts`.
    - Extracted `ClientContext` + `useClient` → `src/providers/ClientContext.ts` + `src/providers/useClient.ts`.
    - Extracted `ChatContext` + `useChatContext` → `src/providers/ChatContext.ts` + `src/providers/useChatContext.ts`.
    - Extracted `ArtifactSlotContext` + hooks → `src/app/components/artifact-context.ts` + `src/app/components/useArtifact.tsx`.
    - Updated all imports (page.tsx, ThreadList.tsx, useChat.ts, TasksFilesSidebar.tsx, ChatInterface.tsx).
    - ✅ `yarn lint`: 0 errors, 0 warnings.

14. Playwright smoke tests (scaffold).
    - Added: `e2e/smoke.spec.ts` — covers thread rename/delete, file upload, preview panel, Monaco button, paste, remove-all, processing state.
    - Added: `playwright.config.ts` — Chromium project, CI-ready.
    - Added: `yarn test:e2e`, `test:e2e:headed`, `test:e2e:ui` scripts.
    - Added: `.github/workflows/ci.yml` — lint + build job, e2e job (disabled by default).
    - ⚠️ `@playwright/test` not yet in devDependencies (`yarn.lock` regeneration needed before first run).
    - ⚠️ `e2e/` and `playwright.config.ts` excluded from `tsconfig.json` until dep is installed.
