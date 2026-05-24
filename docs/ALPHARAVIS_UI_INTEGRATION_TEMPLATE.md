# AlphaRavis UI Integration Template

This is the canonical contract for browser UIs that connect directly to the
AlphaRavis LangGraph brain. Use it when porting or forking UIs such as
`deep-agents-ui`, AionUi, or a future custom UI.

## Goal

A UI should not guess AlphaRavis wiring. It should know the expected environment
variables, LangGraph SDK calls, thread lifecycle, attachment shape, and state
fields up front.

## Expected Runtime

- Primary UI: `submodules/deep-agents-ui/`
- Browser port: `3000`
- LangGraph API service: `langgraph-api`
- Graph id: `alpha_ravis`
- Preferred verification: `docker compose build --no-cache deep-agents-ui`

## Environment Variables

Use public `NEXT_PUBLIC_*` variables only for values safe to expose in the
browser. Do not put secrets into UI env vars.

```text
NEXT_PUBLIC_API_URL=http://localhost:2024
NEXT_PUBLIC_LANGGRAPH_API_URL=http://localhost:2024
NEXT_PUBLIC_GRAPH_ID=alpha_ravis
NEXT_PUBLIC_ASSISTANT_ID=<resolved assistant id or empty to discover by graph>
```

Recommended behavior:

1. Prefer an explicit assistant id when provided.
2. Otherwise search assistants for `graph_id == "alpha_ravis"`.
3. Fail visibly when no assistant is found; do not silently create a wrong one.

## LangGraph SDK Shape

The UI may use the LangGraph JS SDK directly. The expected calls are:

```ts
const client = new Client({ apiUrl: process.env.NEXT_PUBLIC_LANGGRAPH_API_URL });

const assistants = await client.assistants.search({
  graphId: "alpha_ravis",
  limit: 100,
});

const thread = await client.threads.create();
await client.threads.get(thread.thread_id);
await client.threads.getState(thread.thread_id);
await client.threads.update(thread.thread_id, { metadata: { title } });
await client.threads.delete(thread.thread_id);

const stream = client.runs.stream(thread.thread_id, assistantId, {
  input: { messages: [message] },
  streamMode: ["values", "messages", "updates"],
});
```

Exact SDK parameter names can differ between SDK releases; inspect the installed
`@langchain/langgraph-sdk` before changing code.

## Thread Lifecycle

### New Chat

1. Create or select a thread.
2. Send the first user message through `runs.stream`.
3. Store the returned thread id in URL/query state, e.g. `threadId`.
4. Revalidate the thread list after the first response starts or completes.

### Load Existing Chat

1. Read `threadId` from URL/query state.
2. Load thread state.
3. Render messages from state/checkpoint.
4. Revalidate when stream completes or an interrupt resolves.

### Rename Thread

1. Enter inline edit mode.
2. `Enter` saves.
3. `Escape` cancels.
4. Empty title is rejected in UI.
5. No API call when the title did not change.
6. Show loading state while `client.threads.update` is pending.
7. Revalidate the thread list after success.

### Delete Thread

1. Ask for confirmation before destructive delete.
2. Show loading state while `client.threads.delete` is pending.
3. Revalidate the thread list after success.
4. If the active thread was deleted, select the next available thread or clear
   `threadId` when no thread remains.

## Message And Attachment Shape

Text-only user message:

```ts
{
  type: "human",
  content: "hello"
}
```

Multimodal user message:

```ts
{
  type: "human",
  content: [
    { type: "text", text: "analyse this" },
    {
      type: "image",
      mimeType: "image/png",
      data: "<base64 without data-url prefix>",
      metadata: { name: "screenshot.png" }
    },
    {
      type: "file",
      mimeType: "application/pdf",
      data: "<base64 without data-url prefix>",
      metadata: { filename: "document.pdf" }
    }
  ]
}
```

Upload UX rules:

- Support file picker, drag/drop, and paste.
- Validate MIME types before sending.
- Reject duplicates within one pending message.
- Rename generic pasted images (`image.png`, empty name) to a timestamped
  `pasted-image-...` filename before duplicate checks.
- Show processing and error toasts.
- Provide `remove` and `remove all` controls before send.

## AlphaRavis State Shape

The UI should tolerate missing fields and render progressively. Current useful
fields include:

```ts
type AlphaRavisState = {
  messages?: unknown[];
  todos?: Array<{
    id: string;
    content: string;
    status: "pending" | "in_progress" | "completed" | "cancelled";
  }>;
  files?: Record<string, string>;
  ui?: unknown[];
  skills?: string[];
  artifacts?: unknown[];
};
```

Rules:

- Treat `messages` as canonical conversation output.
- Treat `todos` as agent task/status UI, not as user-editable content.
- Treat `files` as state files produced by the agent; show preview/edit controls
  separately from pending upload attachments.
- Treat `ui` and `artifacts` as optional renderable extras.
- Do not crash when any field is absent.

## Preview Policy

Use the lightweight `DiffViewer` for:

- patch preview
- agent changes review
- small code comparisons
- fast UI feedback

Load Monaco only after explicit user action, e.g. an `Open Monaco editor`
button, for:

- real code preview
- manual code correction
- larger code files
- future file editing workflows
- side-by-side editor behavior similar to VS Code

Default code preview must remain a cheap `<pre>` view so Monaco is not pulled
into the initial UI interaction path.

## Smoke Checklist

Manual browser smoke test after UI changes:

- App opens on port `3000`.
- Empty chat openers render.
- Sending a text message starts a stream.
- File picker attaches image/PDF.
- Drag/drop attaches image/PDF.
- Paste attaches screenshot/image and gives it a useful name.
- Attachment remove and remove-all work before send.
- State file preview opens.
- Diff preview uses the lightweight `DiffViewer`.
- Code preview initially uses cheap text preview.
- Monaco loads only after pressing `Open Monaco editor`.
- Thread rename: Enter saves, Escape cancels, empty title rejected.
- Thread delete confirms first and recovers when active thread is deleted.

## Verification Commands

```bash
docker run --rm -v "$PWD/submodules/deep-agents-ui:/app" -w /app node:20-alpine yarn install --frozen-lockfile
docker run --rm -v "$PWD/submodules/deep-agents-ui:/app" -w /app node:20-alpine yarn lint
docker compose build --no-cache deep-agents-ui
```

If local host tooling is missing, use Docker-based verification. Do not modify
private `.env` files just to run the UI.
