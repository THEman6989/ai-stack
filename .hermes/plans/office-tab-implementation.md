# AlphaRavis Office Tab — Implementation Plan

> Standalone plan. Self-contained. No external chat context needed.
> Start here in a fresh session: read this file, then implement.

## Overview

Integrate **OfficeCLI** (`iOfficeAI/OfficeCLI`, Apache 2.0) into AlphaRavis
as a dedicated **"Office" tab** in the Deep Agents UI. OfficeCLI is a single
binary (~30 MB) that gives AI agents full control over Word (.docx), Excel
(.xlsx), and PowerPoint (.pptx) — create, read, edit, render, validate.

**Submodule**: `submodules/OfficeCLI` (already cloned)

**Goal**: Users upload/create/edit Office documents in a dedicated UI tab.
The LangGraph agent calls OfficeCLI via the `terminal` tool. Live preview
renders via OfficeCLI's built-in HTML/PNG engine.

## Why not mammoth / officeparser / python-docx?

| Tool | Can do |
|------|--------|
| mammoth | DOCX → HTML only |
| officeparser | Read text from DOCX/XLSX/PPTX |
| python-docx | Create/edit DOCX (Python only) |
| **OfficeCLI** | **All of the above + 3-Layer DOM + Formulas + Pivot + Render + Validate + Merge + MCP + Watch** |

OfficeCLI replaces all npm/Python Office libraries. One binary. No dependencies.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Deep Agents UI (Next.js)                       │
│  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────────┐   │
│  │  Chat   │ │ Threads  │ │  Tasks   │ │  OFFICE  (new tab) │   │
│  │  Tab    │ │  Tab     │ │  Tab     │ │                    │   │
│  │         │ │          │ │          │ │ ┌────────────────┐ │   │
│  │         │ │          │ │          │ │ │ Document List  │ │   │
│  │         │ │          │ │          │ │ │ - Upload       │ │   │
│  │         │ │          │ │          │ │ │ - Create New   │ │   │
│  │         │ │          │ │          │ │ │ - Recent       │ │   │
│  │         │ │          │ │          │ │ └────────────────┘ │   │
│  │         │ │          │ │          │ │ ┌────────────────┐ │   │
│  │         │ │          │ │          │ │ │ Live Preview   │ │   │
│  │         │ │          │ │          │ │ │ (HTML iframe)  │ │   │
│  │         │ │          │ │          │ │ │ port 26315     │ │   │
│  │         │ │          │ │          │ │ └────────────────┘ │   │
│  │         │ │          │ │          │ │ ┌────────────────┐ │   │
│  │         │ │          │ │          │ │ │ Agent Chat     │ │   │
│  │         │ │          │ │          │ │ │ "Add a slide"  │ │   │
│  │         │ │          │ │          │ │ └────────────────┘ │   │
│  └─────────┘ └──────────┘ └──────────┘ └────────────────────┘   │
│                                                                  │
│  State: { files: Record<id, {name, type, path, previewUrl}> }   │
└──────────────────────────┬───────────────────────────────────────┘
                           │ LangGraph SDK (REST/SSE)
┌──────────────────────────┼───────────────────────────────────────┐
│                    LangGraph API                                  │
│  ┌──────────────────────┼────────────────────────────────────┐   │
│  │              AlphaRavis Agent                              │   │
│  │                                                           │   │
│  │  terminal("officecli add deck.pptx / --type slide ...")   │   │
│  │  terminal("officecli view deck.pptx screenshot -o /tmp/") │   │
│  │  → Agent sees PNG, fixes layout                           │   │
│  │                                                           │   │
│  │  Result: file path → UI offers download                   │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                  │
│  /usr/local/bin/officecli  (single binary, embedded .NET 10)    │
│  /workspace/office-output/  (mounted volume for documents)      │
└──────────────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: Docker Integration (Day 1)

**Goal**: OfficeCLI binary available in langgraph-api container.

**1.1 Add OfficeCLI to Dockerfile**

File: `docker/langgraph-api/Dockerfile` (or wherever langgraph-api is built)

```dockerfile
# Install OfficeCLI — single binary, ~30 MB
RUN curl -fsSL https://raw.githubusercontent.com/iOfficeAI/OfficeCLI/main/install.sh | bash

# Verify
RUN officecli --version
```

Or manual download (more deterministic):

```dockerfile
ARG OFFICECLI_VERSION=latest
RUN ARCH=$(uname -m | sed 's/x86_64/x64/;s/aarch64/arm64/') && \
    curl -fsSL -o /usr/local/bin/officecli \
    "https://github.com/iOfficeAI/OfficeCLI/releases/${OFFICECLI_VERSION}/download/officecli-linux-${ARCH}" && \
    chmod +x /usr/local/bin/officecli
```

**1.2 Add output volume**

In `docker-compose.yml`:

```yaml
langgraph-api:
  volumes:
    - office_output:/workspace/office-output
  environment:
    - OFFICE_OUTPUT_DIR=/workspace/office-output

volumes:
  office_output:
```

**1.3 Install headless browser for PNG rendering**

OfficeCLI needs a headless browser for `view screenshot` and `watch`:

```dockerfile
RUN apt-get update && apt-get install -y chromium-browser && rm -rf /var/lib/apt/lists/*
ENV OFFICECLI_CHROMIUM_PATH=/usr/bin/chromium-browser
```

**1.4 Verify in container**

```bash
docker compose exec langgraph-api officecli --version
docker compose exec langgraph-api officecli create /tmp/test.pptx
docker compose exec langgraph-api officecli view /tmp/test.pptx stats
```

---

### Phase 2: Agent Tool Integration (Day 2-3)

**Goal**: LangGraph agent can call OfficeCLI naturally.

**2.1 The agent already has `terminal` tool — OfficeCLI works immediately**

No code changes needed for basic usage. The agent can:

```bash
# Create a presentation
officecli create /workspace/office-output/deck.pptx

# Add slides
officecli add /workspace/office-output/deck.pptx / --type slide --prop title="Q4 Report"

# Add content
officecli add /workspace/office-output/deck.pptx '/slide[1]' --type shape \
  --prop text="Revenue: $4.2M" --prop x=2cm --prop y=5cm

# View outline
officecli view /workspace/office-output/deck.pptx outline

# Export PNG for vision feedback
officecli view /workspace/office-output/deck.pptx screenshot -o /tmp/deck.png

# Validate
officecli validate /workspace/office-output/deck.pptx
```

**2.2 Optional: OfficeCLI MCP Server**

OfficeCLI has a built-in MCP server. If LangGraph's MCP client can connect:

```bash
# In langgraph-api container:
officecli mcp start --port 26316 &
```

Then register in `langgraph-app/mcp.json`:

```json
{
  "officecli": {
    "transport": "stdio",
    "command": "officecli",
    "args": ["mcp", "start"]
  }
}
```

This exposes all OfficeCLI operations as structured MCP tools
(no shell parsing needed, JSON natively).

**2.3 Agent prompt addition**

Add to the agent's system prompt (in `prompt_assembly.py` or graph state):

```
You have access to OfficeCLI for creating and editing Office documents.
Key commands:

CREATE:  officecli create <file>.docx|.xlsx|.pptx
ADD:     officecli add <file> <path> --type <element> --prop key=value
SET:     officecli set <file> <path> --prop key=value
VIEW:    officecli view <file> outline|text|annotated|html|screenshot
GET:     officecli get <file> <path> --json
QUERY:   officecli query <file> "selector" --json
MERGE:   officecli merge template.docx output.docx '{"key":"value"}'
WATCH:   officecli watch <file>  (live preview at http://localhost:26315)
VALIDATE: officecli validate <file>

Always prefer L1 (view) → L2 (add/set) → L3 (raw). Use --json for
deterministic output. Check issues with `view <file> issues --json`.

Output directory: /workspace/office-output/
```

---

### Phase 3: UI — Office Tab (Day 3-7)

**Goal**: New "Office" tab in Deep Agents UI with document list, live preview, agent chat.

**3.1 Tab Navigation**

File: `src/app/page.tsx` — add Office tab alongside Chat/Threads/Tasks:

```tsx
const TABS = [
  { id: "chat", label: "Chat", icon: MessageSquare },
  { id: "threads", label: "Threads", icon: MessagesSquare },
  { id: "tasks", label: "Tasks", icon: CheckSquare },
  { id: "office", label: "Office", icon: FileText },  // NEW
] as const;
```

**3.2 Office Page Component**

File: `src/app/components/OfficePanel.tsx` (new)

```tsx
"use client";

import { useState } from "react";
import { FileText, Plus, Upload, Eye, Download, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useChatContext } from "@/providers/useChatContext";

interface OfficeDocument {
  id: string;
  name: string;
  type: "docx" | "xlsx" | "pptx";
  path: string;       // /workspace/office-output/doc.docx
  previewUrl?: string; // http://localhost:26315 (watch mode)
  createdAt: Date;
}

export function OfficePanel() {
  const [documents, setDocuments] = useState<OfficeDocument[]>([]);
  const [activeDoc, setActiveDoc] = useState<OfficeDocument | null>(null);
  const [previewMode, setPreviewMode] = useState<"none" | "html" | "watch">("none");

  const handleCreate = async (type: "docx" | "xlsx" | "pptx") => {
    // Send message to agent: "Create a new blank {type} document"
    // Agent calls: officecli create /workspace/office-output/new.{type}
    // UI refreshes document list
  };

  const handleUpload = async (file: File) => {
    // Upload to server, parse with officecli view <file> text
    // Show in document list
  };

  const startWatch = async (doc: OfficeDocument) => {
    // Agent calls: officecli watch <file>
    // Preview URL: http://localhost:26315
    setActiveDoc({ ...doc, previewUrl: "http://localhost:26315" });
    setPreviewMode("watch");
  };

  return (
    <div className="flex h-full">
      {/* Document List Sidebar */}
      <div className="w-64 border-r p-3">
        <div className="flex gap-2 mb-3">
          <Button size="sm" onClick={() => handleCreate("pptx")}>
            <Plus className="h-4 w-4" /> PPTX
          </Button>
          <Button size="sm" onClick={() => handleCreate("docx")}>
            <Plus className="h-4 w-4" /> DOCX
          </Button>
          <Button size="sm" onClick={() => handleCreate("xlsx")}>
            <Plus className="h-4 w-4" /> XLSX
          </Button>
        </div>
        <ScrollArea>
          {documents.map((doc) => (
            <div
              key={doc.id}
              className={`p-2 cursor-pointer rounded ${activeDoc?.id === doc.id ? "bg-accent" : ""}`}
              onClick={() => setActiveDoc(doc)}
            >
              <div className="flex items-center gap-2">
                <FileText className="h-4 w-4" />
                <span className="text-sm truncate">{doc.name}</span>
              </div>
            </div>
          ))}
        </ScrollArea>
      </div>

      {/* Main Area */}
      <div className="flex-1 flex flex-col">
        {activeDoc ? (
          <>
            {/* Toolbar */}
            <div className="flex gap-2 p-2 border-b">
              <Button size="sm" variant="outline" onClick={() => startWatch(activeDoc)}>
                <Eye className="h-4 w-4" /> Live Preview
              </Button>
              <Button size="sm" variant="outline" onClick={() => downloadDoc(activeDoc)}>
                <Download className="h-4 w-4" /> Download
              </Button>
              <Button size="sm" variant="outline" onClick={() => deleteDoc(activeDoc)}>
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>

            {/* Preview / Chat Split */}
            <div className="flex-1 flex">
              {/* Live Preview (iframe) */}
              {previewMode === "watch" && activeDoc.previewUrl && (
                <div className="flex-1">
                  <iframe
                    src={activeDoc.previewUrl}
                    className="w-full h-full border-0"
                    title="Office Live Preview"
                  />
                </div>
              )}

              {/* Agent Chat (embedded) */}
              <div className="w-96 border-l">
                <div className="p-2 border-b text-sm font-medium">
                  Agent: {activeDoc.name}
                </div>
                <div className="p-3">
                  <textarea
                    className="w-full p-2 border rounded text-sm"
                    placeholder="Describe changes... (e.g. 'Add a slide with Q4 numbers')"
                    rows={3}
                  />
                  <Button size="sm" className="mt-2">Send to Agent</Button>
                </div>
              </div>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-muted-foreground">
            Select or create a document to start
          </div>
        )}
      </div>
    </div>
  );
}
```

**3.3 Document State in LangGraph**

Add `office_documents` to the LangGraph state:

```python
# In agent_graph.py State definition
class AlphaRavisState(TypedDict):
    # ... existing fields ...
    office_documents: list[dict]  # [{id, name, type, path, preview_port}]
```

**3.4 File Upload → Office Parsing**

File: `src/lib/office-parser.ts` (new, thin wrapper)

```typescript
// Uses OfficeCLI on the server side via API
export async function parseOfficeFile(file: File): Promise<{
  type: "docx" | "xlsx" | "pptx";
  text: string;
  outline?: string;
}> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch("/api/office/parse", {
    method: "POST",
    body: formData,
  });
  return res.json();
}
```

API route (Next.js or in LangGraph):

```python
# In LangGraph API or Next.js API route
import subprocess, json

def parse_office_file(filepath: str) -> dict:
    ext = filepath.rsplit(".", 1)[-1]
    result = subprocess.run(
        ["officecli", "get", filepath, "/", "--depth", "2", "--json"],
        capture_output=True, text=True, timeout=30
    )
    return json.loads(result.stdout)
```

---

### Phase 4: Watch Mode & Live Preview (Day 4-5)

**Goal**: Real-time preview as agent edits documents.

**4.1 How `officecli watch` works**

```bash
officecli watch /workspace/office-output/deck.pptx
# → Starts HTTP server on http://localhost:26315
# → Every `add`/`set`/`remove` auto-refreshes the browser
# → Excel: inline cell editing, drag-to-reposition charts
```

**4.2 Docker networking**

Expose port 26315 from the container:

```yaml
# docker-compose.yml
langgraph-api:
  ports:
    - "127.0.0.1:26315:26315"
```

Or use a reverse proxy through the existing API bridge:

```nginx
# In api-bridge or nginx config
location /office-preview/ {
    proxy_pass http://langgraph-api:26315/;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

**4.3 UI iframe**

```tsx
// Secure preview URL through bridge
const previewUrl = `/office-preview/?doc=${encodeURIComponent(doc.path)}`;
<iframe src={previewUrl} className="w-full h-full" />
```

**4.4 Agent → Preview Loop**

```
1. Agent runs: officecli watch /workspace/office-output/deck.pptx
2. UI shows iframe at /office-preview/
3. User types "Add a red title 'Q4 Results'"
4. Agent runs: officecli add deck.pptx / --type slide --prop title="Q4 Results"
5. Watch server detects file change → pushes update to iframe
6. User sees new slide appear instantly
```

---

### Phase 5: Advanced Features (Week 2-3)

**5.1 Template Gallery**

Pre-built templates stored in `/workspace/office-output/templates/`:

```bash
# Agent creates from template
officecli merge /workspace/office-output/templates/report.docx \
  /workspace/office-output/report-2026.docx \
  '{"title":"Annual Report 2026","author":"AlphaRavis"}'
```

**5.2 Batch Operations**

```bash
# Agent generates 100 invoices
for i in $(seq 1 100); do
  officecli merge invoice-template.docx "invoice-$i.docx" \
    "{\"number\":\"$i\",\"total\":\"$((RANDOM % 10000))\"}"
done
```

**5.3 Document Validation Pipeline**

```bash
# Pre-delivery check
officecli validate report.docx
officecli view report.docx issues --json
# Fix issues automatically
officecli set report.docx /body/p[1]/r[1] --prop font=Arial
```

**5.4 Round-trip Learning**

Agent extracts structure from existing template:

```bash
officecli dump template.docx -o blueprint.json
# Agent studies blueprint.json
# Agent generates variations:
officecli batch new-report.docx --input variations.json
```

**5.5 MCP Tool Registration**

If LangGraph's MCP client supports external servers:

```json
// langgraph-app/mcp.json
{
  "officecli": {
    "command": "officecli",
    "args": ["mcp", "start"],
    "transport": "stdio"
  }
}
```

This gives the agent typed tool calls instead of raw shell commands.

---

## OfficeCLI Command Cheat Sheet

### Create
```bash
officecli create presentation.pptx
officecli create document.docx
officecli create spreadsheet.xlsx
```

### View (L1 — Read)
```bash
officecli view file.pptx outline        # Slide structure
officecli view file.pptx text           # Plain text extraction
officecli view file.pptx annotated      # With element tags
officecli view file.pptx stats          # Page/slide counts
officecli view file.pptx issues --json  # Problems found
officecli view file.pptx html           # Rendered HTML
officecli view file.pptx screenshot     # PNG per slide
officecli view file.pptx screenshot -o /tmp/out --page 1-3
officecli view file.docx --render html  # Word HTML rendering
```

### Get / Query (L2 — DOM)
```bash
officecli get file.pptx / --depth 2 --json          # Full tree
officecli get file.pptx '/slide[1]/shape[1]' --json # One element
officecli query file.docx "paragraph[style=Heading1]"
officecli query file.xlsx "cell[value>1000]"
```

### Add / Set / Remove (L2 — Mutate)
```bash
# PowerPoint
officecli add deck.pptx / --type slide --prop title="Hello" --prop background=1A1A2E
officecli add deck.pptx '/slide[1]' --type shape --prop text="Content" --prop x=2cm --prop y=5cm
officecli add deck.pptx '/slide[1]' --type picture --prop src=chart.png --prop x=5cm --prop y=5cm
officecli set deck.pptx '/slide[1]/shape[1]' --prop text="Updated" --prop color=FF0000
officecli remove deck.pptx '/slide[2]'
officecli move deck.pptx '/slide[3]' --to / --index 1

# Word
officecli add report.docx /body --type paragraph --prop text="Chapter 1"
officecli add report.docx /body --type table --prop rows=3 --prop cols=4
officecli set report.docx /body/p[1]/r[1] --prop bold=true --prop size=14
officecli add report.docx /body --type picture --prop src=logo.png

# Excel
officecli add budget.xlsx / --type sheet --prop name="Q1"
officecli set budget.xlsx '/Sheet1/A1' --prop value="Revenue"
officecli set budget.xlsx '/Sheet1/B2' --prop value="=SUM(B3:B10)"
officecli add budget.xlsx '/Sheet1' --type pivottable \
  --prop source='Data!A1:E1000' --prop rows='Category' --prop values='Amount:sum'
officecli add budget.xlsx '/Sheet1' --type chart --prop type=bar \
  --prop source='A1:B10' --prop x=5cm --prop y=2cm
```

### Template Merge
```bash
officecli merge template.docx output.docx '{"name":"Acme","date":"2026-05-24"}'
officecli merge template.pptx output.pptx data.json
```

### Watch (Live Preview)
```bash
officecli watch file.pptx    # → http://localhost:26315
# Every add/set/remove auto-refreshes browser
```

### Validate
```bash
officecli validate report.docx
officecli view report.docx issues --json
```

### Dump / Batch (Round-trip)
```bash
officecli dump template.docx -o blueprint.json
officecli batch new.docx --input blueprint.json
```

### Resident Mode
```bash
officecli open report.docx       # Keep in memory
officecli set report.docx ...    # Instant, no file I/O
officecli set report.docx ...    # Multiple fast edits
officecli close report.docx      # Save and release
```

---

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `submodules/OfficeCLI/` | Git submodule (already cloned) |
| `src/app/components/OfficePanel.tsx` | Office tab UI component |
| `src/lib/office-parser.ts` | OfficeCLI wrapper for upload parsing |
| `docs/AIONUI_OFFICE_INTEGRATION.md` | Architecture doc (already exists) |
| `.hermes/plans/office-tab-implementation.md` | This plan |

### Modified Files

| File | Change |
|------|--------|
| `docker/langgraph-api/Dockerfile` | Install OfficeCLI binary + chromium |
| `docker-compose.yml` | Add office_output volume, expose port 26315 |
| `src/app/page.tsx` | Add "Office" tab to navigation |
| `langgraph-app/agent_graph.py` | Add `office_documents` to state |
| `langgraph-app/prompt_assembly.py` | Add OfficeCLI instructions to system prompt |
| `langgraph-app/mcp.json` | Optional: register OfficeCLI MCP server |

---

## Verification Checklist

- [ ] `docker compose exec langgraph-api officecli --version`
- [ ] `officecli create /tmp/test.pptx && officecli view /tmp/test.pptx stats`
- [ ] Agent can generate a PPTX: "Create a 3-slide presentation about AI"
- [ ] Agent can see its output: `officecli view ... screenshot` → vision model feedback
- [ ] Office tab appears in UI at http://localhost:3000
- [ ] File upload: .docx parsed via OfficeCLI, text shown in preview
- [ ] Live preview: `officecli watch` → iframe shows real-time updates
- [ ] Template merge: `officecli merge template.docx out.docx '{"key":"val"}'`
- [ ] Download button: user downloads generated .pptx/.docx/.xlsx
- [ ] `docker compose build --no-cache langgraph-api` passes
