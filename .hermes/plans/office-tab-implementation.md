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

### Phase 2b: Lazy MCP — Token-effiziente Tool-Registrierung (Day 3-4)

**Problem**: Aktuell lädt `mcp_client.py` → `load_robust_mcp_tools()` ALLE
MCP-Tools **eager** beim Graph-Start (`make_graph()` in Zeile 14125). Jeder
Request — auch "hi" — bekommt sämtliche Tool-Definitionen in den System-Prompt.
Bei OfficeCLI mit 20+ Commands sind das ~2000 Tokens Overhead PRO Request.

**CLI vs MCP — Runtime-Vergleich:**

| | Direct CLI (`terminal`) | MCP eager (aktuell) | MCP lazy (Ziel) |
|---|---|---|---|
| **Prompt-Tokens** | ~50 (nur terminal) | ~2000+ (alle Tools immer) | 0 (Tools on-demand) |
| **Tool-Call-Qualität** | String-Parsing | Typisiert, validiert | Typisiert, validiert |
| **Agent-Denkarbeit** | Hoch (Syntax lernen) | Niedrig (Schema da) | Niedrig (Schema da) |
| **Live Preview** | ✅ Gleich | ✅ Gleich | ✅ Gleich |
| **Fehler-Recovery** | stderr lesen | Structured JSON | Structured JSON |

**Fazit**: Direct CLI funktioniert, aber Lazy MCP ist das Optimum —
token-effizient wie CLI, präzise wie MCP. Eager MCP ist raus.

**Ziel-Architektur Lazy MCP:**

```
Graph-Start (make_graph):
  KEINE MCP-Tools laden
  
Request "Erstelle eine PPTX":
  Agent ruft: mcp_discover("officecli")
  → mcp_client lädt OfficeCLI-Tools ON-DEMAND
  → Tools werden dem aktuellen Request hinzugefügt
  → Nächster Request (ohne Office): KEINE OfficeCLI-Tools im Prompt
  
Cache: Geladene Tools bleiben für Session (TTL 5 min)
       Neue Requests ohne Office-Bezug: kein Overhead
```

**Implementierung in `mcp_client.py`:**

Neue Funktion `load_mcp_tools_lazy()`:

```python
# Module-level cache: server_name → (tools, expiry_timestamp)
_lazy_tool_cache: dict[str, tuple[list, float]] = {}
_LAZY_CACHE_TTL = 300  # 5 minutes

async def load_mcp_tools_lazy(
    server_name: str,
    stack: contextlib.AsyncExitStack,
) -> list[Any]:
    """Load MCP tools on-demand with TTL cache."""
    now = time.monotonic()
    if server_name in _lazy_tool_cache:
        tools, expiry = _lazy_tool_cache[server_name]
        if now < expiry:
            return tools
    
    # Load tools from server
    config = load_mcp_config()[0]
    server_config = config["mcpServers"][server_name]
    manager = RobustMCPServerManager(...)
    tools = await manager.connect(stack)
    
    _lazy_tool_cache[server_name] = (tools, now + _LAZY_CACHE_TTL)
    return tools
```

In `agent_graph.py` — `@tool` decorator für Discovery:

```python
@tool
async def mcp_discover(server_hint: str) -> str:
    """Load MCP tools from a server. Use when you need document/office capabilities.
    
    Known servers: officecli (Word/Excel/PowerPoint), pixelle (image gen)
    """
    tools = await load_mcp_tools_lazy(server_hint, _global_stack)
    # Dynamically register tools for current invocation
    for tool in tools:
        _register_tool_for_current_run(tool)
    return f"Loaded {len(tools)} tools from {server_hint}"
```

**Env-Var für graduelle Migration:**

```bash
# .env
ALPHARAVIS_MCP_LAZY=true           # Enable lazy loading
ALPHARAVIS_MCP_LAZY_EAGER_LIST=    # Servers to still load eagerly (empty = all lazy)
ALPHARAVIS_MCP_LAZY_CACHE_TTL=300  # Cache TTL in seconds
```

**Migration:**
1. `mcp_client.py`: `load_mcp_tools_lazy()` neue Funktion
2. `agent_graph.py`: `make_graph()` lädt KEINE Tools mehr eager
3. `agent_graph.py`: `mcp_discover` @tool für On-Demand-Loading
4. Alte `load_robust_mcp_tools()` bleibt für `ALPHARAVIS_MCP_LAZY=false`
5. Bestehende Server (Pixelle SSE) funktionieren unverändert

**Token-Ersparnis (geschätzt):**
- Ohne Office-Request: 0 Office-Tokens (statt ~2000)
- Mit Office-Request: ~2000 Tokens ONCE (dann gecached)
- Bei 100 Requests/Tag, 10% Office: 180K Tokens gespart

**Lazy-Granularität: Server-Level vs. Tool-Level**

MCP-Protokoll kann nur den GANZEN Server laden — `list_tools()` gibt immer
alle Tools zurück. Man kann nicht einzelne Tools anfragen.

ABER: Nach dem Laden können wir **selektiv injecten**:

```
Server-Verbindung (1x):
  mcp_discover("officecli")
  → Verbindet zu officecli MCP
  → list_tools() → 20 Tools (add, set, view, merge, watch, ...)
  → Cache: alle 20 für 5min

Tool-Injection (pro Request):
  Agent sagt "add a slide"
  → System erkennt: nur "add" wird gebraucht
  → Injiziert NUR officecli_add in den Prompt
  → 18 andere Office-Tools bleiben draussen
  
  Agent sagt "create full presentation with pivot table"
  → System injiziert: create, add, set, merge, view → 5 Tools
```

Das heisst: Server-Lazy (keine Verbindung bis gebraucht) + Tool-Lazy
(nur relevante Tools in den Prompt). **Token-Kosten pro Request = nur
die 2-5 Tools die der Agent tatsächlich braucht.**

**Agent-Referenz-Dokumentation:**
`docs/OFFICECLI_AGENT_REFERENCE.md` — kompakte CLI-Referenz für den Agent.
Kann in den System-Prompt injected werden (∼3K chars, ∼750 Tokens).
Deckt alle Befehle, Path-Syntax, Units/Colors, Error-Recovery.

**Integration via bestehendes Toolset-System — KEIN separater CLI/ MCP-Mode nötig**

AlphaRavis hat bereits ein Lazy-Toolset-System in `alpharavis_toolsets.py`:

```
Agent sieht NUR Toolset-Registry (Kategorien):
  coding/read: Read repository, artifacts, reviewed skills...
  media/image: Generate images through Pixelle...
  system/power: Owner-gated power lifecycle...

Erst wenn Agent ein Toolset anfordert → materialize_toolsets() → echte Tools injected
```

OfficeCLI wird einfach als weiteres Toolset registriert:

```python
# In alpharavis_toolsets.py — TOOLSETS dict
"office/documents": Toolset(
    "office/documents",
    "Create, read, edit, and validate Office documents (.docx, .xlsx, .pptx). "
    "Generate presentations, spreadsheets, and Word documents via OfficeCLI. "
    "Includes live preview, template merge, and batch operations.",
    mcp_categories=("officecli", "office"),
),
```

In `mcp.json` wird der OfficeCLI-MCP-Server registriert:

```json
{
  "officecli": {
    "command": "officecli",
    "args": ["mcp", "start"],
    "transport": "stdio"
  }
}
```

**Was der Agent sieht (im Prompt):**
```
office/documents: Create, read, edit, and validate Office documents (.docx, .xlsx, .pptx)...
```
→ ~25 Tokens. Eine Zeile. Keine 20 Einzel-Tools.

**Was passiert wenn der Agent Office braucht:**
1. `infer_toolsets_from_text("Erstelle eine PPTX")` → matcht "office/documents"
2. `materialize_toolsets(["office/documents"])` → injected ALLE OfficeCLI-MCP-Tools
3. Agent nutzt `officecli_add`, `officecli_create`, `officecli_view` etc.

**Token-Bilanz:**
- Request ohne Office: 0 Office-Tokens
- Request mit Office: ~2000 Tokens (20 Tools), ABER nur wenn Office angefordert
- Server-Verbindung: EAGER beim Start (1x), kostet ~2 Sekunden, keine Tokens

**Kein CLI-Mode nötig.** Das Toolset-System ist bereits das "Ordner"-Konzept
das du beschrieben hast. OfficeCLI funktioniert exakt wie Pixelle, RAG, etc.

**Warum das besser ist als ein separater CLI/ MCP-Toggle:**
- Kein UI-Toggle nötig — Agent entscheidet selbst wann er Office-Tools braucht
- Kein `office_mode` State — weniger Komplexität
- CLI-Referenz (`OFFICECLI_AGENT_REFERENCE.md`) bleibt als Fallback für Prompt-Injection
- MCP-Tools sind typisiert, validiert, structured — kein Shell-Parsing
- Funktioniert im normalen Chat UND im Office-Tab — kein Unterschied

**Spätere Optimierung (Phase 5+): Server-Level Lazy MCP**
Falls der OfficeCLI-MCP-Server-Start zu lange dauert (>5s), kann die
Server-Verbindung von eager auf lazy umgestellt werden. Das ist aber
ein separater Umbau in `mcp_client.py` und nicht Office-spezifisch.

### Phase 3: UI — Office Tab (Day 5-8)

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

### Phase 5: Advanced Features — Basis abgeschlossen

**Status (2026-05-25): implemented as thin OfficeCLI plan/launcher layer.**
Phase 5 intentionally does not introduce a background job engine. The UI and
media-gallery expose safe plan endpoints and Agent launchers; managed execution,
progress tracking, and persistence are now Phase 6.

Implemented Phase-5 surface:

- `GET /office/templates` lists DOCX/PPTX/XLSX templates under
  `/workspace/office-output/templates/`.
- `POST /office/template-merge` returns a quoted `officecli merge` plan plus
  validation/issue commands.
- `POST /office/batch` returns a quoted `officecli batch` plan and validation
  follow-up.
- `POST /office/validate` returns `officecli validate` plus
  `officecli view ... issues --json`.
- `POST /office/roundtrip` returns `officecli dump ... -o <blueprint>` plus a
  follow-up `officecli batch ... --input <blueprint>` plan.
- The Office panel fetches those plans, sends the commands to the agent, and
  falls back to local prompt templates if the plan endpoint is unavailable.

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
# Agent generates variants from an input JSON file
officecli batch /workspace/office-output/templates/invoice-template.docx \
  --input /workspace/office-output/batch-input.json
```

**5.3 Document Validation Pipeline**

```bash
# Pre-delivery check
officecli validate /workspace/office-output/report.docx
officecli view /workspace/office-output/report.docx issues --json
```

**5.4 Round-trip Learning Launcher**

Agent extracts structure from an existing document:

```bash
officecli dump /workspace/office-output/template.docx \
  -o /workspace/office-output/template-blueprint.json
# Agent studies blueprint.json and can launch variations later.
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

This gives the agent typed tool calls instead of raw shell commands. It remains
feature-flagged/default-off to avoid eager token/tool overhead.

---

### Phase 6: Managed Office Workflows — implementiert

**Status (2026-05-25): implemented as a non-destructive managed plan/status
layer.** Phase 6 keeps the existing Agent-driven execution model, but removes
manual prompt typing for common workflows: the Office panel now calls
media-gallery workflow endpoints, receives quoted OfficeCLI command plans, and
sends those plans to the agent. Every write-oriented flow creates sibling/copy
artifacts instead of overwriting the source document.

Implemented Phase-6 surface:

- `POST /office/preview` returns an OfficeCLI plan for `<name>-preview.html` and
  `<name>-preview.png` generation. It is non-destructive and only creates
  sibling preview artifacts.
- `POST /office/repair` returns a repair plan that validates the original,
  reads issue JSON, writes `<name>-repaired.<ext>`, and validates the repaired
  copy. The original is explicitly not overwritten.
- `POST /office/watch/start`, `POST /office/watch/stop`, and
  `GET /office/watch/status` provide a managed watch lifecycle/status layer.
  The standalone OfficeCLI watch URL remains available for compatibility, while
  the Office panel embeds the preview directly in an iframe Preview frame.
- `GET /office/blueprints/suggest` returns a user-facing hint: if an operator
  likes an existing/polished document, they can turn it into a blueprint.
- `POST /office/blueprints/create` returns an `officecli dump` plan that writes
  `<name>-blueprint.json` next to the source document.
- `GET /office/blueprints` lists existing `*-blueprint.json` artifacts.
- The Office panel exposes direct buttons for `Generate preview`, `Repair`,
  `Make blueprint`, and managed `Watch` so the operator does not need to type
  those prompts manually.

**6.1 Automatic preview generation**

The preview workflow is explicit and non-destructive:

```bash
officecli view '/workspace/office-output/report.docx' html \
  -o '/workspace/office-output/report-preview.html'
officecli view '/workspace/office-output/report.docx' screenshot \
  -o '/workspace/office-output/report-preview.png'
```

The media-gallery `/office/files` response already links existing preview
artifacts via `preview_available`, `preview_image_url`, and `preview_html_url`,
so the UI can show `Preview ready`, `Preview PNG`, and `Preview HTML` without
listing preview files as separate documents.

**6.2 Managed watch lifecycle**

The Watch button now asks `/office/watch/start` for a plan and opens the
embedded Preview frame in the Office tab. The separate `http://localhost:26315`
watch page stays supported for compatibility and direct debugging.

```bash
nohup officecli watch '/workspace/office-output/report.docx' --port 26315 \
  > /tmp/officecli-watch.log 2>&1 &
officecli unwatch '/workspace/office-output/report.docx'
```

**6.3 Repair button after validation**

Validation remains a separate Phase-5 button, but Phase 6 adds a direct Repair
button beside it. The repair workflow is intentionally copy-first:

```bash
officecli validate '/workspace/office-output/report.docx'
officecli view '/workspace/office-output/report.docx' issues --json
officecli repair '/workspace/office-output/report.docx' \
  -o '/workspace/office-output/report-repaired.docx'
officecli validate '/workspace/office-output/report-repaired.docx'
```

If `officecli repair` is unavailable for a document type, the agent uses the
issue JSON to apply safe `officecli set/add` fixes to the repaired copy only.

**6.4 Blueprint Library hints and creation**

The UI now surfaces a lightweight helper text in the Template/Blueprint area:

> If you like documents you already have, you can make a blueprint out of it and
> reuse the structure later.

The `Make blueprint` button calls `/office/blueprints/create` and launches:

```bash
officecli dump '/workspace/office-output/nice-reference.docx' \
  -o '/workspace/office-output/nice-reference-blueprint.json'
officecli view '/workspace/office-output/nice-reference.docx' outline --json
```

Blueprint JSON files are listed through `/office/blueprints` and become reusable
layout/style recipes for future document creation.

**6.5 Validation result persistence, Batch progress, Template merge forms**

The extended Phase-6 implementation stores Office workflow state through the
existing `langgraph-app/run_state_manager.py` instead of introducing an
Office-only state manager. The same Mongo-backed collection now accepts generic
workflow records keyed by namespace/workflow id.

Implemented workflow-state endpoints:

- `GET /office/validation-results` lists persisted validation records.
- `POST /office/validation-results` records status, summary, issue count, and
  issue details for a file.
- `/office/files` enriches document cards with `validation_status`,
  `validation_badge`, `validation_issues`, and `validation_summary` from the
  latest persisted validation record.
- `GET /office/batch/jobs` lists managed batch jobs.
- `POST /office/batch/jobs` creates a batch job record with row-level progress
  counters and a safe OfficeCLI command plan.
- `GET /office/batch/jobs/{job_id}` returns one batch job status.
- `POST /office/batch/jobs/{job_id}/progress` updates completed/failed/pending
  counters and row error details.
- `POST /office/templates/placeholders` detects `{{placeholder}}` tokens from a
  template file and returns a typed field list. If plain extraction cannot see
  all placeholders in a binary Office document, the returned command plan asks
  the agent to run `officecli view ... text --json` and merge AI-detected fields
  into the same flow.
- `POST /office/templates/merge-form` builds a safe merge plan from a selected
  template, output path, and collected form data.

The Office panel now shows validation badges/issues on document cards, a
Workflow State section with persisted validation results and batch progress, a
Managed Batch button, and Template/Blueprint controls backed by those endpoints.
Real OfficeCLI E2E coverage remains gated by the presence of real OfficeCLI and
sample documents, but the browser/backend contract is now stable.

---

### Phase 7: Dedicated Office Agent — implemented

Phase 7 makes Office a first-class peer in the existing AlphaRavis swarm instead
of leaving substantial Office work on the generalist path. The Office tab still
uses the existing media-gallery endpoints for lightweight list/upload/status/plan
actions, but every generated Office workflow prompt now targets
`active_agent=office_agent` when `NEXT_PUBLIC_OFFICE_AGENT_ENABLED=true`.

Implemented architecture:

1. `langgraph-app/alpharavis_toolsets.py` now defines `agent/office`.
   - It includes `office/documents`, `artifacts`, and narrowly scoped
     memory/reporting helpers.
   - `office/documents` keeps local OfficeCLI execution and Office MCP category
     selection, but no longer inherits Hermes delegation through
     `coding/execute`; code/system fixes should be handed off explicitly.
2. `_build_graph()` now registers `office_agent` in `agent_toolset_names` and
   creates an `office_worker` behind `ALPHARAVIS_ENABLE_OFFICE_AGENT`.
3. Peer handoff `transfer_to_office` is exposed from generalist, UI, research,
   debugger, Hermes, and context agents when the feature is enabled.
4. `office_agent` owns Office policy:
   - inspect before modifying;
   - copy-first/non-destructive repair;
   - validate after create/edit/merge/repair;
   - generate/refresh previews when useful;
   - use the existing run_state_manager-backed media-gallery workflow APIs for
     validation, batch, and placeholder state;
   - delegate to research/debugger/Hermes/context only when that specialist is
     actually needed.
5. `submodules/deep-agents-ui` can submit an `active_agent` override through
   `useChat.sendMessage(..., { activeAgent })`. The Office tab wraps all
   generated prompts with an Office-Agent marker and sends
   `{ active_agent: "office_agent" }` when enabled.
6. Docker Compose enables the Office Agent path for the Office tab by default:
   - `ALPHARAVIS_ENABLE_OFFICE_AGENT=${ALPHARAVIS_ENABLE_OFFICE_AGENT:-true}`
   - `NEXT_PUBLIC_OFFICE_AGENT_ENABLED=${NEXT_PUBLIC_OFFICE_AGENT_ENABLED:-true}`
   - `NEXT_PUBLIC_OFFICE_AGENT_NAME=${NEXT_PUBLIC_OFFICE_AGENT_NAME:-office_agent}`

Routing rule after Phase 7:

- Small/direct UI action: use the existing endpoint directly.
  Examples: list files, refresh state, upload, list templates, read validation
  results, read batch status, detect placeholders.
- Substantial Office workflow: go through `office_agent`.
  Examples: create a deck/report/spreadsheet, multi-step edits, template merge,
  managed batch generation, validation+repair pipeline, preview/watch workflow,
  round-trip blueprint work.
- Other agents should transfer substantial Office work to `office_agent` instead
  of loading broad Office/Hermes execution tools themselves.

Verification targets:

- `tests/test_alpharavis_toolsets.py` covers `agent/office`, Office keyword
  inference, and Office MCP locality.
- `tests/test_office_agent_phase7.py` covers the graph feature flag, handoff
  wiring, Office policy prompt, and Office-tab active-agent routing.
- `tests/test_deep_agents_office_ui.py` covers the visible Office UI contract.


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
| `docs/OFFICECLI_AGENT_REFERENCE.md` | Compact CLI reference for agent system prompt |
| `.hermes/plans/office-tab-implementation.md` | This plan |

### Modified Files

| File | Change |
|------|--------|
| `docker/langgraph-api/Dockerfile` | Install OfficeCLI binary + chromium |
| `docker-compose.yml` | Add office_output volume, expose port 26315 |
| `src/app/page.tsx` | Add "Office" tab to navigation |
| `langgraph-app/agent_graph.py` | Add `office/documents` toolset to `TOOLSETS`; register OfficeCLI MCP tools |
| `langgraph-app/mcp_client.py` | Add `load_mcp_tools_lazy()` with TTL cache and selective tool injection |
| `langgraph-app/prompt_assembly.py` | Add OfficeCLI CLI reference to system prompt (from OFFICECLI_AGENT_REFERENCE.md) |
| `langgraph-app/mcp.json` | Register OfficeCLI MCP server (stdio transport) |

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
