# AionUi / OfficeCLI — Office/Docs Integration Analysis & AlphaRavis Porting Plan

> Erstellt 2026-05-24 · Letzte Aktualisierung: 2026-05-25 (AlphaRavis base path implemented)

## 1. Die zwei Projekte: AionUi vs. OfficeCLI

iOfficeAI hat zwei getrennte Open-Source-Projekte, die zusammenarbeiten:

| | AionUi | OfficeCLI |
|---|---|---|
| **Repo** | `iOfficeAI/AionUi` (26.4k ⭐) | `iOfficeAI/OfficeCLI` (5.1k ⭐) |
| **Typ** | Desktop-App (Electron) | CLI-Tool (Single Binary) |
| **Zweck** | GUI für AI-Agents | Office-Dokumente für AI-Agents |
| **Lizenz** | Apache 2.0 | Apache 2.0 |
| **Sprache** | TypeScript/React | .NET (embedded runtime) |

**AionUi** = Die GUI, die mehrere CLI-Agents in Tabs hostet. Ein "Cowork Space".

**OfficeCLI** = Das Werkzeug, das diese Agents aufrufen um Office-Dokumente zu
erstellen, lesen, editieren. Kein Agent-Manager — ein Document-Tool.

OfficeCLI ist das für AlphaRavis relevanteste Projekt, weil es direkt von jedem
Agent (Claude Code, Codex, Hermes) als CLI-Tool aufgerufen werden kann — ohne
AionUi-GUI.

## 2. OfficeCLI — Das Office-Suite-Tool für AI Agents

> "OfficeCLI is the world's first Office suite designed for AI agents. Give any
> AI agent full control over Word, Excel, and PowerPoint — in one line of code."

**Single Binary. Zero Dependencies. Kein Office installiert.**

```bash
# Installation (Linux/macOS)
curl -fsSL https://raw.githubusercontent.com/iOfficeAI/OfficeCLI/main/install.sh | bash

# PowerPoint in einer Zeile
officecli add deck.pptx / --type slide --prop title="Q4 Report"

# Excel mit Pivot-Tabelle
officecli add sales.xlsx '/Sheet1' --type pivottable \
  --prop source='Data!A1:E10000' --prop rows='Region,Category' \
  --prop cols=Quarter --prop values='Revenue:sum'

# Live-Vorschau
officecli watch deck.pptx   # → http://localhost:26315
```

### 2.1 Warum OfficeCLI besser als mammoth ist

| Fähigkeit | mammoth (npm) | OfficeCLI |
|---|---|---|
| DOCX lesen | ✅ → HTML | ✅ Text, Struktur, Styles, Formeln, JSON |
| DOCX editieren | ❌ | ✅ — jedes Element änderbar |
| DOCX erstellen | ❌ | ✅ — `officecli create doc.docx` |
| XLSX Support | ❌ | ✅ — Zellen, Formeln, Pivot, Charts, CSV Import |
| PPTX Support | ❌ | ✅ — Slides, Shapes, Animationen, 3D-Modelle |
| Template Merge | ❌ | ✅ — `{{key}}` → JSON Merge |
| Rendering | Nur HTML | HTML + PNG Screenshots + Live-Watch-Server |
| MCP Server | ❌ | ✅ — `officecli mcp claude` |
| Formel-Engine | ❌ | ✅ 150+ Excel-Funktionen, Auto-Evaluation |
| Deterministic JSON | ❌ | ✅ — `--json` Flag auf jedem Command |
| Resident Mode | ❌ | ✅ Named-Pipe, Multi-Step Workflows |
| Batch Mode | ❌ | ✅ Atomic Multi-Command |
| Installation | npm install | 1 Binary, auto-install |

**mammoth kann nur DOCX → HTML. OfficeCLI kann alles — lesen, schreiben,
editieren, rendern, mit MCP-Server, in Docker, headless. Für AI Agents gemacht.**

### 2.2 Three-Layer Architecture (L1 → L2 → L3)

OfficeCLI hat ein progressives Komplexitätsmodell das Token-Verbrauch minimiert:

| Layer | Zweck | Beispiel |
|-------|-------|----------|
| L1: Read | Semantische Views | `officecli view report.docx annotated` |
| L2: DOM | Strukturierte Element-Ops | `officecli add budget.xlsx / --type sheet` |
| L3: Raw XML | XPath Direktzugriff | `officecli raw deck.pptx '/slide[1]'` |

Agents starten auf L1 (billig, read-only), eskalieren zu L2 (DOM-Manipulation),
fallen zurück auf L3 (raw XML) nur wenn nötig.

### 2.3 Built-in Features

- **Rendering Engine**: HTML + PNG (headless browser), Live-Watch-Server
- **Formula Engine**: 150+ Excel-Funktionen (SUM, VLOOKUP, FILTER, UNIQUE...)
- **Pivot Engine**: Native OOXML Pivot-Tabellen mit einem Command
- **Template Merge**: `{{key}}` → JSON, Agent designed Layout einmal, Batch füllt N-mal
- **Round-trip dump**: DOCX → JSON → DOCX, Agent lernt von existierenden Templates
- **MCP Server**: `officecli mcp claude|cursor|vscode|lmstudio`
- **Self-healing**: Strukturierte Error-Codes mit Suggestions

### 2.4 OfficeCLI Skill File

```bash
curl -fsSL https://officecli.ai/SKILL.md
```

Ein Agent-Skill-File das dem Agent beibringt wie er OfficeCLI installiert und
alle Commands nutzt. Auto-Install erkennt Claude Code, Cursor, Windsurf, GitHub
Copilot und konfiguriert sich selbst.

## 3. AionUi — Die GUI (Hintergrund)

AionUi ist die Electron-Desktop-App die OfficeCLI unter der Haube nutzt:

- **Tech Stack**: Electron 37 + React 19 + TypeScript + Arco Design + UnoCSS
- **Agent-Hosting**: CLI-Agents (Claude Code, Codex, Hermes, Gemini CLI) in Tabs
- **IPC Bridge**: `@office-ai/platform` (MIT), 24+ Bridge-Module
- **Channel-Plugins**: Telegram (grammY), Lark/Feishu, DingTalk, WeCom
- **npm-Pakete**: `officeparser`, `mammoth`, `docx`, `xlsx-republish`, `pptx2json`
  (für In-App Vorschau, NICHT für Agent-Tooling — Agent-Tooling macht OfficeCLI)

**Für AlphaRavis ist AionUi als Ganzes nicht relevant** — wir haben bereits
deep-agents-ui als WebUI. Relevant ist OfficeCLI als Tool für die Agents.

## 4. Architektur: Wie OfficeCLI in AlphaRavis passt

```
┌─────────────────────────────────────────────────────────────┐
│                     LangGraph API                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              AlphaRavis Agent                          │  │
│  │  "Erstelle eine Präsentation über Q4-Zahlen"          │  │
│  │                                                       │  │
│  │  agent_graph.py:                                      │  │
│  │    tool_call("terminal", {                            │  │
│  │      "command": "officecli add deck.pptx /             │  │
│  │        --type slide --prop title='Q4 Report'"         │  │
│  │    })                                                 │  │
│  │                                                       │  │
│  │    tool_call("terminal", {                            │  │
│  │      "command": "officecli view deck.pptx screenshot   │  │
│  │        -o /tmp/deck.png"                              │  │
│  │    })                                                 │  │
│  │    → Agent SIEHT das Ergebnis (PNG)                   │  │
│  │    → Agent korrigiert Layout-Fehler                   │  │
│  │                                                       │  │
│  │  Ergebnis: deck.pptx → Download via UI                │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                     Docker Container                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  /usr/local/bin/officecli  (Single Binary, ~30 MB)   │  │
│  │  - Kein Office installiert                            │  │
│  │  - Embedded .NET Runtime                              │  │
│  │  - Headless Chrome für PNG-Rendering                  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Workflow:**
1. User sagt: "Erstelle eine PPTX über Q4"
2. LangGraph Agent ruft `officecli create deck.pptx` + `officecli add ...`
3. Agent ruft `officecli view deck.pptx screenshot` → sieht das Ergebnis
4. Agent korrigiert: `officecli set deck.pptx /slide[1]/shape[1] --prop color=FF0000`
5. Datei wird über UI zum Download angeboten

## 5. Vergleich: npm-Pakete vs. OfficeCLI

Die npm-Pakete die AionUi benutzt sind für **In-App Vorschau**, nicht für Agent-Tooling:

| Paket | Zweck in AionUi | Durch OfficeCLI ersetzt? |
|-------|----------------|-------------------------|
| `mammoth` | DOCX → HTML Vorschau | ✅ OfficeCLI `view html` |
| `officeparser` | DOCX/XLSX/PPTX Text-Extraktion | ✅ OfficeCLI `view text` / `get --json` |
| `docx` | DOCX-Erstellung via JS | ✅ OfficeCLI `create` + `add` + `set` |
| `xlsx-republish` | XLSX Lesen | ✅ OfficeCLI `get --json` |
| `pptx2json` | PPTX → JSON | ✅ OfficeCLI `dump` / `get --json` |

**Fazit**: Für AlphaRavis brauchen wir NUR OfficeCLI (1 Binary, ~30 MB).
Keine npm-Pakete für Office-Funktionalität. mammoth/officeparser nur wenn wir
eine reine Web-Vorschau ohne Agent-Tooling bräuchten.

## 6. AlphaRavis-Integration: Konkreter Plan

Status 2026-05-25: Der Basis-Pfad ist implementiert. `langgraph-api` installiert
OfficeCLI + Chromium, Docker Compose mountet `./office-output` nach
`/workspace/office-output` und veröffentlicht den Watch-Port `26315`. AlphaRavis
hat das default-off Toolset `office/documents`, default-off OfficeCLI-Prompting,
per `enabled_env` gegatetes OfficeCLI-MCP, und die DeepAgentsUI-Fork akzeptiert
DOCX/PPTX/XLSX Uploads plus Office-Tab-Launcher. Noch offen sind automatische
Download-/Preview-APIs, verwaltete Watch-Prozesse pro Datei/Session, und Live
Browser-Smoke im UI.

### Phase 1 — OfficeCLI im Docker-Container (implemented base path)

1. Dockerfile: OfficeCLI Binary downloaden und in PATH legen
2. `docker-compose.yml`: Volume für Output-Dateien
3. Agent hat automatisch Zugriff via `terminal()` Tool
4. MCP-Server optional: `officecli mcp` → registriert Office-Tools

```dockerfile
# Im langgraph-api Dockerfile:
RUN curl -fsSL https://raw.githubusercontent.com/iOfficeAI/OfficeCLI/main/install.sh | bash
```

### Phase 2 — Agent-Kontext & Preview (2-3 Tage)

1. Agent lädt Office-Dokumente hoch → OfficeCLI extrahiert Text → Agent-Kontext
2. Agent erstellt/editiert Dokumente → OfficeCLI `watch` für Live-Preview
3. PNG-Screenshots als Vision-Feedback für den Agent (Render → Look → Fix Loop)

### Phase 3 — UI-Integration (optional, 3-5 Tage)

1. File-Upload: `.docx`, `.xlsx`, `.pptx` akzeptieren
2. Preview-Panel: OfficeCLI `view html` Output anzeigen
3. Download-Button für generierte Dokumente

## 7. Feature-Vergleich: AionUi vs. Deep Agents UI

| Feature | AionUi | Deep Agents UI (AlphaRavis Fork) |
|---------|--------|----------------------------------|
| **Framework** | Electron 37 (Desktop) | Next.js 16 (Web) |
| **UI Library** | Arco Design | shadcn/ui (Radix + Tailwind) |
| **Agent-Protokoll** | ACP + CLI stdio | LangGraph SDK (REST/SSE) |
| **Office Tool** | OfficeCLI (built-in) | ❌ → OfficeCLI-Integration geplant |
| **Code-Editor** | Monaco + Codemirror | Monaco (lazy) + DiffViewer |
| **IM Channels** | ✅ Telegram, Lark, DingTalk | ❌ (Hermes Agent separat) |
| **Multi-Agent** | ✅ Parallel CLI Sessions | ✅ Subagent-Indikatoren |
| **File Upload** | ✅ Mit Office-Parsing | ✅ Grundlegend (Bilder/PDF) |
| **Threads** | ✅ | ✅ |
| **Tasks/Todos** | ❌ | ✅ |
| **Tool Approval** | ❌ | ✅ |
| **MCP Support** | ✅ | ✅ (Hermes + LangGraph MCP Client) |
| **Deployment** | Desktop App | Docker Container |

## 8. Lizenz-Kompatibilität

Alle Projekte sind Apache 2.0 oder MIT — vollständig kompatibel:

- **OfficeCLI**: Apache 2.0 ✅ — `github.com/iOfficeAI/OfficeCLI`
- **AionUi**: Apache 2.0 ✅ — `github.com/iOfficeAI/AionUi`
- `@office-ai/aioncli-core`: Apache 2.0 ✅ — `github.com/office-sec/aioncli` (Gemini CLI Fork)
- `@office-ai/platform`: MIT ✅
- `mammoth`: BSD 2-Clause ✅
- `officeparser`, `docx`: MIT ✅

**Keine proprietären Abhängigkeiten.** Alles Open Source.
