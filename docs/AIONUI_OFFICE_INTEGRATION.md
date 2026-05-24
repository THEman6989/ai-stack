# AionUi Office/Docs Integration — Analysis & AlphaRavis Porting Plan

> Erstellt 2026-05-24 · Letzte Aktualisierung: 2026-05-24

## 1. Was ist AionUi?

AionUi (`iOfficeAI/AionUi`, 26.4k ⭐) ist eine Electron-basierte Desktop-App, die
CLI-basierte AI-Agents (Claude Code, Codex, Hermes Agent, OpenCode, Gemini CLI
u.a.) in eine moderne Chat-UI einbettet. Sie fungiert als "Cowork App" —
Terminal-Agent trifft GUI.

- **Repo**: https://github.com/iOfficeAI/AionUi
- **Tech Stack**: Electron 37 + React 19 + TypeScript 5.8 + UnoCSS + Arco Design
- **Lizenz**: Apache 2.0
- **Betriebsmodi**: Desktop (Electron), WebUI (`--webui`), CLI (`electron-vite dev`)

## 2. Office/Docs-Integration: Komponenten

### 2.1 `@office-ai/aioncli-core` — CLI Core

Das zentrale CLI-Framework (`^0.30.6`). Stellt die Brücke zwischen Electron
Main Process und den eingebetteten CLI-Agents her. Zuständig für:

- Prozess-Spawning (pty-basierte Terminal-Sessions)
- Message-Routing (stdin/stdout/stderr → Chat UI)
- Tool-Ausführung und Sandboxing
- Plattform-übergreifende Pfad-Auflösung

### 2.2 `@office-ai/platform` — IPC Bridge

Die Inter-Process-Communication-Bridge (`^0.3.16`) zwischen Electron Main und
Renderer. 24+ Bridge-Module:

| Bridge | Funktion |
|--------|----------|
| `conversationBridge` | Konversation CRUD, Streaming |
| `databaseBridge` | SQLite (better-sqlite3) Zugriff |
| `fileBridge` | Dateisystem-Operationen |
| `agentBridge` | Agent-Start/Stop/Konfiguration |
| `previewBridge` | Datei-Preview (Office, Code, Media) |
| `settingsBridge` | Persistente Einstellungen |

### 2.3 Office-Dokument-Parser

| Paket | Version | Zweck |
|-------|---------|-------|
| `officeparser` | ^5.2.2 | Generischer Office-Parser (DOCX, XLSX, PPTX) |
| `docx` | ^9.5.1 | DOCX-Erstellung (Word-Dokumente generieren) |
| `mammoth` | ^1.11.0 | DOCX → HTML/Markdown Konvertierung |
| `xlsx-republish` | ^0.20.3 | XLSX/Excel Lesen & Schreiben |
| `pptx2json` | ^0.0.10 | PowerPoint → JSON Extraktion |

### 2.4 Preview-Rendering

| Paket | Zweck |
|-------|-------|
| `@monaco-editor/react` ^4.7.0 | Code-Editor (Monaco) |
| `@uiw/react-codemirror` ^4.25.2 | Leichtgewichtiger Code-Editor |
| `@codemirror/lang-markdown` ^6.5.0 | Markdown-Syntax-Highlighting |
| `diff2html` ^3.4.55 | Diff-Rendering (side-by-side, line-by-line) |
| `mermaid` ^11.13.0 | Diagramme |
| `katex` ^0.16.22 | Mathematische Formeln |
| `react-markdown` ^10.1.0 | Markdown-Rendering |

## 3. Architektur: Wie AionUi Office-Dokumente verarbeitet

```
┌─────────────────────────────────────────────────────────────────┐
│                        Renderer Process                          │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────────────┐  │
│  │ Chat UI  │  │ Preview Panel│  │ Workspace (Tabs)          │  │
│  │          │  │  - Code      │  │  - DOCX/XLSX/PPTX Vorschau│  │
│  │          │  │  - Office    │  │  - Monaco Editor          │  │
│  │          │  │  - Diff      │  │  - Mermaid Diagrams       │  │
│  │          │  │  - Media     │  │                           │  │
│  └────┬─────┘  └──────┬───────┘  └─────────────┬─────────────┘  │
│       │               │                        │                 │
│       └───────────────┼────────────────────────┘                 │
│                       │ @office-ai/platform Bridge               │
├───────────────────────┼─────────────────────────────────────────┤
│                  Main Process                                    │
│  ┌────────────────────┼──────────────────────────────────────┐  │
│  │         Agent Manager (aioncli-core)                       │  │
│  │  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │  │
│  │  │ Claude  │  │  Codex   │  │ Hermes   │  │ Gemini   │   │  │
│  │  │  Code   │  │   CLI    │  │  Agent   │  │   CLI    │   │  │
│  │  └────┬────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │  │
│  │       │            │             │             │          │  │
│  │       └────────────┼─────────────┼─────────────┘          │  │
│  │                    │ pty/stdin   │ stdout                  │  │
│  │              ┌─────┴─────────────┴──────┐                  │  │
│  │              │  Office Document Parser   │                  │  │
│  │              │  - officeparser           │                  │  │
│  │              │  - mammoth (DOCX→HTML)    │                  │  │
│  │              │  - pptx2json              │                  │  │
│  │              │  - xlsx-republish         │                  │  │
│  │              └───────────────────────────┘                  │  │
│  └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Datenfluss Office-Dokument:**

1. **Upload**: User lädt DOCX/XLSX/PPTX über File-Upload in Chat UI
2. **Parsing**: `officeparser` extrahiert Text/Struktur → Markdown/JSON
3. **Agent-Kontext**: Extrahierter Text wird als Kontext an den Agent gesendet
4. **Preview**: `mammoth` rendert DOCX → HTML im Preview Panel
5. **Erstellung**: Agent generiert neue Dokumente via `docx` package → Download

## 4. Feature-Channel-System (Plugin-Architektur)

AionUi hat ein Plugin-System für IM-Plattformen:

```
ChannelManager (Singleton)
├── PluginManager      — Lädt/verwaltet Plattform-Plugins
├── SessionManager     — User-Sessions, Pairing
├── PairingService     — QR-Code/Code-basierte Gerätekopplung
├── ActionExecutor     — System-Actions, Dialog-Routing
└── ChannelMessageService — Stream-basierte Nachrichtenverarbeitung
```

**Implementierte Channel-Plugins:**
- **Telegram** (grammY Bot Framework)
- **Lark/Feishu** (offizielles SDK)
- **DingTalk** (dingtalk-stream)
- **WeCom** (WeChat Work)

**Plugin-Schnittstelle (vereinfacht):**
```typescript
interface ChannelPlugin {
  name: string;
  init(): Promise<void>;
  onMessage(msg: UnifiedMessage): Promise<UnifiedResponse>;
  sendMessage(target: string, content: string): Promise<void>;
  onPairRequest(code: string): Promise<boolean>;
}
```

## 5. Relevanz für AlphaRavis

### 5.1 Was AlphaRavis von AionUi lernen kann

| Feature | AionUi | AlphaRavis Status | Priorität |
|---------|--------|-------------------|-----------|
| Office-Dokument-Parsing | `officeparser` + `mammoth` | ❌ Nicht vorhanden | **Hoch** |
| DOCX-Erstellung | `docx` package | ❌ Nicht vorhanden | Mittel |
| XLSX/PPTX | `xlsx-republish` + `pptx2json` | ❌ Nicht vorhanden | Mittel |
| Preview-Panel | Monaco + Codemirror + Diff2HTML | ✅ Monaco Lazy-Loaded, DiffViewer | ✅ Erledigt |
| Channel-Plugins (IM) | Telegram, Lark, DingTalk, WeCom | ❌ Nicht vorhanden | Niedrig |
| Workspace Tabs | Multi-Tab Dokumenten-Editor | ❌ Nicht vorhanden | Niedrig |
| File-Upload Pipeline | officeparser → Markdown → Agent | ✅ Grundlegend vorhanden | ✅ Erledigt |

### 5.2 Konkrete Integrationspunkte

**A) Office-Dokument-Upload → Agent-Kontext**

Aktuell kann deep-agents-ui nur Bilder und PDFs uploaden. Mit den AionUi-Paketen
könnten wir DOCX/XLSX/PPTX parsen und als Markdown-Text in den Agent-Kontext
einspeisen.

Benötigte Änderungen:
1. `file-validation.ts`: Accept-Liste um `.docx`, `.xlsx`, `.pptx` erweitern
2. Neues Modul `src/lib/office-parser.ts`:
   - `mammoth` für DOCX → Markdown
   - `xlsx-republish` für XLSX → CSV/JSON
   - `pptx2json` für PPTX → Text-Extraktion
3. `useFileUpload.ts`: Office-Dateien vor dem Upload parsen
4. Content Block: Text-Inhalt als `text` block einfügen

**B) Preview-Panel für Office-Dokumente**

Das FilePreviewPanel könnte DOCX via mammoth als HTML rendern:

```typescript
// src/lib/office-parser.ts
import mammoth from "mammoth";

export async function parseDocx(buffer: ArrayBuffer): Promise<string> {
  const result = await mammoth.convertToHtml({ arrayBuffer: buffer });
  return result.value; // HTML string
}
```

**C) Agent-generierte Dokumente**

Wenn der Agent ein Dokument erstellen soll (z.B. "Erstelle mir eine DOCX-Datei"),
könnte `docx` im Backend verwendet werden:

```typescript
import { Document, Packer, Paragraph, TextRun } from "docx";

const doc = new Document({
  sections: [{
    children: [new Paragraph({ children: [new TextRun("Hello")] })],
  }],
});
const buffer = await Packer.toBuffer(doc);
```

### 5.3 Abhängigkeiten für AlphaRavis

Minimal-Set für Office-Support:

```json
{
  "dependencies": {
    "mammoth": "^1.11.0",
    "officeparser": "^5.2.2"
  },
  "devDependencies": {
    "@types/mammoth": "^1.6.0"
  }
}
```

Optional (wenn wir Dokument-Erstellung wollen):
```json
{
  "docx": "^9.5.1",
  "xlsx-republish": "^0.20.3"
}
```

## 6. Channel/IM-Integration (Zukunft)

AionUi's Channel-System ist interessant für AlphaRavis' 24/7-Betrieb.
Mögliche spätere Integration:

- **Telegram Bot**: Nutzer sendet Nachricht → LangGraph API → Antwort zurück
- **Lark/Feishu**: Gleiches Prinzip, für Business-User
- **Pairing**: QR-Code-basierte Geräte-Authentifizierung

Aktuell **nicht priorisiert** — AlphaRavis hat bereits Hermes Agent mit
Telegram/Discord-Integration. Channel-Plugins würden den Scope sprengen.

## 7. AionUi vs. Deep Agents UI — Feature-Vergleich

| Feature | AionUi | Deep Agents UI (AlphaRavis Fork) |
|---------|--------|----------------------------------|
| **Framework** | Electron 37 (Desktop) | Next.js 16 (Web) |
| **UI Library** | Arco Design | shadcn/ui (Radix + Tailwind) |
| **Agent-Protokoll** | ACP + CLI stdio | LangGraph SDK (REST/SSE) |
| **Code-Editor** | Monaco + Codemirror | Monaco (lazy) + DiffViewer |
| **Office Docs** | ✅ DOCX, XLSX, PPTX | ❌ |
| **IM Channels** | ✅ Telegram, Lark, DingTalk | ❌ (Hermes Agent separat) |
| **Multi-Agent** | ✅ Parallel CLI Sessions | ✅ Subagent-Indikatoren |
| **File Upload** | ✅ Mit Office-Parsing | ✅ Grundlegend (Bilder/PDF) |
| **Threads** | ✅ | ✅ |
| **Tasks/Todos** | ❌ | ✅ |
| **Tool Approval** | ❌ | ✅ |
| **MCP Support** | ✅ | ✅ (Hermes + LangGraph MCP Client) |
| **Deployment** | Desktop App | Docker Container |

## 8. Empfehlung: Nächste Schritte

**Phase 1 — Office-Upload (1-2 Tage)**
- `mammoth` + `officeparser` als dependencies hinzufügen
- `src/lib/office-parser.ts` Modul erstellen
- `file-validation.ts`: Accept-Types erweitern
- `useFileUpload.ts`: Office-Parsing-Pipeline integrieren
- Preview-Panel: mammoth HTML-Rendering für DOCX

**Phase 2 — Dokument-Erstellung (optional, 2-3 Tage)**
- `docx` package für Agent-generierte Dokumente
- Download-Button in Chat-Nachrichten
- Template-basierte Dokumentgenerierung

**Phase 3 — Channel-Plugins (Zukunft, 1-2 Wochen)**
- Plugin-Architektur inspiriert von AionUi ChannelManager
- Telegram/Lark-Integration via LangGraph API
- Pairing-Service für Mobile-Desktop-Kopplung

---

## 9. Lizenz-Kompatibilität

- AionUi: Apache 2.0 ✅ (kompatibel mit AlphaRavis)
- `@office-ai/aioncli-core`: Apache 2.0 ✅ — Repo: `github.com/office-sec/aioncli`
- `@office-ai/platform`: MIT ✅ — (kein öffentliches Repo, aber MIT via npm)
- `mammoth`: BSD 2-Clause ✅
- `officeparser`: MIT ✅
- `docx`: MIT ✅

**Korrektur (2026-05-24)**: Die `@office-ai/*` Pakete sind entgegen erster
Annahme Open Source (Apache 2.0 / MIT). Sie können für AlphaRavis verwendet
werden, sofern der Scope `@office-ai` keine Nutzungsbeschränkungen auferlegt.
Empfehlung: `@office-ai/aioncli-core` nur dann einbinden, wenn AlphaRavis das
AionUi-CLI-Framework direkt nutzen will (aktuell nicht der Fall). Für reines
Office-Parsing/Dokument-Handling reichen `mammoth`, `officeparser`, `docx`.
