# Upload-Erweiterung & Media-Handling — Implementierungsplan

**Datum:** 2026-05-25
**Status:** ✅ Implementiert (Phasen 1-9)
**Bereich:** Frontend (deep-agents-ui), Bridge, Media Gallery, Media Analysis

---

## 1. Zusammenfassung

Das Upload-System im AI-Stack (deep-agents-ui Chat + Office-Panel) soll erweitert werden:

1. **Alle Dateitypen im Upload-Button** (ChatInterface) — Videos, Office, Text
2. **Video-Upload-Flow** — aus Kontext extrahieren, in Media Gallery speichern, URL-Link im Kontext
3. **Office-Panel** erweitern um ODF, Bilder, Videos, alle Dateien
4. **2GB-Download-Limit** von Hard-Limit auf Warn-Dialog umbauen
5. **Video-Player im UI** als späteren Plan vermerken

---

## 2. Bestandsaufnahme

### 2.1 Was ist schon da?

| Komponente | Status |
|---|---|
| **FFmpeg/ffprobe** | ✅ Installiert (n8.1.1), alle Codecs vorhanden |
| **Media Gallery** (:8130) | ✅ Upload/Register/Stream für images, videos, audio, documents |
| **Bridge Media Mirroring** | ✅ Auto-Register für images + videos (via `BRIDGE_MEDIA_GALLERY_AUTO_REGISTER_VIDEOS=true`) |
| **Video Analysis Pipeline** | ✅ frame extraction, ffprobe, keyframes, captioning |
| **register_media_asset Tool** | ✅ Agent kann Medien registrieren ohne Kontext-Müll |
| **OfficeCLI** | ✅ DOCX, XLSX, PPTX (Core). Plugins für .doc, .hwpx |
| **Frontend Upload** | ⚠️ Nur Bilder+PDF im `accept`, aber Code validiert auch Office |
| **Frontend Drag/Drop** | ✅ Global, mit File-Typ-Validierung |
| **Frontend Paste** | ✅ Bilder aus Zwischenablage |

### 2.2 Was fehlt?

| Lücke | Details |
|---|---|
| **Chat Upload akzeptiert keine Videos** | `accept` nur `image/*,application/pdf` |
| **Video-Base64 im Kontext** | Würde Kontext sprengen; kein Redirect-Flow |
| **ODF in OfficeCLI** | ❌ OfficeCLI unterstützt NUR `.docx`, `.xlsx`, `.pptx`. Keine ODF-Formate. Plugin-System existiert (`format-handler`), aber niemand hat einen ODF-Plugin geschrieben |
| **ODF in Media Gallery** | Keine MIME-Types registriert |
| **Office Panel nur Office** | Keine Bilder, PDFs, Videos akzeptiert |
| **2GB Hard-Limit** | `media_analysis.py` wirft RuntimeError statt Warnung |

### 2.3 ODF-Recherche-Ergebnisse

**OfficeCLI:** Hat KEINEN ODF-Support. Source-Code (`BlankDocCreator.cs`):
```
throw new NotSupportedException("Unsupported file type. Supported: .docx, .xlsx, .pptx,
  or any extension served by an installed format-handler plugin that implements create.");
```
→ Nur via `format-handler` Plugin erweiterbar. Existiert nicht.

**OnlyOffice DocumentServer:**
- GitHub: `ONLYOFFICE/DocumentServer` (6.5k Stars)
- Docker: `onlyoffice/documentserver` (97M+ Pulls)
- Topics: `odt, ods, odp` ✅ (ODF wird nativ unterstützt!)
- REST API für Document-Conversion
- Größe: ~2-3 GB (Voll-Image), `onlyoffice/docbuilder` existiert als schlankere Variante

**Bewertung:** OnlyOffice ist der beste Open-Source-Konverter für ODF→DOCX mit hoher Fidelity (Tabellen, Bilder, Formatierung deutlich besser als LibreOffice).

---

## 3. Schritt-für-Schritt-Plan

### Phase 1: Frontend Upload-Button erweitern (ChatInterface)

**Datei:** `submodules/deep-agents-ui/src/app/components/ChatInterface.tsx`

**Zeile 630** — `accept`-Attribut ändern:

```tsx
// ALT:
accept="image/jpeg,image/png,image/gif,image/webp,application/pdf"

// NEU (alles was FFmpeg + Office + Web kann):
accept="image/*,video/*,audio/*,.pdf,.doc,.docx,.ppt,.pptx,.xls,.xlsx,.odt,.odp,.ods,.txt,.md,.csv,.json,.log,.html,.htm"
```

**Datei:** `submodules/deep-agents-ui/src/lib/file-validation.ts`

**SUPPORTED_FILE_TYPES** erweitern:

```ts
export const SUPPORTED_FILE_TYPES = [
  // Bilder
  "image/jpeg", "image/png", "image/gif", "image/webp",
  "image/heic", "image/heif", "image/svg+xml",
  // Videos (alle die FFmpeg kann)
  "video/mp4", "video/webm", "video/quicktime",
  "video/x-matroska", "video/x-msvideo", "video/x-m4v",
  "video/x-flv", "video/x-ms-wmv", "video/MP2T",
  "video/3gpp", "video/3gpp2", "video/ogg",
  // Audio
  "audio/mpeg", "audio/mp4", "audio/wav", "audio/x-wav",
  "audio/ogg", "audio/webm", "audio/flac",
  // PDF
  "application/pdf",
  // Office (Microsoft)
  ...OFFICE_FILE_TYPES,
  // Office (OpenDocument / ODF)
  "application/vnd.oasis.opendocument.text",
  "application/vnd.oasis.opendocument.presentation",
  "application/vnd.oasis.opendocument.spreadsheet",
  // Text
  "text/plain", "text/markdown", "text/csv",
  "text/html", "application/json",
] as const;
```

**Datei:** `submodules/deep-agents-ui/src/lib/multimodal-utils.ts`

`fileToContentBlock()` für neue Typen erweitern:
- Videos → `type: "file"` (niemals Base64!), mit `mimeType`
- Audio → `type: "file"`
- Office/Text → `type: "file"`
- Nur Bilder bleiben `type: "image"` mit Base64 (die sind klein genug)

**WICHTIG:** Videos und Audio NIEMALS als Base64 in den Kontext! Stattdessen:
1. Datei an Media Gallery Endpoint (`:8130/assets/upload`) senden
2. `public_url` zurückbekommen
3. ContentBlock als `{type: "file", mimeType: "video/mp4", data: "", metadata: {url: "http://...", filename: "..."}}`

### Phase 2: Video-Upload-Flow — Frontend → Media Gallery → Kontext

**Neuer Endpoint oder bestehender nutzen:**

Media Gallery hat bereits:
- `POST /assets/upload` — multipart upload (Zeile 1327)
- `POST /assets/register` — URL/FILE_ID registration (Zeile 2878 in agent_graph.py)

**Frontend-Flow für große Dateien (>10MB oder alle Videos/Audio):**

```typescript
// Pseudocode
async function uploadLargeFile(file: File): Promise<ContentBlock> {
  const formData = new FormData();
  formData.append("file", file);
  
  const response = await fetch(`${MEDIA_GALLERY_URL}/assets/upload`, {
    method: "POST",
    body: formData,
  });
  
  const { public_url, asset_id } = await response.json();
  
  return {
    type: "file",
    mimeType: file.type,
    data: "",  // KEIN Base64!
    metadata: {
      filename: file.name,
      url: public_url,
      asset_id: asset_id,
      media_gallery_url: public_url,
    },
  };
}
```

**Neue Hook:** `useLargeFileUpload.ts` (oder Erweiterung von `useFileUpload.ts`)

- Kleine Bilder (<10MB): Base64 wie bisher
- Alles andere (Videos, Audio, Office, PDFs, Text): Upload zu Media Gallery, URL in ContentBlock

**In `file-validation.ts`:**
```ts
const LARGE_FILE_THRESHOLD = 10 * 1024 * 1024; // 10 MB
const ALWAYS_UPLOAD_TYPES = ["video/", "audio/", "application/"];
```

### Phase 3: Bridge — Video-Kontext-Handling prüfen

**Datei:** `langgraph-app/bridge_server.py`

Aktuell:
- `BRIDGE_MEDIA_GALLERY_AUTO_REGISTER_VIDEOS=true` (default!)
- Bei `BRIDGE_MEDIA_CONTEXT_MODE=metadata` werden Media-Parts durch Metadaten + URL ersetzt

**Prüfen:** Funktioniert der Flow bereits korrekt, wenn das Frontend Videos als `{type:"file", metadata:{url:"..."}}` sendet?

→ Bridge sollte den `url`-Key erkennen und in der Media Gallery registrieren (via `_mirror_media_part_to_media_gallery`)

→ Nach Registrierung URL in `video_url`/`file_url` ersetzen

→ Im Agent-Kontext erscheint dann nur die URL + Metadaten, nicht das Roh-Video

**Falls nicht:** Bridge-Logik für `type:"file"` mit `mimeType: "video/*"` ergänzen.

### Phase 4: LangGraph — Video-Tool bestätigen

**Datei:** `langgraph-app/agent_graph.py`

- `register_media_asset` Tool (Zeile 2878) kann bereits `video` registrieren
- `plan_media_analysis` Tool (Zeile 2931) kennt Video-Pipeline

**Prüfen:** Ob der Agent bei einem `{type:"file", metadata:{url:"http://...", filename:"video.mp4"}}` ContentBlock:
1. Das Video erkennt
2. Es via `register_media_asset` registrieren kann
3. Es später via `media_download` Tool (existiert das?) abrufen kann

**Falls `media_download` Tool fehlt:** Neues Tool hinzufügen:
```python
@tool
async def media_download(asset_id: str):
    """Download a media asset from the gallery to local workspace."""
    # Ruft /assets/{asset_id}/download auf
```

### Phase 5: Office-Panel erweitern

**Datei:** `submodules/deep-agents-ui/src/app/components/OfficePanel.tsx`

**Upload-Accept erweitern** (Zeile 785):

```tsx
// ALT:
accept=".docx,.pptx,.xlsx,..."

// NEU:
accept=".docx,.pptx,.xlsx,.odt,.odp,.ods,image/*,.pdf,.txt,.md,.csv,.json,video/*"
```

**Upload-Handling:** Der Office-Upload-Endpoint (`:8130/office/upload`) akzeptiert derzeit nur Office-Dateien (via `_store_office_uploaded_file`). Für Bilder/PDFs/Videos muss entweder:

a) Der Endpoint um allgemeine Dateien erweitert werden, oder
b) Im Frontend je nach Dateityp unterschiedliche Endpoints ansteuern

**Empfehlung:** Variante (b) — Office-Dateien → `:8130/office/upload`, alles andere → `:8130/assets/upload`

### Phase 6: ODF-Support (Optional, Feature-Gated)

> ⚠️ **ODF ist EINDEUTIG NICHT in OfficeCLI unterstützt.** Der Source-Code bestätigt:
> Nur `.docx`, `.xlsx`, `.pptx` — oder `format-handler` Plugins (existieren nicht für ODF).

#### Architektur: OnlyOffice als ODF→DOCX Konverter

```
ODF-Upload (.odt/.odp/.ods)
  → Media Gallery (:8130) speichert Original
  → OnlyOffice DocumentServer REST API: ODF → DOCX konvertieren
  → DOCX an OfficeCLI weitergeben
  → Agent arbeitet mit DOCX (native OfficeCLI)
```

#### Feature-Gate: `ALPHARAVIS_ENABLE_ODF_UPLOAD`

```bash
# .env
ALPHARAVIS_ENABLE_ODF_UPLOAD=true   # Aktiviert OnlyOffice-Container + ODF-Uploads
# ALPHARAVIS_ENABLE_ODF_UPLOAD=false  # ODF wird abgewiesen
```

**Logik:**
- `false` (default): ODF-Dateien werden beim Upload rejected → User sieht "ODF not supported. Set ALPHARAVIS_ENABLE_ODF_UPLOAD=true to enable."
- `true`: OnlyOffice-Container wird in docker-compose hochgefahren, ODF-Uploads akzeptiert

#### 6.1 docker-compose.yaml — OnlyOffice Service

```yaml
services:
  onlyoffice:
    image: onlyoffice/documentserver:latest
    container_name: onlyoffice
    ports:
      - "8088:80"
    environment:
      - JWT_ENABLED=false
    volumes:
      - onlyoffice_data:/var/www/onlyoffice/Data
      - onlyoffice_log:/var/log/onlyoffice
    profiles:
      - odf
    restart: unless-stopped

volumes:
  onlyoffice_data:
  onlyoffice_log:
```

Der `profiles: [odf]` Eintrag sorgt dafür, dass der Container NUR gestartet wird, wenn `make up ODF=true` oder `docker compose --profile odf up` verwendet wird. Ohne Profil bleibt er aus.

#### 6.2 Frontend — Conditional ODF Accept

```tsx
// ChatInterface.tsx + OfficePanel.tsx
const ODF_ACCEPT = process.env.NEXT_PUBLIC_ALPHARAVIS_ENABLE_ODF_UPLOAD === "true"
  ? ".odt,.odp,.ods,"
  : "";

accept={`${ODF_ACCEPT}.docx,.pptx,.xlsx,image/*,.pdf,...`}
```

#### 6.3 Backend — ODF-to-DOCX Conversion Service

**Neue Datei:** `langgraph-app/odf_converter.py`

```python
import httpx
import os

ONLYOFFICE_URL = os.getenv("ALPHARAVIS_ONLYOFFICE_URL", "http://onlyoffice:80")

async def convert_odf_to_ooxml(odf_path: str, target_format: str = "docx") -> str:
    """Convert ODF to OOXML via OnlyOffice DocumentServer API."""
    # OnlyOffice Conversion API:
    # POST /ConvertService.ashx mit multipart/form-data
    # Returns converted file
    ...
```

#### 6.4 Media Gallery — ODF MIME-Types

```python
# media_server.py UPLOAD_MIME_TYPES
"application/vnd.oasis.opendocument.text": "document",
"application/vnd.oasis.opendocument.presentation": "document",
"application/vnd.oasis.opendocument.spreadsheet": "document",
```

#### 6.5 Feature-Gate Logik im Media Server

```python
ODF_MIME_TYPES = {
    "application/vnd.oasis.opendocument.text",
    "application/vnd.oasis.opendocument.presentation", 
    "application/vnd.oasis.opendocument.spreadsheet",
}

def _is_odf_upload(mime_type: str) -> bool:
    return mime_type in ODF_MIME_TYPES

# In upload handler:
if _is_odf_upload(content_type) and not _env_bool("ALPHARAVIS_ENABLE_ODF_UPLOAD", "false"):
    raise HTTPException(status_code=415, detail="ODF uploads are not enabled. Set ALPHARAVIS_ENABLE_ODF_UPLOAD=true.")
```

### Phase 7: 2GB-Download-Limit → Warn-Dialog

**Datei:** `langgraph-app/media_analysis.py`

**Aktuell (Zeile 196-197):**
```python
if size > max_bytes:
    raise RuntimeError(f"download exceeds limit {max_bytes} bytes")
```

**Ziel:** Kein Hard-Limit mehr, sondern Warnung mit User-Override.

Da `_download_media` ein Backend-Call ist (kein UI), gibt es zwei Optionen:

1. **Env-Variable als Soft-Limit:** `ALPHARAVIS_VIDEO_ANALYSIS_MAX_DOWNLOAD_BYTES` behalten, aber als Warn-Schwelle. Bei Überschreitung: loggen, aber trotzdem downloaden.
2. **User-Interaktion im Agent-Loop:** Vor dem Download fragt der Agent den User via `request_user_input` (gibt's das?), ob die große Datei trotzdem geladen werden soll.

**Empfehlung für Phase 1:** Variante 1 (Soft-Limit). In `_download_media`:
```python
if size > max_bytes:
    logger.warning(f"Download exceeds recommended limit of {max_bytes} bytes (actual: {size}). Proceeding anyway per user preference.")
    # Nicht abbrechen, weitermachen
```
Und einen neuen Env-Flag `ALPHARAVIS_VIDEO_ANALYSIS_HONOR_SIZE_LIMIT=false` (default false), der das alte Hard-Limit-Verhalten optional wiederherstellt.

**Später (UI-Plan):** Einen echten Warn-Dialog im Frontend, der bei `POST /assets/register` mit `download=true` und großer Datei eine `409 Conflict` mit `{"warning": "file_exceeds_recommended_size", "size": ...}` zurückgibt. Frontend zeigt Dialog: "Diese Datei ist X GB groß. Trotzdem herunterladen?" → User klickt Ja → erneuter Request mit `force_download=true`.

### Phase 8: Video-Player im UI (späterer Plan)

**Das ist ein separater, späterer Plan.** Hier nur als Stichpunkte:

- Video-Player-Komponente in `deep-agents-ui`
- Thumbnail-Generierung via `ffmpegthumbnailer` oder ffmpeg
- Streaming via Media Gallery (`/media/{asset_id}/stream`)
- Unterstützte Container/Codecs via HTML5 `<video>` + Media Source Extensions
- HLS/DASH für sehr große Videos
- Verlinkung aus Chat-Nachrichten (klickbare Video-Thumbnails)

### Phase 9: Compare & OnlyOffice Interactive Editor (späterer Plan)

#### 9.1 Subtil-Vergleich im Office Panel

**Konzept:** Aufklappbare Vergleichsleiste, nicht dauerhaft sichtbar.

```
📄 final.docx  [782 KB]  ✅ Konvertiert

   ┌─ Vergleich (▶ zum Aufklappen) ────────────────────────┐
   │                                                        │
   │  original.pdf  →  converted.docx                       │
   │  [▶ Vorschau]     [▶ Vorschau]    ✓ 3/3 Seiten okay    │
   │                                                        │
   │  converted.docx  →  edited.docx                        │
   │  [▶ Vorschau]     [▶ Vorschau]    ⚡ 12 Änderungen     │
   │    └─ Absatz 3: fett                                 │
   │    └─ Tabelle 1: +2 Zeilen                          │
   │                                                        │
   │  edited.docx  →  final.pdf                             │
   │  [▶ Vorschau]     [▶ Vorschau]    ✓ Layout identisch   │
   │                                                        │
   └────────────────────────────────────────────────────────┘
```

**Viewer je nach Dateityp:**
- **DOCX/XLSX/PPTX** → `officecli view html` als iframe (schnell, kein Container nötig)
- **PDF** → Browser-Nativ via Media-Server-URL (`:8130/media/...`)
- **ODF** → Vorher via OnlyOffice konvertiert, dann wie DOCX

**"Open in OnlyOffice" Button:**
- Jede DOCX/XLSX/PPTX-Datei bekommt kleinen Button
- Klick → Datei wird an OnlyOffice geschickt → öffnet interaktiven Editor in neuem Tab/iframe
- User editiert manuell, klickt "Speichern"
- OnlyOffice-Callback informiert uns → Datei wird zurückgeholt
- Im Panel erscheint: "📝 Manuell editiert" → Agent kann mit "Weiter bearbeiten" fortfahren

#### 9.2 OnlyOffice Callback-Flow

```
┌──────────┐      ┌──────────────┐      ┌───────────┐
│  User    │      │  Media       │      │ OnlyOffice│
│  klickt  │ ───→ │  Server      │ ───→ │ Container │
│ "Edit"   │      │  :8130       │      │ :8088     │
└──────────┘      └──────────────┘      └───────────┘
                        │                      │
                        │  POST /office/edit   │
                        │  Body: {file_url}    │
                        │                      │
                        │              OnlyOffice öffnet
                        │              Editor im Browser
                        │                      │
                        │              User editiert...
                        │              Klickt "Save"
                        │                      │
                        │  ← Callback ────────┘
                        │  POST /office/callback
                        │  {status:"saved", url:"..."}
                        │
                        │  Media Server holt
                        │  geänderte Datei zurück
                        │
                  ┌─────┴─────┐
                  │  Datei    │
                  │  updated  │ → Panel: "📝 Editiert"
                  └───────────┘
```

**Endpunkte (Media Server, :8130):**
- `POST /office/edit` → lädt Datei zu OnlyOffice hoch, gibt Editor-URL zurück
- `POST /office/callback` → OnlyOffice ruft auf nach Save, wir holen Datei zurück

#### 9.3 "Weiter bearbeiten mit editierter Datei"

Nach manuellem Edit:
- Panel zeigt "📝 Manuell editiert von Amin um 14:32"
- Button: "Weiter bearbeiten" → sendet Datei als neue Version an Agent
- Agent sieht im Kontext: "User hat die Datei manuell editiert. Verwende edited_final.docx (ersetzt final.docx)"
- Agent setzt Workflow fort (z.B. PDF-Export der editierten Version)

---

## 4. Dateien (Übersicht)

| Datei | Änderung |
|---|---|
| `submodules/deep-agents-ui/src/app/components/ChatInterface.tsx` | `accept` erweitern |
| `submodules/deep-agents-ui/src/lib/file-validation.ts` | `SUPPORTED_FILE_TYPES` + ODF |
| `submodules/deep-agents-ui/src/lib/multimodal-utils.ts` | Video/Audio handling (kein Base64!) |
| `submodules/deep-agents-ui/src/app/hooks/useFileUpload.ts` | Große-Dateien-Upload-Logik |
| `submodules/deep-agents-ui/src/app/components/OfficePanel.tsx` | `accept` + Upload-Logik erweitern |
| `langgraph-app/media_server.py` | `UPLOAD_MIME_TYPES` + ODF, Upload-Endpoint |
| `langgraph-app/media_analysis.py` | 2GB Soft-Limit |
| `langgraph-app/bridge_server.py` | Video-File-Block handling prüfen |
| `langgraph-app/agent_graph.py` | `media_download` Tool (falls nötig) |

---

## 5. Offene Fragen

1. ~~**Ist LibreOffice im Container installiert?**~~ → Ja (26.2.3.2 lokal), aber für ODF-Konvertierung nehmen wir OnlyOffice (bessere Fidelity)
2. **OnlyOffice Container-Größe:** ~2-3 GB. Akzeptabel für den User?
3. **Soll `BRIDGE_MEDIA_CONTEXT_MODE` auf `metadata` bleiben?** → Ja, ist default und korrekt für Video-URL-Ersatz
4. **Video-Base64 im Frontend komplett verhindern?** → Ja, für alle `video/*` und `audio/*` MIME-Types IMMER Gallery-Upload
5. **Thumbnail-Generierung für Videos?** → Späterer Plan, aber `ffmpeg` ist schon da
6. **Max File Size für Gallery Upload?** → Aktuell kein Limit (FastAPI default). Sollten wir eins setzen?
7. **ODF in LibreChat:** LibreChat nutzt LibreOffice für Konvertierung → ODF nativ unterstützt. Kein Problem.

---

## 6. Risiken

- **Video-Upload-Latenz:** Große Videos brauchen Zeit zum Hochladen. Fortschrittsbalken im UI nötig.
- **Media Gallery Storage:** Videos fressen Plattenplatz. Retention-Policy oder Quota später nötig.
- **ODF→DOCX Konvertierung:** Nicht 100% verlustfrei. Formatierung kann abweichen.
- **Bridge-Kompatibilität:** Wenn das Frontend `{type:"file", metadata:{url:"..."}}` sendet, muss die Bridge das korrekt parsen und an die Media Gallery delegieren.
