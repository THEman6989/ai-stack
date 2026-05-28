# ComfyUI Proxy/Submit Hardening

## Ziel

Die bestehende AlphaRavis-ComfyUI-Integration gezielt härten, ohne neue Default-ON Features einzuführen:

1. `/comfyui/view` muss bei `ALPHARAVIS_COMFYUI_API_BASE=unix:///workspace/runtime/comfyui.sock` wirklich über den Unix-Socket/Relay laufen.
2. Der ComfyUI-Tab muss Proxy-Antworten mit `ok:false` und Submit-Blocks mit `blocked:true` als Fehler/Block behandeln, nicht als Erfolg.
3. Live Submit im UI muss immer über Media-Gallery laufen: zuerst Proxy-Preflight, dann Proxy-Submit.
4. Agent-Submit soll optional/konfigurierbar denselben Media-Gallery-Submit-Pfad nutzen, damit Policy/Gating zentral bleiben.

## Kontext

- Host-ComfyUI läuft auf `http://localhost:8188`.
- Container nutzen wegen Docker→Host-Barriere den Relay-Socket `runtime/comfyui.sock` → `unix:///workspace/runtime/comfyui.sock`.
- `media-gallery` stellt `/comfyui/status`, `/queue`, `/models`, `/history`, `/preflight`, `/prompt`, `/view` bereit.
- Live Submit bleibt über `ALPHARAVIS_ENABLE_COMFYUI_WORKFLOW_SUBMIT=false` default OFF.
- UI-Feature-Flags bleiben default OFF/build-time.

## Umsetzung

### 1. `/comfyui/view` UDS-fähig

- In `langgraph-app/comfyui_client.py` eine öffentliche Methode ergänzen, z.B. `view_bytes(filename, subfolder='', file_type='output')`.
- Diese Methode nutzt intern `_async_client()` und `_url('/view?...')`, damit Unix-Socket-Transport und normale HTTP-Bases identisch funktionieren.
- `langgraph-app/media_server.py` `/comfyui/view` nutzt diese Methode statt `client.view_url(...)` + plain `httpx.AsyncClient()`.
- Test in `tests/test_comfyui_client.py`/`tests/test_media_server.py`: UDS-Client ruft `/view` über Client-Transport auf, nicht über Public URL.

### 2. UI App-Level-Errors korrekt behandeln

- In `ComfyUIPanel.tsx` Fetch-Helper oder Call-Sites so anpassen, dass Proxy-Payloads mit `ok === false` als Fehler gelten.
- Für Submit explizit prüfen:
  - `submitResult.ok === false`
  - `submitResult.blocked === true`
  - `submitResult.result?.blocked === true`
- UI-Log/Result: `blocked by backend` statt `submitted`.

### 3. Live Submit nur Proxy-Preflight + Proxy-Submit

- `submitWorkflow()` darf im Live-Modus nicht mehr direct-preflighten.
- Es nimmt immer `proxyBase`/DEFAULT_PROXY und ruft:
  - `POST /preflight`
  - bei ready: `POST /prompt`
- Direct/local preflight bleibt nur für Draft/Inspect-Modus.

### 4. Agent-Submit optional über Media-Gallery vereinheitlichen

- In `agent_graph.py` `submit_comfyui_workflow` optional über `MEDIA_GALLERY_URL/comfyui/prompt` routen.
- Neue Env ist default ON oder ohne neue Env? Minimal-invasiv bevorzugt: Env `ALPHARAVIS_COMFYUI_AGENT_SUBMIT_VIA_MEDIA_GALLERY=true`, aber kein neues Risiko, da eigentlicher Submit weiter durch `ALPHARAVIS_ENABLE_COMFYUI_WORKFLOW_SUBMIT=false` geblockt bleibt.
- Fallback: wenn Media-Gallery nicht erreichbar oder Env false, bestehender `client.submit_workflow()` Pfad bleibt möglich.

## Tests / Verification

- RED/GREEN:
  - gezielte Tests für neuen Client `/view` Byte-Fetch.
  - media-server `/comfyui/view` nutzt Client-Methode.
  - Agent-Submit ruft Media-Gallery `/comfyui/prompt`, wenn Env aktiv ist.
- Bestehende Tests:
  - `python -m pytest -q tests/test_comfyui_client.py tests/test_media_server.py tests/test_alpharavis_toolsets.py`
- UI:
  - `npx eslint src/app/components/ComfyUIPanel.tsx src/app/page.tsx` im deep-agents-ui Submodule.
  - falls möglich: `docker compose build deep-agents-ui`.
- Runtime smoke:
  - `GET http://127.0.0.1:8130/comfyui/status` → `ok:true`
  - `GET http://127.0.0.1:8130/comfyui/view?...` → nicht mehr 502 bei existierendem Output.
  - Submit disabled smoke: `/comfyui/prompt` → `ok:false`, `blocked:true`.

## Docs

- `docs/ALPHARAVIS_CHANGES.md`: Änderung + Verification.
- `docs/ALPHARAVIS_ARCHITECTURE.md`: `/comfyui/view` ist UDS-backed; Submit-Pfad zentralisiert.
- `docs/ALPHARAVIS_USAGE_NOTES.md`: Live Submit nutzt immer Proxy; Agent-Submit-Route/Fallback erwähnen.
- `docs/ALPHARAVIS_OPEN_TASKS.md`: diese vier Hardening-Punkte als umgesetzt/verified markieren.

## Risiken

- UI-Tests sind im Submodule aktuell dünn; ohne neuen Test-Runner primär ESLint/Docker-Build + manuelle Browser-Smokes.
- `agent_graph.py` ist groß; Änderung muss minimal bleiben und keine Graph-Imports brechen.
- Runtime-Env in bestehenden Containern braucht nach Backend-Codeänderung ggf. Hot reload oder `docker compose up -d --force-recreate langgraph-api media-gallery`.
