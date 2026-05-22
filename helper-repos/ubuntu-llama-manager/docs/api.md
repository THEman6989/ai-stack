# API

Die API ist für den späteren AI Stack vorbereitet. Der AI Stack soll Client
dieser API sein; dieses Projekt bleibt eigenständig.

Standard:

```text
http://0.0.0.0:8099
```

Von einem anderen Rechner aus nutzt du die Server-IP:

```bash
curl http://<server-ip>:8099/health
```

## Auth

Gefährliche Endpunkte benötigen:

```http
Authorization: Bearer <API_TOKEN>
```

`API_TOKEN` steht nur in der lokalen Runtime-Config, nicht im Repo.
Wenn `API_HOST="0.0.0.0"` gesetzt ist, muss `API_TOKEN` stark sein, weil die
API im Netzwerk erreichbar ist.

## Öffentliche Endpunkte

```http
GET /health
GET /status
GET /models
GET /models/{id}
GET /llama/status
GET /llama/config
GET /llama-secondary/status
GET /llama-secondary/config
GET /llama/instances
GET /llama/instances/{id}
GET /reboot/status
GET /esp/status
GET /esp/control
GET /diagnostics/gpu
POST /esp/heartbeat
```

Beispiel:

```bash
curl http://127.0.0.1:8099/health
```

Antwort:

```json
{
  "ok": true,
  "service": "ubuntu-llama-manager",
  "config_loaded": true
}
```

## Geschützte Endpunkte

```http
POST /llama/start
POST /llama/stop
POST /llama/restart
POST /llama/config
POST /llama/force-kill
POST /llama/switch-model
POST /llama-secondary/start
POST /llama-secondary/stop
POST /llama-secondary/restart
POST /llama-secondary/config
POST /llama/instances/{id}/config
POST /reboot/enable
POST /reboot/disable
POST /reboot/now
POST /power/shutdown
POST /diagnostics/handle-gpu-fault
POST /ai-stack/diagnose-llama
POST /ai-stack/llama-no-response
POST /recovery/llama-no-response
POST /esp/action
POST /esp/cancel
POST /esp/request-power-cycle
POST /esp/request-power-on
POST /esp/request-power-off
```

Beispiel:

```bash
curl -X POST \
  -H "Authorization: Bearer $API_TOKEN" \
  http://127.0.0.1:8099/llama/restart
```

`POST /power/shutdown` ist der normale AIStack-Shutdown: Es wird nur
`systemctl poweroff` ausgeführt, ohne ESP-Powercycle. Für geplante
GPU-Reinitialisierung nutze `POST /reboot/now` oder den Auto-Reboot-Timer.

Wichtig: Die Ubuntu-Manager-API kann den PC nur steuern, solange Ubuntu laeuft.
Wenn der PC ausgeschaltet ist, ist `http://<server-ip>:8099` nicht erreichbar.
Zum Einschalten muss der AI Stack oder ein Bediengeraet den ESP direkt
ansprechen, z. B. `POST http://<esp-ip>/action` mit `ESP_AUTH_TOKEN`.

## ESP Web Control

Die Test-Webseite laeuft ueber die Manager-API auf Port `8099`:

```text
http://<server-ip>:8099/esp/control
```

Sie bietet Buttons fuer:

- Power kurz: `power-on`, 1 Sekunde
- Neustart: `power-cycle`, D1/Power 8 Sekunden halten, 20 Sekunden warten,
  D1/Power kurz zum Einschalten druecken
- Power lang: `power-off`, 8 Sekunden
- Pin-Test: Power/Reset `HIGH`, `LOW` oder `FLOAT`

Die Seite fragt nach `API_TOKEN` und speichert ihn lokal im Browser. Der Manager
leitet Aktionen an `ESP_WEBHOOK_URL` weiter und setzt dabei `ESP_WEBHOOK_TOKEN`.

Direkt zum Pin-Test-Tab:

```text
http://<server-ip>:8099/esp/control#pin-test
```

Direkte API fuer die Webseite:

```bash
curl -X POST \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"action":"power-on","hold_seconds":1,"delay_before_action_seconds":0}' \
  http://127.0.0.1:8099/esp/action
```

Unterstuetzte Aktionen:

- `power-on`
- `power-off`
- `power-cycle`
- `reset`

Direkter Pin-Test ueber den Manager:

```bash
curl -X POST \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"pin":"power","level":"high","hold_seconds":5}' \
  http://127.0.0.1:8099/esp/pin-test
```

Geplante ESP-Aktion abbrechen:

```bash
curl -X POST -H "Authorization: Bearer $API_TOKEN" \
  http://127.0.0.1:8099/esp/cancel
```

## ESP Direkt-API

Die Manager-API leitet ESP-Aktionen weiter, solange Ubuntu laeuft:

```bash
curl -X POST \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"action":"power-on","hold_seconds":1}' \
  http://<server-ip>:8099/esp/action
```

Wenn der PC aus ist, laeuft die Manager-API nicht. Dann muss der ESP direkt
angesprochen werden:

```bash
curl -X POST \
  -H "Authorization: Bearer <ESP_AUTH_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"action":"power-on","hold_seconds":1}' \
  http://<esp-ip>/action
```

Direkte ESP-Endpunkte:

```http
GET  http://<esp-ip>/health
GET  http://<esp-ip>/status
POST http://<esp-ip>/action
POST http://<esp-ip>/cancel
POST http://<esp-ip>/pin-test
```

Direkte ESP-Aktionen:

- `power-on`: Powerbutton kurz druecken
- `power-off`: Powerbutton lange halten
- `power-cycle`: Powerbutton lange halten, warten, Powerbutton kurz druecken
- `reset`: Reset-Pin kurz druecken, falls angeschlossen

`POST /action`, `/cancel` und `/pin-test` auf dem ESP nutzen
`Authorization: Bearer <ESP_AUTH_TOKEN>`, nicht `API_TOKEN`.

## Status

```http
GET /status
```

Antwort enthält:

- Llama-Service-Status
- Prozessprüfung
- Portprüfung
- Auto-Reboot-Status
- GPU-Powerlimit-Service
- API-Port

## Modelle

```http
GET /models
GET /models/{id}
```

`GET /models` scannt die konfigurierten lokalen Modellordner, z. B.
`HF_CACHE_DIR` oder `MODEL_SCAN_DIRS`, und liefert Modell-IDs, Namen, Pfade,
Dateien, Groesse und Aenderungszeit. Der AI Stack kann daraus ein Modell
auswaehlen und den Wert als `model` an die Llama-Config-API senden.

## Llama-Instanzen Fuer AI Stack

Der Manager kennt zwei Llama-Instanzen:

- `primary`: `ubuntu-llama.service`, Config-Key `LLAMA_COMMAND`
- `secondary`: `ubuntu-llama-8001.service`, Config-Key `LLAMA_SECONDARY_COMMAND`

Aktuellen Zustand und Startbefehl beider Instanzen abfragen:

```bash
curl http://127.0.0.1:8099/llama/instances
```

Einzelne Instanz:

```bash
curl http://127.0.0.1:8099/llama/instances/primary
curl http://127.0.0.1:8099/llama/instances/secondary
```

Die Antwort enthaelt unter anderem:

```json
{
  "id": "secondary",
  "service": "ubuntu-llama-8001.service",
  "configured": true,
  "active": true,
  "workdir": "/home/amin/experi/llama.cpp",
  "command_key": "LLAMA_SECONDARY_COMMAND",
  "command": "./build/bin/llama-server -hf ... -c 8192 ...",
  "host": "127.0.0.1",
  "port": 8001,
  "port_open": true,
  "log_file": "/home/amin/llama-8001.log"
}
```

### Model Oder Kontext Aendern

Der AI Stack kann nur Modell und Kontextgroesse patchen. Alle anderen Flags
bleiben erhalten:

```bash
curl -X POST \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"unsloth/Qwen3.5-2B-GGUF:Q4_1","model_flag":"hf","context_size":16384,"restart":true}' \
  http://127.0.0.1:8099/llama/instances/secondary/config
```

`context_size` ersetzt `-c`, `--ctx-size`, `--ctx_size`, `--context` oder
`--context-size`. Wenn kein Kontext-Flag existiert, wird `-c <wert>` angehaengt.

### Kompletten Command-Block Ersetzen

Wenn der AI Stack den ganzen Startblock neu erzeugt, kann er ihn komplett
senden. Dadurch wird nur der Command-Key der gewaehlten Instanz ersetzt:

```bash
curl -X POST \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"command":"./build/bin/llama-server -hf unsloth/Qwen3.5-2B-GGUF:Q4_1 --host 0.0.0.0 --port 8001 --no-mmproj -ngl 99 -c 8192 -b 1024 -ub 1024 -ctk q8_0 -ctv q8_0 -fa on --reasoning off --chat-template-kwargs '\''{\"enable_thinking\":false}'\''","restart":true}' \
  http://127.0.0.1:8099/llama/instances/secondary/config
```

Kurzformen:

```http
GET  /llama/config
POST /llama/config
GET  /llama-secondary/config
POST /llama-secondary/config
```

`restart=true` ist Standard. Dann macht der Manager bewusst `systemctl stop`
und danach `systemctl start` fuer genau diese Instanz. Das alte Modell ist also
aus, bevor der neue Startbefehl geladen wird. Bei `restart=false` wird nur die
lokale Config geschrieben; der neue Befehl wird dann beim naechsten
Service-Start aktiv.

## Modell Wechseln

Der AI Stack kann nur das Modell wechseln, während Kontextgröße, Jinja,
Batch-Parameter und weitere Flags erhalten bleiben.

```bash
curl -X POST \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"unsloth/Qwen3.6-35B-A3B-GGUF:Q8_0","model_flag":"hf","restart":true}' \
  http://127.0.0.1:8099/llama/switch-model
```

`model_flag`:

- `auto`: ersetzt vorhandenes `-hf`, `--hf`, `-m` oder `--model`
- `hf`: setzt HuggingFace-Modell via `-hf`
- `local`: setzt lokale Datei via `-m`

Der Manager schreibt `LLAMA_COMMAND` in der lokalen Config um und startet den
systemd-Service per Stop/Start neu, wenn `restart=true` ist. Dadurch wird das
alte Modell vor dem neuen Start entladen.

## Recovery bei AI-Stack-Fehlern

Wenn der AI Stack keine Antwort mehr vom Llama-Server bekommt:

```bash
curl -X POST \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"reason":"ai-stack-timeout","probe_timeout_seconds":20}' \
  http://127.0.0.1:8099/ai-stack/llama-no-response
```

Nur Diagnose ohne Kill/Restart/ESP-Aktion:

```bash
curl -X POST \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"reason":"ai-stack-timeout","probe_timeout_seconds":20}' \
  http://127.0.0.1:8099/ai-stack/diagnose-llama
```

Alias fuer Rueckwaertskompatibilitaet:

```http
POST /recovery/llama-no-response
```

Der Manager entscheidet in dieser Reihenfolge:

- Probe an Llama senden: `POST /completion` mit kurzem Prompt.
- Wenn Llama generierten Text liefert: keine Recovery ausfuehren.
- Wenn Llama keine Tokens/Text liefert: ROCm/PCIe/Kernel-Diagnose ausfuehren.
- GPU/ROCm kritisch: ESP-Powercycle/GPU-Fault-Flow ausfuehren.
- keine GPU-Fehler: Llama hart mit `pkill -9` beenden und Service neu starten.

`POST /ai-stack/diagnose-llama` fuehrt nur die ersten drei Schritte aus und
liefert `decision`, `probe` und `gpu` zurueck. `POST /ai-stack/llama-no-response`
setzt die Entscheidung auch direkt um.

Wichtige Probe-Config:

```bash
LLAMA_PROBE_BASE_URL=""
LLAMA_PROBE_PATH="/completion"
LLAMA_PROBE_PROMPT="Reply with exactly: ok"
LLAMA_PROBE_MAX_TOKENS="8"
LLAMA_PROBE_TIMEOUT_SECONDS="20"
LLAMA_PROBE_REQUIRE_CONTENT="true"
```

Beispiel-Antwort:

```json
{
  "ok": true,
  "decision": "llama-hung",
  "reason": "ai-stack-timeout",
  "probe": {
    "ok": false,
    "url": "http://127.0.0.1:8033/completion",
    "content_received": false
  },
  "gpu": {
    "critical": false
  }
}
```

GPU-Diagnose direkt:

```bash
curl http://127.0.0.1:8099/diagnostics/gpu
```

Der Scanner durchsucht `MODEL_SCAN_DIRS` oder `HF_CACHE_DIR`. Er erkennt unter
anderem:

- `.gguf`
- `.safetensors`
- `.bin`
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`

Beispiel-Antwort:

```json
{
  "ok": true,
  "models": [
    {
      "id": "snapshot-abc123",
      "name": "example/model",
      "path": "/home/amin/.cache/huggingface/...",
      "files": ["model.gguf", "tokenizer.json"],
      "size_bytes": 123456789,
      "modified_at": "2026-05-13T00:00:00Z"
    }
  ]
}
```
