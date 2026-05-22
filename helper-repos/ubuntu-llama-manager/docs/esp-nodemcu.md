# NodeMCU V3 / ESP8266

Die ESP-Schnittstelle ist vorbereitet, aber noch nicht an echte Hardware
gebunden. Das Programm läuft vollständig weiter, wenn kein ESP vorhanden ist.

## Firmware flashen

Der NodeMCU V3 mit CH340/CH340C taucht unter Ubuntu typischerweise als
`/dev/ttyUSB0` auf:

```bash
ls -l /dev/ttyUSB0
```

Wenn die Datei zur Gruppe `dialout` gehört, muss dein User Mitglied dieser
Gruppe sein:

```bash
sudo usermod -aG dialout amin
```

Danach einmal neu einloggen oder rebooten. Alternativ geht Upload kurzfristig
mit `sudo`, aber dauerhaft ist `dialout` angenehmer.

Die Firmware liegt hier:

```bash
cd /home/amin/experi/ubuntu-llama-manager/firmware/nodemcu-v3
cp .env.example .env
nano .env
```

Wichtige Werte in `.env`:

```bash
WIFI_SSID="DEIN_WLAN_NAME"
WIFI_PASSWORD="DEIN_WLAN_PASSWORT"
MANAGER_BASE_URL="http://192.168.178.153:8099"
MANAGER_API_TOKEN="1234"
ESP_AUTH_TOKEN="1234"
ESP_WEBHOOK_URL="http://192.168.178.80/action"
```

Du bearbeitest nur `.env`. Build/Upload erzeugt intern automatisch
`config.generated.h`; diese Datei ist in `.gitignore`, damit WLAN-Passwort und
Tokens nicht ins Repo kommen.

Ubuntu Llama Manager liest dieselbe `.env` ebenfalls. Dadurch musst du Tokens
nicht doppelt pflegen:

- `MANAGER_API_TOKEN` wird im Manager als `API_TOKEN` verwendet.
- `ESP_AUTH_TOKEN` wird im Manager als `ESP_WEBHOOK_TOKEN` verwendet.
- `ESP_WEBHOOK_URL` wird nur vom Manager verwendet und kann nach dem Flashen
  mit der ESP-IP gesetzt werden.

Arduino CLI installieren, falls noch nicht vorhanden:

```bash
mkdir -p ~/.local/bin
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | BINDIR=$HOME/.local/bin sh
export PATH="$HOME/.local/bin:$PATH"
arduino-cli version
```

Build und Upload:

```bash
./build.sh
PORT=/dev/ttyUSB0 ./upload.sh
```

Seriellen Monitor mit 9600 Baud öffnen:

```bash
PORT=/dev/ttyUSB0 BAUD=9600 ./monitor.sh
```

Wenn direkt nach Reset nur unlesbare Zeichen kommen, ist das meistens der
ESP8266-Bootloader mit 74880 Baud. Warte kurz oder drücke RESET einmal, während
der 9600-Baud-Monitor offen ist. Den Bootloader selbst kannst du so ansehen:

```bash
PORT=/dev/ttyUSB0 BAUD=74880 ./monitor.sh
```

Wenn Upload wegen Rechten scheitert, pruefe `dialout` oder teste einmal:

```bash
sudo env PATH="$PATH" PORT=/dev/ttyUSB0 ./upload.sh
```

## Firmware-Endpunkte

Die Firmware stellt auf dem ESP bereit:

```http
GET /health
GET /status
POST /action
POST /cancel
POST /pin-test
```

`POST /action` erwartet Bearer-Auth mit `ESP_AUTH_TOKEN`:

```bash
curl -X POST \
  -H "Authorization: Bearer <ESP_AUTH_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"action":"power-cycle","reason":"manual-test","delay_before_action_seconds":3,"hold_seconds":8,"wait_seconds":20}' \
  http://<esp-ip>/action
```

`power-cycle` bedeutet: nach Delay Power-Taste lange halten, warten, dann kurz
druecken zum Einschalten. Nutze fuer den Power-Optokoppler einen sicheren Pin
wie `D1`/GPIO5. Vermeide Boot-Strapping-Pins `D3`, `D4` und `D8`.

Bei PC817-/Optokoppler-Modulen sollte die Firmware mit
`GPIO_IDLE_FLOAT="true"` laufen. Dann sind `D1` und `D2` im Leerlauf
hochohmig; die rote LED am Optokoppler darf im Leerlauf nicht leuchten. Wenn
sie dauerhaft leuchtet oder der PC sofort ausgeht, die Mainboard-Ausgaenge
(`U1/G`, `U2/G`) abziehen und zuerst nur die ESP-zu-Optokoppler-Seite testen.

Direkter Eingangstest fuer die rote PC817-LED:

```bash
curl -X POST \
  -H "Authorization: Bearer <ESP_AUTH_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"pin":"power","level":"high","hold_seconds":5}' \
  http://<esp-ip>/pin-test

curl -X POST \
  -H "Authorization: Bearer <ESP_AUTH_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"pin":"power","level":"low","hold_seconds":5}' \
  http://<esp-ip>/pin-test
```

Der Level, bei dem die LED leuchtet, ist die aktive Polaritaet fuer dein Modul.
Leuchtet `high`, nutze `OUTPUT_ACTIVE_HIGH="true"`. Leuchtet `low`, nutze
`OUTPUT_ACTIVE_HIGH="false"`.

## Heartbeat

```http
POST /esp/heartbeat
Content-Type: application/json
```

Body:

```json
{
  "device_id": "nodemcu-v3-main",
  "status": "online",
  "uptime_seconds": 12345
}
```

Antwort:

```json
{
  "ok": true,
  "heartbeat": {
    "device_id": "nodemcu-v3-main",
    "status": "online",
    "uptime_seconds": 12345,
    "received_at": "2026-05-13T00:00:00Z"
  }
}
```

## Status

```http
GET /esp/status
```

Antwort:

```json
{
  "ok": true,
  "esp": {
    "esp_online": true,
    "last_heartbeat": "2026-05-13T00:00:00Z",
    "device_id": "nodemcu-v3-main",
    "direct_status": {
      "ok": true,
      "url": "http://192.168.178.113/status"
    },
    "pending_request": null
  }
}
```

Wenn der ESP den Manager nicht per Heartbeat erreicht, fragt der Manager
zusaetzlich direkt `GET /status` am ESP ab, sobald `ESP_WEBHOOK_URL` gesetzt
ist. Dadurch bleibt `/esp/status` brauchbar, auch wenn der Heartbeat in einem
Netzwerk noch blockiert ist.

## Vorbereitete Power-Aktionen

Diese Endpunkte sind geschützt und führen aktuell keine echte Hardware-Aktion
aus. Sie legen nur einen Request in `state/esp-request.json` ab.

```http
POST /esp/request-power-cycle
POST /esp/request-power-on
POST /esp/request-power-off
```

Beispiel:

```bash
curl -X POST \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"reason":"manual test"}' \
  http://127.0.0.1:8099/esp/request-power-cycle
```

Später kann eine Firmware diese Requests pollen oder eine direkte
Kommunikation ergänzt werden.

## Direkter Webhook

Optional kann der Manager bei kritischen GPU-Fehlern direkt eine ESP-URL
aufrufen. Der gleiche Webhook kann auch für geplante oder manuelle Reboots
genutzt werden:

```bash
ESP_WEBHOOK_URL="http://192.168.1.50/action"
ESP_WEBHOOK_TOKEN="optional-token"
ESP_POWER_HOLD_SECONDS="8"
ESP_POWER_WAIT_SECONDS="20"
ESP_POWER_DELAY_BEFORE_ACTION_SECONDS="20"
ESP_NOTIFY_SETTLE_SECONDS="2"
ESP_POWER_ACTION_ON_GPU_FAULT="power-cycle"
GPU_FAULT_REQUIRE_ESP_WEBHOOK="true"
GPU_FAULT_ESP_RETRIES="3"
GPU_FAULT_ESP_RETRY_SECONDS="2"
REBOOT_USE_ESP_POWER_CYCLE="true"
REBOOT_REQUIRE_ESP_WEBHOOK="true"
REBOOT_LOCAL_SHUTDOWN_AFTER_ESP="true"
REBOOT_ESP_SHUTDOWN_COMMAND="/usr/bin/systemctl poweroff"
ESP_POWER_ACTION_ON_REBOOT="power-on"
GPU_FAULT_LOCAL_SHUTDOWN_AFTER_ESP="true"
GPU_FAULT_SHUTDOWN_COMMAND="/usr/bin/systemctl poweroff"
```

Der gesendete JSON-Body enthält `action`, `reason`, `hold_seconds`,
`wait_seconds`, `delay_before_action_seconds` und eine kurze
Diagnose-Zusammenfassung. Der ESP sollte trotzdem eigene Default-Werte haben,
falls Felder fehlen.

Wenn der ESP selbst kurz stromlos wird, bootet er danach normal neu, verbindet
sich wieder mit WLAN und startet seinen HTTP-Server. Bereits geplante Aktionen
liegen nur im RAM und sind nach ESP-Stromverlust weg. Wenn der ganze PC
stromlos war, muss entweder das BIOS/UEFI den Rechner nach Stromrueckkehr
einschalten oder der ESP muss wieder versorgt werden und den Powerbutton per
Optokoppler druecken koennen.

Bei `REBOOT_REQUIRE_ESP_WEBHOOK="true"` führt der Manager keinen Poweroff aus,
wenn der ESP-WebHook fehlt oder nicht antwortet. Das verhindert, dass der PC
aus bleibt, solange noch kein ESP angeschlossen ist.

Bei `GPU_FAULT_REQUIRE_ESP_WEBHOOK="true"` gilt das gleiche für AMDGPU/PCIe/RAS-
Fehler: Der Manager fährt Ubuntu erst herunter, wenn der ESP den Powercycle-
Auftrag bestätigt hat. Vorher versucht er den Webhook gemäß
`GPU_FAULT_ESP_RETRIES` mehrfach.

Beim normalen Auto-Reboot nutzt der Manager `ESP_POWER_ACTION_ON_REBOOT="power-on"`:
Er beauftragt den ESP, später kurz den Powerbutton zu drücken, und fährt Ubuntu
danach sauber per `systemctl poweroff` herunter.

Bei GPU-/Kernel-Fehlern nutzt der Manager
`ESP_POWER_ACTION_ON_GPU_FAULT="power-cycle"`: Er beauftragt den ESP mit einem
harten Powercycle nach Delay und versucht vorher trotzdem ein sauberes
`systemctl poweroff`. Wenn Ubuntu wegen AMDGPU/PCIe hängt, übernimmt der ESP.
Der ESP hält dabei den Powerbutton-Pin über den Optokoppler für
`ESP_POWER_HOLD_SECONDS`, standardmäßig 8 Sekunden, kurzgeschlossen.

Bei GPU-kritischem Shutdown ist die Reihenfolge:

1. Manager erkennt ROCm/PCIe/RAS-Fehler.
2. Manager sendet ESP-Request für Hard-Reset/Powercycle und wartet auf
   Bestätigung.
3. ESP wartet `delay_before_action_seconds`, standardmäßig 20 Sekunden.
4. Manager startet nach `ESP_NOTIFY_SETTLE_SECONDS` den Software-Shutdown.
5. ESP betätigt danach den Optokoppler/Powerknopf als harte Absicherung und
   schaltet den PC nach `wait_seconds` wieder ein.

## Sicherheit

- Power-Aktionen brauchen Bearer-Auth.
- Heartbeat kann optional ebenfalls Auth erzwingen:

```bash
ESP_HEARTBEAT_AUTH_REQUIRED="true"
```

- Ohne ESP bleibt die API normal nutzbar.
