# Ubuntu Llama Manager

Eigenständiger Manager für einen `llama.cpp`-Server auf Ubuntu/ROCm-Systemen.
Das Projekt startet und überwacht den Llama-Server, kann AMD-GPU-Powerlimits
setzen, lokale Modelle scannen, eine HTTP/JSON-API bereitstellen und den PC in
einem festen Intervall kontrolliert neu starten.

Der Intervall-Reboot ist eine gewünschte Stabilitätsfunktion, kein Fehler.

## Funktionen

- `llama.cpp` per systemd starten, stoppen, neustarten und überwachen
- kontrollierter Auto-Reboot per `llama-reboot.timer`
- optionaler Watchdog-Modus, der erst ab Llama-Start zählt
- optionales MI50/MI60-Powerlimit per `rocm-smi`
- optionales ROCm-Tuning für Performance-Level, GPU-Takt, Speicher-Takt und PCIe-Level
- optionaler ROCm/GPU-Health-Monitor mit Recovery-Entscheidung
- HTTP/JSON-API für einen späteren AI Stack
- vorbereitete NodeMCU V3 / ESP8266-Schnittstelle
- HuggingFace-/GGUF-Modellscan
- lokale Runtime-Config ohne Secrets im Repo

## Installation

Frischer lokaler Checkout oder bestehender Ordner:

```bash
cd /home/amin/experi/ubuntu-llama-manager
cp .env.example ubuntu-llama.conf
nano ubuntu-llama.conf
./install.sh check
sudo ./install.sh apply
```

## Makefile Kurzbefehle

Der einfachste Weg ist `make`. Das Makefile ruft intern weiter die normalen
Skripte auf, du musst dir aber weniger Pfade merken.

Alles installieren, Firmware flashen und danach Status anzeigen:

```bash
cd /home/amin/experi/ubuntu-llama-manager
make install PORT=/dev/ttyUSB0
```

`make install` macht standardmäßig:

- Config und Tests prüfen
- NodeMCU-Firmware aus `firmware/nodemcu-v3/.env` generieren, bauen und flashen
- `sudo ./install.sh apply` ausführen
- `sudo ./install.sh firewall-allow` ausführen
- Status anzeigen

Nur Manager/systemd aktualisieren, ohne Firmware-Flash:

```bash
make install FIRMWARE=false
```

Nur prüfen:

```bash
make check
```

Nur Manager installieren/neu anwenden:

```bash
make manager-install
```

Nur Manager-Services neu starten:

```bash
make manager-restart
```

Nur Status anzeigen:

```bash
make status
```

Wenn du UFW nicht anfassen willst:

```bash
make install FIREWALL=false
```

API und Llama-Ports nur im lokalen LAN/WLAN freigeben:

```bash
make firewall-lan
```

Nur den zweiten Llama-Server auf Port `8001` im lokalen LAN/WLAN freigeben:

```bash
make firewall-llama-lan
```

Wenn du beide Llama-Server im lokalen LAN/WLAN freigeben willst:

```bash
make firewall-llama-lan LLAMA_FIREWALL_PORTS="8033 8001"
```

`make firewall-wlan` ist ein Alias fuer `make firewall-lan`. Standardnetz ist
`FIREWALL_LAN=192.168.178.0/24`.

Firmware einzeln flashen und seriellen Monitor starten:

```bash
make flash PORT=/dev/ttyUSB0
make monitor PORT=/dev/ttyUSB0
```

Diese Aliase gehen ebenfalls:

```bash
make upload PORT=/dev/ttyUSB0
make firmware-upload PORT=/dev/ttyUSB0
make firmware-monitor PORT=/dev/ttyUSB0
```

Nur Firmware bauen, ohne zu flashen:

```bash
make firmware-build
```

Der Monitor nutzt standardmäßig `BAUD=9600`. Wenn du den ESP8266-Bootloader
sehen willst:

```bash
make monitor PORT=/dev/ttyUSB0 BAUD=74880
```

Wichtige Make-Variablen:

```bash
PORT=/dev/ttyUSB0      # serieller ESP-Port
BAUD=9600             # Monitor-Baudrate
FIRMWARE=false        # bei make install Firmware-Flash überspringen
FIREWALL=false        # bei make install UFW-Freigabe überspringen
FIREWALL_LAN=192.168.178.0/24
LLAMA_FIREWALL_PORTS="8001"
SUDO=sudo             # sudo-Befehl fuer install.sh
```

`ubuntu-llama.conf` ist absichtlich in `.gitignore`. Für GitHub bleibt nur
`.env.example` im Repo.

Wenn du noch vom alten Namen kommst:

```bash
sudo ./install.sh remove-legacy
sudo ./install.sh migrate-path
sudo ./install.sh apply
```

Wenn `ubuntu-llama.service` nach einem Reboot `enabled`, aber trotzdem
`inactive (dead)` ist und `journalctl -u ubuntu-llama.service -b` keine
Eintraege zeigt, gibt es zwei typische Ursachen:

- alte kaputte `rakam-*`-Symlinks in systemd
- systemd-Ordering-Cycle wie `Job ubuntu-llama.service/start deleted to break ordering cycle`

Dann einmal aufraeumen und neu anwenden:

```bash
cd /home/amin/experi/ubuntu-llama-manager
sudo ./install.sh remove-legacy
sudo ./install.sh apply
sudo systemctl start ubuntu-llama.service
```

Danach pruefen:

```bash
find /etc/systemd/system/multi-user.target.wants -maxdepth 1 -type l -name 'rakam-*' -print
systemctl status ubuntu-llama.service
journalctl -u ubuntu-llama.service -b
```

Prüfen:

```bash
systemctl status ubuntu-llama.service ubuntu-manager-api.service ubuntu-gpu-health.service
systemctl list-timers
curl http://127.0.0.1:8099/health
```

## Nach Änderungen aktualisieren

Wenn du Code, `ubuntu-llama.conf`, systemd-Units, ESP-Webhook-Werte oder
API-/Reboot-Einstellungen geändert hast, musst du nicht neu klonen oder alles
neu einrichten. Einmal anwenden reicht:

```bash
cd /home/amin/experi/ubuntu-llama-manager
sudo ./install.sh apply
```

`apply` schreibt die systemd-Units neu, lädt systemd neu, startet die
verwalteten Services passend zur Config und übernimmt neue Werte wie
`ESP_WEBHOOK_URL`, GPU-Powerlimits, Llama-Startbefehl oder API-Host/Port.

Wenn ESP, AI-Stack oder ein anderes Gerät im LAN die API erreichen sollen und
`ufw` aktiv ist, danach den API-Port freigeben:

```bash
sudo ./install.sh firewall-allow
```

Danach kurz prüfen:

```bash
curl http://127.0.0.1:8099/health
curl http://192.168.178.113/status
curl http://127.0.0.1:8099/esp/status
```

## Wichtige Konfiguration

```bash
ENABLE_LLAMA_SERVICE="true"
LLAMA_WORKDIR="/home/amin/experi/llama.cpp"
LLAMA_COMMAND="./build/bin/llama-server ..."
LLAMA_PORT="8033"
LLAMA_PORT_BUSY_WAIT_SECONDS="30"

# Optionaler zweiter llama.cpp-Server: eigener Service auf Port 8001.
ENABLE_LLAMA_SECONDARY_SERVICE="true"
START_LLAMA_SECONDARY_ON_BOOT="true"
LLAMA_SECONDARY_WORKDIR="/home/amin/experi/llama.cpp"
LLAMA_SECONDARY_COMMAND="./build/bin/llama-server -hf unsloth/Qwen3.5-2B-GGUF:Q4_1 --host 0.0.0.0 --port 8001 --no-mmproj -ngl 99 -c 8192 -b 1024 -ub 1024 -ctk q8_0 -ctv q8_0 -fa on --reasoning off --chat-template-kwargs '{\"enable_thinking\":false}'"
LLAMA_SECONDARY_HOST="127.0.0.1"
LLAMA_SECONDARY_PORT="8001"
LLAMA_SECONDARY_LOG_FILE="/home/amin/llama-8001.log"

ENABLE_AUTO_REBOOT="true"
REBOOT_INTERVAL_HOURS="3"
REBOOT_BACKEND="timer"
REBOOT_USE_ESP_POWER_CYCLE="false"

ENABLE_API_SERVICE="true"
API_HOST="0.0.0.0"
API_PORT="8099"
API_TOKEN="change-me"

# Optional: gemeinsame ESP/Firmware-Env mit WLAN und Tokens.
ESP_ENV_FILE="/home/amin/experi/ubuntu-llama-manager/firmware/nodemcu-v3/.env"
```

Wenn systemd andere Umgebungsvariablen braucht als deine interaktive Shell,
setze:

```bash
LLAMA_ENV_FILE="/absolute/path/to/llama.env"
LLAMA_PRE_START_SLEEP_SECONDS="10"
```

`-ngl 999` ist erlaubt, wenn du bewusst alle Layer auf die sichtbaren GPUs
laden willst. Wenn der Dienst dabei sofort mit ROCm OOM abbricht, ist der
Autostart nicht das Problem: Dann sieht Linux/ROCm nicht genug GPU-Speicher
oder nicht alle erwarteten Karten. PM2 oder systemd würden dann nur denselben
Crash neu starten.

Der zweite Server ersetzt den ersten nicht. Er bekommt `ubuntu-llama-8001.service`,
lauscht auf Port `8001` und schreibt nach `/home/amin/llama-8001.log`.

```bash
systemctl status ubuntu-llama.service ubuntu-llama-8001.service
journalctl -u ubuntu-llama-8001.service -f
curl http://127.0.0.1:8099/llama-secondary/status
```

## Bedienung

```bash
./install.sh status
./bin/llama-control.sh status
./bin/llama-control.sh logs
./install.sh gpu-show
```

Alte `rakam-*`-systemd-Units entfernen:

```bash
sudo ./install.sh remove-legacy
sudo ./install.sh apply
```

Alten Installationspfad umziehen, falls `/home/amin/experi/ubuntu-llama-manager`
noch nur ein Symlink auf `/home/amin/experi/rakam-llama-guard` ist:

```bash
sudo ./install.sh migrate-path
```

Reboot-Verhalten testen:

```bash
./bin/test-reboot.sh arm
sudo reboot
./bin/test-reboot.sh after
```

Oder Marker setzen und direkt kontrolliert rebooten:

```bash
sudo ./bin/test-reboot.sh reboot
./bin/test-reboot.sh after
```

Llama verwalten:

```bash
sudo ./bin/llama-control.sh start
sudo ./bin/llama-control.sh stop
sudo ./bin/llama-control.sh restart
```

## API

Nach `sudo ./install.sh apply` läuft die API standardmäßig auf allen Interfaces:

```bash
curl http://127.0.0.1:8099/health
curl http://<server-ip>:8099/status
curl http://<server-ip>:8099/models
```

Wenn `ufw` aktiv ist und der ESP oder ein anderes Gerät im LAN die API nicht
erreicht, den API-Port freigeben:

```bash
sudo ./install.sh firewall-allow
```

Optional enger auf dein lokales LAN/WLAN begrenzen:

```bash
sudo env API_FIREWALL_ALLOW_FROM="192.168.178.0/24" ./install.sh firewall-allow
```

Für den zweiten Llama-Server auf Port `8001`:

```bash
make firewall-llama-lan
```

Das öffnet Port `8001` nur für dein lokales LAN/WLAN, standardmäßig
`192.168.178.0/24`. Wenn du den großen Server auf `8033` auch freigeben willst:

```bash
make firewall-llama-lan LLAMA_FIREWALL_PORTS="8033 8001"
```

Alias, falls du an WLAN denkst:

```bash
make firewall-wlan
```

Oder dauerhaft in `ubuntu-llama.conf`:

```bash
API_FIREWALL_ALLOW_FROM="192.168.178.0/24"
sudo ./install.sh firewall-allow
```

Gefährliche Endpunkte benötigen:

```bash
Authorization: Bearer <API_TOKEN>
```

Setze ein starkes `API_TOKEN`, wenn `API_HOST="0.0.0.0"` aktiv ist.

Mehr dazu: [docs/api.md](docs/api.md)

Wichtige AI-Stack-Flows:

```bash
# Llama hängt oder antwortet nicht: Manager entscheidet GPU-Fehler vs. Llama-Hänger
curl -X POST \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"reason":"ai-stack-timeout","probe_timeout_seconds":20}' \
  http://127.0.0.1:8099/ai-stack/llama-no-response

# Ablauf: Llama-Generation testen, dann GPU/ROCm/PCIe pruefen.
# Wenn Llama noch Tokens liefert: keine Aktion.
# Wenn Llama haengt und GPU okay ist: Llama hart killen und neu starten.
# Wenn GPU kritisch ist: ESP-Powercycle/GPU-Fault-Flow ausloesen.

# Nur Diagnose ohne Aktion:
curl -X POST \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"reason":"ai-stack-timeout","probe_timeout_seconds":20}' \
  http://127.0.0.1:8099/ai-stack/diagnose-llama

# Modell wechseln und Llama neu starten
curl -X POST -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"unsloth/Qwen3.6-35B-A3B-GGUF:Q8_0","model_flag":"hf","restart":true}' \
  http://127.0.0.1:8099/llama/switch-model
```

## ESP Web Control

Zum Testen der Optokoppler-Verkabelung gibt es eine kleine lokale Webseite:

```text
http://127.0.0.1:8099/esp/control
http://<server-ip>:8099/esp/control
```

Bei dir ist der Server aktuell typischerweise:

```text
http://192.168.178.153:8099/esp/control
```

Die Webseite laeuft auf dem Manager-Port `8099`. Sie ist im ganzen LAN
erreichbar, wenn in `ubuntu-llama.conf` gesetzt ist:

```bash
API_HOST="0.0.0.0"
API_PORT="8099"
```

Danach Manager neu anwenden und UFW freigeben:

```bash
cd /home/amin/experi/ubuntu-llama-manager
sudo ./install.sh apply
sudo ./install.sh firewall-allow
```

Buttons:

- Power kurz: sendet `power-on`, drueckt `IN1`/Power fuer 1 Sekunde
- Neustart: sendet `power-cycle`, haelt D1/Power 8 Sekunden, wartet 20 Sekunden,
  drueckt D1/Power danach kurz zum Einschalten
- Power lang: sendet `power-off`, haelt `IN1`/Power fuer 8 Sekunden
- Pin-Test: testet Power/Reset als `HIGH`, `LOW` oder `FLOAT`

Direkt zum Pin-Test-Tab:

```text
http://192.168.178.153:8099/esp/control#pin-test
```

Beim Pin-Test sollten die Mainboard-Ausgaenge `U1/G` und `U2/G` abgezogen
sein. Beobachte zuerst nur die rote PC817-LED.

Die Webseite braucht `API_TOKEN`. Der Browser sendet an den Manager, und der
Manager sendet dann mit `ESP_WEBHOOK_TOKEN` an den ESP. Dadurch muss der
ESP-Token nicht in der Webseite stehen.

Vor dem Test pruefen:

```bash
curl http://127.0.0.1:8099/health
curl http://127.0.0.1:8099/esp/status
curl http://192.168.178.113/status
```

Die zweite API ist direkt auf dem ESP erreichbar, normalerweise Port `80`:

```text
http://<esp-ip>/status
http://<esp-ip>/action
http://<esp-ip>/cancel
```

Wichtig: Wenn der Ubuntu-PC ausgeschaltet ist, ist die Manager-API auf
`http://<server-ip>:8099` ebenfalls aus. Einschalten geht dann nur direkt ueber
den ESP:

```bash
curl -X POST \
  -H "Authorization: Bearer <ESP_AUTH_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"action":"power-on","hold_seconds":1}' \
  http://<esp-ip>/action
```

Der Manager nutzt `ESP_WEBHOOK_URL`, z. B.:

```bash
ESP_WEBHOOK_URL="http://192.168.178.113/action"
```

Direkter API-Test:

```bash
curl -X POST \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"action":"power-on","hold_seconds":1,"delay_before_action_seconds":0}' \
  http://127.0.0.1:8099/esp/action
```

Pin-Test direkt ueber den Manager:

```bash
curl -X POST \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"pin":"power","level":"high","hold_seconds":5}' \
  http://127.0.0.1:8099/esp/pin-test
```

Neustart ueber D1 direkt testen:

```bash
curl -X POST \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"action":"power-cycle","hold_seconds":8,"wait_seconds":20,"delay_before_action_seconds":0}' \
  http://127.0.0.1:8099/esp/action
```

Power lang direkt testen:

```bash
curl -X POST \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"action":"power-off","hold_seconds":8,"delay_before_action_seconds":0}' \
  http://127.0.0.1:8099/esp/action
```

Geplante ESP-Aktion abbrechen:

```bash
curl -X POST \
  -H "Authorization: Bearer $API_TOKEN" \
  http://127.0.0.1:8099/esp/cancel
```

Wichtig: Power lang kann den PC hart ausschalten. Erst Power kurz und Reset
testen, dann Power lang.

## Systemd

Bevorzugter Auto-Reboot:

```bash
sudo systemctl enable --now llama-reboot.timer
sudo systemctl disable --now llama-reboot.timer
systemctl list-timers
systemctl status llama-reboot.timer
journalctl -u llama-reboot.service
```

Mehr dazu: [docs/systemd.md](docs/systemd.md)

GPU/ROCm-Monitor:

```bash
ENABLE_GPU_HEALTH_MONITOR="true"
GPU_HEALTH_POLL_SECONDS="10"
GPU_HEALTH_CRITICAL_ACTION="shutdown"
GPU_HEALTH_PCIE_REPLAY_THRESHOLD="0"
```

Damit prüft der Manager alle 10 Sekunden ROCm/PCIe/RAS/Kernel-Fehler und
entscheidet selbstständig.

GPU-Tuning:

```bash
ENABLE_GPU_POWER_LIMIT="true"
POWER_LIMIT_WATTS="160"
ENABLE_GPU_CLOCK_TUNING="true"
GPU_PERF_LEVEL="manual"
GPU_SCLK_LEVELS="5"
GPU_MCLK_LEVELS="1"
GPU_PCIE_LEVELS="0"
```

Anwenden und prüfen:

```bash
sudo ./install.sh apply
./install.sh gpu-show
cat logs/gpu-power.log
```

Verfügbare MI50-Level anzeigen:

```bash
for d in /sys/class/drm/card*/device; do
  [ -r "$d/pp_dpm_sclk" ] || continue
  echo "=== $d ==="
  cat "$d/product_name" 2>/dev/null || true
  echo "-- power --"
  for h in "$d"/hwmon/hwmon*; do
    [ -r "$h/power1_cap" ] && cat "$h/power1_cap" "$h/power1_cap_max"
  done
  echo "-- sclk --"; cat "$d/pp_dpm_sclk"
  echo "-- mclk --"; cat "$d/pp_dpm_mclk"
  echo "-- pcie --"; cat "$d/pp_dpm_pcie"
done
```

## ESP8266 / NodeMCU

Firmware vorbereiten:

```bash
cd /home/amin/experi/ubuntu-llama-manager
cp firmware/nodemcu-v3/.env.example firmware/nodemcu-v3/.env
nano firmware/nodemcu-v3/.env
make firmware-build
```

Direkt im Firmware-Ordner geht weiterhin:

```bash
cd /home/amin/experi/ubuntu-llama-manager/firmware/nodemcu-v3
cp .env.example .env
nano .env
./build.sh
```

In `.env` stehen WLAN, Manager-URL und Tokens. Diese Datei wird nicht
committed. Du bearbeitest nur diese Datei; `config.generated.h` wird intern
automatisch erzeugt und ignoriert. Der Manager kann dieselbe `.env` lesen:

```bash
WIFI_SSID="dein-wlan"
WIFI_PASSWORD="dein-wlan-passwort"
MANAGER_BASE_URL="http://192.168.178.153:8099"
MANAGER_API_TOKEN="1234"
ESP_AUTH_TOKEN="1234"
ESP_WEBHOOK_URL="http://192.168.178.80/action"
POWER_BUTTON_PIN="D1"
RESET_BUTTON_PIN="D2"
OUTPUT_ACTIVE_HIGH="true"
GPIO_IDLE_FLOAT="true"
```

Für PC817-/Optokoppler-Module ist `GPIO_IDLE_FLOAT="true"` der sichere
Standard: D1/D2 sind im Leerlauf hochohmig und werden nur während eines echten
Power-/Reset-Klicks geschaltet. Die rote LED am PC817 darf im Leerlauf nicht
leuchten. Wenn sie dauerhaft leuchtet, sind Eingangspolarität oder Verkabelung
noch falsch und die Mainboard-Pins sollten abgezogen bleiben.

USB-Rechte unter Ubuntu 24.04:

```bash
sudo usermod -aG dialout amin
```

Danach neu einloggen oder rebooten. Prüfen:

```bash
groups
ls -l /dev/ttyUSB0
test -r /dev/ttyUSB0 && echo readable
test -w /dev/ttyUSB0 && echo writable
```

Flashen:

```bash
cd /home/amin/experi/ubuntu-llama-manager
make flash PORT=/dev/ttyUSB0
```

Wenn der Upload bei `Connecting...` hängt:

```text
FLASH gedrückt halten
kurz RESET drücken
RESET loslassen
FLASH nach 1-2 Sekunden loslassen
```

Serieller Monitor mit 9600 Baud:

```bash
cd /home/amin/experi/ubuntu-llama-manager
make monitor PORT=/dev/ttyUSB0
```

Bootloader-Ausgabe ansehen:

```bash
make monitor PORT=/dev/ttyUSB0 BAUD=74880
```

Wenn im Monitor die ESP-IP steht, `ESP_WEBHOOK_URL` in
`firmware/nodemcu-v3/.env` setzen und danach den Manager neu anwenden:

```bash
cd /home/amin/experi/ubuntu-llama-manager
sudo ./install.sh apply
sudo ./install.sh firewall-allow
```

ESP-Reboot/Power-Cycle aktivieren:

```bash
REBOOT_USE_ESP_POWER_CYCLE="true"
REBOOT_REQUIRE_ESP_WEBHOOK="true"
REBOOT_LOCAL_SHUTDOWN_AFTER_ESP="true"
REBOOT_ESP_SHUTDOWN_COMMAND="/usr/bin/systemctl poweroff"
ESP_POWER_ACTION_ON_REBOOT="power-on"
GPU_FAULT_LOCAL_SHUTDOWN_AFTER_ESP="true"
GPU_FAULT_SHUTDOWN_COMMAND="/usr/bin/systemctl poweroff"
ESP_POWER_ACTION_ON_GPU_FAULT="power-cycle"
ESP_POWER_HOLD_SECONDS="8"
ESP_POWER_WAIT_SECONDS="20"
ESP_POWER_DELAY_BEFORE_ACTION_SECONDS="20"
GPU_FAULT_REQUIRE_ESP_WEBHOOK="true"
GPU_FAULT_ESP_RETRIES="3"
GPU_FAULT_ESP_RETRY_SECONDS="2"
```

Normaler Auto-Reboot ohne GPU-Fehler: Der Manager sagt dem ESP, dass er später
kurz Power drücken soll (`power-on`), fährt Ubuntu dann sauber mit
`systemctl poweroff` herunter, und der ESP schaltet den PC nach seiner Delay-Zeit
wieder ein.

GPU-/Kernel-Fehler: Der Manager sagt dem ESP, dass er nach Delay einen harten
Power-Cycle machen soll (`power-cycle`), versucht aber vorher trotzdem ein
sauberes `systemctl poweroff`. Wenn Ubuntu wegen AMDGPU/PCIe hängt, übernimmt
der ESP über den Optokoppler. Dabei wird der Powerbutton-Pin standardmäßig
8 Sekunden kurzgeschlossen.

Bei `GPU_FAULT_REQUIRE_ESP_WEBHOOK="true"` startet der Manager den lokalen
Shutdown erst, wenn der ESP den Powercycle-Auftrag bestätigt hat. Ohne Antwort
vom ESP wird nicht heruntergefahren, damit der PC nicht einfach aus bleibt.

Normaler AIStack-Shutdown soll ohne ESP laufen:

```http
POST /power/shutdown
Authorization: Bearer <API_TOKEN>
```

Mehr dazu: [docs/esp-nodemcu.md](docs/esp-nodemcu.md)

## Troubleshooting

Typische Ursachen, wenn systemd nicht startet, obwohl manueller Start klappt:

- relative Pfade oder `~` im Befehl
- falsches Working Directory
- fehlende ROCm/HIP/PATH-Variablen
- falscher User
- GPU, Netzwerk oder Mounts noch nicht bereit
- Befehl enthält `nohup`, `&` oder `disown`

Mehr dazu: [docs/troubleshooting.md](docs/troubleshooting.md)

## GitHub

Dieses Projekt ist eigenständig und soll nicht ins AI-Stack-Repo. Lokal:

```bash
git init
git add .
git commit -m "Improve manager APIs, reboot timer, and ESP integration"
```

Mehr dazu: [docs/github-upload.md](docs/github-upload.md)
