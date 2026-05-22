# Systemd

Das Projekt erzeugt systemd-Units aus der lokalen Config.

## Units

```text
ubuntu-gpu-power.service
ubuntu-gpu-health.service
ubuntu-llama.service
ubuntu-llama-8001.service
ubuntu-manager-api.service
llama-reboot.service
llama-reboot.timer
ubuntu-reboot-watch.service
```

`ubuntu-reboot-watch.service` ist der Fallback für `REBOOT_TIMER_MODE="llama-start"`.
Der bevorzugte Weg ist `llama-reboot.timer`.

## Installieren

```bash
cd /home/amin/experi/ubuntu-llama-manager
./install.sh check
sudo ./install.sh apply
```

## Auto-Reboot

Config:

```bash
ENABLE_AUTO_REBOOT="true"
REBOOT_INTERVAL_HOURS="3"
REBOOT_BACKEND="timer"
```

Timer-Befehle:

```bash
sudo systemctl enable --now llama-reboot.timer
sudo systemctl disable --now llama-reboot.timer
systemctl list-timers
systemctl status llama-reboot.timer
journalctl -u llama-reboot.service
```

Der Timer ruft `bin/reboot-now.sh` auf. Das Skript prüft vor dem Reboot nochmal
die Config, schreibt Logs und führt dann kontrolliert `systemctl reboot` aus.

Wenn Reboot/Power-Cycle über den ESP laufen soll:

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
ESP_NOTIFY_SETTLE_SECONDS="2"
GPU_FAULT_REQUIRE_ESP_WEBHOOK="true"
GPU_FAULT_ESP_RETRIES="3"
GPU_FAULT_ESP_RETRY_SECONDS="2"
```

Dann ist die Reihenfolge:

1. `bin/reboot-now.sh` schreibt `state/esp-request.json`.
2. Wenn `ESP_WEBHOOK_URL` gesetzt ist, sendet es den Request direkt an den ESP.
3. Bei `REBOOT_REQUIRE_ESP_WEBHOOK="true"` wird ohne erfolgreiche ESP-Meldung
   kein Poweroff ausgeführt.
4. Beim normalen Auto-Reboot sendet der Manager `ESP_POWER_ACTION_ON_REBOOT`,
   standardmäßig `power-on`, und führt danach `REBOOT_ESP_SHUTDOWN_COMMAND`
   aus, standardmäßig `systemctl poweroff`.
5. Der ESP wartet seine Delay-Zeit und drückt dann kurz Power, damit der PC
   wieder startet.
6. Bei `gpu-health` wird stattdessen `ESP_POWER_ACTION_ON_GPU_FAULT` genutzt,
   standardmäßig `power-cycle`. Das ist die harte Absicherung, wenn AMDGPU/PCIe
   hängt und Ubuntu nicht mehr zuverlässig ausgeht. Der ESP hält den
   Powerbutton-Optokoppler dabei `ESP_POWER_HOLD_SECONDS` Sekunden, standardmäßig
   8 Sekunden, aktiv.
   Bei `GPU_FAULT_REQUIRE_ESP_WEBHOOK="true"` wird der lokale Shutdown nur
   gestartet, wenn der ESP den Auftrag bestätigt hat.

## Llama-Service

```bash
sudo systemctl start ubuntu-llama.service
sudo systemctl stop ubuntu-llama.service
sudo systemctl restart ubuntu-llama.service
systemctl status ubuntu-llama.service
journalctl -u ubuntu-llama.service
```

Optionaler zweiter Server:

```bash
sudo systemctl start ubuntu-llama-8001.service
sudo systemctl stop ubuntu-llama-8001.service
sudo systemctl restart ubuntu-llama-8001.service
systemctl status ubuntu-llama-8001.service
journalctl -u ubuntu-llama-8001.service
```

Config:

```bash
ENABLE_LLAMA_SECONDARY_SERVICE="true"
START_LLAMA_SECONDARY_ON_BOOT="true"
LLAMA_SECONDARY_WORKDIR="/home/amin/experi/llama.cpp"
LLAMA_SECONDARY_PORT="8001"
LLAMA_SECONDARY_LOG_FILE="/home/amin/llama-8001.log"
```

Der Llama-Befehl muss im Vordergrund laufen. Kein `nohup`, kein `&`, kein
`disown`.

## API-Service

```bash
systemctl status ubuntu-manager-api.service
journalctl -u ubuntu-manager-api.service
```

Die API läuft standardmäßig auf `0.0.0.0:8099`, also auf allen Interfaces.
Wenn `ufw` aktiv ist, gib den Port frei:

```bash
sudo ./install.sh firewall-allow
```

## GPU-Powerlimit

```bash
systemctl status ubuntu-gpu-power.service
journalctl -u ubuntu-gpu-power.service
```

Config:

```bash
ENABLE_GPU_POWER_LIMIT="true"
POWER_LIMIT_GPU_IDS="all"
POWER_LIMIT_WATTS="160"
```

Der gleiche Service kann auch Takt-/Performance-Level setzen:

```bash
ENABLE_GPU_CLOCK_TUNING="true"
GPU_PERF_LEVEL="manual"
GPU_SCLK_LEVELS="5"
GPU_MCLK_LEVELS="1"
GPU_PCIE_LEVELS="0"
RESET_CLOCKS_ON_DISABLE="false"
```

Leere Werte werden übersprungen. Die Level sind die `rocm-smi`-DPM-Level der
Karte; vorher anzeigen:

```bash
./install.sh gpu-show
rocm-smi --showclkfrq
rocm-smi --showperflevel
```

## GPU-Health-Monitor

Der Health-Monitor prüft regelmäßig ROCm, PCIe Replay Counts, RAS, XGMI und
Kernel/AER-Logs. Er entscheidet dadurch autonom, ohne dass der AI Stack erst
nachfragen muss:

```bash
ENABLE_GPU_HEALTH_MONITOR="true"
GPU_HEALTH_POLL_SECONDS="10"
GPU_HEALTH_CRITICAL_ACTION="shutdown"
GPU_HEALTH_PCIE_REPLAY_THRESHOLD="0"
```

Mögliche Aktionen:

- `none`: nur loggen und ESP-Request vorbereiten
- `force-kill-llama`: Llama hart killen
- `reboot`: kontrollierter Software-Reboot
- `shutdown`: `systemctl poweroff`

Für den späteren ESP-Hard-Reset:

```bash
ESP_WEBHOOK_URL="http://<esp-ip>/action"
ESP_POWER_HOLD_SECONDS="8"
ESP_POWER_WAIT_SECONDS="20"
ESP_POWER_DELAY_BEFORE_ACTION_SECONDS="20"
ESP_NOTIFY_SETTLE_SECONDS="2"
GPU_FAULT_REQUIRE_ESP_WEBHOOK="true"
```

Bei `GPU_HEALTH_CRITICAL_ACTION="shutdown"` wird zuerst der ESP-Powercycle
beauftragt. Der ESP soll dann `ESP_POWER_DELAY_BEFORE_ACTION_SECONDS` warten,
bevor er den Optokoppler/Powerknopf betätigt. Danach wartet der Manager kurz
`ESP_NOTIFY_SETTLE_SECONDS`, damit der Request raus ist, und startet dann
`systemctl poweroff`.

Der Service läuft als root-systemd-Service. Dadurch braucht er kein
sudo-Passwort in der Config. Falls du ihn irgendwann manuell als normaler User
laufen lässt, ist eine enge sudoers-Regel sicherer als ein Passwort in `.env`.

Die letzte Entscheidung steht hier:

```bash
cat state/last-gpu-decision.json
```

Geprüft werden unter anderem:

- `rocm-smi --showreplaycount`
- `rocm-smi --showrasinfo`
- `rocm-smi --showxgmierr`
- Kernel-Logs auf PCIe/AER/amdgpu-Fehler

PCIe/AER-Fehler kommen zuverlässig aus Kernel-Logs (`journalctl -k`) und teils
aus ROCm-Zählern. Dafür ist root/systemd der richtige Weg; kein sudo-Passwort
in der Config speichern.
