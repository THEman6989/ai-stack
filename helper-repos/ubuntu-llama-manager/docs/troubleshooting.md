# Troubleshooting

## Manuell klappt es, systemd nicht

Prüfe zuerst:

```bash
systemctl status ubuntu-llama.service
journalctl -u ubuntu-llama.service -n 200
tail -n 200 /home/amin/llama.log
```

Typische Ursachen:

- `LLAMA_WORKDIR` ist falsch oder existiert beim Boot noch nicht.
- `LLAMA_COMMAND` nutzt relative Pfade außerhalb von `LLAMA_WORKDIR`.
- `LLAMA_COMMAND` enthält `nohup`, `&` oder `disown`.
- systemd hat andere Environment-Variablen als deine Shell.
- ROCm/HIP-Pfade fehlen.
- GPU, Netzwerk oder Mounts sind beim Service-Start noch nicht bereit.
- Der Dienst läuft als falscher User.
- `-ngl 999` oder `--gpu-layers all` erzwingt alle Layer auf die sichtbaren
  GPUs. Das ist okay, wenn genug GPUs/VRAM sichtbar sind. Wenn llama.cpp mit
  ROCm OOM beendet, sieht Linux/ROCm zu wenig GPU-Speicher.

Hilfen:

```bash
LLAMA_ENV_FILE="/absolute/path/to/llama.env"
LLAMA_PRE_START_SLEEP_SECONDS="10"
```

Wenn `/home/amin/llama.log` so etwas zeigt:

```text
cudaMalloc failed: out of memory
failed to allocate ROCm0 buffer
main: exiting due to model loading error
```

dann ist der Autostart selbst nicht kaputt. Der konfigurierte
`LLAMA_COMMAND` passt nicht in den gerade sichtbaren VRAM. Prüfe zuerst, ob
alle erwarteten Karten wirklich sichtbar sind:

```bash
rocm-smi --showid --showmeminfo vram
tail -n 200 /home/amin/llama.log
```

Wenn du bewusst alles auf GPU laden willst, darf `-ngl 999` bleiben. Dann muss
aber genug sichtbarer VRAM vorhanden sein. Als Fallback zum Starten mit
CPU-Overflow kann man testweise nutzen:

```bash
-ngl auto
```

statt:

```bash
-ngl 999
```

## Port prüfen

```bash
./bin/llama-control.sh port
ss -ltnp 'sport = :8033'
```

Wenn im Llama-Log steht:

```text
couldn't bind HTTP server socket, hostname: 0.0.0.0, port: 8033
```

dann ist nicht das Modell selbst abgestürzt. Der Server konnte den Port nicht
öffnen, meistens weil noch ein alter Service oder ein manuell gestarteter
`llama-server` läuft. Prüfe dann:

```bash
systemctl is-enabled ubuntu-llama.service rakam-llama.service
systemctl status ubuntu-llama.service rakam-llama.service
ss -ltnp 'sport = :8033'
pgrep -af llama-server
```

Beim Umstieg vom alten Projektnamen müssen die alten Units aus sein:

```bash
sudo ./install.sh remove-legacy
sudo ./install.sh apply
```

Wenn der neue Pfad noch ein Symlink auf den alten Ordner ist:

```bash
ls -ld /home/amin/experi/ubuntu-llama-manager /home/amin/experi/rakam-llama-guard
sudo ./install.sh migrate-path
```

Danach schreibt `install.sh` die `ubuntu-*`-Units neu, sodass sie direkt auf
`/home/amin/experi/ubuntu-llama-manager` zeigen.

`start-llama.sh` prüft `LLAMA_PORT` vor dem Modellladen. Mit
`LLAMA_PORT_BUSY_WAIT_SECONDS` kannst du einstellen, wie lange der Service auf
einen freien Port wartet, bevor er klar mit Fehlercode `98` abbricht.

## Prozess prüfen

```bash
./bin/llama-control.sh process
pgrep -af llama-server
```

## Auto-Reboot prüfen

```bash
systemctl list-timers
systemctl status llama-reboot.timer
journalctl -u llama-reboot.service
tail -n 100 logs/reboot.log
```

Reboot-Startverhalten gezielt testen:

```bash
./bin/test-reboot.sh arm
sudo reboot
./bin/test-reboot.sh after
```

Oder direkt ueber das Testskript:

```bash
sudo ./bin/test-reboot.sh reboot
./bin/test-reboot.sh after
```

Der Check schreibt einen Report nach `state/reboot-test-report.txt` und prueft
unter anderem Boot-ID, alte `rakam-*`-Units, `ubuntu-llama.service`,
Llama-Port, API, GPU-Health-Service und `llama-reboot.timer`.

Wenn `REBOOT_TIMER_MODE="llama-start"` gesetzt ist, wird statt des Timers der
Watchdog genutzt:

```bash
systemctl status ubuntu-reboot-watch.service
tail -n 100 logs/reboot-watch.log
```

## GPU-Powerlimit prüfen

```bash
./install.sh gpu-show
journalctl -u ubuntu-gpu-power.service
tail -n 100 logs/gpu-power.log
```

GPU-Takt-/Speicher-Level prüfen:

```bash
rocm-smi --showclkfrq
rocm-smi --showclocks
rocm-smi --showperflevel
```

Setzen über Config:

```bash
ENABLE_GPU_CLOCK_TUNING="true"
GPU_PERF_LEVEL="manual"
GPU_SCLK_LEVELS="5"
GPU_MCLK_LEVELS="1"
GPU_PCIE_LEVELS="0"
```

## GPU-/ROCm-Fehler prüfen

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m ubuntu_manager.monitor --config ubuntu-llama.conf --once
curl http://127.0.0.1:8099/diagnostics/gpu
systemctl status ubuntu-gpu-health.service
journalctl -u ubuntu-gpu-health.service -n 200
cat state/last-gpu-decision.json
```

Wichtig: Überschriften wie `XGMI Error status` sind nicht automatisch Fehler.
Der Monitor sucht nach echten Kernel-/ROCm-Fehlermustern oder nicht-null RAS
Uncorrectable Errors.

## Llama hängt, GPU ist aber okay

Vom AI Stack aus:

```bash
curl -X POST \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"reason":"ai-stack-timeout"}' \
  http://127.0.0.1:8099/recovery/llama-no-response
```

Wenn keine GPU-Fehler gefunden werden, wird Llama hart beendet und neu
gestartet.

## API prüfen

```bash
curl http://127.0.0.1:8099/health
curl http://<server-ip>:8099/health
curl http://127.0.0.1:8099/status
systemctl status ubuntu-manager-api.service
journalctl -u ubuntu-manager-api.service
```

Wenn die API im Netzwerk erreichbar sein soll:

```bash
API_HOST="0.0.0.0"
API_PORT="8099"
```

Dann `sudo ./install.sh apply` oder mindestens `sudo systemctl restart
ubuntu-manager-api.service` ausführen. Gefährliche Endpunkte bleiben per
Bearer-Token geschützt.

Wenn `curl http://127.0.0.1:8099/health` funktioniert, aber ESP/AI-Stack aus
dem LAN nicht durchkommen, ist oft `ufw` aktiv:

```bash
sudo ./install.sh firewall-allow
```

Optional nur fuer das lokale LAN/WLAN:

```bash
sudo env API_FIREWALL_ALLOW_FROM="192.168.178.0/24" ./install.sh firewall-allow
```

Llama-Port `8001` nur im lokalen LAN/WLAN freigeben:

```bash
make firewall-llama-lan
```

Mehrere Llama-Ports:

```bash
make firewall-llama-lan LLAMA_FIREWALL_PORTS="8033 8001"
```

Oder `API_FIREWALL_ALLOW_FROM` dauerhaft in `ubuntu-llama.conf` setzen und dann:

```bash
sudo ./install.sh firewall-allow
```

## PCIe/AER-Rechte

`rocm-smi --showreplaycount` ist oft ohne sudo lesbar. Kernel-/PCIe-/AER-Fehler
kommen aber zuverlässig über `journalctl -k`; dafür braucht der Monitor root
oder passende Journal-Rechte. Der `ubuntu-gpu-health.service` läuft deshalb als
root-systemd-Service und braucht kein sudo-Passwort in `ubuntu-llama.conf`.

## Config neu anwenden

Nach Änderungen an `ubuntu-llama.conf`:

```bash
./install.sh check
sudo ./install.sh apply
```
