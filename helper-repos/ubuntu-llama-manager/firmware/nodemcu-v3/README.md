# NodeMCU V3 Firmware

ESP8266 power-controller firmware for Ubuntu Llama Manager.

## Configure

```bash
cd /home/amin/experi/ubuntu-llama-manager/firmware/nodemcu-v3
cp .env.example .env
nano .env
```

Set:

```bash
WIFI_SSID="DEIN_WLAN_NAME"
WIFI_PASSWORD="DEIN_WLAN_PASSWORT"
MANAGER_BASE_URL="http://192.168.178.153:8099"
MANAGER_API_TOKEN="1234"
ESP_AUTH_TOKEN="1234"
ESP_WEBHOOK_URL="http://192.168.178.80/action"
POWER_BUTTON_PIN="D1"
RESET_BUTTON_PIN="D2"
OUTPUT_ACTIVE_HIGH="true"
GPIO_IDLE_FLOAT="true"
```

You only edit `.env`. Build/upload generates the internal
`config.generated.h` automatically; it is ignored by git.

Ubuntu Llama Manager also reads this `.env`, so tokens stay in one place:
`MANAGER_API_TOKEN` becomes `API_TOKEN`, and `ESP_AUTH_TOKEN` becomes
`ESP_WEBHOOK_TOKEN`.

`ESP_WEBHOOK_URL` is read by the manager, not by the firmware. After flashing,
set it to the ESP address, for example `http://192.168.178.113/action`, then
restart or re-apply the manager service.

For PC817/opto-coupler modules keep `GPIO_IDLE_FLOAT="true"` while testing.
That leaves the GPIO pins high impedance when idle and drives them only during
a real button action. The red PC817 LED must be off while idle and should light
only during a short/long press.

## Build

```bash
./build.sh
```

## Upload

```bash
PORT=/dev/ttyUSB0 ./upload.sh
```

If `/dev/ttyUSB0` is not readable/writable:

```bash
sudo usermod -aG dialout amin
```

Then log out and back in, or reboot.

Temporary upload with sudo:

```bash
sudo env PATH="$HOME/.local/bin:$PATH" PORT=/dev/ttyUSB0 ./upload.sh
```

## Monitor

```bash
PORT=/dev/ttyUSB0 BAUD=9600 ./monitor.sh
```

If you only see unreadable characters directly after reset, that is usually
the ESP8266 boot ROM at 74880 baud. Press RESET once while this monitor is open,
or check the boot ROM with:

```bash
PORT=/dev/ttyUSB0 BAUD=74880 ./monitor.sh
```

After ESP power loss the firmware boots again, reconnects to WiFi, and starts
the HTTP server. Pending actions are not persisted across ESP power loss.

## HTTP Endpoints

```text
GET  /health
GET  /status
POST /action
POST /cancel
POST /pin-test
```

Use these direct ESP endpoints when the Ubuntu host is powered off. In that
state the manager API on port `8099` is offline, so power-on must go directly to
the ESP.

Example:

```bash
curl -X POST \
  -H "Authorization: Bearer <ESP_AUTH_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"action":"power-cycle","reason":"manual-test","delay_before_action_seconds":3,"hold_seconds":8,"wait_seconds":20}' \
  http://<esp-ip>/action
```

Direct opto-coupler input test without the mainboard connected:

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
