#include <Arduino.h>
#include <ESP8266HTTPClient.h>
#include <ESP8266WebServer.h>
#include <ESP8266WiFi.h>
#include <WiFiClient.h>

#include "config.generated.h"

ESP8266WebServer server(80);

enum PendingAction {
  ACTION_NONE,
  ACTION_POWER_CYCLE,
  ACTION_POWER_ON,
  ACTION_POWER_OFF,
  ACTION_RESET
};

struct ActionState {
  PendingAction action = ACTION_NONE;
  unsigned long scheduledAtMs = 0;
  unsigned long delayMs = 0;
  unsigned long holdMs = 0;
  unsigned long waitMs = 0;
  String reason = "";
  String status = "idle";
};

ActionState pending;
unsigned long lastHeartbeatMs = 0;
String lastAction = "none";
String lastError = "";

int activeLevel() {
  return OUTPUT_ACTIVE_HIGH ? HIGH : LOW;
}

int inactiveLevel() {
  return OUTPUT_ACTIVE_HIGH ? LOW : HIGH;
}

void releaseButtonPin(int pin) {
  digitalWrite(pin, inactiveLevel());
  if (GPIO_IDLE_FLOAT) {
    pinMode(pin, INPUT);
  } else {
    pinMode(pin, OUTPUT);
  }
}

void prepareButtonPin(int pin) {
  digitalWrite(pin, inactiveLevel());
  if (GPIO_IDLE_FLOAT) {
    pinMode(pin, INPUT);
  } else {
    pinMode(pin, OUTPUT);
  }
}

void setButtonPin(int pin, bool active) {
  if (active) {
    pinMode(pin, OUTPUT);
    delay(2);
    digitalWrite(pin, activeLevel());
  } else {
    releaseButtonPin(pin);
  }
}

void pressButton(int pin, unsigned long holdMs) {
  setButtonPin(pin, true);
  delay(holdMs);
  setButtonPin(pin, false);
}

String jsonEscape(const String &value) {
  String out;
  out.reserve(value.length() + 8);
  for (size_t i = 0; i < value.length(); i++) {
    char c = value[i];
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (c == '\n') {
      out += "\\n";
    } else {
      out += c;
    }
  }
  return out;
}

void sendJson(int code, const String &body) {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(code, "application/json", body);
}

bool authorized() {
  String expected = String(ESP_AUTH_TOKEN);
  if (expected.length() == 0) {
    return true;
  }
  String supplied = server.header("Authorization");
  return supplied == "Bearer " + expected;
}

String extractJsonString(const String &body, const String &key, const String &fallback) {
  String needle = "\"" + key + "\"";
  int keyPos = body.indexOf(needle);
  if (keyPos < 0) {
    return fallback;
  }
  int colon = body.indexOf(':', keyPos + needle.length());
  if (colon < 0) {
    return fallback;
  }
  int firstQuote = body.indexOf('"', colon + 1);
  if (firstQuote < 0) {
    return fallback;
  }
  int secondQuote = body.indexOf('"', firstQuote + 1);
  if (secondQuote < 0) {
    return fallback;
  }
  return body.substring(firstQuote + 1, secondQuote);
}

long extractJsonInt(const String &body, const String &key, long fallback) {
  String needle = "\"" + key + "\"";
  int keyPos = body.indexOf(needle);
  if (keyPos < 0) {
    return fallback;
  }
  int colon = body.indexOf(':', keyPos + needle.length());
  if (colon < 0) {
    return fallback;
  }
  int start = colon + 1;
  while (start < (int)body.length() && (body[start] == ' ' || body[start] == '\t')) {
    start++;
  }
  int end = start;
  while (end < (int)body.length() && (isDigit(body[end]) || body[end] == '-')) {
    end++;
  }
  if (end <= start) {
    return fallback;
  }
  return body.substring(start, end).toInt();
}

const char *actionName(PendingAction action) {
  switch (action) {
    case ACTION_POWER_CYCLE:
      return "power-cycle";
    case ACTION_POWER_ON:
      return "power-on";
    case ACTION_POWER_OFF:
      return "power-off";
    case ACTION_RESET:
      return "reset";
    case ACTION_NONE:
    default:
      return "none";
  }
}

PendingAction parseAction(const String &value) {
  if (value == "power-cycle") {
    return ACTION_POWER_CYCLE;
  }
  if (value == "power-on") {
    return ACTION_POWER_ON;
  }
  if (value == "power-off") {
    return ACTION_POWER_OFF;
  }
  if (value == "reset") {
    return ACTION_RESET;
  }
  return ACTION_NONE;
}

String statusJson() {
  String ip = WiFi.isConnected() ? WiFi.localIP().toString() : "";
  return "{"
    "\"ok\":true,"
    "\"device_id\":\"" + jsonEscape(ESP_DEVICE_ID) + "\","
    "\"wifi_connected\":" + String(WiFi.isConnected() ? "true" : "false") + ","
    "\"ip\":\"" + jsonEscape(ip) + "\","
    "\"rssi\":" + String(WiFi.isConnected() ? WiFi.RSSI() : 0) + ","
    "\"uptime_seconds\":" + String(millis() / 1000) + ","
    "\"pending_action\":\"" + String(actionName(pending.action)) + "\","
    "\"action_status\":\"" + jsonEscape(pending.status) + "\","
    "\"last_action\":\"" + jsonEscape(lastAction) + "\","
    "\"last_error\":\"" + jsonEscape(lastError) + "\""
    "}";
}

void scheduleAction(PendingAction action, const String &reason, long delaySeconds, long holdSeconds, long waitSeconds) {
  pending.action = action;
  pending.reason = reason;
  pending.scheduledAtMs = millis();
  pending.delayMs = max(0L, delaySeconds) * 1000UL;
  pending.holdMs = max(1L, holdSeconds) * 1000UL;
  pending.waitMs = max(0L, waitSeconds) * 1000UL;
  pending.status = "queued";
  lastAction = String(actionName(action)) + " queued";
}

void executePendingAction() {
  if (pending.action == ACTION_NONE) {
    return;
  }
  if (millis() - pending.scheduledAtMs < pending.delayMs) {
    return;
  }

  PendingAction action = pending.action;
  pending.action = ACTION_NONE;
  pending.status = "executing";
  lastAction = String(actionName(action)) + " executing";

  if (action == ACTION_POWER_ON) {
    pressButton(POWER_BUTTON_PIN, pending.holdMs);
  } else if (action == ACTION_POWER_OFF) {
    pressButton(POWER_BUTTON_PIN, pending.holdMs);
  } else if (action == ACTION_POWER_CYCLE) {
    pressButton(POWER_BUTTON_PIN, pending.holdMs);
    delay(pending.waitMs);
    pressButton(POWER_BUTTON_PIN, DEFAULT_SHORT_PRESS_SECONDS * 1000UL);
  } else if (action == ACTION_RESET) {
    pressButton(RESET_BUTTON_PIN, DEFAULT_SHORT_PRESS_SECONDS * 1000UL);
  }

  pending.status = "done";
  lastAction = String(actionName(action)) + " done";
}

void handleHealth() {
  sendJson(200, "{\"ok\":true,\"service\":\"nodemcu-v3-power-controller\"}");
}

void handleStatus() {
  sendJson(200, statusJson());
}

void handleCancel() {
  if (!authorized()) {
    sendJson(401, "{\"ok\":false,\"error\":\"unauthorized\"}");
    return;
  }
  pending = ActionState();
  lastAction = "cancelled";
  sendJson(200, "{\"ok\":true,\"status\":\"cancelled\"}");
}

void handleAction() {
  if (!authorized()) {
    sendJson(401, "{\"ok\":false,\"error\":\"unauthorized\"}");
    return;
  }

  String body = server.arg("plain");
  String actionText = extractJsonString(body, "action", "");
  PendingAction action = parseAction(actionText);
  if (action == ACTION_NONE) {
    sendJson(400, "{\"ok\":false,\"error\":\"unknown action\"}");
    return;
  }

  String reason = extractJsonString(body, "reason", "api-request");
  long delaySeconds = extractJsonInt(body, "delay_before_action_seconds", DEFAULT_DELAY_BEFORE_ACTION_SECONDS);
  long defaultHoldSeconds = (action == ACTION_POWER_ON || action == ACTION_RESET)
    ? DEFAULT_SHORT_PRESS_SECONDS
    : DEFAULT_FORCE_OFF_HOLD_SECONDS;
  long holdSeconds = extractJsonInt(body, "hold_seconds", defaultHoldSeconds);
  long waitSeconds = extractJsonInt(body, "wait_seconds", DEFAULT_WAIT_AFTER_OFF_SECONDS);
  scheduleAction(action, reason, delaySeconds, holdSeconds, waitSeconds);

  sendJson(202, "{"
    "\"ok\":true,"
    "\"status\":\"queued\","
    "\"action\":\"" + String(actionName(action)) + "\","
    "\"delay_before_action_seconds\":" + String(delaySeconds) + ","
    "\"hold_seconds\":" + String(holdSeconds) + ","
    "\"wait_seconds\":" + String(waitSeconds) +
    "}");
}

int pinFromName(const String &name) {
  if (name == "power" || name == "d1" || name == "D1") {
    return POWER_BUTTON_PIN;
  }
  if (name == "reset" || name == "d2" || name == "D2") {
    return RESET_BUTTON_PIN;
  }
  return -1;
}

void drivePinLevel(int pin, const String &level) {
  if (level == "high") {
    pinMode(pin, OUTPUT);
    delay(2);
    digitalWrite(pin, HIGH);
  } else if (level == "low") {
    pinMode(pin, OUTPUT);
    delay(2);
    digitalWrite(pin, LOW);
  } else {
    releaseButtonPin(pin);
  }
}

void handlePinTest() {
  if (!authorized()) {
    sendJson(401, "{\"ok\":false,\"error\":\"unauthorized\"}");
    return;
  }

  String body = server.arg("plain");
  String pinName = extractJsonString(body, "pin", "power");
  String level = extractJsonString(body, "level", "float");
  long holdSeconds = extractJsonInt(body, "hold_seconds", 3);
  int pin = pinFromName(pinName);
  if (pin < 0 || (level != "high" && level != "low" && level != "float")) {
    sendJson(400, "{\"ok\":false,\"error\":\"use pin power/reset and level high/low/float\"}");
    return;
  }

  drivePinLevel(pin, level);
  if (level != "float" && holdSeconds > 0) {
    delay(max(1L, holdSeconds) * 1000UL);
    releaseButtonPin(pin);
  }

  lastAction = "pin-test " + pinName + " " + level;
  sendJson(200, "{"
    "\"ok\":true,"
    "\"pin\":\"" + jsonEscape(pinName) + "\","
    "\"level\":\"" + jsonEscape(level) + "\","
    "\"hold_seconds\":" + String(holdSeconds) +
    "}");
}

void sendHeartbeat() {
  if (!WiFi.isConnected()) {
    return;
  }

  String url = String(MANAGER_BASE_URL) + "/esp/heartbeat";
  String body = "{"
    "\"device_id\":\"" + jsonEscape(ESP_DEVICE_ID) + "\","
    "\"status\":\"online\","
    "\"uptime_seconds\":" + String(millis() / 1000) + ","
    "\"ip\":\"" + jsonEscape(WiFi.localIP().toString()) + "\","
    "\"rssi\":" + String(WiFi.RSSI()) +
    "}";

  WiFiClient heartbeatClient;
  HTTPClient http;
  http.setTimeout(3000);
  if (!http.begin(heartbeatClient, url)) {
    lastError = "heartbeat begin failed";
    Serial.println(lastError);
    return;
  }
  http.addHeader("Content-Type", "application/json");
  if (String(MANAGER_API_TOKEN).length() > 0) {
    http.addHeader("Authorization", "Bearer " + String(MANAGER_API_TOKEN));
  }
  int code = http.POST(body);
  if (code < 200 || code >= 300) {
    lastError = "heartbeat http " + String(code);
    Serial.println(lastError);
  } else {
    lastError = "";
  }
  http.end();
}

void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  unsigned long started = millis();
  while (!WiFi.isConnected() && millis() - started < 30000UL) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  if (WiFi.isConnected()) {
    Serial.print("WiFi connected, IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("WiFi not connected yet; will keep retrying in loop.");
  }
}

void setup() {
  Serial.begin(9600);
  delay(200);
  prepareButtonPin(POWER_BUTTON_PIN);
  prepareButtonPin(RESET_BUTTON_PIN);
  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, HIGH);

  Serial.println();
  Serial.println("NodeMCU V3 power controller starting");
  connectWifi();

  server.on("/health", HTTP_GET, handleHealth);
  server.on("/status", HTTP_GET, handleStatus);
  server.on("/action", HTTP_POST, handleAction);
  server.on("/cancel", HTTP_POST, handleCancel);
  server.on("/pin-test", HTTP_POST, handlePinTest);
  server.begin();
  Serial.println("HTTP server started on port 80");
}

void loop() {
  if (!WiFi.isConnected()) {
    connectWifi();
  }

  server.handleClient();
  executePendingAction();

  if (millis() - lastHeartbeatMs >= HEARTBEAT_INTERVAL_SECONDS * 1000UL) {
    lastHeartbeatMs = millis();
    sendHeartbeat();
  }
}
