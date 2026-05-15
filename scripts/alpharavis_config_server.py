from __future__ import annotations

import argparse
import json
import socket
import sys
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
EXAMPLE_PATH = ROOT / ".env(exaple)"

BOOLEAN_VALUES = {"true", "false"}
SECRET_MARKERS = ("KEY", "PASSWORD", "SECRET", "TOKEN", "PASS", "CREDS")
URL_MARKERS = ("URL", "URI", "API_BASE", "BASE_URL", "HOST")


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def ensure_env() -> None:
    if not ENV_PATH.exists():
        ENV_PATH.write_text(EXAMPLE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        return

    current = read_env(ENV_PATH)
    defaults = read_env(EXAMPLE_PATH)
    missing = [(key, value) for key, value in defaults.items() if key not in current]
    if not missing:
        return

    with ENV_PATH.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write("\n\n# Added by make config from .env(exaple)\n")
        for key, value in missing:
            fh.write(f"{key}={value}\n")


def update_env_value(key: str, value: str) -> None:
    lines = ENV_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[str] = []
    found = False
    for line in lines:
        if line.strip().startswith("#") or "=" not in line:
            out.append(line)
            continue
        current_key = line.split("=", 1)[0].strip()
        if current_key == key:
            out.append(f"{key}={value}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")


def _clean_comment(line: str) -> str:
    return line.strip()[1:].strip()


def _is_section_title(text: str) -> bool:
    return bool(text) and text.upper() == text and not text.startswith("=")


def parse_env_template(path: Path) -> list[dict[str, object]]:
    sections: list[dict[str, object]] = []
    current_section = "General"
    comments: list[str] = []
    awaiting_section_title = False
    defaults = read_env(path)

    def section_bucket(title: str) -> dict[str, object]:
        for section in sections:
            if section["title"] == title:
                return section
        section = {"title": title, "entries": []}
        sections.append(section)
        return section

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if stripped.startswith("# ====="):
            awaiting_section_title = True
            comments = []
            continue
        if stripped.startswith("#"):
            text = _clean_comment(stripped)
            if awaiting_section_title and _is_section_title(text):
                current_section = text
                section_bucket(current_section)
                awaiting_section_title = False
                comments = []
                continue
            if text and not text.startswith("="):
                comments.append(text)
            continue
        awaiting_section_title = False
        if not stripped:
            comments = []
            continue
        if "=" not in stripped:
            comments = []
            continue
        key = stripped.split("=", 1)[0].strip()
        default = defaults.get(key, "")
        description = " ".join(comments[-5:])
        section_bucket(current_section)["entries"].append(
            {
                "key": key,
                "default": default,
                "description": description,
            }
        )
        comments = []
    return [section for section in sections if section["entries"]]


def infer_kind(key: str, value: str, description: str) -> str:
    lowered = value.strip().lower()
    desc_lower = description.lower()
    if lowered in BOOLEAN_VALUES or "allowed values: true, false" in desc_lower:
        return "bool"
    if value.startswith(("http://", "https://")) or any(marker in key for marker in URL_MARKERS):
        return "url"
    if key.endswith("_PORT") or key.endswith("_SECONDS") or key.endswith("_LIMIT") or key.endswith("_CHARS"):
        return "number"
    return "text"


def build_config_model() -> dict[str, object]:
    ensure_env()
    template = parse_env_template(EXAMPLE_PATH)
    current = read_env(ENV_PATH)
    sections: list[dict[str, object]] = []
    for section in template:
        entries: list[dict[str, object]] = []
        for raw_entry in section["entries"]:
            key = str(raw_entry["key"])
            default = str(raw_entry["default"])
            value = current.get(key, default)
            description = str(raw_entry.get("description", ""))
            entries.append(
                {
                    "key": key,
                    "value": value,
                    "default": default,
                    "description": description,
                    "kind": infer_kind(key, value or default, description),
                    "secret": any(marker in key for marker in SECRET_MARKERS),
                    "changed": value != default,
                }
            )
        sections.append({"title": section["title"], "entries": entries})
    return {
        "envPath": str(ENV_PATH),
        "examplePath": str(EXAMPLE_PATH),
        "sections": sections,
    }


def _template_keys() -> set[str]:
    keys: set[str] = set()
    for section in parse_env_template(EXAMPLE_PATH):
        for entry in section["entries"]:
            keys.add(str(entry["key"]))
    return keys


def apply_config_updates(values: dict[str, object]) -> int:
    ensure_env()
    allowed = _template_keys()
    updated = 0
    for key, value in values.items():
        if key not in allowed:
            continue
        text = "" if value is None else str(value).replace("\r", "").replace("\n", " ")
        update_env_value(key, text)
        updated += 1
    return updated


def defaults_payload() -> dict[str, str]:
    return read_env(EXAMPLE_PATH)


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AlphaRavis Config</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101418;
      --panel: #171d22;
      --panel-2: #20272e;
      --line: #36414b;
      --text: #eef3f7;
      --muted: #9ba9b5;
      --accent: #2fb67d;
      --accent-2: #d9a441;
      --danger: #e05d5d;
      --focus: #75b8ff;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); }
    header {
      position: sticky; top: 0; z-index: 4;
      display: flex; align-items: center; justify-content: space-between; gap: 16px;
      padding: 16px 22px; background: rgba(16, 20, 24, 0.96);
      border-bottom: 1px solid var(--line);
    }
    h1 { margin: 0; font-size: 20px; font-weight: 700; letter-spacing: 0; }
    .path { color: var(--muted); font-size: 12px; margin-top: 4px; overflow-wrap: anywhere; }
    .search { width: min(460px, 40vw); }
    input, select {
      width: 100%; min-height: 38px; border: 1px solid var(--line); border-radius: 6px;
      padding: 8px 10px; background: #0e1216; color: var(--text); font: inherit;
    }
    input:focus, select:focus, button:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
    main { padding: 20px 22px 92px; }
    nav {
      display: flex; gap: 8px; margin-bottom: 18px;
      overflow-x: auto; padding-bottom: 8px;
      scrollbar-width: none; -ms-overflow-style: none;
    }
    nav::-webkit-scrollbar { display: none; }
    nav button, .actions button, .row button {
      min-height: 36px; border: 1px solid var(--line); border-radius: 6px;
      background: var(--panel-2); color: var(--text); padding: 7px 11px; cursor: pointer;
      transition: background 0.1s, border-color 0.1s;
    }
    nav button { white-space: nowrap; flex-shrink: 0; }
    nav button:hover, .row button:hover { background: var(--line); }
    nav button.active { border-color: var(--accent); color: #dff7ec; background: rgba(47, 182, 125, 0.1); }
    section { margin-bottom: 28px; }
    h2 { font-size: 17px; margin: 0 0 12px; letter-spacing: 0; }
    .grid { display: grid; gap: 10px; }
    .row {
      display: grid; grid-template-columns: minmax(240px, 34%) minmax(240px, 1fr) 96px;
      gap: 12px; align-items: start; padding: 12px;
      background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
    }
    .key { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; overflow-wrap: anywhere; }
    .desc { color: var(--muted); margin-top: 6px; font-size: 12px; line-height: 1.4; }
    .default { color: var(--muted); margin-top: 8px; font-size: 12px; overflow-wrap: anywhere; }
    .changed .key::after { content: " changed"; color: var(--accent-2); font-family: inherit; font-size: 11px; }
    .bool { display: inline-flex; border: 1px solid var(--line); border-radius: 6px; overflow: hidden; min-height: 38px; }
    .bool button { border: 0; border-radius: 0; min-width: 72px; background: #0e1216; }
    .bool button.selected { background: var(--accent); color: #04110b; font-weight: 700; }
    .reset-one { width: 100%; }
    .footer {
      position: fixed; left: 18px; right: 18px; bottom: 18px; display: flex; gap: 10px; z-index: 5;
      background: rgba(16, 20, 24, 0.94); padding: 10px; border: 1px solid var(--line); border-radius: 8px;
      backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
    }
    .save { background: var(--accent) !important; color: #04110b !important; border-color: var(--accent) !important; font-weight: 700; flex: 1; }
    .reset-all { border-color: var(--danger) !important; color: #ffdede !important; }
    .status { color: var(--muted); align-self: center; min-width: 140px; font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .hidden { display: none; }
    @media (max-width: 780px) {
      header { display: block; padding: 12px 16px; }
      .search { width: 100%; margin-top: 12px; }
      .row { grid-template-columns: 1fr; gap: 8px; }
      .row > div:nth-child(2) { margin-top: 4px; }
      main { padding: 16px 16px 120px; }
    }
    @media (max-width: 480px) {
      .footer { bottom: 0; left: 0; right: 0; border-radius: 0; border-left: 0; border-right: 0; flex-wrap: wrap; justify-content: space-between; }
      .status { width: 100%; text-align: center; margin-bottom: 4px; order: -1; }
      .reset-all, .save { flex: 1; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>AlphaRavis Config</h1>
      <div class="path" id="path"></div>
    </div>
    <input id="filter" class="search" placeholder="Filter settings">
  </header>
  <main>
    <nav id="tabs"></nav>
    <div id="content"></div>
  </main>
  <div class="footer">
    <span class="status" id="status">Loading...</span>
    <button class="reset-all" id="resetAll">Reset all</button>
    <button class="save" id="save">Save</button>
  </div>
  <script>
    let model = null;
    let active = "";
    const values = {};

    const content = document.getElementById("content");
    const tabs = document.getElementById("tabs");
    const filter = document.getElementById("filter");
    const status = document.getElementById("status");

    function fieldId(key) { return "field-" + key.replaceAll("_", "-"); }
    function setStatus(text) { status.textContent = text; }
    function escapeHtml(text) {
      return String(text).replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
    }
    function currentValue(entry) { return Object.prototype.hasOwnProperty.call(values, entry.key) ? values[entry.key] : entry.value; }
    function updateValue(key, value) {
      values[key] = value;
      const row = document.querySelector(`[data-key="${CSS.escape(key)}"]`);
      if (row) row.classList.toggle("changed", value !== row.dataset.default);
    }
    function renderField(entry) {
      const value = currentValue(entry);
      if (entry.kind === "bool") {
        return `<div class="bool" role="group" aria-label="${escapeHtml(entry.key)}">
          <button type="button" data-bool="${escapeHtml(entry.key)}" data-value="true" class="${value === "true" ? "selected" : ""}">True</button>
          <button type="button" data-bool="${escapeHtml(entry.key)}" data-value="false" class="${value === "false" ? "selected" : ""}">False</button>
        </div>`;
      }
      const type = entry.secret ? "password" : (entry.kind === "number" ? "number" : "text");
      return `<input id="${fieldId(entry.key)}" data-input="${escapeHtml(entry.key)}" type="${type}" value="${escapeHtml(value)}" autocomplete="off">`;
    }
    function render() {
      const query = filter.value.trim().toLowerCase();
      tabs.innerHTML = model.sections.map(section => `<button type="button" class="${section.title === active ? "active" : ""}" data-tab="${escapeHtml(section.title)}">${escapeHtml(section.title)}</button>`).join("");
      content.innerHTML = model.sections.map(section => {
        const rows = section.entries.filter(entry => {
          const haystack = `${entry.key} ${entry.description} ${currentValue(entry)}`.toLowerCase();
          return (!active || section.title === active) && (!query || haystack.includes(query));
        }).map(entry => {
          const value = currentValue(entry);
          return `<div class="row ${value !== entry.default ? "changed" : ""}" data-key="${escapeHtml(entry.key)}" data-default="${escapeHtml(entry.default)}">
            <div>
              <div class="key">${escapeHtml(entry.key)}</div>
              <div class="desc">${escapeHtml(entry.description || "No description in .env(exaple).")}</div>
              <div class="default">Default: ${escapeHtml(entry.default || "(empty)")}</div>
            </div>
            <div>${renderField(entry)}</div>
            <button type="button" class="reset-one" data-reset="${escapeHtml(entry.key)}">Reset</button>
          </div>`;
        }).join("");
        return rows ? `<section><h2>${escapeHtml(section.title)}</h2><div class="grid">${rows}</div></section>` : "";
      }).join("");
      bindRows();
    }
    function findEntry(key) {
      for (const section of model.sections) {
        const found = section.entries.find(entry => entry.key === key);
        if (found) return found;
      }
      return null;
    }
    function bindRows() {
      document.querySelectorAll("[data-input]").forEach(input => {
        input.addEventListener("input", event => updateValue(event.target.dataset.input, event.target.value));
      });
      document.querySelectorAll("[data-bool]").forEach(button => {
        button.addEventListener("click", event => {
          const key = event.target.dataset.bool;
          updateValue(key, event.target.dataset.value);
          render();
        });
      });
      document.querySelectorAll("[data-reset]").forEach(button => {
        button.addEventListener("click", event => {
          const entry = findEntry(event.target.dataset.reset);
          if (!entry) return;
          updateValue(entry.key, entry.default);
          render();
        });
      });
    }
    tabs.addEventListener("click", event => {
      if (!event.target.dataset.tab) return;
      active = event.target.dataset.tab;
      render();
    });
    filter.addEventListener("input", render);
    document.getElementById("resetAll").addEventListener("click", () => {
      if (!confirm("Reset all settings to .env(exaple) defaults?")) return;
      for (const section of model.sections) for (const entry of section.entries) values[entry.key] = entry.default;
      render();
      setStatus("All reset locally");
    });
    document.getElementById("save").addEventListener("click", async () => {
      setStatus("Saving...");
      const payload = {};
      for (const section of model.sections) for (const entry of section.entries) payload[entry.key] = currentValue(entry);
      const response = await fetch("/api/config", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ values: payload }) });
      const result = await response.json();
      if (!response.ok) {
        setStatus(result.error || "Save failed");
        return;
      }
      model = await (await fetch("/api/config")).json();
      for (const key of Object.keys(values)) delete values[key];
      document.getElementById("path").textContent = model.envPath;
      setStatus(`Saved ${result.updated} values`);
      render();
    });
    async function boot() {
      model = await (await fetch("/api/config")).json();
      active = model.sections[0]?.title || "";
      document.getElementById("path").textContent = model.envPath;
      setStatus("Ready");
      render();
    }
    boot().catch(err => setStatus(String(err)));
  </script>
</body>
</html>
"""


class ConfigHandler(BaseHTTPRequestHandler):
    server_version = "AlphaRavisConfig/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self) -> None:
        body = HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self._send_html()
            return
        if self.path == "/api/config":
            self._send_json(build_config_model())
            return
        if self.path == "/api/defaults":
            self._send_json(defaults_payload())
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path != "/api/config":
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            values = payload.get("values", {})
            if not isinstance(values, dict):
                raise ValueError("values must be an object")
            updated = apply_config_updates(values)
        except Exception as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self._send_json({"ok": True, "updated": updated, "envPath": str(ENV_PATH)})


def find_free_port(host: str, preferred: int) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free port found from {preferred} to {preferred + 49}")


def serve(host: str, port: int, *, open_browser: bool) -> None:
    selected_port = find_free_port(host, port)
    server = ThreadingHTTPServer((host, selected_port), ConfigHandler)
    url = f"http://{host}:{selected_port}"
    print(f"AlphaRavis config UI: {url}")
    print("Press Ctrl+C to stop the config server.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nConfig server stopped.")
    finally:
        server.server_close()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AlphaRavis .env config web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true", help="Do not try to open the browser automatically.")
    args = parser.parse_args(argv)
    serve(args.host, args.port, open_browser=not args.no_open)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
