from __future__ import annotations

import os
import time
import uuid
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel


BRIDGE_BASE_URL = os.getenv("TEST_UI_BRIDGE_BASE_URL", "http://api-bridge:8123/v1").rstrip("/")
BRIDGE_MODEL = os.getenv("TEST_UI_MODEL", "my-agent")
BRIDGE_TIMEOUT_SECONDS = float(os.getenv("TEST_UI_BRIDGE_TIMEOUT_SECONDS", "240"))

app = FastAPI(title="AlphaRavis Bridge Test UI")


class ChatRequest(BaseModel):
    message: str
    messages: list[dict[str, Any]] = []
    protocol: str = "responses"
    stream: bool = False
    session_id: str = ""
    trace_id: str = ""


def _extract_responses_text(payload: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        for part in item.get("content", []):
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                chunks.append(part["text"])
    return "".join(chunks)


def _extract_chat_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content", "")
    return content if isinstance(content, str) else str(content)


def _extract_trace(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    trace = metadata.get("alpha_trace")
    if not isinstance(trace, dict):
        trace = payload.get("alpharavis_trace")
    return trace if isinstance(trace, dict) else {}


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(HTML, headers={"Cache-Control": "no-store"})


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "bridge_base_url": BRIDGE_BASE_URL,
        "model": BRIDGE_MODEL,
    }


@app.post("/api/send")
async def send_chat(request: ChatRequest) -> JSONResponse:
    text = request.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="message is required")

    started = time.perf_counter()
    trace_id = request.trace_id.strip() or f"trace_{uuid.uuid4().hex[:12]}"
    session_id = request.session_id.strip() or f"session_{uuid.uuid4().hex[:12]}"
    protocol = request.protocol.strip().lower()
    if protocol not in {"responses", "chat"}:
        protocol = "responses"

    async with httpx.AsyncClient(timeout=BRIDGE_TIMEOUT_SECONDS) as client:
        if protocol == "chat":
            history = [
                {"role": str(item.get("role") or "user"), "content": str(item.get("content") or "")}
                for item in request.messages
                if item.get("role") in {"user", "assistant"} and item.get("content")
            ]
            payload = {
                "model": BRIDGE_MODEL,
                "messages": [*history, {"role": "user", "content": text}],
                "stream": False,
                "max_tokens": 512,
                "metadata": {
                    "conversation_id": f"bridge-test-ui-{session_id}",
                    "trace_id": trace_id,
                    "trace_source": "bridge-test-ui",
                },
            }
            response = await client.post(
                f"{BRIDGE_BASE_URL}/chat/completions",
                json=payload,
                headers={"x-alpha-trace-id": trace_id},
            )
            response.raise_for_status()
            raw = response.json()
            answer = _extract_chat_text(raw)
        else:
            history_text = ""
            for item in request.messages[-20:]:
                role = str(item.get("role") or "")
                content = str(item.get("content") or "").strip()
                if role in {"user", "assistant"} and content:
                    history_text += f"{role}: {content}\n"
            prompt = text if not history_text else f"Chat history:\n{history_text}\nuser: {text}"
            payload = {
                "model": BRIDGE_MODEL,
                "input": prompt,
                "stream": False,
                "max_output_tokens": 512,
                "metadata": {
                    "conversation_id": f"bridge-test-ui-{session_id}",
                    "trace_id": trace_id,
                    "trace_source": "bridge-test-ui",
                },
            }
            response = await client.post(
                f"{BRIDGE_BASE_URL}/responses",
                json=payload,
                headers={"x-alpha-trace-id": trace_id},
            )
            response.raise_for_status()
            raw = response.json()
            answer = _extract_responses_text(raw)

    elapsed_seconds = round(time.perf_counter() - started, 3)
    trace = _extract_trace(raw)
    if not trace:
        trace = {"trace_id": trace_id, "protocol": protocol, "steps": []}
    trace.setdefault("trace_id", trace_id)
    trace.setdefault("protocol", protocol)
    trace.setdefault("steps", [])
    trace["test_ui_server_elapsed_seconds"] = elapsed_seconds
    trace["steps"] = [
        {"name": "test_ui.server.received", "elapsed_seconds": 0.0},
        *[step for step in trace.get("steps", []) if isinstance(step, dict)],
        {"name": "test_ui.server.completed", "elapsed_seconds": elapsed_seconds},
    ]

    return JSONResponse(
        {
            "answer": answer,
            "protocol": protocol,
            "elapsed_seconds": elapsed_seconds,
            "trace": trace,
            "raw": raw,
        }
    )


HTML = """<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AlphaRavis Bridge Test UI</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #111318; color: #eef1f5; min-height: 100vh; }
    main { max-width: 960px; margin: 0 auto; padding: 24px; display: grid; gap: 16px; }
    header { display: flex; align-items: center; justify-content: space-between; gap: 12px; border-bottom: 1px solid #2d3340; padding-bottom: 12px; }
    h1 { font-size: 20px; margin: 0; font-weight: 650; }
    .status { color: #9aa4b2; font-size: 13px; }
    #chat { min-height: 48vh; max-height: 68vh; overflow: auto; display: flex; flex-direction: column; gap: 10px; padding: 4px 2px; }
    .msg { border: 1px solid #2d3340; background: #181c24; border-radius: 8px; padding: 10px 12px; white-space: pre-wrap; line-height: 1.45; }
    .user { align-self: flex-end; max-width: 78%; background: #16324a; border-color: #24557e; }
    .assistant { align-self: flex-start; max-width: 86%; }
    .meta { color: #9aa4b2; font-size: 12px; margin-bottom: 4px; }
    form { display: grid; gap: 10px; border-top: 1px solid #2d3340; padding-top: 14px; }
    textarea { width: 100%; min-height: 96px; resize: vertical; border: 1px solid #3a4252; border-radius: 8px; background: #0d1016; color: #eef1f5; padding: 12px; font: inherit; }
    .controls { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    select, button { border: 1px solid #3a4252; border-radius: 8px; background: #171b23; color: #eef1f5; padding: 9px 12px; font: inherit; }
    button.primary { background: #2d6cdf; border-color: #2d6cdf; }
    button:disabled { opacity: .55; cursor: wait; }
    details { border: 1px solid #2d3340; border-radius: 8px; padding: 8px 10px; background: #0d1016; }
    .trace { border: 1px solid #2d3340; border-radius: 8px; background: #0d1016; padding: 10px; }
    .trace h2 { font-size: 14px; margin: 0 0 8px; font-weight: 650; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { border-top: 1px solid #202633; padding: 6px 4px; text-align: left; vertical-align: top; }
    th { color: #9aa4b2; font-weight: 600; }
    .bar-wrap { height: 8px; background: #161b24; border-radius: 999px; overflow: hidden; min-width: 120px; }
    .bar { height: 100%; background: #2d6cdf; width: 0; }
    pre { overflow: auto; font-size: 12px; color: #cbd5e1; }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>AlphaRavis Bridge Test UI</h1>
      <div id="status" class="status">bereit</div>
    </header>
    <section id="chat" aria-live="polite"></section>
    <form id="form">
      <textarea id="input" placeholder="Nachricht eingeben..." autofocus></textarea>
      <div class="controls">
        <select id="protocol" title="Bridge-Protokoll">
          <option value="responses">Responses</option>
          <option value="chat">Chat Completions</option>
        </select>
        <button class="primary" id="send" type="submit">Senden</button>
        <button id="clear" type="button">Verlauf leeren</button>
      </div>
    </form>
    <section class="trace">
      <h2>Trace</h2>
      <div id="trace-summary" class="status">noch keine Anfrage</div>
      <table>
        <thead><tr><th>t</th><th>Schritt</th><th>Dauer</th><th>Details</th><th></th></tr></thead>
        <tbody id="trace-body"></tbody>
      </table>
    </section>
    <details>
      <summary>Letzte rohe Bridge-Antwort</summary>
      <pre id="raw">{}</pre>
    </details>
  </main>
  <script>
    const chat = document.getElementById('chat');
    const form = document.getElementById('form');
    const input = document.getElementById('input');
    const statusEl = document.getElementById('status');
    const rawEl = document.getElementById('raw');
    const traceSummary = document.getElementById('trace-summary');
    const traceBody = document.getElementById('trace-body');
    const sendBtn = document.getElementById('send');
    const clearBtn = document.getElementById('clear');
    const protocol = document.getElementById('protocol');
    const messages = [];

    function makeId(prefix) {
      if (window.crypto && typeof window.crypto.randomUUID === 'function') {
        return `${prefix}_${window.crypto.randomUUID().split('-').join('').slice(0, 12)}`;
      }
      const random = Math.random().toString(16).slice(2, 14);
      return `${prefix}_${Date.now().toString(16)}${random}`.slice(0, prefix.length + 13);
    }

    function storedSessionId() {
      try {
        const existing = window.localStorage.getItem('alpharavis-test-ui-session');
        if (existing) return existing;
        const created = makeId('session');
        window.localStorage.setItem('alpharavis-test-ui-session', created);
        return created;
      } catch (error) {
        return makeId('session');
      }
    }

    const sessionId = storedSessionId();

    window.addEventListener('error', (event) => {
      statusEl.textContent = `JS-Fehler: ${event.message || 'unbekannt'}`;
    });

    function renderTrace(trace, browserSeconds) {
      traceBody.innerHTML = '';
      trace = trace || {};
      const steps = Array.isArray(trace.steps) ? trace.steps : [];
      const maxElapsed = Math.max(browserSeconds || 0, ...steps.map((step) => Number(step.elapsed_seconds || 0)), 0.001);
      traceSummary.textContent = `${trace.trace_id || 'trace'} | ${steps.length} Schritte | ${browserSeconds.toFixed(2)}s Browser`;
      for (const step of steps) {
        const elapsed = Number(step.elapsed_seconds || 0);
        const duration = step.duration_seconds == null ? '' : `${Number(step.duration_seconds).toFixed(3)}s`;
        const details = Object.entries(step)
          .filter(([key]) => !['name', 'elapsed_seconds', 'duration_seconds'].includes(key))
          .map(([key, value]) => `${key}=${typeof value === 'string' ? value : JSON.stringify(value)}`)
          .join(' ');
        const tr = document.createElement('tr');
        const cells = [
          `${elapsed.toFixed(3)}s`,
          step.name || '',
          duration,
          details,
        ];
        for (const text of cells) {
          const td = document.createElement('td');
          td.textContent = text;
          tr.appendChild(td);
        }
        const barCell = document.createElement('td');
        const wrap = document.createElement('div');
        wrap.className = 'bar-wrap';
        const bar = document.createElement('div');
        bar.className = 'bar';
        bar.style.width = `${Math.min(100, (elapsed / maxElapsed) * 100)}%`;
        wrap.appendChild(bar);
        barCell.appendChild(wrap);
        tr.appendChild(barCell);
        traceBody.appendChild(tr);
      }
    }

    function render() {
      chat.innerHTML = '';
      for (const msg of messages) {
        const el = document.createElement('div');
        el.className = `msg ${msg.role}`;
        const meta = document.createElement('div');
        meta.className = 'meta';
        meta.textContent = msg.role;
        const body = document.createElement('div');
        body.textContent = msg.content || '(leer)';
        el.append(meta, body);
        chat.appendChild(el);
      }
      chat.scrollTop = chat.scrollHeight;
    }

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const text = input.value.trim();
      if (!text) return;
      input.value = '';
      messages.push({ role: 'user', content: text });
      render();
      sendBtn.disabled = true;
      statusEl.textContent = 'sendet...';
      const started = performance.now();
      const traceId = makeId('trace');
      try {
        const res = await fetch('/api/send', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            message: text,
            messages: messages.slice(0, -1),
            protocol: protocol.value,
            session_id: sessionId,
            trace_id: traceId
          })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || res.statusText);
        messages.push({ role: 'assistant', content: data.answer || '' });
        rawEl.textContent = JSON.stringify(data.raw, null, 2);
        const browserSeconds = (performance.now() - started) / 1000;
        renderTrace(data.trace || {}, browserSeconds);
        statusEl.textContent = `${data.protocol} | ${data.elapsed_seconds}s server | ${browserSeconds.toFixed(2)}s browser`;
      } catch (error) {
        messages.push({ role: 'assistant', content: `FEHLER: ${error.message || error}` });
        statusEl.textContent = 'Fehler';
      } finally {
        sendBtn.disabled = false;
        render();
        input.focus();
      }
    });

    clearBtn.addEventListener('click', () => {
      messages.length = 0;
      rawEl.textContent = '{}';
      traceBody.innerHTML = '';
      traceSummary.textContent = 'noch keine Anfrage';
      statusEl.textContent = 'bereit';
      render();
      input.focus();
    });

    input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
        if (typeof form.requestSubmit === 'function') {
          form.requestSubmit();
        } else {
          sendBtn.click();
        }
      }
    });

    render();
  </script>
</body>
</html>
"""
