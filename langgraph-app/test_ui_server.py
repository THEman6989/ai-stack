from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel


BRIDGE_BASE_URL = os.getenv("TEST_UI_BRIDGE_BASE_URL", "http://api-bridge:8123/v1").rstrip("/")
BRIDGE_MODEL = os.getenv("TEST_UI_MODEL", "my-agent")
BRIDGE_TIMEOUT_SECONDS = float(os.getenv("TEST_UI_BRIDGE_TIMEOUT_SECONDS", "240"))

app = FastAPI(title="AlphaRavis Bridge Test UI")


class ChatRequest(BaseModel):
    message: str
    messages: list[dict[str, Any]] = []
    protocol: str = "responses"
    stream: bool = True
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


def _protocol(raw: str) -> str:
    protocol = raw.strip().lower()
    return protocol if protocol in {"responses", "chat"} else "responses"


def _bridge_request_payload(
    request: ChatRequest,
    *,
    text: str,
    protocol: str,
    session_id: str,
    trace_id: str,
    stream: bool,
) -> tuple[str, dict[str, Any]]:
    metadata = {
        "conversation_id": f"bridge-test-ui-{session_id}",
        "trace_id": trace_id,
        "trace_source": "bridge-test-ui",
    }
    if protocol == "chat":
        history = [
            {"role": str(item.get("role") or "user"), "content": str(item.get("content") or "")}
            for item in request.messages
            if item.get("role") in {"user", "assistant"} and item.get("content")
        ]
        return (
            f"{BRIDGE_BASE_URL}/chat/completions",
            {
                "model": BRIDGE_MODEL,
                "messages": [*history, {"role": "user", "content": text}],
                "stream": stream,
                "max_tokens": 512,
                "metadata": metadata,
            },
        )

    input_items: list[dict[str, Any]] = []
    for item in request.messages[-20:]:
        role = str(item.get("role") or "")
        content = str(item.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            input_items.append(
                {
                    "type": "message",
                    "role": role,
                    "content": [{"type": "output_text" if role == "assistant" else "input_text", "text": content}],
                }
            )
    input_items.append(
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        }
    )
    return (
        f"{BRIDGE_BASE_URL}/responses",
        {
            "model": BRIDGE_MODEL,
            "input": input_items,
            "stream": stream,
            "max_output_tokens": 512,
            "metadata": metadata,
        },
    )


def _test_ui_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(HTML, headers={"Cache-Control": "no-store"})


@app.get("/observer", response_class=HTMLResponse)
async def observer() -> HTMLResponse:
    return HTMLResponse(OBSERVER_HTML, headers={"Cache-Control": "no-store"})


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "bridge_base_url": BRIDGE_BASE_URL,
        "model": BRIDGE_MODEL,
    }


@app.get("/api/observer")
async def observer_records(limit: int = 80) -> JSONResponse:
    async with httpx.AsyncClient(timeout=BRIDGE_TIMEOUT_SECONDS) as client:
        response = await client.get(f"{BRIDGE_BASE_URL.removesuffix('/v1')}/_alpharavis/bridge-observer", params={"limit": limit})
        response.raise_for_status()
        return JSONResponse(response.json())


@app.delete("/api/observer")
async def clear_observer_records() -> JSONResponse:
    async with httpx.AsyncClient(timeout=BRIDGE_TIMEOUT_SECONDS) as client:
        response = await client.delete(f"{BRIDGE_BASE_URL.removesuffix('/v1')}/_alpharavis/bridge-observer")
        response.raise_for_status()
        return JSONResponse(response.json())


@app.post("/api/send")
async def send_chat(request: ChatRequest) -> JSONResponse:
    text = request.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="message is required")

    started = time.perf_counter()
    trace_id = request.trace_id.strip() or f"trace_{uuid.uuid4().hex[:12]}"
    session_id = request.session_id.strip() or f"session_{uuid.uuid4().hex[:12]}"
    protocol = _protocol(request.protocol)

    async with httpx.AsyncClient(timeout=BRIDGE_TIMEOUT_SECONDS) as client:
        url, payload = _bridge_request_payload(
            request,
            text=text,
            protocol=protocol,
            session_id=session_id,
            trace_id=trace_id,
            stream=False,
        )
        response = await client.post(
            url,
            json=payload,
            headers={"x-alpha-trace-id": trace_id},
        )
        response.raise_for_status()
        raw = response.json()
        if protocol == "chat":
            response.raise_for_status()
            answer = _extract_chat_text(raw)
        else:
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


@app.post("/api/send_stream")
async def send_chat_stream(request: ChatRequest) -> StreamingResponse:
    text = request.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="message is required")

    trace_id = request.trace_id.strip() or f"trace_{uuid.uuid4().hex[:12]}"
    session_id = request.session_id.strip() or f"session_{uuid.uuid4().hex[:12]}"
    protocol = _protocol(request.protocol)
    url, payload = _bridge_request_payload(
        request,
        text=text,
        protocol=protocol,
        session_id=session_id,
        trace_id=trace_id,
        stream=True,
    )

    async def proxy_events() -> AsyncIterator[str]:
        started = time.perf_counter()
        yield _test_ui_event(
            "test_ui.started",
            {
                "protocol": protocol,
                "trace_id": trace_id,
                "model": BRIDGE_MODEL,
            },
        )
        try:
            async with httpx.AsyncClient(timeout=BRIDGE_TIMEOUT_SECONDS) as client:
                async with client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers={"x-alpha-trace-id": trace_id},
                ) as response:
                    if response.status_code >= 400:
                        body = (await response.aread()).decode(errors="replace")
                        yield _test_ui_event(
                            "test_ui.error",
                            {
                                "protocol": protocol,
                                "trace_id": trace_id,
                                "status_code": response.status_code,
                                "detail": body[:4000],
                                "elapsed_seconds": round(time.perf_counter() - started, 3),
                            },
                        )
                        return
                    async for chunk in response.aiter_text():
                        if chunk:
                            yield chunk
        except Exception as exc:
            yield _test_ui_event(
                "test_ui.error",
                {
                    "protocol": protocol,
                    "trace_id": trace_id,
                    "detail": str(exc),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                },
            )
            return

        yield _test_ui_event(
            "test_ui.completed",
            {
                "protocol": protocol,
                "trace_id": trace_id,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            },
        )

    return StreamingResponse(
        proxy_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store"},
    )


OBSERVER_HTML = """<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AlphaRavis Bridge Observer</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; background: #0f1117; color: #eef1f5; }
    main { width: 100%; min-height: 100vh; padding: 16px; display: grid; grid-template-rows: auto minmax(320px, 1fr) minmax(260px, 42vh); gap: 12px; }
    header { display: flex; align-items: center; justify-content: space-between; gap: 12px; border-bottom: 1px solid #2b3240; padding-bottom: 10px; }
    h1 { margin: 0; font-size: 20px; font-weight: 700; }
    .tools { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    button, select, input { border: 1px solid #3a4252; border-radius: 7px; background: #171b23; color: #eef1f5; padding: 8px 10px; font: inherit; }
    button.active { background: #2d6cdf; border-color: #2d6cdf; }
    .status { color: #9aa4b2; font-size: 12px; }
    .table-wrap { border: 1px solid #2b3240; border-radius: 8px; overflow: auto; background: #11151d; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; min-width: 1180px; }
    th, td { border-bottom: 1px solid #222938; padding: 7px 8px; text-align: left; vertical-align: top; }
    th { position: sticky; top: 0; z-index: 1; background: #151a24; color: #9aa4b2; font-weight: 650; }
    tr { cursor: pointer; }
    tr:hover, tr.selected { background: #1a2230; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .pill { display: inline-block; border: 1px solid #3a4252; border-radius: 999px; padding: 1px 7px; color: #cbd5e1; }
    .hard { border-color: #8f3b3b; color: #f59b9b; }
    .done { border-color: #2f8f5b; color: #7dd3a8; }
    .detail { border: 1px solid #2b3240; border-radius: 8px; background: #0c1017; display: grid; grid-template-rows: auto 1fr; min-height: 0; }
    .detail-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; border-bottom: 1px solid #222938; padding: 8px; }
    .tabs, .mode { display: flex; gap: 6px; align-items: center; }
    pre { margin: 0; padding: 12px; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12px; line-height: 1.45; color: #d6deeb; }
    .empty { color: #9aa4b2; }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>AlphaRavis Bridge Observer</h1>
        <div id="status" class="status">lädt...</div>
      </div>
      <div class="tools">
        <label class="status">Limit <input id="limit" type="number" min="1" max="200" value="80" style="width:72px"></label>
        <button id="refresh" type="button">Aktualisieren</button>
        <button id="clear" type="button">Leeren</button>
        <a class="status" href="/">Test UI</a>
      </div>
    </header>
    <section class="table-wrap" aria-label="Bridge Requests">
      <table>
        <thead>
          <tr>
            <th>Zeit</th>
            <th>Protokoll</th>
            <th>Status</th>
            <th>Stream</th>
            <th>Model</th>
            <th>Thread-Key</th>
            <th>Thread-ID</th>
            <th>Raw Msg</th>
            <th>Raw Tokens</th>
            <th>Model Msg</th>
            <th>Model Tokens</th>
            <th>State Msg</th>
            <th>Output</th>
            <th>Reasoning</th>
          </tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
    </section>
    <section class="detail">
      <div class="detail-head">
        <div class="tabs">
          <button id="sendTab" class="active" type="button">Senden</button>
          <button id="receiveTab" type="button">Empfang</button>
        </div>
        <div class="mode">
          <button id="contextMode" class="active" type="button">Nur Kontext</button>
          <button id="fullMode" type="button">Vollansicht</button>
        </div>
      </div>
      <pre id="detail" class="empty">Keine Anfrage ausgewählt.</pre>
    </section>
  </main>
  <script>
    const rowsEl = document.getElementById('rows');
    const detailEl = document.getElementById('detail');
    const statusEl = document.getElementById('status');
    const limitEl = document.getElementById('limit');
    const refreshBtn = document.getElementById('refresh');
    const clearBtn = document.getElementById('clear');
    const sendTab = document.getElementById('sendTab');
    const receiveTab = document.getElementById('receiveTab');
    const contextMode = document.getElementById('contextMode');
    const fullMode = document.getElementById('fullMode');
    let records = [];
    let selectedId = '';
    let activeTab = 'send';
    let activeMode = 'context';

    function fmtTime(value) {
      if (!value) return '';
      return new Date(value * 1000).toLocaleTimeString();
    }
    function short(value, length = 42) {
      const text = String(value || '');
      return text.length > length ? `${text.slice(0, length)}…` : text;
    }
    function pretty(value) {
      return JSON.stringify(value ?? {}, null, 2);
    }
    function selectedRecord() {
      return records.find((record) => record.id === selectedId) || records[0] || null;
    }
    function statusClass(status) {
      if (status === 'hard_cutoff') return 'pill hard';
      if (status === 'completed') return 'pill done';
      return 'pill';
    }
    function renderRows() {
      rowsEl.innerHTML = '';
      for (const record of records) {
        const tr = document.createElement('tr');
        if (record.id === selectedId) tr.className = 'selected';
        const send = record.send || {};
        const receive = record.receive || {};
        const stateProfile = send.langgraph_state_profile || {};
        const cells = [
          fmtTime(record.created_at),
          record.protocol,
          record.status,
          record.stream ? 'ja' : 'nein',
          record.model,
          short(record.thread_key, 48),
          short(record.thread_id, 36),
          send.raw_message_count,
          send.raw_token_estimate,
          send.model_context_message_count,
          send.model_context_token_estimate,
          stateProfile.message_count,
          receive.output_chars,
          receive.reasoning_chars,
        ];
        cells.forEach((value, index) => {
          const td = document.createElement('td');
          if (index === 2) {
            const span = document.createElement('span');
            span.className = statusClass(record.status);
            span.textContent = value || 'received';
            td.appendChild(span);
          } else {
            td.textContent = value ?? '';
            if ([5, 6].includes(index)) td.className = 'mono';
          }
          tr.appendChild(td);
        });
        tr.addEventListener('click', () => {
          selectedId = record.id;
          renderRows();
          renderDetail();
        });
        rowsEl.appendChild(tr);
      }
    }
    function renderDetail() {
      const record = selectedRecord();
      if (!record) {
        detailEl.className = 'empty';
        detailEl.textContent = 'Keine Anfrage ausgewählt.';
        return;
      }
      detailEl.className = '';
      if (activeTab === 'send') {
        const send = record.send || {};
        detailEl.textContent = activeMode === 'context'
          ? pretty({
              thread_key: record.thread_key,
              thread_id: record.thread_id,
              model_context_token_estimate: send.model_context_token_estimate,
              langgraph_state_profile: send.langgraph_state_profile || {},
              model_context_messages: send.model_context_messages || [],
              model_context: send.model_context || {},
            })
          : pretty(send);
      } else {
        const receive = record.receive || {};
        detailEl.textContent = activeMode === 'context'
          ? pretty({
              status: receive.status,
              output_text: receive.output_text || '',
              reasoning_text: receive.reasoning_text || '',
              event_counts: receive.event_counts || {},
            })
          : pretty(receive);
      }
    }
    async function loadRecords() {
      const limit = Math.max(1, Math.min(200, Number(limitEl.value || 80)));
      const response = await fetch(`/api/observer?limit=${limit}`, { cache: 'no-store' });
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json();
      records = Array.isArray(payload.records) ? payload.records : [];
      if (!selectedId && records[0]) selectedId = records[0].id;
      if (selectedId && !records.some((record) => record.id === selectedId) && records[0]) selectedId = records[0].id;
      renderRows();
      renderDetail();
      statusEl.textContent = `${records.length} Requests`;
    }
    function setTab(tab) {
      activeTab = tab;
      sendTab.classList.toggle('active', tab === 'send');
      receiveTab.classList.toggle('active', tab === 'receive');
      renderDetail();
    }
    function setMode(mode) {
      activeMode = mode;
      contextMode.classList.toggle('active', mode === 'context');
      fullMode.classList.toggle('active', mode === 'full');
      renderDetail();
    }
    refreshBtn.addEventListener('click', () => loadRecords().catch((error) => { statusEl.textContent = error.message; }));
    clearBtn.addEventListener('click', async () => {
      await fetch('/api/observer', { method: 'DELETE' });
      records = [];
      selectedId = '';
      renderRows();
      renderDetail();
      statusEl.textContent = 'geleert';
    });
    sendTab.addEventListener('click', () => setTab('send'));
    receiveTab.addEventListener('click', () => setTab('receive'));
    contextMode.addEventListener('click', () => setMode('context'));
    fullMode.addEventListener('click', () => setMode('full'));
    loadRecords().catch((error) => { statusEl.textContent = error.message; });
    window.setInterval(() => loadRecords().catch(() => {}), 2500);
  </script>
</body>
</html>
"""


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
    .header-left { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .nav-button { display: inline-flex; align-items: center; justify-content: center; border: 1px solid #3a4252; border-radius: 8px; background: #171b23; color: #eef1f5; padding: 8px 11px; font-size: 13px; text-decoration: none; }
    .nav-button:hover { background: #202633; }
    .status { color: #9aa4b2; font-size: 13px; }
    .live-panels { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
    .live-panel { border: 1px solid #2d3340; border-radius: 8px; background: #0d1016; min-height: 112px; display: grid; grid-template-rows: auto 1fr; overflow: hidden; }
    .live-panel.expanded { grid-column: 1 / -1; min-height: 280px; }
    .live-panel-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; border-bottom: 1px solid #202633; padding: 6px 8px 6px 10px; }
    .live-panel h2 { margin: 0; color: #9aa4b2; font-size: 11px; font-weight: 650; text-transform: uppercase; letter-spacing: 0; }
    .panel-toggle { border-radius: 6px; padding: 4px 7px; font-size: 11px; line-height: 1; }
    .live-panel pre { margin: 0; padding: 10px; max-height: 180px; overflow: auto; color: #cbd5e1; white-space: pre-wrap; overflow-wrap: anywhere; }
    .live-panel.expanded pre { max-height: 440px; }
    #chat { min-height: 48vh; max-height: 68vh; overflow: auto; display: flex; flex-direction: column; gap: 10px; padding: 4px 2px; }
    .msg { border: 1px solid #2d3340; background: #181c24; border-radius: 8px; padding: 10px 12px; white-space: pre-wrap; line-height: 1.45; }
    .user { align-self: flex-end; max-width: 78%; background: #16324a; border-color: #24557e; }
    .assistant { align-self: flex-start; max-width: 86%; }
    .meta { color: #9aa4b2; font-size: 12px; margin-bottom: 4px; }
    .route-badge { display: inline-block; margin-left: 8px; border: 1px solid #3a4252; border-radius: 999px; padding: 1px 7px; color: #cbd5e1; font-size: 11px; }
    .route-fast { border-color: #2f8f5b; color: #7dd3a8; }
    .route-agent { border-color: #8a6d2e; color: #f3c969; }
    .route-hard { border-color: #8f3b3b; color: #f59b9b; }
    .reasoning-details { margin-top: 8px; border-top: 1px solid #2d3340; padding-top: 8px; color: #cbd5e1; }
    .reasoning-details summary { cursor: pointer; color: #9aa4b2; font-size: 12px; user-select: none; }
    .reasoning-section { margin-top: 8px; display: grid; gap: 4px; }
    .reasoning-label { color: #9aa4b2; font-size: 11px; font-weight: 650; text-transform: uppercase; letter-spacing: 0; }
    .reasoning-body { font-size: 12px; line-height: 1.45; color: #cbd5e1; white-space: pre-wrap; }
    .reasoning-status { color: #9aa4b2; }
    form { display: grid; gap: 10px; border-top: 1px solid #2d3340; padding-top: 14px; }
    textarea { width: 100%; min-height: 96px; resize: vertical; border: 1px solid #3a4252; border-radius: 8px; background: #0d1016; color: #eef1f5; padding: 12px; font: inherit; }
    .controls { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    select, button { border: 1px solid #3a4252; border-radius: 8px; background: #171b23; color: #eef1f5; padding: 9px 12px; font: inherit; }
    button.primary { background: #2d6cdf; border-color: #2d6cdf; }
    button:disabled { opacity: .55; cursor: wait; }
    details { border: 1px solid #2d3340; border-radius: 8px; padding: 8px 10px; background: #0d1016; }
    .trace { border: 1px solid #2d3340; border-radius: 8px; background: #0d1016; padding: 10px; }
    .trace-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 8px; }
    .trace h2 { font-size: 14px; margin: 0; font-weight: 650; }
    .trace-toggle { display: inline-flex; align-items: center; gap: 6px; color: #9aa4b2; font-size: 12px; user-select: none; }
    .trace-toggle input { margin: 0; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { border-top: 1px solid #202633; padding: 6px 4px; text-align: left; vertical-align: top; }
    th { color: #9aa4b2; font-weight: 600; }
    .bar-wrap { height: 8px; background: #161b24; border-radius: 999px; overflow: hidden; min-width: 120px; }
    .bar { height: 100%; background: #2d6cdf; width: 0; }
    pre { overflow: auto; font-size: 12px; color: #cbd5e1; }
    @media (max-width: 760px) { .live-panels { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <header>
      <div class="header-left">
        <a class="nav-button" href="/observer">Observer</a>
        <h1>AlphaRavis Bridge Test UI</h1>
      </div>
      <div id="status" class="status">bereit</div>
    </header>
    <section class="live-panels" aria-label="Stream-Details">
      <div class="live-panel">
        <div class="live-panel-head"><h2>Status</h2><button class="panel-toggle" type="button" data-panel-toggle>Gross</button></div>
        <pre id="live-status">(leer)</pre>
      </div>
      <div class="live-panel">
        <div class="live-panel-head"><h2>Reasoning</h2><button class="panel-toggle" type="button" data-panel-toggle>Gross</button></div>
        <pre id="live-reasoning">(leer)</pre>
      </div>
      <div class="live-panel">
        <div class="live-panel-head"><h2>Planer</h2><button class="panel-toggle" type="button" data-panel-toggle>Gross</button></div>
        <pre id="live-plan">(leer)</pre>
      </div>
    </section>
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
      <div class="trace-head">
        <h2>Trace</h2>
        <label class="trace-toggle"><input id="trace-delta-details" type="checkbox"> Delta-Details</label>
      </div>
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
    const liveStatusEl = document.getElementById('live-status');
    const liveReasoningEl = document.getElementById('live-reasoning');
    const livePlanEl = document.getElementById('live-plan');
    const rawEl = document.getElementById('raw');
    const traceSummary = document.getElementById('trace-summary');
    const traceBody = document.getElementById('trace-body');
    const traceDeltaDetails = document.getElementById('trace-delta-details');
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

    function resetSessionId() {
      const created = makeId('session');
      try {
        window.localStorage.setItem('alpharavis-test-ui-session', created);
      } catch (error) {
        // Ignore storage failures; the in-memory session id is still reset.
      }
      return created;
    }

    let sessionId = storedSessionId();
    let lastTrace = null;
    let lastTraceBrowserSeconds = 0;

    window.addEventListener('error', (event) => {
      statusEl.textContent = `JS-Fehler: ${event.message || 'unbekannt'}`;
    });

    function isCompactDeltaStep(step) {
      if (!step || typeof step !== 'object') return false;
      if (step.event === 'response.output_text.delta') return true;
      if (step.event === 'response.reasoning.delta' && ['internal_plan', 'model'].includes(step.reasoning_kind)) return true;
      if (step.event === 'message' && ['internal_plan', 'model'].includes(step.reasoning_kind)) return true;
      return step.event === 'message' && step.text_delta === true;
    }

    function summarizeTraceSteps(steps) {
      const summarized = [];
      let group = null;
      function flushGroup() {
        if (!group) return;
        summarized.push({
          name: `${group.name || 'Delta empfangen'} (${group.count} Deltas, ${group.chars} Zeichen)`,
          elapsed_seconds: group.firstElapsed,
          duration_seconds: Math.max(0, group.lastElapsed - group.firstElapsed),
          event: group.event,
          reasoning_kind: group.reasoningKind,
          sequence_number: `${group.firstSequence ?? '?'}..${group.lastSequence ?? '?'}`,
          delta_chars: group.chars,
        });
        group = null;
      }
      for (const step of steps) {
        if (!isCompactDeltaStep(step)) {
          flushGroup();
          summarized.push(step);
          continue;
        }
        const elapsed = Number(step.elapsed_seconds || 0);
        const chars = Number(step.delta_chars || 0);
        if (!group || group.event !== step.event) {
          flushGroup();
          group = {
            event: step.event,
            name: step.name,
            reasoningKind: step.reasoning_kind,
            count: 0,
            chars: 0,
            firstElapsed: elapsed,
            lastElapsed: elapsed,
            firstSequence: step.sequence_number,
            lastSequence: step.sequence_number,
          };
        }
        group.count += 1;
        group.chars += chars;
        group.lastElapsed = elapsed;
        group.lastSequence = step.sequence_number;
      }
      flushGroup();
      return summarized;
    }

    function renderTrace(trace, browserSeconds) {
      lastTrace = trace || {};
      lastTraceBrowserSeconds = browserSeconds || 0;
      traceBody.innerHTML = '';
      trace = trace || {};
      const rawSteps = Array.isArray(trace.steps) ? trace.steps : [];
      const steps = traceDeltaDetails.checked ? rawSteps : summarizeTraceSteps(rawSteps);
      const maxElapsed = Math.max(browserSeconds || 0, ...steps.map((step) => Number(step.elapsed_seconds || 0)), 0.001);
      const hiddenSteps = rawSteps.length - steps.length;
      const compactSuffix = hiddenSteps > 0 ? ` | ${hiddenSteps} Delta-Zeilen zusammengefasst` : '';
      traceSummary.textContent = `${trace.trace_id || 'trace'} | ${steps.length} Schritte | ${browserSeconds.toFixed(2)}s Browser${compactSuffix}`;
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
        if (msg.role === 'assistant') {
          const badge = document.createElement('span');
          badge.className = `route-badge ${routeClass(msg.route)}`;
          badge.textContent = routeLabel(msg.route);
          meta.appendChild(badge);
        }
        const body = document.createElement('div');
        body.textContent = msg.content || '(leer)';
        el.append(meta, body);
        if (msg.role === 'assistant' && (msg.reasoningStatus || msg.internalPlan || msg.reasoning)) {
          const details = document.createElement('details');
          details.className = 'reasoning-details';
          details.open = Boolean(msg.reasoningOpen);
          details.addEventListener('toggle', () => {
            msg.reasoningOpen = details.open;
          });
          const summary = document.createElement('summary');
          summary.textContent = 'Reasoning';
          details.appendChild(summary);
          if (msg.reasoningStatus) {
            const statusSection = document.createElement('div');
            statusSection.className = 'reasoning-section';
            const statusLabel = document.createElement('div');
            statusLabel.className = 'reasoning-label';
            statusLabel.textContent = 'Status';
            const statusBody = document.createElement('div');
            statusBody.className = 'reasoning-body reasoning-status';
            statusBody.textContent = msg.reasoningStatus;
            statusSection.append(statusLabel, statusBody);
            details.appendChild(statusSection);
          }
          if (msg.internalPlan) {
            const planSection = document.createElement('div');
            planSection.className = 'reasoning-section';
            const planLabel = document.createElement('div');
            planLabel.className = 'reasoning-label';
            planLabel.textContent = 'Interner Plan';
            const planBody = document.createElement('div');
            planBody.className = 'reasoning-body';
            planBody.textContent = msg.internalPlan;
            planSection.append(planLabel, planBody);
            details.appendChild(planSection);
          }
          if (msg.reasoning) {
            const reasoningSection = document.createElement('div');
            reasoningSection.className = 'reasoning-section';
            const reasoningLabel = document.createElement('div');
            reasoningLabel.className = 'reasoning-label';
            reasoningLabel.textContent = 'Modell-Reasoning';
            const reasoningBody = document.createElement('div');
            reasoningBody.className = 'reasoning-body';
            reasoningBody.textContent = msg.reasoning;
            reasoningSection.append(reasoningLabel, reasoningBody);
            details.appendChild(reasoningSection);
          }
          el.appendChild(details);
        }
        chat.appendChild(el);
      }
      chat.scrollTop = chat.scrollHeight;
      const currentAssistant = [...messages].reverse().find((msg) => msg.role === 'assistant');
      renderLivePanels(currentAssistant || null);
    }

    function renderLivePanels(msg) {
      const statusText = msg && msg.reasoningStatus ? msg.reasoningStatus.trim() : '';
      const reasoningText = msg && msg.reasoning ? msg.reasoning.trim() : '';
      const planText = msg && msg.internalPlan ? msg.internalPlan.trim() : '';
      liveStatusEl.textContent = statusText || '(leer)';
      liveReasoningEl.textContent = reasoningText || '(leer)';
      livePlanEl.textContent = planText || '(leer)';
      liveStatusEl.scrollTop = liveStatusEl.scrollHeight;
      liveReasoningEl.scrollTop = liveReasoningEl.scrollHeight;
      livePlanEl.scrollTop = livePlanEl.scrollHeight;
    }

    function parseSseBlock(block) {
      const lines = block.split(/\\r?\\n/);
      let eventName = 'message';
      const dataLines = [];
      for (const line of lines) {
        if (line.startsWith('event:')) {
          eventName = line.slice(6).trim();
        } else if (line.startsWith('data:')) {
          dataLines.push(line.slice(5).trimStart());
        }
      }
      if (!dataLines.length) return null;
      const dataText = dataLines.join('\\n');
      if (dataText === '[DONE]') {
        return { event: eventName, done: true, data: '[DONE]' };
      }
      try {
        return { event: eventName, data: JSON.parse(dataText) };
      } catch (error) {
        return { event: eventName, data: dataText };
      }
    }

    function responseTextDelta(eventName, data) {
      if (!data || typeof data !== 'object') return '';
      if (eventName === 'response.output_text.delta' && typeof data.delta === 'string') {
        return data.delta;
      }
      return '';
    }

    function chatTextDelta(data) {
      if (!data || typeof data !== 'object') return '';
      const choice = Array.isArray(data.choices) ? data.choices[0] : null;
      const delta = choice && choice.delta && typeof choice.delta === 'object' ? choice.delta : {};
      return typeof delta.content === 'string' ? delta.content : '';
    }

    function reasoningDelta(protocolName, eventName, data, currentReasoning) {
      if (!data || typeof data !== 'object') return '';
      if (protocolName === 'responses') {
        if (eventName === 'response.reasoning.delta' && typeof data.delta === 'string') {
          return data.delta;
        }
        if (eventName === 'response.reasoning.done' && !currentReasoning && typeof data.text === 'string') {
          return data.text;
        }
        return '';
      }
      const choice = Array.isArray(data.choices) ? data.choices[0] : null;
      const delta = choice && choice.delta && typeof choice.delta === 'object' ? choice.delta : {};
      if (typeof delta.reasoning_content === 'string') return delta.reasoning_content;
      if (typeof delta.reasoning === 'string') return delta.reasoning;
      return '';
    }

    function reasoningKind(data, text, msg) {
      if (data && typeof data === 'object' && typeof data.alpha_reasoning_kind === 'string') {
        return data.alpha_reasoning_kind;
      }
      const choice = data && Array.isArray(data.choices) ? data.choices[0] : null;
      const delta = choice && choice.delta && typeof choice.delta === 'object' ? choice.delta : {};
      if (typeof delta.alpha_reasoning_kind === 'string') return delta.alpha_reasoning_kind;
      const value = String(text || '').trimStart();
      if (value.startsWith('Status:')) return 'status';
      if (value.startsWith('Interner Plan')) return 'internal_plan';
      if (msg && msg.reasoningMode === 'internal_plan') return 'internal_plan';
      return 'model';
    }

    function cleanInternalReasoning(text) {
      return String(text || '').replace(/^Interner Plan \\([^)]*\\):\\n?/, '');
    }

    function routeClass(routeName) {
      if (routeName === 'fast_path') return 'route-fast';
      if (routeName === 'agent_path') return 'route-agent';
      if (routeName === 'hard_stop') return 'route-hard';
      return '';
    }

    function routeLabel(routeName) {
      if (routeName === 'fast_path') return 'Fast Path';
      if (routeName === 'agent_path') return 'Agent Path';
      if (routeName === 'hard_stop') return 'Hard Stop';
      return 'Route offen';
    }

    function routeFromText(text) {
      const value = String(text || '').toLowerCase();
      if (!value) return '';
      if (value.includes('fast-path aktiv') || value.includes('fast_chat')) return 'fast_path';
      if (value.includes('hard_stop') || value.includes('hard context')) return 'hard_stop';
      if (
        value.includes('swarm') ||
        value.includes('planner') ||
        value.includes('memory_kernel') ||
        value.includes('skill_library') ||
        value.includes('handoff_context_guard') ||
        value.includes('crisis_preflight')
      ) {
        return 'agent_path';
      }
      return '';
    }

    function routeFromEvent(protocolName, eventName, data, textDelta, reasoning) {
      if (eventName === 'response.output_text.delta' || eventName === 'message') {
        const fromText = routeFromText(textDelta);
        if (fromText) return fromText;
      }
      if (eventName === 'response.reasoning.delta') {
        const fromReasoning = routeFromText(reasoning || (data && data.delta));
        if (fromReasoning) return fromReasoning;
      }
      return '';
    }

    function streamStatusText(eventName, data) {
      if (eventName === 'test_ui.started') return 'Stream gestartet';
      if (eventName === 'test_ui.completed') return 'Stream abgeschlossen';
      if (eventName === 'test_ui.error') return 'Stream-Fehler';
      if (eventName === 'response.reasoning.delta' && data && typeof data.delta === 'string') {
        if (data.alpha_reasoning_kind === 'internal_plan') return 'Interner Plan empfangen';
        if (data.alpha_reasoning_kind === 'model') return 'Modell-Reasoning empfangen';
        return data.delta.trim();
      }
      if (eventName === 'response.output_text.delta') return 'Antworttext empfangen';
      if (eventName === 'message' && data && Array.isArray(data.choices)) {
        const choice = data.choices[0] || {};
        const delta = choice.delta && typeof choice.delta === 'object' ? choice.delta : {};
        if (choice.finish_reason) return `Chat abgeschlossen: ${choice.finish_reason}`;
        if (delta.alpha_reasoning_kind === 'internal_plan') return 'Interner Plan empfangen';
        if (delta.alpha_reasoning_kind === 'model' || delta.reasoning_content || delta.reasoning) {
          return 'Modell-Reasoning empfangen';
        }
        if (chatTextDelta(data)) return 'Antworttext empfangen';
      }
      return eventName;
    }

    function traceStepForEvent(parsed, started) {
      const data = parsed.data && typeof parsed.data === 'object' ? parsed.data : {};
      const choice = data && Array.isArray(data.choices) ? data.choices[0] : null;
      const choiceDelta = choice && choice.delta && typeof choice.delta === 'object' ? choice.delta : {};
      const chatReasoning = typeof choiceDelta.reasoning_content === 'string'
        ? choiceDelta.reasoning_content
        : (typeof choiceDelta.reasoning === 'string' ? choiceDelta.reasoning : '');
      return {
        name: streamStatusText(parsed.event, data),
        elapsed_seconds: (performance.now() - started) / 1000,
        event: parsed.event,
        sequence_number: data.sequence_number,
        delta_chars: typeof data.delta === 'string' ? data.delta.length : (chatReasoning ? chatReasoning.length : undefined),
        reasoning_kind: typeof data.alpha_reasoning_kind === 'string'
          ? data.alpha_reasoning_kind
          : (typeof choiceDelta.alpha_reasoning_kind === 'string' ? choiceDelta.alpha_reasoning_kind : undefined),
        text_delta: parsed.event === 'response.output_text.delta' || Boolean(chatTextDelta(data)),
      };
    }

    async function consumeSseResponse(res, handlers) {
      if (!res.body) throw new Error('Streaming wird von diesem Browser nicht unterstützt.');
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const blocks = buffer.split(/\\r?\\n\\r?\\n/);
        buffer = blocks.pop() || '';
        for (const block of blocks) {
          const parsed = parseSseBlock(block.trim());
          if (parsed) handlers.onEvent(parsed);
        }
        if (done) break;
      }
      const tail = buffer.trim();
      if (tail) {
        const parsed = parseSseBlock(tail);
        if (parsed) handlers.onEvent(parsed);
      }
    }

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const text = input.value.trim();
      if (!text) return;
      input.value = '';
      messages.push({ role: 'user', content: text });
      const assistantMsg = {
        role: 'assistant',
        content: '',
        reasoning: '',
        reasoningStatus: '',
        internalPlan: '',
        reasoningMode: '',
        reasoningOpen: false,
        route: ''
      };
      messages.push(assistantMsg);
      render();
      sendBtn.disabled = true;
      statusEl.textContent = 'streamt...';
      const started = performance.now();
      const traceId = makeId('trace');
      const rawEvents = [];
      const streamSteps = [];
      try {
        const res = await fetch('/api/send_stream', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            message: text,
            messages: messages.slice(0, -2),
            protocol: protocol.value,
            stream: true,
            session_id: sessionId,
            trace_id: traceId
          })
        });
        if (!res.ok) {
          const errorText = await res.text();
          throw new Error(errorText || res.statusText);
        }
        await consumeSseResponse(res, {
          onEvent(parsed) {
            rawEvents.push(parsed);
            if (parsed.done) return;
            const data = parsed.data;
            if (parsed.event === 'test_ui.error') {
              const detail = data && typeof data === 'object' ? data.detail || data.status_code || 'unbekannt' : data;
              throw new Error(`Stream-Fehler: ${detail}`);
            }
            const reasoning = reasoningDelta(
              protocol.value,
              parsed.event,
              data,
              `${assistantMsg.reasoningStatus}${assistantMsg.internalPlan}${assistantMsg.reasoning}`
            );
            if (reasoning) {
              const kind = reasoningKind(data, reasoning, assistantMsg);
              assistantMsg.reasoningMode = kind;
              if (kind === 'status') {
                assistantMsg.reasoningStatus += reasoning;
              } else if (kind === 'internal_plan') {
                assistantMsg.internalPlan += cleanInternalReasoning(reasoning);
              } else {
                assistantMsg.reasoning += reasoning;
              }
            }
            const delta = protocol.value === 'chat' ? chatTextDelta(data) : responseTextDelta(parsed.event, data);
            if (delta) {
              assistantMsg.content += delta;
            }
            const inferredRoute = routeFromEvent(protocol.value, parsed.event, data, delta, reasoning);
            if (inferredRoute && !assistantMsg.route) {
              assistantMsg.route = inferredRoute;
            }
            if (reasoning || delta || inferredRoute) {
              render();
            }
            const step = traceStepForEvent(parsed, started);
            streamSteps.push(step);
            if (streamSteps.length > 160) streamSteps.splice(0, streamSteps.length - 160);
            const browserSeconds = (performance.now() - started) / 1000;
            renderTrace({ trace_id: traceId, protocol: protocol.value, steps: streamSteps }, browserSeconds);
            statusEl.textContent = `${protocol.value} stream | ${routeLabel(assistantMsg.route)} | ${streamStatusText(parsed.event, data)}`;
          }
        });
        rawEl.textContent = JSON.stringify(rawEvents, null, 2);
        const browserSeconds = (performance.now() - started) / 1000;
        renderTrace({ trace_id: traceId, protocol: protocol.value, steps: streamSteps }, browserSeconds);
        statusEl.textContent = `${protocol.value} stream | ${routeLabel(assistantMsg.route)} | ${browserSeconds.toFixed(2)}s browser`;
        if (!assistantMsg.content) assistantMsg.content = '(kein sichtbarer Antworttext gestreamt)';
      } catch (error) {
        assistantMsg.content = `FEHLER: ${error.message || error}`;
        statusEl.textContent = 'Fehler';
      } finally {
        sendBtn.disabled = false;
        render();
        input.focus();
      }
    });

    clearBtn.addEventListener('click', () => {
      messages.length = 0;
      sessionId = resetSessionId();
      rawEl.textContent = '{}';
      traceBody.innerHTML = '';
      traceSummary.textContent = 'noch keine Anfrage';
      lastTrace = null;
      lastTraceBrowserSeconds = 0;
      statusEl.textContent = 'neue Session bereit';
      render();
      input.focus();
    });

    document.querySelectorAll('[data-panel-toggle]').forEach((button) => {
      button.addEventListener('click', () => {
        const panel = button.closest('.live-panel');
        if (!panel) return;
        const expanded = panel.classList.toggle('expanded');
        button.textContent = expanded ? 'Klein' : 'Gross';
      });
    });

    traceDeltaDetails.addEventListener('change', () => {
      if (lastTrace) renderTrace(lastTrace, lastTraceBrowserSeconds);
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
