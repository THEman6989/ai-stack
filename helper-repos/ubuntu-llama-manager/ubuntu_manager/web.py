from __future__ import annotations


ESP_CONTROL_HTML = r"""<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ESP Power Control</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #111418;
      --panel: #1a2027;
      --panel-2: #202832;
      --text: #eef2f6;
      --muted: #9aa8b5;
      --line: #303a46;
      --green: #39d98a;
      --blue: #6db7ff;
      --red: #ff5f68;
      --amber: #f8c35d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font: 15px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      width: min(980px, calc(100vw - 28px));
      margin: 0 auto;
      padding: 28px 0;
    }
    header {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 18px;
      padding-bottom: 18px;
      border-bottom: 1px solid var(--line);
    }
    h1 {
      margin: 0;
      font-size: 28px;
      font-weight: 760;
      letter-spacing: 0;
    }
    .sub {
      margin-top: 6px;
      color: var(--muted);
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-top: 20px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    .wide { grid-column: 1 / -1; }
    label {
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }
    input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #0d1014;
      color: var(--text);
      padding: 11px 12px;
      font: inherit;
    }
    .actions {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }
    .tabs {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .tab {
      min-height: 46px;
      color: var(--muted);
    }
    .tab.active {
      color: var(--text);
      border-color: color-mix(in srgb, var(--blue) 60%, var(--line));
    }
    button {
      min-height: 76px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
      color: var(--text);
      padding: 14px 12px;
      font: inherit;
      font-weight: 720;
      cursor: pointer;
    }
    button:hover { border-color: #526273; }
    button:disabled { opacity: .55; cursor: not-allowed; }
    .short { border-color: color-mix(in srgb, var(--green) 45%, var(--line)); }
    .reset { border-color: color-mix(in srgb, var(--blue) 45%, var(--line)); }
    .long { border-color: color-mix(in srgb, var(--red) 50%, var(--line)); }
    .float { border-color: color-mix(in srgb, var(--amber) 45%, var(--line)); }
    .cancel { min-height: 48px; color: var(--amber); }
    .btn-title { display: block; font-size: 17px; }
    .btn-sub { display: block; margin-top: 4px; color: var(--muted); font-weight: 520; font-size: 13px; }
    .status {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }
    .metric {
      background: #0d1014;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 78px;
    }
    .metric strong { display: block; font-size: 13px; color: var(--muted); margin-bottom: 6px; }
    .ok { color: var(--green); }
    .bad { color: var(--red); }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
      color: #d8e0e8;
      font-size: 13px;
    }
    .warn {
      border-color: color-mix(in srgb, var(--amber) 45%, var(--line));
      color: #ffe0a3;
    }
    .hidden { display: none; }
    @media (max-width: 760px) {
      header, .status, .actions, .grid { grid-template-columns: 1fr; display: grid; }
      .grid { gap: 12px; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>ESP Power Control</h1>
        <div class="sub">Manager API auf Port 8099, ESP-Aktionen ueber den konfigurierten Webhook.</div>
      </div>
      <button class="cancel" id="refresh">Status aktualisieren</button>
    </header>

    <section class="grid">
      <div class="panel wide warn">
        Diese Seite drueckt echte Mainboard-Taster ueber den Optokoppler. Power lang kann den PC hart ausschalten.
      </div>

      <div class="panel wide">
        <label for="token">API Token</label>
        <input id="token" type="password" autocomplete="current-password" placeholder="API_TOKEN / MANAGER_API_TOKEN">
      </div>

      <div class="panel wide tabs">
        <button class="tab active" id="tabPower" type="button">Power Control</button>
        <button class="tab" id="tabPinTest" type="button">Pin-Test</button>
      </div>

      <div class="panel wide" id="powerPanel">
        <div class="actions">
          <button class="short" id="powerShort">
            <span class="btn-title">Power kurz</span>
            <span class="btn-sub">1 Sekunde, wie normaler Power-Taster</span>
          </button>
          <button class="reset" id="resetBtn">
            <span class="btn-title">Neustart</span>
            <span class="btn-sub">Power 8s aus, 20s warten, Power kurz an</span>
          </button>
          <button class="long" id="powerLong">
            <span class="btn-title">Power lang</span>
            <span class="btn-sub">8 Sekunden halten, Force-Off</span>
          </button>
        </div>
      </div>

      <div class="panel wide hidden" id="pinTestPanel">
        <label>Optokoppler Eingangstest ohne Mainboard-Ausgang</label>
        <div class="actions">
          <button class="short" id="powerHigh">
            <span class="btn-title">Power HIGH</span>
            <span class="btn-sub">D1 fuer 5 Sekunden HIGH</span>
          </button>
          <button class="long" id="powerLow">
            <span class="btn-title">Power LOW</span>
            <span class="btn-sub">D1 fuer 5 Sekunden LOW</span>
          </button>
          <button class="float" id="powerFloat">
            <span class="btn-title">Power FLOAT</span>
            <span class="btn-sub">D1 sofort hochohmig</span>
          </button>
          <button class="short" id="resetHigh">
            <span class="btn-title">Reset HIGH</span>
            <span class="btn-sub">D2 fuer 5 Sekunden HIGH</span>
          </button>
          <button class="long" id="resetLow">
            <span class="btn-title">Reset LOW</span>
            <span class="btn-sub">D2 fuer 5 Sekunden LOW</span>
          </button>
          <button class="float" id="resetFloat">
            <span class="btn-title">Reset FLOAT</span>
            <span class="btn-sub">D2 sofort hochohmig</span>
          </button>
        </div>
      </div>

      <div class="panel wide status">
        <div class="metric"><strong>Manager</strong><span id="managerState">...</span></div>
        <div class="metric"><strong>ESP</strong><span id="espState">...</span></div>
      </div>

      <div class="panel wide">
        <label>Antwort</label>
        <pre id="output">Bereit.</pre>
      </div>
    </section>
  </main>

  <script>
    const tokenInput = document.getElementById('token');
    const output = document.getElementById('output');
    const managerState = document.getElementById('managerState');
    const espState = document.getElementById('espState');
    const saved = localStorage.getItem('ubuntuManagerApiToken') || '';
    tokenInput.value = saved;
    tokenInput.addEventListener('input', () => localStorage.setItem('ubuntuManagerApiToken', tokenInput.value));

    function authHeaders(extra = {}) {
      const token = tokenInput.value.trim();
      return Object.assign({
        'Content-Type': 'application/json',
        ...(token ? {'Authorization': 'Bearer ' + token} : {})
      }, extra);
    }

    function show(value) {
      output.textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
    }

    async function getJson(path) {
      const res = await fetch(path);
      const data = await res.json();
      return {res, data};
    }

    async function postAction(payload) {
      if (!tokenInput.value.trim()) {
        show('Bitte API_TOKEN eintragen.');
        return;
      }
      setBusy(true);
      try {
        const res = await fetch('/esp/action', {
          method: 'POST',
          headers: authHeaders(),
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        show(data);
        await refreshStatus();
      } catch (err) {
        show(String(err));
      } finally {
        setBusy(false);
      }
    }

    async function postPinTest(payload) {
      if (!tokenInput.value.trim()) {
        show('Bitte API_TOKEN eintragen.');
        return;
      }
      setBusy(true);
      try {
        const res = await fetch('/esp/pin-test', {
          method: 'POST',
          headers: authHeaders(),
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        show(data);
        await refreshStatus();
      } catch (err) {
        show(String(err));
      } finally {
        setBusy(false);
      }
    }

    function selectTab(tab) {
      const power = tab === 'power';
      document.getElementById('powerPanel').classList.toggle('hidden', !power);
      document.getElementById('pinTestPanel').classList.toggle('hidden', power);
      document.getElementById('tabPower').classList.toggle('active', power);
      document.getElementById('tabPinTest').classList.toggle('active', !power);
      location.hash = power ? 'power' : 'pin-test';
    }

    function setBusy(busy) {
      for (const btn of document.querySelectorAll('button')) btn.disabled = busy;
    }

    async function refreshStatus() {
      try {
        const health = await getJson('/health');
        managerState.innerHTML = health.data.ok ? '<span class="ok">online</span>' : '<span class="bad">Fehler</span>';
      } catch {
        managerState.innerHTML = '<span class="bad">nicht erreichbar</span>';
      }
      try {
        const status = await getJson('/esp/status');
        const esp = status.data.esp || {};
        const online = esp.esp_online;
        const detail = esp.direct_status && esp.direct_status.body ? ' ' + (esp.direct_status.body.ip || '') : '';
        espState.innerHTML = online ? '<span class="ok">online' + detail + '</span>' : '<span class="bad">offline</span>';
      } catch {
        espState.innerHTML = '<span class="bad">nicht erreichbar</span>';
      }
    }

    document.getElementById('powerShort').addEventListener('click', () => postAction({
      action: 'power-on',
      reason: 'web-power-short',
      hold_seconds: 1,
      wait_seconds: 0,
      delay_before_action_seconds: 0
    }));
    document.getElementById('resetBtn').addEventListener('click', () => {
      if (!confirm('Neustart ueber D1 ausfuehren? Power wird 8 Sekunden gehalten, dann wartet der ESP 20 Sekunden und schaltet wieder ein.')) return;
      postAction({
        action: 'power-cycle',
        reason: 'web-power-cycle-restart',
        hold_seconds: 8,
        wait_seconds: 20,
        delay_before_action_seconds: 0
      });
    });
    document.getElementById('powerLong').addEventListener('click', () => {
      if (!confirm('Power wirklich 8 Sekunden halten? Das kann den PC hart ausschalten.')) return;
      postAction({
        action: 'power-off',
        reason: 'web-power-long',
        hold_seconds: 8,
        wait_seconds: 0,
        delay_before_action_seconds: 0
      });
    });
    document.getElementById('tabPower').addEventListener('click', () => selectTab('power'));
    document.getElementById('tabPinTest').addEventListener('click', () => selectTab('pin-test'));
    document.getElementById('powerHigh').addEventListener('click', () => postPinTest({pin: 'power', level: 'high', hold_seconds: 5}));
    document.getElementById('powerLow').addEventListener('click', () => postPinTest({pin: 'power', level: 'low', hold_seconds: 5}));
    document.getElementById('powerFloat').addEventListener('click', () => postPinTest({pin: 'power', level: 'float', hold_seconds: 0}));
    document.getElementById('resetHigh').addEventListener('click', () => postPinTest({pin: 'reset', level: 'high', hold_seconds: 5}));
    document.getElementById('resetLow').addEventListener('click', () => postPinTest({pin: 'reset', level: 'low', hold_seconds: 5}));
    document.getElementById('resetFloat').addEventListener('click', () => postPinTest({pin: 'reset', level: 'float', hold_seconds: 0}));
    document.getElementById('refresh').addEventListener('click', refreshStatus);
    if (location.hash === '#pin-test' || location.pathname === '/esp/test') selectTab('pin-test');
    refreshStatus();
  </script>
</body>
</html>
"""
