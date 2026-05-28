# DeepAgent Gateway — AI-Stack Integration

Dieses Dokument beschreibt den Endpoint im AI-Stack, den das DeepAgent Gateway
zum Flushen von queued Messages benötigt.

## Endpoint: POST /api/queue/ingest

Der Gateway sendet queued Messages als JSON-Payload an diesen Endpoint.

### Request

```http
POST /api/queue/ingest HTTP/1.1
Content-Type: application/json
Authorization: Bearer <AI_STACK_QUEUE_INGEST_TOKEN>  (optional)

{
  "messages": [
    {
      "id": "uuid-vom-gateway",
      "session_id": "uuid",
      "role": "user",
      "content": "Hello, wie geht's?",
      "created_at": "2025-06-15T10:30:00Z",
      "status": "queued",
      "media": [
        {
          "media_id": "uuid",
          "filename": "screenshot.png",
          "mime_type": "image/png",
          "file_path": "/var/lib/deepagent-gateway/media/uuid.png"
        }
      ]
    }
  ]
}
```

### Response

```json
{
  "accepted": ["uuid-1", "uuid-2"],
  "duplicates": ["uuid-3"],
  "failed": []
}
```

- `accepted`: Message-IDs, die erfolgreich angenommen und als User-Prompt
  in DeepAgent/LangGraph eingespeist werden.
- `duplicates`: Message-IDs, die bereits verarbeitet wurden (Idempotenz).
- `failed`: Message-IDs, die nicht verarbeitet werden konnten.

## Implementierung im AI-Stack

Der Endpoint ist implementiert in `langgraph-app/queue_ingest.py` und als Router
in `langgraph-app/bridge_server.py` eingebunden (`POST /api/queue/ingest`).

### Architektur

```python
# langgraph-app/queue_ingest.py
# Exportiert einen FastAPI APIRouter, der in bridge_server.py via
# app.include_router(queue_ingest_router) eingebunden wird.
#
# Features:
# - In-Memory Idempotenz (message_id → verarbeitet)
# - Submittet jede neue Message als User-Prompt an LangGraph
# - Thread-basiert pro session_id ("queue-{session_id}")
# - Optionaler Auth-Token (QUEUE_INGEST_TOKEN / GATEWAY_ADMIN_TOKEN)
```

### Idempotenz

Der AI-Stack muss sich gemerkte message_ids speichern (z.B. in Redis oder SQLite),
um doppelte Verarbeitung zu verhindern. Der Gateway sendet bei jedem Flush
potentiell dieselben Messages erneut (wenn der vorherige Flush nicht bestätigt wurde).

### Media-Handling

Wenn eine Message `media`-Einträge hat, muss der AI-Stack die Mediendateien
vom Gateway abrufen:

```
GET /api/gateway/media/{media_id}
```

Dafür muss der AI-Stack das Gateway erreichen können (internes Netzwerk).

## Unit-Test

```python
# tests/test_queue_ingest.py

async def test_queue_ingest_idempotency():
    """Stellt sicher, dass doppelte message_ids als duplicates erkannt werden."""
    payload = {
        "messages": [
            {"id": "msg-1", "session_id": "s1", "role": "user", "content": "Test"}
        ]
    }
    # Erster Request
    r1 = await client.post("/api/queue/ingest", json=payload)
    assert r1.json()["accepted"] == ["msg-1"]

    # Zweiter Request (gleiche message_id)
    r2 = await client.post("/api/queue/ingest", json=payload)
    assert r2.json()["duplicates"] == ["msg-1"]
```
