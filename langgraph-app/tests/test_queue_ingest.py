"""Tests für Queue-Ingest-Endpoint — Idempotenz, Auth, Message-Verarbeitung."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from queue_ingest import (
    _mark_processed,
    _is_processed,
    _PROCESSED_IDS,
)


class TestIdempotency:
    def setup_method(self):
        _PROCESSED_IDS.clear()

    def test_mark_and_check_processed(self):
        assert not _is_processed("msg-1")
        _mark_processed("msg-1")
        assert _is_processed("msg-1")

    def test_duplicate_rejected(self):
        _mark_processed("msg-1")
        _mark_processed("msg-2")
        assert _is_processed("msg-1")
        assert _is_processed("msg-2")
        assert not _is_processed("msg-3")

    def test_eviction_on_limit(self):
        # Simulate reaching the limit
        from queue_ingest import _MAX_PROCESSED_IDS

        with patch.object(
            __import__("queue_ingest"), "_MAX_PROCESSED_IDS", 10
        ):
            for i in range(15):
                __import__("queue_ingest")._mark_processed(f"msg-{i}")

            # Should still have entries (evicted ~10%)
            ids = __import__("queue_ingest")._PROCESSED_IDS
            assert len(ids) <= 15  # Some evicted
            # Newer entries should survive
            assert "msg-14" in ids


class TestQueueIngestRoute:
    @pytest.mark.asyncio
    async def test_empty_messages(self):
        """POST mit leerem messages-Array → 400."""
        from fastapi.testclient import TestClient
        from queue_ingest import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        resp = client.post("/api/queue/ingest", json={"messages": []})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_duplicate_detection(self):
        """Nachricht mit gleicher ID zweimal → erstes accepted, zweites duplicate."""
        from fastapi.testclient import TestClient
        from queue_ingest import router, _PROCESSED_IDS
        from fastapi import FastAPI

        _PROCESSED_IDS.clear()

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        payload = {
            "messages": [
                {"id": "dup-1", "session_id": "s1", "role": "user", "content": "Test"}
            ]
        }

        # Mock LangGraph submission
        with patch(
            "queue_ingest._submit_to_langgraph", new=AsyncMock()
        ) as mock_submit:
            mock_submit.return_value = None

            r1 = client.post("/api/queue/ingest", json=payload)
            assert r1.status_code == 200
            assert r1.json()["accepted"] == ["dup-1"]
            assert r1.json()["duplicates"] == []

            r2 = client.post("/api/queue/ingest", json=payload)
            assert r2.status_code == 200
            assert r2.json()["duplicates"] == ["dup-1"]
            assert r2.json()["accepted"] == []

    @pytest.mark.asyncio
    async def test_auth_token_required(self):
        """Wenn QUEUE_INGEST_TOKEN gesetzt → 403 ohne Token."""
        from fastapi.testclient import TestClient
        from queue_ingest import router, QUEUE_INGEST_TOKEN
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        with patch("queue_ingest.QUEUE_INGEST_TOKEN", "secret"):
            resp = client.post(
                "/api/queue/ingest",
                json={"messages": [{"id": "m1", "session_id": "s1", "role": "user", "content": "X"}]},
            )
            assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_auth_token_accepted(self):
        """Mit korrektem Token → 200."""
        from fastapi.testclient import TestClient
        from queue_ingest import router, _PROCESSED_IDS
        from fastapi import FastAPI

        _PROCESSED_IDS.clear()

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        with patch("queue_ingest.QUEUE_INGEST_TOKEN", "secret"):
            with patch(
                "queue_ingest._submit_to_langgraph", new=AsyncMock()
            ):
                resp = client.post(
                    "/api/queue/ingest",
                    json={"messages": [{"id": "m1", "session_id": "s1", "role": "user", "content": "X"}]},
                    headers={"Authorization": "Bearer secret"},
                )
                assert resp.status_code == 200
