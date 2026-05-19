from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))
os.environ.setdefault("ALPHARAVIS_MEDIA_ROOT", "/tmp/alpharavis-media-server-test")

if "fastapi" not in sys.modules and importlib.util.find_spec("fastapi") is None:
    fastapi_stub = types.ModuleType("fastapi")

    class HTTPException(Exception):
        pass

    class FastAPI:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def get(self, *args, **kwargs):
            return lambda fn: fn

        def post(self, *args, **kwargs):
            return lambda fn: fn

        def mount(self, *args, **kwargs) -> None:
            return None

    fastapi_stub.FastAPI = FastAPI
    fastapi_stub.HTTPException = HTTPException
    sys.modules["fastapi"] = fastapi_stub

    responses_stub = types.ModuleType("fastapi.responses")
    responses_stub.HTMLResponse = str
    sys.modules["fastapi.responses"] = responses_stub

    staticfiles_stub = types.ModuleType("fastapi.staticfiles")

    class StaticFiles:
        def __init__(self, *args, **kwargs) -> None:
            pass

    staticfiles_stub.StaticFiles = StaticFiles
    sys.modules["fastapi.staticfiles"] = staticfiles_stub

if "pydantic" not in sys.modules and importlib.util.find_spec("pydantic") is None:
    pydantic_stub = types.ModuleType("pydantic")

    class BaseModel:
        pass

    def Field(default=None, *, default_factory=None, **kwargs):
        return default_factory() if default_factory is not None else default

    pydantic_stub.BaseModel = BaseModel
    pydantic_stub.Field = Field
    sys.modules["pydantic"] = pydantic_stub

import media_server  # noqa: E402


def test_download_asset_accepts_inline_video_data(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(media_server, "MEDIA_ROOT", tmp_path)
    target = tmp_path / "video.mp4"

    result = asyncio.run(media_server._download_asset("data:video/mp4;base64,QUJD", target))

    assert target.read_bytes() == b"ABC"
    assert result["bytes"] == 3
    assert result["path"] == str(target)


def test_stored_source_url_omits_inline_blob() -> None:
    source_url = media_server._stored_source_url("data:video/mp4;base64,QUJD")

    assert source_url == "data:video/mp4;base64,[inline-data-omitted]"


class _FakeCursor(list):
    def __init__(self, rows: list[dict]) -> None:
        super().__init__(rows)
        self.sort_field = ""
        self.sort_direction = 0

    def sort(self, field: str, direction: int):
        self.sort_field = field
        self.sort_direction = direction
        super().sort(key=lambda row: row.get(field) or "", reverse=direction < 0)
        return self

    def limit(self, limit: int):
        return _FakeCursor(list(self[:limit]))


class _FakeCollection:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.queries: list[dict] = []
        self.cursor: _FakeCursor | None = None

    def find(self, query: dict):
        self.queries.append(query)
        cursor = _FakeCursor([dict(row) for row in self.rows])
        self.cursor = cursor
        return cursor


def test_assets_support_thread_group_filters_and_sort(monkeypatch) -> None:
    collection = _FakeCollection(
        [
            {"_id": "b", "asset_id": "b", "title": "Beta", "created_at": 2},
            {"_id": "a", "asset_id": "a", "title": "Alpha", "created_at": 1},
        ]
    )
    monkeypatch.setattr(media_server, "_collection", lambda: collection)

    result = asyncio.run(
        media_server.list_assets(
            thread_key="chat-1",
            group_id="group-1",
            media_type="image",
            sort="title",
            order="asc",
        )
    )

    assert collection.queries[0] == {
        "media_type": "image",
        "thread_key": "chat-1",
        "$or": [{"group_id": "group-1"}, {"derivation_group_id": "group-1"}],
    }
    assert collection.cursor is not None
    assert collection.cursor.sort_field == "title"
    assert collection.cursor.sort_direction == 1
    assert [asset["asset_id"] for asset in result["assets"]] == ["a", "b"]


def test_gallery_can_group_by_thread_and_sort_by_name(monkeypatch) -> None:
    collection = _FakeCollection(
        [
            {
                "_id": "video",
                "asset_id": "video",
                "title": "Video",
                "media_type": "video",
                "asset_kind": "original",
                "role": "input",
                "thread_key": "chat-1",
                "group_id": "chat-1",
                "derivation_group_id": "chat-1",
                "source_key": "video",
                "public_url": "http://localhost:8130/media/video.mp4",
                "created_at": 2,
            },
            {
                "_id": "image",
                "asset_id": "image",
                "title": "Image",
                "media_type": "image",
                "asset_kind": "original",
                "role": "input",
                "thread_key": "chat-1",
                "group_id": "chat-1",
                "derivation_group_id": "chat-1",
                "source_key": "image",
                "public_url": "http://localhost:8130/media/image.png",
                "created_at": 1,
            },
        ]
    )
    monkeypatch.setattr(media_server, "_collection", lambda: collection)

    html = asyncio.run(media_server.gallery(group_by="thread", sort="title", order="asc"))

    assert "chat-1 (2)" in html
    assert html.index("Image") < html.index("Video")
    assert "name='group_by'" in html
    assert "name='sort'" in html
