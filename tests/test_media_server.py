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
