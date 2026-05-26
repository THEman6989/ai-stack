from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))

import odf_converter  # noqa: E402


class _FakeResponse:
    def __init__(
        self,
        status_code: int,
        *,
        json_data: dict | None = None,
        content: bytes = b"",
        text: str = "",
        headers: dict | None = None,
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data or {}
        self.content = content
        self.text = text
        self.headers = headers or {"content-type": "application/json"}

    def json(self) -> dict:
        return self._json_data


class _FakeAsyncClient:
    last_payload: dict | None = None
    last_url: str = ""
    response: _FakeResponse = _FakeResponse(
        200,
        json_data={"endConvert": True, "fileUrl": "http://onlyoffice/cache/out.docx"},
        text='{"endConvert": true, "fileUrl": "http://onlyoffice/cache/out.docx"}',
    )

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, url: str, *, json: dict):
        type(self).last_url = url
        type(self).last_payload = json
        return type(self).response

    async def get(self, url: str):
        return _FakeResponse(200, content=b"DOCXDATA")


def test_convert_odf_to_ooxml_uses_onlyoffice_json_url_contract(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "sample.odt"
    input_path.write_bytes(b"ODT")
    monkeypatch.setattr(odf_converter.httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(odf_converter, "ONLYOFFICE_URL", "http://onlyoffice:80")
    _FakeAsyncClient.response = _FakeResponse(
        200,
        json_data={"endConvert": True, "fileUrl": "http://onlyoffice/cache/out.docx"},
        text='{"endConvert": true, "fileUrl": "http://onlyoffice/cache/out.docx"}',
    )

    result = asyncio.run(
        odf_converter.convert_odf_to_ooxml(
            str(input_path),
            "application/vnd.oasis.opendocument.text",
            str(tmp_path),
            source_url="http://media-gallery:8130/media/sample.odt",
        )
    )

    payload = _FakeAsyncClient.last_payload
    assert _FakeAsyncClient.last_url == "http://onlyoffice:80/ConvertService.ashx"
    assert payload is not None
    assert payload["url"] == "http://media-gallery:8130/media/sample.odt"
    assert payload["filetype"] == "odt"
    assert payload["outputtype"] == "docx"
    assert payload["async"] is False
    assert payload["key"].startswith("alpharavis-")
    assert result["output_format"] == "docx"
    assert Path(str(result["output_path"])).read_bytes() == b"DOCXDATA"


def test_convert_odf_to_ooxml_accepts_onlyoffice_xml_response(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "sample.odt"
    input_path.write_bytes(b"ODT")
    monkeypatch.setattr(odf_converter.httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(odf_converter, "ONLYOFFICE_URL", "http://onlyoffice:80")
    _FakeAsyncClient.response = _FakeResponse(
        200,
        text='<?xml version="1.0" encoding="utf-8"?><FileResult><EndConvert>true</EndConvert><FileUrl>http://onlyoffice/cache/out.docx</FileUrl></FileResult>',
        headers={"content-type": "text/xml; charset=UTF-8"},
    )

    result = asyncio.run(
        odf_converter.convert_odf_to_ooxml(
            str(input_path),
            "application/vnd.oasis.opendocument.text",
            str(tmp_path),
            source_url="http://media-gallery:8130/media/sample.odt",
        )
    )

    assert result["output_format"] == "docx"
    assert Path(str(result["output_path"])).read_bytes() == b"DOCXDATA"


def test_parse_onlyoffice_xml_error_reports_download_block() -> None:
    response = _FakeResponse(
        200,
        text='<?xml version="1.0" encoding="utf-8"?><FileResult><Error>-4</Error></FileResult>',
        headers={"content-type": "text/xml; charset=UTF-8"},
    )
    data = odf_converter._parse_conversion_response(response)
    assert data == {"error": "-4"}
    assert "download" in odf_converter._format_onlyoffice_error("-4")


def test_is_odf_detects_odf_mime_types() -> None:
    assert odf_converter.is_odf("application/vnd.oasis.opendocument.text")
    assert odf_converter.is_odf("application/vnd.oasis.opendocument.presentation")
    assert odf_converter.is_odf("application/vnd.oasis.opendocument.spreadsheet")
    assert not odf_converter.is_odf("application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    assert not odf_converter.is_odf("")
    assert not odf_converter.is_odf("text/plain")


def test_target_format_maps_odf_to_ooxml() -> None:
    assert odf_converter._target_format("application/vnd.oasis.opendocument.text") == "docx"
    assert odf_converter._target_format("application/vnd.oasis.opendocument.presentation") == "pptx"
    assert odf_converter._target_format("application/vnd.oasis.opendocument.spreadsheet") == "xlsx"
    assert odf_converter._target_format("unknown/type") == "docx"


def test_target_mime_and_ext_are_consistent() -> None:
    odt = "application/vnd.oasis.opendocument.text"
    assert odf_converter._target_mime(odt).startswith("application/vnd.openxmlformats-officedocument")
    assert odf_converter._target_ext(odt) == ".docx"
    assert odf_converter._target_mime("application/vnd.oasis.opendocument.presentation").startswith("application/vnd.openxmlformats-officedocument")
    assert odf_converter._target_ext("application/vnd.oasis.opendocument.presentation") == ".pptx"


def test_format_onlyoffice_error_unknown_code() -> None:
    assert odf_converter._format_onlyoffice_error("-99") == "-99"
    assert odf_converter._format_onlyoffice_error("-4") == "-4 (OnlyOffice could not download the source document URL)"


def test_parse_conversion_response_rejects_plain_text() -> None:
    import pytest as pytest_mod
    resp = _FakeResponse(200, text="plain text", headers={"content-type": "text/plain"})
    with pytest_mod.raises(ValueError, match="Unsupported"):
        odf_converter._parse_conversion_response(resp)
