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
os.environ.setdefault("ALPHARAVIS_OFFICE_OUTPUT_ROOT", "/tmp/alpharavis-office-output-test")

if "fastapi" not in sys.modules and importlib.util.find_spec("fastapi") is None:
    fastapi_stub = types.ModuleType("fastapi")

    class HTTPException(Exception):
        def __init__(self, status_code: int = 500, detail: str = "") -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class FastAPI:
        def __init__(self, *args, **kwargs) -> None:
            self.user_middleware = []

        def add_middleware(self, cls, **options) -> None:
            self.user_middleware.append(types.SimpleNamespace(cls=cls, options=options))

        def get(self, *args, **kwargs):
            return lambda fn: fn

        def post(self, *args, **kwargs):
            return lambda fn: fn

        def mount(self, *args, **kwargs) -> None:
            return None

    fastapi_stub.FastAPI = FastAPI
    fastapi_stub.HTTPException = HTTPException
    fastapi_stub.Request = object
    sys.modules["fastapi"] = fastapi_stub

    middleware_stub = types.ModuleType("fastapi.middleware")
    cors_stub = types.ModuleType("fastapi.middleware.cors")

    class CORSMiddleware:
        pass

    cors_stub.CORSMiddleware = CORSMiddleware
    sys.modules["fastapi.middleware"] = middleware_stub
    sys.modules["fastapi.middleware.cors"] = cors_stub

    responses_stub = types.ModuleType("fastapi.responses")
    responses_stub.HTMLResponse = str
    responses_stub.Response = str
    responses_stub.RedirectResponse = str
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
        def __init__(self, **kwargs) -> None:
            for name, value in kwargs.items():
                setattr(self, name, value)

    def Field(default=None, *, default_factory=None, **kwargs):
        return default_factory() if default_factory is not None else default

    pydantic_stub.BaseModel = BaseModel
    pydantic_stub.Field = Field
    sys.modules["pydantic"] = pydantic_stub

import media_server  # noqa: E402


def test_office_output_listing_returns_supported_files(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "office-output"
    root.mkdir()
    (root / "deck.pptx").write_bytes(b"PPTX")
    (root / "notes.txt").write_text("ignore")
    nested = root / "nested"
    nested.mkdir()
    (nested / "report.docx").write_bytes(b"DOCX")

    monkeypatch.setattr(media_server, "OFFICE_OUTPUT_ROOT", root.resolve())
    monkeypatch.setattr(media_server, "OFFICE_OUTPUT_PUBLIC_BASE_URL", "http://localhost:8130/office-output")

    files = media_server._list_office_output_files(limit=10)

    assert {item["relative_path"] for item in files} == {"deck.pptx", "nested/report.docx"}
    assert all(item["download_url"].startswith("http://localhost:8130/office-output/") for item in files)


def test_office_output_record_links_existing_preview_artifacts(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "office-output"
    root.mkdir()
    (root / "deck.pptx").write_bytes(b"PPTX")
    (root / "deck-preview.png").write_bytes(b"PNG")
    (root / "deck-preview.html").write_text("<html></html>", encoding="utf-8")
    monkeypatch.setattr(media_server, "OFFICE_OUTPUT_ROOT", root.resolve())
    monkeypatch.setattr(media_server, "OFFICE_OUTPUT_PUBLIC_BASE_URL", "http://localhost:8130/office-output")

    record = media_server._office_output_record(root / "deck.pptx")

    assert record["preview_available"] is True
    assert record["preview_image_url"] == "http://localhost:8130/office-output/deck-preview.png"
    assert record["preview_html_url"] == "http://localhost:8130/office-output/deck-preview.html"


def test_office_output_listing_hides_sibling_preview_artifacts(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "office-output"
    root.mkdir()
    (root / "deck.pptx").write_bytes(b"PPTX")
    (root / "deck-preview.png").write_bytes(b"PNG")
    (root / "deck-preview.html").write_text("<html></html>", encoding="utf-8")
    monkeypatch.setattr(media_server, "OFFICE_OUTPUT_ROOT", root.resolve())
    monkeypatch.setattr(media_server, "OFFICE_OUTPUT_PUBLIC_BASE_URL", "http://localhost:8130/office-output")

    files = media_server._list_office_output_files(limit=10)

    assert [item["relative_path"] for item in files] == ["deck.pptx"]
    assert files[0]["preview_available"] is True


async def _call_list_office_output_files() -> dict:
    return await media_server.list_office_output_files(limit=10)


def test_office_files_endpoint_uses_office_output_root(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "office-output"
    root.mkdir()
    (root / "sheet.xlsx").write_bytes(b"XLSX")
    monkeypatch.setattr(media_server, "OFFICE_OUTPUT_ROOT", root.resolve())

    result = asyncio.run(_call_list_office_output_files())

    assert result["root"] == str(root.resolve())
    assert result["files"][0]["filename"] == "sheet.xlsx"


class _FakeComfyClient:
    def __init__(self, base_url: str = "") -> None:
        self.base_url = base_url or "http://comfypc:8188"

    async def queue(self) -> dict:
        return {"queue_running": ["run"], "queue_pending": ["one", "two"]}

    async def models(self, folder: str = "checkpoints") -> list[str]:
        return [f"{folder}/model.safetensors"]

    async def system_stats(self) -> dict:
        return {"system": {"os": "test"}}

    async def history_outputs(self, prompt_id: str) -> dict:
        return {
            "prompt_id": prompt_id,
            "history": {prompt_id: {"outputs": {}}},
            "outputs": [
                {
                    "node_id": "9",
                    "output_type": "images",
                    "filename": "ComfyUI_00001_.png",
                    "subfolder": "",
                    "type": "output",
                    "url": f"{self.base_url}/view?filename=ComfyUI_00001_.png&subfolder=&type=output",
                }
            ],
        }

    async def preflight_workflow(self, workflow: dict, *, check_server: bool = True) -> dict:
        return {"ready": bool(workflow), "format": "api", "server_checked": check_server}

    async def submit_workflow(self, workflow: dict, *, client_id: str = "alpharavis") -> dict:
        return {"prompt_id": "submitted-123", "client_id": client_id, "node_count": len(workflow)}

async def _call_comfy_queue() -> dict:

    return await media_server.comfyui_queue_endpoint()


async def _call_comfy_models() -> dict:
    return await media_server.comfyui_models_endpoint("checkpoints")


async def _call_comfy_history() -> dict:
    return await media_server.comfyui_history_endpoint("abc")


def test_comfyui_proxy_endpoints_use_configured_client(monkeypatch) -> None:
    monkeypatch.setattr(media_server, "ComfyUIClient", _FakeComfyClient)
    monkeypatch.setattr(media_server, "resolve_comfyui_base_url", lambda remote_pcs: "http://comfypc:8188")

    queue = asyncio.run(_call_comfy_queue())
    models = asyncio.run(_call_comfy_models())
    history = asyncio.run(_call_comfy_history())

    assert queue["ok"] is True
    assert queue["base_url"] == "http://comfypc:8188"
    assert len(queue["queue"]["queue_pending"]) == 2
    assert models["models"] == ["checkpoints/model.safetensors"]
    assert history["ok"] is True
    assert history["prompt_id"] == "abc"
    assert history["outputs"][0]["filename"] == "ComfyUI_00001_.png"


def test_comfyui_workflow_preflight_endpoint_uses_client(monkeypatch) -> None:
    monkeypatch.setattr(media_server, "ComfyUIClient", _FakeComfyClient)
    monkeypatch.setattr(media_server, "resolve_comfyui_base_url", lambda remote_pcs: "http://comfypc:8188")

    request = media_server.ComfyUIWorkflowRequest(workflow={"1": {"class_type": "CheckpointLoaderSimple"}}, check_server=False)
    result = asyncio.run(media_server.comfyui_preflight_endpoint(request))

    assert result["ok"] is True
    assert result["base_url"] == "http://comfypc:8188"
    assert result["preflight"] == {"ready": True, "format": "api", "server_checked": False}


def test_comfyui_workflow_prompt_endpoint_uses_submit_gate(monkeypatch) -> None:
    monkeypatch.setattr(media_server, "ComfyUIClient", _FakeComfyClient)
    monkeypatch.setattr(media_server, "resolve_comfyui_base_url", lambda remote_pcs: "http://comfypc:8188")

    request = media_server.ComfyUIWorkflowRequest(workflow={"1": {"class_type": "CheckpointLoaderSimple"}}, client_id="ui-test")
    result = asyncio.run(media_server.comfyui_prompt_endpoint(request))

    assert result["ok"] is True
    assert result["result"] == {"prompt_id": "submitted-123", "client_id": "ui-test", "node_count": 1}


def test_comfyui_register_outputs_endpoint_writes_media_records(monkeypatch) -> None:
    collection = _FakeCollection([])
    monkeypatch.setattr(media_server, "_collection", lambda: collection)

    request = media_server.ComfyUIRegisterOutputsRequest()
    request.prompt_id = "abc"
    request.source_base_url = "http://localhost:8188"
    request.download = False
    request.outputs = [
        {
            "node_id": "9",
            "output_type": "images",
            "filename": "ComfyUI_00001_.png",
            "subfolder": "",
            "type": "output",
        }
    ]

    result = asyncio.run(media_server.comfyui_register_outputs_endpoint(request))

    assert result["ok"] is True
    assert len(collection.replacements) == 1
    record = collection.replacements[0][1]
    assert record["origin"] == "comfyui_output"
    assert record["asset_kind"] == "processed"
    assert record["media_type"] == "image"
    assert record["group_id"] == "abc"
    assert record["download_url"].startswith("http://localhost:8188/view?filename=ComfyUI_00001_.png")
    assert record["metadata"]["prompt_id"] == "abc"


async def _call_list_office_templates() -> dict:
    return await media_server.list_office_templates(limit=10)


def test_office_templates_endpoint_lists_templates_subdir(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "office-output"
    templates = root / "templates"
    templates.mkdir(parents=True)
    (templates / "report.docx").write_bytes(b"DOCX")
    (root / "normal.docx").write_bytes(b"DOCX")
    monkeypatch.setattr(media_server, "OFFICE_OUTPUT_ROOT", root.resolve())
    monkeypatch.setattr(media_server, "OFFICE_OUTPUT_PUBLIC_BASE_URL", "http://localhost:8130/office-output")

    result = asyncio.run(_call_list_office_templates())

    assert result["root"] == str(templates.resolve())
    assert [item["relative_path"] for item in result["files"]] == ["templates/report.docx"]


def test_store_office_uploaded_file_writes_supported_file(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "office-output"
    root.mkdir()
    monkeypatch.setattr(media_server, "OFFICE_OUTPUT_ROOT", root.resolve())
    monkeypatch.setattr(media_server, "OFFICE_OUTPUT_PUBLIC_BASE_URL", "http://localhost:8130/office-output")

    record = media_server._store_office_uploaded_file(
        filename="Quarterly Report.pptx",
        content=b"PPTXDATA",
    )

    stored = root / record["relative_path"]
    assert stored.read_bytes() == b"PPTXDATA"
    assert record["filename"] == "quarterly-report.pptx"
    assert record["download_url"] == "http://localhost:8130/office-output/quarterly-report.pptx"


def test_store_office_uploaded_file_rejects_unsupported_extension(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "office-output"
    root.mkdir()
    monkeypatch.setattr(media_server, "OFFICE_OUTPUT_ROOT", root.resolve())

    try:
        media_server._store_office_uploaded_file(filename="notes.txt", content=b"TEXT")
    except media_server.HTTPException as exc:
        assert getattr(exc, "status_code", None) == 400
        assert "unsupported office file type" in str(getattr(exc, "detail", ""))
    else:  # pragma: no cover - defensive assertion for stubbed environments
        raise AssertionError("unsupported office upload was accepted")


def test_office_validate_phase5_plan_quotes_paths() -> None:
    plan = media_server._office_validate_plan("nested/Quarterly Report.pptx")

    assert plan["operation"] == "validate"
    assert plan["phase"] == 5
    assert plan["file"] == "/workspace/office-output/nested/Quarterly Report.pptx"
    assert "officecli validate '/workspace/office-output/nested/Quarterly Report.pptx'" in plan["commands"][0]
    assert "issues --json" in plan["commands"][1]


def test_office_template_merge_phase5_plan_includes_json_payload() -> None:
    plan = media_server._office_template_merge_plan(
        template="templates/report-template.docx",
        output="reports/report-merged.docx",
        data={"title": "Annual Report", "author": "AlphaRavis"},
    )

    assert plan["operation"] == "template_merge"
    assert plan["phase"] == 5
    assert plan["template"] == "/workspace/office-output/templates/report-template.docx"
    assert plan["output"] == "/workspace/office-output/reports/report-merged.docx"
    assert "officecli merge" in plan["commands"][0]
    assert "Annual Report" in plan["commands"][0]


def test_office_batch_phase5_plan_uses_safe_input_path() -> None:
    plan = media_server._office_batch_plan("templates/invoice.docx", "batch/input.json")

    assert plan["operation"] == "batch"
    assert plan["phase"] == 5
    assert plan["file"] == "/workspace/office-output/templates/invoice.docx"
    assert plan["input"] == "/workspace/office-output/batch/input.json"
    assert "officecli batch" in plan["commands"][0]
    assert "officecli validate" in plan["commands"][1]


def test_office_roundtrip_phase5_plan_writes_blueprint_under_output() -> None:
    plan = media_server._office_roundtrip_plan("decks/demo deck.pptx")

    assert plan["operation"] == "roundtrip"
    assert plan["phase"] == 5
    assert plan["blueprint"] == "/workspace/office-output/decks/demo deck-blueprint.json"
    assert "officecli dump '/workspace/office-output/decks/demo deck.pptx'" in plan["commands"][0]


def test_office_phase5_plan_rejects_path_traversal() -> None:
    try:
        media_server._office_validate_plan("../secret.docx")
    except media_server.HTTPException as exc:
        assert getattr(exc, "status_code", None) == 400
        assert "unsafe office path" in str(getattr(exc, "detail", ""))
    else:  # pragma: no cover - defensive assertion for stubbed environments
        raise AssertionError("unsafe office path was accepted")


def test_office_phase6_preview_plan_generates_html_and_png_without_overwriting() -> None:
    plan = media_server._office_preview_plan("reports/Quarterly Report.docx")

    assert plan["phase"] == 6
    assert plan["operation"] == "preview"
    assert plan["status"] == "planned"
    assert plan["preview_html"] == "/workspace/office-output/reports/Quarterly Report-preview.html"
    assert plan["preview_image"] == "/workspace/office-output/reports/Quarterly Report-preview.png"
    assert "officecli view '/workspace/office-output/reports/Quarterly Report.docx' html" in plan["commands"][0]
    assert "screenshot -o '/workspace/office-output/reports/Quarterly Report-preview.png'" in plan["commands"][1]


def test_office_phase6_repair_plan_writes_repaired_copy() -> None:
    plan = media_server._office_repair_plan("reports/Quarterly Report.docx")

    assert plan["phase"] == 6
    assert plan["operation"] == "repair"
    assert plan["status"] == "planned"
    assert plan["file"] == "/workspace/office-output/reports/Quarterly Report.docx"
    assert plan["output"] == "/workspace/office-output/reports/Quarterly Report-repaired.docx"
    assert plan["output"] != plan["file"]
    assert any("officecli validate" in command for command in plan["commands"])
    assert any("repaired" in note.lower() for note in plan["notes"])


def test_office_phase6_watch_manager_tracks_preview_status() -> None:
    media_server._OFFICE_WATCH_STATE.clear()

    started = media_server._office_watch_plan("reports/Quarterly Report.docx", action="start")
    status = media_server._office_watch_status("reports/Quarterly Report.docx")
    stopped = media_server._office_watch_plan("reports/Quarterly Report.docx", action="stop")

    assert started["phase"] == 6
    assert started["operation"] == "watch_start"
    assert started["status"] == "planned"
    assert started["preview_url"]
    assert "iframe" in started["ui_hint"].lower()
    assert status["status"] == "planned"
    assert status["file"] == started["file"]
    assert stopped["operation"] == "watch_stop"
    assert stopped["status"] == "stopped"


def test_office_phase6_blueprint_suggestion_and_create_plan(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "office-output"
    root.mkdir()
    (root / "reference.docx").write_bytes(b"DOCX")
    (root / "reference-blueprint.json").write_text('{"kind":"docx"}', encoding="utf-8")
    monkeypatch.setattr(media_server, "OFFICE_OUTPUT_ROOT", root.resolve())
    monkeypatch.setattr(media_server, "OFFICE_OUTPUT_PUBLIC_BASE_URL", "http://localhost:8130/office-output")

    suggestion = media_server._office_blueprint_suggestion()
    plan = media_server._office_blueprint_create_plan("reference.docx")
    blueprints = media_server._list_office_blueprints(limit=10)

    assert suggestion["phase"] == 6
    assert "If you like documents" in suggestion["message"]
    assert "blueprint" in suggestion["message"].lower()
    assert plan["operation"] == "blueprint_create"
    assert plan["blueprint"] == "/workspace/office-output/reference-blueprint.json"
    assert "officecli dump '/workspace/office-output/reference.docx'" in plan["commands"][0]
    assert [item["relative_path"] for item in blueprints] == ["reference-blueprint.json"]


def test_office_phase6_rejects_unsafe_paths() -> None:
    for builder in (media_server._office_preview_plan, media_server._office_repair_plan, media_server._office_blueprint_create_plan):
        try:
            builder("../secret.docx")
        except media_server.HTTPException as exc:
            assert getattr(exc, "status_code", None) == 400
        else:  # pragma: no cover - defensive assertion for stubbed environments
            raise AssertionError("unsafe office path was accepted")


def test_office_validation_result_persists_badge_in_run_state_manager(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "office-output"
    root.mkdir()
    doc = root / "demo.docx"
    doc.write_bytes(b"DOCX")
    saved: dict[str, dict] = {}

    def fake_save(namespace: str, workflow_id: str, record: dict):
        saved[f"{namespace}:{workflow_id}"] = dict(record)
        return {"saved": True, "record": dict(record)}

    def fake_load(namespace: str, workflow_id: str):
        record = saved.get(f"{namespace}:{workflow_id}")
        return dict(record) if record else None

    monkeypatch.setattr(media_server, "OFFICE_OUTPUT_ROOT", root.resolve())
    monkeypatch.setattr(media_server, "_office_state_save", fake_save)
    monkeypatch.setattr(media_server, "_office_state_load", fake_load)

    result = media_server._office_record_validation_result(
        file="demo.docx",
        status="warning",
        issues=[{"level": "warning", "message": "missing alt text"}],
        summary="1 warning",
    )
    record = media_server._office_output_record(doc.resolve())

    assert result["status"] == "warning"
    assert result["namespace"] == "office_validation"
    assert record["validation_status"] == "warning"
    assert record["validation_badge"] == "warning"
    assert record["validation_issues"] == [{"level": "warning", "message": "missing alt text"}]


def test_office_batch_job_creates_managed_progress_record(monkeypatch) -> None:
    saved: dict[str, dict] = {}

    def fake_save(namespace: str, workflow_id: str, record: dict):
        saved[f"{namespace}:{workflow_id}"] = dict(record)
        return {"saved": True, "record": dict(record)}

    def fake_load(namespace: str, workflow_id: str):
        record = saved.get(f"{namespace}:{workflow_id}")
        return dict(record) if record else None

    monkeypatch.setattr(media_server, "_office_state_save", fake_save)
    monkeypatch.setattr(media_server, "_office_state_load", fake_load)

    job = media_server._office_batch_job_plan(
        template="templates/invoice.docx",
        input_path="batch/customers.json",
        output_dir="batch/output",
        total=3,
    )
    loaded = media_server._office_batch_job_status(job["job_id"])
    updated = media_server._office_update_batch_job(
        job["job_id"],
        status="running",
        completed=1,
        failed=0,
        errors=[],
    )

    assert job["phase"] == 6
    assert job["operation"] == "batch_job"
    assert job["status"] == "planned"
    assert job["progress"] == {"total": 3, "completed": 0, "failed": 0, "percent": 0}
    assert "--output-dir '/workspace/office-output/batch/output'" in job["commands"][0]
    assert loaded["job_id"] == job["job_id"]
    assert updated["progress"]["completed"] == 1
    assert updated["progress"]["percent"] == 33


def test_office_template_placeholder_detection_and_merge_form_plan(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "office-output"
    template_dir = root / "templates"
    template_dir.mkdir(parents=True)
    template = template_dir / "invoice.docx"
    template.write_text("Hello {{customer_name}}, total {{ total }}. Again {{customer_name}}.", encoding="utf-8")
    monkeypatch.setattr(media_server, "OFFICE_OUTPUT_ROOT", root.resolve())

    placeholders = media_server._office_template_placeholders("templates/invoice.docx")
    plan = media_server._office_template_merge_form_plan(
        template="templates/invoice.docx",
        output="invoices/acme.docx",
        data={"customer_name": "ACME", "total": "42"},
    )

    assert placeholders["phase"] == 6
    assert placeholders["operation"] == "template_placeholders"
    assert placeholders["placeholders"] == ["customer_name", "total"]
    assert placeholders["fields"] == [
        {"name": "customer_name", "label": "customer_name", "required": True, "type": "text"},
        {"name": "total", "label": "total", "required": True, "type": "text"},
    ]
    assert plan["operation"] == "template_merge_form"
    assert plan["missing_fields"] == []
    assert "officecli merge '/workspace/office-output/templates/invoice.docx' '/workspace/office-output/invoices/acme.docx'" in plan["commands"][0]
    assert "customer_name" in plan["commands"][0]


def test_media_gallery_cors_allows_browser_ui_origins() -> None:
    middleware = [item for item in media_server.app.user_middleware if item.cls.__name__ == "CORSMiddleware"]

    assert middleware
    allow_origins = middleware[0].options["allow_origins"]
    assert "http://localhost:3000" in allow_origins
    assert "http://127.0.0.1:3000" in allow_origins


def test_media_gallery_cors_origins_are_env_configurable(monkeypatch) -> None:
    monkeypatch.setenv("ALPHARAVIS_MEDIA_CORS_ALLOW_ORIGINS", "https://ui.example, http://localhost:9999/")

    assert media_server._cors_allow_origins() == ["https://ui.example", "http://localhost:9999"]


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
        self.replacements: list[tuple[dict, dict, bool]] = []

    def find(self, query: dict):
        self.queries.append(query)
        cursor = _FakeCursor([dict(row) for row in self.rows])
        self.cursor = cursor
        return cursor

    def replace_one(self, query: dict, record: dict, upsert: bool = False) -> None:
        self.replacements.append((query, record, upsert))


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

    assert "<strong>chat-1</strong>" in html
    assert "2 Assets" in html
    assert html.index("image.png") < html.index("video.mp4")
    assert "name='group_by'" in html
    assert "name='sort'" in html
    assert "href='/favicon.svg'" in html
    assert "<div class='mark'>MG</div>" in html
    assert "@media(max-width:540px)" in html
    assert "class='thumb'" in html
    assert "source_key=" not in html
    assert "provider=" not in html


def test_gallery_defaults_to_date_sections_and_hides_group_ids(monkeypatch) -> None:
    collection = _FakeCollection(
        [
            {
                "_id": "image",
                "asset_id": "image",
                "title": "inferior-file-name.png",
                "media_type": "image",
                "asset_kind": "processed",
                "role": "output",
                "thread_key": "chat-1",
                "group_id": "private-group-id",
                "derivation_group_id": "private-group-id",
                "source_key": "inferior-source-key",
                "public_url": "http://localhost:8130/media/image.png",
                "created_at": 1_700_000_000,
            },
        ]
    )
    monkeypatch.setattr(media_server, "_collection", lambda: collection)

    html = asyncio.run(media_server.gallery())

    assert "14.11.2023" in html
    assert "private-group-id" not in html
    assert "inferior-source-key" not in html
    assert "inferior-file-name.png" not in html
    assert "Date sections" in html


def test_gallery_upload_form_is_browser_native(monkeypatch) -> None:
    collection = _FakeCollection([])
    monkeypatch.setattr(media_server, "_collection", lambda: collection)

    html = asyncio.run(media_server.gallery())

    assert "action='/assets/upload'" in html
    assert "enctype='multipart/form-data'" in html
    assert "type='file'" in html
    assert "accept='image/*,video/*,audio/*" in html


def test_store_uploaded_asset_writes_file_and_record(monkeypatch, tmp_path: Path) -> None:
    collection = _FakeCollection([])
    monkeypatch.setattr(media_server, "MEDIA_ROOT", tmp_path)
    monkeypatch.setattr(media_server, "_collection", lambda: collection)

    record = media_server._store_uploaded_asset(
        filename="photo.jpg",
        content_type="image/jpeg",
        content=b"JPEGDATA",
        title="My Upload",
    )

    stored = tmp_path / record["relative_path"]
    assert stored.read_bytes() == b"JPEGDATA"
    assert record["media_type"] == "image"
    assert record["asset_kind"] == "original"
    assert record["origin"] == "gallery_upload"
    assert record["public_url"].endswith(record["relative_path"].replace(os.sep, "/"))
    assert collection.replacements[0][0] == {"_id": record["asset_id"]}
    assert collection.replacements[0][2] is True


def test_internal_media_url_uses_container_reachable_base(monkeypatch) -> None:
    monkeypatch.setattr(media_server, "MEDIA_INTERNAL_BASE_URL", "http://media-gallery:8130")

    assert media_server._internal_media_url("2026-05-26/gallery upload/doc.odt") == (
        "http://media-gallery:8130/media/2026-05-26/gallery upload/doc.odt"
    )


def test_maybe_convert_odf_upload_registers_converted_asset(monkeypatch, tmp_path: Path) -> None:
    import odf_converter

    collection = _FakeCollection([])
    monkeypatch.setattr(media_server, "MEDIA_ROOT", tmp_path)
    monkeypatch.setattr(media_server, "MEDIA_INTERNAL_BASE_URL", "http://media-gallery:8130")
    monkeypatch.setattr(media_server, "_collection", lambda: collection)

    original = media_server._store_uploaded_asset(
        filename="sample.odt",
        content_type="application/vnd.oasis.opendocument.text",
        content=b"ODT",
        title="sample.odt",
    )
    seen: dict[str, str] = {}

    async def fake_convert(input_path: str, mime_type: str, output_dir: str, *, source_url: str):
        seen["input_path"] = input_path
        seen["mime_type"] = mime_type
        seen["source_url"] = source_url
        output_path = Path(output_dir) / "sample_converted.docx"
        output_path.write_bytes(b"DOCX")
        return {
            "output_path": str(output_path),
            "output_format": "docx",
            "output_mime": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "output_ext": ".docx",
            "output_size": 4,
        }

    monkeypatch.setattr(odf_converter, "convert_odf_to_ooxml", fake_convert)

    converted = asyncio.run(media_server._maybe_convert_odf_upload(original))

    assert converted is not None
    assert converted["mime_type"] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert converted["asset_kind"] == "converted"
    assert converted["origin"] == "onlyoffice_conversion"
    assert converted["parent_asset_id"] == original["asset_id"]
    assert converted["metadata"]["conversion_provider"] == "onlyoffice"
    assert converted["metadata"]["original_asset_id"] == original["asset_id"]
    assert seen["source_url"].startswith("http://media-gallery:8130/media/")
    assert len(collection.replacements) >= 3  # original, converted upload, converted metadata update


def test_maybe_convert_odf_upload_skips_non_odf(monkeypatch, tmp_path: Path) -> None:
    collection = _FakeCollection([])
    monkeypatch.setattr(media_server, "MEDIA_ROOT", tmp_path)
    monkeypatch.setattr(media_server, "_collection", lambda: collection)

    record = media_server._store_uploaded_asset(
        filename="image.png",
        content_type="image/png",
        content=b"PNG",
        title="image.png",
    )
    result = asyncio.run(media_server._maybe_convert_odf_upload(record))
    assert result is None


def test_parse_gallery_upload_multipart_extracts_file_and_fields() -> None:
    body = (
        b"--abc\r\n"
        b'Content-Disposition: form-data; name="title"\r\n\r\n'
        b"Nice file\r\n"
        b"--abc\r\n"
        b'Content-Disposition: form-data; name="file"; filename="clip.mp4"\r\n'
        b"Content-Type: video/mp4\r\n\r\n"
        b"MP4DATA\r\n"
        b"--abc--\r\n"
    )

    fields, uploaded = media_server._parse_gallery_upload_multipart("multipart/form-data; boundary=abc", body)

    assert fields == {"title": "Nice file"}
    assert uploaded["filename"] == "clip.mp4"
    assert uploaded["content_type"] == "video/mp4"
    assert uploaded["content"] == b"MP4DATA"
