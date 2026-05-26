from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = ROOT / "submodules" / "deep-agents-ui"


def read(path: str) -> str:
    return (UI_ROOT / path).read_text(encoding="utf-8")


def test_file_validation_accepts_office_openxml_mime_types():
    content = read("src/lib/file-validation.ts")

    assert "OFFICE_FILE_TYPES" in content
    assert "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in content
    assert "application/vnd.openxmlformats-officedocument.presentationml.presentation" in content
    assert "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in content
    assert "DOCX, PPTX, XLSX" in content


def test_multimodal_utils_preserves_office_files_as_file_blocks():
    content = read("src/lib/multimodal-utils.ts")

    assert "supportedOfficeTypes" in content
    assert "metadata: { filename: file.name }" in content
    assert "application/vnd.openxmlformats-officedocument" in content


def test_office_panel_exists_and_sends_agent_officecli_prompt():
    panel_path = UI_ROOT / "src" / "app" / "components" / "OfficePanel.tsx"

    assert panel_path.is_file()
    content = panel_path.read_text(encoding="utf-8")
    assert content.startswith('"use client";')
    assert "export const OfficePanel" in content
    assert "useChatContext" in content
    assert "sendMessage" in content
    assert "OfficeCLI" in content
    assert "/workspace/office-output" in content
    assert "NEXT_PUBLIC_OFFICE_OUTPUT_FILES_URL" in content
    assert "NEXT_PUBLIC_OFFICE_OUTPUT_UPLOAD_URL" in content
    assert "NEXT_PUBLIC_OFFICE_TEMPLATES_URL" in content
    assert "NEXT_PUBLIC_OFFICE_TEMPLATE_MERGE_URL" in content
    assert "NEXT_PUBLIC_OFFICE_VALIDATE_URL" in content
    assert "NEXT_PUBLIC_OFFICE_BATCH_URL" in content
    assert "NEXT_PUBLIC_OFFICE_ROUNDTRIP_URL" in content
    assert "http://localhost:8130/office/files" in content
    assert "http://localhost:8130/office/template-merge" in content
    assert "http://localhost:8130/office/validate" in content
    assert "http://localhost:8130/office/batch" in content
    assert "http://localhost:8130/office/roundtrip" in content
    assert "fetchOutputFiles" in content
    assert "fetchTemplates" in content
    assert "handleUpload" in content
    assert "type=\"file\"" in content
    assert "uploadAccept" in content
    assert ".docx,.pptx,.xlsx" in content
    assert "officeOutputPath" in content
    assert "relative_path" in content
    assert "useEffect" in content
    assert "outputFiles" in content
    assert "formatSize" in content
    assert "formatDate" in content
    assert "OfficeOutputFile" in content
    assert "download_url" in content
    assert "preview_available" in content
    assert "preview_image_url" in content
    assert "preview_html_url" in content
    assert "Preview ready" in content
    assert "Preview PNG" in content
    assert "Preview HTML" in content
    assert "Refresh" in content
    assert "Download" in content
    assert "handleScreenshot" in content
    assert "Camera" in content
    assert "Screenshot" in content
    assert "officecli view" in content
    assert "screenshot -o" in content
    assert "handleWatch" in content
    assert "handleTemplateMerge" in content
    assert "handleBatch" in content
    assert "handleValidate" in content
    assert "handleRoundTrip" in content
    assert "OfficePhase5Plan" in content
    assert "fetchOfficePlan" in content
    assert "planCommands" in content
    assert "operation" in content
    assert "phase" in content
    assert "Template Gallery" in content
    assert "officecli merge" in content
    assert "officecli validate" in content
    assert "officecli dump" in content
    assert "officecli batch" in content
    assert "PREVIEW_PORT" in content
    assert "previewPort" in content
    assert "nohup officecli watch" in content
    assert "officecli unwatch" in content
    assert "background=true" not in content
    assert "process(action=" not in content
    assert "handleStopWatch" in content
    assert "MonitorPlay" in content
    assert "MonitorStop" in content
    assert "watchFile" in content
    assert "Live Preview:" in content
    assert "officecli watch" in content
    assert "iframe" in content
    assert "sandbox" in content
    assert "useRef" in content
    assert "prevLoading" in content
    assert "Auto-refresh" in content
    assert "NEXT_PUBLIC_OFFICE_PREVIEW_GENERATE_URL" in content
    assert "NEXT_PUBLIC_OFFICE_REPAIR_URL" in content
    assert "NEXT_PUBLIC_OFFICE_WATCH_START_URL" in content
    assert "NEXT_PUBLIC_OFFICE_WATCH_STOP_URL" in content
    assert "NEXT_PUBLIC_OFFICE_BLUEPRINTS_URL" in content
    assert "NEXT_PUBLIC_OFFICE_BLUEPRINT_CREATE_URL" in content
    assert "NEXT_PUBLIC_OFFICE_BLUEPRINT_SUGGEST_URL" in content
    assert "OfficePhase6Plan" in content
    assert "fetchOfficeWorkflowPlan" in content
    assert "handleGeneratePreview" in content
    assert "handleRepair" in content
    assert "handleBlueprintCreate" in content
    assert "blueprintHint" in content
    assert "If you like documents" in content
    assert "Repair" in content
    assert "Generate preview" in content
    assert "Make blueprint" in content
    assert "Preview frame" in content
    assert "NEXT_PUBLIC_OFFICE_VALIDATION_RESULTS_URL" in content
    assert "NEXT_PUBLIC_OFFICE_BATCH_JOBS_URL" in content
    assert "NEXT_PUBLIC_OFFICE_BATCH_STATUS_URL" in content
    assert "NEXT_PUBLIC_OFFICE_TEMPLATE_PLACEHOLDERS_URL" in content
    assert "NEXT_PUBLIC_OFFICE_TEMPLATE_MERGE_FORM_URL" in content
    assert "fetchValidationResults" in content
    assert "validationBadge" in content
    assert "Validation issues" in content
    assert "Batch progress" in content
    assert "handleManagedBatch" in content
    assert "fetchTemplatePlaceholders" in content
    assert "templatePlaceholders" in content
    assert "Template merge form" in content
    assert "placeholder" in content


def test_home_page_wires_chat_and_office_tabs():
    content = read("src/app/page.tsx")

    assert "OfficePanel" in content
    assert 'activeView, setActiveView' in content
    assert 'setActiveView("chat")' in content
    assert 'setActiveView("office")' in content
    assert "Office" in content


def test_media_gallery_exposes_host_ownership_env_for_office_uploads():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    env_example = (ROOT / ".env(exaple)").read_text(encoding="utf-8")

    assert "ALPHARAVIS_OFFICE_OUTPUT_HOST_UID=${ALPHARAVIS_OFFICE_OUTPUT_HOST_UID:-1000}" in compose
    assert "ALPHARAVIS_OFFICE_OUTPUT_HOST_GID=${ALPHARAVIS_OFFICE_OUTPUT_HOST_GID:-1000}" in compose
    assert "NEXT_PUBLIC_OFFICE_TEMPLATE_MERGE_URL=${NEXT_PUBLIC_OFFICE_TEMPLATE_MERGE_URL:-http://localhost:8130/office/template-merge}" in compose
    assert "NEXT_PUBLIC_OFFICE_VALIDATE_URL=${NEXT_PUBLIC_OFFICE_VALIDATE_URL:-http://localhost:8130/office/validate}" in compose
    assert "NEXT_PUBLIC_OFFICE_BATCH_URL=${NEXT_PUBLIC_OFFICE_BATCH_URL:-http://localhost:8130/office/batch}" in compose
    assert "NEXT_PUBLIC_OFFICE_ROUNDTRIP_URL=${NEXT_PUBLIC_OFFICE_ROUNDTRIP_URL:-http://localhost:8130/office/roundtrip}" in compose
    assert "NEXT_PUBLIC_OFFICE_PREVIEW_GENERATE_URL=${NEXT_PUBLIC_OFFICE_PREVIEW_GENERATE_URL:-http://localhost:8130/office/preview}" in compose
    assert "NEXT_PUBLIC_OFFICE_REPAIR_URL=${NEXT_PUBLIC_OFFICE_REPAIR_URL:-http://localhost:8130/office/repair}" in compose
    assert "NEXT_PUBLIC_OFFICE_WATCH_START_URL=${NEXT_PUBLIC_OFFICE_WATCH_START_URL:-http://localhost:8130/office/watch/start}" in compose
    assert "NEXT_PUBLIC_OFFICE_WATCH_STOP_URL=${NEXT_PUBLIC_OFFICE_WATCH_STOP_URL:-http://localhost:8130/office/watch/stop}" in compose
    assert "NEXT_PUBLIC_OFFICE_BLUEPRINTS_URL=${NEXT_PUBLIC_OFFICE_BLUEPRINTS_URL:-http://localhost:8130/office/blueprints}" in compose
    assert "NEXT_PUBLIC_OFFICE_BLUEPRINT_CREATE_URL=${NEXT_PUBLIC_OFFICE_BLUEPRINT_CREATE_URL:-http://localhost:8130/office/blueprints/create}" in compose
    assert "NEXT_PUBLIC_OFFICE_BLUEPRINT_SUGGEST_URL=${NEXT_PUBLIC_OFFICE_BLUEPRINT_SUGGEST_URL:-http://localhost:8130/office/blueprints/suggest}" in compose
    assert "NEXT_PUBLIC_OFFICE_VALIDATION_RESULTS_URL=${NEXT_PUBLIC_OFFICE_VALIDATION_RESULTS_URL:-http://localhost:8130/office/validation-results}" in compose
    assert "NEXT_PUBLIC_OFFICE_BATCH_JOBS_URL=${NEXT_PUBLIC_OFFICE_BATCH_JOBS_URL:-http://localhost:8130/office/batch/jobs}" in compose
    assert "NEXT_PUBLIC_OFFICE_BATCH_STATUS_URL=${NEXT_PUBLIC_OFFICE_BATCH_STATUS_URL:-http://localhost:8130/office/batch/jobs}" in compose
    assert "NEXT_PUBLIC_OFFICE_TEMPLATE_PLACEHOLDERS_URL=${NEXT_PUBLIC_OFFICE_TEMPLATE_PLACEHOLDERS_URL:-http://localhost:8130/office/templates/placeholders}" in compose
    assert "NEXT_PUBLIC_OFFICE_TEMPLATE_MERGE_FORM_URL=${NEXT_PUBLIC_OFFICE_TEMPLATE_MERGE_FORM_URL:-http://localhost:8130/office/templates/merge-form}" in compose
    assert "ALPHARAVIS_OFFICE_OUTPUT_HOST_UID=1000" in env_example
    assert "ALPHARAVIS_OFFICE_OUTPUT_HOST_GID=1000" in env_example
    assert "NEXT_PUBLIC_OFFICE_TEMPLATE_MERGE_URL=http://localhost:8130/office/template-merge" in env_example
    assert "NEXT_PUBLIC_OFFICE_VALIDATE_URL=http://localhost:8130/office/validate" in env_example
    assert "NEXT_PUBLIC_OFFICE_BATCH_URL=http://localhost:8130/office/batch" in env_example
    assert "NEXT_PUBLIC_OFFICE_ROUNDTRIP_URL=http://localhost:8130/office/roundtrip" in env_example
    assert "NEXT_PUBLIC_OFFICE_PREVIEW_GENERATE_URL=http://localhost:8130/office/preview" in env_example
    assert "NEXT_PUBLIC_OFFICE_REPAIR_URL=http://localhost:8130/office/repair" in env_example
    assert "NEXT_PUBLIC_OFFICE_WATCH_START_URL=http://localhost:8130/office/watch/start" in env_example
    assert "NEXT_PUBLIC_OFFICE_WATCH_STOP_URL=http://localhost:8130/office/watch/stop" in env_example
    assert "NEXT_PUBLIC_OFFICE_BLUEPRINTS_URL=http://localhost:8130/office/blueprints" in env_example
    assert "NEXT_PUBLIC_OFFICE_BLUEPRINT_CREATE_URL=http://localhost:8130/office/blueprints/create" in env_example
    assert "NEXT_PUBLIC_OFFICE_BLUEPRINT_SUGGEST_URL=http://localhost:8130/office/blueprints/suggest" in env_example
    assert "NEXT_PUBLIC_OFFICE_VALIDATION_RESULTS_URL=http://localhost:8130/office/validation-results" in env_example
    assert "NEXT_PUBLIC_OFFICE_BATCH_JOBS_URL=http://localhost:8130/office/batch/jobs" in env_example
    assert "NEXT_PUBLIC_OFFICE_BATCH_STATUS_URL=http://localhost:8130/office/batch/jobs" in env_example
    assert "NEXT_PUBLIC_OFFICE_TEMPLATE_PLACEHOLDERS_URL=http://localhost:8130/office/templates/placeholders" in env_example
    assert "NEXT_PUBLIC_OFFICE_TEMPLATE_MERGE_FORM_URL=http://localhost:8130/office/templates/merge-form" in env_example
