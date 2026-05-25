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
    assert "http://localhost:8130/office/files" in content
    assert "fetchOutputFiles" in content
    assert "useEffect" in content
    assert "outputFiles" in content
    assert "formatSize" in content
    assert "formatDate" in content
    assert "OfficeOutputFile" in content
    assert "download_url" in content
    assert "Refresh" in content
    assert "Download" in content
    assert "handleScreenshot" in content
    assert "Camera" in content
    assert "Screenshot" in content
    assert "officecli view" in content
    assert "screenshot -o" in content


def test_home_page_wires_chat_and_office_tabs():
    content = read("src/app/page.tsx")

    assert "OfficePanel" in content
    assert 'activeView, setActiveView' in content
    assert 'setActiveView("chat")' in content
    assert 'setActiveView("office")' in content
    assert "Office" in content
