import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ubuntu_manager.config import Settings
from ubuntu_manager.recovery import handle_gpu_fault, parse_llama_probe_content


class FakeManager:
    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir

    def write_json_state(self, name: str, payload: dict) -> Path:
        path = self.state_dir / name
        path.write_text("{}", encoding="utf-8")
        return path


class RecoveryProbeTest(unittest.TestCase):
    def test_parse_llama_cpp_completion_content(self) -> None:
        self.assertEqual(parse_llama_probe_content({"content": " ok"}), " ok")

    def test_parse_response_field(self) -> None:
        self.assertEqual(parse_llama_probe_content({"response": "ok"}), "ok")

    def test_parse_openai_text_choices(self) -> None:
        self.assertEqual(parse_llama_probe_content({"choices": [{"text": "o"}, {"text": "k"}]}), "ok")

    def test_parse_openai_chat_choices(self) -> None:
        self.assertEqual(parse_llama_probe_content({"choices": [{"message": {"content": "ok"}}]}), "ok")

    def test_missing_content_is_empty(self) -> None:
        self.assertEqual(parse_llama_probe_content({"choices": [{}]}), "")

    def test_gpu_fault_refuses_shutdown_without_esp_confirmation(self) -> None:
        with TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            settings = Settings(
                config_path=state_dir / "ubuntu-llama.conf",
                project_root=state_dir,
                raw={
                    "UBUNTU_STATE_DIR": str(state_dir),
                    "GPU_HEALTH_CRITICAL_ACTION": "shutdown",
                    "GPU_FAULT_REQUIRE_ESP_WEBHOOK": "true",
                    "ESP_WEBHOOK_URL": "",
                    "ESP_POWER_ACTION_ON_GPU_FAULT": "power-cycle",
                },
            )
            diagnostics = {"critical": True, "matches": [], "command_failures": []}
            with patch("ubuntu_manager.recovery.run_command") as run_command:
                result = handle_gpu_fault(settings, FakeManager(state_dir), diagnostics)
            self.assertFalse(result["ok"])
            self.assertFalse(result["action"]["executed"])
            run_command.assert_not_called()

    def test_gpu_fault_uses_configured_shutdown_command_after_esp_confirmation(self) -> None:
        class Response:
            status = 202

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return b'{"ok":true}'

        with TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            settings = Settings(
                config_path=state_dir / "ubuntu-llama.conf",
                project_root=state_dir,
                raw={
                    "UBUNTU_STATE_DIR": str(state_dir),
                    "GPU_HEALTH_CRITICAL_ACTION": "shutdown",
                    "GPU_FAULT_REQUIRE_ESP_WEBHOOK": "true",
                    "GPU_FAULT_LOCAL_SHUTDOWN_AFTER_ESP": "true",
                    "GPU_FAULT_SHUTDOWN_COMMAND": "/bin/echo poweroff",
                    "ESP_WEBHOOK_URL": "http://esp/action",
                    "ESP_POWER_ACTION_ON_GPU_FAULT": "power-cycle",
                    "ESP_NOTIFY_SETTLE_SECONDS": "0",
                },
            )
            diagnostics = {"critical": True, "matches": [], "command_failures": []}
            with patch("ubuntu_manager.recovery.urllib.request.urlopen", return_value=Response()):
                with patch("ubuntu_manager.recovery.run_command", return_value={"ok": True}) as run_command:
                    result = handle_gpu_fault(settings, FakeManager(state_dir), diagnostics)
            self.assertTrue(result["ok"])
            self.assertTrue(result["esp"]["webhook"]["ok"])
            run_command.assert_called_once_with(["/bin/echo", "poweroff"], timeout=5)


if __name__ == "__main__":
    unittest.main()
