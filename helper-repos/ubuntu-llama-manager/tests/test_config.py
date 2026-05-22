from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ubuntu_manager.config import Settings
from ubuntu_manager.llama_config import switch_context_in_command, switch_model_in_command


class SettingsTest(unittest.TestCase):
    def test_reboot_interval_hours_wins_over_legacy(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / "test.env"
            config.write_text(
                'REBOOT_INTERVAL_HOURS="3"\n'
                'REBOOT_AFTER_SECONDS="10000"\n',
                encoding="utf-8",
            )

            settings = Settings.load(config)

            self.assertEqual(settings.reboot_interval_seconds, 10800)

    def test_model_scan_dirs_are_colon_separated(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "test.env"
            config.write_text(f'MODEL_SCAN_DIRS="{root}:{root / "other"}"\n', encoding="utf-8")

            settings = Settings.load(config)

            self.assertEqual(settings.model_scan_dirs, [root, root / "other"])

    def test_switch_hf_model_keeps_other_flags(self) -> None:
        command = "./build/bin/llama-server -hf old/model --jinja -c 32768 --port 8033"

        updated = switch_model_in_command(command, "new/model:Q8_0", "hf")

        self.assertIn("-hf new/model:Q8_0", updated)
        self.assertIn("--jinja", updated)
        self.assertIn("-c 32768", updated)

    def test_switch_context_keeps_model_and_other_flags(self) -> None:
        command = "./build/bin/llama-server -hf model/name --port 8001 -c 8192 -ngl 99"

        updated = switch_context_in_command(command, 16384)

        self.assertIn("-hf model/name", updated)
        self.assertIn("-c 16384", updated)
        self.assertIn("--port 8001", updated)

    def test_switch_context_adds_flag_when_missing(self) -> None:
        command = "./build/bin/llama-server -hf model/name --port 8001"

        updated = switch_context_in_command(command, "4096")

        self.assertIn("-c 4096", updated)


if __name__ == "__main__":
    unittest.main()
