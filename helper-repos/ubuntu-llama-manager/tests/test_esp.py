import unittest

from ubuntu_manager.esp import sibling_url_from_webhook, status_url_from_webhook


class EspStatusTest(unittest.TestCase):
    def test_status_url_from_action_webhook(self) -> None:
        self.assertEqual(
            status_url_from_webhook("http://192.168.178.113/action"),
            "http://192.168.178.113/status",
        )

    def test_status_url_keeps_prefix(self) -> None:
        self.assertEqual(
            status_url_from_webhook("http://192.168.178.113/api/action"),
            "http://192.168.178.113/api/status",
        )

    def test_status_url_rejects_empty_or_relative_values(self) -> None:
        self.assertEqual(status_url_from_webhook(""), "")
        self.assertEqual(status_url_from_webhook("/action"), "")

    def test_sibling_url_from_action_webhook(self) -> None:
        self.assertEqual(
            sibling_url_from_webhook("http://192.168.178.113/action", "cancel"),
            "http://192.168.178.113/cancel",
        )


if __name__ == "__main__":
    unittest.main()
