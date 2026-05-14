from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tailscale_https_routes as routes  # noqa: E402


def test_plan_includes_service_dashboard_by_default(capsys):
    assert routes.main(["plan", "--tailscale-host", "node.tailnet.ts.net"]) == 0

    output = capsys.readouterr().out

    assert "AlphaRavis Dashboard: http://localhost:8090 -> https://node.tailnet.ts.net:8090" in output


def test_plan_can_exclude_service_dashboard(capsys):
    assert routes.main(["plan", "--tailscale-host", "node.tailnet.ts.net", "--exclude-dashboard"]) == 0

    output = capsys.readouterr().out

    assert "AlphaRavis Dashboard" not in output


def test_disable_without_tailscale_binary_removes_overrides(tmp_path: Path, monkeypatch):
    output = tmp_path / "tailscale_service_urls.json"
    output.write_text('{"routes":[]}\n', encoding="utf-8")
    monkeypatch.setattr(routes.shutil, "which", lambda name: None)

    assert routes.main(["disable", "--output", str(output)]) == 0

    assert not output.exists()


def test_auto_sudo_retries_after_permission_error(monkeypatch):
    calls = []

    def fake_run(cmd, *, check=True, capture=True, sudo_password=None):
        calls.append((cmd, sudo_password))
        if sudo_password is None:
            return subprocess.CompletedProcess(cmd, 1, "", "permission denied")
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    monkeypatch.setattr(routes, "run", fake_run)
    monkeypatch.setattr(routes.getpass, "getpass", lambda prompt: "secret")

    result = routes.TailscaleRunner("auto").run(["tailscale", "serve", "status"])

    assert result.returncode == 0
    assert calls == [
        (["tailscale", "serve", "status"], None),
        (["tailscale", "serve", "status"], "secret"),
    ]


def test_auto_sudo_does_not_retry_non_permission_error(monkeypatch):
    def fake_run(cmd, *, check=True, capture=True, sudo_password=None):
        return subprocess.CompletedProcess(cmd, 2, "", "unknown flag")

    monkeypatch.setattr(routes, "run", fake_run)

    try:
        routes.TailscaleRunner("auto").run(["tailscale", "serve", "status"])
    except subprocess.CalledProcessError as exc:
        assert exc.returncode == 2
        assert exc.stderr == "unknown flag"
    else:
        raise AssertionError("expected CalledProcessError")
