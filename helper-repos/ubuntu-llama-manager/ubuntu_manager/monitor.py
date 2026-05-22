from __future__ import annotations

import argparse
import json
import time

from .config import Settings
from .gpu import diagnose_gpu
from .recovery import handle_gpu_fault
from .services import Manager, now_iso


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor ROCm/GPU health and trigger configured recovery.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    settings = Settings.load(args.config)
    manager = Manager(settings)
    poll_seconds = max(5, settings.int("GPU_HEALTH_POLL_SECONDS", 30))

    while True:
        diagnostics = diagnose_gpu(settings)
        snapshot = {
            "time": now_iso(),
            "critical": diagnostics["critical"],
            "decision": diagnostics.get("decision", "unknown"),
            "pcie_replay_counts": diagnostics.get("pcie_replay_counts", []),
            "command_failures": diagnostics["command_failures"],
            "matches": diagnostics["matches"][:25],
        }
        manager.write_json_state("last-gpu-decision.json", snapshot)
        manager.write_json_state("gpu-health.json", snapshot)
        print(json.dumps(snapshot, sort_keys=True), flush=True)

        if diagnostics["critical"]:
            result = handle_gpu_fault(settings, manager, diagnostics)
            print(json.dumps({"recovery": result}, sort_keys=True), flush=True)
            if settings.get("GPU_HEALTH_CRITICAL_ACTION", "none").lower() not in {"none", "log"}:
                return

        if args.once:
            return
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
