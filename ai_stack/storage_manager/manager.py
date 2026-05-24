"""Storage Manager — orchestrates scan, budget check, and cleanup.

Two main entry points:
  get_storage_status()  — scan + budget comparison (read-only, no deletions)
  run_storage_cleanup() — scan + clean any service over budget
"""

import time
import logging
from dataclasses import dataclass, field

from ai_stack.storage_manager.config import StorageConfig, get_storage_config
from ai_stack.storage_manager.budget import StorageBudget, ServiceBudget
from ai_stack.storage_manager.scanner import scan_all_services, ServiceUsage
from ai_stack.storage_manager.cleaner import clean_service, CleanupResult

logger = logging.getLogger(__name__)


@dataclass
class StorageStatus:
    """Complete snapshot of storage state."""

    config: StorageConfig
    budget: StorageBudget
    usage: dict[str, ServiceUsage]
    timestamp: float = field(default_factory=time.time)
    total_used_bytes: int = 0
    total_budget_bytes: int = 0
    warnings: list[str] = field(default_factory=list)
    critical: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.total_used_bytes = sum(u.bytes_used for u in self.usage.values())
        self.total_budget_bytes = self.config.max_total_bytes
        for svc_name, svc_budget in self.budget.services.items():
            used = self.usage.get(svc_name, ServiceUsage(svc_name, 0)).bytes_used
            if used > svc_budget.total_bytes:
                self.critical.append(svc_name)
            elif used > svc_budget.warn_bytes:
                self.warnings.append(svc_name)

    def total_used_gb(self) -> float:
        return self.total_used_bytes / (1024 ** 3)

    def usage_pct(self) -> float:
        if self.total_budget_bytes == 0:
            return 0.0
        return (self.total_used_bytes / self.total_budget_bytes) * 100.0

    def format_table(self) -> str:
        """Formatted table for the agent/user."""
        lines = [
            f"Storage Status  ({self.usage_pct():.1f}% of {self.total_budget_bytes / (1024**3):.1f} GB used)",
            f"{'Service':<20s} {'%Budget':>7s} {'Used MB':>10s} {'Budget MB':>10s} {'Status':>10s}",
            "-" * 65,
        ]
        for svc_name in sorted(self.usage.keys()):
            usage = self.usage[svc_name]
            svc_budget = self.budget.get(svc_name)
            if svc_budget is None:
                budget_mb = 0
                pct = 0.0
                status = "—"
            else:
                budget_mb = svc_budget.total_bytes / (1024 * 1024)
                pct = svc_budget.percentage
                if svc_name in self.critical:
                    status = "CRITICAL"
                elif svc_name in self.warnings:
                    status = "WARN"
                else:
                    status = "ok"

            used_mb = usage.bytes_used / (1024 * 1024)
            lines.append(
                f"{svc_name:<20s} {pct:6.1f}% {used_mb:10.1f} {budget_mb:10.1f} {status:>10s}"
            )

        if self.warnings:
            lines.append(f"\n  Warnings ({len(self.warnings)}): {', '.join(self.warnings)}")
        if self.critical:
            lines.append(f"\n  Critical ({len(self.critical)}): {', '.join(self.critical)} — auto-cleanup will run")

        return "\n".join(lines)


@dataclass
class CleanupReport:
    """Report after running cleanup."""

    status: StorageStatus
    results: list[CleanupResult] = field(default_factory=list)
    total_freed_bytes: int = 0

    def format_report(self) -> str:
        if not self.results:
            return "No cleanup needed — all services within budget."

        lines = ["Cleanup Report:"]
        for r in self.results:
            if r.bytes_freed > 0 or r.files_deleted > 0 or r.mongo_docs_deleted > 0:
                parts = []
                if r.bytes_freed > 0:
                    parts.append(f"{r.bytes_freed / (1024**2):.1f} MB freed")
                if r.files_deleted > 0:
                    parts.append(f"{r.files_deleted} files deleted")
                if r.mongo_docs_deleted > 0:
                    parts.append(f"{r.mongo_docs_deleted} MongoDB docs deleted")
                if r.dry_run:
                    parts.append("[DRY RUN]")
                lines.append(f"  {r.service_name}: {', '.join(parts)}")
            if r.error:
                lines.append(f"  {r.service_name}: ERROR — {r.error}")
        lines.append(f"\nTotal freed: {self.total_freed_bytes / (1024**2):.1f} MB")
        return "\n".join(lines)


def get_storage_status() -> StorageStatus:
    """Scan disk usage and compare against budgets. Read-only — no deletions."""
    config = get_storage_config()
    if not config.enabled:
        return StorageStatus(
            config=config,
            budget=StorageBudget(config=config),
            usage={},
            warnings=["Storage manager is DISABLED. Set ALPHARAVIS_STORAGE_MANAGER_ENABLED=true"],
        )

    budget = StorageBudget(config=config)
    usage = scan_all_services(config.data_root, config.mongo_uri, config.pg_uri)
    return StorageStatus(config=config, budget=budget, usage=usage)


def run_storage_cleanup(force: bool = False, service_filter: str = "") -> CleanupReport:
    """Scan and clean services that exceed their budget.

    Args:
        force: If True, clean ALL services regardless of budget.
        service_filter: If non-empty, only clean this specific service.

    Returns a CleanupReport with per-service results.
    """
    config = get_storage_config()
    if not config.enabled:
        return CleanupReport(
            status=StorageStatus(
                config=config,
                budget=StorageBudget(config=config),
                usage={},
            ),
            results=[CleanupResult(service_name="storage_manager", bytes_freed=0,
                                    files_deleted=0, mongo_docs_deleted=0,
                                    error="Storage manager is DISABLED. Set ALPHARAVIS_STORAGE_MANAGER_ENABLED=true")],
        )

    budget = StorageBudget(config=config)
    usage = scan_all_services(config.data_root, config.mongo_uri, config.pg_uri)
    status = StorageStatus(config=config, budget=budget, usage=usage)

    results: list[CleanupResult] = []
    total_freed = 0

    services_to_clean = list(usage.keys()) if force else status.critical.copy()
    if service_filter:
        services_to_clean = [service_filter]

    for svc_name in services_to_clean:
        svc_usage = usage.get(svc_name)
        if svc_usage is None:
            continue
        svc_budget = budget.get(svc_name)
        if svc_budget is None:
            continue
        budget_bytes = svc_budget.total_bytes
        result = clean_service(svc_name, svc_usage, budget_bytes, config)
        results.append(result)
        total_freed += result.bytes_freed

    return CleanupReport(status=status, results=results, total_freed_bytes=total_freed)


# ── singleton for LangGraph tool use ──
_storage_manager: "StorageManager | None" = None


class StorageManager:
    """Lightweight wrapper for LangGraph tool integration."""

    def status(self) -> str:
        return get_storage_status().format_table()

    def cleanup(self, force: bool = False, service: str = "") -> str:
        report = run_storage_cleanup(force=force, service_filter=service)
        return report.format_report()

    def budget_summary(self) -> str:
        config = get_storage_config()
        budget = StorageBudget(config=config)
        return budget.summary()


def get_storage_manager() -> StorageManager:
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = StorageManager()
    return _storage_manager
