"""Storage budget calculations."""

from dataclasses import dataclass, field
from typing import Optional

from ai_stack.storage_manager.config import StorageConfig, get_storage_config


@dataclass
class ServiceBudget:
    """Budget for a single data service."""

    service_name: str
    total_bytes: int                # hard cap in bytes
    warn_bytes: int                 # warn threshold in bytes (90% of cap by default)
    percentage: float               # percentage of global cap
    warning: bool = False           # True when current usage exceeds warn threshold
    critical: bool = False          # True when current usage exceeds hard cap


@dataclass
class StorageBudget:
    """Global storage budget with per-service breakdown."""

    config: StorageConfig = field(default_factory=get_storage_config)
    services: dict[str, ServiceBudget] = field(default_factory=dict)
    total_global_bytes: int = 0
    total_budget_bytes: int = 0

    def __post_init__(self):
        self.total_budget_bytes = self.config.max_total_bytes
        self.services = {}
        for svc, pct in self.config.service_percentages.items():
            budget = self.config.budget_bytes(svc)
            self.services[svc] = ServiceBudget(
                service_name=svc,
                total_bytes=budget,
                warn_bytes=self.config.warn_bytes(svc),
                percentage=pct,
            )

    def get(self, service: str) -> Optional[ServiceBudget]:
        return self.services.get(service)

    def total_budget_gb(self) -> float:
        return self.total_budget_bytes / (1024 ** 3)

    def summary(self) -> str:
        """Human-readable budget summary."""
        lines = [
            f"Storage Budget: {self.total_budget_gb():.1f} GB total cap",
            f"  warn threshold: {self.config.warn_threshold_pct:.0f}%",
            f"  dry_run: {self.config.dry_run}",
            "",
        ]
        for svc in sorted(self.services.values(), key=lambda s: s.total_bytes, reverse=True):
            pct = svc.percentage
            gb = svc.total_bytes / (1024 ** 3)
            warn_gb = svc.warn_bytes / (1024 ** 3)
            lines.append(
                f"  {svc.service_name:20s}  {pct:5.1f}%  = {gb:6.2f} GB  (warn at {warn_gb:5.2f} GB)"
            )
        return "\n".join(lines)
