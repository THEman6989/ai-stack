"""Storage Manager configuration — all values from env vars, no hardcoding.

Percentage-based allocation: each data service gets X% of ALPHARAVIS_STORAGE_MAX_TOTAL_GB.

Services and their data roots (relative to project root / workspace):
  librechat      — ./librechat-data/       (MongoDB chat history)
  media_gallery  — ./media-data/            (Media Gallery files)
  hermes         — ./hermes-data/           (Hermes sessions/skills/audio)
  pixelle        — ./pixelle-data/          (Pixelle generated images)
  openwebui      — ./openwebui-data/        (OpenWebUI chat history)
  logs           — ./logs/                  (All service logs)
"""

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default


# ── default percentages (must sum to 100) ──────────────────────────────
_DEFAULT_PERCENTAGES: dict[str, float] = {
    "librechat": 10.0,        # MongoDB — chat history
    "media_gallery": 40.0,    # Media files — user's priority
    "hermes": 5.0,            # Hermes agent data
    "vectordb": 30.0,         # Archives / pgvector
    "pixelle": 5.0,           # Generated images
    "openwebui": 5.0,         # OpenWebUI chat history
    "logs": 3.0,              # Log files
    "other": 2.0,             # Everything else not explicitly tracked
}


@dataclass
class StorageConfig:
    """Immutable config snapshot. Always build fresh via get_storage_config()."""

    enabled: bool
    max_total_bytes: int          # hard cap in bytes
    warn_threshold_pct: float     # warn when usage > this % of budget (e.g. 90.0)
    dry_run: bool                 # if True, only report — never delete
    service_percentages: dict[str, float]  # service_name → percentage
    data_root: str                # absolute path to project root (for data dirs)
    mongo_uri: str                # MongoDB connection string for chat cleanup
    pg_uri: str                   # PostgreSQL connection string for vectordb cleanup

    def budget_bytes(self, service: str) -> int:
        """Budget in bytes for a service, based on its percentage of total."""
        pct = self.service_percentages.get(service, self.service_percentages.get("other", 10.0))
        return int(self.max_total_bytes * pct / 100.0)

    def warn_bytes(self, service: str) -> int:
        """Warn threshold in bytes for a service."""
        return int(self.budget_bytes(service) * self.warn_threshold_pct / 100.0)


def get_storage_config() -> StorageConfig:
    """Build storage config from environment variables.

    Feature flag: ALPHARAVIS_STORAGE_MANAGER_ENABLED (default: false)
    Total cap:    ALPHARAVIS_STORAGE_MAX_TOTAL_GB  (default: 50)
    Warn at:      ALPHARAVIS_STORAGE_WARN_THRESHOLD_PCT (default: 90)
    Dry run:      ALPHARAVIS_STORAGE_DRY_RUN (default: false)
    Percentages:  ALPHARAVIS_STORAGE_{SERVICE}_PCT (e.g. ALPHARAVIS_STORAGE_MEDIA_GALLERY_PCT=40)
    Data root:    ALPHARAVIS_STORAGE_DATA_ROOT or current working directory
    MongoDB URI:  MONGODB_URI or mongodb://mongodb:27017
    """
    enabled = _env_bool("ALPHARAVIS_STORAGE_MANAGER_ENABLED", "false")

    max_total_gb = max(1, _env_int("ALPHARAVIS_STORAGE_MAX_TOTAL_GB", 50))
    max_total_bytes = max_total_gb * 1024 * 1024 * 1024

    warn_threshold_pct = max(0.0, min(100.0, _env_float("ALPHARAVIS_STORAGE_WARN_THRESHOLD_PCT", 90.0)))

    dry_run = _env_bool("ALPHARAVIS_STORAGE_DRY_RUN", "false")

    # build per-service percentages from env vars, falling back to defaults
    percentages: dict[str, float] = {}
    env_prefix = "ALPHARAVIS_STORAGE_"
    env_suffix = "_PCT"
    for service, default_pct in _DEFAULT_PERCENTAGES.items():
        env_name = f"ALPHARAVIS_STORAGE_{service.upper()}_PCT"
        percentages[service] = _env_float(env_name, default_pct)

    # normalize so they sum to 100 (if user overrides don't sum perfectly)
    total_pct = sum(percentages.values())
    if total_pct > 0 and abs(total_pct - 100.0) > 0.01:
        scale = 100.0 / total_pct
        percentages = {k: round(v * scale, 2) for k, v in percentages.items()}

    data_root = os.getenv("ALPHARAVIS_STORAGE_DATA_ROOT", "/workspace")
    if not os.path.isabs(data_root):
        data_root = os.path.abspath(data_root)

    mongo_uri = os.getenv("MONGODB_URI", "mongodb://mongodb:27017")

    pg_uri = os.getenv(
        "ALPHARAVIS_PGVECTOR_DATABASE_URL",
        os.getenv("ALPHARAVIS_STORAGE_PG_URI", "postgresql://postgres:${POSTGRES_PASSWORD}@vectordb:5432/rag_api"),
    )
    # Expand env var references in pg_uri
    pg_uri = pg_uri.replace("${POSTGRES_PASSWORD}", os.getenv("POSTGRES_PASSWORD", ""))

    return StorageConfig(
        enabled=enabled,
        max_total_bytes=max_total_bytes,
        warn_threshold_pct=warn_threshold_pct,
        dry_run=dry_run,
        service_percentages=percentages,
        data_root=data_root,
        mongo_uri=mongo_uri,
        pg_uri=pg_uri,
    )
