"""AlphaRavis Storage Manager — percentage-based disk quota enforcement.

Data directories managed (from docker-compose.yml host mounts):
  librechat-data  — MongoDB chat history
  media-data      — Media Gallery files
  hermes-data     — Hermes agent sessions/skills/audio
  pixelle-data    — Pixelle generated images
  openwebui-data  — OpenWebUI chat history
  logs            — All service logs

Each service gets a configurable percentage of the total cap.
When a service exceeds its budget, oldest entries are auto-deleted.
"""

from ai_stack.storage_manager.config import StorageConfig, get_storage_config
from ai_stack.storage_manager.budget import StorageBudget
from ai_stack.storage_manager.scanner import scan_all_services
from ai_stack.storage_manager.cleaner import clean_service
from ai_stack.storage_manager.manager import StorageManager, get_storage_manager

__all__ = [
    "StorageConfig",
    "get_storage_config",
    "StorageBudget",
    "scan_all_services",
    "clean_service",
    "StorageManager",
    "get_storage_manager",
]
