"""Storage scanner — measure disk usage per service.

Scans host-mounted data directories (from docker-compose.yml) and
MongoDB database stats. Returns ServiceUsage objects with byte counts.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ServiceUsage:
    """Measured disk usage for a single data service."""

    service_name: str
    bytes_used: int
    file_count: int = 0
    largest_file: str = ""
    largest_file_bytes: int = 0
    db_stats: dict = field(default_factory=dict)  # extra MongoDB stats if applicable
    error: str = ""


# ── which data directories (relative to data_root) belong to which service ──
SERVICE_DIRS: dict[str, str] = {
    "librechat": "librechat-data",
    "media_gallery": "media-data",
    "hermes": "hermes-data",
    "pixelle": "pixelle-data",
    "openwebui": "openwebui-data",
    "logs": "logs",
    "vectordb": "data/pgdata",
}

# ── services measured via PostgreSQL dbStats, not just filesystem du ──
DB_SERVICES: set[str] = {"librechat", "vectordb"}


def _du_dir(path: str) -> tuple[int, int, str, int]:
    """Walk a directory, return (total_bytes, file_count, largest_path, largest_bytes)."""
    total = 0
    count = 0
    largest_path = ""
    largest_bytes = 0
    try:
        for dirpath, _, filenames in os.walk(path):
            for fname in filenames:
                fp = os.path.join(dirpath, fname)
                try:
                    st = os.lstat(fp)
                    sz = st.st_size
                    total += sz
                    count += 1
                    if sz > largest_bytes:
                        largest_bytes = sz
                        largest_path = fp
                except OSError:
                    continue
    except OSError as e:
        logger.debug("Cannot walk %s: %s", path, e)
        return 0, 0, "", 0
    return total, count, largest_path, largest_bytes


def scan_directory_service(data_root: str, service_name: str, rel_dir: str) -> ServiceUsage:
    """Scan a host-mounted directory for a service."""
    full_path = os.path.join(data_root, rel_dir)
    if not os.path.isdir(full_path):
        return ServiceUsage(service_name=service_name, bytes_used=0, error=f"dir not found: {full_path}")
    total, count, largest, largest_sz = _du_dir(full_path)
    return ServiceUsage(
        service_name=service_name,
        bytes_used=total,
        file_count=count,
        largest_file=largest,
        largest_file_bytes=largest_sz,
    )


def scan_mongo_service(mongo_uri: str) -> ServiceUsage:
    """Query MongoDB for database storage stats (librechat + langgraph).

    Returns dbStats for the LibreChat and langgraph databases.
    """
    try:
        from pymongo import MongoClient
    except ImportError:
        return ServiceUsage(
            service_name="librechat",
            bytes_used=0,
            error="pymongo not installed — cannot query MongoDB stats",
        )

    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db_stats: dict = {}
        total = 0
        for db_name in ("LibreChat", "langgraph"):
            try:
                stats = client[db_name].command("dbStats")
                data_size = stats.get("dataSize", 0)
                index_size = stats.get("indexSize", 0)
                db_stats[db_name] = {
                    "dataSize": data_size,
                    "indexSize": index_size,
                    "objects": stats.get("objects", 0),
                }
                total += data_size + index_size
            except Exception as e:
                db_stats[db_name] = {"error": str(e)}
        client.close()
        return ServiceUsage(
            service_name="librechat",
            bytes_used=total,
            db_stats=db_stats,
        )
    except Exception as e:
        return ServiceUsage(
            service_name="librechat",
            bytes_used=0,
            error=f"MongoDB connection failed: {e}",
        )


def scan_vectordb_service(pg_uri: str) -> ServiceUsage:
    """Query PostgreSQL for vector database stats (rag_api + litellm).

    Connects to vectordb container and sums data+index sizes across
    rag_api and litellm databases.
    """
    try:
        import psycopg2
    except ImportError:
        return ServiceUsage(
            service_name="vectordb",
            bytes_used=0,
            error="psycopg2 not installed — cannot query PostgreSQL stats",
        )

    try:
        conn = psycopg2.connect(pg_uri, connect_timeout=5)
        conn.autocommit = True
        cur = conn.cursor()
        db_stats: dict = {}
        total = 0

        for db_name in ("rag_api", "litellm"):
            try:
                cur.execute(f"SELECT pg_database_size('{db_name}')")
                row = cur.fetchone()
                if row:
                    size = row[0]
                    db_stats[db_name] = {"total_size": size}
                    total += size
                else:
                    db_stats[db_name] = {"error": "database not found"}
            except Exception as e:
                db_stats[db_name] = {"error": str(e)}

        cur.close()
        conn.close()
        return ServiceUsage(
            service_name="vectordb",
            bytes_used=total,
            db_stats=db_stats,
        )
    except Exception as e:
        return ServiceUsage(
            service_name="vectordb",
            bytes_used=0,
            error=f"PostgreSQL connection failed: {e}",
        )


def scan_all_services(data_root: str, mongo_uri: str, pg_uri: str = "") -> dict[str, ServiceUsage]:
    """Scan all services and return usage per service.

    MongoDB is measured via dbStats (for librechat).
    PostgreSQL is measured via pg_database_size (for vectordb).
    All other services are measured via filesystem du.
    """
    results: dict[str, ServiceUsage] = {}

    # ── filesystem-based services ──
    for service_name, rel_dir in SERVICE_DIRS.items():
        if service_name in DB_SERVICES:
            continue
        results[service_name] = scan_directory_service(data_root, service_name, rel_dir)

    # ── MongoDB (librechat + langgraph) ──
    results["librechat"] = scan_mongo_service(mongo_uri)

    # ── also include file-based librechat extras (images etc.) ──
    file_usage = scan_directory_service(data_root, "librechat", "librechat-data")
    if file_usage.bytes_used > 0:
        existing = results["librechat"]
        # add file bytes on top of MongoDB bytes
        results["librechat"] = ServiceUsage(
            service_name="librechat",
            bytes_used=existing.bytes_used + file_usage.bytes_used,
            file_count=file_usage.file_count,
            largest_file=file_usage.largest_file,
            largest_file_bytes=file_usage.largest_file_bytes,
            db_stats=existing.db_stats,
            error=existing.error or file_usage.error,
        )

    # ── PostgreSQL / vectordb (rag_api + litellm) ──
    if pg_uri:
        results["vectordb"] = scan_vectordb_service(pg_uri)
    else:
        results["vectordb"] = ServiceUsage(
            service_name="vectordb",
            bytes_used=0,
            error="pg_uri not configured — cannot query vectordb stats",
        )

    # ── also include filesystem portion of pgdata ──
    file_usage_pg = scan_directory_service(data_root, "vectordb", "data/pgdata")
    if file_usage_pg.bytes_used > 0:
        existing_pg = results["vectordb"]
        results["vectordb"] = ServiceUsage(
            service_name="vectordb",
            bytes_used=existing_pg.bytes_used + file_usage_pg.bytes_used,
            file_count=file_usage_pg.file_count,
            largest_file=file_usage_pg.largest_file,
            largest_file_bytes=file_usage_pg.largest_file_bytes,
            db_stats=existing_pg.db_stats,
            error=existing_pg.error or file_usage_pg.error,
        )

    return results
