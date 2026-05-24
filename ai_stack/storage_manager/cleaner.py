"""Storage cleaner — delete oldest entries when a service exceeds its budget.

Cleanup strategies per service:
  librechat      — MongoDB: delete oldest conversations by createdAt
  vectordb       — PostgreSQL: delete oldest archive chunks by created_at
  media_gallery  — Filesystem: delete oldest files by mtime
  hermes         — Filesystem: delete oldest files by mtime
                   (session cleanup delegated to hermes-agent itself)
  pixelle        — Filesystem: delete oldest files by mtime
  openwebui      — Filesystem: delete oldest files by mtime
  logs           — Filesystem: delete oldest log files by mtime
"""

import os
import logging
from dataclasses import dataclass, field

from ai_stack.storage_manager.config import StorageConfig
from ai_stack.storage_manager.scanner import ServiceUsage, SERVICE_DIRS

logger = logging.getLogger(__name__)


@dataclass
class CleanupResult:
    """Result of a cleanup operation on one service."""

    service_name: str
    bytes_freed: int
    files_deleted: int
    mongo_docs_deleted: int
    pg_docs_deleted: int = 0
    error: str = ""
    dry_run: bool = False


def _oldest_files(path: str, count: int = 200) -> list[tuple[str, int]]:
    """Return the N oldest files in a directory tree, sorted by mtime.

    Returns list of (full_path, size_bytes) sorted oldest-first.
    """
    entries: list[tuple[str, int, float]] = []
    try:
        for dirpath, _, filenames in os.walk(path):
            for fname in filenames:
                fp = os.path.join(dirpath, fname)
                try:
                    st = os.lstat(fp)
                    entries.append((fp, st.st_size, st.st_mtime))
                except OSError:
                    continue
    except OSError:
        pass
    entries.sort(key=lambda x: x[2])  # oldest first
    return [(fp, sz) for fp, sz, _ in entries[:count]]


def _clean_filesystem(
    service_name: str,
    data_root: str,
    rel_dir: str,
    excess_bytes: int,
    dry_run: bool,
) -> CleanupResult:
    """Delete oldest files from a directory until excess_bytes is freed."""
    full_path = os.path.join(data_root, rel_dir)
    if not os.path.isdir(full_path):
        return CleanupResult(
            service_name=service_name,
            bytes_freed=0,
            files_deleted=0,
            mongo_docs_deleted=0,
            dry_run=dry_run,
            error=f"directory not found: {full_path}",
        )

    freed = 0
    deleted = 0
    oldest = _oldest_files(full_path, count=500)

    for fp, sz in oldest:
        if freed >= excess_bytes:
            break
        try:
            if dry_run:
                logger.info("  [DRY RUN] would delete: %s (%d bytes)", fp, sz)
            else:
                os.remove(fp)
                logger.info("  deleted: %s (%d bytes)", fp, sz)
            freed += sz
            deleted += 1
        except OSError as e:
            logger.warning("  failed to delete %s: %s", fp, e)

    return CleanupResult(
        service_name=service_name,
        bytes_freed=freed,
        files_deleted=deleted,
        mongo_docs_deleted=0,
        dry_run=dry_run,
        error="" if freed >= excess_bytes else f"freed {freed}/{excess_bytes} bytes — still over budget",
    )


def _clean_mongo(
    mongo_uri: str,
    excess_bytes: int,
    dry_run: bool,
) -> CleanupResult:
    """Delete oldest conversations from MongoDB to free excess_bytes.

    Deletes oldest LibreChat conversations by createdAt, then langgraph
    checkpoints by created_at. Does NOT drop the database.
    """
    try:
        from pymongo import MongoClient
    except ImportError:
        return CleanupResult(
            service_name="librechat",
            bytes_freed=0,
            files_deleted=0,
            mongo_docs_deleted=0,
            dry_run=dry_run,
            error="pymongo not installed",
        )

    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        total_deleted = 0
        target_bytes = excess_bytes

        # ── LibreChat conversations (oldest first) ──
        db = client["LibreChat"]
        coll_names = ["conversations", "messages", "convos"]  # try all known names
        for coll_name in coll_names:
            if coll_name not in db.list_collection_names():
                continue
            coll = db[coll_name]
            # get oldest docs
            cursor = coll.find({}, {"_id": 1}).sort("createdAt", 1).limit(500)
            ids_to_delete = [doc["_id"] for doc in cursor]

            if not ids_to_delete:
                continue

            if dry_run:
                logger.info("  [DRY RUN] would delete %d docs from LibreChat.%s",
                            len(ids_to_delete), coll_name)
                total_deleted += len(ids_to_delete)
                break
            else:
                result = coll.delete_many({"_id": {"$in": ids_to_delete}})
                logger.info("  deleted %d docs from LibreChat.%s", result.deleted_count, coll_name)
                total_deleted += result.deleted_count

                # approximate bytes freed (MongoDB will reclaim async)
                # use avg doc size from previous scan
                if result.deleted_count > 0:
                    break

        # ── langgraph checkpoints ──
        try:
            lg_db = client["langgraph"]
            for coll_name in lg_db.list_collection_names():
                if "checkpoint" not in coll_name.lower():
                    continue
                coll = lg_db[coll_name]
                cursor = coll.find({}, {"_id": 1}).sort("created_at", 1).limit(500)
                ids_to_delete = [doc["_id"] for doc in cursor]
                if not ids_to_delete:
                    continue
                if dry_run:
                    logger.info("  [DRY RUN] would delete %d docs from langgraph.%s",
                                len(ids_to_delete), coll_name)
                    total_deleted += len(ids_to_delete)
                else:
                    result = coll.delete_many({"_id": {"$in": ids_to_delete}})
                    logger.info("  deleted %d docs from langgraph.%s", result.deleted_count, coll_name)
                    total_deleted += result.deleted_count
        except Exception as e:
            logger.warning("  langgraph cleanup: %s", e)

        client.close()

        # MongoDB frees disk asynchronously — we can't measure exact bytes freed
        # Return an estimate: assume avg doc is ~2KB
        estimated_freed = total_deleted * 2048

        return CleanupResult(
            service_name="librechat",
            bytes_freed=estimated_freed,
            files_deleted=0,
            mongo_docs_deleted=total_deleted,
            dry_run=dry_run,
        )

    except Exception as e:
        return CleanupResult(
            service_name="librechat",
            bytes_freed=0,
            files_deleted=0,
            mongo_docs_deleted=0,
            dry_run=dry_run,
            error=str(e),
        )


def _clean_vectordb(
    pg_uri: str,
    excess_bytes: int,
    dry_run: bool,
) -> CleanupResult:
    """Delete oldest archive chunks from pgvector to free excess_bytes.

    Targets langchain_pg_embedding (archive chunks) ordered by created_at.
    Does NOT drop tables or delete collections — only old chunks.
    """
    try:
        import psycopg2
    except ImportError:
        return CleanupResult(
            service_name="vectordb",
            bytes_freed=0,
            files_deleted=0,
            mongo_docs_deleted=0,
            dry_run=dry_run,
            error="psycopg2 not installed",
        )

    try:
        conn = psycopg2.connect(pg_uri, connect_timeout=5)
        conn.autocommit = True
        cur = conn.cursor()

        total_deleted = 0
        avg_row_bytes = 2048  # fallback estimate, updated from table stats if available

        # ── Delete oldest langchain_pg_embedding rows ──
        # These store the actual archive/document chunks.
        # Estimate average row size from the table stats.
        try:
            cur.execute(
                "SELECT reltuples::bigint, pg_total_relation_size('langchain_pg_embedding') "
                "FROM pg_class WHERE relname = 'langchain_pg_embedding'"
            )
            row = cur.fetchone()
            if row and row[0] and row[0] > 0:
                avg_row_bytes = max(256, row[1] // row[0])
            else:
                avg_row_bytes = 2048  # fallback estimate

            target_rows = max(1, excess_bytes // avg_row_bytes)

            if dry_run:
                cur.execute(
                    "SELECT COUNT(*) FROM ("
                    "  SELECT uuid FROM langchain_pg_embedding ORDER BY created_at LIMIT %s"
                    ") AS sub", (target_rows,)
                )
                count = cur.fetchone()[0]
                logger.info("  [DRY RUN] would delete %d rows from langchain_pg_embedding", count)
                total_deleted += count
            else:
                cur.execute(
                    "DELETE FROM langchain_pg_embedding WHERE uuid IN ("
                    "  SELECT uuid FROM langchain_pg_embedding ORDER BY created_at LIMIT %s"
                    ")",
                    (target_rows,),
                )
                total_deleted += cur.rowcount or 0
                logger.info("  deleted %d rows from langchain_pg_embedding", cur.rowcount or 0)
        except Exception as e:
            logger.warning("  langchain_pg_embedding cleanup: %s", e)

        # ── Also clean old embedding jobs ──
        try:
            cur.execute(
                "DELETE FROM alpharavis_embedding_jobs "
                "WHERE status IN ('completed', 'failed') AND created_at < NOW() - INTERVAL '30 days'"
            )
            total_deleted += cur.rowcount or 0
            if cur.rowcount and cur.rowcount > 0:
                logger.info("  deleted %d old embedding jobs", cur.rowcount)
        except Exception as e:
            logger.debug("  embedding_jobs cleanup: %s", e)

        cur.close()
        conn.close()

        estimated_freed = total_deleted * avg_row_bytes if total_deleted > 0 else 0

        return CleanupResult(
            service_name="vectordb",
            bytes_freed=estimated_freed,
            files_deleted=0,
            mongo_docs_deleted=0,
            pg_docs_deleted=total_deleted,
            dry_run=dry_run,
        )

    except Exception as e:
        return CleanupResult(
            service_name="vectordb",
            bytes_freed=0,
            files_deleted=0,
            mongo_docs_deleted=0,
            dry_run=dry_run,
            error=str(e),
        )


def clean_service(
    service_name: str,
    usage: ServiceUsage,
    budget_bytes: int,
    config: StorageConfig,
) -> CleanupResult:
    """Clean a single service if it exceeds its budget.

    Returns the cleanup result. If usage is within budget, returns a no-op result.
    """
    excess = usage.bytes_used - budget_bytes
    if excess <= 0:
        return CleanupResult(
            service_name=service_name,
            bytes_freed=0,
            files_deleted=0,
            mongo_docs_deleted=0,
            dry_run=config.dry_run,
            error="",
        )

    logger.info("Cleaning %s: %d MB used, %d MB budget, %d MB excess",
                service_name,
                usage.bytes_used // (1024 * 1024),
                budget_bytes // (1024 * 1024),
                excess // (1024 * 1024))

    if service_name == "librechat":
        return _clean_mongo(config.mongo_uri, excess, config.dry_run)
    elif service_name == "vectordb":
        return _clean_vectordb(config.pg_uri, excess, config.dry_run)
    elif service_name in SERVICE_DIRS:
        return _clean_filesystem(service_name, config.data_root,
                                 SERVICE_DIRS[service_name], excess, config.dry_run)
    else:
        return CleanupResult(
            service_name=service_name,
            bytes_freed=0,
            files_deleted=0,
            mongo_docs_deleted=0,
            dry_run=config.dry_run,
            error=f"unknown service: {service_name}",
        )
