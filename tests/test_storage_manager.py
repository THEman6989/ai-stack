"""Tests for ai_stack.storage_manager — config, budget, scanner, cleaner, manager."""

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from ai_stack.storage_manager.config import StorageConfig, get_storage_config
from ai_stack.storage_manager.budget import StorageBudget, ServiceBudget
from ai_stack.storage_manager.scanner import ServiceUsage, scan_directory_service, scan_all_services
from ai_stack.storage_manager.cleaner import (
    _oldest_files,
    _clean_filesystem,
    clean_service,
    CleanupResult,
)
from ai_stack.storage_manager.manager import (
    get_storage_status,
    get_storage_manager,
    StorageStatus,
)


# ── config tests ──


def test_config_defaults():
    """Default config when no env vars set."""
    with patch.dict(os.environ, {}, clear=True):
        os.environ["ALPHARAVIS_STORAGE_DATA_ROOT"] = "/tmp"
        cfg = get_storage_config()
        assert cfg.enabled is False
        assert cfg.max_total_bytes == 50 * 1024 ** 3
        assert cfg.warn_threshold_pct == 90.0
        assert cfg.dry_run is False
        assert cfg.service_percentages["media_gallery"] == 40.0
        assert cfg.service_percentages["librechat"] == 10.0
        assert cfg.service_percentages["vectordb"] == 30.0
        assert cfg.pg_uri


def test_config_enabled():
    """Enabled via env var."""
    with patch.dict(os.environ, {"ALPHARAVIS_STORAGE_MANAGER_ENABLED": "true"}, clear=False):
        os.environ["ALPHARAVIS_STORAGE_DATA_ROOT"] = "/tmp"
        cfg = get_storage_config()
        assert cfg.enabled is True


def test_config_custom_cap():
    """Custom total cap."""
    with patch.dict(os.environ, {
        "ALPHARAVIS_STORAGE_MANAGER_ENABLED": "true",
        "ALPHARAVIS_STORAGE_MAX_TOTAL_GB": "10",
        "ALPHARAVIS_STORAGE_DATA_ROOT": "/tmp",
    }, clear=False):
        cfg = get_storage_config()
        assert cfg.max_total_bytes == 10 * 1024 ** 3


def test_config_custom_percentages():
    """Custom per-service percentages, normalized to 100."""
    with patch.dict(os.environ, {
        "ALPHARAVIS_STORAGE_MANAGER_ENABLED": "true",
        "ALPHARAVIS_STORAGE_MAX_TOTAL_GB": "10",
        "ALPHARAVIS_STORAGE_MEDIA_GALLERY_PCT": "60",
        "ALPHARAVIS_STORAGE_LIBRECHAT_PCT": "20",
        "ALPHARAVIS_STORAGE_HERMES_PCT": "5",
        "ALPHARAVIS_STORAGE_VECTORDB_PCT": "0",
        "ALPHARAVIS_STORAGE_PIXELLE_PCT": "5",
        "ALPHARAVIS_STORAGE_OPENWEBUI_PCT": "5",
        "ALPHARAVIS_STORAGE_LOGS_PCT": "3",
        "ALPHARAVIS_STORAGE_OTHER_PCT": "2",
        "ALPHARAVIS_STORAGE_DATA_ROOT": "/tmp",
    }, clear=False):
        cfg = get_storage_config()
        total = sum(cfg.service_percentages.values())
        assert abs(total - 100.0) < 0.02
        assert cfg.service_percentages["media_gallery"] == 60.0


def test_budget_bytes():
    """Budget bytes calculation for a service."""
    cfg = StorageConfig(
        enabled=True,
        max_total_bytes=50 * 1024 ** 3,
        warn_threshold_pct=90.0,
        dry_run=False,
        service_percentages={"media_gallery": 40.0},
        data_root="/tmp",
        mongo_uri="mongodb://localhost",
        pg_uri="postgresql://localhost/rag_api",
    )
    assert cfg.budget_bytes("media_gallery") == 20 * 1024 ** 3


def test_budget_summary():
    """Budget summary string."""
    cfg = StorageConfig(
        enabled=True,
        max_total_bytes=50 * 1024 ** 3,
        warn_threshold_pct=90.0,
        dry_run=False,
        service_percentages={"media_gallery": 40.0, "librechat": 30.0, "logs": 10.0},
        data_root="/tmp",
        mongo_uri="mongodb://localhost",
        pg_uri="postgresql://localhost/rag_api",
    )
    budget = StorageBudget(config=cfg)
    summary = budget.summary()
    assert "50.0 GB" in summary
    assert "media_gallery" in summary
    assert "40.0%" in summary


# ── scanner tests ──


def test_scan_directory_service(tmp_path):
    """Scan a directory and measure usage."""
    test_dir = tmp_path / "testdata"
    test_dir.mkdir()
    (test_dir / "file1.txt").write_text("hello")
    (test_dir / "file2.txt").write_text("world!")

    usage = scan_directory_service(str(tmp_path), "test", "testdata")
    assert usage.service_name == "test"
    assert usage.bytes_used == 11
    assert usage.file_count == 2


def test_scan_directory_not_found():
    """Non-existent directory returns zero usage."""
    usage = scan_directory_service("/tmp", "test", "nonexistent_dir_12345")
    assert usage.bytes_used == 0
    assert "dir not found" in usage.error


def test_scan_all_services(tmp_path):
    """Scan all services with mock data."""
    (tmp_path / "media-data").mkdir()
    (tmp_path / "media-data" / "img.png").write_text("fakeimage")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "app.log").write_text("log line 1\nlog line 2")

    usage = scan_all_services(str(tmp_path), "mongodb://localhost:99999")
    assert "media_gallery" in usage
    assert "logs" in usage
    assert usage["media_gallery"].bytes_used == 9
    assert usage["logs"].bytes_used == 21


# ── cleaner tests ──


def test_oldest_files(tmp_path):
    """Find oldest files in a directory."""
    test_dir = tmp_path / "cleandata"
    test_dir.mkdir()
    f1 = test_dir / "old.txt"
    f1.write_text("old")
    f2 = test_dir / "new.txt"
    f2.write_text("newer")

    os.utime(f1, (0, 0))

    oldest = _oldest_files(str(test_dir), count=10)
    assert len(oldest) == 2
    assert "old.txt" in oldest[0][0]
    assert oldest[0][1] == 3


def test_clean_filesystem_dry_run(tmp_path):
    """Dry-run filesystem cleanup does not delete."""
    test_dir = tmp_path / "drydata"
    test_dir.mkdir()
    (test_dir / "a.txt").write_text("x" * 100)
    (test_dir / "b.txt").write_text("y" * 200)

    result = _clean_filesystem("test", str(tmp_path), "drydata", 50, dry_run=True)
    assert result.dry_run is True
    assert result.files_deleted > 0
    assert (test_dir / "a.txt").exists() or (test_dir / "b.txt").exists()


def test_clean_filesystem_real(tmp_path):
    """Real filesystem cleanup deletes oldest files."""
    test_dir = tmp_path / "realdata"
    test_dir.mkdir()
    f1 = test_dir / "old.txt"
    f1.write_text("x" * 500)
    f2 = test_dir / "new.txt"
    f2.write_text("y" * 100)

    os.utime(f1, (0, 0))

    result = _clean_filesystem("test", str(tmp_path), "realdata", 250, dry_run=False)
    assert result.dry_run is False
    assert result.files_deleted >= 1
    assert result.bytes_freed >= 250
    assert not f1.exists()
    assert f2.exists()


def test_clean_service_within_budget():
    """No cleanup when service is within budget."""
    cfg = StorageConfig(
        enabled=True,
        max_total_bytes=1000,
        warn_threshold_pct=90.0,
        dry_run=False,
        service_percentages={"test": 100.0},
        data_root="/tmp",
        mongo_uri="mongodb://localhost",
        pg_uri="postgresql://localhost/rag_api",
    )
    usage = ServiceUsage(service_name="test", bytes_used=500)
    result = clean_service("test", usage, 1000, cfg)
    assert result.bytes_freed == 0
    assert result.files_deleted == 0


def test_clean_service_over_budget(tmp_path):
    """Cleanup when service exceeds budget."""
    test_dir = tmp_path / "media-data"
    test_dir.mkdir()
    (test_dir / "big.txt").write_text("x" * 500)

    cfg = StorageConfig(
        enabled=True,
        max_total_bytes=1000,
        warn_threshold_pct=90.0,
        dry_run=False,
        service_percentages={"media_gallery": 100.0},
        data_root=str(tmp_path),
        mongo_uri="mongodb://localhost",
        pg_uri="postgresql://localhost/rag_api",
    )
    usage = ServiceUsage(service_name="media_gallery", bytes_used=500)
    result = clean_service("media_gallery", usage, cfg.budget_bytes("media_gallery"), cfg)
    assert result.bytes_freed == 0

    usage2 = ServiceUsage(service_name="media_gallery", bytes_used=1500)
    result2 = clean_service("media_gallery", usage2, cfg.budget_bytes("media_gallery"), cfg)
    assert result2.bytes_freed >= 500


# ── manager tests ──


def test_get_storage_manager():
    """Singleton storage manager."""
    mgr1 = get_storage_manager()
    mgr2 = get_storage_manager()
    assert mgr1 is mgr2


def test_storage_status_disabled():
    """Status when disabled."""
    with patch.dict(os.environ, {
        "ALPHARAVIS_STORAGE_MANAGER_ENABLED": "false",
        "ALPHARAVIS_STORAGE_DATA_ROOT": "/tmp",
    }, clear=True):
        status = get_storage_status()
        assert "DISABLED" in status.warnings[0]


def test_storage_status_table():
    """Status table format."""
    status = StorageStatus(
        config=StorageConfig(
            enabled=True,
            max_total_bytes=50 * 1024 ** 3,
            warn_threshold_pct=90.0,
            dry_run=False,
            service_percentages={"logs": 10.0, "media_gallery": 40.0},
            data_root="/tmp",
            mongo_uri="mongodb://localhost",
            pg_uri="postgresql://localhost/rag_api",
        ),
        budget=StorageBudget(config=StorageConfig(
            enabled=True, max_total_bytes=50 * 1024 ** 3,
            warn_threshold_pct=90.0, dry_run=False,
            service_percentages={"logs": 10.0, "media_gallery": 40.0},
            data_root="/tmp", mongo_uri="mongodb://localhost",
            pg_uri="postgresql://localhost/rag_api",
        )),
        usage={
            "logs": ServiceUsage("logs", 1 * 1024 ** 3),
            "media_gallery": ServiceUsage("media_gallery", 25 * 1024 ** 3),
        },
    )
    table = status.format_table()
    assert "logs" in table
    assert "media_gallery" in table
    assert "CRITICAL" in table


def test_pct_normalization():
    """Percentages that don't sum to 100 get normalized."""
    with patch.dict(os.environ, {
        "ALPHARAVIS_STORAGE_MANAGER_ENABLED": "true",
        "ALPHARAVIS_STORAGE_MAX_TOTAL_GB": "10",
        "ALPHARAVIS_STORAGE_MEDIA_GALLERY_PCT": "50",
        "ALPHARAVIS_STORAGE_LIBRECHAT_PCT": "25",
        "ALPHARAVIS_STORAGE_DATA_ROOT": "/tmp",
    }, clear=False):
        cfg = get_storage_config()
        total = sum(cfg.service_percentages.values())
        assert abs(total - 100.0) < 0.02


# ── vectordb scanner tests ──


def test_scan_vectordb_service_mock():
    """scan_vectordb_service with mocked psycopg2."""
    mock_psycopg2 = MagicMock()
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    mock_cur.fetchone.side_effect = [(1024 * 1024 * 10,), (1024 * 1024 * 5,)]
    mock_psycopg2.connect.return_value = mock_conn

    with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
        from ai_stack.storage_manager.scanner import scan_vectordb_service
        usage = scan_vectordb_service("postgresql://localhost/rag_api")

        assert usage.service_name == "vectordb"
        assert usage.bytes_used == 15 * 1024 * 1024


def test_scan_vectordb_no_psycopg2():
    """scan_vectordb_service when psycopg2 is not installed."""
    with patch.dict("sys.modules", {"psycopg2": None}):
        from ai_stack.storage_manager.scanner import scan_vectordb_service
        usage = scan_vectordb_service("postgresql://localhost/rag_api")
        assert usage.bytes_used == 0
        assert "not installed" in usage.error


# ── vectordb cleaner tests ──


def test_clean_vectordb_dry_run():
    """Dry-run vectordb cleanup returns a CleanupResult."""
    mock_psycopg2 = MagicMock()
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    # Return valid integers for the table stats query
    mock_cur.fetchone.return_value = (100, 102400)
    mock_psycopg2.connect.return_value = mock_conn

    with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
        from ai_stack.storage_manager.cleaner import _clean_vectordb
        result = _clean_vectordb("postgresql://localhost/rag_api", 1024 * 1024, dry_run=True)
        assert result.dry_run is True
        assert isinstance(result.pg_docs_deleted, int)


def test_clean_vectordb_no_psycopg2():
    """vectordb cleanup when psycopg2 is not installed."""
    with patch.dict("sys.modules", {"psycopg2": None}):
        from ai_stack.storage_manager.cleaner import _clean_vectordb
        result = _clean_vectordb("postgresql://localhost/rag_api", 1024, dry_run=False)
        assert "not installed" in result.error


# ── Integration: clean_service dispatches to vectordb ──


def test_clean_service_vectordb_dispatch():
    """clean_service dispatches vectordb to _clean_vectordb."""
    cfg = StorageConfig(
        enabled=True,
        max_total_bytes=50 * 1024 ** 3,
        warn_threshold_pct=90.0,
        dry_run=True,
        service_percentages={"vectordb": 30.0},
        data_root="/tmp",
        mongo_uri="mongodb://localhost",
        pg_uri="postgresql://localhost/rag_api",
    )

    mock_psycopg2 = MagicMock()
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    mock_cur.fetchone.side_effect = [(100, 100 * 1024), (10,)]
    mock_psycopg2.connect.return_value = mock_conn

    with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
        usage = ServiceUsage(service_name="vectordb", bytes_used=30 * 1024 ** 3)
        result = clean_service("vectordb", usage, cfg.budget_bytes("vectordb"), cfg)
        assert result.service_name == "vectordb"
        assert result.dry_run is True


def test_scan_all_services_includes_vectordb(tmp_path):
    """scan_all_services includes vectordb when pg_uri provided."""
    usage = scan_all_services(str(tmp_path), "mongodb://localhost:99999",
                              pg_uri="postgresql://localhost/rag_api")
    assert "vectordb" in usage
    usage2 = scan_all_services(str(tmp_path), "mongodb://localhost:99999")
    assert "vectordb" in usage2
