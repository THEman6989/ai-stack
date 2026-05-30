#!/usr/bin/env python3
"""Backfill PGVector contract fields for existing rows.

Dry-run by default. Pass --apply to actually update.

Fills:
  - source_id  = source_key    where empty
  - version    = metadata->>'source_digest' or metadata->>'version' or 'v1'
  - raw_ref    = '{}'::jsonb   where empty (PGVector default)

Usage:
  python scripts/backfill_pgvector_contract.py            # dry-run
  python scripts/backfill_pgvector_contract.py --apply    # apply changes
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph-app"))


def _env_pgvector_url() -> str:
    return os.getenv(
        "ALPHARAVIS_PGVECTOR_URL",
        "postgresql://postgres:postgres@localhost:5432/alpharavis",
    )


def _pgvector_table() -> str:
    return os.getenv("ALPHARAVIS_PGVECTOR_TABLE", "alpharavis_memory_vectors")


def _require_psycopg():
    try:
        import psycopg  # noqa: F401
    except ImportError:
        print("psycopg (psycopg3) is required. Install: pip install psycopg[binary]")
        sys.exit(1)


def _connect():
    _require_psycopg()
    import psycopg
    return psycopg.connect(_env_pgvector_url())


def _report(dry: bool, label: str, sql: str, count: int) -> None:
    prefix = "[DRY-RUN] " if dry else "[APPLY]   "
    print(f"{prefix}{label}: {count} rows  ({sql.strip()[:80]}...)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill PGVector contract fields")
    parser.add_argument("--apply", action="store_true", help="Actually update rows")
    args = parser.parse_args()

    table = _pgvector_table()
    dry = not args.apply

    print(f"Table: {table}")
    print(f"Mode:  {'DRY-RUN (no changes)' if dry else 'APPLY (will update)'}")
    print()

    conn = _connect()

    try:
        with conn.cursor() as cur:
            # 1. Report current state
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            total = cur.fetchone()[0]
            print(f"Total rows: {total}")

            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE source_id = ''")
            missing_source_id = cur.fetchone()[0]
            print(f"Missing source_id: {missing_source_id}")

            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE version = '' OR version = 'v1'")
            missing_version = cur.fetchone()[0]
            print(f"Missing version: {missing_version}")

            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE raw_ref IS NULL")
            null_raw_ref = cur.fetchone()[0]
            print(f"NULL raw_ref: {null_raw_ref}")

            if not dry:
                print()

            # 2. Fill source_id
            sql_source_id = f"UPDATE {table} SET source_id = source_key WHERE source_id = ''"
            if dry:
                _report(dry, "source_id", sql_source_id, missing_source_id)
            else:
                cur.execute(sql_source_id)
                _report(dry, "source_id", sql_source_id, cur.rowcount)

            # 3. Fill version
            sql_version = (
                f"UPDATE {table} "
                "SET version = COALESCE(NULLIF(metadata->>'version', ''), "
                "NULLIF(metadata->>'source_digest', ''), version, 'v1') "
                "WHERE version = '' OR version = 'v1'"
            )
            if dry:
                _report(dry, "version", sql_version, missing_version)
            else:
                cur.execute(sql_version)
                _report(dry, "version", sql_version, cur.rowcount)

            # 4. Fill raw_ref default
            sql_raw_ref = (
                f"UPDATE {table} "
                "SET raw_ref = '{}'::jsonb "
                "WHERE raw_ref IS NULL"
            )
            if dry:
                _report(dry, "raw_ref", sql_raw_ref, null_raw_ref)
            else:
                cur.execute(sql_raw_ref)
                _report(dry, "raw_ref", sql_raw_ref, cur.rowcount)

            if not dry:
                conn.commit()
                print()
                print("Backfill applied.")

            # 5. Final state
            print()
            if not dry:
                cur.execute(f"SELECT COUNT(*) FROM {table} WHERE source_id = ''")
                print(f"Remaining empty source_id: {cur.fetchone()[0]}")
                cur.execute(f"SELECT COUNT(*) FROM {table} WHERE version = '' OR version = 'v1'")
                print(f"Remaining default version: {cur.fetchone()[0]}")
                cur.execute(f"SELECT COUNT(*) FROM {table} WHERE raw_ref IS NULL")
                print(f"Remaining NULL raw_ref: {cur.fetchone()[0]}")

            cur.execute(
                f"SELECT source_type, COUNT(*) FROM {table} GROUP BY source_type ORDER BY COUNT(*) DESC"
            )
            print("\nRows by source_type:")
            for row in cur.fetchall():
                print(f"  {row[0]}: {row[1]}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
