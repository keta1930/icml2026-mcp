"""Validate the current runtime SQLite schema, transactions, and privacy hashes."""

from __future__ import annotations

import re

import pytest

from icml_paper_search.runtime.database import (
    RuntimeDatabase,
    hash_capability,
    hash_client_identity,
    hash_query,
)


def test_initialize_enables_wal_and_current_schema(tmp_path):
    database = RuntimeDatabase(tmp_path / "state" / "runtime.sqlite3")
    database.initialize()

    with database.connection() as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

    assert {
        "sessions",
        "session_seen",
        "session_queries",
        "rate_buckets",
        "rate_daily",
        "provider_windows",
        "provider_reservations",
        "embedding_cache",
    } <= tables


def test_transaction_commits_and_rolls_back(tmp_path):
    database = RuntimeDatabase(tmp_path / "runtime.sqlite3")
    with database.transaction() as connection:
        connection.execute("INSERT INTO runtime_meta(key, value) VALUES ('ok', 'yes')")

    with pytest.raises(RuntimeError):
        with database.transaction() as connection:
            connection.execute(
                "INSERT INTO runtime_meta(key, value) VALUES ('rolled', 'no')"
            )
            raise RuntimeError("stop")

    with database.connection() as connection:
        keys = {row[0] for row in connection.execute("SELECT key FROM runtime_meta")}
    assert keys == {"schema_id", "ok"}


def test_foreign_schema_is_rejected_without_mutation(tmp_path):
    path = tmp_path / "runtime.sqlite3"
    import sqlite3

    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE unrelated(value TEXT)")
    with pytest.raises(RuntimeError, match="current schema"):
        RuntimeDatabase(path).initialize()
    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert tables == {"unrelated"}


def test_restart_is_idempotent_and_close_is_final(tmp_path):
    path = tmp_path / "runtime.sqlite3"
    RuntimeDatabase(path).initialize()
    restarted = RuntimeDatabase(path)
    restarted.initialize()
    restarted.close()

    with pytest.raises(RuntimeError, match="is closed"), restarted.connection():
        pass


def test_hashes_are_stable_scoped_and_do_not_contain_raw_values():
    capability = "ses_raw-sensitive-token"
    client = "203.0.113.7"
    query = "  robust   vision  "
    hashes = {
        hash_capability(capability),
        hash_client_identity(client, "deployment-salt"),
        hash_query(query),
    }

    assert len(hashes) == 3
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes)
    assert capability not in "".join(hashes)
    assert hash_query(query) == hash_query("robust vision")
    assert hash_client_identity(client, "other-salt") not in hashes
