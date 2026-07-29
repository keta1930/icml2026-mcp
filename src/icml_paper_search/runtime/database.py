"""SQLite runtime state for sessions, limits, provider quotas, and caching."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

SCHEMA_ID = "icml-paper-search-milvus"

_SCHEMA = (
    "CREATE TABLE runtime_meta (key TEXT PRIMARY KEY, value BLOB NOT NULL)",
    """CREATE TABLE sessions (
        token_hash TEXT PRIMARY KEY CHECK(length(token_hash) = 64),
        label TEXT NOT NULL,
        created_at REAL NOT NULL,
        last_active_at REAL NOT NULL,
        default_top_k INTEGER,
        empty_rounds INTEGER NOT NULL DEFAULT 0 CHECK(empty_rounds >= 0),
        query_count INTEGER NOT NULL DEFAULT 0 CHECK(query_count >= 0),
        version INTEGER NOT NULL DEFAULT 0 CHECK(version >= 0))""",
    "CREATE INDEX sessions_last_active ON sessions(last_active_at)",
    """CREATE TABLE session_seen (
        token_hash TEXT NOT NULL REFERENCES sessions(token_hash) ON DELETE CASCADE,
        paper_id TEXT NOT NULL,
        title TEXT NOT NULL,
        first_seen_at REAL NOT NULL,
        ordinal INTEGER NOT NULL CHECK(ordinal >= 1),
        PRIMARY KEY(token_hash, paper_id),
        UNIQUE(token_hash, ordinal))""",
    """CREATE TABLE session_queries (
        token_hash TEXT NOT NULL REFERENCES sessions(token_hash) ON DELETE CASCADE,
        ordinal INTEGER NOT NULL CHECK(ordinal >= 1),
        query_hash TEXT NOT NULL CHECK(length(query_hash) = 64),
        filters_desc TEXT NOT NULL,
        new_hits INTEGER NOT NULL CHECK(new_hits >= 0),
        duplicates_suppressed INTEGER NOT NULL CHECK(duplicates_suppressed >= 0),
        candidates_examined INTEGER NOT NULL CHECK(candidates_examined >= 0),
        created_at REAL NOT NULL,
        PRIMARY KEY(token_hash, ordinal))""",
    """CREATE TABLE rate_buckets (
        scope TEXT NOT NULL,
        identity_hash TEXT NOT NULL,
        tokens REAL NOT NULL CHECK(tokens >= 0),
        updated_at REAL NOT NULL,
        PRIMARY KEY(scope, identity_hash))""",
    """CREATE TABLE rate_daily (
        scope TEXT NOT NULL,
        identity_hash TEXT NOT NULL,
        utc_day TEXT NOT NULL,
        request_count INTEGER NOT NULL CHECK(request_count >= 0),
        PRIMARY KEY(scope, identity_hash, utc_day))""",
    """CREATE TABLE provider_windows (
        minute_key TEXT PRIMARY KEY,
        request_count INTEGER NOT NULL CHECK(request_count >= 0),
        consumed_tokens INTEGER NOT NULL CHECK(consumed_tokens >= 0),
        reserved_tokens INTEGER NOT NULL CHECK(reserved_tokens >= 0),
        updated_at REAL NOT NULL)""",
    """CREATE TABLE provider_reservations (
        reservation_id TEXT PRIMARY KEY,
        minute_key TEXT NOT NULL,
        reserved_tokens INTEGER NOT NULL CHECK(reserved_tokens > 0),
        actual_tokens INTEGER,
        status TEXT NOT NULL CHECK(status IN ('reserved', 'settled')),
        created_at REAL NOT NULL,
        settled_at REAL)""",
    "CREATE INDEX provider_reservations_status ON provider_reservations(status, created_at)",
    """CREATE TABLE embedding_cache (
        cache_key TEXT PRIMARY KEY CHECK(length(cache_key) = 64),
        vector BLOB NOT NULL,
        dimension INTEGER NOT NULL CHECK(dimension > 0),
        token_count INTEGER NOT NULL DEFAULT 0 CHECK(token_count >= 0),
        created_at REAL NOT NULL,
        last_access_at REAL NOT NULL,
        expires_at REAL NOT NULL)""",
    "CREATE INDEX embedding_cache_expiry ON embedding_cache(expires_at, last_access_at)",
)


class RuntimeDatabase:
    """Create a short-lived SQLite connection for each operation."""

    def __init__(self, path: Path | str, *, busy_timeout_ms: int = 5_000) -> None:
        self.path = Path(path)
        self.busy_timeout_ms = busy_timeout_ms
        self._initialize_lock = threading.Lock()
        self._initialized = False
        self._closed = False

    def initialize(self) -> None:
        """Create the current schema or reject a database with another schema."""
        with self._initialize_lock:
            self._ensure_open()
            if self._initialized:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = self._open_connection()
            try:
                connection.execute("PRAGMA journal_mode=WAL")
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                    )
                }
                if not tables:
                    self._create_schema(connection)
                self._validate_schema(connection)
            finally:
                connection.close()
            self._initialized = True

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield one connection owned by the calling thread."""
        self.initialize()
        connection = self._open_connection()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(
        self, mode: Literal["DEFERRED", "IMMEDIATE", "EXCLUSIVE"] = "IMMEDIATE"
    ) -> Iterator[sqlite3.Connection]:
        """Run a short transaction with automatic commit or rollback."""
        with self.connection() as connection:
            connection.execute(f"BEGIN {mode}")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    def close(self) -> None:
        """Prevent new operations."""
        with self._initialize_lock:
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("runtime database is closed")

    def _open_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout_ms / 1_000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.execute("BEGIN IMMEDIATE")
        try:
            for statement in _SCHEMA:
                connection.execute(statement)
            connection.execute(
                "INSERT INTO runtime_meta(key, value) VALUES ('schema_id', ?)",
                (SCHEMA_ID,),
            )
        except BaseException:
            connection.rollback()
            raise
        else:
            connection.commit()

    @staticmethod
    def _validate_schema(connection: sqlite3.Connection) -> None:
        try:
            row = connection.execute(
                "SELECT value FROM runtime_meta WHERE key = 'schema_id'"
            ).fetchone()
        except sqlite3.OperationalError as exc:
            raise RuntimeError(
                "runtime database does not match the current schema"
            ) from exc
        if row is None or row[0] != SCHEMA_ID:
            raise RuntimeError("runtime database does not match the current schema")


def hash_capability(token: str) -> str:
    """Hash a session capability before persistence."""
    if not token:
        raise ValueError("capability token must not be empty")
    return _sha256(b"icml-capability\0" + token.encode())


def hash_client_identity(identity: str, salt: str | bytes) -> str:
    """Hash a source identity with a deployment-private salt."""
    salt_bytes = salt.encode() if isinstance(salt, str) else salt
    if not identity.strip() or not salt_bytes:
        raise ValueError("client identity and salt must not be empty")
    return _sha256(b"icml-client\0" + salt_bytes + b"\0" + identity.strip().encode())


def hash_query(query: str) -> str:
    """Normalize and hash a query so raw text is not persisted."""
    normalized = " ".join(query.split())
    if not normalized:
        raise ValueError("query must not be empty")
    return _sha256(b"icml-query\0" + normalized.encode())


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
