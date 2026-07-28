"""Runtime SQLite 的连接、迁移与隐私 hash 基础设施。"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

SCHEMA_VERSION = 2

_MIGRATION_1 = (
    "CREATE TABLE runtime_meta (key TEXT PRIMARY KEY, value BLOB NOT NULL)",
    """CREATE TABLE sessions (
        token_hash TEXT PRIMARY KEY CHECK(length(token_hash) = 64), label TEXT NOT NULL,
        created_at REAL NOT NULL, last_active_at REAL NOT NULL, default_top_k INTEGER,
        empty_rounds INTEGER NOT NULL DEFAULT 0 CHECK(empty_rounds >= 0),
        query_count INTEGER NOT NULL DEFAULT 0 CHECK(query_count >= 0),
        status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'closed')),
        version INTEGER NOT NULL DEFAULT 0 CHECK(version >= 0))""",
    "CREATE INDEX sessions_active_last_seen ON sessions(status, last_active_at)",
    """CREATE TABLE session_seen (
        token_hash TEXT NOT NULL REFERENCES sessions(token_hash) ON DELETE CASCADE,
        paper_id TEXT NOT NULL, title TEXT NOT NULL, first_seen_at REAL NOT NULL,
        ordinal INTEGER NOT NULL CHECK(ordinal >= 1), PRIMARY KEY(token_hash, paper_id),
        UNIQUE(token_hash, ordinal))""",
    """CREATE TABLE session_queries (
        token_hash TEXT NOT NULL REFERENCES sessions(token_hash) ON DELETE CASCADE,
        ordinal INTEGER NOT NULL CHECK(ordinal >= 1),
        query_hash TEXT NOT NULL CHECK(length(query_hash) = 64), query_summary TEXT,
        filters_desc TEXT NOT NULL, new_hits INTEGER NOT NULL CHECK(new_hits >= 0),
        duplicates_suppressed INTEGER NOT NULL CHECK(duplicates_suppressed >= 0),
        candidates_examined INTEGER NOT NULL CHECK(candidates_examined >= 0),
        created_at REAL NOT NULL, PRIMARY KEY(token_hash, ordinal))""",
    """CREATE TABLE rate_buckets (
        scope TEXT NOT NULL, identity_hash TEXT NOT NULL, tokens REAL NOT NULL CHECK(tokens >= 0),
        updated_at REAL NOT NULL, PRIMARY KEY(scope, identity_hash))""",
    """CREATE TABLE rate_daily (
        scope TEXT NOT NULL, identity_hash TEXT NOT NULL, utc_day TEXT NOT NULL,
        request_count INTEGER NOT NULL CHECK(request_count >= 0),
        PRIMARY KEY(scope, identity_hash, utc_day))""",
    """CREATE TABLE budget_ledger (
        scope TEXT NOT NULL, period_kind TEXT NOT NULL CHECK(period_kind IN ('day', 'month')),
        period_key TEXT NOT NULL, consumed_tokens INTEGER NOT NULL DEFAULT 0 CHECK(consumed_tokens >= 0),
        reserved_tokens INTEGER NOT NULL DEFAULT 0 CHECK(reserved_tokens >= 0), updated_at REAL NOT NULL,
        PRIMARY KEY(scope, period_kind, period_key))""",
    """CREATE TABLE budget_reservations (
        reservation_id TEXT PRIMARY KEY, scope TEXT NOT NULL, utc_day TEXT NOT NULL,
        utc_month TEXT NOT NULL, reserved_tokens INTEGER NOT NULL CHECK(reserved_tokens > 0),
        settled_tokens INTEGER CHECK(settled_tokens >= 0),
        status TEXT NOT NULL CHECK(status IN ('reserved', 'settled', 'released')),
        created_at REAL NOT NULL, settled_at REAL)""",
    "CREATE INDEX budget_reservations_scope_status ON budget_reservations(scope, status, created_at)",
    """CREATE TABLE embedding_cache (
        cache_key TEXT PRIMARY KEY CHECK(length(cache_key) = 64), vector BLOB NOT NULL,
        dimension INTEGER NOT NULL CHECK(dimension > 0),
        token_count INTEGER NOT NULL DEFAULT 0 CHECK(token_count >= 0),
        created_at REAL NOT NULL, last_access_at REAL NOT NULL, expires_at REAL NOT NULL)""",
    "CREATE INDEX embedding_cache_expiry ON embedding_cache(expires_at, last_access_at)",
)

_MIGRATION_2 = (
    "ALTER TABLE budget_ledger ADD COLUMN overrun_tokens INTEGER NOT NULL DEFAULT 0 CHECK(overrun_tokens >= 0)",
    "ALTER TABLE budget_reservations ADD COLUMN actual_tokens INTEGER CHECK(actual_tokens >= 0)",
    "ALTER TABLE budget_reservations ADD COLUMN overrun_tokens INTEGER NOT NULL DEFAULT 0 CHECK(overrun_tokens >= 0)",
)

_MIGRATIONS = {
    1: _MIGRATION_1,
    2: _MIGRATION_2,
}


class RuntimeDatabase:
    """为每次操作创建线程内独占的 SQLite connection。"""

    def __init__(self, path: Path | str, *, busy_timeout_ms: int = 5_000) -> None:
        self.path = Path(path)
        self.busy_timeout_ms = busy_timeout_ms
        self._initialize_lock = threading.Lock()
        self._initialized = False
        self._closed = False

    def initialize(self) -> None:
        """启用 WAL/FTS5 并幂等应用 runtime schema migration。"""
        with self._initialize_lock:
            self._ensure_open()
            if self._initialized:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = self._open_connection()
            try:
                connection.execute("PRAGMA journal_mode=WAL")
                self._preflight_fts5(connection)
                self._apply_migrations(connection)
            finally:
                connection.close()
            self._initialized = True

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """返回只在当前调用线程使用的短生命周期 connection。"""
        self.initialize()
        connection = self._open_connection()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self, mode: Literal["DEFERRED", "IMMEDIATE", "EXCLUSIVE"] = "IMMEDIATE") -> Iterator[sqlite3.Connection]:
        """在短事务中执行纯 SQLite 工作并自动提交或回滚。"""
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
        """阻止后续连接；已退出的短连接无需额外回收。"""
        with self._initialize_lock:
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("runtime database 已关闭")

    def _open_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=self.busy_timeout_ms / 1_000, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    @staticmethod
    def _preflight_fts5(connection: sqlite3.Connection) -> None:
        try:
            connection.execute("CREATE VIRTUAL TABLE temp.fts5_preflight USING fts5(text)")
            connection.execute("DROP TABLE temp.fts5_preflight")
        except sqlite3.OperationalError as exc:
            raise RuntimeError("当前 Python SQLite 未启用 FTS5") from exc

    @staticmethod
    def _apply_migrations(connection: sqlite3.Connection) -> None:
        connection.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")
        current = connection.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations").fetchone()[0]
        if current > SCHEMA_VERSION:
            raise RuntimeError("runtime database schema 比当前程序更新")
        for version in range(current + 1, SCHEMA_VERSION + 1):
            connection.execute("BEGIN IMMEDIATE")
            try:
                for statement in _MIGRATIONS[version]:
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO schema_migrations(version) VALUES (?)", (version,)
                )
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()


def hash_capability(token: str) -> str:
    """不可逆摘要 session capability，禁止把 raw token 写入数据库。"""
    if not token:
        raise ValueError("capability token 不能为空")
    return _sha256(b"icml-capability-v1\0" + token.encode())


def hash_client_identity(identity: str, salt: str | bytes) -> str:
    """用部署私有 salt 摘要来源身份，不保存原始 IP。"""
    salt_bytes = salt.encode() if isinstance(salt, str) else salt
    if not identity.strip() or not salt_bytes:
        raise ValueError("client identity 与 salt 不能为空")
    return _sha256(b"icml-client-v1\0" + salt_bytes + b"\0" + identity.strip().encode())


def hash_query(query: str) -> str:
    """规范空白后摘要 query，避免审计表保存原文。"""
    normalized = " ".join(query.split())
    if not normalized:
        raise ValueError("query 不能为空")
    return _sha256(b"icml-query-v1\0" + normalized.encode())


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
