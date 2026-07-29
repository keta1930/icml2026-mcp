"""Persistent source and global request limits."""

from __future__ import annotations

import secrets
from datetime import datetime

from ..config import Settings, get_settings
from ..domain.errors import RateLimited
from .clock import utc_day, utc_timestamp
from .database import RuntimeDatabase, hash_client_identity


class SQLiteRateLimiter:
    """Enforce atomic token buckets and UTC daily request counts."""

    def __init__(
        self, database: RuntimeDatabase, settings: Settings | None = None
    ) -> None:
        self._database = database
        self._settings = settings or get_settings()
        self._database.initialize()
        self._identity_salt = self._load_identity_salt()

    def hash_identity(self, identity: str) -> str:
        """Hash source identities with a persistent deployment salt."""
        return hash_client_identity(identity, self._identity_salt)

    def check_search(
        self, client_hash: str, now: datetime | float | None = None
    ) -> None:
        """Consume the source, global, and source-daily search allowance."""
        self._check(
            client_hash,
            scope="search-ip",
            rate_per_minute=self._settings.ip_rate_per_minute,
            burst=self._settings.ip_rate_burst,
            daily_limit=self._settings.ip_searches_per_day,
            global_rate_per_minute=self._settings.global_rate_per_minute,
            now=now,
        )

    def check_session_open(
        self, client_hash: str, now: datetime | float | None = None
    ) -> None:
        """Consume the source allowance for opening a search session."""
        self._check(
            client_hash,
            scope="session-open-ip",
            rate_per_minute=self._settings.ip_session_open_rate_per_minute,
            burst=self._settings.ip_session_open_burst,
            daily_limit=self._settings.ip_session_opens_per_day,
            global_rate_per_minute=None,
            now=now,
        )

    def _check(
        self,
        client_hash: str,
        *,
        scope: str,
        rate_per_minute: int,
        burst: int,
        daily_limit: int,
        global_rate_per_minute: int | None,
        now: datetime | float | None,
    ) -> None:
        timestamp = utc_timestamp(now)
        day = utc_day(timestamp)
        with self._database.transaction() as connection:
            daily = connection.execute(
                """SELECT request_count FROM rate_daily
                   WHERE scope = ? AND identity_hash = ? AND utc_day = ?""",
                (scope, client_hash, day),
            ).fetchone()
            if daily is not None and daily["request_count"] >= daily_limit:
                raise RateLimited()

            source_tokens = self._available_tokens(
                connection,
                scope=scope,
                identity_hash=client_hash,
                timestamp=timestamp,
                rate_per_minute=rate_per_minute,
                capacity=burst,
            )
            global_tokens = None
            if global_rate_per_minute is not None:
                global_tokens = self._available_tokens(
                    connection,
                    scope="search-global",
                    identity_hash="global",
                    timestamp=timestamp,
                    rate_per_minute=global_rate_per_minute,
                    capacity=global_rate_per_minute,
                )
            if source_tokens < 1.0 or (
                global_tokens is not None and global_tokens < 1.0
            ):
                raise RateLimited()

            self._save_bucket(
                connection, scope, client_hash, source_tokens - 1.0, timestamp
            )
            if global_tokens is not None:
                self._save_bucket(
                    connection,
                    "search-global",
                    "global",
                    global_tokens - 1.0,
                    timestamp,
                )
            connection.execute(
                """INSERT INTO rate_daily(scope, identity_hash, utc_day, request_count)
                   VALUES (?, ?, ?, 1)
                   ON CONFLICT(scope, identity_hash, utc_day)
                   DO UPDATE SET request_count = request_count + 1""",
                (scope, client_hash, day),
            )

    def _load_identity_salt(self) -> bytes:
        candidate = secrets.token_bytes(32)
        with self._database.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO runtime_meta(key, value) VALUES ('client_identity_salt', ?)",
                (candidate,),
            )
            value = connection.execute(
                "SELECT value FROM runtime_meta WHERE key = 'client_identity_salt'"
            ).fetchone()["value"]
        return bytes(value)

    @staticmethod
    def _available_tokens(
        connection,
        *,
        scope: str,
        identity_hash: str,
        timestamp: float,
        rate_per_minute: int,
        capacity: int,
    ) -> float:
        row = connection.execute(
            "SELECT tokens, updated_at FROM rate_buckets WHERE scope = ? AND identity_hash = ?",
            (scope, identity_hash),
        ).fetchone()
        if row is None:
            return float(capacity)
        elapsed = max(0.0, timestamp - row["updated_at"])
        return min(float(capacity), row["tokens"] + elapsed * rate_per_minute / 60.0)

    @staticmethod
    def _save_bucket(
        connection, scope: str, identity_hash: str, tokens: float, timestamp: float
    ) -> None:
        connection.execute(
            """INSERT INTO rate_buckets(scope, identity_hash, tokens, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(scope, identity_hash)
               DO UPDATE SET tokens = excluded.tokens, updated_at = excluded.updated_at""",
            (scope, identity_hash, tokens, timestamp),
        )
