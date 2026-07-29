"""Persistent DashScope request and input-token limits."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime

from ..config import Settings, get_settings
from ..domain.errors import RateLimited
from .clock import utc_minute, utc_timestamp
from .database import RuntimeDatabase


@dataclass(frozen=True, slots=True)
class ProviderReservation:
    """Capacity reserved for one embedding-provider request."""

    reservation_id: str
    reserved_tokens: int
    minute_key: str


@dataclass(frozen=True, slots=True)
class ProviderQuotaStatus:
    """Request and token usage for one UTC minute."""

    request_count: int
    request_limit: int
    consumed_tokens: int
    reserved_tokens: int
    token_limit: int


class SQLiteProviderQuota:
    """Atomically enforce provider RPM and TPM limits."""

    def __init__(
        self, database: RuntimeDatabase, settings: Settings | None = None
    ) -> None:
        self._database = database
        self._settings = settings or get_settings()
        self._database.initialize()

    def reserve(
        self, tokens: int, *, now: datetime | float | None = None
    ) -> ProviderReservation:
        """Reserve one request and its estimated input tokens."""
        if tokens < 1:
            raise ValueError("reserved tokens must be a positive integer")
        timestamp = utc_timestamp(now)
        minute_key = utc_minute(timestamp)
        reservation_id = f"req_{secrets.token_urlsafe(18)}"
        with self._database.transaction() as connection:
            request_count, consumed, reserved = self._window_row(
                connection, minute_key, timestamp
            )
            if request_count >= self._settings.provider_requests_per_minute:
                raise RateLimited()
            if consumed + reserved + tokens > self._settings.provider_tokens_per_minute:
                raise RateLimited()
            connection.execute(
                """UPDATE provider_windows
                   SET request_count = request_count + 1,
                       reserved_tokens = reserved_tokens + ?, updated_at = ?
                   WHERE minute_key = ?""",
                (tokens, timestamp, minute_key),
            )
            connection.execute(
                """INSERT INTO provider_reservations(
                       reservation_id, minute_key, reserved_tokens, status, created_at
                   ) VALUES (?, ?, ?, 'reserved', ?)""",
                (reservation_id, minute_key, tokens, timestamp),
            )
        return ProviderReservation(reservation_id, tokens, minute_key)

    def settle(
        self,
        reservation: ProviderReservation | str,
        actual_tokens: int | None = None,
        *,
        request_sent: bool = True,
        now: datetime | float | None = None,
    ) -> int:
        """Replace a reservation with actual usage or release an unsent request."""
        if actual_tokens is not None and actual_tokens < 0:
            raise ValueError("actual tokens must not be negative")
        timestamp = utc_timestamp(now)
        reservation_id = (
            reservation.reservation_id
            if isinstance(reservation, ProviderReservation)
            else reservation
        )
        with self._database.transaction() as connection:
            row = connection.execute(
                """SELECT minute_key, reserved_tokens, status
                   FROM provider_reservations WHERE reservation_id = ?""",
                (reservation_id,),
            ).fetchone()
            if row is None or row["status"] != "reserved":
                raise ValueError(
                    "the provider reservation does not exist or is settled"
                )
            charged = (
                int(actual_tokens)
                if actual_tokens is not None
                else (row["reserved_tokens"] if request_sent else 0)
            )
            connection.execute(
                """UPDATE provider_windows
                   SET request_count = request_count - ?,
                       reserved_tokens = reserved_tokens - ?,
                       consumed_tokens = consumed_tokens + ?, updated_at = ?
                   WHERE minute_key = ?""",
                (
                    0 if request_sent else 1,
                    row["reserved_tokens"],
                    charged,
                    timestamp,
                    row["minute_key"],
                ),
            )
            connection.execute(
                """UPDATE provider_reservations
                   SET actual_tokens = ?, status = 'settled', settled_at = ?
                   WHERE reservation_id = ?""",
                (charged, timestamp, reservation_id),
            )
        return charged

    def status(self, *, now: datetime | float | None = None) -> ProviderQuotaStatus:
        """Read provider usage for the current UTC minute."""
        timestamp = utc_timestamp(now)
        minute_key = utc_minute(timestamp)
        with self._database.transaction() as connection:
            requests, consumed, reserved = self._window_row(
                connection, minute_key, timestamp
            )
        return ProviderQuotaStatus(
            request_count=requests,
            request_limit=self._settings.provider_requests_per_minute,
            consumed_tokens=consumed,
            reserved_tokens=reserved,
            token_limit=self._settings.provider_tokens_per_minute,
        )

    @staticmethod
    def _window_row(
        connection, minute_key: str, timestamp: float
    ) -> tuple[int, int, int]:
        connection.execute(
            """INSERT OR IGNORE INTO provider_windows(
                   minute_key, request_count, consumed_tokens, reserved_tokens, updated_at
               ) VALUES (?, 0, 0, 0, ?)""",
            (minute_key, timestamp),
        )
        row = connection.execute(
            """SELECT request_count, consumed_tokens, reserved_tokens
               FROM provider_windows WHERE minute_key = ?""",
            (minute_key,),
        ).fetchone()
        return row["request_count"], row["consumed_tokens"], row["reserved_tokens"]
