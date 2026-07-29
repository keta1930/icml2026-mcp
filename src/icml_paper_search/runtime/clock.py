"""UTC time helpers shared by persistent runtime controls."""

from __future__ import annotations

from datetime import UTC, datetime


def utc_timestamp(value: datetime | float | None = None) -> float:
    """Convert a supported time value to a UTC timestamp."""
    if value is None:
        return datetime.now(UTC).timestamp()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.timestamp()
    return float(value)


def utc_day(timestamp: float) -> str:
    """Return the UTC day containing a timestamp."""
    return datetime.fromtimestamp(timestamp, UTC).strftime("%Y-%m-%d")


def utc_minute(timestamp: float) -> str:
    """Return the UTC minute containing a timestamp."""
    return datetime.fromtimestamp(timestamp, UTC).strftime("%Y-%m-%dT%H:%M")
