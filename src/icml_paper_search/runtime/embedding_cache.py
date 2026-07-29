"""Persistent query-embedding cache and governed provider access."""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, TypeVar

from ..config import Settings, get_settings
from ..domain.errors import EmbeddingConfigurationError
from .clock import utc_timestamp
from .database import RuntimeDatabase, hash_query
from .provider_quota import SQLiteProviderQuota

T = TypeVar("T")
QUERY_TOKEN_PROTOCOL_OVERHEAD = 16


def embedding_cache_key(
    query: str,
    *,
    provider: str = "dashscope",
    region: str = "cn-beijing",
    model: str,
    dimension: int,
    text_type: str = "query",
) -> str:
    """Generate a stable cache key that does not contain the raw query."""
    normalized = " ".join(query.split())
    material = "\0".join(
        (provider, region, model, str(dimension), text_type, hash_query(normalized))
    )
    return hash_query(material)


def query_token_upper_bound(text: str) -> int:
    """Reserve query tokens using UTF-8 bytes plus fixed protocol overhead."""
    return max(1, len(text.encode("utf-8")) + QUERY_TOKEN_PROTOCOL_OVERHEAD)


class EmbeddingCache:
    """Cache successful query vectors with bounded TTL/LRU eviction."""

    def __init__(
        self, database: RuntimeDatabase, settings: Settings | None = None
    ) -> None:
        self._database = database
        self._settings = settings or get_settings()
        self._database.initialize()

    def get(
        self, cache_key: str, *, now: datetime | float | None = None
    ) -> list[float] | None:
        """Read an unexpired vector and refresh its LRU timestamp."""
        if self._settings.embedding_cache_max_entries == 0:
            return None
        timestamp = utc_timestamp(now)
        with self._database.transaction() as connection:
            row = connection.execute(
                "SELECT vector, dimension, expires_at FROM embedding_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is None:
                return None
            if row["expires_at"] <= timestamp:
                connection.execute(
                    "DELETE FROM embedding_cache WHERE cache_key = ?", (cache_key,)
                )
                return None
            try:
                raw = json.loads(bytes(row["vector"]).decode("utf-8"))
                vector = [float(value) for value in raw]
            except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
                raw, vector = None, []
            if (
                not isinstance(raw, list)
                or len(vector) != row["dimension"]
                or not all(math.isfinite(value) for value in vector)
            ):
                connection.execute(
                    "DELETE FROM embedding_cache WHERE cache_key = ?", (cache_key,)
                )
                return None
            connection.execute(
                "UPDATE embedding_cache SET last_access_at = ? WHERE cache_key = ?",
                (timestamp, cache_key),
            )
            return vector

    def put(
        self,
        cache_key: str,
        vector: Sequence[float],
        *,
        token_count: int = 0,
        now: datetime | float | None = None,
    ) -> None:
        """Write a successful vector and enforce TTL/LRU limits."""
        max_entries = self._settings.embedding_cache_max_entries
        if max_entries == 0:
            return
        values = [float(value) for value in vector]
        if len(values) != self._settings.embedding_dim or not all(
            math.isfinite(value) for value in values
        ):
            raise ValueError("cached vector dimensions or values are invalid")
        timestamp = utc_timestamp(now)
        expires_at = timestamp + self._settings.embedding_cache_ttl_seconds
        payload = json.dumps(values, separators=(",", ":")).encode("utf-8")
        with self._database.transaction() as connection:
            connection.execute(
                "DELETE FROM embedding_cache WHERE expires_at <= ?", (timestamp,)
            )
            connection.execute(
                """INSERT INTO embedding_cache(
                       cache_key, vector, dimension, token_count,
                       created_at, last_access_at, expires_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(cache_key) DO UPDATE SET
                       vector = excluded.vector, dimension = excluded.dimension,
                       token_count = excluded.token_count, created_at = excluded.created_at,
                       last_access_at = excluded.last_access_at, expires_at = excluded.expires_at""",
                (
                    cache_key,
                    payload,
                    len(values),
                    max(0, token_count),
                    timestamp,
                    timestamp,
                    expires_at,
                ),
            )
            overflow = connection.execute(
                "SELECT MAX(COUNT(*) - ?, 0) FROM embedding_cache", (max_entries,)
            ).fetchone()[0]
            if overflow:
                connection.execute(
                    """DELETE FROM embedding_cache WHERE cache_key IN (
                           SELECT cache_key FROM embedding_cache
                           ORDER BY last_access_at, created_at, cache_key LIMIT ?
                       )""",
                    (overflow,),
                )

    def prune(self, *, now: datetime | float | None = None) -> int:
        """Remove expired cache entries."""
        with self._database.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM embedding_cache WHERE expires_at <= ?",
                (utc_timestamp(now),),
            )
            return cursor.rowcount


@dataclass(slots=True)
class _Flight[T]:
    event: threading.Event
    result: T | None = None
    error: BaseException | None = None


class SingleFlight:
    """Coalesce identical in-process cache misses."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._flights: dict[str, _Flight[Any]] = {}

    def run(self, key: str, operation: Callable[[], T]) -> T:
        """Run one operation while callers for the same key await its result."""
        with self._lock:
            flight = self._flights.get(key)
            owner = flight is None
            if flight is None:
                flight = _Flight(event=threading.Event())
                self._flights[key] = flight
        if owner:
            try:
                flight.result = operation()
            except BaseException as exc:  # noqa: BLE001 - wake waiters before re-raising
                flight.error = exc
            finally:
                flight.event.set()
                with self._lock:
                    self._flights.pop(key, None)
        else:
            flight.event.wait()
        if flight.error is not None:
            raise flight.error
        return flight.result  # type: ignore[return-value]


class UsageAwareEncoder(Protocol):
    """Provider interface required for governed query embeddings."""

    def encode_query_with_usage(self, text: str) -> Any: ...

    def encode_documents(
        self, texts: Sequence[str], *, show_progress: bool = False
    ) -> list[list[float]]: ...

    def close(self) -> None: ...


class GovernedQueryEncoder:
    """Compose cache, request coalescing, provider quota, and concurrency."""

    def __init__(
        self,
        provider: UsageAwareEncoder,
        cache: EmbeddingCache,
        quota: SQLiteProviderQuota,
        settings: Settings | None = None,
        *,
        single_flight: SingleFlight | None = None,
    ) -> None:
        self._provider = provider
        self._cache = cache
        self._quota = quota
        self._settings = settings or get_settings()
        self._single_flight = single_flight or SingleFlight()
        self._provider_slots = threading.BoundedSemaphore(
            self._settings.provider_concurrency
        )

    def encode_query(self, text: str) -> list[float]:
        """Return a cached vector or make one governed provider call."""
        key = embedding_cache_key(
            text,
            model=self._settings.embedding_model,
            dimension=self._settings.embedding_dim,
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        return self._single_flight.run(key, lambda: self._encode_cache_miss(key, text))

    def encode_documents(
        self, texts: Sequence[str], *, show_progress: bool = False
    ) -> list[list[float]]:
        """Limit concurrent document-embedding calls."""
        with self._provider_slots:
            return self._provider.encode_documents(texts, show_progress=show_progress)

    def close(self) -> None:
        """Close the underlying provider."""
        self._provider.close()

    def _encode_cache_miss(self, key: str, text: str) -> list[float]:
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        reservation = self._quota.reserve(query_token_upper_bound(text))
        try:
            with self._provider_slots:
                batch = self._provider.encode_query_with_usage(text)
        except EmbeddingConfigurationError:
            self._quota.settle(reservation, request_sent=False)
            raise
        except BaseException:
            self._quota.settle(reservation, request_sent=True)
            raise

        vectors = getattr(batch, "vectors", None)
        total_tokens = getattr(batch, "total_tokens", None)
        if not vectors or len(vectors) != 1 or total_tokens is None:
            self._quota.settle(reservation, request_sent=True)
            raise ValueError("the provider usage contract is invalid")
        vector = [float(value) for value in vectors[0]]
        self._quota.settle(reservation, int(total_tokens), request_sent=True)
        self._cache.put(key, vector, token_count=int(total_tokens))
        return vector
