"""公共 MCP 的持久限流、预算、缓存与有界并发控制。"""

from __future__ import annotations

import asyncio
import json
import math
import secrets
import threading
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Generic, Protocol, TypeVar

import anyio

from .config import Settings, get_settings
from .errors import (
    BudgetExhausted,
    EmbeddingConfigurationError,
    RateLimited,
    ServerBusy,
)
from .state import RuntimeDatabase, hash_client_identity, hash_query

T = TypeVar("T")

QUERY_TOKEN_PROTOCOL_OVERHEAD = 16


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime | float | None) -> float:
    if value is None:
        return _now().timestamp()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.timestamp()
    return float(value)


def _period_keys(timestamp: float) -> tuple[str, str]:
    value = datetime.fromtimestamp(timestamp, timezone.utc)
    return value.strftime("%Y-%m-%d"), value.strftime("%Y-%m")


class SQLiteRateLimiter:
    """用 SQLite token bucket 与 UTC 日计数实施原子限流。"""

    def __init__(self, database: RuntimeDatabase, settings: Settings | None = None) -> None:
        self._database = database
        self._settings = settings or get_settings()
        self._database.initialize()
        self._identity_salt = self._load_identity_salt()

    def hash_identity(self, identity: str) -> str:
        """用数据库持久 salt 摘要来源身份，使重启后限流连续。"""
        return hash_client_identity(identity, self._identity_salt)

    def check_search(
        self, client_hash: str, now: datetime | float | None = None
    ) -> None:
        """通过时原子扣减 IP/global token 并增加 IP 日计数。"""
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
        """在创建 session 前扣减持久化的来源 IP 额度。"""
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
        """在单个事务中检查并扣减一类请求的所有额度。"""
        timestamp = _timestamp(now)
        utc_day, _ = _period_keys(timestamp)
        with self._database.transaction() as connection:
            daily = connection.execute(
                """SELECT request_count FROM rate_daily
                   WHERE scope = ? AND identity_hash = ? AND utc_day = ?""",
                (scope, client_hash, utc_day),
            ).fetchone()
            if daily is not None and daily["request_count"] >= daily_limit:
                raise RateLimited()

            ip_tokens = self._available_tokens(
                connection,
                scope=scope,
                identity_hash=client_hash,
                timestamp=timestamp,
                rate_per_minute=rate_per_minute,
                capacity=burst,
            )
            global_tokens: float | None = None
            if global_rate_per_minute is not None:
                global_tokens = self._available_tokens(
                    connection,
                    scope="search-global",
                    identity_hash="global",
                    timestamp=timestamp,
                    rate_per_minute=global_rate_per_minute,
                    capacity=global_rate_per_minute,
                )
            if ip_tokens < 1.0 or (
                global_tokens is not None and global_tokens < 1.0
            ):
                raise RateLimited()

            self._save_bucket(connection, scope, client_hash, ip_tokens - 1.0, timestamp)
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
                (scope, client_hash, utc_day),
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


@dataclass(frozen=True, slots=True)
class BudgetReservation:
    """表示已在日/月额度中占位的 provider 调用。"""

    reservation_id: str
    reserved_tokens: int


@dataclass(frozen=True, slots=True)
class BudgetStatus:
    """返回当前 scope 的在线预算快照。"""

    daily_consumed: int
    daily_reserved: int
    daily_overrun: int
    daily_limit: int
    monthly_consumed: int
    monthly_reserved: int
    monthly_overrun: int
    monthly_limit: int


class SQLiteBudgetLedger:
    """在 provider 调用前预留并在响应后保守对账 Token。"""

    def __init__(self, database: RuntimeDatabase, settings: Settings | None = None) -> None:
        self._database = database
        self._settings = settings or get_settings()
        self._database.initialize()

    def reserve(
        self,
        tokens: int,
        *,
        scope: str = "online",
        now: datetime | float | None = None,
    ) -> BudgetReservation:
        """同时满足日/月硬上限时创建原子预算预留。"""
        if tokens < 1:
            raise ValueError("预留 Token 必须为正整数")
        timestamp = _timestamp(now)
        utc_day, utc_month = _period_keys(timestamp)
        reservation_id = f"res_{secrets.token_urlsafe(18)}"
        with self._database.transaction() as connection:
            day = self._ledger_row(connection, scope, "day", utc_day, timestamp)
            month = self._ledger_row(connection, scope, "month", utc_month, timestamp)
            if day[2] or month[2]:
                raise BudgetExhausted()
            if day[0] + day[1] + tokens > self._settings.daily_token_budget:
                raise BudgetExhausted()
            if month[0] + month[1] + tokens > self._settings.monthly_token_budget:
                raise BudgetExhausted()
            for kind, key in (("day", utc_day), ("month", utc_month)):
                connection.execute(
                    """UPDATE budget_ledger
                       SET reserved_tokens = reserved_tokens + ?, updated_at = ?
                       WHERE scope = ? AND period_kind = ? AND period_key = ?""",
                    (tokens, timestamp, scope, kind, key),
                )
            connection.execute(
                """INSERT INTO budget_reservations(
                       reservation_id, scope, utc_day, utc_month, reserved_tokens,
                       status, created_at
                   ) VALUES (?, ?, ?, ?, ?, 'reserved', ?)""",
                (reservation_id, scope, utc_day, utc_month, tokens, timestamp),
            )
        return BudgetReservation(reservation_id, tokens)

    def settle(
        self,
        reservation: BudgetReservation | str,
        actual_tokens: int | None = None,
        *,
        request_sent: bool = True,
        now: datetime | float | None = None,
    ) -> int:
        """按实际 usage 对账；usage 未知且已发送时按预留计费。"""
        timestamp = _timestamp(now)
        reservation_id = (
            reservation.reservation_id
            if isinstance(reservation, BudgetReservation)
            else reservation
        )
        with self._database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM budget_reservations WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()
            if row is None or row["status"] != "reserved":
                raise ValueError("预算预留不存在或已结算")
            charged = row["reserved_tokens"] if request_sent and actual_tokens is None else 0
            actual: int | None = None
            if actual_tokens is not None:
                if actual_tokens < 0:
                    raise ValueError("实际 Token 不能为负数")
                actual = int(actual_tokens)
                charged = actual
            overrun = max(0, charged - row["reserved_tokens"]) if actual is not None else 0
            status = "settled" if request_sent or actual_tokens is not None else "released"
            for kind, key in (("day", row["utc_day"]), ("month", row["utc_month"])):
                connection.execute(
                    """UPDATE budget_ledger
                       SET reserved_tokens = reserved_tokens - ?,
                           consumed_tokens = consumed_tokens + ?,
                           overrun_tokens = overrun_tokens + ?, updated_at = ?
                       WHERE scope = ? AND period_kind = ? AND period_key = ?""",
                    (
                        row["reserved_tokens"],
                        charged,
                        overrun,
                        timestamp,
                        row["scope"],
                        kind,
                        key,
                    ),
                )
            connection.execute(
                """UPDATE budget_reservations
                   SET settled_tokens = ?, actual_tokens = ?, overrun_tokens = ?,
                       status = ?, settled_at = ?
                   WHERE reservation_id = ?""",
                (charged, actual, overrun, status, timestamp, reservation_id),
            )
            return charged

    def status(
        self, *, scope: str = "online", now: datetime | float | None = None
    ) -> BudgetStatus:
        """读取当前 UTC 日/月的已消费与已预留额度。"""
        timestamp = _timestamp(now)
        utc_day, utc_month = _period_keys(timestamp)
        with self._database.transaction() as connection:
            day = self._ledger_row(connection, scope, "day", utc_day, timestamp)
            month = self._ledger_row(connection, scope, "month", utc_month, timestamp)
        return BudgetStatus(
            daily_consumed=day[0],
            daily_reserved=day[1],
            daily_overrun=day[2],
            daily_limit=self._settings.daily_token_budget,
            monthly_consumed=month[0],
            monthly_reserved=month[1],
            monthly_overrun=month[2],
            monthly_limit=self._settings.monthly_token_budget,
        )

    @staticmethod
    def _ledger_row(connection, scope: str, kind: str, key: str, timestamp: float):
        connection.execute(
            """INSERT OR IGNORE INTO budget_ledger(
                   scope, period_kind, period_key, consumed_tokens, reserved_tokens,
                   overrun_tokens, updated_at
               ) VALUES (?, ?, ?, 0, 0, 0, ?)""",
            (scope, kind, key, timestamp),
        )
        row = connection.execute(
            """SELECT consumed_tokens, reserved_tokens, overrun_tokens FROM budget_ledger
               WHERE scope = ? AND period_kind = ? AND period_key = ?""",
            (scope, kind, key),
        ).fetchone()
        return row["consumed_tokens"], row["reserved_tokens"], row["overrun_tokens"]


def embedding_cache_key(
    query: str,
    *,
    provider: str = "dashscope",
    region: str = "cn-beijing",
    model: str,
    dimension: int,
    text_type: str = "query",
) -> str:
    """生成不含原 query 的稳定 cache key。"""
    normalized = " ".join(query.split())
    material = "\0".join(
        (provider, region, model, str(dimension), text_type, hash_query(normalized))
    )
    return hash_query(material)


def query_token_upper_bound(text: str) -> int:
    """按 UTF-8 byte 数加固定协议开销预留 query Token。"""
    return max(1, len(text.encode("utf-8")) + QUERY_TOKEN_PROTOCOL_OVERHEAD)


class EmbeddingCache:
    """只缓存成功 query vector，并按 TTL/LRU 有界回收。"""

    def __init__(self, database: RuntimeDatabase, settings: Settings | None = None) -> None:
        self._database = database
        self._settings = settings or get_settings()
        self._database.initialize()

    def get(
        self, cache_key: str, *, now: datetime | float | None = None
    ) -> list[float] | None:
        """读取未过期向量并刷新 LRU 时间。"""
        if self._settings.embedding_cache_max_entries == 0:
            return None
        timestamp = _timestamp(now)
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
            vector: Any = None
            try:
                vector = json.loads(bytes(row["vector"]).decode("utf-8"))
                parsed = [float(value) for value in vector]
            except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
                parsed = []
            if (
                not isinstance(vector, list)
                or len(parsed) != row["dimension"]
                or not all(math.isfinite(value) for value in parsed)
            ):
                connection.execute(
                    "DELETE FROM embedding_cache WHERE cache_key = ?", (cache_key,)
                )
                return None
            connection.execute(
                "UPDATE embedding_cache SET last_access_at = ? WHERE cache_key = ?",
                (timestamp, cache_key),
            )
            return parsed

    def put(
        self,
        cache_key: str,
        vector: Sequence[float],
        *,
        token_count: int = 0,
        now: datetime | float | None = None,
    ) -> None:
        """写入成功向量并在同一事务内清理 TTL/LRU。"""
        max_entries = self._settings.embedding_cache_max_entries
        if max_entries == 0:
            return
        values = [float(value) for value in vector]
        if len(values) != self._settings.embedding_dim or not all(
            math.isfinite(value) for value in values
        ):
            raise ValueError("缓存向量维度或数值无效")
        timestamp = _timestamp(now)
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
        """显式清理过期 cache。"""
        with self._database.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM embedding_cache WHERE expires_at <= ?", (_timestamp(now),)
            )
            return cursor.rowcount


@dataclass(slots=True)
class _Flight(Generic[T]):
    event: threading.Event
    result: T | None = None
    error: BaseException | None = None


class SingleFlight:
    """合并进程内相同 cache miss，让等待线程共享同一结果。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._flights: dict[str, _Flight[Any]] = {}

    def run(self, key: str, operation: Callable[[], T]) -> T:
        """由首个线程执行 operation，其余线程等待同一结果。"""
        with self._lock:
            flight = self._flights.get(key)
            owner = flight is None
            if flight is None:
                flight = _Flight(event=threading.Event())
                self._flights[key] = flight
        if owner:
            try:
                flight.result = operation()
            except BaseException as exc:
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
    """描述治理包装器需要的同步 provider 接口。"""

    def encode_query_with_usage(self, text: str) -> Any: ...

    def encode_documents(
        self, texts: Sequence[str], *, show_progress: bool = False
    ) -> list[list[float]]: ...

    def close(self) -> None: ...


class GovernedQueryEncoder:
    """串接 cache、single-flight、预算与 provider 并发控制。"""

    def __init__(
        self,
        provider: UsageAwareEncoder,
        cache: EmbeddingCache,
        budget: SQLiteBudgetLedger,
        settings: Settings | None = None,
        *,
        single_flight: SingleFlight | None = None,
    ) -> None:
        self._provider = provider
        self._cache = cache
        self._budget = budget
        self._settings = settings or get_settings()
        self._single_flight = single_flight or SingleFlight()
        self._provider_slots = threading.BoundedSemaphore(
            self._settings.provider_concurrency
        )

    def encode_query(self, text: str) -> list[float]:
        """优先命中持久 cache，否则合并请求后受预算约束地调用 provider。"""
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
        """建库由 CLI 独立预算约束，本包装器只限制 provider 并发。"""
        with self._provider_slots:
            return self._provider.encode_documents(texts, show_progress=show_progress)

    def close(self) -> None:
        """关闭底层 provider 的线程资源。"""
        self._provider.close()

    def _encode_cache_miss(self, key: str, text: str) -> list[float]:
        # 等待 single-flight 时可能已有另一个刚完成的请求写入 cache。
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        reservation = self._budget.reserve(query_token_upper_bound(text))
        try:
            with self._provider_slots:
                batch = self._provider.encode_query_with_usage(text)
        except EmbeddingConfigurationError:
            self._budget.settle(reservation, request_sent=False)
            raise
        except BaseException:
            self._budget.settle(reservation, request_sent=True)
            raise

        vectors = getattr(batch, "vectors", None)
        total_tokens = getattr(batch, "total_tokens", None)
        if not vectors or len(vectors) != 1 or total_tokens is None:
            self._budget.settle(reservation, request_sent=True)
            raise ValueError("provider usage 契约无效")
        vector = [float(value) for value in vectors[0]]
        self._budget.settle(reservation, int(total_tokens), request_sent=True)
        self._cache.put(key, vector, token_count=int(total_tokens))
        return vector


class AdmissionController:
    """限制 active+waiting search 数量，并给等待设置硬超时。"""

    def __init__(
        self, queue_size: int, timeout_seconds: float, *, worker_size: int | None = None
    ) -> None:
        active_size = worker_size if worker_size is not None else queue_size
        if queue_size < 1 or active_size < 1 or timeout_seconds <= 0:
            raise ValueError("queue_size、worker_size 与 timeout_seconds 必须为正数")
        if active_size > queue_size:
            raise ValueError("worker_size 不能大于 queue_size")
        self._capacity = queue_size
        self._workers = asyncio.Semaphore(active_size)
        self._timeout_seconds = timeout_seconds
        self._admitted = 0
        self._peak_admitted = 0

    @property
    def admitted_count(self) -> int:
        """返回当前 active+waiting 请求数。"""
        return self._admitted

    @property
    def peak_admitted_count(self) -> int:
        """返回进程内观测到的最大 admission 请求数。"""
        return self._peak_admitted

    async def run_search(self, operation: Callable[[], Awaitable[T]]) -> T:
        """在 admission slot 中执行 async 搜索操作。"""
        # FastMCP 使用单一 event loop；首个 await 前的计数与预留是原子的。
        if self._admitted >= self._capacity:
            raise ServerBusy()
        self._admitted += 1
        self._peak_admitted = max(self._peak_admitted, self._admitted)
        worker_acquired = False
        try:
            try:
                await asyncio.wait_for(
                    self._workers.acquire(), self._timeout_seconds
                )
                worker_acquired = True
            except TimeoutError as exc:
                raise ServerBusy() from exc
            return await operation()
        finally:
            if worker_acquired:
                self._workers.release()
            self._admitted -= 1


@dataclass(slots=True)
class ThreadLimiters:
    """隔离慢搜索与轻量状态操作使用的 worker token。"""

    search: anyio.CapacityLimiter
    state: anyio.CapacityLimiter
    chroma: anyio.CapacityLimiter

    @classmethod
    def from_settings(cls, settings: Settings) -> ThreadLimiters:
        """按单进程容量构造相互独立的 AnyIO limiter。"""
        return cls(
            search=anyio.CapacityLimiter(max(settings.provider_concurrency, 4)),
            state=anyio.CapacityLimiter(16),
            chroma=anyio.CapacityLimiter(settings.chroma_concurrency),
        )
