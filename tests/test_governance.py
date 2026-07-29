"""Offline validation of public-service rate, budget, cache, and concurrency controls."""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from icml_paper_search.config import Settings
from icml_paper_search.errors import BudgetExhausted, RateLimited, ServerBusy
from icml_paper_search.governance import (
    AdmissionController,
    EmbeddingCache,
    GovernedQueryEncoder,
    SQLiteBudgetLedger,
    SQLiteRateLimiter,
    SingleFlight,
    embedding_cache_key,
    query_token_upper_bound,
)
from icml_paper_search.state import RuntimeDatabase


def _settings(tmp_path, **overrides) -> Settings:
    values = {
        "data_dir": tmp_path,
        "index_root": tmp_path / "index",
        "state_db_path": tmp_path / "runtime.sqlite3",
        "max_search_depth": 50,
        "ip_rate_per_minute": 6,
        "ip_rate_burst": 3,
        "ip_searches_per_day": 4,
        "global_rate_per_minute": 120,
        "daily_token_budget": 20,
        "monthly_token_budget": 100,
        "embedding_cache_max_entries": 2,
        "embedding_cache_ttl_seconds": 60,
        "provider_concurrency": 2,
    }
    values.update(overrides)
    return Settings(**values)


def test_rate_limit_is_atomic_and_persists_across_restart(tmp_path):
    """Persist IP burst and daily counts in SQLite across object recreation."""
    settings = _settings(tmp_path)
    database = RuntimeDatabase(settings.state_db_path)
    limiter = SQLiteRateLimiter(database, settings)
    client = limiter.hash_identity("203.0.113.10")
    now = datetime(2026, 7, 28, tzinfo=timezone.utc)
    for _ in range(settings.ip_rate_burst):
        limiter.check_search(client, now)
    with pytest.raises(RateLimited):
        limiter.check_search(client, now)

    restarted = SQLiteRateLimiter(RuntimeDatabase(settings.state_db_path), settings)
    assert restarted.hash_identity("203.0.113.10") == client
    with pytest.raises(RateLimited):
        restarted.check_search(client, now)
    restarted.check_search(client, now + timedelta(seconds=10))


def test_session_open_limit_is_persistent_per_ip(tmp_path):
    """Limit session creation by IP burst, minute rate, and UTC daily total."""
    settings = _settings(
        tmp_path,
        ip_session_open_rate_per_minute=3,
        ip_session_open_burst=2,
        ip_session_opens_per_day=3,
    )
    path = settings.state_db_path
    limiter = SQLiteRateLimiter(RuntimeDatabase(path), settings)
    client = limiter.hash_identity("203.0.113.12")
    now = datetime(2026, 7, 28, tzinfo=timezone.utc)

    limiter.check_session_open(client, now)
    limiter.check_session_open(client, now)
    with pytest.raises(RateLimited):
        limiter.check_session_open(client, now)

    restarted = SQLiteRateLimiter(RuntimeDatabase(path), settings)
    assert restarted.hash_identity("203.0.113.12") == client
    restarted.check_session_open(client, now + timedelta(seconds=20))
    with pytest.raises(RateLimited):
        restarted.check_session_open(client, now + timedelta(minutes=1))


def test_budget_reserve_settle_release_and_unknown_usage(tmp_path):
    """Release unsent calls and conservatively charge unknown sent usage."""
    settings = _settings(tmp_path)
    ledger = SQLiteBudgetLedger(RuntimeDatabase(settings.state_db_path), settings)
    now = datetime(2026, 7, 28, tzinfo=timezone.utc)

    released = ledger.reserve(5, now=now)
    assert ledger.settle(released, request_sent=False, now=now) == 0
    unknown = ledger.reserve(7, now=now)
    assert ledger.settle(unknown, request_sent=True, now=now) == 7
    actual = ledger.reserve(5, now=now)
    assert ledger.settle(actual, 3, now=now) == 3
    status = SQLiteBudgetLedger(
        RuntimeDatabase(settings.state_db_path), settings
    ).status(now=now)
    assert status.daily_consumed == 10
    assert status.daily_reserved == 0
    with pytest.raises(BudgetExhausted):
        ledger.reserve(11, now=now)


def test_budget_overrun_is_recorded_and_fails_closed_after_restart(tmp_path):
    """Persist provider overruns and fail closed for the affected periods."""
    settings = _settings(tmp_path, daily_token_budget=20, monthly_token_budget=40)
    path = settings.state_db_path
    now = datetime(2026, 7, 28, tzinfo=timezone.utc)
    ledger = SQLiteBudgetLedger(RuntimeDatabase(path), settings)

    reservation = ledger.reserve(15, now=now)
    assert ledger.settle(reservation, actual_tokens=21, now=now) == 21
    status = ledger.status(now=now)
    assert status.daily_consumed == 21
    assert status.daily_overrun == 6

    restarted = SQLiteBudgetLedger(RuntimeDatabase(path), settings)
    with pytest.raises(BudgetExhausted):
        restarted.reserve(1, now=now)
    with RuntimeDatabase(path).connection() as connection:
        row = connection.execute(
            """SELECT actual_tokens, overrun_tokens FROM budget_reservations
               WHERE reservation_id = ?""",
            (reservation.reservation_id,),
        ).fetchone()
    assert tuple(row) == (21, 6)


def test_query_budget_reservation_is_utf8_upper_bound():
    """Cover UTF-8 bytes and fixed provider protocol overhead in query reservations."""
    ascii_bound = query_token_upper_bound("abc")
    unicode_bound = query_token_upper_bound("衣服")

    assert ascii_bound > len("abc".encode("utf-8"))
    assert unicode_bound - ascii_bound == len("衣服".encode("utf-8")) - 3


def test_cache_uses_hash_key_ttl_and_lru(tmp_path):
    """Do not cache raw queries and retain the two newest entries by TTL/LRU."""
    settings = _settings(tmp_path)
    database = RuntimeDatabase(settings.state_db_path)
    cache = EmbeddingCache(database, settings)
    now = datetime(2026, 7, 28, tzinfo=timezone.utc)
    vector = [0.0] * settings.embedding_dim
    keys = [
        embedding_cache_key(
            f"secret query {index}",
            model=settings.embedding_model,
            dimension=settings.embedding_dim,
        )
        for index in range(3)
    ]
    cache.put(keys[0], vector, now=now)
    cache.put(keys[1], vector, now=now + timedelta(seconds=1))
    assert cache.get(keys[0], now=now + timedelta(seconds=2)) == vector
    cache.put(keys[2], vector, now=now + timedelta(seconds=3))
    assert cache.get(keys[1], now=now + timedelta(seconds=4)) is None
    assert cache.get(keys[0], now=now + timedelta(seconds=61)) is None
    with database.connection() as connection:
        dump = "\n".join(connection.iterdump())
    assert "secret query" not in dump


def test_single_flight_merges_concurrent_operation():
    """Execute one provider operation for concurrent misses on the same key."""
    flight = SingleFlight()
    lock = threading.Lock()
    calls = 0

    def operation() -> int:
        nonlocal calls
        with lock:
            calls += 1
        time.sleep(0.05)
        return 42

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _index: flight.run("same", operation), range(8)))
    assert results == [42] * 8
    assert calls == 1


@dataclass(frozen=True)
class Batch:
    vectors: tuple[tuple[float, ...], ...]
    total_tokens: int


class FakeProvider:
    """Usage-aware provider that records maximum concurrency."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def encode_query_with_usage(self, text: str) -> Batch:
        with self.lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.03)
        with self.lock:
            self.active -= 1
        return Batch(((float(len(text)),) * self.dimension,), 1)

    def encode_documents(self, texts, *, show_progress=False):
        return [[0.0] * self.dimension for _text in texts]

    def close(self) -> None:
        return None


def test_governed_encoder_caches_and_bounds_provider_concurrency(tmp_path):
    """Coalesce identical queries and bound provider in-flight calls across queries."""
    settings = _settings(
        tmp_path,
        daily_token_budget=100,
        monthly_token_budget=200,
        embedding_cache_max_entries=10,
    )
    database = RuntimeDatabase(settings.state_db_path)
    provider = FakeProvider(settings.embedding_dim)
    encoder = GovernedQueryEncoder(
        provider,
        EmbeddingCache(database, settings),
        SQLiteBudgetLedger(database, settings),
        settings,
    )
    queries = ["same"] * 4 + [f"q{index}" for index in range(4)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        vectors = list(executor.map(encoder.encode_query, queries))
    assert all(len(vector) == settings.embedding_dim for vector in vectors)
    assert provider.calls == 5
    assert provider.max_active <= settings.provider_concurrency
    assert encoder.encode_query("same") == vectors[0]
    assert provider.calls == 5


def test_admission_timeout_and_cancellation_release_capacity():
    """Release admission capacity immediately on queue timeout or cancellation."""

    async def scenario() -> None:
        controller = AdmissionController(
            queue_size=2, timeout_seconds=0.02, worker_size=1
        )
        entered = asyncio.Event()
        release = asyncio.Event()

        async def blocking() -> None:
            entered.set()
            await release.wait()

        first = asyncio.create_task(controller.run_search(blocking))
        await entered.wait()
        with pytest.raises(ServerBusy):
            await controller.run_search(lambda: asyncio.sleep(0))
        assert controller.admitted_count == 1

        waiting = asyncio.create_task(
            controller.run_search(lambda: asyncio.sleep(0))
        )
        await asyncio.sleep(0)
        assert controller.admitted_count == 2
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert controller.admitted_count == 1
        release.set()
        await first
        assert controller.admitted_count == 0
        assert await controller.run_search(lambda: asyncio.sleep(0, result=42)) == 42

    asyncio.run(scenario())


def test_admission_rejects_over_64_without_unbounded_waiters():
    """Cap active-plus-waiting at 64 and reject excess bursts before queueing."""

    async def scenario() -> None:
        controller = AdmissionController(
            queue_size=64, timeout_seconds=2.0, worker_size=1
        )
        release = asyncio.Event()
        completed = 0

        async def blocking() -> None:
            nonlocal completed
            await release.wait()
            completed += 1

        admitted = [
            asyncio.create_task(controller.run_search(blocking)) for _ in range(64)
        ]
        for _ in range(100):
            if controller.admitted_count == 64:
                break
            await asyncio.sleep(0)
        assert controller.admitted_count == 64

        async def rejected() -> None:
            with pytest.raises(ServerBusy):
                await controller.run_search(lambda: asyncio.sleep(0))

        await asyncio.gather(*(rejected() for _ in range(40)))
        assert controller.admitted_count == 64
        assert controller.peak_admitted_count == 64

        for task in admitted[-8:]:
            task.cancel()
        await asyncio.gather(*admitted[-8:], return_exceptions=True)
        assert controller.admitted_count == 56

        replacements = [
            asyncio.create_task(controller.run_search(blocking)) for _ in range(8)
        ]
        for _ in range(100):
            if controller.admitted_count == 64:
                break
            await asyncio.sleep(0)
        assert controller.admitted_count == 64

        release.set()
        await asyncio.gather(*admitted[:-8], *replacements)
        assert completed == 64
        assert controller.admitted_count == 0
        assert controller.peak_admitted_count == 64

    asyncio.run(scenario())
