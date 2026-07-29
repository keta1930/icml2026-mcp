"""Validate persistent limits, provider quotas, caching, and admission control."""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from icml_paper_search.config import Settings
from icml_paper_search.domain.errors import RateLimited, ServerBusy
from icml_paper_search.runtime.concurrency import AdmissionController
from icml_paper_search.runtime.database import RuntimeDatabase
from icml_paper_search.runtime.embedding_cache import (
    EmbeddingCache,
    GovernedQueryEncoder,
    SingleFlight,
    embedding_cache_key,
    query_token_upper_bound,
)
from icml_paper_search.runtime.provider_quota import SQLiteProviderQuota
from icml_paper_search.runtime.rate_limits import SQLiteRateLimiter


def _settings(tmp_path, **overrides) -> Settings:
    values = {
        "data_dir": tmp_path,
        "index_root": tmp_path / "index",
        "state_db_path": tmp_path / "runtime.sqlite3",
        "max_search_depth": 50,
        "ip_rate_per_minute": 6,
        "ip_rate_burst": 3,
        "ip_searches_per_day": 4,
        "global_rate_per_minute": 3,
        "provider_requests_per_minute": 4,
        "provider_tokens_per_minute": 100,
        "embedding_cache_max_entries": 2,
        "embedding_cache_ttl_seconds": 60,
        "provider_concurrency": 2,
    }
    values.update(overrides)
    return Settings(**values)


def test_rate_limit_is_atomic_and_persists_across_restart(tmp_path):
    settings = _settings(tmp_path)
    path = settings.state_db_path
    limiter = SQLiteRateLimiter(RuntimeDatabase(path), settings)
    client = limiter.hash_identity("203.0.113.10")
    now = datetime(2026, 7, 28, tzinfo=UTC)
    for _ in range(settings.ip_rate_burst):
        limiter.check_search(client, now)
    with pytest.raises(RateLimited):
        limiter.check_search(client, now)

    restarted = SQLiteRateLimiter(RuntimeDatabase(path), settings)
    assert restarted.hash_identity("203.0.113.10") == client
    with pytest.raises(RateLimited):
        restarted.check_search(client, now)
    restarted.check_search(client, now + timedelta(seconds=20))


def test_provider_quota_enforces_rpm_tpm_and_releases_unsent_calls(tmp_path):
    settings = _settings(
        tmp_path,
        provider_requests_per_minute=3,
        provider_tokens_per_minute=20,
    )
    path = settings.state_db_path
    quota = SQLiteProviderQuota(RuntimeDatabase(path), settings)
    now = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)

    released = quota.reserve(8, now=now)
    assert quota.settle(released, request_sent=False, now=now) == 0
    sent = quota.reserve(8, now=now)
    assert quota.settle(sent, actual_tokens=5, now=now) == 5
    pending = quota.reserve(10, now=now)
    status = SQLiteProviderQuota(RuntimeDatabase(path), settings).status(now=now)
    assert (status.request_count, status.consumed_tokens, status.reserved_tokens) == (
        2,
        5,
        10,
    )
    with pytest.raises(RateLimited):
        quota.reserve(6, now=now)

    quota.settle(pending, request_sent=True, now=now)
    final = quota.status(now=now)
    assert (final.request_count, final.consumed_tokens, final.reserved_tokens) == (
        2,
        15,
        0,
    )


def test_provider_quota_rolls_over_on_utc_minute(tmp_path):
    settings = _settings(
        tmp_path,
        provider_requests_per_minute=1,
        provider_tokens_per_minute=10,
        global_rate_per_minute=1,
    )
    quota = SQLiteProviderQuota(RuntimeDatabase(settings.state_db_path), settings)
    now = datetime(2026, 7, 28, 12, 0, 59, tzinfo=UTC)
    quota.reserve(10, now=now)
    with pytest.raises(RateLimited):
        quota.reserve(1, now=now)
    assert quota.reserve(10, now=now + timedelta(seconds=1)).minute_key.endswith(
        "12:01"
    )


def test_query_token_reservation_is_utf8_upper_bound():
    ascii_bound = query_token_upper_bound("abc")
    unicode_bound = query_token_upper_bound("衣服")
    assert ascii_bound > 3
    assert unicode_bound - ascii_bound == len("衣服".encode()) - 3


def test_cache_uses_hash_key_ttl_and_lru(tmp_path):
    settings = _settings(tmp_path)
    database = RuntimeDatabase(settings.state_db_path)
    cache = EmbeddingCache(database, settings)
    now = datetime(2026, 7, 28, tzinfo=UTC)
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
        results = list(executor.map(lambda _: flight.run("same", operation), range(8)))
    assert results == [42] * 8
    assert calls == 1


@dataclass(frozen=True)
class Batch:
    vectors: tuple[tuple[float, ...], ...]
    total_tokens: int


class FakeProvider:
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
        return [[0.0] * self.dimension for _ in texts]

    def close(self) -> None:
        return None


def test_governed_encoder_caches_coalesces_and_bounds_provider(tmp_path):
    settings = _settings(
        tmp_path,
        provider_requests_per_minute=100,
        provider_tokens_per_minute=10_000,
        embedding_cache_max_entries=10,
    )
    database = RuntimeDatabase(settings.state_db_path)
    provider = FakeProvider(settings.embedding_dim)
    encoder = GovernedQueryEncoder(
        provider,
        EmbeddingCache(database, settings),
        SQLiteProviderQuota(database, settings),
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


def test_admission_rejects_over_capacity_and_releases_slots():
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
        waiting = asyncio.create_task(controller.run_search(lambda: asyncio.sleep(0)))
        await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        release.set()
        await first
        assert controller.admitted_count == 0

    asyncio.run(scenario())
