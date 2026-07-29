"""Bounded asynchronous admission and worker pools."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

import anyio

from ..config import Settings
from ..domain.errors import ServerBusy

T = TypeVar("T")


class AdmissionController:
    """Bound active-plus-waiting searches and apply a queue timeout."""

    def __init__(
        self, queue_size: int, timeout_seconds: float, *, worker_size: int | None = None
    ) -> None:
        active_size = worker_size if worker_size is not None else queue_size
        if queue_size < 1 or active_size < 1 or timeout_seconds <= 0:
            raise ValueError(
                "queue_size, worker_size, and timeout_seconds must be positive"
            )
        if active_size > queue_size:
            raise ValueError("worker_size must not exceed queue_size")
        self._capacity = queue_size
        self._workers = asyncio.Semaphore(active_size)
        self._timeout_seconds = timeout_seconds
        self._admitted = 0
        self._peak_admitted = 0

    @property
    def admitted_count(self) -> int:
        """Return the current active-plus-waiting request count."""
        return self._admitted

    @property
    def peak_admitted_count(self) -> int:
        """Return the largest admitted request count observed."""
        return self._peak_admitted

    async def run_search(self, operation: Callable[[], Awaitable[T]]) -> T:
        """Run an asynchronous operation inside an admission slot."""
        if self._admitted >= self._capacity:
            raise ServerBusy()
        self._admitted += 1
        self._peak_admitted = max(self._peak_admitted, self._admitted)
        worker_acquired = False
        try:
            try:
                await asyncio.wait_for(self._workers.acquire(), self._timeout_seconds)
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
    """Separate worker tokens for search and lightweight state operations."""

    search: anyio.CapacityLimiter
    state: anyio.CapacityLimiter

    @classmethod
    def from_settings(cls, settings: Settings) -> ThreadLimiters:
        """Construct the process worker limiters."""
        return cls(
            search=anyio.CapacityLimiter(max(settings.provider_concurrency, 4)),
            state=anyio.CapacityLimiter(16),
        )
