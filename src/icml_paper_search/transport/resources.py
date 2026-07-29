"""Application resource ownership and ASGI lifespan integration."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import anyio
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette

from ..config import Settings
from ..runtime.concurrency import AdmissionController, ThreadLimiters
from ..runtime.database import RuntimeDatabase
from ..runtime.rate_limits import SQLiteRateLimiter
from ..search.service import PaperSearchService


@dataclass(slots=True)
class ApplicationResources:
    """Own the process-wide database, search service, and concurrency controls."""

    settings: Settings
    database: RuntimeDatabase | None = None
    service: PaperSearchService | None = None
    rate_limiter: SQLiteRateLimiter | None = None
    admission: AdmissionController | None = None
    limiters: ThreadLimiters | None = None
    initialized: bool = False
    _closed: bool = False

    def initialize(self) -> None:
        """Initialize local state and lazy external clients without calling DashScope."""
        if self.initialized:
            return
        if self._closed:
            raise RuntimeError("application resources are closed")
        try:
            database = self.database or RuntimeDatabase(
                self.settings.state_db_path,
                busy_timeout_ms=self.settings.sqlite_busy_timeout_ms,
            )
            self.database = database
            database.initialize()
            self.service = self.service or PaperSearchService(
                self.settings, database=database
            )
            self.rate_limiter = self.rate_limiter or SQLiteRateLimiter(
                database, self.settings
            )
            self.admission = self.admission or AdmissionController(
                self.settings.search_queue_size,
                self.settings.queue_timeout_seconds,
                worker_size=min(
                    self.settings.search_queue_size,
                    max(self.settings.provider_concurrency, 4),
                ),
            )
            self.limiters = self.limiters or ThreadLimiters.from_settings(self.settings)
            self.initialized = True
        except Exception:
            self.close()
            raise

    def readiness(self) -> None:
        """Check the runtime database and active Milvus index without calling DashScope."""
        self._require_initialized()
        assert self.database is not None
        assert self.service is not None
        with self.database.connection() as connection:
            connection.execute("SELECT 1").fetchone()
        self.service.validate()

    def close(self) -> None:
        """Close the service and runtime database in dependency order."""
        if self._closed:
            return
        self._closed = True
        try:
            if self.service is not None:
                self.service.close()
        finally:
            try:
                if self.database is not None:
                    self.database.close()
            finally:
                self.initialized = False

    def _require_initialized(self) -> None:
        if not self.initialized:
            raise RuntimeError("application resources are not initialized")


class ApplicationFastMCP(FastMCP):
    """Compose process resources with the FastMCP session-manager lifespan."""

    def __init__(self, *args: Any, resources: ApplicationResources, **kwargs: Any):
        self._icml_resources = resources
        super().__init__(*args, **kwargs)

    def streamable_http_app(self) -> Starlette:
        """Create an ASGI app managing resources and Streamable HTTP sessions."""
        app = super().streamable_http_app()
        session_manager_lifespan = app.router.lifespan_context
        resources = self._icml_resources

        @asynccontextmanager
        async def application_lifespan(starlette_app: Starlette):
            await anyio.to_thread.run_sync(resources.initialize)
            try:
                async with session_manager_lifespan(starlette_app) as state:
                    yield state
            finally:
                await anyio.to_thread.run_sync(resources.close)

        app.router.lifespan_context = application_lifespan
        return app
