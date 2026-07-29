"""Async FastMCP entry point with bounded offload and safe public error boundaries."""

from __future__ import annotations

import ipaddress
import logging
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any, Callable

import anyio
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse

from .catalog import get_catalog
from .config import Settings, get_settings
from .errors import AppError, InvalidRequestError, public_error_text
from .formatting import format_papers, format_search_outcome, format_session_summary
from .governance import AdmissionController, SQLiteRateLimiter, ThreadLimiters
from .models import SearchFilters
from .service import PaperSearchService
from .state import RuntimeDatabase

logger = logging.getLogger(__name__)

INSTRUCTIONS = """\
Hybrid search over 6,341 ICML 2026 papers using Chroma dense retrieval and SQLite BM25.
For multi-round research, create a capability session, keep its `session_id` private, and
reuse it across `search_papers` calls so previously returned papers are suppressed.
Consecutive empty rounds indicate that the current direction may be exhausted; rephrase
the query, increase `search_depth`, or close the session. Without a `session_id`, searches
are stateless. Anonymous callers cannot enumerate other sessions.\
"""


@dataclass(slots=True)
class ApplicationResources:
    """Create and close shared resources once within the ASGI process lifespan."""

    settings: Settings
    database: RuntimeDatabase | None = None
    service: PaperSearchService | None = None
    rate_limiter: SQLiteRateLimiter | None = None
    admission: AdmissionController | None = None
    limiters: ThreadLimiters | None = None
    initialized: bool = False
    _closed: bool = False

    def initialize(self) -> None:
        """Initialize state and lazy clients without calling DashScope."""
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
            self.limiters = self.limiters or ThreadLimiters.from_settings(
                self.settings
            )
            self.initialized = True
        except Exception:
            self.close()
            raise

    def readiness(self) -> None:
        """Check only local state and index without reading a key or calling the provider."""
        self._require_initialized()
        assert self.database is not None
        assert self.service is not None
        with self.database.connection() as connection:
            connection.execute("SELECT 1").fetchone()
        self.service.validate()

    def close(self) -> None:
        """Close the engine, provider, and runtime database in dependency order."""
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
    """Compose ASGI process resources with the FastMCP session-manager lifespan."""

    def __init__(self, *args: Any, resources: ApplicationResources, **kwargs: Any):
        self._icml_resources = resources
        super().__init__(*args, **kwargs)

    def streamable_http_app(self) -> Starlette:
        """Create an ASGI app managing process resources and Streamable HTTP sessions."""
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


def create_mcp(
    settings: Settings | None = None,
    *,
    resources: ApplicationResources | None = None,
) -> FastMCP:
    """Construct a single-process FastMCP server with injectable test resources."""
    config = settings or get_settings()
    app_resources = resources or ApplicationResources(config)

    server = ApplicationFastMCP(
        "icml-paper-search",
        resources=app_resources,
        instructions=INSTRUCTIONS,
        host=config.server_host,
        port=config.server_port,
        transport_security=_transport_security(config),
    )
    _register_tools(server)

    @server.custom_route("/health/live", methods=["GET"])
    async def liveness(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @server.custom_route("/health/ready", methods=["GET"])
    async def readiness(_request: Request) -> JSONResponse:
        try:
            app_resources._require_initialized()
            assert app_resources.limiters is not None
            await anyio.to_thread.run_sync(
                app_resources.readiness, limiter=app_resources.limiters.state
            )
        except Exception:
            return JSONResponse({"status": "not_ready"}, status_code=503)
        return JSONResponse({"status": "ready"})

    # Tests and embedded deployments may inspect this handle; it is excluded from MCP schema.
    setattr(server, "_icml_resources", app_resources)
    return server


def _register_tools(server: FastMCP) -> None:
    """Register the six compatibility-stable asynchronous MCP tools."""
    server.tool(
        title="Open Search Session",
        description=(
            "Create an unenumerable capability session for automatic deduplication "
            "across search rounds."
        ),
    )(open_search_session)
    server.tool(
        title="Search Papers",
        description=(
            "Search ICML 2026 papers with Chroma dense retrieval and SQLite BM25 "
            "weighted RRF; with `session_id`, return only papers not yet seen in "
            "that session."
        ),
    )(search_papers)
    server.tool(
        title="Close Search Session",
        description="Close a capability session and return a bounded search summary.",
    )(close_search_session)
    server.tool(
        title="Search Session Status",
        description=(
            "Inspect one capability session by `session_id`; anonymous callers cannot "
            "enumerate sessions."
        ),
    )(search_session_status)
    server.tool(
        title="Get Papers",
        description=(
            "Fetch full abstracts by OpenReview ID and optionally mark the papers as seen "
            "in a capability session atomically."
        ),
    )(get_papers)
    server.tool(
        title="List Paper Areas",
        description="List available `area` and `paper_type` values with corpus counts.",
    )(list_paper_areas)


async def open_search_session(
    ctx: Context,
    label: Annotated[
        str,
        Field(default="", max_length=120, description="Optional topic label for the session."),
    ] = "",
    default_top_k: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            le=50,
            description="Default number of results returned per round for this session.",
        ),
    ] = None,
) -> str:
    """Open a capability session asynchronously."""
    resources = _resources(ctx)

    async def operation() -> str:
        client_ip = _resolve_client_ip(ctx, resources.settings)
        assert resources.rate_limiter is not None
        client_hash = resources.rate_limiter.hash_identity(client_ip)
        await _run_state(
            resources,
            lambda: resources.rate_limiter.check_session_open(client_hash),
        )
        summary = await _run_state(
            resources,
            lambda: _service(resources).open_session(
                label=label, default_top_k=default_top_k
            ),
        )
        return (
            format_session_summary(summary, title="Search session opened")
            + "\n\nKeep this session_id private; the anonymous service cannot list or "
            "recover it for you."
        )

    return await _safe_tool_call(operation)


async def search_papers(
    ctx: Context,
    query: Annotated[
        str,
        Field(
            min_length=1,
            max_length=2_000,
            description="Natural-language research topic or method.",
        ),
    ],
    session_id: Annotated[
        str | None,
        Field(
            default=None,
            description="Capability session ID used for persistent deduplication.",
        ),
    ] = None,
    top_k: Annotated[
        int | None,
        Field(default=None, ge=1, le=50, description="Maximum number of new papers to return."),
    ] = None,
    search_depth: Annotated[
        int | None, Field(default=None, ge=1, description="Fixed relevance-window depth.")
    ] = None,
    area: Annotated[str | None, Field(default=None, description="Research area filter.")] = None,
    paper_type: Annotated[
        str | None, Field(default=None, description="Paper type filter.")
    ] = None,
    spotlight_only: Annotated[
        bool, Field(default=False, description="Return only spotlight papers.")
    ] = False,
    min_score: Annotated[
        float | None,
        Field(
            default=None,
            ge=0.0,
            le=1.0,
            description="Minimum normalized weighted-RRF score.",
        ),
    ] = None,
    include_abstract: Annotated[
        bool, Field(default=True, description="Include abstract previews in the result.")
    ] = True,
) -> str:
    """Run governed fixed-window hybrid search asynchronously."""
    resources = _resources(ctx)
    filters = SearchFilters(
        area=area, paper_type=paper_type, spotlight_only=spotlight_only
    )

    async def operation() -> str:
        client_ip = _resolve_client_ip(ctx, resources.settings)
        assert resources.rate_limiter is not None
        client_hash = resources.rate_limiter.hash_identity(client_ip)
        await _run_state(
            resources,
            lambda: resources.rate_limiter.check_search(client_hash),
        )
        assert resources.admission is not None

        async def admitted() -> str:
            outcome = await _run_search(
                resources,
                lambda: _service(resources).search(
                    query,
                    session_id=session_id,
                    top_k=top_k,
                    search_depth=search_depth,
                    filters=filters,
                    min_score=min_score,
                ),
            )
            return format_search_outcome(
                outcome, resources.settings, include_abstract=include_abstract
            )

        return await resources.admission.run_search(admitted)

    return await _safe_tool_call(operation)


async def close_search_session(
    ctx: Context,
    session_id: Annotated[
        str, Field(min_length=1, description="Capability session ID to close.")
    ],
) -> str:
    """Close a session asynchronously and return a safe summary."""
    resources = _resources(ctx)

    async def operation() -> str:
        summary = await _run_state(
            resources, lambda: _service(resources).close_session(session_id)
        )
        return format_session_summary(
            summary, title="Search session closed", include_session_id=False
        )

    return await _safe_tool_call(operation)


async def search_session_status(
    ctx: Context,
    session_id: Annotated[
        str | None, Field(default=None, description="Capability session ID to inspect.")
    ] = None,
    include_papers: Annotated[
        bool, Field(default=False, description="Include a bounded list of paper titles.")
    ] = False,
) -> str:
    """Inspect one session asynchronously while rejecting anonymous enumeration."""
    resources = _resources(ctx)

    async def operation() -> str:
        if session_id is None:
            raise InvalidRequestError(public_message="`session_id` is required.")
        summary = await _run_state(
            resources,
            lambda: _service(resources).session_status(
                session_id, include_papers=include_papers
            ),
        )
        return format_session_summary(
            summary, title="Search session status", include_session_id=False
        )

    return await _safe_tool_call(operation)


async def get_papers(
    ctx: Context,
    paper_ids: Annotated[
        list[str],
        Field(min_length=1, max_length=50, description="OpenReview paper IDs."),
    ],
    session_id: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Capability session in which returned papers are atomically marked as seen."
            ),
        ),
    ] = None,
) -> str:
    """Fetch papers asynchronously and atomically record session-seen state."""
    resources = _resources(ctx)

    async def operation() -> str:
        papers = await _run_search(
            resources,
            lambda: _service(resources).get_papers(
                paper_ids, session_id=session_id
            ),
        )
        found = {paper.paper_id for paper in papers}
        missing = [paper_id for paper_id in paper_ids if paper_id not in found]
        return format_papers(papers, resources.settings, missing_ids=missing)

    return await _safe_tool_call(operation)


async def list_paper_areas(ctx: Context) -> str:
    """Read static corpus dimensions in a worker thread."""
    resources = _resources(ctx)

    async def operation() -> str:
        return await _run_state(
            resources, lambda: get_catalog(resources.settings).render()
        )

    return await _safe_tool_call(operation)


async def _safe_tool_call(operation: Callable[[], Any]) -> str:
    """Return only public short codes and omit exception messages from server logs."""
    try:
        return await operation()
    except Exception as exc:
        error = exc if isinstance(exc, AppError) else AppError()
        logger.warning(
            "tool request failed code=%s exception_type=%s",
            error.code,
            type(exc).__name__,
        )
        return public_error_text(exc)


async def _run_search(resources: ApplicationResources, operation: Callable[[], Any]):
    """Run a complete synchronous search in a bounded search worker."""
    resources._require_initialized()
    assert resources.limiters is not None
    return await anyio.to_thread.run_sync(operation, limiter=resources.limiters.search)


async def _run_state(resources: ApplicationResources, operation: Callable[[], Any]):
    """Run short SQLite or catalog operations in a separate state worker."""
    resources._require_initialized()
    assert resources.limiters is not None
    return await anyio.to_thread.run_sync(operation, limiter=resources.limiters.state)


def _resources(ctx: Context) -> ApplicationResources:
    """Get shared resources from the stable process-level FastMCP container."""
    resources = getattr(ctx.fastmcp, "_icml_resources", None)
    if not isinstance(resources, ApplicationResources):
        raise RuntimeError("FastMCP application resources are invalid")
    resources._require_initialized()
    return resources


def _service(resources: ApplicationResources) -> PaperSearchService:
    resources._require_initialized()
    assert resources.service is not None
    return resources.service


def _resolve_client_ip(ctx: Context, settings: Settings) -> str:
    """Use the peer by default and accept the override header only from trusted proxies."""
    request = ctx.request_context.request
    peer = "unknown"
    if isinstance(request, Request) and request.client is not None:
        peer = request.client.host
    if not settings.trusted_proxy_enabled or not isinstance(request, Request):
        return _normalized_ip(peer)
    if not _in_trusted_network(peer, settings.trusted_proxy_networks):
        return _normalized_ip(peer)
    forwarded = request.headers.get(settings.trusted_proxy_header, "")
    return _normalized_ip(forwarded) if forwarded else _normalized_ip(peer)


def _normalized_ip(value: str) -> str:
    try:
        return ipaddress.ip_address(value.strip()).compressed
    except ValueError:
        return "unknown"


def _in_trusted_network(peer: str, networks: tuple[str, ...]) -> bool:
    try:
        address = ipaddress.ip_address(peer)
        return any(address in ipaddress.ip_network(network) for network in networks)
    except ValueError:
        return False


def _transport_security(settings: Settings) -> TransportSecuritySettings:
    """Configure Host and Origin allowlists for direct, local, and TLS-sidecar access."""
    host = settings.server_host
    port = settings.server_port
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=list(dict.fromkeys((*settings.allowed_hosts, host, f"{host}:{port}"))),
        allowed_origins=list(settings.allowed_origins),
    )


mcp = create_mcp()


def main() -> None:
    """Start the MCP server with single-process Streamable HTTP."""
    settings = get_settings()
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info(
        "starting streamable-http MCP server host=%s port=%s",
        settings.server_host,
        settings.server_port,
    )
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
