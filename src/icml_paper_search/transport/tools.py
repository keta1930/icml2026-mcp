"""MCP tool schemas, handlers, and public error boundaries."""

from __future__ import annotations

import ipaddress
import logging
from collections.abc import Callable
from typing import Annotated, Any

import anyio
from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field
from starlette.requests import Request

from ..config import Settings
from ..corpus.catalog import get_catalog
from ..domain.errors import AppError, InvalidRequestError, public_error_text
from ..domain.models import SearchFilters
from ..search.formatting import (
    format_papers,
    format_search_outcome,
    format_session_summary,
)
from ..search.service import PaperSearchService
from .resources import ApplicationResources

logger = logging.getLogger(__name__)

INSTRUCTIONS = """\
Search a fixed collection of 6,341 ICML 2026 papers.
For multi-round research, call `open_search_session`, keep its `session_id` private, and
reuse it in later `search_papers` calls. Papers already returned in that session will be
left out. Searches without a `session_id` are stateless, and different sessions remain
independent. If several searches return no new papers, rephrase the query, change filters,
or close the session.\
"""


def register_tools(server: FastMCP) -> None:
    """Register the six public MCP tools."""
    server.tool(
        title="Open Search Session",
        description=(
            "Start a search session and return its ID. Pass that ID to later search "
            "calls to leave out papers already returned in the session."
        ),
    )(open_search_session)
    server.tool(
        title="Search Papers",
        description=(
            "Search 6,341 ICML 2026 papers. With `session_id`, leave out papers "
            "already returned in that session."
        ),
    )(search_papers)
    server.tool(
        title="Close Search Session",
        description="Close a search session and return a short summary.",
    )(close_search_session)
    server.tool(
        title="Search Session Status",
        description="Show progress for one search session.",
    )(search_session_status)
    server.tool(
        title="Get Papers",
        description=(
            "Fetch papers by OpenReview ID. With `session_id`, record found papers "
            "as returned in that session."
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
        Field(
            default="",
            max_length=120,
            description="Optional topic label for the session.",
        ),
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
    """Open a search session."""
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
            description="Search session ID used to leave out previously returned papers.",
        ),
    ] = None,
    top_k: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            le=50,
            description="Maximum number of new papers to return.",
        ),
    ] = None,
    search_depth: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description="Number of ranked candidates to check before session deduplication.",
        ),
    ] = None,
    area: Annotated[
        str | None, Field(default=None, description="Research area filter.")
    ] = None,
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
            description="Minimum normalized hybrid-ranking score.",
        ),
    ] = None,
    include_abstract: Annotated[
        bool,
        Field(default=True, description="Include abstract previews in the result."),
    ] = True,
) -> str:
    """Run a governed hybrid search."""
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
        str, Field(min_length=1, description="Search session ID to close.")
    ],
) -> str:
    """Close a session and return its summary."""
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
        str, Field(min_length=1, description="Search session ID to inspect.")
    ],
    include_papers: Annotated[
        bool,
        Field(default=False, description="Include a bounded list of paper titles."),
    ] = False,
) -> str:
    """Inspect one session without allowing anonymous enumeration."""
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
                "Search session in which found papers are recorded as returned."
            ),
        ),
    ] = None,
) -> str:
    """Fetch papers and atomically record session-seen state."""
    resources = _resources(ctx)

    async def operation() -> str:
        papers = await _run_search(
            resources,
            lambda: _service(resources).get_papers(paper_ids, session_id=session_id),
        )
        found = {paper.paper_id for paper in papers}
        missing = [paper_id for paper_id in paper_ids if paper_id not in found]
        return format_papers(papers, resources.settings, missing_ids=missing)

    return await _safe_tool_call(operation)


async def list_paper_areas(ctx: Context) -> str:
    """Read static corpus dimensions."""
    resources = _resources(ctx)

    async def operation() -> str:
        return await _run_state(
            resources, lambda: get_catalog(resources.settings).render()
        )

    return await _safe_tool_call(operation)


async def _safe_tool_call(operation: Callable[[], Any]) -> str:
    """Return public error text and keep exception details out of server logs."""
    try:
        return await operation()
    except Exception as exc:  # noqa: BLE001 - public tool error boundary
        error = exc if isinstance(exc, AppError) else AppError()
        logger.warning(
            "tool request failed code=%s exception_type=%s",
            error.code,
            type(exc).__name__,
        )
        return public_error_text(exc)


async def _run_search(resources: ApplicationResources, operation: Callable[[], Any]):
    resources._require_initialized()
    assert resources.limiters is not None
    return await anyio.to_thread.run_sync(operation, limiter=resources.limiters.search)


async def _run_state(resources: ApplicationResources, operation: Callable[[], Any]):
    resources._require_initialized()
    assert resources.limiters is not None
    return await anyio.to_thread.run_sync(operation, limiter=resources.limiters.state)


def _resources(ctx: Context) -> ApplicationResources:
    resources = getattr(ctx.fastmcp, "_icml_resources", None)
    if not isinstance(resources, ApplicationResources):
        raise TypeError("FastMCP application resources are invalid")
    resources._require_initialized()
    return resources


def _service(resources: ApplicationResources) -> PaperSearchService:
    resources._require_initialized()
    assert resources.service is not None
    return resources.service


def _resolve_client_ip(ctx: Context, settings: Settings) -> str:
    """Trust the forwarding header only when the direct peer is an allowed proxy."""
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
