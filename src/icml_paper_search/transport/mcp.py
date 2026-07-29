"""FastMCP server construction and Streamable HTTP entry point."""

from __future__ import annotations

import logging
import sys

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse

from ..config import Settings, get_settings
from .resources import ApplicationFastMCP, ApplicationResources
from .tools import INSTRUCTIONS, register_tools

logger = logging.getLogger(__name__)


def create_mcp(
    settings: Settings | None = None,
    *,
    resources: ApplicationResources | None = None,
) -> FastMCP:
    """Construct a single-process FastMCP server with injectable resources."""
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
    register_tools(server)

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
        except Exception:  # noqa: BLE001 - readiness must fail closed
            return JSONResponse({"status": "not_ready"}, status_code=503)
        return JSONResponse({"status": "ready"})

    server._icml_resources = app_resources
    return server


def _transport_security(settings: Settings) -> TransportSecuritySettings:
    """Configure Host and Origin allowlists for local and TLS-sidecar access."""
    host = settings.server_host
    port = settings.server_port
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=list(
            dict.fromkeys((*settings.allowed_hosts, host, f"{host}:{port}"))
        ),
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
