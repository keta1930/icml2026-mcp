"""异步 FastMCP 入口：有界 offload、匿名治理与安全错误边界。"""

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
ICML 2026 论文库（6341 篇）的 Chroma dense + SQLite BM25 混合检索。

用 capability session 做主题调研：
1. `open_search_session` 创建会话并妥善保存 session_id；
2. 带同一 session_id 多次调用 `search_papers`，已返回论文不会重复；
3. 连续空轮表示当前检索方向接近挖尽，可换措辞、调大 search_depth 或结束会话。

不传 session_id 时是无状态检索。匿名服务不会列出其他调用方的 session。\
"""


@dataclass(slots=True)
class ApplicationResources:
    """在 FastMCP lifespan 中一次创建并统一关闭进程资源。"""

    settings: Settings
    database: RuntimeDatabase | None = None
    service: PaperSearchService | None = None
    rate_limiter: SQLiteRateLimiter | None = None
    admission: AdmissionController | None = None
    limiters: ThreadLimiters | None = None
    initialized: bool = False

    def initialize(self) -> None:
        """初始化 state 与延迟客户端，不触发 DashScope 调用。"""
        if self.initialized:
            return
        database = self.database or RuntimeDatabase(
            self.settings.state_db_path,
            busy_timeout_ms=self.settings.sqlite_busy_timeout_ms,
        )
        database.initialize()
        self.database = database
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

    def readiness(self) -> None:
        """只检查本地 state/index，不读取 key 或访问 provider。"""
        self._require_initialized()
        assert self.database is not None
        assert self.service is not None
        with self.database.connection() as connection:
            connection.execute("SELECT 1").fetchone()
        self.service.validate()

    def close(self) -> None:
        """按依赖顺序关闭引擎/provider 与 runtime database。"""
        if not self.initialized:
            return
        if self.service is not None:
            self.service.close()
        if self.database is not None:
            self.database.close()
        self.initialized = False

    def _require_initialized(self) -> None:
        if not self.initialized:
            raise RuntimeError("application resources 尚未初始化")


def create_mcp(
    settings: Settings | None = None,
    *,
    resources: ApplicationResources | None = None,
) -> FastMCP:
    """构造可注入测试资源的单进程 FastMCP server。"""
    config = settings or get_settings()
    app_resources = resources or ApplicationResources(config)

    @asynccontextmanager
    async def lifespan(_server: FastMCP):
        await anyio.to_thread.run_sync(app_resources.initialize)
        try:
            yield app_resources
        finally:
            await anyio.to_thread.run_sync(app_resources.close)

    server = FastMCP(
        "icml-paper-search",
        instructions=INSTRUCTIONS,
        host=config.server_host,
        port=config.server_port,
        lifespan=lifespan,
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

    # 测试和嵌入式部署可读取资源句柄，但它不会进入 MCP schema。
    setattr(server, "_icml_resources", app_resources)
    return server


def _register_tools(server: FastMCP) -> None:
    """注册六个兼容名称的 async MCP tool。"""
    server.tool(
        title="开启检索会话",
        description="创建不可枚举的 capability session，供后续多轮检索自动去重。",
    )(open_search_session)
    server.tool(
        title="检索论文",
        description=(
            "用 Chroma dense + SQLite BM25 weighted-RRF 检索 ICML 2026 论文；"
            "带 session_id 时只返回该会话未见论文。"
        ),
    )(search_papers)
    server.tool(
        title="关闭检索会话",
        description="关闭 capability session 并返回有界检索汇总。",
    )(close_search_session)
    server.tool(
        title="查看会话状态",
        description="按 session_id 查看单个会话；匿名模式禁止无参枚举。",
    )(search_session_status)
    server.tool(
        title="按 id 取论文",
        description="按 OpenReview id 精确取回完整摘要，可原子登记到会话去重集。",
    )(get_papers)
    server.tool(
        title="列出可过滤维度",
        description="列出语料的 area 与 paper_type 取值及分布。",
    )(list_paper_areas)


async def open_search_session(
    ctx: Context,
    label: Annotated[
        str,
        Field(default="", max_length=120, description="会话主题标签"),
    ] = "",
    default_top_k: Annotated[
        int | None,
        Field(default=None, ge=1, le=50, description="该会话默认每轮返回数"),
    ] = None,
) -> str:
    """异步开启 capability session。"""
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
            format_session_summary(summary, title="已开启检索会话")
            + "\n\n请妥善保存该 session_id；匿名服务无法替你列出或找回它。"
        )

    return await _safe_tool_call(operation)


async def search_papers(
    ctx: Context,
    query: Annotated[
        str,
        Field(min_length=1, max_length=2_000, description="自然语言研究主题或方法"),
    ],
    session_id: Annotated[
        str | None, Field(default=None, description="capability session id")
    ] = None,
    top_k: Annotated[
        int | None, Field(default=None, ge=1, le=50, description="返回的新论文数上限")
    ] = None,
    search_depth: Annotated[
        int | None, Field(default=None, ge=1, description="固定相关性窗口深度")
    ] = None,
    area: Annotated[str | None, Field(default=None, description="研究领域过滤")] = None,
    paper_type: Annotated[
        str | None, Field(default=None, description="论文类型过滤")
    ] = None,
    spotlight_only: Annotated[
        bool, Field(default=False, description="只返回 spotlight 论文")
    ] = False,
    min_score: Annotated[
        float | None,
        Field(default=None, ge=0.0, le=1.0, description="weighted-RRF 得分下限"),
    ] = None,
    include_abstract: Annotated[
        bool, Field(default=True, description="是否附带摘要摘录")
    ] = True,
) -> str:
    """异步执行有治理的 fixed-window 混合检索。"""
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
    session_id: Annotated[str, Field(min_length=1, description="要关闭的 session id")],
) -> str:
    """异步关闭 session 并返回安全汇总。"""
    resources = _resources(ctx)

    async def operation() -> str:
        summary = await _run_state(
            resources, lambda: _service(resources).close_session(session_id)
        )
        return format_session_summary(
            summary, title="已关闭检索会话", include_session_id=False
        )

    return await _safe_tool_call(operation)


async def search_session_status(
    ctx: Context,
    session_id: Annotated[
        str | None, Field(default=None, description="要查看的 capability session id")
    ] = None,
    include_papers: Annotated[
        bool, Field(default=False, description="是否列出有界论文标题清单")
    ] = False,
) -> str:
    """异步查看单个 session，拒绝匿名全局枚举。"""
    resources = _resources(ctx)

    async def operation() -> str:
        if session_id is None:
            raise InvalidRequestError(public_message="必须提供 session_id。")
        summary = await _run_state(
            resources,
            lambda: _service(resources).session_status(
                session_id, include_papers=include_papers
            ),
        )
        return format_session_summary(
            summary, title="检索会话状态", include_session_id=False
        )

    return await _safe_tool_call(operation)


async def get_papers(
    ctx: Context,
    paper_ids: Annotated[
        list[str],
        Field(min_length=1, max_length=50, description="OpenReview id 列表"),
    ],
    session_id: Annotated[
        str | None, Field(default=None, description="登记到该 capability session")
    ] = None,
) -> str:
    """异步取回论文并原子登记 session seen。"""
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
    """在线程池中读取静态语料维度。"""
    resources = _resources(ctx)

    async def operation() -> str:
        return await _run_state(
            resources, lambda: get_catalog(resources.settings).render()
        )

    return await _safe_tool_call(operation)


async def _safe_tool_call(operation: Callable[[], Any]) -> str:
    """只返回公开短码，服务端日志也不写异常 message。"""
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
    """把完整同步搜索放入有界 search worker。"""
    resources._require_initialized()
    assert resources.limiters is not None
    return await anyio.to_thread.run_sync(operation, limiter=resources.limiters.search)


async def _run_state(resources: ApplicationResources, operation: Callable[[], Any]):
    """把短 SQLite/catalog 操作放入独立 state worker。"""
    resources._require_initialized()
    assert resources.limiters is not None
    return await anyio.to_thread.run_sync(operation, limiter=resources.limiters.state)


def _resources(ctx: Context) -> ApplicationResources:
    """从 FastMCP request context 取得 lifespan 资源。"""
    resources = ctx.request_context.lifespan_context
    if not isinstance(resources, ApplicationResources):
        raise RuntimeError("FastMCP lifespan context 无效")
    resources._require_initialized()
    return resources


def _service(resources: ApplicationResources) -> PaperSearchService:
    resources._require_initialized()
    assert resources.service is not None
    return resources.service


def _resolve_client_ip(ctx: Context, settings: Settings) -> str:
    """默认使用 peer，仅对可信私网代理接受固定覆盖 header。"""
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
    """为直连、本地和 mita TLS sidecar 配置 Host/Origin 白名单。"""
    host = settings.server_host
    port = settings.server_port
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=list(dict.fromkeys((*settings.allowed_hosts, host, f"{host}:{port}"))),
        allowed_origins=list(settings.allowed_origins),
    )


mcp = create_mcp()


def main() -> None:
    """用单进程 Streamable HTTP 启动 MCP server。"""
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
