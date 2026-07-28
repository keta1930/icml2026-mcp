"""异步 MCP schema、offload、可信代理与安全错误边界。"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from mcp.server.fastmcp import Context
from mcp.server.lowlevel.server import RequestContext
from starlette.requests import Request
from starlette.testclient import TestClient

from icml_paper_search.config import Settings
from icml_paper_search.governance import ThreadLimiters
from icml_paper_search.models import (
    Paper,
    SearchFilters,
    SearchHit,
    SearchOutcome,
    SessionSummary,
)
from icml_paper_search.server import (
    ApplicationResources,
    _resolve_client_ip,
    _safe_tool_call,
    _transport_security,
    create_mcp,
    open_search_session,
    search_papers,
    search_session_status,
)
from icml_paper_search.service import PaperSearchService
from icml_paper_search.sessions import SessionManager, SQLiteSessionStore
from icml_paper_search.state import RuntimeDatabase


def _settings(tmp_path, **overrides) -> Settings:
    values = {
        "data_dir": tmp_path,
        "index_root": tmp_path / "index",
        "state_db_path": tmp_path / "runtime.sqlite3",
        "max_search_depth": 50,
        "ip_rate_per_minute": 60,
        "ip_rate_burst": 10,
    }
    values.update(overrides)
    return Settings(**values)


def _context(resources: ApplicationResources, *, peer="198.51.100.10", headers=()):
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": list(headers),
        "client": (peer, 43210),
        "server": ("test", 20441),
        "scheme": "https",
        "query_string": b"",
    }
    request_context = RequestContext(
        request_id="test",
        meta=None,
        session=object(),
        lifespan_context=resources,
        request=Request(scope),
    )
    return Context(
        request_context=request_context,
        fastmcp=SimpleNamespace(_icml_resources=resources),
    )


class LifecycleResources(ApplicationResources):
    """记录真实 ASGI 与 transport 生命周期对进程资源的影响。"""

    def __init__(self, settings: Settings, *, fail_initialize: bool = False) -> None:
        super().__init__(settings)
        self.fail_initialize = fail_initialize
        self.initialize_calls = 0
        self.close_calls = 0

    def initialize(self) -> None:
        if self.initialized:
            return
        self.initialize_calls += 1
        if self.fail_initialize:
            raise RuntimeError("startup failed")
        self.limiters = ThreadLimiters.from_settings(self.settings)
        self.initialized = True

    def readiness(self) -> None:
        self._require_initialized()

    def close(self) -> None:
        if not self.initialized:
            return
        self.close_calls += 1
        self.initialized = False


class LocalLifecycleService:
    """提供零 provider 的本地 readiness 与可计数关闭边界。"""

    def __init__(self) -> None:
        self.validate_calls = 0
        self.close_calls = 0

    def validate(self) -> None:
        self.validate_calls += 1

    def close(self) -> None:
        self.close_calls += 1


class FailingDatabase:
    """模拟 state 初始化失败并记录清理。"""

    def __init__(self) -> None:
        self.initialize_calls = 0
        self.close_calls = 0

    def initialize(self) -> None:
        self.initialize_calls += 1
        raise RuntimeError("state startup failed")

    def close(self) -> None:
        self.close_calls += 1


_MCP_HEADERS = {
    "accept": "application/json, text/event-stream",
    "content-type": "application/json",
    "mcp-protocol-version": "2025-11-25",
}


def _open_transport(client: TestClient, request_id: int) -> str:
    """通过真实 Streamable HTTP initialize 建立一个 transport。"""
    response = client.post(
        "/mcp",
        headers=_MCP_HEADERS,
        json={
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "lifespan-test", "version": "1"},
            },
        },
    )
    assert response.status_code == 200
    session_id = response.headers["mcp-session-id"]
    initialized = client.post(
        "/mcp",
        headers={**_MCP_HEADERS, "mcp-session-id": session_id},
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
    )
    assert initialized.status_code == 202
    return session_id


def _list_tools(client: TestClient, session_id: str, request_id: int) -> None:
    """在指定 transport 上执行一个真实 MCP tool-list 请求。"""
    response = client.post(
        "/mcp",
        headers={**_MCP_HEADERS, "mcp-session-id": session_id},
        json={"jsonrpc": "2.0", "id": request_id, "method": "tools/list"},
    )
    assert response.status_code == 200
    assert "search_papers" in response.text


def _call_status_without_capability(
    client: TestClient, session_id: str, request_id: int
) -> None:
    """调用一个读取进程资源、但不会访问 provider 的真实 MCP tool。"""
    response = client.post(
        "/mcp",
        headers={**_MCP_HEADERS, "mcp-session-id": session_id},
        json={
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": "search_session_status", "arguments": {}},
        },
    )
    assert response.status_code == 200
    assert "error[invalid_request]" in response.text


def _close_transport(client: TestClient, session_id: str) -> None:
    """终止一个真实 Streamable HTTP transport。"""
    response = client.delete(
        "/mcp", headers={**_MCP_HEADERS, "mcp-session-id": session_id}
    )
    assert response.status_code == 200


def test_six_tools_are_async_and_context_is_not_in_schema(tmp_path):
    """保留六个名称，同时避免同步 tool 阻塞 event loop。"""
    server = create_mcp(_settings(tmp_path))
    tools = {tool.name: tool for tool in server._tool_manager.list_tools()}
    assert set(tools) == {
        "open_search_session",
        "search_papers",
        "close_search_session",
        "search_session_status",
        "get_papers",
        "list_paper_areas",
    }
    assert all(tool.is_async for tool in tools.values())
    assert all("ctx" not in tool.parameters.get("properties", {}) for tool in tools.values())
    assert tools["search_papers"].parameters["properties"]["query"]["maxLength"] == 2000
    assert tools["get_papers"].parameters["properties"]["paper_ids"]["maxItems"] == 50


def test_asgi_startup_initializes_resources_before_first_transport(tmp_path):
    """ASGI startup 后即 ready，且 FastMCP session manager 同时运行。"""
    resources = LifecycleResources(_settings(tmp_path))
    server = create_mcp(resources.settings, resources=resources)
    app = server.streamable_http_app()

    with TestClient(app, base_url="http://127.0.0.1:20441") as client:
        assert resources.initialize_calls == 1
        assert resources.close_calls == 0
        assert server.session_manager._task_group is not None
        response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "ready"}

    assert resources.close_calls == 1
    assert not resources.initialized


def test_two_transports_do_not_own_shared_application_resources(tmp_path):
    """关闭一个 transport 不得关闭另一个 transport 使用的共享资源。"""
    resources = LifecycleResources(_settings(tmp_path))
    server = create_mcp(resources.settings, resources=resources)

    with TestClient(
        server.streamable_http_app(), base_url="http://127.0.0.1:20441"
    ) as client:
        first = _open_transport(client, 1)
        second = _open_transport(client, 2)
        _list_tools(client, first, 3)
        _list_tools(client, second, 4)

        _close_transport(client, first)

        assert resources.initialize_calls == 1
        assert resources.close_calls == 0
        assert client.get("/health/ready").status_code == 200
        _call_status_without_capability(client, second, 5)
        _close_transport(client, second)
        assert client.get("/health/ready").status_code == 200

    assert resources.close_calls == 1


def test_asgi_startup_failure_fails_closed(tmp_path):
    """资源初始化失败必须阻断 ASGI startup，不能先对外服务。"""
    resources = LifecycleResources(_settings(tmp_path), fail_initialize=True)
    server = create_mcp(resources.settings, resources=resources)

    with pytest.raises(RuntimeError, match="startup failed"):
        with TestClient(
            server.streamable_http_app(), base_url="http://127.0.0.1:20441"
        ):
            pytest.fail("startup failure 后不应进入服务上下文")

    assert resources.initialize_calls == 1
    assert resources.close_calls == 0


def test_real_resources_start_locally_and_close_once(tmp_path, monkeypatch):
    """真实资源容器在 startup 不读取 key，shutdown 后重复关闭无副作用。"""
    settings = _settings(tmp_path)
    service = LocalLifecycleService()
    resources = ApplicationResources(settings, service=service)  # type: ignore[arg-type]
    server = create_mcp(settings, resources=resources)

    def reject_key_resolution(_settings: Settings) -> str:
        raise AssertionError("ASGI lifecycle 不得读取 provider key")

    monkeypatch.setattr(Settings, "resolve_dashscope_api_key", reject_key_resolution)

    with TestClient(
        server.streamable_http_app(), base_url="http://127.0.0.1:20441"
    ) as client:
        assert settings.state_db_path.is_file()
        assert client.get("/health/ready").status_code == 200
        assert service.validate_calls == 1

    assert service.close_calls == 1
    resources.close()
    assert service.close_calls == 1


def test_partial_startup_failure_closes_owned_resources(tmp_path):
    """真实资源初始化中途失败时清理一次，且 session manager 不启动。"""
    database = FailingDatabase()
    resources = ApplicationResources(
        _settings(tmp_path), database=database  # type: ignore[arg-type]
    )
    server = create_mcp(resources.settings, resources=resources)

    with pytest.raises(RuntimeError, match="state startup failed"):
        with TestClient(
            server.streamable_http_app(), base_url="http://127.0.0.1:20441"
        ):
            pytest.fail("state startup failure 后不应进入服务上下文")

    assert database.initialize_calls == 1
    assert database.close_calls == 1
    assert server.session_manager._task_group is None
    resources.close()
    assert database.close_calls == 1


def test_safe_error_does_not_echo_internal_message():
    """未知异常 message 可能含 provider body，公开文本不得透传。"""

    async def scenario() -> str:
        async def fail() -> str:
            raise RuntimeError("sentinel-secret-query-token")

        return await _safe_tool_call(fail)

    text = asyncio.run(scenario())
    assert text.startswith("error[internal_error]")
    assert "sentinel" not in text


def test_client_ip_only_accepts_header_from_trusted_proxy(tmp_path):
    """公网 peer 不能伪造覆盖 header，Compose 私网代理可以。"""
    settings = _settings(
        tmp_path,
        trusted_proxy_enabled=True,
        trusted_proxy_networks=("172.20.0.0/16",),
    )
    resources = ApplicationResources(settings, initialized=True)
    header = [(b"x-icml-client-ip", b"203.0.113.8")]

    untrusted = _context(resources, peer="198.51.100.20", headers=header)
    trusted = _context(resources, peer="172.20.0.2", headers=header)
    assert _resolve_client_ip(untrusted, settings) == "198.51.100.20"
    assert _resolve_client_ip(trusted, settings) == "203.0.113.8"


def test_transport_domains_are_explicit_configuration(tmp_path):
    """库默认不绑定 mita，自托管域名必须通过配置显式放行。"""
    settings = _settings(
        tmp_path,
        allowed_hosts=("papers.example.org:20441",),
        allowed_origins=("https://papers.example.org:20441",),
    )
    security = _transport_security(settings)
    assert "papers.example.org:20441" in security.allowed_hosts
    assert "https://papers.example.org:20441" in security.allowed_origins
    assert all("agent4sci.com" not in host for host in security.allowed_hosts)


class SlowService:
    """模拟阻塞搜索，同时让轻量 status 立即完成。"""

    def __init__(self) -> None:
        self.entered = threading.Event()

    def search(self, query: str, **_kwargs) -> SearchOutcome:
        self.entered.set()
        time.sleep(0.12)
        return SearchOutcome(
            query=query,
            hits=[],
            search_depth=5,
            candidates_examined=0,
            duplicates_suppressed=0,
            filters=SearchFilters(),
        )

    def session_status(self, _session_id: str, *, include_papers: bool):
        now = datetime.now(timezone.utc)
        return SessionSummary(
            session_id="ses_test",
            label="test",
            created_at=now,
            last_active_at=now,
            query_count=0,
            unique_papers=0,
            consecutive_empty_rounds=0,
            exhausted=False,
            default_top_k=None,
            queries=[],
            papers=[],
        )

    def close(self) -> None:
        return None


class CountingSessionService(SlowService):
    """记录真正进入 session 写入边界的调用数。"""

    def __init__(self) -> None:
        super().__init__()
        self.open_calls = 0
        self.lock = threading.Lock()

    def open_session(self, *, label: str, default_top_k: int | None):
        with self.lock:
            self.open_calls += 1
            ordinal = self.open_calls
        now = datetime.now(timezone.utc)
        return SessionSummary(
            session_id=f"ses_test_{ordinal}",
            label=label or "untitled",
            created_at=now,
            last_active_at=now,
            query_count=0,
            unique_papers=0,
            consecutive_empty_rounds=0,
            exhausted=False,
            default_top_k=default_top_k,
            queries=[],
            papers=[],
        )


class ConcurrentEngine:
    """返回同一候选窗口并记录搜索线程并发。"""

    def __init__(self, count: int = 60) -> None:
        self.papers = [
            Paper(
                paper_id=f"p{index:03d}",
                title=f"Paper {index}",
                abstract="",
                authors=(),
                area="Test",
                subarea="",
                paper_type="Poster",
                spotlight=False,
                arxiv_id="",
            )
            for index in range(count)
        ]
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def search(self, _query: str, *, limit: int, filters: SearchFilters):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.01)
            return [
                SearchHit(paper=paper, score=0.99 - index * 0.001)
                for index, paper in enumerate(self.papers[:limit])
            ]
        finally:
            with self.lock:
                self.active -= 1

    def fetch(self, paper_ids):
        by_id = {paper.paper_id: paper for paper in self.papers}
        return [by_id[paper_id] for paper_id in paper_ids if paper_id in by_id]

    def close(self) -> None:
        return None


def test_slow_search_does_not_block_session_status(tmp_path):
    """慢 provider/search worker 不占用 event loop 或 state limiter。"""

    async def scenario() -> None:
        settings = _settings(tmp_path)
        service = SlowService()
        resources = ApplicationResources(settings, service=service)  # type: ignore[arg-type]
        resources.initialize()
        ctx = _context(resources)
        search_task = asyncio.create_task(search_papers(ctx, "query"))
        await asyncio.to_thread(service.entered.wait)
        started = time.monotonic()
        status = await search_session_status(ctx, "ses_test")
        elapsed = time.monotonic() - started
        assert "检索会话状态" in status
        assert elapsed < 0.08
        await search_task
        resources.close()

    asyncio.run(scenario())


def test_anonymous_status_requires_capability(tmp_path):
    """无参 status 返回稳定 invalid_request，永不枚举全局 session。"""

    async def scenario() -> str:
        settings = _settings(tmp_path)
        resources = ApplicationResources(settings, service=SlowService())  # type: ignore[arg-type]
        resources.initialize()
        try:
            return await search_session_status(_context(resources), None)
        finally:
            resources.close()

    text = asyncio.run(scenario())
    assert text.startswith("error[invalid_request]")
    assert "session_id" in text


def test_open_session_rate_limit_runs_before_session_write(tmp_path):
    """单个匿名来源的容量攻击在 session 写入前被持久限流。"""

    async def scenario() -> None:
        settings = _settings(
            tmp_path,
            ip_session_open_rate_per_minute=2,
            ip_session_open_burst=2,
            ip_session_opens_per_day=2,
        )
        service = CountingSessionService()
        resources = ApplicationResources(settings, service=service)  # type: ignore[arg-type]
        resources.initialize()
        try:
            results = await asyncio.gather(
                *(
                    open_search_session(_context(resources), label=f"attack-{index}")
                    for index in range(40)
                )
            )
        finally:
            resources.close()

        assert service.open_calls == 2
        assert sum(text.startswith("已开启检索会话") for text in results) == 2
        assert sum(text.startswith("error[rate_limited]") for text in results) == 38

    asyncio.run(scenario())


def test_40_clients_mix_search_and_status_with_same_session(tmp_path):
    """40 个客户端混合调用时不阻塞状态请求且 session 无重复。"""

    async def scenario() -> None:
        settings = _settings(
            tmp_path,
            ip_rate_per_minute=120,
            ip_rate_burst=120,
            global_rate_per_minute=120,
            provider_concurrency=4,
            search_queue_size=64,
            queue_timeout_seconds=2,
        )
        database = RuntimeDatabase(settings.state_db_path)
        database.initialize()
        engine = ConcurrentEngine()
        sessions = SessionManager(
            settings, SQLiteSessionStore(database, settings)
        )
        service = PaperSearchService(
            settings, engine=engine, sessions=sessions, database=database
        )
        resources = ApplicationResources(
            settings, database=database, service=service
        )
        resources.initialize()
        session_id = service.open_session(label="shared").session_id
        try:
            searches = [
                search_papers(
                    _context(resources, peer=f"198.51.100.{index + 1}"),
                    f"query {index}",
                    session_id=session_id,
                    top_k=1,
                    search_depth=40,
                    include_abstract=False,
                )
                for index in range(20)
            ]
            statuses = [
                search_session_status(
                    _context(resources, peer=f"203.0.113.{index + 1}"),
                    session_id,
                )
                for index in range(20)
            ]
            results = await asyncio.gather(*searches, *statuses)
        finally:
            resources.close()

        search_results = results[:20]
        status_results = results[20:]
        paper_ids = [
            match.group(1)
            for text in search_results
            if (match := re.search(r"id=(p\d{3})", text)) is not None
        ]
        assert len(paper_ids) == 20
        assert len(set(paper_ids)) == 20
        assert all("error[" not in text for text in search_results)
        assert all("检索会话状态" in text for text in status_results)
        assert engine.max_active <= settings.provider_concurrency
        assert resources.admission is not None
        assert resources.admission.peak_admitted_count <= settings.search_queue_size

    asyncio.run(scenario())
