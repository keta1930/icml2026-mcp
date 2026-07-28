"""异步 MCP schema、offload、可信代理与安全错误边界。"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from datetime import datetime, timezone

from mcp.server.fastmcp import Context
from mcp.server.lowlevel.server import RequestContext
from starlette.requests import Request

from icml_paper_search.config import Settings
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
    return Context(request_context=request_context)


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
