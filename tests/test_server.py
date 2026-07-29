"""Async MCP schema, offload, trusted-proxy, and safe-error boundary tests."""

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
    INSTRUCTIONS,
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
    """Record how real ASGI and transport lifecycles affect process resources."""

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
    """Provide local readiness and countable close behavior without a provider."""

    def __init__(self) -> None:
        self.validate_calls = 0
        self.close_calls = 0

    def validate(self) -> None:
        self.validate_calls += 1

    def close(self) -> None:
        self.close_calls += 1


class FailingDatabase:
    """Simulate state initialization failure and record cleanup."""

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
    """Establish a transport through a real Streamable HTTP initialize request."""
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
    """Run a real MCP tool-list request on a transport."""
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
    """Call a real MCP tool that reads process resources without provider access."""
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
    """Terminate a real Streamable HTTP transport."""
    response = client.delete(
        "/mcp", headers={**_MCP_HEADERS, "mcp-session-id": session_id}
    )
    assert response.status_code == 200


def test_six_tools_are_async_and_context_is_not_in_schema(tmp_path):
    """Preserve six tool names while keeping synchronous work off the event loop."""
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


def test_initialize_and_tool_schema_match_the_english_contract(tmp_path):
    expected_instructions = (
        "Hybrid search over 6,341 ICML 2026 papers using Chroma dense retrieval and "
        "SQLite BM25.\nFor multi-round research, create a capability session, keep its "
        "`session_id` private, and\nreuse it across `search_papers` calls so previously "
        "returned papers are suppressed.\nConsecutive empty rounds indicate that the "
        "current direction may be exhausted; rephrase\nthe query, increase `search_depth`, "
        "or close the session. Without a `session_id`, searches\nare stateless. Anonymous "
        "callers cannot enumerate other sessions."
    )
    expected_tools = [
        (
            "open_search_session",
            "Open Search Session",
            "Create an unenumerable capability session for automatic deduplication "
            "across search rounds.",
        ),
        (
            "search_papers",
            "Search Papers",
            "Search ICML 2026 papers with Chroma dense retrieval and SQLite BM25 weighted "
            "RRF; with `session_id`, return only papers not yet seen in that session.",
        ),
        (
            "close_search_session",
            "Close Search Session",
            "Close a capability session and return a bounded search summary.",
        ),
        (
            "search_session_status",
            "Search Session Status",
            "Inspect one capability session by `session_id`; anonymous callers cannot "
            "enumerate sessions.",
        ),
        (
            "get_papers",
            "Get Papers",
            "Fetch full abstracts by OpenReview ID and optionally mark the papers as seen "
            "in a capability session atomically.",
        ),
        (
            "list_paper_areas",
            "List Paper Areas",
            "List available `area` and `paper_type` values with corpus counts.",
        ),
    ]
    expected_descriptions = {
        "open_search_session": {
            "label": "Optional topic label for the session.",
            "default_top_k": "Default number of results returned per round for this session.",
        },
        "search_papers": {
            "query": "Natural-language research topic or method.",
            "session_id": "Capability session ID used for persistent deduplication.",
            "top_k": "Maximum number of new papers to return.",
            "search_depth": "Fixed relevance-window depth.",
            "area": "Research area filter.",
            "paper_type": "Paper type filter.",
            "spotlight_only": "Return only spotlight papers.",
            "min_score": "Minimum normalized weighted-RRF score.",
            "include_abstract": "Include abstract previews in the result.",
        },
        "close_search_session": {"session_id": "Capability session ID to close."},
        "search_session_status": {
            "session_id": "Capability session ID to inspect.",
            "include_papers": "Include a bounded list of paper titles.",
        },
        "get_papers": {
            "paper_ids": "OpenReview paper IDs.",
            "session_id": (
                "Capability session in which returned papers are atomically marked as seen."
            ),
        },
        "list_paper_areas": {},
    }

    server = create_mcp(_settings(tmp_path))
    tools = server._tool_manager.list_tools()

    assert INSTRUCTIONS == expected_instructions
    assert [(tool.name, tool.title, tool.description) for tool in tools] == expected_tools
    for tool in tools:
        properties = tool.parameters.get("properties", {})
        assert {
            name: schema["description"] for name, schema in properties.items()
        } == expected_descriptions[tool.name]
        assert not re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", tool.description)

    by_name = {tool.name: tool.parameters for tool in tools}
    assert by_name["open_search_session"].get("required", []) == []
    assert by_name["open_search_session"]["properties"]["label"] == {
        "default": "",
        "description": "Optional topic label for the session.",
        "maxLength": 120,
        "title": "Label",
        "type": "string",
    }
    assert by_name["open_search_session"]["properties"]["default_top_k"]["anyOf"] == [
        {"maximum": 50, "minimum": 1, "type": "integer"},
        {"type": "null"},
    ]
    search_schema = by_name["search_papers"]
    assert search_schema["required"] == ["query"]
    assert search_schema["properties"]["query"]["minLength"] == 1
    assert search_schema["properties"]["query"]["maxLength"] == 2_000
    assert search_schema["properties"]["session_id"]["default"] is None
    assert search_schema["properties"]["top_k"]["anyOf"] == [
        {"maximum": 50, "minimum": 1, "type": "integer"},
        {"type": "null"},
    ]
    assert search_schema["properties"]["search_depth"]["anyOf"] == [
        {"minimum": 1, "type": "integer"},
        {"type": "null"},
    ]
    assert search_schema["properties"]["area"]["default"] is None
    assert search_schema["properties"]["paper_type"]["default"] is None
    assert search_schema["properties"]["spotlight_only"]["default"] is False
    assert search_schema["properties"]["min_score"]["anyOf"] == [
        {"maximum": 1.0, "minimum": 0.0, "type": "number"},
        {"type": "null"},
    ]
    assert search_schema["properties"]["include_abstract"]["default"] is True
    assert by_name["close_search_session"]["required"] == ["session_id"]
    assert by_name["close_search_session"]["properties"]["session_id"]["minLength"] == 1
    assert by_name["search_session_status"].get("required", []) == []
    assert by_name["search_session_status"]["properties"]["session_id"]["default"] is None
    assert by_name["search_session_status"]["properties"]["include_papers"]["default"] is False
    assert by_name["get_papers"]["required"] == ["paper_ids"]
    assert by_name["get_papers"]["properties"]["paper_ids"]["minItems"] == 1
    assert by_name["get_papers"]["properties"]["paper_ids"]["maxItems"] == 50
    assert by_name["get_papers"]["properties"]["session_id"]["default"] is None
    assert by_name["list_paper_areas"]["properties"] == {}


def test_streamable_http_rejects_plain_get_and_health_remains_available(tmp_path):
    resources = LifecycleResources(_settings(tmp_path))
    server = create_mcp(resources.settings, resources=resources)

    with TestClient(
        server.streamable_http_app(), base_url="http://127.0.0.1:20441"
    ) as client:
        response = client.get("/mcp", headers={"accept": "application/json"})

        assert response.status_code == 406
        assert response.json()["error"]["code"] == -32600
        assert client.get("/health/live").json() == {"status": "ok"}
        assert client.get("/health/ready").json() == {"status": "ready"}


def test_asgi_startup_initializes_resources_before_first_transport(tmp_path):
    """Become ready at ASGI startup while running the FastMCP session manager."""
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
    """Closing one transport must not close resources used by another transport."""
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
    """Resource initialization failure must stop ASGI startup before serving."""
    resources = LifecycleResources(_settings(tmp_path), fail_initialize=True)
    server = create_mcp(resources.settings, resources=resources)

    with pytest.raises(RuntimeError, match="startup failed"):
        with TestClient(
            server.streamable_http_app(), base_url="http://127.0.0.1:20441"
        ):
            pytest.fail("the service context must not open after startup failure")

    assert resources.initialize_calls == 1
    assert resources.close_calls == 0


def test_real_resources_start_locally_and_close_once(tmp_path, monkeypatch):
    """Start real resources without reading a key and close idempotently."""
    settings = _settings(tmp_path)
    service = LocalLifecycleService()
    resources = ApplicationResources(settings, service=service)  # type: ignore[arg-type]
    server = create_mcp(settings, resources=resources)

    def reject_key_resolution(_settings: Settings) -> str:
        raise AssertionError("the ASGI lifecycle must not read the provider key")

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
    """Clean partial real-resource startup once without starting the session manager."""
    database = FailingDatabase()
    resources = ApplicationResources(
        _settings(tmp_path), database=database  # type: ignore[arg-type]
    )
    server = create_mcp(resources.settings, resources=resources)

    with pytest.raises(RuntimeError, match="state startup failed"):
        with TestClient(
            server.streamable_http_app(), base_url="http://127.0.0.1:20441"
        ):
            pytest.fail("the service context must not open after state startup failure")

    assert database.initialize_calls == 1
    assert database.close_calls == 1
    assert server.session_manager._task_group is None
    resources.close()
    assert database.close_calls == 1


def test_safe_error_does_not_echo_internal_message():
    """Do not expose provider bodies embedded in unknown exception messages."""

    async def scenario() -> str:
        async def fail() -> str:
            raise RuntimeError("sentinel-secret-query-token")

        return await _safe_tool_call(fail)

    text = asyncio.run(scenario())
    assert text.startswith("error[internal_error]")
    assert "sentinel" not in text


def test_client_ip_only_accepts_header_from_trusted_proxy(tmp_path):
    """Reject spoofed override headers from public peers and accept trusted proxies."""
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
    """Keep the library deployment-neutral and require explicit self-hosted domains."""
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
    """Simulate a blocking search while lightweight status completes immediately."""

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
    """Count calls that actually enter the session write boundary."""

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
    """Return one candidate window while recording search-thread concurrency."""

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
    """Keep slow provider/search workers off the event loop and state limiter."""

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
        assert "Search session status" in status
        assert elapsed < 0.08
        await search_task
        resources.close()

    asyncio.run(scenario())


def test_anonymous_status_requires_capability(tmp_path):
    """Return stable invalid_request for status without enumerating sessions."""

    async def scenario() -> str:
        settings = _settings(tmp_path)
        resources = ApplicationResources(settings, service=SlowService())  # type: ignore[arg-type]
        resources.initialize()
        try:
            return await search_session_status(_context(resources), None)
        finally:
            resources.close()

    text = asyncio.run(scenario())
    assert text == "error[invalid_request]: `session_id` is required."


def test_open_session_rate_limit_runs_before_session_write(tmp_path):
    """Persistently rate-limit one anonymous source before session writes."""

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
        assert sum(text.startswith("Search session opened") for text in results) == 2
        assert sum(text.startswith("error[rate_limited]") for text in results) == 38

    asyncio.run(scenario())


def test_40_clients_mix_search_and_status_with_same_session(tmp_path):
    """Serve 40 mixed clients without blocking status or duplicating session papers."""

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
        assert all("Search session status" in text for text in status_results)
        assert engine.max_active <= settings.provider_concurrency
        assert resources.admission is not None
        assert resources.admission.peak_admitted_count <= settings.search_queue_size

    asyncio.run(scenario())
