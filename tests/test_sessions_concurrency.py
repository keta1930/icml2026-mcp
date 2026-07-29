"""Concurrency, persistence, and boundary tests for SQLite capability sessions."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from icml_paper_search.config import Settings
from icml_paper_search.domain.errors import SessionCapacityReached, SessionNotFoundError
from icml_paper_search.domain.models import Paper, SearchHit
from icml_paper_search.runtime.database import RuntimeDatabase, hash_capability
from icml_paper_search.runtime.sessions import SessionManager, SQLiteSessionStore


class Clock:
    """Provide an advanceable UTC test clock."""

    def __init__(self) -> None:
        self.value = datetime(2026, 7, 28, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value


def _settings(tmp_path, **overrides) -> Settings:
    values = {
        "data_dir": tmp_path,
        "index_root": tmp_path / "index",
        "state_db_path": tmp_path / "runtime.sqlite3",
        "default_top_k": 5,
        "max_top_k": 20,
        "max_search_depth": 40,
        "max_sessions": 3,
        "session_idle_ttl_seconds": 60,
        "max_session_rounds": 20,
        "max_session_audit_records": 20,
        "session_summary_paper_limit": 3,
    }
    values.update(overrides)
    return Settings(**values)


def _manager(settings: Settings, clock: Clock) -> SessionManager:
    database = RuntimeDatabase(settings.state_db_path)
    return SessionManager(settings, SQLiteSessionStore(database, settings), clock=clock)


def _hits(count: int) -> list[SearchHit]:
    return [
        SearchHit(
            paper=Paper(
                paper_id=f"p{index:03d}",
                title=f"Paper {index}",
                abstract="",
                authors=(),
                area="",
                subarea="",
                paper_type="",
                spotlight=False,
                arxiv_id="",
            ),
            score=1.0 - index * 0.01,
        )
        for index in range(count)
    ]


def _commit(manager: SessionManager, token: str, hits: list[SearchHit]):
    return manager.commit_round(
        token,
        hits,
        wanted=5,
        query_hash="a" * 64,
        filters_desc="",
        candidates_examined=len(hits),
    )


def test_concurrent_rounds_never_return_same_paper(tmp_path):
    """Base concurrent transactions on committed seen state before taking the next page."""
    clock = Clock()
    manager = _manager(_settings(tmp_path), clock)
    token = manager.open().session_id
    hits = _hits(10)

    with ThreadPoolExecutor(max_workers=2) as executor:
        rounds = list(
            executor.map(lambda _index: _commit(manager, token, hits), range(2))
        )

    first = {hit.paper_id for hit in rounds[0].hits}
    second = {hit.paper_id for hit in rounds[1].hits}
    assert not first & second
    assert first | second == {hit.paper_id for hit in hits}


def test_capability_survives_restart_but_raw_token_is_never_stored(tmp_path):
    """Keep sessions usable after reopening while storing only capability hashes."""
    settings = _settings(tmp_path)
    clock = Clock()
    token = _manager(settings, clock).open(label="restart").session_id
    _commit(_manager(settings, clock), token, _hits(2))

    restarted = _manager(settings, clock)
    assert restarted.status(token).unique_papers == 2
    with sqlite3.connect(settings.state_db_path) as connection:
        dump = "\n".join(connection.iterdump())
    assert token not in dump
    assert hash_capability(token) in dump


def test_capacity_rejects_without_evicting_active_session(tmp_path):
    """Reject at capacity without cross-caller LRU eviction."""
    settings = _settings(tmp_path, max_sessions=1)
    clock = Clock()
    manager = _manager(settings, clock)
    token = manager.open().session_id

    with pytest.raises(SessionCapacityReached):
        manager.open()
    assert manager.status(token).session_id == token


def test_expired_session_is_reclaimed_and_not_distinguishable(tmp_path):
    """Return the same error for expired and random capabilities."""
    settings = _settings(tmp_path, max_sessions=1)
    clock = Clock()
    manager = _manager(settings, clock)
    expired = manager.open().session_id
    clock.value += timedelta(seconds=settings.session_idle_ttl_seconds + 1)
    replacement = manager.open().session_id

    assert replacement != expired
    with pytest.raises(SessionNotFoundError):
        manager.status(expired)
    with pytest.raises(SessionNotFoundError):
        manager.status("ses_invalid")


def test_close_is_atomic_and_summary_is_bounded(tmp_path):
    """Return a bounded close summary and invalidate the capability immediately."""
    settings = _settings(tmp_path, session_summary_paper_limit=3)
    clock = Clock()
    manager = _manager(settings, clock)
    token = manager.open().session_id
    _commit(manager, token, _hits(5))

    summary = manager.close(token)
    assert len(summary.papers) == 3
    assert summary.papers_truncated
    with pytest.raises(SessionNotFoundError):
        manager.close(token)
