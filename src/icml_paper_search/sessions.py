"""检索会话：记住已返回过的论文，让后续 query 只吐新条目。

会话状态刻意保持为纯内存 + 无 IO，`SessionStore` 协议留出持久化扩展点。
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Protocol, runtime_checkable

from .config import Settings, get_settings
from .models import QueryRecord, SearchHit, SessionSummary


class SessionNotFoundError(KeyError):
    """引用了不存在或已关闭的会话。"""

    def __init__(self, session_id: str) -> None:
        super().__init__(session_id)
        self.session_id = session_id

    def __str__(self) -> str:
        return f"检索会话 {self.session_id!r} 不存在或已关闭"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class SearchSession:
    """单个检索会话的可变状态。"""

    session_id: str
    label: str
    created_at: datetime
    last_active_at: datetime
    default_top_k: int | None = None
    exhausted_after_empty_rounds: int = 3
    consecutive_empty_rounds: int = 0
    # paper_id -> title，dict 兼作去重集合并保留首次检索顺序
    seen: dict[str, str] = field(default_factory=dict)
    queries: list[QueryRecord] = field(default_factory=list)

    def is_new(self, paper_id: str) -> bool:
        return paper_id not in self.seen

    def filter_new(self, hits: Iterable[SearchHit]) -> list[SearchHit]:
        """滤掉本会话已返回过的条目，不改变会话状态。"""
        return [hit for hit in hits if hit.paper_id not in self.seen]

    def mark_seen(self, hits: Iterable[SearchHit]) -> None:
        """把条目登记为已返回，后续 query 不再展示。"""
        for hit in hits:
            self.seen.setdefault(hit.paper_id, hit.paper.title)

    def mark_ids_seen(self, papers: Iterable[tuple[str, str]]) -> None:
        """直接按 (paper_id, title) 登记，用于 get_papers 这类显式取回。"""
        for paper_id, title in papers:
            self.seen.setdefault(paper_id, title)

    def record_round(
        self,
        query: str,
        filters_desc: str,
        new_hits: list[SearchHit],
        duplicates_suppressed: int,
    ) -> None:
        """记录一轮检索：登记新条目、更新连续空轮计数与审计日志。"""
        self.mark_seen(new_hits)
        self.consecutive_empty_rounds = 0 if new_hits else self.consecutive_empty_rounds + 1
        self.queries.append(
            QueryRecord(
                query=query,
                filters_desc=filters_desc,
                new_hits=len(new_hits),
                duplicates_suppressed=duplicates_suppressed,
                at=_now(),
            )
        )
        self.touch()

    def touch(self) -> None:
        self.last_active_at = _now()

    @property
    def exhausted(self) -> bool:
        """连续空轮达到阈值，说明当前检索方向基本挖尽。"""
        return self.consecutive_empty_rounds >= self.exhausted_after_empty_rounds

    def summary(self, *, include_papers: bool = True, include_queries: bool = True) -> SessionSummary:
        return SessionSummary(
            session_id=self.session_id,
            label=self.label,
            created_at=self.created_at,
            last_active_at=self.last_active_at,
            query_count=len(self.queries),
            unique_papers=len(self.seen),
            consecutive_empty_rounds=self.consecutive_empty_rounds,
            exhausted=self.exhausted,
            default_top_k=self.default_top_k,
            queries=list(self.queries) if include_queries else [],
            papers=list(self.seen.items()) if include_papers else [],
        )


@runtime_checkable
class SessionStore(Protocol):
    """会话存储抽象；换成 SQLite / Redis 只需实现这四个方法。"""

    def put(self, session: SearchSession) -> None: ...

    def get(self, session_id: str) -> SearchSession | None: ...

    def pop(self, session_id: str) -> SearchSession | None: ...

    def all(self) -> list[SearchSession]: ...


class InMemorySessionStore:
    """进程内会话存储。会话存活期等于 server 进程生命周期（HTTP 长驻进程内跨请求共享）。"""

    def __init__(self) -> None:
        self._sessions: dict[str, SearchSession] = {}

    def put(self, session: SearchSession) -> None:
        self._sessions[session.session_id] = session

    def get(self, session_id: str) -> SearchSession | None:
        return self._sessions.get(session_id)

    def pop(self, session_id: str) -> SearchSession | None:
        return self._sessions.pop(session_id, None)

    def all(self) -> list[SearchSession]:
        return list(self._sessions.values())


class SessionManager:
    """会话的生命周期管理：创建、查找、关闭、按空闲时间与数量回收。

    所有公开方法都在一把锁下执行，可安全用于并发的 MCP 请求。
    """

    def __init__(
        self,
        settings: Settings | None = None,
        store: SessionStore | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._store = store or InMemorySessionStore()
        self._lock = threading.RLock()

    def open(self, label: str = "", default_top_k: int | None = None) -> SearchSession:
        """新建会话；超出上限时先回收最久未活跃的会话。"""
        with self._lock:
            self._evict_expired()
            self._enforce_capacity()
            now = _now()
            session = SearchSession(
                session_id=f"ses_{uuid.uuid4().hex[:12]}",
                label=label or "untitled",
                created_at=now,
                last_active_at=now,
                default_top_k=default_top_k,
                exhausted_after_empty_rounds=self._settings.exhausted_after_empty_rounds,
            )
            self._store.put(session)
            return session

    def require(self, session_id: str) -> SearchSession:
        """取会话，不存在则抛 `SessionNotFoundError`。"""
        with self._lock:
            session = self._store.get(session_id)
            if session is None:
                raise SessionNotFoundError(session_id)
            session.touch()
            return session

    def close(self, session_id: str) -> SessionSummary:
        """关闭会话并返回最终统计。"""
        with self._lock:
            session = self._store.pop(session_id)
            if session is None:
                raise SessionNotFoundError(session_id)
            return session.summary()

    def list(self) -> list[SessionSummary]:
        """列出全部存活会话（不含逐条论文与 query 明细）。"""
        with self._lock:
            self._evict_expired()
            sessions = sorted(
                self._store.all(), key=lambda s: s.last_active_at, reverse=True
            )
            return [
                s.summary(include_papers=False, include_queries=False) for s in sessions
            ]

    def _evict_expired(self) -> None:
        ttl = self._settings.session_idle_ttl_seconds
        cutoff = _now().timestamp() - ttl
        for session in self._store.all():
            if session.last_active_at.timestamp() < cutoff:
                self._store.pop(session.session_id)

    def _enforce_capacity(self) -> None:
        sessions = self._store.all()
        overflow = len(sessions) - self._settings.max_sessions + 1
        if overflow <= 0:
            return
        for session in sorted(sessions, key=lambda s: s.last_active_at)[:overflow]:
            self._store.pop(session.session_id)
