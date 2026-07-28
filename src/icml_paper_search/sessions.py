"""SQLite capability session 的原子生命周期与去重操作。"""

from __future__ import annotations

import secrets
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from .config import Settings, get_settings
from .errors import InvalidRequestError, SessionCapacityReached, SessionNotFoundError
from .models import QueryRecord, RoundCommit, SearchHit, SessionSummary
from .state import RuntimeDatabase, hash_capability


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_timestamp(value: datetime) -> float:
    """把时区安全的时间转换为 SQLite 使用的 UTC timestamp。"""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


def _as_datetime(value: float) -> datetime:
    return datetime.fromtimestamp(value, tz=timezone.utc)


@runtime_checkable
class SessionStore(Protocol):
    """定义不暴露可变 session 的原子持久化边界。"""

    def open(
        self,
        token_hash: str,
        *,
        label: str,
        default_top_k: int | None,
        now: datetime,
    ) -> None: ...

    def commit_round(
        self,
        token_hash: str,
        candidates: Sequence[SearchHit],
        *,
        wanted: int,
        query_hash: str,
        query_summary: str | None,
        filters_desc: str,
        candidates_examined: int,
        now: datetime,
    ) -> RoundCommit: ...

    def mark_ids_seen(
        self,
        token_hash: str,
        papers: Iterable[tuple[str, str]],
        *,
        now: datetime,
    ) -> int: ...

    def status(
        self,
        token_hash: str,
        *,
        include_papers: bool,
        include_queries: bool,
        now: datetime,
    ) -> SessionSummary: ...

    def close(
        self,
        token_hash: str,
        *,
        include_papers: bool,
        now: datetime,
    ) -> SessionSummary: ...

    def cleanup_expired(self, *, now: datetime) -> int: ...


class SQLiteSessionStore:
    """用短 `BEGIN IMMEDIATE` 事务维护 session 的一致状态。"""

    def __init__(self, database: RuntimeDatabase, settings: Settings | None = None) -> None:
        self._database = database
        self._settings = settings or get_settings()
        self._database.initialize()

    def open(
        self,
        token_hash: str,
        *,
        label: str,
        default_top_k: int | None,
        now: datetime,
    ) -> None:
        """仅在回收过期项后仍有容量时创建 session。"""
        timestamp = _as_timestamp(now)
        cutoff = timestamp - self._settings.session_idle_ttl_seconds
        with self._database.transaction() as connection:
            self._delete_expired(connection, cutoff)
            active_count = connection.execute(
                "SELECT COUNT(*) FROM sessions WHERE status = 'active'"
            ).fetchone()[0]
            if active_count >= self._settings.max_sessions:
                raise SessionCapacityReached()
            connection.execute(
                """INSERT INTO sessions(
                       token_hash, label, created_at, last_active_at, default_top_k
                   ) VALUES (?, ?, ?, ?, ?)""",
                (token_hash, label, timestamp, timestamp, default_top_k),
            )

    def commit_round(
        self,
        token_hash: str,
        candidates: Sequence[SearchHit],
        *,
        wanted: int,
        query_hash: str,
        query_summary: str | None,
        filters_desc: str,
        candidates_examined: int,
        now: datetime,
    ) -> RoundCommit:
        """原子过滤已见项、登记返回项并提交轮次审计。"""
        if wanted < 1:
            raise InvalidRequestError()
        timestamp = _as_timestamp(now)
        cutoff = timestamp - self._settings.session_idle_ttl_seconds
        with self._database.transaction() as connection:
            session = self._require_active(connection, token_hash, cutoff)
            if session["query_count"] >= self._settings.max_session_rounds:
                raise InvalidRequestError(public_message="该检索会话已达到轮次上限。")

            seen_rows = connection.execute(
                "SELECT paper_id FROM session_seen WHERE token_hash = ?",
                (token_hash,),
            ).fetchall()
            seen = {row["paper_id"] for row in seen_rows}
            fresh: list[SearchHit] = []
            candidate_ids: set[str] = set()
            duplicates = 0
            for hit in candidates:
                if hit.paper_id in seen or hit.paper_id in candidate_ids:
                    duplicates += 1
                    continue
                candidate_ids.add(hit.paper_id)
                fresh.append(hit)

            remaining = max(0, self._settings.max_session_seen - len(seen))
            returned = fresh[: min(wanted, remaining)]
            first_ordinal = len(seen) + 1
            connection.executemany(
                """INSERT INTO session_seen(
                       token_hash, paper_id, title, first_seen_at, ordinal
                   ) VALUES (?, ?, ?, ?, ?)""",
                [
                    (
                        token_hash,
                        hit.paper_id,
                        hit.paper.title,
                        timestamp,
                        first_ordinal + offset,
                    )
                    for offset, hit in enumerate(returned)
                ],
            )

            empty_rounds = 0 if returned else session["empty_rounds"] + 1
            query_count = session["query_count"] + 1
            version = session["version"] + 1
            connection.execute(
                """UPDATE sessions
                   SET last_active_at = ?, empty_rounds = ?, query_count = ?, version = ?
                   WHERE token_hash = ?""",
                (timestamp, empty_rounds, query_count, version, token_hash),
            )
            if query_count <= self._settings.max_session_audit_records:
                connection.execute(
                    """INSERT INTO session_queries(
                           token_hash, ordinal, query_hash, query_summary, filters_desc,
                           new_hits, duplicates_suppressed, candidates_examined, created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        token_hash,
                        query_count,
                        query_hash,
                        query_summary,
                        filters_desc,
                        len(returned),
                        duplicates,
                        candidates_examined,
                        timestamp,
                    ),
                )
            unique_total = len(seen) + len(returned)

        return RoundCommit(
            hits=returned,
            duplicates_suppressed=duplicates,
            consecutive_empty_rounds=empty_rounds,
            session_unique_total=unique_total,
            exhausted=empty_rounds >= self._settings.exhausted_after_empty_rounds,
            query_count=query_count,
            version=version,
        )

    def mark_ids_seen(
        self,
        token_hash: str,
        papers: Iterable[tuple[str, str]],
        *,
        now: datetime,
    ) -> int:
        """原子登记精确取回且实际存在的论文。"""
        timestamp = _as_timestamp(now)
        cutoff = timestamp - self._settings.session_idle_ttl_seconds
        unique_by_id: dict[str, str] = {}
        for paper_id, title in papers:
            unique_by_id.setdefault(paper_id, title)
        unique_papers = list(unique_by_id.items())
        with self._database.transaction() as connection:
            session = self._require_active(connection, token_hash, cutoff)
            rows = connection.execute(
                "SELECT paper_id FROM session_seen WHERE token_hash = ?",
                (token_hash,),
            ).fetchall()
            seen = {row["paper_id"] for row in rows}
            additions = [paper for paper in unique_papers if paper[0] not in seen]
            remaining = max(0, self._settings.max_session_seen - len(seen))
            additions = additions[:remaining]
            connection.executemany(
                """INSERT INTO session_seen(
                       token_hash, paper_id, title, first_seen_at, ordinal
                   ) VALUES (?, ?, ?, ?, ?)""",
                [
                    (token_hash, paper_id, title, timestamp, len(seen) + offset + 1)
                    for offset, (paper_id, title) in enumerate(additions)
                ],
            )
            connection.execute(
                "UPDATE sessions SET last_active_at = ?, version = ? WHERE token_hash = ?",
                (timestamp, session["version"] + 1, token_hash),
            )
            return len(seen) + len(additions)

    def status(
        self,
        token_hash: str,
        *,
        include_papers: bool,
        include_queries: bool,
        now: datetime,
    ) -> SessionSummary:
        """读取有界快照并刷新 session 的活跃时间。"""
        timestamp = _as_timestamp(now)
        cutoff = timestamp - self._settings.session_idle_ttl_seconds
        with self._database.transaction() as connection:
            row = self._require_active(connection, token_hash, cutoff)
            connection.execute(
                "UPDATE sessions SET last_active_at = ? WHERE token_hash = ?",
                (timestamp, token_hash),
            )
            return self._summary_from_row(
                connection,
                row,
                session_id="",
                last_active_at=timestamp,
                include_papers=include_papers,
                include_queries=include_queries,
            )

    def close(
        self,
        token_hash: str,
        *,
        include_papers: bool,
        now: datetime,
    ) -> SessionSummary:
        """在同一事务内读取最终摘要并删除 capability 状态。"""
        timestamp = _as_timestamp(now)
        cutoff = timestamp - self._settings.session_idle_ttl_seconds
        with self._database.transaction() as connection:
            row = self._require_active(connection, token_hash, cutoff)
            summary = self._summary_from_row(
                connection,
                row,
                session_id="",
                last_active_at=timestamp,
                include_papers=include_papers,
                include_queries=True,
            )
            connection.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
            return summary

    def cleanup_expired(self, *, now: datetime) -> int:
        """删除超过严格 idle TTL 的 active session。"""
        cutoff = _as_timestamp(now) - self._settings.session_idle_ttl_seconds
        with self._database.transaction() as connection:
            return self._delete_expired(connection, cutoff)

    @staticmethod
    def _delete_expired(connection, cutoff: float) -> int:
        cursor = connection.execute(
            "DELETE FROM sessions WHERE status = 'active' AND last_active_at < ?",
            (cutoff,),
        )
        return cursor.rowcount

    @staticmethod
    def _require_active(connection, token_hash: str, cutoff: float):
        row = connection.execute(
            "SELECT * FROM sessions WHERE token_hash = ? AND status = 'active'",
            (token_hash,),
        ).fetchone()
        if row is None:
            raise SessionNotFoundError()
        if row["last_active_at"] < cutoff:
            connection.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
            raise SessionNotFoundError()
        return row

    def _summary_from_row(
        self,
        connection,
        row,
        *,
        session_id: str,
        last_active_at: float,
        include_papers: bool,
        include_queries: bool,
    ) -> SessionSummary:
        query_rows = []
        if include_queries:
            query_rows = connection.execute(
                """SELECT * FROM session_queries
                   WHERE token_hash = ? ORDER BY ordinal LIMIT ?""",
                (row["token_hash"], self._settings.max_session_audit_records),
            ).fetchall()
        paper_rows = []
        if include_papers and self._settings.session_summary_paper_limit:
            paper_rows = connection.execute(
                """SELECT paper_id, title FROM session_seen
                   WHERE token_hash = ? ORDER BY ordinal LIMIT ?""",
                (row["token_hash"], self._settings.session_summary_paper_limit),
            ).fetchall()
        unique_papers = connection.execute(
            "SELECT COUNT(*) FROM session_seen WHERE token_hash = ?",
            (row["token_hash"],),
        ).fetchone()[0]
        queries = [
            QueryRecord(
                query_hash=query["query_hash"],
                query_summary=query["query_summary"],
                filters_desc=query["filters_desc"],
                new_hits=query["new_hits"],
                duplicates_suppressed=query["duplicates_suppressed"],
                candidates_examined=query["candidates_examined"],
                at=_as_datetime(query["created_at"]),
            )
            for query in query_rows
        ]
        return SessionSummary(
            session_id=session_id,
            label=row["label"],
            created_at=_as_datetime(row["created_at"]),
            last_active_at=_as_datetime(last_active_at),
            query_count=row["query_count"],
            unique_papers=unique_papers,
            consecutive_empty_rounds=row["empty_rounds"],
            exhausted=row["empty_rounds"] >= self._settings.exhausted_after_empty_rounds,
            default_top_k=row["default_top_k"],
            queries=queries,
            papers=[(paper["paper_id"], paper["title"]) for paper in paper_rows],
            audit_truncated=row["query_count"] > len(queries) if include_queries else False,
            papers_truncated=unique_papers > len(paper_rows) if include_papers else False,
        )


class SessionManager:
    """生成 raw capability，并在存储边界只使用其 SHA-256 摘要。"""

    def __init__(
        self,
        settings: Settings | None = None,
        store: SessionStore | None = None,
        *,
        clock=_utc_now,
    ) -> None:
        self._settings = settings or get_settings()
        if store is None:
            database = RuntimeDatabase(
                self._settings.state_db_path,
                busy_timeout_ms=self._settings.sqlite_busy_timeout_ms,
            )
            store = SQLiteSessionStore(database, self._settings)
        self._store = store
        self._clock = clock

    def open(self, label: str = "", default_top_k: int | None = None) -> SessionSummary:
        """创建至少 256-bit 的不可枚举 capability session。"""
        normalized_label = label.strip() or "untitled"
        if len(normalized_label) > self._settings.session_label_max_chars:
            raise InvalidRequestError(public_message="会话标签过长。")
        token = f"ses_{secrets.token_urlsafe(32)}"
        now = self._clock()
        self._store.open(
            hash_capability(token),
            label=normalized_label,
            default_top_k=default_top_k,
            now=now,
        )
        return SessionSummary(
            session_id=token,
            label=normalized_label,
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

    def commit_round(
        self,
        session_id: str,
        candidates: Sequence[SearchHit],
        *,
        wanted: int,
        query_hash: str,
        query_summary: str | None,
        filters_desc: str,
        candidates_examined: int,
    ) -> RoundCommit:
        """提交与本次返回严格一致的原子轮次。"""
        return self._store.commit_round(
            self._token_hash(session_id),
            candidates,
            wanted=wanted,
            query_hash=query_hash,
            query_summary=query_summary,
            filters_desc=filters_desc,
            candidates_examined=candidates_examined,
            now=self._clock(),
        )

    def mark_ids_seen(self, session_id: str, papers: Iterable[tuple[str, str]]) -> int:
        """登记精确取回的论文，并验证 capability 仍有效。"""
        return self._store.mark_ids_seen(
            self._token_hash(session_id), papers, now=self._clock()
        )

    def status(
        self,
        session_id: str,
        *,
        include_papers: bool = False,
        include_queries: bool = True,
    ) -> SessionSummary:
        """返回调用方持有 capability 对应的有界状态。"""
        summary = self._store.status(
            self._token_hash(session_id),
            include_papers=include_papers,
            include_queries=include_queries,
            now=self._clock(),
        )
        summary.session_id = session_id
        return summary

    def close(self, session_id: str, *, include_papers: bool = True) -> SessionSummary:
        """关闭 capability 并返回有界最终摘要。"""
        summary = self._store.close(
            self._token_hash(session_id), include_papers=include_papers, now=self._clock()
        )
        summary.session_id = session_id
        return summary

    def cleanup_expired(self) -> int:
        """触发显式 TTL 清理。"""
        return self._store.cleanup_expired(now=self._clock())

    @staticmethod
    def _token_hash(session_id: str) -> str:
        if not session_id or not session_id.startswith("ses_"):
            raise SessionNotFoundError()
        return hash_capability(session_id)
