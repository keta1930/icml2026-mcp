"""把本地混合检索与 SQLite capability session 编排成业务能力。"""

from __future__ import annotations

import logging

from .config import Settings, get_settings
from .embedding import DashScopeEncoder
from .errors import InvalidRequestError
from .governance import EmbeddingCache, GovernedQueryEncoder, SQLiteBudgetLedger
from .models import Paper, SearchFilters, SearchOutcome, SessionSummary
from .retrieval import ChromaHybridSearchEngine, SearchEngine
from .sessions import SessionManager, SQLiteSessionStore
from .state import RuntimeDatabase, hash_query

logger = logging.getLogger(__name__)


class PaperSearchService:
    """保留 fixed-window 语义，并用原子 session 提交消除竞态。"""

    def __init__(
        self,
        settings: Settings | None = None,
        engine: SearchEngine | None = None,
        sessions: SessionManager | None = None,
        *,
        database: RuntimeDatabase | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._database = database
        self._owns_database = False
        if sessions is None or engine is None:
            if self._database is None:
                self._database = RuntimeDatabase(
                    self._settings.state_db_path,
                    busy_timeout_ms=self._settings.sqlite_busy_timeout_ms,
                )
                self._database.initialize()
                self._owns_database = True
        if sessions is None:
            assert self._database is not None
            sessions = SessionManager(
                self._settings, SQLiteSessionStore(self._database, self._settings)
            )
        if engine is None:
            assert self._database is not None
            provider = DashScopeEncoder(self._settings)
            governed_encoder = GovernedQueryEncoder(
                provider,
                EmbeddingCache(self._database, self._settings),
                SQLiteBudgetLedger(self._database, self._settings),
                self._settings,
            )
            engine = ChromaHybridSearchEngine(self._settings, governed_encoder)
        self._engine = engine
        self._sessions = sessions

    def open_session(
        self, label: str = "", default_top_k: int | None = None
    ) -> SessionSummary:
        """创建一个后续按 capability 自动去重的检索会话。"""
        top_k = (
            self._clamp_top_k(default_top_k)
            if default_top_k is not None
            else None
        )
        return self._sessions.open(label=label, default_top_k=top_k)

    def close_session(self, session_id: str) -> SessionSummary:
        """原子关闭会话并返回有界汇总。"""
        return self._sessions.close(session_id, include_papers=True)

    def session_status(
        self, session_id: str, *, include_papers: bool = False
    ) -> SessionSummary:
        """只按调用方持有的 capability 查看单个 session。"""
        return self._sessions.status(
            session_id, include_papers=include_papers, include_queries=True
        )

    def search(
        self,
        query: str,
        *,
        session_id: str | None = None,
        top_k: int | None = None,
        search_depth: int | None = None,
        filters: SearchFilters | None = None,
        min_score: float | None = None,
    ) -> SearchOutcome:
        """执行一次固定深度检索，并在 session 内原子提交实际返回项。"""
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise InvalidRequestError(public_message="query 不能为空。")
        if len(normalized_query) > self._settings.max_query_chars:
            raise InvalidRequestError(public_message="query 超过长度上限。")
        if min_score is not None and not 0.0 <= float(min_score) <= 1.0:
            raise InvalidRequestError(public_message="min_score 必须在 [0,1] 内。")

        filters = filters or SearchFilters()
        session = None
        if session_id is not None:
            # 远端 IO 前先做 capability/TTL 检查，但最终正确性由 commit_round 决定。
            session = self._sessions.status(
                session_id, include_papers=False, include_queries=False
            )
        wanted = self._clamp_top_k(
            top_k if top_k is not None else (session.default_top_k if session else None)
        )
        depth = self._resolve_depth(wanted, search_depth)

        candidates = self._engine.search(
            normalized_query, limit=depth, filters=filters
        )
        if min_score is not None:
            candidates = [hit for hit in candidates if hit.score >= min_score]
        candidates_examined = len(candidates)

        if session_id is None:
            returned = list(candidates[:wanted])
            duplicates = 0
            empty_rounds = 0
            unique_total = 0
            exhausted = False
            fresh_count = len(candidates)
        else:
            commit = self._sessions.commit_round(
                session_id,
                candidates,
                wanted=wanted,
                query_hash=hash_query(normalized_query),
                query_summary=None,
                filters_desc=filters.describe(),
                candidates_examined=candidates_examined,
            )
            returned = commit.hits
            duplicates = commit.duplicates_suppressed
            empty_rounds = commit.consecutive_empty_rounds
            unique_total = commit.session_unique_total
            exhausted = commit.exhausted
            fresh_count = candidates_examined - duplicates

        window_full = candidates_examined >= depth
        truncated = window_full and fresh_count < wanted
        outcome = SearchOutcome(
            query=normalized_query,
            hits=returned,
            search_depth=depth,
            candidates_examined=candidates_examined,
            duplicates_suppressed=duplicates,
            filters=filters,
            min_score=min_score,
            session_id=session_id,
            consecutive_empty_rounds=empty_rounds,
            session_unique_total=unique_total,
            exhausted=exhausted,
            truncated=truncated,
        )
        logger.info(
            "search query_hash=%s query_chars=%d depth=%d new=%d dup=%d candidates=%d",
            hash_query(normalized_query)[:16],
            len(normalized_query),
            depth,
            len(returned),
            duplicates,
            candidates_examined,
        )
        return outcome

    def get_papers(
        self, paper_ids: list[str], *, session_id: str | None = None
    ) -> list[Paper]:
        """按输入顺序精确取回论文，并原子登记实际存在的结果。"""
        if len(paper_ids) > self._settings.max_get_paper_ids:
            raise InvalidRequestError(public_message="paper_ids 超过数量上限。")
        papers = self._engine.fetch(paper_ids)
        if session_id is not None:
            self._sessions.mark_ids_seen(
                session_id, ((paper.paper_id, paper.title) for paper in papers)
            )
        return papers

    def validate(self):
        """执行不访问 provider 的本地索引 readiness 检查。"""
        validate = getattr(self._engine, "validate", None)
        if not callable(validate):
            return None
        return validate()

    def close(self) -> None:
        """关闭引擎及由本服务创建的 runtime database。"""
        close_engine = getattr(self._engine, "close", None)
        if callable(close_engine):
            close_engine()
        if self._owns_database and self._database is not None:
            self._database.close()

    def _clamp_top_k(self, top_k: int | None) -> int:
        settings = self._settings
        if top_k is None:
            return settings.default_top_k
        return max(1, min(int(top_k), settings.max_top_k))

    def _resolve_depth(self, wanted: int, search_depth: int | None) -> int:
        """显式 depth 优先，始终夹在 wanted 与全局上限之间。"""
        settings = self._settings
        depth = (
            search_depth
            if search_depth is not None
            else wanted * settings.depth_multiplier
        )
        return max(wanted, min(int(depth), settings.max_search_depth))
