"""Orchestrate Milvus hybrid retrieval and SQLite capability sessions."""

from __future__ import annotations

import logging

from ..config import Settings, get_settings
from ..domain.errors import InvalidRequestError
from ..domain.models import Paper, SearchFilters, SearchOutcome, SessionSummary
from ..embeddings.dashscope import DashScopeEncoder
from ..milvus.repository import MilvusPaperRepository, SearchEngine
from ..runtime.database import RuntimeDatabase, hash_query
from ..runtime.embedding_cache import EmbeddingCache, GovernedQueryEncoder
from ..runtime.provider_quota import SQLiteProviderQuota
from ..runtime.sessions import SessionManager, SQLiteSessionStore

logger = logging.getLogger(__name__)


class PaperSearchService:
    """Preserve fixed-window semantics and eliminate races with atomic session commits."""

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
        if (sessions is None or engine is None) and self._database is None:
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
                SQLiteProviderQuota(self._database, self._settings),
                self._settings,
            )
            engine = MilvusPaperRepository(self._settings, governed_encoder)
        self._engine = engine
        self._sessions = sessions

    def open_session(
        self, label: str = "", default_top_k: int | None = None
    ) -> SessionSummary:
        """Create a search session with capability-based automatic deduplication."""
        top_k = self._clamp_top_k(default_top_k) if default_top_k is not None else None
        return self._sessions.open(label=label, default_top_k=top_k)

    def close_session(self, session_id: str) -> SessionSummary:
        """Atomically close a session and return a bounded summary."""
        return self._sessions.close(session_id, include_papers=True)

    def session_status(
        self, session_id: str, *, include_papers: bool = False
    ) -> SessionSummary:
        """Inspect one session only through a caller-held capability."""
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
        """Run one fixed-depth search and atomically commit returned items to the session."""
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise InvalidRequestError(public_message="query must not be empty.")
        if len(normalized_query) > self._settings.max_query_chars:
            raise InvalidRequestError(
                public_message="query exceeds the maximum length."
            )
        if min_score is not None and not 0.0 <= float(min_score) <= 1.0:
            raise InvalidRequestError(public_message="min_score must be within [0,1].")

        filters = filters or SearchFilters()
        session = None
        if session_id is not None:
            # Check capability and TTL before remote I/O; commit_round ensures final correctness.
            session = self._sessions.status(
                session_id, include_papers=False, include_queries=False
            )
        wanted = self._clamp_top_k(
            top_k if top_k is not None else (session.default_top_k if session else None)
        )
        depth = self._resolve_depth(wanted, search_depth)

        candidates = self._engine.search(normalized_query, limit=depth, filters=filters)
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
        """Fetch papers in input order and atomically record those that exist."""
        if len(paper_ids) > self._settings.max_get_paper_ids:
            raise InvalidRequestError(
                public_message="paper_ids exceeds the maximum item count."
            )
        papers = self._engine.fetch(paper_ids)
        if session_id is not None:
            self._sessions.mark_ids_seen(
                session_id, ((paper.paper_id, paper.title) for paper in papers)
            )
        return papers

    def validate(self):
        """Run local index readiness checks without provider access."""
        validate = getattr(self._engine, "validate", None)
        if not callable(validate):
            return None
        return validate()

    def close(self) -> None:
        """Close the engine and the runtime database created by this service."""
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
        """Prefer explicit depth while clamping it between wanted and the global limit."""
        settings = self._settings
        depth = (
            search_depth
            if search_depth is not None
            else wanted * settings.depth_multiplier
        )
        return max(wanted, min(int(depth), settings.max_search_depth))
