"""编排层：把检索引擎与会话去重拼成对外的业务能力。

这一层不依赖 MCP，可被 CLI、测试或其他 transport 直接复用。
"""

from __future__ import annotations

import logging

from .config import Settings, get_settings
from .models import Paper, SearchFilters, SearchOutcome, SessionSummary
from .retrieval import MilvusSearchEngine, SearchEngine
from .sessions import SessionManager

logger = logging.getLogger(__name__)


class PaperSearchService:
    """带会话去重的论文检索服务。"""

    def __init__(
        self,
        settings: Settings | None = None,
        engine: SearchEngine | None = None,
        sessions: SessionManager | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._engine = engine or MilvusSearchEngine(self._settings)
        self._sessions = sessions or SessionManager(self._settings)

    # --- 会话 ---

    def open_session(
        self, label: str = "", default_top_k: int | None = None
    ) -> SessionSummary:
        """开一个检索会话，后续 query 在其中自动去重。"""
        top_k = self._clamp_top_k(default_top_k) if default_top_k else None
        session = self._sessions.open(label=label, default_top_k=top_k)
        return session.summary()

    def close_session(self, session_id: str) -> SessionSummary:
        """关闭会话并交出汇总（含完整已检索论文清单）。"""
        return self._sessions.close(session_id)

    def session_status(self, session_id: str, *, include_papers: bool = False) -> SessionSummary:
        """查看单个会话的进展。"""
        return self._sessions.require(session_id).summary(include_papers=include_papers)

    def list_sessions(self) -> list[SessionSummary]:
        """列出存活会话。"""
        return self._sessions.list()

    # --- 检索 ---

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
        """混合检索；给定 session_id 时只返回该会话尚未见过的论文。

        每个 query 有一个固定的相关性窗口（默认 `top_k × depth_multiplier`）：
        去重只在窗口内进行，命中全部被此前轮次覆盖时就返回空。

        这里刻意不做"不够就往更深处捞"的自动扩召回——那样虽然每轮都能凑满页，
        但会把结果一路下探到弱相关区，且"该方向挖尽"的信号要几十轮才出现。
        固定窗口让空结果在可预期的几轮内到来，且返回的论文始终是该 query 的强相关项。
        想挖更深时显式调大 `search_depth` 或 `top_k`。
        """
        query = query.strip()
        if not query:
            raise ValueError("query 不能为空")

        filters = filters or SearchFilters()
        session = self._sessions.require(session_id) if session_id else None
        wanted = self._clamp_top_k(
            top_k if top_k is not None else (session.default_top_k if session else None)
        )
        depth = self._resolve_depth(wanted, search_depth)

        candidates = self._engine.search(query, limit=depth, filters=filters)
        if min_score is not None:
            candidates = [hit for hit in candidates if hit.score >= min_score]

        fresh = session.filter_new(candidates) if session else list(candidates)
        returned = fresh[:wanted]
        duplicates = len(candidates) - len(fresh)
        # 窗口被填满且新条目不够，说明窗口外还有结果，调大 depth 能继续挖。
        window_full = len(candidates) >= depth
        truncated = window_full and len(fresh) < wanted

        if session is not None:
            session.record_round(query, filters.describe(), returned, duplicates)

        outcome = SearchOutcome(
            query=query,
            hits=returned,
            search_depth=depth,
            candidates_examined=len(candidates),
            duplicates_suppressed=duplicates,
            filters=filters,
            min_score=min_score,
            session_id=session_id,
            consecutive_empty_rounds=session.consecutive_empty_rounds if session else 0,
            session_unique_total=len(session.seen) if session else 0,
            exhausted=session.exhausted if session else False,
            truncated=truncated,
        )
        logger.info(
            "search %r session=%s depth=%d new=%d dup=%d candidates=%d",
            query,
            session_id,
            depth,
            len(returned),
            duplicates,
            len(candidates),
        )
        return outcome

    def get_papers(
        self, paper_ids: list[str], *, session_id: str | None = None
    ) -> list[Paper]:
        """按 OpenReview id 精确取论文全文摘要；会一并登记进会话去重集。"""
        papers = self._engine.fetch(paper_ids)
        if session_id and papers:
            session = self._sessions.require(session_id)
            session.mark_ids_seen([(p.paper_id, p.title) for p in papers])
        return papers

    def _clamp_top_k(self, top_k: int | None) -> int:
        cfg = self._settings
        if top_k is None:
            return cfg.default_top_k
        return max(1, min(int(top_k), cfg.max_top_k))

    def _resolve_depth(self, wanted: int, search_depth: int | None) -> int:
        """检索窗口深度：显式指定优先，否则按 top_k 等比放大；至少覆盖 top_k。"""
        cfg = self._settings
        depth = search_depth if search_depth is not None else wanted * cfg.depth_multiplier
        return max(wanted, min(int(depth), cfg.max_search_depth))
