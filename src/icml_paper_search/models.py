"""与具体向量数据库无关的论文、检索和 session 领域类型。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

OPENREVIEW_FORUM = "https://openreview.net/forum?id={}"
ARXIV_ABS = "https://arxiv.org/abs/{}"


@dataclass(frozen=True, slots=True)
class Paper:
    """描述一篇 ICML 2026 论文的检索侧视图。"""
    paper_id: str
    title: str
    abstract: str
    authors: tuple[str, ...]
    area: str
    subarea: str
    paper_type: str
    spotlight: bool
    arxiv_id: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Paper:
        """从 Chroma metadata 或 SQLite row 构造论文。"""
        return cls(
            paper_id=str(value.get("paper_id", value.get("orid", "")) or ""),
            title=str(value.get("title", "") or ""),
            abstract=str(value.get("abstract", "") or ""),
            authors=tuple(_load_authors(value.get("authors"))),
            area=str(value.get("area", "") or ""),
            subarea=str(value.get("subarea", value.get("sub", "")) or ""),
            paper_type=str(value.get("paper_type", value.get("type", "")) or ""),
            spotlight=_load_bool(value.get("spotlight", value.get("spot", False))),
            arxiv_id=str(value.get("arxiv_id", value.get("arxiv", "")) or ""),
        )

    @property
    def openreview_url(self) -> str:
        """返回论文的 OpenReview 页面。"""
        return OPENREVIEW_FORUM.format(self.paper_id)

    @property
    def arxiv_url(self) -> str:
        """存在 arXiv id 时返回摘要页。"""
        return ARXIV_ABS.format(self.arxiv_id) if self.arxiv_id else ""


def _load_authors(raw: Any) -> list[str]:
    """容错解析 metadata 中的作者列表。"""
    if isinstance(raw, (list, tuple)):
        return [str(author) for author in raw]
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return [str(raw)]
    return [str(author) for author in parsed] if isinstance(parsed, list) else [str(parsed)]


def _load_bool(raw: Any) -> bool:
    """兼容 SQLite 整数和 Chroma 布尔 metadata。"""
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes"}
    return bool(raw)


@dataclass(frozen=True, slots=True)
class SearchHit:
    """表示一条带 `[0,1]` weighted RRF 得分的命中。"""
    paper: Paper
    score: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("SearchHit.score 必须在 [0,1] 内")

    @property
    def paper_id(self) -> str:
        """返回命中的论文主键。"""
        return self.paper.paper_id


@dataclass(frozen=True, slots=True)
class SearchFilters:
    """描述与检索后端无关的领域过滤条件。"""
    area: str | None = None
    paper_type: str | None = None
    spotlight_only: bool = False

    def describe(self) -> str:
        """返回面向用户的过滤条件摘要。"""
        parts = [
            f"area={self.area}" if self.area else "",
            f"type={self.paper_type}" if self.paper_type else "",
            "spotlight-only" if self.spotlight_only else "",
        ]
        return ", ".join(part for part in parts if part)

    @property
    def is_empty(self) -> bool:
        """说明是否没有设置任何过滤条件。"""
        return not (self.area or self.paper_type or self.spotlight_only)


@dataclass(slots=True)
class SearchOutcome:
    """描述一次检索及其 session 去重统计。"""
    query: str
    hits: list[SearchHit]
    search_depth: int
    candidates_examined: int
    duplicates_suppressed: int
    filters: SearchFilters = field(default_factory=SearchFilters)
    min_score: float | None = None
    session_id: str | None = None
    consecutive_empty_rounds: int = 0
    session_unique_total: int = 0
    exhausted: bool = False
    truncated: bool = False

    @property
    def is_empty(self) -> bool:
        """说明本轮是否没有返回新命中。"""
        return not self.hits


@dataclass(frozen=True, slots=True)
class QueryRecord:
    """保存不含完整 query 的有界 session 审计记录。"""
    query_hash: str
    query_summary: str | None
    filters_desc: str
    new_hits: int
    duplicates_suppressed: int
    candidates_examined: int
    at: datetime


@dataclass(frozen=True, slots=True)
class RoundCommit:
    """返回与 SQLite 原子提交完全一致的一轮状态。"""
    hits: list[SearchHit]
    duplicates_suppressed: int
    consecutive_empty_rounds: int
    session_unique_total: int
    exhausted: bool
    query_count: int
    version: int


@dataclass(slots=True)
class SessionSummary:
    """描述一个 capability session 的有界对外快照。"""
    session_id: str
    label: str
    created_at: datetime
    last_active_at: datetime
    query_count: int
    unique_papers: int
    consecutive_empty_rounds: int
    exhausted: bool
    default_top_k: int | None
    queries: list[QueryRecord]
    papers: list[tuple[str, str]]
    audit_truncated: bool = False
    papers_truncated: bool = False
