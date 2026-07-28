"""领域模型：论文、检索命中、过滤条件与检索结果。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

OPENREVIEW_FORUM = "https://openreview.net/forum?id={}"
ARXIV_ABS = "https://arxiv.org/abs/{}"


@dataclass(frozen=True, slots=True)
class Paper:
    """一篇 ICML 2026 论文的检索侧视图。"""

    paper_id: str  # OpenReview id，全局主键
    title: str
    abstract: str
    authors: tuple[str, ...]
    area: str
    subarea: str
    paper_type: str
    spotlight: bool
    arxiv_id: str

    @classmethod
    def from_entity(cls, entity: dict[str, Any]) -> Paper:
        """从 Milvus entity 反序列化。"""
        return cls(
            paper_id=entity.get("orid", ""),
            title=entity.get("title", ""),
            abstract=entity.get("abstract", ""),
            authors=tuple(_load_authors(entity.get("authors"))),
            area=entity.get("area", ""),
            subarea=entity.get("sub", ""),
            paper_type=entity.get("type", ""),
            spotlight=bool(entity.get("spot", False)),
            arxiv_id=entity.get("arxiv", ""),
        )

    @property
    def openreview_url(self) -> str:
        return OPENREVIEW_FORUM.format(self.paper_id)

    @property
    def arxiv_url(self) -> str:
        return ARXIV_ABS.format(self.arxiv_id) if self.arxiv_id else ""


def _load_authors(raw: Any) -> list[str]:
    """authors 在 Milvus 中以 JSON 字符串存储，容错解析。"""
    if isinstance(raw, list):
        return [str(a) for a in raw]
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return [str(raw)]
    return [str(a) for a in parsed] if isinstance(parsed, list) else [str(parsed)]


@dataclass(frozen=True, slots=True)
class SearchHit:
    """一条检索命中：论文 + 混合排序得分。"""

    paper: Paper
    score: float

    @property
    def paper_id(self) -> str:
        return self.paper.paper_id


@dataclass(frozen=True, slots=True)
class SearchFilters:
    """检索过滤条件，转换为 Milvus 布尔表达式。"""

    area: str | None = None
    paper_type: str | None = None
    spotlight_only: bool = False

    def to_expr(self) -> str:
        """拼 Milvus filter 表达式，无条件时返回空串。"""
        clauses: list[str] = []
        if self.area:
            clauses.append(f'area == "{_escape(self.area)}"')
        if self.paper_type:
            clauses.append(f'type == "{_escape(self.paper_type)}"')
        if self.spotlight_only:
            clauses.append("spot == true")
        return " and ".join(clauses)

    def describe(self) -> str:
        """人类可读的过滤条件摘要。"""
        parts = [
            f"area={self.area}" if self.area else "",
            f"type={self.paper_type}" if self.paper_type else "",
            "spotlight-only" if self.spotlight_only else "",
        ]
        return ", ".join(p for p in parts if p)

    @property
    def is_empty(self) -> bool:
        return not self.to_expr()


def _escape(value: str) -> str:
    """转义 Milvus 字符串字面量中的引号与反斜杠。"""
    return value.replace("\\", "\\\\").replace('"', '\\"')


@dataclass(slots=True)
class SearchOutcome:
    """一次会话内检索的完整结果，含去重统计。"""

    query: str
    hits: list[SearchHit]  # 本轮新增（未被去重）的命中
    search_depth: int  # 本轮的相关性窗口深度
    candidates_examined: int  # 窗口内实际取回的候选数
    duplicates_suppressed: int  # 因此前已检索过而被隐藏的条目数
    filters: SearchFilters = field(default_factory=SearchFilters)
    min_score: float | None = None
    session_id: str | None = None
    consecutive_empty_rounds: int = 0
    session_unique_total: int = 0
    exhausted: bool = False
    truncated: bool = False  # 窗口已填满且新条目不足，调大 search_depth 还能挖到

    @property
    def is_empty(self) -> bool:
        return not self.hits


@dataclass(slots=True)
class QueryRecord:
    """会话内一次 query 的审计记录。"""

    query: str
    filters_desc: str
    new_hits: int
    duplicates_suppressed: int
    at: datetime


@dataclass(slots=True)
class SessionSummary:
    """会话的对外状态快照。"""

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
    papers: list[tuple[str, str]]  # (paper_id, title)，按首次检索到的顺序
