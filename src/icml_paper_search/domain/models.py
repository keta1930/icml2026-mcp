"""Paper-search domain models."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

OPENREVIEW_FORUM = "https://openreview.net/forum?id={}"
ARXIV_ABS = "https://arxiv.org/abs/{}"


@dataclass(frozen=True, slots=True)
class Paper:
    """Describe the retrieval-facing view of an ICML 2026 paper."""

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
        """Construct a paper from the current Milvus entity shape."""
        return cls(
            paper_id=str(value.get("paper_id", "") or ""),
            title=str(value.get("title", "") or ""),
            abstract=str(value.get("abstract", "") or ""),
            authors=tuple(_load_authors(value.get("authors_json"))),
            area=str(value.get("area", "") or ""),
            subarea=str(value.get("subarea", "") or ""),
            paper_type=str(value.get("paper_type", "") or ""),
            spotlight=bool(value.get("spotlight", False)),
            arxiv_id=str(value.get("arxiv_id", "") or ""),
        )

    @property
    def openreview_url(self) -> str:
        """Return the paper's OpenReview page."""
        return OPENREVIEW_FORUM.format(self.paper_id)

    @property
    def arxiv_url(self) -> str:
        """Return the abstract page when an arXiv ID is available."""
        return ARXIV_ABS.format(self.arxiv_id) if self.arxiv_id else ""


def _load_authors(raw: Any) -> list[str]:
    """Parse the JSON author list stored in the current Milvus schema."""
    if not isinstance(raw, str) or not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return [str(author) for author in parsed] if isinstance(parsed, list) else []


@dataclass(frozen=True, slots=True)
class SearchHit:
    """Represent a hit with a Milvus RRF score in `[0,1]`."""

    paper: Paper
    score: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("SearchHit.score must be within [0,1]")

    @property
    def paper_id(self) -> str:
        """Return the primary key of the matched paper."""
        return self.paper.paper_id


@dataclass(frozen=True, slots=True)
class SearchFilters:
    """Describe domain filters independent of the retrieval backend."""

    area: str | None = None
    paper_type: str | None = None
    spotlight_only: bool = False

    def describe(self) -> str:
        """Return a user-facing filter summary."""
        parts = [
            f"area={self.area}" if self.area else "",
            f"type={self.paper_type}" if self.paper_type else "",
            "spotlight-only" if self.spotlight_only else "",
        ]
        return ", ".join(part for part in parts if part)

    @property
    def is_empty(self) -> bool:
        """Return whether no filters are set."""
        return not (self.area or self.paper_type or self.spotlight_only)


@dataclass(slots=True)
class SearchOutcome:
    """Describe one search and its session deduplication statistics."""

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
        """Return whether this round produced no new hits."""
        return not self.hits


@dataclass(frozen=True, slots=True)
class QueryRecord:
    """Store a bounded session audit record without the full query."""

    query_hash: str
    filters_desc: str
    new_hits: int
    duplicates_suppressed: int
    candidates_examined: int
    at: datetime


@dataclass(frozen=True, slots=True)
class RoundCommit:
    """Return round state exactly matching the atomic SQLite commit."""

    hits: list[SearchHit]
    duplicates_suppressed: int
    consecutive_empty_rounds: int
    session_unique_total: int
    exhausted: bool
    query_count: int
    version: int


@dataclass(slots=True)
class SessionSummary:
    """Describe a bounded public snapshot of a capability session."""

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
