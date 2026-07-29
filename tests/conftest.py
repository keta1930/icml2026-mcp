"""Offline fixtures backed by temporary SQLite state and a fake engine."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from icml_paper_search.config import Settings
from icml_paper_search.domain.models import Paper, SearchFilters, SearchHit
from icml_paper_search.runtime.sessions import SessionManager
from icml_paper_search.search.service import PaperSearchService


def make_paper(
    n: int, *, area: str = "Deep Learning", spotlight: bool = False
) -> Paper:
    return Paper(
        paper_id=f"p{n:03d}",
        title=f"Paper {n}",
        abstract=f"Abstract of paper {n}. " * 20,
        authors=(f"Author {n}A", f"Author {n}B"),
        area=area,
        subarea="",
        paper_type="Poster",
        spotlight=spotlight,
        arxiv_id=f"2605.{n:05d}",
    )


class FakeEngine:
    """Return papers in fixed order with scores decreasing by rank."""

    def __init__(self, papers: list[Paper]) -> None:
        self.papers = papers
        self.search_calls: list[int] = []

    def search(
        self, query: str, *, limit: int, filters: SearchFilters
    ) -> list[SearchHit]:
        self.search_calls.append(limit)
        pool = [p for p in self.papers if _matches(p, filters)]
        return [
            SearchHit(paper=p, score=0.99 - i * 0.001)
            for i, p in enumerate(pool[:limit])
        ]

    def fetch(self, paper_ids: Sequence[str]) -> list[Paper]:
        by_id = {p.paper_id: p for p in self.papers}
        return [by_id[pid] for pid in dict.fromkeys(paper_ids) if pid in by_id]


def _matches(paper: Paper, filters: SearchFilters) -> bool:
    if filters.area and paper.area != filters.area:
        return False
    if filters.paper_type and paper.paper_type != filters.paper_type:
        return False
    if filters.spotlight_only and not paper.spotlight:
        return False
    return True


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        index_root=tmp_path / "index",
        state_db_path=tmp_path / "runtime.sqlite3",
        default_top_k=5,
        max_top_k=20,
        depth_multiplier=2,
        max_search_depth=40,
        exhausted_after_empty_rounds=2,
        max_sessions=3,
        max_session_rounds=100,
        max_session_audit_records=100,
    )


@pytest.fixture
def papers() -> list[Paper]:
    return [make_paper(n) for n in range(60)]


@pytest.fixture
def engine(papers) -> FakeEngine:
    return FakeEngine(papers)


@pytest.fixture
def service(settings, engine) -> PaperSearchService:
    return PaperSearchService(
        settings=settings, engine=engine, sessions=SessionManager(settings)
    )
