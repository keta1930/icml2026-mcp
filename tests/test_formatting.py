"""Validate compact agent-facing result, empty-state, and session rendering."""

from __future__ import annotations

from icml_paper_search.config import Settings
from icml_paper_search.domain.models import (
    Paper,
    SearchFilters,
    SearchHit,
    SearchOutcome,
)
from icml_paper_search.search.formatting import (
    format_papers,
    format_search_outcome,
    format_session_summary,
)


def test_outcome_header_reports_dedup_stats(service, settings):
    session_id = service.open_session().session_id
    service.search("q", session_id=session_id, top_k=5, search_depth=20)
    outcome = service.search("q", session_id=session_id, top_k=5, search_depth=20)

    text = format_search_outcome(outcome, settings)
    assert "new=5" in text
    assert "duplicates_suppressed=5" in text
    assert "window=20/20" in text
    assert "session_total=" in text


def test_empty_outcome_explains_exhaustion(service, settings):
    session_id = service.open_session().session_id
    while not service.search("q", session_id=session_id).is_empty:
        pass
    for _ in range(settings.exhausted_after_empty_rounds):
        outcome = service.search("q", session_id=session_id)

    text = format_search_outcome(outcome, settings)
    assert "this direction is exhausted" in text


def test_empty_outcome_without_candidates_flags_filters(service, settings):
    outcome = service.search("q", filters=SearchFilters(spotlight_only=True))
    text = format_search_outcome(outcome, settings)
    assert "filters may be too narrow" in text
    assert "no matching results in the corpus" not in text


def test_empty_outcome_blames_min_score_not_the_corpus(service, settings):
    """Attribute an empty result to min_score rather than the corpus."""
    outcome = service.search("q", min_score=1.0)
    text = format_search_outcome(outcome, settings)
    assert "min_score=1.0" in text
    assert "no matching results in the corpus" not in text


def test_deeper_hint_suggests_a_concrete_depth(service, settings):
    """Give a directly usable search_depth after a complete window."""
    session_id = service.open_session().session_id
    service.search("q", session_id=session_id, top_k=5, search_depth=5)
    outcome = service.search("q", session_id=session_id, top_k=5, search_depth=5)

    text = format_search_outcome(outcome, settings)
    assert "search_depth" in text
    assert "10" in text


def test_deeper_hint_at_max_depth_stops_suggesting(service, settings):
    """Recommend a new direction instead of more depth at the global limit."""
    session_id = service.open_session().session_id
    depth = settings.max_search_depth
    while not service.search("q", session_id=session_id, search_depth=depth).is_empty:
        pass
    outcome = service.search("q", session_id=session_id, search_depth=depth)

    text = format_search_outcome(outcome, settings)
    assert "Rephrase the query or change direction" in text


def test_abstract_is_truncated_in_listings(service):
    settings = Settings(abstract_preview_chars=40)
    outcome = service.search("q", top_k=1)
    text = format_search_outcome(outcome, settings)
    assert "…" in text


def test_include_abstract_false_drops_abstract(service, settings):
    outcome = service.search("q", top_k=1)
    text = format_search_outcome(outcome, settings, include_abstract=False)
    assert "Abstract of paper" not in text
    assert outcome.hits[0].paper.title in text


def test_format_papers_keeps_full_abstract_and_lists_missing(service, settings):
    papers = service.get_papers(["p000"])
    text = format_papers(papers, settings, missing_ids=["nope"])
    assert "…" not in text
    assert text.startswith("Retrieved 1 paper(s)")
    assert "IDs not found: nope" in text


def test_session_summary_includes_query_trace(service):
    session_id = service.open_session(label="研究主题").session_id
    service.search("alpha", session_id=session_id, top_k=2)
    summary = service.close_session(session_id)

    text = format_session_summary(summary, title="Search session closed")
    assert "研究主题" in text
    assert "hash:" in text
    assert "alpha" not in text
    assert "Retrieved papers" in text


def test_mixed_language_business_values_are_preserved_verbatim(settings):
    paper = Paper(
        paper_id="mixed",
        title="中文论文标题",
        abstract="中文摘要内容",
        authors=("测试作者",),
        area="机器学习领域",
        subarea="子领域",
        paper_type="海报",
        spotlight=False,
        arxiv_id="",
    )
    outcome = SearchOutcome(
        query="中文查询",
        hits=[SearchHit(paper=paper, score=0.5)],
        search_depth=5,
        candidates_examined=1,
        duplicates_suppressed=0,
        filters=SearchFilters(area="机器学习领域"),
    )

    text = format_search_outcome(outcome, settings)

    for value in (
        "中文查询",
        "中文论文标题",
        "中文摘要内容",
        "测试作者",
        "机器学习领域",
        "子领域",
        "海报",
    ):
        assert value in text
    assert "new=1" in text
    assert "stateless (deduplication disabled)" in text
