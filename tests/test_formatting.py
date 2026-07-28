"""输出渲染：确认关键统计与空结果解释都出现在 agent 可见的文本里。"""

from __future__ import annotations

from icml_paper_search.config import Settings
from icml_paper_search.formatting import (
    format_papers,
    format_search_outcome,
    format_session_summary,
)
from icml_paper_search.models import SearchFilters


def test_outcome_header_reports_dedup_stats(service, settings):
    session_id = service.open_session().session_id
    service.search("q", session_id=session_id, top_k=5, search_depth=20)
    outcome = service.search("q", session_id=session_id, top_k=5, search_depth=20)

    text = format_search_outcome(outcome, settings)
    assert "新增 5 条" in text
    assert "去重隐藏 5 条" in text
    assert "窗口 20/20 条" in text
    assert "会话累计" in text


def test_empty_outcome_explains_exhaustion(service, settings):
    session_id = service.open_session().session_id
    while not service.search("q", session_id=session_id).is_empty:
        pass
    for _ in range(settings.exhausted_after_empty_rounds):
        outcome = service.search("q", session_id=session_id)

    text = format_search_outcome(outcome, settings)
    assert "检索完毕" in text


def test_empty_outcome_without_candidates_flags_filters(service, settings):
    outcome = service.search("q", filters=SearchFilters(spotlight_only=True))
    text = format_search_outcome(outcome, settings)
    assert "过滤条件" in text
    assert "没有匹配结果" not in text  # 是被过滤掉的，不是语料里没有


def test_empty_outcome_blames_min_score_not_the_corpus(service, settings):
    """被 min_score 筛空时要说清是分数门槛，别谎报语料无此主题。"""
    outcome = service.search("q", min_score=1.0)
    text = format_search_outcome(outcome, settings)
    assert "min_score=1.0" in text
    assert "没有匹配结果" not in text


def test_deeper_hint_suggests_a_concrete_depth(service, settings):
    """窗口看完时给出可直接照用的 search_depth 建议值。"""
    session_id = service.open_session().session_id
    service.search("q", session_id=session_id, top_k=5, search_depth=5)
    outcome = service.search("q", session_id=session_id, top_k=5, search_depth=5)

    text = format_search_outcome(outcome, settings)
    assert "search_depth" in text
    assert "10" in text


def test_deeper_hint_at_max_depth_stops_suggesting(service, settings):
    """已到深度上限时不再劝深挖，而是让 agent 换方向。"""
    session_id = service.open_session().session_id
    depth = settings.max_search_depth
    while not service.search("q", session_id=session_id, search_depth=depth).is_empty:
        pass
    outcome = service.search("q", session_id=session_id, search_depth=depth)

    text = format_search_outcome(outcome, settings)
    assert "换个措辞" in text or "换研究主题" in text


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
    assert "nope" in text


def test_session_summary_includes_query_trace(service):
    session_id = service.open_session(label="topic").session_id
    service.search("alpha", session_id=session_id, top_k=2)
    summary = service.close_session(session_id)

    text = format_session_summary(summary, title="已关闭检索会话")
    assert "topic" in text
    assert "hash:" in text
    assert "alpha" not in text
    assert "已检索论文" in text
