"""会话去重机制的行为验证——这是本项目的核心不变式。"""

from __future__ import annotations

import pytest

from icml_paper_search.errors import InvalidRequestError, SessionCapacityReached
from icml_paper_search.models import SearchFilters
from icml_paper_search.sessions import SessionNotFoundError


def test_repeated_query_returns_no_duplicates(service):
    """同一 query 连续跑：第二轮起只给新论文，不重复已返回的条目。"""
    session_id = service.open_session(label="dedup").session_id

    first = service.search("diffusion", session_id=session_id)
    second = service.search("diffusion", session_id=session_id)

    first_ids = {h.paper_id for h in first.hits}
    second_ids = {h.paper_id for h in second.hits}
    assert first_ids
    assert not first_ids & second_ids
    assert second.duplicates_suppressed >= len(first_ids)


def test_identical_query_eventually_returns_empty(service):
    """反复跑同一 query 最终会返回空——候选池被全部去重掉。"""
    session_id = service.open_session().session_id
    outcomes = [service.search("diffusion", session_id=session_id) for _ in range(12)]

    assert outcomes[-1].is_empty
    assert outcomes[-1].duplicates_suppressed > 0


def test_consecutive_empty_rounds_mark_exhausted(service, settings):
    """连续空轮累计到阈值即判定该方向检索完毕。"""
    session_id = service.open_session().session_id
    while not service.search("diffusion", session_id=session_id).is_empty:
        pass

    for _ in range(settings.exhausted_after_empty_rounds):
        outcome = service.search("diffusion", session_id=session_id)

    assert outcome.is_empty
    assert outcome.exhausted
    assert outcome.consecutive_empty_rounds >= settings.exhausted_after_empty_rounds


def test_new_hits_reset_empty_round_counter(service):
    """出现新论文时空轮计数归零，避免误判为已挖尽。"""
    session_id = service.open_session().session_id
    # 深度 1 的窗口：第二轮必空
    service.search("a", session_id=session_id, top_k=1, search_depth=1)
    empty = service.search("a", session_id=session_id, top_k=1, search_depth=1)
    assert empty.is_empty
    assert empty.consecutive_empty_rounds == 1

    fresh = service.search("a", session_id=session_id, top_k=5, search_depth=10)
    assert fresh.hits
    assert fresh.consecutive_empty_rounds == 0


def test_window_is_walked_page_by_page(service):
    """固定窗口按页推进：depth=10/top_k=5 时两轮走完窗口，第三轮归零。"""
    session_id = service.open_session().session_id

    first = service.search("q", session_id=session_id, top_k=5, search_depth=10)
    second = service.search("q", session_id=session_id, top_k=5, search_depth=10)
    third = service.search("q", session_id=session_id, top_k=5, search_depth=10)

    assert len(first.hits) == 5
    assert len(second.hits) == 5  # 窗口内还有新条目，仍给满页
    assert third.is_empty  # 窗口耗尽 → 零新增信号
    assert third.duplicates_suppressed == 10


def test_window_does_not_dig_past_requested_depth(service, engine):
    """窗口不做自动扩召回：只发一次固定 limit 的检索，不下探到弱相关区。"""
    session_id = service.open_session().session_id
    service.search("q", session_id=session_id, top_k=5, search_depth=10)
    engine.search_calls.clear()

    outcome = service.search("q", session_id=session_id, top_k=5, search_depth=10)

    assert engine.search_calls == [10]
    assert outcome.candidates_examined == 10
    assert outcome.search_depth == 10


def test_depth_defaults_to_multiple_of_top_k(service, settings, engine):
    """未指定 search_depth 时窗口 = top_k × depth_multiplier。"""
    service.search("q", top_k=4)
    assert engine.search_calls[-1] == 4 * settings.depth_multiplier


def test_depth_is_clamped_to_max_and_at_least_top_k(service, settings, engine):
    """窗口深度受上限约束，且不小于 top_k。"""
    service.search("q", top_k=5, search_depth=settings.max_search_depth * 10)
    assert engine.search_calls[-1] == settings.max_search_depth

    service.search("q", top_k=8, search_depth=2)
    assert engine.search_calls[-1] == 8


def test_truncated_flags_more_results_beyond_window(service):
    """窗口填满但新条目不足时打上 truncated，提示还能往深处挖。"""
    session_id = service.open_session().session_id
    service.search("q", session_id=session_id, top_k=5, search_depth=5)
    outcome = service.search("q", session_id=session_id, top_k=5, search_depth=5)

    assert outcome.is_empty
    assert outcome.truncated


def test_not_truncated_when_corpus_is_smaller_than_window(service, papers):
    """语料本身不足窗口深度时不该谎报"还有更深的结果"。"""
    outcome = service.search("q", top_k=5, search_depth=len(papers) + 10)
    assert not outcome.truncated


def test_sessions_are_isolated(service):
    """并行会话各自独立记账，互不影响。"""
    a = service.open_session(label="a").session_id
    b = service.open_session(label="b").session_id

    first = service.search("q", session_id=a, top_k=5)
    other = service.search("q", session_id=b, top_k=5)

    assert [h.paper_id for h in first.hits] == [h.paper_id for h in other.hits]


def test_stateless_search_does_not_dedupe(service):
    """不传 session_id 时不去重，两次结果一致。"""
    first = service.search("q", top_k=5)
    second = service.search("q", top_k=5)

    assert [h.paper_id for h in first.hits] == [h.paper_id for h in second.hits]
    assert first.duplicates_suppressed == 0


def test_get_papers_marks_seen(service):
    """显式取回的论文也进去重集，后续检索不再重复吐出。"""
    session_id = service.open_session().session_id
    papers = service.get_papers(["p000", "p001"], session_id=session_id)
    assert [p.paper_id for p in papers] == ["p000", "p001"]

    outcome = service.search("q", session_id=session_id, top_k=5)
    assert not {"p000", "p001"} & {h.paper_id for h in outcome.hits}


def test_get_papers_skips_unknown_ids(service):
    """不存在的 id 被静默跳过，不影响其余取回。"""
    papers = service.get_papers(["p000", "nope"])
    assert [p.paper_id for p in papers] == ["p000"]


def test_close_session_returns_full_summary(service):
    """关闭会话给出 query 轨迹与完整论文清单，之后 id 失效。"""
    session_id = service.open_session(label="wrap").session_id
    outcome = service.search("q", session_id=session_id, top_k=5)

    summary = service.close_session(session_id)
    assert summary.query_count == 1
    assert summary.unique_papers == len(outcome.hits)
    assert len(summary.papers) == len(outcome.hits)
    assert len(summary.queries[0].query_hash) == 64
    assert summary.queries[0].query_summary is None

    with pytest.raises(SessionNotFoundError):
        service.search("q", session_id=session_id)


def test_unknown_session_raises(service):
    with pytest.raises(SessionNotFoundError):
        service.search("q", session_id="ses_missing")


def test_filters_narrow_the_candidate_pool(service, papers):
    """过滤条件生效：spotlight-only 在全 non-spotlight 语料上返回空。"""
    outcome = service.search("q", filters=SearchFilters(spotlight_only=True))
    assert outcome.candidates_examined == 0
    assert outcome.is_empty


def test_top_k_is_clamped(service, settings):
    """top_k 被夹到 max_top_k，防止单轮吐爆 context。"""
    outcome = service.search("q", top_k=settings.max_top_k + 100)
    assert len(outcome.hits) <= settings.max_top_k


def test_empty_query_rejected(service):
    with pytest.raises(InvalidRequestError):
        service.search("   ")


def test_session_capacity_rejects_without_cross_user_eviction(service, settings):
    """会话数超上限时稳定拒绝，不回收其他调用方的活跃 capability。"""
    ids = [service.open_session(label=str(n)).session_id for n in range(settings.max_sessions)]
    with pytest.raises(SessionCapacityReached):
        service.open_session(label="overflow")
    assert service.session_status(ids[0]).session_id == ids[0]
