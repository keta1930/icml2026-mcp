"""Behavioral validation of session deduplication, a core project invariant."""

from __future__ import annotations

import pytest

from icml_paper_search.errors import InvalidRequestError, SessionCapacityReached
from icml_paper_search.models import SearchFilters
from icml_paper_search.sessions import SessionNotFoundError


def test_repeated_query_returns_no_duplicates(service):
    """Return only unseen papers from the second run of an identical query onward."""
    session_id = service.open_session(label="dedup").session_id

    first = service.search("diffusion", session_id=session_id)
    second = service.search("diffusion", session_id=session_id)

    first_ids = {h.paper_id for h in first.hits}
    second_ids = {h.paper_id for h in second.hits}
    assert first_ids
    assert not first_ids & second_ids
    assert second.duplicates_suppressed >= len(first_ids)


def test_identical_query_eventually_returns_empty(service):
    """Repeated identical queries eventually empty the deduplicated candidate pool."""
    session_id = service.open_session().session_id
    outcomes = [service.search("diffusion", session_id=session_id) for _ in range(12)]

    assert outcomes[-1].is_empty
    assert outcomes[-1].duplicates_suppressed > 0


def test_consecutive_empty_rounds_mark_exhausted(service, settings):
    """Mark a direction exhausted at the consecutive-empty-round threshold."""
    session_id = service.open_session().session_id
    while not service.search("diffusion", session_id=session_id).is_empty:
        pass

    for _ in range(settings.exhausted_after_empty_rounds):
        outcome = service.search("diffusion", session_id=session_id)

    assert outcome.is_empty
    assert outcome.exhausted
    assert outcome.consecutive_empty_rounds >= settings.exhausted_after_empty_rounds


def test_new_hits_reset_empty_round_counter(service):
    """Reset the empty-round count when new papers appear."""
    session_id = service.open_session().session_id
    # A depth-one window must be empty on the second round.
    service.search("a", session_id=session_id, top_k=1, search_depth=1)
    empty = service.search("a", session_id=session_id, top_k=1, search_depth=1)
    assert empty.is_empty
    assert empty.consecutive_empty_rounds == 1

    fresh = service.search("a", session_id=session_id, top_k=5, search_depth=10)
    assert fresh.hits
    assert fresh.consecutive_empty_rounds == 0


def test_window_is_walked_page_by_page(service):
    """Page through a fixed depth-10 window in two top-k-5 rounds."""
    session_id = service.open_session().session_id

    first = service.search("q", session_id=session_id, top_k=5, search_depth=10)
    second = service.search("q", session_id=session_id, top_k=5, search_depth=10)
    third = service.search("q", session_id=session_id, top_k=5, search_depth=10)

    assert len(first.hits) == 5
    assert len(second.hits) == 5  # Remaining unseen items still fill the page.
    assert third.is_empty  # An exhausted window signals zero new results.
    assert third.duplicates_suppressed == 10


def test_window_does_not_dig_past_requested_depth(service, engine):
    """Use one fixed-limit retrieval without expanding into weaker results."""
    session_id = service.open_session().session_id
    service.search("q", session_id=session_id, top_k=5, search_depth=10)
    engine.search_calls.clear()

    outcome = service.search("q", session_id=session_id, top_k=5, search_depth=10)

    assert engine.search_calls == [10]
    assert outcome.candidates_examined == 10
    assert outcome.search_depth == 10


def test_depth_defaults_to_multiple_of_top_k(service, settings, engine):
    """Use top_k times depth_multiplier when search_depth is omitted."""
    service.search("q", top_k=4)
    assert engine.search_calls[-1] == 4 * settings.depth_multiplier


def test_depth_is_clamped_to_max_and_at_least_top_k(service, settings, engine):
    """Clamp window depth to the limit while keeping it at least top_k."""
    service.search("q", top_k=5, search_depth=settings.max_search_depth * 10)
    assert engine.search_calls[-1] == settings.max_search_depth

    service.search("q", top_k=8, search_depth=2)
    assert engine.search_calls[-1] == 8


def test_truncated_flags_more_results_beyond_window(service):
    """Mark a full window truncated when too few unseen items remain."""
    session_id = service.open_session().session_id
    service.search("q", session_id=session_id, top_k=5, search_depth=5)
    outcome = service.search("q", session_id=session_id, top_k=5, search_depth=5)

    assert outcome.is_empty
    assert outcome.truncated


def test_not_truncated_when_corpus_is_smaller_than_window(service, papers):
    """Do not claim deeper results when the corpus itself is smaller than the window."""
    outcome = service.search("q", top_k=5, search_depth=len(papers) + 10)
    assert not outcome.truncated


def test_sessions_are_isolated(service):
    """Keep accounting isolated across concurrent sessions."""
    a = service.open_session(label="a").session_id
    b = service.open_session(label="b").session_id

    first = service.search("q", session_id=a, top_k=5)
    other = service.search("q", session_id=b, top_k=5)

    assert [h.paper_id for h in first.hits] == [h.paper_id for h in other.hits]


def test_stateless_search_does_not_dedupe(service):
    """Keep searches stateless and identical when no session_id is provided."""
    first = service.search("q", top_k=5)
    second = service.search("q", top_k=5)

    assert [h.paper_id for h in first.hits] == [h.paper_id for h in second.hits]
    assert first.duplicates_suppressed == 0


def test_get_papers_marks_seen(service):
    """Add explicitly fetched papers to the deduplication set."""
    session_id = service.open_session().session_id
    papers = service.get_papers(["p000", "p001"], session_id=session_id)
    assert [p.paper_id for p in papers] == ["p000", "p001"]

    outcome = service.search("q", session_id=session_id, top_k=5)
    assert not {"p000", "p001"} & {h.paper_id for h in outcome.hits}


def test_get_papers_skips_unknown_ids(service):
    """Skip unknown IDs without affecting other fetched papers."""
    papers = service.get_papers(["p000", "nope"])
    assert [p.paper_id for p in papers] == ["p000"]


def test_close_session_returns_full_summary(service):
    """Return query history and papers on close, then invalidate the capability."""
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
    """Apply spotlight-only filtering to an entirely non-spotlight corpus."""
    outcome = service.search("q", filters=SearchFilters(spotlight_only=True))
    assert outcome.candidates_examined == 0
    assert outcome.is_empty


def test_top_k_is_clamped(service, settings):
    """Clamp top_k to max_top_k to bound per-round context."""
    outcome = service.search("q", top_k=settings.max_top_k + 100)
    assert len(outcome.hits) <= settings.max_top_k


def test_empty_query_rejected(service):
    with pytest.raises(InvalidRequestError):
        service.search("   ")


def test_session_capacity_rejects_without_cross_user_eviction(service, settings):
    """Reject capacity overflow without evicting another caller's active capability."""
    ids = [service.open_session(label=str(n)).session_id for n in range(settings.max_sessions)]
    with pytest.raises(SessionCapacityReached):
        service.open_session(label="overflow")
    assert service.session_status(ids[0]).session_id == ids[0]
