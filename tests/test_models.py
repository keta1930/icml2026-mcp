"""Validate backend-neutral domain models."""

from __future__ import annotations

import pytest

from icml_paper_search.domain.models import Paper, SearchFilters, SearchHit


def test_paper_from_mapping_accepts_current_milvus_entity():
    paper = Paper.from_mapping(
        {
            "paper_id": "abc123",
            "title": "T",
            "abstract": "A",
            "authors_json": '["Alice", "Bob"]',
            "area": "Theory",
            "subarea": "Learning Theory",
            "paper_type": "Oral",
            "spotlight": True,
            "arxiv_id": "2605.00001",
        }
    )

    assert paper.paper_id == "abc123"
    assert paper.authors == ("Alice", "Bob")
    assert paper.subarea == "Learning Theory"
    assert paper.paper_type == "Oral"
    assert paper.spotlight is True
    assert paper.openreview_url.endswith("abc123")
    assert paper.arxiv_url == "https://arxiv.org/abs/2605.00001"


def test_paper_from_mapping_rejects_invalid_author_json_as_empty():
    assert Paper.from_mapping({"authors_json": "not json"}).authors == ()
    assert Paper.from_mapping({"authors_json": "null"}).authors == ()
    assert Paper.from_mapping({"authors_json": '["X"]'}).authors == ("X",)
    assert Paper.from_mapping({"paper_id": "x"}).arxiv_url == ""


def test_filters_only_describe_domain_values():
    filters = SearchFilters(
        area='Deep "Learning"', paper_type="Oral", spotlight_only=True
    )

    assert filters.describe() == 'area=Deep "Learning", type=Oral, spotlight-only'
    assert not filters.is_empty
    assert SearchFilters().is_empty
    assert SearchFilters().describe() == ""
    assert not hasattr(filters, "to_expr")


def test_search_hit_score_is_normalized():
    paper = Paper.from_mapping({"paper_id": "p1"})

    assert SearchHit(paper=paper, score=0.0).score == 0.0
    assert SearchHit(paper=paper, score=1.0).score == 1.0
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        SearchHit(paper=paper, score=1.01)
