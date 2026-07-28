"""领域模型：Milvus entity 反序列化与 filter 表达式拼装。"""

from __future__ import annotations

from icml_paper_search.models import Paper, SearchFilters


def test_paper_from_entity_parses_json_authors():
    paper = Paper.from_entity(
        {
            "orid": "abc123",
            "title": "T",
            "abstract": "A",
            "authors": '["Alice", "Bob"]',
            "area": "Theory",
            "sub": "",
            "type": "Oral",
            "spot": True,
            "arxiv": "2605.00001",
        }
    )
    assert paper.authors == ("Alice", "Bob")
    assert paper.spotlight is True
    assert paper.openreview_url.endswith("abc123")
    assert paper.arxiv_url == "https://arxiv.org/abs/2605.00001"


def test_paper_from_entity_tolerates_bad_authors():
    """authors 字段脏了也不该让整条结果炸掉。"""
    assert Paper.from_entity({"authors": "not json"}).authors == ("not json",)
    assert Paper.from_entity({"authors": None}).authors == ()
    assert Paper.from_entity({"authors": ["X"]}).authors == ("X",)


def test_paper_without_arxiv_has_no_arxiv_url():
    assert Paper.from_entity({"orid": "x"}).arxiv_url == ""


def test_filters_build_conjunctive_expr():
    filters = SearchFilters(area="Deep Learning", paper_type="Oral", spotlight_only=True)
    assert filters.to_expr() == (
        'area == "Deep Learning" and type == "Oral" and spot == true'
    )
    assert not filters.is_empty


def test_empty_filters_produce_no_expr():
    filters = SearchFilters()
    assert filters.to_expr() == ""
    assert filters.is_empty
    assert filters.describe() == ""


def test_filter_values_are_escaped():
    """带引号的取值不能逃出字符串字面量。"""
    expr = SearchFilters(area='Deep "Learning"').to_expr()
    assert expr == 'area == "Deep \\"Learning\\""'
