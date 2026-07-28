"""本地语料元数据：入库记录组装与 area/type 分布统计。"""

from __future__ import annotations

import json

import pytest

from icml_paper_search.catalog import get_catalog, load_records
from icml_paper_search.config import Settings

INDEX = {
    "papers": [
        {
            "i": 1,
            "pid": "1",
            "orid": "a1",
            "title": "First",
            "authors": ["Alice"],
            "area": "Theory",
            "sub": "Learning Theory",
            "type": "Poster",
            "spot": True,
            "arxiv": "2605.1",
        },
        {
            "i": 2,
            "pid": "2",
            "orid": "a2",
            "title": "Second",
            "authors": [],
            "area": "Theory",
            "type": "Oral",
        },
        {"i": 3, "pid": "3", "orid": "a3", "title": "Third", "authors": ["Bob"]},
    ]
}
ABSTRACTS = {"a1": "Abstract one.", "a2": "Abstract two."}


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / "index.json").write_text(json.dumps(INDEX), encoding="utf-8")
    (tmp_path / "abstracts.json").write_text(json.dumps(ABSTRACTS), encoding="utf-8")
    return tmp_path


def test_load_records_joins_abstracts(data_dir):
    records = load_records(Settings(data_dir=data_dir))

    assert [r.paper_id for r in records] == ["a1", "a2", "a3"]
    assert records[0].text == "First\nAbstract one."
    assert records[0].fields["abstract"] == "Abstract one."
    # 缺摘要的论文仍入库，text 退化为 title
    assert records[2].fields["abstract"] == ""
    assert records[2].text == "Third"


def test_load_records_serializes_authors_and_defaults(data_dir):
    records = load_records(Settings(data_dir=data_dir))

    assert json.loads(records[0].fields["authors"]) == ["Alice"]
    assert records[0].fields["spot"] is True
    # 缺失的可选字段补空值，不是 None
    assert records[2].fields["area"] == ""
    assert records[2].fields["spot"] is False


def test_catalog_counts_areas_and_spotlights(data_dir):
    catalog = get_catalog(Settings(data_dir=data_dir))

    assert catalog.total == 3
    assert catalog.spotlights == 1
    assert dict(catalog.areas) == {"Theory": 2, "Uncategorized": 1}
    assert dict(catalog.paper_types) == {"Poster": 1, "Oral": 1, "Unknown": 1}


def test_catalog_accepts_settings_instance(data_dir):
    """Settings 不可 hash，缓存必须按路径做键，否则这里会 TypeError。"""
    first = get_catalog(Settings(data_dir=data_dir))
    second = get_catalog(Settings(data_dir=data_dir))
    assert first is second


def test_catalog_render_lists_filter_values(data_dir):
    text = get_catalog(Settings(data_dir=data_dir)).render()
    assert "Theory  (2)" in text
    assert "paper_type 可选值" in text


def test_missing_data_file_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="index.json"):
        load_records(Settings(data_dir=tmp_path))
