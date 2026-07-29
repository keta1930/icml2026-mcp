"""Local corpus record assembly and area/type distribution tests."""

from __future__ import annotations

import json

import pytest

from icml_paper_search.config import Settings
from icml_paper_search.corpus.catalog import (
    Catalog,
    corpus_hash,
    data_file_hashes,
    get_catalog,
    load_records,
)

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
    assert records[0].document == "First\nAbstract one."
    assert records[0].abstract == "Abstract one."
    # Papers without abstracts remain indexable using title-only text.
    assert records[2].abstract == ""
    assert records[2].document == "Third"


def test_load_records_serializes_authors_and_defaults(data_dir):
    records = load_records(Settings(data_dir=data_dir))

    entity = records[0].milvus_entity([0.1, 0.2])
    assert json.loads(entity["authors_json"]) == ["Alice"]
    assert entity["spotlight"] is True
    assert entity["dense_vector"] == [0.1, 0.2]
    # Missing optional fields use empty values rather than None.
    assert records[2].area == ""
    assert records[2].spotlight is False


def test_catalog_counts_areas_and_spotlights(data_dir):
    catalog = get_catalog(Settings(data_dir=data_dir))

    assert catalog.total == 3
    assert catalog.spotlights == 1
    assert dict(catalog.areas) == {"Theory": 2, "Uncategorized": 1}
    assert dict(catalog.paper_types) == {"Poster": 1, "Oral": 1, "Unknown": 1}


def test_catalog_accepts_settings_instance(data_dir):
    """Cache by path because Settings is unhashable."""
    first = get_catalog(Settings(data_dir=data_dir))
    second = get_catalog(Settings(data_dir=data_dir))
    assert first is second


def test_catalog_render_lists_filter_values(data_dir):
    text = get_catalog(Settings(data_dir=data_dir)).render()
    assert "Theory (2)" in text
    assert "Available paper_type values:" in text


def test_catalog_render_preserves_taxonomy_values_verbatim():
    text = Catalog(
        total=2,
        areas=(("机器学习", 2),),
        paper_types=(("海报", 2),),
        spotlights=1,
    ).render()

    assert "ICML 2026 corpus: 2 papers, including 1 spotlight papers." in text
    assert "机器学习 (2)" in text
    assert "海报 (2)" in text


def test_missing_data_file_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="index.json"):
        load_records(Settings(data_dir=tmp_path))


def test_corpus_hash_is_stable_across_record_order(data_dir):
    records = load_records(Settings(data_dir=data_dir))
    assert corpus_hash(records) == corpus_hash(list(reversed(records)))


def test_data_file_hashes_cover_both_snapshots(data_dir):
    hashes = data_file_hashes(Settings(data_dir=data_dir))
    assert set(hashes) == {"index.json", "abstracts.json"}
    assert all(len(value) == 64 for value in hashes.values())
