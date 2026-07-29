"""Milvus native dense/BM25 retrieval and repository boundary tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from icml_paper_search.config import Settings
from icml_paper_search.domain.models import SearchFilters
from icml_paper_search.indexing.manifest import IndexManifest
from icml_paper_search.milvus.repository import (
    MilvusPaperRepository,
    _filter_expression,
    _normalize_rrf_score,
)


def _manifest(settings: Settings) -> IndexManifest:
    return IndexManifest(
        format="milvus-dense-bm25-rrf",
        build_id="a" * 16,
        collection_name="icml2026_mcp_" + "a" * 16,
        corpus_hash="b" * 64,
        paper_ids_hash="c" * 64,
        record_count=2,
        data_file_hashes={"index.json": "d" * 64, "abstracts.json": "e" * 64},
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
        milvus_version="2.5.27",
        pymilvus_version="2.5.18",
        dense_metric="COSINE",
        sparse_metric="BM25",
        fusion="RRF",
        built_at=datetime.now(UTC).isoformat(),
    )


def _entity(paper_id: str, title: str) -> dict[str, object]:
    return {
        "paper_id": paper_id,
        "title": title,
        "abstract": f"Abstract for {title}",
        "authors_json": '["Alice", "Bob"]',
        "area": "Theory",
        "subarea": "Learning Theory",
        "paper_type": "Poster",
        "spotlight": False,
        "arxiv_id": "2605.00001",
    }


class FakeEncoder:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.queries: list[str] = []

    def encode_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.25] * self.dimension

    def encode_documents(self, texts, *, show_progress=False):
        return [[0.25] * self.dimension for _ in texts]

    def close(self) -> None:
        return None


class FakeClient:
    def __init__(self) -> None:
        self.hybrid_calls = []
        self.get_calls = []
        self.closed = False

    def hybrid_search(self, collection_name, reqs, ranker, limit, **kwargs):
        self.hybrid_calls.append((collection_name, reqs, ranker, limit, kwargs))
        return [
            [
                {"distance": 2 / 61, "entity": _entity("p1", "First")},
                {"distance": 1 / 61 + 1 / 62, "entity": _entity("p2", "Second")},
            ]
        ][:limit]

    def get(self, collection, ids, **kwargs):
        self.get_calls.append((collection, ids, kwargs))
        rows = {"p1": _entity("p1", "First"), "p2": _entity("p2", "Second")}
        return [rows[paper_id] for paper_id in reversed(ids) if paper_id in rows]

    def close(self) -> None:
        self.closed = True


def _repository(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        index_root=tmp_path / "index",
        state_db_path=tmp_path / "state.sqlite3",
    )
    encoder = FakeEncoder(settings.embedding_dim)
    client = FakeClient()
    repository = MilvusPaperRepository(settings, encoder, client_factory=lambda: client)
    repository._manifest = _manifest(settings)
    return repository, encoder, client


def test_search_uses_dense_and_bm25_with_unweighted_rrf(tmp_path):
    repository, encoder, client = _repository(tmp_path)
    filters = SearchFilters(
        area='Deep "Learning"', paper_type="Poster", spotlight_only=True
    )

    hits = repository.search("robust learning", limit=2, filters=filters)

    assert encoder.queries == ["robust learning"]
    collection, requests, ranker, limit, kwargs = client.hybrid_calls[0]
    assert collection == repository.manifest.collection_name
    assert [request._anns_field for request in requests] == [
        "dense_vector",
        "sparse_vector",
    ]
    assert requests[0]._data == [[0.25] * repository._settings.embedding_dim]
    assert requests[1]._data == ["robust learning"]
    assert requests[0]._expr == requests[1]._expr
    assert requests[0]._expr == (
        'area == "Deep \\"Learning\\"" and paper_type == "Poster" and spotlight == true'
    )
    assert ranker.dict() == {"strategy": "rrf", "params": {"k": 60}}
    assert limit == 2
    assert kwargs["output_fields"][0] == "paper_id"
    assert [hit.paper_id for hit in hits] == ["p1", "p2"]
    assert hits[0].score == 1.0
    assert 0.98 < hits[1].score < 1.0


def test_fetch_preserves_first_requested_order_and_removes_duplicates(tmp_path):
    repository, _, client = _repository(tmp_path)
    papers = repository.fetch(["p2", "missing", "p1", "p2"])
    assert [paper.paper_id for paper in papers] == ["p2", "p1"]
    assert client.get_calls[0][1] == ["p2", "missing", "p1"]


def test_filter_expression_uses_exact_current_fields():
    assert _filter_expression(SearchFilters()) == ""
    assert _filter_expression(SearchFilters(area="Theory")) == 'area == "Theory"'
    assert _filter_expression(SearchFilters(spotlight_only=True)) == "spotlight == true"


def test_rrf_score_normalization_uses_the_actual_search_path_count():
    assert _normalize_rrf_score(2 / 61, path_count=2) == 1.0
    assert _normalize_rrf_score(3 / 61, path_count=3) == 1.0
    assert _normalize_rrf_score(1 / 61, path_count=2) == 0.5
    assert _normalize_rrf_score(-1.0, path_count=2) == 0.0
    with pytest.raises(ValueError, match="path_count must be positive"):
        _normalize_rrf_score(0.0, path_count=0)


def test_close_closes_encoder_and_thread_clients(tmp_path):
    repository, _, client = _repository(tmp_path)
    repository.fetch(["p1"])
    repository.close()
    assert client.closed
