"""Chroma + FTS5 持久化、过滤、RRF 与 fixed-window 测试。"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from icml_paper_search.bm25 import BM25IndexBuilder, RankedPaperId
from icml_paper_search.catalog import (
    PaperRecord,
    corpus_hash,
    data_file_hashes,
    paper_ids_hash,
)
from icml_paper_search.config import Settings
from icml_paper_search.errors import IndexUnavailable
from icml_paper_search.manifest import (
    IndexManifest,
    IndexPaths,
    collection_metadata_for,
    collection_name_for,
    promote_staging,
    write_manifest,
)
from icml_paper_search.models import SearchFilters
from icml_paper_search.retrieval import ChromaHybridSearchEngine, _weighted_rrf

DIMENSION = 1024


class FakeEncoder:
    """返回固定 query 向量，保证测试不访问网络。"""

    def __init__(self, vector):
        self.vector = vector
        self.calls = 0

    def encode_query(self, _text):
        self.calls += 1
        return list(self.vector)

    def encode_documents(self, texts, *, show_progress=False):
        del show_progress
        return [list(self.vector) for _text in texts]


def make_record(
    paper_id: str,
    title: str,
    abstract: str,
    *,
    area: str,
    paper_type: str = "Poster",
    spot: bool = False,
) -> PaperRecord:
    text = f"{title}\n{abstract}"
    return PaperRecord(
        fields={
            "orid": paper_id,
            "i": 1,
            "title": title,
            "abstract": abstract,
            "text": text,
            "authors": json.dumps(["Alice"]),
            "area": area,
            "sub": "",
            "type": paper_type,
            "spot": spot,
            "pid": "1",
            "arxiv": "",
        },
        text=text,
    )


def write_source_files(path, records):
    """写出与测试索引对应的原始语料快照。"""
    papers = []
    abstracts = {}
    for record in records:
        fields = record.fields
        papers.append(
            {
                "i": fields["i"],
                "pid": fields["pid"],
                "orid": record.paper_id,
                "title": fields["title"],
                "authors": json.loads(fields["authors"]),
                "area": fields["area"],
                "sub": fields["sub"],
                "type": fields["type"],
                "spot": fields["spot"],
                "arxiv": fields["arxiv"],
            }
        )
        abstracts[record.paper_id] = fields["abstract"]
    (path / "index.json").write_text(
        json.dumps({"papers": papers}), encoding="utf-8"
    )
    (path / "abstracts.json").write_text(json.dumps(abstracts), encoding="utf-8")


@pytest.fixture
def local_index(tmp_path):
    import chromadb

    settings = Settings(index_root=tmp_path / "index", data_dir=tmp_path)
    records = [
        make_record("p1", "Vision transformer", "attention image model", area="Vision"),
        make_record(
            "p2", "Graph theory", "spectral graph learning", area="Theory", spot=True
        ),
        make_record("p3", "Language agents", "tool using language model", area="NLP"),
    ]
    vectors = []
    for index in range(len(records)):
        vector = [0.0] * DIMENSION
        vector[index] = 1.0
        vectors.append(vector)

    write_source_files(tmp_path, records)

    digest = corpus_hash(records)
    build_id = f"v1-{digest[:16]}-{DIMENSION}d"
    paths = IndexPaths.for_root(settings.index_root)
    staging = paths.staging(build_id)
    staging.mkdir(parents=True)
    collection_name = collection_name_for(build_id)
    manifest = IndexManifest.create(
        build_id=build_id,
        corpus_hash=digest,
        paper_ids_hash=paper_ids_hash(record.paper_id for record in records),
        record_count=len(records),
        data_file_hashes=data_file_hashes(settings),
        settings=settings,
        collection_name=collection_name,
    )
    client = chromadb.PersistentClient(path=str(staging / "chroma"))
    collection = client.create_collection(
        name=collection_name,
        metadata=collection_metadata_for(manifest),
        embedding_function=None,
    )
    collection.add(
        ids=[record.paper_id for record in records],
        embeddings=vectors,
        documents=[record.text for record in records],
        metadatas=[record.fields for record in records],
    )
    with BM25IndexBuilder(staging / "bm25.sqlite3") as builder:
        builder.add_records(records)
        builder.finalize(corpus_hash=digest)
    write_manifest(staging, manifest)
    promote_staging(settings.index_root, build_id)
    return settings, records, vectors


def test_persistent_hybrid_search_filter_and_ordered_fetch(local_index):
    settings, _records, vectors = local_index
    engine = ChromaHybridSearchEngine(settings, FakeEncoder(vectors[1]))

    hits = engine.search(
        "spectral graph", limit=3, filters=SearchFilters(area="Theory", spotlight_only=True)
    )
    assert [hit.paper_id for hit in hits] == ["p2"]
    assert 0.0 <= hits[0].score <= 1.0
    fetched = engine.fetch(["p3", "missing", "p1", "p3"])
    assert [paper.paper_id for paper in fetched] == ["p3", "p1"]
    engine.close()

    reopened = ChromaHybridSearchEngine(settings, FakeEncoder(vectors[0]))
    assert reopened.count() == 3
    assert reopened.validate().record_count == 3
    assert reopened.fetch(["p1"])[0].title == "Vision transformer"
    reopened.close()


def test_weighted_rrf_normalizes_and_uses_stable_tie_break():
    dense = [RankedPaperId("dense-only", 1, 0.9), RankedPaperId("both", 2, 0.8)]
    bm25 = [RankedPaperId("both", 1, -2.0), RankedPaperId("bm25-only", 2, -1.0)]

    ranked = _weighted_rrf(
        dense, bm25, dense_weight=0.8, bm25_weight=0.2, rrf_k=60, limit=3
    )
    assert [paper_id for paper_id, _score in ranked] == [
        "both",
        "dense-only",
        "bm25-only",
    ]
    assert all(0.0 <= score <= 1.0 for _paper_id, score in ranked)

    tied = _weighted_rrf(
        [RankedPaperId("b", 1, 0.0), RankedPaperId("a", 1, 0.0)],
        [],
        dense_weight=1.0,
        bm25_weight=0.0,
        rrf_k=60,
        limit=2,
    )
    assert [paper_id for paper_id, _score in tied] == ["a", "b"]


def test_search_calls_each_retriever_once_with_requested_limit():
    settings = Settings()
    encoder = FakeEncoder([0.0] * DIMENSION)
    engine = ChromaHybridSearchEngine(settings, encoder)
    engine._manifest = IndexManifest.create(
        build_id="v1-fixed-window",
        corpus_hash="0" * 64,
        paper_ids_hash=paper_ids_hash(["p1", "p2"]),
        record_count=2,
        data_file_hashes={"index.json": "0" * 64, "abstracts.json": "1" * 64},
        settings=settings,
        collection_name="icml-papers-fixed-window",
    )

    class Collection:
        def __init__(self):
            self.query_calls = []
            self.get_calls = 0

        def query(self, **kwargs):
            self.query_calls.append(kwargs)
            return {"ids": [["p1", "p2"]], "distances": [[0.1, 0.2]]}

        def get(self, **_kwargs):
            self.get_calls += 1
            metadatas = [
                make_record("p1", "One", "alpha", area="A").fields,
                make_record("p2", "Two", "beta", area="A").fields,
            ]
            return {"ids": ["p1", "p2"], "metadatas": metadatas, "documents": ["", ""]}

    class Sparse:
        def __init__(self):
            self.calls = []

        def search(self, query, *, limit, filters):
            self.calls.append((query, limit, filters))
            return [RankedPaperId("p2", 1, -1.0)]

    collection = Collection()
    sparse = Sparse()
    engine._collection = collection
    engine._bm25 = sparse

    hits = engine.search("query", limit=2, filters=SearchFilters())

    assert len(hits) == 2
    assert encoder.calls == 1
    assert len(collection.query_calls) == 1
    assert collection.query_calls[0]["n_results"] == 2
    assert sparse.calls[0][1] == 2


def test_manifest_config_mismatch_fails_closed(local_index):
    settings, _records, vectors = local_index
    paths = IndexPaths.for_root(settings.index_root)
    version = paths.version(paths.current.read_text().strip())
    manifest = IndexManifest(**json.loads((version / "manifest.json").read_text()))
    write_manifest(version, replace(manifest, embedding_dim=768))

    engine = ChromaHybridSearchEngine(settings, FakeEncoder(vectors[0]))
    with pytest.raises(IndexUnavailable):
        engine.count()


def test_collection_contract_metadata_mismatch_fails_closed(local_index):
    import chromadb

    settings, _records, vectors = local_index
    version = IndexPaths.for_root(settings.index_root).version(
        IndexPaths.for_root(settings.index_root).current.read_text().strip()
    )
    manifest = IndexManifest(**json.loads((version / "manifest.json").read_text()))
    client = chromadb.PersistentClient(path=str(version / "chroma"))
    collection = client.get_collection(manifest.collection_name, embedding_function=None)
    metadata = dict(collection.metadata or {})
    metadata.pop("hnsw:space")
    metadata["dense_metric"] = "l2"
    collection.modify(metadata=metadata)

    engine = ChromaHybridSearchEngine(settings, FakeEncoder(vectors[0]))
    with pytest.raises(IndexUnavailable):
        engine.count()


def test_same_count_with_different_paper_ids_fails_closed(local_index):
    import chromadb

    settings, _records, vectors = local_index
    paths = IndexPaths.for_root(settings.index_root)
    version = paths.version(paths.current.read_text().strip())
    manifest = IndexManifest(**json.loads((version / "manifest.json").read_text()))
    client = chromadb.PersistentClient(path=str(version / "chroma"))
    collection = client.get_collection(manifest.collection_name, embedding_function=None)
    collection.delete(ids=["p3"])
    replacement = make_record("p4", "Replacement", "different", area="NLP")
    collection.add(
        ids=[replacement.paper_id],
        embeddings=[vectors[2]],
        documents=[replacement.text],
        metadatas=[replacement.fields],
    )

    engine = ChromaHybridSearchEngine(settings, FakeEncoder(vectors[0]))
    with pytest.raises(IndexUnavailable):
        engine.count()


def test_source_snapshot_hash_mismatch_fails_closed(local_index):
    settings, _records, vectors = local_index
    settings.abstracts_path.write_text('{"p1": "changed"}', encoding="utf-8")

    engine = ChromaHybridSearchEngine(settings, FakeEncoder(vectors[0]))
    with pytest.raises(IndexUnavailable):
        engine.count()


def test_source_corpus_fingerprint_mismatch_fails_closed(local_index):
    settings, _records, vectors = local_index
    paths = IndexPaths.for_root(settings.index_root)
    version = paths.version(paths.current.read_text().strip())
    manifest = IndexManifest(**json.loads((version / "manifest.json").read_text()))
    settings.abstracts_path.write_text('{"p1": "changed"}', encoding="utf-8")
    write_manifest(
        version,
        replace(manifest, data_file_hashes=data_file_hashes(settings)),
    )

    engine = ChromaHybridSearchEngine(settings, FakeEncoder(vectors[0]))
    with pytest.raises(IndexUnavailable):
        engine.count()
