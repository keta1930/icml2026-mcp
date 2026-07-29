"""Milvus index planning, building, validation, and resume tests."""

from __future__ import annotations

import json

import pytest

from icml_paper_search.config import Settings
from icml_paper_search.corpus.catalog import load_records
from icml_paper_search.domain.errors import IndexBuildLimitExceeded, IndexUnavailable
from icml_paper_search.embeddings.dashscope import EmbeddingBatch
from icml_paper_search.indexing.builder import _estimate_tokens, build_index, plan_build
from icml_paper_search.indexing.manifest import load_current_manifest

DIMENSION = 1024


class FakeUsageEncoder:
    def __init__(self, *, tokens_per_batch=3, fail_on_call=None):
        self.tokens_per_batch = tokens_per_batch
        self.fail_on_call = fail_on_call
        self.calls: list[list[str]] = []
        self.closed = False

    def encode_documents_with_usage(self, texts, *, show_progress=False):
        self.calls.append(list(texts))
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("provider failure")
        vectors = []
        for index, _ in enumerate(texts):
            vector = [0.0] * DIMENSION
            vector[index] = 1.0
            vectors.append(tuple(vector))
        token_count = (
            sum(_estimate_tokens(text) for text in texts)
            if self.tokens_per_batch == "estimate"
            else self.tokens_per_batch
        )
        return EmbeddingBatch(tuple(vectors), int(token_count))

    def encode_documents(self, texts, *, show_progress=False):
        return [
            list(vector) for vector in self.encode_documents_with_usage(texts).vectors
        ]

    def close(self):
        self.closed = True


class FakeSchema:
    def __init__(self):
        self.fields = []
        self.functions = []

    def add_field(self, name, datatype, **kwargs):
        params = {"dim": kwargs["dim"]} if "dim" in kwargs else {}
        self.fields.append({"name": name, "type": datatype, "params": params, **kwargs})

    def add_function(self, function):
        self.functions.append(function)


class FakeIndexes:
    def __init__(self):
        self.values = []

    def add_index(self, **kwargs):
        self.values.append(kwargs)


class FakeMilvus:
    def __init__(self):
        self.collections = {}
        self.hybrid_calls = []

    def create_schema(self, **_):
        return FakeSchema()

    def prepare_index_params(self):
        return FakeIndexes()

    def create_collection(self, name, *, schema, index_params, **_):
        self.collections[name] = {
            "schema": schema,
            "indexes": index_params,
            "rows": {},
        }

    def has_collection(self, name, **_):
        return name in self.collections

    def drop_collection(self, name, **_):
        self.collections.pop(name)

    def load_collection(self, *_args, **_kwargs):
        return None

    def flush(self, *_args, **_kwargs):
        return None

    def insert(self, name, entities, **_):
        rows = self.collections[name]["rows"]
        for entity in entities:
            rows[entity["paper_id"]] = dict(entity)
        return {"insert_count": len(entities)}

    def describe_collection(self, name, **_):
        schema = self.collections[name]["schema"]
        function = schema.functions[0]
        return {
            "fields": schema.fields,
            "functions": [
                {
                    "name": function.name,
                    "type": function.type,
                    "input_field_names": list(function.input_field_names),
                    "output_field_names": list(function.output_field_names),
                }
            ],
        }

    def list_indexes(self, name, **_):
        return [
            index["index_name"] for index in self.collections[name]["indexes"].values
        ]

    def describe_index(self, name, index_name, **_):
        index = next(
            index
            for index in self.collections[name]["indexes"].values
            if index["index_name"] == index_name
        )
        return {**index, **index.get("params", {})}

    def get_collection_stats(self, name, **_):
        return {"row_count": len(self.collections[name]["rows"])}

    def query(self, name, **_):
        return [{"paper_id": paper_id} for paper_id in self.collections[name]["rows"]]

    def get(self, name, ids, **_):
        rows = self.collections[name]["rows"]
        return [rows[paper_id] for paper_id in ids if paper_id in rows]

    def hybrid_search(self, name, requests, ranker, limit, **_):
        self.hybrid_calls.append((requests, ranker))
        row = next(iter(self.collections[name]["rows"].values()))
        return [[{"distance": 2 / 61, "entity": row}]][:limit]


def write_corpus(path, count=3, *, suffix=""):
    papers = []
    abstracts = {}
    for index in range(count):
        paper_id = f"p{index}{suffix}"
        papers.append(
            {
                "orid": paper_id,
                "title": f"Paper {index} {suffix}",
                "authors": ["Alice"],
                "area": "Theory",
                "sub": "",
                "type": "Poster",
                "spot": index == 0,
                "arxiv": "",
            }
        )
        abstracts[paper_id] = f"graph learning abstract {index}"
    (path / "index.json").write_text(json.dumps({"papers": papers}), encoding="utf-8")
    (path / "abstracts.json").write_text(json.dumps(abstracts), encoding="utf-8")


@pytest.fixture
def build_settings(tmp_path):
    write_corpus(tmp_path)
    return Settings(
        data_dir=tmp_path,
        index_root=tmp_path / "index-store",
        state_db_path=tmp_path / "state.sqlite3",
        embedding_batch_size=2,
    )


def test_preflight_is_offline_and_deterministic(build_settings):
    first = plan_build(build_settings)
    second = plan_build(build_settings)
    result = build_index(build_settings, max_input_tokens=None)
    assert first == second
    assert first.record_count == 3
    assert first.collection_name.endswith(first.build_id)
    assert result.executed is False
    assert not build_settings.index_root.exists()


def test_build_creates_native_schema_and_current_manifest(build_settings):
    client = FakeMilvus()
    encoder = FakeUsageEncoder(tokens_per_batch=4)
    result = build_index(
        build_settings,
        encoder,
        max_input_tokens=100_000,
        execute=True,
        client_factory=lambda: client,
    )
    manifest = load_current_manifest(build_settings.index_root)
    collection = client.collections[manifest.collection_name]
    fields = {field["name"] for field in collection["schema"].fields}
    assert {"dense_vector", "sparse_vector", "document", "paper_id"} <= fields
    assert len(collection["schema"].functions) == 1
    assert {index["metric_type"] for index in collection["indexes"].values} == {
        "COSINE",
        "BM25",
    }
    assert result.manifest_path == build_settings.index_root / "current.json"
    assert result.total_tokens == 8
    assert [len(batch) for batch in encoder.calls] == [2, 1]
    assert client.hybrid_calls[0][1].dict() == {
        "strategy": "rrf",
        "params": {"k": 60},
    }


def test_input_limit_stops_before_provider_and_can_resume(build_settings):
    records = load_records(build_settings)
    limits = [_estimate_tokens(record.document) for record in records]
    client = FakeMilvus()
    first = FakeUsageEncoder(tokens_per_batch="estimate")
    with pytest.raises(IndexBuildLimitExceeded):
        build_index(
            build_settings,
            first,
            max_input_tokens=limits[0] + limits[1] - 1,
            execute=True,
            client_factory=lambda: client,
        )
    assert [len(batch) for batch in first.calls] == [1]
    checkpoint = json.loads(plan_build(build_settings).checkpoint_path.read_text())
    assert checkpoint["completed_ids"] == [records[0].paper_id]

    resumed = FakeUsageEncoder(tokens_per_batch="estimate")
    result = build_index(
        build_settings,
        resumed,
        max_input_tokens=sum(limits),
        execute=True,
        resume=True,
        client_factory=lambda: client,
    )
    assert result.resumed
    assert [len(batch) for batch in resumed.calls] == [2]
    assert result.total_tokens == sum(limits)


def test_provider_usage_over_limit_is_recorded_and_resume_is_rejected(build_settings):
    first_limit = _estimate_tokens(load_records(build_settings)[0].document)
    client = FakeMilvus()
    encoder = FakeUsageEncoder(tokens_per_batch=first_limit + 1)
    with pytest.raises(IndexBuildLimitExceeded):
        build_index(
            build_settings,
            encoder,
            max_input_tokens=first_limit,
            execute=True,
            client_factory=lambda: client,
        )
    failure = json.loads((build_settings.index_root / "build-failure.json").read_text())
    assert failure["reason"] == "provider_usage_exceeded_build_limit"
    assert failure["actual_tokens"] == first_limit + 1
    with pytest.raises(IndexUnavailable):
        build_index(
            build_settings,
            FakeUsageEncoder(),
            max_input_tokens=100_000,
            execute=True,
            resume=True,
            client_factory=lambda: client,
        )


def test_valid_current_collection_is_reused_without_provider_call(build_settings):
    client = FakeMilvus()
    build_index(
        build_settings,
        FakeUsageEncoder(),
        max_input_tokens=100_000,
        execute=True,
        client_factory=lambda: client,
    )
    encoder = FakeUsageEncoder()
    result = build_index(
        build_settings,
        encoder,
        max_input_tokens=100_000,
        execute=True,
        client_factory=lambda: client,
    )
    assert result.reused
    assert encoder.calls == []
